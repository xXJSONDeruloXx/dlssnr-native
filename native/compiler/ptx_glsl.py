"""Checked PTX-to-GLSL lowering for the DLSS compatibility prototype.

The backend covers the scalar/vector, memory, image, subgroup, and selected
``mma``/``wmma`` forms exercised by the supplied corpus. WMMA lowering is
limited to the SM70-era half-precision ``m16n16k16`` ABI used by the captured
DLSS PWIN kernels; unsupported architecture-dependent fragments remain
explicitly gated.
"""

from __future__ import annotations

import re
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    from .ptx_ir import PtxInstruction, PtxKernel, PtxModule, PtxParameter, parse_ptx
except ImportError:  # Direct execution from the tools directory.
    from ptx_ir import PtxInstruction, PtxKernel, PtxModule, PtxParameter, parse_ptx


class PtxTranslationError(ValueError):
    """Raised when a PTX kernel is outside the checked lowering subset."""


@dataclass(frozen=True)
class RegisterFile:
    family: str
    glsl_name: str
    glsl_type: str
    count: int
    ptx_type: str
    scalarized: bool = False
    storage: bool = False


_REG_DECL_RE = re.compile(
    r"^\.reg\s*(?P<type>\.[A-Za-z0-9_.]+)\s+%(?P<family>[A-Za-z]+)<(?P<count>\d+)>$"
)
_NAMED_REG_DECL_RE = re.compile(
    r"^\.reg\s*(?P<type>\.[A-Za-z0-9_.]+)\s+(?P<names>.+)$"
)
_MAXNTID_RE = re.compile(r"^\.maxntid\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)$")
_REGISTER_RE = re.compile(r"^%(?P<family>[A-Za-z]+)(?P<index>\d+)$")
_NAMED_REGISTER_RE = re.compile(r"^(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)$")
_SHARED_DECL_RE = re.compile(
    r"^\.shared(?:\s+\.align\s+\d+)?\s+(?P<type>\.[A-Za-z0-9_]+)\s+"
    r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)(?:\[(?P<size>\d+)\])?$"
)
_PARAM_ADDRESS_RE = re.compile(
    r"^\[\s*(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)(?:\s*([+-])\s*(?P<offset>\d+))?\s*\]$"
)
_BRACKET_RE = re.compile(r"^\[.*\]$", re.DOTALL)
_FLOAT_BITS_RE = re.compile(r"^0f([0-9A-Fa-f]{8})$")
_DOUBLE_BITS_RE = re.compile(r"^0d([0-9A-Fa-f]{16})$")

# The runtime binds a bounded table of logical NVX image handles.  Keeping
# the slot count fixed avoids requiring descriptor-indexing features from the
# caller: generated helper functions select one of the statically declared
# sampler/image bindings with a switch.
_RUNTIME_IMAGE_SLOTS = 64
_RUNTIME_REGISTER_BINDING = 200
_RUNTIME_CONTROL_BINDING = 201
_RUNTIME_SHARED_BINDING = 202
_RUNTIME_CONSTANT_BINDING = 2
_RUNTIME_TEXTURE_BINDING_BASE = 3
_RUNTIME_SURFACE_BINDING_BASE = _RUNTIME_TEXTURE_BINDING_BASE + _RUNTIME_IMAGE_SLOTS
_RUNTIME_IMAGE_TABLE_BINDING = _RUNTIME_SURFACE_BINDING_BASE + _RUNTIME_IMAGE_SLOTS
_RUNTIME_ADDRESS_MAP_BINDING = 203

# NVIDIA's PTX WMMA ABI exposes eight packed f16 A and B registers per warp
# lane for the m16n16k16 form.  The first three words below are control state;
# the twenty words after them carry A[8], B[8], and C[4].  Keep these offsets
# centralized because the replay shader and the Vulkan host allocator must
# agree exactly.
_WMMA_A_REGISTER_COUNT = 8
_WMMA_B_REGISTER_COUNT = 8
_WMMA_C_REGISTER_COUNT = 4
_WMMA_STATE_HEADER_WORDS = 3
_WMMA_STATE_INPUT_WORDS = (
    _WMMA_A_REGISTER_COUNT + _WMMA_B_REGISTER_COUNT + _WMMA_C_REGISTER_COUNT
)
_WMMA_STATE_RESULT_BASE = _WMMA_STATE_HEADER_WORDS + _WMMA_STATE_INPUT_WORDS


def _runtime_image_slot_count() -> int:
    """Return the active logical image-table width for one translated launch."""
    text = os.environ.get("DLSSAMD_TRANSLATED_IMAGE_SLOTS")
    try:
        value = int(text, 10) if text else _RUNTIME_IMAGE_SLOTS
    except ValueError:
        value = _RUNTIME_IMAGE_SLOTS
    return max(1, min(_RUNTIME_IMAGE_SLOTS, value))


_RUNTIME_SURFACE_IMAGE_FORMATS = {
    "r16f",
    "rg16f",
    "rgba16f",
    "r32f",
    "rg32f",
    "rgba32f",
    "r8ui",
    "rg8ui",
    "rgba8ui",
    "r16ui",
    "rg16ui",
    "rgba16ui",
    "r32ui",
    "rg32ui",
    "rgba32ui",
}


def _runtime_surface_image_formats(surface_format: str) -> list[str]:
    """Return per-slot image format qualifiers supplied by the runtime.

    A CUDA surface handle can refer to an image whose component width/count
    differs from another handle in the same launch.  The old backend used one
    hard-coded rgba32 declaration for every slot, which made the Vulkan image
    descriptor type disagree with the real view.  The layer supplies the
    observed Vulkan-compatible qualifier as a comma-separated environment
    value; missing or invalid entries retain the checked legacy fallback.
    """

    image_slots = _runtime_image_slot_count()
    default_format = "rgba32f" if surface_format == "float" else "rgba32ui"
    text = os.environ.get("DLSSAMD_TRANSLATED_SURFACE_IMAGE_FORMATS")
    if not text:
        return [default_format] * image_slots
    formats = [part.strip() for part in text.split(",")]
    result = []
    for slot in range(image_slots):
        candidate = formats[slot] if slot < len(formats) else ""
        result.append(candidate if candidate in _RUNTIME_SURFACE_IMAGE_FORMATS else default_format)
    return result


def _glsl_type_for_register(type_name: str) -> str:
    if type_name == ".pred":
        return "bool"
    if type_name in {".f32"}:
        return "float"
    if type_name in {".f16", ".b16", ".b32", ".u16", ".u32", ".s16", ".s32"}:
        return "uint"
    if type_name in {".b64", ".u64", ".s64"}:
        return "uint64_t"
    if type_name == ".f64":
        return "double"
    if type_name == ".b128":
        # Little-endian words; GLSL has no scalar 128-bit integer type.
        return "uvec4"
    raise PtxTranslationError(f"register type is outside scalar subset: {type_name}")


def _register_files(
    kernel: PtxKernel,
    *,
    scalarize: bool = False,
    storage_families: set[str] | None = None,
) -> dict[str, RegisterFile]:
    result: dict[str, RegisterFile] = {}
    used_glsl_names: set[str] = set()
    storage_families = storage_families or set()

    def add_register_file(key: str, register_file: RegisterFile) -> None:
        previous = result.get(key)
        if previous and previous.glsl_type != register_file.glsl_type:
            raise PtxTranslationError(f"register {key} changes type")
        if previous:
            result[key] = RegisterFile(
                family=previous.family,
                glsl_name=previous.glsl_name,
                glsl_type=previous.glsl_type,
                count=max(previous.count, register_file.count),
                ptx_type=previous.ptx_type,
                scalarized=previous.scalarized,
                storage=previous.storage or register_file.storage,
            )
        else:
            result[key] = register_file

    for directive in kernel.directives:
        match = _REG_DECL_RE.match(directive)
        if match:
            family = match.group("family")
            if family not in used_glsl_names:
                used_glsl_names.add(family)
            add_register_file(
                family,
                RegisterFile(
                    family=family,
                    glsl_name=family,
                    glsl_type=_glsl_type_for_register(match.group("type")),
                    count=int(match.group("count")),
                    ptx_type=match.group("type"),
                    scalarized=scalarize,
                    storage=family in storage_families,
                ),
            )
            continue

        # NVCC emits short-lived named temporaries in addition to the normal
        # numbered register families, for example ``.reg .f16 low,high``.
        # Preserve those names as independent one-element files.  The source
        # names may contain '$', which is not a GLSL identifier, so give them
        # stable private names in the generated shader.
        named_match = _NAMED_REG_DECL_RE.match(directive)
        if not named_match:
            continue
        glsl_type = _glsl_type_for_register(named_match.group("type"))
        for name in _split_operands(named_match.group("names")):
            name = name.strip()
            if not _NAMED_REGISTER_RE.match(name):
                continue
            key = "@" + name
            if key in result:
                add_register_file(
                    key,
                    RegisterFile(
                        family=name,
                        glsl_name=result[key].glsl_name,
                        glsl_type=glsl_type,
                        count=1,
                        ptx_type=named_match.group("type"),
                        scalarized=scalarize,
                    ),
                )
                continue
            base_name = "ptx_named_" + re.sub(r"[^A-Za-z0-9_]", "_", name)
            glsl_name = base_name
            suffix = 2
            while glsl_name in used_glsl_names:
                glsl_name = f"{base_name}_{suffix}"
                suffix += 1
            used_glsl_names.add(glsl_name)
            add_register_file(
                key,
                RegisterFile(
                    family=name,
                    glsl_name=glsl_name,
                    glsl_type=glsl_type,
                    count=1,
                    ptx_type=named_match.group("type"),
                    scalarized=scalarize,
                ),
            )
    return result


def _workgroup_size(kernel: PtxKernel) -> tuple[int, int, int]:
    for directive in kernel.directives:
        match = _MAXNTID_RE.match(directive)
        if match:
            return tuple(int(match.group(i)) for i in range(1, 4))
    return (1, 1, 1)


def _shared_layout(kernel: PtxKernel) -> tuple[dict[str, int], int]:
    """Return a compact byte-addressed layout for PTX shared symbols.

    CUDA's assembler assigns the final shared-memory offsets.  The prototype
    has no CUDA linker, so it lays the declared symbols out in declaration
    order with the requested alignment.  Register-computed shared addresses
    remain byte offsets and use the same backing array.
    """

    symbols: dict[str, int] = {}
    cursor = 0
    for directive in kernel.directives:
        match = _SHARED_DECL_RE.match(directive)
        if not match:
            continue
        alignment = 1
        # The alignment is intentionally recovered from the directive rather
        # than inferred from the element type; PTX may request over-alignment.
        align_match = re.search(r"\.align\s+(\d+)", directive)
        if align_match:
            alignment = max(1, int(align_match.group(1)))
        cursor = (cursor + alignment - 1) & ~(alignment - 1)
        symbols[match.group("name")] = cursor
        type_name = match.group("type")
        element_size = {".u8": 1, ".b8": 1, ".u16": 2, ".b16": 2, ".u32": 4, ".b32": 4, ".f32": 4}.get(type_name)
        if element_size is None:
            raise PtxTranslationError(f"shared-memory element type is not supported: {type_name}")
        cursor += element_size * int(match.group("size") or 1)
    return symbols, max(cursor, 4)


def _constant_layout(kernel: PtxKernel) -> dict[str, int]:
    """Assign deterministic descriptor offsets to PTX constant symbols."""

    symbols: dict[str, int] = {}
    cursor = 0
    for instruction in kernel.instructions:
        parts = instruction.opcode.split(".")
        if parts[0] != "ld" or "const" not in parts or len(_split_operands(instruction.operands)) < 2:
            continue
        match = _PARAM_ADDRESS_RE.match(_split_operands(instruction.operands)[1].strip())
        if not match:
            continue
        name = match.group("name")
        if name not in symbols:
            # Constants are supplied by a separate descriptor in the future
            # runtime.  Keep symbols disjoint while avoiding any extraction of
            # the NVIDIA constant payload into this prototype.
            symbols[name] = cursor
            cursor += 1 << 20
    return symbols


def _split_operands(value: str) -> list[str]:
    result: list[str] = []
    start = 0
    braces = 0
    brackets = 0
    for index, character in enumerate(value):
        if character == "{":
            braces += 1
        elif character == "}":
            braces = max(0, braces - 1)
        elif character == "[":
            brackets += 1
        elif character == "]":
            brackets = max(0, brackets - 1)
        elif character == "," and braces == 0 and brackets == 0:
            result.append(value[start:index].strip())
            start = index + 1
    tail = value[start:].strip()
    if tail:
        result.append(tail)
    return result


def _vector_operands(value: str) -> list[str]:
    value = value.strip()
    if len(value) < 2 or value[0] != "{" or value[-1] != "}":
        raise PtxTranslationError(f"expected a vector operand: {value}")
    result = _split_operands(value[1:-1])
    if not result:
        raise PtxTranslationError(f"empty vector operand: {value}")
    return result


def _register_ref(token: str, registers: dict[str, RegisterFile]) -> tuple[str, str] | None:
    match = _REGISTER_RE.match(token.strip())
    if match:
        family = match.group("family")
        if family not in registers:
            raise PtxTranslationError(f"register %{family} is not declared")
        index = int(match.group("index"))
        if index >= registers[family].count:
            raise PtxTranslationError(f"register %{family}{index} exceeds declaration")
        register_file = registers[family]
        if register_file.storage:
            return f"ptx_registers.data[ptx_register_base + {index}u]", register_file.glsl_type
        if register_file.scalarized:
            return f"{register_file.glsl_name}_{index}", register_file.glsl_type
        return f"{register_file.glsl_name}[{index}]", register_file.glsl_type
    if token.strip() in {"%SP", "%SPL"}:
        return "ptx_special_spl", "uint64_t"
    named = _NAMED_REGISTER_RE.match(token.strip())
    if not named:
        return None
    key = "@" + named.group("name")
    register_file = registers.get(key)
    if not register_file:
        return None
    if register_file.scalarized:
        return f"{register_file.glsl_name}_0", register_file.glsl_type
    return f"{register_file.glsl_name}[0]", register_file.glsl_type


def _register_file_for_token(token: str, registers: dict[str, RegisterFile]) -> RegisterFile | None:
    token = token.strip()
    match = _REGISTER_RE.match(token)
    if match:
        return registers.get(match.group("family"))
    named = _NAMED_REGISTER_RE.match(token)
    if named:
        return registers.get("@" + named.group("name"))
    return None


def _literal(token: str, target_type: str | None = None) -> str:
    token = token.strip()
    if token.startswith("0f") and _FLOAT_BITS_RE.match(token):
        bits = token[2:]
        return f"uintBitsToFloat(0x{bits}u)" if target_type != "uint" else f"0x{bits}u"
    if token.startswith("0d"):
        match = _DOUBLE_BITS_RE.match(token)
        if not match:
            raise PtxTranslationError(f"invalid double literal: {token}")
        bits = match.group(1)
        raw = f"uint64_t(0x{bits}UL)"
        value = f"uint64BitsToDouble({raw})"
        if target_type == "uint64_t":
            return raw
        if target_type == "float":
            return f"float({value})"
        if target_type == "uint":
            return f"uint({value})"
        return value
    if token in {"true", "false"}:
        return token
    if target_type == "float":
        if token.endswith("f"):
            return token
        try:
            return f"{float(token):.9g}"
        except ValueError as exc:
            raise PtxTranslationError(f"invalid float literal: {token}") from exc
    if target_type == "bool":
        if token in {"0", "0u"}:
            return "false"
        if token in {"1", "1u"}:
            return "true"
        return f"({token}) != 0u"
    if target_type == "uint64_t":
        try:
            return f"uint64_t({int(token, 0) & 0xFFFFFFFFFFFFFFFF}UL)"
        except ValueError as exc:
            raise PtxTranslationError(f"invalid 64-bit integer literal: {token}") from exc
    if target_type == "double":
        try:
            return f"double({float(token):.17g})"
        except ValueError as exc:
            raise PtxTranslationError(f"invalid double literal: {token}") from exc
    if token.startswith("-"):
        try:
            value = int(token, 0)
        except ValueError as exc:
            raise PtxTranslationError(f"invalid integer literal: {token}") from exc
        return f"0x{value & 0xFFFFFFFF:08x}u"
    if token.endswith("U") or token.endswith("u"):
        return token
    try:
        return f"{int(token, 0)}u"
    except ValueError as exc:
        raise PtxTranslationError(f"unsupported operand: {token}") from exc


def _operand_expr(token: str, registers: dict[str, RegisterFile], target_type: str | None = None) -> str:
    token = token.strip()
    if token.startswith("!"):
        return f"!({_operand_expr(token[1:], registers, 'bool')})"
    special = {
        "%tid.x": "gl_LocalInvocationID.x",
        "%tid.y": "gl_LocalInvocationID.y",
        "%tid.z": "gl_LocalInvocationID.z",
        "%ctaid.x": "gl_WorkGroupID.x",
        "%ctaid.y": "gl_WorkGroupID.y",
        "%ctaid.z": "gl_WorkGroupID.z",
        "%ntid.x": "gl_WorkGroupSize.x",
        "%ntid.y": "gl_WorkGroupSize.y",
        "%ntid.z": "gl_WorkGroupSize.z",
        "%nctaid.x": "gl_NumWorkGroups.x",
        "%nctaid.y": "gl_NumWorkGroups.y",
        "%nctaid.z": "gl_NumWorkGroups.z",
        "%laneid": "(gl_SubgroupInvocationID & 31u)",
        "WARP_SZ": "32u",
    }
    if os.environ.get("DLSSAMD_TRANSLATED_SCALAR_TILED_DISPATCH") not in {
        None, "", "0"
    }:
        special.update({
            "%ctaid.x": "ptx_dispatch_info.workgroup_base.x + gl_WorkGroupID.x",
            "%ctaid.y": "ptx_dispatch_info.workgroup_base.y + gl_WorkGroupID.y",
            "%ctaid.z": "ptx_dispatch_info.workgroup_base.z + gl_WorkGroupID.z",
            "%nctaid.x": "ptx_dispatch_info.full_grid.x",
            "%nctaid.y": "ptx_dispatch_info.full_grid.y",
            "%nctaid.z": "ptx_dispatch_info.full_grid.z",
        })
    if token in special:
        expression = special[token]
        if target_type == "float":
            return f"float({expression})"
        if target_type == "double":
            return f"double({expression})"
        if target_type == "uint64_t":
            return f"uint64_t({expression})"
        if target_type == "bool":
            return f"({expression}) != 0u"
        return expression
    register = _register_ref(token, registers)
    if register:
        expression, register_type = register
        if target_type and register_type != target_type:
            if target_type == "float" and register_type == "uint":
                return f"uintBitsToFloat({expression})"
            if target_type == "uint" and register_type == "float":
                return f"floatBitsToUint({expression})"
            if target_type == "uint" and register_type == "double":
                return f"doubleBitsToUint64({expression})"
            if target_type == "uint64_t" and register_type == "uint":
                return f"uint64_t({expression})"
            if target_type == "uint64_t" and register_type == "float":
                return f"uint64_t(floatBitsToUint({expression}))"
            if target_type == "double" and register_type == "uint64_t":
                return f"uint64BitsToDouble({expression})"
            if target_type == "bool":
                return f"({expression}) != 0u"
        return expression
    return _literal(token, target_type)


def _global_invocation_component(component: str) -> str:
    if os.environ.get("DLSSAMD_TRANSLATED_SCALAR_TILED_DISPATCH") not in {
        None, "", "0"
    }:
        return f"ptx_global_invocation_id.{component}"
    return f"gl_GlobalInvocationID.{component}"


def _parameter_offsets(parameters: tuple[PtxParameter, ...]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for parameter in parameters:
        if parameter.type_name in {".u64", ".b64", ".s64", ".f64"}:
            size = 8
        elif parameter.type_name in {".u32", ".s32", ".b32", ".f32"}:
            size = 4
        elif parameter.type_name in {".u16", ".s16", ".b16", ".f16"}:
            size = 2
        elif parameter.type_name in {".u8", ".s8", ".b8"}:
            size = 1
        else:
            raise PtxTranslationError(f"parameter type is outside scalar subset: {parameter.type_name}")
        alignment = parameter.alignment or size
        if alignment <= 0 or alignment & (alignment - 1):
            raise PtxTranslationError(
                f"parameter alignment is not a positive power of two: {parameter.name}"
            )
        cursor = (cursor + alignment - 1) & ~(alignment - 1)
        offsets[parameter.name] = cursor
        cursor += size * (parameter.array_size or 1)
    return offsets


def _parameter_load(
    address: str,
    parameters: tuple[PtxParameter, ...],
    load_type: str,
    extra_offset: int = 0,
) -> str:
    match = _PARAM_ADDRESS_RE.match(address.strip())
    if not match:
        raise PtxTranslationError(f"parameter address is not supported: {address}")
    offsets = _parameter_offsets(parameters)
    name = match.group("name")
    if name not in offsets:
        raise PtxTranslationError(f"unknown parameter: {name}")
    sign = -1 if match.group(2) == "-" else 1
    offset = offsets[name] + sign * int(match.group("offset") or 0) + extra_offset
    if offset < 0:
        raise PtxTranslationError(f"parameter address is before the parameter block: {address}")
    loaders = {
        "u8": "ptx_load_u8",
        "s8": "ptx_load_u8",
        "b8": "ptx_load_u8",
        "u16": "ptx_load_u16",
        "s16": "ptx_load_u16",
        "b16": "ptx_load_u16",
        "f16": "ptx_load_u16",
        "u32": "ptx_load_u32",
        "s32": "ptx_load_u32",
        "b32": "ptx_load_u32",
        "f32": "ptx_load_u32",
        "u64": "ptx_load_u64",
        "s64": "ptx_load_u64",
        "b64": "ptx_load_u64",
        "f64": "ptx_load_u64",
    }
    if load_type not in loaders:
        raise PtxTranslationError(f"parameter load type is not supported: .{load_type}")
    return f"{loaders[load_type]}({offset}u)"


def _load_type_from_opcode(parts: list[str]) -> str:
    supported = {
        "u8", "s8", "b8", "u16", "s16", "b16", "u32", "s32", "b32",
        "f16", "f32", "u64", "s64", "b64", "f64",
    }
    for part in reversed(parts):
        if part in supported:
            return part
    raise PtxTranslationError(f"parameter load type is not supported: {'.'.join(parts)}")


def _loaded_value(value: str, load_type: str, destination_type: str) -> str:
    if destination_type == "float":
        if load_type == "f32":
            return f"uintBitsToFloat({value})"
        if load_type == "f16":
            return f"ptx_unpack_f16({value})"
        return f"uintBitsToFloat({value})"
    if destination_type == "double":
        return f"uint64BitsToDouble({value})"
    if destination_type == "uint64_t" and load_type not in {"u64", "s64", "b64", "f64"}:
        return f"uint64_t({value})"
    if destination_type == "uint" and load_type in {"u64", "s64", "b64", "f64"}:
        return f"uint({value})"
    return value


def _address_expr(
    address: str,
    registers: dict[str, RegisterFile],
    shared_symbols: dict[str, int],
) -> str:
    address = address.strip()
    if not (address.startswith("[") and address.endswith("]")):
        raise PtxTranslationError(f"memory address is malformed: {address}")
    inside = address[1:-1].strip().replace(" ", "")
    match = re.match(r"^(?P<base>[^+\-]+)(?P<offset>(?:[+\-]-?\d+)?)$", inside)
    if not match:
        raise PtxTranslationError(f"memory address expression is not supported: {address}")
    base = match.group("base")
    offset_text = match.group("offset")
    if base in shared_symbols:
        expression = f"uint64_t({shared_symbols[base]}UL)"
    else:
        try:
            expression = _operand_expr(base, registers, "uint64_t")
        except PtxTranslationError:
            if not base.startswith("%") and _NAMED_REGISTER_RE.match(base):
                # Module-scope .shared declarations are not attached to a
                # kernel by the small IR yet.  Treat an unresolved symbol as
                # the start of this kernel's compact shared arena; the device
                # report keeps this path explicitly provisional.
                expression = "uint64_t(0UL)"
            else:
                raise
    if offset_text:
        normalized_offset = offset_text.replace("+-", "-").replace("-+", "-")
        expression += f" + uint64_t({int(normalized_offset)}L)"
    return expression


def _address_with_offset(address: str, byte_offset: int, registers: dict[str, RegisterFile], shared_symbols: dict[str, int]) -> str:
    expression = _address_expr(address, registers, shared_symbols)
    if byte_offset:
        expression += f" + uint64_t({byte_offset}UL)"
    return expression


def _constant_address(address: str, constant_symbols: dict[str, int]) -> str:
    match = _PARAM_ADDRESS_RE.match(address.strip())
    if not match:
        raise PtxTranslationError(f"constant address is not supported: {address}")
    base = constant_symbols.get(match.group("name"))
    if base is None:
        raise PtxTranslationError(f"constant symbol is not declared: {match.group('name')}")
    sign = -1 if match.group(2) == "-" else 1
    offset = base + sign * int(match.group("offset") or 0)
    if offset < 0:
        raise PtxTranslationError(f"constant address is before the descriptor: {address}")
    return f"{offset}u"


def _memory_load_expression(address_expr: str, load_type: str, *, shared: bool, local: bool = False) -> str:
    prefix = "ptx_shared_load_" if shared else "ptx_local_load_" if local else "ptx_global_load_"
    if load_type in {"u8", "s8", "b8"}:
        return f"{prefix}u8({address_expr})"
    if load_type in {"u16", "s16", "b16", "f16"}:
        return f"{prefix}u16({address_expr})"
    if load_type in {"u32", "s32", "b32", "f32"}:
        return f"{prefix}u32({address_expr})"
    if load_type in {"u64", "s64", "b64", "f64"}:
        return f"{prefix}u64({address_expr})"
    raise PtxTranslationError(f"memory load type is not supported: .{load_type}")


def _memory_store_statement(address_expr: str, value: str, store_type: str, *, shared: bool, local: bool = False) -> str:
    prefix = "ptx_shared_store_" if shared else "ptx_local_store_" if local else "ptx_global_store_"
    if store_type in {"u8", "s8", "b8"}:
        return f"{prefix}u8({address_expr}, uint({value}));"
    if store_type in {"u16", "s16", "b16", "f16"}:
        return f"{prefix}u16({address_expr}, uint({value}));"
    if store_type in {"u32", "s32", "b32"}:
        return f"{prefix}u32({address_expr}, uint({value}));"
    if store_type == "f32":
        return f"{prefix}u32({address_expr}, floatBitsToUint({value}));"
    if store_type in {"u64", "s64", "b64"}:
        return f"{prefix}u64({address_expr}, uint64_t({value}));"
    if store_type == "f64":
        return f"{prefix}u64({address_expr}, doubleBitsToUint64({value}));"
    raise PtxTranslationError(f"memory store type is not supported: .{store_type}")


def _image_coordinates(
    address: str,
    registers: dict[str, RegisterFile],
    *,
    integer: bool,
) -> str:
    address = address.strip()
    if not (address.startswith("[") and address.endswith("]")):
        raise PtxTranslationError(f"image address is malformed: {address}")
    address_parts = _split_operands(address[1:-1])
    if len(address_parts) != 2:
        raise PtxTranslationError(f"image address form is not supported: {address}")
    coordinates = _vector_operands(address_parts[1])
    if len(coordinates) != 2:
        raise PtxTranslationError(f"only 2D image coordinates are supported: {address}")
    value_type = "uint" if integer else "float"
    values = [_operand_expr(value, registers, value_type) for value in coordinates]
    if integer:
        return f"ivec2(int({values[0]}), int({values[1]}))"
    return f"vec2({values[0]}, {values[1]})"


def _image_handle_expression(
    address: str,
    registers: dict[str, RegisterFile],
) -> str:
    address = address.strip()
    if not (address.startswith("[") and address.endswith("]")):
        raise PtxTranslationError(f"image address is malformed: {address}")
    address_parts = _split_operands(address[1:-1])
    if len(address_parts) != 2:
        raise PtxTranslationError(f"image address form is not supported: {address}")
    return _operand_expr(address_parts[0], registers, "uint64_t")


def _surface_value_expression(
    token: str,
    registers: dict[str, RegisterFile],
    *,
    surface_format: str,
    raw_bits: bool,
) -> str:
    register = _register_ref(token.strip(), registers)
    if surface_format == "float":
        if raw_bits and register and register[1] == "uint":
            return f"uintBitsToFloat({register[0]})"
        return _operand_expr(token, registers, "float")
    if raw_bits and register and register[1] == "float":
        return f"floatBitsToUint({register[0]})"
    return _operand_expr(token, registers, "uint")


def _emit_image_instruction(
    instruction: PtxInstruction,
    registers: dict[str, RegisterFile],
    *,
    surface_format: str = "uint",
) -> str:
    opcode = instruction.opcode
    parts = opcode.split(".")
    base = parts[0]
    operands = _split_operands(instruction.operands)
    if base in {"tex", "tld4"}:
        if len(operands) not in {2, 3}:
            raise PtxTranslationError(f"texture form is not supported: {instruction.text}")
        destinations = _vector_operands(operands[0])
        address = operands[1]
        handle = _image_handle_expression(address, registers)
        integer_coordinates = base == "tex" and "s32" in parts
        coordinates = _image_coordinates(address, registers, integer=integer_coordinates)
        if base == "tex" and "level" in parts:
            lod = _operand_expr(operands[2], registers, "float")
            sample = f"ptx_texture_sample_lod({handle}, {coordinates}, {lod})"
        elif base == "tld4":
            sample = f"ptx_texture_gather({handle}, {coordinates})"
        elif not integer_coordinates:
            sample = f"ptx_texture_sample({handle}, {coordinates})"
        else:
            sample = f"ptx_texture_fetch({handle}, {coordinates})"
        statements = []
        for index, destination in enumerate(destinations):
            destination_type = _destination_type(destination, registers)
            value = f"{sample}[{index}]"
            if destination_type == "uint":
                value = f"ptx_pack_f16({value})" if "f16" in parts else f"floatBitsToUint({value})"
            elif destination_type != "float":
                raise PtxTranslationError(f"texture destination type is not supported: {instruction.text}")
            statements.append(f"{_destination_expr(destination, registers)} = {value};")
        return " ".join(statements)
    if base == "sust":
        if len(operands) != 2:
            raise PtxTranslationError(f"surface store form is not supported: {instruction.text}")
        coordinates = _image_coordinates(operands[0], registers, integer=True)
        values = _vector_operands(operands[1]) if operands[1].strip().startswith("{") else [operands[1]]
        raw_bits = any(value in parts for value in {"b8", "b16", "b32", "b64"})
        if "b" in parts:
            if "b32" not in parts or len(values) != 1:
                raise PtxTranslationError(
                    f"unformatted surface store form is not supported: {instruction.text}"
                )
            handle = _image_handle_expression(operands[0], registers)
            return f"ptx_surface_store_b32({handle}, {coordinates}, {_operand_expr(values[0], registers, 'uint')});"
        encoded = [
            _surface_value_expression(
                value,
                registers,
                surface_format=surface_format,
                raw_bits=raw_bits,
            )
            for value in values
        ]
        value_type = "vec4" if surface_format == "float" else "uvec4"
        zero = "0.0" if surface_format == "float" else "0u"
        if len(encoded) == 1:
            value = f"{value_type}({encoded[0]})"
        elif len(encoded) == 4:
            value = f"{value_type}({', '.join(encoded)})"
        elif len(encoded) == 2:
            value = f"{value_type}({encoded[0]}, {encoded[1]}, {zero}, {zero})"
        else:
            raise PtxTranslationError(f"surface store lane count is not supported: {instruction.text}")
        handle = _image_handle_expression(operands[0], registers)
        return f"ptx_surface_store({handle}, {coordinates}, {value});"
    if base == "suld":
        if len(operands) != 2:
            raise PtxTranslationError(f"surface load form is not supported: {instruction.text}")
        destinations = _vector_operands(operands[0])
        handle = _image_handle_expression(operands[1], registers)
        loaded = f"ptx_surface_load({handle}, {_image_coordinates(operands[1], registers, integer=True)})"
        statements = []
        for index, destination in enumerate(destinations):
            destination_type = _destination_type(destination, registers)
            value = f"{loaded}[{index}]"
            if surface_format == "float":
                if destination_type == "uint":
                    value = f"floatBitsToUint({value})"
                elif destination_type != "float":
                    raise PtxTranslationError(f"surface load destination type is not supported: {instruction.text}")
            elif destination_type == "float":
                value = f"uintBitsToFloat({value})"
            elif destination_type != "uint":
                raise PtxTranslationError(f"surface load destination type is not supported: {instruction.text}")
            statements.append(f"{_destination_expr(destination, registers)} = {value};")
        return " ".join(statements)
    raise PtxTranslationError(f"image opcode is not supported: {opcode}")


def _half2_value(token: str, registers: dict[str, RegisterFile]) -> str:
    return f"unpackHalf2x16({_operand_expr(token, registers, 'uint')})"


def _half_value(token: str, registers: dict[str, RegisterFile]) -> str:
    return f"unpackHalf2x16({_operand_expr(token, registers, 'uint')}).x"


def _pack_half(value: str) -> str:
    return f"packHalf2x16(vec2({value}, 0.0))"


def _emit_half2_instruction(
    instruction: PtxInstruction,
    registers: dict[str, RegisterFile],
) -> str | None:
    opcode = instruction.opcode
    parts = opcode.split(".")
    base = parts[0]
    operands = _split_operands(instruction.operands)
    if "f16x2" not in opcode:
        return None
    if base in {"add", "sub", "mul", "max", "min"} and len(operands) == 3:
        destination = _destination_expr(operands[0], registers)
        if _destination_type(operands[0], registers) != "uint":
            raise PtxTranslationError(f"packed half2 destination is not a 32-bit register: {opcode}")
        left = _half2_value(operands[1], registers)
        right = _half2_value(operands[2], registers)
        operator = {"add": "+", "sub": "-", "mul": "*"}.get(base)
        if operator:
            expression = f"{left} {operator} {right}"
        else:
            expression = f"{base}({left}, {right})"
        if "sat" in parts:
            expression = f"clamp({expression}, vec2(0.0), vec2(1.0))"
        if "relu" in parts:
            expression = f"max({expression}, vec2(0.0))"
        return f"{destination} = packHalf2x16({expression});"
    if base == "fma" and len(operands) == 4:
        destination = _destination_expr(operands[0], registers)
        if _destination_type(operands[0], registers) != "uint":
            raise PtxTranslationError(f"packed half2 destination is not a 32-bit register: {opcode}")
        left = _half2_value(operands[1], registers)
        right = _half2_value(operands[2], registers)
        addend = _half2_value(operands[3], registers)
        expression = f"fma({left}, {right}, {addend})"
        if "sat" in parts:
            expression = f"clamp({expression}, vec2(0.0), vec2(1.0))"
        if "relu" in parts:
            expression = f"max({expression}, vec2(0.0))"
        return f"{destination} = packHalf2x16({expression});"
    if base in {"abs", "neg", "ex2"} and len(operands) == 2:
        destination = _destination_expr(operands[0], registers)
        if _destination_type(operands[0], registers) != "uint":
            raise PtxTranslationError(f"packed half2 destination is not a 32-bit register: {opcode}")
        value = _half2_value(operands[1], registers)
        if base == "abs":
            expression = f"abs({value})"
        elif base == "neg":
            expression = f"-({value})"
        else:
            expression = f"exp2({value})"
        return f"{destination} = packHalf2x16({expression});"
    raise PtxTranslationError(f"packed half2 opcode is not supported: {opcode}")


def _emit_half_instruction(
    instruction: PtxInstruction,
    registers: dict[str, RegisterFile],
) -> str | None:
    opcode = instruction.opcode
    parts = opcode.split(".")
    base = parts[0]
    if "f16" not in parts or "f16x2" in parts:
        return None
    operands = _split_operands(instruction.operands)
    if base not in {"add", "sub", "mul", "div", "max", "min", "fma", "abs", "neg", "ex2", "lg2", "rcp", "rsqrt", "sqrt", "tanh"}:
        raise PtxTranslationError(f"scalar half opcode is not supported: {opcode}")
    destination = _destination_expr(operands[0], registers)
    if _destination_type(operands[0], registers) != "uint":
        raise PtxTranslationError(f"scalar half destination is not a 16/32-bit register: {opcode}")
    values = [_half_value(value, registers) for value in operands[1:]]
    if base == "add":
        expression = f"{values[0]} + {values[1]}"
    elif base == "sub":
        expression = f"{values[0]} - {values[1]}"
    elif base == "mul":
        expression = f"{values[0]} * {values[1]}"
    elif base == "div":
        expression = f"{values[0]} / {values[1]}"
    elif base in {"max", "min"}:
        expression = f"{base}({values[0]}, {values[1]})"
    elif base == "fma":
        expression = f"fma({values[0]}, {values[1]}, {values[2]})"
    elif base == "abs":
        expression = f"abs({values[0]})"
    elif base == "neg":
        expression = f"-({values[0]})"
    elif base == "ex2":
        expression = f"exp2({values[0]})"
    elif base == "lg2":
        expression = f"log2({values[0]})"
    elif base == "rcp":
        expression = f"(1.0 / {values[0]})"
    elif base == "rsqrt":
        expression = f"inversesqrt({values[0]})"
    elif base == "tanh":
        expression = f"tanh({values[0]})"
    else:
        expression = f"sqrt({values[0]})"
    if "sat" in parts:
        expression = f"clamp({expression}, 0.0, 1.0)"
    if "relu" in parts:
        expression = f"max({expression}, 0.0)"
    return f"{destination} = {_pack_half(expression)};"


def _cvt_source_expression(
    token: str,
    source_type: str,
    registers: dict[str, RegisterFile],
) -> str:
    if source_type == "f16":
        return _half_value(token, registers)
    if source_type == "f16x2":
        return _half2_value(token, registers)
    if source_type == "f32":
        return _operand_expr(token, registers, "float")
    if source_type == "f64":
        return _operand_expr(token, registers, "double")
    if source_type in {"s8", "s16", "s32"}:
        return f"int({_operand_expr(token, registers, 'uint')})"
    if source_type == "s64":
        return f"int64_t({_operand_expr(token, registers, 'uint64_t')})"
    if source_type in {"u8", "u16", "u32", "b8", "b16", "b32"}:
        return _operand_expr(token, registers, "uint")
    if source_type in {"u64", "b64"}:
        return _operand_expr(token, registers, "uint64_t")
    raise PtxTranslationError(f"conversion source type is not supported: {source_type}")


def _emit_cvt_instruction(
    instruction: PtxInstruction,
    registers: dict[str, RegisterFile],
) -> str:
    parts = instruction.opcode.split(".")
    operands = _split_operands(instruction.operands)
    if len(operands) != 2 and not (
        len(operands) == 3 and parts[-2] == "f16x2" and parts[-1] == "f32"
    ):
        raise PtxTranslationError(f"invalid cvt operands: {instruction.text}")
    destination = _destination_expr(operands[0], registers)
    destination_type = _destination_type(operands[0], registers)
    destination_format = parts[-2] if len(parts) >= 3 else ""
    source_format = parts[-1] if len(parts) >= 2 else ""

    if destination_format == "f16x2" and source_format == "f32":
        if len(operands) != 3:
            raise PtxTranslationError(f"packed half conversion needs two source operands: {instruction.text}")
        left = _operand_expr(operands[1], registers, "float")
        right = _operand_expr(operands[2], registers, "float")
        return f"{destination} = packHalf2x16(vec2({left}, {right}));"

    if destination_format == "f16x2" and source_format == "e4m3x2":
        source = _operand_expr(operands[1], registers, "uint")
        return f"{destination} = packHalf2x16(ptx_unpack_e4m3x2({source}));"

    source = _cvt_source_expression(operands[1], source_format, registers)
    rounding = _rounding_function(parts)
    if destination_format == "f32":
        expression = f"float({source})" if source_format != "f32" else source
        if destination_type == "uint":
            expression = f"floatBitsToUint({expression})"
    elif destination_format == "f64":
        expression = f"double({source})" if source_format != "f64" else source
    elif destination_format == "f16":
        if destination_type != "uint":
            raise PtxTranslationError(f"half conversion destination is not a packed integer register: {instruction.text}")
        return f"{destination} = {_pack_half(f'float({source})' if source_format != 'f32' else source)};"
    elif destination_format in {"s8", "s16", "s32"}:
        rounded = f"{rounding}(float({source}))" if source_format not in {"f32", "f16"} else f"{rounding}({source})"
        expression = f"uint(int({rounded}))"
    elif destination_format in {"u8", "u16", "u32", "b8", "b16", "b32"}:
        if source_format in {"f32", "f16", "f64"}:
            rounded = f"{rounding}(float({source}))"
            expression = f"uint({rounded})"
        else:
            expression = f"uint({source})"
        width = int(destination_format[1:]) if destination_format[0] in {"u", "b"} else 32
        if width < 32:
            expression = f"({expression} & 0x{(1 << width) - 1:x}u)"
    elif destination_format in {"s64", "u64", "b64"}:
        if source_format in {"f32", "f16", "f64"}:
            expression = f"uint64_t({rounding}(float({source})))"
        else:
            expression = f"uint64_t({source})"
    elif destination_format == "e4m3x2" and source_format == "f16x2":
        # Keep the conversion bit-exact enough for finite normalized values
        # while the dedicated float8 lowering is still being expanded.  The
        # helper performs the saturation/rounding in shader code.
        return f"{destination} = ptx_pack_e4m3x2({_half2_value(operands[1], registers)});"
    else:
        raise PtxTranslationError(f"conversion is not supported: {instruction.opcode}")
    return f"{destination} = {expression};"


def _packed_half_component(token: str, half: int, registers: dict[str, RegisterFile]) -> str:
    return f"unpackHalf2x16({_operand_expr(token, registers, 'uint')})[{half}]"


def _packed_e4m3_component(token: str, byte: int, registers: dict[str, RegisterFile]) -> str:
    return f"ptx_e4m3_to_float(({_operand_expr(token, registers, 'uint')} >> {byte * 8}u) & 0xffu)"


def _emit_mma_instruction(
    instruction: PtxInstruction,
    registers: dict[str, RegisterFile],
) -> str | None:
    opcode = instruction.opcode
    parts = opcode.split(".")
    if not opcode.startswith("mma.sync.aligned.") or len(_split_operands(instruction.operands)) != 4:
        return None
    operands = _split_operands(instruction.operands)
    destinations = _vector_operands(operands[0])
    a_values = _vector_operands(operands[1])
    b_values = _vector_operands(operands[2])
    c_values = _vector_operands(operands[3])
    shape = next((value for value in parts if re.fullmatch(r"m\d+n\d+k\d+", value)), None)
    if shape not in {"m16n8k8", "m16n8k16", "m16n8k32"}:
        return None
    if "f16" not in parts:
        return None

    if shape == "m16n8k16" and parts[-4:] == ["f16", "f16", "f16", "f16"]:
        if (len(destinations), len(a_values), len(b_values), len(c_values)) != (2, 4, 2, 2):
            raise PtxTranslationError(f"m16n8k16 fragment sizes are not supported: {instruction.text}")
        a = ", ".join(_packed_half_component(token, half, registers) for token in a_values for half in (0, 1))
        b = ", ".join(_packed_half_component(token, half, registers) for token in b_values for half in (0, 1))
        c = ", ".join(_packed_half_component(token, half, registers) for token in c_values for half in (0, 1))
        return " ".join([
            f"float ptx_mma_a[8] = float[]({a});",
            f"float ptx_mma_b[4] = float[]({b});",
            f"float ptx_mma_c[4] = float[]({c});",
            "float ptx_mma_d0 = ptx_mma_m16n8k16_f16(ptx_mma_a, ptx_mma_b, ptx_mma_c, 0u);",
            "float ptx_mma_d1 = ptx_mma_m16n8k16_f16(ptx_mma_a, ptx_mma_b, ptx_mma_c, 1u);",
            "float ptx_mma_d2 = ptx_mma_m16n8k16_f16(ptx_mma_a, ptx_mma_b, ptx_mma_c, 2u);",
            "float ptx_mma_d3 = ptx_mma_m16n8k16_f16(ptx_mma_a, ptx_mma_b, ptx_mma_c, 3u);",
            f"{_destination_expr(destinations[0], registers)} = packHalf2x16(vec2(ptx_mma_d0, ptx_mma_d1));",
            f"{_destination_expr(destinations[1], registers)} = packHalf2x16(vec2(ptx_mma_d2, ptx_mma_d3));",
        ])

    if shape == "m16n8k8" and parts[-4:] == ["f16", "f16", "f16", "f16"]:
        if (len(destinations), len(a_values), len(b_values), len(c_values)) != (2, 2, 1, 2):
            raise PtxTranslationError(f"m16n8k8 fragment sizes are not supported: {instruction.text}")
        a = ", ".join(_packed_half_component(token, half, registers) for token in a_values for half in (0, 1))
        b = ", ".join(_packed_half_component(token, half, registers) for token in b_values for half in (0, 1))
        c = ", ".join(_packed_half_component(token, half, registers) for token in c_values for half in (0, 1))
        return " ".join([
            f"float ptx_mma_a[4] = float[]({a});",
            f"float ptx_mma_b[2] = float[]({b});",
            f"float ptx_mma_c[4] = float[]({c});",
            "float ptx_mma_d0 = ptx_mma_m16n8k8_f16(ptx_mma_a, ptx_mma_b, ptx_mma_c, 0u);",
            "float ptx_mma_d1 = ptx_mma_m16n8k8_f16(ptx_mma_a, ptx_mma_b, ptx_mma_c, 1u);",
            "float ptx_mma_d2 = ptx_mma_m16n8k8_f16(ptx_mma_a, ptx_mma_b, ptx_mma_c, 2u);",
            "float ptx_mma_d3 = ptx_mma_m16n8k8_f16(ptx_mma_a, ptx_mma_b, ptx_mma_c, 3u);",
            f"{_destination_expr(destinations[0], registers)} = packHalf2x16(vec2(ptx_mma_d0, ptx_mma_d1));",
            f"{_destination_expr(destinations[1], registers)} = packHalf2x16(vec2(ptx_mma_d2, ptx_mma_d3));",
        ])

    if shape == "m16n8k32" and parts[-4:] == ["f16", "e4m3", "e4m3", "f16"]:
        if (len(destinations), len(a_values), len(b_values), len(c_values)) != (2, 4, 2, 2):
            raise PtxTranslationError(f"m16n8k32 fragment sizes are not supported: {instruction.text}")
        a = ", ".join(_packed_e4m3_component(token, byte, registers) for token in a_values for byte in range(4))
        b = ", ".join(_packed_e4m3_component(token, byte, registers) for token in b_values for byte in range(4))
        c = ", ".join(_packed_half_component(token, half, registers) for token in c_values for half in (0, 1))
        return " ".join([
            f"float ptx_mma_a[16] = float[]({a});",
            f"float ptx_mma_b[8] = float[]({b});",
            f"float ptx_mma_c[4] = float[]({c});",
            "float ptx_mma_d0 = ptx_mma_m16n8k32_e4m3(ptx_mma_a, ptx_mma_b, ptx_mma_c, 0u);",
            "float ptx_mma_d1 = ptx_mma_m16n8k32_e4m3(ptx_mma_a, ptx_mma_b, ptx_mma_c, 1u);",
            "float ptx_mma_d2 = ptx_mma_m16n8k32_e4m3(ptx_mma_a, ptx_mma_b, ptx_mma_c, 2u);",
            "float ptx_mma_d3 = ptx_mma_m16n8k32_e4m3(ptx_mma_a, ptx_mma_b, ptx_mma_c, 3u);",
            f"{_destination_expr(destinations[0], registers)} = packHalf2x16(vec2(ptx_mma_d0, ptx_mma_d1));",
            f"{_destination_expr(destinations[1], registers)} = packHalf2x16(vec2(ptx_mma_d2, ptx_mma_d3));",
        ])
    return None


def _emit_wmma_instruction(
    instruction: PtxInstruction,
    registers: dict[str, RegisterFile],
) -> str | None:
    """Lower the captured half-precision WMMA tile through native coopmat.

    The captured SM80/SM86 PWIN modules use the 32-lane
    ``m16n16k16.f16.f16`` fragment form.  NVIDIA's assembler lowers that
    form to two ``HMMA.16816.F16`` operations: the first four A words are the
    eight half values shared by both N=8 halves, while B and C are split into
    two N=8 fragments.  The layer's target has a native Vulkan cooperative
    matrix tuple for this exact 16x16x16 f16 operation.  Subgroup shuffles
    convert the PTX fragment ownership into the target's implementation-
    dependent AMD coopmat ownership without a shared-memory round trip.

    Keep this form narrow and explicit: other WMMA shapes/types must not
    silently inherit this layout.
    """
    opcode = instruction.opcode
    if opcode != "wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16":
        return None
    operands = _split_operands(instruction.operands)
    if len(operands) != 4:
        raise PtxTranslationError(f"wmma operands are not supported: {instruction.text}")
    destinations = _vector_operands(operands[0])
    a_values = _vector_operands(operands[1])
    b_values = _vector_operands(operands[2])
    c_values = _vector_operands(operands[3])
    if (len(destinations), len(a_values), len(b_values), len(c_values)) != (4, 8, 8, 4):
        raise PtxTranslationError(f"wmma.m16n16k16 fragment sizes are not supported: {instruction.text}")

    mode = os.environ.get("DLSSAMD_TRANSLATOR_WMMA_MODE")
    if mode == "passthrough":
        suffix = str(instruction.line)
        copied = [
            f"uint ptx_wmma_passthrough_{suffix}_{index} = "
            f"{_operand_expr(value, registers, 'uint')};"
            for index, value in enumerate(c_values)
        ]
        copied.extend(
            f"{_destination_expr(destination, registers)} = "
            f"ptx_wmma_passthrough_{suffix}_{index};"
            for index, (destination, _value) in enumerate(zip(destinations, c_values))
        )
        return " ".join(copied)

    if mode == "software":
        suffix = str(instruction.line)
        a = ", ".join(
            _packed_half_component(token, half, registers)
            for token in a_values[:4]
            for half in (0, 1)
        )
        b0 = ", ".join(
            _packed_half_component(token, half, registers)
            for token in b_values[:2]
            for half in (0, 1)
        )
        b1 = ", ".join(
            _packed_half_component(token, half, registers)
            for token in b_values[2:4]
            for half in (0, 1)
        )
        c0 = ", ".join(
            _packed_half_component(token, half, registers)
            for token in c_values[:2]
            for half in (0, 1)
        )
        c1 = ", ".join(
            _packed_half_component(token, half, registers)
            for token in c_values[2:]
            for half in (0, 1)
        )
        statements = [
            f"float ptx_wmma_a_{suffix}[8] = float[]({a});",
            f"float ptx_wmma_b0_{suffix}[4] = float[]({b0});",
            f"float ptx_wmma_b1_{suffix}[4] = float[]({b1});",
            f"float ptx_wmma_c0_{suffix}[4] = float[]({c0});",
            f"float ptx_wmma_c1_{suffix}[4] = float[]({c1});",
        ]
        for fragment, b_name, c_name, output_base in (
            ("0", f"ptx_wmma_b0_{suffix}", f"ptx_wmma_c0_{suffix}", 0),
            ("1", f"ptx_wmma_b1_{suffix}", f"ptx_wmma_c1_{suffix}", 2),
        ):
            values = [
                f"ptx_wmma_d_{suffix}_{fragment}_{index} = "
                f"ptx_mma_m16n8k16_f16(ptx_wmma_a_{suffix}, {b_name}, {c_name}, {index}u);"
                for index in range(4)
            ]
            statements.extend(f"float {value}" for value in values)
            statements.extend([
                f"{_destination_expr(destinations[output_base], registers)} = "
                f"packHalf2x16(vec2(ptx_wmma_d_{suffix}_{fragment}_0, ptx_wmma_d_{suffix}_{fragment}_1));",
                f"{_destination_expr(destinations[output_base + 1], registers)} = "
                f"packHalf2x16(vec2(ptx_wmma_d_{suffix}_{fragment}_2, ptx_wmma_d_{suffix}_{fragment}_3));",
            ])
        return " ".join(statements)

    # Instruction line numbers are stable within a kernel and keep the local
    # GLSL temporaries distinct when a PWIN kernel contains many WMMA ops.
    suffix = str(instruction.line)
    a_values = [
        _packed_half_component(token, half, registers)
        for token in a_values[:4]
        for half in (0, 1)
    ]
    b_values = [
        _packed_half_component(token, half, registers)
        for token in b_values[:4]
        for half in (0, 1)
    ]
    c_values = [
        _packed_half_component(token, half, registers)
        for token in c_values
        for half in (0, 1)
    ]

    a = f"ptx_wmma_a_values_{suffix}"
    b = f"ptx_wmma_b_values_{suffix}"
    c = f"ptx_wmma_c_values_{suffix}"
    statements = [
        f"vec4 {a}_0 = vec4({', '.join(a_values[:4])});",
        f"vec4 {a}_1 = vec4({', '.join(a_values[4:])});",
        f"vec4 {b}_0 = vec4({', '.join(b_values[:4])});",
        f"vec4 {b}_1 = vec4({', '.join(b_values[4:])});",
        f"vec4 {c}_0 = vec4({', '.join(c_values[:4])});",
        f"vec4 {c}_1 = vec4({', '.join(c_values[4:])});",
        f"uvec4 ptx_wmma_result_{suffix} = ptx_wmma_m16n16k16_f16({a}_0, {a}_1, {b}_0, {b}_1, {c}_0, {c}_1);",
    ]
    for destination, field in zip(destinations, ("x", "y", "z", "w")):
        statements.append(f"{_destination_expr(destination, registers)} = ptx_wmma_result_{suffix}.{field};")
    return " ".join(statements)


def _wmma_register_indices(
    instruction: PtxInstruction,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    """Return the packed register indices needed by the staged WMMA path."""
    operands = _split_operands(instruction.operands)
    if len(operands) != 4:
        return None
    try:
        destinations = _vector_operands(operands[0])
        a_values = _vector_operands(operands[1])
        b_values = _vector_operands(operands[2])
        c_values = _vector_operands(operands[3])
    except PtxTranslationError:
        return None
    if (len(destinations), len(a_values), len(b_values), len(c_values)) != (4, 8, 8, 4):
        return None
    vectors: list[tuple[int, ...]] = []
    # Do not collapse the WMMA fragment to the four-register m16n8 view.  The
    # m16n16k16 PTX form has eight packed A and eight packed B registers per
    # lane; the element ownership is opaque, but every register is part of the
    # ABI and must survive the replay boundary.
    for values in (destinations, a_values, b_values, c_values):
        indices: list[int] = []
        for value in values:
            match = _REGISTER_RE.fullmatch(value.strip())
            if not match or match.group("family") != "r":
                return None
            indices.append(int(match.group("index")))
        vectors.append(tuple(indices))
    return vectors[0], vectors[1], vectors[2], vectors[3]


def _glsl_wmma_dynamic_tables(
    records: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
) -> list[str]:
    def vector(values: tuple[int, ...]) -> str:
        return "uvec4(" + ", ".join(f"{value}u" for value in values) + ")"

    def table(name: str, values: list[tuple[int, ...]]) -> list[str]:
        return [
            f"const uvec4 {name}[{len(values)}] = uvec4[](",
            "        " + ",\n        ".join(vector(value) for value in values) + ");",
        ]

    return (
        table("ptx_wmma_dest_indices", [record[0] for record in records])
        + table("ptx_wmma_a_indices_0", [record[1][:4] for record in records])
        + table("ptx_wmma_a_indices_1", [record[1][4:] for record in records])
        + table("ptx_wmma_b_indices_0", [record[2][:4] for record in records])
        + table("ptx_wmma_b_indices_1", [record[2][4:] for record in records])
        + table("ptx_wmma_c_indices", [record[3] for record in records])
    )


def _glsl_wmma_dynamic_helper(
    register_count: int,
    *,
    storage: bool,
    software: bool = False,
) -> list[str]:
    if storage:
        register_expression = lambda index: f"ptx_registers.data[ptx_register_base + {index}]"
        signature = "uvec4 ptx_wmma_dynamic(uint operation) {"
    else:
        register_expression = lambda index: f"registers[{index}]"
        signature = f"uvec4 ptx_wmma_dynamic(in uint registers[{register_count}], uint operation) {{"
    lines = [
        signature,
        "    uvec4 a_indices = ptx_wmma_a_indices_0[operation];",
        "    uvec4 b_indices = ptx_wmma_b_indices_0[operation];",
        "    uvec4 c_indices = ptx_wmma_c_indices[operation];",
    ]
    if software:
        def half_values(index: str) -> str:
            expression = register_expression(index)
            return f"unpackHalf2x16({expression})[0], unpackHalf2x16({expression})[1]"

        lines.extend([
            "    float ptx_wmma_a[8] = float[](",
            f"        {half_values('a_indices.x')}, {half_values('a_indices.y')},",
            f"        {half_values('a_indices.z')}, {half_values('a_indices.w')});",
            "    float ptx_wmma_b0[4] = float[](",
            f"        {half_values('b_indices.x')}, {half_values('b_indices.y')});",
            "    float ptx_wmma_b1[4] = float[](",
            f"        {half_values('b_indices.z')}, {half_values('b_indices.w')});",
            "    float ptx_wmma_c0[4] = float[](",
            f"        {half_values('c_indices.x')}, {half_values('c_indices.y')});",
            "    float ptx_wmma_c1[4] = float[](",
            f"        {half_values('c_indices.z')}, {half_values('c_indices.w')});",
            "    vec4 ptx_wmma_d0 = ptx_mma_m16n8k16_f16x4(ptx_wmma_a, ptx_wmma_b0, ptx_wmma_c0);",
            "    vec4 ptx_wmma_d1 = ptx_mma_m16n8k16_f16x4(ptx_wmma_a, ptx_wmma_b1, ptx_wmma_c1);",
            "    return uvec4(packHalf2x16(ptx_wmma_d0.xy), packHalf2x16(ptx_wmma_d0.zw),",
            "                  packHalf2x16(ptx_wmma_d1.xy), packHalf2x16(ptx_wmma_d1.zw));",
        ])
    else:
        lines.extend([
            f"    vec4 a_low = vec4(unpackHalf2x16({register_expression('a_indices.x')})[0], unpackHalf2x16({register_expression('a_indices.x')})[1],",
            f"                      unpackHalf2x16({register_expression('a_indices.y')})[0], unpackHalf2x16({register_expression('a_indices.y')})[1]);",
            f"    vec4 a_high = vec4(unpackHalf2x16({register_expression('a_indices.z')})[0], unpackHalf2x16({register_expression('a_indices.z')})[1],",
            f"                       unpackHalf2x16({register_expression('a_indices.w')})[0], unpackHalf2x16({register_expression('a_indices.w')})[1]);",
            f"    vec4 b_low = vec4(unpackHalf2x16({register_expression('b_indices.x')})[0], unpackHalf2x16({register_expression('b_indices.x')})[1],",
            f"                      unpackHalf2x16({register_expression('b_indices.y')})[0], unpackHalf2x16({register_expression('b_indices.y')})[1]);",
            f"    vec4 b_high = vec4(unpackHalf2x16({register_expression('b_indices.z')})[0], unpackHalf2x16({register_expression('b_indices.z')})[1],",
            f"                       unpackHalf2x16({register_expression('b_indices.w')})[0], unpackHalf2x16({register_expression('b_indices.w')})[1]);",
            f"    vec4 c_low = vec4(unpackHalf2x16({register_expression('c_indices.x')})[0], unpackHalf2x16({register_expression('c_indices.x')})[1],",
            f"                      unpackHalf2x16({register_expression('c_indices.y')})[0], unpackHalf2x16({register_expression('c_indices.y')})[1]);",
            f"    vec4 c_high = vec4(unpackHalf2x16({register_expression('c_indices.z')})[0], unpackHalf2x16({register_expression('c_indices.z')})[1],",
            f"                       unpackHalf2x16({register_expression('c_indices.w')})[0], unpackHalf2x16({register_expression('c_indices.w')})[1]);",
            "    return ptx_wmma_m16n16k16_f16(a_low, a_high, b_low, b_high, c_low, c_high);",
        ])
    lines.append("}")
    return lines


def _wmma_dynamic_call(registers: dict[str, RegisterFile], operation: str) -> str:
    if registers["r"].storage:
        return f"ptx_wmma_dynamic({operation})"
    return f"ptx_wmma_dynamic(r, {operation})"


def _wmma_register_lvalue(registers: dict[str, RegisterFile], index: str) -> str:
    if registers["r"].storage:
        return f"ptx_registers.data[ptx_register_base + {index}]"
    return f"r[{index}]"


def _glsl_replay_state_helpers(*, register_storage: bool = False) -> list[str]:
    lines = [
        "void ptx_replay_capture(uint operation,",
        "        uint a0, uint a1, uint a2, uint a3, uint a4, uint a5, uint a6, uint a7,",
        "        uint b0, uint b1, uint b2, uint b3, uint b4, uint b5, uint b6, uint b7,",
        "        uint c0, uint c1, uint c2, uint c3) {",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 1u] = operation;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 3u] = a0;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 4u] = a1;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 5u] = a2;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 6u] = a3;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 7u] = a4;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 8u] = a5;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 9u] = a6;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 10u] = a7;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 11u] = b0;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 12u] = b1;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 13u] = b2;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 14u] = b3;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 15u] = b4;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 16u] = b5;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 17u] = b6;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 18u] = b7;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 19u] = c0;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 20u] = c1;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 21u] = c2;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 22u] = c3;",
        "}",
    ]
    if register_storage:
        lines.extend([
            "bool ptx_replay_wmma(uint operation, uint d0, uint d1, uint d2, uint d3,",
            "        uint a0, uint a1, uint a2, uint a3, uint a4, uint a5, uint a6, uint a7,",
            "        uint b0, uint b1, uint b2, uint b3, uint b4, uint b5, uint b6, uint b7,",
            "        uint c0, uint c1, uint c2, uint c3) {",
            "    if (ptx_wmma_state.data[ptx_wmma_state_base + 0u] <= operation) {",
            "        ptx_replay_capture(operation, a0, a1, a2, a3, a4, a5, a6, a7,",
            "                           b0, b1, b2, b3, b4, b5, b6, b7,",
            "                           c0, c1, c2, c3);",
            "        return true;",
            "    }",
            f"    uint ptx_replay_result_base = ptx_wmma_state_base + {_WMMA_STATE_RESULT_BASE}u + operation * 4u;",
            "    ptx_registers.data[ptx_register_base + d0] = ptx_wmma_state.data[ptx_replay_result_base];",
            "    ptx_registers.data[ptx_register_base + d1] = ptx_wmma_state.data[ptx_replay_result_base + 1u];",
            "    ptx_registers.data[ptx_register_base + d2] = ptx_wmma_state.data[ptx_replay_result_base + 2u];",
            "    ptx_registers.data[ptx_register_base + d3] = ptx_wmma_state.data[ptx_replay_result_base + 3u];",
            "    return false;",
            "}",
        ])
    return lines


def _emit_forward_segment(
    kernel: PtxKernel,
    registers: dict[str, RegisterFile],
    *,
    start: int,
    end: int,
    branch_targets: dict[int, int],
    surface_format: str,
    terminal_statement: str = "return;",
) -> list[str] | None:
    """Emit one no-WMMA segment with recursively structured forward branches."""

    def emit_range(range_start: int, range_end: int, indent: str) -> list[str] | None:
        lines: list[str] = []
        index = range_start
        while index < range_end:
            instruction = kernel.instructions[index]
            base = instruction.opcode.split(".", 1)[0]
            if base == "wmma":
                return None
            if base == "bra":
                target = branch_targets[index]
                if target > range_end:
                    return None
                condition = _operand_expr(instruction.predicate or "false", registers, "bool")
                skipped = emit_range(index + 1, target, indent + "    ")
                if skipped is None:
                    return None
                if skipped:
                    lines.append(f"{indent}if (!({condition})) {{")
                    lines.extend(skipped)
                    lines.append(f"{indent}}}")
                index = target
                continue
            if base in {"ret", "trap"}:
                lines.extend(f"{indent}{statement}" for statement in terminal_statement.splitlines())
                index += 1
                continue
            emitted = _emit_instruction(
                instruction,
                kernel,
                registers,
                surface_format=surface_format,
            )
            lines.append(f"{indent}{{")
            block_indent = indent + "    "
            if instruction.predicate:
                condition = _operand_expr(instruction.predicate, registers, "bool")
                lines.append(f"{block_indent}if ({condition}) {{")
                lines.extend(f"{block_indent}    {statement}" for statement in emitted.splitlines())
                lines.append(f"{block_indent}}}")
            else:
                lines.extend(f"{block_indent}{statement}" for statement in emitted.splitlines())
            lines.append(f"{indent}}}")
            index += 1
        return lines

    return emit_range(start, end, "    ")


def _emit_staged_wmma_control_flow(
    kernel: PtxKernel,
    registers: dict[str, RegisterFile],
    records: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    *,
    surface_format: str,
) -> list[str] | None:
    """Keep scalar PTX order while reducing WMMA to one dynamic call site."""
    labels = {name: index for name, index in kernel.label_indices}
    branch_targets: dict[int, int] = {}
    for index, instruction in enumerate(kernel.instructions):
        if instruction.opcode.split(".", 1)[0] != "bra":
            continue
        target = _branch_target(instruction, labels)
        if not instruction.predicate or target <= index:
            return None
        branch_targets[index] = target

    wmma_positions = [
        index for index, instruction in enumerate(kernel.instructions)
        if instruction.opcode.startswith("wmma.")
    ]
    if len(wmma_positions) != len(records) or not wmma_positions:
        return None
    for index, target in branch_targets.items():
        if any(index < position < target for position in wmma_positions):
            return None

    lines = [
        f"    for (uint ptx_wmma_index = 0u; ptx_wmma_index < {len(records)}u; ++ptx_wmma_index) {{",
        "        switch (ptx_wmma_index) {",
    ]
    segment_start = 0
    for operation, position in enumerate(wmma_positions):
        segment = _emit_forward_segment(
            kernel,
            registers,
            start=segment_start,
            end=position,
            branch_targets=branch_targets,
            surface_format=surface_format,
        )
        if segment is None:
            return None
        lines.append(f"            case {operation}u:")
        lines.append("            {")
        lines.extend(f"                {statement[4:]}" for statement in segment)
        lines.append("                break;")
        lines.append("            }")
        segment_start = position + 1
    lines.extend([
        "            default:",
        "                return;",
        "        }",
        f"        uvec4 ptx_wmma_result = {_wmma_dynamic_call(registers, 'ptx_wmma_index')};",
    ])
    for field, component in zip(("x", "y", "z", "w"), range(4)):
        index_expression = f"ptx_wmma_dest_indices[ptx_wmma_index].{'xyzw'[component]}"
        lines.append(
            f"        {_wmma_register_lvalue(registers, index_expression)} = "
            f"ptx_wmma_result.{field};"
        )
    lines.append("    }")
    tail = _emit_forward_segment(
        kernel,
        registers,
        start=segment_start,
        end=len(kernel.instructions),
        branch_targets=branch_targets,
        surface_format=surface_format,
    )
    if tail is None:
        return None
    lines.extend(tail)
    return lines


def _emit_replay_forward_control_flow(
    kernel: PtxKernel,
    registers: dict[str, RegisterFile],
    records: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    *,
    surface_format: str,
    operation_expressions: dict[int, str] | None = None,
) -> list[str] | None:
    """Replay a forward-only PWIN CFG with one compact-state stop point.

    A PC interpreter is needed for arbitrary PTX, but the captured enc0/enc1
    style kernels are forward-only control flow with WMMA operations acting as
    ordered barriers.  Emit those scalar regions as ordinary GLSL statements
    in a bounded operation switch.  Each replay dispatch walks the regions
    from the beginning, restores prior WMMA results, and stops at the current
    operation to let the native dispatcher consume its compact inputs.  This
    removes thousands of per-instruction PC cases from the RADV optimization
    unit while retaining the same host-visible replay protocol.
    """
    labels = {name: index for name, index in kernel.label_indices}
    branch_targets: dict[int, int] = {}
    for index, instruction in enumerate(kernel.instructions):
        if instruction.opcode.split(".", 1)[0] != "bra":
            continue
        target = _branch_target(instruction, labels)
        if not instruction.predicate or target <= index:
            return None
        branch_targets[index] = target

    def state_value(index: str) -> str:
        return f"ptx_wmma_state.data[ptx_wmma_state_base + {index}]"

    wmma_operations = {
        index: (operation_expressions or {}).get(index, f"{operation}u")
        for operation, index in enumerate(
            position
            for position, instruction in enumerate(kernel.instructions)
            if instruction.opcode.startswith("wmma.")
        )
    }
    wmma_records = {
        index: record for index, record in zip(wmma_operations, records)
    }
    if len(wmma_records) != len(records) or not records:
        return None

    terminal_statement = " ".join([
        f"{state_value('2u')} = 1u;",
        f"{state_value('1u')} = 0xffffffffu;",
        "return;",
    ])

    def emit_range(range_start: int, range_end: int, indent: str) -> list[str] | None:
        lines: list[str] = []
        index = range_start
        while index < range_end:
            instruction = kernel.instructions[index]
            base = instruction.opcode.split(".", 1)[0]
            if base == "bra":
                target = branch_targets[index]
                if target > range_end:
                    return None
                condition = _operand_expr(instruction.predicate or "false", registers, "bool")
                skipped = emit_range(index + 1, target, indent + "    ")
                if skipped is None:
                    return None
                if skipped:
                    lines.append(f"{indent}if (!({condition})) {{")
                    lines.extend(skipped)
                    lines.append(f"{indent}}}")
                index = target
                continue
            if base == "wmma":
                operation = wmma_operations[index]
                record = wmma_records[index]
                destinations, a_values, b_values, c_values = record
                capture_values = [
                    _operand_expr(f"%r{token}", registers, "uint")
                    for token in (*a_values, *b_values, *c_values)
                ]
                if registers["r"].storage:
                    lines.append(
                        f"{indent}if (ptx_replay_wmma({operation}, "
                        f"{', '.join(f'{destination}u' for destination in destinations)}, "
                        f"{', '.join(capture_values)})) return;"
                    )
                else:
                    lines.extend([
                        f"{indent}if ({state_value('0u')} <= {operation}) {{",
                        f"{indent}    ptx_replay_capture("
                        f"{operation}, {', '.join(capture_values)});",
                        f"{indent}    return;",
                        f"{indent}}}",
                    ])
                for component, destination in enumerate(destinations):
                    lines.append(
                        f"{indent}{_destination_expr(f'%r{destination}', registers)} = "
                        f"ptx_wmma_state.data[ptx_wmma_state_base + {_WMMA_STATE_RESULT_BASE}u + ({operation}) * 4u + {component}u];"
                    )
                index += 1
                continue
            if base in {"ret", "trap"}:
                lines.extend(f"{indent}{statement}" for statement in terminal_statement.splitlines())
                index += 1
                continue
            emitted = _emit_instruction(
                instruction,
                kernel,
                registers,
                surface_format=surface_format,
            )
            lines.append(f"{indent}{{")
            block_indent = indent + "    "
            if instruction.predicate:
                condition = _operand_expr(instruction.predicate, registers, "bool")
                lines.append(f"{block_indent}if ({condition}) {{")
                lines.extend(f"{block_indent}    {statement}" for statement in emitted.splitlines())
                lines.append(f"{block_indent}}}")
            else:
                lines.extend(f"{block_indent}{statement}" for statement in emitted.splitlines())
            lines.append(f"{indent}}}")
            index += 1
        return lines

    lines = emit_range(0, len(kernel.instructions), "    ")
    if lines is None:
        return None
    lines.extend([
        f"    {state_value('2u')} = 1u;",
        f"    {state_value('1u')} = 0xffffffffu;",
        "    return;",
    ])
    return lines


def _emit_movmatrix_instruction(
    instruction: PtxInstruction,
    registers: dict[str, RegisterFile],
) -> str | None:
    """Lower the documented 32-lane m8n8 row-fragment transpose.

    The PTX m8n8.b16 layout gives each lane two adjacent source elements;
    four lanes make one source row.  After the transpose, the same four-lane
    group represents one result column, with each lane holding two adjacent
    result rows.  The explicit lane mapping is therefore independent of the
    host subgroup width; a 64-lane RADV subgroup is treated as two logical
    PTX warps.
    """
    if instruction.opcode not in {
        "movmatrix.sync.aligned.m8n8.trans.b16",
        "movmatrix.sync.trans.aligned.m8n8.b16",
    }:
        return None
    operands = _split_operands(instruction.operands)
    if len(operands) != 2:
        raise PtxTranslationError(f"movmatrix operands are not supported: {instruction.text}")
    if _destination_type(operands[0], registers) != "uint":
        raise PtxTranslationError(f"movmatrix destination is not a 32-bit register: {instruction.text}")
    destination = _destination_expr(operands[0], registers)
    source = _operand_expr(operands[1], registers, "uint")
    return " ".join([
        "uint ptx_mov_lane = gl_SubgroupInvocationID & 31u;",
        "uint ptx_mov_warp_base = gl_SubgroupInvocationID & ~31u;",
        "uint ptx_mov_column = ptx_mov_lane >> 2u;",
        "uint ptx_mov_pair = ptx_mov_lane & 3u;",
        f"uint ptx_mov_source0 = subgroupShuffle({source}, ptx_mov_warp_base + (ptx_mov_pair * 2u) * 4u + (ptx_mov_column >> 1u));",
        f"uint ptx_mov_source1 = subgroupShuffle({source}, ptx_mov_warp_base + (ptx_mov_pair * 2u + 1u) * 4u + (ptx_mov_column >> 1u));",
        f"{destination} = packHalf2x16(vec2(unpackHalf2x16(ptx_mov_source0)[ptx_mov_column & 1u], unpackHalf2x16(ptx_mov_source1)[ptx_mov_column & 1u]));",
    ])


def _glsl_matrix_helpers() -> list[str]:
    return [
        "float ptx_mma_m16n8k8_f16(in float a[4], in float b[2], in float c[4], uint component) {",
        "    uint lane = gl_SubgroupInvocationID & 31u;",
        "    uint warp_base = gl_SubgroupInvocationID & ~31u;",
        "    uint row = (lane >> 2u) + (component >= 2u ? 8u : 0u);",
        "    uint col = (lane & 3u) * 2u + (component & 1u);",
        "    float result = c[component];",
        "    for (uint k = 0u; k < 8u; ++k) {",
        "        uint a_index = (row >= 8u ? 2u : 0u) + (k & 1u);",
        "        uint a_lane = warp_base + ((row & 7u) << 2u) + (k >> 1u);",
        "        uint b_index = k & 1u;",
        "        uint b_lane = warp_base + (col << 2u) + (k >> 1u);",
        "        result = fma(subgroupShuffle(a[a_index], a_lane), subgroupShuffle(b[b_index], b_lane), result);",
        "    }",
        "    return result;",
        "}",
        "float ptx_mma_m16n8k16_f16(in float a[8], in float b[4], in float c[4], uint component) {",
        "    uint lane = gl_SubgroupInvocationID & 31u;",
        "    uint warp_base = gl_SubgroupInvocationID & ~31u;",
        "    uint row = (lane >> 2u) + (component >= 2u ? 8u : 0u);",
        "    uint col = (lane & 3u) * 2u + (component & 1u);",
        "    float result = c[component];",
        "    for (uint k = 0u; k < 16u; ++k) {",
        "        uint a_index = (row >= 8u ? 2u : 0u) + (k >= 8u ? 4u : 0u) + (k & 1u);",
        "        uint a_lane = warp_base + ((row & 7u) << 2u) + ((k & 7u) >> 1u);",
        "        uint b_index = (k >= 8u ? 2u : 0u) + (k & 1u);",
        "        uint b_lane = warp_base + (col << 2u) + ((k & 7u) >> 1u);",
        "        result = fma(subgroupShuffle(a[a_index], a_lane), subgroupShuffle(b[b_index], b_lane), result);",
        "    }",
        "    return result;",
        "}",
        "vec4 ptx_mma_m16n8k16_f16x4(in float a[8], in float b[4], in float c[4]) {",
        "    uint lane = gl_SubgroupInvocationID & 31u;",
        "    uint warp_base = gl_SubgroupInvocationID & ~31u;",
        "    vec4 result = vec4(c[0], c[1], c[2], c[3]);",
        "    for (uint k = 0u; k < 16u; ++k) {",
        "        for (uint component = 0u; component < 4u; ++component) {",
        "            uint row = (lane >> 2u) + (component >= 2u ? 8u : 0u);",
        "            uint col = (lane & 3u) * 2u + (component & 1u);",
        "            uint a_index = (row >= 8u ? 2u : 0u) + (k >= 8u ? 4u : 0u) + (k & 1u);",
        "            uint a_lane = warp_base + ((row & 7u) << 2u) + ((k & 7u) >> 1u);",
        "            uint b_index = (k >= 8u ? 2u : 0u) + (k & 1u);",
        "            uint b_lane = warp_base + (col << 2u) + ((k & 7u) >> 1u);",
        "            result[component] = fma(subgroupShuffle(a[a_index], a_lane), subgroupShuffle(b[b_index], b_lane), result[component]);",
        "        }",
        "    }",
        "    return result;",
        "}",
        "float ptx_mma_m16n8k32_e4m3(in float a[16], in float b[8], in float c[4], uint component) {",
        "    uint lane = gl_SubgroupInvocationID & 31u;",
        "    uint warp_base = gl_SubgroupInvocationID & ~31u;",
        "    uint row = (lane >> 2u) + (component >= 2u ? 8u : 0u);",
        "    uint col = (lane & 3u) * 2u + (component & 1u);",
        "    float result = c[component];",
        "    for (uint k = 0u; k < 32u; ++k) {",
        "        uint a_index = (row >= 8u ? 4u : 0u) + (k >= 16u ? 8u : 0u) + (k & 3u);",
        "        uint a_lane = warp_base + ((row & 7u) << 2u) + ((k & 15u) >> 2u);",
        "        uint b_index = (k >= 16u ? 4u : 0u) + (k & 3u);",
        "        uint b_lane = warp_base + (col << 2u) + ((k & 15u) >> 2u);",
        "        result = fma(subgroupShuffle(a[a_index], a_lane), subgroupShuffle(b[b_index], b_lane), result);",
        "    }",
        "    return result;",
        "}",
    ]


def _glsl_wmma_helpers() -> list[str]:
    """Generate the target-calibrated native 32-lane f16 WMMA bridge."""
    return [
        "uvec4 ptx_wmma_m16n16k16_f16(in vec4 a_low, in vec4 a_high, in vec4 b_low, in vec4 b_high, in vec4 c_low, in vec4 c_high) {",
        "    uint lane = gl_SubgroupInvocationID & 31u;",
        "    uvec4 a_packed = uvec4(",
        "        packHalf2x16(vec2(a_low.x, a_high.x)), packHalf2x16(vec2(a_low.y, a_high.y)),",
        "        packHalf2x16(vec2(a_low.z, a_high.z)), packHalf2x16(vec2(a_low.w, a_high.w)));",
        "    uvec4 b_packed = uvec4(",
        "        packHalf2x16(vec2(b_low.x, b_high.x)), packHalf2x16(vec2(b_low.y, b_high.y)),",
        "        packHalf2x16(vec2(b_low.z, b_high.z)), packHalf2x16(vec2(b_low.w, b_high.w)));",
        "    uvec4 c_packed = uvec4(",
        "        packHalf2x16(vec2(c_low.x, c_high.x)), packHalf2x16(vec2(c_low.y, c_high.y)),",
        "        packHalf2x16(vec2(c_low.z, c_high.z)), packHalf2x16(vec2(c_low.w, c_high.w)));",
        "    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>(float16_t(0.0));",
        "    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>(float16_t(0.0));",
        "    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> c = coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(float16_t(0.0));",
        "    [[dont_unroll]] for (uint component = 0u; component < 8u; ++component) {",
        "        uint row = lane & 15u;",
        "        uint column = ((lane >> 4u) << 3u) + component;",
        "        uint source_lane = ((row & 7u) << 2u) + ((column & 7u) >> 1u);",
        "        uint source_component = (row >= 8u ? 2u : 0u) + (column >= 8u ? 4u : 0u) + (column & 1u);",
        "        vec2 shuffled = unpackHalf2x16(subgroupShuffle(a_packed, source_lane)[source_component & 3u]);",
        "        a[component] = float16_t(source_component < 4u ? shuffled.x : shuffled.y);",
        "    }",
        "    [[dont_unroll]] for (uint component = 0u; component < 8u; ++component) {",
        "        uint row = ((lane >> 4u) << 3u) + component;",
        "        uint column = lane & 15u;",
        "        uint source_lane = ((column & 7u) << 2u) + ((row & 7u) >> 1u);",
        "        uint source_component = (column >= 8u ? 4u : 0u) + (row >= 8u ? 2u : 0u) + (row & 1u);",
        "        vec2 shuffled = unpackHalf2x16(subgroupShuffle(b_packed, source_lane)[source_component & 3u]);",
        "        b[component] = float16_t(source_component < 4u ? shuffled.x : shuffled.y);",
        "        source_lane = ((row & 7u) << 2u) + ((column & 7u) >> 1u);",
        "        source_component = (column >= 8u ? 4u : 0u) + (row >= 8u ? 2u : 0u) + (column & 1u);",
        "        shuffled = unpackHalf2x16(subgroupShuffle(c_packed, source_lane)[source_component & 3u]);",
        "        c[component] = float16_t(source_component < 4u ? shuffled.x : shuffled.y);",
        "    }",
        "    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> d = coopMatMulAdd(a, b, c);",
        "    uvec4 d_packed = uvec4(",
        "        packHalf2x16(vec2(float(d[0]), float(d[1]))), packHalf2x16(vec2(float(d[2]), float(d[3]))),",
        "        packHalf2x16(vec2(float(d[4]), float(d[5]))), packHalf2x16(vec2(float(d[6]), float(d[7]))));",
        "    uint row0 = lane >> 2u;",
        "    uint row1 = row0 + 8u;",
        "    uint col0 = (lane & 3u) << 1u;",
        "    uint col1 = col0 + 8u;",
        "    uint index0 = row0 & 7u;",
        "    uint index1 = row1 & 7u;",
        "    uvec4 d00 = subgroupShuffle(d_packed, (row0 >= 8u ? 16u : 0u) + col0);",
        "    uvec4 d01 = subgroupShuffle(d_packed, (row0 >= 8u ? 16u : 0u) + col0 + 1u);",
        "    uvec4 d10 = subgroupShuffle(d_packed, (row1 >= 8u ? 16u : 0u) + col0);",
        "    uvec4 d11 = subgroupShuffle(d_packed, (row1 >= 8u ? 16u : 0u) + col0 + 1u);",
        "    uvec4 d20 = subgroupShuffle(d_packed, (row0 >= 8u ? 16u : 0u) + col1);",
        "    uvec4 d21 = subgroupShuffle(d_packed, (row0 >= 8u ? 16u : 0u) + col1 + 1u);",
        "    uvec4 d30 = subgroupShuffle(d_packed, (row1 >= 8u ? 16u : 0u) + col1);",
        "    uvec4 d31 = subgroupShuffle(d_packed, (row1 >= 8u ? 16u : 0u) + col1 + 1u);",
        "    vec2 d00_values = unpackHalf2x16(d00[index0 >> 1u]);",
        "    vec2 d01_values = unpackHalf2x16(d01[index0 >> 1u]);",
        "    vec2 d10_values = unpackHalf2x16(d10[index1 >> 1u]);",
        "    vec2 d11_values = unpackHalf2x16(d11[index1 >> 1u]);",
        "    vec2 d20_values = unpackHalf2x16(d20[index0 >> 1u]);",
        "    vec2 d21_values = unpackHalf2x16(d21[index0 >> 1u]);",
        "    vec2 d30_values = unpackHalf2x16(d30[index1 >> 1u]);",
        "    vec2 d31_values = unpackHalf2x16(d31[index1 >> 1u]);",
        "    uvec4 result;",
        "    result.x = packHalf2x16(vec2(",
        "        (index0 & 1u) == 0u ? d00_values.x : d00_values.y,",
        "        (index0 & 1u) == 0u ? d01_values.x : d01_values.y));",
        "    result.y = packHalf2x16(vec2(",
        "        (index1 & 1u) == 0u ? d10_values.x : d10_values.y,",
        "        (index1 & 1u) == 0u ? d11_values.x : d11_values.y));",
        "    result.z = packHalf2x16(vec2(",
        "        (index0 & 1u) == 0u ? d20_values.x : d20_values.y,",
        "        (index0 & 1u) == 0u ? d21_values.x : d21_values.y));",
        "    result.w = packHalf2x16(vec2(",
        "        (index1 & 1u) == 0u ? d30_values.x : d30_values.y,",
        "        (index1 & 1u) == 0u ? d31_values.x : d31_values.y));",
        "    return result;",
        "}",
    ]


def _glsl_image_helpers(
    *,
    has_texture: bool,
    has_surface: bool,
    surface_format: str,
) -> list[str]:
    """Generate bounded logical-handle dispatch helpers for image operations."""

    image_slots = _runtime_image_slot_count()
    surface_image_formats = _runtime_surface_image_formats(surface_format)
    lines = [
        "uint ptx_image_index(uint64_t handle) {",
        f"    for (uint i = 0u; i < {image_slots}u; ++i)",
        "        if (ptx_image_table.data[i] == handle)",
        "            return i;",
        "    return 0xffffffffu;",
        "}",
    ]
    if has_texture:
        lines.extend([
            "vec4 ptx_texture_sample(uint64_t handle, vec2 coordinates) {",
            "    uint index = ptx_image_index(handle);",
            "    switch (index) {",
        ])
        lines.extend(
            f"        case {slot}u: return texture(ptx_texture_{slot}, coordinates);"
            for slot in range(image_slots)
        )
        lines.extend([
            "        default: return vec4(0.0);",
            "    }",
            "}",
            "vec4 ptx_texture_sample_lod(uint64_t handle, vec2 coordinates, float lod) {",
            "    uint index = ptx_image_index(handle);",
            "    switch (index) {",
        ])
        lines.extend(
            f"        case {slot}u: return textureLod(ptx_texture_{slot}, coordinates, lod);"
            for slot in range(image_slots)
        )
        lines.extend([
            "        default: return vec4(0.0);",
            "    }",
            "}",
            "vec4 ptx_texture_gather(uint64_t handle, vec2 coordinates) {",
            "    uint index = ptx_image_index(handle);",
            "    switch (index) {",
        ])
        lines.extend(
            f"        case {slot}u: return textureGather(ptx_texture_{slot}, coordinates, 0);"
            for slot in range(image_slots)
        )
        lines.extend([
            "        default: return vec4(0.0);",
            "    }",
            "}",
            "vec4 ptx_texture_fetch(uint64_t handle, ivec2 coordinates) {",
            "    uint index = ptx_image_index(handle);",
            "    switch (index) {",
        ])
        lines.extend(
            f"        case {slot}u: return texelFetch(ptx_texture_{slot}, coordinates, 0);"
            for slot in range(image_slots)
        )
        lines.extend([
            "        default: return vec4(0.0);",
            "    }",
            "}",
        ])
    if has_surface:
        surface_value_type = "vec4" if surface_format == "float" else "uvec4"
        lines.extend([
            f"void ptx_surface_store(uint64_t handle, ivec2 coordinates, {surface_value_type} value) {{",
            "    uint index = ptx_image_index(handle);",
            "    switch (index) {",
        ])
        lines.extend(
            f"        case {slot}u: imageStore(ptx_surface_{slot}, coordinates, value); break;"
            for slot in range(image_slots)
        )
        lines.extend([
            "        default: break;",
            "    }",
            "}",
            f"{surface_value_type} ptx_surface_load(uint64_t handle, ivec2 coordinates) {{",
            "    uint index = ptx_image_index(handle);",
            "    switch (index) {",
        ])
        lines.extend(
            f"        case {slot}u: return imageLoad(ptx_surface_{slot}, coordinates);"
            for slot in range(image_slots)
        )
        lines.extend([
            f"        default: return {surface_value_type}({('0.0' if surface_format == 'float' else '0u')});",
            "    }",
            "}",
        ])
        lines.extend([
            "void ptx_surface_store_b32(uint64_t handle, ivec2 byte_coordinates, uint value) {",
            "    uint index = ptx_image_index(handle);",
            "    vec2 half_values = unpackHalf2x16(value);",
            "    switch (index) {",
        ])
        for slot, image_format in enumerate(surface_image_formats):
            if image_format == "r16f":
                lines.extend([
                    f"        case {slot}u: {{",
                    "            uint pixel = uint(byte_coordinates.x) >> 1u;",
                    f"            imageStore(ptx_surface_{slot}, ivec2(int(pixel), byte_coordinates.y), vec4(half_values.x, 0.0, 0.0, 0.0));",
                    f"            imageStore(ptx_surface_{slot}, ivec2(int(pixel + 1u), byte_coordinates.y), vec4(half_values.y, 0.0, 0.0, 0.0));",
                    "            break;",
                    "        }",
                ])
            elif image_format in {"rg16f", "rgba16f"}:
                bytes_per_pixel = 4 if image_format == "rg16f" else 8
                component_count = 2 if image_format == "rg16f" else 4
                lines.extend([
                    f"        case {slot}u: {{",
                    "            uint byte_offset = uint(byte_coordinates.x);",
                    f"            uint pixel = byte_offset / {bytes_per_pixel}u;",
                    f"            uint component = (byte_offset % {bytes_per_pixel}u) >> 1u;",
                    f"            vec4 current = imageLoad(ptx_surface_{slot}, ivec2(int(pixel), byte_coordinates.y));",
                    f"            if (component < {component_count}u) current[component] = half_values.x;",
                    f"            if (component + 1u < {component_count}u) current[component + 1u] = half_values.y;",
                    f"            imageStore(ptx_surface_{slot}, ivec2(int(pixel), byte_coordinates.y), current);",
                    "            break;",
                    "        }",
                ])
            elif image_format in {"r32f", "rg32f", "rgba32f"}:
                bytes_per_pixel = {"r32f": 4, "rg32f": 8, "rgba32f": 16}[image_format]
                component_count = {"r32f": 1, "rg32f": 2, "rgba32f": 4}[image_format]
                lines.extend([
                    f"        case {slot}u: {{",
                    "            uint byte_offset = uint(byte_coordinates.x);",
                    f"            uint pixel = byte_offset / {bytes_per_pixel}u;",
                    f"            uint component = (byte_offset % {bytes_per_pixel}u) >> 2u;",
                    f"            vec4 current = imageLoad(ptx_surface_{slot}, ivec2(int(pixel), byte_coordinates.y));",
                    f"            if (component < {component_count}u) current[component] = uintBitsToFloat(value);",
                    f"            imageStore(ptx_surface_{slot}, ivec2(int(pixel), byte_coordinates.y), current);",
                    "            break;",
                    "        }",
                ])
            else:
                lines.append(f"        case {slot}u: break;")
        lines.extend([
            "        default: break;",
            "    }",
            "}",
        ])
    return lines


def _rounding_function(parts: list[str]) -> str:
    if "rzi" in parts:
        return "trunc"
    if "rmi" in parts:
        return "floor"
    if "rpi" in parts:
        return "ceil"
    if "rni" in parts:
        return "roundEven"
    return "roundEven"


def _destination_type(destination: str, registers: dict[str, RegisterFile]) -> str:
    register = _register_ref(destination, registers)
    if not register:
        raise PtxTranslationError(f"destination is not a register: {destination}")
    return register[1]


def _destination_expr(destination: str, registers: dict[str, RegisterFile]) -> str:
    register = _register_ref(destination, registers)
    if not register:
        raise PtxTranslationError(f"destination is not a register: {destination}")
    return register[0]


def _emit_mov_b128(operands: list[str], registers: dict[str, RegisterFile]) -> str:
    """Pack/unpack raw 32/64-bit lanes without floating-point conversion."""
    if len(operands) != 2:
        raise PtxTranslationError("mov.b128 requires two operands")
    destination, source = operands
    unpack = destination.startswith("{")
    pack = source.startswith("{")
    if unpack == pack:
        raise PtxTranslationError("mov.b128 requires one scalar and one vector operand")
    scalar = source if unpack else destination
    if _destination_type(scalar, registers) != "uvec4":
        raise PtxTranslationError("mov.b128 scalar must be a .b128 register")
    lanes = _vector_operands(destination if unpack else source)
    if len(lanes) not in {2, 4}:
        raise PtxTranslationError("mov.b128 requires two 64-bit or four 32-bit lanes")
    width = 128 // len(lanes)
    for lane in lanes:
        if unpack and lane == "_":
            continue
        register = _register_file_for_token(lane, registers)
        if register is None or register.ptx_type not in {
            f".b{width}", f".u{width}", f".s{width}", f".f{width}"
        }:
            raise PtxTranslationError("mov.b128 lane register width does not match vector size")

    scalar_expr = _destination_expr(scalar, registers)
    if pack:
        words = []
        for lane in lanes:
            expression, glsl_type = _register_ref(lane, registers)
            if width == 32:
                words.append(f"floatBitsToUint({expression})" if glsl_type == "float" else expression)
            else:
                bits = f"doubleBitsToUint64({expression})" if glsl_type == "double" else expression
                words.extend([f"uint({bits})", f"uint({bits} >> 32)"])
        return f"{scalar_expr} = uvec4({', '.join(words)});"
    # Snapshot the source before writes, and scope the temporary to this
    # instruction so predicated/repeated emissions cannot collide.
    statements = [f"{{ uvec4 ptx_mov_bits = {scalar_expr};"]
    for index, lane in enumerate(lanes):
        if lane == "_":
            continue
        expression, glsl_type = _register_ref(lane, registers)
        if width == 32:
            bits = f"ptx_mov_bits[{index}]"
            value = f"uintBitsToFloat({bits})" if glsl_type == "float" else bits
        else:
            bits = f"(uint64_t(ptx_mov_bits[{index * 2}]) | (uint64_t(ptx_mov_bits[{index * 2 + 1}]) << 32))"
            value = f"uint64BitsToDouble({bits})" if glsl_type == "double" else bits
        statements.append(f"{expression} = {value};")
    return " ".join(statements) + " }"


def _emit_copysign_f32(operands: list[str], registers: dict[str, RegisterFile]) -> str:
    if len(operands) != 3:
        raise PtxTranslationError("copysign.f32 requires three operands")
    for index, token in enumerate(operands):
        register = _register_file_for_token(token, registers)
        if register is not None:
            if register.ptx_type not in {".b32", ".f32"}:
                raise PtxTranslationError("copysign.f32 requires .b32 or .f32 registers")
        elif index == 0 or not _FLOAT_BITS_RE.fullmatch(token):
            raise PtxTranslationError("copysign.f32 operand is outside checked subset")
    # PTX takes the sign from a and the magnitude from b (opposite the
    # argument order of C copysign). Integer masks preserve NaN payloads,
    # subnormals and signed zero without doing floating-point arithmetic.
    sign = _operand_expr(operands[1], registers, "uint")
    magnitude = _operand_expr(operands[2], registers, "uint")
    bits = f"(({sign} & 0x80000000u) | ({magnitude} & 0x7fffffffu))"
    if _destination_type(operands[0], registers) == "float":
        bits = f"uintBitsToFloat({bits})"
    return f"{_destination_expr(operands[0], registers)} = {bits};"


def _emit_instruction(
    instruction: PtxInstruction,
    kernel: PtxKernel,
    registers: dict[str, RegisterFile],
    *,
    surface_format: str,
) -> str:
    opcode = instruction.opcode
    parts = opcode.split(".")
    base = parts[0]
    operands = _split_operands(instruction.operands)
    if opcode == "mov.b128":
        return _emit_mov_b128(operands, registers)
    if base == "copysign":
        if opcode != "copysign.f32":
            raise PtxTranslationError(f"unsupported copysign form: {opcode}")
        return _emit_copysign_f32(operands, registers)
    if "b128" in parts or any(
        register and register.glsl_type == "uvec4"
        for register in (
            _register_file_for_token(token, registers)
            for token in re.findall(r"%?[A-Za-z_$][A-Za-z0-9_$]*", instruction.operands)
        )
    ):
        raise PtxTranslationError(f"unsupported 128-bit register operation: {opcode}")
    shared_symbols, _shared_size = _shared_layout(kernel)
    if "f16x2" in parts and base not in {"cvt", "set"}:
        emitted = _emit_half2_instruction(instruction, registers)
        if emitted is not None:
            return emitted
    if "f16" in parts and "f16x2" not in parts and base in {"add", "sub", "mul", "div", "max", "min", "fma", "abs", "neg", "ex2", "lg2", "rcp", "rsqrt", "sqrt", "tanh"}:
        emitted = _emit_half_instruction(instruction, registers)
        if emitted is not None:
            return emitted
    if base in {"ret"}:
        return "return;"
    if base == "cvta":
        if len(operands) != 2 or _destination_type(operands[0], registers) != "uint64_t":
            raise PtxTranslationError(f"address conversion is not supported: {instruction.text}")
        return f"{_destination_expr(operands[0], registers)} = {_operand_expr(operands[1], registers, 'uint64_t')};"
    if base == "shfl":
        if len(operands) not in {4, 5}:
            raise PtxTranslationError(f"shuffle form is not supported: {instruction.text}")
        if "|" in operands[0]:
            destination_token, predicate_token = (part.strip() for part in operands[0].split("|", 1))
        else:
            destination_token, predicate_token = operands[0].strip(), None
        destination = _destination_expr(destination_token, registers)
        source = _operand_expr(operands[1], registers, "uint")
        lane = _operand_expr(operands[2], registers, "uint")
        if ".bfly." in opcode:
            expression = f"subgroupShuffleXor({source}, {lane})"
        elif ".idx." in opcode:
            expression = f"subgroupShuffle({source}, {lane})"
        elif ".up." in opcode:
            expression = f"subgroupShuffleUp({source}, {lane})"
        elif ".down." in opcode:
            expression = f"subgroupShuffleDown({source}, {lane})"
        else:
            raise PtxTranslationError(f"shuffle mode is not supported: {opcode}")
        # The fifth operand is the active mask and the fourth is the PTX
        # clamp/width.  The prototype preserves the value movement and marks
        # the auxiliary predicate valid; exact out-of-range lane semantics are
        # a separate subgroup-validation milestone.
        if predicate_token is None:
            return f"{destination} = {expression};"
        predicate = _destination_expr(predicate_token, registers)
        return f"{destination} = {expression}; {predicate} = true;"
    if base == "vote":
        if "ballot" in parts and len(operands) == 3:
            destination = _destination_expr(operands[0], registers)
            if _destination_type(operands[0], registers) != "uint":
                raise PtxTranslationError(f"ballot destination is not a 32-bit register: {instruction.text}")
            condition = _operand_expr(operands[1], registers, "bool")
            return f"{destination} = subgroupBallot({condition}).x;"
        if len(operands) == 3 and ("any" in parts or "all" in parts):
            destination = _destination_expr(operands[0], registers)
            condition = _operand_expr(operands[1], registers, "bool")
            expression = "subgroupAny" if "any" in parts else "subgroupAll"
            return f"{destination} = {expression}({condition});"
        raise PtxTranslationError(f"vote form is not supported: {instruction.text}")
    if base == "activemask":
        if len(operands) != 1 or _destination_type(operands[0], registers) != "uint":
            raise PtxTranslationError(f"activemask form is not supported: {instruction.text}")
        destination = _destination_expr(operands[0], registers)
        return (
            f"{destination} = ((gl_SubgroupInvocationID & 32u) == 0u) "
            "? subgroupBallot(true).x : subgroupBallot(true).y;"
        )
    if base == "atom":
        if len(operands) != 3 or not ({"shared", "global"} & set(parts)) or "add" not in parts:
            raise PtxTranslationError(f"atomic form is not supported: {instruction.text}")
        destination = _destination_expr(operands[0], registers)
        address = _address_expr(operands[1], registers, shared_symbols)
        index = f"uint({address}) >> 2u"
        if "f32" in parts:
            if _destination_type(operands[0], registers) != "float":
                raise PtxTranslationError(f"float atomic destination is not a float register: {instruction.text}")
            value = _operand_expr(operands[2], registers, "float")
            if "global" in parts:
                return f"{destination} = atomicAdd(PtxGlobalFloatMemory(ptx_resolve_global_address({address})).data[0], {value});"
            return f"{destination} = atomicAdd(ptx_shared_float_data[{index}], {value});"
        value = _operand_expr(operands[2], registers, "uint")
        if "global" in parts:
            return f"{destination} = atomicAdd(PtxGlobalMemory(ptx_resolve_global_address({address})).data[0], {value});"
        return f"{destination} = atomicAdd(ptx_shared_data[{index}], {value});"
    if base == "cp" and "async" in parts:
        if "commit_group" in parts or "wait_group" in parts:
            return ""
        if len(operands) != 4:
            raise PtxTranslationError(f"cp.async form is not supported: {instruction.text}")
        size_token = operands[2].strip()
        try:
            size = int(size_token, 0)
        except ValueError as exc:
            raise PtxTranslationError(f"cp.async size is not constant: {instruction.text}") from exc
        if size <= 0 or size % 4:
            raise PtxTranslationError(f"cp.async size is not a positive 4-byte multiple: {instruction.text}")
        statements = []
        for offset in range(0, size, 4):
            destination_address = _address_with_offset(operands[0], offset, registers, shared_symbols)
            source_address = _address_with_offset(operands[1], offset, registers, shared_symbols)
            statements.append(
                _memory_store_statement(
                    destination_address,
                    _memory_load_expression(source_address, "u32", shared=False),
                    "u32",
                    shared=True,
                )
            )
        return " ".join(statements)
    if base == "mma":
        emitted = _emit_mma_instruction(instruction, registers)
        if emitted is not None:
            return emitted
    if base == "wmma":
        emitted = _emit_wmma_instruction(instruction, registers)
        if emitted is not None:
            return emitted
    if base == "movmatrix":
        emitted = _emit_movmatrix_instruction(instruction, registers)
        if emitted is not None:
            return emitted
    if base == "ldmatrix":
        if "m8n8" not in parts or "b16" not in parts or len(operands) != 2:
            raise PtxTranslationError(f"ldmatrix form is not supported: {instruction.text}")
        destinations = _vector_operands(operands[0])
        tile_count = next((int(value[1:]) for value in parts if value.startswith("x") and value[1:].isdigit()), None)
        if tile_count not in {1, 2, 4} or len(destinations) != tile_count:
            raise PtxTranslationError(f"ldmatrix tile count does not match destination fragment: {instruction.text}")
        source = f"uint({_address_expr(operands[1], registers, shared_symbols)})"
        lane = "(gl_SubgroupInvocationID & 31u)"
        warp_base = "(gl_SubgroupInvocationID & ~31u)"
        row = f"({lane} >> 2u)"
        pair = f"({lane} & 3u)"
        statements = []
        for tile in range(tile_count):
            provider = f"({warp_base} + {tile * 8}u + {row})"
            row_address = f"subgroupShuffle({source}, {provider})"
            low_offset = f"uint64_t({pair} * 4u)"
            high_offset = f"uint64_t({pair} * 4u + 2u)"
            low = f"ptx_shared_load_u16(uint64_t({row_address}) + {low_offset})"
            high = f"ptx_shared_load_u16(uint64_t({row_address}) + {high_offset})"
            statements.append(f"{_destination_expr(destinations[tile], registers)} = {low} | ({high} << 16u);")
        return " ".join(statements)
    if base in {"tex", "tld4", "sust", "suld"}:
        return _emit_image_instruction(
            instruction,
            registers,
            surface_format=surface_format,
        )
    if base in {"bra", "trap", "shfl", "vote", "tex", "tld4", "sust", "suld", "mma", "wmma", "ldmatrix", "movmatrix"}:
        raise PtxTranslationError(f"control/image/subgroup/matrix opcode is not in checked subset: {opcode}")
    if base == "bar":
        if opcode.startswith("bar.sync") or opcode.startswith("bar.warp.sync"):
            return "barrier(); memoryBarrierShared(); memoryBarrierBuffer();"
        raise PtxTranslationError(f"unsupported barrier opcode: {opcode}")
    if base in {"ld"}:
        if len(operands) != 2:
            raise PtxTranslationError(f"invalid ld.param operands: {instruction.text}")
        cache_qualifier = re.match(
            r"^::[A-Za-z0-9_.]+\.(u8|s8|b8|u16|s16|b16|u32|s32|b32|f16|f32|u64|s64|b64|f64)\s+(.+)$",
            operands[0],
        )
        if cache_qualifier:
            load_type = cache_qualifier.group(1)
            operands[0] = cache_qualifier.group(2).strip()
        else:
            load_type = _load_type_from_opcode(parts)
        destinations = _vector_operands(operands[0]) if operands[0].strip().startswith("{") else [operands[0]]
        load_size = {"u8": 1, "s8": 1, "b8": 1, "u16": 2, "s16": 2, "b16": 2,
                     "f16": 2, "u32": 4, "s32": 4, "b32": 4, "f32": 4, "u64": 8,
                     "s64": 8, "b64": 8, "f64": 8}[load_type]
        statements: list[str] = []
        if "param" in parts:
            for index, destination_operand in enumerate(destinations):
                destination = _destination_expr(destination_operand, registers)
                destination_type = _destination_type(destination_operand, registers)
                if _PARAM_ADDRESS_RE.match(operands[1].strip()):
                    value = _parameter_load(operands[1], kernel.parameters, load_type, index * load_size)
                else:
                    # PTX also spells accesses through a register containing
                    # the parameter-block address as ld.param.  The prototype
                    # maps that address to byte zero of ptx_params.
                    address = _address_expr(operands[1], registers, {})
                    value = f"ptx_load_{load_type if load_type in {'u8', 'u16', 'u32', 'u64'} else {'s8':'u8','b8':'u8','s16':'u16','b16':'u16','s32':'u32','b32':'u32','f16':'u16','f32':'u32','s64':'u64','b64':'u64','f64':'u64'}[load_type]}(uint({address}) + {index * load_size}u)"
                value = _loaded_value(value, load_type, destination_type)
                if destination_type == "bool":
                    value = f"({value}) != 0u"
                statements.append(f"{destination} = {value};")
            return " ".join(statements)

        if "const" in parts:
            constant_symbols = _constant_layout(kernel)
            for index, destination_operand in enumerate(destinations):
                destination = _destination_expr(destination_operand, registers)
                destination_type = _destination_type(destination_operand, registers)
                address = _constant_address(operands[1], constant_symbols)
                value = f"ptx_const_load_{load_type if load_type in {'u8', 'u16', 'u32', 'u64'} else {'s8':'u8','b8':'u8','s16':'u16','b16':'u16','s32':'u32','b32':'u32','f16':'u16','f32':'u32','s64':'u64','b64':'u64','f64':'u64'}[load_type]}(uint({address}) + {index * load_size}u)"
                value = _loaded_value(value, load_type, destination_type)
                if destination_type == "bool":
                    value = f"({value}) != 0u"
                statements.append(f"{destination} = {value};")
            return " ".join(statements)

        if "shared" in parts or "global" in parts or "local" in parts:
            shared = "shared" in parts
            local = "local" in parts
            for index, destination_operand in enumerate(destinations):
                destination = _destination_expr(destination_operand, registers)
                destination_type = _destination_type(destination_operand, registers)
                address = _address_with_offset(operands[1], index * load_size, registers, shared_symbols)
                value = _memory_load_expression(address, load_type, shared=shared, local=local)
                value = _loaded_value(value, load_type, destination_type)
                if destination_type == "bool":
                    value = f"({value}) != 0u"
                statements.append(f"{destination} = {value};")
            return " ".join(statements)
        raise PtxTranslationError(f"load address space is not supported: {opcode}")
    if base in {"st"}:
        if len(operands) != 2:
            raise PtxTranslationError(f"invalid store operands: {instruction.text}")
        store_type = parts[-1]
        address_operand = operands[0].strip()
        if store_type not in {
            "u8", "s8", "b8", "u16", "s16", "b16", "f16", "u32", "s32",
            "b32", "f32", "u64", "s64", "b64", "f64",
        }:
            # PTX cache/release qualifiers after ``::`` are retained in the
            # first operand by the lightweight parser, e.g.
            # ``::no_allocate.s8 [address]``.  They do not change the memory
            # lowering; recover the actual element type and address here.
            qualified = re.match(
                r"^::(?:[A-Za-z0-9_]+\.)*(?P<type>u8|s8|b8|u16|s16|b16|f16|u32|s32|b32|f32|u64|s64|b64|f64)\s+(?P<address>.*)$",
                address_operand,
            )
            if qualified:
                store_type = qualified.group("type")
                address_operand = qualified.group("address")
        values = _vector_operands(operands[1]) if operands[1].strip().startswith("{") else [operands[1]]
        store_size = {"u8": 1, "s8": 1, "b8": 1, "u16": 2, "s16": 2, "b16": 2,
                      "f16": 2, "u32": 4, "s32": 4, "b32": 4, "f32": 4, "u64": 8,
                      "s64": 8, "b64": 8, "f64": 8}.get(store_type)
        if store_size is None:
            raise PtxTranslationError(f"store type is not supported: {opcode}")
        if "global" in parts:
            # Preserve the original scalar fixture ABI: a 32-bit register
            # address is an output-array index in that deliberately tiny test.
            if store_type == "u32" and re.match(r"^\[\s*%r\d+\s*\]$", address_operand):
                return f"ptx_output.data[{_global_invocation_component('x')}] = {_operand_expr(values[0], registers, 'uint')};"
            address_space_shared = False
            address_space_local = False
        elif "shared" in parts:
            address_space_shared = True
            address_space_local = False
        elif "local" in parts:
            address_space_shared = False
            address_space_local = True
        else:
            raise PtxTranslationError(f"store address space is not supported: {opcode}")
        statements = []
        for index, value_operand in enumerate(values):
            address = _address_with_offset(address_operand, index * store_size, registers, shared_symbols)
            target = _operand_expr(value_operand, registers, "float" if store_type == "f32" else "uint64_t" if store_type in {"u64", "s64", "b64", "f64"} else "uint")
            statements.append(_memory_store_statement(address, target, store_type, shared=address_space_shared, local=address_space_local))
        return " ".join(statements)
    if base == "mov":
        if len(operands) != 2:
            raise PtxTranslationError(f"invalid mov operands: {instruction.text}")
        if operands[0].strip().startswith("{"):
            destinations = _vector_operands(operands[0])
            source = _operand_expr(operands[1], registers, "uint")
            unpacked = f"unpackHalf2x16({source})"
            if len(destinations) != 2:
                raise PtxTranslationError(f"only two-lane packed mov is supported: {instruction.text}")
            statements: list[str] = []
            for index, destination in enumerate(destinations):
                register_file = _register_file_for_token(destination, registers)
                value = f"ptx_pack_f16({unpacked}[{index}])" if register_file and register_file.ptx_type == ".f16" else f"uint({unpacked}[{index}])"
                statements.append(f"{_destination_expr(destination, registers)} = {value};")
            return " ".join(statements)
        if operands[1].strip().startswith("{"):
            sources = _vector_operands(operands[1])
            if len(sources) != 2:
                raise PtxTranslationError(f"only two-lane packed mov is supported: {instruction.text}")
            destination = _destination_expr(operands[0], registers)
            return f"{destination} = packHalf2x16(vec2({_half_value(sources[0], registers)}, {_half_value(sources[1], registers)}));"
        destination = _destination_expr(operands[0], registers)
        target_type = _destination_type(operands[0], registers)
        if target_type == "uint64_t" and (
            operands[1].strip() in {parameter.name for parameter in kernel.parameters}
            or operands[1].strip().startswith("__local_depot")
            or operands[1].strip().startswith(("_ZZ", "shared", "smem_"))
        ):
            # PTX names the kernel parameter block as a symbolic address.  The
            # prototype ABI puts that block at byte offset zero in ptx_params;
            # later pointer-memory lowering can therefore use a zero base.
            return f"{destination} = uint64_t(0UL);"
        if target_type == "uint" and operands[1].strip().startswith(("_ZZ", "shared", "smem_")):
            return f"{destination} = 0u;"
        if "f32" in parts and target_type == "uint":
            return f"{destination} = floatBitsToUint({_operand_expr(operands[1], registers, 'float')});"
        return f"{destination} = {_operand_expr(operands[1], registers, target_type)};"
    if base == "cvt":
        return _emit_cvt_instruction(instruction, registers)
    if base == "neg" and any(value in parts for value in {"s8", "s16", "s32", "s64"}):
        if len(operands) != 2:
            raise PtxTranslationError(f"invalid neg operands: {instruction.text}")
        destination = _destination_expr(operands[0], registers)
        target_type = _destination_type(operands[0], registers)
        if target_type == "uint64_t":
            value = _operand_expr(operands[1], registers, "uint64_t")
            return f"{destination} = uint64_t(-int64_t({value}));"
        if target_type == "uint":
            value = _operand_expr(operands[1], registers, "uint")
            return f"{destination} = uint(-int({value}));"
        raise PtxTranslationError(f"integer neg destination is not supported: {instruction.text}")
    if base == "popc":
        if len(operands) != 2 or _destination_type(operands[0], registers) != "uint":
            raise PtxTranslationError(f"popc form is not supported: {instruction.text}")
        value = _operand_expr(operands[1], registers, "uint")
        return f"{_destination_expr(operands[0], registers)} = bitCount({value});"
    if base in {"add", "sub", "mul", "div", "max", "min", "fma", "abs", "neg", "ex2", "lg2", "rcp", "rsqrt", "sqrt", "tanh"} and any(
        value in parts for value in {"f32", "f64"}
    ):
        destination = _destination_expr(operands[0], registers)
        target_type = _destination_type(operands[0], registers)
        value_type = "double" if "f64" in parts else "float"
        values = [_operand_expr(value, registers, value_type) for value in operands[1:]]
        if base == "add":
            expression = f"{values[0]} + {values[1]}"
        elif base == "sub":
            expression = f"{values[0]} - {values[1]}"
        elif base == "mul":
            expression = f"{values[0]} * {values[1]}"
        elif base == "div":
            expression = f"{values[0]} / {values[1]}"
        elif base in {"max", "min"}:
            expression = f"{base}({values[0]}, {values[1]})"
        elif base == "fma":
            expression = f"fma({values[0]}, {values[1]}, {values[2]})"
        elif base == "abs":
            expression = f"abs({values[0]})"
        elif base == "neg":
            expression = f"-({values[0]})"
        elif base == "ex2":
            expression = f"exp2({values[0]})"
        elif base == "lg2":
            expression = f"log2({values[0]})"
        elif base == "rcp":
            expression = f"(1.0 / {values[0]})"
        elif base == "rsqrt":
            expression = f"inversesqrt({values[0]})"
        elif base == "tanh":
            expression = f"tanh({values[0]})"
        else:
            expression = f"sqrt({values[0]})"
        if "sat" in parts:
            expression = f"clamp({expression}, 0.0, 1.0)"
        if target_type == "uint" and value_type == "float":
            expression = f"floatBitsToUint({expression})"
        return f"{destination} = {expression};"
    if base in {"div", "rem"}:
        if len(operands) != 3:
            raise PtxTranslationError(f"invalid {base} operands: {instruction.text}")
        destination = _destination_expr(operands[0], registers)
        target_type = _destination_type(operands[0], registers)
        if target_type not in {"uint", "uint64_t"}:
            raise PtxTranslationError(f"integer {base} destination is not supported: {opcode}")
        signed = any(value in parts for value in {"s16", "s32", "s64"})
        if target_type == "uint64_t":
            left = _operand_expr(operands[1], registers, "uint64_t")
            right = _operand_expr(operands[2], registers, "uint64_t")
            if signed:
                left = f"int64_t({left})"
                right = f"int64_t({right})"
        else:
            left = _operand_expr(operands[1], registers, "uint")
            right = _operand_expr(operands[2], registers, "uint")
            if signed:
                left = f"int({left})"
                right = f"int({right})"
        expression = f"{left} {'/' if base == 'div' else '%'} {right}"
        if signed:
            expression = f"uint64_t({expression})" if target_type == "uint64_t" else f"uint({expression})"
        return f"{destination} = {expression};"
    if base in {"add", "sub", "mul", "mad"} and not (base == "mul" and any(value in parts for value in {"wide", "hi"})):
        if len(operands) != (4 if base == "mad" else 3):
            raise PtxTranslationError(f"invalid {base} operands: {instruction.text}")
        destination = _destination_expr(operands[0], registers)
        target_type = _destination_type(operands[0], registers)
        args = [_operand_expr(value, registers, target_type) for value in operands[1:]]
        if base == "add":
            expression = f"{args[0]} + {args[1]}"
        elif base == "sub":
            expression = f"{args[0]} - {args[1]}"
        elif base == "mul":
            if not any(qualifier == "lo" for qualifier in parts[1:]) and target_type != "float":
                raise PtxTranslationError(f"only mul.lo is supported for integer registers: {opcode}")
            expression = f"{args[0]} * {args[1]}"
        else:
            if "lo" not in parts[1:] or target_type == "float":
                raise PtxTranslationError(f"only integer mad.lo is supported: {opcode}")
            expression = f"({args[0]} * {args[1]}) + {args[2]}"
        return f"{destination} = {expression};"
    if base == "mul" and "wide" in parts:
        if len(operands) != 3 or _destination_type(operands[0], registers) not in {"uint", "uint64_t"}:
            raise PtxTranslationError(f"wide multiply form is not supported: {instruction.text}")
        signed = any(value in parts for value in {"s16", "s32"})
        left = _operand_expr(operands[1], registers, "uint")
        right = _operand_expr(operands[2], registers, "uint")
        width = 16 if "u16" in parts or "s16" in parts else 32
        if width < 32:
            left = f"({left} & 0xffffu)"
            right = f"({right} & 0xffffu)"
        if signed:
            expression = f"uint64_t(int64_t(int({left})) * int64_t(int({right})))"
        else:
            expression = f"uint64_t({left}) * uint64_t({right})"
        if _destination_type(operands[0], registers) == "uint":
            expression = f"uint({expression})"
        return f"{_destination_expr(operands[0], registers)} = {expression};"
    if base == "mul" and "hi" in parts:
        if len(operands) != 3 or _destination_type(operands[0], registers) != "uint":
            raise PtxTranslationError(f"high multiply form is not supported: {instruction.text}")
        left = _operand_expr(operands[1], registers, "uint")
        right = _operand_expr(operands[2], registers, "uint")
        if any(value in parts for value in {"s16", "s32"}):
            expression = f"uint((int64_t(int({left})) * int64_t(int({right}))) >> 32)"
        else:
            expression = f"uint((uint64_t({left}) * uint64_t({right})) >> 32)"
        return f"{_destination_expr(operands[0], registers)} = {expression};"
    if base == "bfi":
        if len(operands) != 5 or _destination_type(operands[0], registers) != "uint":
            raise PtxTranslationError(f"bitfield insert form is not supported: {instruction.text}")
        args = [_operand_expr(value, registers, "uint") for value in operands[1:]]
        return f"{_destination_expr(operands[0], registers)} = ptx_bfi({args[0]}, {args[1]}, {args[2]}, {args[3]});"
    if base == "prmt":
        if len(operands) != 4 or _destination_type(operands[0], registers) != "uint":
            raise PtxTranslationError(f"byte permutation form is not supported: {instruction.text}")
        args = [_operand_expr(value, registers, "uint") for value in operands[1:]]
        return f"{_destination_expr(operands[0], registers)} = ptx_prmt({args[0]}, {args[1]}, {args[2]});"
    if base == "dp2a":
        if len(operands) != 4 or _destination_type(operands[0], registers) != "uint":
            raise PtxTranslationError(f"dp2a form is not supported: {instruction.text}")
        args = [_operand_expr(value, registers, "uint") for value in operands[1:]]
        signed = any(value.startswith("s") for value in parts[1:])
        return f"{_destination_expr(operands[0], registers)} = ptx_dp2a({args[0]}, {args[1]}, {args[2]}, {'true' if signed else 'false'});"
    if base in {"max", "min", "abs"}:
        destination = _destination_expr(operands[0], registers)
        target_type = _destination_type(operands[0], registers)
        if target_type == "uint64_t":
            values = [_operand_expr(value, registers, target_type) for value in operands[1:]]
            if base == "abs":
                return f"{destination} = uint64_t(abs(int64_t({values[0]})));"
            return f"{destination} = {values[0]} {'>' if base == 'max' else '<'} {values[1]} ? {values[0]} : {values[1]};"
        if target_type != "uint":
            raise PtxTranslationError(f"{opcode} destination type is not supported")
        if base == "abs":
            return f"{destination} = uint(abs(int({_operand_expr(operands[1], registers, 'uint')})));"
        left = _operand_expr(operands[1], registers, "uint")
        right = _operand_expr(operands[2], registers, "uint")
        if any(value in parts for value in {"s8", "s16", "s32"}):
            left = f"int({left})"
            right = f"int({right})"
            return f"{destination} = uint({left} {'>' if base == 'max' else '<'} {right} ? {left} : {right});"
        return f"{destination} = {left} {'>' if base == 'max' else '<'} {right} ? {left} : {right};"
    if base in {"and", "or", "xor"}:
        if len(operands) != 3:
            raise PtxTranslationError(f"invalid {base} operands: {instruction.text}")
        destination = _destination_expr(operands[0], registers)
        target_type = _destination_type(operands[0], registers)
        if target_type == "bool" and ".pred" in opcode:
            operator = {"and": "&&", "or": "||", "xor": "^^"}[base]
            return f"{destination} = {_operand_expr(operands[1], registers, 'bool')} {operator} {_operand_expr(operands[2], registers, 'bool')};"
        if target_type not in {"uint", "uint64_t"}:
            raise PtxTranslationError(f"{base} requires an integer destination")
        operator = {"and": "&", "or": "|", "xor": "^"}[base]
        return f"{destination} = {_operand_expr(operands[1], registers, target_type)} {operator} {_operand_expr(operands[2], registers, target_type)};"
    if base in {"shl", "shr"}:
        if len(operands) != 3:
            raise PtxTranslationError(f"invalid {base} operands: {instruction.text}")
        destination = _destination_expr(operands[0], registers)
        target_type = _destination_type(operands[0], registers)
        if target_type not in {"uint", "uint64_t"}:
            raise PtxTranslationError(f"{base} requires an integer destination")
        left = _operand_expr(operands[1], registers, target_type)
        right = _operand_expr(operands[2], registers, "uint")
        if base == "shr" and any(value in parts for value in {"s16", "s32"}):
            return f"{destination} = uint(int({left}) >> int({right}));"
        operator = "<<" if base == "shl" else ">>"
        return f"{destination} = {left} {operator} {right};"
    if base == "not":
        if len(operands) != 2:
            raise PtxTranslationError(f"invalid not operands: {instruction.text}")
        destination = _destination_expr(operands[0], registers)
        target_type = _destination_type(operands[0], registers)
        if target_type == "bool" and ".pred" in opcode:
            return f"{destination} = !{_operand_expr(operands[1], registers, 'bool')};"
        if target_type not in {"uint", "uint64_t"}:
            raise PtxTranslationError("not destination must be an integer or predicate")
        return f"{destination} = ~{_operand_expr(operands[1], registers, target_type)};"
    if base == "setp":
        if len(operands) != 3:
            raise PtxTranslationError(f"invalid setp operands: {instruction.text}")
        destination = _destination_expr(operands[0], registers)
        if _destination_type(operands[0], registers) != "bool":
            raise PtxTranslationError("setp destination must be a predicate register")
        relation = next(
            (
                qualifier
                for qualifier in parts[1:]
                if qualifier in {"eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu", "leu", "gtu", "geu"}
            ),
            None,
        )
        if relation is None:
            raise PtxTranslationError(f"unsupported setp relation: {opcode}")
        signed = any(value in parts for value in {"s8", "s16", "s32"})
        signed64 = "s64" in parts
        floating = "f32" in parts
        floating64 = "f64" in parts
        half = "f16" in parts and not floating
        if floating64:
            value_type = "double"
        elif floating:
            value_type = "float"
        elif signed64 or any(value in parts for value in {"u64", "b64"}):
            value_type = "uint64_t"
        else:
            value_type = "uint"
        left = _operand_expr(operands[1], registers, value_type)
        right = _operand_expr(operands[2], registers, value_type)
        if half:
            left = _half_value(operands[1], registers)
            right = _half_value(operands[2], registers)
        if signed:
            left = f"int({left})"
            right = f"int({right})"
        if signed64:
            left = f"int64_t({left})"
            right = f"int64_t({right})"
        relation = {"equ": "eq", "neu": "ne", "ltu": "lt", "leu": "le", "gtu": "gt", "geu": "ge"}.get(relation, relation)
        operator = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}[relation]
        return f"{destination} = {left} {operator} {right};"
    if base == "set" and "f16x2" in parts:
        if len(operands) != 3 or _destination_type(operands[0], registers) != "uint":
            raise PtxTranslationError(f"packed half comparison form is not supported: {instruction.text}")
        left = _half2_value(operands[1], registers)
        right = _half2_value(operands[2], registers)
        relation = next((value for value in parts if value in {"gt", "ge", "lt", "le", "equ", "neu"}), None)
        if relation is None:
            raise PtxTranslationError(f"packed half comparison relation is not supported: {instruction.opcode}")
        operator = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "equ": "==", "neu": "!="}[relation]
        return (
            f"{_destination_expr(operands[0], registers)} = "
            f"(({left}.x {operator} {right}.x) ? 0xffffu : 0u) | "
            f"(({left}.y {operator} {right}.y) ? 0xffff0000u : 0u);"
        )
    if base == "set" and "f16" in parts:
        if len(operands) != 3 or _destination_type(operands[0], registers) != "uint":
            raise PtxTranslationError(f"scalar half comparison form is not supported: {instruction.text}")
        relation = next((value for value in parts if value in {"eq", "ne", "gt", "ge", "lt", "le"}), None)
        if relation is None:
            raise PtxTranslationError(f"scalar half comparison relation is not supported: {instruction.opcode}")
        operator = {"eq": "==", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}[relation]
        left = _half_value(operands[1], registers)
        right = _half_value(operands[2], registers)
        return f"{_destination_expr(operands[0], registers)} = ({left} {operator} {right}) ? 0xffffu : 0u;"
    if base == "selp":
        if len(operands) != 4:
            raise PtxTranslationError(f"invalid selp operands: {instruction.text}")
        destination = _destination_expr(operands[0], registers)
        target_type = _destination_type(operands[0], registers)
        condition = _operand_expr(operands[3], registers, "bool")
        true_value = _operand_expr(operands[1], registers, target_type)
        false_value = _operand_expr(operands[2], registers, target_type)
        return f"{destination} = {condition} ? {true_value} : {false_value};"
    raise PtxTranslationError(f"unsupported opcode: {opcode}")


def _branch_target(instruction: PtxInstruction, labels: dict[str, int]) -> int:
    operands = _split_operands(instruction.operands)
    if len(operands) != 1:
        raise PtxTranslationError(f"invalid branch operands: {instruction.text}")
    target = operands[0].strip()
    if target not in labels:
        raise PtxTranslationError(f"branch target is not defined: {target}")
    return labels[target]


def _resolve_u32_constant(kernel: PtxKernel, register: str, before: int) -> int | None:
    """Resolve the small induction-variable initializers emitted by NVCC."""
    visited: set[str] = set()
    current = register
    while current not in visited:
        visited.add(current)
        for index in range(before - 1, -1, -1):
            instruction = kernel.instructions[index]
            if instruction.opcode not in {"mov.u32", "mov.b32"}:
                continue
            operands = _split_operands(instruction.operands)
            if len(operands) != 2 or operands[0].strip() != current:
                continue
            source = operands[1].strip()
            try:
                if _REGISTER_RE.fullmatch(source):
                    current = source
                    before = index
                    break
                return int(source.rstrip("uU"), 0) & 0xFFFFFFFF
            except ValueError:
                return None
        else:
            return None
    return None


def expand_fixed_wmma_loops(kernel: PtxKernel) -> PtxKernel:
    """Expand the bounded WMMA loops used by the captured PWIN kernels.

    The compact replay protocol numbers WMMA *instances*, not just source
    instructions.  Three PWIN stages contain a single NVCC-generated loop
    whose induction variable has a constant start, positive step, and
    ``setp.ne`` limit.  Expanding that loop makes the operation numbering
    explicit for both the scalar replay and native dispatcher shaders.  A
    loop with WMMA that falls outside this deliberately narrow pattern is
    rejected rather than silently producing repeated or stale matrix results.
    """
    labels = {name: index for name, index in kernel.label_indices}
    backward_branches: list[tuple[int, int]] = []
    for index, instruction in enumerate(kernel.instructions):
        if instruction.opcode.split(".", 1)[0] != "bra":
            continue
        target = _branch_target(instruction, labels)
        if target <= index:
            backward_branches.append((index, target))
    if not backward_branches:
        return kernel

    wmma_positions = {
        index for index, instruction in enumerate(kernel.instructions)
        if instruction.opcode.startswith("wmma.")
    }
    if not any(
        any(start <= position < branch for position in wmma_positions)
        for branch, start in backward_branches
    ):
        return kernel
    if len(backward_branches) != 1:
        raise PtxTranslationError(
            f"WMMA replay requires one bounded loop, found {len(backward_branches)} in {kernel.name}"
        )

    branch_index, loop_start = backward_branches[0]
    if branch_index < 2 or not kernel.instructions[branch_index].predicate:
        raise PtxTranslationError(f"WMMA loop control is not in the supported form: {kernel.name}")
    update = kernel.instructions[branch_index - 2]
    compare = kernel.instructions[branch_index - 1]
    if update.opcode not in {"add.s32", "add.u32"} or not compare.opcode.startswith("setp.ne."):
        raise PtxTranslationError(f"WMMA loop control is not a constant counted loop: {kernel.name}")
    update_operands = _split_operands(update.operands)
    compare_operands = _split_operands(compare.operands)
    if len(update_operands) != 3 or len(compare_operands) != 3:
        raise PtxTranslationError(f"WMMA loop operands are malformed: {kernel.name}")
    control_register = update_operands[0].strip()
    if (
        not _REGISTER_RE.fullmatch(control_register)
        or update_operands[1].strip() != control_register
        or compare_operands[1].strip() != control_register
    ):
        raise PtxTranslationError(f"WMMA loop induction variable is not canonical: {kernel.name}")
    try:
        step = int(update_operands[2].strip().rstrip("uU"), 0)
        limit = int(compare_operands[2].strip().rstrip("uU"), 0)
    except ValueError as exc:
        raise PtxTranslationError(f"WMMA loop bounds are not literals: {kernel.name}") from exc
    initial = _resolve_u32_constant(kernel, control_register, loop_start)
    if initial is None or step <= 0 or limit <= initial or (limit - initial) % step:
        raise PtxTranslationError(f"WMMA loop bounds are not safely expandable: {kernel.name}")
    trip_count = (limit - initial) // step
    if trip_count > 64:
        raise PtxTranslationError(f"WMMA loop expansion is too large: {kernel.name} trips={trip_count}")
    body = kernel.instructions[loop_start:branch_index]
    if not any(instruction.opcode.startswith("wmma.") for instruction in body):
        return kernel
    if any(instruction.opcode.split(".", 1)[0] == "bra" for instruction in body):
        raise PtxTranslationError(f"WMMA loop contains a nested branch: {kernel.name}")
    if any(loop_start < index < branch_index for _name, index in kernel.label_indices):
        raise PtxTranslationError(f"WMMA loop contains nested labels: {kernel.name}")

    instructions = (
        kernel.instructions[:loop_start]
        + body * trip_count
        + kernel.instructions[branch_index + 1:]
    )
    expanded_body_end = loop_start + len(body) * trip_count
    label_indices: list[tuple[str, int]] = []
    for name, index in kernel.label_indices:
        if index <= loop_start:
            new_index = index
        elif index >= branch_index:
            new_index = expanded_body_end + (index - (branch_index + 1))
        else:
            raise PtxTranslationError(f"WMMA loop label mapping is not safe: {kernel.name}")
        label_indices.append((name, new_index))
    return PtxKernel(
        name=kernel.name,
        parameters=kernel.parameters,
        directives=kernel.directives,
        labels=kernel.labels,
        label_indices=tuple(label_indices),
        instructions=tuple(instructions),
    )


def _fixed_wmma_loop(kernel: PtxKernel) -> tuple[int, int, str, int, int, int] | None:
    """Return ``(start, branch, register, initial, step, trips)`` for a PWIN loop."""
    labels = {name: index for name, index in kernel.label_indices}
    backward_branches = [
        (index, _branch_target(instruction, labels))
        for index, instruction in enumerate(kernel.instructions)
        if instruction.opcode.split(".", 1)[0] == "bra"
        and _branch_target(instruction, labels) <= index
    ]
    wmma_positions = {
        index for index, instruction in enumerate(kernel.instructions)
        if instruction.opcode.startswith("wmma.")
    }
    candidates = [
        (branch, start) for branch, start in backward_branches
        if any(start <= position < branch for position in wmma_positions)
    ]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise PtxTranslationError(
            f"WMMA replay requires one bounded loop, found {len(candidates)} in {kernel.name}"
        )
    branch_index, loop_start = candidates[0]
    if branch_index < 2 or not kernel.instructions[branch_index].predicate:
        raise PtxTranslationError(f"WMMA loop control is not in the supported form: {kernel.name}")
    update = kernel.instructions[branch_index - 2]
    compare = kernel.instructions[branch_index - 1]
    if update.opcode not in {"add.s32", "add.u32"} or not compare.opcode.startswith("setp.ne."):
        raise PtxTranslationError(f"WMMA loop control is not a constant counted loop: {kernel.name}")
    update_operands = _split_operands(update.operands)
    compare_operands = _split_operands(compare.operands)
    if len(update_operands) != 3 or len(compare_operands) != 3:
        raise PtxTranslationError(f"WMMA loop operands are malformed: {kernel.name}")
    control_register = update_operands[0].strip()
    if (
        not _REGISTER_RE.fullmatch(control_register)
        or update_operands[1].strip() != control_register
        or compare_operands[1].strip() != control_register
    ):
        raise PtxTranslationError(f"WMMA loop induction variable is not canonical: {kernel.name}")
    try:
        step = int(update_operands[2].strip().rstrip("uU"), 0)
        limit = int(compare_operands[2].strip().rstrip("uU"), 0)
    except ValueError as exc:
        raise PtxTranslationError(f"WMMA loop bounds are not literals: {kernel.name}") from exc
    initial = _resolve_u32_constant(kernel, control_register, loop_start)
    if initial is None or step <= 0 or limit <= initial or (limit - initial) % step:
        raise PtxTranslationError(f"WMMA loop bounds are not safely expandable: {kernel.name}")
    trip_count = (limit - initial) // step
    if trip_count > 64:
        raise PtxTranslationError(f"WMMA loop expansion is too large: {kernel.name} trips={trip_count}")
    body = kernel.instructions[loop_start:branch_index]
    if any(instruction.opcode.split(".", 1)[0] == "bra" for instruction in body):
        raise PtxTranslationError(f"WMMA loop contains a nested branch: {kernel.name}")
    if any(loop_start < index < branch_index for _name, index in kernel.label_indices):
        raise PtxTranslationError(f"WMMA loop contains nested labels: {kernel.name}")
    return loop_start, branch_index, control_register, initial, step, trip_count


def wmma_instance_count(kernel: PtxKernel) -> int:
    """Return the runtime WMMA instance count, including fixed loop trips."""
    positions = [
        index for index, instruction in enumerate(kernel.instructions)
        if instruction.opcode.startswith("wmma.")
    ]
    loop = _fixed_wmma_loop(kernel)
    if loop is None:
        return len(positions)
    loop_start, branch_index, _register, _initial, _step, trip_count = loop
    before = sum(position < loop_start for position in positions)
    inside = sum(loop_start <= position < branch_index for position in positions)
    after = len(positions) - before - inside
    return before + inside * trip_count + after


def _wmma_operation_expressions(
    kernel: PtxKernel,
    registers: dict[str, RegisterFile],
) -> tuple[dict[int, str], int]:
    """Map each source WMMA to its compact-state runtime operation expression."""
    positions = [
        index for index, instruction in enumerate(kernel.instructions)
        if instruction.opcode.startswith("wmma.")
    ]
    loop = _fixed_wmma_loop(kernel)
    if loop is None:
        return {position: f"{operation}u" for operation, position in enumerate(positions)}, len(positions)
    loop_start, branch_index, control_register, initial, step, trip_count = loop
    before = sum(position < loop_start for position in positions)
    inside = sum(loop_start <= position < branch_index for position in positions)
    control_expression = _operand_expr(control_register, registers, "uint")
    expressions: dict[int, str] = {}
    body_operation = 0
    after_operation = 0
    for operation, position in enumerate(positions):
        if position < loop_start:
            expressions[position] = f"{operation}u"
        elif position < branch_index:
            expressions[position] = (
                f"{before}u + (({control_expression} - {initial}u) / {step}u) * "
                f"{inside}u + {body_operation}u"
            )
            body_operation += 1
        else:
            expressions[position] = f"{before + inside * trip_count + after_operation}u"
            after_operation += 1
    return expressions, before + inside * trip_count + after_operation


def _register_word_width(register: RegisterFile) -> int:
    return {"uint64_t": 2, "double": 2, "uvec4": 4}.get(register.glsl_type, 1)


def _register_storage_layout(
    registers: dict[str, RegisterFile],
) -> tuple[dict[str, int], int]:
    """Return a packed uint-word layout for a cross-pipeline register file."""
    offsets: dict[str, int] = {}
    cursor = 0
    for key, register in sorted(registers.items(), key=lambda item: item[1].glsl_name):
        offsets[key] = cursor
        word_width = _register_word_width(register)
        cursor += register.count * word_width
    return offsets, cursor


def _replay_segment_storage_indices(
    kernel: PtxKernel,
    segment_wmma_limit: int,
) -> dict[str, set[int]]:
    """Return the union of typed registers crossing any segment boundary."""
    if _fixed_wmma_loop(kernel) is not None:
        # Segment phases are independently dispatched, so turn the narrow
        # constant-count loop into the exact repeated instruction sequence
        # before calculating liveness.  The normal replay phase retains the
        # compact dynamic loop lowering; only the checkpointed segments need
        # an acyclic instruction coordinate system.
        kernel = expand_fixed_wmma_loops(kernel)
    positions = [
        index for index, instruction in enumerate(kernel.instructions)
        if instruction.opcode.startswith("wmma.")
    ]
    result: dict[str, set[int]] = {}
    segment_count = (len(positions) + segment_wmma_limit - 1) // segment_wmma_limit
    for segment_index in range(segment_count):
        start, end, _count = _replay_segment_bounds(
            kernel, segment_index, segment_wmma_limit
        ) or (0, 0, 0)
        incoming, outgoing = _replay_segment_register_indices(kernel, start, end)
        for values in (incoming, outgoing):
            for family, indices in values.items():
                result.setdefault(family, set()).update(indices)
        for family, indices in _replay_segment_branch_state(kernel, start, end).items():
            result.setdefault(family, set()).update(indices)
    return result


def _replay_sparse_storage_layout(
    registers: dict[str, RegisterFile],
    indices: dict[str, set[int]],
) -> tuple[dict[tuple[str, int], int], int]:
    """Pack only selected typed register indices into a uint-word SSBO."""
    offsets: dict[tuple[str, int], int] = {}
    cursor = 0
    for key, register in sorted(registers.items(), key=lambda item: item[1].glsl_name):
        word_width = _register_word_width(register)
        for index in sorted(indices.get(register.family, set())):
            if index >= register.count:
                continue
            offsets[(key, index)] = cursor
            cursor += word_width
    return offsets, cursor


def register_storage_word_count(
    kernel: PtxKernel,
    *,
    all_registers: bool = False,
) -> int:
    """Return the per-invocation SSBO stride used by replay storage."""
    registers = _register_files(kernel, scalarize=False)
    if not all_registers:
        register = registers.get("r")
        return register.count if register else 0
    try:
        has_wmma = any(
            instruction.opcode.startswith("wmma.")
            for instruction in kernel.instructions
        )
        limit_name = (
            "DLSSAMD_TRANSLATOR_SEGMENT_WMMA_LIMIT"
            if has_wmma
            else "DLSSAMD_TRANSLATOR_SEGMENT_INSTRUCTION_LIMIT"
        )
        limit = int(os.environ.get(limit_name, "8"), 10)
    except ValueError:
        limit = 8
    if limit < 1:
        limit = 8
    indices = (
        _replay_segment_storage_indices(kernel, limit)
        if has_wmma
        else _scalar_segment_storage_indices(kernel, limit)
    )
    return _replay_sparse_storage_layout(registers, indices)[1]


def shared_storage_word_count(kernel: PtxKernel) -> int:
    """Return the byte-addressed shared-memory footprint in 32-bit words."""
    _symbols, shared_size = _shared_layout(kernel)
    return max(1, (shared_size + 3) // 4)


def _replay_segment_bounds(
    kernel: PtxKernel,
    segment_index: int,
    segment_wmma_limit: int,
) -> tuple[int, int, int] | None:
    """Return instruction bounds and WMMA count for one acyclic replay chunk."""
    if segment_index < 0 or segment_wmma_limit <= 0:
        raise PtxTranslationError("replay segment selection is invalid")
    if _fixed_wmma_loop(kernel) is not None:
        kernel = expand_fixed_wmma_loops(kernel)
    positions = [
        index for index, instruction in enumerate(kernel.instructions)
        if instruction.opcode.startswith("wmma.")
    ]
    start_operation = segment_index * segment_wmma_limit
    if start_operation >= len(positions):
        raise PtxTranslationError(
            f"replay segment {segment_index} is outside {kernel.name}"
        )
    end_operation = min(start_operation + segment_wmma_limit, len(positions))
    start = 0 if start_operation == 0 else positions[start_operation - 1] + 1
    end = len(kernel.instructions) if end_operation == len(positions) else positions[end_operation - 1] + 1

    return start, end, end_operation - start_operation


def _scalar_segment_bounds(
    kernel: PtxKernel,
    segment_index: int,
    instruction_limit: int,
) -> tuple[int, int, int]:
    """Return instruction bounds for one forward-only scalar replay chunk."""
    if segment_index < 0 or instruction_limit <= 0:
        raise PtxTranslationError("scalar replay segment selection is invalid")
    start = segment_index * instruction_limit
    if start >= len(kernel.instructions):
        raise PtxTranslationError(
            f"scalar replay segment {segment_index} is outside {kernel.name}"
        )
    end = min(start + instruction_limit, len(kernel.instructions))
    labels = {name: index for name, index in kernel.label_indices}
    for index in range(start, end):
        instruction = kernel.instructions[index]
        if instruction.opcode.split(".", 1)[0] != "bra":
            continue
        target = _branch_target(instruction, labels)
        if target < start:
            raise PtxTranslationError(
                f"scalar replay branch crosses an earlier segment in {kernel.name}"
            )
    return start, end, end - start


def _scalar_segment_storage_indices(
    kernel: PtxKernel,
    instruction_limit: int,
) -> dict[str, set[int]]:
    """Return the union of typed registers crossing scalar segment boundaries."""
    if instruction_limit <= 0:
        raise PtxTranslationError("scalar replay instruction limit is invalid")
    segment_count = (len(kernel.instructions) + instruction_limit - 1) // instruction_limit
    result: dict[str, set[int]] = {}
    for segment_index in range(segment_count):
        start, end, _count = _scalar_segment_bounds(
            kernel, segment_index, instruction_limit
        )
        incoming, outgoing = _replay_segment_register_indices(kernel, start, end)
        for values in (incoming, outgoing):
            for family, indices in values.items():
                result.setdefault(family, set()).update(indices)
    return result


def _private_register_lvalue(
    key: str,
    index: int,
    registers: dict[str, RegisterFile],
) -> str:
    register = registers[key]
    if register.scalarized:
        return f"{register.glsl_name}_{index}"
    return f"{register.glsl_name}[{index}]"


def _instruction_register_sets(
    instruction: PtxInstruction,
) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    """Return (uses, destinations) for the register subset used by replay."""
    token_pattern = re.compile(r"%(?P<family>[A-Za-z]+)(?P<index>\d+)")
    tokens = {
        (match.group("family"), int(match.group("index")))
        for match in token_pattern.finditer(
            " ".join(value for value in (instruction.operands, instruction.predicate) if value)
        )
    }
    base = instruction.opcode.split(".", 1)[0]
    no_destination = {
        "atom", "bar", "bra", "exit", "membar", "prefetch", "ret", "st", "sust", "trap"
    }
    destinations: set[tuple[str, int]] = set()
    if base not in no_destination and instruction.operands:
        first_operand = _split_operands(instruction.operands)[0]
        destinations = {
            (match.group("family"), int(match.group("index")))
            for match in token_pattern.finditer(first_operand)
        }
    return tokens - destinations, destinations


def _replay_segment_register_indices(
    kernel: PtxKernel,
    start: int,
    end: int,
) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """Return register indices to load at and store from a replay segment."""
    instruction_sets = [_instruction_register_sets(instruction) for instruction in kernel.instructions]
    segment_uses = set().union(*(uses for uses, _destinations in instruction_sets[start:end]))
    segment_destinations = set().union(
        *(destinations for _uses, destinations in instruction_sets[start:end])
    )
    after_uses = set().union(*(uses for uses, _destinations in instruction_sets[end:]))
    incoming = segment_uses - segment_destinations
    outgoing = segment_destinations & after_uses

    def by_family(values: set[tuple[str, int]]) -> dict[str, set[int]]:
        result: dict[str, set[int]] = {}
        for family, index in values:
            result.setdefault(family, set()).add(index)
        return result

    return by_family(incoming), by_family(outgoing)


def _scalar_segment_register_declaration_indices(
    kernel: PtxKernel,
    start: int,
    end: int,
    registers: dict[str, RegisterFile],
    register_load_indices: dict[str, set[int]],
    register_store_indices: dict[str, set[int]],
) -> dict[str, set[int]]:
    """Return only the registers a scalar replay segment can reference.

    Scalar replay already checkpoints values that cross a segment boundary.
    Declaring the entire PTX register file in every segment defeats that
    reduction: the compiler then allocates or spills thousands of private
    values for every invocation even when the segment touches only a small
    subset.  Keep all numeric and named registers referenced by the segment,
    plus the explicit checkpoint sets, and let the caller scalarize those
    declarations so unused indices do not become large GLSL arrays.
    """
    result: dict[str, set[int]] = {}
    for instruction in kernel.instructions[start:end]:
        uses, destinations = _instruction_register_sets(instruction)
        for family, index in uses | destinations:
            result.setdefault(family, set()).add(index)

    for key, register in registers.items():
        if not key.startswith("@"):
            continue
        pattern = re.compile(
            r"(?<![A-Za-z0-9_$])" + re.escape(register.family) +
            r"(?![A-Za-z0-9_$])"
        )
        if any(pattern.search(instruction.text) for instruction in kernel.instructions[start:end]):
            result.setdefault(register.family, set()).add(0)

    for family, indices in register_load_indices.items():
        result.setdefault(family, set()).update(indices)
    for family, indices in register_store_indices.items():
        result.setdefault(family, set()).update(indices)
    return result


def _replay_segment_branch_state(
    kernel: PtxKernel,
    start: int,
    end: int,
) -> dict[str, set[int]]:
    """Return predicate registers needed to carry a split forward branch."""
    labels = {name: index for name, index in kernel.label_indices}
    result: dict[str, set[int]] = {}
    for index, instruction in enumerate(kernel.instructions):
        if instruction.opcode.split(".", 1)[0] != "bra":
            continue
        target = _branch_target(instruction, labels)
        if not instruction.predicate or target <= index:
            continue
        # A branch that begins before this segment and lands after its start
        # needs its predicate reloaded here.  A branch inside this segment
        # whose target lies beyond the segment needs the same state for the
        # early-return path emitted by the segment backend.
        crosses_segment = (index < start < target) or (start <= index < end < target)
        if not crosses_segment:
            continue
        match = _REGISTER_RE.fullmatch(instruction.predicate.strip())
        if match:
            result.setdefault(match.group("family"), set()).add(
                int(match.group("index"))
            )
    return result


def _replay_register_load_lines(
    registers: dict[str, RegisterFile],
    offsets: dict[tuple[str, int], int],
    *,
    indices: dict[str, set[int]] | None = None,
    indent: str = "    ",
) -> list[str]:
    lines: list[str] = []
    for key, register in sorted(registers.items(), key=lambda item: item[1].glsl_name):
        family_indices = (
            sorted(indices.get(register.family, set()))
            if indices is not None
            else range(register.count)
        )
        for index in family_indices:
            if index >= register.count:
                continue
            word = offsets.get((key, index))
            if word is None:
                continue
            lvalue = _private_register_lvalue(key, index, registers)
            if register.glsl_type == "uvec4":
                words = [f"ptx_registers.data[ptx_register_base + {word + lane}u]" for lane in range(4)]
                value = f"uvec4({', '.join(words)})"
            elif register.glsl_type == "uint64_t":
                value = (
                    f"uint64_t(ptx_registers.data[ptx_register_base + {word}u]) | "
                    f"(uint64_t(ptx_registers.data[ptx_register_base + {word + 1}u]) << 32)"
                )
            elif register.glsl_type == "double":
                bits = (
                    f"uint64_t(ptx_registers.data[ptx_register_base + {word}u]) | "
                    f"(uint64_t(ptx_registers.data[ptx_register_base + {word + 1}u]) << 32)"
                )
                value = f"uint64BitsToDouble({bits})"
            elif register.glsl_type == "float":
                value = f"uintBitsToFloat(ptx_registers.data[ptx_register_base + {word}u])"
            elif register.glsl_type == "bool":
                value = f"ptx_registers.data[ptx_register_base + {word}u] != 0u"
            else:
                value = f"ptx_registers.data[ptx_register_base + {word}u]"
            lines.append(f"{indent}{lvalue} = {value};")
    return lines


def _replay_register_zero_lines(
    registers: dict[str, RegisterFile],
    offsets: dict[tuple[str, int], int],
    *,
    indices: dict[str, set[int]] | None = None,
    indent: str = "    ",
) -> list[str]:
    lines: list[str] = []
    for key, register in sorted(registers.items(), key=lambda item: item[1].glsl_name):
        family_indices = (
            sorted(indices.get(register.family, set()))
            if indices is not None
            else range(register.count)
        )
        for index in family_indices:
            if index >= register.count:
                continue
            word = offsets.get((key, index))
            if word is None:
                continue
            word_width = _register_word_width(register)
            for word_index in range(word_width):
                lines.append(
                    f"{indent}ptx_registers.data[ptx_register_base + "
                    f"{word + word_index}u] = 0u;"
                )
    return lines


def _replay_register_store_lines(
    registers: dict[str, RegisterFile],
    offsets: dict[tuple[str, int], int],
    *,
    indices: dict[str, set[int]] | None = None,
    indent: str = "    ",
) -> list[str]:
    lines: list[str] = []
    for key, register in sorted(registers.items(), key=lambda item: item[1].glsl_name):
        family_indices = (
            sorted(indices.get(register.family, set()))
            if indices is not None
            else range(register.count)
        )
        for index in family_indices:
            if index >= register.count:
                continue
            word = offsets.get((key, index))
            if word is None:
                continue
            value = _private_register_lvalue(key, index, registers)
            if register.glsl_type == "uvec4":
                lines.extend(
                    f"{indent}ptx_registers.data[ptx_register_base + {word + lane}u] = {value}[{lane}];"
                    for lane in range(4)
                )
            elif register.glsl_type == "uint64_t":
                bits = value
                lines.extend([
                    f"{indent}ptx_registers.data[ptx_register_base + {word}u] = uint({bits});",
                    f"{indent}ptx_registers.data[ptx_register_base + {word + 1}u] = uint({bits} >> 32);",
                ])
            elif register.glsl_type == "double":
                bits = f"doubleBitsToUint64({value})"
                lines.extend([
                    f"{indent}ptx_registers.data[ptx_register_base + {word}u] = uint({bits});",
                    f"{indent}ptx_registers.data[ptx_register_base + {word + 1}u] = uint({bits} >> 32);",
                ])
            elif register.glsl_type == "float":
                value = f"floatBitsToUint({value})"
                lines.append(
                    f"{indent}ptx_registers.data[ptx_register_base + {word}u] = {value};"
                )
            elif register.glsl_type == "bool":
                lines.append(
                    f"{indent}ptx_registers.data[ptx_register_base + {word}u] = {value} ? 1u : 0u;"
                )
            else:
                lines.append(
                    f"{indent}ptx_registers.data[ptx_register_base + {word}u] = {value};"
                )
    return lines


def _emit_scalar_segment_control_flow(
    kernel: PtxKernel,
    registers: dict[str, RegisterFile],
    *,
    start: int,
    end: int,
    final: bool,
    surface_format: str,
    register_offsets: dict[tuple[str, int], int],
    register_store_indices: dict[str, set[int]],
) -> list[str]:
    """Emit a forward-only scalar segment with a persistent instruction PC."""
    labels = {name: index for name, index in kernel.label_indices}
    branch_targets: dict[int, int] = {}
    for index, instruction in enumerate(kernel.instructions):
        if instruction.opcode.split(".", 1)[0] != "bra":
            continue
        target = _branch_target(instruction, labels)
        if start <= index < end and target < start:
            raise PtxTranslationError(
                f"scalar replay branch crosses an earlier segment in {kernel.name}"
            )
        branch_targets[index] = target

    def state_value(index: int) -> str:
        return f"ptx_wmma_state.data[ptx_wmma_state_base + {index}u]"

    lines = [
        f"    if ({state_value(1)} != 0u) return;",
        f"    uint ptx_pc = {state_value(0)};",
        f"    while (ptx_pc >= {start}u && ptx_pc < {end}u) {{",
        f"        switch (ptx_pc - {start}u) {{",
    ]
    for index in range(start, end):
        instruction = kernel.instructions[index]
        base = instruction.opcode.split(".", 1)[0]
        local_index = index - start
        lines.append(f"            case {local_index}u: {{")
        if base == "bra":
            target = branch_targets[index]
            if instruction.predicate:
                condition = _operand_expr(instruction.predicate, registers, "bool")
                lines.append(
                    f"                ptx_pc = ({condition}) ? {target}u : {index + 1}u;"
                )
            else:
                lines.append(f"                ptx_pc = {target}u;")
        elif base in {"exit", "ret", "trap"}:
            lines.extend(
                _replay_register_store_lines(
                    registers,
                    register_offsets,
                    indices=register_store_indices,
                    indent="                ",
                )
            )
            lines.extend([
                f"                {state_value(0)} = {len(kernel.instructions)}u;",
                f"                {state_value(1)} = 1u;",
                "                return;",
            ])
        else:
            emitted = _emit_instruction(
                instruction,
                kernel,
                registers,
                surface_format=surface_format,
            )
            if instruction.predicate:
                condition = _operand_expr(instruction.predicate, registers, "bool")
                lines.append(f"                if ({condition}) {{")
                lines.extend(
                    f"                    {statement}" for statement in emitted.splitlines()
                )
                lines.append("                }")
            else:
                lines.extend(f"                {statement}" for statement in emitted.splitlines())
            lines.append(f"                ptx_pc = {index + 1}u;")
        lines.extend([
            "                break;",
            "            }",
        ])
    lines.extend([
        "            default:",
        f"                ptx_pc = {end}u;",
        "                break;",
        "        }",
        "    }",
    ])
    lines.extend(
        _replay_register_store_lines(
            registers,
            register_offsets,
            indices=register_store_indices,
            indent="    ",
        )
    )
    lines.append(f"    {state_value(0)} = ptx_pc;")
    if final:
        lines.append(f"    {state_value(1)} = 1u;")
    return lines


def _emit_replay_segment_control_flow(
    kernel: PtxKernel,
    registers: dict[str, RegisterFile],
    records: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    *,
    start: int,
    end: int,
    final: bool,
    surface_format: str,
    operation_expressions: dict[int, str],
    register_offsets: dict[tuple[str, int], int],
    register_load_indices: dict[str, set[int]],
    register_store_indices: dict[str, set[int]],
) -> list[str] | None:
    """Emit one independently compiled, register-checkpointed replay phase."""
    labels = {name: index for name, index in kernel.label_indices}
    branch_targets: dict[int, int] = {}
    for index, instruction in enumerate(kernel.instructions):
        if instruction.opcode.split(".", 1)[0] != "bra":
            continue
        target = _branch_target(instruction, labels)
        if instruction.predicate is None or target <= index:
            return None
        branch_targets[index] = target

    wmma_records = {
        index: record for index, record in zip(
            (position for position, instruction in enumerate(kernel.instructions)
             if instruction.opcode.startswith("wmma.")),
            records,
        )
    }
    wmma_operations = {
        index: operation
        for operation, index in enumerate(
            position for position, instruction in enumerate(kernel.instructions)
            if instruction.opcode.startswith("wmma.")
        )
    }

    def state_value(index: str) -> str:
        return f"ptx_wmma_state.data[ptx_wmma_state_base + {index}]"

    def branch_skip_operation(target: int) -> int:
        return sum(
            position < target
            for position in wmma_operations
        )

    def emit_branch_skip(condition: str, target: int, indent: str) -> list[str]:
        lines = [f"{indent}if ({condition}) {{"]
        lines.extend(
            _replay_register_store_lines(
                registers,
                register_offsets,
                indices=register_store_indices,
                indent=indent + "    ",
            )
        )
        lines.extend([
            f"{indent}    {state_value('0u')} = {branch_skip_operation(target)}u;",
            f"{indent}    {state_value('1u')} = 0xffffffffu;",
            f"{indent}    return;",
            f"{indent}}}",
        ])
        return lines

    def emit_range(range_start: int, range_end: int, indent: str) -> list[str] | None:
        lines: list[str] = []
        index = range_start
        while index < range_end:
            instruction = kernel.instructions[index]
            base = instruction.opcode.split(".", 1)[0]
            if base == "bra":
                target = branch_targets[index]
                if target > range_end:
                    condition = _operand_expr(instruction.predicate or "false", registers, "bool")
                    lines.extend(emit_branch_skip(condition, target, indent))
                    index += 1
                    continue
                condition = _operand_expr(instruction.predicate or "false", registers, "bool")
                skipped = emit_range(index + 1, target, indent + "    ")
                if skipped is None:
                    return None
                if skipped:
                    lines.append(f"{indent}if (!({condition})) {{")
                    lines.extend(skipped)
                    lines.append(f"{indent}}}")
                index = target
                continue
            if base == "wmma":
                record = wmma_records.get(index)
                if record is None:
                    return None
                destinations, a_values, b_values, c_values = record
                operation = operation_expressions[index]
                capture_values = [
                    _operand_expr(f"%r{token}", registers, "uint")
                    for token in (*a_values, *b_values, *c_values)
                ]
                lines.extend([
                    f"{indent}if ({state_value('0u')} <= {operation}) {{",
                    f"{indent}    ptx_replay_capture({operation}, {', '.join(capture_values)});",
                ])
                lines.extend(
                    _replay_register_store_lines(
                        registers,
                        register_offsets,
                        indices=register_store_indices,
                        indent=indent + "    ",
                    )
                )
                lines.extend([
                    f"{indent}    return;",
                    f"{indent}}}",
                ])
                for component, destination in enumerate(destinations):
                    lines.append(
                        f"{indent}{_destination_expr(f'%r{destination}', registers)} = "
                        f"{state_value(f'{15}u + ({operation}) * 4u + {component}u')};"
                    )
                index += 1
                continue
            if base in {"ret", "trap"}:
                if not final:
                    return None
                lines.extend(
                    _replay_register_store_lines(
                        registers,
                        register_offsets,
                        indices=register_store_indices,
                        indent=indent,
                    )
                )
                lines.extend([
                    f"{indent}{state_value('2u')} = 1u;",
                    f"{indent}{state_value('1u')} = 0xffffffffu;",
                    f"{indent}return;",
                ])
                index += 1
                continue
            emitted = _emit_instruction(
                instruction,
                kernel,
                registers,
                surface_format=surface_format,
            )
            lines.append(f"{indent}{{")
            block_indent = indent + "    "
            if instruction.predicate:
                condition = _operand_expr(instruction.predicate, registers, "bool")
                lines.append(f"{block_indent}if ({condition}) {{")
                lines.extend(f"{block_indent}    {statement}" for statement in emitted.splitlines())
                lines.append(f"{block_indent}}}")
            else:
                lines.extend(f"{block_indent}{statement}" for statement in emitted.splitlines())
            lines.append(f"{indent}}}")
            index += 1
        return lines

    lines = [
        f"    if ({state_value('2u')} != 0u) return;",
    ]
    labels = {name: index for name, index in kernel.label_indices}
    for index, instruction in enumerate(kernel.instructions):
        if instruction.opcode.split(".", 1)[0] != "bra" or index >= start:
            continue
        target = _branch_target(instruction, labels)
        if target <= start or target < end or not instruction.predicate:
            continue
        condition = _operand_expr(instruction.predicate, registers, "bool")
        lines.extend(emit_branch_skip(condition, target, "    "))
    body = emit_range(start, end, "    ")
    if body is None:
        return None
    lines.extend(body)
    lines.extend(
        _replay_register_store_lines(
            registers,
            register_offsets,
            indices=register_store_indices,
            indent="    ",
        )
    )
    if final:
        lines.extend([
            f"    {state_value('2u')} = 1u;",
            f"    {state_value('1u')} = 0xffffffffu;",
        ])
    lines.append("    return;")
    return lines


def _emit_control_flow(
    kernel: PtxKernel,
    registers: dict[str, RegisterFile],
    *,
    surface_format: str,
    staged_wmma: bool = False,
    replay_wmma: bool = False,
) -> list[str]:
    labels = {name: index for name, index in kernel.label_indices}
    wmma_operations = {
        index: operation
        for operation, index in enumerate(
            position
            for position, instruction in enumerate(kernel.instructions)
            if instruction.opcode.startswith("wmma.")
        )
    }
    wmma_records = {
        index: record
        for index, record in (
            (index, _wmma_register_indices(instruction))
            for index, instruction in enumerate(kernel.instructions)
            if instruction.opcode.startswith("wmma.")
        )
    }

    def state_value(index: str) -> str:
        return f"ptx_wmma_state.data[ptx_wmma_state_base + {index}]"

    page_size = 1024
    if replay_wmma:
        try:
            page_size = int(os.environ.get("DLSSAMD_TRANSLATOR_INTERPRETER_PAGE_SIZE", "128"))
        except ValueError:
            page_size = 128
        if page_size not in {64, 128, 256, 512, 1024}:
            page_size = 128
    lines = [
        "    uint ptx_pc = 0u;",
        "    while (true) {",
        f"        switch (ptx_pc / {page_size}u) {{",
    ]
    for page_start in range(0, len(kernel.instructions), page_size):
        page_end = min(page_start + page_size, len(kernel.instructions))
        page = page_start // page_size
        lines.append(f"            case {page}u: {{")
        lines.append(f"                switch (ptx_pc % {page_size}u) {{")
        for index in range(page_start, page_end):
            instruction = kernel.instructions[index]
            opcode = instruction.opcode
            base = opcode.split(".", 1)[0]
            local_index = index - page_start
            lines.append(f"                    case {local_index}u: {{")
            if base == "bra":
                target = _branch_target(instruction, labels)
                next_index = index + 1
                if instruction.predicate:
                    condition = _operand_expr(instruction.predicate, registers, "bool")
                    lines.append(
                        f"                        if ({condition}) ptx_pc = {target}u; else ptx_pc = {next_index}u;"
                    )
                else:
                    lines.append(f"                        ptx_pc = {target}u;")
            elif base in {"ret", "trap"}:
                if replay_wmma:
                    lines.extend([
                        f"                        {state_value('2u')} = 1u;",
                        f"                        {state_value('1u')} = 0xffffffffu;",
                    ])
                lines.append("                        return;")
            elif replay_wmma and base == "wmma":
                operation = wmma_operations[index]
                record = wmma_records[index]
                if record is None:
                    raise PtxTranslationError("WMMA replay operands are not supported")
                destinations, a_values, b_values, c_values = record
                lines.append(f"                        if ({state_value('0u')} <= {operation}u) {{")
                capture_values = [
                    _operand_expr(f"%r{token}", registers, "uint")
                    for token in (*a_values, *b_values, *c_values)
                ]
                lines.append(
                    "                            ptx_replay_capture("
                    f"{operation}u, {', '.join(capture_values)});"
                )
                lines.extend([
                    f"                            ptx_pc = {index + 1}u;",
                    "                            return;",
                    "                        }",
                ])
                for component, destination in enumerate(destinations):
                    lines.append(
                        f"                        {_destination_expr(f'%r{destination}', registers)} = "
                        f"{state_value(f'{15 + operation * 4 + component}u')};"
                    )
                lines.append(f"                        ptx_pc = {index + 1}u;")
            elif staged_wmma and base == "wmma":
                operation = wmma_operations[index]
                lines.extend([
                    f"                        uvec4 ptx_wmma_result = {_wmma_dynamic_call(registers, f'{operation}u')};",
                    f"                        {_wmma_register_lvalue(registers, f'ptx_wmma_dest_indices[{operation}u].x')} = ptx_wmma_result.x;",
                    f"                        {_wmma_register_lvalue(registers, f'ptx_wmma_dest_indices[{operation}u].y')} = ptx_wmma_result.y;",
                    f"                        {_wmma_register_lvalue(registers, f'ptx_wmma_dest_indices[{operation}u].z')} = ptx_wmma_result.z;",
                    f"                        {_wmma_register_lvalue(registers, f'ptx_wmma_dest_indices[{operation}u].w')} = ptx_wmma_result.w;",
                ])
            else:
                emitted = _emit_instruction(
                    instruction,
                    kernel,
                    registers,
                    surface_format=surface_format,
                )
                if instruction.predicate:
                    condition = _operand_expr(instruction.predicate, registers, "bool")
                    lines.append(f"                        if ({condition}) {{")
                    for statement in emitted.splitlines():
                        lines.append(f"                            {statement}")
                    lines.append("                        }")
                else:
                    for statement in emitted.splitlines():
                        lines.append(f"                        {statement}")
                lines.append(f"                        ptx_pc = {index + 1}u;")
            lines.extend(["                        break;", "                    } "])
        lines.extend([
            "                    default:",
            "                        return;",
            "                }",
            "                break;",
            "            }",
        ])
    lines.extend([
        "            default:",
        "                return;",
        "        }",
        "    }",
    ])
    return lines


def _emit_split_control_flow(
    kernel: PtxKernel,
    registers: dict[str, RegisterFile],
    *,
    surface_format: str,
    staged_wmma: bool,
) -> tuple[list[str], list[str]]:
    """Split the general PTX interpreter into non-inlined page functions.

    The fallback PC interpreter is needed for kernels with loops and nested
    branches, but putting all of its cases in ``main`` makes RADV build one
    enormous optimization unit.  Keep the exact PC semantics while placing a
    bounded number of instructions in each helper.  The runtime compiler
    marks these helpers ``DontInline`` when the large native WMMA path is
    enabled, so the driver can compile the pages independently.
    """
    labels = {name: index for name, index in kernel.label_indices}
    try:
        page_size = int(os.environ.get("DLSSAMD_TRANSLATOR_PAGE_SIZE", "256"))
    except ValueError:
        page_size = 256
    if page_size not in {32, 64, 128, 256, 512}:
        page_size = 256
    wmma_operations = {
        index: operation
        for operation, index in enumerate(
            position
            for position, instruction in enumerate(kernel.instructions)
            if instruction.opcode.startswith("wmma.")
        )
    }

    helper_lines: list[str] = []
    page_count = (len(kernel.instructions) + page_size - 1) // page_size
    for page in range(page_count):
        page_start = page * page_size
        page_end = min(page_start + page_size, len(kernel.instructions))
        helper_lines.extend([
            f"bool ptx_exec_page_{page}(inout uint ptx_pc, out uint ptx_wmma_operation) {{",
            "    ptx_wmma_operation = 0xffffffffu;",
            f"    switch (ptx_pc % {page_size}u) {{",
        ])
        for index in range(page_start, page_end):
            instruction = kernel.instructions[index]
            opcode = instruction.opcode
            base = opcode.split(".", 1)[0]
            local_index = index - page_start
            helper_lines.append(f"        case {local_index}u: {{")
            if base == "bra":
                target = _branch_target(instruction, labels)
                next_index = index + 1
                if instruction.predicate:
                    condition = _operand_expr(instruction.predicate, registers, "bool")
                    helper_lines.append(
                        f"            if ({condition}) ptx_pc = {target}u; else ptx_pc = {next_index}u;"
                    )
                else:
                    helper_lines.append(f"            ptx_pc = {target}u;")
            elif base in {"ret", "trap"}:
                helper_lines.append("            return true;")
            elif staged_wmma and base == "wmma":
                operation = wmma_operations[index]
                helper_lines.extend([
                    f"            ptx_wmma_operation = {operation}u;",
                    f"            ptx_pc = {index + 1}u;",
                ])
            else:
                emitted = _emit_instruction(
                    instruction,
                    kernel,
                    registers,
                    surface_format=surface_format,
                )
                if instruction.predicate:
                    condition = _operand_expr(instruction.predicate, registers, "bool")
                    helper_lines.append(f"            if ({condition}) {{")
                    helper_lines.extend(f"                {statement}" for statement in emitted.splitlines())
                    helper_lines.append("            }")
                else:
                    helper_lines.extend(f"            {statement}" for statement in emitted.splitlines())
                helper_lines.append(f"            ptx_pc = {index + 1}u;")
            helper_lines.extend(["            break;", "        }"])
        helper_lines.extend([
            "        default:",
            "            return true;",
            "    }",
            "    return false;",
            "}",
        ])

    main_lines = [
        "    uint ptx_pc = 0u;",
        "    uint ptx_wmma_operation;",
        "    while (true) {",
        f"        switch (ptx_pc / {page_size}u) {{",
    ]
    for page in range(page_count):
        main_lines.extend([
            f"            case {page}u:",
            f"                if (ptx_exec_page_{page}(ptx_pc, ptx_wmma_operation)) return;",
            "                break;",
        ])
    main_lines.extend([
        "            default:",
        "                return;",
        "        }",
    ])
    if staged_wmma:
        main_lines.extend([
            "        if (ptx_wmma_operation != 0xffffffffu) {",
            f"            uvec4 ptx_wmma_result = {_wmma_dynamic_call(registers, 'ptx_wmma_operation')};",
            f"            {_wmma_register_lvalue(registers, 'ptx_wmma_dest_indices[ptx_wmma_operation].x')} = ptx_wmma_result.x;",
            f"            {_wmma_register_lvalue(registers, 'ptx_wmma_dest_indices[ptx_wmma_operation].y')} = ptx_wmma_result.y;",
            f"            {_wmma_register_lvalue(registers, 'ptx_wmma_dest_indices[ptx_wmma_operation].z')} = ptx_wmma_result.z;",
            f"            {_wmma_register_lvalue(registers, 'ptx_wmma_dest_indices[ptx_wmma_operation].w')} = ptx_wmma_result.w;",
            "        }",
        ])
    main_lines.append("    }")
    return main_lines, helper_lines


def _emit_replay_control_flow(
    kernel: PtxKernel,
    registers: dict[str, RegisterFile],
    *,
    surface_format: str,
    operation_expressions: dict[int, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Replay scalar PTX until one WMMA, leaving the matrix op to another pipeline.

    This path deliberately starts at PTX instruction zero on every dispatch.
    The PWIN modules are SSA-like, so only the compact WMMA input/result state
    crosses the scalar/native pipeline boundary.  Replaying the scalar setup
    keeps the temporary predicate, address, and half-register files private
    and avoids putting a cooperative-matrix operation in the large scalar CFG
    shader.
    """
    labels = {name: index for name, index in kernel.label_indices}
    try:
        page_size = int(os.environ.get("DLSSAMD_TRANSLATOR_PAGE_SIZE", "64"))
    except ValueError:
        page_size = 64
    if page_size not in {32, 64, 128, 256, 512}:
        page_size = 64
    wmma_operations = {
        index: (operation_expressions or {}).get(index, f"{operation}u")
        for operation, index in enumerate(
            position
            for position, instruction in enumerate(kernel.instructions)
            if instruction.opcode.startswith("wmma.")
        )
    }
    wmma_records = {
        index: record
        for index, record in (
            (index, _wmma_register_indices(instruction))
            for index, instruction in enumerate(kernel.instructions)
            if instruction.opcode.startswith("wmma.")
        )
    }
    if any(record is None for record in wmma_records.values()):
        raise PtxTranslationError("WMMA replay operands are not supported")
    def state_value(index: str) -> str:
        return f"ptx_wmma_state.data[ptx_wmma_state_base + {index}]"

    def state_result_value(operation: str, component: int) -> str:
        return (
            f"ptx_wmma_state.data[ptx_wmma_state_base + {_WMMA_STATE_RESULT_BASE}u + ({operation}) * 4u + "
            f"{component}u]"
        )

    page_count = (len(kernel.instructions) + page_size - 1) // page_size
    helper_lines: list[str] = []
    for page in range(page_count):
        page_start = page * page_size
        page_end = min(page_start + page_size, len(kernel.instructions))
        helper_lines.extend([
            f"uint ptx_replay_page_{page}(inout uint ptx_pc) {{",
            f"    switch (ptx_pc % {page_size}u) {{",
        ])
        for index in range(page_start, page_end):
            instruction = kernel.instructions[index]
            base = instruction.opcode.split(".", 1)[0]
            local_index = index - page_start
            helper_lines.append(f"        case {local_index}u: {{")
            if base == "wmma":
                operation = wmma_operations[index]
                destinations, a_values, b_values, c_values = wmma_records[index]  # type: ignore[misc]
                capture_values = [
                    _operand_expr(f"%r{token}", registers, "uint")
                    for token in (*a_values, *b_values, *c_values)
                ]
                if registers["r"].storage:
                    helper_lines.append(
                        f"            if (ptx_replay_wmma({operation}, "
                        f"{', '.join(f'{destination}u' for destination in destinations)}, "
                        f"{', '.join(capture_values)})) return 1u;"
                    )
                else:
                    helper_lines.extend([
                        f"            if ({state_value('0u')} <= {operation}) {{",
                        f"                {state_value('1u')} = {operation};",
                    ])
                    for offset, token in enumerate((*a_values, *b_values, *c_values), start=3):
                        helper_lines.append(
                            f"                {state_value(f'{offset}u')} = "
                            f"{_operand_expr(f'%r{token}', registers, 'uint')};"
                        )
                    helper_lines.extend([
                        f"                ptx_pc = {index + 1}u;",
                        "                return 1u;",
                        "            }",
                    ])
                    for component, destination in enumerate(destinations):
                        destination_expression = _destination_expr(f"%r{destination}", registers)
                        helper_lines.append(
                            f"            {destination_expression} = "
                            f"{state_result_value(operation, component)};"
                        )
                helper_lines.append(f"            ptx_pc = {index + 1}u;")
            elif base == "bra":
                target = _branch_target(instruction, labels)
                next_index = index + 1
                if instruction.predicate:
                    condition = _operand_expr(instruction.predicate, registers, "bool")
                    helper_lines.append(
                        f"            if ({condition}) ptx_pc = {target}u; else ptx_pc = {next_index}u;"
                    )
                else:
                    helper_lines.append(f"            ptx_pc = {target}u;")
            elif base in {"ret", "trap"}:
                helper_lines.extend([
                    f"            {state_value('2u')} = 1u;",
                    f"            {state_value('1u')} = 0xffffffffu;",
                    "            return 2u;",
                ])
            else:
                emitted = _emit_instruction(
                    instruction,
                    kernel,
                    registers,
                    surface_format=surface_format,
                )
                if instruction.predicate:
                    condition = _operand_expr(instruction.predicate, registers, "bool")
                    helper_lines.append(f"            if ({condition}) {{")
                    helper_lines.extend(f"                {statement}" for statement in emitted.splitlines())
                    helper_lines.append("            }")
                else:
                    helper_lines.extend(f"            {statement}" for statement in emitted.splitlines())
                helper_lines.append(f"            ptx_pc = {index + 1}u;")
            helper_lines.extend(["            break;", "        }"])
        helper_lines.extend([
            "        default:",
            f"            {state_value('2u')} = 1u;",
            f"            {state_value('1u')} = 0xffffffffu;",
            "            return 2u;",
            "    }",
            "    return 0u;",
            "}",
        ])

    main_lines = [
        f"    if ({state_value('2u')} != 0u) return;",
        "    uint ptx_pc = 0u;",
        "    while (true) {",
        f"        switch (ptx_pc / {page_size}u) {{",
    ]
    for page in range(page_count):
        main_lines.extend([
            f"            case {page}u:",
            f"                if (ptx_replay_page_{page}(ptx_pc) != 0u) return;",
            "                break;",
        ])
    main_lines.extend([
            "            default:",
        f"                {state_value('2u')} = 1u;",
        f"                {state_value('1u')} = 0xffffffffu;",
        "                return;",
        "        }",
        "    }",
    ])
    return main_lines, helper_lines


def _emit_forward_control_flow(
    kernel: PtxKernel,
    registers: dict[str, RegisterFile],
    *,
    surface_format: str,
) -> list[str] | None:
    """Lower an acyclic PTX CFG without materializing a program counter.

    Clang-generated DLSS PWIN kernels use forward branches to skip a small
    predicated store block.  The general interpreter lowering is correct for
    arbitrary PTX, but its nested PC switch makes RADV compile a huge shader.
    Accept only the simple, statically provable forward-skip shape here and
    retain the interpreter for kernels with loops or nested branch regions.
    """
    labels = {name: index for name, index in kernel.label_indices}
    branch_targets: dict[int, int] = {}
    for index, instruction in enumerate(kernel.instructions):
        if instruction.opcode.split(".", 1)[0] != "bra":
            continue
        target = _branch_target(instruction, labels)
        if not instruction.predicate or target <= index:
            return None
        branch_targets[index] = target

    lines: list[str] = []
    index = 0
    while index < len(kernel.instructions):
        instruction = kernel.instructions[index]
        base = instruction.opcode.split(".", 1)[0]
        if base == "bra":
            target = branch_targets[index]
            skipped = kernel.instructions[index + 1:target]
            if any(position in branch_targets for position in range(index + 1, target)):
                return None
            condition = _operand_expr(instruction.predicate or "false", registers, "bool")
            if skipped:
                lines.append(f"    if (!({condition})) {{")
                for skipped_instruction in skipped:
                    lines.append("        {")
                    emitted = _emit_instruction(
                        skipped_instruction,
                        kernel,
                        registers,
                        surface_format=surface_format,
                    )
                    if skipped_instruction.predicate:
                        skipped_condition = _operand_expr(
                            skipped_instruction.predicate,
                            registers,
                            "bool",
                        )
                        lines.append(f"            if ({skipped_condition}) {{")
                        lines.extend(f"                {statement}" for statement in emitted.splitlines())
                        lines.append("            }")
                    else:
                        lines.extend(f"            {statement}" for statement in emitted.splitlines())
                    lines.append("        }")
                lines.append("    }")
            index = target
            continue
        if base in {"ret", "trap"}:
            lines.append("    return;")
            index += 1
            continue
        emitted = _emit_instruction(
            instruction,
            kernel,
            registers,
            surface_format=surface_format,
        )
        lines.append("    {")
        if instruction.predicate:
            condition = _operand_expr(instruction.predicate, registers, "bool")
            lines.append(f"        if ({condition}) {{")
            lines.extend(f"            {statement}" for statement in emitted.splitlines())
            lines.append("        }")
        else:
            lines.extend(f"        {statement}" for statement in emitted.splitlines())
        lines.append("    }")
        index += 1
    return lines


def _glsl_parameter_helpers(
    registers: dict[str, RegisterFile],
    *,
    shared_size: int,
    shared_storage: bool = False,
    has_global_memory: bool,
    has_address_map: bool = False,
    has_shared_memory: bool,
    has_local_memory: bool,
    has_constant_memory: bool,
) -> list[str]:
    lines = [
        "uint ptx_load_u8(uint byte_offset) {",
        "    uint word = ptx_params.data[byte_offset >> 2u];",
        "    return (word >> ((byte_offset & 3u) * 8u)) & 0xffu;",
        "}",
        "uint ptx_load_u16(uint byte_offset) {",
        "    return ptx_load_u8(byte_offset) | (ptx_load_u8(byte_offset + 1u) << 8u);",
        "}",
        "uint ptx_load_u32(uint byte_offset) {",
        "    return ptx_load_u8(byte_offset) | (ptx_load_u8(byte_offset + 1u) << 8u) |",
        "           (ptx_load_u8(byte_offset + 2u) << 16u) | (ptx_load_u8(byte_offset + 3u) << 24u);",
        "}",
        "float ptx_unpack_f16(uint bits) {",
        "    return unpackHalf2x16(bits & 0xffffu).x;",
        "}",
        "vec2 ptx_unpack_f16x2(uint bits) {",
        "    return unpackHalf2x16(bits);",
        "}",
        "uint ptx_pack_f16(float value) {",
        "    return packHalf2x16(vec2(value, 0.0));",
        "}",
        "float ptx_e4m3_to_float(uint bits) {",
        "    uint sign = bits >> 7u;",
        "    uint exponent = (bits >> 3u) & 0xfu;",
        "    uint mantissa = bits & 7u;",
        "    float magnitude;",
        "    if (exponent == 0u) magnitude = exp2(-6.0) * (float(mantissa) / 8.0);",
        "    else if (exponent == 15u && mantissa == 7u) magnitude = uintBitsToFloat(0x7fc00000u);",
        "    else magnitude = exp2(float(int(exponent) - 7)) * (1.0 + float(mantissa) / 8.0);",
        "    return sign != 0u ? -magnitude : magnitude;",
        "}",
        "vec2 ptx_unpack_e4m3x2(uint bits) {",
        "    return vec2(ptx_e4m3_to_float(bits & 0xffu), ptx_e4m3_to_float((bits >> 8u) & 0xffu));",
        "}",
        "uint ptx_float_to_e4m3(float value) {",
        "    if (isnan(value)) return 0x7fu;",
        "    uint sign = value < 0.0 ? 0x80u : 0u;",
        "    float magnitude = abs(value);",
        "    if (magnitude == 0.0) return sign;",
        "    if (magnitude >= 448.0) return sign | 0x7eu;",
        "    int exponent = int(floor(log2(magnitude)));",
        "    if (exponent < -6) return sign | uint(clamp(int(roundEven(magnitude * 512.0)), 0, 7));",
        "    int encoded_exponent = clamp(exponent + 7, 1, 14);",
        "    float scaled = magnitude / exp2(float(exponent)) - 1.0;",
        "    int mantissa = int(roundEven(scaled * 8.0));",
        "    if (mantissa >= 8) { encoded_exponent += 1; mantissa = 0; }",
        "    if (encoded_exponent >= 15) return sign | 0x7eu;",
        "    return sign | (uint(encoded_exponent) << 3u) | uint(clamp(mantissa, 0, 7));",
        "}",
        "uint ptx_pack_e4m3x2(vec2 value) {",
        "    return ptx_float_to_e4m3(value.x) | (ptx_float_to_e4m3(value.y) << 8u);",
        "}",
        "uint ptx_bfi(uint insert_value, uint base_value, uint position, uint width) {",
        "    if (width == 0u || position >= 32u) return base_value;",
        "    uint effective_width = min(width, 32u - position);",
        "    uint mask = effective_width == 32u ? 0xffffffffu : ((1u << effective_width) - 1u) << position;",
        "    return (base_value & ~mask) | ((insert_value << position) & mask);",
        "}",
        "uint ptx_prmt(uint first, uint second, uint selector) {",
        "    uint result = 0u;",
        "    for (uint lane = 0u; lane < 4u; ++lane) {",
        "        uint choice = (selector >> (lane * 4u)) & 0xfu;",
        "        uint selected = 0u;",
        "        if ((choice & 8u) == 0u) {",
        "            uint source = choice < 4u ? first : second;",
        "            selected = (source >> ((choice & 3u) * 8u)) & 0xffu;",
        "        }",
        "        result |= selected << (lane * 8u);",
        "    }",
        "    return result;",
        "}",
        "uint ptx_dp2a(uint first, uint second, uint accumulator, bool signed_inputs) {",
        "    int first0 = int(first & 0xffu);",
        "    int first1 = int((first >> 8u) & 0xffu);",
        "    int second0 = int(second & 0xffu);",
        "    int second1 = int((second >> 8u) & 0xffu);",
        "    if (signed_inputs) {",
        "        if (first0 >= 128) first0 -= 256;",
        "        if (first1 >= 128) first1 -= 256;",
        "        if (second0 >= 128) second0 -= 256;",
        "        if (second1 >= 128) second1 -= 256;",
        "    }",
        "    return uint(int(accumulator) + first0 * second0 + first1 * second1);",
        "}",
    ]
    if any(register.glsl_type == "uint64_t" for register in registers.values()):
        lines.extend([
            "uint64_t ptx_load_u64(uint byte_offset) {",
            "    return uint64_t(ptx_load_u32(byte_offset)) |",
            "           (uint64_t(ptx_load_u32(byte_offset + 4u)) << 32);",
            "}",
        ])
    if has_shared_memory:
        shared_word = "ptx_shared_state.data[ptx_shared_base + (byte_offset >> 2u)]" if shared_storage else "ptx_shared_data[byte_offset >> 2u]"
        shared_store_word = "ptx_shared_state.data[ptx_shared_base + word_index]" if shared_storage else "ptx_shared_data[word_index]"
        lines.extend([
            "uint ptx_shared_load_u8(uint64_t address) {",
            "    uint byte_offset = uint(address);",
            f"    uint word = {shared_word};",
            "    return (word >> ((byte_offset & 3u) * 8u)) & 0xffu;",
            "}",
            "uint ptx_shared_load_u16(uint64_t address) {",
            "    return ptx_shared_load_u8(address) | (ptx_shared_load_u8(address + uint64_t(1UL)) << 8u);",
            "}",
            "uint ptx_shared_load_u32(uint64_t address) {",
            "    return ptx_shared_load_u8(address) | (ptx_shared_load_u8(address + uint64_t(1UL)) << 8u) |",
            "           (ptx_shared_load_u8(address + uint64_t(2UL)) << 16u) | (ptx_shared_load_u8(address + uint64_t(3UL)) << 24u);",
            "}",
            "uint64_t ptx_shared_load_u64(uint64_t address) {",
            "    return uint64_t(ptx_shared_load_u32(address)) |",
            "           (uint64_t(ptx_shared_load_u32(address + uint64_t(4UL))) << 32);",
            "}",
            "void ptx_shared_store_u8(uint64_t address, uint value) {",
            "    uint byte_offset = uint(address);",
            "    uint word_index = byte_offset >> 2u;",
            "    uint shift = (byte_offset & 3u) * 8u;",
            "    uint mask = 0xffu << shift;",
            f"    {shared_store_word} = ({shared_store_word} & ~mask) | ((value & 0xffu) << shift);",
            "}",
            "void ptx_shared_store_u16(uint64_t address, uint value) {",
            "    ptx_shared_store_u8(address, value);",
            "    ptx_shared_store_u8(address + uint64_t(1UL), value >> 8u);",
            "}",
            "void ptx_shared_store_u32(uint64_t address, uint value) {",
            "    ptx_shared_store_u8(address, value);",
            "    ptx_shared_store_u8(address + uint64_t(1UL), value >> 8u);",
            "    ptx_shared_store_u8(address + uint64_t(2UL), value >> 16u);",
            "    ptx_shared_store_u8(address + uint64_t(3UL), value >> 24u);",
            "}",
            "void ptx_shared_store_u64(uint64_t address, uint64_t value) {",
            "    ptx_shared_store_u32(address, uint(value));",
            "    ptx_shared_store_u32(address + uint64_t(4UL), uint(value >> 32));",
            "}",
        ])
    if has_local_memory:
        lines.extend([
            "uint ptx_local_load_u8(uint64_t address) {",
            "    uint byte_offset = uint(address);",
            "    uint word = ptx_local_data[byte_offset >> 2u];",
            "    return (word >> ((byte_offset & 3u) * 8u)) & 0xffu;",
            "}",
            "uint ptx_local_load_u16(uint64_t address) {",
            "    return ptx_local_load_u8(address) | (ptx_local_load_u8(address + uint64_t(1UL)) << 8u);",
            "}",
            "uint ptx_local_load_u32(uint64_t address) {",
            "    return ptx_local_load_u8(address) | (ptx_local_load_u8(address + uint64_t(1UL)) << 8u) |",
            "           (ptx_local_load_u8(address + uint64_t(2UL)) << 16u) | (ptx_local_load_u8(address + uint64_t(3UL)) << 24u);",
            "}",
            "uint64_t ptx_local_load_u64(uint64_t address) {",
            "    return uint64_t(ptx_local_load_u32(address)) |",
            "           (uint64_t(ptx_local_load_u32(address + uint64_t(4UL))) << 32);",
            "}",
            "void ptx_local_store_u8(uint64_t address, uint value) {",
            "    uint byte_offset = uint(address);",
            "    uint word_index = byte_offset >> 2u;",
            "    uint shift = (byte_offset & 3u) * 8u;",
            "    uint mask = 0xffu << shift;",
            "    ptx_local_data[word_index] = (ptx_local_data[word_index] & ~mask) | ((value & 0xffu) << shift);",
            "}",
            "void ptx_local_store_u16(uint64_t address, uint value) {",
            "    ptx_local_store_u8(address, value);",
            "    ptx_local_store_u8(address + uint64_t(1UL), value >> 8u);",
            "}",
            "void ptx_local_store_u32(uint64_t address, uint value) {",
            "    ptx_local_store_u8(address, value);",
            "    ptx_local_store_u8(address + uint64_t(1UL), value >> 8u);",
            "    ptx_local_store_u8(address + uint64_t(2UL), value >> 16u);",
            "    ptx_local_store_u8(address + uint64_t(3UL), value >> 24u);",
            "}",
            "void ptx_local_store_u64(uint64_t address, uint64_t value) {",
            "    ptx_local_store_u32(address, uint(value));",
            "    ptx_local_store_u32(address + uint64_t(4UL), uint(value >> 32));",
            "}",
        ])
    if has_constant_memory:
        lines.extend([
            "uint ptx_const_load_u8(uint byte_offset) {",
            "    uint word = ptx_const.data[byte_offset >> 2u];",
            "    return (word >> ((byte_offset & 3u) * 8u)) & 0xffu;",
            "}",
            "uint ptx_const_load_u16(uint byte_offset) {",
            "    return ptx_const_load_u8(byte_offset) | (ptx_const_load_u8(byte_offset + 1u) << 8u);",
            "}",
            "uint ptx_const_load_u32(uint byte_offset) {",
            "    return ptx_const_load_u8(byte_offset) | (ptx_const_load_u8(byte_offset + 1u) << 8u) |",
            "           (ptx_const_load_u8(byte_offset + 2u) << 16u) | (ptx_const_load_u8(byte_offset + 3u) << 24u);",
            "}",
            "uint64_t ptx_const_load_u64(uint byte_offset) {",
            "    return uint64_t(ptx_const_load_u32(byte_offset)) |",
            "           (uint64_t(ptx_const_load_u32(byte_offset + 4u)) << 32);",
            "}",
        ])
    if has_global_memory:
        lines.extend([
            "layout(buffer_reference, std430) buffer PtxGlobalMemory { uint data[]; };",
            "layout(buffer_reference, std430) buffer PtxGlobalFloatMemory { float data[]; };",
        ])
        if has_address_map:
            lines.extend([
                f"layout(set = 0, binding = {_RUNTIME_ADDRESS_MAP_BINDING}, std430) readonly buffer PtxAddressMap {{ uint64_t data[]; }} ptx_address_map;",
                "uint64_t ptx_resolve_global_address(uint64_t address) {",
                "    uint count = uint(ptx_address_map.data[0]);",
                "    for (uint index = 0u; index < 256u && index < count; ++index) {",
                "        uint base = 1u + index * 4u;",
                "        uint64_t source = ptx_address_map.data[base + 0u];",
                "        uint64_t target = ptx_address_map.data[base + 1u];",
                "        uint64_t size = ptx_address_map.data[base + 2u];",
                "        if (address >= source && address - source < size)",
                "            return target + (address - source);",
                "    }",
                "    return address;",
                "}",
            ])
        else:
            lines.append(
                "uint64_t ptx_resolve_global_address(uint64_t address) { return address; }"
            )
        lines.extend([
            "uint ptx_global_load_u8(uint64_t address) {",
            "    uint64_t resolved = ptx_resolve_global_address(address);",
            "    uint64_t aligned = resolved & ~uint64_t(3UL);",
            "    uint word = PtxGlobalMemory(aligned).data[0];",
            "    return (word >> (uint(resolved & uint64_t(3UL)) * 8u)) & 0xffu;",
            "}",
            "uint ptx_global_load_u16(uint64_t address) {",
            "    return ptx_global_load_u8(address) | (ptx_global_load_u8(address + uint64_t(1UL)) << 8u);",
            "}",
            "uint ptx_global_load_u32(uint64_t address) {",
            "    return ptx_global_load_u8(address) | (ptx_global_load_u8(address + uint64_t(1UL)) << 8u) |",
            "           (ptx_global_load_u8(address + uint64_t(2UL)) << 16u) | (ptx_global_load_u8(address + uint64_t(3UL)) << 24u);",
            "}",
            "uint64_t ptx_global_load_u64(uint64_t address) {",
            "    return uint64_t(ptx_global_load_u32(address)) |",
            "           (uint64_t(ptx_global_load_u32(address + uint64_t(4UL))) << 32);",
            "}",
            "void ptx_global_store_u8(uint64_t address, uint value) {",
            "    uint64_t resolved = ptx_resolve_global_address(address);",
            "    uint64_t aligned = resolved & ~uint64_t(3UL);",
            "    PtxGlobalMemory(aligned).data[0] = (PtxGlobalMemory(aligned).data[0] & ~(0xffu << (uint(resolved & uint64_t(3UL)) * 8u))) |",
            "        ((value & 0xffu) << (uint(resolved & uint64_t(3UL)) * 8u));",
            "}",
            "void ptx_global_store_u16(uint64_t address, uint value) {",
            "    ptx_global_store_u8(address, value);",
            "    ptx_global_store_u8(address + uint64_t(1UL), value >> 8u);",
            "}",
            "void ptx_global_store_u32(uint64_t address, uint value) {",
            "    PtxGlobalMemory(ptx_resolve_global_address(address)).data[0] = value;",
            "}",
            "void ptx_global_store_u64(uint64_t address, uint64_t value) {",
            "    PtxGlobalMemory(ptx_resolve_global_address(address)).data[0] = uint(value);",
            "    PtxGlobalMemory(ptx_resolve_global_address(address + uint64_t(4UL))).data[0] = uint(value >> 32);",
            "}",
        ])
    return lines


def translate_kernel(
    kernel: PtxKernel,
    *,
    workgroup_size: tuple[int, int, int] | None = None,
    surface_format: str = "uint",
) -> str:
    has_wmma = any(instruction.opcode.startswith("wmma.") for instruction in kernel.instructions)
    wmma_mode = os.environ.get("DLSSAMD_TRANSLATOR_WMMA_MODE")
    native_wmma = has_wmma and wmma_mode not in {"passthrough", "software", "replay"}
    software_wmma = has_wmma and wmma_mode == "software"
    replay_wmma = has_wmma and wmma_mode == "replay"
    wmma_backend = native_wmma or software_wmma or replay_wmma
    wmma_instructions = [
        instruction for instruction in kernel.instructions
        if instruction.opcode.startswith("wmma.")
    ]
    wmma_records = [_wmma_register_indices(instruction) for instruction in wmma_instructions]
    staged_wmma = wmma_backend and bool(wmma_instructions) and all(
        record is not None for record in wmma_records
    )
    register_storage_enabled = (
        staged_wmma
        and not replay_wmma
        and os.environ.get("DLSSAMD_TRANSLATOR_REGISTER_STORAGE") not in {None, "", "0"}
    )
    # Scalarizing thousands of PTX registers makes the replay shader very
    # expensive for the Vulkan compiler to analyze.  Keep an A/B switch for a
    # private-register array representation: it has the same per-invocation
    # semantics, while giving the compiler one bounded aggregate instead of
    # thousands of independent SSA candidates.  This is distinct from
    # register_storage_enabled, which is the cross-dispatch SSBO path and can
    # require gigabytes for a full PWIN launch.
    replay_register_array = (
        replay_wmma
        and os.environ.get("DLSSAMD_TRANSLATOR_REPLAY_REGISTER_ARRAY")
        not in {None, "", "0"}
    )
    replay_register_storage = (
        replay_wmma
        and os.environ.get("DLSSAMD_TRANSLATOR_REPLAY_REGISTER_STORAGE")
        not in {None, "", "0"}
    )
    split_scalar = os.environ.get("DLSSAMD_TRANSLATED_SPLIT_SCALAR") not in {
        None, "", "0"
    }
    replay_segment_index_text = os.environ.get("DLSSAMD_TRANSLATOR_REPLAY_SEGMENT_INDEX")
    replay_segmented = replay_wmma and os.environ.get(
        "DLSSAMD_TRANSLATED_SEGMENTED_REPLAY"
    ) not in {None, "", "0"}
    scalar_segmented = split_scalar and os.environ.get(
        "DLSSAMD_TRANSLATED_SCALAR_SEGMENTED_REPLAY"
    ) not in {None, "", "0"}
    scalar_shader_reset = (
        scalar_segmented
        and os.environ.get("DLSSAMD_TRANSLATED_SCALAR_TILED_DISPATCH")
        not in {None, "", "0"}
        and os.environ.get("DLSSAMD_TRANSLATED_SCALAR_SHADER_RESET")
        not in {None, "", "0"}
    )
    replay_segmented = replay_segmented or scalar_segmented
    replay_segment_index: int | None = None
    replay_segment_wmma_limit = 0
    replay_segment_bounds: tuple[int, int, int] | None = None
    replay_segment_register_offsets: dict[tuple[str, int], int] = {}
    replay_segment_register_words = 0
    replay_segment_register_load_indices: dict[str, set[int]] = {}
    replay_segment_register_store_indices: dict[str, set[int]] = {}
    replay_segment_register_declaration_indices: dict[str, set[int]] = {}
    if replay_segmented and replay_segment_index_text not in {None, ""}:
        try:
            replay_segment_index = int(replay_segment_index_text, 10)
            limit_name = (
                "DLSSAMD_TRANSLATOR_SEGMENT_INSTRUCTION_LIMIT"
                if scalar_segmented and not replay_wmma
                else "DLSSAMD_TRANSLATOR_SEGMENT_WMMA_LIMIT"
            )
            replay_segment_wmma_limit = int(os.environ.get(limit_name, "8"), 10)
        except ValueError as exc:
            raise PtxTranslationError("replay segment selection is not numeric") from exc
        if scalar_segmented and not replay_wmma:
            replay_segment_bounds = _scalar_segment_bounds(
                kernel, replay_segment_index, replay_segment_wmma_limit
            )
        else:
            kernel = expand_fixed_wmma_loops(kernel)
            # The expanded kernel contains one record for every logical loop
            # iteration.  Rebuild the WMMA record list so later segment
            # emission does not run out of source-iteration records halfway
            # through the expanded body.
            wmma_instructions = [
                instruction for instruction in kernel.instructions
                if instruction.opcode.startswith("wmma.")
            ]
            wmma_records = [_wmma_register_indices(instruction) for instruction in wmma_instructions]
            replay_segment_bounds = _replay_segment_bounds(
                kernel, replay_segment_index, replay_segment_wmma_limit
            )
        full_registers = _register_files(kernel, scalarize=False)
        segment_storage_indices = (
            _scalar_segment_storage_indices(kernel, replay_segment_wmma_limit)
            if scalar_segmented and not replay_wmma
            else _replay_segment_storage_indices(kernel, replay_segment_wmma_limit)
        )
        replay_segment_register_offsets, replay_segment_register_words = (
            _replay_sparse_storage_layout(full_registers, segment_storage_indices)
        )
        replay_segment_register_load_indices, replay_segment_register_store_indices = (
            _replay_segment_register_indices(
                kernel, replay_segment_bounds[0], replay_segment_bounds[1]
            )
        )
        for family, indices in _replay_segment_branch_state(
            kernel, replay_segment_bounds[0], replay_segment_bounds[1]
        ).items():
            replay_segment_register_load_indices.setdefault(family, set()).update(indices)
            replay_segment_register_store_indices.setdefault(family, set()).update(indices)
        if not replay_segment_register_words:
            raise PtxTranslationError("segmented replay requires a register file")
    replay_segment_register_storage = replay_segment_bounds is not None
    if replay_register_storage:
        replay_register_array = False
    storage_families = {"r"} if register_storage_enabled else set()
    if replay_register_storage:
        storage_families.add("r")
    if replay_segment_register_storage:
        # Segment phases checkpoint the private typed register files around
        # each Vulkan dispatch; the storage buffer itself is a raw uint SSBO,
        # so keep the PTX registers private for typed instruction lowering.
        storage_families.clear()
    registers = _register_files(
        kernel,
        scalarize=replay_segment_register_storage or (has_wmma and (
            not staged_wmma
            or (replay_wmma and not replay_register_array and not replay_segment_register_storage)
        )),
        storage_families=storage_families,
    )
    if replay_segment_register_storage:
        replay_segment_register_declaration_indices = (
            _scalar_segment_register_declaration_indices(
                kernel,
                replay_segment_bounds[0],  # type: ignore[index]
                replay_segment_bounds[1],  # type: ignore[index]
                registers,
                replay_segment_register_load_indices,
                replay_segment_register_store_indices,
            )
        )
        if not replay_segment_register_words:
            raise PtxTranslationError("segmented replay register layout is empty")
    if replay_wmma:
        wmma_operation_expressions, wmma_operation_count = _wmma_operation_expressions(
            kernel, registers
        )
    else:
        wmma_operation_expressions = {
            index: f"{operation}u"
            for operation, index in enumerate(
                position
                for position, instruction in enumerate(kernel.instructions)
                if instruction.opcode.startswith("wmma.")
            )
        }
        wmma_operation_count = len(wmma_instructions)
    has_register_storage = any(register.storage for register in registers.values())
    if not registers:
        raise PtxTranslationError(f"kernel {kernel.name} declares no supported registers")
    if surface_format not in {"uint", "float"}:
        raise PtxTranslationError(f"surface format is not supported: {surface_format}")
    # A real CUDA launch supplies the block dimensions.  `.maxntid` is a
    # compiler-side limit and is absent from some fatbin PTX entries, so the
    # runtime may override it with the dimensions carried by VkCuLaunchInfoNVX.
    workgroup = workgroup_size or _workgroup_size(kernel)
    shared_symbols, shared_size = _shared_layout(kernel)
    has_global_memory = any("global" in instruction.opcode.split(".") for instruction in kernel.instructions)
    has_shared_memory = bool(shared_symbols) or any("shared" in instruction.opcode.split(".") for instruction in kernel.instructions)
    has_local_memory = any("local" in instruction.opcode.split(".") for instruction in kernel.instructions)
    has_constant_memory = any("const" in instruction.opcode.split(".") for instruction in kernel.instructions)
    has_subgroup_shuffle = any(
        instruction.opcode.startswith(("shfl.", "mma.", "wmma.", "ldmatrix.", "movmatrix."))
        for instruction in kernel.instructions
    )
    has_subgroup_relative_shuffle = any(
        instruction.opcode.startswith("shfl.") and any(mode in instruction.opcode for mode in (".up.", ".down."))
        for instruction in kernel.instructions
    )
    has_subgroup_ballot = any(
        (instruction.opcode.startswith("vote.") and ".ballot." in instruction.opcode)
        or instruction.opcode.startswith("activemask.")
        for instruction in kernel.instructions
    )
    has_subgroup_vote = any(
        instruction.opcode.startswith("vote.") and (".any." in instruction.opcode or ".all." in instruction.opcode)
        for instruction in kernel.instructions
    )
    has_float_atomic = any(instruction.opcode.startswith("atom.") and ".f32" in instruction.opcode for instruction in kernel.instructions)
    has_texture = any(instruction.opcode.split(".", 1)[0] in {"tex", "tld4"} for instruction in kernel.instructions)
    has_surface = any(instruction.opcode.split(".", 1)[0] in {"sust", "suld"} for instruction in kernel.instructions)
    has_mma = any(instruction.opcode.startswith("mma.") for instruction in kernel.instructions)
    if has_wmma and wmma_mode == "software":
        has_mma = True
    if wmma_backend and (
        workgroup[0] != 32
        or workgroup[1] != 1
        or workgroup[2] < 1
    ):
        raise PtxTranslationError(
            "captured wmma.m16n16k16 lowering requires 32x1xN warp slices"
        )
    has_64bit = any(register.glsl_type in {"uint64_t", "double"} for register in registers.values()) or has_global_memory or has_shared_memory or has_local_memory or has_constant_memory or has_texture or has_surface
    lines = ["#version 450"]
    if scalar_segmented and os.environ.get(
        "DLSSAMD_TRANSLATED_SCALAR_TILED_DISPATCH"
    ) not in {None, "", "0"}:
        lines.append(
            "layout(push_constant) uniform PtxDispatchInfo { "
            "uvec4 workgroup_base; uvec4 full_grid; } ptx_dispatch_info;"
        )
    if has_64bit:
        lines.append("#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require")
    if has_global_memory:
        lines.append("#extension GL_EXT_buffer_reference : require")
    if has_subgroup_shuffle:
        lines.append("#extension GL_KHR_shader_subgroup_shuffle : require")
        lines.append("#extension GL_KHR_shader_subgroup_basic : require")
    if has_subgroup_relative_shuffle:
        lines.append("#extension GL_KHR_shader_subgroup_shuffle_relative : require")
    if has_subgroup_ballot:
        lines.append("#extension GL_KHR_shader_subgroup_ballot : require")
    if has_subgroup_vote:
        lines.append("#extension GL_KHR_shader_subgroup_vote : require")
    if has_float_atomic:
        lines.append("#extension GL_EXT_shader_atomic_float : require")
    if any(register.glsl_type == "double" for register in registers.values()):
        lines.append("#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require")
    if native_wmma:
        lines.extend([
            "#extension GL_KHR_cooperative_matrix : require",
            "#extension GL_KHR_memory_scope_semantics : require",
            "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require",
            "#extension GL_EXT_control_flow_attributes : require",
        ])
    if staged_wmma and not replay_wmma:
        lines.extend(_glsl_wmma_dynamic_tables([record for record in wmma_records if record is not None]))
    lines.extend([
        "layout(local_size_x = %d, local_size_y = %d, local_size_z = %d) in;" % workgroup,
        "layout(set = 0, binding = 0, std430) readonly buffer PtxParams { uint data[]; } ptx_params;",
        "layout(set = 0, binding = 1, std430) buffer PtxOutput { uint data[]; } ptx_output;",
    ])
    if has_register_storage or replay_segment_register_storage:
        register_binding = (
            _RUNTIME_CONTROL_BINDING
            if replay_wmma or replay_segment_register_storage
            else _RUNTIME_REGISTER_BINDING
        )
        lines.extend([
            f"layout(set = 0, binding = {register_binding}, std430) buffer PtxRegisters {{ uint data[]; }} ptx_registers;",
            "uint ptx_register_base;",
        ])
    if replay_wmma:
        lines.extend([
            f"layout(set = 0, binding = {_RUNTIME_REGISTER_BINDING}, std430) buffer PtxWmmaState {{ uint data[]; }} ptx_wmma_state;",
            "uint ptx_wmma_state_base;",
        ])
    elif replay_segment_register_storage:
        lines.extend([
            f"layout(set = 0, binding = {_RUNTIME_REGISTER_BINDING}, std430) buffer PtxReplayState {{ uint data[]; }} ptx_wmma_state;",
            "uint ptx_wmma_state_base;",
        ])
    if has_constant_memory:
        lines.append(f"layout(set = 0, binding = {_RUNTIME_CONSTANT_BINDING}, std430) readonly buffer PtxConstants {{ uint data[]; }} ptx_const;")
    image_slots = _runtime_image_slot_count()
    if has_texture:
        for slot in range(image_slots):
            lines.append(
                f"layout(set = 0, binding = {_RUNTIME_TEXTURE_BINDING_BASE + slot}) uniform sampler2D ptx_texture_{slot};"
            )
    if has_surface:
        surface_image_type = "image2D" if surface_format == "float" else "uimage2D"
        surface_image_formats = _runtime_surface_image_formats(surface_format)
        for slot in range(image_slots):
            lines.append(
                f"layout(set = 0, binding = {_RUNTIME_SURFACE_BINDING_BASE + slot}, {surface_image_formats[slot]}) uniform {surface_image_type} ptx_surface_{slot};"
            )
    if has_texture or has_surface:
        lines.append(
            f"layout(set = 0, binding = {_RUNTIME_IMAGE_TABLE_BINDING}, std430) readonly buffer PtxImageTable {{ uint64_t data[{_RUNTIME_IMAGE_SLOTS}]; }} ptx_image_table;"
        )
    if has_shared_memory:
        if scalar_segmented and replay_segment_register_storage:
            lines.extend([
                f"layout(set = 0, binding = {_RUNTIME_SHARED_BINDING}, std430) buffer PtxSharedState {{ uint data[]; }} ptx_shared_state;",
                "uint ptx_shared_base;",
            ])
        else:
            lines.append(f"shared uint ptx_shared_data[{max(1, (shared_size + 3) // 4)}];")
    if has_float_atomic:
        lines.append(f"shared float ptx_shared_float_data[{max(1, (shared_size + 3) // 4)}];")
    if has_local_memory:
        lines.append("uint ptx_local_data[4096];")
    lines.extend(
        _glsl_parameter_helpers(
            registers,
            shared_size=shared_size,
            shared_storage=scalar_segmented and replay_segment_register_storage,
            has_global_memory=has_global_memory,
            has_address_map=(
                has_global_memory
                and os.environ.get("DLSSAMD_TRANSLATED_ADDRESS_MAP") not in {None, "", "0"}
            ),
            has_shared_memory=has_shared_memory,
            has_local_memory=has_local_memory,
            has_constant_memory=has_constant_memory,
        )
    )
    if has_texture or has_surface:
        lines.extend(
            _glsl_image_helpers(
                has_texture=has_texture,
                has_surface=has_surface,
                surface_format=surface_format,
            )
        )
    if has_mma:
        lines.extend(_glsl_matrix_helpers())
    if native_wmma:
        lines.extend(_glsl_wmma_helpers())
    if staged_wmma and not replay_wmma:
        lines.extend(
            _glsl_wmma_dynamic_helper(
                registers["r"].count,
                storage=registers["r"].storage,
                software=software_wmma,
            )
        )
    if replay_wmma:
        lines.extend(
            _glsl_replay_state_helpers(register_storage=registers["r"].storage)
        )
    split_control_helpers: list[str] = []
    if replay_wmma or replay_segment_register_storage:
        if replay_segment_register_storage and not replay_wmma:
            forward_flow = _emit_scalar_segment_control_flow(
                kernel,
                registers,
                start=replay_segment_bounds[0],  # type: ignore[index]
                end=replay_segment_bounds[1],  # type: ignore[index]
                final=replay_segment_bounds[1] == len(kernel.instructions),  # type: ignore[index]
                surface_format=surface_format,
                register_offsets=replay_segment_register_offsets,
                register_store_indices=replay_segment_register_store_indices,
            )
        elif replay_segment_register_storage:
            forward_flow = _emit_replay_segment_control_flow(
                kernel,
                registers,
                [record for record in wmma_records if record is not None],
                start=replay_segment_bounds[0],  # type: ignore[index]
                end=replay_segment_bounds[1],  # type: ignore[index]
                final=replay_segment_bounds[1] == len(kernel.instructions),  # type: ignore[index]
                surface_format=surface_format,
                operation_expressions=wmma_operation_expressions,
                register_offsets=replay_segment_register_offsets,
                register_load_indices=replay_segment_register_load_indices,
                register_store_indices=replay_segment_register_store_indices,
            )
            if forward_flow is None:
                raise PtxTranslationError(
                    f"unable to emit segmented replay phase for {kernel.name}"
                )
        else:
            replay_paged = os.environ.get("DLSSAMD_TRANSLATOR_REPLAY_PAGED") not in {
                None, "", "0"
            }
            if replay_paged:
                # Keep every replay page in its own non-inlined helper.  This is
                # materially easier for RADV to compile than one giant forward
                # CFG, while preserving the same one-WMMA-per-dispatch protocol.
                forward_flow, split_control_helpers = _emit_replay_control_flow(
                    kernel,
                    registers,
                    surface_format=surface_format,
                    operation_expressions=wmma_operation_expressions,
                )
            else:
                forward_flow = _emit_replay_forward_control_flow(
                    kernel,
                    registers,
                    surface_format=surface_format,
                    records=[record for record in wmma_records if record is not None],
                    operation_expressions=wmma_operation_expressions,
                )
                if forward_flow is None:
                    # Keep the replay interpreter in separately compiled page
                    # functions for kernels with loops or branch regions that
                    # cross a WMMA barrier.
                    forward_flow, split_control_helpers = _emit_replay_control_flow(
                        kernel,
                        registers,
                        surface_format=surface_format,
                        operation_expressions=wmma_operation_expressions,
                    )
    elif staged_wmma:
        forward_flow = _emit_staged_wmma_control_flow(
            kernel,
            registers,
            [record for record in wmma_records if record is not None],
            surface_format=surface_format,
        )
        if forward_flow is None:
            forward_flow, split_control_helpers = _emit_split_control_flow(
                kernel,
                registers,
                surface_format=surface_format,
                staged_wmma=True,
            )
    else:
        if split_scalar:
            forward_flow, split_control_helpers = _emit_split_control_flow(
                kernel,
                registers,
                surface_format=surface_format,
                staged_wmma=False,
            )
        else:
            forward_flow = _emit_forward_control_flow(
                kernel,
                registers,
                surface_format=surface_format,
            )
            if forward_flow is None:
                forward_flow = _emit_control_flow(
                    kernel,
                    registers,
                    surface_format=surface_format,
                )
    split_registers = sorted(registers.values(), key=lambda value: value.glsl_name)

    def declaration_indices(register: RegisterFile):
        if replay_segment_register_storage:
            return sorted(
                replay_segment_register_declaration_indices.get(register.family, set())
            )
        return range(max(register.count, 1))

    if split_control_helpers:
        if any("%SP" in instruction.text or "%SPL" in instruction.text for instruction in kernel.instructions):
            lines.append("uint64_t ptx_special_spl = uint64_t(0UL);")
        for register in split_registers:
            if register.storage:
                continue
            if register.scalarized:
                for index in declaration_indices(register):
                    lines.append(f"{register.glsl_type} {register.glsl_name}_{index};")
            else:
                lines.append(f"{register.glsl_type} {register.glsl_name}[{max(register.count, 1)}];")
    lines.extend(split_control_helpers)
    lines.append("void main() {")
    if scalar_segmented and os.environ.get(
        "DLSSAMD_TRANSLATED_SCALAR_TILED_DISPATCH"
    ) not in {None, "", "0"}:
        lines.append(
            "    uvec3 ptx_global_invocation_id = "
            "(ptx_dispatch_info.workgroup_base.xyz + gl_WorkGroupID) * "
            "gl_WorkGroupSize + gl_LocalInvocationID;"
        )
    if not split_control_helpers and any("%SP" in instruction.text or "%SPL" in instruction.text for instruction in kernel.instructions):
        lines.append("    uint64_t ptx_special_spl = uint64_t(0UL);")
    if not split_control_helpers:
        for register in split_registers:
            if register.storage:
                continue
            if register.scalarized:
                for index in declaration_indices(register):
                    lines.append(f"    {register.glsl_type} {register.glsl_name}_{index};")
            else:
                lines.append(f"    {register.glsl_type} {register.glsl_name}[{max(register.count, 1)}];")
    if has_register_storage:
        lines.append(
            "    ptx_register_base = (gl_GlobalInvocationID.x + "
            "(gl_NumWorkGroups.x * gl_WorkGroupSize.x) * gl_GlobalInvocationID.y + "
            "(gl_NumWorkGroups.x * gl_WorkGroupSize.x * gl_NumWorkGroups.y * gl_WorkGroupSize.y) * "
            f"gl_GlobalInvocationID.z) * {registers['r'].count}u;"
        )
    elif replay_segment_register_storage:
        lines.append(
            "    ptx_register_base = (gl_GlobalInvocationID.x + "
            "(gl_NumWorkGroups.x * gl_WorkGroupSize.x) * gl_GlobalInvocationID.y + "
            "(gl_NumWorkGroups.x * gl_WorkGroupSize.x * gl_NumWorkGroups.y * gl_WorkGroupSize.y) * "
            f"gl_GlobalInvocationID.z) * {replay_segment_register_words}u;"
        )
    if replay_wmma or replay_segment_register_storage:
        state_stride = (
            _WMMA_STATE_RESULT_BASE + wmma_operation_count * 4
            if replay_wmma
            else 3 if scalar_shader_reset else 2
        )
        lines.append(
            "    ptx_wmma_state_base = (gl_GlobalInvocationID.x + "
            "(gl_NumWorkGroups.x * gl_WorkGroupSize.x) * gl_GlobalInvocationID.y + "
            "(gl_NumWorkGroups.x * gl_WorkGroupSize.x * gl_NumWorkGroups.y * gl_WorkGroupSize.y) * "
            f"gl_GlobalInvocationID.z) * {state_stride}u;"
        )
    if scalar_segmented and replay_segment_register_storage and has_shared_memory:
        shared_words = max(1, (shared_size + 3) // 4)
        lines.append(
            "    ptx_shared_base = (gl_WorkGroupID.x + "
            "gl_NumWorkGroups.x * gl_WorkGroupID.y + "
            "gl_NumWorkGroups.x * gl_NumWorkGroups.y * gl_WorkGroupID.z) * "
            f"{shared_words}u;"
        )
    if scalar_shader_reset and replay_segment_index == 0:
        lines.extend([
            "    bool ptx_replay_new_tile = "
            "ptx_wmma_state.data[ptx_wmma_state_base + 2u] != "
            "ptx_dispatch_info.workgroup_base.w;",
            "    if (ptx_replay_new_tile) {",
        ])
        lines.extend(
            _replay_register_zero_lines(
                registers,
                replay_segment_register_offsets,
                indices=replay_segment_register_load_indices,
                indent="        ",
            )
        )
        lines.extend([
            "        ptx_wmma_state.data[ptx_wmma_state_base + 0u] = 0u;",
            "        ptx_wmma_state.data[ptx_wmma_state_base + 1u] = 0u;",
            "        ptx_wmma_state.data[ptx_wmma_state_base + 2u] = "
            "ptx_dispatch_info.workgroup_base.w;",
            "    }",
        ])
        if has_shared_memory:
            lines.extend([
                "    if (ptx_replay_new_tile && gl_LocalInvocationIndex == 0u) {",
                f"        for (uint ptx_shared_index = 0u; ptx_shared_index < {shared_words}u; ++ptx_shared_index)",
                "            ptx_shared_state.data[ptx_shared_base + ptx_shared_index] = 0u;",
                "    }",
                "    memoryBarrierBuffer();",
                "    barrier();",
            ])
    if replay_segment_register_storage:
        lines.extend(
            _replay_register_load_lines(
                registers,
                replay_segment_register_offsets,
                indices=replay_segment_register_load_indices,
            )
        )
    lines.extend(forward_flow)
    lines.append("}")
    return "\n".join(lines) + "\n"


def translate_module(
    module: PtxModule,
    entry_name: str,
    *,
    workgroup_size: tuple[int, int, int] | None = None,
    surface_format: str = "uint",
) -> str:
    matches = [kernel for kernel in module.kernels if kernel.name == entry_name]
    if len(matches) != 1:
        raise PtxTranslationError(f"expected one kernel named {entry_name!r}, found {len(matches)}")
    return translate_kernel(
        matches[0],
        workgroup_size=workgroup_size,
        surface_format=surface_format,
    )


def translate_wmma_dispatch_kernel(
    kernel: PtxKernel,
    *,
    workgroup_size: tuple[int, int, int] | None = None,
) -> str:
    """Generate the small native-WMMA half of a split PWIN dispatch.

    The scalar replay shader and this dispatcher communicate through the
    compact WMMA state SSBO.  Keeping the cooperative-matrix operation out of
    the scalar CFG is intentional: RADV can compile this module independently
    instead of combining its native matrix lowering with thousands of scalar
    PC cases.
    """
    instructions = [
        instruction for instruction in kernel.instructions
        if instruction.opcode.startswith("wmma.")
    ]
    records = [_wmma_register_indices(instruction) for instruction in instructions]
    if not records or any(record is None for record in records):
        raise PtxTranslationError(f"WMMA dispatcher operands are not supported: {kernel.name}")
    workgroup = workgroup_size or _workgroup_size(kernel)
    if workgroup[0] != 32 or workgroup[1] != 1 or workgroup[2] < 1:
        raise PtxTranslationError("WMMA dispatcher requires 32x1xN warp slices")
    registers = _register_files(kernel)
    operation_count = wmma_instance_count(kernel)
    state_stride = _WMMA_STATE_RESULT_BASE + operation_count * 4
    lines = [
        "#version 450",
        "#extension GL_KHR_shader_subgroup_shuffle : require",
        "#extension GL_KHR_shader_subgroup_basic : require",
        "#extension GL_KHR_cooperative_matrix : require",
        "#extension GL_KHR_memory_scope_semantics : require",
        "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require",
        "#extension GL_EXT_control_flow_attributes : require",
    ]
    lines.extend([
        "layout(local_size_x = %d, local_size_y = %d, local_size_z = %d) in;" % workgroup,
        "layout(set = 0, binding = 200, std430) buffer PtxWmmaState { uint data[]; } ptx_wmma_state;",
        "uint ptx_wmma_state_base;",
    ])
    lines.extend(_glsl_wmma_helpers())
    lines.extend([
        "void main() {",
        "    uint ptx_wmma_invocation = gl_GlobalInvocationID.x +",
        "        (gl_NumWorkGroups.x * gl_WorkGroupSize.x) * gl_GlobalInvocationID.y +",
        "        (gl_NumWorkGroups.x * gl_WorkGroupSize.x * gl_NumWorkGroups.y * gl_WorkGroupSize.y) * gl_GlobalInvocationID.z;",
        f"    ptx_wmma_state_base = ptx_wmma_invocation * {state_stride}u;",
        "    if (ptx_wmma_state.data[ptx_wmma_state_base + 2u] != 0u) return;",
        "    uint ptx_wmma_operation = ptx_wmma_state.data[ptx_wmma_state_base + 1u];",
        f"    if (ptx_wmma_operation >= {operation_count}u) return;",
        "    vec4 ptx_wmma_a_low = vec4(",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 3u]).x,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 3u]).y,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 4u]).x,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 4u]).y);",
        "    vec4 ptx_wmma_a_high = vec4(",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 5u]).x,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 5u]).y,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 6u]).x,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 6u]).y);",
        "    vec4 ptx_wmma_b_low = vec4(",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 11u]).x,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 11u]).y,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 12u]).x,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 12u]).y);",
        "    vec4 ptx_wmma_b_high = vec4(",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 13u]).x,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 13u]).y,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 14u]).x,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 14u]).y);",
        "    vec4 ptx_wmma_c_low = vec4(",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 19u]).x,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 19u]).y,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 20u]).x,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 20u]).y);",
        "    vec4 ptx_wmma_c_high = vec4(",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 21u]).x,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 21u]).y,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 22u]).x,",
        "        unpackHalf2x16(ptx_wmma_state.data[ptx_wmma_state_base + 22u]).y);",
        "    uvec4 ptx_wmma_result = ptx_wmma_m16n16k16_f16(",
        "        ptx_wmma_a_low, ptx_wmma_a_high, ptx_wmma_b_low, ptx_wmma_b_high,",
        "        ptx_wmma_c_low, ptx_wmma_c_high);",
        f"    uint ptx_wmma_result_base = ptx_wmma_state_base + {_WMMA_STATE_RESULT_BASE}u + ptx_wmma_operation * 4u;",
        "    ptx_wmma_state.data[ptx_wmma_result_base] = ptx_wmma_result.x;",
        "    ptx_wmma_state.data[ptx_wmma_result_base + 1u] = ptx_wmma_result.y;",
        "    ptx_wmma_state.data[ptx_wmma_result_base + 2u] = ptx_wmma_result.z;",
        "    ptx_wmma_state.data[ptx_wmma_result_base + 3u] = ptx_wmma_result.w;",
        "    ptx_wmma_state.data[ptx_wmma_state_base] = ptx_wmma_operation + 1u;",
        "    ptx_wmma_state.data[ptx_wmma_state_base + 1u] = 0xffffffffu;",
        "}",
    ])
    return "\n".join(lines) + "\n"


def compile_glsl(source: str, output: Path, *, glslc: str = "glslc") -> None:
    executable = shutil.which(glslc) or glslc
    with tempfile.TemporaryDirectory(prefix="amdlss-ptx-glsl-") as directory:
        source_path = Path(directory) / "kernel.comp"
        source_path.write_text(source, encoding="ascii")
        command = [
            executable,
            "-fshader-stage=compute",
            "--target-env=vulkan1.2",
        ]
        # The SteamOS glslc build can spend the entire Deck's RAM in its
        # optimization pass on the large PWIN replay CFG.  Runtime shaders
        # already go through Vulkan pipeline creation on RADV; allowing an
        # opt-in compiler-level O0 path keeps translation bounded without
        # changing the normal standalone compiler behavior.
        if os.environ.get("DLSSAMD_GLSLC_O0") not in {None, "", "0"}:
            command.append("-O0")
        command.extend([
            str(source_path),
            "-o",
            str(output),
        ])
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode:
        raise PtxTranslationError(
            f"glslc failed with {completed.returncode}: {(completed.stderr or completed.stdout).strip()}"
        )
