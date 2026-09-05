"""Length-bounded CUDA fatbin/PTX inspection primitives.

The supplied NGX DLL uses the public fatbin magic/header and the newer entry
layout observed in current CUDA binaries. PTX entries in that DLL are raw LZ4
blocks. This module intentionally extracts metadata by default; callers must
opt in before writing decoded PTX to disk.
"""

from __future__ import annotations

import hashlib
import ctypes
import ctypes.util
import re
import struct
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


FATBIN_MAGIC = 0xBA55ED50
FATBIN_MAGIC_BYTES = struct.pack("<I", FATBIN_MAGIC)
FATBIN_VERSION = 1

FATBIN_KIND_PTX = 0x0001
FATBIN_KIND_ELF = 0x0002

FATBIN_FLAG_COMPRESS = 0x1000
FATBIN_FLAG_LZ4 = 0x2000
FATBIN_FLAG_LZ4_ALT = 0x4000
FATBIN_FLAG_ZSTD = 0x8000

MAX_INPUT_SIZE = 2 * 1024 * 1024 * 1024
MAX_FATBINS = 4096
MAX_ENTRIES_PER_FATBIN = 4096
MAX_DECOMPRESSED_PAYLOAD = 128 * 1024 * 1024


class FatbinError(ValueError):
    """Raised when a fatbin candidate is malformed or unsafe to parse."""


@dataclass(frozen=True)
class FatbinHeader:
    offset: int
    version: int
    header_size: int
    fat_size: int
    end_offset: int


@dataclass(frozen=True)
class FatbinEntry:
    offset: int
    kind: int
    version: int
    header_size: int
    padded_payload_size: int
    payload_size: int
    code_version_major: int
    code_version_minor: int
    architecture: int
    flags: int
    uncompressed_size: int
    payload_offset: int
    payload: bytes

    @property
    def compression(self) -> str:
        if self.flags & FATBIN_FLAG_ZSTD:
            return "zstd"
        if self.flags & (FATBIN_FLAG_LZ4 | FATBIN_FLAG_LZ4_ALT):
            return "lz4"
        if self.flags & FATBIN_FLAG_COMPRESS:
            return "zlib-or-legacy"
        return "none"

    @property
    def architecture_name(self) -> Optional[str]:
        return f"sm_{self.architecture}" if self.architecture else None


@dataclass(frozen=True)
class FatbinRecord:
    header: FatbinHeader
    entries: tuple[FatbinEntry, ...]

    @property
    def raw_size(self) -> int:
        return self.header.end_offset - self.header.offset


@dataclass(frozen=True)
class PtxAnalysis:
    version: str
    target: str
    address_size: int
    entry_names: tuple[str, ...]
    opcode_counts: dict[str, int]
    instruction_count: int
    features: tuple[str, ...]
    source: bytes


def _ensure_range(data: bytes, offset: int, size: int, label: str) -> None:
    if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
        raise FatbinError(f"{label} is outside the input: offset=0x{offset:x}, size=0x{size:x}")


def _u16(data: bytes, offset: int, label: str) -> int:
    _ensure_range(data, offset, 2, label)
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int, label: str) -> int:
    _ensure_range(data, offset, 4, label)
    return struct.unpack_from("<I", data, offset)[0]


def _u64(data: bytes, offset: int, label: str) -> int:
    _ensure_range(data, offset, 8, label)
    return struct.unpack_from("<Q", data, offset)[0]


def _checked_add(left: int, right: int, label: str) -> int:
    result = left + right
    if result < left:
        raise FatbinError(f"integer overflow while calculating {label}")
    return result


def lz4_decompress_block(
    compressed: bytes,
    expected_size: int,
    *,
    max_output_size: int = MAX_DECOMPRESSED_PAYLOAD,
) -> bytes:
    """Decode a raw LZ4 block with strict bounds checks.

    CUDA fatbin PTX entries use a raw block, not an LZ4 frame. The decoder is
    intentionally self-contained so the inspection tool works on the stock
    Python installation on SteamOS.
    """

    if expected_size < 0 or expected_size > max_output_size:
        raise FatbinError(f"LZ4 output size is not allowed: {expected_size}")

    output = bytearray()

    def read_length(initial: int) -> int:
        length = initial
        if initial != 15:
            return length
        while True:
            if source_index_ref[0] >= len(compressed):
                raise FatbinError("truncated LZ4 length extension")
            value = compressed[source_index_ref[0]]
            source_index_ref[0] += 1
            length += value
            if value != 255:
                return length

    # A one-item mutable holder keeps the helper local without hiding the
    # cursor mutation in a second parser abstraction.
    source_index_ref = [0]
    while source_index_ref[0] < len(compressed):
        token = compressed[source_index_ref[0]]
        source_index_ref[0] += 1

        literal_length = read_length(token >> 4)
        literal_end = source_index_ref[0] + literal_length
        if literal_end > len(compressed):
            raise FatbinError("LZ4 literal run exceeds the compressed payload")
        output.extend(compressed[source_index_ref[0] : literal_end])
        source_index_ref[0] = literal_end
        if len(output) > max_output_size:
            raise FatbinError("LZ4 output exceeds the configured limit")

        # A final literal-only sequence is valid and has no match tuple.
        if source_index_ref[0] == len(compressed):
            break

        if source_index_ref[0] + 2 > len(compressed):
            raise FatbinError("truncated LZ4 match offset")
        match_offset = compressed[source_index_ref[0]] | (
            compressed[source_index_ref[0] + 1] << 8
        )
        source_index_ref[0] += 2
        if match_offset == 0 or match_offset > len(output):
            raise FatbinError(f"invalid LZ4 match offset: {match_offset}")

        match_length = read_length(token & 0x0F) + 4
        if len(output) + match_length > max_output_size:
            raise FatbinError("LZ4 output exceeds the configured limit")
        for _ in range(match_length):
            output.append(output[-match_offset])

    if len(output) != expected_size:
        raise FatbinError(
            f"LZ4 output size mismatch: expected {expected_size}, got {len(output)}"
        )
    return bytes(output)


def _decompress_zlib(compressed: bytes, expected_size: int) -> bytes:
    if expected_size <= 0 or expected_size > MAX_DECOMPRESSED_PAYLOAD:
        raise FatbinError(f"zlib output size is not allowed: {expected_size}")
    decompressor = zlib.decompressobj()
    result = decompressor.decompress(compressed, MAX_DECOMPRESSED_PAYLOAD + 1)
    if len(result) > MAX_DECOMPRESSED_PAYLOAD or decompressor.unconsumed_tail:
        raise FatbinError("zlib output exceeds the configured limit")
    result += decompressor.flush()
    if len(result) > MAX_DECOMPRESSED_PAYLOAD:
        raise FatbinError("zlib output exceeds the configured limit")
    if len(result) != expected_size:
        raise FatbinError(
            f"zlib output size mismatch: expected {expected_size}, got {len(result)}"
        )
    if decompressor.unused_data:
        raise FatbinError("zlib payload has trailing data")
    return result


def _decompress_zstd(compressed: bytes, expected_size: int) -> bytes:
    """Decode a Zstandard frame through the system libzstd when available."""

    if expected_size <= 0 or expected_size > MAX_DECOMPRESSED_PAYLOAD:
        raise FatbinError(f"Zstandard output size is not allowed: {expected_size}")
    library_name = ctypes.util.find_library("zstd")
    if not library_name:
        raise FatbinError("Zstandard PTX entries require libzstd")
    try:
        library = ctypes.CDLL(library_name)
        decompress = library.ZSTD_decompress
        is_error = library.ZSTD_isError
        error_name = library.ZSTD_getErrorName
    except (AttributeError, OSError) as exc:
        raise FatbinError("Zstandard PTX entries require a compatible libzstd") from exc

    decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    decompress.restype = ctypes.c_size_t
    is_error.argtypes = [ctypes.c_size_t]
    is_error.restype = ctypes.c_uint
    error_name.argtypes = [ctypes.c_size_t]
    error_name.restype = ctypes.c_char_p

    output = ctypes.create_string_buffer(expected_size)
    source = ctypes.create_string_buffer(compressed)
    result = decompress(output, expected_size, source, len(compressed))
    if is_error(result):
        detail = error_name(result)
        message = detail.decode("ascii", "replace") if detail else "unknown error"
        raise FatbinError(f"Zstandard decompression failed: {message}")
    if result != expected_size:
        raise FatbinError(
            f"Zstandard output size mismatch: expected {expected_size}, got {result}"
        )
    return output.raw[:expected_size]


def decompress_ptx_entry(entry: FatbinEntry) -> bytes:
    if entry.kind != FATBIN_KIND_PTX:
        raise FatbinError("attempted to decompress a non-PTX entry")

    if entry.compression == "lz4":
        return lz4_decompress_block(entry.payload, entry.uncompressed_size)
    if entry.compression == "zlib-or-legacy":
        try:
            return _decompress_zlib(entry.payload, entry.uncompressed_size)
        except (FatbinError, zlib.error) as exc:
            raise FatbinError(
                "compressed PTX uses an unsupported legacy compression variant"
            ) from exc
    if entry.compression == "zstd":
        return _decompress_zstd(entry.payload, entry.uncompressed_size)

    if entry.uncompressed_size and len(entry.payload) != entry.uncompressed_size:
        raise FatbinError(
            "uncompressed PTX size mismatch: "
            f"expected {entry.uncompressed_size}, got {len(entry.payload)}"
        )
    if len(entry.payload) > MAX_DECOMPRESSED_PAYLOAD:
        raise FatbinError("PTX payload exceeds the configured limit")
    return entry.payload


def parse_fatbin_at(data: bytes, offset: int) -> FatbinRecord:
    """Parse one validated fatbin beginning at ``offset``."""

    _ensure_range(data, offset, 16, "fatbin header")
    magic = _u32(data, offset, "fatbin magic")
    if magic != FATBIN_MAGIC:
        raise FatbinError(f"invalid fatbin magic at 0x{offset:x}")

    version = _u16(data, offset + 4, "fatbin version")
    header_size = _u16(data, offset + 6, "fatbin header size")
    fat_size = _u64(data, offset + 8, "fatbin size")
    if version != FATBIN_VERSION:
        raise FatbinError(f"unsupported fatbin version: {version}")
    if header_size < 16 or header_size % 8:
        raise FatbinError(f"invalid fatbin header size: {header_size}")
    if fat_size == 0 or fat_size > MAX_INPUT_SIZE:
        raise FatbinError(f"invalid fatbin payload size: {fat_size}")

    end_offset = _checked_add(offset, header_size, "fatbin end")
    end_offset = _checked_add(end_offset, fat_size, "fatbin end")
    if end_offset > len(data):
        raise FatbinError(f"fatbin extends past input: end=0x{end_offset:x}")

    header = FatbinHeader(offset, version, header_size, fat_size, end_offset)
    entries: list[FatbinEntry] = []
    cursor = offset + header_size

    while cursor < end_offset:
        if len(entries) >= MAX_ENTRIES_PER_FATBIN:
            raise FatbinError("fatbin entry count exceeds the configured limit")
        _ensure_range(data, cursor, 0x20, "fatbin entry header")
        kind = _u16(data, cursor, "entry kind")
        entry_version = _u16(data, cursor + 2, "entry version")
        entry_header_size = _u32(data, cursor + 4, "entry header size")
        padded_payload_size = _u64(data, cursor + 8, "entry padded payload size")
        if entry_header_size < 0x20 or entry_header_size % 8:
            raise FatbinError(
                f"invalid entry header size at 0x{cursor:x}: {entry_header_size}"
            )
        if padded_payload_size % 8:
            raise FatbinError(
                f"entry payload is not 8-byte aligned at 0x{cursor:x}"
            )

        payload_offset = _checked_add(cursor, entry_header_size, "entry payload")
        entry_end = _checked_add(
            payload_offset, padded_payload_size, "entry end"
        )
        if entry_end > end_offset:
            raise FatbinError(f"entry extends past its fatbin at 0x{cursor:x}")

        declared_payload_size = _u32(data, cursor + 0x10, "entry payload size")
        if declared_payload_size == 0 or declared_payload_size > padded_payload_size:
            payload_size = padded_payload_size
        else:
            payload_size = declared_payload_size

        code_version_minor = _u16(data, cursor + 0x18, "code version minor")
        code_version_major = _u16(data, cursor + 0x1A, "code version major")
        architecture = _u32(data, cursor + 0x1C, "entry architecture")
        flags = _u64(data, cursor + 0x28, "entry flags") if entry_header_size >= 0x30 else 0
        uncompressed_size = (
            _u64(data, cursor + 0x38, "entry uncompressed size")
            if entry_header_size >= 0x40
            else 0
        )

        if kind == FATBIN_KIND_PTX and uncompressed_size > MAX_DECOMPRESSED_PAYLOAD:
            raise FatbinError(
                f"PTX entry declares an oversized output: {uncompressed_size}"
            )

        _ensure_range(data, payload_offset, payload_size, "entry payload")
        payload = bytes(data[payload_offset : payload_offset + payload_size])
        entries.append(
            FatbinEntry(
                offset=cursor,
                kind=kind,
                version=entry_version,
                header_size=entry_header_size,
                padded_payload_size=padded_payload_size,
                payload_size=payload_size,
                code_version_major=code_version_major,
                code_version_minor=code_version_minor,
                architecture=architecture,
                flags=flags,
                uncompressed_size=uncompressed_size,
                payload_offset=payload_offset,
                payload=payload,
            )
        )
        cursor = entry_end

    if cursor != end_offset:
        raise FatbinError(f"fatbin entry walk ended at 0x{cursor:x}, expected 0x{end_offset:x}")
    return FatbinRecord(header, tuple(entries))


def scan_fatbins(data: bytes) -> tuple[list[FatbinRecord], list[str]]:
    """Find and validate fatbins embedded in arbitrary host-binary bytes."""

    if len(data) > MAX_INPUT_SIZE:
        raise FatbinError(f"input exceeds the configured limit: {len(data)} bytes")

    records: list[FatbinRecord] = []
    errors: list[str] = []
    search_offset = 0
    while search_offset < len(data):
        offset = data.find(FATBIN_MAGIC_BYTES, search_offset)
        if offset < 0:
            break
        try:
            record = parse_fatbin_at(data, offset)
        except FatbinError as exc:
            errors.append(f"0x{offset:x}: {exc}")
            search_offset = offset + 1
            continue
        records.append(record)
        if len(records) > MAX_FATBINS:
            raise FatbinError("fatbin count exceeds the configured limit")
        search_offset = max(offset + len(FATBIN_MAGIC_BYTES), record.header.end_offset)
    return records, errors


_VERSION_RE = re.compile(rb"(?m)^\s*\.version\s+([0-9]+(?:\.[0-9]+)?)")
_TARGET_RE = re.compile(rb"(?m)^\s*\.target\s+([^\r\n]+)")
_ADDRESS_SIZE_RE = re.compile(rb"(?m)^\s*\.address_size\s+([0-9]+)")
_ENTRY_RE = re.compile(rb"\.entry\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
_OPCODE_RE = re.compile(r"^(?:@!?[%A-Za-z0-9_.$]+\s+)?([A-Za-z][A-Za-z0-9_.]*)\b")
_LABEL_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*:\s*")


def _clean_ptx_source(source: bytes) -> bytes:
    if not source:
        raise FatbinError("PTX payload is empty")
    source = source.rstrip(b"\x00")
    if b"\x00" in source:
        raise FatbinError("PTX payload contains an embedded NUL")
    try:
        source.decode("ascii")
    except UnicodeDecodeError as exc:
        raise FatbinError("PTX payload is not ASCII text") from exc
    return source


def _opcode_counts(source: bytes) -> Counter[str]:
    text = source.decode("ascii")
    counts: Counter[str] = Counter()
    in_block_comment = False
    for raw_line in text.splitlines():
        line = raw_line
        if in_block_comment:
            end = line.find("*/")
            if end < 0:
                continue
            line = line[end + 2 :]
            in_block_comment = False
        while "/*" in line:
            start = line.find("/*")
            end = line.find("*/", start + 2)
            if end < 0:
                line = line[:start]
                in_block_comment = True
                break
            line = line[:start] + line[end + 2 :]
        line = line.split("//", 1)[0].strip()
        if not line or line.startswith((".", "{", "}", ";")):
            continue
        line = _LABEL_RE.sub("", line).strip()
        if not line:
            continue
        match = _OPCODE_RE.match(line)
        if match:
            counts[match.group(1)] += 1
    return counts


def analyze_ptx(source: bytes) -> PtxAnalysis:
    source = _clean_ptx_source(source)
    version_match = _VERSION_RE.search(source)
    target_match = _TARGET_RE.search(source)
    address_match = _ADDRESS_SIZE_RE.search(source)
    entry_matches = _ENTRY_RE.findall(source)
    if not version_match or not target_match or not address_match:
        raise FatbinError("PTX is missing a required version, target, or address-size directive")
    if not entry_matches:
        raise FatbinError("PTX contains no .entry function")

    try:
        version = version_match.group(1).decode("ascii")
        target = target_match.group(1).decode("ascii").strip()
        address_size = int(address_match.group(1))
        entry_names = tuple(name.decode("ascii") for name in entry_matches)
    except (UnicodeDecodeError, ValueError) as exc:
        raise FatbinError("PTX metadata is malformed") from exc

    opcode_counts = _opcode_counts(source)
    features: set[str] = set()
    opcode_names = tuple(opcode_counts)
    if any(name.startswith(("mma", "wmma")) for name in opcode_names):
        features.add("matrix")
    if b"e4m3" in source or b"e5m2" in source or b"fp8" in source.lower():
        features.add("fp8")
    if any(name.startswith(("tex.", "tld")) for name in opcode_names):
        features.add("texture")
    if any(name.startswith(("sust", "suld")) for name in opcode_names):
        features.add("surface")
    if any(name.startswith("bar.") for name in opcode_names):
        features.add("barrier")
    if any(name.startswith(("shfl.", "vote.")) for name in opcode_names):
        features.add("shuffle")
    if any(name.startswith("cp.async") for name in opcode_names):
        features.add("cp_async")
    if any(name.startswith(("atom.", "red.")) for name in opcode_names):
        features.add("atomics")

    return PtxAnalysis(
        version=version,
        target=target,
        address_size=address_size,
        entry_names=entry_names,
        opcode_counts=dict(opcode_counts),
        instruction_count=sum(opcode_counts.values()),
        features=tuple(sorted(features)),
        source=source,
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _entry_kind_name(kind: int) -> str:
    return {FATBIN_KIND_PTX: "ptx", FATBIN_KIND_ELF: "elf"}.get(kind, f"unknown_{kind:#x}")


def inspect_bytes(
    data: bytes,
    *,
    file_name: str = "input",
    extract_dir: Optional[Path] = None,
) -> dict:
    records, scan_errors = scan_fatbins(data)
    if not records:
        detail = f"; first scan error: {scan_errors[0]}" if scan_errors else ""
        raise FatbinError(f"no valid CUDA fatbins found in {file_name}{detail}")

    if extract_dir is not None:
        extract_dir.mkdir(parents=True, exist_ok=True)

    module_rows: list[dict] = []
    all_opcodes: Counter[str] = Counter()
    all_entry_names: list[str] = []
    feature_modules: Counter[str] = Counter()
    ptx_versions: Counter[str] = Counter()
    ptx_targets: Counter[str] = Counter()
    compression_modes: Counter[str] = Counter()
    fatbin_kind_counts: Counter[str] = Counter()
    fatbins_with_ptx = 0
    ptx_entry_count = 0
    ptx_total_size = 0
    ptx_max_size = 0

    for fatbin_index, record in enumerate(records):
        for entry in record.entries:
            fatbin_kind_counts[_entry_kind_name(entry.kind)] += 1
        ptx_entries = [entry for entry in record.entries if entry.kind == FATBIN_KIND_PTX]
        if ptx_entries:
            fatbins_with_ptx += 1
        for ptx_index, entry in enumerate(ptx_entries):
            source = decompress_ptx_entry(entry)
            analysis = analyze_ptx(source)
            all_opcodes.update(analysis.opcode_counts)
            all_entry_names.extend(analysis.entry_names)
            for feature in analysis.features:
                feature_modules[feature] += 1
            ptx_versions[analysis.version] += 1
            ptx_targets[analysis.target] += 1
            compression_modes[entry.compression] += 1
            ptx_entry_count += len(analysis.entry_names)
            ptx_total_size += len(source)
            ptx_max_size = max(ptx_max_size, len(source))

            row = {
                "fatbin_index": fatbin_index,
                "fatbin_offset": record.header.offset,
                "fatbin_size": record.raw_size,
                "entry_offset": entry.offset,
                "entry_kind": _entry_kind_name(entry.kind),
                "entry_header_size": entry.header_size,
                "payload_offset": entry.payload_offset,
                "compressed_size": entry.payload_size,
                "uncompressed_size": len(source),
                "declared_uncompressed_size": entry.uncompressed_size,
                "compression": entry.compression,
                "flags": f"0x{entry.flags:x}",
                "code_version": f"{entry.code_version_major}.{entry.code_version_minor}",
                "architecture": entry.architecture_name,
                "ptx_sha256": _sha256(source),
                "entry_names": list(analysis.entry_names),
                "ptx_version": analysis.version,
                "target": analysis.target,
                "address_size": analysis.address_size,
                "instruction_count": analysis.instruction_count,
                "features": list(analysis.features),
                "unique_opcodes": len(analysis.opcode_counts),
            }
            if extract_dir is not None:
                output_name = f"ptx-{fatbin_index:03d}-{_sha256(source)[:12]}.ptx"
                output_path = extract_dir / output_name
                output_path.write_bytes(source)
                row["extracted_file"] = output_name
            module_rows.append(row)

    unique_names = sorted(set(all_entry_names))
    manifest = {
        "schema": 1,
        "tool": "dlss_fatbin_inspect",
        "file": Path(file_name).name,
        "file_size": len(data),
        "file_sha256": _sha256(data),
        "fatbins_found": len(records),
        "fatbins_with_ptx": fatbins_with_ptx,
        "fatbin_entry_counts": dict(sorted(fatbin_kind_counts.items())),
        "ptx_modules": len(module_rows),
        "ptx_entry_points": ptx_entry_count,
        "unique_ptx_entry_points": len(unique_names),
        "ptx_versions": dict(sorted(ptx_versions.items())),
        "ptx_targets": dict(sorted(ptx_targets.items())),
        "compression_modes": dict(sorted(compression_modes.items())),
        "feature_module_counts": dict(sorted(feature_modules.items())),
        "ptx_total_uncompressed_bytes": ptx_total_size,
        "ptx_max_uncompressed_bytes": ptx_max_size,
        "unique_opcodes": len(all_opcodes),
        "instruction_count": sum(all_opcodes.values()),
        "opcode_counts": dict(sorted(all_opcodes.items())),
        "entry_names": unique_names,
        "modules": module_rows,
        "scan_warnings": scan_errors,
    }
    return manifest


def inspect_path(path: Path, *, extract_dir: Optional[Path] = None) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FatbinError(f"input is not a regular file: {path}")
    if path.stat().st_size > MAX_INPUT_SIZE:
        raise FatbinError(f"input exceeds the configured limit: {path.stat().st_size} bytes")
    return inspect_bytes(path.read_bytes(), file_name=path.name, extract_dir=extract_dir)
