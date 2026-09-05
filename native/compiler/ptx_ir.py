"""Small, loss-aware PTX parser for the recovered DLSS corpus.

This is intentionally a front end, not a compiler. It preserves the original
statement text, predicate, opcode and operand string so later lowering passes
can add typed semantics without reparsing the compressed source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class PtxParseError(ValueError):
    """Raised when a PTX module cannot be converted to the normalized IR."""


@dataclass(frozen=True)
class PtxParameter:
    type_name: str
    name: str
    array_size: int | None
    alignment: int | None = None


@dataclass(frozen=True)
class PtxInstruction:
    line: int
    predicate: str | None
    opcode: str
    operands: str
    text: str


@dataclass(frozen=True)
class PtxKernel:
    name: str
    parameters: tuple[PtxParameter, ...]
    directives: tuple[str, ...]
    labels: tuple[str, ...]
    label_indices: tuple[tuple[str, int], ...]
    instructions: tuple[PtxInstruction, ...]


@dataclass(frozen=True)
class PtxModule:
    version: str
    target: str
    address_size: int
    kernels: tuple[PtxKernel, ...]

    @property
    def instruction_count(self) -> int:
        return sum(len(kernel.instructions) for kernel in self.kernels)

    @property
    def opcode_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for kernel in self.kernels:
            for instruction in kernel.instructions:
                counts[instruction.opcode] = counts.get(instruction.opcode, 0) + 1
        return dict(sorted(counts.items()))


_VERSION_RE = re.compile(r"(?m)^\s*\.version\s+([0-9]+(?:\.[0-9]+)?)")
_TARGET_RE = re.compile(r"(?m)^\s*\.target\s+([^\r\n]+)")
_ADDRESS_SIZE_RE = re.compile(r"(?m)^\s*\.address_size\s+([0-9]+)")
_ENTRY_RE = re.compile(r"\.entry\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
_PARAM_RE = re.compile(
    r"\.param(?:\s+\.align\s+(?P<align>\d+))?\s+"
    r"(?P<type>\.[A-Za-z0-9_.]+)\s+"
    r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)"
    r"(?:\[(?P<size>\d+)\])?"
)
_PREDICATE_RE = re.compile(r"^@(?P<predicate>!?[%A-Za-z0-9_.$]+)\s+")
_OPCODE_RE = re.compile(
    r"^(?P<opcode>[A-Za-z][A-Za-z0-9_.]*(?:::[A-Za-z0-9_.]+)*)(?P<rest>(?:\s.*)?)$", re.DOTALL
)
_LABEL_RE = re.compile(r"^(?P<label>[A-Za-z_$][A-Za-z0-9_$]*):\s*(?P<rest>.*)$", re.DOTALL)
_IDENTIFIER_RE = re.compile(r"(?<![\w.$%])[%A-Za-z_$][\w.$]*(?![\w.$])")
_REG_RE = re.compile(r"^\.reg\s*(\.[\w]+)\s+(.+)$", re.DOTALL)
_REG_NAME_RE = re.compile(r"([%A-Za-z_$][\w$]*)(?:<(\d+)>)?$")


def _remove_comments(text: str) -> str:
    result: list[str] = []
    block_comment = False
    for line in text.splitlines(keepends=True):
        current = line
        pieces: list[str] = []
        while current:
            if block_comment:
                end = current.find("*/")
                if end < 0:
                    pieces.append("\n" if current.endswith("\n") else "")
                    current = ""
                    continue
                current = current[end + 2 :]
                block_comment = False
                continue
            start_block = current.find("/*")
            start_line = current.find("//")
            if start_line >= 0 and (start_block < 0 or start_line < start_block):
                pieces.append(current[:start_line])
                if line.endswith("\n"):
                    pieces.append("\n")
                current = ""
                continue
            if start_block < 0:
                pieces.append(current)
                current = ""
                continue
            pieces.append(current[:start_block])
            current = current[start_block + 2 :]
            block_comment = True
        result.extend(pieces)
    return "".join(result)


def _find_body_end(text: str, opening_brace: int) -> int:
    depth = 1
    in_string = escaped = False
    for cursor in range(opening_brace + 1, len(text)):
        character = text[cursor]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return cursor
    raise PtxParseError("PTX kernel body is missing its closing brace")


def _split_statements(body: str, first_line: int) -> list[tuple[int, str]]:
    statements: list[tuple[int, str]] = []
    start = 0
    line = first_line
    statement_line: int | None = None
    vector_depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(body):
        if character == "\n":
            line += 1
        elif statement_line is None and not character.isspace():
            statement_line = line
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if not body[start:index].strip():
                statements.append((statement_line or line, "{"))
                start = index + 1
                statement_line = None
            else:
                vector_depth += 1
        elif character == "}":
            if vector_depth:
                vector_depth -= 1
            elif body[start:index].strip():
                raise PtxParseError(f"missing semicolon before closing scope on line {line}")
            else:
                statements.append((statement_line or line, "}"))
                start = index + 1
                statement_line = None
        elif character == ":" and re.fullmatch(r"[A-Za-z_$][\w$]*", body[start:index].strip()):
            statements.append((statement_line or line, body[start:index + 1].strip()))
            start = index + 1
            statement_line = None
        elif character == ";" and not vector_depth:
            value = body[start:index].strip()
            if value:
                statements.append((statement_line or line, value))
            start = index + 1
            statement_line = None
    tail = body[start:].strip()
    if tail or vector_depth or in_string:
        raise PtxParseError(f"unterminated PTX statement on line {statement_line or line}")
    return statements


def _parse_parameters(header: str) -> tuple[PtxParameter, ...]:
    parameters: list[PtxParameter] = []
    for match in _PARAM_RE.finditer(header):
        size = match.group("size")
        parameters.append(
            PtxParameter(
                type_name=match.group("type"),
                name=match.group("name"),
                array_size=int(size) if size is not None else None,
                alignment=(
                    int(match.group("align"))
                    if match.group("align") is not None else None
                ),
            )
        )
    return tuple(parameters)


def _parse_kernel_body(name: str, header: str, body: str, first_line: int) -> PtxKernel:
    directives: list[str] = [
        line.strip()
        for line in header.splitlines()
        if line.strip().startswith(".") and not line.strip().startswith(".param")
    ]
    labels: list[str] = []
    label_indices: list[tuple[str, int]] = []
    instructions: list[PtxInstruction] = []
    # Alpha-rename lexical declarations before flattening the instruction
    # stream. Two passes allow forward branches while preventing sibling or
    # child scopes from leaking labels/registers into their parent.
    scopes: list[dict[str, str]] = [{}]
    parents = [-1]
    scope = 0
    records: list[tuple[int, str, int]] = []
    occupied = set(_IDENTIFIER_RE.findall(body))
    serial = 0

    def fresh(register: bool) -> str:
        nonlocal serial
        while True:
            serial += 1
            value, suffix = serial, ""
            while value:
                value, digit = divmod(value - 1, 26)
                suffix = chr(65 + digit) + suffix
            candidate = ("%ptxScope" + suffix + "0") if register else ("ptxLabel" + suffix)
            if candidate not in occupied and (not register or candidate[:-1] not in occupied):
                occupied.add(candidate)
                return candidate

    def declare(token: str, resolved: str) -> None:
        if token in scopes[scope]:
            raise PtxParseError(f"duplicate declaration in lexical scope: {token}")
        scopes[scope][token] = resolved

    for line_number, statement in _split_statements(body, first_line):
        if statement == "{":
            parents.append(scope)
            scope = len(scopes)
            scopes.append({})
            continue
        if statement == "}":
            if scope == 0:
                raise PtxParseError("unbalanced PTX lexical scope")
            scope = parents[scope]
            continue
        reg = _REG_RE.fullmatch(statement)
        if reg:
            for item in reg.group(2).split(","):
                match = _REG_NAME_RE.fullmatch(item.strip())
                if not match:
                    raise PtxParseError(f"invalid register declaration on line {line_number}")
                token, count = match.groups()
                if count is not None and int(count) <= 0:
                    raise PtxParseError("register array must have positive size")
                # Preserve ordinary root families for existing matrix/replay
                # recognition. Normalize named and nested registers into the
                # same numbered representation used by liveness analysis.
                if scope == 0 and count and re.fullmatch(r"%[A-Za-z]+", token):
                    directives.append(f".reg {reg.group(1)} {token}<{count}>")
                    for index in range(int(count)):
                        declare(f"{token}{index}", f"{token}{index}")
                else:
                    for index in range(int(count) if count else 1):
                        original = f"{token}{index}" if count else token
                        resolved = fresh(True)
                        declare(original, resolved)
                        directives.append(f".reg {reg.group(1)} {resolved[:-1]}<1>")
            continue
        label = _LABEL_RE.fullmatch(statement)
        if label:
            token = label.group("label")
            declare(token, token if scope == 0 else fresh(False))
        records.append((line_number, statement, scope))
    if scope:
        raise PtxParseError("unclosed PTX lexical scope")

    def resolve(token: str, scope: int) -> str:
        while scope >= 0:
            if token in scopes[scope]:
                return scopes[scope][token]
            scope = parents[scope]
        return token

    for line_number, raw_statement, scope in records:
        statement = raw_statement
        label_match = _LABEL_RE.fullmatch(statement)
        if label_match:
            label = resolve(label_match.group("label"), scope)
            labels.append(label)
            label_indices.append((label, len(instructions)))
            continue
        if statement.startswith("."):
            directives.append(statement)
            continue
        predicate_match = _PREDICATE_RE.match(statement)
        predicate = predicate_match.group("predicate") if predicate_match else None
        if predicate_match:
            statement = statement[predicate_match.end() :].strip()
        opcode_match = _OPCODE_RE.match(statement)
        if not opcode_match:
            raise PtxParseError(f"could not parse PTX statement on line {line_number}: {raw_statement!r}")
        instructions.append(
            PtxInstruction(
                line=line_number,
                predicate=_IDENTIFIER_RE.sub(lambda m: resolve(m.group(), scope), predicate) if predicate else None,
                opcode=opcode_match.group("opcode"),
                operands=_IDENTIFIER_RE.sub(lambda m: resolve(m.group(), scope), opcode_match.group("rest").strip()),
                text=raw_statement.strip(),
            )
        )
    return PtxKernel(
        name=name,
        parameters=_parse_parameters(header),
        directives=tuple(directives),
        labels=tuple(labels),
        label_indices=tuple(label_indices),
        instructions=tuple(instructions),
    )


def parse_ptx(source: bytes | str) -> PtxModule:
    if isinstance(source, bytes):
        source = source.rstrip(b"\x00")
        try:
            source = source.decode("ascii")
        except UnicodeDecodeError as exc:
            raise PtxParseError("PTX source is not ASCII") from exc
    text = _remove_comments(source)
    version_match = _VERSION_RE.search(text)
    target_match = _TARGET_RE.search(text)
    address_match = _ADDRESS_SIZE_RE.search(text)
    if not version_match or not target_match or not address_match:
        raise PtxParseError("PTX is missing version, target, or address-size metadata")

    kernels: list[PtxKernel] = []
    cursor = 0
    while True:
        entry_match = _ENTRY_RE.search(text, cursor)
        if entry_match is None:
            break
        close_paren = text.find(")", entry_match.end())
        if close_paren < 0:
            raise PtxParseError(f"kernel {entry_match.group(1)} has no closing parameter list")
        opening_brace = text.find("{", close_paren)
        if opening_brace < 0:
            raise PtxParseError(f"kernel {entry_match.group(1)} has no body")
        body_end = _find_body_end(text, opening_brace)
        first_body_line = text.count("\n", 0, opening_brace + 1) + 1
        kernels.append(
            _parse_kernel_body(
                entry_match.group(1),
                text[entry_match.end() : opening_brace],
                text[opening_brace + 1 : body_end],
                first_body_line,
            )
        )
        cursor = body_end + 1

    if not kernels:
        raise PtxParseError("PTX contains no .entry kernel")
    return PtxModule(
        version=version_match.group(1),
        target=target_match.group(1).strip(),
        address_size=int(address_match.group(1)),
        kernels=tuple(kernels),
    )
