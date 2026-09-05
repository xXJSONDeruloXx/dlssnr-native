"""Synthetic PTX only: scope resolution and bit-exact register lowering."""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from native.compiler.ptx_ir import PtxParseError, parse_ptx
from native.compiler.ptx_glsl import (
    PtxTranslationError,
    _emit_instruction,
    _instruction_register_sets,
    _glsl_parameter_helpers,
    _register_files,
    _register_storage_layout,
    _replay_register_load_lines,
    _replay_register_store_lines,
    _replay_register_zero_lines,
    _replay_sparse_storage_layout,
    compile_glsl,
    translate_kernel,
)


def kernel(body):
    return parse_ptx(
        '.version 8.3\n.target sm_89\n.address_size 64\n'
        '.entry synthetic() {\n' + body + '\n}'
    ).kernels[0]


def emit(body):
    parsed = kernel(body)
    registers = _register_files(parsed)
    return _emit_instruction(parsed.instructions[-1], parsed, registers, surface_format='uint')


class LexicalScopeTests(unittest.TestCase):
    def test_multiline_labels_and_vector_braces(self):
        parsed = kernel('''
            .reg .b128 %q<1>;
            .reg .b32 %r<4>;
            .reg .pred %p<1>;
            @%p0 bra done;
            first:
            second:
            mov.b128 %q0, {
                %r0, %r1,
                %r2, %r3
            };
            done:
            ret;
        ''')
        self.assertEqual(parsed.label_indices, (('first', 1), ('second', 1), ('done', 2)))
        self.assertEqual([i.opcode for i in parsed.instructions], ['bra', 'mov.b128', 'ret'])
        self.assertEqual(parsed.instructions[1].line, 12)
        self.assertIn('uvec4(r[0], r[1], r[2], r[3])', translate_kernel(parsed))

    def test_shadowed_named_registers_have_independent_types_and_lifetimes(self):
        parsed = kernel('''
            .reg .b32 scratch;
            mov.b32 scratch, 7;
            { .reg .b128 scratch;
              .reg .b32 %r<4>;
              mov.b128 scratch, {%r0, %r1, %r2, %r3};
              { .reg .pred scratch; mov.pred scratch, 1; }
              mov.b128 {%r0, %r1, %r2, %r3}, scratch;
            }
            mov.b32 scratch, 9;
            { .reg .f32 scratch; mov.f32 scratch, 0f80000000; }
        ''')
        registers = _register_files(parsed)
        self.assertEqual(len(registers), 8)
        first = parsed.instructions[0].operands.split(',')[0]
        self.assertEqual(first, parsed.instructions[4].operands.split(',')[0])
        inner = parsed.instructions[1].operands.split(',')[0]
        self.assertNotEqual(first, inner)
        self.assertTrue(parsed.instructions[3].operands.endswith(inner))
        self.assertIn('uvec4', translate_kernel(parsed))

    def test_shadowed_numbered_family_and_predicate_resolve_by_scope(self):
        parsed = kernel('''
            .reg .b32 %r<2>; .reg .pred %p<1>;
            mov.b32 %r0, 1;
            { .reg .b64 %r<2>; .reg .pred %p<1>;
              @!%p0 mov.b64 %r0, %r1;
            }
            @%p0 mov.b32 %r0, %r1;
        ''')
        inside, outside = parsed.instructions[1:]
        self.assertNotEqual(inside.predicate, '!%p0')
        self.assertNotIn('%r0', inside.operands)
        self.assertEqual(outside.predicate, '%p0')
        self.assertEqual(outside.operands, '%r0, %r1')
        self.assertIn('uint64_t', translate_kernel(parsed))
        uses, definitions = _instruction_register_sets(inside)
        self.assertEqual(len(uses), 2)  # source plus predicate
        self.assertEqual(len(definitions), 1)

    def test_forward_and_shadowed_labels_resolve_to_nearest_scope(self):
        parsed = kernel('''
            .reg .pred %p<1>;
            bra end;
            { bra end; end: ret; }
            { bra end; end: ret; }
            end: ret;
        ''')
        self.assertEqual(len(set(parsed.labels)), 3)
        self.assertEqual(parsed.instructions[0].operands, 'end')
        self.assertEqual(dict(parsed.label_indices)[parsed.instructions[1].operands], 2)
        self.assertEqual(dict(parsed.label_indices)[parsed.instructions[3].operands], 4)
        translate_kernel(parsed)

    def test_compact_register_directives_are_scoped(self):
        parsed = kernel('.reg.b32 value; { .reg.b16 value; mov.b16 value, 1; } '
                        'mov.b32 value, 2;')
        self.assertEqual({r.ptx_type for r in _register_files(parsed).values()}, {'.b16', '.b32'})
        self.assertNotEqual(parsed.instructions[0].operands.split(',')[0],
                            parsed.instructions[1].operands.split(',')[0])
        translate_kernel(parsed)

    def test_double_colon_opcode_qualifiers_are_preserved(self):
        parsed = kernel('.reg .b64 %rd<1>; .reg .b32 %r<1>; '
                        'ld.global.L1::evict_normal.L2::128B.b32 %r0, [%rd0];')
        self.assertEqual(parsed.instructions[0].opcode, 'ld.global.L1::evict_normal.L2::128B.b32')
        self.assertEqual(parsed.instructions[0].operands, '%r0, [%rd0]')

    def test_child_can_branch_to_enclosing_label(self):
        parsed = kernel('{ { bra finish; } } finish: ret;')
        self.assertEqual(parsed.instructions[0].operands, 'finish')
        self.assertEqual(parsed.label_indices, (('finish', 1),))

    def test_sibling_label_is_not_visible(self):
        with self.assertRaises(PtxTranslationError):
            translate_kernel(kernel('{ private: ret; } { bra private; }'))

    def test_sibling_register_is_not_visible(self):
        with self.assertRaises(PtxTranslationError):
            translate_kernel(kernel('{ .reg .b32 private; } { mov.b32 private, 1; }'))

    def test_duplicate_declarations_rejected(self):
        for body in (
            '.reg .b32 %r<2>; .reg .b32 %r1;',
            '.reg .b32 value; .reg .b64 value;',
            'same: ret; same: ret;',
        ):
            with self.subTest(body=body), self.assertRaises(PtxParseError):
                kernel(body)

    def test_generated_names_cannot_capture_source_identifiers(self):
        parsed = kernel('''
            .reg .b32 %ptxScopeA<1>;
            { .reg .b32 value; mov.b32 value, %ptxScopeA0; }
        ''')
        destination, source = parsed.instructions[0].operands.split(', ')
        self.assertNotEqual(destination, source)
        translate_kernel(parsed)

    def test_closing_brace_does_not_swallow_next_same_line_entry(self):
        module = parse_ptx('.version 8.3\n.target sm_89\n.address_size 64\n'
                           '.entry one() {ret;} .entry two() {{ret;}}')
        self.assertEqual([k.name for k in module.kernels], ['one', 'two'])
        self.assertEqual([len(k.instructions) for k in module.kernels], [1, 1])

    def test_unterminated_statements_and_invalid_declarations_rejected(self):
        for body in ('mov.b32 %r0, 1', '{ ret }', '.reg .b32 %r<0>;',
                     '.reg .b32 invalid-name;'):
            with self.subTest(body=body), self.assertRaises(PtxParseError):
                kernel(body)


class WideRegisterTests(unittest.TestCase):
    def test_pack_four_words_in_low_to_high_order(self):
        self.assertEqual(
            emit('.reg .b128 %q<1>; .reg .b32 %r<4>; '
                 'mov.b128 %q0, {%r0, %r1, %r2, %r3};'),
            'q[0] = uvec4(r[0], r[1], r[2], r[3]);',
        )

    def test_unpack_four_words_and_discard_lane(self):
        code = emit('.reg .b128 %q<1>; .reg .b32 %r<4>; '
                    'mov.b128 {%r0, _, %r2, %r3}, %q0;')
        self.assertIn('uvec4 ptx_mov_bits = q[0];', code)
        self.assertIn('r[0] = ptx_mov_bits[0];', code)
        self.assertIn('r[2] = ptx_mov_bits[2];', code)
        self.assertIn('r[3] = ptx_mov_bits[3];', code)
        self.assertNotIn('r[1] =', code)

    def test_pack_two_doublewords_in_low_to_high_order(self):
        self.assertEqual(
            emit('.reg .b128 %q<1>; .reg .b64 %rd<2>; mov.b128 %q0, {%rd0, %rd1};'),
            'q[0] = uvec4(uint(rd[0]), uint(rd[0] >> 32), uint(rd[1]), uint(rd[1] >> 32));',
        )

    def test_unpack_two_doublewords(self):
        code = emit('.reg .b128 %q<1>; .reg .b64 %rd<2>; mov.b128 {%rd0, %rd1}, %q0;')
        for lane in range(2):
            self.assertIn(f'rd[{lane}] = (uint64_t(ptx_mov_bits[{2 * lane}]) | '
                          f'(uint64_t(ptx_mov_bits[{2 * lane + 1}]) << 32));', code)

    def test_float_lanes_are_reinterpreted_not_converted(self):
        for kind, count, pack, unpack in (
            ('f32', 4, 'floatBitsToUint', 'uintBitsToFloat'),
            ('f64', 2, 'doubleBitsToUint64', 'uint64BitsToDouble'),
        ):
            declarations = f'.reg .b128 %q<1>; .reg .{kind} %f<{count}>; '
            vector = '{' + ', '.join(f'%f{i}' for i in range(count)) + '}'
            self.assertIn(pack, emit(declarations + f'mov.b128 %q0, {vector};'))
            self.assertIn(unpack, emit(declarations + f'mov.b128 {vector}, %q0;'))

    def test_invalid_wide_forms_fail_closed(self):
        declarations = '.reg .b128 %q<2>; .reg .b32 %r<4>; .reg .b16 %h<4>; '
        for instruction in (
            'mov.b128 %q0, {%r0, %r1};',
            'mov.b128 %q0, {%h0, %h1, %h2, %h3};',
            'mov.b128 %q0, {%r0, %r1, %r2};',
            'mov.b128 %r0, {%r0, %r1, %r2, %r3};',
            'mov.b128 %q0, {%r0, %r1, %r2, _};',
            'mov.b128 %q0, {0, 0, 0, 0};',
            'mov.b128 %q0, %q1;',
            'mov.b32 %r0, %q0;',
            'add.b128 %q0, %q0, %q1;',
            'ld.global.b128 %q0, [%r0];',
        ):
            with self.subTest(instruction=instruction), self.assertRaises(PtxTranslationError):
                emit(declarations + instruction)

    def test_four_word_checkpoint_storage(self):
        parsed = kernel('.reg .b128 %q<2>; .reg .b32 %r<1>; ret;')
        registers = _register_files(parsed, scalarize=True)
        offsets, words = _register_storage_layout(registers)
        self.assertEqual((offsets, words), ({'q': 0, 'r': 8}, 9))
        sparse, words = _replay_sparse_storage_layout(registers, {'q': {1}, 'r': {0}})
        self.assertEqual((sparse, words), ({('q', 1): 0, ('r', 0): 4}, 5))
        loads = '\n'.join(_replay_register_load_lines(registers, sparse))
        stores = '\n'.join(_replay_register_store_lines(registers, sparse))
        zeroes = _replay_register_zero_lines(registers, sparse)
        self.assertIn('q_1 = uvec4(', loads)
        self.assertEqual(len(zeroes), 5)
        for lane in range(4):
            self.assertIn(f'ptx_register_base + {lane}u', loads)
            self.assertIn(f'ptx_register_base + {lane}u] = q_1[{lane}];', stores)

    def test_predicated_wide_unpack_is_scoped(self):
        code = translate_kernel(kernel('''
            .reg .b128 %q<1>; .reg .b32 %r<4>; .reg .pred %p<1>;
            @%p0 mov.b128 {%r0, %r1, %r2, %r3}, %q0;
            @!%p0 mov.b128 {%r3, %r2, %r1, %r0}, %q0;
        '''))
        self.assertEqual(code.count('{ uvec4 ptx_mov_bits ='), 2)
        self.assertIn('if (p[0])', code)
        self.assertIn('if (!(p[0]))', code)


class CopySignTests(unittest.TestCase):
    def test_actual_emitted_bit_expression_preserves_special_values(self):
        code = emit('.reg .b32 %r<3>; copysign.f32 %r0, %r1, %r2;')
        expression = re.sub(r'(?<=[0-9a-f])u\b', '', code.split(' = ', 1)[1].rstrip(';'))
        # Evaluate the emitted integer expression, not a second implementation
        # of the emitter. Deliberately include NaN payloads and signed zero.
        values = [0, 0x80000000, 1, 0x80000001, 0x7f800000,
                  0xff800000, 0x7fc12345, 0xff812345, 0x3f800000]
        for sign in values:
            for magnitude in values:
                with self.subTest(sign=hex(sign), magnitude=hex(magnitude)):
                    actual = eval(expression, {'__builtins__': {}}, {'r': [0, sign, magnitude]})
                    self.assertEqual(actual, (sign & (1 << 31)) | (magnitude & ((1 << 31) - 1)))

    def test_float_registers_and_raw_immediates(self):
        code = emit('.reg .f32 %f<3>; copysign.f32 %f0, %f1, %f2;')
        self.assertEqual(code, 'f[0] = uintBitsToFloat(((floatBitsToUint(f[1]) & 0x80000000u) | '
                         '(floatBitsToUint(f[2]) & 0x7fffffffu)));')
        self.assertIn('0x80000000u & 0x80000000u', emit(
            '.reg .b32 %r<1>; copysign.f32 %r0, 0f80000000, 0f7fc12345;'))

    def test_unsupported_modifiers_and_widths_are_rejected(self):
        for instruction in ('copysign.ftz.f32 %r0, %r1, %r2;',
                            'copysign.f64 %r0, %r1, %r2;',
                            'copysign.f32 %r0, %r1;',
                            'copysign.f32 %r0, %q0, %r2;'):
            with self.subTest(instruction=instruction), self.assertRaises(PtxTranslationError):
                emit('.reg .b32 %r<3>; .reg .b128 %q<1>; ' + instruction)

    @unittest.skipUnless(shutil.which('glslc'), 'glslc is not installed')
    def test_synthetic_shader_compiles(self):
        source = translate_kernel(kernel('''
            .reg .b128 %q<1>; .reg .b32 %r<4>;
            mov.b32 %r0, 0x80000000; mov.b32 %r1, 1;
            mov.b32 %r2, 0x7fc12345; mov.b32 %r3, 0;
            mov.b128 %q0, {%r0, %r1, %r2, %r3};
            mov.b128 {%r3, %r2, %r1, %r0}, %q0;
            copysign.f32 %r0, %r1, %r2;
            ret;
        '''))
        with tempfile.TemporaryDirectory() as directory:
            compile_glsl(source, Path(directory) / 'synthetic.spv')


class E4M3DecodeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('clang++'), 'clang++ is not installed')
    def test_all_256_encodings_against_ieee_bit_reference(self):
        helpers = '\n'.join(_glsl_parameter_helpers(
            {}, shared_size=0, has_global_memory=False, has_shared_memory=False,
            has_local_memory=False, has_constant_memory=False,
        ))
        # Execute the actual emitted scalar helper as C++ (its syntax is
        # shared with GLSL), replacing only GLSL's bitcast builtin. No GPU or
        # proprietary input is needed for exhaustive decoder validation.
        helper = re.search(r'float ptx_e4m3_to_float\(uint bits\) \{.*?\n\}', helpers, re.S).group()
        source = '''
            #include <cmath>
            #include <cstdint>
            #include <cstring>
            #include <cstdio>
            using uint = uint32_t;
            float uintBitsToFloat(uint bits) {
                float value; std::memcpy(&value, &bits, 4); return value;
            }
        ''' + helper + '''
            int main() {
                for (uint bits = 0; bits < 256; ++bits) {
                    float value = ptx_e4m3_to_float(bits);
                    uint raw; std::memcpy(&raw, &value, 4);
                    std::printf("%08x\\n", raw);
                }
            }
        '''
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'synthetic_decoder'
            subprocess.run(['clang++', '-std=c++11', '-x', 'c++', '-', '-o', str(binary)],
                           input=source, text=True, capture_output=True, check=True)
            result = subprocess.run([str(binary)], text=True, capture_output=True, check=True)
        actual = [int(line, 16) for line in result.stdout.splitlines()]
        self.assertEqual(len(actual), 256)
        for byte, decoded in enumerate(actual):
            sign = (byte & 128) << 24
            exponent, mantissa = (byte >> 3) & 15, byte & 7
            if exponent == 15 and mantissa == 7:
                expected = sign | 0x7fc00000
            elif exponent:
                expected = sign | ((exponent + 120) << 23) | (mantissa << 20)
            elif mantissa:
                highest = mantissa.bit_length() - 1
                expected = sign | ((highest + 118) << 23) | ((mantissa - (1 << highest)) << (23 - highest))
            else:
                expected = sign
            with self.subTest(encoding=hex(byte)):
                self.assertEqual(decoded, expected)


if __name__ == '__main__':
    unittest.main()
