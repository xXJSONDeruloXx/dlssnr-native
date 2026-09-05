"""Numerical check through the actual standalone shared-library ABI."""
import argparse
import ctypes as c
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from native.probes.matrix_layout import expected


def run(library, shader):
    lib = c.CDLL(str(library.resolve()))
    pointer = c.c_void_p
    def api(name, restype, *arguments):
        function = getattr(lib, name)
        function.restype, function.argtypes = restype, list(arguments)
        return function
    create = api('nr_create', c.c_int, c.POINTER(pointer))
    destroy = api('nr_destroy', None, pointer)
    allocate = api('nr_buffer_create', c.c_int, pointer, c.c_size_t, c.POINTER(pointer))
    free = api('nr_buffer_destroy', None, pointer, pointer)
    address = api('nr_buffer_address', c.c_uint64, pointer, pointer)
    write = api('nr_buffer_write', c.c_int, pointer, pointer, c.c_size_t, pointer, c.c_size_t)
    read = api('nr_buffer_read', c.c_int, pointer, pointer, c.c_size_t, pointer, c.c_size_t)
    pipeline_create = api('nr_pipeline_create', c.c_int, pointer, pointer, c.c_size_t, c.POINTER(pointer))
    pipeline_destroy = api('nr_pipeline_destroy', None, pointer, pointer)
    dispatch = api('nr_dispatch', c.c_int, pointer, pointer, pointer, pointer, c.c_uint32, c.c_uint32, c.c_uint32)
    name = api('nr_device_name', c.c_char_p, pointer)
    def check(value):
        if value:
            raise RuntimeError('Native Vulkan result: ' + str(value))
    ctx, params, output, pipeline = pointer(), pointer(), pointer(), pointer()
    try:
        check(create(c.byref(ctx)))
        check(allocate(ctx, 16, c.byref(params)))
        check(allocate(ctx, 513*4, c.byref(output)))
        gpu_address = c.c_uint64(address(ctx, output))
        if not gpu_address.value:
            raise RuntimeError('Missing GPU buffer address')
        check(write(ctx, params, 0, c.byref(gpu_address), 8))
        if write(ctx, params, 16, c.byref(gpu_address), 8) == 0:
            raise RuntimeError('Out-of-bounds write accepted')
        binary = shader.read_bytes()
        words = (c.c_uint32 * (len(binary)//4)).from_buffer_copy(binary)
        check(pipeline_create(ctx, words, len(binary), c.byref(pipeline)))
        check(dispatch(ctx, pipeline, params, output, 1, 1, 1))
        result = (c.c_uint32 * 513)()
        check(read(ctx, output, 0, result, c.sizeof(result)))
        mismatches = sum(a != b for a,b in zip(result[:512], expected()))
        report = {'device':name(ctx).decode(), 'elements':512, 'mismatches':mismatches,
                  'subgroup_size':result[512], 'library_dispatch':True,
                  'pass':mismatches==0 and result[512] in (32,64)}
        print(json.dumps(report))
        if not report['pass']:
            raise RuntimeError('Native matrix readback mismatch')
    finally:
        if pipeline: pipeline_destroy(ctx, pipeline)
        if output: free(ctx, output)
        if params: free(ctx, params)
        if ctx: destroy(ctx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--spirv', type=Path, required=True)
    args = parser.parse_args()
    run(args.library, args.spirv)
