"""Independent coordinate fixtures for the compiler's two NR MMA layouts."""
import argparse
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from native.compiler.ptx_glsl import _glsl_matrix_helpers


def shader():
    return '''#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
layout(local_size_x=64) in;
layout(set=0,binding=1,std430) buffer Output { uint data[]; } output_data;
float av(uint r,uint k,uint seed) { return float(int((r*3u+k*5u+seed)%17u)-8)*0.25; }
float bv(uint k,uint c,uint seed) { return float(int((k*7u+c*11u+seed)%13u)-6)*0.125; }
float cv(uint r,uint c,uint seed) { return float(int((r+c*3u+seed)%9u)-4)*0.0625; }
''' + '\n'.join(_glsl_matrix_helpers()) + '''
void main() {
    uint lane=gl_LocalInvocationID.x%32u, warp=gl_LocalInvocationID.x/32u;
    uint seed=warp*9u;
    float a16[8], b16[4], a32[16], b32[8], c[4];
    for(uint i=0u;i<8u;i++) {
        uint r=lane/4u+((i%4u)/2u)*8u;
        uint k=(lane%4u)*2u+i%2u+(i/4u)*8u;
        a16[i]=av(r,k,seed);
    }
    for(uint i=0u;i<4u;i++) b16[i]=bv((lane%4u)*2u+i%2u+(i/2u)*8u,lane/4u,seed);
    for(uint i=0u;i<16u;i++) {
        uint r=lane/4u+((i%8u)/4u)*8u;
        uint k=(lane%4u)*4u+i%4u+(i/8u)*16u;
        a32[i]=av(r,k,seed);
    }
    for(uint i=0u;i<8u;i++) b32[i]=bv((lane%4u)*4u+i%4u+(i/4u)*16u,lane/4u,seed);
    for(uint i=0u;i<4u;i++) c[i]=cv(lane/4u+(i/2u)*8u,(lane%4u)*2u+i%2u,seed);
    for(uint i=0u;i<4u;i++) {
        uint r=lane/4u+(i/2u)*8u, col=(lane%4u)*2u+i%2u;
        float x=ptx_mma_m16n8k16_f16(a16,b16,c,i);
        float y=ptx_mma_m16n8k32_e4m3(a32,b32,c,i);
        output_data.data[warp*256u+r*8u+col]=floatBitsToUint(unpackHalf2x16(packHalf2x16(vec2(x,0)))[0]);
        output_data.data[warp*256u+128u+r*8u+col]=floatBitsToUint(unpackHalf2x16(packHalf2x16(vec2(y,0)))[0]);
    }
    if(gl_LocalInvocationID.x==0u) output_data.data[512]=gl_SubgroupSize;
}
'''


def expected():
    result = []
    for warp in range(2):
        seed = warp * 9
        for size in (16, 32):
            for row in range(16):
                for column in range(8):
                    value = ((row + column*3 + seed) % 9 - 4) / 16
                    for k in range(size):
                        value += ((row*3+k*5+seed) % 17-8)/4 * ((k*7+column*11+seed) % 13-6)/8
                    value = struct.unpack('<e', struct.pack('<e', value))[0]
                    result.append(struct.unpack('<I', struct.pack('<f', value))[0])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shader', type=Path)
    parser.add_argument('--validate', type=Path)
    args = parser.parse_args()
    if args.shader:
        args.shader.write_text(shader())
    if args.validate:
        observed = json.loads(args.validate.read_text())['output']
        reference = expected()
        mismatches = [i for i, (a,b) in enumerate(zip(observed[:512], reference)) if a != b]
        if len(observed) != 513 or observed[-1] not in (32,64):
            raise SystemExit('Unexpected readback size or subgroup width')
        print(json.dumps({'elements':512,'mismatches':len(mismatches),
                          'first_mismatches':mismatches[:12], 'subgroup_size':observed[-1],
                          'pass':not mismatches}))
        raise SystemExit(bool(mismatches))
