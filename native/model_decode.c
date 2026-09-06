#include "model_decode.h"
#include <math.h>
#include <string.h>

float nr_e4m3fn_decode(uint8_t bits) {
    unsigned exponent=(bits>>3)&15u;
    unsigned mantissa=bits&7u;
    if (exponent==15u && mantissa==7u) return NAN;
    float value=exponent ? ldexpf(1.0f+(float)mantissa/8.0f,(int)exponent-7)
                         : (float)mantissa/512.0f;
    return bits&128u ? -value : value;
}

float nr_fp16_decode(uint16_t bits) {
    unsigned sign=bits>>15;
    unsigned exponent=(bits>>10)&31u;
    unsigned mantissa=bits&1023u;
    float value;
    if (!exponent) value=mantissa ? ldexpf((float)mantissa,-24) : 0.0f;
    else if (exponent==31u) value=mantissa ? NAN : INFINITY;
    else value=ldexpf(1.0f+(float)mantissa/1024.0f,(int)exponent-15);
    return sign ? -value : value;
}

uint32_t nr_tin_channel_permutation(uint32_t channel) {
    return (channel&~14u)|((channel&2u)<<2)|((channel&4u)>>1)|((channel&8u)>>1);
}

static int dimensions_ok(uint32_t rows, uint32_t columns, size_t output_count) {
    if (!rows || !columns || rows%16 || columns%16) return 0;
    if ((size_t)rows>SIZE_MAX/(size_t)columns) return 0;
    return (size_t)rows*columns==output_count;
}

static size_t fp8_index(uint32_t row, uint32_t column, uint32_t rows) {
    uint32_t nb=row/16, rr=row%16, d4=rr/8, d2=rr%8;
    uint32_t kb=column/32, cc=column%32, d5=cc/8, d3=(cc%8)/2, d6=cc%2;
    return (size_t)kb*(rows/16u)*512u+(size_t)nb*512u+
           (size_t)d2*64u+(size_t)d3*16u+(size_t)d4*8u+(size_t)d5*2u+d6;
}
static size_t fp16_index(uint32_t row, uint32_t column, uint32_t rows) {
    uint32_t nb=row/16, rr=row%16, d4=rr/8, d2=rr%8;
    uint32_t kb=column/16, cc=column%16, d5=cc/8, d3=(cc%8)/2, d6=cc%2;
    return (size_t)kb*(rows/16u)*256u+(size_t)nb*256u+
           (size_t)d2*32u+(size_t)d3*8u+(size_t)d4*4u+(size_t)d5*2u+d6;
}

int nr_unpack_matrix_fp8(const void *raw, size_t raw_size,
                         uint32_t rows, uint32_t columns, int tin,
                         float *output, size_t output_count) {
    if (!raw || !output || columns%32 || !dimensions_ok(rows,columns,output_count))
        return -1;
    size_t required=(size_t)rows*columns;
    if (raw_size<required) return -2;
    const uint8_t *bytes=(const uint8_t *)raw;
    for (uint32_t r=0;r<rows;r++) for (uint32_t col=0;col<columns;col++) {
        uint32_t sr=tin?nr_tin_channel_permutation(r):r;
        uint32_t sc=tin?nr_tin_channel_permutation(col):col;
        size_t index=fp8_index(sr,sc,rows);
        if (index>=raw_size) return -3;
        output[(size_t)r*columns+col]=nr_e4m3fn_decode(bytes[index]);
    }
    return 0;
}

int nr_unpack_matrix_fp16(const void *raw, size_t raw_size,
                          uint32_t rows, uint32_t columns, int tin,
                          float *output, size_t output_count) {
    if (!raw || !output || !dimensions_ok(rows,columns,output_count))
        return -1;
    size_t values=(size_t)rows*columns;
    if (values>SIZE_MAX/2 || raw_size<values*2) return -2;
    const uint8_t *bytes=(const uint8_t *)raw;
    for (uint32_t r=0;r<rows;r++) for (uint32_t col=0;col<columns;col++) {
        uint32_t sr=tin?nr_tin_channel_permutation(r):r;
        uint32_t sc=tin?nr_tin_channel_permutation(col):col;
        size_t index=fp16_index(sr,sc,rows)*2;
        if (index+1>=raw_size) return -3;
        uint16_t bits=(uint16_t)bytes[index]|(uint16_t)bytes[index+1]<<8;
        output[(size_t)r*columns+col]=nr_fp16_decode(bits);
    }
    return 0;
}
