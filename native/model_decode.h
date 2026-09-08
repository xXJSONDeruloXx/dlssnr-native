#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

float nr_e4m3fn_decode(uint8_t bits);
float nr_fp16_decode(uint16_t bits);
uint32_t nr_tin_channel_permutation(uint32_t channel);
int nr_unpack_matrix_fp8(const void *raw, size_t raw_size,
                         uint32_t rows, uint32_t columns, int tin,
                         float *output, size_t output_count);
int nr_unpack_matrix_fp16(const void *raw, size_t raw_size,
                          uint32_t rows, uint32_t columns, int tin,
                          float *output, size_t output_count);

#ifdef __cplusplus
}
#endif
