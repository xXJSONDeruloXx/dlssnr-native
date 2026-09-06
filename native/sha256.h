#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t state[8];
    uint64_t bytes;
    uint8_t block[64];
    size_t used;
} nr_sha256;

void nr_sha256_init(nr_sha256 *ctx);
void nr_sha256_update(nr_sha256 *ctx, const void *data, size_t size);
void nr_sha256_final(nr_sha256 *ctx, uint8_t digest[32]);
