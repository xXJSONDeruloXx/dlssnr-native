#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct nr_context nr_context;
typedef struct nr_buffer nr_buffer;
typedef struct nr_pipeline nr_pipeline;
int nr_create(nr_context **out);
void nr_destroy(nr_context *ctx);
int nr_buffer_create(nr_context *ctx, size_t size, nr_buffer **out);
uint64_t nr_buffer_address(nr_context *ctx, nr_buffer *buffer);
int nr_buffer_write(nr_context *ctx, nr_buffer *buffer, size_t offset, const void *data, size_t size);
int nr_buffer_read(nr_context *ctx, nr_buffer *buffer, size_t offset, void *data, size_t size);
void nr_buffer_destroy(nr_context *ctx, nr_buffer *buffer);
int nr_pipeline_create(nr_context *ctx, const void *spirv, size_t size, nr_pipeline **out);
void nr_pipeline_destroy(nr_context *ctx, nr_pipeline *pipeline);
int nr_dispatch(nr_context *ctx, nr_pipeline *pipeline, nr_buffer *parameters,
                nr_buffer *output, uint32_t x, uint32_t y, uint32_t z);
const char *nr_device_name(nr_context *ctx);
#ifdef __cplusplus
}
#endif
