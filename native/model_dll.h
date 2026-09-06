#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct nr_model_dll nr_model_dll;

int nr_model_dll_open(const char *path, nr_model_dll **out);
void nr_model_dll_close(nr_model_dll *model);
uint32_t nr_model_dll_tensor_count(const nr_model_dll *model);
const char *nr_model_dll_sha256(const nr_model_dll *model);
const char *nr_model_dll_weights_sha256(const nr_model_dll *model);
const void *nr_model_dll_tensor(const nr_model_dll *model, const char *name, size_t *size);
const char *nr_model_dll_tensor_name(const nr_model_dll *model, uint32_t index);
const void *nr_model_dll_tensor_at(const nr_model_dll *model, uint32_t index, size_t *size);

#ifdef __cplusplus
}
#endif
