#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct nr_model_package nr_model_package;

int nr_model_package_open(const char *path, nr_model_package **out);
void nr_model_package_close(nr_model_package *model);
uint32_t nr_model_package_tensor_count(const nr_model_package *model);
const char *nr_model_package_dll_sha256(const nr_model_package *model);
const char *nr_model_package_weights_sha256(const nr_model_package *model);
const void *nr_model_package_tensor(const nr_model_package *model, const char *name, size_t *size);
const char *nr_model_package_tensor_name(const nr_model_package *model, uint32_t index);
const void *nr_model_package_tensor_at(const nr_model_package *model, uint32_t index, size_t *size);

#ifdef __cplusplus
}
#endif
