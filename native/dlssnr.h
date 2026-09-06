#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define DLSSNR_ABI_VERSION 1u

enum {
    DLSSNR_OK = 0,
    DLSSNR_ERROR_INVALID_ARGUMENT = -1000,
    DLSSNR_ERROR_MODEL = -1001,
    DLSSNR_ERROR_BACKEND = -1002,
    DLSSNR_ERROR_UNSUPPORTED = -1003
};

typedef struct dlssnr_model dlssnr_model;
typedef struct dlssnr_runtime dlssnr_runtime;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t frame;
    uint32_t reset;
} dlssnr_stage1_profile;

uint32_t dlssnr_abi_version(void);
int dlssnr_model_open(const char *path, dlssnr_model **out);
void dlssnr_model_close(dlssnr_model *model);
uint32_t dlssnr_model_tensor_count(const dlssnr_model *model);
const char *dlssnr_model_dll_sha256(const dlssnr_model *model);
const char *dlssnr_model_weights_sha256(const dlssnr_model *model);
const void *dlssnr_model_tensor(const dlssnr_model *model, const char *name, size_t *size);

int dlssnr_runtime_create(dlssnr_runtime **out);
void dlssnr_runtime_destroy(dlssnr_runtime *runtime);
const char *dlssnr_runtime_device_name(const dlssnr_runtime *runtime);
int dlssnr_runtime_set_model(dlssnr_runtime *runtime, dlssnr_model *model);
int dlssnr_stage1_profile_get(dlssnr_stage1_profile *profile);
int dlssnr_stage1_evaluate_rgb_f32(dlssnr_runtime *runtime,
                                    const float *input_rgb,
                                    size_t input_floats,
                                    float *output_rgb,
                                    size_t output_floats);

#ifdef __cplusplus
}
#endif
