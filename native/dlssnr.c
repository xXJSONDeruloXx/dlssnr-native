#include "dlssnr.h"
#include "model_dll.h"
#include "model_package.h"
#include "runtime.h"
#include <stdlib.h>

struct dlssnr_model {
    int source;
    union {
        nr_model_package *package;
        nr_model_dll *dll;
    } u;
};
struct dlssnr_runtime {
    nr_context *gpu;
    dlssnr_model *model;
};

uint32_t dlssnr_abi_version(void) {
    return DLSSNR_ABI_VERSION;
}
int dlssnr_model_open(const char *path, dlssnr_model **out) {
    if (!path || !out) return DLSSNR_ERROR_INVALID_ARGUMENT;
    *out=NULL;
    dlssnr_model *model=calloc(1,sizeof(*model));
    if (!model) return DLSSNR_ERROR_MODEL;
    if (!nr_model_package_open(path,&model->u.package)) {
        model->source=DLSSNR_MODEL_SOURCE_PACKAGE;
    } else if (!nr_model_dll_open(path,&model->u.dll)) {
        model->source=DLSSNR_MODEL_SOURCE_DLL;
    } else {
        free(model);
        return DLSSNR_ERROR_MODEL;
    }
    *out=model;
    return DLSSNR_OK;
}
void dlssnr_model_close(dlssnr_model *model) {
    if (!model) return;
    if (model->source==DLSSNR_MODEL_SOURCE_PACKAGE) nr_model_package_close(model->u.package);
    else if (model->source==DLSSNR_MODEL_SOURCE_DLL) nr_model_dll_close(model->u.dll);
    free(model);
}
int dlssnr_model_source_kind(const dlssnr_model *model) {
    return model ? model->source : 0;
}
uint32_t dlssnr_model_tensor_count(const dlssnr_model *model) {
    if (!model) return 0;
    return model->source==DLSSNR_MODEL_SOURCE_PACKAGE
        ? nr_model_package_tensor_count(model->u.package)
        : nr_model_dll_tensor_count(model->u.dll);
}
const char *dlssnr_model_dll_sha256(const dlssnr_model *model) {
    if (!model) return NULL;
    return model->source==DLSSNR_MODEL_SOURCE_PACKAGE
        ? nr_model_package_dll_sha256(model->u.package)
        : nr_model_dll_sha256(model->u.dll);
}
const char *dlssnr_model_weights_sha256(const dlssnr_model *model) {
    if (!model) return NULL;
    return model->source==DLSSNR_MODEL_SOURCE_PACKAGE
        ? nr_model_package_weights_sha256(model->u.package)
        : nr_model_dll_weights_sha256(model->u.dll);
}
const char *dlssnr_model_tensor_name(const dlssnr_model *model, uint32_t index) {
    if (!model) return NULL;
    return model->source==DLSSNR_MODEL_SOURCE_PACKAGE
        ? nr_model_package_tensor_name(model->u.package,index)
        : nr_model_dll_tensor_name(model->u.dll,index);
}
const void *dlssnr_model_tensor_at(const dlssnr_model *model, uint32_t index, size_t *size) {
    if (size) *size=0;
    if (!model) return NULL;
    return model->source==DLSSNR_MODEL_SOURCE_PACKAGE
        ? nr_model_package_tensor_at(model->u.package,index,size)
        : nr_model_dll_tensor_at(model->u.dll,index,size);
}
const void *dlssnr_model_tensor(const dlssnr_model *model, const char *name, size_t *size) {
    if (size) *size=0;
    if (!model) return NULL;
    return model->source==DLSSNR_MODEL_SOURCE_PACKAGE
        ? nr_model_package_tensor(model->u.package,name,size)
        : nr_model_dll_tensor(model->u.dll,name,size);
}

int dlssnr_runtime_create(dlssnr_runtime **out) {
    if (!out) return DLSSNR_ERROR_INVALID_ARGUMENT;
    *out=NULL;
    dlssnr_runtime *runtime=calloc(1,sizeof(*runtime));
    if (!runtime) return DLSSNR_ERROR_BACKEND;
    if (nr_create(&runtime->gpu)) {
        free(runtime);
        return DLSSNR_ERROR_BACKEND;
    }
    *out=runtime;
    return DLSSNR_OK;
}
void dlssnr_runtime_destroy(dlssnr_runtime *runtime) {
    if (!runtime) return;
    nr_destroy(runtime->gpu);
    free(runtime);
}
const char *dlssnr_runtime_device_name(const dlssnr_runtime *runtime) {
    return runtime ? nr_device_name(runtime->gpu) : NULL;
}
int dlssnr_runtime_set_model(dlssnr_runtime *runtime, dlssnr_model *model) {
    if (!runtime || !model) return DLSSNR_ERROR_INVALID_ARGUMENT;
    runtime->model=model;
    return DLSSNR_OK;
}
int dlssnr_stage1_profile_get(dlssnr_stage1_profile *profile) {
    if (!profile) return DLSSNR_ERROR_INVALID_ARGUMENT;
    profile->width=1280;
    profile->height=720;
    profile->channels=3;
    profile->frame=0;
    profile->reset=1;
    return DLSSNR_OK;
}
int dlssnr_stage1_evaluate_rgb_f32(dlssnr_runtime *runtime,
                                    const float *input_rgb,
                                    size_t input_floats,
                                    float *output_rgb,
                                    size_t output_floats) {
    const size_t required=(size_t)1280*720*3;
    if (!runtime || !runtime->model || !input_rgb || !output_rgb ||
        input_floats!=required || output_floats!=required)
        return DLSSNR_ERROR_INVALID_ARGUMENT;
    return DLSSNR_ERROR_UNSUPPORTED;
}
