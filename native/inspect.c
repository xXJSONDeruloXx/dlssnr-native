#include "dlssnr.h"
#include <stdio.h>
#include <string.h>

static const char *source_name(int source) {
    if (source==DLSSNR_MODEL_SOURCE_DLL) return "dll";
    if (source==DLSSNR_MODEL_SOURCE_PACKAGE) return "package";
    return "unknown";
}

int main(int argc, char **argv) {
    int device=0;
    const char *path=NULL;
    for (int i=1;i<argc;i++) {
        if (!strcmp(argv[i],"--device")) device=1;
        else if (!path) path=argv[i];
        else {
            fprintf(stderr,"usage: %s [--device] MODEL\n",argv[0]);
            return 64;
        }
    }
    if (!path) {
        fprintf(stderr,"usage: %s [--device] MODEL\n",argv[0]);
        return 64;
    }
    dlssnr_model *model=NULL;
    int result=dlssnr_model_open(path,&model);
    if (result) {
        fprintf(stderr,"model_open=%d\n",result);
        return 2;
    }
    printf("abi=%u\n",dlssnr_abi_version());
    printf("source=%s\n",source_name(dlssnr_model_source_kind(model)));
    printf("tensors=%u\n",dlssnr_model_tensor_count(model));
    printf("dll_sha256=%s\n",dlssnr_model_dll_sha256(model));
    printf("weights_sha256=%s\n",dlssnr_model_weights_sha256(model));
    dlssnr_stage1_profile profile;
    if (!dlssnr_stage1_profile_get(&profile))
        printf("stage1=%ux%u rgb frame=%u reset=%u\n",
               profile.width,profile.height,profile.frame,profile.reset);
    dlssnr_stage1_inventory inventory;
    if (!dlssnr_model_stage1_inventory(model,&inventory))
        printf("stage1_tensors=%u/%u auxiliary=%u\n",
               inventory.present_required_tensors,inventory.required_tensors,
               inventory.auxiliary_tensors);
    if (device) {
        dlssnr_runtime *runtime=NULL;
        result=dlssnr_runtime_create(&runtime);
        if (result) {
            fprintf(stderr,"runtime_create=%d\n",result);
            dlssnr_model_close(model);
            return 3;
        }
        result=dlssnr_runtime_set_model(runtime,model);
        if (result) {
            fprintf(stderr,"runtime_set_model=%d\n",result);
            dlssnr_runtime_destroy(runtime);
            dlssnr_model_close(model);
            return 4;
        }
        printf("device=%s\n",dlssnr_runtime_device_name(runtime));
        printf("stage1_model_bound=1\n");
        dlssnr_runtime_destroy(runtime);
    }
    dlssnr_model_close(model);
    return 0;
}
