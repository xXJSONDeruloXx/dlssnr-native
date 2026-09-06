#include "stage1_graph.h"
#include <stdio.h>
#include <string.h>

static int emit(char *name, size_t size, unsigned block, unsigned layer,
                enum nr_stage1_tensor_role role,
                enum nr_stage1_tensor_role *out_role) {
    if (!name || !size) return -1;
    int n=snprintf(name,size,"block%u.layer%u.layer",block,layer);
    if (n<0 || (size_t)n>=size) return -1;
    if (out_role) *out_role=role;
    return 0;
}

uint32_t nr_stage1_required_tensor_count(void) {
    return NR_STAGE1_REQUIRED_TENSORS;
}

int nr_stage1_required_tensor(uint32_t index, char *name, size_t name_size,
                              enum nr_stage1_tensor_role *role) {
    if (index>=NR_STAGE1_REQUIRED_TENSORS) return -1;
    if (index==0) return emit(name,name_size,0,0,NR_STAGE1_PRE,role);
    index--;

    if (index<22) return emit(name,name_size,index+1,0,NR_STAGE1_ENCODER,role);
    index-=22;

    if (index<32) {
        unsigned block=23+index/4;
        unsigned layer=index%4;
        return emit(name,name_size,block,layer,NR_STAGE1_CORE_ENCODER,role);
    }
    index-=32;

    if (index==0) return emit(name,name_size,30,4,NR_STAGE1_VIT_INPUT,role);
    index--;

    if (index<40) {
        unsigned block=31+index/5;
        unsigned layer=index%5;
        return emit(name,name_size,block,layer,NR_STAGE1_VIT,role);
    }
    index-=40;

    if (index==0) return emit(name,name_size,39,0,NR_STAGE1_VIT_OUTPUT,role);
    index--;

    if (index<32) {
        unsigned block=40+index/4;
        unsigned layer=index%4;
        return emit(name,name_size,block,layer,NR_STAGE1_CORE_DECODER,role);
    }
    index-=32;

    if (index<22) return emit(name,name_size,48+index,0,NR_STAGE1_DECODER,role);
    index-=22;

    if (index==0) return emit(name,name_size,70,0,NR_STAGE1_POST,role);
    return -1;
}

int nr_stage1_is_required_tensor(const char *name) {
    if (!name) return 0;
    char candidate[64];
    for (uint32_t i=0;i<NR_STAGE1_REQUIRED_TENSORS;i++) {
        if (nr_stage1_required_tensor(i,candidate,sizeof(candidate),NULL)) return 0;
        if (!strcmp(candidate,name)) return 1;
    }
    return 0;
}

const char *nr_stage1_role_name(enum nr_stage1_tensor_role role) {
    switch (role) {
    case NR_STAGE1_PRE:return "pre";
    case NR_STAGE1_ENCODER:return "encoder";
    case NR_STAGE1_CORE_ENCODER:return "core_encoder";
    case NR_STAGE1_VIT_INPUT:return "vit_input";
    case NR_STAGE1_VIT:return "vit";
    case NR_STAGE1_VIT_OUTPUT:return "vit_output";
    case NR_STAGE1_CORE_DECODER:return "core_decoder";
    case NR_STAGE1_DECODER:return "decoder";
    case NR_STAGE1_POST:return "post";
    default:return "unknown";
    }
}
