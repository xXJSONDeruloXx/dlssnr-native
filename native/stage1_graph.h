#pragma once
#include <stddef.h>
#include <stdint.h>

#define NR_STAGE1_REQUIRED_TENSORS 152u

enum nr_stage1_tensor_role {
    NR_STAGE1_PRE = 1,
    NR_STAGE1_ENCODER,
    NR_STAGE1_CORE_ENCODER,
    NR_STAGE1_VIT_INPUT,
    NR_STAGE1_VIT,
    NR_STAGE1_VIT_OUTPUT,
    NR_STAGE1_CORE_DECODER,
    NR_STAGE1_DECODER,
    NR_STAGE1_POST
};

uint32_t nr_stage1_required_tensor_count(void);
int nr_stage1_required_tensor(uint32_t index, char *name, size_t name_size,
                              enum nr_stage1_tensor_role *role);
int nr_stage1_is_required_tensor(const char *name);
const char *nr_stage1_role_name(enum nr_stage1_tensor_role role);
