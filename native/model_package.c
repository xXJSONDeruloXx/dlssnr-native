#define _POSIX_C_SOURCE 200809L
#include "model_package.h"
#include "sha256.h"
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define NR_MODEL_MAGIC "DLSSNRM1"
#define NR_MODEL_VERSION 1u
#define NR_MODEL_TENSORS 153u
#define NR_HEADER_SIZE 88u
#define NR_ENTRY_SIZE 56u
#define NR_ALIGNMENT 64u

static const uint8_t expected_dll[32] = {
    0x6e,0xb2,0x09,0xe7,0x64,0xf3,0x98,0x72,0x62,0x5d,0xbd,0x6a,0xba,0xf4,0x5e,0x2b,
    0xb6,0x32,0x2f,0x6f,0x27,0x0f,0x78,0x1f,0x70,0xc0,0x59,0xae,0x30,0xb3,0x92,0x7f
};
static const uint8_t expected_weights[32] = {
    0xc7,0xf8,0x6a,0xb2,0x33,0x35,0x6f,0xe7,0x3d,0x7f,0x66,0xb7,0x49,0x55,0x9c,0x85,
    0x83,0x8c,0xdc,0xe3,0x14,0x9b,0x22,0x84,0x80,0x15,0x48,0x01,0xdf,0xf8,0xbc,0x60
};

typedef struct {
    const char *name;
    uint16_t name_len;
    uint64_t offset;
    uint64_t size;
    uint8_t digest[32];
} nr_tensor_entry;

struct nr_model_package {
    int fd;
    void *mapping;
    size_t mapping_size;
    uint64_t payload_start;
    nr_tensor_entry entries[NR_MODEL_TENSORS];
    char dll_sha256[65];
    char weights_sha256[65];
};

static uint16_t u16(const uint8_t *p) {
    return (uint16_t)p[0] | (uint16_t)p[1] << 8;
}
static uint32_t u32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static uint64_t u64(const uint8_t *p) {
    uint64_t value=0;
    for (unsigned i=0;i<8;i++) value |= (uint64_t)p[i] << (8*i);
    return value;
}
static void hex32(char out[65], const uint8_t *data) {
    static const char hex[]="0123456789abcdef";
    for (unsigned i=0;i<32;i++) {
        out[i*2]=hex[data[i]>>4];
        out[i*2+1]=hex[data[i]&15];
    }
    out[64]=0;
}
static int range_ok(uint64_t start, uint64_t size, uint64_t limit) {
    return start<=limit && size<=limit-start;
}
static int span_compare(const char *a, uint16_t an, const char *b, uint16_t bn) {
    size_t n=an<bn?an:bn;
    int cmp=memcmp(a,b,n);
    if (cmp) return cmp;
    return an<bn?-1:an>bn?1:0;
}
static int safe_name(const char *name, uint16_t size) {
    if (size<15 || memcmp(name,"block",5)) return 0;
    for (uint16_t i=0;i<size;i++) {
        unsigned char c=(unsigned char)name[i];
        if (!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='.'||c=='_'))
            return 0;
    }
    return 1;
}

int nr_model_package_open(const char *path, nr_model_package **out) {
    if (!path || !out) return -1;
    *out=NULL;
    int fd=open(path,O_RDONLY);
    if (fd<0) return -2;
    struct stat st;
    if (fstat(fd,&st) || st.st_size<(off_t)NR_HEADER_SIZE) { close(fd); return -3; }
    if ((uintmax_t)st.st_size>SIZE_MAX) { close(fd); return -3; }
    size_t bytes=(size_t)st.st_size;
    uint8_t *map=mmap(NULL,bytes,PROT_READ,MAP_PRIVATE,fd,0);
    if (map==MAP_FAILED) { close(fd); return -4; }
    if (memcmp(map,NR_MODEL_MAGIC,8) || u32(map+8)!=NR_MODEL_VERSION ||
        u32(map+12)!=NR_MODEL_TENSORS || memcmp(map+16,expected_dll,32) ||
        memcmp(map+48,expected_weights,32)) {
        munmap(map,bytes); close(fd); return -5;
    }
    uint64_t payload_start=u64(map+80);
    if (payload_start<NR_HEADER_SIZE || payload_start>(uint64_t)bytes ||
        payload_start%NR_ALIGNMENT) {
        munmap(map,bytes); close(fd); return -6;
    }
    nr_model_package *model=calloc(1,sizeof(*model));
    if (!model) { munmap(map,bytes); close(fd); return -7; }
    model->fd=fd; model->mapping=map; model->mapping_size=bytes; model->payload_start=payload_start;
    hex32(model->dll_sha256,map+16);
    hex32(model->weights_sha256,map+48);
    uint64_t cursor=NR_HEADER_SIZE;
    uint64_t previous_end=0;
    for (uint32_t i=0;i<NR_MODEL_TENSORS;i++) {
        if (!range_ok(cursor,NR_ENTRY_SIZE,payload_start)) goto invalid;
        const uint8_t *entry=map+cursor;
        uint16_t name_len=u16(entry);
        uint64_t offset=u64(entry+8);
        uint64_t size=u64(entry+16);
        if (memcmp(entry+2,(uint8_t[6]){0},6)) goto invalid;
        cursor+=NR_ENTRY_SIZE;
        if (!name_len || name_len>256 || !range_ok(cursor,name_len,payload_start) ||
            !size || offset%NR_ALIGNMENT || offset>UINT64_MAX-payload_start ||
            !range_ok(payload_start+offset,size,bytes)) goto invalid;
        const char *name=(const char *)(map+cursor);
        if (!safe_name(name,name_len)) goto invalid;
        if (i && span_compare(model->entries[i-1].name,model->entries[i-1].name_len,name,name_len)>=0)
            goto invalid;
        if (offset<previous_end) goto invalid;
        model->entries[i].name=name;
        model->entries[i].name_len=name_len;
        model->entries[i].offset=offset;
        model->entries[i].size=size;
        memcpy(model->entries[i].digest,entry+24,32);
        previous_end=offset+size;
        cursor+=name_len;
    }
    for (uint64_t i=cursor;i<payload_start;i++) if (map[i]) goto invalid;
    if (payload_start+previous_end!=(uint64_t)bytes) goto invalid;

    nr_sha256 aggregate;
    nr_sha256_init(&aggregate);
    for (uint32_t i=0;i<NR_MODEL_TENSORS;i++) {
        nr_tensor_entry *entry=&model->entries[i];
        const uint8_t *payload=map+payload_start+entry->offset;
        nr_sha256 payload_hash;
        uint8_t digest[32];
        char hex[65],line[384];
        nr_sha256_init(&payload_hash);
        nr_sha256_update(&payload_hash,payload,(size_t)entry->size);
        nr_sha256_final(&payload_hash,digest);
        if (memcmp(digest,entry->digest,32)) goto invalid;
        hex32(hex,digest);
        int n=snprintf(line,sizeof(line),"%.*s\t%llu\t%s\n",
                       (int)entry->name_len,entry->name,
                       (unsigned long long)entry->size,hex);
        if (n<0 || (size_t)n>=sizeof(line)) goto invalid;
        nr_sha256_update(&aggregate,line,(size_t)n);
    }
    uint8_t aggregate_digest[32];
    nr_sha256_final(&aggregate,aggregate_digest);
    if (memcmp(aggregate_digest,expected_weights,32)) goto invalid;

    *out=model;
    return 0;
invalid:
    nr_model_package_close(model);
    return -8;
}

void nr_model_package_close(nr_model_package *model) {
    if (!model) return;
    if (model->mapping && model->mapping!=MAP_FAILED) munmap(model->mapping,model->mapping_size);
    if (model->fd>=0) close(model->fd);
    free(model);
}
uint32_t nr_model_package_tensor_count(const nr_model_package *model) {
    return model ? NR_MODEL_TENSORS : 0;
}
const char *nr_model_package_dll_sha256(const nr_model_package *model) {
    return model ? model->dll_sha256 : NULL;
}
const char *nr_model_package_weights_sha256(const nr_model_package *model) {
    return model ? model->weights_sha256 : NULL;
}
const char *nr_model_package_tensor_name(const nr_model_package *model, uint32_t index) {
    if (!model || index>=NR_MODEL_TENSORS) return NULL;
    static _Thread_local char name[257];
    uint16_t n=model->entries[index].name_len;
    memcpy(name,model->entries[index].name,n); name[n]=0;
    return name;
}
const void *nr_model_package_tensor_at(const nr_model_package *model, uint32_t index, size_t *size) {
    if (size) *size=0;
    if (!model || index>=NR_MODEL_TENSORS) return NULL;
    const nr_tensor_entry *entry=&model->entries[index];
    if (size) *size=(size_t)entry->size;
    return (const uint8_t *)model->mapping+model->payload_start+entry->offset;
}
const void *nr_model_package_tensor(const nr_model_package *model, const char *name, size_t *size) {
    if (size) *size=0;
    if (!model || !name) return NULL;
    size_t n=strlen(name);
    uint32_t lo=0,hi=NR_MODEL_TENSORS;
    while (lo<hi) {
        uint32_t mid=lo+(hi-lo)/2;
        const nr_tensor_entry *entry=&model->entries[mid];
        int cmp=span_compare(name,(uint16_t)(n>UINT16_MAX?UINT16_MAX:n),
                             entry->name,entry->name_len);
        if (!cmp && n<=UINT16_MAX)
            return nr_model_package_tensor_at(model,mid,size);
        if (cmp<0) hi=mid; else lo=mid+1;
    }
    return NULL;
}
