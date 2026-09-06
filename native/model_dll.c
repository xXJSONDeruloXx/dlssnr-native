#define _POSIX_C_SOURCE 200809L
#include "model_dll.h"
#include "sha256.h"
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define NR_TENSORS 153u

static const uint8_t expected_dll[32] = {
    0x6e,0xb2,0x09,0xe7,0x64,0xf3,0x98,0x72,0x62,0x5d,0xbd,0x6a,0xba,0xf4,0x5e,0x2b,
    0xb6,0x32,0x2f,0x6f,0x27,0x0f,0x78,0x1f,0x70,0xc0,0x59,0xae,0x30,0xb3,0x92,0x7f
};
static const uint8_t expected_weights[32] = {
    0xc7,0xf8,0x6a,0xb2,0x33,0x35,0x6f,0xe7,0x3d,0x7f,0x66,0xb7,0x49,0x55,0x9c,0x85,
    0x83,0x8c,0xdc,0xe3,0x14,0x9b,0x22,0x84,0x80,0x15,0x48,0x01,0xdf,0xf8,0xbc,0x60
};

typedef struct {
    char *name;
    const uint8_t *payload;
    uint64_t size;
    uint8_t digest[32];
} nr_dll_tensor;

struct nr_model_dll {
    int fd;
    uint8_t *mapping;
    size_t mapping_size;
    nr_dll_tensor tensors[NR_TENSORS];
    char dll_sha256[65];
    char weights_sha256[65];
};

static uint16_t u16(const uint8_t *p) {
    return (uint16_t)p[0] | (uint16_t)p[1]<<8;
}
static uint32_t u32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
static int range_ok(uint64_t start, uint64_t size, uint64_t limit) {
    return start<=limit && size<=limit-start;
}
static void hex32(char out[65], const uint8_t data[32]) {
    static const char hex[]="0123456789abcdef";
    for (unsigned i=0;i<32;i++) {
        out[i*2]=hex[data[i]>>4];
        out[i*2+1]=hex[data[i]&15];
    }
    out[64]=0;
}
static int safe_tensor_name(const char *s) {
    if (strncmp(s,"block",5)) return 0;
    const char *p=s+5;
    if (*p<'0'||*p>'9') return 0;
    while (*p>='0'&&*p<='9') p++;
    if (strncmp(p,".layer",6)) return 0;
    p+=6;
    if (*p<'0'||*p>'9') return 0;
    while (*p>='0'&&*p<='9') p++;
    if (*p++!='.') return 0;
    if (!*p) return 0;
    for (;*p;p++) {
        unsigned char c=(unsigned char)*p;
        if (!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_')) return 0;
    }
    return 1;
}
static int compare_tensors(const void *a, const void *b) {
    const nr_dll_tensor *aa=(const nr_dll_tensor *)a;
    const nr_dll_tensor *bb=(const nr_dll_tensor *)b;
    return strcmp(aa->name,bb->name);
}

typedef struct { uint32_t va,size,ptr; } nr_section;

static int map_rva(const nr_section *sections, uint16_t count, uint32_t address,
                   uint32_t size, size_t file_size, uint64_t *offset) {
    for (uint16_t i=0;i<count;i++) {
        uint64_t va=sections[i].va,raw=sections[i].size,ptr=sections[i].ptr;
        if ((uint64_t)address>=va && (uint64_t)address+size>=address &&
            (uint64_t)address+size<=va+raw) {
            uint64_t result=ptr+((uint64_t)address-va);
            if (!range_ok(result,size,file_size)) return -1;
            *offset=result;
            return 0;
        }
    }
    return -1;
}
static int resource_entries(const uint8_t *file, size_t file_size,
                            uint64_t resource_offset, uint32_t resource_size,
                            uint32_t relative, uint64_t *entries, uint32_t *count) {
    if (relative>resource_size || resource_size-relative<16) return -1;
    uint64_t off=resource_offset+relative;
    if (!range_ok(off,16,file_size)) return -1;
    uint32_t n=(uint32_t)u16(file+off+12)+(uint32_t)u16(file+off+14);
    if (n>1024 || !range_ok(off+16,(uint64_t)n*8,file_size)) return -1;
    *entries=off+16;*count=n;
    return 0;
}
static int resource_name_equals(const uint8_t *file, size_t file_size,
                                uint64_t resource_offset, uint32_t resource_size,
                                uint32_t key, const char *expected) {
    if (!(key&0x80000000u)) return 0;
    uint32_t relative=key&0x7fffffffu;
    if (relative>resource_size || resource_size-relative<2) return 0;
    uint64_t off=resource_offset+relative;
    if (!range_ok(off,2,file_size)) return 0;
    uint16_t length=u16(file+off);
    size_t expected_len=strlen(expected);
    if (length!=expected_len || !range_ok(off+2,(uint64_t)length*2,file_size)) return 0;
    for (uint16_t i=0;i<length;i++)
        if (u16(file+off+2+(uint64_t)i*2)!=(uint8_t)expected[i]) return 0;
    return 1;
}
static int find_weights_resource(const uint8_t *file, size_t file_size,
                                 const nr_section *sections, uint16_t section_count,
                                 uint32_t resource_rva, uint32_t resource_size,
                                 const uint8_t **payload, uint32_t *payload_size) {
    uint64_t resource_offset;
    if (map_rva(sections,section_count,resource_rva,resource_size,file_size,&resource_offset)) return -1;
    uint64_t root_entries;uint32_t root_count;
    if (resource_entries(file,file_size,resource_offset,resource_size,0,&root_entries,&root_count)) return -1;
    for (uint32_t i=0;i<root_count;i++) {
        uint32_t key=u32(file+root_entries+(uint64_t)i*8);
        uint32_t target=u32(file+root_entries+(uint64_t)i*8+4);
        if (key!=10u || !(target&0x80000000u)) continue;
        uint64_t type_entries;uint32_t type_count;
        if (resource_entries(file,file_size,resource_offset,resource_size,target&0x7fffffffu,&type_entries,&type_count)) return -1;
        for (uint32_t j=0;j<type_count;j++) {
            uint32_t name=u32(file+type_entries+(uint64_t)j*8);
            uint32_t name_target=u32(file+type_entries+(uint64_t)j*8+4);
            if (!resource_name_equals(file,file_size,resource_offset,resource_size,name,"WEIGHTS_HT") ||
                !(name_target&0x80000000u)) continue;
            uint64_t language_entries;uint32_t language_count;
            if (resource_entries(file,file_size,resource_offset,resource_size,name_target&0x7fffffffu,
                                 &language_entries,&language_count) || !language_count) return -1;
            uint32_t leaf=u32(file+language_entries+4);
            if (leaf&0x80000000u || leaf>resource_size || resource_size-leaf<16) return -1;
            uint64_t data_entry=resource_offset+leaf;
            if (!range_ok(data_entry,16,file_size)) return -1;
            uint32_t va=u32(file+data_entry),size=u32(file+data_entry+4);
            uint64_t data_offset;
            if (map_rva(sections,section_count,va,size,file_size,&data_offset)) return -1;
            *payload=file+data_offset;*payload_size=size;
            return 0;
        }
    }
    return -1;
}
static int parse_tensors(nr_model_dll *model, const uint8_t *data, uint32_t size) {
    if (size<16 || u32(data)!=size) return -1;
    uint32_t name_length=u32(data+8);
    uint64_t pos=16;
    uint32_t count=0;
    while (pos<size && count<NR_TENSORS) {
        if (!name_length || name_length>256 || !range_ok(pos,(uint64_t)name_length+28,size)) return -1;
        char *name=malloc((size_t)name_length+1);
        if (!name) return -1;
        memcpy(name,data+pos,name_length);name[name_length]=0;
        if (!safe_tensor_name(name)) { free(name); return -1; }
        pos+=name_length;
        uint32_t payload_size=u32(data+pos+16);
        pos+=28;
        if (!payload_size || !range_ok(pos,payload_size,size)) { free(name); return -1; }
        model->tensors[count].name=name;
        model->tensors[count].payload=data+pos;
        model->tensors[count].size=payload_size;
        nr_sha256 hash;
        nr_sha256_init(&hash);nr_sha256_update(&hash,data+pos,payload_size);
        nr_sha256_final(&hash,model->tensors[count].digest);
        count++;
        pos+=payload_size;
        if (pos==size) break;
        if (!range_ok(pos,8,size)) return -1;
        for (unsigned z=0;z<8;z++) if (data[pos+z]) return -1;
        pos+=8;
        if (size-pos<20) {
            if (size-pos && (size-pos!=12 || u32(data+pos)!=1 || u32(data+pos+4)!=0 ||
                            u32(data+pos+8)!=payload_size/2)) return -1;
            pos=size;
            break;
        }
        name_length=u32(data+pos+12);
        pos+=20;
    }
    if (count!=NR_TENSORS) return -1;
    qsort(model->tensors,NR_TENSORS,sizeof(model->tensors[0]),compare_tensors);
    nr_sha256 aggregate;
    nr_sha256_init(&aggregate);
    for (uint32_t i=0;i<NR_TENSORS;i++) {
        if (i && !strcmp(model->tensors[i-1].name,model->tensors[i].name)) return -1;
        char hex[65],line[384];
        hex32(hex,model->tensors[i].digest);
        int n=snprintf(line,sizeof(line),"%s\t%llu\t%s\n",model->tensors[i].name,
                       (unsigned long long)model->tensors[i].size,hex);
        if (n<0 || (size_t)n>=sizeof(line)) return -1;
        nr_sha256_update(&aggregate,line,(size_t)n);
    }
    uint8_t digest[32];
    nr_sha256_final(&aggregate,digest);
    if (memcmp(digest,expected_weights,32)) return -1;
    hex32(model->weights_sha256,digest);
    return 0;
}

int nr_model_dll_open(const char *path, nr_model_dll **out) {
    if (!path || !out) return -1;
    *out=NULL;
    int fd=open(path,O_RDONLY);
    if (fd<0) return -2;
    struct stat st;
    if (fstat(fd,&st) || st.st_size<64 || (uintmax_t)st.st_size>SIZE_MAX) { close(fd); return -3; }
    size_t bytes=(size_t)st.st_size;
    uint8_t *map=mmap(NULL,bytes,PROT_READ,MAP_PRIVATE,fd,0);
    if (map==MAP_FAILED) { close(fd); return -4; }
    nr_sha256 file_hash;uint8_t file_digest[32];
    nr_sha256_init(&file_hash);nr_sha256_update(&file_hash,map,bytes);nr_sha256_final(&file_hash,file_digest);
    if (memcmp(file_digest,expected_dll,32) || map[0]!='M' || map[1]!='Z') {
        munmap(map,bytes);close(fd);return -5;
    }
    if (!range_ok(0x3c,4,bytes)) { munmap(map,bytes);close(fd);return -6; }
    uint32_t pe=u32(map+0x3c);
    if (!range_ok(pe,24,bytes) || memcmp(map+pe,"PE\0\0",4)) { munmap(map,bytes);close(fd);return -6; }
    uint16_t section_count=u16(map+pe+6),optional_size=u16(map+pe+20);
    uint64_t opt=(uint64_t)pe+24;
    if (!section_count || section_count>128 || !range_ok(opt,optional_size,bytes) || optional_size<96) {
        munmap(map,bytes);close(fd);return -6;
    }
    uint16_t magic=u16(map+opt);
    uint64_t dd=opt+(magic==0x20b?112:magic==0x10b?96:UINT64_MAX);
    if (dd==UINT64_MAX || !range_ok(dd+16,8,opt+optional_size)) { munmap(map,bytes);close(fd);return -6; }
    uint32_t resource_rva=u32(map+dd+16),resource_size=u32(map+dd+20);
    if (!resource_rva || !resource_size) { munmap(map,bytes);close(fd);return -6; }
    uint64_t section_table=opt+optional_size;
    if (!range_ok(section_table,(uint64_t)section_count*40,bytes)) { munmap(map,bytes);close(fd);return -6; }
    nr_section *sections=calloc(section_count,sizeof(*sections));
    if (!sections) { munmap(map,bytes);close(fd);return -7; }
    for (uint16_t i=0;i<section_count;i++) {
        uint64_t off=section_table+(uint64_t)i*40;
        sections[i].va=u32(map+off+12);
        sections[i].size=u32(map+off+16);
        sections[i].ptr=u32(map+off+20);
    }
    const uint8_t *resource;uint32_t resource_bytes;
    int found=find_weights_resource(map,bytes,sections,section_count,resource_rva,resource_size,
                                    &resource,&resource_bytes);
    free(sections);
    if (found) { munmap(map,bytes);close(fd);return -8; }
    nr_model_dll *model=calloc(1,sizeof(*model));
    if (!model) { munmap(map,bytes);close(fd);return -7; }
    model->fd=fd;model->mapping=map;model->mapping_size=bytes;
    hex32(model->dll_sha256,file_digest);
    if (parse_tensors(model,resource,resource_bytes)) {
        nr_model_dll_close(model);
        return -9;
    }
    *out=model;
    return 0;
}
void nr_model_dll_close(nr_model_dll *model) {
    if (!model) return;
    for (uint32_t i=0;i<NR_TENSORS;i++) free(model->tensors[i].name);
    if (model->mapping && model->mapping!=MAP_FAILED) munmap(model->mapping,model->mapping_size);
    if (model->fd>=0) close(model->fd);
    free(model);
}
uint32_t nr_model_dll_tensor_count(const nr_model_dll *model) { return model?NR_TENSORS:0; }
const char *nr_model_dll_sha256(const nr_model_dll *model) { return model?model->dll_sha256:NULL; }
const char *nr_model_dll_weights_sha256(const nr_model_dll *model) { return model?model->weights_sha256:NULL; }
const char *nr_model_dll_tensor_name(const nr_model_dll *model, uint32_t index) {
    return model&&index<NR_TENSORS?model->tensors[index].name:NULL;
}
const void *nr_model_dll_tensor_at(const nr_model_dll *model, uint32_t index, size_t *size) {
    if (size) *size=0;
    if (!model || index>=NR_TENSORS) return NULL;
    if (size) *size=(size_t)model->tensors[index].size;
    return model->tensors[index].payload;
}
const void *nr_model_dll_tensor(const nr_model_dll *model, const char *name, size_t *size) {
    if (size) *size=0;
    if (!model || !name) return NULL;
    uint32_t lo=0,hi=NR_TENSORS;
    while (lo<hi) {
        uint32_t mid=lo+(hi-lo)/2;
        int cmp=strcmp(name,model->tensors[mid].name);
        if (!cmp) return nr_model_dll_tensor_at(model,mid,size);
        if (cmp<0) hi=mid; else lo=mid+1;
    }
    return NULL;
}
