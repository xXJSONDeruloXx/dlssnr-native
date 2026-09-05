/* SPDX-License-Identifier: GPL-3.0-only
 * Native Vulkan buffer/compute execution. No model schedule is implied by
 * successful dispatch; this is the numerical backend used to validate it.
 * One context is externally synchronized by its caller.
 */
#include "runtime.h"
#include <vulkan/vulkan.h>
#include <stdlib.h>
#include <string.h>

struct nr_context {
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueue queue;
    VkCommandPool commands;
    VkDescriptorSetLayout descriptors;
    VkPipelineLayout layout;
    VkDescriptorPool pool;
    VkPhysicalDeviceMemoryProperties memory;
    VkPhysicalDeviceProperties properties;
};
struct nr_buffer { VkBuffer handle; VkDeviceMemory memory; size_t size; void *mapped; };
struct nr_pipeline { VkPipeline handle; };
#define TRY(call) do { result=(call); if(result!=VK_SUCCESS) goto fail; } while(0)

int nr_create(nr_context **out) {
    if(!out)return VK_ERROR_INITIALIZATION_FAILED;
    *out=NULL;VkResult result=VK_ERROR_INITIALIZATION_FAILED;
    nr_context *c=calloc(1,sizeof(*c));if(!c)return VK_ERROR_OUT_OF_HOST_MEMORY;
    VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.pApplicationName="dlssnr-native",.apiVersion=VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
    TRY(vkCreateInstance(&ici,NULL,&c->instance));
    uint32_t count=0;TRY(vkEnumeratePhysicalDevices(c->instance,&count,NULL));
    if(!count){result=VK_ERROR_INITIALIZATION_FAILED;goto fail;}
    VkPhysicalDevice *devices=calloc(count,sizeof(*devices));if(!devices){result=VK_ERROR_OUT_OF_HOST_MEMORY;goto fail;}
    result=vkEnumeratePhysicalDevices(c->instance,&count,devices);
    if(result==VK_SUCCESS)for(uint32_t i=0;i<count;i++){
        VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(devices[i],&p);
        if(p.vendorID==0x1002 && p.apiVersion>=VK_API_VERSION_1_2){c->physical=devices[i];c->properties=p;break;}
    }
    free(devices);if(result!=VK_SUCCESS || !c->physical){result=VK_ERROR_FEATURE_NOT_PRESENT;goto fail;}
    vkGetPhysicalDeviceMemoryProperties(c->physical,&c->memory);
    uint32_t queues=0;vkGetPhysicalDeviceQueueFamilyProperties(c->physical,&queues,NULL);
    VkQueueFamilyProperties *qp=calloc(queues,sizeof(*qp));if(!qp){result=VK_ERROR_OUT_OF_HOST_MEMORY;goto fail;}
    vkGetPhysicalDeviceQueueFamilyProperties(c->physical,&queues,qp);
    uint32_t family=UINT32_MAX;
    for(uint32_t i=0;i<queues;i++)if(qp[i].queueFlags&VK_QUEUE_COMPUTE_BIT){family=i;break;}
    free(qp);if(family==UINT32_MAX){result=VK_ERROR_FEATURE_NOT_PRESENT;goto fail;}
    VkPhysicalDeviceVulkan12Features f12={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 features={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&f12};
    vkGetPhysicalDeviceFeatures2(c->physical,&features);
    if(!f12.bufferDeviceAddress || !features.features.shaderInt64){result=VK_ERROR_FEATURE_NOT_PRESENT;goto fail;}
    VkPhysicalDeviceVulkan12Features enabled={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,.bufferDeviceAddress=VK_TRUE,.scalarBlockLayout=f12.scalarBlockLayout};
    VkPhysicalDeviceFeatures core={.shaderInt64=VK_TRUE,.shaderFloat64=features.features.shaderFloat64};
    float priority=1;
    VkDeviceQueueCreateInfo queue={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=family,.queueCount=1,.pQueuePriorities=&priority};
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.pNext=&enabled,.queueCreateInfoCount=1,.pQueueCreateInfos=&queue,.pEnabledFeatures=&core};
    TRY(vkCreateDevice(c->physical,&dci,NULL,&c->device));vkGetDeviceQueue(c->device,family,0,&c->queue);
    VkCommandPoolCreateInfo cp={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=family,.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
    TRY(vkCreateCommandPool(c->device,&cp,NULL,&c->commands));
    VkDescriptorSetLayoutBinding bindings[2]={{.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT},{.binding=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT}};
    VkDescriptorSetLayoutCreateInfo ds={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=2,.pBindings=bindings};
    TRY(vkCreateDescriptorSetLayout(c->device,&ds,NULL,&c->descriptors));
    VkPipelineLayoutCreateInfo pl={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&c->descriptors};
    TRY(vkCreatePipelineLayout(c->device,&pl,NULL,&c->layout));
    VkDescriptorPoolSize ps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2};
    VkDescriptorPoolCreateInfo pool={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&ps};
    TRY(vkCreateDescriptorPool(c->device,&pool,NULL,&c->pool));
    *out=c;return 0;
fail:nr_destroy(c);return result;
}

void nr_destroy(nr_context *c) {
    if(!c)return;
    if(c->device){
        vkDeviceWaitIdle(c->device);
        if(c->pool)vkDestroyDescriptorPool(c->device,c->pool,NULL);
        if(c->layout)vkDestroyPipelineLayout(c->device,c->layout,NULL);
        if(c->descriptors)vkDestroyDescriptorSetLayout(c->device,c->descriptors,NULL);
        if(c->commands)vkDestroyCommandPool(c->device,c->commands,NULL);
        vkDestroyDevice(c->device,NULL);
    }
    if(c->instance)vkDestroyInstance(c->instance,NULL);
    free(c);
}
const char *nr_device_name(nr_context *c){return c?c->properties.deviceName:NULL;}
int nr_buffer_create(nr_context *c,size_t size,nr_buffer **out) {
    if(!c||!out||!size)return VK_ERROR_INITIALIZATION_FAILED;
    *out=NULL;VkResult result;
    nr_buffer *b=calloc(1,sizeof(*b));if(!b)return VK_ERROR_OUT_OF_HOST_MEMORY;b->size=size;
    VkBufferCreateInfo ci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=size,.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    TRY(vkCreateBuffer(c->device,&ci,NULL,&b->handle));
    VkMemoryRequirements req;vkGetBufferMemoryRequirements(c->device,b->handle,&req);
    uint32_t type=UINT32_MAX;
    for(uint32_t i=0;i<c->memory.memoryTypeCount;i++)if((req.memoryTypeBits&(1u<<i)) && (c->memory.memoryTypes[i].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){type=i;break;}
    if(type==UINT32_MAX){result=VK_ERROR_FEATURE_NOT_PRESENT;goto fail;}
    VkMemoryAllocateFlagsInfo flags={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo alloc={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.pNext=&flags,.allocationSize=req.size,.memoryTypeIndex=type};
    TRY(vkAllocateMemory(c->device,&alloc,NULL,&b->memory));
    TRY(vkBindBufferMemory(c->device,b->handle,b->memory,0));
    TRY(vkMapMemory(c->device,b->memory,0,size,0,&b->mapped));
    memset(b->mapped,0,size);*out=b;return 0;
fail:nr_buffer_destroy(c,b);return result;
}
void nr_buffer_destroy(nr_context *c,nr_buffer *b){if(!c||!b)return;if(b->mapped)vkUnmapMemory(c->device,b->memory);if(b->handle)vkDestroyBuffer(c->device,b->handle,NULL);if(b->memory)vkFreeMemory(c->device,b->memory,NULL);free(b);}
uint64_t nr_buffer_address(nr_context *c,nr_buffer *b){if(!c||!b)return 0;VkBufferDeviceAddressInfo i={.sType=VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,.buffer=b->handle};return vkGetBufferDeviceAddress(c->device,&i);}
int nr_buffer_write(nr_context *c,nr_buffer *b,size_t offset,const void *data,size_t size){if(!c||!b||!data||offset>b->size||size>b->size-offset)return VK_ERROR_INITIALIZATION_FAILED;memcpy((char*)b->mapped+offset,data,size);return 0;}
int nr_buffer_read(nr_context *c,nr_buffer *b,size_t offset,void *data,size_t size){if(!c||!b||!data||offset>b->size||size>b->size-offset)return VK_ERROR_INITIALIZATION_FAILED;memcpy(data,(char*)b->mapped+offset,size);return 0;}
int nr_pipeline_create(nr_context *c,const void *spirv,size_t size,nr_pipeline **out){
    if(!c||!out||!spirv||size<20||size%4)return VK_ERROR_INITIALIZATION_FAILED;
    *out=NULL;VkResult result;VkShaderModule module=VK_NULL_HANDLE;
    nr_pipeline *p=calloc(1,sizeof(*p));if(!p)return VK_ERROR_OUT_OF_HOST_MEMORY;
    VkShaderModuleCreateInfo shader={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=size,.pCode=spirv};
    TRY(vkCreateShaderModule(c->device,&shader,NULL,&module));
    VkComputePipelineCreateInfo ci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=module,.pName="main"},.layout=c->layout};
    TRY(vkCreateComputePipelines(c->device,VK_NULL_HANDLE,1,&ci,NULL,&p->handle));
    vkDestroyShaderModule(c->device,module,NULL);*out=p;return 0;
fail:if(module)vkDestroyShaderModule(c->device,module,NULL);nr_pipeline_destroy(c,p);return result;
}
void nr_pipeline_destroy(nr_context *c,nr_pipeline *p){if(!c||!p)return;if(p->handle)vkDestroyPipeline(c->device,p->handle,NULL);free(p);}
int nr_dispatch(nr_context *c,nr_pipeline *p,nr_buffer *parameters,nr_buffer *output,uint32_t x,uint32_t y,uint32_t z){
    if(!c||!p||!parameters||!output||!x||!y||!z)return VK_ERROR_INITIALIZATION_FAILED;
    if(x>c->properties.limits.maxComputeWorkGroupCount[0]||y>c->properties.limits.maxComputeWorkGroupCount[1]||z>c->properties.limits.maxComputeWorkGroupCount[2])return VK_ERROR_INITIALIZATION_FAILED;
    VkResult result;VkCommandBuffer cmd=VK_NULL_HANDLE;
    TRY(vkResetDescriptorPool(c->device,c->pool,0));
    VkDescriptorSet set;VkDescriptorSetAllocateInfo da={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=c->pool,.descriptorSetCount=1,.pSetLayouts=&c->descriptors};
    TRY(vkAllocateDescriptorSets(c->device,&da,&set));
    VkDescriptorBufferInfo buffers[2]={{parameters->handle,0,parameters->size},{output->handle,0,output->size}};
    VkWriteDescriptorSet writes[2];memset(writes,0,sizeof(writes));
    for(uint32_t i=0;i<2;i++)writes[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=set,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&buffers[i]};
    vkUpdateDescriptorSets(c->device,2,writes,0,NULL);
    VkCommandBufferAllocateInfo ca={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=c->commands,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    TRY(vkAllocateCommandBuffers(c->device,&ca,&cmd));
    VkCommandBufferBeginInfo begin={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    TRY(vkBeginCommandBuffer(cmd,&begin));
    VkMemoryBarrier before={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&before,0,NULL,0,NULL);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,p->handle);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,c->layout,0,1,&set,0,NULL);vkCmdDispatch(cmd,x,y,z);
    VkMemoryBarrier after={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&after,0,NULL,0,NULL);
    TRY(vkEndCommandBuffer(cmd));
    VkSubmitInfo submit={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cmd};
    TRY(vkQueueSubmit(c->queue,1,&submit,VK_NULL_HANDLE));TRY(vkQueueWaitIdle(c->queue));
fail:if(cmd)vkFreeCommandBuffers(c->device,c->commands,1,&cmd);return result;
}
