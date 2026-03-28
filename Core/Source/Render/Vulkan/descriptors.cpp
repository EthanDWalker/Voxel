#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/util.h"
#include <cstdlib>
#include <vector>

namespace Core {
VulkanDescriptorPool DescriptorBuilder::Pool;
std::vector<VkDescriptorSetLayoutBinding> DescriptorBuilder::Bindings;
std::vector<void *> DescriptorBuilder::Writes;

VulkanDescriptor::~VulkanDescriptor() {
  vkDestroyDescriptorSetLayout(VulkanContext::device, layout, nullptr);
}

void VulkanDescriptor::UpdateFromWrite(const VkWriteDescriptorSet &info) {
  vkUpdateDescriptorSets(VulkanContext::device, 1, &info, 0, nullptr);
}

void DescriptorBuilder::StartUp() { Pool.StartUp(); }

void DescriptorBuilder::ShutDown() {
  Pool.Destroy();
  for (auto *write : Writes) {
    free(write);
  }
}

void DescriptorBuilder::Reset() {
  Bindings.clear();
  for (auto *write : Writes) {
    free(write);
  }
  Writes.clear();
}

void DescriptorBuilder::Build(VkShaderStageFlags stage_flags, VulkanDescriptor *descriptor) {
  if (descriptor->layout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(VulkanContext::device, descriptor->layout, nullptr);
    descriptor->layout = VK_NULL_HANDLE;
  }
  for (auto &binding : Bindings) {
    binding.stageFlags = stage_flags;
  }

  std::vector<VkDescriptorBindingFlags> binding_flags{};
  binding_flags.resize(Bindings.size());

  for (auto &flag : binding_flags) {
    flag = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
  }

  VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_info{};
  binding_flags_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
  binding_flags_info.bindingCount = binding_flags.size();
  binding_flags_info.pBindingFlags = binding_flags.data();

  VkDescriptorSetLayoutCreateInfo ds_layout_ci{};
  ds_layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  ds_layout_ci.bindingCount = Bindings.size();
  ds_layout_ci.pBindings = Bindings.data();
  ds_layout_ci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
  ds_layout_ci.pNext = &binding_flags_info;

  VK_CHECK(vkCreateDescriptorSetLayout(VulkanContext::device, &ds_layout_ci, nullptr, &descriptor->layout));

  if (descriptor->set != VK_NULL_HANDLE) {
    vkFreeDescriptorSets(VulkanContext::device, Pool.current_pool, 1, &descriptor->set);
    descriptor->set = VK_NULL_HANDLE;
  }
  Pool.Allocate(descriptor);

  std::vector<VkWriteDescriptorSet> binding_writes;

  for (auto &binding : Bindings) {
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstBinding = binding.binding;
    write.dstSet = descriptor->set;
    write.descriptorCount = binding.descriptorCount;
    write.descriptorType = binding.descriptorType;
    switch (binding.descriptorType) {
    case VK_DESCRIPTOR_TYPE_SAMPLER:
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: {
      write.pImageInfo = (VkDescriptorImageInfo *)Writes[binding.binding];
      break;
    }
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: {
      write.pBufferInfo = (VkDescriptorBufferInfo *)Writes[binding.binding];
      break;
    }
    case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: {
      write.pNext = Writes[binding.binding];
      break;
    }
    default: {
      abort();
      break;
    }
    }
    binding_writes.push_back(write);
  }
  vkUpdateDescriptorSets(VulkanContext::device, binding_writes.size(), binding_writes.data(), 0, nullptr);
  Reset();
}

void VulkanDescriptorPool::StartUp() { NewPool(); }

void VulkanDescriptorPool::NewPool() {
  std::array<VkDescriptorPoolSize, POOL_RATIOS.size()> pool_sizes{};

  for (u32 i = 0; i < POOL_RATIOS.size(); i++) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = POOL_RATIOS[i].first;
    pool_size.descriptorCount = POOL_RATIOS[i].second * alloc_scaler;
    pool_sizes[i] = pool_size;
  }

  VkDescriptorPoolCreateInfo descriptor_pool_ci{};
  descriptor_pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptor_pool_ci.pPoolSizes = pool_sizes.data();
  descriptor_pool_ci.poolSizeCount = pool_sizes.size();
  descriptor_pool_ci.maxSets = 256;
  descriptor_pool_ci.flags =
      VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

  VK_CHECK(vkCreateDescriptorPool(VulkanContext::device, &descriptor_pool_ci, nullptr, &current_pool));

  used_pools.push_back(current_pool);
}

void VulkanDescriptorPool::Allocate(VulkanDescriptor *descriptor, void *pNext) {
  VkDescriptorSetAllocateInfo ds_alloc_info{};
  ds_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ds_alloc_info.descriptorPool = current_pool;
  ds_alloc_info.pSetLayouts = &descriptor->layout;
  ds_alloc_info.descriptorSetCount = 1;
  ds_alloc_info.pNext = pNext;

  VkResult result = vkAllocateDescriptorSets(VulkanContext::device, &ds_alloc_info, &descriptor->set);

  switch (result) {
  case VK_ERROR_OUT_OF_POOL_MEMORY:
    break;
  case VK_ERROR_OUT_OF_DEVICE_MEMORY:
    break;
  case VK_ERROR_FRAGMENTED_POOL:
    break;
  default: {
    return;
  }
  }

  NewPool();
  ds_alloc_info.descriptorPool = current_pool;

  VK_CHECK(vkAllocateDescriptorSets(VulkanContext::device, &ds_alloc_info, &descriptor->set));
}

void VulkanDescriptorPool::Destroy() {
  for (VkDescriptorPool pool : used_pools) {
    vkDestroyDescriptorPool(VulkanContext::device, pool, nullptr);
  }
}
} // namespace Core
