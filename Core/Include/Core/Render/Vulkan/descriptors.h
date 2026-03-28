#pragma once
#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/image.h"
#include "Core/Render/Vulkan/sampler.h"
#include "Core/Render/types.h"
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace Core {

struct VulkanDescriptor {
  VulkanDescriptor() = default;

  VulkanDescriptor(const VulkanDescriptor &) = delete;
  VulkanDescriptor &operator=(const VulkanDescriptor &) = delete;

  VulkanDescriptor(VulkanDescriptor &&) = default;
  VulkanDescriptor &operator=(VulkanDescriptor &&) = default;

  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;

  operator VkDescriptorSet() { return this->set; }

  ~VulkanDescriptor();

  void UpdateFromWrite(const VkWriteDescriptorSet &info);

  template <DeviceResourceType type>
  void Update(const u32 binding, const BaseVulkanImage *image, const VulkanSampler *sampler, const u32 array_index = 0) {
    static_assert(type == DeviceResourceType::CombinedImageSampler, "must pass combined image sampler");

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstArrayElement = array_index;
    write.descriptorCount = 1;
    write.dstBinding = binding;
    write.dstSet = set;

    VkDescriptorImageInfo image_info{};
    image_info.imageView = image ? image->view : VK_NULL_HANDLE;
    image_info.sampler = sampler ? sampler->obj : VK_NULL_HANDLE;
    image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &image_info;

    UpdateFromWrite(write);
  }

  template <DeviceResourceType type, typename T = std::nullptr_t>
  void Update(const u32 binding, const T resource = nullptr, const u32 array_index = 0) {
    static_assert(std::is_pointer_v<T> || std::is_null_pointer_v<T>, "must pass in pointer");

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstArrayElement = array_index;
    write.descriptorCount = 1;
    write.dstBinding = binding;
    write.dstSet = set;

    if constexpr (type == DeviceResourceType::Buffer) {
      VkDescriptorBufferInfo buffer_info{};
      if (std::is_null_pointer_v<T>) {
        buffer_info.buffer = nullptr;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      } else {
        buffer_info.buffer = resource->obj;
        write.descriptorType = ((resource->usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) != 0)
                                   ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                   : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      }

      buffer_info.offset = 0;
      buffer_info.range = VK_WHOLE_SIZE;
      write.pBufferInfo = &buffer_info;

      UpdateFromWrite(write);
    } else if constexpr (type == DeviceResourceType::SampledImage ||
                         type == DeviceResourceType::StorageImage ||
                         type == DeviceResourceType::RWStorageImage) {
      VkDescriptorImageInfo image_info{};
      if (std::is_null_pointer_v<T>) {
        image_info.imageView = VK_NULL_HANDLE;
      } else {
        image_info.imageView = resource->view;
      }
      switch (type) {
      case DeviceResourceType::SampledImage: {
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        break;
      }
      case DeviceResourceType::StorageImage: {
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        break;
      }
      case DeviceResourceType::RWStorageImage: {
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        break;
      }
      }

      write.pImageInfo = &image_info;
      UpdateFromWrite(write);
    } else if constexpr (type == DeviceResourceType::Sampler) {
      VkDescriptorImageInfo image_info{};

      if constexpr (std::is_null_pointer_v<T>) {
        image_info.sampler = VK_NULL_HANDLE;
      } else if constexpr (true) {
        image_info.sampler = resource->obj;
      }

      write.pImageInfo = &image_info;
      UpdateFromWrite(write);
    }
  }
};

struct VulkanDescriptorPool {
  static constexpr std::array<std::pair<VkDescriptorType, f32>, 6> POOL_RATIOS{{
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3.0f},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3.0f},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1.0f},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1.0f},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1.0f},
      {VK_DESCRIPTOR_TYPE_SAMPLER, 0.1f},
  }};

  u32 alloc_scaler = 100;

  VkDescriptorPool current_pool = VK_NULL_HANDLE;
  std::vector<VkDescriptorPool> used_pools;

  void Destroy();
  void StartUp();
  void Allocate(VulkanDescriptor *descriptor, void *pNext = nullptr);
  void NewPool();
};

struct DescriptorBuilder {
  static VulkanDescriptorPool Pool;

  static std::vector<VkDescriptorSetLayoutBinding> Bindings;
  static std::vector<void *> Writes;

  static void StartUp();
  static void ShutDown();

  template <DeviceResourceType type, typename T = std::nullptr_t>
  static void Bind(const T resource = nullptr, const u32 count = 1) {
    static_assert(std::is_pointer_v<T> || std::is_null_pointer_v<T>, "must pass in pointer to resource");

    if constexpr (type == DeviceResourceType::Buffer) {
      static_assert(std::is_same_v<T, VulkanBuffer *> || std::is_null_pointer_v<T>,
                    "must provide vulkan buffer for resource type of buffer");
      VkDescriptorSetLayoutBinding new_binding{};
      new_binding.binding = Bindings.size();
      if constexpr (std::is_same_v<T, VulkanBuffer *>) {
        new_binding.descriptorType =
            ((((VulkanBuffer *)resource)->usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) != 0)
                ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      } else {
        new_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      }
      new_binding.descriptorCount = count;

      Bindings.push_back(new_binding);

      VkDescriptorBufferInfo *buffer_write_array =
          (VkDescriptorBufferInfo *)malloc(sizeof(VkDescriptorBufferInfo) * count);

      for (u32 i = 0; i < count; i++) {
        VkDescriptorBufferInfo buffer_write{};
        if constexpr (std::is_null_pointer_v<T>) {
          buffer_write.buffer = VK_NULL_HANDLE;
        } else if constexpr (true) {
          buffer_write.buffer = resource->obj;
        }
        buffer_write.offset = 0;
        buffer_write.range = VK_WHOLE_SIZE;
        buffer_write_array[i] = buffer_write;
      }

      Writes.push_back(buffer_write_array);
    } else if constexpr (type == DeviceResourceType::SampledImage ||
                         type == DeviceResourceType::StorageImage ||
                         type == DeviceResourceType::RWStorageImage) {
      static_assert(std::is_base_of_v<BaseVulkanImage, std::remove_pointer_t<T>> ||
                        std::is_same_v<BaseVulkanImage, std::remove_pointer_t<T>> ||
                        std::is_null_pointer_v<T>,
                    "must provide base vulkan image for resource type of image");

      VkDescriptorSetLayoutBinding new_binding{};
      new_binding.binding = Bindings.size();
      new_binding.descriptorCount = count;

      switch (type) {
      case DeviceResourceType::SampledImage: {
        new_binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        break;
      }
      case DeviceResourceType::StorageImage:
      case DeviceResourceType::RWStorageImage: {
        new_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        break;
      }
      }

      Bindings.push_back(new_binding);

      VkDescriptorImageInfo *image_write_array =
          (VkDescriptorImageInfo *)malloc(sizeof(VkDescriptorImageInfo) * count);
      for (u32 i = 0; i < count; i++) {
        VkDescriptorImageInfo image_write{};

        if constexpr (std::is_null_pointer_v<T>) {
          image_write.imageView = VK_NULL_HANDLE;
        } else if constexpr (true) {
          image_write.imageView = resource->view;
        }

        switch (type) {
        case DeviceResourceType::SampledImage:
        case DeviceResourceType::StorageImage: {
          image_write.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
          break;
        }
        case DeviceResourceType::RWStorageImage: {
          image_write.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
          break;
        }
        }
        image_write.sampler = VK_NULL_HANDLE;
        image_write_array[i] = image_write;
      }
      Writes.push_back(image_write_array);
    } else if constexpr (type == DeviceResourceType::Sampler) {
      static_assert(std::is_same_v<T, VulkanSampler *> || std::is_null_pointer_v<T>,
                    "must pass vulkan sampler to that resource type");
      VkDescriptorSetLayoutBinding new_binding{};
      new_binding.binding = Bindings.size();
      new_binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
      new_binding.descriptorCount = 1;

      Bindings.push_back(new_binding);

      VkDescriptorImageInfo *image_write_array =
          (VkDescriptorImageInfo *)malloc(sizeof(VkDescriptorImageInfo) * count);
      for (u32 i = 0; i < count; i++) {
        VkDescriptorImageInfo image_write{};
        if constexpr (std::is_null_pointer_v<T>) {
          image_write.sampler = VK_NULL_HANDLE;
        } else if constexpr (true) {
          image_write.sampler = resource->obj;
        }
        image_write.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image_write_array[i] = image_write;
      }
      Writes.push_back(image_write_array);
    } else if constexpr (false) {
      static_assert(false, "binding not supported");
    }
  }

  template <DeviceResourceType type>
  void Bind(const BaseVulkanImage *image, const VulkanSampler *sampler, const u32 count = 1) {
    static_assert(type == DeviceResourceType::CombinedImageSampler, "must pass combinded image sampler");

    VkDescriptorSetLayoutBinding new_binding{};
    new_binding.binding = Bindings.size();
    new_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    new_binding.descriptorCount = 1;

    Bindings.push_back(new_binding);

    VkDescriptorImageInfo *image_write_array =
        (VkDescriptorImageInfo *)malloc(sizeof(VkDescriptorImageInfo) * count);
    for (u32 i = 0; i < count; i++) {
      VkDescriptorImageInfo image_write{};
      image_write.sampler = sampler ? sampler->obj : VK_NULL_HANDLE;
      image_write.imageView = image ? image->view : VK_NULL_HANDLE;
      image_write.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      image_write_array[i] = image_write;
    }
    Writes.push_back(image_write_array);
  }

  static void Build(VkShaderStageFlags stage_flags, VulkanDescriptor *descriptor);
  static void Reset();
};
} // namespace Core
