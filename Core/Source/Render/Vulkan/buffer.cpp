#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/util.h"

#define VMA_IMPLEMENTATION
#include "vma/vk_mem_alloc.h"

namespace Core {

VulkanBuffer::~VulkanBuffer() { vmaDestroyBuffer(VulkanContext::allocator, obj, allocation); }
void VulkanBuffer::Destroy() { vmaDestroyBuffer(VulkanContext::allocator, obj, allocation); }

void VulkanBuffer::Create(u64 size, VkBufferUsageFlags usage, bool host) {
  this->usage = usage;
  this->size = size;

  VkBufferCreateInfo buffer_ci{};
  buffer_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_ci.size = size;
  buffer_ci.usage = usage;

  VmaAllocationCreateInfo alloc_ci{};
  alloc_ci.usage = host ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  alloc_ci.flags = host ? VMA_ALLOCATION_CREATE_MAPPED_BIT |
                              VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                        : 0;

  VmaAllocationInfo info{};

  VK_CHECK(
      vmaCreateBuffer(VulkanContext::allocator, &buffer_ci, &alloc_ci, &obj, &allocation, &info));

  if (host) {
    this->address = allocation->GetMappedData();
  } else if ((usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0) {
    VkBufferDeviceAddressInfo device_address_info{};
    device_address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    device_address_info.buffer = obj;
    this->address = (void *)vkGetBufferDeviceAddress(VulkanContext::device, &device_address_info);
  }
}
} // namespace Core
