#include "Core/Render/Vulkan/command_pool.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/util.h"

namespace Core {
VulkanCommandPool::~VulkanCommandPool() {
  vkDestroyCommandPool(VulkanContext::device, obj, nullptr);
}

void VulkanCommandPool::Create(u32 queue_index) {
  VkCommandPoolCreateInfo command_pool_ci{};
  command_pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  command_pool_ci.queueFamilyIndex = queue_index;
  vkCreateCommandPool(VulkanContext::device, &command_pool_ci, nullptr, &obj);
}

void VulkanCommandPool::Reset() { VK_CHECK(vkResetCommandPool(VulkanContext::device, obj, 0)); }

VkCommandBuffer VulkanCommandPool::AllocateCommandBuffer(VkCommandBufferLevel buffer_level) {
  VkCommandBuffer cmd;
  VkCommandBufferAllocateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  info.commandBufferCount = 1;
  info.level = buffer_level;
  info.commandPool = obj;
  VK_CHECK(vkAllocateCommandBuffers(VulkanContext::device, &info, &cmd));
  return cmd;
}
} // namespace Core
