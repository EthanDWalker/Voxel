#pragma once
#include "Core/Render/Vulkan/command_buffer.h"
#include "volk.h"

namespace Core {
struct VulkanSwapchain {
  VulkanSwapchain() = default;

  VulkanSwapchain(const VulkanSwapchain &) = delete;
  VulkanSwapchain &operator=(const VulkanSwapchain &) = delete;

  VulkanSwapchain(VulkanSwapchain &&) = default;
  VulkanSwapchain &operator=(VulkanSwapchain &&) = default;

  static const u8 FRAME_OVERLAP = 3;

  BaseVulkanImage images[FRAME_OVERLAP];
  VulkanCommandPool command_pools[FRAME_OVERLAP];
  VulkanCommandBuffer command_buffers[FRAME_OVERLAP];
  VkSemaphore swapchain_semaphores[FRAME_OVERLAP];
  VkSemaphore render_semaphores[FRAME_OVERLAP];
  VkFence render_fences[FRAME_OVERLAP];
  VkSwapchainKHR obj;
  VkFormat format;
  Vec2u32 extent;
  u32 frame_number;
  u32 image_index;

  void Create(Vec2u32 extent);
  void Resize(Vec2u32 extent);

  void AcquireNextImage(bool &resize);
  VulkanCommandBuffer &BeginCommandBuffer();
  VulkanCommandBuffer &GetActiveCommandBuffer();
  void SubmitCommandBuffer();
  void Present(bool &resize);

  BaseVulkanImage &GetImage();

  ~VulkanSwapchain();
};
} // namespace Core
