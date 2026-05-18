#pragma once
#include "Core/Render/Vulkan/command_buffer.h"
#include "volk.h"

namespace Core {

struct FrameData {
  u32 frame_index;
};
struct VulkanSwapchain {
  VulkanSwapchain() = default;

  VulkanSwapchain(const VulkanSwapchain &) = delete;
  VulkanSwapchain &operator=(const VulkanSwapchain &) = delete;

  VulkanSwapchain(VulkanSwapchain &&) = delete;
  VulkanSwapchain &operator=(VulkanSwapchain &&) = delete;

  static const u8 FRAME_OVERLAP = 3;

  BaseVulkanImage images[FRAME_OVERLAP];
  VulkanCommandPool command_pools[FRAME_OVERLAP];
  VulkanCommandBuffer command_buffers[FRAME_OVERLAP];
  VkSemaphore swapchain_semaphores[FRAME_OVERLAP];
  VkSemaphore render_semaphores[FRAME_OVERLAP];
  VkFence render_fences[FRAME_OVERLAP];
  VkSemaphore frame_number_semaphore;

  VkSwapchainKHR obj;
  VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
  Vec2u32 extent;
  u32 frame_index = 0;
  u32 image_index;
  u32 frame_number = 0;

  void Create(const Vec2u32 extent);
  void Resize(const Vec2u32 extent);

  void AcquireNextImage(bool &resize);
  void BeginCommandBuffer();
  VulkanCommandBuffer &GetActiveCommandBuffer();
  void SubmitCommandBuffer();
  void Present(bool &resize);

  BaseVulkanImage &GetImage();

  ~VulkanSwapchain();
};
} // namespace Core
