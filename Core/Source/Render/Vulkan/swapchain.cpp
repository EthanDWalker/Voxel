#include "Core/Render/Vulkan/swapchain.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/command_pool.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/info.h"
#include "Core/Render/Vulkan/util.h"
#include "Core/Render/frame.h"
#include "VkBootstrap.h"

namespace Core {
VulkanSwapchain::~VulkanSwapchain() {
  ZoneScoped;
  vkDeviceWaitIdle(VulkanContext::device);
  vkDestroySwapchainKHR(VulkanContext::device, obj, nullptr);
  for (size_t i = 0; i < FRAME_OVERLAP; i++) {
    vkDestroyImageView(VulkanContext::device, images[i].view, nullptr);
    vkDestroySemaphore(VulkanContext::device, swapchain_semaphores[i], nullptr);
    vkDestroySemaphore(VulkanContext::device, render_semaphores[i], nullptr);
    vkDestroyFence(VulkanContext::device, render_fences[i], nullptr);
  }
}

void VulkanSwapchain::Create(Vec2u32 extent) {
  ZoneScoped;
  vkb::SwapchainBuilder swapchain_builder{
      VulkanContext::physical_device,
      VulkanContext::device,
      VulkanContext::surface,
  };

  this->format = VK_FORMAT_B8G8R8A8_UNORM;

  vkb::Swapchain vkb_swapchain =
      swapchain_builder
          .set_desired_format(
              VkSurfaceFormatKHR{.format = this->format, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
          .set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
          .set_required_min_image_count(FRAME_OVERLAP)
          .set_desired_extent(extent.width, extent.height)
          .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
          .build()
          .value();

  this->extent = Vec2u32::From(vkb_swapchain.extent);
  this->obj = vkb_swapchain.swapchain;
  std::vector<VkImage> images = vkb_swapchain.get_images().value();
  std::vector<VkImageView> image_views = vkb_swapchain.get_image_views().value();

  VkFenceCreateInfo fence_ci{};
  fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  VkSemaphoreCreateInfo semaphore_ci{};
  semaphore_ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  for (u64 i = 0; i < FRAME_OVERLAP; i++) {
    this->images[i].obj = images[i];
    this->images[i].view = image_views[i];
    command_pools[i].Create(VulkanContext::graphics_queue_index);
    command_buffers[i].CreatePrimary(command_pools[i]);

    VK_CHECK(vkCreateFence(VulkanContext::device, &fence_ci, nullptr, &render_fences[i]));

    VK_CHECK(vkCreateSemaphore(VulkanContext::device, &semaphore_ci, nullptr, &render_semaphores[i]));

    VK_CHECK(vkCreateSemaphore(VulkanContext::device, &semaphore_ci, nullptr, &swapchain_semaphores[i]));
  }
}

void VulkanSwapchain::Resize(Vec2u32 extent) {
  ZoneScoped;

  vkb::SwapchainBuilder swapchain_builder{
      VulkanContext::physical_device,
      VulkanContext::device,
      VulkanContext::surface,
  };

  this->format = VK_FORMAT_B8G8R8A8_UNORM;

  vkDeviceWaitIdle(VulkanContext::device);

  vkb::Swapchain vkb_swapchain =
      swapchain_builder
          .set_desired_format(
              VkSurfaceFormatKHR{.format = this->format, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
          .set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
          .set_required_min_image_count(FRAME_OVERLAP)
          .set_old_swapchain(obj)
          .set_desired_extent(extent.width, extent.height)
          .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
          .build()
          .value();

  vkDestroySwapchainKHR(VulkanContext::device, obj, nullptr);

  this->extent = Vec2u32::From(vkb_swapchain.extent);
  this->obj = vkb_swapchain.swapchain;
  std::vector<VkImage> images = vkb_swapchain.get_images().value();
  std::vector<VkImageView> image_views = vkb_swapchain.get_image_views().value();

  for (u64 i = 0; i < FRAME_OVERLAP; i++) {
    vkDestroyImageView(VulkanContext::device, this->images[i].view, nullptr);

    this->images[i].obj = images[i];
    this->images[i].view = image_views[i];
  }
}

void VulkanSwapchain::AcquireNextImage(bool &resize) {
  ZoneScoped;
  u32 frame_index = frame_number % FRAME_OVERLAP;

  VK_CHECK(vkWaitForFences(VulkanContext::device, 1, &render_fences[frame_index], VK_TRUE,
                           std::numeric_limits<u32>::max()));
  VK_CHECK(vkResetFences(VulkanContext::device, 1, &render_fences[frame_index]));

  {
    VkResult e = vkAcquireNextImageKHR(VulkanContext::device, obj, std::numeric_limits<u32>::max(),
                                       swapchain_semaphores[frame_index], nullptr, &image_index);
    if (e == VK_ERROR_OUT_OF_DATE_KHR) {
      resize = true;
      return;
    }
    VK_CHECK(e);
  }
  resize = false;
}

VulkanCommandBuffer &VulkanSwapchain::GetActiveCommandBuffer() {
  ZoneScoped;
  u32 frame_index = frame_number % FRAME_OVERLAP;
  return command_buffers[frame_index];
}

void VulkanSwapchain::BeginCommandBuffer() {
  ZoneScoped;
  u32 frame_index = frame_number % FRAME_OVERLAP;

  VK_CHECK(vkResetCommandPool(VulkanContext::device, command_pools[frame_index].obj, 0));

  command_buffers[frame_index].Begin();
}

void VulkanSwapchain::SubmitCommandBuffer() {
  ZoneScoped;
  u32 frame_index = frame_number % FRAME_OVERLAP;

  VkImageMemoryBarrier2 image_barrier{};
  image_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
  image_barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  image_barrier.image = images[image_index].obj;
  image_barrier.subresourceRange = ImageSubresourceRange(VK_IMAGE_ASPECT_COLOR_BIT);

  VkDependencyInfo dep_info{};
  dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  dep_info.imageMemoryBarrierCount = 1;
  dep_info.pImageMemoryBarriers = &image_barrier;

  vkCmdPipelineBarrier2(command_buffers[frame_index].obj, &dep_info);

  command_buffers[frame_index].End();

  VkCommandBufferSubmitInfo cmd_submit_info = CommandBufferSubmitInfo(command_buffers[frame_index].obj);

  VkSemaphoreSubmitInfo wait_semaphore_info = SemaphoreSubmitInfo(
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, swapchain_semaphores[frame_index]);

  VkSemaphoreSubmitInfo signal_semaphore_info =
      SemaphoreSubmitInfo(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, render_semaphores[frame_index]);

  VkSubmitInfo2 submit_info = SubmitInfo(&cmd_submit_info, &signal_semaphore_info, &wait_semaphore_info);

  VK_CHECK(vkQueueSubmit2(VulkanContext::graphics_queue, 1, &submit_info, render_fences[frame_index]));
}

void VulkanSwapchain::Present(bool &resize) {
  ZoneScoped;
  u32 frame_index = frame_number % FRAME_OVERLAP;

  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.pSwapchains = &obj;
  present_info.swapchainCount = 1;
  present_info.pWaitSemaphores = &render_semaphores[frame_index];
  present_info.waitSemaphoreCount = 1;
  present_info.pImageIndices = &image_index;

  VkResult e = vkQueuePresentKHR(VulkanContext::graphics_queue, &present_info);
  if (e == VK_ERROR_OUT_OF_DATE_KHR) {
    resize = true;
    return;
  }
  VK_CHECK(e);

  resize = false;
  frame_number++;
}

BaseVulkanImage &VulkanSwapchain::GetImage() {
  ZoneScoped;
  return images[image_index];
}
} // namespace Core
