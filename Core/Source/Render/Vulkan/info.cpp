#include "Core/Render/Vulkan/info.h"
#include "Core/Render/Vulkan/image_util.h"
#include <span>

namespace Core {
VkCommandBufferBeginInfo CommandBufferBeginInfo(VkCommandBufferUsageFlags flags) {
  ZoneScoped;
  VkCommandBufferBeginInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  info.flags = flags;
  return info;
}

VkImageSubresourceRange ImageSubresourceRange(VkImageAspectFlags aspect_mask) {
  ZoneScoped;
  VkImageSubresourceRange info{};
  info.aspectMask = aspect_mask;
  info.baseMipLevel = 0;
  info.layerCount = VK_REMAINING_MIP_LEVELS;
  info.baseArrayLayer = 0;
  info.levelCount = VK_REMAINING_ARRAY_LAYERS;
  return info;
}

VkSemaphoreSubmitInfo SemaphoreSubmitInfo(VkPipelineStageFlags2 stage_mask, VkSemaphore semaphore) {
  ZoneScoped;
  VkSemaphoreSubmitInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
  info.semaphore = semaphore;
  info.stageMask = stage_mask;
  info.value = 1;
  return info;
}

VkCommandBufferSubmitInfo CommandBufferSubmitInfo(VkCommandBuffer cmd) {
  ZoneScoped;
  VkCommandBufferSubmitInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
  info.commandBuffer = cmd;
  return info;
}

VkSubmitInfo2 SubmitInfo(VkCommandBufferSubmitInfo *cmd,
                         VkSemaphoreSubmitInfo *signal_semaphore_info,
                         VkSemaphoreSubmitInfo *wait_semaphore_info) {
  ZoneScoped;
  VkSubmitInfo2 info{};
  info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
  info.pCommandBufferInfos = cmd;
  info.commandBufferInfoCount = 1;
  info.pSignalSemaphoreInfos = signal_semaphore_info;
  info.signalSemaphoreInfoCount = signal_semaphore_info ? 1 : 0;
  info.pWaitSemaphoreInfos = wait_semaphore_info;
  info.waitSemaphoreInfoCount = wait_semaphore_info ? 1 : 0;
  return info;
}

VkImageCreateInfo ImageCI(VkFormat format, VkImageUsageFlags usage_flags, Vec3u32 extent,
                          u32 mip_levels) {
  ZoneScoped;
  VkImageCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.extent = Vec3u32::To<VkExtent3D>(extent);
  info.format = format;
  info.imageType = extent.depth == 1 ? VK_IMAGE_TYPE_2D : VK_IMAGE_TYPE_3D;
  info.mipLevels = mip_levels;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = usage_flags;
  return info;
}

VkImageViewCreateInfo ImageViewCI(VkFormat format, VkImage image, u32 mip_levels) {
  ZoneScoped;
  VkImageViewCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  info.image = image;
  info.format = format;
  info.subresourceRange.baseMipLevel = 0;
  info.subresourceRange.levelCount = mip_levels;
  info.subresourceRange.baseArrayLayer = 0;
  info.subresourceRange.layerCount = 1;
  info.subresourceRange.aspectMask =
      IsDepthFormat(format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
  return info;
}

VkRenderingAttachmentInfo AttachmentInfo(VkImageView view, VkImageView resolve_view,
                                         VkClearValue *clear, VkImageLayout layout) {
  ZoneScoped;
  VkRenderingAttachmentInfo info{};
  info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
  info.imageLayout = layout;
  info.imageView = view;
  if (clear) {
    info.clearValue = *clear;
    info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  } else {
    info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  }
  info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  info.resolveImageView = resolve_view;
  info.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  info.resolveMode = resolve_view ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE;
  return info;
}

VkRenderingAttachmentInfo DepthAttachmentInfo(VkImageView image_view, VkImageLayout layout,
                                              VkAttachmentLoadOp load_op,
                                              VkAttachmentStoreOp store_op) {
  ZoneScoped;
  VkRenderingAttachmentInfo info{};
  info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
  info.imageView = image_view;
  info.imageLayout = layout;
  info.loadOp = load_op;
  info.storeOp = store_op;
  info.clearValue.depthStencil.depth = 0.0f;
  return info;
}

VkRenderingInfo RenderingInfo(Vec3u32 render_extent,
                              std::span<VkRenderingAttachmentInfo> color_attachments,
                              VkRenderingAttachmentInfo *depth_attachment) {
  ZoneScoped;
  VkRenderingInfo info{};
  info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
  info.renderArea = VkRect2D{VkOffset2D{0, 0}, {render_extent.width, render_extent.height}};
  info.layerCount = 1;
  info.pColorAttachments = color_attachments.data();
  info.colorAttachmentCount = color_attachments.size();
  info.pDepthAttachment = depth_attachment;
  return info;
}

VkViewport Viewport(Vec3u32 extent) {
  ZoneScoped;
  VkViewport viewport{};
  viewport.x = 0;
  viewport.y = 0;
  viewport.width = extent.width;
  viewport.height = extent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  return viewport;
}

VkRect2D Scissor(Vec3u32 extent) {
  ZoneScoped;
  VkRect2D scissor{};
  scissor.offset.x = 0;
  scissor.offset.y = 0;
  scissor.extent.width = extent.width;
  scissor.extent.height = extent.height;
  return scissor;
}
} // namespace Core
