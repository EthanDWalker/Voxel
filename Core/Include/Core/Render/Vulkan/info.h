#pragma once
#include "volk.h"
#include <span>

namespace Core {
VkCommandBufferBeginInfo CommandBufferBeginInfo(VkCommandBufferUsageFlags flags);
VkImageSubresourceRange ImageSubresourceRange(VkImageAspectFlags aspect_mask);
VkSemaphoreSubmitInfo SemaphoreSubmitInfo(VkPipelineStageFlags2 stage_mask, VkSemaphore semaphore);
VkCommandBufferSubmitInfo CommandBufferSubmitInfo(VkCommandBuffer cmd);
VkSubmitInfo2 SubmitInfo(VkCommandBufferSubmitInfo *cmd,
                         VkSemaphoreSubmitInfo *signal_semaphore_info,
                         VkSemaphoreSubmitInfo *wait_semaphore_info);
VkImageCreateInfo ImageCI(VkFormat format, VkImageUsageFlags usage_flags, Vec3u32 extent,
                          u32 mip_levels = 1);
VkImageViewCreateInfo ImageViewCI(VkFormat format, VkImage image, u32 mip_levels);
VkRenderingAttachmentInfo AttachmentInfo(VkImageView view, VkImageView resolve_view,
                                         VkClearValue *clear, VkImageLayout layout);
VkRenderingAttachmentInfo DepthAttachmentInfo(VkImageView image_view, VkImageLayout layout,
                                              VkAttachmentLoadOp load_op,
                                              VkAttachmentStoreOp store_op);
VkRenderingInfo RenderingInfo(Vec3u32 render_extent,
                              std::span<VkRenderingAttachmentInfo> color_attachments,
                              VkRenderingAttachmentInfo *depth_attachment);
VkViewport Viewport(Vec3u32 extent);
VkRect2D Scissor(Vec3u32 extent);
} // namespace Core
