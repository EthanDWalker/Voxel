#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/image_util.h"
#include "Core/Render/Vulkan/indirect_draw.h"
#include "Core/Render/Vulkan/info.h"
#include "Core/Render/Vulkan/util.h"
#include "Core/Util/fail.h"
#include "Core/Util/log.h"

namespace Core {
void VulkanCommandBuffer::Begin() {
  VkCommandBufferBeginInfo cmd_begin = CommandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

  VK_CHECK(vkBeginCommandBuffer(obj, &cmd_begin));
}

void VulkanCommandBuffer::End() { VK_CHECK(vkEndCommandBuffer(obj)); }

void VulkanCommandBuffer::PushConstants(VkShaderStageFlagBits stages, u32 size, const void *data) {
  Assert(bound_pipeline_layout, "Must bind pipeline before pushing constants");
  vkCmdPushConstants(obj, *bound_pipeline_layout, stages, push_constant_offset, size, data);
  push_constant_offset += size;
}

void VulkanCommandBuffer::Dispatch(Vec3u32 groups) { vkCmdDispatch(obj, groups.x, groups.y, groups.z); }

void VulkanCommandBuffer::CopyImageToImage(const BaseVulkanImage &src_image,
                                           const BaseVulkanImage &dst_image) {
  VkImageCopy2 copy{};
  copy.sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2;
  copy.extent = Vec3u32::To<VkExtent3D>(src_image.GetVec3u32());
  copy.dstSubresource.mipLevel = 0;
  copy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy.dstSubresource.baseArrayLayer = 0;
  copy.dstSubresource.layerCount = 1;
  copy.srcSubresource.mipLevel = 0;
  copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy.srcSubresource.baseArrayLayer = 0;
  copy.srcSubresource.layerCount = 1;

  VkCopyImageInfo2 info{};
  info.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2;
  info.regionCount = 1;
  info.pRegions = &copy;

  info.srcImage = src_image.obj;
  info.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

  info.dstImage = dst_image.obj;
  info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  info.regionCount = 1;
  vkCmdCopyImage2(obj, &info);
}

void VulkanCommandBuffer::FillBuffer(const VulkanBuffer &buffer, const u64 fill_size, const u32 data,
                                     const u32 offset) {
  vkCmdFillBuffer(obj, buffer.obj, offset, fill_size, data);
}

void VulkanCommandBuffer::UploadBufferToBuffer(const VulkanBuffer &src_buffer, const VulkanBuffer &dst_buffer,
                                               const u64 size, const u64 src_offset, const u64 dst_offset) {
  Assert((dst_buffer.usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) != 0 &&
             (VK_BUFFER_USAGE_TRANSFER_SRC_BIT & src_buffer.usage) != 0,
         "When trying to upload buffer to buffer proper bits arent set");
  VkBufferCopy buffer_copy{};
  buffer_copy.size = size;
  buffer_copy.dstOffset = dst_offset;
  buffer_copy.srcOffset = src_offset;

  vkCmdCopyBuffer(obj, src_buffer.obj, dst_buffer.obj, 1, &buffer_copy);
}

void VulkanCommandBuffer::DrawIndirect(const VulkanIndirectDrawCommand &command) {
  vkCmdDrawIndexedIndirectCount(obj, command.draw_buffer.obj, 0, command.draw_count_buffer.obj, 0,
                                command.max_draw_count, sizeof(VkDrawIndexedIndirectCommand));
}

void VulkanCommandBuffer::UploadBufferToImage(const VulkanBuffer &buffer, const BaseVulkanImage &image,
                                              const u32 src_offset, const u32 mip_level) {
  VkBufferImageCopy copy_region{};
  copy_region.bufferOffset = src_offset;
  copy_region.bufferRowLength = 0;
  copy_region.bufferImageHeight = 0;
  copy_region.imageSubresource.aspectMask =
      IsDepthFormat(image.format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
  copy_region.imageSubresource.mipLevel = mip_level;
  copy_region.imageSubresource.baseArrayLayer = 0;
  copy_region.imageSubresource.layerCount = 1;
  copy_region.imageExtent = Vec3u32::To<VkExtent3D>(Max(image.GetVec3u32() / (1 << mip_level), Vec3u32(1)));

  vkCmdCopyBufferToImage(obj, buffer.obj, image.obj, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);
}

void VulkanCommandBuffer::Blit(const BaseVulkanImage &src_image, const BaseVulkanImage &dst_image,
                               const u32 src_mip_level, const u32 dst_mip_level) {
  VkImageMemoryBarrier2 barriers[2] = {};

  VkImageMemoryBarrier2 &src_barrier = barriers[0];
  src_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
  src_barrier.image = src_image.obj;
  src_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  src_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  src_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  src_barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  src_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
  src_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  src_barrier.subresourceRange.baseArrayLayer = 0;
  src_barrier.subresourceRange.baseMipLevel = src_mip_level;
  src_barrier.subresourceRange.layerCount = 1;
  src_barrier.subresourceRange.levelCount = 1;

  VkImageMemoryBarrier2 &dst_barrier = barriers[1];
  dst_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
  dst_barrier.image = dst_image.obj;
  dst_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  dst_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  dst_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  dst_barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  dst_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  dst_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  dst_barrier.subresourceRange.baseArrayLayer = 0;
  dst_barrier.subresourceRange.baseMipLevel = dst_mip_level;
  dst_barrier.subresourceRange.layerCount = 1;
  dst_barrier.subresourceRange.levelCount = 1;

  VkDependencyInfo dependency_info{};
  dependency_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  dependency_info.pImageMemoryBarriers = barriers;
  dependency_info.imageMemoryBarrierCount = 2;

  vkCmdPipelineBarrier2(obj, &dependency_info);

  VkImageBlit2 blit{};
  blit.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
  blit.srcSubresource.layerCount = 1;
  blit.srcSubresource.baseArrayLayer = 0;
  blit.srcSubresource.aspectMask =
      IsDepthFormat(src_image.format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
  blit.srcSubresource.mipLevel = src_mip_level;

  blit.dstSubresource.layerCount = 1;
  blit.dstSubresource.baseArrayLayer = 0;
  blit.dstSubresource.aspectMask =
      IsDepthFormat(dst_image.format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
  blit.dstSubresource.mipLevel = dst_mip_level;

  blit.srcOffsets[0] = {0, 0, 0};
  blit.dstOffsets[0] = {0, 0, 0};

  blit.srcOffsets[1] =
      Vec3u32::To<VkOffset3D>(Max(src_image.GetVec3u32() / (1 << src_mip_level), Vec3u32(1)));
  blit.dstOffsets[1] =
      Vec3u32::To<VkOffset3D>(Max(dst_image.GetVec3u32() / (1 << dst_mip_level), Vec3u32(1)));

  VkBlitImageInfo2 info{};
  info.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
  info.srcImage = src_image.obj;
  info.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  info.dstImage = dst_image.obj;
  info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  info.filter = VK_FILTER_LINEAR;
  info.pRegions = &blit;
  info.regionCount = 1;

  vkCmdBlitImage2(obj, &info);
}

void VulkanCommandBuffer::BeginRendering(const std::vector<BaseVulkanImage *> &attachment_images,
                                         BaseVulkanImage *depth_image, Vec2u32 extent,
                                         VkAttachmentLoadOp depth_load_op,
                                         VkAttachmentStoreOp depth_store_op) {
  VkViewport viewport = Viewport({extent.width, extent.height, 1});
  vkCmdSetViewport(obj, 0, 1, &viewport);
  VkRect2D scissor = Scissor({extent.width, extent.height, 1});
  vkCmdSetScissor(obj, 0, 1, &scissor);

  VkClearColorValue clear_color_value{};
  clear_color_value = {0.0f, 0.0f, 0.0f, 1.0f};

  VkClearValue clear_value{};
  clear_value.color = clear_color_value;

  std::vector<VkRenderingAttachmentInfo> attachements;
  VkRenderingAttachmentInfo depth_attachment;
  VkRenderingInfo rendering_info = RenderingInfo({extent.width, extent.height, 1}, {}, nullptr);
  if (attachment_images.size() != 0) {
    attachements.reserve(attachment_images.size());

    for (BaseVulkanImage *image : attachment_images) {
      attachements.push_back(AttachmentInfo(image->view, VK_NULL_HANDLE, &clear_value,
                                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL));
    }

    rendering_info.colorAttachmentCount = attachements.size();
    rendering_info.pColorAttachments = attachements.data();
  }
  if (depth_image) {
    depth_attachment = DepthAttachmentInfo(depth_image->view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                           depth_load_op, depth_store_op);
    rendering_info.pDepthAttachment = &depth_attachment;
  }

  vkCmdBeginRendering(obj, &rendering_info);
}

void VulkanCommandBuffer::BeginMultiRendering(const u32 viewport_count, const u32 layer_count,
                                              const std::vector<BaseVulkanImage *> &attachment_images,
                                              BaseVulkanImage *depth_image, Vec2u32 extent,
                                              VkAttachmentLoadOp depth_load_op,
                                              VkAttachmentStoreOp depth_store_op) {
  std::vector<VkViewport> viewports(viewport_count);
  std::vector<VkRect2D> scissors(viewport_count);
  for (u32 i = 0; i < viewport_count; i++) {
    viewports[i] = Viewport({extent.width, extent.height, 1});
    scissors[i] = Scissor({extent.width, extent.height, 1});
  }
  vkCmdSetViewport(obj, 0, viewport_count, viewports.data());
  vkCmdSetScissor(obj, 0, viewport_count, scissors.data());

  VkClearColorValue clear_color_value{};
  clear_color_value = {0.0f, 0.0f, 0.0f, 1.0f};

  VkClearValue clear_value{};
  clear_value.color = clear_color_value;

  std::vector<VkRenderingAttachmentInfo> attachements;
  VkRenderingAttachmentInfo depth_attachment;
  VkRenderingInfo rendering_info = RenderingInfo({extent.width, extent.height, 1}, {}, nullptr);
  rendering_info.layerCount = layer_count;
  if (attachment_images.size() != 0) {
    attachements.reserve(attachment_images.size());

    for (const BaseVulkanImage *image : attachment_images) {
      attachements.push_back(AttachmentInfo(image->view, VK_NULL_HANDLE, &clear_value,
                                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL));
    }

    rendering_info.colorAttachmentCount = attachements.size();
    rendering_info.pColorAttachments = attachements.data();
  }
  if (depth_image) {
    depth_attachment = DepthAttachmentInfo(depth_image->view, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                           depth_load_op, depth_store_op);
    rendering_info.pDepthAttachment = &depth_attachment;
  }

  vkCmdBeginRendering(obj, &rendering_info);
}

void VulkanCommandBuffer::EndRendering() {
  vkCmdEndRendering(obj);
  bound_pipeline_layout = nullptr;
}

void VulkanCommandBuffer::BindIndexBuffer(const VulkanBuffer &index_buffer) {
  Assert((index_buffer.usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) != 0, "Must bind an index buffer");
  vkCmdBindIndexBuffer(obj, index_buffer.obj, 0, VK_INDEX_TYPE_UINT32);
}

void VulkanCommandBuffer::BindDescriptors(const std::vector<VkDescriptorSet> &descriptors) {
  Assert(bound_pipeline_layout, "You must bind a pipeline to bind descriptor");
  vkCmdBindDescriptorSets(obj, bind_point, *bound_pipeline_layout, 0, descriptors.size(), descriptors.data(),
                          0, nullptr);
}

void VulkanCommandBuffer::Draw(u32 vertex_count, u32 instance_count, u32 vertex_offset, u32 instance_offset) {
  vkCmdDraw(obj, vertex_count, instance_count, vertex_offset, instance_offset);
}

void VulkanCommandBuffer::DrawIndexed(u32 index_count, u32 instance_count, u32 first_index, i32 vertex_offset,
                                      u32 first_instance) {
  vkCmdDrawIndexed(obj, index_count, instance_count, first_index, vertex_offset, first_instance);
}
} // namespace Core
