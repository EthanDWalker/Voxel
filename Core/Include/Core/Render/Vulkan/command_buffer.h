#pragma once

#include "Core/Render/Vulkan/command_pool.h"
#include "Core/Render/Vulkan/image.h"
#include "Core/Render/Vulkan/pipeline.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "volk.h"

namespace Core {
struct VulkanCommandBuffer {
  VulkanCommandBuffer() = default;

  VulkanCommandBuffer(const VulkanCommandBuffer &) = delete;
  VulkanCommandBuffer &operator=(const VulkanCommandBuffer &) = delete;

  VulkanCommandBuffer(VulkanCommandBuffer &&) = default;
  VulkanCommandBuffer &operator=(VulkanCommandBuffer &&) = default;

  VkCommandBuffer obj = VK_NULL_HANDLE;
  VkPipelineBindPoint bind_point;
  const VkPipelineLayout *bound_pipeline_layout = nullptr;
  u32 push_constant_offset = 0;

  void CreatePrimary(VulkanCommandPool &pool) {
    obj = pool.AllocateCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
  }

  void CreateSecondary(VulkanCommandPool &pool) {
    obj = pool.AllocateCommandBuffer(VK_COMMAND_BUFFER_LEVEL_SECONDARY);
  }

  void Begin();
  void End();

  void BeginRendering(const std::vector<BaseVulkanImage *> &attachment_images,
                      BaseVulkanImage *depth_image, Vec2u32 extent,
                      VkAttachmentLoadOp depth_load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
                      VkAttachmentStoreOp depth_store_op = VK_ATTACHMENT_STORE_OP_STORE);
  void BeginMultiRendering(const u32 viewport_count, const u32 layer_count,
                           const std::vector<BaseVulkanImage *> &attachment_images,
                           BaseVulkanImage *depth_image, Vec2u32 extent,
                           VkAttachmentLoadOp depth_load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
                           VkAttachmentStoreOp depth_store_op = VK_ATTACHMENT_STORE_OP_STORE);
  void EndRendering();

  void Dispatch(Vec3u32 groups);
  void BindIndexBuffer(const VulkanBuffer &index_buffer);
  void BindDescriptors(const std::vector<VkDescriptorSet> &descriptors);
  void PushConstants(VkShaderStageFlagBits stages, u32 size, const void *data);
  void Draw(u32 vertex_count, u32 instance_count = 1, u32 vertex_offset = 0,
            u32 instance_offset = 0);
  void DrawIndexed(u32 index_count, u32 instance_count = 1, u32 first_index = 0,
                   i32 vertex_offset = 0, u32 first_instance = 0);

  void UploadBufferToBuffer(const VulkanBuffer &src_buffer, const VulkanBuffer &dst_buffer,
                            const u64 size, const u64 src_offset = 0, const u64 dst_offset = 0);
  void FillBuffer(const VulkanBuffer &buffer, const u64 fill_size, const u32 data,
                  const u32 offset = 0);
  void UploadBufferToImage(const VulkanBuffer &buffer, const BaseVulkanImage &image,
                           const u32 src_offset = 0, const u32 mip_level = 0);
  void Blit(const BaseVulkanImage &src_image, const BaseVulkanImage &dst_image,
            const u32 src_mip_level, const u32 dst_mip_level);

  template <SubPassType T> void BindSubPass(const VulkanSubPass<T> &sub_pass) {
    VkDependencyInfo dependency_info{};
    dependency_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency_info.pBufferMemoryBarriers = sub_pass.buffer_barriers.data();
    dependency_info.bufferMemoryBarrierCount = sub_pass.buffer_barriers.size();
    dependency_info.pImageMemoryBarriers = sub_pass.image_barriers.data();
    dependency_info.imageMemoryBarrierCount = sub_pass.image_barriers.size();

    vkCmdPipelineBarrier2(obj, &dependency_info);
  }

  void CopyImageToImage(const BaseVulkanImage &src_image, const BaseVulkanImage &dst_image);

  template <PipelineType T> void BindPipeline(const VulkanPipeline<T> &pipeline) {
    if constexpr (T == PipelineType::Compute) {
      bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
    } else if constexpr (T == PipelineType::Graphic) {
      bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
    } else {
      static_assert(false, "This pipeline type is not supported");
    }
    push_constant_offset = 0;
    vkCmdBindPipeline(obj, bind_point, pipeline.obj);
    bound_pipeline_layout = &pipeline.layout;
  }

  void DrawIndirect(const VulkanIndirectDrawCommand &command);
};
} // namespace Core
