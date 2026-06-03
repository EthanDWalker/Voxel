#include "Core/Util/compress.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/other_buffer.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/types.h"
#include "Core/Util/context.h"
#include "Core/Util/types.h"

namespace Core {

struct BC1Block {
  u32 colors;
  u32 indices;
};

void CompressBC1(const Vec2u32 extent, VulkanBuffer<BufferType::StagingBuffer> &input_buffer,
                 VulkanBuffer<BufferType::StagingBuffer> &output_buffer) {
  Assert((input_buffer.usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0,
         "input bc1 buffer must have VK_BUFFER_USAGE_STORAGE_BUFFER_BIT");
  const Vec2u32 blocks = Vec2u32((extent.width + 3) / 4, (extent.height + 3) / 4);

  output_buffer.Create(blocks.x * blocks.y * sizeof(BC1Block), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  util_context->bc1_compression_descriptor.Update<DeviceResourceType::Buffer>(0, &input_buffer);
  util_context->bc1_compression_descriptor.Update<DeviceResourceType::Buffer>(1, &output_buffer);

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    {
      VulkanSubPass<SubPassType::Compute> compression_pass;
      compression_pass.AddDependency<DeviceResourceType::Buffer>(input_buffer);
      compression_pass.AddDependency<DeviceResourceType::RWBuffer>(output_buffer);

      cmd.BindSubPass(compression_pass);

      BC1CompressionPushConstants push_constants{};
      push_constants.extent = extent;

      cmd.BindPipeline(util_context->bc1_compression_pipeline);
      cmd.BindDescriptors({util_context->bc1_compression_descriptor});
      cmd.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(BC1CompressionPushConstants), &push_constants);
      cmd.Dispatch(Vec3u32((blocks / 8) + 1, 1));
    }
  });
}
} // namespace Core
