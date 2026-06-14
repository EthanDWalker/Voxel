#include "Core/Render/debug.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/context.h"
#include "Core/Render/types.h"

namespace Core {
void DebugDrawChunkBoundaries() {
  VulkanCommandBuffer &cmd = render_context->swapchain.GetActiveCommandBuffer();

  cmd.BeginDebugPass("chunk boundaries draw");

  VulkanSubPass<SubPassType::Graphic> pass;
  pass.AddDependency<DeviceResourceType::ColorAttachment>(render_context->swapchain.GetImage());
  pass.AddDependency<DeviceResourceType::Buffer>(
      render_context->camera_buffer[render_context->swapchain.frame_index]);

  cmd.BindSubPass(pass);

  cmd.BeginRendering({&render_context->swapchain.GetImage()}, nullptr, render_context->swapchain.extent,
                     false);

  cmd.BindPipeline(render_context->debug_chunk_boundaries_pipeline);
  cmd.BindDescriptors({
      render_context->voxel_tree.descriptor,
      render_context->camera_descriptor[render_context->swapchain.frame_index],
      render_context->image_descriptor,
  });

  cmd.Draw(64);

  cmd.EndRendering();
  cmd.EndDebugPass();
};
}; // namespace Core
