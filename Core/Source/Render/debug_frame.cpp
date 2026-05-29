#include "Core/Render/debug_frame.h"

<<<<<<< Updated upstream
namespace Core {
void DrawDebugAABBs() {
  VulkanCommandBuffer &cmd = render_context->swapchain.GetActiveCommandBuffer();

  const u32 resource_index = render_context->swapchain.frame_index;

  cmd.BeginDebugPass("aabb debug pass");
  VulkanSubPass<SubPassType::Graphic> debug_pass;
  debug_pass.AddDependency<DeviceResourceType::Buffer>(render_context->mesh_counted_buffer);
  debug_pass.AddDependency<DeviceResourceType::Buffer>(render_context->camera_buffer[resource_index]);
  debug_pass.AddDependency<DeviceResourceType::ColorAttachment>(render_context->swapchain.GetImage());

  cmd.BindSubPass(debug_pass);

  cmd.BeginRendering({&render_context->swapchain.GetImage()}, nullptr, render_context->swapchain.extent,
                     /*clear=*/false);
  cmd.BindPipeline(render_context->debug_aabb_pipeline);
  cmd.BindDescriptors({
      render_context->mesh_descriptor,
      render_context->camera_descriptor[resource_index],
  });
  cmd.Draw(render_context->instance_counted_buffer.cpu_append_count);
  cmd.EndRendering();
  cmd.EndDebugPass();
}
} // namespace Core
=======
namespace Core {} // namespace Core
>>>>>>> Stashed changes
