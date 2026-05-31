#include "Core/Render/debug_frame.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/context.h"
#include "Core/Render/types.h"

namespace Core {
void DrawDebugAABBs() {
  VulkanCommandBuffer &cmd = render_context->swapchain.GetActiveCommandBuffer();

  const u32 resource_index = render_context->swapchain.frame_index;

  {
    cmd.BeginDebugPass("debug aabb pass");

    VulkanSubPass<SubPassType::Graphic> pass;
    pass.AddDependency<DeviceResourceType::ColorAttachment>(render_context->swapchain.GetImage());
    pass.AddDependency<DeviceResourceType::Buffer>(render_context->camera_buffer[resource_index]);
    for (u32 i = 0; i < render_context->mesh_voxel_aabb_buffer_arr.size(); i++) {
      pass.AddDependency<DeviceResourceType::Buffer>(*render_context->mesh_voxel_aabb_buffer_arr[i]);
    }

    cmd.BindSubPass(pass);

    cmd.BeginRendering({&render_context->swapchain.GetImage()}, nullptr,
                       render_context->swapchain.GetImage().GetVec2u32(), /*clear=*/false);
    cmd.BindPipeline(render_context->debug_aabb_pipeline);
    cmd.BindDescriptors({
        render_context->mesh_descriptor,
        render_context->camera_descriptor[resource_index],
    });

    for (u32 i = 0; i < render_context->instance_counted_buffer.cpu_append_count; i++) {
      cmd.Draw(render_context->mesh_voxel_aabb_buffer_arr[i]->cpu_append_count, 1, 0, i);
    }

    cmd.EndRendering();

    cmd.EndDebugPass();
  }
}
} // namespace Core
