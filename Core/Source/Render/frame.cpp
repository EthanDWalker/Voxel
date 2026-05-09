#include "Core/Render/frame.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/camera.h"
#include "Core/Render/context.h"
#include "Core/Render/sparse_voxel_tree.h"
#include "Core/Render/types.h"
#include "Core/window.h"
#include <cstring>

namespace Core {
void BeginFrame(bool &resize) {
  FrameMark;
  ZoneScoped;
  render_context->swapchain.AcquireNextImage(resize);
  render_context->swapchain.BeginCommandBuffer();
}

void WaitIdle() {
  ZoneScoped;
  vkDeviceWaitIdle(VulkanContext::device);
}

void EndFrame(bool &resize) {
  ZoneScoped;
  render_context->swapchain.SubmitCommandBuffer();
  render_context->swapchain.Present(resize);
  if (resize) {
    render_context->swapchain.Resize(Window::GetSize());
  }
}

void Frame(Camera &camera) {
  ZoneScoped;
  camera.Update();

  VulkanCommandBuffer &cmd = render_context->swapchain.GetActiveCommandBuffer();

  {
    VulkanSubPass<SubPassType::Transfer> upload_pass;
    upload_pass.AddDependency<DeviceResourceType::TransferSrc>(render_context->frame_staging_buffer);
    upload_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->main_image);
    upload_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->camera_buffer);

    cmd.BindSubPass(upload_pass);

    u64 offset = 0;
    memcpy((char *)render_context->frame_staging_buffer.host_address + offset, &camera.ubo,
           sizeof(Camera::UBO));

    cmd.UploadBufferToBuffer(render_context->frame_staging_buffer, render_context->camera_buffer,
                             sizeof(Camera::UBO), offset);

    offset += sizeof(Camera::UBO);
  }

  {
    VulkanSubPass<SubPassType::Raytrace> triangle_id_pass;
    triangle_id_pass.AddDependency(render_context->top_level_acceleration_structure);
    triangle_id_pass.AddDependency<DeviceResourceType::Buffer>(render_context->camera_buffer);
    triangle_id_pass.AddDependency<DeviceResourceType::StorageImage>(render_context->main_image);
    triangle_id_pass.AddDependency<DeviceResourceType::Buffer>(render_context->mesh_triangle_offset_buffer);
    triangle_id_pass.AddDependency<DeviceResourceType::RWBuffer>(
        render_context->triangle_id_dense_set.header_buffer);
    triangle_id_pass.AddDependency<DeviceResourceType::RWBuffer>(
        render_context->triangle_id_dense_set.value_buffer);
    triangle_id_pass.AddDependency<DeviceResourceType::RWBuffer>(
        render_context->triangle_id_dense_set.key_buffer);

    triangle_id_pass.ReserveBufferDependencies(render_context->mesh_count);
    for (u32 i = 0; i < render_context->mesh_count; i++) {
      triangle_id_pass.AddDependency<DeviceResourceType::Buffer>(
          *render_context->triangle_voxelized_depth_buffers[i]);
    }

    cmd.BindSubPass(triangle_id_pass);

    cmd.BindPipeline(render_context->triangle_id_pipeline);
    cmd.BindDescriptors({
        render_context->top_level_acceleration_structure_descriptor,
        render_context->camera_descriptor,
        render_context->image_descriptor,
        render_context->mesh_descriptor,
        render_context->triangle_id_dense_set.descriptor,
    });
    cmd.TraceRays(render_context->beam_prepass_image.GetVec3u32(),
                  render_context->triangle_id_shader_binding_table);
  }

  {
    VulkanSubPass<SubPassType::Compute> beam_pass;
    beam_pass.AddDependency<DeviceResourceType::RWStorageImage>(render_context->beam_prepass_image);
    beam_pass.AddDependency<DeviceResourceType::Buffer>(render_context->camera_buffer);

    cmd.BindSubPass(beam_pass);

    cmd.BindPipeline(render_context->beam_prepass_pipeline);
    cmd.BindDescriptors({
        render_context->image_descriptor,
        render_context->camera_descriptor,
        render_context->voxel_tree.descriptor,
        render_context->triangle_id_dense_set.descriptor,
        render_context->mesh_descriptor,
    });
    cmd.Dispatch(Vec3u32(render_context->beam_prepass_image.GetVec2u32() / 8 + 1, 1));
  }

  {
    VulkanSubPass<SubPassType::Graphic> reallocate_pass;
    reallocate_pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->triangle_id_dense_set.header_buffer);
    reallocate_pass.AddDependency<DeviceResourceType::Buffer>(
        render_context->triangle_id_dense_set.value_buffer);

    reallocate_pass.ReserveBufferDependencies(render_context->triangle_voxelized_depth_buffers.size());
    reallocate_pass.ReserveBufferDependencies(render_context->vertex_buffers.size());
    reallocate_pass.ReserveBufferDependencies(render_context->index_buffers.size());
    for (u32 i = 0; i < render_context->mesh_count; i++) {
      reallocate_pass.AddDependency<DeviceResourceType::RWBuffer>(
          *render_context->triangle_voxelized_depth_buffers[i]);
      reallocate_pass.AddDependency<DeviceResourceType::Buffer>(*render_context->vertex_buffers[i]);
      reallocate_pass.AddDependency<DeviceResourceType::Buffer>(*render_context->index_buffers[i]);
    }

    reallocate_pass.AddDependency<DeviceResourceType::Buffer>(render_context->mesh_triangle_offset_buffer);

    cmd.BindSubPass(reallocate_pass);

    cmd.BeginRendering({}, nullptr, 1 << (5 * 2));
    cmd.BindPipeline(render_context->reallocate_pipeline);
    cmd.BindDescriptors({
        render_context->mesh_descriptor,
        render_context->voxel_tree.descriptor,
        render_context->triangle_id_dense_set.descriptor,
    });
    cmd.Draw(render_context->triangle_id_dense_set.size);
    cmd.EndRendering();
  }

  {
    VulkanSubPass<SubPassType::Compute> main_pass;
    main_pass.AddDependency<DeviceResourceType::RWStorageImage>(render_context->main_image);
    main_pass.AddDependency<DeviceResourceType::Buffer>(render_context->camera_buffer);
    main_pass.AddDependency<DeviceResourceType::Buffer>(render_context->directional_light_buffer);

    cmd.BindSubPass(main_pass);

    cmd.BindPipeline(render_context->main_pipeline);
    cmd.BindDescriptors({
        render_context->image_descriptor,
        render_context->camera_descriptor,
        render_context->voxel_tree.descriptor,
        render_context->light_descriptor,
    });
    cmd.Dispatch(Vec3u32(render_context->main_image.GetVec2u32() / 8 + 1, 1));
  }

  {
    VulkanSubPass<SubPassType::Transfer> transer_pass;
    transer_pass.AddDependency<DeviceResourceType::TransferSrc>(render_context->main_image);
    transer_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->swapchain.GetImage());

    transer_pass.AddDependency<DeviceResourceType::TransferDst>(
        render_context->triangle_id_dense_set.key_buffer);
    transer_pass.AddDependency<DeviceResourceType::TransferSrc>(
        render_context->triangle_id_dense_set.header_host_buffer);
    transer_pass.AddDependency<DeviceResourceType::TransferDst>(
        render_context->triangle_id_dense_set.header_buffer);

    cmd.BindSubPass(transer_pass);

    cmd.CopyImageToImage(render_context->main_image, render_context->swapchain.GetImage());
    cmd.FillBuffer(render_context->triangle_id_dense_set.key_buffer,
                   render_context->triangle_id_dense_set.key_buffer.size, DeviceDenseSet::EMPTY_KEY);

    cmd.UploadBufferToBuffer(render_context->triangle_id_dense_set.header_host_buffer,
                             render_context->triangle_id_dense_set.header_buffer,
                             render_context->triangle_id_dense_set.header_buffer.size);
  }
}

void Resize(Vec2u32 extent) {
  ZoneScoped;
  WaitIdle();

  render_context->main_image.Recreate(extent, render_context->main_image.format,
                                      render_context->main_image.usage);
  render_context->image_descriptor.Update<DeviceResourceType::RWStorageImage>(0, &render_context->main_image);

  render_context->beam_prepass_image.Recreate(extent / PREPASS_SCALE,
                                              render_context->beam_prepass_image.format,
                                              render_context->beam_prepass_image.usage, /*referenced=*/true);
  render_context->image_descriptor.Update<DeviceResourceType::RWStorageImage>(
      1, &render_context->beam_prepass_image);

  Core::Log("{}", extent.String());
}
} // namespace Core
