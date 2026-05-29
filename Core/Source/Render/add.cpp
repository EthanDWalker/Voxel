#include "Core/Render/add.h"
#include "Core/Render/Vulkan/acceleration_structure.h"
#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/image.h"
#include "Core/Render/Vulkan/other_buffer.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/Vulkan/util.h"
#include "Core/Render/context.h"
#include "Core/Render/types.h"
#include "Core/Util/Parse/object.h"
#include <memory>

namespace Core {
u32 AddDirectionalLight(const DirectionalLight &dir_light) {
  ZoneScoped;

  VulkanBuffer<BufferType::StagingBuffer> staging_buffer = "directional light staging buffer";
  staging_buffer.Create(sizeof(DirectionalLight));
  memcpy(staging_buffer.host_address, &dir_light, sizeof(DirectionalLight));

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    VulkanSubPass<SubPassType::Transfer> transfer_pass;
    transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(staging_buffer);
    transfer_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->directional_light_buffer);

    cmd.BindSubPass(transfer_pass);

    render_context->directional_light_buffer.Append(cmd, staging_buffer);
  });

  return render_context->directional_light_buffer.cpu_append_count - 1;
}

void AddObject(const ObjectData &object) {
  ZoneScoped;

  const u32 initial_mesh_index = render_context->mesh_node_brick_arr.size();

  const f32 VOXEL_AABB_SIZE = 0.5f;

  std::vector<std::unique_ptr<VulkanBuffer<BufferType::StructuredBuffer, Vertex>>> vertex_buffer_arr;
  vertex_buffer_arr.reserve(object.mesh_data_arr.size());
  std::vector<std::unique_ptr<VulkanBuffer<BufferType::StructuredBuffer, Index>>> index_buffer_arr;
  index_buffer_arr.reserve(object.mesh_data_arr.size());

  for (u32 i = 0; i < object.mesh_data_arr.size(); i++) {
    vertex_buffer_arr
        .emplace_back(std::make_unique<VulkanBuffer<BufferType::StructuredBuffer, Vertex>>("vertex buffer"))
        ->Create(object.mesh_data_arr[i].vertex_count,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    index_buffer_arr
        .emplace_back(std::make_unique<VulkanBuffer<BufferType::StructuredBuffer, Index>>("index buffer"))
        ->Create(object.mesh_data_arr[i].index_count,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    render_context->voxelize_descriptor.Update<DeviceResourceType::Buffer>(2, vertex_buffer_arr[i].get(), i);
    render_context->voxelize_descriptor.Update<DeviceResourceType::Buffer>(3, index_buffer_arr[i].get(), i);
  }

  VulkanBuffer<BufferType::StagingBuffer> mesh_aabb_staging_buffer = "mesh aabb staging buffer";
  mesh_aabb_staging_buffer.Create(sizeof(AABB) * object.mesh_data_arr.size());

  std::vector<std::unique_ptr<VulkanBuffer<BufferType::CountedBuffer, AABB>>> mesh_voxel_aabb_buffer_arr;
  mesh_voxel_aabb_buffer_arr.reserve(object.mesh_data_arr.size());

  VulkanBuffer<BufferType::StagingBuffer> mesh_voxel_aabb_count_buffer = "mesh voxel aabb count buffer";
  mesh_voxel_aabb_count_buffer.Create(object.mesh_data_arr.size() * sizeof(u32));

  for (u32 i = 0; i < object.mesh_data_arr.size(); i++) {
    AABB &aabb = ((AABB *)mesh_aabb_staging_buffer.host_address)[i];
    const Vec3f32 aabb_center = (object.mesh_data_arr[i].aabb.min + object.mesh_data_arr[i].aabb.max) / 2.0f;

    aabb.min = Ceil(Abs(object.mesh_data_arr[i].aabb.min) / VOXEL_AABB_SIZE) * VOXEL_AABB_SIZE *
               Sign(object.mesh_data_arr[i].aabb.min);
    aabb.max = Ceil(Abs(object.mesh_data_arr[i].aabb.max) / VOXEL_AABB_SIZE) * VOXEL_AABB_SIZE *
               Sign(object.mesh_data_arr[i].aabb.max);
    aabb.min = Select(aabb.min > object.mesh_data_arr[i].aabb.min, aabb.min - VOXEL_AABB_SIZE, aabb.min);
    aabb.max = Select(aabb.max < object.mesh_data_arr[i].aabb.max, aabb.max + VOXEL_AABB_SIZE, aabb.max);

    Core::Log("min {} max {}", aabb.min.String(), aabb.max.String());

    Assert(All(aabb.min <= object.mesh_data_arr[i].aabb.min),
           "fitted aabb min is smaller than mesh aabb min ({}) ({})", aabb.min.String(),
           object.mesh_data_arr[i].aabb.min.String());
    Assert(All(aabb.max >= object.mesh_data_arr[i].aabb.max),
           "fitted aabb max is smaller than mesh aabb max ({}) ({})", aabb.max.String(),
           object.mesh_data_arr[i].aabb.max.String());

    const Vec3u32 voxel_aabb_extent = VecTypeCast<u32>(Ceil((aabb.max - aabb.min) / VOXEL_AABB_SIZE));
    const u32 voxel_aabb_count = voxel_aabb_extent.x * voxel_aabb_extent.y * voxel_aabb_extent.z;

    render_context->mesh_node_brick_arr
        .emplace_back(std::make_unique<VulkanBuffer<BufferType::StructuredBuffer, BranchNode>>(
            "voxel node brick buffer"))
        ->Create(voxel_aabb_count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    render_context->voxelize_descriptor.Update<DeviceResourceType::Buffer>(
        0, render_context->mesh_node_brick_arr.back().get(), i);
    render_context->mesh_descriptor.Update<DeviceResourceType::Buffer>(
        1, render_context->mesh_node_brick_arr.back().get(), render_context->mesh_node_brick_arr.size() - 1);

    mesh_voxel_aabb_buffer_arr
        .emplace_back(
            std::make_unique<VulkanBuffer<BufferType::CountedBuffer, AABB>>("mesh voxel aabb buffer"))
        ->Create(voxel_aabb_count, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                       VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    render_context->voxelize_descriptor.Update<DeviceResourceType::Buffer>(
        1, mesh_voxel_aabb_buffer_arr[i].get(), i);
  }

  VulkanBuffer<BufferType::StructuredBuffer, AABB> mesh_aabb_buffer = "mesh aabb buffer";
  mesh_aabb_buffer.Create(object.mesh_data_arr.size(),
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  render_context->voxelize_descriptor.Update<DeviceResourceType::Buffer>(4, &mesh_aabb_buffer);

  std::vector<std::unique_ptr<VulkanImage<ImageType::Planar>>> albedo_image_arr;
  albedo_image_arr.resize(object.material_data_arr.size());

  for (u32 i = 0; i < object.material_data_arr.size(); i++) {
    if (!object.material_data_arr[i].initialized)
      continue;

    albedo_image_arr[i] = std::make_unique<VulkanImage<ImageType::Planar>>();
    albedo_image_arr[i]->Create(object.material_data_arr[i].albedo_extent, VK_FORMAT_BC1_RGB_UNORM_BLOCK,
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    render_context->voxelize_descriptor.Update<DeviceResourceType::SampledImage>(5, albedo_image_arr[i].get(),
                                                                                 i);
  }

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    cmd.BeginDebugPass("voxelize info transfer");
    {
      VulkanSubPass<SubPassType::Transfer> transfer_pass;
      transfer_pass.ReserveBufferDependencies(vertex_buffer_arr.size() * 2);
      for (u32 i = 0; i < vertex_buffer_arr.size(); i++) {
        transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(
            *object.mesh_data_arr[i].vertex_host_buffer);
        transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*vertex_buffer_arr[i]);
      }

      transfer_pass.ReserveBufferDependencies(index_buffer_arr.size() * 2);
      for (u32 i = 0; i < index_buffer_arr.size(); i++) {
        transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(
            *object.mesh_data_arr[i].index_host_buffer);
        transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*index_buffer_arr[i]);
      }

      transfer_pass.ReserveBufferDependencies(albedo_image_arr.size());
      transfer_pass.ReserveImageDependencies(albedo_image_arr.size());
      for (u32 i = 0; i < albedo_image_arr.size(); i++) {
        if (!object.material_data_arr[i].initialized)
          continue;

        transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(
            *object.material_data_arr[i].compressed_albedo_data_buffer);
        transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*albedo_image_arr[i]);
      }

      cmd.BindSubPass(transfer_pass);

      for (u32 i = 0; i < vertex_buffer_arr.size(); i++) {
        cmd.UploadBufferToBuffer(*object.mesh_data_arr[i].vertex_host_buffer, *vertex_buffer_arr[i],
                                 vertex_buffer_arr[i]->size);
      }
      for (u32 i = 0; i < index_buffer_arr.size(); i++) {
        cmd.UploadBufferToBuffer(*object.mesh_data_arr[i].index_host_buffer, *index_buffer_arr[i],
                                 index_buffer_arr[i]->size);
      }
      for (u32 i = 0; i < albedo_image_arr.size(); i++) {
        if (!object.material_data_arr[i].initialized)
          continue;

        cmd.UploadBufferToImage(*object.material_data_arr[i].compressed_albedo_data_buffer,
                                *albedo_image_arr[i]);
      }
      cmd.UploadBufferToBuffer(mesh_aabb_staging_buffer, mesh_aabb_buffer, mesh_aabb_staging_buffer.size);

      cmd.EndDebugPass();
    }

    {
      cmd.BeginDebugPass("voxelize pass");
      VulkanSubPass<SubPassType::Graphic> voxelize_pass;
      voxelize_pass.ReserveBufferDependencies(vertex_buffer_arr.size());
      for (u32 i = 0; i < vertex_buffer_arr.size(); i++) {
        voxelize_pass.AddDependency<DeviceResourceType::Buffer>(*vertex_buffer_arr[i]);
      }

      voxelize_pass.ReserveBufferDependencies(index_buffer_arr.size());
      for (u32 i = 0; i < index_buffer_arr.size(); i++) {
        voxelize_pass.AddDependency<DeviceResourceType::Buffer>(*index_buffer_arr[i]);
      }

      voxelize_pass.ReserveBufferDependencies(mesh_voxel_aabb_buffer_arr.size());
      for (u32 i = 0; i < mesh_voxel_aabb_buffer_arr.size(); i++) {
        voxelize_pass.AddDependency<DeviceResourceType::RWBuffer>(*mesh_voxel_aabb_buffer_arr[i]);
      }

      voxelize_pass.ReserveBufferDependencies(mesh_voxel_aabb_buffer_arr.size());
      for (u32 i = 0; i < mesh_voxel_aabb_buffer_arr.size(); i++) {
        voxelize_pass.AddDependency<DeviceResourceType::RWBuffer>(
            *render_context->mesh_node_brick_arr[i + initial_mesh_index]);
      }

      voxelize_pass.ReserveImageDependencies(albedo_image_arr.size());
      for (u32 i = 0; i < albedo_image_arr.size(); i++) {
        if (!object.material_data_arr[i].initialized)
          continue;

        voxelize_pass.AddDependency<DeviceResourceType::SampledImage>(*albedo_image_arr[i]);
      }

      voxelize_pass.AddDependency<DeviceResourceType::Buffer>(mesh_aabb_buffer);

      cmd.BindSubPass(voxelize_pass);

      cmd.BeginRendering({}, nullptr, Vec2u32(1000), false);
      cmd.BindPipeline(render_context->allocate_pipeline);
      cmd.BindDescriptors({
          render_context->voxelize_descriptor,
      });

      for (u32 i = 0; i < object.mesh_data_arr.size(); i++) {
        AllocateInfo alloc_info;
        alloc_info.mesh_index = i;
        alloc_info.albedo_index = object.mesh_data_arr[i].material_index;

        cmd.PushConstants(VK_SHADER_STAGE_ALL_GRAPHICS, sizeof(AllocateInfo), &alloc_info);
        cmd.Draw(object.mesh_data_arr[i].index_count / 3);
        cmd.ClearPushConstants();
      }

      cmd.EndRendering();

      cmd.EndDebugPass();
    }

    {
      cmd.BeginDebugPass("transfer voxelize info");

      VulkanSubPass<SubPassType::Transfer> transfer_pass;
      transfer_pass.ReserveBufferDependencies(mesh_voxel_aabb_buffer_arr.size());
      for (u32 i = 0; i < mesh_voxel_aabb_buffer_arr.size(); i++) {
        transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(*mesh_voxel_aabb_buffer_arr[i]);
      }
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(mesh_voxel_aabb_count_buffer);

      cmd.BindSubPass(transfer_pass);

      for (u32 i = 0; i < mesh_voxel_aabb_buffer_arr.size(); i++) {
        cmd.UploadBufferToBuffer(*mesh_voxel_aabb_buffer_arr[i], mesh_voxel_aabb_count_buffer, sizeof(u32),
                                 offsetof(VulkanBuffer<BufferType::CountedBuffer>::Header, count),
                                 i * sizeof(u32));
      }

      cmd.EndDebugPass();
    }
  });

  for (u32 i = 0; i < object.mesh_data_arr.size(); i++) {
    mesh_voxel_aabb_buffer_arr[i]->cpu_append_count = ((u32 *)mesh_voxel_aabb_count_buffer.host_address)[i];
    Core::Log("{}", mesh_voxel_aabb_buffer_arr[i]->cpu_append_count);
    render_context->bottom_level_acceleration_structure_arr
        .emplace_back(std::make_unique<VulkanAccelerationStructure<AccelerationStructureType::AABB>>())
        ->Create(*mesh_voxel_aabb_buffer_arr[i]);
  }

  VulkanBuffer<BufferType::StagingBuffer> instance_staging_buffer = "instance staging buffer";
  instance_staging_buffer.Create(sizeof(Instance) * object.instance_data_arr.size());

  for (u32 i = 0; i < object.instance_data_arr.size(); i++) {
    Instance &instance = ((Instance *)instance_staging_buffer.host_address)[i];
    instance.instanceCustomIndex = object.instance_data_arr[i].mesh_index;
    instance.transform = Mat4ToVkTransform(object.instance_data_arr[i].matrix);
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = 0;
    instance.mask = 0xFF;
    instance.accelerationStructureReference = GetDeviceAddress(
        render_context
            ->bottom_level_acceleration_structure_arr[instance.instanceCustomIndex + initial_mesh_index]
            ->obj);
  }

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    {
      cmd.BeginDebugPass("instance pass");

      VulkanSubPass<SubPassType::Transfer> transfer_pass;
      transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(instance_staging_buffer);
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->instance_counted_buffer);

      cmd.BindSubPass(transfer_pass);

      render_context->instance_counted_buffer.Append(cmd, instance_staging_buffer,
                                                     object.instance_data_arr.size());

      cmd.EndDebugPass();
    }
  });

  render_context->top_level_acceleration_structure.Recreate(render_context->instance_counted_buffer);
  render_context->mesh_descriptor.Update<DeviceResourceType::AccelerationStructure>(
      0, &render_context->top_level_acceleration_structure);
}
}; // namespace Core
