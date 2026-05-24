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
#include "Core/Render/sparse_voxel_tree.h"
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

  return render_context->directional_light_count++;
}

void VoxelizeMeshes(const std::vector<InstanceData> &instance_data_arr, const std::vector<Mesh> &mesh_arr,
                    const u32 max_depth) {
  ZoneScoped;

  SparseVoxelTree::TreeHeader *const header =
      (SparseVoxelTree::TreeHeader *const)render_context->voxel_tree.tree_header_host_buffer.host_address;

  for (u32 depth = 1; depth < max_depth; depth++) {
    const u32 page_offset = render_context->voxel_tree.branch_pages.size();
    const u32 new_page_offset = header->branch_count >> SparseVoxelTree::PAGE_SIZE_EXP;

    render_context->voxel_tree.AllocateBranchPages(new_page_offset - page_offset);

    VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
      {
        cmd.BeginDebugPass("svo allocate transfer pass");
        VulkanSubPass<SubPassType::Transfer> transfer_pass;
        transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(
            render_context->voxel_tree.empty_page_host_buffer);

        transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(
            render_context->voxel_tree.tree_header_host_buffer);
        transfer_pass.AddDependency<DeviceResourceType::TransferDst>(
            render_context->voxel_tree.tree_header_buffer);

        for (u32 i = page_offset; i <= new_page_offset; i++) {
          transfer_pass.AddDependency<DeviceResourceType::TransferDst>(
              *render_context->voxel_tree.branch_pages[i]);
        }

        cmd.BindSubPass(transfer_pass);

        for (u32 i = page_offset; i <= new_page_offset; i++) {
          cmd.UploadBufferToBuffer(render_context->voxel_tree.empty_page_host_buffer,
                                   *render_context->voxel_tree.branch_pages[i],
                                   render_context->voxel_tree.branch_pages[i]->size);
        }

        cmd.UploadBufferToBuffer(render_context->voxel_tree.tree_header_host_buffer,
                                 render_context->voxel_tree.tree_header_buffer,
                                 render_context->voxel_tree.tree_header_buffer.size);
        cmd.EndDebugPass();
      }

      {
        cmd.BeginDebugPass("svo allocate pass");
        VulkanSubPass<SubPassType::Graphic> allocate_pass;
        allocate_pass.AddDependency<DeviceResourceType::RWBuffer>(
            render_context->voxel_tree.tree_header_buffer);

        allocate_pass.ReserveBufferDependencies(render_context->voxel_tree.branch_pages.size());
        for (u32 i = 0; i < render_context->voxel_tree.branch_pages.size(); i++) {
          allocate_pass.AddDependency<DeviceResourceType::RWBuffer>(
              *render_context->voxel_tree.branch_pages[i]);
        }

        cmd.BindSubPass(allocate_pass);

        cmd.BeginRendering({}, nullptr, Vec2u32(1 << (max_depth * 2)), false);
        cmd.BindPipeline(render_context->allocate_pipeline);
        cmd.BindDescriptors({render_context->voxelize_descriptor, render_context->voxel_tree.descriptor});

        AllocateInfo alloc_info;
        alloc_info.depth = depth;
        alloc_info.leaf = (depth == (max_depth - 1));

        for (u32 i = 0; i < instance_data_arr.size(); i++) {
          alloc_info.instance_index = i;
          cmd.PushConstants(VK_SHADER_STAGE_ALL_GRAPHICS, sizeof(AllocateInfo), &alloc_info);
          cmd.Draw(mesh_arr[instance_data_arr[i].mesh_index].index_count / 3);
          cmd.ClearPushConstants();
        }
        cmd.EndRendering();
        cmd.EndDebugPass();
      }

      {
        cmd.BeginDebugPass("svo allocate transfer to host");
        VulkanSubPass<SubPassType::Transfer> transfer_pass;
        transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(
            render_context->voxel_tree.tree_header_buffer);
        transfer_pass.AddDependency<DeviceResourceType::TransferDst>(
            render_context->voxel_tree.tree_header_host_buffer);

        cmd.BindSubPass(transfer_pass);

        cmd.UploadBufferToBuffer(render_context->voxel_tree.tree_header_buffer,
                                 render_context->voxel_tree.tree_header_host_buffer,
                                 render_context->voxel_tree.tree_header_host_buffer.size);
        cmd.EndDebugPass();
      }
    });
  }

  const u32 page_offset = render_context->voxel_tree.leaf_pages.size();
  const u32 new_page_offset = header->leaf_count >> SparseVoxelTree::PAGE_SIZE_EXP;

  render_context->voxel_tree.AllocateLeafPages(new_page_offset - page_offset);

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    {
      cmd.BeginDebugPass("svo allocate child mask transfer");
      VulkanSubPass<SubPassType::Transfer> transfer_pass;
      for (u32 i = page_offset; i <= new_page_offset; i++) {
        transfer_pass.AddDependency<DeviceResourceType::TransferDst>(
            *render_context->voxel_tree.leaf_pages[i]);
      }

      cmd.BindSubPass(transfer_pass);

      for (u32 i = page_offset; i <= new_page_offset; i++) {
        cmd.FillBuffer(*render_context->voxel_tree.leaf_pages[i],
                       render_context->voxel_tree.leaf_pages[i]->size, 0);
      }
      cmd.EndDebugPass();
    }

    {
      cmd.BeginDebugPass("svo allocate child mask");
      VulkanSubPass<SubPassType::Graphic> child_mask_pass;
      child_mask_pass.AddDependency<DeviceResourceType::RWBuffer>(
          render_context->voxel_tree.tree_header_buffer);

      child_mask_pass.ReserveBufferDependencies(render_context->voxel_tree.branch_pages.size());
      for (u32 i = 0; i < render_context->voxel_tree.branch_pages.size(); i++) {
        child_mask_pass.AddDependency<DeviceResourceType::RWBuffer>(
            *render_context->voxel_tree.branch_pages[i]);
      }

      child_mask_pass.ReserveBufferDependencies(render_context->voxel_tree.leaf_pages.size());
      for (u32 i = 0; i < render_context->voxel_tree.leaf_pages.size(); i++) {
        child_mask_pass.AddDependency<DeviceResourceType::RWBuffer>(
            *render_context->voxel_tree.leaf_pages[i]);
      }

      cmd.BindSubPass(child_mask_pass);

      cmd.BeginRendering({}, nullptr, Vec2u32(1 << (max_depth * 2)), false);
      cmd.BindPipeline(render_context->allocate_child_mask_pipeline);
      cmd.BindDescriptors({render_context->voxelize_descriptor, render_context->voxel_tree.descriptor});

      for (u32 i = 0; i < instance_data_arr.size(); i++) {
        AllocateInfo alloc_info;
        alloc_info.depth = max_depth - 1;
        alloc_info.leaf = true;
        alloc_info.instance_index = i;

        cmd.PushConstants(VK_SHADER_STAGE_ALL_GRAPHICS, sizeof(AllocateInfo), &alloc_info);
        cmd.Draw(mesh_arr[instance_data_arr[i].mesh_index].index_count / 3);
        cmd.ClearPushConstants();
      }

      cmd.EndRendering();
      cmd.EndDebugPass();
    }

    {
      cmd.BeginDebugPass("svo allocate child mask transfer to host");
      VulkanSubPass<SubPassType::Compute> transition;
      transition.ReserveBufferDependencies(render_context->voxel_tree.branch_pages.size());
      for (u32 i = 0; i < render_context->voxel_tree.branch_pages.size(); i++) {
        transition.AddDependency<DeviceResourceType::Buffer>(*render_context->voxel_tree.branch_pages[i]);
      }
      transition.ReserveBufferDependencies(render_context->voxel_tree.leaf_pages.size());
      for (u32 i = 0; i < render_context->voxel_tree.leaf_pages.size(); i++) {
        transition.AddDependency<DeviceResourceType::Buffer>(*render_context->voxel_tree.leaf_pages[i]);
      }

      cmd.BindSubPass(transition);
      cmd.EndDebugPass();
    }
  });
}

void AddObject(const ObjectData &object) {
  ZoneScoped;

  const u32 max_depth = SparseVoxelTree::MAX_DEPTH;

  VulkanBuffer<BufferType::StructuredBuffer, Mesh> mesh_buffer = "mesh buffer";
  mesh_buffer.Create(object.mesh_data_arr.size(),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  render_context->voxelize_descriptor.Update<DeviceResourceType::Buffer>(2, &mesh_buffer);

  VulkanBuffer<BufferType::StagingBuffer> mesh_staging_buffer = "mesh staging buffer";
  mesh_staging_buffer.Create(mesh_buffer.size);

  std::vector<Mesh> mesh_arr;
  mesh_arr.resize(object.mesh_data_arr.size());

  for (u32 i = 0; i < mesh_arr.size(); i++) {
    mesh_arr[i].index_count = object.mesh_data_arr[i].index_count;
    mesh_arr[i].min_bound = object.mesh_data_arr[i].aabb.min;
    mesh_arr[i].albedo_image_index = object.mesh_data_arr[i].material_index;
    mesh_arr[i].max_bound = object.mesh_data_arr[i].aabb.max;
  }

  memcpy(mesh_staging_buffer.host_address, mesh_arr.data(), mesh_staging_buffer.size);

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

    render_context->voxelize_descriptor.Update<DeviceResourceType::Buffer>(0, vertex_buffer_arr[i].get(), i);
    render_context->voxelize_descriptor.Update<DeviceResourceType::Buffer>(1, index_buffer_arr[i].get(), i);
  }

  std::vector<std::unique_ptr<VulkanImage<ImageType::Planar>>> albedo_image_arr;
  albedo_image_arr.resize(object.material_data_arr.size());

  for (u32 i = 0; i < object.material_data_arr.size(); i++) {
    if (!object.material_data_arr[i].initialized)
      continue;

    albedo_image_arr[i] = std::make_unique<VulkanImage<ImageType::Planar>>();
    albedo_image_arr[i]->Create(object.material_data_arr[i].albedo_extent, VK_FORMAT_BC1_RGB_UNORM_BLOCK,
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    render_context->voxelize_descriptor.Update<DeviceResourceType::SampledImage>(4, albedo_image_arr[i].get(),
                                                                                 i);
  }

  VulkanBuffer<BufferType::StagingBuffer> instance_staging_buffer = "instance staging buffer";
  instance_staging_buffer.Create(sizeof(Instance) * object.instance_data_arr.size());

  for (u32 i = 0; i < object.instance_data_arr.size(); i++) {
    Instance &instance = ((Instance *)instance_staging_buffer.host_address)[i];
    instance.transform = Mat4ToVkTransform(object.instance_data_arr[i].matrix);
    instance.instanceCustomIndex = object.instance_data_arr[i].mesh_index;
  }

  VulkanBuffer<BufferType::StructuredBuffer, Instance> instance_buffer = "instance buffer";
  instance_buffer.Create(object.instance_data_arr.size(),
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  render_context->voxelize_descriptor.Update<DeviceResourceType::Buffer>(3, &instance_buffer);

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

      transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(mesh_staging_buffer);
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(mesh_buffer);
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->mesh_counted_buffer);

      transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(instance_staging_buffer);
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(instance_buffer);
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(render_context->instance_counted_buffer);

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

      cmd.UploadBufferToBuffer(mesh_staging_buffer, mesh_buffer, mesh_buffer.size);
      render_context->mesh_counted_buffer.Append(cmd, mesh_staging_buffer, object.mesh_data_arr.size());

      cmd.UploadBufferToBuffer(instance_staging_buffer, instance_buffer, instance_buffer.size);
      render_context->instance_counted_buffer.Append(cmd, instance_staging_buffer,
                                                     object.instance_data_arr.size());
      cmd.EndDebugPass();
    }

    {
      cmd.BeginDebugPass("voxelize graphic transfer pass");
      VulkanSubPass<SubPassType::Graphic> graphic_pass;
      graphic_pass.ReserveBufferDependencies(vertex_buffer_arr.size());
      for (u32 i = 0; i < vertex_buffer_arr.size(); i++) {
        graphic_pass.AddDependency<DeviceResourceType::Buffer>(*vertex_buffer_arr[i]);
      }

      graphic_pass.ReserveBufferDependencies(index_buffer_arr.size());
      for (u32 i = 0; i < index_buffer_arr.size(); i++) {
        graphic_pass.AddDependency<DeviceResourceType::Buffer>(*index_buffer_arr[i]);
      }

      graphic_pass.ReserveImageDependencies(albedo_image_arr.size());
      for (u32 i = 0; i < albedo_image_arr.size(); i++) {
        if (!object.material_data_arr[i].initialized)
          continue;

        graphic_pass.AddDependency<DeviceResourceType::SampledImage>(*albedo_image_arr[i]);
      }

      graphic_pass.AddDependency<DeviceResourceType::Buffer>(mesh_buffer);
      graphic_pass.AddDependency<DeviceResourceType::Buffer>(instance_buffer);

      cmd.BindSubPass(graphic_pass);
      cmd.EndDebugPass();
    }
  });

  VoxelizeMeshes(object.instance_data_arr, mesh_arr, max_depth);
}
}; // namespace Core
