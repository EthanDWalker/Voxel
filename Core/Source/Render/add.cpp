#include "Core/Render/add.h"
#include "Core/Render/Vulkan/acceleration_structure.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/image.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/context.h"
#include "Core/Render/sparse_voxel_tree.h"
#include "Core/Render/types.h"
#include "Core/Util/Parse/gltf.h"
#include <cstring>
#include <memory>

namespace Core {
u32 AddDirectionalLight(const DirectionalLight &dir_light) {
  ZoneScoped;
  VulkanBuffer staging_buffer = "directional light staging buffer";
  staging_buffer.Create(sizeof(DirectionalLight), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, /*host=*/true);
  memcpy(staging_buffer.host_address, &dir_light, sizeof(DirectionalLight));

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    cmd.UploadBufferToBuffer(
        staging_buffer, render_context->directional_light_buffer, sizeof(DirectionalLight), 0,
        sizeof(DirectionalLight) * render_context->directional_light_count + sizeof(u32));

    cmd.FillBuffer(render_context->directional_light_buffer, sizeof(u32),
                   render_context->directional_light_count + 1);
  });

  return render_context->directional_light_count++;
}

void VoxelizeMesh(const Mesh &mesh, const u32 max_depth) {
  AllocateInfo alloc_info;
  alloc_info.mesh_id = mesh.id;

  SparseVoxelTree::TreeHeader *header =
      (SparseVoxelTree::TreeHeader *)render_context->voxel_tree.tree_header_host_buffer.host_address;

  for (u32 depth = 1; depth < max_depth; depth++) {
    const u32 page_offset = render_context->voxel_tree.branch_pages.size();
    const u32 new_page_offset = header->branch_count >> SparseVoxelTree::PAGE_SIZE_EXP;

    render_context->voxel_tree.AllocateBranchPages(new_page_offset - page_offset);

    VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
      {
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
      }

      {
        VulkanSubPass<SubPassType::Graphic> allocate_pass;
        allocate_pass.AddDependency<DeviceResourceType::Buffer>(*render_context->vertex_buffers[mesh.id]);
        allocate_pass.AddDependency<DeviceResourceType::Buffer>(*render_context->index_buffers[mesh.id]);
        allocate_pass.AddDependency<DeviceResourceType::RWBuffer>(
            render_context->voxel_tree.tree_header_buffer);
        allocate_pass.AddDependency<DeviceResourceType::SampledImage>(
            *render_context->albedo_images[mesh.id]);

        allocate_pass.ReserveBufferDependencies(render_context->voxel_tree.branch_pages.size());
        for (u32 i = 0; i < render_context->voxel_tree.branch_pages.size(); i++) {
          allocate_pass.AddDependency<DeviceResourceType::RWBuffer>(
              *render_context->voxel_tree.branch_pages[i]);
        }

        cmd.BindSubPass(allocate_pass);

        alloc_info.depth = depth;
        alloc_info.leaf = (depth == (max_depth - 1));
        cmd.BeginRendering({}, nullptr, Vec2u32(1 << (max_depth * 2)));
        cmd.BindPipeline(render_context->allocate_pipeline);
        cmd.BindDescriptors({render_context->mesh_descriptor, render_context->voxel_tree.descriptor});
        cmd.PushConstants(VK_SHADER_STAGE_ALL_GRAPHICS, sizeof(AllocateInfo), &alloc_info);
        cmd.Draw(mesh.index_count / 3);
        cmd.EndRendering();
      }

      {
        VulkanSubPass<SubPassType::Transfer> transfer_pass;
        transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(
            render_context->voxel_tree.tree_header_buffer);
        transfer_pass.AddDependency<DeviceResourceType::TransferDst>(
            render_context->voxel_tree.tree_header_host_buffer);

        cmd.BindSubPass(transfer_pass);

        cmd.UploadBufferToBuffer(render_context->voxel_tree.tree_header_buffer,
                                 render_context->voxel_tree.tree_header_host_buffer,
                                 render_context->voxel_tree.tree_header_host_buffer.size);
      }
    });
  }

  const u32 page_offset = render_context->voxel_tree.leaf_pages.size();
  const u32 new_page_offset = header->leaf_count >> SparseVoxelTree::PAGE_SIZE_EXP;

  render_context->voxel_tree.AllocateLeafPages(new_page_offset - page_offset);

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    {
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
    }
    {
      VulkanSubPass<SubPassType::Graphic> child_mask_pass;
      child_mask_pass.AddDependency<DeviceResourceType::Buffer>(*render_context->vertex_buffers[mesh.id]);
      child_mask_pass.AddDependency<DeviceResourceType::Buffer>(*render_context->index_buffers[mesh.id]);
      child_mask_pass.AddDependency<DeviceResourceType::RWBuffer>(
          render_context->voxel_tree.tree_header_buffer);
      child_mask_pass.AddDependency<DeviceResourceType::SampledImage>(
          *render_context->albedo_images[mesh.id]);

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

      cmd.BeginRendering({}, nullptr, Vec2u32(1 << (max_depth * 2)));
      cmd.BindPipeline(render_context->allocate_child_mask_pipeline);
      cmd.BindDescriptors({render_context->mesh_descriptor, render_context->voxel_tree.descriptor});
      cmd.PushConstants(VK_SHADER_STAGE_ALL_GRAPHICS, sizeof(AllocateInfo), &alloc_info);
      cmd.Draw(mesh.index_count / 3);
      cmd.EndRendering();
    }

    {
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
    }
  });
}

Mesh AddMesh(const MeshData &mesh_data) {
  ZoneScoped;

  const u32 max_depth = SparseVoxelTree::MAX_DEPTH - 2;

  VkPhysicalDeviceAccelerationStructurePropertiesKHR properties{};
  properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;

  VkPhysicalDeviceProperties2 device_properties{};
  device_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  device_properties.pNext = &properties;

  vkGetPhysicalDeviceProperties2(VulkanContext::physical_device, &device_properties);

  VulkanBuffer &vertex_buffer =
      *render_context->vertex_buffers.emplace_back(std::make_unique<VulkanBuffer>("vertex buffer"));
  VulkanBuffer &index_buffer =
      *render_context->index_buffers.emplace_back(std::make_unique<VulkanBuffer>("index buffer"));
  VulkanBuffer &triangle_voxelized_depth_buffer =
      *render_context->triangle_voxelized_depth_buffers.emplace_back(
          std::make_unique<VulkanBuffer>("triangle voxelized depth buffer"));

  vertex_buffer.Create(mesh_data.vertex_host_buffer->size,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  index_buffer.Create(mesh_data.index_host_buffer->size,
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  const u32 triangle_count = mesh_data.index_host_buffer->size / (sizeof(Index) * 3);
  triangle_voxelized_depth_buffer.Create(triangle_count * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  VulkanImage<ImageType::Planar> &albedo_image =
      *render_context->albedo_images.emplace_back(std::make_unique<VulkanImage<ImageType::Planar>>());
  albedo_image.Create(mesh_data.material.albedo_extent, VK_FORMAT_R8G8B8A8_UNORM,
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

  const u64 albedo_image_data_size =
      mesh_data.material.albedo_extent.width * mesh_data.material.albedo_extent.height * 4;
  VulkanBuffer albedo_staging_buffer = "image data staging buffer";
  constexpr bool host = true;
  albedo_staging_buffer.Create(albedo_image_data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host);
  memcpy(albedo_staging_buffer.host_address, mesh_data.material.albedo_data, albedo_image_data_size);

  const u32 mesh_index = render_context->mesh_count;
  render_context->mesh_count++;

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    VulkanSubPass<SubPassType::Transfer> transfer_pass;

    transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(*mesh_data.vertex_host_buffer);
    transfer_pass.AddDependency<DeviceResourceType::TransferDst>(vertex_buffer);

    transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(*mesh_data.index_host_buffer);
    transfer_pass.AddDependency<DeviceResourceType::TransferDst>(index_buffer);
    transfer_pass.AddDependency<DeviceResourceType::TransferDst>(triangle_voxelized_depth_buffer);

    transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(albedo_staging_buffer);
    transfer_pass.AddDependency<DeviceResourceType::TransferDst>(albedo_image);

    cmd.BindSubPass(transfer_pass);

    cmd.UploadBufferToBuffer(*mesh_data.vertex_host_buffer, vertex_buffer,
                             mesh_data.vertex_host_buffer->size);
    cmd.UploadBufferToBuffer(*mesh_data.index_host_buffer, index_buffer, mesh_data.index_host_buffer->size);
    cmd.UploadBufferToImage(albedo_staging_buffer, albedo_image);
    cmd.FillBuffer(render_context->mesh_triangle_offset_buffer, sizeof(u32), render_context->total_triangles,
                   sizeof(u32) * mesh_index);
    cmd.FillBuffer(triangle_voxelized_depth_buffer, triangle_voxelized_depth_buffer.size, 0);
  });

  render_context->bottom_level_acceleration_structures
      .emplace_back(std::make_unique<VulkanAccelerationStructure>())
      ->CreateBottomLevel(vertex_buffer, index_buffer);

  render_context->total_triangles += triangle_count;

  Mesh mesh;
  mesh.id = mesh_index;
  mesh.vertex_count = mesh_data.vertex_count;
  mesh.index_count = mesh_data.index_count;

  render_context->mesh_descriptor.Update<DeviceResourceType::Buffer>(1, &vertex_buffer, mesh_index);
  render_context->mesh_descriptor.Update<DeviceResourceType::Buffer>(2, &index_buffer, mesh_index);
  render_context->mesh_descriptor.Update<DeviceResourceType::Buffer>(3, &triangle_voxelized_depth_buffer,
                                                                     mesh_index);
  render_context->mesh_descriptor.Update<DeviceResourceType::SampledImage>(4, &albedo_image, mesh_index);

  VoxelizeMesh(mesh, max_depth);

  return mesh;
}
}; // namespace Core
