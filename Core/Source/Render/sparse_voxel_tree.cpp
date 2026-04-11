#include "Core/Render/sparse_voxel_tree.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/sampler.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/types.h"
#include "Core/Util/Parse/gltf.h"
#include <memory>

namespace Core {
struct VoxelizePushConstants {
  u32 count;
  u32 depth;
};

const u32 INDEX_BUFFER_BINDING = 0;
const u32 VERTEX_BUFFER_BINDING = 1;
const u32 ALBEDO_IMAGE_BINDING = 2;

const u32 TREE_BUFFER_BINDING = 0;
const u32 TREE_LEAF_BUFFER_BINDING = 1;

SparseVoxelTree::SparseVoxelTree() {
  constexpr bool host = true;

  sampler.Create(SamplerFilter::Linear, SamplerFilter::Linear);

  tree_header_buffer.Create(sizeof(TreeHeader), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  tree_header_host_buffer.Create(sizeof(TreeHeader),
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, host);

  empty_page_host_buffer.Create(sizeof(Node) * PAGE_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host);

  for (u32 i = 0; i < PAGE_SIZE; i++) {
    ((Node *)empty_page_host_buffer.address)[i] = {.child_mask = 0, .child_ptr = SENTINAL, .color = 0};
  }

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>();          // index buffer (not in use atm)
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>();          // vertex buffer
  DescriptorBuilder::Bind<DeviceResourceType::SampledImage>();    // albedo image
  DescriptorBuilder::Bind<DeviceResourceType::Sampler>(&sampler); // sampler
  DescriptorBuilder::Build(VK_SHADER_STAGE_COMPUTE_BIT, &voxelize_descriptor);

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, MAX_PAGES); // voxel tree
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, MAX_PAGES); // voxel tree leafs
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&tree_header_buffer);
  DescriptorBuilder::Build(VK_SHADER_STAGE_COMPUTE_BIT, &tree_descriptor);

  TreeHeader header{};
  header._min_bound = MIN_BOUND;
  header._max_bound = MAX_BOUND;
  header._max_voxelize_depth = MAX_VOXLELIZE_DEPTH;
  header._page_size_exp = PAGE_SIZE_EXP;
  header.level_voxel_count[0] = 64;

  for (u32 i = 0; i < MAX_VOXLELIZE_DEPTH - 1; i++) {
    const u32 max_level_nodes = (1 << ((i + 1) * 6));
    if (max_level_nodes <= PAGE_SIZE) {
      pages[i]
          .emplace_back(std::make_unique<VulkanBuffer>())
          ->Create(sizeof(Node) * max_level_nodes,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);
    } else {
      pages[i]
          .emplace_back(std::make_unique<VulkanBuffer>())
          ->Create(sizeof(Node) * PAGE_SIZE,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);
    }
    tree_descriptor.Update<DeviceResourceType::Buffer>(TREE_BUFFER_BINDING, pages[i].back().get(), i);

    header.level_page_offset[i] = i;
  }

  leaf_pages.emplace_back(std::make_unique<VulkanBuffer>())
      ->Create(sizeof(u32) * PAGE_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               HOST);
  tree_descriptor.Update<DeviceResourceType::Buffer>(TREE_LEAF_BUFFER_BINDING, leaf_pages.back().get(), 0);

  memcpy(tree_header_host_buffer.address, &header, sizeof(TreeHeader));

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    VulkanSubPass<SubPassType::Transfer> transfer_pass;
    transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(tree_header_host_buffer);
    transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(empty_page_host_buffer);
    transfer_pass.AddDependency<DeviceResourceType::TransferDst>(tree_header_buffer);
    for (u32 i = 0; i < pages.size(); i++) {
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*pages[i][0]);
    }
    for (u32 i = 0; i < leaf_pages.size(); i++) {
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*leaf_pages[i]);
    }

    cmd.BindSubPass(transfer_pass);

    cmd.UploadBufferToBuffer(tree_header_host_buffer, tree_header_buffer, sizeof(TreeHeader));
    for (u32 i = 0; i < pages.size(); i++) {
      cmd.UploadBufferToBuffer(empty_page_host_buffer, *pages[i][0], pages[i][0]->size);
    }
    for (u32 i = 0; i < leaf_pages.size(); i++) {
      cmd.FillBuffer(*leaf_pages[i], leaf_pages[i]->size, 0);
    }
  });

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptor(voxelize_descriptor);
    pipeline_builder.AddDescriptor(tree_descriptor);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "allocate.slang");
    pipeline_builder.AddPushConstantRange(sizeof(VoxelizePushConstants));
    PipelineBuildManager::Build(pipeline_builder, allocate_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptor(voxelize_descriptor);
    pipeline_builder.AddDescriptor(tree_descriptor);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "allocate_child_mask.slang");
    pipeline_builder.AddPushConstantRange(sizeof(VoxelizePushConstants));
    PipelineBuildManager::Build(pipeline_builder, allocate_child_mask_pipeline);
  }

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptor(tree_descriptor);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "calculate_radiance.slang");
    pipeline_builder.AddPushConstantRange(sizeof(VoxelizePushConstants));
    PipelineBuildManager::Build(pipeline_builder, calculate_radiance_pipeline);
  }
}

void SubdivideTriangle(const f32 side_length_threashold, const std::array<Vertex, 3> triangle,
                       std::vector<Vertex> &vertices) {
  const f32 side_length_threashold_2 = side_length_threashold * side_length_threashold;

  // see if we can subdivide further
  for (u32 i = 0; i < 3; i++) {
    const Vec3f32 edge = triangle[(i + 1) % 3].position - triangle[i].position;
    const f32 edge_length_2 = Dot(edge, edge);

    if (edge_length_2 >= side_length_threashold_2) {
      const Vertex midpoint = {.position = edge * 0.5f + triangle[i].position,
                               .uv = (triangle[(i + 1) % 3].uv + triangle[i].uv) * 0.5f};

      SubdivideTriangle(side_length_threashold,
                        {
                            triangle[i],
                            midpoint,
                            triangle[(i + 2) % 3],
                        },
                        vertices);

      SubdivideTriangle(side_length_threashold,
                        {
                            midpoint,
                            triangle[(i + 1) % 3],
                            triangle[(i + 2) % 3],
                        },
                        vertices);

      return;
    }
  }

  // if we cant subdivide further, add to arrays

  const u32 init_index = vertices.size();

  vertices.push_back(triangle[0]);
  vertices.push_back(triangle[1]);
  vertices.push_back(triangle[2]);
}

void SparseVoxelTree::VoxelizeMesh(const MeshData &mesh_data) {
  // first, if some triangles are very large we split them into smaller ones
  const f32 split_threashold = 700.0f;

  TreeHeader *header = (TreeHeader *)tree_header_host_buffer.address;

  std::vector<Vertex> split_vertices;
  split_vertices.reserve(mesh_data.vertex_arr.size());

  for (u32 i = 0; i < mesh_data.index_arr.size(); i += 3) {
    SubdivideTriangle(split_threashold,
                      {
                          mesh_data.vertex_arr[mesh_data.index_arr[i + 0]],
                          mesh_data.vertex_arr[mesh_data.index_arr[i + 1]],
                          mesh_data.vertex_arr[mesh_data.index_arr[i + 2]],
                      },
                      split_vertices);
  }

  VulkanBuffer vertex_buffer;
  vertex_buffer.Create(sizeof(Vertex) * split_vertices.size(),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  constexpr bool host = true;

  VulkanImage<ImageType::Planar> albedo_image;
  albedo_image.Create(mesh_data.material.albedo_extent, VK_FORMAT_R8G8B8A8_UNORM,
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

  const u64 image_data_size =
      mesh_data.material.albedo_extent.width * mesh_data.material.albedo_extent.height * 4;

  VulkanBuffer staging_buffer;
  staging_buffer.Create(vertex_buffer.size + image_data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host);

  voxelize_descriptor.Update<DeviceResourceType::Buffer>(VERTEX_BUFFER_BINDING, &vertex_buffer);
  voxelize_descriptor.Update<DeviceResourceType::SampledImage>(ALBEDO_IMAGE_BINDING, &albedo_image);

  {
    for (u32 depth = 1; depth < MAX_VOXLELIZE_DEPTH; depth++) {
      TreeHeader *header = ((TreeHeader *)tree_header_host_buffer.address);

      const u32 page_offset = pages[depth - 1].size();
      const u32 new_page_offset = header->level_voxel_count[depth - 1] >> PAGE_SIZE_EXP;

      if (new_page_offset >= page_offset) {
        for (u32 i = page_offset; i <= new_page_offset; i++) {
          pages[depth - 1]
              .emplace_back(std::make_unique<VulkanBuffer>())
              ->Create(sizeof(Node) * PAGE_SIZE,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);

          tree_descriptor.Update<DeviceResourceType::Buffer>(
              TREE_BUFFER_BINDING, pages[depth - 1].back().get(), header->level_page_offset[depth - 1] + i);
        }

        u32 index = header->level_page_offset[depth - 1] + pages[depth - 1].size();
        for (u32 i = depth; i < MAX_VOXLELIZE_DEPTH - 1; i++) {
          header->level_page_offset[i] = index;
          for (u32 j = 0; j < pages[i].size(); j++) {
            tree_descriptor.Update<DeviceResourceType::Buffer>(TREE_BUFFER_BINDING, pages[i][j].get(), index);
            index++;
          }
        }
      }

      VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
        {
          VulkanSubPass<SubPassType::Transfer> transfer_pass;
          transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(staging_buffer);
          transfer_pass.AddDependency<DeviceResourceType::TransferDst>(vertex_buffer);
          transfer_pass.AddDependency<DeviceResourceType::TransferDst>(albedo_image);
          transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(empty_page_host_buffer);

          transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(tree_header_host_buffer);
          transfer_pass.AddDependency<DeviceResourceType::TransferDst>(tree_header_buffer);

          for (u32 i = page_offset; i <= new_page_offset; i++) {
            transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*pages[depth - 1][i]);
          }

          cmd.BindSubPass(transfer_pass);

          u64 offset = 0;

          if (depth == 1) {
            memcpy((char *)staging_buffer.address + offset, split_vertices.data(), vertex_buffer.size);
            cmd.UploadBufferToBuffer(staging_buffer, vertex_buffer, vertex_buffer.size, offset);
            offset += vertex_buffer.size;

            memcpy((char *)staging_buffer.address + offset, mesh_data.material.albedo_data, image_data_size);
            cmd.UploadBufferToImage(staging_buffer, albedo_image, offset);
            offset += image_data_size;
          }

          for (u32 i = page_offset; i <= new_page_offset; i++) {
            cmd.UploadBufferToBuffer(empty_page_host_buffer, *pages[depth - 1][i], pages[depth - 1][i]->size);
          }

          cmd.UploadBufferToBuffer(tree_header_host_buffer, tree_header_buffer, tree_header_buffer.size);
        }

        {
          VulkanSubPass<SubPassType::Compute> allocate_pass;
          allocate_pass.AddDependency<DeviceResourceType::Buffer>(vertex_buffer);
          allocate_pass.AddDependency<DeviceResourceType::RWBuffer>(tree_header_buffer);
          for (u32 i = 0; i < pages.size(); i++) {
            for (u32 j = 0; j < pages[i].size(); j++) {
              allocate_pass.AddDependency<DeviceResourceType::RWBuffer>(*pages[i][j]);
            }
          }

          cmd.BindSubPass(allocate_pass);

          VoxelizePushConstants pc;
          pc.count = split_vertices.size() / 3;
          pc.depth = depth;

          cmd.BindPipeline(allocate_pipeline);
          cmd.BindDescriptors({voxelize_descriptor, tree_descriptor});
          cmd.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(pc), &pc);
          cmd.Dispatch(Vec3u32((pc.count + 63) / 64, 1, 1));
        }

        {
          VulkanSubPass<SubPassType::Transfer> transfer_pass;
          transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(tree_header_buffer);
          transfer_pass.AddDependency<DeviceResourceType::TransferDst>(tree_header_host_buffer);

          cmd.BindSubPass(transfer_pass);

          cmd.UploadBufferToBuffer(tree_header_buffer, tree_header_host_buffer, tree_header_host_buffer.size);
        }
      });
    }
  }

  const u32 page_offset = leaf_pages.size();
  const u32 new_page_offset = header->leaf_voxel_count >> PAGE_SIZE_EXP;

  for (u32 i = page_offset; i <= new_page_offset; i++) {
    leaf_pages.emplace_back(std::make_unique<VulkanBuffer>())
        ->Create(sizeof(u32) * PAGE_SIZE,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);

    tree_descriptor.Update<DeviceResourceType::Buffer>(TREE_LEAF_BUFFER_BINDING, leaf_pages.back().get(), i);
  }

  SCOPED_TIMER("child mask pass")

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    {
      VulkanSubPass<SubPassType::Transfer> transfer_pass;
      for (u32 i = page_offset; i <= new_page_offset; i++) {
        transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*leaf_pages[i]);
      }

      cmd.BindSubPass(transfer_pass);

      for (u32 i = page_offset; i <= new_page_offset; i++) {
        cmd.FillBuffer(*leaf_pages[i], leaf_pages[i]->size, 0);
      }
    }
    {
      VulkanSubPass<SubPassType::Compute> child_mask_pass;
      child_mask_pass.AddDependency<DeviceResourceType::Buffer>(vertex_buffer);
      child_mask_pass.AddDependency<DeviceResourceType::RWBuffer>(tree_header_buffer);
      for (u32 i = 0; i < pages.size(); i++) {
        for (u32 j = 0; j < pages[i].size(); j++) {
          child_mask_pass.AddDependency<DeviceResourceType::RWBuffer>(*pages[i][j]);
        }
      }
      for (u32 i = 0; i < leaf_pages.size(); i++) {
        child_mask_pass.AddDependency<DeviceResourceType::RWBuffer>(*leaf_pages[i]);
      }

      cmd.BindSubPass(child_mask_pass);

      VoxelizePushConstants pc;
      pc.count = split_vertices.size() / 3;
      pc.depth = MAX_VOXLELIZE_DEPTH - 1;

      cmd.BindPipeline(allocate_child_mask_pipeline);
      cmd.BindDescriptors({voxelize_descriptor, tree_descriptor});
      cmd.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(pc), &pc);
      cmd.Dispatch(Vec3u32((pc.count + 63) / 64, 1, 1));
    }
  });

  for (i32 i = MAX_VOXLELIZE_DEPTH - 2; i >= 0; i--) {
    VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
      {
        VulkanSubPass<SubPassType::Compute> radiance_pass;
        radiance_pass.AddDependency<DeviceResourceType::RWBuffer>(tree_header_buffer);
        for (u32 i = 0; i < pages.size(); i++) {
          for (u32 j = 0; j < pages[i].size(); j++) {
            radiance_pass.AddDependency<DeviceResourceType::RWBuffer>(*pages[i][j]);
          }
        }
        for (u32 i = 0; i < leaf_pages.size(); i++) {
          radiance_pass.AddDependency<DeviceResourceType::RWBuffer>(*leaf_pages[i]);
        }

        cmd.BindSubPass(radiance_pass);

        VoxelizePushConstants pc;
        pc.count = header->level_voxel_count[i];
        pc.depth = i;

        cmd.BindPipeline(calculate_radiance_pipeline);
        cmd.BindDescriptors({tree_descriptor});
        cmd.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(pc), &pc);
        cmd.Dispatch(Vec3u32((pc.count + 63) / 64, 1, 1));
      }
    });
  }
}
} // namespace Core
