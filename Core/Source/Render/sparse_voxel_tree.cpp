#include "Core/Render/sparse_voxel_tree.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/sampler.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/types.h"
#include "Core/Util/Parse/gltf.h"
#include "Core/Util/log.h"

namespace Core {
struct alignas(GPU_ALIGNMENT) TreeHeader {
  u32 voxel_count;
  u32 allocated_page_count;
  u32 _page_size;
  f32 _min_bound;
  f32 _max_bound;
  u32 _max_voxelize_depth;
};

struct VoxelizePushConstants {
  u32 triangle_count;
};

SparseVoxelTree::SparseVoxelTree() {
  constexpr bool host = true;

  sampler.Create(SamplerFilter::Nearest, SamplerFilter::Nearest);

  tree_header_buffer.Create(sizeof(TreeHeader), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  tree_header_host_buffer.Create(sizeof(TreeHeader),
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, host);

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>();       // index buffer (not in use atm)
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>();       // vertex buffer
  DescriptorBuilder::Bind<DeviceResourceType::SampledImage>(); // albedo image
  DescriptorBuilder::Bind<DeviceResourceType::Sampler>(&sampler);
  DescriptorBuilder::Build(VK_SHADER_STAGE_COMPUTE_BIT, &voxelize_descriptor);

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, MAX_PAGES);
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&tree_header_buffer);
  DescriptorBuilder::Build(VK_SHADER_STAGE_COMPUTE_BIT, &tree_descriptor);

  for (u32 i = 0; i < 1; i++) {
    // add initial page
    pages.emplace_back(std::make_unique<VulkanBuffer>())
        ->Create(sizeof(Voxel) * PAGE_SIZE,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    tree_descriptor.Update<DeviceResourceType::Buffer>(0, pages.back().get(), pages.size() - 1);
  }

  TreeHeader header{};
  header.voxel_count = 8; // allocated top voxel split
  header.allocated_page_count = pages.size();
  header._page_size = PAGE_SIZE;
  header._min_bound = MIN_BOUND;
  header._max_bound = MAX_BOUND;
  header._max_voxelize_depth = MAX_VOXLELIZE_DEPTH;

  memcpy(tree_header_host_buffer.address, &header, sizeof(TreeHeader));

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    VulkanSubPass<SubPassType::Transfer> transfer_pass;
    transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(tree_header_host_buffer);
    transfer_pass.AddDependency<DeviceResourceType::TransferDst>(tree_header_buffer);
    for (u32 i = 0; i < pages.size(); i++) {
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*pages[i]);
    }

    cmd.BindSubPass(transfer_pass);

    cmd.UploadBufferToBuffer(tree_header_host_buffer, tree_header_buffer, sizeof(TreeHeader));
    for (u32 i = 0; i < pages.size(); i++) {
      cmd.FillBuffer(*pages[i], pages[i]->size, 0);
    }
  });

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptor(voxelize_descriptor);
    pipeline_builder.AddDescriptor(tree_descriptor);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "voxelize.slang");
    pipeline_builder.AddPushConstantRange(sizeof(VoxelizePushConstants));
    PipelineBuildManager::Build(pipeline_builder, voxelize_pipeline);
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

  const f32 box_size = MAX_BOUND - MIN_BOUND;
  const f32 grid_n = 1 << MAX_VOXLELIZE_DEPTH;
  const f32 voxel_size = box_size / grid_n;

  std::vector<Vertex> split_vertices;
  split_vertices.reserve(mesh_data.vertex_arr.size());

  {
    for (u32 i = 0; i < mesh_data.index_arr.size(); i += 3) {
      SubdivideTriangle(split_threashold,
                        {
                            mesh_data.vertex_arr[mesh_data.index_arr[i + 0]],
                            mesh_data.vertex_arr[mesh_data.index_arr[i + 1]],
                            mesh_data.vertex_arr[mesh_data.index_arr[i + 2]],
                        },
                        split_vertices);
    }
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

  // voxelize_descriptor.Update(0, index_buffer);
  voxelize_descriptor.Update<DeviceResourceType::Buffer>(1, &vertex_buffer);
  voxelize_descriptor.Update<DeviceResourceType::SampledImage>(2, &albedo_image);

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    {
      VulkanSubPass<SubPassType::Transfer> transfer_pass;
      transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(staging_buffer);
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(vertex_buffer);
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(albedo_image);

      cmd.BindSubPass(transfer_pass);

      u64 offset = 0;

      memcpy((char *)staging_buffer.address + offset, split_vertices.data(), vertex_buffer.size);
      cmd.UploadBufferToBuffer(staging_buffer, vertex_buffer, vertex_buffer.size, offset);
      offset += vertex_buffer.size;

      memcpy((char *)staging_buffer.address + offset, mesh_data.material.albedo_data, image_data_size);
      cmd.UploadBufferToImage(staging_buffer, albedo_image, offset);
      offset += image_data_size;
    }

    {
      VulkanSubPass<SubPassType::Compute> voxelize_pass;
      voxelize_pass.AddDependency<DeviceResourceType::Buffer>(vertex_buffer);
      voxelize_pass.AddDependency<DeviceResourceType::RWBuffer>(tree_header_buffer);
      voxelize_pass.AddDependency<DeviceResourceType::SampledImage>(albedo_image);
      for (u32 i = 0; i < pages.size(); i++) {
        voxelize_pass.AddDependency<DeviceResourceType::RWBuffer>(*pages[i]);
      }

      cmd.BindSubPass(voxelize_pass);

      VoxelizePushConstants pc;
      pc.triangle_count = split_vertices.size() / 3;

      cmd.BindPipeline(voxelize_pipeline);
      cmd.BindDescriptors({voxelize_descriptor, tree_descriptor});
      cmd.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(pc), &pc);
      cmd.Dispatch(Vec3u32((pc.triangle_count + 63) / 64, 1, 1));
    }

    {
      VulkanSubPass<SubPassType::Transfer> transfer_pass;
      transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(tree_header_buffer);
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(tree_header_host_buffer);

      cmd.BindSubPass(transfer_pass);

      cmd.UploadBufferToBuffer(tree_header_buffer, tree_header_host_buffer, tree_header_host_buffer.size);
    }
  });

  TreeHeader *header = ((TreeHeader *)tree_header_host_buffer.address);

  while (header->voxel_count / PAGE_SIZE >= pages.size()) {
    const u32 page_count = pages.size();
    for (u32 i = 0; i < (header->voxel_count / PAGE_SIZE) - (page_count - 1); i++) {
      pages.emplace_back(std::make_unique<VulkanBuffer>())
          ->Create(sizeof(Voxel) * PAGE_SIZE,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
      tree_descriptor.Update<DeviceResourceType::Buffer>(0, pages.back().get(), pages.size() - 1);
    }

    header->allocated_page_count = pages.size();

    VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
      {
        VulkanSubPass<SubPassType::Transfer> transfer_pass;
        transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(tree_header_host_buffer);
        transfer_pass.AddDependency<DeviceResourceType::TransferDst>(tree_header_buffer);
        for (u32 i = page_count; i < pages.size(); i++) {
          transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*pages[i]);
        }

        cmd.BindSubPass(transfer_pass);

        cmd.UploadBufferToBuffer(tree_header_host_buffer, tree_header_buffer, tree_header_buffer.size);
        for (u32 i = page_count; i < pages.size(); i++) {
          cmd.FillBuffer(*pages[i], pages[i]->size, 0);
        }
      }

      {
        VulkanSubPass<SubPassType::Compute> voxelize_pass;
        voxelize_pass.AddDependency<DeviceResourceType::Buffer>(vertex_buffer);
        voxelize_pass.AddDependency<DeviceResourceType::RWBuffer>(tree_header_buffer);
        voxelize_pass.AddDependency<DeviceResourceType::SampledImage>(albedo_image);
        for (u32 i = 0; i < pages.size(); i++) {
          voxelize_pass.AddDependency<DeviceResourceType::RWBuffer>(*pages[i]);
        }

        cmd.BindSubPass(voxelize_pass);

        VoxelizePushConstants pc;
        pc.triangle_count = split_vertices.size() / 3;

        cmd.BindPipeline(voxelize_pipeline);
        cmd.BindDescriptors({voxelize_descriptor, tree_descriptor});
        cmd.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(pc), &pc);
        cmd.Dispatch(Vec3u32((pc.triangle_count + 63) / 64, 1, 1));
      }

      {
        VulkanSubPass<SubPassType::Transfer> transfer_pass;
        transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(tree_header_buffer);
        transfer_pass.AddDependency<DeviceResourceType::TransferDst>(tree_header_host_buffer);

        cmd.BindSubPass(transfer_pass);

        cmd.UploadBufferToBuffer(tree_header_buffer, tree_header_host_buffer, tree_header_host_buffer.size);
      }
    });

    header = ((TreeHeader *)tree_header_host_buffer.address);
  }
}
} // namespace Core
