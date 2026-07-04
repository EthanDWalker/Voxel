#include "Core/Render/sparse_voxel_tree.h"
#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/other_buffer.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/types.h"
#include <memory>

namespace Core {
const u32 TREE_BUFFER_BINDING = 0;
const u32 TREE_LUMINANCE_BUFFER_BINDING = 1;
const u32 TREE_LEAF_BUFFER_BINDING = 2;

SparseVoxelTree::SparseVoxelTree() {
  ZoneScoped;
  constexpr bool host = true;

  tree_header_buffer.Create(1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  tree_header_host_buffer.Create(sizeof(TreeHeader));

  material_buffer.Create(MAX_MATERIALS,
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, MAX_PAGES); // tree
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, MAX_PAGES); // tree luminance
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, MAX_PAGES); // tree leafs
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&tree_header_buffer);
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&material_buffer);
  DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_ALL_GRAPHICS,
                                 descriptor_layout);
  DescriptorBuilder::BuildSet(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_ALL_GRAPHICS, descriptor_layout,
                              descriptor);
  DescriptorBuilder::Reset();

  TreeHeader header{};
  header.branch_count = 64;
  header.allocated_leaf_count = 1;
  header.allocated_branch_count = 1;

  branch_pages
      .emplace_back(
          std::make_unique<VulkanBuffer<BufferType::StructuredBuffer, BranchNode>>("branch page buffer"))
      ->Create(PAGE_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  descriptor.Update<DeviceResourceType::Buffer>(TREE_BUFFER_BINDING, branch_pages.back().get(), 0);

  branch_luminance_pages
      .emplace_back(
          std::make_unique<VulkanBuffer<BufferType::StructuredBuffer, f32>>("branch luminance page buffer"))
      ->Create(PAGE_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  descriptor.Update<DeviceResourceType::Buffer>(TREE_LUMINANCE_BUFFER_BINDING, branch_luminance_pages.back().get(), 0);

  leaf_pages
      .emplace_back(
          std::make_unique<VulkanBuffer<BufferType::StructuredBuffer, LeafNode>>("leaf page buffer"))
      ->Create(PAGE_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  descriptor.Update<DeviceResourceType::Buffer>(TREE_LEAF_BUFFER_BINDING, leaf_pages.back().get(), 0);

  {
    Material torch_material{};
    torch_material.rough = 0.0f;
    torch_material.emissive = 3.0f;
    torch_material.metallic = 0.0f;
    torch_material.reflect = 0.25f;
    torch_material.albedo = Vec4f32(1.0f, 0.7f, 0.0f, 1.0f);
    material_arr.emplace_back(torch_material);
  }

  {
    Material stone_material{};
    stone_material.rough = 0.95f;
    stone_material.emissive = 0.0f;
    stone_material.metallic = 0.0f;
    stone_material.reflect = 0.25f;
    stone_material.albedo = Vec4f32(0.5333333333f, 0.5490196078f, 0.5529411765f, 1.0f);
    material_arr.emplace_back(stone_material);
  }

  VulkanBuffer<BufferType::StagingBuffer> material_staging_buffer = "material staging buffer";
  material_staging_buffer.Create(sizeof(Material) * material_arr.size());
  memcpy(material_staging_buffer.host_address, material_arr.data(), material_arr.size() * sizeof(Material));

  memcpy(tree_header_host_buffer.host_address, &header, sizeof(TreeHeader));

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    cmd.BeginDebugPass("sparse voxel tree init");
    VulkanSubPass<SubPassType::Transfer> transfer_pass;
    transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(tree_header_host_buffer);
    transfer_pass.AddDependency<DeviceResourceType::TransferDst>(tree_header_buffer);
    for (u32 i = 0; i < branch_pages.size(); i++) {
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*branch_pages[i]);
    }
    for (u32 i = 0; i < leaf_pages.size(); i++) {
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*leaf_pages[i]);
    }
    transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(material_staging_buffer);
    transfer_pass.AddDependency<DeviceResourceType::TransferDst>(material_buffer);

    cmd.BindSubPass(transfer_pass);

    cmd.UploadBufferToBuffer(tree_header_host_buffer, tree_header_buffer, sizeof(TreeHeader));
    for (u32 i = 0; i < branch_pages.size(); i++) {
      cmd.FillBuffer(*branch_pages[i], branch_pages[i]->size, 0);
    }
    for (u32 i = 0; i < leaf_pages.size(); i++) {
      cmd.FillBuffer(*leaf_pages[i], leaf_pages[i]->size, 0);
    }
    material_buffer.Append(cmd, material_staging_buffer, material_arr.size());
    cmd.EndDebugPass();
  });
}

void SparseVoxelTree::ResizeBranch(const u32 count) {
  ZoneScoped;
  const u32 new_page_count = count >> PAGE_SIZE_EXP;
  for (u32 i = branch_pages.size(); i <= new_page_count; i++) {
    branch_pages
        .emplace_back(
            std::make_unique<VulkanBuffer<BufferType::StructuredBuffer, BranchNode>>("branch page buffer"))
        ->Create(PAGE_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    branch_luminance_pages
        .emplace_back(
            std::make_unique<VulkanBuffer<BufferType::StructuredBuffer, f32>>("branch luminance page buffer"))
        ->Create(PAGE_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    descriptor.Update<DeviceResourceType::Buffer>(TREE_BUFFER_BINDING, branch_pages.back().get(), i);
    descriptor.Update<DeviceResourceType::Buffer>(TREE_LUMINANCE_BUFFER_BINDING,
                                                  branch_luminance_pages.back().get(), i);
  }
}

void SparseVoxelTree::ResizeLeaf(const u32 count) {
  ZoneScoped;
  const u32 new_page_count = count >> PAGE_SIZE_EXP;
  for (u32 i = leaf_pages.size(); i <= new_page_count; i++) {
    leaf_pages
        .emplace_back(
            std::make_unique<VulkanBuffer<BufferType::StructuredBuffer, LeafNode>>("leaf page buffer"))
        ->Create(PAGE_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    descriptor.Update<DeviceResourceType::Buffer>(TREE_LEAF_BUFFER_BINDING, leaf_pages.back().get(), i);
  }
}
} // namespace Core
