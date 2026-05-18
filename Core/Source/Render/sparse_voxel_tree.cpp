#include "Core/Render/sparse_voxel_tree.h"
#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/types.h"
#include <memory>

namespace Core {
const u32 VERTEX_BUFFER_BINDING = 0;
const u32 ALBEDO_IMAGE_BINDING = 1;

const u32 TREE_BUFFER_BINDING = 0;
const u32 TREE_LEAF_BUFFER_BINDING = 1;

struct AllocatePushConstants {
  u32 depth;
  u32 leaf;
};

SparseVoxelTree::SparseVoxelTree() {
  ZoneScoped;
  constexpr bool host = true;

  tree_header_buffer.Create(1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  tree_header_host_buffer.BuildAddStagingBinding(sizeof(TreeHeader));
  tree_header_host_buffer.Build();

  empty_page_host_buffer.BuildAddStagingBinding(sizeof(BranchNode) * PAGE_SIZE);
  empty_page_host_buffer.Build();

  for (u32 i = 0; i < PAGE_SIZE; i++) {
    ((BranchNode *)empty_page_host_buffer.host_address)[i] = {
        .child_mask = 0,
        .child_ptr = SENTINAL,
    };
  }

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, MAX_PAGES); // tree
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr, MAX_PAGES); // tree leafs
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&tree_header_buffer);
  DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_ALL_GRAPHICS,
                                 descriptor_layout);
  DescriptorBuilder::BuildSet(VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_ALL_GRAPHICS, descriptor_layout,
                              descriptor);
  DescriptorBuilder::Reset();

  TreeHeader header{};
  header.branch_count = 64;

  branch_pages
      .emplace_back(
          std::make_unique<VulkanBuffer<BufferType::StructuredBuffer, BranchNode>>("branch page buffer"))
      ->Create(1 << PAGE_SIZE_EXP, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  descriptor.Update<DeviceResourceType::Buffer>(TREE_BUFFER_BINDING, branch_pages.back().get(), 0);

  leaf_pages
      .emplace_back(
          std::make_unique<VulkanBuffer<BufferType::StructuredBuffer, LeafNode>>("leaf page buffer"))
      ->Create(1 << PAGE_SIZE_EXP, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  descriptor.Update<DeviceResourceType::Buffer>(TREE_LEAF_BUFFER_BINDING, leaf_pages.back().get(), 0);

  memcpy(tree_header_host_buffer.host_address, &header, sizeof(TreeHeader));

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    VulkanSubPass<SubPassType::Transfer> transfer_pass;
    transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(tree_header_host_buffer);
    transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(empty_page_host_buffer);
    transfer_pass.AddDependency<DeviceResourceType::TransferDst>(tree_header_buffer);
    for (u32 i = 0; i < branch_pages.size(); i++) {
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*branch_pages[i]);
    }
    for (u32 i = 0; i < leaf_pages.size(); i++) {
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(*leaf_pages[i]);
    }

    cmd.BindSubPass(transfer_pass);

    cmd.UploadBufferToBuffer(tree_header_host_buffer, tree_header_buffer, sizeof(TreeHeader));
    for (u32 i = 0; i < branch_pages.size(); i++) {
      cmd.UploadBufferToBuffer(empty_page_host_buffer, *branch_pages[i], branch_pages[i]->size);
    }
    for (u32 i = 0; i < leaf_pages.size(); i++) {
      cmd.FillBuffer(*leaf_pages[i], leaf_pages[i]->size, 0);
    }
  });
}

void SparseVoxelTree::AllocateBranchPages(const u32 count) {
  ZoneScoped;
  const u32 page_offset = branch_pages.size();
  const u32 new_page_offset = page_offset + count;
  for (u32 i = page_offset; i <= new_page_offset; i++) {
    branch_pages
        .emplace_back(
            std::make_unique<VulkanBuffer<BufferType::StructuredBuffer, BranchNode>>("branch page buffer"))
        ->Create(1 << PAGE_SIZE_EXP, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    descriptor.Update<DeviceResourceType::Buffer>(TREE_BUFFER_BINDING, branch_pages.back().get(), i);
  }
}

void SparseVoxelTree::AllocateLeafPages(const u32 count) {
  ZoneScoped;
  const u32 page_offset = leaf_pages.size();
  const u32 new_page_offset = page_offset + count;
  for (u32 i = page_offset; i <= new_page_offset; i++) {
    leaf_pages
        .emplace_back(
            std::make_unique<VulkanBuffer<BufferType::StructuredBuffer, LeafNode>>("leaf page buffer"))
        ->Create(1 << PAGE_SIZE_EXP, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    descriptor.Update<DeviceResourceType::Buffer>(TREE_LEAF_BUFFER_BINDING, leaf_pages.back().get(), i);
  }
}
} // namespace Core
