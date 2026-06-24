#include "Core/Render/device_hash_set.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/Vulkan/swapchain.h"
#include "Core/Render/types.h"

namespace Core {
void DeviceHashSet::Recreate(const u32 size, const VkShaderStageFlags shader_stages) {
  this->size = size;

  DeviceHashSetHeader header{};
  header.size = size;

  for (u32 i = 0; i < swapped_data.size(); i++) {
    swapped_data[i].key_buffer.Destroy();
    swapped_data[i].key_buffer.Create(size,
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    swapped_data[i].diffuse_data_buffer.Destroy();
    swapped_data[i].diffuse_data_buffer.Create(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    swapped_data[i].specular_data_buffer.Destroy();
    swapped_data[i].specular_data_buffer.Create(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  }

  for (u32 i = 0; i < swapped_data.size(); i++) {
    swapped_data[i].descriptor.Update<DeviceResourceType::Buffer>(KEY_BINDING, &swapped_data[i].key_buffer);
    swapped_data[i].descriptor.Update<DeviceResourceType::Buffer>(
        BACK_KEY_BINDING, &swapped_data[LastIndex(i, VulkanSwapchain::FRAME_OVERLAP)].key_buffer);

    swapped_data[i].descriptor.Update<DeviceResourceType::Buffer>(DIFFUSE_DATA_BINDING,
                                                                  &swapped_data[i].diffuse_data_buffer);
    swapped_data[i].descriptor.Update<DeviceResourceType::Buffer>(
        BACK_DIFFUSE_DATA_BINDING,
        &swapped_data[LastIndex(i, VulkanSwapchain::FRAME_OVERLAP)].diffuse_data_buffer);

    swapped_data[i].descriptor.Update<DeviceResourceType::Buffer>(SPECULAR_DATA_BINDING,
                                                                  &swapped_data[i].specular_data_buffer);
    swapped_data[i].descriptor.Update<DeviceResourceType::Buffer>(
        BACK_SPECULAR_DATA_BINDING,
        &swapped_data[LastIndex(i, VulkanSwapchain::FRAME_OVERLAP)].specular_data_buffer);
  }

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    cmd.BeginDebugPass("device hash set reinit");
    VulkanSubPass<SubPassType::Transfer> copy_pass;

    for (u32 i = 0; i < swapped_data.size(); i++) {
      copy_pass.AddDependency<DeviceResourceType::TransferDst>(swapped_data[i].key_buffer);
      copy_pass.AddDependency<DeviceResourceType::TransferSrc>(swapped_data[i].header_staging_buffer);
      copy_pass.AddDependency<DeviceResourceType::TransferDst>(swapped_data[i].header_buffer);
      copy_pass.AddDependency<DeviceResourceType::TransferDst>(swapped_data[i].diffuse_data_buffer);
      copy_pass.AddDependency<DeviceResourceType::TransferDst>(swapped_data[i].specular_data_buffer);
    }

    cmd.BindSubPass(copy_pass);

    for (u32 i = 0; i < swapped_data.size(); i++) {
      cmd.FillBuffer(swapped_data[i].key_buffer, swapped_data[i].key_buffer.size, EMPTY_KEY);
      cmd.FillBuffer(swapped_data[i].diffuse_data_buffer, swapped_data[i].diffuse_data_buffer.size, 0);
      cmd.FillBuffer(swapped_data[i].specular_data_buffer, swapped_data[i].specular_data_buffer.size, 0);
      cmd.UploadBufferToBuffer(swapped_data[i].header_staging_buffer, swapped_data[i].header_buffer,
                               sizeof(DeviceHashSetHeader));
    }
    cmd.EndDebugPass();
  });
}

void DeviceHashSet::Create(const u32 size, const VkShaderStageFlags shader_stages) {
  this->size = size;

  DeviceHashSetHeader header{};
  header.size = size;

  for (u32 i = 0; i < swapped_data.size(); i++) {
    swapped_data[i].key_buffer.Create(size,
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    swapped_data[i].diffuse_data_buffer.Create(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    swapped_data[i].specular_data_buffer.Create(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    swapped_data[i].header_buffer.Create(1, VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    swapped_data[i].header_staging_buffer.Create(sizeof(DeviceHashSetHeader));
    memcpy(swapped_data[i].header_staging_buffer.host_address, &header, sizeof(DeviceHashSetHeader));
  }

  for (u32 i = 0; i < swapped_data.size(); i++) {
    DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&swapped_data[i].header_buffer);
    DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&swapped_data[i].key_buffer);
    DescriptorBuilder::Bind<DeviceResourceType::Buffer>(
        &swapped_data[LastIndex(i, VulkanSwapchain::FRAME_OVERLAP)].key_buffer);

    DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&swapped_data[i].diffuse_data_buffer);
    DescriptorBuilder::Bind<DeviceResourceType::Buffer>(
        &swapped_data[LastIndex(i, VulkanSwapchain::FRAME_OVERLAP)].diffuse_data_buffer);

    DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&swapped_data[i].specular_data_buffer);
    DescriptorBuilder::Bind<DeviceResourceType::Buffer>(
        &swapped_data[LastIndex(i, VulkanSwapchain::FRAME_OVERLAP)].specular_data_buffer);

    if (i == 0) {
      DescriptorBuilder::BuildLayout(shader_stages, descriptor_layout);
    }

    DescriptorBuilder::BuildSet(shader_stages, descriptor_layout, swapped_data[i].descriptor);
    DescriptorBuilder::Reset();
  }

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    cmd.BeginDebugPass("device hash set init");
    VulkanSubPass<SubPassType::Transfer> copy_pass;

    for (u32 i = 0; i < swapped_data.size(); i++) {
      copy_pass.AddDependency<DeviceResourceType::TransferDst>(swapped_data[i].key_buffer);
      copy_pass.AddDependency<DeviceResourceType::TransferSrc>(swapped_data[i].header_staging_buffer);
      copy_pass.AddDependency<DeviceResourceType::TransferDst>(swapped_data[i].header_buffer);
      copy_pass.AddDependency<DeviceResourceType::TransferDst>(swapped_data[i].specular_data_buffer);
      copy_pass.AddDependency<DeviceResourceType::TransferDst>(swapped_data[i].diffuse_data_buffer);
    }

    cmd.BindSubPass(copy_pass);

    for (u32 i = 0; i < swapped_data.size(); i++) {
      cmd.FillBuffer(swapped_data[i].diffuse_data_buffer, swapped_data[i].diffuse_data_buffer.size, 0);
      cmd.FillBuffer(swapped_data[i].specular_data_buffer, swapped_data[i].specular_data_buffer.size, 0);
      cmd.UploadBufferToBuffer(swapped_data[i].header_staging_buffer, swapped_data[i].header_buffer,
                               sizeof(DeviceHashSetHeader));
      cmd.FillBuffer(swapped_data[i].key_buffer, swapped_data[i].key_buffer.size, EMPTY_KEY);
    }
    cmd.EndDebugPass();
  });
}
} // namespace Core
