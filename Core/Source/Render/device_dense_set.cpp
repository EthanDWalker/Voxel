#include "Core/Render/device_dense_set.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include <cstring>

namespace Core {
DeviceDenseSet::DeviceDenseSet(const u32 size, const VkShaderStageFlags stage_flags) {
  this->size = size;
  header_buffer.Create(sizeof(Header),
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       HOST);
  header_host_buffer.Create(
      sizeof(Header), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, /*host=*/true);
  value_buffer.Create(sizeof(Value) * size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST);
  key_buffer.Create(sizeof(u32) * size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    HOST);

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&header_buffer);
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&value_buffer);
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&key_buffer);
  DescriptorBuilder::Build(stage_flags, &descriptor);

  Header header{};
  header.size = size;
  header.value_count = 0;
  memcpy(header_host_buffer.host_address, &header, sizeof(Header));

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    {
      VulkanSubPass<SubPassType::Transfer> transfer_pass;
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(key_buffer);

      transfer_pass.AddDependency<DeviceResourceType::TransferSrc>(header_host_buffer);
      transfer_pass.AddDependency<DeviceResourceType::TransferDst>(header_buffer);

      cmd.BindSubPass(transfer_pass);

      cmd.FillBuffer(key_buffer, key_buffer.size, EMPTY_KEY);
      cmd.UploadBufferToBuffer(header_host_buffer, header_buffer, header_buffer.size);
    }
  });
}

} // namespace Core
