#include "Core/Render/Vulkan/indirect_draw.h"
#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/types.h"

namespace Core {
void VulkanIndirectDrawCommand::Create(const u32 max_draw_count) {
  ZoneScoped;
  draw_buffer.Create(sizeof(VkDrawIndexedIndirectCommand) * max_draw_count,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);

  draw_count_buffer.Create(sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&draw_buffer);
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(&draw_count_buffer);
  DescriptorBuilder::Build(VK_SHADER_STAGE_COMPUTE_BIT, &descriptor);

  this->max_draw_count = max_draw_count;
}
} // namespace Core
