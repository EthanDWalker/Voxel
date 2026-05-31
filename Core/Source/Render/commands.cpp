#include "Core/Render/commands.h"
#include "Core/Render/Vulkan/other_buffer.h"
#include "Core/Render/Vulkan/util.h"
#include "Core/Render/context.h"

namespace Core {
void PushEditInstanceCommand(const u32 instance_index, const Mat4f32 &matrix) {
  VulkanBuffer<BufferType::StagingBuffer> instance_staging_buffer = "instance staging buffer";
  instance_staging_buffer.Create(sizeof(Instance));
  Instance &instance = ((Instance *)instance_staging_buffer.host_address)[0];
  instance.instanceCustomIndex = 1;
  instance.instanceShaderBindingTableRecordOffset = 0;
  instance.accelerationStructureReference =
      GetDeviceAddress(render_context->bottom_level_acceleration_structure_arr[1]->obj);
  instance.flags = 0;
  instance.mask = 0xFF;
  instance.transform = Mat4ToVkTransform(matrix);

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    {
      cmd.BeginDebugPass("instance transfer pass");
      VulkanSubPass<SubPassType::Transfer> pass;
      pass.AddDependency<DeviceResourceType::TransferSrc>(instance_staging_buffer);
      pass.AddDependency<DeviceResourceType::TransferDst>(render_context->instance_counted_buffer);

      cmd.BindSubPass(pass);

      cmd.UploadBufferToBuffer(
          instance_staging_buffer, render_context->instance_counted_buffer, sizeof(Instance), 0,
          sizeof(VulkanBuffer<BufferType::CountedBuffer>::Header) + instance_index * sizeof(Instance));
    }
  });

  render_context->top_level_acceleration_structure.Recreate(render_context->instance_counted_buffer);
  render_context->mesh_descriptor.Update<DeviceResourceType::AccelerationStructure>(0, &render_context->top_level_acceleration_structure);
}
} // namespace Core
