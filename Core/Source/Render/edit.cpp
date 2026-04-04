#include "Core/Render/edit.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/context.h"
#include "Core/Render/types.h"
#include <cstring>

namespace Core {
u32 AddDirectionalLight(const DirectionalLight &dir_light) {
  Assert(All(Normalize(dir_light.direction) == dir_light.direction), "must normalize dir light direction");

  VulkanBuffer staging_buffer;
  staging_buffer.Create(sizeof(DirectionalLight), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, /*host=*/true);
  memcpy(staging_buffer.address, &dir_light, sizeof(DirectionalLight));

  VulkanContext::Submit([&](VulkanCommandBuffer &cmd) {
    cmd.UploadBufferToBuffer(
        staging_buffer, render_context->directional_light_buffer, sizeof(DirectionalLight), 0,
        sizeof(DirectionalLight) * render_context->directional_light_count + sizeof(u32));

    cmd.FillBuffer(render_context->directional_light_buffer, sizeof(u32), render_context->directional_light_count + 1);
  });

  return render_context->directional_light_count++;
}
}; // namespace Core
