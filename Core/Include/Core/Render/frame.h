#pragma once

#include "Core/Render/Vulkan/command_buffer.h"
#include "camera.h"

namespace Core {
void Frame(Camera &camera);
VulkanCommandBuffer &BeginFrame(bool &resize);
void EndFrame(bool &resize);
void WaitIdle();
void Resize(Vec2u32 extent);
} // namespace Core
