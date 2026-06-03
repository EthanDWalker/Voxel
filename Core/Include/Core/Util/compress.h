#pragma once

#include "Core/Render/Vulkan/other_buffer.h"

namespace Core {
// assumes buffers have R8G8B8A8 data
void CompressBC1(const Vec2u32 extent, VulkanBuffer<BufferType::StagingBuffer> &input_buffer,
                 VulkanBuffer<BufferType::StagingBuffer> &output_buffer);
}; // namespace Core
