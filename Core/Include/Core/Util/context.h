#pragma once

#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/pipeline.h"

namespace Core {
struct UtilContext {
  VulkanPipeline<PipelineType::Compute> bc1_compression_pipeline;
  VulkanDescriptorLayout bc1_compression_descriptor_layout;
  VulkanDescriptor bc1_compression_descriptor;

  UtilContext();
  ~UtilContext();
};

extern std::unique_ptr<UtilContext> util_context;
} // namespace Core
