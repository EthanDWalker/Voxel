#include "Core/Util/context.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/types.h"

namespace Core {
std::unique_ptr<UtilContext> util_context;

UtilContext::UtilContext() {
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr);
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr);
  DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_COMPUTE_BIT, bc1_compression_descriptor_layout);
  DescriptorBuilder::BuildSet(VK_SHADER_STAGE_COMPUTE_BIT, bc1_compression_descriptor_layout,
                              bc1_compression_descriptor);
  DescriptorBuilder::Reset();

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptorLayout(bc1_compression_descriptor_layout);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "bc1.slang");
    pipeline_builder.AddPushConstantRange(sizeof(Vec2u32)); // extent
    PipelineBuildManager::Build(pipeline_builder, bc1_compression_pipeline);
  }
}

UtilContext::~UtilContext() {}
} // namespace Core
