#include "Core/Util/context.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/types.h"
#include "Core/Util/types.h"

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
    pipeline_builder.AddPushConstantRange(sizeof(BC1CompressionPushConstants));
    PipelineBuildManager::Build(pipeline_builder, bc1_compression_pipeline);
  }

  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr); // vertex buffer
  DescriptorBuilder::Bind<DeviceResourceType::Buffer>(nullptr); // index buffer
  DescriptorBuilder::BuildLayout(VK_SHADER_STAGE_COMPUTE_BIT, generate_terrain_descriptor_layout);
  DescriptorBuilder::BuildSet(VK_SHADER_STAGE_COMPUTE_BIT, generate_terrain_descriptor_layout,
                              generate_terrain_descriptor);
  DescriptorBuilder::Reset();

  {
    auto &pipeline_builder = PipelineBuildManager::New<PipelineType::Compute>();
    pipeline_builder.AddDescriptorLayout(generate_terrain_descriptor_layout);
    pipeline_builder.SetShader(std::filesystem::path(SHADER_DIR) / "generate_terrain.slang");
    pipeline_builder.AddPushConstantRange(sizeof(GenerateTerrainPushConstants));
    PipelineBuildManager::Build(pipeline_builder, generate_terrain_pipeline);
  }
}

UtilContext::~UtilContext() {}
} // namespace Core
