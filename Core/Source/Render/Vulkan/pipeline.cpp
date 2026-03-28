#include "Core/Render/Vulkan/pipeline.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/util.h"
#include "Core/Util/fail.h"
#include "Core/Util/log.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace Core {
std::vector<PipelineBuilder<PipelineType::Graphic>>
    PipelineBuildManager::graphics_pipeline_builder_arr;
std::vector<VulkanPipeline<PipelineType::Graphic> *>
    PipelineBuildManager::graphics_pipeline_ptr_arr;

std::vector<PipelineBuilder<PipelineType::Compute>>
    PipelineBuildManager::compute_pipeline_builder_arr;
std::vector<VulkanPipeline<PipelineType::Compute> *> PipelineBuildManager::compute_pipeline_ptr_arr;

enum class SlangTokens : u8 {
  Import,
  EndLine,
};

constexpr std::string_view SlangTokenStrings[] = {
    "import",
    ";",
};

void BaseVulkanPipeline::Destroy() {
  vkDestroyPipeline(VulkanContext::device, obj, nullptr);
  vkDestroyPipelineLayout(VulkanContext::device, layout, nullptr);
}

BaseVulkanPipeline::~BaseVulkanPipeline() {
  vkDestroyPipeline(VulkanContext::device, obj, nullptr);
  vkDestroyPipelineLayout(VulkanContext::device, layout, nullptr);
}

void GetShaderImports(const std::filesystem::path &shader_src,
                      std::vector<std::filesystem::path> &paths) {
  std::ifstream src_file(shader_src);

  std::string token;
  std::getline(src_file, token, ' ');

  while (token == SlangTokenStrings[static_cast<u8>(SlangTokens::Import)]) {
    std::getline(src_file, token, ';');
    paths.push_back(shader_src.parent_path() / (token + ".slang"));
    std::getline(src_file, token, '\n');
    std::getline(src_file, token, ' ');
  }
}

void LoadShaderModule(const std::filesystem::path &shader_src, VkShaderModule *out_shader_module) {
  std::string shader_bin = shader_src.string() + ".spv";

  std::vector<std::filesystem::path> import_paths;

  GetShaderImports(shader_src, import_paths);

  bool import_updated = false;
  if (std::filesystem::exists(shader_bin)) {
    for (const std::filesystem::path import : import_paths) {
      Assert(std::filesystem::exists(import), "invalid import found in shader file {}",
             shader_src.string());
      if (std::filesystem::last_write_time(import) > std::filesystem::last_write_time(shader_bin)) {
        import_updated = true;
        break;
      }
    }
  }

  if (!std::filesystem::exists(std::filesystem::path(shader_bin)) ||
      (std::filesystem::last_write_time(shader_src) >
       std::filesystem::last_write_time(shader_bin)) ||
      import_updated) {
    Log("compiling shader {}", shader_src.string());
    system(std::format("slangc {} -o {} -capability SPIRV_1_6 -target spirv -profile sm_6_6"
                       " -matrix-layout-column-major -fvk-use-entrypoint-name -O3",
                       shader_src.string(), shader_bin)
               .c_str());
  }

  std::ifstream file(shader_bin, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    Log("Failed to load shader file: {}", shader_src.string());
    abort();
  }

  u64 file_size = (u64)file.tellg();
  std::vector<u32> buffer(file_size / sizeof(u32));

  file.seekg(0);

  file.read((char *)buffer.data(), file_size);

  file.close();

  VkShaderModuleCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.pNext = nullptr;
  ci.codeSize = buffer.size() * sizeof(u32);
  ci.pCode = buffer.data();

  VkShaderModule shader_module;
  if (vkCreateShaderModule(VulkanContext::device, &ci, nullptr, &shader_module) != VK_SUCCESS) {
    Log("Failed to create shader module");
    abort();
  }
  *out_shader_module = shader_module;
}

void PipelineBuildManager::RecreatePipelines() {
  vkDeviceWaitIdle(VulkanContext::device);
  for (u32 i = 0; i < graphics_pipeline_ptr_arr.size(); i++) {
    graphics_pipeline_ptr_arr[i]->Destroy();
    graphics_pipeline_builder_arr[i].Build(*graphics_pipeline_ptr_arr[i]);
  }
  for (u32 i = 0; i < compute_pipeline_ptr_arr.size(); i++) {
    compute_pipeline_ptr_arr[i]->Destroy();
    compute_pipeline_builder_arr[i].Build(*compute_pipeline_ptr_arr[i]);
  }
}

void PipelineBuilder<PipelineType::Compute>::AddPushConstantRange(u32 size) {
  VkPushConstantRange range{};
  u32 offset = 0;
  for (auto &range : push_constant_ranges) {
    offset += range.offset;
  }
  range.offset = offset;
  range.size = size;
  range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_constant_ranges.push_back(range);
}

void PipelineBuilder<PipelineType::Compute>::AddDescriptor(const VulkanDescriptor &descriptor) {
  descriptor_set_layouts.push_back(descriptor.layout);
}

void PipelineBuilder<PipelineType::Compute>::Build(
    VulkanPipeline<PipelineType::Compute> &pipeline) {
  VkShaderModule shader;

  Assert(std::filesystem::exists(shader_src), "compute source ({}) doesnt exist",
         shader_src.string());

  LoadShaderModule(shader_src, &shader);

  VkPipelineLayoutCreateInfo pipeline_layout_ci{};
  pipeline_layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_ci.pPushConstantRanges = push_constant_ranges.data();
  pipeline_layout_ci.pushConstantRangeCount = push_constant_ranges.size();
  pipeline_layout_ci.pSetLayouts = descriptor_set_layouts.data();
  pipeline_layout_ci.setLayoutCount = descriptor_set_layouts.size();

  VK_CHECK(vkCreatePipelineLayout(VulkanContext::device, &pipeline_layout_ci, nullptr,
                                  &pipeline.layout));

  VkPipelineShaderStageCreateInfo shader_ci{};
  shader_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shader_ci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  shader_ci.module = shader;
  shader_ci.pName = "CMain";

  VkComputePipelineCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  info.layout = pipeline.layout;
  info.stage = shader_ci;

  VK_CHECK(vkCreateComputePipelines(VulkanContext::device, VK_NULL_HANDLE, 1, &info, nullptr,
                                    &pipeline.obj));

  vkDestroyShaderModule(VulkanContext::device, shader, nullptr);
}

// need to set shaders and depth image format
void PipelineBuilder<PipelineType::Graphic>::Default() {
  SetCullMode(VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);
  SetPolygonMode(VK_POLYGON_MODE_FILL);
  SetInputTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  SetDepthTest();
  SetDepthFormat(VK_FORMAT_D32_SFLOAT);
  SetNoMultisampling();
  SetViewportCount(1);
}

void PipelineBuilder<PipelineType::Graphic>::SetInputTopology(VkPrimitiveTopology topology) {
  input_assembly.topology = topology;
  input_assembly.primitiveRestartEnable = VK_FALSE;
}

void PipelineBuilder<PipelineType::Graphic>::SetPolygonMode(VkPolygonMode mode) {
  rasterization.polygonMode = mode;
  rasterization.lineWidth = 1.0f;
}

void PipelineBuilder<PipelineType::Graphic>::SetCullMode(VkCullModeFlags cull_mode,
                                                         VkFrontFace front_face) {
  rasterization.cullMode = cull_mode;
  rasterization.frontFace = front_face;
}

void PipelineBuilder<PipelineType::Graphic>::EnableConservativeRasterization() {
  VkPhysicalDeviceConservativeRasterizationPropertiesEXT props{};
  props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT;

  VkPhysicalDeviceProperties2KHR device_props{};
  device_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
  device_props.pNext = &props;
  vkGetPhysicalDeviceProperties2(VulkanContext::physical_device, &device_props);

  conservative_rasterization.conservativeRasterizationMode =
      VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;

  rasterization.pNext = &conservative_rasterization;
}

void PipelineBuilder<PipelineType::Graphic>::SetMultisampling(VkSampleCountFlagBits sample_count) {
  multisample.sampleShadingEnable = VK_FALSE;
  multisample.rasterizationSamples = sample_count;
  multisample.minSampleShading = 0.2f;
}

void PipelineBuilder<PipelineType::Graphic>::SetNoMultisampling() {
  multisample.sampleShadingEnable = VK_FALSE;
  multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multisample.minSampleShading = 1.0f;
  multisample.pSampleMask = nullptr;
  multisample.alphaToCoverageEnable = VK_FALSE;
  multisample.alphaToOneEnable = VK_FALSE;
}

void PipelineBuilder<PipelineType::Graphic>::SetBlendingAdditive(u8 index) {
  color_attachments[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  color_attachments[index].blendEnable = VK_TRUE;
  color_attachments[index].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  color_attachments[index].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
  color_attachments[index].colorBlendOp = VK_BLEND_OP_ADD;
  color_attachments[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  color_attachments[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  color_attachments[index].alphaBlendOp = VK_BLEND_OP_ADD;
}

void PipelineBuilder<PipelineType::Graphic>::SetBlendingAlpha(u8 index) {
  color_attachments[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  color_attachments[index].blendEnable = VK_TRUE;
  color_attachments[index].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  color_attachments[index].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  color_attachments[index].colorBlendOp = VK_BLEND_OP_ADD;
  color_attachments[index].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  color_attachments[index].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  color_attachments[index].alphaBlendOp = VK_BLEND_OP_ADD;
}

void PipelineBuilder<PipelineType::Graphic>::SetNoBlending(u8 index) {
  color_attachments[index].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  color_attachments[index].blendEnable = VK_FALSE;
}

void PipelineBuilder<PipelineType::Graphic>::AddColorAttachment(VkFormat format) {
  color_attachment_formats.push_back(format);
  color_attachments.push_back({});
  render_info.colorAttachmentCount++;
  render_info.pColorAttachmentFormats = color_attachment_formats.data();
  SetNoBlending(color_attachments.size() - 1);
}

void PipelineBuilder<PipelineType::Graphic>::SetNoDepthTest() {
  depth_stencil.depthTestEnable = VK_FALSE;
  depth_stencil.depthWriteEnable = VK_FALSE;
  depth_stencil.depthCompareOp = VK_COMPARE_OP_NEVER;
  depth_stencil.depthBoundsTestEnable = VK_FALSE;
  depth_stencil.stencilTestEnable = VK_FALSE;
  depth_stencil.front = {};
  depth_stencil.back = {};
  depth_stencil.minDepthBounds = 0.f;
  depth_stencil.maxDepthBounds = 1.f;
}

void PipelineBuilder<PipelineType::Graphic>::SetDepthTest() {
  depth_stencil.depthTestEnable = VK_TRUE;
  depth_stencil.depthWriteEnable = VK_TRUE;
  depth_stencil.depthCompareOp = VK_COMPARE_OP_GREATER;
  depth_stencil.depthBoundsTestEnable = VK_FALSE;
  depth_stencil.stencilTestEnable = VK_FALSE;
  depth_stencil.front = {};
  depth_stencil.back = {};
  depth_stencil.minDepthBounds = 0.f;
  depth_stencil.maxDepthBounds = 1.f;
}

void PipelineBuilder<PipelineType::Graphic>::SetDepthFormat(VkFormat format) {
  render_info.depthAttachmentFormat = format;
}

void PipelineBuilder<PipelineType::Graphic>::AddPushConstantRange(VkShaderStageFlags stage_flags,
                                                                  u32 size) {
  VkPushConstantRange range{};
  u32 offset = 0;
  for (auto range : push_constant_ranges) {
    offset += range.size;
  }
  range.offset = offset;
  range.size = size;
  range.stageFlags = stage_flags;
  push_constant_ranges.push_back(range);
}

void PipelineBuilder<PipelineType::Graphic>::AddDescriptor(const VulkanDescriptor &descriptor) {
  descriptor_set_layouts.push_back(descriptor.layout);
}

void PipelineBuilder<PipelineType::Graphic>::SetViewportCount(u32 count) {
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.pNext = nullptr;

  viewport_state.viewportCount = count;
  viewport_state.scissorCount = count;
}

void PipelineBuilder<PipelineType::Graphic>::Build(
    VulkanPipeline<PipelineType::Graphic> &pipeline) {
  Assert(std::filesystem::exists(vert_shader_src_path), "vert source ({}) doesnt exist",
         vert_shader_src_path.string());
  Assert(std::filesystem::exists(frag_shader_src_path), "frag source ({}) doesnt exist",
         frag_shader_src_path.string());
  Assert(!geom_shader_src_path.has_value() || std::filesystem::exists(geom_shader_src_path.value()),
         "geom source ({}) doesnt exist", geom_shader_src_path.value_or("").string());

  VkShaderModule vert_shader;
  VkShaderModule frag_shader;
  std::optional<VkShaderModule> geom_shader;

  LoadShaderModule(vert_shader_src_path, &vert_shader);
  LoadShaderModule(frag_shader_src_path, &frag_shader);
  if (geom_shader_src_path.has_value()) {
    geom_shader.emplace();
    LoadShaderModule(geom_shader_src_path.value(), &geom_shader.value());
  }

  VkPipelineLayoutCreateInfo pipeline_layout_ci{};
  pipeline_layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_ci.pPushConstantRanges = push_constant_ranges.data();
  pipeline_layout_ci.pushConstantRangeCount = push_constant_ranges.size();
  pipeline_layout_ci.pSetLayouts = descriptor_set_layouts.data();
  pipeline_layout_ci.setLayoutCount = descriptor_set_layouts.size();

  VK_CHECK(vkCreatePipelineLayout(VulkanContext::device, &pipeline_layout_ci, nullptr,
                                  &pipeline.layout));

  VkPipelineColorBlendStateCreateInfo color_blending = {};
  color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending.logicOpEnable = VK_FALSE;
  color_blending.pAttachments = color_attachments.data();
  color_blending.attachmentCount = color_attachments.size();

  VkPipelineVertexInputStateCreateInfo vertex_input_info{};
  vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkGraphicsPipelineCreateInfo pipeline_ci = {};
  pipeline_ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_ci.pNext = &render_info;

  std::vector<VkPipelineShaderStageCreateInfo> shader_stages = {};
  shader_stages.resize(2);
  shader_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shader_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  shader_stages[0].module = vert_shader;
  shader_stages[0].pName = "VMain";

  shader_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shader_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  shader_stages[1].module = frag_shader;
  shader_stages[1].pName = "FMain";

  if (geom_shader.has_value()) {
    shader_stages.resize(3);
    shader_stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[2].stage = VK_SHADER_STAGE_GEOMETRY_BIT;
    shader_stages[2].module = geom_shader.value();
    shader_stages[2].pName = "GMain";
  };

  pipeline_ci.stageCount = shader_stages.size();
  pipeline_ci.pStages = shader_stages.data();
  pipeline_ci.pVertexInputState = &vertex_input_info;
  pipeline_ci.pInputAssemblyState = &input_assembly;
  pipeline_ci.pViewportState = &viewport_state;
  pipeline_ci.pRasterizationState = &rasterization;
  pipeline_ci.pMultisampleState = &multisample;
  pipeline_ci.pColorBlendState = &color_blending;
  pipeline_ci.pDepthStencilState = &depth_stencil;
  pipeline_ci.layout = pipeline.layout;

  VkDynamicState state[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

  VkPipelineDynamicStateCreateInfo dynamic_info{};
  dynamic_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_info.pDynamicStates = &state[0];
  dynamic_info.dynamicStateCount = 2;

  pipeline_ci.pDynamicState = &dynamic_info;
  if (vkCreateGraphicsPipelines(VulkanContext::device, VK_NULL_HANDLE, 1, &pipeline_ci, nullptr,
                                &pipeline.obj) != VK_SUCCESS) {
    Assert(false, "Failed to create graphics pipeline");
  }

  vkDestroyShaderModule(VulkanContext::device, vert_shader, nullptr);
  vkDestroyShaderModule(VulkanContext::device, frag_shader, nullptr);
  if (geom_shader.has_value()) {
    vkDestroyShaderModule(VulkanContext::device, geom_shader.value(), nullptr);
    geom_shader.reset();
  }
}
} // namespace Core
