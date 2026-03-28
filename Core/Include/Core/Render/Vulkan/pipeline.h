#pragma once

#include "Core/Render/Vulkan/descriptors.h"
#include "volk.h"
#include <filesystem>
#include <optional>
#include <vector>

namespace Core {

enum class PipelineType {
  Graphic,
  Compute,
};

struct BaseVulkanPipeline {
  VkPipeline obj = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;

  BaseVulkanPipeline() = default;

  BaseVulkanPipeline(const BaseVulkanPipeline &) = delete;
  BaseVulkanPipeline &operator=(const BaseVulkanPipeline &) = delete;

  BaseVulkanPipeline(BaseVulkanPipeline &&) = default;
  BaseVulkanPipeline &operator=(BaseVulkanPipeline &&) = default;

  void Destroy();

  ~BaseVulkanPipeline();
};

template <PipelineType T> struct VulkanPipeline : BaseVulkanPipeline {};

template <PipelineType T> struct PipelineBuilder {};

template <> struct PipelineBuilder<PipelineType::Compute> {
  std::vector<VkPushConstantRange> push_constant_ranges = {};
  std::vector<VkDescriptorSetLayout> descriptor_set_layouts = {};
  std::filesystem::path shader_src;

  void SetShader(const std::filesystem::path &comp) { shader_src = comp; }

  void AddPushConstantRange(u32 size);
  void AddDescriptor(const VulkanDescriptor &descriptor);

  void Build(VulkanPipeline<PipelineType::Compute> &pipeline);
};

template <> struct PipelineBuilder<PipelineType::Graphic> {
  std::vector<VkPushConstantRange> push_constant_ranges = {};
  std::vector<VkDescriptorSetLayout> descriptor_set_layouts = {};
  std::vector<VkPipelineColorBlendAttachmentState> color_attachments = {};
  std::vector<VkFormat> color_attachment_formats = {};

  VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  VkPipelineTessellationStateCreateInfo tessellation = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
  VkPipelineViewportStateCreateInfo viewport = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  VkPipelineRasterizationStateCreateInfo rasterization = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  VkPipelineDepthStencilStateCreateInfo depth_stencil = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  VkPipelineDynamicStateCreateInfo dynamic_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  VkPipelineRenderingCreateInfo render_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  VkPipelineRasterizationConservativeStateCreateInfoEXT conservative_rasterization = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT};

  std::filesystem::path vert_shader_src_path;
  std::filesystem::path frag_shader_src_path;
  std::optional<std::filesystem::path> geom_shader_src_path;

  void SetShaders(const std::filesystem::path &vert, const std::filesystem::path &frag,
                  const std::filesystem::path &geom = "") {
    vert_shader_src_path = vert;
    frag_shader_src_path = frag;
    if (!geom.empty()) {
      geom_shader_src_path = geom;
    }
  }

  void Default();

  void SetInputTopology(VkPrimitiveTopology topology);

  void SetPolygonMode(VkPolygonMode mode);

  void SetViewportCount(u32 count);

  void SetCullMode(VkCullModeFlags cull_mode, VkFrontFace front_face);

  void SetNoMultisampling();

  void SetMultisampling(VkSampleCountFlagBits sample_count);

  void SetNoBlending(u8 index);

  void SetBlendingAdditive(u8 index);

  void SetBlendingAlpha(u8 index);

  void EnableConservativeRasterization();

  void AddColorAttachment(VkFormat format);

  void SetNoDepthTest();

  void SetDepthTest();

  void SetDepthFormat(VkFormat format);

  void AddPushConstantRange(VkShaderStageFlags stage_flags, u32 size);

  void AddDescriptor(const VulkanDescriptor &descriptor);

  void Build(VulkanPipeline<PipelineType::Graphic> &pipeline);
};

struct PipelineBuildManager {
  static std::vector<PipelineBuilder<PipelineType::Graphic>> graphics_pipeline_builder_arr;
  static std::vector<VulkanPipeline<PipelineType::Graphic> *> graphics_pipeline_ptr_arr;

  static std::vector<PipelineBuilder<PipelineType::Compute>> compute_pipeline_builder_arr;
  static std::vector<VulkanPipeline<PipelineType::Compute> *> compute_pipeline_ptr_arr;

  template <PipelineType T> static PipelineBuilder<T> &New() {
    if constexpr (T == PipelineType::Graphic) {
      return graphics_pipeline_builder_arr.emplace_back();
    } else if constexpr (T == PipelineType::Compute) {
      return compute_pipeline_builder_arr.emplace_back();
    } else {
      static_assert(false, "Invalid type");
    }
  }

  template <PipelineType T>
  static void Build(PipelineBuilder<T> &builder, VulkanPipeline<T> &pipeline) {
    if constexpr (T == PipelineType::Graphic) {
      graphics_pipeline_ptr_arr.push_back(&pipeline);
      builder.Build(pipeline);
    } else if constexpr (T == PipelineType::Compute) {
      compute_pipeline_ptr_arr.push_back(&pipeline);
      builder.Build(pipeline);
    } else {
      static_assert(false, "Invalid type");
    }
  }

  static void RecreatePipelines();
};

} // namespace Core
