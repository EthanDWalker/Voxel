#pragma once

#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/image.h"
#include "Core/Render/Vulkan/pipeline.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/Vulkan/swapchain.h"
#include "sparse_voxel_tree.h"

namespace Core {
struct Spec {
  u32 max_directional_lights = 10;
};

const u32 BEAM_PREPASS_SCALE = 4;

struct RenderContext {
  Spec current_spec;
  VulkanSwapchain swapchain;
  VulkanImage<ImageType::Planar> main_image;
  VulkanImage<ImageType::Planar> beam_prepass_image;

  VulkanBuffer frame_staging_buffer;

  VulkanPipeline<PipelineType::Compute> main_pipeline;
  VulkanPipeline<PipelineType::Compute> beam_prepass_pipeline;

  VulkanDescriptor image_descriptor;
  VulkanDescriptor camera_descriptor;
  VulkanBuffer camera_buffer;

  VulkanDescriptor light_descriptor;
  VulkanBuffer directional_light_buffer;
  u32 directional_light_count;

  SparseVoxelTree voxel_tree;

  VulkanSubPass<SubPassType::Compute> beam_pass;
  VulkanSubPass<SubPassType::Compute> main_draw_pass;
  VulkanSubPass<SubPassType::Transfer> upload_pass;

  void Create(const Spec &spec);
  void CreatePipelines();
  void RecreatePipelines();

  ~RenderContext();
};

extern RenderContext *render_context;
} // namespace Core
