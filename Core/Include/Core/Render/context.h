#pragma once

#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/image.h"
#include "Core/Render/Vulkan/other_buffer.h"
#include "Core/Render/Vulkan/pipeline.h"
#include "Core/Render/Vulkan/swapchain.h"
#include "Core/Render/camera.h"
#include "Core/Render/device_hash_set.h"
#include "Core/Render/types.h"
#include "sparse_voxel_tree.h"
#include <chrono>
#include <functional>
#include <mutex>

namespace Core {
struct RenderSpec {
  u32 max_raycasts = 10;
};

const u32 BEAM_PREPASS_SCALE_EXP = 2;
const u32 INDIRECT_LIGHT_SCALE_EXP = 1;

const f32 DEFAULT_DIFFUSE_LIGHT_ALPHA = 0.01f;
const f32 DEFAULT_SPECULAR_LIGHT_ALPHA = 0.2f;

struct RenderContext {
  RenderSpec current_spec;
  VulkanSwapchain swapchain;

  std::array<VulkanBuffer<BufferType::StagingBuffer>, VulkanSwapchain::FRAME_OVERLAP> frame_staging_buffer = {
      "frame staging buffer 0",
      "frame staging buffer 1",
      "frame staging buffer 2",
  };

  std::array<VulkanBuffer<BufferType::StructuredBuffer, Camera>, VulkanSwapchain::FRAME_OVERLAP>
      camera_buffer = {
          "camera buffer 0",
          "camera buffer 1",
          "camera buffer 2",
  };

  VulkanPipeline<PipelineType::Compute> main_pipeline;
  VulkanPipeline<PipelineType::Compute> beam_prepass_pipeline;
  VulkanPipeline<PipelineType::Compute> lighting_pipeline;
  VulkanPipeline<PipelineType::Compute> chunk_allocate_pipeline;
  VulkanPipeline<PipelineType::Compute> tone_map_pipeline;

  VulkanDescriptorLayout image_descriptor_layout;
  VulkanDescriptor image_descriptor;
  VulkanImage<ImageType::Planar> main_image;
  VulkanImage<ImageType::Planar> beam_prepass_image;

  std::array<VulkanBuffer<BufferType::StructuredBuffer, FrameLuminanceData>, VulkanSwapchain::FRAME_OVERLAP>
      frame_luminance_data_buffer = {
          "frame luminance data buffer 0",
          "frame luminance data buffer 1",
          "frame luminance data buffer 2",
  };

  VulkanDescriptorLayout frame_luminance_descriptor_layout;
  std::array<VulkanDescriptor, VulkanSwapchain::FRAME_OVERLAP> frame_luminance_descriptor = {};

  VulkanDescriptorLayout camera_descriptor_layout;
  std::array<VulkanDescriptor, VulkanSwapchain::FRAME_OVERLAP> camera_descriptor = {};

  VulkanDescriptorLayout swapchain_descriptor_layout;
  std::array<VulkanDescriptor, VulkanSwapchain::FRAME_OVERLAP> swapchain_descriptor = {};

  SparseVoxelTree voxel_tree;

  DeviceHashSet light_hash_set;
  f32 diffuse_light_alpha = DEFAULT_DIFFUSE_LIGHT_ALPHA;
  f32 specular_light_alpha = DEFAULT_SPECULAR_LIGHT_ALPHA;

  VulkanPipeline<PipelineType::Compute> clear_volume_pipeline;
  std::vector<VoxelVolume> clear_volume_cmds;
  std::mutex clear_volume_cmd_mutex;

  VulkanPipeline<PipelineType::Compute> fill_volume_pipeline;
  std::vector<VoxelVolume> fill_volume_cmds;
  std::mutex fill_volume_cmd_mutex;

  VulkanSampler albedo_sampler;

  VulkanBuffer<BufferType::StagingBuffer> raycast_staging_buffer = "raycast staging buffer";
  VulkanBuffer<BufferType::StructuredBuffer, Raycast> raycast_cmds_buffer = "raycast cmd buffer";
  VulkanBuffer<BufferType::StructuredBuffer, RaycastResult> raycast_results_buffer = "raycast result buffer";

  VulkanDescriptorLayout raycast_descriptor_layout;
  VulkanDescriptor raycast_descriptor;

  std::vector<Raycast> raycast_cmds;
  std::vector<std::function<void(RaycastResult)>> raycast_cmd_callbacks;
  std::mutex raycast_cmd_mutex;
  VulkanPipeline<PipelineType::Compute> raycast_cmd_pipeline;

  std::chrono::time_point<std::chrono::steady_clock> last_frame_time;
  std::chrono::time_point<std::chrono::steady_clock> start_time;

  RenderContext(const RenderSpec &spec);
  void CreatePipelines();
  void RecreatePipelines();

  ~RenderContext();
};

extern std::unique_ptr<RenderContext> render_context;
} // namespace Core
