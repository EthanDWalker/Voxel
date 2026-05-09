#pragma once

#include "Core/Render/Vulkan/acceleration_structure.h"
#include "Core/Render/Vulkan/buffer.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/image.h"
#include "Core/Render/Vulkan/pipeline.h"
#include "Core/Render/Vulkan/shader_binding_table.h"
#include "Core/Render/Vulkan/swapchain.h"
#include "Core/Render/device_dense_set.h"
#include "Core/Render/types.h"
#include "sparse_voxel_tree.h"
#include <functional>
#include <mutex>

namespace Core {
struct Spec {
  u32 max_directional_lights = 10;
  u32 max_raycasts = 10;
  u32 max_instances = 1'000;
  u32 max_meshes = 1'000;
};

const u32 PREPASS_SCALE = 4;

struct RenderContext {
  Spec current_spec;
  VulkanSwapchain swapchain;
  VulkanImage<ImageType::Planar> main_image;
  VulkanImage<ImageType::Planar> beam_prepass_image;

  VulkanBuffer frame_staging_buffer = "frame staging buffer";

  VulkanPipeline<PipelineType::Compute> main_pipeline;
  VulkanPipeline<PipelineType::Compute> beam_prepass_pipeline;
  VulkanPipeline<PipelineType::Compute> clear_volume_pipeline;

  VulkanPipeline<PipelineType::Graphic> allocate_pipeline;
  VulkanPipeline<PipelineType::Graphic> allocate_child_mask_pipeline;
  VulkanPipeline<PipelineType::Graphic> reallocate_pipeline;
  VulkanPipeline<PipelineType::Raytrace> triangle_id_pipeline;
  VulkanShaderBindingTable triangle_id_shader_binding_table;

  DeviceDenseSet triangle_id_dense_set = DeviceDenseSet(
      10'000, VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR);

  VulkanDescriptor image_descriptor;
  VulkanDescriptor camera_descriptor;
  VulkanBuffer camera_buffer = "camera buffer";

  VulkanDescriptor light_descriptor;
  VulkanBuffer directional_light_buffer = "directional light buffer";
  u32 directional_light_count;

  SparseVoxelTree voxel_tree;

  std::vector<VoxelVolume> clear_volume_cmds;
  std::mutex clear_volume_cmd_mutex;

  VulkanDescriptor mesh_descriptor;
  VulkanBuffer mesh_triangle_offset_buffer = "mesh triangle offset buffer";
  u32 total_triangles = 0;
  u32 mesh_count = 0;
  std::vector<std::unique_ptr<VulkanBuffer>> vertex_buffers;
  std::vector<std::unique_ptr<VulkanBuffer>> index_buffers;
  std::vector<std::unique_ptr<VulkanBuffer>> triangle_voxelized_depth_buffers;
  VulkanSampler albedo_sampler;
  std::vector<std::unique_ptr<VulkanImage<ImageType::Planar>>> albedo_images;
  std::vector<std::unique_ptr<VulkanAccelerationStructure>> bottom_level_acceleration_structures;

  VulkanBuffer raycast_staging_buffer = "raycast staging buffer";
  VulkanBuffer raycast_cmds_buffer = "raycast cmd buffer";
  VulkanBuffer raycast_results_buffer = "raycast result buffer";
  VulkanDescriptor raycast_descriptor;
  std::vector<Raycast> raycast_cmds;
  std::vector<std::function<void(RaycastResult)>> raycast_callbacks;
  std::mutex raycast_mutex;
  VulkanPipeline<PipelineType::Compute> raycast_pipeline;

  std::mutex add_instance_cmd_mutex;
  std::vector<Instance> add_instance_cmds;
  VulkanBuffer instance_buffer = "instance buffer";
  VulkanAccelerationStructure top_level_acceleration_structure;
  VulkanDescriptor top_level_acceleration_structure_descriptor;
  u32 instance_count;

  void Create(const Spec &spec);
  void CreatePipelines();
  void RecreatePipelines();

  ~RenderContext();
};

extern RenderContext *render_context;
} // namespace Core
