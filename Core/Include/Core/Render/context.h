#pragma once

#include "Core/Render/Vulkan/acceleration_structure.h"
#include "Core/Render/Vulkan/descriptors.h"
#include "Core/Render/Vulkan/image.h"
#include "Core/Render/Vulkan/other_buffer.h"
#include "Core/Render/Vulkan/pipeline.h"
#include "Core/Render/Vulkan/swapchain.h"
#include "Core/Render/camera.h"
#include "Core/Render/types.h"
#include <chrono>

namespace Core {
struct Spec {
  u32 max_directional_lights = 10;
  u32 max_raycasts = 10;
  u32 max_meshes = 10'000;
  u32 max_average_rays_per_pixel = 2;
  u32 max_cached_indirect_lighting_per_pixel = 2;
};

const u32 BEAM_PREPASS_SCALE_EXP = 2;
const u32 INDIRECT_LIGHT_SCALE_EXP = 1;

struct RenderContext {
  Spec current_spec;
  VulkanSwapchain swapchain;
  VulkanImage<ImageType::Planar> main_image;

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

  VulkanPipeline<PipelineType::Graphic> allocate_pipeline;

  VulkanDescriptorLayout image_descriptor_layout;
  VulkanDescriptor image_descriptor;

  VulkanDescriptorLayout camera_descriptor_layout;
  std::array<VulkanDescriptor, VulkanSwapchain::FRAME_OVERLAP> camera_descriptor = {};

  VulkanDescriptorLayout light_descriptor_layout;
  VulkanDescriptor light_descriptor;
  VulkanBuffer<BufferType::CountedBuffer, DirectionalLight> directional_light_buffer =
      "directional light buffer";
<<<<<<< Updated upstream
  u32 directional_light_count;

  SparseVoxelTree voxel_tree;

  DeviceHashSet indirect_light_hash_set;
  IndirectDispatchBuffer<PipelineType::Compute, IndirectLightingRayDispatch> indirect_light_dispatch_buffer;

  std::vector<VoxelVolume> clear_volume_cmds;
  std::mutex clear_volume_cmd_mutex;

  VulkanBuffer<BufferType::CountedBuffer, Instance> instance_counted_buffer = "instance counted buffer";
  VulkanBuffer<BufferType::CountedBuffer, Mesh> mesh_counted_buffer = "mesh counted buffer";
=======

  VulkanBuffer<BufferType::CountedBuffer, Instance> instance_counted_buffer = "instance counted buffer";
  VulkanAccelerationStructure<AccelerationStructureType::TopLevel> top_level_acceleration_structure;
  std::vector<std::unique_ptr<VulkanBuffer<BufferType::StructuredBuffer, BranchNode>>> mesh_node_brick_arr;
  std::vector<std::unique_ptr<VulkanAccelerationStructure<AccelerationStructureType::AABB>>>
      bottom_level_acceleration_structure_arr;

>>>>>>> Stashed changes
  VulkanDescriptorLayout mesh_descriptor_layout;
  VulkanDescriptor mesh_descriptor;

  VulkanDescriptorLayout voxelize_descriptor_layout;
  VulkanDescriptor voxelize_descriptor;
  VulkanSampler albedo_sampler;

  std::chrono::time_point<std::chrono::steady_clock> last_frame_time;
  std::chrono::time_point<std::chrono::steady_clock> start_time;

  void Create(const Spec &spec);
  void CreatePipelines();
  void RecreatePipelines();

  ~RenderContext();
};

extern RenderContext *render_context;
} // namespace Core
