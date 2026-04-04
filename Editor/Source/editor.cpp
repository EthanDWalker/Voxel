#include "editor.h"
#include "Core/Render/context.h"
#include "Core/Render/edit.h"
#include "Core/Render/frame.h"
#include "Core/Util/Parse/gltf.h"
#include "Core/Util/log.h"
#include "Core/Util/timer.h"
#include "Core/input.h"
#include "Core/window.h"
#include <bitset>
#include <filesystem>

void PrintNode(const u32 node) {
  if (node & 0x800000) {
    Core::Log("child mask: {} far ptr index: {}", std::bitset<8>(node >> 24).to_string(), node & 0x7FFFFF);
  } else {
    Core::Log("child mask: {} child offset: {}", std::bitset<8>(node >> 24).to_string(), node & 0x7FFFFF);
  }
}

void TraverseNode(const u32 node, u32 &ptr) {
  if ((node & 0x800000) == 0) {
    ptr += (node & 0x7FFFFF);
  } else {
    const u32 far_ptr = node & 0x7FFFFF;
    const u32 far_ptr_page_index = far_ptr >> Core::SparseVoxelTree::FAR_PTR_PAGE_SIZE_EXP;
    const u32 far_ptr_page_offset =
        far_ptr - (far_ptr_page_index << Core::SparseVoxelTree::FAR_PTR_PAGE_SIZE_EXP);

    ptr += ((u32 *)Core::render_context->voxel_tree.far_ptr_pages[far_ptr_page_index]
                ->address)[far_ptr_page_offset];
  }
}

void Editor::StartUp() {
  SCOPED_TIMER("START UP")
  camera.Create(Core::render_context->main_image.GetVec2u32());

  Core::MeshFileData mesh_file_data;
  Core::ParseGlbFile("C:/Users/ethan/Developer/Voxel/Editor/Assets/Sponza/Sponza.glb", mesh_file_data);

  for (u32 i = 0; i < mesh_file_data.mesh_data_arr.size(); i++) {
    Core::render_context->voxel_tree.VoxelizeMesh(mesh_file_data.mesh_data_arr[i]);
  }

  const Core::DirectionalLight dir_light = {
      .direction = Normalize(Vec3f32(-0.1f, -1.0f, -0.1f)),
      .intesity = 1.0f,
      .color = Vec3f32(1.0f),
  };

  for (u32 i = 0; i < Core::SparseVoxelTree::MAX_VOXLELIZE_DEPTH - 1; i++) {
    Core::Log("level {} has {} voxels", i, Core::render_context->voxel_tree.level_voxels[i]);
  }

  Core::AddDirectionalLight(dir_light);
}

void Editor::Run() {
  f32 delta_time = 0.0f;

  const u32 sample_size = 100;

  f32 frame_test_acc = 0.0f;
  u32 current_samples = 0;

  while (!Core::Window::ShouldClose()) {
    Core::Timer timer{};
    Core::InputContext::Update();
    bool resize = false;

    {
      if (Core::InputContext::GetHeld(Core::Input::MOUSE_RIGHT)) {
        camera.yaw -= Core::InputContext::delta_mouse_pos.x * camera.sensitivity;
        camera.pitch += Core::InputContext::delta_mouse_pos.y * camera.sensitivity;
      }
      camera.pitch = Clamp(camera.pitch, -89.0f, 89.0f);

      delta_time *= camera.speed;

      if (Core::InputContext::GetHeld(Core::Input::W))
        camera.position += delta_time * camera.front;
      if (Core::InputContext::GetHeld(Core::Input::S))
        camera.position -= delta_time * camera.front;
      if (Core::InputContext::GetHeld(Core::Input::D))
        camera.position += delta_time * camera.right;
      if (Core::InputContext::GetHeld(Core::Input::A))
        camera.position -= delta_time * camera.right;
      if (Core::InputContext::GetHeld(Core::Input::E))
        camera.position += delta_time * camera.up;
      if (Core::InputContext::GetHeld(Core::Input::Q))
        camera.position -= delta_time * camera.up;
    }

    Core::VulkanCommandBuffer &cmd = Core::BeginFrame(resize);

    Core::Frame(camera);

    Core::EndFrame(resize);

    if (resize) {
      Core::Resize(Core::Window::GetSize());
      camera.Resize(Core::render_context->main_image.GetVec2u32());
    }

    if (Core::InputContext::GetHeld(Core::Input::R) && Core::InputContext::GetHeld(Core::Input::LEFT_CONTROL))
      Core::render_context->RecreatePipelines();

    if (Core::InputContext::GetPressed(Core::Input::ESCAPE))
      Core::Window::SetShouldClose(true);

    if (Core::InputContext::GetPressed(Core::Input::V)) {
      Core::Log("gathering samples...");
      frame_test_acc = 0.0f;
      current_samples = 0;
    }

    if (Core::InputContext::GetHeld(Core::Input::F))
      camera.speed = Abs(Core::SparseVoxelTree::MAX_BOUND) / 100.0f;
    else
      camera.speed = Abs(Core::SparseVoxelTree::MAX_BOUND);

    if (current_samples < sample_size) {
      frame_test_acc += timer.ElapsedMillis();
      current_samples++;
    } else if (current_samples == sample_size) {
      Core::Log("frame time {} ms", frame_test_acc / float(sample_size));
      current_samples++;
    }

    Core::Window::SetTitle(std::format("{} fps", Round(1.0f / timer.Elapsed())));
    delta_time = timer.Elapsed();
  }
}

void Editor::ShutDown() {}
