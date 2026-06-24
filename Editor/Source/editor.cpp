#include "editor.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/submission_pass.h"
#include "Core/Render/add.h"
#include "Core/Render/commands.h"
#include "Core/Render/context.h"
#include "Core/Render/debug.h"
#include "Core/Render/frame.h"
#include "Core/Render/sparse_voxel_tree.h"
#include "Core/Render/types.h"
#include "Core/Util/log.h"
#include "Core/Util/timer.h"
#include "Core/input.h"
#include "Core/window.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

void Editor::StartUp() {
  camera.Create(Core::render_context->main_image.GetVec2u32());

  constexpr u32 seed = 832910;

  for (u32 x = 0; x < 4; x++) {
    for (u32 y = 0; y < 4; y++) {
      for (u32 z = 0; z < 4; z++) {
        SCOPED_TIMER(std::format("voxelize chunk {} {} {}", x, y, z));
        const Vec3u32 chunk_index = Vec3u32(x, y, z);
        Core::VoxelizeChunk(Vec3u32(x, y, z), seed, Core::SparseVoxelTree::MAX_DEPTH);
      }
    }
  }

  Core::SparseVoxelTree &tree = Core::render_context->voxel_tree;

  Core::SparseVoxelTree::TreeHeader *tree_header =
      (Core::SparseVoxelTree::TreeHeader *)tree.tree_header_host_buffer.host_address;

  Core::Log("branch voxel count {} (pages {})", tree_header->branch_count, tree.branch_pages.size());
  Core::Log("leaf count {} (pages {})", tree_header->leaf_count, tree.leaf_pages.size());

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.ConfigWindowsMoveFromTitleBarOnly = true;
  io.ConfigFlags = ImGuiConfigFlags_NavEnableKeyboard;

  ImGui_ImplGlfw_InitForVulkan((GLFWwindow *)Core::Window::handle, true);
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.UseDynamicRendering = true;
  init_info.Instance = Core::VulkanContext::instance;
  init_info.PhysicalDevice = Core::VulkanContext::physical_device;
  init_info.Device = Core::VulkanContext::device;
  init_info.QueueFamily = Core::VulkanContext::graphics_queue_index;
  init_info.Queue = Core::VulkanContext::graphics_queue;
  init_info.MinImageCount = Core::render_context->swapchain.FRAME_OVERLAP;
  init_info.ImageCount = Core::render_context->swapchain.FRAME_OVERLAP;
  init_info.DescriptorPoolSize = 100;
  init_info.PipelineInfoMain.PipelineRenderingCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
  init_info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
  init_info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats =
      &Core::render_context->swapchain.format;
  ImGui_ImplVulkan_Init(&init_info);
}

void Editor::Run() {
  f32 delta_time = 0.0f;

  f32 frame_test_acc = 0.0f;
  u32 current_samples = 0;

  while (!Core::Window::ShouldClose()) {
    Core::Timer timer{};
    bool resize = false;
    Core::InputContext::Update();

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiViewport *viewport = ImGui::GetMainViewport();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

    const char *label = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    const ImVec2 text_size = ImGui::CalcTextSize(label);
    const ImVec2 padding = ImGui::GetStyle().FramePadding;

    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize({viewport->WorkSize.x, text_size.y + padding.y * 2.0f});

    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                                   ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings;

    if (ImGui::Begin("Toolbar", nullptr, flags)) {
      if (ImGui::Button("Lighting")) {
        ImGui::OpenPopup("Lighting Settings");
      }

      if (ImGui::BeginPopup("Lighting Settings")) {
        ImGui::SliderFloat("Diffuse Alpha", &Core::render_context->diffuse_light_alpha, 0.0f, 0.5f);
        ImGui::SliderFloat("Specular Alpha", &Core::render_context->specular_light_alpha, 0.0f, 0.5f);
        ImGui::EndPopup();
      }

      ImGui::SameLine();
      if (ImGui::Button("Camera")) {
        ImGui::OpenPopup("Camera Settings");
      }

      if (ImGui::BeginPopup("Camera Settings")) {
        ImGui::Text("Camera Position: %s", camera.position.String().c_str());
        ImGui::EndPopup();
      }
      ImGui::SameLine();

      if (ImGui::Button("Material")) {
        ImGui::OpenPopup("Material Settings");
      }

      if (ImGui::BeginPopup("Material Settings")) {
        static u32 material_index = 0;
        ImGui::Text("Material Index: %d", material_index);
        ImGui::SameLine();
        if (ImGui::Button("-")) {
          material_index =
              Clamp<i32>(material_index - 1, 0, Core::render_context->voxel_tree.material_arr.size() - 1);
        }
        ImGui::SameLine();
        if (ImGui::Button("+")) {
          material_index =
              Clamp<i32>(material_index + 1, 0, Core::render_context->voxel_tree.material_arr.size() - 1);
        }

        ImGui::ColorEdit4("albedo",
                          (f32 *)&Core::render_context->voxel_tree.material_arr[material_index].albedo);
        ImGui::SliderFloat("roughness", &Core::render_context->voxel_tree.material_arr[material_index].rough,
                           0.0f, 1.0f);
        ImGui::SliderFloat(
            "metallic", &Core::render_context->voxel_tree.material_arr[material_index].metallic, 0.0f, 1.0f);
        ImGui::SliderFloat(
            "reflection", &Core::render_context->voxel_tree.material_arr[material_index].reflect, 0.0f, 1.0f);
        ImGui::SliderFloat(
            "emmision", &Core::render_context->voxel_tree.material_arr[material_index].emmisive, 0.0f, 5.0f);

        Core::VulkanBuffer<Core::BufferType::StagingBuffer> staging_buffer = "material staging buffer";
        staging_buffer.Create(sizeof(Core::Material));
        memcpy(staging_buffer.host_address, &Core::render_context->voxel_tree.material_arr[material_index],
               sizeof(Core::Material));

        Core::VulkanContext::Submit([&](Core::VulkanCommandBuffer &cmd) {
          cmd.BeginDebugPass("material upload");

          Core::VulkanSubPass<Core::SubPassType::Transfer> pass;
          pass.AddDependency<Core::DeviceResourceType::TransferSrc>(staging_buffer);
          pass.AddDependency<Core::DeviceResourceType::TransferDst>(
              Core::render_context->voxel_tree.material_buffer);

          cmd.BindSubPass(pass);

          cmd.UploadBufferToBuffer(staging_buffer, Core::render_context->voxel_tree.material_buffer,
                                   sizeof(Core::Material), 0,
                                   sizeof(Core::VulkanBuffer<Core::BufferType::CountedBuffer>::Header) +
                                       sizeof(Core::Material) * material_index);

          cmd.EndDebugPass();
        });

        ImGui::EndPopup();
      }
      ImGui::SameLine();
    }

    ImGui::End();

    ImGui::PopStyleVar(4);

    ImGui::EndFrame();

    if (!ImGui::GetIO().WantCaptureMouse) {
      if (Core::InputContext::GetPressed(Core::Input::MOUSE_MIDDLE)) {
        Core::Raycast query{};
        query.origin = camera.position;

        const Vec2f32 mouse_pos = Vec2f32::From(Core::InputContext::mouse_pos);

        const Vec4f32 view =
            PerspectiveReverseZInverse(camera.z_near, 1000.0f, camera.fov_y, camera.aspect_ratio) *
            Vec4f32(mouse_pos, 1.0f, 1.0f);

        query.dir = Vec4f32::DownCast<Vec3f32>(
            LookAtInverse(camera.position, camera.position + camera.front, camera.up) *
            Vec4f32(Normalize(Vec4f32::DownCast<Vec3f32>(view)), 0.0f));

        Core::QueueRaycastCmd(query, [&](const Core::RaycastResult &result) {
          Core::QueueFillVolumeCmd(Core::VoxelVolume{
              .min_tree_index = Core::GetTreeIndex(result.hit_position - 5.0f),
              .depth = Core::SparseVoxelTree::MAX_DEPTH,
              .max_tree_index = Core::GetTreeIndex(result.hit_position + 5.0f),
              .material_index = 0,
          });
        });
      }

      if (Core::InputContext::GetHeld(Core::Input::MOUSE_LEFT)) {
        Core::Raycast query{};
        query.origin = camera.position;

        const Vec2f32 mouse_pos = Vec2f32::From(Core::InputContext::mouse_pos);

        const Vec4f32 view =
            PerspectiveReverseZInverse(camera.z_near, 1000.0f, camera.fov_y, camera.aspect_ratio) *
            Vec4f32(mouse_pos, 1.0f, 1.0f);

        query.dir = Vec4f32::DownCast<Vec3f32>(
            LookAtInverse(camera.position, camera.position + camera.front, camera.up) *
            Vec4f32(Normalize(Vec4f32::DownCast<Vec3f32>(view)), 0.0f));

        Core::QueueRaycastCmd(query, [&](const Core::RaycastResult &result) {
          Core::QueueClearVolumeCmd(Core::VoxelVolume{
              .min_tree_index = Core::GetTreeIndex(result.hit_position - 5.0f),
              .max_tree_index = Core::GetTreeIndex(result.hit_position + 5.0f),
          });
        });
      }
    }

    if (!ImGui::GetIO().WantCaptureKeyboard || false) {
      if (Core::InputContext::GetHeld(Core::Input::R) &&
          Core::InputContext::GetHeld(Core::Input::LEFT_CONTROL))
        Core::render_context->RecreatePipelines();

      if (Core::InputContext::GetPressed(Core::Input::ESCAPE))
        Core::Window::SetShouldClose(true);

      if (Core::InputContext::GetHeld(Core::Input::F)) {
        camera.speed = 50.0f;
      } else {
        camera.speed = Abs(Core::SparseVoxelTree::MAX_BOUND);
      }

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
    }

    Core::FlushRaycastCmds();
    Core::FlushClearVolumeCmds();
    Core::FlushFillVolumeCmds();

    Core::BeginFrame(resize);

    Core::Frame(camera);

    if (Core::InputContext::GetHeld(Core::Input::B)) {
      Core::DebugDrawChunkBoundaries();
    }

    Core::VulkanCommandBuffer &cmd = Core::render_context->swapchain.GetActiveCommandBuffer();
    {
      cmd.BeginDebugPass("ui render");
      Core::VulkanSubPass<Core::SubPassType::Graphic> pass;
      pass.AddDependency<Core::DeviceResourceType::ColorAttachment>(
          Core::render_context->swapchain.GetImage());

      cmd.BindSubPass(pass);

      cmd.BeginRendering({&Core::render_context->swapchain.GetImage()}, nullptr,
                         Core::render_context->swapchain.extent, false);
      ImGui::Render();
      ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd.obj);
      cmd.EndRendering();

      cmd.EndDebugPass();
    }

    Core::EndFrame(resize);

    if (resize) {
      Core::Resize(Core::Window::GetSize());
      camera.Resize(Core::render_context->main_image.GetVec2u32());
    }

    if (frame_test_acc < 500.0f) {
      frame_test_acc += timer.ElapsedMillis();
      current_samples++;
    } else {
      const f32 ms_per_frame = frame_test_acc / f32(current_samples);
      Core::Window::SetTitle(
          std::format("{:.2f} ms ({:.0f} fps)", ms_per_frame, 1.0f / (ms_per_frame / 1000.0f)));
      frame_test_acc = 0.0f;
      current_samples = 0;
    }

    delta_time = timer.Elapsed();
  }
}

void Editor::ShutDown() {
  Core::WaitIdle();
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}
