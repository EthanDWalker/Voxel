#pragma once
#include "Core/Render/camera.h"

struct Editor {
  Core::Camera camera;
  u32 material_index = 0;

  void StartUp();

  void Run();

  void ShutDown();
};
