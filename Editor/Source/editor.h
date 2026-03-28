#pragma once
#include "Core/Render/camera.h"

struct Editor {
  Core::Camera camera;

  void StartUp();

  void Run();

  void ShutDown();
};
