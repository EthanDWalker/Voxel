#pragma once

#include "camera.h"

namespace Core {
void Frame(Camera &camera);
void BeginFrame(bool &resize);
void EndFrame(bool &resize);
void WaitIdle();
void Resize(Vec2u32 extent);
} // namespace Core
