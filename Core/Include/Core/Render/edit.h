#pragma once
#include "Core/Render/types.h"
#include <functional>

namespace Core {
u32 AddDirectionalLight(const DirectionalLight &dir_light);
void ClearVolume(const VoxelVolume &volume);

void QueueRaycast(const Raycast &raycast, const std::function<void(const RaycastResult &result)> &&callback);
void FlushRaycasts();
}; // namespace Core
