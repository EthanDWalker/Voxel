#pragma once

#include "Core/Render/types.h"
#include <functional>

namespace Core {
void QueueClearVolumeCmd(const VoxelVolume &volume);
void FlushClearVolumeCmds();

void QueueRaycastCmd(const Raycast &raycast,
                     const std::function<void(const RaycastResult &result)> &&callback);
void FlushRaycastCmds();
} // namespace Core
