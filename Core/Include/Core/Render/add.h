#pragma once
#include "Core/Render/types.h"
#include "Core/Util/Parse/object.h"

namespace Core {
u32 AddDirectionalLight(const DirectionalLight &dir_light);

void AddObject(const ObjectData &object, const u32 max_depth);

void VoxelizeChunk(const Vec3u32 index, const u32 seed, const u32 max_depth);
}; // namespace Core
