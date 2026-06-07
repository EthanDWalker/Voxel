#pragma once
#include "Core/Render/types.h"
#include "Core/Util/Parse/object.h"

namespace Core {
u32 AddDirectionalLight(const DirectionalLight &dir_light);

void AddObject(const ObjectData &object, const u32 max_depth);

void VoxelizeTerrain(const Vec2u32 extent, const Vec2f32 center, const f32 density, const i32 seed, const u32 max_depth);
}; // namespace Core
