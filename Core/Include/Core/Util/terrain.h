#pragma once

#include "Core/Util/Parse/object.h"

namespace Core {
void GenerateTerrainObject(const Vec2u32 extent, const Vec2f32 density, const i32 seed, ObjectData &object_data);
};
