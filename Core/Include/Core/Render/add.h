#pragma once
#include "Core/Render/types.h"
#include "Core/Util/Parse/gltf.h"

namespace Core {
u32 AddDirectionalLight(const DirectionalLight &dir_light);

void AddObject(const ObjectData &object);
}; // namespace Core
