#pragma once
#include "Core/Render/types.h"
#include "Core/Util/Parse/gltf.h"

namespace Core {
u32 AddDirectionalLight(const DirectionalLight &dir_light);

Mesh AddMesh(const MeshData &mesh_data);
}; // namespace Core
