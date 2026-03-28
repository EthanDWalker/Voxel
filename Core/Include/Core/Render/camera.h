#pragma once
#include "volk.h"
#include "types.h"

namespace Core {
struct Camera {
  struct alignas(GPU_ALIGNMENT) UBO {
    Mat4f32 projection;
    Mat4f32 view;
    Vec3f32 position;
    f32 aspect_ratio;
    Vec3f32 front;
    f32 fov_y;
    Vec3f32 up;
    f32 z_near;
  };

  UBO ubo;

  Vec3f32 position = {0.0f, 0.0f, 1.0f};
  Vec3f32 velocity = {0.0f, 0.0f, 0.0f};

  Vec3f32 front = {0.0f, 0.0f, -1.0f};
  const Vec3f32 world_up = Vec3f32(0.0f, 1.0f, 0.0f);
  Vec3f32 right = Cross(front, world_up);
  Vec3f32 up = Cross(right, front);

  f32 yaw = -90.0f;
  f32 pitch = 0.0f;

  f32 speed = 5.0f;
  f32 sensitivity = 100.0f;

  f32 aspect_ratio;

  f32 z_near = 0.1f;

  f32 fov_y = Radians(70.0f);

  void Create(const Vec2u32 extent);
  void Resize(const Vec2u32 extent);

  void Update();
};
} // namespace Core
