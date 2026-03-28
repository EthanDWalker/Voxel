#include "Core/Render/camera.h"

namespace Core {
void Camera::Create(const Vec2u32 extent) {
  aspect_ratio = extent.width / static_cast<f32>(extent.height);
}

void Camera::Resize(const Vec2u32 extent) {
  aspect_ratio = extent.width / static_cast<f32>(extent.height);
}

void Camera::Update() {
  const Vec3f32 direction{
      std::cos(Radians(yaw)) * std::cos(Radians(pitch)),
      std::sin(Radians(pitch)),
      std::sin(Radians(yaw)) * std::cos(Radians(pitch)),
  };
  front = Normalize(direction);
  right = Normalize(Cross(front, world_up));
  up = Normalize(Cross(right, front));

  ubo.view = LookAt(position, position + front, up);
  ubo.projection = InfinitePerspectiveReverseZ(z_near, fov_y, aspect_ratio);
  ubo.position = position;
  ubo.front = front;
  ubo.up = up;
  ubo.fov_y = fov_y;
  ubo.aspect_ratio = aspect_ratio;
  ubo.z_near = z_near;
}

} // namespace Core
