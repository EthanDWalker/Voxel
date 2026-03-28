#pragma once

#include "matrix.hpp"
#include "vector.hpp"

inline Mat4f32 InfinitePerspectiveReverseZ(const f32 z_near, const f32 fov_y,
                                           const f32 aspect_ratio) {
  const f32 tan_half_fov_y = tan(fov_y / 2.0f);

  Mat4f32 proj;

  proj[0][0] = 1.0f / (aspect_ratio * tan_half_fov_y);
  proj[1][1] = -1.0f / tan_half_fov_y;
  proj[2][3] = -1.0f;
  proj[3][2] = z_near;

  return proj;
}

constexpr const Mat4f32 Ortho(f32 l, f32 r, f32 b, f32 t, f32 n, f32 f) {
  return Mat4f32(Vec4f32(1.0f, 0.0f, 0.0f, 0.0f), Vec4f32(0.0f, 1.0f, 0.0f, 0.0f),
                 Vec4f32(0.0f, 0.0f, 0.5f, 0.0f), Vec4f32(0.0f, 0.0f, 0.5f, 1.0f)) *
         Mat4f32(Vec4f32(2.0f / (r - l), 0.0f, 0.0f, -(r + l) / (r - l)),
                 Vec4f32(0.0f, 2.0f / (t - b), 0.0f, -(t + b) / (t - b)),
                 Vec4f32(0.0f, 0.0f, -2.0f / (f - n), -(f + n) / (f - n)),
                 Vec4f32(0.0f, 0.0f, 0.0f, 1.0f));
}

inline const Mat4f32 PerspectiveReverseZ(const f32 z_near, const f32 z_far, const f32 fov_y,
                                         const f32 aspect_ratio) {

  const f32 tan_half_fov_y = tan(fov_y / 2.0f);

  Mat4f32 proj;
  proj[0][0] = 1.0f / (aspect_ratio * tan_half_fov_y);
  proj[1][1] = 1.0f / (tan_half_fov_y);
  proj[2][2] = z_near / (z_far - z_near);
  proj[2][3] = -1.0f;
  proj[3][2] = z_far * z_near / (z_far - z_near);
  return proj;
}

inline const Mat4f32 InfinitePerspectiveReverseZInverse(const f32 z_near, const f32 fov_y,
                                                        const f32 aspect_ratio) {
  const f32 tan_half_fov_y = tan(fov_y / 2.0f);

  Mat4f32 inv_proj;

  inv_proj[0][0] = aspect_ratio * tan_half_fov_y;
  inv_proj[1][1] = -tan_half_fov_y;
  inv_proj[2][3] = 1.0f / z_near;
  inv_proj[3][2] = -1.0f;

  return inv_proj;
}

inline const Mat4f32 PerspectiveReverseZInverse(const f32 z_near, const f32 z_far, const f32 fov_y,
                                                const f32 aspect_ratio) {
  const f32 tan_half_fov_y = tan(fov_y / 2.0f);

  Mat4f32 inv_proj;
  inv_proj[0][0] = aspect_ratio * tan_half_fov_y;
  inv_proj[1][1] = -tan_half_fov_y;
  inv_proj[2][3] = 1.0f / ((z_far / (z_near - z_far)) * (z_near - 1.0f));
  inv_proj[3][2] = -1.0f;
  inv_proj[3][3] = 1.0f / (z_near - 1.0f);
  return inv_proj;
}

inline const Mat4f32 LookAt(const Vec3f32 eye, const Vec3f32 center, const Vec3f32 up) {
  const vec f = Normalize(center - eye);
  const vec s = Normalize(Cross(f, up));
  const vec u = Normalize(Cross(s, f));

  // clang-format off
  Mat4f32 view = Mat4f32(
    Vec4f32(s, -Dot(s, eye)),
    Vec4f32(u, -Dot(u, eye)),
    Vec4f32(f * -1.0f, Dot(f, eye)),
    Vec4f32(0.0f, 0.0f, 0.0f, 1.0f)
  );
  // clang-format on

  return view;
}

inline const Mat4f32 LookAtInverse(const Vec3f32 eye, const Vec3f32 center, const Vec3f32 up) {
  const vec f = Normalize(center - eye);
  const vec s = Normalize(Cross(f, up));
  const vec u = Normalize(Cross(s, f));

  // clang-format off
  Mat4f32 inv_view = Mat4f32(
    Vec4f32(s.x, u.x, -f.x, eye.x),
    Vec4f32(s.y, u.y, -f.y, eye.y),
    Vec4f32(s.z, u.z, -f.z, eye.z),
    Vec4f32(0.0f, 0.0f, 0.0f, 1.0f)
  );
  // clang-format on

  return inv_view;
}
