#pragma once
#include <cmath>
#include <string>

#include "matrix.hpp"
#include "vector.hpp"

struct Quat {
  f32 x, y, z, w;

  constexpr Quat() : x(0.0f), y(0.0f), z(0.0f), w(1.0f) {}

  Quat(const vec<3, f32> n, const f32 radians) {
    const f32 half_sin_theta_n = std::sin(radians * 0.5f) / Magnitude(n);
    x = n.x * half_sin_theta_n;
    y = n.y * half_sin_theta_n;
    z = n.z * half_sin_theta_n;
    w = std::cos(radians * 0.5f);
  }

  // clang-format off
  constexpr Quat(const f32 x, const f32 y, const f32 z, const f32 w) : x(x), y(y), z(z), w(w) {}

  constexpr f32 &operator [] (const u8 index) const { return *((f32 *)this + index); }
  constexpr Quat operator + (const Quat o) const { return Quat(x + o.x, y + o.y, z + o.z, w + o.w); }
  constexpr Quat operator - (const Quat o) const { return Quat(x - o.x, y - o.y, z - o.z, w - o.w); }
  constexpr Quat operator * (const f32 s) const { return Quat(x * s, y * s, z * s, w * s); }
  constexpr Quat operator / (const f32 s) const { assert(s != 0.0f); return *this * (1.0f / s); }

  constexpr const Quat operator*(const Quat o) const {
    return Quat(
      w * o.x + x * o.w + y * o.z - z * o.y, // i
      w * o.y - x * o.z + y * o.w + z * o.x, // j
      w * o.z + x * o.y - y * o.x + z * o.w, // k
      w * o.w - x * o.x - y * o.y - z * o.z // w
    );
  }
  constexpr Quat operator*(const Quat o) {
    return Quat(
      w * o.x + x * o.w + y * o.z - z * o.y, // i
      w * o.y - x * o.z + y * o.w + z * o.x, // j
      w * o.z + x * o.y - y * o.x + z * o.w, // k
      w * o.w - x * o.x - y * o.y - z * o.z // w
    );
  }
  // clang-format on

  constexpr operator vec<3, f32>() const { return {x, y, z}; }

  constexpr const std::string String() const {
    std::string final = "";
    return final + std::to_string(w) + " + " + std::to_string(x) + "i + " + std::to_string(y) +
           "j + " + std::to_string(z) + "k";
  }
};

inline f32 Magnitude(const Quat v) {
  f32 total = 0;
  for (u8 i = 0; i < 4 /* Quat size */; i++) {
    total += v[i] * v[i];
  }
  return sqrt(total);
}

constexpr const f32 Magnitude2(const Quat v) {
  f32 total = 0;
  for (u8 i = 0; i < 4 /* Quat size */; i++) {
    total += v[i] * v[i];
  }
  return total;
}

constexpr Quat Normalize(const Quat q) { return q / Magnitude(q); }

constexpr Quat Conjugate(const Quat q) { return Quat(-q.x, -q.y, -q.z, q.w); };

constexpr Quat Inverse(const Quat q) { return Conjugate(q) / Magnitude2(q); };

constexpr vec<3, f32> Rotate(const Quat q, const vec<3, f32> v) {
  return q * Quat(v.x, v.y, v.z, 0.0f) * Inverse(q);
}

constexpr Quat FromRotationMatrix(const Mat4f32 &m) {
  f32 t;
  Quat q;
  if (m[2][2] < 0) {
    if (m[0][0] > m[1][1]) {
      t = 1.0f + m[0][0] - m[1][1] - m[2][2];
      q = Quat(t, m[1][0] + m[0][1], m[0][2] + m[2][0], m[2][1] - m[1][2]);
    } else {
      t = 1.0f - m[0][0] + m[1][1] - m[2][2];
      q = Quat(m[1][0] + m[0][1], t, m[2][1] + m[1][2], m[0][2] - m[2][0]);
    }
  } else {
    if (m[0][0] < -m[1][1]) {
      t = 1.0f - m[0][0] - m[1][1] + m[2][2];
      q = Quat(m[0][2] + m[2][0], m[2][1] + m[1][2], t, m[1][0] - m[0][1]);
    } else {
      t = 1.0f + m[0][0] + m[1][1] + m[2][2];
      q = Quat(m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1], t);
    }
  }
  q = q * (0.5f / sqrt(t));
  return q;
}

constexpr Mat3f32 ToRotationMatrix(const Quat q) {
  // clang-format off
  return Mat3f32(
Vec3f32(2.0f * (q.w * q.w + q.x * q.x) - 1.0f, 2.0f * (q.x * q.y - q.w * q.z), 2.0f * (q.x * q.z + q.w * q.y)),
Vec3f32(2.0f * (q.x * q.y + q.w * q.z), 2.0f * (q.w * q.w + q.y * q.y) - 1.0f, 2.0f * (q.y * q.z - q.w * q.x)),
Vec3f32(2.0f * (q.x * q.z - q.w * q.y), 2.0f * (q.y * q.z + q.w * q.x), 2.0f * (q.w * q.w + q.z * q.z) - 1.0f)
  );
  // clang-format on
}

inline Vec3f32 ToEuler(const Quat q) {
  return Vec3f32(atan(2.0f * (q.w * q.x + q.y * q.z) / (1.0f - 2.0f * (q.x * q.x + q.y * q.y))),
                 asin(2.0f * (q.w * q.y - q.z * q.x)),
                 atan(2.0f * (q.w * q.z + q.x * q.y) / (1.0f - 2.0f * (q.y * q.y + q.z * q.z))));
}
