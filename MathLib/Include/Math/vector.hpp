#pragma once
#include "types.hpp"
#include <assert.h>
#include <bit>
#include <limits>
#include <string>

template <typename Derived, u8 length, typename EntryType> struct vec_base {
  Derived &Get() { return static_cast<Derived &>(*this); }

  constexpr const Derived &Get() const { return static_cast<const Derived &>(*this); }

  template <typename U> static constexpr Derived From(const U data) {
    static_assert(sizeof(U) == sizeof(Derived), "Types should be of equal size");
    return std::bit_cast<Derived>(data);
  }

  template <typename U> static constexpr U To(const Derived data) {
    static_assert(sizeof(U) == sizeof(Derived), "Types should be of equal size");
    return std::bit_cast<U>(data);
  }

  template <typename U> static constexpr U DownCast(const Derived data) {
    static_assert(sizeof(U) < sizeof(Derived), "If sizes are equal use To() fn");
    U final;
    for (u8 i = 0; i < sizeof(U) / sizeof(EntryType); i++) {
      final[i] = data[i];
    }
    return final;
  }

  constexpr const EntryType &operator[](const u8 index) const {
    assert(index < length && "Index out of bounds");
    return *((const EntryType *const)&Get() + index);
  }

  constexpr EntryType &operator[](const u8 index) {
    assert(index < length && "Index out of bounds");
    return *((EntryType *const)&Get() + index);
  }

  constexpr Derived operator+(const Derived other) const {
    Derived final;
    for (u8 i = 0; i < length; i++) {
      final[i] = Get()[i] + other[i];
    }
    return final;
  }

  constexpr void operator+=(const Derived other) { Get() = Get() + other; }
  constexpr void operator-=(const Derived other) { Get() = Get() - other; }
  constexpr void operator*=(const Derived other) { Get() = Get() * other; }
  constexpr void operator/=(const Derived other) { Get() = Get() / other; }

  friend constexpr Derived operator*(const EntryType s, const Derived v) { return v * s; }
  friend constexpr Derived operator/(const EntryType s, const Derived v) { return v / s; }
  constexpr Derived operator-() const { return Get() * -1.0f; }

  constexpr Derived operator-(const Derived other) const {
    Derived final;
    for (u8 i = 0; i < length; i++) {
      final[i] = Get()[i] - other[i];
    }
    return final;
  }

  constexpr Derived operator*(const EntryType other) const {
    Derived final;
    for (u8 i = 0; i < length; i++) {
      final[i] = Get()[i] * other;
    }
    return final;
  }

  constexpr Derived operator/(const EntryType other) const {
    Derived final;
    for (u8 i = 0; i < length; i++) {
      final[i] = Get()[i] / other;
    }
    return final;
  }

  constexpr Derived operator/(const Derived other) const {
    Derived final;
    for (u8 i = 0; i < length; i++) {
      final[i] = Get()[i] / other[i];
    }
    return final;
  }
  constexpr Derived operator*(const Derived other) const {
    Derived final;
    for (u8 i = 0; i < length; i++) {
      final[i] = Get()[i] * other[i];
    }
    return final;
  }

  constexpr const std::string String() const {
    std::string v_string = "{";
    for (u8 i = 0; i < length; i++) {
      v_string += std::to_string((*this)[i]);
      if (i != length - 1) {
        v_string += ',';
      }
    }
    v_string += '}';
    return v_string;
  };
};

template <u8 length, typename T> struct vec : vec_base<vec<length, T>, length, T> {};

template <typename T> struct vec<2, T> : vec_base<vec<2, T>, 2, T> {
  // clang-format off
  union { T x, r, width; };
  union { T y, g, height; };
  // clang-format on

  constexpr vec<2, T>() : x(0), y(0) {}
  constexpr vec<2, T>(const T fill) : x(fill), y(fill) {}
  constexpr vec<2, T>(const T x, const T y) : x(x), y(y) {}
};

template <typename T> struct vec<3, T> : vec_base<vec<3, T>, 3, T> {
  // clang-format off
  union { T x, r, width; };
  union { T y, g, height; };
  union { T z, b, depth; };
  // clang-format on

  constexpr vec<3, T>() : x(0), y(0), z(0) {}
  constexpr vec<3, T>(const T fill) : x(fill), y(fill), z(fill) {}
  constexpr vec<3, T>(const vec<2, T> v, const T z) : x(v.x), y(v.y), z(z) {}
  constexpr vec<3, T>(const T x, const T y, const T z) : x(x), y(y), z(z) {}
};

template <typename T> struct vec<4, T> : vec_base<vec<4, T>, 4, T> {
  // clang-format off
  union { T x, r, width; };
  union { T y, g, height; };
  union { T z, b, depth; };
  union { T w, a; };
  // clang-format on

  constexpr vec<4, T>() : x(0), y(0), z(0), w(0) {}
  constexpr vec<4, T>(const T fill) : x(fill), y(fill), z(fill), w(fill) {}
  constexpr vec<4, T>(const vec<2, T> v, const T z, const T w) : x(v.x), y(v.y), z(z), w(w) {}
  constexpr vec<4, T>(const vec<3, T> v, const T w) : x(v.x), y(v.y), z(v.z), w(w) {}
  constexpr vec<4, T>(const T x, const T y, const T z, const T w) : x(x), y(y), z(z), w(w) {}
};

template <u8 length, typename T>
constexpr const f32 Dot(const vec<length, T> v0, const vec<length, T> v1) {
  f32 total = 0;
  for (u8 i = 0; i < length; i++) {
    total += v0[i] * v1[i];
  }
  return total;
}

template <typename T> constexpr const vec<3, T> Cross(const vec<3, T> v0, const vec<3, T> v1) {
  // clang-format off
  return vec<3, T>(
    v0.y * v1.z - v0.z * v1.y,
    v0.z * v1.x - v0.x * v1.z,
    v0.x * v1.y - v0.y * v1.x
  );
  // clang-format on
}

template <u8 length, typename T> constexpr const T Magnitude(const vec<length, T> v) {
  f32 total = 0;
  for (u8 i = 0; i < length; i++) {
    total += v[i] * v[i];
  }
  return total == 0 ? 1.0f : sqrt(total);
}

template <u8 length, typename T> constexpr const T Magnitude2(const vec<length, T> v) {
  f32 total = 0;
  for (u8 i = 0; i < length; i++) {
    total += v[i] * v[i];
  }
  return total;
}

template <u8 length, typename T> constexpr const vec<length, T> Normalize(const vec<length, T> v) {
  return v / Magnitude(v);
}

template <typename OtherT, u8 length, typename T>
constexpr vec<length, OtherT> VecTypeCast(const vec<length, T> from) {
  vec<length, OtherT> final;
  for (u8 i = 0; i < length; i++) {
    final[i] = static_cast<OtherT>(from[i]);
  }
  return final;
}

template <u8 length, typename T>
constexpr vec<length, T> Min(const vec<length, T> v0, const vec<length, T> v1) {
  vec<length, T> final;
  for (u8 i = 0; i < length; i++) {
    final[i] = (v0[i] < v1[i]) ? v0[i] : v1[i];
  }
  return final;
}

template <u8 length, typename T>
constexpr T MaxElement(const vec<length, T> v) {
  T final = std::numeric_limits<T>::min();
  for (u8 i = 0; i < length; i++) {
    final = v[i] > final ? v[i] : final;
  }
  return final;
}

template <u8 length, typename T>
constexpr vec<length, T> Max(const vec<length, T> v0, const vec<length, T> v1) {
  vec<length, T> final;
  for (u8 i = 0; i < length; i++) {
    final[i] = (v0[i] > v1[i]) ? v0[i] : v1[i];
  }
  return final;
}

template <u8 length, typename T> constexpr vec<length, T> Ceil(const vec<length, T> v) {
  vec<length, T> final;
  for (u8 i = 0; i < length; i++) {
    final[i] = ceil(v[i]);
  }
  return final;
}

template <u8 length, typename T> constexpr vec<length, T> Floor(const vec<length, T> v) {
  vec<length, T> final;
  for (u8 i = 0; i < length; i++) {
    final[i] = floor(v[i]);
  }
  return final;
}

template <u8 length, typename T> constexpr vec<length, T> Abs(const vec<length, T> v) {
  vec<length, T> final;
  for (u8 i = 0; i < length; i++) {
    final[i] = abs(v[i]);
  }
  return final;
}

template <u8 length, typename T> constexpr vec<length, T> Sign(const vec<length, T> v) {
  vec<length, T> final;
  for (u8 i = 0; i < length; i++) {
    final[i] = v[i] == 0 ? 1 : (v[i] < 0 ? -1 : 1);
  }
  return final;
}

template <u8 length, typename T>
constexpr const vec<length, bool> operator<(const vec<length, T> v0, const vec<length, T> v1) {
  vec<length, bool> final;
  for (u8 i = 0; i < length; i++) {
    final[i] = v0[i] < v1[i];
  }
  return final;
}

template <u8 length, typename T>
constexpr const vec<length, bool> operator>(const vec<length, T> v0, const vec<length, T> v1) {
  vec<length, bool> final;
  for (u8 i = 0; i < length; i++) {
    final[i] = v0[i] > v1[i];
  }
  return final;
}

template <u8 length, typename T>
constexpr const vec<length, bool> operator>=(const vec<length, T> v0, const vec<length, T> v1) {
  vec<length, bool> final;
  for (u8 i = 0; i < length; i++) {
    final[i] = v0[i] >= v1[i];
  }
  return final;
}

template <u8 length, typename T>
constexpr const vec<length, bool> operator<=(const vec<length, T> v0, const vec<length, T> v1) {
  vec<length, bool> final;
  for (u8 i = 0; i < length; i++) {
    final[i] = v0[i] <= v1[i];
  }
  return final;
}

template <u8 length, typename T>
constexpr const vec<length, bool> operator==(const vec<length, T> v0, const vec<length, T> v1) {
  vec<length, bool> final;
  for (u8 i = 0; i < length; i++) {
    final[i] = v0[i] == v1[i];
  }
  return final;
}

template <u8 length, typename T>
constexpr const vec<length, bool> operator!=(const vec<length, T> v0, const vec<length, T> v1) {
  vec<length, bool> final;
  for (u8 i = 0; i < length; i++) {
    final[i] = v0[i] != v1[i];
  }
  return final;
}

template <u8 length> constexpr const bool All(const vec<length, bool> v) {
  for (u8 i = 0; i < length; i++) {
    if (!v[i])
      return false;
  }
  return true;
}

template <u8 length> constexpr const bool Any(const vec<length, bool> v) {
  for (u8 i = 0; i < length; i++) {
    if (v[i])
      return true;
  }
  return false;
}

template <u8 length, typename T>
constexpr const vec<length, T> Select(const vec<length, bool> v, const vec<length, T> if_true,
                                      const vec<length, T> if_false) {
  vec<length, T> final;
  for (u8 i = 0; i < length; i++) {
    final[i] = v[i] ? if_true[i] : if_false[i];
  }
  return final;
}

using Vec2f32 = vec<2, f32>;
using Vec3f32 = vec<3, f32>;
using Vec4f32 = vec<4, f32>;

using Vec2f64 = vec<2, f64>;
using Vec3f64 = vec<3, f64>;
using Vec4f64 = vec<4, f64>;

using Vec2u32 = vec<2, u32>;
using Vec3u32 = vec<3, u32>;
using Vec4u32 = vec<4, u32>;

using Vec2i32 = vec<2, i32>;
using Vec3i32 = vec<3, i32>;
using Vec4i32 = vec<4, i32>;
