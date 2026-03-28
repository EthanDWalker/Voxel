#pragma once

#include "types.hpp"
#include "vector.hpp"

template <typename T> constexpr const T Radians(const T degrees) {
  return degrees * static_cast<T>(0.01745329251994329576923690768489);
}

template <typename T> constexpr T Min(const T x, const T y) { return (x > y) ? y : x; }
template <typename T> constexpr T Max(const T x, const T y) { return (x < y) ? y : x; }
template <typename T> constexpr T Clamp(const T x, const T min, const T max) {
  return Min(Max(x, min), max);
}

constexpr u32 Pow(u32 x, u32 y) {
  u32 z = 1;
  u32 base = x;
  while (y) {
    if (y & 1) {
      z *= base;
    }
    y >>= 1;
    base *= base;
  }
  return z;
}

template <u8 length> constexpr vec<length, u32> Pow(const vec<length, u32> v, u32 x) {
  vec<length, u32> final;
  for (u8 i = 0; i < length; i++) {
    final[i] = Pow(v[i], x);
  }
  return final;
}

template <typename T> constexpr T Abs(const T v) { return (v < 0) ? -v : v; }

constexpr f32 Round(const f32 v) { return (i32)(v + 0.5f); }

constexpr f32 Cos(const f32 radians) {
  const f32 positive_radians = Abs(radians);
  constexpr const f32 PI_OVER_180 = 0.0174532925199f;
  constexpr const f32 INV_PI_OVER_180 = 57.2957795131;
  const f32 r = ((u32((positive_radians + EPSILON) * INV_PI_OVER_180) % 180)) * PI_OVER_180;

  // 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8! - x^10/10!
  // clang-format off
  const f32 cos = 
    1.0f 
    - r * r * (/*1/2!*/ 0.5f) 
    + r * r * r * r * (/*1/4!*/ 0.0416666666667f) 
    - r * r * r * r * r * r * (/*1/6!*/ 0.00138888888889f) 
    + r * r * r * r * r * r * r * r * (/*1/8!*/ 0.0000248015873016f) 
    - r * r * r * r * r * r * r * r * r * r * (/*1/10!*/ 2.7557319224e-7);

  constexpr const f32 PI_OVER_2 = 1.57079632679f;
  constexpr const f32 INV_2_PI_OVER_180 = 28.6478897565f;
  return cos * (u32(((u32((Abs(positive_radians) + EPSILON) * INV_2_PI_OVER_180) % 180)) / 90.0f) * -2.0f + 1.0f);
  // clang-format on
}

constexpr f32 Sin(const f32 radians) {
  const f32 positive_radians = Abs(radians);
  constexpr const f32 PI_OVER_180 = 0.0174532925199f;
  constexpr const f32 INV_PI_OVER_180 = 57.2957795131;
  const f32 r = ((u32((Abs(positive_radians) + EPSILON) * INV_PI_OVER_180) % 180)) * PI_OVER_180;

  // 1 - x + x^3/3! - x^5/5! + x^7/7! - x^9/9!
  // clang-format off
  const f32 sin = 
    r
    - r * r * r * (/*1/3!*/ 0.166666666667f) 
    + r * r * r * r * r * (/*1/5!*/ 0.00833333333333f) 
    - r * r * r * r * r * r * r *(/*1/7!*/ 0.000198412698413f) 
    + r * r * r * r * r * r * r * r * r * (/*1/9!*/ 0.0000027557319224f);

  constexpr const f32 PI_OVER_2 = 1.57079632679f;
  constexpr const f32 INV_2_PI_OVER_180 = 28.6478897565f;
  return sin * (u32(((u32((Abs(positive_radians) + EPSILON) * INV_2_PI_OVER_180) % 180)) / 90.0f) * -2.0f + 1.0f);
  // clang-format on
}

static u32 FloatAsU32(const float x) { return *(u32 *)&x; }
static f32 U32AsFloat(const u32 x) { return *(f32 *)&x; }

struct f16 {
  u16 data;

  void FromF32(const f32 data) {
    // IEEE-754 16-bit floating-point format (without infinity): 1-5-10,
    // exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const u32 b = FloatAsU32(data) +
                  0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
    const u32 e = (b & 0x7F800000) >> 23; // exponent
    const u32 m = b & 0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 =
                                  // decimal indicator flag - initial rounding
    this->data = (b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
                 ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
                 (e > 143) * 0x7FFF; // sign : normalized : denormalized : saturate
  }

  f16() { FromF32(0.0f); }
  f16(const f32 data) { FromF32(data); }

  operator f32() const {
    // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15,
    // +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const u32 e = (this->data & 0x7C00) >> 10; // exponent
    const u32 m = (this->data & 0x03FF) << 13; // mantissa
    const u32 v = FloatAsU32((float)m) >>
                  23; // evil log2 bit hack to count leading zeros in denormalized format
    return U32AsFloat(
        (this->data & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
        ((e == 0) & (m != 0)) *
            ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000))); // sign : normalized : denormalized
  }
};

static u16 PackUnorm16(const f32 v, const f32 min, const f32 max) {
  const f32 n = Clamp(Abs((v - min) / (max - min)), 0.0f, 1.0f);
  return (u16)Round(n * 65535.0f);
}

static u32 PackSnorm10(f32 v) {
  int i = (int)Round(Clamp(v, -1.0f, 1.0f) * 511.0f);
  i = Clamp(i, -511, 511);
  return (u32)(i & 0x3FF);
}

static u32 PackRGB10A2(const Vec3f32 n, u32 a) {
  return (PackSnorm10(n.x)) | ((PackSnorm10(n.y) << 10)) | ((PackSnorm10(n.z) << 20)) |
         ((a & 0x3) << 30);
}
