#include "Include/Math/matrix.hpp"
#include "Include/Math/vector.hpp"
#include "Include/Math/quaternion.hpp"
#include "Include/Math/other.hpp"
#include <assert.h>
#include <format>
#include <iostream>

template <typename T> static void Log(T val) {
  std::cout << "[MathLib] " << std::format("{}", val) << '\n';
}

template <class... Types> static void Log(std::format_string<Types...> fmt, Types &&...args) {
  std::cout << "[MathLib] " << std::format(fmt, std::forward<Types>(args)...) << '\n';
}

template <u8 rows, u8 cols, typename T>
const void TestEqual(const mat<rows, cols, T> x, const mat<rows, cols, T> y) {
  // Log("Test: {} == {} Result: {}", x.String(), y.String(), x == y ? "Pass" : "FAIL");
}

template <u8 length, typename T>
const void TestEqual(const vec<length, T> x, const vec<length, T> y) {
  Log("Test: {} == {} Result: {}", x.String(), y.String(), x == y ? "Pass" : "FAIL");
}

template <typename T, typename U> const void TestEqual(const T x, const U y) {
  Log("Test: {} == {} Result: {}", x, y, x == y ? "Pass" : "FAIL");
}

int main() {
  Vec3f32 i = Vec3f32(1.0f, 0.0f, 0.0f);
  Vec3f32 j = Vec3f32(0.0f, 1.0f, 0.0f);
  Vec3f32 k = Vec3f32(0.0f, 0.0f, 1.0f);
  TestEqual(Cross(i, j), k);
  TestEqual(i - i, Vec3f32());
  TestEqual(Dot(i + i, i), 2.0f);
  TestEqual(Mat4f32(1.0f) * 2.0f, Mat4f32(2.0f));
  TestEqual(Det(Mat4f32(2.0f)), 16.0f);

  Quat q = Quat(i, Radians(45.0f));

  Log(Rotate(q, j).String());
  return 0;
}
