#pragma once
#include <type_traits>
#include "vector.hpp"
#include "types.hpp"

template <u8 rows, u8 cols, typename T> struct mat {
  vec<cols, vec<rows, T>> data;

  template <typename CastType> constexpr static CastType To(const mat &m) {
    static_assert(sizeof(CastType) == sizeof(mat), "Types must be of equal size");
    return *(CastType *)&m;
  }

  template <typename CastType> constexpr static mat From(const CastType &m) {
    static_assert(sizeof(CastType) == sizeof(mat), "Types must be of equal size");
    return *(mat *)&m;
  }

  constexpr mat() : data(0.0f) {}

  template <typename... Rows> constexpr mat(const Rows... args) {
    static_assert(sizeof...(Rows) == rows, "Incorrect row count");
    u8 i = 0;
    (SetRow(i++, args), ...);
  }

  constexpr mat(T value) {
    static_assert(rows == cols, "Must have a square matrix to fill diagonal");
    for (u8 i = 0; i < rows; i++) {
      data[i][i] = value;
    }
  }

  constexpr void SetRow(const u8 r, const vec<cols, T> v) {
    for (u8 col = 0; col < cols; col++) {
      data[col][r] = v[col];
    }
  }

  template <u8 other_rows, u8 other_cols, typename O>
  mat<rows, other_rows, O> constexpr operator*(const mat<other_rows, other_cols, O> &other) const {
    static_assert(std::is_same_v<T, O>, "Matrixes must have same type of data");
    static_assert(cols == other_rows, "Invalid matrix multiplication");
    mat<rows, other_cols, T> final;
    for (u8 row = 0; row < rows; row++) {
      for (u8 col = 0; col < other_cols; col++) {
        for (u8 v = 0; v < cols; v++) {
          final.data[col][row] += this->data[v][row] * other.data[col][v];
        }
      }
    }
    return final;
  }

  constexpr vec<rows, T> operator*(const vec<rows, T> other) const {
    vec<rows, T> final = 0.0f;
    for (u8 row = 0; row < rows; row++) {
      for (u8 col = 0; col < cols; col++) {
        final[row] += data[col][row] * other[col];
      }
    }
    return final;
  }

  mat<rows, cols, T> constexpr operator*(const T s) const {
    mat<rows, cols, T> final;
    for (u8 row = 0; row < rows; row++) {
      for (u8 col = 0; col < cols; col++) {
        final[col][row] = this->data[col][row] * s;
      }
    }
    return final;
  }

  constexpr const vec<rows, T> &operator[](const u8 index) const {
    assert(index < cols && "Index out of bounds");
    return data[index];
  }

  constexpr vec<rows, T> &operator[](const u8 index) {
    assert(index < cols && "Index out of bounds");
    return data[index];
  }

  constexpr const std::string String() const {
    std::string string;
    string += '\n';
    for (u8 row = 0; row < rows; row++) {
      string += '|';
      string += ' ';
      for (u8 col = 0; col < cols; col++) {
        string += std::to_string(data[col][row]);
        string += ' ';
      }
      string += '|';
      string += '\n';
    }
    return string;
  }
};

// https://informatika.stei.itb.ac.id/~rinaldi.munir/Matdis/2016-2017/Makalah2016/Makalah-Matdis-2016-051.pdf
template <u8 rows, u8 cols, typename T>
constexpr const void SwapRowsWithZeroPivot(mat<rows, cols, T> &m, bool &singular) {
  static_assert(rows == cols, "only accepts square matrices");
  singular = false;
  u8 i = 0;

  while (i < rows && !singular) {
    if (m[i][i] == 0) {
      u8 j = 0;
      while (j < rows && m[j][i] == 0) {
        j++;
        if (m[j][i] != 0) {
          for (u8 k = 0; k < rows; k++) {
            T temp = m[i][k];
            m[i][k] = m[j][k];
            m[j][k] = -temp;
          }
        } else {
          singular = true;
        }
      }
    }
    i++;
  }
}

// https://informatika.stei.itb.ac.id/~rinaldi.munir/Matdis/2016-2017/Makalah2016/Makalah-Matdis-2016-051.pdf
template <u8 rows, u8 cols, typename T> constexpr const T Det(const mat<rows, cols, T> &m) {
  static_assert(rows == cols, "Cannot take determinate of non-square matrix");
  mat U = m;
  bool singular = false;
  SwapRowsWithZeroPivot(U, singular);
  if (singular)
    return 0.0f;

  for (u8 i = 0; i < rows; i++) {
    for (u8 j = i + 1; j < rows; j++) {
      T ratio = U[i][j] / U[i][i];
      for (u8 k = 0; k < rows; k++) {
        U[k][j] = U[k][j] - ratio * U[k][i];
      }
    }
  }

  T det = static_cast<T>(1.0f);

  for (u8 i = 0; i < rows; i++) {
    det = det * U[i][i];
  }
  return det;
}

using Mat3f32 = mat<3, 3, f32>;
using Mat4f32 = mat<4, 4, f32>;
