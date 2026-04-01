#pragma once
#include <chrono>
#include <iostream>

using f32 = float;

namespace Core {
struct Timer {
  Timer() { Reset(); }
  void Reset() { m_start = std::chrono::high_resolution_clock::now(); }
  f32 Elapsed() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - m_start)
               .count() *
           0.001f * 0.001f;
  }
  f32 ElapsedMillis() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - m_start)
               .count() *
           0.001f;
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

#define SCOPED_TIMER(x) Core::ScopedTimer scoped_timer{x};

struct ScopedTimer {
  ScopedTimer(std::string name) {
    m_start = std::chrono::high_resolution_clock::now();
    this->name = name;
  }

  ~ScopedTimer() {
    std::cout << std::format("[ScopedTimer]: {} : {} ms\n", name,
                             ElapsedMillis());
  }

  f32 ElapsedMillis() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - m_start)
               .count() *
           0.001f;
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
  std::string name;
};
} // namespace Core
