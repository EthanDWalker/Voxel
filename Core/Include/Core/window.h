#pragma once

namespace Core {
struct Window {
  static void *handle;

  static Vec2u32 GetSize();
  static bool ShouldClose();
  static void SetShouldClose(const bool v);
  static void SetTitle(const std::string &title);
};
} // namespace Core
