#include "Core/window.h"
#include "GLFW/glfw3.h"

namespace Core {
void *Window::handle = nullptr;

Vec2u32 Window::GetSize() {
  ZoneScoped;
  Vec2u32 size;
  glfwGetWindowSize((GLFWwindow *)handle, (i32 *)&size.width, (i32 *)&size.height);

  return size;
}

bool Window::ShouldClose() {
  ZoneScoped;
  return glfwWindowShouldClose((GLFWwindow *)handle);
}

void Window::SetShouldClose(const bool v) {
  ZoneScoped;
  glfwSetWindowShouldClose((GLFWwindow *)handle, v);
}

void Window::SetTitle(const std::string &title) {
  ZoneScoped;
  glfwSetWindowTitle((GLFWwindow *)handle, title.c_str());
}
} // namespace Core
