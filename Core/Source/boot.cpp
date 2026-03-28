#include "Core/Render/Vulkan/context.h"
#include "Core/Util/thread_pool.h"
#include "Core/input.h"
#include "Core/window.h"
#include "GLFW/glfw3.h"

namespace Core {
void StartUp() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  const u32 width = 1600;
  const u32 height = 900;
  const char *title = "Engine";

  Window::handle = glfwCreateWindow(width, height, title, nullptr, nullptr);
  ThreadPool::StartUp();

  VulkanContext::StartUp();

  InputContext::StartUp();
}

void ShutDown() {
  VulkanContext::ShutDown();

  ThreadPool::ShutDown();

  glfwDestroyWindow((GLFWwindow *)Window::handle);
  glfwTerminate();
}
} // namespace Core
