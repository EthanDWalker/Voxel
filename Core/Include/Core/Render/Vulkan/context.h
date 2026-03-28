#pragma once

#include "Core/Render/Vulkan/command_buffer.h"
#include "volk.h"

#define VMA_LEAK_LOG_FORMAT(format, ...)                                                           \
  do {                                                                                             \
    printf("[Core] VMA: " format "\n", ##__VA_ARGS__);                                             \
  } while (false)

#include "vma/vk_mem_alloc.h"
#include <functional>

namespace Core {
struct VulkanContext {
  static void StartUp();
  static void ShutDown();

  static void Submit(const std::function<void(VulkanCommandBuffer &cmd)> &&function);

  static VkInstance instance;
  static VkDevice device;
  static VkPhysicalDevice physical_device;
  static VkSurfaceKHR surface;
  static VmaAllocator allocator;
  static VkDebugUtilsMessengerEXT debug_messenger;
  static VkQueue graphics_queue;
  static VkQueue compute_queue;
  static u32 graphics_queue_index;
  static u32 compute_queue_index;
};
} // namespace Core
