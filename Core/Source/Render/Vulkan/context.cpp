#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/command_buffer.h"
#include "Core/Render/Vulkan/command_pool.h"
#include "Core/Render/Vulkan/info.h"
#include "Core/Render/Vulkan/util.h"
#include "Core/Util/thread_pool.h"
#include "Core/window.h"

#define VOLK_IMPLEMENTATION
#include "volk.h"

#include "GLFW/glfw3.h"
#include "VkBootstrap.h"
#include <functional>

namespace Core {
VkInstance VulkanContext::instance;
VkDevice VulkanContext::device;
VkPhysicalDevice VulkanContext::physical_device;
VkSurfaceKHR VulkanContext::surface;
VmaAllocator VulkanContext::allocator;
VkDebugUtilsMessengerEXT VulkanContext::debug_messenger;
VkQueue VulkanContext::graphics_queue;
VkQueue VulkanContext::compute_queue;
u32 VulkanContext::graphics_queue_index;
u32 VulkanContext::compute_queue_index;

static thread_local VkCommandPool command_pool;
static thread_local VkCommandBuffer command_buffer;
static thread_local VkFence fence;

inline VKAPI_ATTR VkBool32 VKAPI_CALL VulkanDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT *data, void *) {
  auto ms = vkb::to_string_message_severity(severity);
  auto mt = vkb::to_string_message_type(type);
  if (type & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) {
    Core::Log("[{}: {}] - {}\n{}\n", ms, mt, data->pMessageIdName, data->pMessage);
  } else if (type & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    Core::Assert(false, "[{}: {}] - {}\n", ms, mt, data->pMessage);
  } else {
    Core::Log("[{}: {}] - {}\n", ms, mt, data->pMessage);
  }

  return VK_FALSE;
}

void VulkanContext::StartUp() {
  volkInitialize();
  vkb::InstanceBuilder instance_builder;

  auto instance_return = instance_builder.set_app_name("Core")
                             .request_validation_layers()
                             .set_debug_callback(VulkanDebugCallback)
                             .require_api_version(1, 3)
                             .build();

  assert(instance_return);

  vkb::Instance vkb_instance = instance_return.value();

  instance = vkb_instance.instance;
  volkLoadInstance(instance);
  debug_messenger = vkb_instance.debug_messenger;

  VK_CHECK(glfwCreateWindowSurface(instance, (GLFWwindow *)Window::handle, nullptr, &surface));

  VkPhysicalDeviceFeatures features{};
  features.multiDrawIndirect = true;
  features.fillModeNonSolid = true;
  features.shaderInt64 = true;
  features.shaderInt16 = true;

  VkPhysicalDeviceVulkan13Features features_13{};
  features_13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  features_13.dynamicRendering = true;
  features_13.synchronization2 = true;
  features_13.shaderIntegerDotProduct = true;

  VkPhysicalDeviceVulkan12Features features_12{};
  features_12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features_12.bufferDeviceAddress = true;
  features_12.drawIndirectCount = true;
  features_12.descriptorIndexing = true;
  features_12.shaderSampledImageArrayNonUniformIndexing = true;
  features_12.runtimeDescriptorArray = true;
  features_12.descriptorBindingVariableDescriptorCount = true;
  features_12.descriptorBindingPartiallyBound = true;
  features_12.descriptorBindingUniformBufferUpdateAfterBind = true;
  features_12.descriptorBindingSampledImageUpdateAfterBind = true;
  features_12.descriptorBindingStorageBufferUpdateAfterBind = true;
  features_12.descriptorBindingStorageImageUpdateAfterBind = true;
  features_12.shaderFloat16 = true;
  features_12.shaderOutputLayer = true;
  features_12.shaderBufferInt64Atomics = true;

  VkPhysicalDeviceVulkan11Features features_11{};
  features_11.storageBuffer16BitAccess = true;

  VkPhysicalDeviceRobustness2FeaturesEXT robustness_2{};
  robustness_2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT;
  robustness_2.pNext = nullptr;
  robustness_2.nullDescriptor = true;

  vkb::PhysicalDeviceSelector physical_device_selector{vkb_instance};

  vkb::PhysicalDevice vkb_physical_device =
      physical_device_selector.set_minimum_version(1, 3)
          .set_required_features_13(features_13)
          .set_required_features_12(features_12)
          .set_required_features_11(features_11)
          .set_required_features(features)
          .set_surface(surface)
          .add_required_extension(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME)
          .add_required_extension_features(robustness_2)
          .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
          .select()
          .value();
  physical_device = vkb_physical_device.physical_device;

  vkb::DeviceBuilder device_builder{vkb_physical_device};

  vkb::Device vkb_device = device_builder.build().value();
  device = vkb_device.device;
  volkLoadDevice(device);

  graphics_queue = vkb_device.get_queue(vkb::QueueType::graphics).value();
  graphics_queue_index = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

  compute_queue = vkb_device.get_queue(vkb::QueueType::compute).value();
  compute_queue_index = vkb_device.get_queue_index(vkb::QueueType::compute).value();

  VmaVulkanFunctions vulkan_functions{};
  vulkan_functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  vulkan_functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

  VmaAllocatorCreateInfo allocator_ci{};
  allocator_ci.device = device;
  allocator_ci.instance = instance;
  allocator_ci.physicalDevice = physical_device;
  allocator_ci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  allocator_ci.pVulkanFunctions = &vulkan_functions;
  VK_CHECK(vmaCreateAllocator(&allocator_ci, &allocator));

  VkCommandPoolCreateInfo command_pool_ci{};
  command_pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  command_pool_ci.queueFamilyIndex = VulkanContext::graphics_queue_index;
  command_pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  VK_CHECK(vkCreateCommandPool(device, &command_pool_ci, nullptr, &command_pool));

  VkCommandBufferAllocateInfo command_buffer_ci{};
  command_buffer_ci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_buffer_ci.commandBufferCount = 1;
  command_buffer_ci.commandPool = command_pool;
  command_buffer_ci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

  VK_CHECK(vkAllocateCommandBuffers(device, &command_buffer_ci, &command_buffer));

  VkFenceCreateInfo fence_ci{};
  fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  VK_CHECK(vkCreateFence(VulkanContext::device, &fence_ci, nullptr, &fence));

  ThreadPool::CreateThreadLocalData([=](const u32 id) {
    VK_CHECK(vkCreateCommandPool(device, &command_pool_ci, nullptr, &command_pool));

    VK_CHECK(vkAllocateCommandBuffers(device, &command_buffer_ci, &command_buffer));

    VK_CHECK(vkCreateFence(VulkanContext::device, &fence_ci, nullptr, &fence));
  });

  DescriptorBuilder::StartUp();
}

void VulkanContext::Submit(const std::function<void(VulkanCommandBuffer &cmd)> &&function) {
  VK_CHECK(vkResetFences(device, 1, &fence));
  VK_CHECK(vkResetCommandPool(device, command_pool, 0));

  VulkanCommandBuffer cmd;
  cmd.obj = command_buffer;

  cmd.Begin();

  function(cmd);

  cmd.End();

  VkCommandBufferSubmitInfo cmd_submit_info = CommandBufferSubmitInfo(cmd.obj);
  VkSubmitInfo2 submit_info = SubmitInfo(&cmd_submit_info, nullptr, nullptr);

  VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit_info, fence));

  VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, std::numeric_limits<u64>::max()));
}

void VulkanContext::ShutDown() {
  vkDeviceWaitIdle(device);

  DescriptorBuilder::ShutDown();

  vkDestroyFence(device, fence, nullptr);
  vkDestroyCommandPool(device, command_pool, nullptr);

  ThreadPool::DestroyThreadLocalData([](const u32 id) {
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
  });

  vkDestroySurfaceKHR(instance, surface, nullptr);
  vmaDestroyAllocator(allocator);
  vkDestroyDevice(device, nullptr);
  vkb::destroy_debug_utils_messenger(instance, debug_messenger);
  vkDestroyInstance(instance, nullptr);
}
} // namespace Core
