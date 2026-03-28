#include "Core/Render/Vulkan/sampler.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/util.h"

namespace Core {
void VulkanSampler::Create(const SamplerFilter filter, const SamplerFilter mip_filter, const f32 lod_bias) {
  VkSamplerCreateInfo sampler_ci{};
  sampler_ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_ci.minFilter = static_cast<VkFilter>(filter);
  sampler_ci.magFilter = static_cast<VkFilter>(filter);
  sampler_ci.maxLod = VK_LOD_CLAMP_NONE;
  sampler_ci.mipmapMode = static_cast<VkSamplerMipmapMode>(mip_filter);
  sampler_ci.mipLodBias = lod_bias;

  VK_CHECK(vkCreateSampler(VulkanContext::device, &sampler_ci, nullptr, &obj));
}

VulkanSampler::~VulkanSampler() { vkDestroySampler(VulkanContext::device, obj, nullptr); }
} // namespace Core
