#include "Core/Render/Vulkan/sampler.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/util.h"

namespace Core {

VkFilter ToVkFilter(const SamplerFilter filter) {
  ZoneScoped;
  switch (filter) {
  case SamplerFilter::Linear: {
    return VK_FILTER_LINEAR;
  }
  case SamplerFilter::Nearest: {
    return VK_FILTER_NEAREST;
  }
  default: {
    abort();
  }
  }
}

VkSamplerMipmapMode ToVkSamplerMipMapMode(const SamplerFilter filter) {
  ZoneScoped;
  switch (filter) {
  case SamplerFilter::Linear: {
    return VK_SAMPLER_MIPMAP_MODE_LINEAR;
  }
  case SamplerFilter::Nearest: {
    return VK_SAMPLER_MIPMAP_MODE_NEAREST;
  }
  default: {
    abort();
  }
  }
}

void VulkanSampler::Create(const SamplerFilter filter, const SamplerFilter mip_filter, const f32 lod_bias) {
  ZoneScoped;
  VkSamplerCreateInfo sampler_ci{};
  sampler_ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_ci.minFilter = ToVkFilter(filter);
  sampler_ci.magFilter = ToVkFilter(filter);
  sampler_ci.maxLod = VK_LOD_CLAMP_NONE;
  sampler_ci.mipmapMode = ToVkSamplerMipMapMode(mip_filter);
  sampler_ci.mipLodBias = lod_bias;
  sampler_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;

  VK_CHECK(vkCreateSampler(VulkanContext::device, &sampler_ci, nullptr, &obj));
}

VulkanSampler::~VulkanSampler() { vkDestroySampler(VulkanContext::device, obj, nullptr); }
} // namespace Core
