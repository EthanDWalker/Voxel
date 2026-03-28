#pragma once
#include "volk.h"

#include "vma/vk_mem_alloc.h"

namespace Core {

enum class ImageType : u8 {
  Planar,
  CubeMap,
};

struct BaseVulkanImage {
  BaseVulkanImage() = default;

  BaseVulkanImage(const BaseVulkanImage &) = delete;
  BaseVulkanImage &operator=(const BaseVulkanImage &) = delete;

  BaseVulkanImage(BaseVulkanImage &&) = default;
  BaseVulkanImage &operator=(BaseVulkanImage &&) = default;

  VkImage obj = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkImageUsageFlags usage = static_cast<VkImageUsageFlags>(0);
  u32 width = 0;
  u32 height = 0;
  u32 depth = 0;

  void CreateBase(const VkImageCreateInfo &image_ci, VkImageViewCreateInfo &image_view_ci);
  void DestroyBase();

  Vec2u32 GetVec2u32() const { return {width, height}; }
  Vec3u32 GetVec3u32() const { return {width, height, depth}; }
};

template <ImageType T> struct VulkanImage;

template <> struct VulkanImage<ImageType::Planar> : BaseVulkanImage {
  void Recreate(Vec2u32 extent, VkFormat format, VkImageUsageFlags usage_flags,
                bool mipmapped = false);

  void Create(Vec2u32 extent, VkFormat format, VkImageUsageFlags usage_flags,
              bool mipmapped = false);

  ~VulkanImage<ImageType::Planar>();
};

template <> struct VulkanImage<ImageType::CubeMap> : BaseVulkanImage {
  // ordered according to vulkan spec
  enum class CubeFace : u8 {
    PositiveX = 0,
    NegativeX = 1,
    PositiveY = 2,
    NegativeY = 3,
    PositiveZ = 4,
    NegativeZ = 5,
    Count = 6,
  };

  VkImageView face_views[static_cast<u8>(CubeFace::Count)] = {VK_NULL_HANDLE};

  void Recreate(u32 side_length, VkFormat format, VkImageUsageFlags usage_flags,
                bool mipmapped = false);

  void Create(u32 side_length, VkFormat format, VkImageUsageFlags usage_flags,
              bool mipmapped = false);

  ~VulkanImage<ImageType::CubeMap>();
};
} // namespace Core
