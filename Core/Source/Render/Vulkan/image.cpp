#include "Core/Render/Vulkan/image.h"
#include "Core/Render/Vulkan/context.h"
#include "Core/Render/Vulkan/info.h"
#include "Core/Render/Vulkan/util.h"
#include <tracy/Tracy.hpp>

namespace Core {

void BaseVulkanImage::DestroyBase() {
  ZoneScoped;
  vmaDestroyImage(VulkanContext::allocator, obj, allocation);
  vkDestroyImageView(VulkanContext::device, view, nullptr);
}

void BaseVulkanImage::CreateBase(const VkImageCreateInfo &image_ci, VkImageViewCreateInfo &image_view_ci) {
  ZoneScoped;
  Assert(image_ci.format != VK_FORMAT_UNDEFINED, "Format cannot be undefined");
  Assert(image_ci.extent.width > 0, "Width cannot be less than 0");
  Assert(image_ci.extent.height > 0, "Height cannot be less than 0");
  Assert(image_ci.usage != 0, "Image has no usage");
  if (image_ci.mipLevels != 1) {
    Assert((image_ci.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0,
           "If mip mapped transfer dst bit must be set");
    Assert((image_ci.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0,
           "If mip mapped transfer src bit must be set");
  }

  this->format = image_ci.format;
  this->width = image_ci.extent.width;
  this->height = image_ci.extent.height;
  this->depth = image_ci.extent.depth;
  this->usage = image_ci.usage;

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  VK_CHECK(vmaCreateImage(VulkanContext::allocator, &image_ci, &alloc_info, &this->obj, &this->allocation,
                          nullptr));
  image_view_ci.image = obj;
  VK_CHECK(vkCreateImageView(VulkanContext::device, &image_view_ci, nullptr, &this->view));
}

void BaseVulkanImage::CreateBaseRef(const BaseVulkanImage &base_image, VkImageViewCreateInfo &image_view_ci) {
  ZoneScoped;

  this->format = base_image.format;
  this->width = base_image.width;
  this->height = base_image.height;
  this->depth = base_image.depth;
  this->usage = base_image.usage;
  this->allocation = base_image.allocation;
  this->obj = base_image.obj;

  image_view_ci.image = obj;
  VK_CHECK(vkCreateImageView(VulkanContext::device, &image_view_ci, nullptr, &this->view));
}

void BaseVulkanImage::DestroyBaseRef() {
  ZoneScoped;
  vkDestroyImageView(VulkanContext::device, view, nullptr);
}

// ----------PLANAR----------

VulkanImage<ImageType::Planar>::~VulkanImage<ImageType::Planar>() {
  ZoneScoped;
  DestroyBase();
}

void VulkanImage<ImageType::Planar>::Recreate(Vec2u32 extent, VkFormat format, VkImageUsageFlags usage_flags,
                                              bool mipmapped) {
  ZoneScoped;
  DestroyBase();
  Create(extent, format, usage_flags, mipmapped);
}

void VulkanImage<ImageType::Planar>::Create(Vec2u32 extent, VkFormat format, VkImageUsageFlags usage_flags,
                                            bool mipmapped) {
  ZoneScoped;
  const u32 mip_levels = mipmapped ? CalculateMipLevels(extent) : 1;
  VkImageCreateInfo image_ci = ImageCI(format, usage_flags, Vec3u32(extent, 1), mip_levels);
  VkImageViewCreateInfo image_view_ci = ImageViewCI(format, obj, mip_levels);
  CreateBase(image_ci, image_view_ci);
}

// ----------VOLUME----------

void VulkanImage<ImageType::Volume>::Create(Vec3u32 extent, VkFormat format, VkImageUsageFlags usage_flags,
                                            bool referenced, bool mipmapped) {
  ZoneScoped;
  const u32 mip_levels = mipmapped ? CalculateMipLevels(extent) : 1;
  VkImageCreateInfo image_ci = ImageCI(format, usage_flags, extent, mip_levels);
  if (referenced) {
    image_ci.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
  }
  VkImageViewCreateInfo image_view_ci = ImageViewCI(format, obj, mip_levels);
  image_view_ci.viewType = VK_IMAGE_VIEW_TYPE_3D;
  CreateBase(image_ci, image_view_ci);
}

void VulkanImage<ImageType::Volume>::Recreate(Vec3u32 extent, VkFormat format, VkImageUsageFlags usage_flags,
                                              bool referenced, bool mipmapped) {
  ZoneScoped;
  DestroyBase();
  Create(extent, format, usage_flags, mipmapped);
}

VulkanImage<ImageType::Volume>::~VulkanImage<ImageType::Volume>() {
  ZoneScoped;
  DestroyBase();
}

// ----------VOLUME REF--------

void VulkanImage<ImageType::VolumeRef>::Create(const VulkanImage<ImageType::Volume> &image, VkFormat format,
                                               bool mipmapped) {
  ZoneScoped;
  VkImageViewCreateInfo image_view_ci =
      ImageViewCI(format, image.obj, mipmapped ? CalculateMipLevels(image.GetVec3u32()) : 1);
  image_view_ci.viewType = VK_IMAGE_VIEW_TYPE_3D;
  CreateBaseRef(image, image_view_ci);
}

void VulkanImage<ImageType::VolumeRef>::Recreate(const VulkanImage<ImageType::Volume> &image, VkFormat format,
                                                 bool mipmapped) {
  ZoneScoped;
  DestroyBaseRef();
  Create(image, format, mipmapped);
}

VulkanImage<ImageType::VolumeRef>::~VulkanImage<ImageType::VolumeRef>() {
  ZoneScoped;
  DestroyBaseRef();
}

// ----------CUBE MAP----------

void VulkanImage<ImageType::CubeMap>::Recreate(u32 side_length, VkFormat format,
                                               VkImageUsageFlags usage_flags, bool mipmapped) {
  ZoneScoped;
  DestroyBase();
  Create(side_length, format, usage_flags, mipmapped);
}

VulkanImage<ImageType::CubeMap>::~VulkanImage<ImageType::CubeMap>() {
  ZoneScoped;
  DestroyBase();
  for (u32 i = 0; i < static_cast<u32>(CubeFace::Count); i++) {
    vkDestroyImageView(VulkanContext::device, face_views[i], nullptr);
  }
}

void VulkanImage<ImageType::CubeMap>::Create(u32 side_length, VkFormat format, VkImageUsageFlags usage_flags,
                                             bool mipmapped) {
  ZoneScoped;
  const u32 mip_levels = mipmapped ? CalculateMipLevels(Vec2u32(side_length)) : 1;
  VkImageCreateInfo image_ci = ImageCI(format, usage_flags, Vec3u32(Vec2u32(side_length), 1), mip_levels);
  image_ci.arrayLayers = static_cast<u32>(CubeFace::Count);
  image_ci.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

  VkImageViewCreateInfo image_view_ci = ImageViewCI(format, obj, mip_levels);
  image_view_ci.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
  image_view_ci.subresourceRange.layerCount = static_cast<u32>(CubeFace::Count);

  CreateBase(image_ci, image_view_ci);

  image_view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  image_view_ci.subresourceRange.layerCount = 1;
  for (u32 i = 0; i < static_cast<u32>(CubeFace::Count); i++) {
    image_view_ci.subresourceRange.baseArrayLayer = i;
    VK_CHECK(vkCreateImageView(VulkanContext::device, &image_view_ci, nullptr, &face_views[i]));
  }
}
} // namespace Core
