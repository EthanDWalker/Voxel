#pragma once

#include "vector.hpp"

static bool RayAABBIntersect(const Vec3f32 ray_dir, const Vec3f32 ray_origin,
                             const Vec3f32 bounds_min, const Vec3f32 bounds_max, float &t_min,
                             float &t_max) {
  f32 tmin, tmax, tymin, tymax, tzmin, tzmax;
  if (ray_dir.x >= 0) {
    tmin = (bounds_min.x - ray_origin.x) / ray_dir.x;
    tmax = (bounds_max.x - ray_origin.x) / ray_dir.x;
  } else {
    tmin = (bounds_max.x - ray_origin.x) / ray_dir.x;
    tmax = (bounds_min.x - ray_origin.x) / ray_dir.x;
  }
  if (ray_dir.y >= 0) {
    tymin = (bounds_min.y - ray_origin.y) / ray_dir.y;
    tymax = (bounds_max.y - ray_origin.y) / ray_dir.y;
  } else {
    tymin = (bounds_max.y - ray_origin.y) / ray_dir.y;
    tymax = (bounds_min.y - ray_origin.y) / ray_dir.y;
  }
  if ((tmin > tymax) || (tymin > tmax))
    return false;
  if (tymin > tmin)
    tmin = tymin;
  if (tymax < tmax)
    tmax = tymax;
  if (ray_dir.z >= 0) {
    tzmin = (bounds_min.z - ray_origin.z) / ray_dir.z;
    tzmax = (bounds_max.z - ray_origin.z) / ray_dir.z;
  } else {
    tzmin = (bounds_max.z - ray_origin.z) / ray_dir.z;
    tzmax = (bounds_min.z - ray_origin.z) / ray_dir.z;
  }
  if ((tmin > tzmax) || (tzmin > tmax))
    return false;
  if (tzmin > tmin)
    tmin = tzmin;
  if (tzmax < tmax)
    tmax = tzmax;

  t_min = tmin;
  t_max = tmax;

  return tmax > 0.0f;
}
