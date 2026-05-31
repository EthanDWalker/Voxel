#include "common.slang"

struct AtomicBranchNode {
  Atomic<uint64_t> child_mask;
  Atomic<uint32_t> child_ptr;
};

struct BranchNode {
  uint64_t child_mask;
  uint32_t child_ptr;
};

struct AtomicPackedLeafNode {
  Atomic<uint32_t> data;
};

struct PackedLeafNode {
  uint32_t data;
};

struct LeafNode {
  float4 color;
  float3 normal;
};

struct AtomicLeafHeader {
  Atomic<uint32_t> allocated_leaf_count;
  const uint32_t page_size_exp;
};

struct LeafHeader {
  const uint32_t allocated_leaf_count;
  const uint32_t page_size_exp;
};

// 5 bits r
// 6 bits g
// 6 bits b
// 8 bit phi
// 8 bit theta
uint32_t PackLeafNode(const LeafNode node) {
  uint32_t value = 0;

  value |= uint32_t(node.color.r * float((1 << 5) - 1)) << 27;
  value |= uint32_t(node.color.g * float((1 << 6) - 1)) << 21;
  value |= uint32_t(node.color.b * float((1 << 5) - 1)) << 16;

  const float phi_ratio = ((1 << 8) - 1) * (1.0f / PI);
  value |= uint32_t(acos(node.normal.z) * phi_ratio) << 8;

  const float theta_ratio = ((1 << 8) - 1) * (1.0f / (2.0f * PI));
  value |= uint32_t((atan2(node.normal.y, node.normal.x) + PI) * theta_ratio) << 0;

  return value;
}

LeafNode UnpackLeafNode(const uint32_t data) {
  LeafNode node;

  const float theta_ratio = ((1 << 8) - 1) * (1.0f / (2.0f * PI));
  const float theta = (((data >> 0) & 0xFF) / theta_ratio) - PI;

  const float phi_ratio = ((1 << 8) - 1) * (1.0f / PI);
  const float phi = (((data >> 8) & 0xFF) / phi_ratio);

  node.normal.z = cos(phi);
  node.normal.y = sin(phi) * sin(theta);
  node.normal.x = sin(phi) * cos(theta);

  node.normal = normalize(node.normal);

  node.color.rgb = ((data >> uint32_t3(27, 21, 16)) & uint32_t3(0x1F, 0x3F, 0x1F)) /
                   float3((1 << 5) - 1, (1 << 6) - 1, (1 << 5) - 1);
  node.color.a = 1.0f;

  return node;
}

// must be at least 3
static const uint32_t VOXEL_AABB_SIZE_EXP = 5;

static const float VOXEL_AABB_SIZE = float(1 << VOXEL_AABB_SIZE_EXP);
static const uint32_t VOXEL_SIZE_EXP = VOXEL_AABB_SIZE_EXP - 2;
static const float VOXEL_SIZE = 1 << VOXEL_SIZE_EXP;
