#pragma once

namespace Core {
struct GenerateTerrainPushConstants {
  Vec2u32 extent;
  i32 seed;
};

struct BC1CompressionPushConstants {
  Vec2u32 extent;
};
}; // namespace Core
