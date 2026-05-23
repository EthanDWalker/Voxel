#include "Include/common.slang"

float3 UnpackRGB565(uint16_t packed) {
  uint32_t r = (packed >> 11) & 0x1F;
  uint32_t g = (packed >> 5) & 0x3F;
  uint32_t b = packed & 0x1F;

  uint32_t r8 = (r << 3) | (r >> 2);
  uint32_t g8 = (g << 2) | (g >> 4);
  uint32_t b8 = (b << 3) | (b >> 2);

  return float3(r8 / 255.0f, g8 / 255.0f, b8 / 255.0f);
}

struct BC1Block {
  uint32_t colors;
  uint32_t indices;

  property float3 color0 {
    get { return UnpackRGB565((uint16_t)(colors >> 16)); }
    set {
      uint16_t r = (uint16_t)round(newValue.x * 31.0f);
      uint16_t g = (uint16_t)round(newValue.y * 63.0f);
      uint16_t b = (uint16_t)round(newValue.z * 31.0f);
      uint32_t color = (r << 11) | (g << 5) | b;

      colors &= 0x0000FFFF;
      colors |= color << 16;
    }
  }

  property float3 color1 {
    get { return UnpackRGB565((uint16_t)(colors & 0xFFFF)); }
    set {
      uint16_t r = (uint16_t)round(newValue.x * 31.0f);
      uint16_t g = (uint16_t)round(newValue.y * 63.0f);
      uint16_t b = (uint16_t)round(newValue.z * 31.0f);
      uint16_t color = (r << 11) | (g << 5) | b;

      colors &= 0xFFFF0000;
      colors |= color;
    }
  }

  [[mutating]] void ComputeColorPalette(const float3 pixels[16]) {
    color0 = float3(1e30f);
    color1 = float3(-1e30f);

    for (uint32_t i = 0; i < 16; i++) {
      color0 = min(color0, pixels[i]);
      color1 = max(color1, pixels[i]);
    }
  }

  [[mutating]]
  void SetIndice(const uint32_t2 index, const float3 color) {
    const uint32_t offset = (index.x + index.y * 4) * 2;

    const float2 color_multipliers[4] = {
      float2(1.0f, 0.0f),
      float2(0.0f, 1.0f),
      float2(2.0f / 3.0f, 1.0f / 3.0f),
      float2(1.0f / 3.0f, 2.0f / 3.0f),
    };

    uint32_t type = 0;
    float min_error = 1e30f;

    for (uint32_t i = 0; i < 4; i++) {
      const float3 p = color0 * color_multipliers[i].x + color1 * color_multipliers[i].y;
      const float3 d = p - color;
      const float error = dot(d, d);
      if (error < min_error) {
        min_error = error;
        type = i;
      }
    }

    indices &= ~(0x3u << offset);
    indices |= ((uint32_t)type << offset);
  }
};

[[vk::binding(0, 0)]]
RWStructuredBuffer<uint32_t> input_texture;

[[vk::binding(1, 0)]]
RWStructuredBuffer<BC1Block> output_texture;

void Write(const uint32_t2 block_index, const uint32_t2 block_count, const uint32_t2 extent) {
  float3 pixels[16];
  const uint32_t2 offset = {
    block_index.x * 4,
    block_index.y * 4,
  };

  const uint32_t texel_width = 4;
  const uint32_t texel_height = 4;
  for (uint32_t x = 0; x < texel_width; x++) {
    for (uint32_t y = 0; y < texel_height; y++) {
      const float4 color = UnpackRGBA8(input_texture[(offset.x + x) + ((offset.y + y) * extent.x)]);
      pixels[x + texel_width * y] = color.rgb;
    }
  }

  BC1Block block = {};
  block.ComputeColorPalette(pixels);

  for (uint32_t i = 0; i < 4; i++) {
    for (uint32_t j = 0; j < 4; j++) {
      block.SetIndice(uint32_t2(i, j), pixels[i + texel_width * j]);
    }
  }

  output_texture[block_index.x + (block_index.y * block_count.x)] = block;
}

[[shader("compute")]]
[[numthreads(8, 8, 1)]]
void CMain(const uint32_t2 block_index: SV_DispatchThreadID, const uniform uint32_t2 extent) {
  const uint32_t2 block_count = uint32_t2((extent.x + 3) / 4, (extent.y + 3) / 4);

  if (block_index.x >= block_count.x || block_index.y >= block_count.y) {
    return;
  }

  Write(block_index, block_count, extent);
}
