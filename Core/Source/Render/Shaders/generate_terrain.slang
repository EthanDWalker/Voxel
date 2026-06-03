#include "Include/common.slang"
#include "Include/gradients.slang"

[[vk::binding(0, 0)]]
RWStructuredBuffer<Vertex> vertex_arr;

[[vk::binding(1, 0)]]
RWStructuredBuffer<Index> index_arr;

struct PushConstants {
  uint32_t2 extent;
  int32_t seed;
};

// https://github.com/Auburn/FastNoiseLite/blob/master/HLSL

static const uint32_t3 PRIME = uint32_t3(501125321, 1136930381, 1720413743);

float InterpQuintic(const float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }

int32_t Hash2D(const int32_t seed, const int32_t x, const int32_t y) {
  int32_t hash = seed ^ x ^ y;

  hash *= 0x27d4eb2d;
  return hash;
}

float GradCoord2D(const int32_t seed, const int32_t x, const int32_t y, const float xd, const float yd) {
  int hash = Hash2D(seed, x, y);
  hash ^= hash >> 15;
  hash &= 127 << 1;
  return xd * GRADIENTS_2D[hash] + yd * GRADIENTS_2D[hash | 1];
}

float SamplePerlin(const int32_t seed, const float2 postion) {
  int x0 = (int)floor(postion.x);
  int y0 = (int)floor(postion.y);

  float xd0 = (float)(postion.x - x0);
  float yd0 = (float)(postion.y - y0);
  float xd1 = xd0 - 1;
  float yd1 = yd0 - 1;

  float xs = InterpQuintic(xd0);
  float ys = InterpQuintic(yd0);

  x0 *= PRIME.x;
  y0 *= PRIME.y;
  int x1 = x0 + PRIME.x;
  int y1 = y0 + PRIME.y;

  float xf0 = lerp(GradCoord2D(seed, x0, y0, xd0, yd0), GradCoord2D(seed, x1, y0, xd1, yd0), xs);
  float xf1 = lerp(GradCoord2D(seed, x0, y1, xd0, yd1), GradCoord2D(seed, x1, y1, xd1, yd1), xs);

  return lerp(xf0, xf1, ys) * 1.4247691104677813f;
}

float Amplify(const float h) { return h * h * 100.0f; }

[numthreads(8, 8, 1)]
void CMain(const uint32_t2 dispatch_id: SV_DispatchThreadID, const uniform PushConstants push_constants) {
  if (any(dispatch_id >= push_constants.extent))
    return;

  const int32_t seed = push_constants.seed;
  const float frequency = 100.0f;

  const uint32_t vertex_arr_index = dispatch_id.x + dispatch_id.y * push_constants.extent.x;

  const float h = Amplify(SamplePerlin(seed, float2(dispatch_id.x, dispatch_id.y) / frequency));

  const float h_x = Amplify(SamplePerlin(seed, float2(dispatch_id.x + 1, dispatch_id.y) / frequency)) - h;
  const float h_y = Amplify(SamplePerlin(seed, float2(dispatch_id.x, dispatch_id.y + 1) / frequency)) - h;

  Vertex vertex;
  vertex.position = float3(dispatch_id.x, h, dispatch_id.y) -
                    (float3(push_constants.extent.x, 0, push_constants.extent.y) * 0.5f);
  vertex.normal = normalize(float3(-h_y, 1.0f, -h_x));
  vertex.uv = float2(1.0f);

  vertex_arr[vertex_arr_index] = vertex;

  if (dispatch_id.y < push_constants.extent.y - 2 && dispatch_id.x != push_constants.extent.x - 1) {
    index_arr[vertex_arr_index * 6 + 0] = vertex_arr_index;
    index_arr[vertex_arr_index * 6 + 1] = vertex_arr_index + 1 + push_constants.extent.x;
    index_arr[vertex_arr_index * 6 + 2] = vertex_arr_index + push_constants.extent.x;

    index_arr[vertex_arr_index * 6 + 3] = vertex_arr_index;
    index_arr[vertex_arr_index * 6 + 4] = vertex_arr_index + 1;
    index_arr[vertex_arr_index * 6 + 5] = vertex_arr_index + 1 + push_constants.extent.x;
  }
}
