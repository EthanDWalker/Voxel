#pragma once
#include "common.slang"

#ifndef HASH_SET_SET_BINDING
#error "must define HASH_SET_SET_BINDING set binding"
#define HASH_SET_SET_BINDING -1
#endif

struct HashSetHeaderData {
  uint32_t size;
  Atomic<uint32_t> insertion_failure;
};

struct AtomicHashSetHeader {
  RWStructuredBuffer<HashSetHeaderData> data;

  property uint32_t size {
    get { return data[0].size; }
  };

  void LogInsertionFailure() { data[0].insertion_failure.increment(); }
};

struct HashSetHeader {
  StructuredBuffer<HashSetHeaderData> data;

  property uint32_t size {
    get { return data[0].size; }
  };
};

#ifdef HASH_SET_ATOMIC
[[vk::binding(0, HASH_SET_SET_BINDING)]]
AtomicHashSetHeader hash_set_header;
[[vk::binding(3, HASH_SET_SET_BINDING)]]
RWStructuredBuffer<Atomic<uint32_t>> occlusion_data;
#else
[[vk::binding(0, HASH_SET_SET_BINDING)]]
HashSetHeader hash_set_header;
[[vk::binding(3, HASH_SET_SET_BINDING)]]
StructuredBuffer<uint32_t> occlusion_data;
#endif

static const uint32_t EMPTY_KEY = 0xFFFFFFFF;
static const uint32_t MAX_DRIFT = 100;
static const float ALPHA = 0.005f;

uint32_t Hash(uint32_t a) {
  a = (a ^ 61) ^ (a >> 16);
  a = a + (a << 3);
  a = a ^ (a >> 4);
  a = a * 0x27d4eb2d;
  a = a ^ (a >> 15);
  return a;
}

#ifdef HASH_SET_ATOMIC
struct AtomicHashSet {
  struct AtomicKeyValuePair {
    Atomic<uint32_t> key;
    Atomic<float> sample_count;
    Atomic<uint64_t> lighting;
  };

  RWStructuredBuffer<AtomicKeyValuePair> set;

  uint32_t Find(const uint32_t id, out bool in_set) {
    const uint32_t hash = Hash(id);
    in_set = false;

    for (uint32_t i = 0; i < MAX_DRIFT; i++) {
      const uint32_t index = (hash + i) % hash_set_header.size;
      const uint32_t ret = set[index].key.load();

      // assumes we do the shadow pass first which works in this case
      if (ret == id) {
        in_set = true;
        return index;
      } else if (ret == EMPTY_KEY) {
        return 0;
      }
    }

    return 0;
  }

  uint32_t FindAndInitialize(const uint32_t id, out bool initialized, out bool insertion_failiure) {
    const uint32_t hash = Hash(id);
    initialized = false;
    insertion_failiure = false;

    for (uint32_t i = 0; i < MAX_DRIFT; i++) {
      const uint32_t index = (hash + i) % hash_set_header.size;
      const uint32_t ret = set[index].key.compareExchange(EMPTY_KEY, id);

      // assumes we do the shadow pass first which works in this case
      if (ret == EMPTY_KEY) {
        set[index].lighting.add((0xFFFFFFFF - EMPTY_KEY) + 1);
        set[index].sample_count.store(0.0f);
        initialized = true;
        return index;
      } else if (ret == id) {
        return index;
      }
    }

    insertion_failiure = true;
    hash_set_header.LogInsertionFailure();
    return false;
  }
};
#endif

static const float SCALE = 0.01f;
static const float INV_SCALE = 1.0f / SCALE;
uint64_t PackIndirectLighting(const float3 data) {
  const uint32_t3 d = uint32_t3(data * INV_SCALE);
  if (any(d != 0)) {
    // printf("(%u %u %u) vs (%f %f %f)", d.x, d.y, d.z, data.x, data.y, data.z);
  }
  return SumComponents(uint64_t3(uint32_t3(data * INV_SCALE)) << uint32_t3(43, 21, 0));
}

float3 UnpackIndirectLighting(const uint64_t data) {
  return ((data >> uint32_t3(43, 21, 0)) & uint32_t3(0x1FFFF, 0x3FFFF, 0x1FFFF)) * SCALE;
}

struct IndirectLightValue {
  uint32_t occlusion_data;
  float sample_count;
  uint64_t lighting_data;

  property bool in_shadow {
    get { return bool(occlusion_data >> 1); }
  };

  property float3 indirect_light {
    get { return UnpackIndirectLighting(lighting_data) / sample_count; }
  };
};

struct HashSet {
  struct KeyValuePair {
    uint32_t key;
    float sample_count;
    uint64_t lighting_data;
  };

  StructuredBuffer<KeyValuePair> set;

  IndirectLightValue GetId(const uint32_t id, out bool in_set) {
    const uint32_t hash = Hash(id);

    for (uint32_t i = 0; i < MAX_DRIFT; i++) {
      const uint32_t index = (hash + i) % hash_set_header.size;
      const uint32_t ret = set[index].key;

      if (ret == id) {
        in_set = true;
        IndirectLightValue value;
        value.lighting_data = set[index].lighting_data;
        value.sample_count = set[index].sample_count;
#ifdef HASH_SET_ATOMIC
        // dont use index from last frame to index into this frame
        value.occlusion_data = 0;
#else
        value.occlusion_data = occlusion_data[index];
#endif

        return value;
      } else if (ret == EMPTY_KEY) {
        break;
      }
    }

    in_set = false;
    return {};
  }
};

#ifdef HASH_SET_ATOMIC
[[vk::binding(1, HASH_SET_SET_BINDING)]]
AtomicHashSet hash_set;
[[vk::binding(2, HASH_SET_SET_BINDING)]]
HashSet last_hash_set;
#else
[[vk::binding(1, HASH_SET_SET_BINDING)]]
HashSet hash_set;
[[vk::binding(2, HASH_SET_SET_BINDING)]]
HashSet last_hash_set;
#endif
