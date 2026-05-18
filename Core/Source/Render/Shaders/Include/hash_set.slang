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
#else
[[vk::binding(0, HASH_SET_SET_BINDING)]]
HashSetHeader hash_set_header;
#endif

static const uint32_t EMPTY_KEY = 0xFFFFFFFF;
static const uint32_t MAX_DRIFT = 100;

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
    // 1 bits in shadow
    // 1 bits shadow ray cast
    Atomic<uint32_t> occlusion_data;
    Atomic<float> lighting;
    Atomic<uint32_t> sample_count;
  };

  RWStructuredBuffer<AtomicKeyValuePair> set;

  bool ClaimShadowRay(const uint32_t id) {
    const uint32_t hash = Hash(id);

    for (uint32_t i = 0; i < MAX_DRIFT; i++) {
      const uint32_t index = (hash + i) % hash_set_header.size;
      const uint32_t ret = set[index].key.compareExchange(EMPTY_KEY, id);

      // assumes we do the shadow pass first which works in this case
      if (ret == EMPTY_KEY) {
        const uint32_t value = 1 << 30;
        // overflows the u32, reseting the value to 0 + value
        set[index].lighting.store(0.0f);
        set[index].occlusion_data.add((0xFFFFFFFF - EMPTY_KEY) + 1 + value);
        set[index].sample_count.add((0xFFFFFFFF - EMPTY_KEY) + 1);
        return true;
      } else if (ret == id) {
        return false;
      }
    }

    hash_set_header.LogInsertionFailure();
    return false;
  }

  void SetInShadow(const uint32_t id) {
    const uint32_t hash = Hash(id);

    for (uint32_t i = 0; i < MAX_DRIFT; i++) {
      const uint32_t index = (hash + i) % hash_set_header.size;
      const uint32_t ret = set[index].key.load();

      // assumes we do the shadow pass first which works in this case
      if (ret == id) {
        const uint32_t value = 1 << 31;
        const uint32_t ret = set[index].occlusion_data.add(value);
      } else if (ret == EMPTY_KEY) {
        break;
      }
    }
  }

  void StoreIndirectLight(const uint32_t id, const float lighting, const uint32_t samples) {
    const uint32_t hash = Hash(id);

    for (uint32_t i = 0; i < MAX_DRIFT; i++) {
      const uint32_t index = (hash + i) % hash_set_header.size;
      const uint32_t ret = set[index].key.load();

      // assumes we do the shadow pass first which works in this case
      if (ret == id) {
        set[index].lighting.add(lighting);
        set[index].sample_count.add(samples);
      } else if (ret == EMPTY_KEY) {
        // id is not in set
        break;
      }
    }
  }
};
#endif

struct IndirectLightValue {
  uint32_t occlusion_data;
  float lighting;
  uint32_t sample_count;

  property bool in_shadow {
    get { return bool(occlusion_data >> 31); }
  };

  property float3 indirect_light {
    get { return lighting / float(sample_count); }
  };
};

struct HashSet {
  struct KeyValuePair {
    uint32_t key;
    IndirectLightValue indirect_light;
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
        return set[index].indirect_light;
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
