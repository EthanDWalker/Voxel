#pragma once
#include "common.slang"

struct HashSetHeader {
  uint32_t size;
};

struct LightingData {
  float3 acc_lighting;
  float sample_count;
};

struct AtomicLightingData {
  Atomic<float> acc_lighting_r;
  Atomic<float> acc_lighting_g;
  Atomic<float> acc_lighting_b;
  Atomic<float> sample_count;
};

struct HashSetKey {
  uint32_t key;
};

struct AtomicHashSetKey {
  Atomic<uint32_t> key;
};

static const uint32_t HASH_SET_HEADER_BINDING = 0;

static const uint32_t HASH_SET_KEY_BINDING = 1;
static const uint32_t HASH_SET_LAST_KEY_BINDING = 2;

static const uint32_t HASH_SET_DIFFUSE_DATA_BINDING = 3;
static const uint32_t HASH_SET_LAST_DIFFUSE_DATA_BINDING = 4;

static const uint32_t HASH_SET_SPECULAR_DATA_BINDING = 5;
static const uint32_t HASH_SET_LAST_SPECULAR_DATA_BINDING = 6;

static const uint32_t EMPTY_KEY = 0xFFFFFFFF;
static const uint32_t MAX_DRIFT = 100;

uint32_t LodMask(const float distance) {
  if (distance < 30.0f) {
    return ~0b0;
  } else if (distance < 90.0f) {
    return ~0b1;
  } else if (distance < 250.0f){
    return ~0b11;
  } else {
    return ~0b111;
  }
}

uint32_t Hash(uint32_t a) {
  a = (a ^ 61) ^ (a >> 16);
  a = a + (a << 3);
  a = a ^ (a >> 4);
  a = a * 0x27d4eb2d;
  a = a ^ (a >> 15);
  return a;
}

Optional<uint32_t> HashSetFind(const uint32_t id, StructuredBuffer<HashSetKey> key_buffer,
                               RWStructuredBuffer<HashSetHeader> hash_set_header) {
  const uint32_t hash = Hash(id);

  for (uint32_t i = 0; i < MAX_DRIFT; i++) {
    const uint32_t index = (hash + i) % hash_set_header[0].size;
    const uint32_t ret = key_buffer[index].key;

    if (ret == id) {
      return index;
    } else if (ret == EMPTY_KEY) {
      return none;
    }
  }

  return none;
}

Optional<uint32_t> HashSetFind(const uint32_t id, StructuredBuffer<HashSetKey> key_buffer,
                               StructuredBuffer<HashSetHeader> hash_set_header) {
  const uint32_t hash = Hash(id);

  for (uint32_t i = 0; i < MAX_DRIFT; i++) {
    const uint32_t index = (hash + i) % hash_set_header[0].size;
    const uint32_t ret = key_buffer[index].key;

    if (ret == id) {
      return index;
    } else if (ret == EMPTY_KEY) {
      return none;
    }
  }

  return none;
}

Optional<uint32_t> HashSetFindOrInitialize(const uint32_t id, out bool initialized,
                                           RWStructuredBuffer<AtomicHashSetKey> key_buffer,
                                           RWStructuredBuffer<HashSetHeader> hash_set_header) {
  const uint32_t hash = Hash(id);

  for (uint32_t i = 0; i < MAX_DRIFT; i++) {
    const uint32_t index = (hash + i) % hash_set_header[0].size;
    const uint32_t ret = key_buffer[index].key.compareExchange(EMPTY_KEY, id);

    if (ret == EMPTY_KEY) {
      initialized = true;
      return index;
    } else if (ret == id) {
      initialized = false;
      return index;
    }
  }

  initialized = false;
  return none;
}
