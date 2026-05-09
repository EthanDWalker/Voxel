#pragma once
#include "common.slang"

#ifndef DENSE_SET_SET_BINDING
#error "must define DENSE_SET_SET_BINDING set binding"
#define DENSE_SET_SET_BINDING -1
#endif

struct BaseDenseSetHeader<T>
    where T : IBuffer {
  T buffer;

  property uint32_t size {
    get { return buffer.Load<uint32_t, 0>(); }
  };

  property uint32_t value_count {
    get { return buffer.Load<uint32_t, 4>(); }
  };
};

struct RWDenseSetHeader : BaseDenseSetHeader<RWByteBuffer> {
  uint32_t AllocateValues(const uint32_t amount) {
    uint32_t val;
    buffer.data.InterlockedAdd(4, amount, val);
    return val;
  }
};

[[vk::binding(0, DENSE_SET_SET_BINDING)]]
RWDenseSetHeader dense_set_header;

static const uint32_t EMPTY_KEY = 0xFFFFFFFF;
static const uint32_t MAX_DRIFT = 10;

[[vk::binding(1, DENSE_SET_SET_BINDING)]]
RWStructuredBuffer<uint32_t> value_arr;

struct DenseSet {
  RWStructuredBuffer<Atomic<uint32_t>> keys;

  uint32_t Hash(uint32_t a) {
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);

    return a;
  }

  void Insert(const uint32_t id, const uint32_t value) {
    const uint32_t hash = Hash(id);

    for (uint32_t i = 0; i < MAX_DRIFT; i++) {
      const uint32_t index = (hash + i) % dense_set_header.size;
      const uint32_t ret = keys[index].compareExchange(EMPTY_KEY, id);

      if (ret == EMPTY_KEY) {
        const uint32_t ptr = dense_set_header.AllocateValues(1);
        if (ptr >= dense_set_header.size) {
          return;
        }

        value_arr[ptr] = value;
        return;
      }
    }
  }
};

[[vk::binding(2, DENSE_SET_SET_BINDING)]]
DenseSet dense_set;
