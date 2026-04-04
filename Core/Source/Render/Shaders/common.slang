public module common;

struct Camera {
  float4x4 projection;
  float4x4 view;
  float3 position;
  float aspect_ratio;
  float3 front;
  float fov_y;
  float3 up;
  float z_near;
};

struct DirectionalLight {
  float3 direction;
  float intesity;
  float3 color;
};

uint32_t PackRGBA8(const float4 data) {
  const float ratio = 255.0f;

  uint32_t value = 0;

  value |= uint32_t(data.r * ratio) << 24;
  value |= uint32_t(data.g * ratio) << 16;
  value |= uint32_t(data.b * ratio) << 8;
  value |= uint32_t(data.a * ratio) << 0;

  return value;
}

float4 UnpackRGBA8(const uint32_t data) {
  const float ratio = 1.0f / 255.0f;

  float4 value;

  value.r = ((data >> 24) & 0xFF) * ratio;
  value.g = ((data >> 16) & 0xFF) * ratio;
  value.b = ((data >> 8) & 0xFF) * ratio;
  value.a = ((data >> 0) & 0xFF) * ratio;

  return value;
}

float4x4 LookAtInverse(const float3 eye, const float3 center, const float3 up) {
  const float3 f = normalize(center - eye);
  const float3 s = normalize(cross(f, up));
  const float3 u = normalize(cross(s, f));

  // clang-format off
  const float4x4 inv_view = float4x4(
    float4(s.x, u.x, -f.x, eye.x),
    float4(s.y, u.y, -f.y, eye.y),
    float4(s.z, u.z, -f.z, eye.z),
    float4(0.0f, 0.0f, 0.0f, 1.0f)
  );
  // clang-format on

  return inv_view;
}

float4x4 PerspectiveReverseZInverse(const float z_near, const float z_far, const float fov_y,
                                    const float aspect_ratio) {
  const float tan_half_fov_y = tan(fov_y / 2.0f);

  float4x4 inv_proj = float4x4(0.0f);
  inv_proj[0][0] = aspect_ratio * tan_half_fov_y;
  inv_proj[1][1] = tan_half_fov_y;
  inv_proj[2][3] = -1.0f;
  inv_proj[3][2] = (-z_far + z_near) / (z_far * z_near);
  inv_proj[3][3] = 1.0f / z_near;

  return inv_proj;
}

struct TreeHeader {
  private ByteAddressBuffer buffer;

  property uint32_t voxel_count {
    get { return buffer.Load<uint32_t>(0); }
  }

  property uint32_t allocated_page_count {
    get { return buffer.Load<uint32_t>(4); }
  }

  property uint32_t page_size {
    get { return buffer.Load<uint32_t>(8); }
  };

  property float min_bound {
    get { return buffer.Load<float>(12); }
  };

  property float max_bound {
    get { return buffer.Load<float>(16); }
  };

  property uint32_t max_depth {
    get { return buffer.Load<uint32_t>(20); }
  };

  property uint32_t allocated_far_ptr_page_count {
    get { return buffer.Load<uint32_t>(28); }
  };

  property uint32_t far_ptr_page_size {
    get { return buffer.Load<uint32_t>(32); }
  };

  property uint32_t far_ptr_page_size_exp {
    get { return buffer.Load<uint32_t>(36); }
  };

  property uint32_t page_size_exp {
    get { return buffer.Load<uint32_t>(40); }
  };

  void GetLevelPageOffset(const uint32_t level) {
    buffer.Load<uint32_t>(48 + (level * sizeof(uint32_t)));
  }
};

struct RWTreeHeader {
  private RWByteAddressBuffer buffer;

  property uint32_t voxel_count {
    get {
      uint32_t val;
      buffer.InterlockedAdd(0, 0, val);
      return val;
    }
  }

  const uint32_t AllocatedVoxels(const uint32_t count) {
    uint32_t val;
    buffer.InterlockedAdd(0, count, val);
    return val;
  }

  property uint32_t allocated_page_count {
    get { return buffer.Load<uint32_t>(4); }
  }

  property uint32_t page_size {
    get { return buffer.Load<uint32_t>(8); }
  };

  property float min_bound {
    get { return buffer.Load<float>(12); }
  };

  property float max_bound {
    get { return buffer.Load<float>(16); }
  };

  property uint32_t max_depth {
    get { return buffer.Load<uint32_t>(20); }
  };

  const uint32_t AllocateFarPtr() {
    uint32_t val;
    buffer.InterlockedAdd(24, 1, val);
    return val;
  }

  property uint32_t allocated_far_ptr_page_count {
    get { return buffer.Load<uint32_t>(28); }
  };

  property uint32_t far_ptr_page_size {
    get { return buffer.Load<uint32_t>(32); }
  };

  property uint32_t far_ptr_page_size_exp {
    get { return buffer.Load<uint32_t>(36); }
  };

  property uint32_t page_size_exp {
    get { return buffer.Load<uint32_t>(40); }
  };

  void Notify(const uint32_t v = 1) { buffer.InterlockedAdd(44, v); }

  void GetLevelPageOffset(const uint32_t level) {
    buffer.Load<uint32_t>(48 + (level * sizeof(uint32_t)));
  }
};

struct CountedBuffer<T> {
  private ByteAddressBuffer buffer;

  property uint32_t count {
    get { return buffer.Load<uint32_t>(0); }
  }

  __subscript(const uint32_t index)->T {
    get { return buffer.Load<T>(sizeof(uint32_t) + sizeof(DirectionalLight) * index); }
  }
};
