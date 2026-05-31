#include "Core/Util/Parse/gltf.h"
#include "Core/Util/Parse/json.h"
#include "Core/Util/Parse/object.h"
#include "Core/Util/compress.h"
#include "Core/Util/fail.h"
#include "Core/Util/timer.h"
#include <cstring>
#include <format>
#include <fstream>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace Core {

struct GlbHeader {
  u32 magic;
  u32 version;
  u32 length;

  static const u32 MAGIC_NUMBER = 0x46546C67;
};

enum class GlbChunkType : u32 {
  Json = 0x4E4F534A,
  Bin = 0x004E4942,
};

enum class GlbAccessorComponentType : u32 {
  i8 = 5120,
  u8 = 5121,
  i16 = 5122,
  u16 = 5123,
  u32 = 5125,
  f32 = 5126,
};

u32 GetGlbAccessorComponentTypeSize(const GlbAccessorComponentType type) {
  ZoneScoped;
  switch (type) {
  case GlbAccessorComponentType::i8:
  case GlbAccessorComponentType::u8: {
    return 1;
  }

  case GlbAccessorComponentType::i16:
  case GlbAccessorComponentType::u16: {
    return 2;
  }

  case GlbAccessorComponentType::u32:
  case GlbAccessorComponentType::f32: {
    return 4;
  }
  default: {
    Assert(false, "unknown glb accessor component type");
    return 0;
  }
  }
}

struct GlbChunkHeader {
  u32 length;
  GlbChunkType type;
};

void ParseGlbFile(const std::filesystem::path &file_path, ObjectData &object_data) {
  ZoneScoped;
  SCOPED_TIMER("glb parsing");
  std::ifstream file(file_path, std::ios::in | std::ios::binary);

  GlbHeader header;
  file.read((char *)&header, sizeof(GlbHeader));

  Assert(header.magic == GlbHeader::MAGIC_NUMBER, ".glb file doesnt contain magic number");

  GlbChunkHeader json_chunk_header;
  file.read((char *)&json_chunk_header, sizeof(GlbChunkHeader));

  Assert(json_chunk_header.type == GlbChunkType::Json, "initial chunk type is not json");

  JsonObject *json_object = ParseJsonStream(file, json_chunk_header.length);

  GlbChunkHeader bin_chunk_header;
  file.read((char *)&bin_chunk_header, sizeof(GlbChunkHeader));

  Assert(bin_chunk_header.type == GlbChunkType::Bin, "bin chunk type is not bin type");

  object_data.name = file_path.filename().stem().string();

  const JsonArray node_array = json_object->FindNoFail("nodes");
  const JsonArray mesh_array = json_object->FindNoFail("meshes");
  const JsonArray accessor_array = json_object->FindNoFail("accessors");
  const JsonArray buffer_view_array = json_object->FindNoFail("bufferViews");

  const JsonArray material_array = json_object->FindNoFail("materials");
  const JsonArray texture_array = json_object->FindNoFail("textures");
  object_data.material_data_arr.resize(texture_array.value_count);

  const JsonArray image_array = json_object->FindNoFail("images");
  const JsonArray sampler_array = json_object->FindNoFail("samplers");

  const u64 bin_offset = sizeof(GlbHeader) + sizeof(GlbChunkHeader) * 2 + json_chunk_header.length;

  std::vector<u32> mesh_index_offset;
  mesh_index_offset.resize(mesh_array.value_count);

  for (u32 i = 0; i < mesh_array.value_count; i++) {
    Assert(mesh_array.value_type_arr[i] == JsonValueType::Object, "mesh array must contain objects");
    const JsonObject mesh_object = *mesh_array.value_arr[i].object;

    const JsonArray primitive_array = mesh_object.FindNoFail("primitives");

    mesh_index_offset[i] = object_data.mesh_data_arr.size();

    for (u32 j = 0; j < primitive_array.value_count; j++) {
      MeshData &mesh = object_data.mesh_data_arr.emplace_back();

      mesh.name = mesh_object.FindNoFail("name").value_arr[0].string +
                  (primitive_array.value_count > 1 ? std::format("({})", j) : "");

      const JsonObject primitive_object = *primitive_array.value_arr[j].object;

      {
        const u32 material_index = (u32)primitive_object.FindNoFail("material").value_arr[0].number;

        const u32 material_texture_index = (u32)material_array.value_arr[material_index]
                                               .object->FindNoFail("pbrMetallicRoughness")
                                               .value_arr[0]
                                               .object->FindNoFail("baseColorTexture")
                                               .value_arr[0]
                                               .object->FindNoFail("index")
                                               .value_arr[0]
                                               .number;

        const u32 material_sampler_index = (u32)texture_array.value_arr[material_texture_index]
                                               .object->FindNoFail("sampler")
                                               .value_arr[0]
                                               .number;

        const u32 material_texture_image_index = (u32)texture_array.value_arr[material_texture_index]
                                                     .object->FindNoFail("source")
                                                     .value_arr[0]
                                                     .number;

        mesh.material_index = material_texture_image_index;

        if (!object_data.material_data_arr[mesh.material_index].initialized) {
          MaterialData &material_data = object_data.material_data_arr.at(mesh.material_index);

          material_data.initialized = true;
          material_data.name =
              material_array.value_arr[material_index].object->FindNoFail("name").value_arr[0].string;

          const u32 material_image_buffer_view_index =
              (u32)image_array.value_arr[material_texture_image_index]
                  .object->FindNoFail("bufferView")
                  .value_arr[0]
                  .number;

          const JsonObject material_buffer_view =
              *buffer_view_array.value_arr[material_image_buffer_view_index].object;

          const u64 data_size = (u32)material_buffer_view.FindNoFail("byteLength").value_arr[0].number;
          u8 *const image_data = (u8 *const)malloc(data_size);

          file.seekg(bin_offset + (u32)material_buffer_view.FindNoFail("byteOffset").value_arr[0].number,
                     std::ios::beg);
          file.read((char *)image_data, data_size);

          u8 *albedo_data =
              stbi_load_from_memory((u8 *)image_data, data_size, (int *)&material_data.albedo_extent.x,
                                    (int *)&material_data.albedo_extent.y, nullptr, 4);
          free(image_data);

          VulkanBuffer<BufferType::StagingBuffer> albedo_data_buffer = "albedo data buffer";
          albedo_data_buffer.Create(material_data.albedo_extent.x * material_data.albedo_extent.y * 4,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
          memcpy(albedo_data_buffer.host_address, albedo_data, albedo_data_buffer.size);
          CompressBC1(material_data.albedo_extent, albedo_data_buffer,
                      *material_data.compressed_albedo_data_buffer);
          free(albedo_data);
        }
      }

      const JsonObject attributes_object = *primitive_object.FindNoFail("attributes").value_arr[0].object;

      const JsonObject index_accessor_object =
          *accessor_array.value_arr[(u32)primitive_object.FindNoFail("indices").value_arr[0].number].object;
      const JsonObject position_accessor_object =
          *accessor_array.value_arr[(u32)attributes_object.FindNoFail("POSITION").value_arr[0].number].object;
      const JsonObject normal_accessor_object =
          *accessor_array.value_arr[(u32)attributes_object.FindNoFail("NORMAL").value_arr[0].number].object;
      const JsonObject uv_accessor_object =
          *accessor_array.value_arr[(u32)attributes_object.FindNoFail("TEXCOORD_0").value_arr[0].number]
               .object;

      const JsonObject index_buffer_view_object =
          *buffer_view_array
               .value_arr[(u32)index_accessor_object.FindNoFail("bufferView").value_arr[0].number]
               .object;
      const JsonObject position_buffer_view_object =
          *buffer_view_array
               .value_arr[(u32)position_accessor_object.FindNoFail("bufferView").value_arr[0].number]
               .object;
      const JsonObject normal_buffer_view_object =
          *buffer_view_array
               .value_arr[(u32)normal_accessor_object.FindNoFail("bufferView").value_arr[0].number]
               .object;
      const JsonObject uv_buffer_view_object =
          *buffer_view_array.value_arr[(u32)uv_accessor_object.FindNoFail("bufferView").value_arr[0].number]
               .object;

      const u32 vertex_count = (u32)position_accessor_object.FindNoFail("count").value_arr[0].number;
      mesh.vertex_host_buffer->Create(sizeof(Vertex) * vertex_count);

      const u32 index_count = (u32)index_accessor_object.FindNoFail("count").value_arr[0].number;
      mesh.index_host_buffer->Create(sizeof(Index) * index_count);

      file.seekg(bin_offset + (u32)position_buffer_view_object.FindNoFail("byteOffset").value_arr[0].number,
                 std::ios::beg);

      mesh.vertex_count = vertex_count;
      for (u32 i = 0; i < vertex_count; i++) {
        Vec3f32 position;
        file.read((char *)&position, sizeof(Vec3f32));
        ((Vertex *)mesh.vertex_host_buffer->host_address)[i].position =
            VecTypeCast<f16>(Vec4f32(position, 0.0f));
      }

      file.seekg(bin_offset + (u32)normal_buffer_view_object.FindNoFail("byteOffset").value_arr[0].number,
                 std::ios::beg);

      for (u32 i = 0; i < vertex_count; i++) {
        Vec3f32 normal;
        file.read((char *)&normal, sizeof(Vec3f32));
        ((Vertex *)mesh.vertex_host_buffer->host_address)[i].normal = PackNormal(normal);
      }

      file.seekg(bin_offset + (u32)uv_buffer_view_object.FindNoFail("byteOffset").value_arr[0].number,
                 std::ios::beg);

      for (u32 i = 0; i < vertex_count; i++) {
        Vec2f32 uv;
        file.read((char *)&uv, sizeof(Vec2f32));
        ((Vertex *)mesh.vertex_host_buffer->host_address)[i].uv = VecTypeCast<f16>(uv);
      }

      file.seekg(bin_offset + (u32)index_buffer_view_object.FindNoFail("byteOffset").value_arr[0].number,
                 std::ios::beg);

      const u32 index_stride = GetGlbAccessorComponentTypeSize(static_cast<GlbAccessorComponentType>(
          index_accessor_object.FindNoFail("componentType").value_arr[0].number));

      mesh.index_count = index_count;
      for (u32 i = 0; i < index_count; i++) {
        file.read((char *)&((Index *)mesh.index_host_buffer->host_address)[i], index_stride);
      }

      mesh.aabb.min.z = position_accessor_object.FindNoFail("min").value_arr[0].number;
      mesh.aabb.min.y = position_accessor_object.FindNoFail("min").value_arr[1].number;
      mesh.aabb.min.x = position_accessor_object.FindNoFail("min").value_arr[2].number;

      mesh.aabb.max.z = position_accessor_object.FindNoFail("max").value_arr[0].number;
      mesh.aabb.max.y = position_accessor_object.FindNoFail("max").value_arr[1].number;
      mesh.aabb.max.x = position_accessor_object.FindNoFail("max").value_arr[2].number;
    }
  }

  for (u32 i = 0; i < node_array.value_count; i++) {
    const JsonObject *node_object = node_array.value_arr[i].object;

    const auto mesh_id_res = node_object->Find("mesh");
    if (!mesh_id_res.has_value()) {
      continue;
    }
    const u32 mesh_id = mesh_id_res->value_arr[0].number;

    Vec3f32 scale = 1.0f;
    if (node_object->Find("scale").has_value()) {
      scale.x = node_object->Find("scale")->value_arr[0].number;
      scale.y = node_object->Find("scale")->value_arr[1].number;
      scale.z = node_object->Find("scale")->value_arr[2].number;
    }

    Quat rotation;
    if (node_object->Find("rotation").has_value()) {
      rotation.y = node_object->Find("rotation")->value_arr[0].number;
      rotation.z = node_object->Find("rotation")->value_arr[1].number;
      rotation.x = node_object->Find("rotation")->value_arr[2].number;
      rotation.w = node_object->Find("rotation")->value_arr[3].number;
    }

    Vec3f32 translation;
    if (node_object->Find("translation").has_value()) {
      translation.x = node_object->Find("translation")->value_arr[0].number;
      translation.y = node_object->Find("translation")->value_arr[1].number;
      translation.z = node_object->Find("translation")->value_arr[2].number;
    }

    const Mat4f32 instance_matrix = InstanceMatrix(translation, rotation, scale);
    const u32 primitive_count = mesh_array.value_arr[mesh_id].object->FindNoFail("primitives").value_count;
    for (u32 j = mesh_index_offset[mesh_id]; j < mesh_index_offset[mesh_id] + primitive_count; j++) {
      InstanceData &instance = object_data.instance_data_arr.emplace_back();
      instance.mesh_index = j;
      instance.matrix = instance_matrix;
    }
  }

  json_object->Free();
  free(json_object);
}
} // namespace Core
