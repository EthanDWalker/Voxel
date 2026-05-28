#include "Core/Util/Parse/object.h"
#include <filesystem>
#include <fstream>

namespace Core {

void WriteMeshFile(const std::filesystem::path &folder_path, const MeshData &mesh_data) {
  ZoneScoped;
  const std::filesystem::path path = folder_path / std::format("{}.mesh", mesh_data.name);

  std::ofstream file = std::ofstream(path, std::ios::binary | std::ios::out);

  Assert(file.is_open(), "failed to open {} file", mesh_data.name);

  MeshFileHeader header;
  header.aabb = mesh_data.aabb;
  header.vertex_count = mesh_data.vertex_count;
  header.index_count = mesh_data.index_count;
  header.material_index = mesh_data.material_index;

  file.write((const char *)&header, sizeof(MeshFileHeader));

  file.write((const char *)mesh_data.vertex_host_buffer->host_address, mesh_data.vertex_host_buffer->size);
  file.write((const char *)mesh_data.index_host_buffer->host_address, mesh_data.index_host_buffer->size);
}

void ReadMeshFile(MeshData &mesh_data, const std::filesystem::path &file_path) {
  ZoneScoped;
  std::ifstream file = std::ifstream(file_path, std::ios::in | std::ios::binary);

  MeshFileHeader header;
  file.read((char *)&header, sizeof(MeshFileHeader));

  mesh_data.vertex_count = header.vertex_count;
  mesh_data.index_count = header.index_count;
  mesh_data.material_index = header.material_index;
  mesh_data.name = file_path.stem().string();
  mesh_data.aabb = header.aabb;
  mesh_data.vertex_host_buffer->Create(header.vertex_count * sizeof(Vertex));
  mesh_data.index_host_buffer->Create(header.index_count * sizeof(Index));

  file.read((char *)mesh_data.vertex_host_buffer->host_address, mesh_data.vertex_host_buffer->size);
  file.read((char *)mesh_data.index_host_buffer->host_address, mesh_data.index_host_buffer->size);
}

void WriteMaterialFile(const std::filesystem::path &folder_path, const MaterialData &material_data) {
  ZoneScoped;
  if (!material_data.initialized) {
    return;
  }
  const std::filesystem::path path = folder_path / std::format("{}.material", material_data.name);

  std::ofstream file = std::ofstream(path, std::ios::binary | std::ios::out);

  Assert(file.is_open(), "failed to open {} file", material_data.name);

  MaterialFileHeader header;
  header.extent = material_data.albedo_extent;
  header.image_data_size = material_data.compressed_albedo_data_buffer->size;

  file.write((const char *)&header, sizeof(MaterialFileHeader));
  file.write((const char *)material_data.compressed_albedo_data_buffer->host_address, header.image_data_size);
}

void ReadMaterialFile(MaterialData &material_data, const std::filesystem::path &file_path) {
  ZoneScoped;
  std::ifstream file = std::ifstream(file_path, std::ios::in | std::ios::binary);

  MaterialFileHeader header;
  file.read((char *)&header, sizeof(MaterialFileHeader));

  material_data.albedo_extent = header.extent;
  material_data.compressed_albedo_data_buffer->Create(header.image_data_size);
  material_data.initialized = true;
  file.read((char *)material_data.compressed_albedo_data_buffer->host_address, header.image_data_size);

  material_data.name = file_path.stem().string();
}

// writes length of string then string w/o null terminal character at the end
void WriteString(std::ofstream &file, const std::string &string) {
  ZoneScoped;
  const u32 name_size = string.size();
  file.write((const char *)&name_size, sizeof(u32));

  if (name_size == 0)
    return;

  const char *name = string.c_str();
  file.write(name, name_size * sizeof(char));
}

void WriteObjectFolder(const std::filesystem::path &folder_path, const ObjectData &object_data) {
  ZoneScoped;
  if (!std::filesystem::exists(folder_path)) {
    std::filesystem::create_directories(folder_path);
  }

  const std::filesystem::path mesh_folder_path = folder_path / "Meshes";
  if (!std::filesystem::exists(mesh_folder_path)) {
    std::filesystem::create_directories(mesh_folder_path);
  }

  for (u32 i = 0; i < object_data.mesh_data_arr.size(); i++) {
    WriteMeshFile(mesh_folder_path, object_data.mesh_data_arr[i]);
  }

  const std::filesystem::path material_folder_path = folder_path / "Materials";
  if (!std::filesystem::exists(material_folder_path)) {
    std::filesystem::create_directories(material_folder_path);
  }

  for (u32 i = 0; i < object_data.material_data_arr.size(); i++) {
    WriteMaterialFile(material_folder_path, object_data.material_data_arr[i]);
  }

  std::ofstream file = std::ofstream(folder_path / std::format("{}.object", object_data.name),
                                     std::ios::out | std::ios::binary);

  ObjectFolderHeader header;
  header.material_descriptor_count = object_data.material_data_arr.size();
  header.mesh_descriptor_count = object_data.mesh_data_arr.size();
  header.instance_data_count = object_data.instance_data_arr.size();

  file.write((const char *)&header, sizeof(ObjectFolderHeader));

  for (u32 i = 0; i < header.material_descriptor_count; i++) {
    WriteString(file, object_data.material_data_arr[i].name);
  }

  for (u32 i = 0; i < header.mesh_descriptor_count; i++) {
    WriteString(file, object_data.mesh_data_arr[i].name);
  }

  file.write((const char *)object_data.instance_data_arr.data(),
             sizeof(InstanceData) * object_data.instance_data_arr.size());
}

// reads string with length as u32 in front of string
void ReadString(std::ifstream &file, std::string &string) {
  ZoneScoped;
  u32 size;
  file.read((char *)&size, sizeof(u32));

  if (size == 0)
    return;

  string.resize(size);

  file.read(string.data(), sizeof(char) * size);
}

void ReadObjectFolder(const std::filesystem::path &folder_path, ObjectData &object_data) {
  ZoneScoped;
  Assert(std::filesystem::exists(folder_path), "object folder {} doesnt exist", folder_path.string());

  std::ifstream file = std::ifstream(folder_path / std::format("{}.object", folder_path.stem().string()),
                                     std::ios::in | std::ios::binary);

  ObjectFolderHeader header;
  file.read((char *)&header, sizeof(ObjectFolderHeader));

  const std::filesystem::path material_folder_path = folder_path / "Materials";
  Assert(std::filesystem::exists(material_folder_path), "material folder {} doesnt exist",
         material_folder_path.string());

  object_data.material_data_arr.resize(header.material_descriptor_count);

  for (u32 i = 0; i < header.material_descriptor_count; i++) {
    std::string material_file_name;
    ReadString(file, material_file_name);

    if (material_file_name.empty())
      continue;

    const std::filesystem::path material_file_path =
        material_folder_path / std::format("{}.material", material_file_name);

    ReadMaterialFile(object_data.material_data_arr[i], material_file_path);
  }

  const std::filesystem::path mesh_folder_path = folder_path / "Meshes";
  Assert(std::filesystem::exists(mesh_folder_path), "meshes folder {} doesnt exist",
         mesh_folder_path.string());

  object_data.mesh_data_arr.resize(header.mesh_descriptor_count);

  for (u32 i = 0; i < header.mesh_descriptor_count; i++) {
    std::string mesh_file_name;
    ReadString(file, mesh_file_name);

    const std::filesystem::path mesh_file_path = mesh_folder_path / std::format("{}.mesh", mesh_file_name);

    ReadMeshFile(object_data.mesh_data_arr[i], mesh_file_path);
  }

  object_data.instance_data_arr.resize(header.instance_data_count);

  file.read((char *)object_data.instance_data_arr.data(), sizeof(InstanceData) * header.instance_data_count);
}

} // namespace Core
