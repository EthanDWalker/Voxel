#pragma once

#include "Core/Util/Parse/object.h"
#include <filesystem>

namespace Core {
void ParseGlbFile(const std::filesystem::path &file_path, ObjectData &object_data);
} // namespace Core
