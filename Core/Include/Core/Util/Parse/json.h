#pragma once

#include "Core/Util/fail.h"
#include <fstream>
#include <optional>

namespace Core {
enum class JsonValueType : u8 {
  String,
  Number,
  Object,
  Array,
};

struct JsonObject;
struct JsonArray;

union JsonValueUnion {
  JsonObject *object;
  JsonArray *array;
  f64 number;
  char *string;
  void *data;
};

struct JsonArray {
  JsonValueUnion *value_arr;
  JsonValueType *value_type_arr;
  u32 value_count;

  void Free();

  u32 PushBack(const u32 append_count = 1);
};

struct JsonObject {
  char **name_arr;
  JsonArray *arr_arr;
  u32 arr_count;

  void Free();

  u32 PushBack();

  const std::optional<JsonArray> Find(const char *name) const;

  const JsonArray FindNoFail(const char *name) const {
    const std::optional<JsonArray> opt = Find(name);
    Assert(opt.has_value(), "Unable to find {}", name);
    return opt.value();
  }
};

JsonObject ParseJsonStream(std::ifstream &file, const u64 file_size);

void PrintJsonGraph(const JsonObject &object, const std::string &indentation = "");
}; // namespace Core
