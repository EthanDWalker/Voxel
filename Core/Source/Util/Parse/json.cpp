#include "Core/Util/Parse/json.h"
#include <cstring>

namespace Core {
enum JsonToken : char {
  BeginObject = '{',
  EndObject = '}',
  BeginArray = '[',
  EndArray = ']',
  StringIndicator = '"',
  ValueIndicator = ':',
  Comma = ',',
  DecimalPoint = '.',
  NegativeSign = '-',
};

void JsonArray::Free() {
  for (u32 i = 0; i < value_count; i++) {
    if (value_type_arr[i] == JsonValueType::Array) {
      value_arr[i].array->Free();
    }
    if (value_type_arr[i] == JsonValueType::Object) {
      value_arr[i].object->Free();
    }
    free(value_arr[i].data);
  }
  free(value_arr);
  free(value_type_arr);
}

void JsonObject::Free() {
  for (u32 i = 0; i < arr_count; i++) {
    free((void *)name_arr[i]);
    arr_arr[i].Free();
  }
  free(name_arr);
}

const std::optional<JsonArray> JsonObject::Find(const char *name) const {
  for (u32 i = 0; i < arr_count; i++) {
    if (strcmp(name, name_arr[i]) == 0) {
      return arr_arr[i];
    }
  }

  return std::nullopt;
}

u32 JsonObject::PushBack() {
  arr_count++;
  name_arr = (char **)realloc(name_arr, sizeof(char *) * arr_count);
  arr_arr = (JsonArray *)realloc(arr_arr, sizeof(JsonArray) * arr_count);
  memset(&arr_arr[arr_count - 1], 0, sizeof(JsonArray));
  return arr_count - 1;
}

u32 JsonArray::PushBack(const u32 append_count) {
  value_count += append_count;
  value_type_arr = (JsonValueType *)realloc(value_type_arr, sizeof(JsonValueType) * value_count);
  value_arr = (JsonValueUnion *)realloc(value_arr, sizeof(void *) * value_count);
  return value_count - append_count;
}

void ParseJsonString(u64 &cursor, const char *file_data, char **string) {
  static char string_buff[1000];

  u32 size = 0;
  for (;; size++) {
    if (size == 1000)
      Assert(false, "string in json file too long (more than 1k chars)");
    if (file_data[cursor + size + 1] == JsonToken::StringIndicator)
      break;
    string_buff[size] = file_data[cursor + size + 1];
  }
  cursor += size + 2;
  *string = (char *)malloc(size + 1);
  memcpy(*string, string_buff, size);
  (*string)[size] = '\0';
}

constexpr const u8 CharToNumber(const char c) {
  // '0' = 48
  // '9' = 57
  return (u8)c - 48;
}

constexpr const bool IsNumber(const char c) {
  // '0' = 48
  // '9' = 57
  return c >= 48 && c <= 57;
}

JsonValueType ParseJsonValue(u64 &cursor, const char *file_data, JsonValueUnion *data) {
  switch (file_data[cursor]) {
  case JsonToken::StringIndicator: {
    ParseJsonString(cursor, file_data, (char **)data);
    return JsonValueType::String;
    break;
  }

  // number value
  default: {
    u32 digits = 0;
    // decimal place cant be zero since json decimal is 0.xx
    u32 decimal_place = 0;
    bool negative = false;
    if (file_data[cursor] == JsonToken::NegativeSign) {
      negative = true;
      cursor++;
    }

    for (;; digits++) {
      if (file_data[cursor + digits] == JsonToken::DecimalPoint) {
        decimal_place = digits;
      } else if (!IsNumber(file_data[cursor + digits])) {
        if (decimal_place == 0) {
          decimal_place = digits;
        }
        break;
      }
    }

    f64 value = 0.0f;
    for (u32 i = 0; i < decimal_place; i++) {
      const char digit = file_data[cursor + i];
      value += std::pow(10, (decimal_place - i) - 1) * CharToNumber(digit);
    }

    for (u32 i = decimal_place + 1; i < digits; i++) {
      const char digit = file_data[cursor + i];
      value += 1.0f / f32(std::pow(10, i - decimal_place)) * CharToNumber(digit);
    }

    value *= negative ? -1.0f : 1.0f;

    *(f64 *)data = value;
    cursor += digits;
    return JsonValueType::Number;
    break;
  }
  }
}

JsonObject *ParseJsonObjectFromCursor(u64 &cursor, const char *file_data);

void ParseJsonArray(u64 &cursor, const char *file_data, JsonArray *array) {
  while (true) {
    if (file_data[cursor] == JsonToken::NegativeSign || IsNumber(file_data[cursor])) {
      const u32 index = array->PushBack();
      array->value_type_arr[index] = ParseJsonValue(cursor, file_data, &array->value_arr[index]);
    }
    switch (file_data[cursor]) {
    case JsonToken::BeginObject: {
      cursor++;
      const u32 index = array->PushBack();
      array->value_type_arr[index] = JsonValueType::Object;
      array->value_arr[index].object = ParseJsonObjectFromCursor(cursor, file_data);
      break;
    }
    case JsonToken::StringIndicator: {
      const u32 index = array->PushBack();
      array->value_type_arr[index] = ParseJsonValue(cursor, file_data, &array->value_arr[index]);
      break;
    }
    case JsonToken::EndArray: {
      return;
      break;
    }
    default: {
      cursor++;
      break;
    }
    }
  }
}

JsonObject *ParseJsonObjectFromCursor(u64 &cursor, const char *file_data) {
  static char string_buff[100];
  JsonObject *current_object = (JsonObject *)malloc(sizeof(JsonObject));
  memset(current_object, 0, sizeof(JsonObject));
  JsonArray *current_array;

  while (true) {
    switch (file_data[cursor]) {
    // make sure this is only found when parsing the root instance object
    case JsonToken::EndObject: {
      cursor++;
      return current_object;
      break;
    }

    case JsonToken::BeginObject: {
      cursor++;
      const u32 index = current_array->PushBack();
      current_array->value_type_arr[index] = JsonValueType::Object;
      current_array->value_arr[index].object = ParseJsonObjectFromCursor(cursor, file_data);
      break;
    }

    case JsonToken::BeginArray: {
      cursor++;
      ParseJsonArray(cursor, file_data, current_array);
      break;
    }

    // only come after string indicator which is parsed
    case JsonToken::ValueIndicator: {
      // parse basic values function
      // dont parse object or array values
      cursor++;
      if (file_data[cursor] == JsonToken::BeginObject || file_data[cursor] == JsonToken::BeginArray) {
        break;
      }
      const u32 index = current_array->PushBack();
      current_array->value_type_arr[index] =
          ParseJsonValue(cursor, file_data, &current_array->value_arr[index]);
      break;
    }

    // make sure to never have string indicator be at end of string
    case JsonToken::StringIndicator: {
      const u32 index = current_object->PushBack();
      ParseJsonString(cursor, file_data, &current_object->name_arr[index]);
      current_array = &current_object->arr_arr[index];
      break;
    }

    // white space char
    default: {
      cursor++;
      break;
    }
    }
  }
}

JsonObject ParseJsonStream(std::ifstream &file, const u64 file_size) {
  Assert(file.is_open(), "Must open file before parsing JSON");

  const char *file_data = (const char *)malloc(file_size);
  file.read((char *)file_data, file_size);

  // first NewObject token
  u64 cursor = 1;
  return *ParseJsonObjectFromCursor(cursor, file_data);
}

void PrintJsonGraph(const JsonObject &object, const std::string &indentation) {
  for (u32 i = 0; i < object.arr_count; i++) {
    Core::Log("{} - {}", indentation, object.name_arr[i]);
    for (u32 j = 0; j < object.arr_arr[i].value_count; j++) {
      JsonValueType value_type = object.arr_arr[i].value_type_arr[j];
      JsonValueUnion value = object.arr_arr[i].value_arr[j];

      if (value_type == JsonValueType::Object) {
        PrintJsonGraph(*value.object, indentation + '\t' + '\t');
      } else if (value_type == JsonValueType::Number) {
        Core::Log("{} - {}", indentation + '\t' + '\t', value.number);
      } else if (value_type == JsonValueType::String) {
        Core::Log("{} - {}", indentation + '\t' + '\t', value.string);
      }
    }
  }
}
} // namespace Core
