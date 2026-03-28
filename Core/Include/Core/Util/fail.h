#pragma once

#include <format>
#include <functional>

namespace Core {
extern std::function<void(const std::string &)> FailFn;

template <class... Types>
static void Assert(const bool condition, std::format_string<Types...> fmt, Types &&...args) {
  if (!condition) {
    FailFn("[Core] " + std::format(fmt, std::forward<Types>(args)...));
    exit(0);
  }
}
}; // namespace Core
