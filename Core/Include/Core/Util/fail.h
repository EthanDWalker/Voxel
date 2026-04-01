#pragma once

#include <format>
#include <iostream>

namespace Core {
template <class... Types>
static void Assert(const bool condition, std::format_string<Types...> fmt, Types &&...args) {
  if (!condition) {
    std::cout << std::format(fmt, std::forward<Types>(args)...) << '\n';
    exit(0);
  }
}
}; // namespace Core
