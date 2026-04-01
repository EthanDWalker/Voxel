#pragma once

#include <format>
#include <iostream>

namespace Core {
template <class... Types> static void Log(std::format_string<Types...> fmt, Types &&...args) {
  std::cout << ("[Core] " + std::format(fmt, std::forward<Types>(args)...)) << '\n';
}
} // namespace Core
