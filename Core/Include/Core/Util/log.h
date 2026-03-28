#pragma once

#include <format>
#include <functional>

namespace Core {
extern std::function<void(const std::string &)> LogFn;

template <class... Types> static void Log(std::format_string<Types...> fmt, Types &&...args) {
  LogFn("[Core] " + std::format(fmt, std::forward<Types>(args)...));
}
}
