#include "Core/Util/log.h"

#include <iostream>

namespace Core {
std::function<void(const std::string &)> LogFn = [](const std::string &log) {
  std::cout << log << '\n';
};
};
