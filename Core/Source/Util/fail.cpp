#include "Core/Util/fail.h"
#include "Core/Util/log.h"

namespace Core {
std::function<void(const std::string &)> FailFn = [](const std::string &log) {
  LogFn(log);
};
};
