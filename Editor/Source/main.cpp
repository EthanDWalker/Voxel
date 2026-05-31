#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>

#include "Core/boot.h"
#include "editor.h"

int main() {
  _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
  Core::StartUp();

  {
    Editor editor{};

    editor.StartUp();

    editor.Run();

    editor.ShutDown();
  }

  Core::ShutDown();
}
