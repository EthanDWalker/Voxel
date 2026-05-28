#include "Core/boot.h"
#include "editor.h"

int main() {
  Core::StartUp();

  {
    Editor editor{};

    editor.StartUp();

    editor.Run();

    editor.ShutDown();
  }

  Core::ShutDown();
}
