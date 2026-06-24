#include "Core/Util/timer.h"
#include "Core/boot.h"
#include "editor.h"

int main() {
  {
    SCOPED_TIMER("core start up");
    Core::StartUp();
  }

  {
    Editor editor{};

    {
      SCOPED_TIMER("editor start up");
      editor.StartUp();
    }

    editor.Run();

    {
      SCOPED_TIMER("editor shut down");
      editor.ShutDown();
    }
  }

  {
    SCOPED_TIMER("core shut down");
    Core::ShutDown();
  }
}
