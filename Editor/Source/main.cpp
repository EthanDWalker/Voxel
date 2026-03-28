#include "Core/Render/context.h"
#include "Core/boot.h"
#include "editor.h"

int main() {
  Core::StartUp();

  {
    Core::RenderContext context{};
    context.Create(Core::Spec());
    Core::render_context = &context;

    Editor editor{};

    editor.StartUp();

    editor.Run();

    editor.ShutDown();
  }

  Core::ShutDown();
}
