#include "Core/input.h"
#include "Core/window.h"
#include "GLFW/glfw3.h"
#include <bitset>

namespace Core {

std::bitset<static_cast<u64>(Input::COUNT)> InputContext::_pressed_input;
std::bitset<static_cast<u64>(Input::COUNT)> InputContext::_held_input;
std::bitset<static_cast<u64>(Input::COUNT)> InputContext::_released_input;

Vec2u32 InputContext::window_size;
Vec2f32 InputContext::mouse_pos;
Vec2f32 InputContext::delta_mouse_pos;

void InputContext::StartUp() {
  ZoneScoped;
  Update();
};

void InputContext::Update() {
  ZoneScoped;
  glfwPollEvents();

  GLFWwindow *window = (GLFWwindow *)Window::handle;

  Vec2i32 window_size_int;
  glfwGetWindowSize(window, &window_size_int.x, &window_size_int.y);
  Vec2f64 mouse_position_double;
  glfwGetCursorPos(window, &mouse_position_double.x, &mouse_position_double.y);

  window_size = Vec2u32::From(window_size_int);

  Vec2f32 new_mouse_position =
      (VecTypeCast<f32>(mouse_position_double) + Vec2f32(0.5f)) / VecTypeCast<f32>(window_size);
  new_mouse_position = new_mouse_position * 2.0f - Vec2f32(1.0f);

  delta_mouse_pos = mouse_pos - new_mouse_position;
  mouse_pos = new_mouse_position;

  _pressed_input = 0;
  _released_input = 0;
  for (u8 i = 0; i < static_cast<u8>(Input::KEYBOARD_COUNT); i++) {
    bool input_down = glfwGetKey(window, KEY_TO_GLFW_KEY[i]);

    _pressed_input[i] = !GetHeld((Input)i) && input_down;
    _released_input[i] = GetHeld((Input)i) && !input_down;
  }

  for (u8 i = static_cast<u8>(Input::KEYBOARD_COUNT); i < static_cast<u8>(Input::COUNT); i++) {
    bool input_down =
        glfwGetMouseButton(window, MOUSE_TO_GLFW_MOUSE[i - static_cast<u8>(Input::KEYBOARD_COUNT)]);

    _pressed_input[i] = !GetHeld((Input)i) && input_down;
    _released_input[i] = GetHeld((Input)i) && !input_down;
  }

  _held_input = 0;
  for (u8 i = 0; i < static_cast<u8>(Input::KEYBOARD_COUNT); i++) {
    bool input_down = glfwGetKey(window, KEY_TO_GLFW_KEY[i]);

    _held_input[i] = input_down;
  }

  for (u8 i = static_cast<u8>(Input::KEYBOARD_COUNT); i < static_cast<u8>(Input::COUNT); i++) {
    bool input_down =
        glfwGetMouseButton(window, MOUSE_TO_GLFW_MOUSE[i - static_cast<u8>(Input::KEYBOARD_COUNT)]);

    _held_input[i] = input_down;
  }
}
} // namespace Core
