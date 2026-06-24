#pragma once

float3 F_Schlick(const float3 f_0, const float v_dot_h) {
  return f_0 + (1.0f - f_0) * pow(1.0f - v_dot_h, 5.0f);
}

float G_1_SmithGGX(const float3 w_o, const float3 n, const float a2) {
  const float n_dot_v = dot(n, w_o);
  const float denom = sqrt(a2 + (1.0f - a2) * n_dot_v * n_dot_v) + n_dot_v;

  return 2.0f * n_dot_v / denom;
}

float G_2_SmithGGX(const float3 w_i, const float3 w_o, const float3 n, const float a2) {
  const float n_dot_l = dot(n, w_i);
  const float n_dot_v = dot(n, w_o);

  const float denom_1 = n_dot_l * sqrt(a2 + (1.0f - a2) * n_dot_l * n_dot_l);
  const float denom_2 = n_dot_v * sqrt(a2 + (1.0f - a2) * n_dot_v * n_dot_v);

  return 2.0f * n_dot_l * n_dot_v / (denom_1 + denom_2);
}
