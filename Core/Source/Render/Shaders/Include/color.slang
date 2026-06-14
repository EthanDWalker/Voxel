#pragma once

static const float MIN_LOG_LUMINANCE = log2(0.003f);
static const float MAX_LOG_LUMINANCE = log2(11.0f);
static const float INV_LOG_LUMINANCE_RANGE = 1.0f / (MAX_LOG_LUMINANCE - MIN_LOG_LUMINANCE);

float TonemapACES(const float x) {
  const float a = 2.51f;
  const float b = 0.03f;
  const float c = 2.43f;
  const float d = 0.59f;
  const float e = 0.14f;
  return (x * (a * x + b)) / (x * (c * x + d) + e);
}

float GetLuminance(const float3 color) { return dot(color, float3(0.2125f, 0.7154f, 0.0721f)); }

float3 RGBToXYZ(const float3 rgb) {
  // clang-format off
  return float3(
    0.4124564f * rgb.r + 0.3575761f * rgb.g + 0.1804375f * rgb.b,
    0.2126729f * rgb.r + 0.7151522f * rgb.g + 0.0721750f * rgb.b,
    0.0193339f * rgb.r + 0.1191920f * rgb.g + 0.9503041f * rgb.b
  );
  // clang-format on
}

float3 XYZtoRGB(const float3 xyz) {
  // clang-format off
  return float3(
    3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
    -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
    0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z
  );
  // clang-format on
}

static const float XYZ_xxY_EPSILON = 1e-6f;

float3 XYZToxyY(const float3 xyz) {
  const float sum = max(xyz.x + xyz.y + xyz.z, XYZ_xxY_EPSILON);
  return float3(float2(xyz.xy) / sum, xyz.y);
}

float3 xyYToXYZ(const float3 xyY) {
  const float x = xyY.x;
  const float y = xyY.y;
  const float Y = xyY.z;

  if (y < XYZ_xxY_EPSILON) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  const float X = x * Y / y;
  const float Z = (1.0f - x - y) * Y / y;

  return float3(X, Y, Z);
}

float3 RGBToxyY(const float3 rgb) { return XYZToxyY(RGBToXYZ(rgb)); }

float3 xyYToRGB(const float3 xyY) { return XYZtoRGB(xyYToXYZ(xyY)); }

float3 LinearToSRGB(const float3 linear) {
  return select(linear <= 0.0031308f, linear * 12.92f, 1.055f * pow(linear, 1.0f / 2.4f) - 0.055f);
}
