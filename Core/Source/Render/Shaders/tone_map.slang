#include "Include/camera.slang"
#include "Include/color.slang"
#include "Include/common.slang"

struct FrameLuminanceData {
  float acc_log_luminance;
  float total_weight;
};

[[vk::binding(0, 0)]]
RWTexture2D<float4> hdr_image;

[[vk::binding(0, 1)]]
RWTexture2D<float4> swapchain_image;

[[vk::binding(0, 2)]]
RWStructuredBuffer<FrameLuminanceData> frame_luminance_data;

[[vk::binding(1, 2)]]
StructuredBuffer<FrameLuminanceData> last_frame_luminance_data;

struct PushConstants {
  float delta_time;
};

[numthreads(8, 8, 1)]
[shader("compute")]
void CMain(const uint32_t2 pixel_index: SV_DispatchThreadID, const uniform PushConstants info) {
  uint32_t2 image_size;
  hdr_image.GetDimensions(image_size.x, image_size.y);

  if (all(image_size <= pixel_index)) {
    return;
  }

  if (all(pixel_index == 0)) {
    frame_luminance_data[0].acc_log_luminance =
        lerp(last_frame_luminance_data[0].acc_log_luminance, frame_luminance_data[0].acc_log_luminance,
             info.delta_time * 2.0f);
  }

  const float3 hdr_color = hdr_image.Load(pixel_index).rgb;

  const float last_average_luminance =
      exp2((last_frame_luminance_data[0].acc_log_luminance / last_frame_luminance_data[0].total_weight) *
               LOG_LUMINANCE_RANGE +
           MIN_LOG_LUMINANCE);

  float3 xyY_color = RGBToxyY(hdr_color);

  xyY_color.z *= CAMERA_EXPOSURE;

  xyY_color.z = xyY_color.z / (9.6f * last_average_luminance);

  xyY_color.z = saturate(TonemapACES(xyY_color.z));

  swapchain_image.Store(pixel_index, float4(LinearToSRGB(xyYToRGB(xyY_color)), 1.0f));
}
