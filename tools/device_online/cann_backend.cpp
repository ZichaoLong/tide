#include "device_backend.h"
#include "cann_api.h"
#include <stdexcept>

namespace tide::device_online {
void validate_kernel_device(c10::Device device) {
  if (device.type()!=resident_device_type) throw std::invalid_argument("resident backend requires an NPU");
#ifdef TIDE_ASCENDC_SOC
  CannApi api;
  const auto soc=CannApi::symbol<const char*(*)()>(api.runtime,"aclrtGetSocName")();
  if (!soc || std::string(soc)!=TIDE_ASCENDC_SOC)
    throw std::runtime_error("resident kernel differs from actual SoC");
#endif
}
void check_device_launch(int status,const char* operation) { CannApi::check(status,operation); }
}
