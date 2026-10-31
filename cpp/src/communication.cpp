#include "tide/communication.h"
#include <c10/core/impl/VirtualGuardImpl.h>
#include <cstdlib>
#include <atomic>
#include <stdexcept>

namespace tide {
namespace {
std::atomic<Index> copies{0}, bytes{0}, legs{0}, fences{0}, bridges{0};
}
std::map<std::string,Index> communication_statistics(bool reset) {
  auto take=[reset](std::atomic<Index>& x){return reset?x.exchange(0):x.load();};
  return {{"copies",take(copies)},{"logical_bytes",take(bytes)},{"tensor_leg_bytes",take(legs)},
          {"endpoint_synchronizations",take(fences)},{"cpu_bridge_copies",take(bridges)}};
}
std::string communication_policy() {
  const auto* raw = std::getenv("TIDE_COMMUNICATION_POLICY");
  std::string policy = raw ? raw : "ordinary";
  if (policy != "ordinary" && policy != "blocking" && policy != "cpu_bridge")
    throw std::invalid_argument("unknown TIDE_COMMUNICATION_POLICY");
  return policy;
}
namespace {
void synchronize_endpoint(at::Device device) {
  if (!device.is_cpu()) c10::impl::VirtualGuardImpl(device.type()).synchronizeDevice(device.index());
}
}
Tensor communication_copy(const Tensor& value, at::Device destination, at::ScalarType dtype, const std::string& policy) {
  if (policy != "ordinary" && policy != "blocking" && policy != "cpu_bridge")
    throw std::invalid_argument("unknown explicit communication policy");
  auto options = value.options().device(destination).dtype(dtype);
  if (value.device() == destination) return value.to(options);
  if (policy != "ordinary") { synchronize_endpoint(value.device()); synchronize_endpoint(destination); }
  const bool bridge = policy == "cpu_bridge" && !value.device().is_cpu() && !destination.is_cpu();
  auto result = (bridge ? value.to(at::kCPU) : value).to(options);
  if (policy != "ordinary") { synchronize_endpoint(value.device()); synchronize_endpoint(destination); }
  ++copies;bytes+=value.nbytes();legs+=value.nbytes()*(bridge?2:1);bridges+=bridge?1:0;
  if(policy!="ordinary")fences+=2*int(!value.device().is_cpu())+2*int(!destination.is_cpu());
  return result;
}
} // namespace tide
