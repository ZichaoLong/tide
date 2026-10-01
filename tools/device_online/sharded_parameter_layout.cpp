#include "sharded_parameter_sources.h"
#include "parameter_plan.h"
#include <limits>
#include <set>
#include <stdexcept>
namespace tide::device_online {
std::vector<ParameterVjp> sharded_parameter_layout(const Graph& graph,const ParameterRegistry& registry,int64_t width,
    const std::vector<at::Device>& devices,int64_t budget,bool controls) {
  if(devices.empty()||budget<1)throw std::invalid_argument("canonical parameter layout needs bounded devices");
  std::set<c10::DeviceIndex> seen;
  for(auto d:devices)if(d.type()!=c10::DeviceType::PrivateUse1||d.index()<0||!seen.insert(d.index()).second)
    throw std::invalid_argument("canonical parameter layout needs distinct explicit NPUs");
  const auto plan=plan_parameters(graph,registry,width,std::numeric_limits<int64_t>::max(),controls);
  std::vector<bool> active;for(auto offset:plan.offsets)active.push_back(offset>=0);
  const auto placement=place_parameter_owners(plan.owners,active,devices.size());
  std::vector<ParameterVjp> out(devices.size());std::vector<int64_t> elements(devices.size(),0);
  long double bytes=64.L*plan.owners.size()+4096.L*devices.size();
  for(size_t i=0;i<plan.owners.size();++i) {
    const auto d=placement[i];auto& p=out[d];const auto size=plan.owners[i].value.numel();
    p.owners.push_back(plan.owners[i]);p.offsets.push_back(active[i]?elements[d]:-1);
    if(active[i]) {
      if(size>std::numeric_limits<int64_t>::max()-elements[d])throw std::invalid_argument("canonical layout extent overflow");
      elements[d]+=size;bytes+=4.L*size;
    }
  }
  if(bytes>budget)throw std::invalid_argument("canonical parameter layout exceeds tensor budget");
  for(size_t d=0;d<devices.size();++d) {
    auto options=at::TensorOptions().device(devices[d]).dtype(at::kFloat);
    out[d].values=at::zeros({std::max<int64_t>(1,elements[d])},options);
    out[d].connected=at::zeros({std::max<int64_t>(1,out[d].owners.size())},options.dtype(at::kBool));
  }
  return out;
}
} // namespace tide::device_online
