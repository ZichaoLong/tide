#pragma once
#include "eager_capacity.h"
#include "memory.h"
#include "failure.h"

namespace tide_flow::eager_capacity {
// Admission and observation are consumer boundaries, never scheduling inputs.
class Admission {
 public:
  Admission(const Packet& p,const Config& c,const std::vector<at::Device>& devices):packet_(p),config_(c) {
    std::vector<Index> budgets;
    for(auto d:devices) {
      auto info=device_memory_info(d);initial_.push_back(info);
      const auto limit=d.is_cpu()?info.free/2:info.free;
      budgets.push_back(c.device_memory_bytes?std::min(c.device_memory_bytes,limit):limit);
    }
    plan_=plan(p,c,budgets);
    if(!plan_.accepted) {
      const std::string error="eager physical sample/head memory envelope exceeds budget before model allocation";
      throw RecordedFailure(error,"{\"schema\":\"tide-online-consumer-v1\",\"state\":\"failed\",\"workload_sha256\":"+quoted(p.sha)+",\"error\":"+quoted(error)+
        ",\"failure_phase\":\"preallocation_memory_admission\",\"memory_admission\":"+json()+"}\n");
    }
  }
  Index rows() const {return plan_.rows;}
  bool observe(const MemoryRecord& memory) {
    peaks_=initial_[0].device.is_cpu()?std::vector<Index>{memory.cpu_peak_growth()}:memory.peak_growth();
    if(peaks_.size()!=plan_.cards.size())throw std::invalid_argument("eager memory observation device mismatch");
    observed_=true;within_=true;
    for(size_t i=0;i<peaks_.size();++i)within_&=peaks_[i]<=plan_.cards[i].peak;
    return within_;
  }
  std::string json() const {
    std::ostringstream shape;record(shape,packet_,config_,plan_);
    auto prefix=shape.str();prefix.pop_back();std::ostringstream out;out<<prefix;
    out<<",\"requested_device_memory_bytes\":"<<config_.device_memory_bytes
       <<",\"cpu_budget_policy\":\"at most half currently available host memory\",\"initial_devices\":[";
    for(size_t i=0;i<initial_.size();++i) {
      if(i)out<<',';const auto& d=initial_[i];
      out<<"{\"device\":"<<quoted(d.device.str())<<",\"free_bytes\":"<<d.free
         <<",\"total_bytes\":"<<d.total<<",\"allocated_bytes\":"<<d.allocated<<",\"counter\":"
         <<quoted(d.device.is_cpu()?"process-lifetime peak RSS; incremental growth is a proxy, not allocator accounting":"framework allocator")<<'}';
    }
    out<<']';if(observed_) {
      out<<",\"observed_peak_growth_bytes\":";array(out,peaks_);
      out<<",\"allocator_within_estimate\":"<<(within_?"true":"false");
    }
    out<<'}';return out.str();
  }
 private:
  const Packet& packet_;const Config& config_;Plan plan_;
  std::vector<DeviceMemoryInfo> initial_;std::vector<Index> peaks_;
  bool observed_=false,within_=false;
};
} // namespace tide_flow::eager_capacity
