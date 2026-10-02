#pragma once
#include <array>
#include <iomanip>
#include <sstream>
#include <vector>

namespace tide_flow {
struct PhaseTiming {
  bool enabled=false;
  std::vector<std::array<double,2>> measured,warmup;
  void add(double total,double sample,bool warming) {
    if(enabled)(warming?warmup:measured).push_back({sample<0?total:sample,sample<0?0:total-sample});
  }
  std::string json() const {
    std::ostringstream out;out<<std::setprecision(17)<<"{\"enabled\":"<<(enabled?"true":"false");
    auto rows=[&](const char* name,const auto& values) {
      out<<",\""<<name<<"\":[";bool first=true;
      for(const auto& v:values){if(!first)out<<',';first=false;
        out<<"{\"sample_work_seconds\":"<<v[0]<<",\"optimizer_seconds\":"<<v[1]<<'}';}
      out<<']';
    };
    rows("measured",measured);rows("warmup",warmup);
    out<<",\"boundary\":\"training: all resolved devices synchronized before optimizer; extra synchronization included in total\""
       <<",\"scope\":\"sample work includes input/forward/loss/backward/accumulation/continuation; optimizer includes final finite checks and publication; inference has zero optimizer time\"}";
    return out.str();
  }
};
} // namespace tide_flow
