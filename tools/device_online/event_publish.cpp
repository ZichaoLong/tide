#include "event_reverse.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_event_publish.h"
#include <map>
#include <stdexcept>

namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
void append_event_publish(CannProgram& p,const std::vector<EventAttentionTape>& groups,const ParameterVjp& registry,
    const at::Tensor& values,const at::Tensor& error,int64_t budget) {
  std::map<std::string,std::pair<int64_t,at::Tensor>> owners;
  for(size_t i=0;i<registry.owners.size();++i)if(registry.offsets[i]>=0)
    for(const auto& name:registry.owners[i].aliases)owners.emplace(name,std::make_pair(registry.offsets[i],registry.owners[i].value));
  int64_t used=0;
  for(const auto& a:groups) {
    const auto dtype=a.qkv.scalar_type();const int64_t fp16=dtype==at::kHalf;
    if((dtype!=at::kFloat&&dtype!=at::kHalf)||a.projection.scalar_type()!=dtype)
      throw std::invalid_argument("invalid attention publication payload dtype");
    std::vector<int64_t> plan,tiles{0};const int64_t w=a.width,kv=w/a.heads*a.kv_heads,cols=w+2*kv;
    for(size_t n=0;n<a.nodes.size();++n)for(int64_t kind=0;kind<4;++kind) {
      const std::string suffix=kind==0?"attn_q":kind==1?"attn_k":kind==2?"attn_v":"attn_out";
      const auto it=owners.find("nodes."+std::to_string(a.nodes[n])+".extra."+suffix);if(it==owners.end())continue;
      const int64_t column=kind==1?w:kind==2?w+kv:0,length=kind==0||kind==3?w:kv;
      if(it->second.second.sizes()!=at::IntArrayRef({w,length})||it->second.second.scalar_type()!=dtype)
        throw std::invalid_argument("attention publication owner shape/dtype mismatch");
      plan.insert(plan.end(),{it->second.first,kind==3?1:0,int64_t(n)*w*(kind==3?w:cols)+column,w,length,kind==3?w:cols});
      tiles.push_back(tiles.back()+w*((length+255)/256));
    }
    if(plan.empty())continue;
    const long double next=used+8.L*(plan.size()+tiles.size());if(next>budget)throw std::invalid_argument("attention publication budget exceeded");
    used=static_cast<int64_t>(next);
    auto table=at::tensor(plan,at::kLong).reshape({-1,6}).to(values.device()),offsets=at::tensor(tiles,at::kLong).to(values.device());
    const int64_t count=plan.size()/6,tasks=tiles.back();
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_event_publish)(32,stream,
      ptr(table),ptr(offsets),ptr(values),ptr(a.qkv),ptr(a.projection),ptr(error),count,tasks,fp16),"publish attention parameter aliases");},
      {table,offsets,values,a.qkv,a.projection,error});
  }
}
} // namespace tide::device_online
