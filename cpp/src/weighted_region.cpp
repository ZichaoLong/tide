// Allocation §2, §4: independent C++ complete region step, no scheduler policy.
#include "weighted_detail.h"
#include "tide/weighted_sending.h"

namespace tide::weighted {
using namespace detail;
namespace {
class Sending final:public RegionKernel {
 public:
  explicit Sending(RegionalConfig c):c_(std::move(c)){c_.validate();}
  Selection step(const RegionWeights& w,const RegionInput& r) const override {
    std::vector<Tensor> received,caps,scores,evidence;std::vector<Index> nodes;
    for(const auto& d:r.candidates){
      nodes.push_back(d.node);received.push_back(d.fields.at("w"));caps.push_back(d.fields.at("mu"));
      scores.push_back(d.fields.at("score"));if(c_.regional())evidence.push_back(d.fields.at("evidence"));
    }
    auto recv=at::stack(received),mu=at::stack(caps),s=at::stack(scores),sum=recv.sum(),mass=mu.sum();
    Tensor control,low,high,result;
    if(c_.region_control_mode=="context"){
      auto z=(recv.unsqueeze(-1)*at::stack(evidence)).sum(0)/sum.clamp_min(1);
      control=head(w.extra,"head",at::cat({z,sum.log1p().reshape({1}),mass.log1p().reshape({1})}));
    }
    if(c_.total()){
      auto rho=control.defined()?control.sigmoid():at::full({},*c_.total_fraction,sum.options());
      low=sum.clamp_max(1)+rho*(mass.clamp_max(*c_.k)-sum.clamp_max(1));high=low;
      if(c_.selection_mode=="regional_total_zero")result=total_zero(s,mu,low,nodes);
    } else if(c_.attention()){
      result=attention_send(s,mu,recv,c_.selection_mode,*c_.allocation_base,at::full({},*c_.allocation_temperature,s.options()));
    } else if(c_.selection_mode=="proportional")result=proportional(mu,*c_.k);
    else {
      s=s-(control.defined()?control:at::full({},*c_.eta,s.options()));
      if(c_.hard())result=hard_selection(s,mu,*c_.k,nodes);
      low=(c_.regional()?sum:mass).clamp_max(1);high=at::full({},*c_.k,s.options());
    }
    if(!result.defined())result=allocate(s,mu,low,high,at::full({},*c_.allocation_temperature,s.options()),*c_.regularizer,c_.ent_alpha).controls;
    Selection answer;answer.history.last_time=r.time;
    for(size_t i=0;i<nodes.size();++i){answer.controls[nodes[i]]=result[i];if(result[i].item<double>()>0)answer.active.insert(nodes[i]);}
    return answer;
  }
  void validate_weights(const RegionWeights& w,const RegionLayout& layout) const override {
    require(layout.spec.budget==c_.capacity(layout.members.size())&&layout.spec.observe_all,"weighted region requires declared k/K and observe-all");
    if(c_.region_control_mode=="context"){
      require(w.extra.at("head.matrix").sizes()==at::IntArrayRef({c_.hidden_width,*c_.region_control_width+2})
        &&w.extra.at("head.hidden_bias").sizes()==at::IntArrayRef({c_.hidden_width})
        &&w.extra.at("head.vector").sizes()==at::IntArrayRef({c_.hidden_width})&&w.extra.at("head.bias").dim()==0,
        "invalid weighted region head shape");
    }
  }
  void validate_history(const History& h,const RegionLayout&) const override {
    require(h.scalars.empty()&&h.node_maps.empty()&&h.tensors.empty(),"weighted history is a singleton");
  }
 private:RegionalConfig c_;
};
}
void configure_region(RegionWeights& w,const RegionalConfig& r){w.kernel=std::make_shared<Sending>(r);}
}
