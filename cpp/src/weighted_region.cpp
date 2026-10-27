// Allocation §2, §4: independent C++ complete region step, no scheduler policy.
#include "weighted_detail.h"
#include "weighted_scoring.h"
#include "tide/weighted_sending.h"
#include "weighted_region_batch.h"

namespace tide::weighted {
using namespace detail;
namespace {
class Sending final:public RegionKernel {
 public:
  explicit Sending(RegionalConfig c):c_(std::move(c)){c_.validate();}
  bool joint_batch() const override{return true;}
  bool batched_autograd() const override{return true;}
  bool history_independent() const override{return true;}
  std::vector<Selection> batch(const RegionWeights& w,const std::vector<RegionInput>& r) const override{return batch::regions(w,r,c_);}
  std::vector<Selection> batch_grad(const RegionWeights& w,const std::vector<RegionInput>& r) const override{return batch::region_vjp(w,r,c_);}
  Selection step(const RegionWeights& w,const RegionInput& r) const override {
    std::map<std::string,std::vector<Tensor>> columns;std::vector<Index> nodes;
    for(const auto& d:r.candidates){
      nodes.push_back(d.node);for(const auto& [name,value]:d.fields)columns[name].push_back(value);
    }
    std::map<std::string,Tensor> fields;for(const auto& [name,values]:columns)fields[name]=at::stack(values);
    auto recv=fields.at("w"),mu=fields.at("mu"),s=region_scores(w,c_,fields),sum=recv.sum(),mass=mu.sum();
    Tensor control,low,high,result,shift;
    if(c_.region_control_mode=="context"){
      auto z=(recv.unsqueeze(-1)*fields.at("evidence")).sum(0)/sum.clamp_min(1);
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
      shift=control.defined()?control:at::full({},*c_.eta,s.options());
      if(c_.hard())result=hard_selection(s-shift,mu,*c_.k,nodes);
      low=(c_.regional()?sum:mass).clamp_max(1);high=at::full({},*c_.k,s.options());
    }
    if(!result.defined())result=allocate(s,mu,low,high,at::full({},*c_.allocation_temperature,s.options()),*c_.regularizer,c_.ent_alpha,shift).controls;
    Selection answer;answer.history.last_time=r.time;
    for(size_t i=0;i<nodes.size();++i){answer.controls[nodes[i]]=result[i];if(result[i].item<double>()>0)answer.active.insert(nodes[i]);}
    return answer;
  }
  void validate_weights(const RegionWeights& w,const RegionLayout& layout) const override {
    require(layout.spec.budget==c_.capacity(layout.members.size())&&layout.spec.observe_all,"weighted region requires declared k/K and observe-all");
    validate_query(w,c_);
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
