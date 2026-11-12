#include "tide/weighted_columns.h"
#include "weighted_allocation_batch.h"
#include "tide/allocation_observation.h"
#include "tide/operator_profile.h"
#include <ATen/record_function.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>

namespace tide::weighted {
struct ColumnVersionGuard {
  std::vector<torch::autograd::SavedVariable> saved;
  bool available=true;
  void save(const Tensor& t){if(t.defined()&&t.is_inference())available=false;else saved.emplace_back(t,false);}
  void check()const{
    if(!available)throw std::runtime_error("allocation column has no retained version counter");
    for(const auto& value:saved)value.unpack();
  }
};
namespace {
using torch::autograd::variable_list;
Tensor row_scalar(const Tensor& t,const Tensor& scores){
  detail::require(t.defined()&&t.device()==scores.device()&&t.scalar_type()==scores.scalar_type(),
                  "allocation column scalar metadata mismatch");
  if(t.dim()==0)return t.expand({scores.size(0)});
  detail::require(t.sizes()==scores.sizes().slice(0,1),"allocation column scalar must be scalar or R");
  return t;
}
class ColumnVjp final:public torch::autograd::Node {
 public:
  std::string kind;std::optional<double> alpha;
  std::vector<torch::autograd::SavedVariable> saved;
  variable_list apply(variable_list&& bars)override{
    if(at::GradMode::is_enabled())throw std::invalid_argument("allocation columns support first-order exact-map VJP only");
    if(saved.empty())throw std::runtime_error("allocation column graph released; use retain_graph");
    variable_list x;for(auto& v:saved)x.push_back(v.unpack());
    // A multi-output regional publication may leave this entire group unused.
    // Check originals above, then preserve absent cotangents as absent roots.
    if(!bars[0].defined())return variable_list(6);
    AllocationColumns a{x[6],x[7],x[8],x[9],x[10],x[11],{},kind,kind+"-exact-map-v1",alpha};
    std::copy_n(x.begin(),6,a.inputs.begin());
    auto result=allocation_columns_vjp(a,bars[0]);
    for(size_t i=0;i<6;++i)if(!should_compute_output(i))result[i]=Tensor{};
    return variable_list(result.begin(),result.end());
  }
  void release_variables()override{saved.clear();}
  std::string name()const override{return "tide::weighted::AllocationColumnsVjp";}
};
}
std::array<Tensor,6> allocation_columns_vjp(const AllocationColumns& a,const Tensor& bar,const Tensor& support){
  if(at::GradMode::is_enabled())throw std::invalid_argument("allocation columns support first-order exact-map VJP only");
  if(a.versions)a.versions->check();
  const auto& x=a.inputs;auto scores=x[0],caps=x[1],temperature=x[4],branch=a.branch;
  detail::require(bar.sizes()==scores.sizes()&&bar.device()==scores.device()&&bar.scalar_type()==scores.scalar_type(),
                  "allocation column adjoint metadata mismatch");
  auto rows=support.defined()?support:at::ones_like(branch,at::kBool);
  detail::require(rows.sizes()==branch.sizes()&&rows.device()==branch.device()&&rows.scalar_type()==at::kBool,
                  "allocation column adjoint support must be bool R on same device");
  std::array<Tensor,6> out;
  auto capped=rows&(branch==batch::Caps),normal=rows&(branch!=batch::Zero)&(branch!=batch::Caps);
  if(capped.any().item<bool>())out[1]=at::where(capped.unsqueeze(1),bar,0);
  auto ids=at::nonzero(normal).flatten();
  if(ids.numel()){
    auto c=a.controls.index({ids}),tau=temperature.index({ids}).unsqueeze(1),grad=bar.index({ids});
    auto b=branch.index({ids});detail::Regularizer rule(a.regularizer,a.alpha);
    auto slopes=rule.slopes(a.response.index({ids}),caps.index({ids}),tau,c);
    auto aa=slopes[0],e=slopes[1],dtau=slopes[2],total=aa.sum(1);
    auto constrained=(b!=batch::Slack)&(total>0);
    auto anchor=grad.gather(1,aa.argmax(1).unsqueeze(1)).squeeze(1),relative=grad-anchor.unsqueeze(1);
    auto offset=(aa*relative).sum(1)/at::where(total>0,total,1);
    auto mean=at::where(constrained,anchor+offset,0);
    auto centered=at::where(constrained.unsqueeze(1),relative-offset.unsqueeze(1),grad);
    auto ds=aa*centered,dc=e*centered,dt=(dtau*centered).sum(1);
    out[0]=at::zeros_like(scores).index_copy_(0,ids,ds);
    auto capbar=at::zeros_like(caps).index_copy_(0,ids,dc);out[1]=out[1].defined()?out[1]+capbar:capbar;
    auto lower=(b==batch::Lower)|(b==batch::Equality);
    for(size_t target:{size_t(2),size_t(3)}){
      auto mask=target==2?lower:~lower;
      if(mask.any().item<bool>())out[target]=at::zeros_like(x[target]).index_copy_(0,ids.index({mask}),mean.index({mask}));
    }
    out[4]=at::zeros_like(temperature).index_copy_(0,ids,dt);
    if(x[5].defined())out[5]=at::zeros_like(x[5]).index_copy_(0,ids,at::where(b!=batch::Slack,0,-ds.sum(1)));
  }
  return out;
}
AllocationColumns allocate_columns(const Tensor& scores,const Tensor& caps,const Tensor& lower,
    const Tensor& upper,const Tensor& temperature,const std::string& kind,std::optional<double> alpha,const Tensor& shift){
  RECORD_FUNCTION("tide::stage::allocation", std::vector<c10::IValue>());
  op_profile::Scope stage_profile(op_profile::Allocation);
  detail::require(scores.dim()==2&&scores.size(0)>0&&scores.size(1)>0,"allocation columns require nonempty R,N");
  std::array<Tensor,6> inputs{scores,caps,row_scalar(lower,scores),row_scalar(upper,scores),row_scalar(temperature,scores),
                            shift.defined()?row_scalar(shift,scores):Tensor{}};
  auto n=batch::solve(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],detail::Regularizer(kind,alpha),inputs[5]);
  AllocationColumns out{n.controls,n.multiplier,n.branch,n.response,n.free,n.residual,inputs,kind,kind+"-exact-map-v1",alpha};
  out.versions=std::make_shared<ColumnVersionGuard>();
  for(const auto& t:inputs)out.versions->save(t);
  for(const auto& t:{n.controls,n.response,n.branch})out.versions->save(t);
  if(allocation_observations.enabled){
    auto host=n.branch.to(at::kCPU).contiguous();auto branches=host.const_data_ptr<Index>();
    at::NoGradGuard guard;
    for(Index i=0;i<scores.size(0);++i){
      AllocationRow row{inputs[0][i],inputs[1][i],inputs[2][i],inputs[3][i],inputs[4][i],inputs[5].defined()?inputs[5][i]:Tensor{}};
      Allocation value{n.controls[i],n.multiplier[i],n.free[i],n.residual[i],batch::branches[branches[i]],out.backward};
      observe_allocation(row,value,kind,alpha,"native-columns");
    }
  }
  if(at::GradMode::is_enabled()){
    auto host=n.branch.to(at::kCPU).contiguous();auto data=host.const_data_ptr<Index>();bool grad=false;
    for(Index i=0;i<host.numel();++i){
      const auto b=data[i];
      if(b==batch::Caps)grad|=inputs[1].requires_grad();
      else if(b!=batch::Zero){
        for(size_t k:{size_t(0),size_t(1),(b==batch::Lower||b==batch::Equality)?size_t(2):size_t(3),size_t(4)})
          grad|=inputs[k].requires_grad();
        grad|=inputs[5].defined()&&inputs[5].requires_grad();
      }
    }
    if(grad){
      auto node=std::make_shared<ColumnVjp>();node->kind=kind;node->alpha=alpha;
      node->set_next_edges(torch::autograd::collect_next_edges(variable_list(inputs.begin(),inputs.end())));
      for(const auto& t:inputs)node->saved.emplace_back(t,false);
      for(const auto& t:{n.controls,n.multiplier,n.branch,n.response,n.free,n.residual})node->saved.emplace_back(t,false);
      out.controls=n.controls.detach();torch::autograd::set_history(out.controls,node);
    }
  }
  return out;
}
}
