// Architecture §2: batched numeric Aggregate with isolated first-order roots.
#include "weighted_aggregate.h"
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <sstream>
#include <stdexcept>

namespace tide::weighted {
namespace {
using torch::autograd::variable_list;
std::pair<bool,std::string> signature(const AggregateInput& r,const SourceClasses& classes,double alpha){
  if(r.sources.empty())throw std::invalid_argument("weighted Aggregate requires nonempty fibers");
  std::ostringstream s;variable_list masses;
  for(const auto& source:r.sources){
    const auto& t=source.atom.value;
    if(alpha)s << ':' << classes.classify(source.atom);
    s << '/' << t.sizes() << t.device() << int(t.scalar_type()) << t.requires_grad();
    masses.push_back(t[0].detach().clamp_min(0));
  }
  return {!(at::stack(masses).sum().item<double>()>0),s.str()};
}
class AggregateVjp final:public torch::autograd::Node {
 public:
  size_t weights=0,sources=0,rows=0;
  std::vector<torch::autograd::SavedVariable> originals,leaves,outputs;
  std::vector<size_t> output_map;
  variable_list apply(variable_list&& bars) override {
    if(at::GradMode::is_enabled())throw std::invalid_argument("weighted Aggregate supports first-order VJP only");
    if(originals.empty())throw std::runtime_error("Aggregate VJP graph already released; use retain_graph");
    for(auto& t:originals)t.unpack();
    variable_list xs,ys;
    for(auto& t:leaves)xs.push_back(t.unpack());
    for(auto& t:outputs)ys.push_back(t.unpack());
    variable_list result(xs.size()),inputs,roots,grads;
    std::vector<size_t> ids,used;
    for(size_t i=0;i<bars.size();++i)if(bars[i].defined()){
      used.push_back(output_map[i]);roots.push_back(ys[output_map[i]]);grads.push_back(bars[i]);
    }
    auto append=[&](size_t i){if(xs[i].requires_grad()&&should_compute_output(i)){
      ids.push_back(i);inputs.push_back(xs[i]);
    }};
    for(size_t i=0;i<weights;++i)append(i);
    for(auto row:used)for(size_t j=0;j<sources;++j)append(weights+row*sources+j);
    if(!inputs.empty()&&!roots.empty()){
      auto values=torch::autograd::grad(roots,inputs,grads,true,false,true);
      for(size_t i=0;i<ids.size();++i)result[ids[i]]=values[i];
    }
    return result;
  }
  void release_variables() override {originals.clear();leaves.clear();outputs.clear();}
  std::string name() const override {return "tide::weighted::BatchedAggregateVjp";}
};
std::vector<AggregateResult> bind_group(NodeWeights w,std::vector<AggregateInput> requests,
    const SourceClasses& classes,double alpha,double epsilon,bool zero){
  auto node=std::make_shared<AggregateVjp>();
  variable_list inputs,leaves;
  auto leaf=[&](Tensor& t){
    inputs.push_back(t);t=t.detach().set_requires_grad(t.requires_grad());leaves.push_back(t);
  };
  // Preserve logical parameter aliases; each physical input atom remains a use.
  std::map<const c10::TensorImpl*,Tensor> aliases;
  for(auto& [name,t]:w.extra)if(name.find("aggregate_program.transforms.")==0){
    auto id=t.unsafeGetTensorImpl();auto found=aliases.find(id);
    if(found!=aliases.end())t=found->second;
    else {leaf(t);aliases.emplace(id,t);}
  }
  node->weights=inputs.size();node->rows=requests.size();node->sources=requests[0].sources.size();
  for(auto& r:requests)for(auto& s:r.sources)leaf(s.atom.value);
  node->set_next_edges(torch::autograd::collect_next_edges(inputs));
  for(const auto& t:inputs)node->originals.emplace_back(t,false);
  for(const auto& t:leaves)node->leaves.emplace_back(t,false);
  std::vector<AggregateResult> refs;
  {
    at::AutoGradMode enable(true);
    if(zero){
      variable_list atoms;
      for(const auto& r:requests)for(const auto& s:r.sources)atoms.push_back(s.atom.value);
      auto payload=at::stack(atoms);
      auto total=payload.select(1,0).clamp_min(0).reshape({static_cast<Index>(requests.size()),-1}).sum(1);
      auto values=at::cat({total.unsqueeze(-1),at::zeros({static_cast<Index>(requests.size()),payload.size(1)-1},payload.options())},-1);
      for(Index i=0;i<values.size(0);++i)refs.push_back({values[i],{}});
    }else refs=aggregate_batch(w,requests,classes,alpha,epsilon);
  }
  std::vector<AggregateResult> result;
  for(size_t i=0;i<refs.size();++i){
    const auto& ref=refs[i].value;node->outputs.emplace_back(ref,false);
    auto value=ref.detach().clone();
    if(ref.requires_grad()){node->output_map.push_back(i);torch::autograd::set_history(value,node);}
    result.push_back({value,{}});
  }
  return result;
}
}
std::vector<AggregateResult> aggregate_batch_vjp(const NodeWeights& w,const std::vector<AggregateInput>& requests,
    const SourceClasses& classes,double alpha,double epsilon){
  std::map<std::pair<bool,std::string>,std::vector<size_t>> groups;
  for(size_t i=0;i<requests.size();++i)groups[signature(requests[i],classes,alpha)].push_back(i);
  std::vector<AggregateResult> result(requests.size());
  for(const auto& [key,ids]:groups){
    std::vector<AggregateInput> rows;
    for(auto i:ids)rows.push_back(requests[i]);
    auto values=bind_group(w,std::move(rows),classes,alpha,epsilon,key.first);
    for(size_t i=0;i<ids.size();++i)result[ids[i]]=std::move(values[i]);
  }
  return result;
}
}
