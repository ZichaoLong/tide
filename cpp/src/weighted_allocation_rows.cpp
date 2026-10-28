// Independent public row roots over the actual KKT solution, not iterations.
#include "weighted_allocation_batch.h"
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <sstream>

namespace tide::weighted {
namespace {
using torch::autograd::variable_list;
class AllocationVjp final:public torch::autograd::Node {
 public:
  explicit AllocationVjp(detail::Regularizer r):rule(std::move(r)){}
  detail::Regularizer rule;
  size_t width=0;
  std::vector<torch::autograd::SavedVariable> inputs,numeric;
  std::vector<size_t> output_map;
  variable_list apply(variable_list&& bars) override {
    if(at::GradMode::is_enabled())throw std::invalid_argument("batch allocation supports first-order exact-map VJP only");
    if(inputs.empty())throw std::runtime_error("allocation VJP graph already released; use retain_graph");
    variable_list xs,ns;
    for(auto& v:inputs)xs.push_back(v.unpack());for(auto& v:numeric)ns.push_back(v.unpack());
    auto c=ns[0],multiplier=ns[1],branch=ns[2],response=ns[3];
    variable_list result(xs.size()),full(xs.size()/width);std::vector<Index> normal;
    for(size_t i=0;i<bars.size();++i)full[output_map[i]]=bars[i];
    for(size_t i=0;i<full.size();++i)if(full[i].defined()){
      const auto b=branch[i].item<Index>();
      if(b==batch::Caps)result[i*width+1]=full[i];else if(b!=batch::Zero)normal.push_back(i);
    }
    if(!normal.empty()){
      auto ids=at::tensor(normal,branch.options());variable_list fields,gs;
      for(size_t j=0;j<width;++j){variable_list col;for(auto i:normal)col.push_back(xs[i*width+j]);fields.push_back(at::stack(col));}
      for(auto i:normal)gs.push_back(full[i]);auto grad=at::stack(gs),tau=fields[4].unsqueeze(1);
      Tensor x=response.index({ids});
      auto slopes=rule.slopes(x,fields[1],tau,c.index({ids}));auto a=slopes[0],e=slopes[1],dtau=slopes[2],total=a.sum(1);
      auto constrained=(branch.index({ids})!=batch::Slack)&(total>0);
      auto anchor=grad.gather(1,a.argmax(1).unsqueeze(1)).squeeze(1),relative=grad-anchor.unsqueeze(1);
      auto offset=(a*relative).sum(1)/at::where(total>0,total,at::ones_like(total));
      auto mean=at::where(constrained,anchor+offset,at::zeros_like(total));
      auto centered=at::where(constrained.unsqueeze(1),relative-offset.unsqueeze(1),grad);
      auto ds=a*centered,dc=e*centered,dt=(dtau*centered).sum(1);
      auto shift=at::where(branch.index({ids})!=batch::Slack,at::zeros_like(total),-ds.sum(1));
      for(size_t j=0;j<normal.size();++j){
        auto i=normal[j];auto base=i*width;auto b=branch[i].item<Index>();
        result[base]=ds[j];result[base+1]=dc[j];result[base+((b==batch::Lower||b==batch::Equality)?2:3)]=mean[j];
        result[base+4]=dt[j];if(width==6)result[base+5]=shift[j];
      }
    }
    for(size_t i=0;i<result.size();++i)if(!should_compute_output(i))result[i]=Tensor{};
    return result;
  }
  void release_variables() override{inputs.clear();numeric.clear();}
  std::string name() const override{return "tide::weighted::AllocationBatchVjp";}
};
}
std::vector<Allocation> allocate_batch(const std::vector<AllocationRow>& rows,const std::string& kind,std::optional<double> alpha){
  detail::Regularizer rule(kind,alpha);std::vector<Allocation> result(rows.size());
  std::map<std::string,std::vector<size_t>> groups;
  for(size_t i=0;i<rows.size();++i){
    const auto& r=rows[i];std::ostringstream key;
    key<<r.scores.sizes()<<r.scores.device()<<int(r.scores.scalar_type())<<r.shift.defined();groups[key.str()].push_back(i);
  }
  for(const auto& [key,ids]:groups){
    const size_t width=rows[ids[0]].shift.defined()?6:5;variable_list inputs,fields;
    for(auto i:ids){const auto& r=rows[i];for(auto t:{r.scores,r.caps,r.lower,r.upper,r.temperature})inputs.push_back(t);if(width==6)inputs.push_back(r.shift);}
    for(size_t j=0;j<width;++j){variable_list column;for(size_t i=0;i<ids.size();++i)column.push_back(inputs[i*width+j]);fields.push_back(at::stack(column));}
    auto numeric=batch::solve(fields[0],fields[1],fields[2],fields[3],fields[4],rule,width==6?fields[5]:Tensor{});
    auto node=std::make_shared<AllocationVjp>(rule);node->width=width;
    node->set_next_edges(torch::autograd::collect_next_edges(inputs));
    for(auto& t:inputs)node->inputs.emplace_back(t,false);
    for(auto& t:{numeric.controls,numeric.multiplier,numeric.branch,numeric.response})node->numeric.emplace_back(t,false);
    for(size_t j=0;j<ids.size();++j){
      const auto b=numeric.branch[j].item<Index>();auto controls=numeric.controls[j].clone();bool grad=false;
      if(at::GradMode::is_enabled()){
        if(b==batch::Caps)grad=inputs[j*width+1].requires_grad();
        else if(b!=batch::Zero){
          for(auto k:{size_t(0),size_t(1),(b==batch::Lower||b==batch::Equality)?size_t(2):size_t(3),size_t(4)})grad|=inputs[j*width+k].requires_grad();
          if(width==6)grad|=inputs[j*width+5].requires_grad();
        }
      }
      if(grad){node->output_map.push_back(j);torch::autograd::set_history(controls,node);}
      result[ids[j]]={controls,numeric.multiplier[j],numeric.free[j],numeric.residual[j],batch::branches[b],kind+"-exact-map-v1"};
    }
  }
  return result;
}
}
