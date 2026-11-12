// Architecture §2.2: batched source MLP and a support-aware explicit transpose.
#include "weighted_column_sources.h"
#include "tide/packing_work.h"
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
namespace tide::weighted::columns {
namespace {
using torch::autograd::variable_list;
class Uses final:public torch::autograd::Node {
 public:
  torch::autograd::SavedVariable original;
  variable_list apply(variable_list&& bars)override{
    if(at::GradMode::is_enabled())throw std::invalid_argument("shared columns support first-order VJP only");
    original.unpack();Tensor total;
    for(const auto& bar:bars)if(bar.defined())total=total.defined()?total+bar:bar;
    return {total};
  }
  void release_variables()override{original.reset_data();}
  std::string name()const override{return "tide::weighted::OrderedColumnUsesVjp";}
};
class SourceVjp final:public torch::autograd::Node {
 public:
  std::vector<torch::autograd::SavedVariable> saved;
  std::vector<size_t> output_map;
  size_t count=0,arity=0,total=0;
  double alpha=0;
  bool shared_a=false,shared_b=false;
  variable_list apply(variable_list&& bars)override{
    if(at::GradMode::is_enabled())throw std::invalid_argument("source columns support first-order VJP only");
    if(saved.empty())throw std::runtime_error("source column graph released");
    variable_list values;for(auto& s:saved)values.push_back(s.unpack());
    variable_list result(count),gs;std::vector<Index> ids;
    for(size_t i=0;i<bars.size();++i)if(bars[i].defined()){ids.push_back(output_map[i]);gs.push_back(bars[i]);}
    if(ids.empty())return result;
    Tensor index;if(ids.size()!=total)index=at::tensor(ids,values[count].options().dtype(at::kLong));
    auto take=[&](const Tensor& x){return index.defined()?x.index_select(0,index):x;};
    variable_list x;for(size_t j=count;j<values.size();++j)x.push_back(take(values[j]));
    auto a=x[0],b=x[1],u=x[2],normalized=x[3],denominator=x[4],hidden=x[5],activated=x[6];
    auto g=at::stack(gs),gv=g*alpha;
    auto gh=shared_b?at::matmul(gv,b[0]):at::bmm(gv,b);
    auto ht=at::silu_backward(gh,hidden),db=at::bmm(gv.transpose(1,2),activated),da=at::bmm(ht.transpose(1,2),u);
    auto gu=shared_a?at::matmul(ht,a[0]):at::bmm(ht,a);
    if(g.scalar_type()==at::kHalf)gu=gu.to(at::kFloat);
    auto dz=g+((gu-normalized*(gu*normalized).mean(-1,true))/denominator).to(g.scalar_type());
    variable_list gradients{dz,da,db,ht.sum(1)};
    if(arity==8){auto al=x[7],ar=x[8],bl=x[9],br=x[10];
      gradients.insert(gradients.end(),{at::bmm(da,ar.transpose(1,2)),at::bmm(al.transpose(1,2),da),
        at::bmm(db,br.transpose(1,2)),at::bmm(bl.transpose(1,2),db)});}
    for(size_t j=0;j<arity;++j)for(size_t k=0;k<ids.size();++k){auto slot=ids[k]*arity+j;
      if(should_compute_output(slot))result[slot]=gradients[j][k];}
    return result;
  }
  void release_variables()override{saved.clear();}
  std::string name()const override{return "tide::weighted::SourceColumnsVjp";}
};
}
std::vector<Tensor> ordered_uses(const Tensor& value,size_t count){
  if(count<2||!at::GradMode::is_enabled()||!value.requires_grad())return std::vector<Tensor>(count,value);
  auto node=std::make_shared<Uses>();node->original=torch::autograd::SavedVariable(value,false);
  node->set_next_edges(torch::autograd::collect_next_edges(value));at::NoGradGuard guard;
  std::vector<Tensor> result;for(size_t i=0;i<count;++i){auto out=value.view_as(value);torch::autograd::set_history(out,node);result.push_back(out);}return result;
}
std::vector<Tensor> source_columns_group(const std::vector<SourceRequest>& requests){
  const auto arity=requests[0].parameters.size()+1;variable_list inputs;bool grad=false;
  for(const auto& r:requests){inputs.push_back(r.content);inputs.insert(inputs.end(),r.parameters.begin(),r.parameters.end());}
  if(at::GradMode::is_enabled())for(const auto& t:inputs)grad|=t.requires_grad();
  at::NoGradGuard guard;variable_list packed;
  for(size_t j=0;j<arity;++j){variable_list column;for(size_t i=0;i<requests.size();++i)column.push_back(inputs[i*arity+j]);packed.push_back(at::stack(column));}
  auto z=packed[0],a=packed[1],b=packed[2],e=packed[3];
  if(arity==8){a=a+at::bmm(packed[4],packed[5]);b=b+at::bmm(packed[6],packed[7]);}
  bool shared_a=true,shared_b=true;
  for(size_t i=1;i<requests.size();++i){
    auto same=[&](size_t j){return inputs[i*arity+j].unsafeGetTensorImpl()==inputs[j].unsafeGetTensorImpl();};
    shared_a&=same(1)&&(arity==4||(same(4)&&same(5)));shared_b&=same(2)&&(arity==4||(same(6)&&same(7)));}
  auto operand=z.scalar_type()==at::kHalf?z.to(at::kFloat):z;
  auto denominator=at::sqrt(requests[0].epsilon+operand.square().mean(-1,true)),normalized=operand/denominator;
  auto u=normalized.to(z.scalar_type());
  auto hidden=(shared_a?at::matmul(u,a[0].t()):at::bmm(u,a.transpose(1,2)))+e.unsqueeze(1);
  auto activated=at::silu(hidden);
  auto values=z+requests[0].alpha*(shared_b?at::matmul(activated,b[0].t()):at::bmm(activated,b.transpose(1,2)));
  work::packing::bucket("source_columns",z.size(1),requests.size());
  auto outputs=values.unbind();
  if(grad){auto node=std::make_shared<SourceVjp>();node->count=inputs.size();node->arity=arity;node->total=requests.size();
    node->alpha=requests[0].alpha;node->shared_a=shared_a;node->shared_b=shared_b;
    node->set_next_edges(torch::autograd::collect_next_edges(inputs));
    for(const auto& t:inputs)node->saved.emplace_back(t,false);
    for(const auto& t:{a,b,u,normalized,denominator,hidden,activated})node->saved.emplace_back(t,false);
    for(size_t j=4;j<arity;++j)node->saved.emplace_back(packed[j],false);
    for(size_t i=0;i<outputs.size();++i){bool used=false;for(size_t j=0;j<arity;++j)used|=inputs[i*arity+j].requires_grad();
      if(used){node->output_map.push_back(i);torch::autograd::set_history(outputs[i],node);}}
  }
  return outputs;
}
}
