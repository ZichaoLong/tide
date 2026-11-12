// KV §3–4: block packing with explicit one-pass, support-aware first transpose.
#include "weighted_column_query_ops.h"
#include "weighted_row_linear.h"
#include "weighted_kv_block.h"
#include "tide/packing_work.h"
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <cmath>
#include <limits>
namespace tide::weighted::columns {
namespace {
using torch::autograd::variable_list;
class Contractions final:public torch::autograd::Node {
 public:
  Contraction kind;
  Index arity=0,batch=0,count=0,total=0;
  std::vector<torch::autograd::SavedVariable> saved;
  std::vector<Index> output_map;
  variable_list apply(variable_list&& bars)override{
    if(at::GradMode::is_enabled())throw std::invalid_argument("node column contractions support first-order VJP only");
    if(saved.empty())throw std::runtime_error("node column contraction graph released");
    variable_list values;for(auto& s:saved)values.push_back(s.unpack());
    variable_list result(count),gs;std::vector<Index> nodes,ids;
    for(size_t i=0;i<bars.size();++i)if(bars[i].defined()){
      const auto n=output_map[i];nodes.push_back(n);gs.push_back(bars[i]);
      for(Index b=0;b<batch;++b)ids.push_back(n*batch+b);
    }
    if(nodes.empty())return result;
    auto index=at::tensor(ids,values[count].options().dtype(at::kLong));
    auto take=[&](const Tensor& x){return static_cast<Index>(nodes.size())==total?x:x.index_select(0,index);};
    variable_list x;for(Index j=0;j<arity;++j)x.push_back(take(values[count+j]));
    auto g=at::cat(gs);variable_list grads;
    if(kind==Contraction::matmul)grads={at::bmm(g,x[1].transpose(1,2)),at::bmm(x[0].transpose(1,2),g)};
    else {
      if(x[arity-2].scalar_type()==at::kHalf){for(auto& t:x)t=t.to(at::kFloat);g=g.to(at::kFloat);}
      auto v=x[arity-2],w=x[arity-1],p=take(values.back());
      auto used=(w>0).any(1),anchor_index=used.to(at::kLong).argmax(-1).reshape({-1,1,1,1});
      auto anchor=v.gather(2,anchor_index.expand({-1,v.size(1),1,v.size(3)}));
      auto delta=at::where(used.unsqueeze(1).unsqueeze(-1),v,anchor)-anchor;
      auto dl=p*(at::matmul(g,delta.transpose(-1,-2))-(g*at::matmul(p,delta)).sum(-1,true));
      dl=at::where((w>0).sum(-1).unsqueeze(1).unsqueeze(-1)>1,dl,at::zeros_like(dl));
      auto dv=at::matmul(p.transpose(-1,-2),g),dw=dl.sum(1)/at::where(w>0,w,at::ones_like(w));
      if(kind==Contraction::logits)grads={dl,dv,dw};
      else {auto scale=std::sqrt(double(x[0].size(-1)));
        grads={at::matmul(dl,x[1])/scale,at::matmul(dl.transpose(-1,-2),x[0])/scale,dv,dw};}
    }
    for(Index j=0;j<arity;++j){auto pieces=grads[j].split(batch);
      for(size_t i=0;i<nodes.size();++i){const auto slot=nodes[i]*arity+j;
        if(should_compute_output(slot))result[slot]=pieces[i].to(values[slot].scalar_type());}}
    return result;
  }
  void release_variables()override{saved.clear();}
  std::string name()const override{return "tide::weighted::NodeColumnContractionsVjp";}
};
}
std::vector<Tensor> project_nodes(const std::vector<Tensor>& values,const std::vector<Tensor>& weights){
  detail::require(values.size()==weights.size(),"node projection count mismatch");
  if(values.empty())return {};
  if(values.size()==1)return {at::matmul(values[0],weights[0].t())};
  std::vector<Tensor> xs;for(const auto& x:values)xs.push_back(x.reshape({-1,x.size(-1)}));
  auto ys=linear_blocks(xs,weights);
  for(size_t i=0;i<ys.size();++i){auto shape=values[i].sizes().vec();shape.back()=weights[i].size(0);ys[i]=ys[i].reshape(shape);}
  return ys;
}
std::vector<Tensor> contract_nodes(Contraction kind,const std::vector<std::vector<Tensor>>& rows){
  if(rows.empty())return {};
  const Index arity=kind==Contraction::matmul?2:kind==Contraction::attention?4:3;
  detail::require(static_cast<Index>(rows[0].size())==arity&&rows[0][0].size(0)>0,"invalid node contraction operands");
  variable_list inputs;bool grad=false;
  for(const auto& row:rows){detail::require(static_cast<Index>(row.size())==arity,"node contraction operand count");
    for(Index j=0;j<arity;++j){const auto& x=row[j];const auto& y=rows[0][j];
      detail::require(x.sizes()==y.sizes()&&x.scalar_type()==y.scalar_type()&&x.device()==y.device(),"node contraction metadata mismatch");
      inputs.push_back(x);grad|=x.requires_grad();}}
  if(rows.size()==1){const auto& x=rows[0];
    if(kind==Contraction::matmul)return {at::bmm(x[0],x[1])};
    if(kind==Contraction::attention)return {block::attention(x[0],x[1],x[2],x[3])};
    return {logit_attention(x[0],x[1],x[2])};}
  grad=grad&&at::GradMode::is_enabled();const auto batch=rows[0][0].size(0);
  at::NoGradGuard guard;variable_list packed;
  for(Index j=0;j<arity;++j){variable_list column;for(const auto& row:rows)column.push_back(row[j]);packed.push_back(at::cat(column));}
  Tensor value,p;
  if(kind==Contraction::matmul)value=at::bmm(packed[0],packed[1]);
  else {
    auto x=packed;
    if(x[arity-2].scalar_type()==at::kHalf)for(auto& t:x)t=t.to(at::kFloat);
    auto logits=kind==Contraction::logits?x[0]:at::matmul(x[0],x[1].transpose(-1,-2))/std::sqrt(double(x[0].size(-1)));
    auto v=x[arity-2],w=x[arity-1],positive=w>0,active=positive.any(-1).unsqueeze(1).unsqueeze(-1);
    logits=(logits+at::where(positive,w,at::ones_like(w)).log().unsqueeze(1)).masked_fill(~positive.unsqueeze(1),-std::numeric_limits<double>::infinity());
    p=at::softmax(at::where(active,logits,at::zeros_like(logits)),-1);p=at::where(active,p,at::zeros_like(p));
    value=at::matmul(p,v).to(packed[arity-2].scalar_type());
  }
  auto outputs=value.split(batch);
  work::packing::bucket(kind==Contraction::matmul?"column_matmul_nodes":kind==Contraction::attention?"column_attention_nodes":"column_logits_nodes",rows[0][0].size(-2),rows.size());
  if(grad){auto node=std::make_shared<Contractions>();node->kind=kind;node->arity=arity;node->batch=batch;node->total=rows.size();node->count=inputs.size();
    node->set_next_edges(torch::autograd::collect_next_edges(inputs));
    for(const auto& t:inputs)node->saved.emplace_back(t,false);
    for(const auto& t:packed)node->saved.emplace_back(t,false);
    if(p.defined())node->saved.emplace_back(p,false);
    for(size_t i=0;i<outputs.size();++i){bool used=false;for(const auto& t:rows[i])used|=t.requires_grad();
      if(used){node->output_map.push_back(i);torch::autograd::set_history(outputs[i],node);}}
  }
  return outputs;
}
}
