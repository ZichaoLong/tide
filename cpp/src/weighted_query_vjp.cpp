#include "weighted_private_rows.h"
#include "tide/operator_profile.h"
#include <ATen/record_function.h>
// Architecture §3/4: actual query batches with independent row/field VJP roots.
#include "weighted_query_vjp.h"
#include "weighted_query_layout.h"
#include "weighted_query_roots.h"
#include "weighted_singleton_vjp.h"
#include "region_metadata.h"
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <optional>
#include <sstream>
#include <type_traits>

namespace tide::weighted::query {
namespace {
using torch::autograd::variable_list;
struct Row {std::optional<State> state;Index time;Tensor content,control;};
template<class F> void tensors(Row& row,F transform){
  if(row.state){transform(row.state->value);for(auto& [name,t]:row.state->slots)transform(t);}
  transform(row.content);if(row.control.defined())transform(row.control);
}
template<class Config> std::string signature(const NodeWeights& w,Row row,const Config& c,bool is_full,const std::string& branch){
  std::ostringstream s;s << (row.state?"state":"initial");
  if(row.state)for(const auto& [name,t]:row.state->slots)s << ':' << name;
  if constexpr(std::is_same_v<Config,KVConfig>){
    if(branch.empty()&&(is_full||c.memory_read=="qry")){
      auto state=row.state?*row.state:w.kernel->initial(w);
      std::pair<Tensor,Tensor> visibility=detail::visible(state,row.time,c);
      const auto& [mask,age]=visibility;
      Tensor writes=state.slots.at("weights").index({mask}).detach();
      s << ":long=" << (writes>0).any().item<bool>();
      if(!is_full){Tensor short_weights=writes*detail::passive_decay(age.index({mask}),c.beta_short,writes);
        s << ":short=" << (short_weights>0).any().item<bool>();}
    }
  }
  s << branch;
  auto meta=[&](const Tensor& t,bool ragged){
    s << '/';auto shape=t.sizes().vec();if(ragged&&!shape.empty())shape[0]=-1;
    s << at::IntArrayRef(shape) << t.device() << int(t.scalar_type()) << t.requires_grad();
  };
  if(row.state){meta(row.state->value,false);for(const auto& [name,t]:row.state->slots)meta(t,!branch.empty());}
  meta(row.content,false);if(row.control.defined())meta(row.control,false);
  return s.str();
}
class QueryVjp final:public torch::autograd::Node {
 public:
  size_t weights=0,row_inputs=0,row_outputs=0,rows=0;
  private_rows::Packing packing;
  std::vector<torch::autograd::SavedVariable> originals,leaves,outputs;
  std::vector<size_t> output_map;
  variable_list apply(variable_list&& bars) override {
  RECORD_FUNCTION("tide::stage::query_backward", std::vector<c10::IValue>());
  op_profile::Scope stage_profile(op_profile::QueryAdjoint);
    if(at::GradMode::is_enabled())throw std::invalid_argument("weighted queries support first-order VJP only");
    if(originals.empty())throw std::runtime_error("query VJP graph already released; use retain_graph");
    for(auto& t:originals)t.unpack();
    variable_list xs,ys,full(rows*row_outputs);
    for(auto& t:leaves)xs.push_back(t.unpack());
    for(auto& t:outputs)ys.push_back(t.unpack());
    for(size_t i=0;i<bars.size();++i)full[output_map[i]]=bars[i];
    std::map<std::vector<size_t>,std::vector<size_t>> groups;
    for(size_t row=0;row<rows;++row){
      std::vector<size_t> mask;
      for(size_t j=0;j<row_outputs;++j)if(full[row*row_outputs+j].defined())mask.push_back(j);
      if(!mask.empty())groups[mask].push_back(row);
    }
    variable_list result(weights+rows*row_inputs);
    for(const auto& [mask,used]:groups){
      variable_list roots,grads,inputs;std::vector<size_t> ids;
      for(auto row:used)for(auto j:mask){roots.push_back(ys[row*row_outputs+j]);grads.push_back(full[row*row_outputs+j]);}
      auto append=[&](size_t i){if(xs[i].requires_grad()){ids.push_back(i);inputs.push_back(xs[i]);}};
      for(size_t i=0;i<weights;++i)append(i);
      for(size_t f=0;f<row_inputs;++f)append(weights+f);
      if(inputs.empty())continue;
      coalesce_roots(roots,grads);
      variable_list values;{
        RECORD_FUNCTION("tide::stage::query_gradient", std::vector<c10::IValue>());
        op_profile::Scope profile(op_profile::QueryGradient);
        values=torch::autograd::grad(roots,inputs,grads,true,false,true);
      }
      for(size_t i=0;i<ids.size();++i)packing.scatter(ids[i],used,values[i],result);
    }
    for(size_t i=0;i<result.size();++i)if(!should_compute_output(i))result[i]=Tensor{};
    return result;
  }
  void release_variables() override {originals.clear();leaves.clear();outputs.clear();}
  std::string name() const override {return "tide::weighted::BatchedQueryVjp";}
};
using Fields=std::vector<TensorFields>;
template<class Config> Fields bind_group(NodeWeights w,std::vector<Row> rows,const Config& c,const RegionalConfig& region,
    bool is_full,Index slots,const Options& options,int dense){
  RECORD_FUNCTION("tide::stage::query_bind", std::vector<c10::IValue>());
  op_profile::Scope stage_profile(op_profile::QueryBind);
  if(rows.size()==1){
    variable_list originals{w.bias};
    const std::string prefix=is_full?"full_program.":"read_program.";
    for(const auto& [name,t]:w.extra)if(name.find("kernel.")==0||name.find(prefix)==0)originals.push_back(t);
    auto& row=rows[0];tensors(row,[&](Tensor& t){originals.push_back(t);});
    Fields values;
    if(is_full){
      for(const auto& v:full(w,{{&*row.state,row.time,ContentView{row.content},row.control}},slots,options,c,dense))
        values.push_back({{"value",v.value}});
    }else values=describe(w,{{row.state?&*row.state:nullptr,row.time,ContentView{row.content}}},c,region,dense);
    variable_list refs;for(const auto& [name,t]:values[0])refs.push_back(t);
    auto outputs=singleton_boundary(originals,refs);size_t i=0;
    for(auto& [name,t]:values[0])t=outputs[i++];
    return values;
  }
  auto node=std::make_shared<QueryVjp>();variable_list inputs,leaves;
  auto leaf=[&](Tensor& t){inputs.push_back(t);t=t.detach().set_requires_grad(t.requires_grad());leaves.push_back(t);};
  std::map<const c10::TensorImpl*,Tensor> aliases;
  auto parameter=[&](Tensor& t){
    auto id=t.unsafeGetTensorImpl();auto found=aliases.find(id);
    if(found!=aliases.end())t=found->second;else{leaf(t);aliases.emplace(id,t);}
  };
  parameter(w.bias);
  const std::string prefix=is_full?"full_program.":"read_program.";
  for(auto& [name,t]:w.extra)if(name.find("kernel.")==0||name.find(prefix)==0)parameter(t);
  node->weights=inputs.size();node->rows=rows.size();
  for(auto& row:rows)tensors(row,[&](Tensor& t){inputs.push_back(t);});
  node->row_inputs=(inputs.size()-node->weights)/rows.size();
  node->set_next_edges(torch::autograd::collect_next_edges(inputs));
  for(const auto& t:inputs)node->originals.emplace_back(t,false);
  leaves=node->packing.pack(inputs,leaves,rows.size());
  for(const auto& t:leaves)node->leaves.emplace_back(t,false);
  Fields refs;
  {
    at::AutoGradMode enable(true);
    auto views=node->packing.views(leaves);size_t k=0;
    for(auto& row:rows)tensors(row,[&](Tensor& t){t=views[k++];});
    if(is_full){
      std::vector<FullInput> requests;
      for(const auto& row:rows)requests.push_back({&*row.state,row.time,ContentView{row.content},row.control});
      for(const auto& value:full(w,requests,slots,options,c,dense))refs.push_back({{"value",value.value}});
    }else{
      std::vector<ReadInput> requests;
      for(const auto& row:rows)requests.push_back({row.state?&*row.state:nullptr,row.time,ContentView{row.content}});
      refs=describe(w,requests,c,region,dense);
    }
  }
  stage_profile.phase(op_profile::QueryPublish);
  node->row_outputs=refs[0].size();Fields result;
  for(size_t i=0;i<rows.size();++i){
    TensorFields out;size_t j=0;
    for(const auto& [name,ref]:refs[i]){
      node->outputs.emplace_back(ref,false);auto value=ref.detach().clone();
      if(ref.requires_grad()){node->output_map.push_back(i*node->row_outputs+j);torch::autograd::set_history(value,node);}
      out[name]=value;++j;
    }
    result.push_back(std::move(out));
  }
  return result;
}
template<class Config> Fields bind_queries(const NodeWeights& w,const std::vector<Row>& rows,const Config& c,const RegionalConfig& r,
    bool is_full,Index slots,const Options& options,int dense){
  std::map<std::string,std::vector<size_t>> groups;Fields result(rows.size());
  std::vector<std::string> branches(rows.size());
  if constexpr(std::is_same_v<Config,KVConfig>){
    if(dense==2){
      States states;Times times;
      for(const auto& row:rows){states.push_back(row.state?*row.state:w.kernel->initial(w));times.push_back(row.time);}
      branches=packed_branches(states,times,c,is_full);
    }else if(dense){
      at::NoGradGuard guard;std::vector<Tensor> flags;
      for(const auto& row:rows){
        auto state=row.state?*row.state:w.kernel->initial(w);auto [mask,age]=detail::visible(state,row.time,c);
        auto writes=state.slots.at("weights");flags.push_back(((writes>0)&mask).any());
        if(!is_full)flags.push_back(((writes*detail::passive_decay(age.clamp_min(0),c.beta_short,writes)>0)&mask).any());
      }
      auto values=region_metadata::flags(flags,true);size_t n=is_full?1:2;
      for(size_t i=0;i<rows.size();++i){branches[i]=":long="+std::to_string(values[i*n]);
        if(!is_full)branches[i]+=":short="+std::to_string(values[i*n+1]);}
    }
  }
  for(size_t i=0;i<rows.size();++i)groups[signature(w,rows[i],c,is_full,branches[i])].push_back(i);
  for(const auto& [key,ids]:groups){
    std::vector<Row> group;for(auto i:ids)group.push_back(rows[i]);
    auto values=bind_group(w,std::move(group),c,r,is_full,slots,options,dense);
    for(size_t i=0;i<ids.size();++i)result[ids[i]]=std::move(values[i]);
  }
  return result;
}
template<class Config> Fields read_impl(const NodeWeights& w,const std::vector<ReadInput>& requests,const Config& c,const RegionalConfig& r,int dense){
  std::vector<Row> rows;
  for(const auto& q:requests)rows.push_back({q.state?std::make_optional(*q.state):std::nullopt,q.time,q.content.value,{}});
  return bind_queries(w,rows,c,r,false,0,Options{},dense);
}
template<class Config> std::vector<FullResult> full_impl(const NodeWeights& w,const std::vector<FullInput>& requests,Index slots,const Options& options,const Config& c,int dense){
  std::vector<Row> rows;std::vector<FullResult> result;
  for(const auto& q:requests)rows.push_back({*q.comparison,q.time,q.content.value,q.control});
  for(const auto& fields:bind_queries(w,rows,c,RegionalConfig{},true,slots,options,dense)){
    FullResult row{fields.at("value"),{}};
    for(Index j=0;j<slots;++j)row.emitted.push_back({j,row.value});result.push_back(std::move(row));
  }
  return result;
}
}
std::vector<TensorFields> read_vjp(const NodeWeights& w,const std::vector<ReadInput>& r,const KVConfig& c,const RegionalConfig& region,int dense){return read_impl(w,r,c,region,dense);}
std::vector<TensorFields> read_vjp(const NodeWeights& w,const std::vector<ReadInput>& r,const RecurrentConfig& c,const RegionalConfig& region,int dense){return read_impl(w,r,c,region,dense);}
std::vector<FullResult> full_vjp(const NodeWeights& w,const std::vector<FullInput>& r,Index slots,const Options& o,const KVConfig& c,int dense){return full_impl(w,r,slots,o,c,dense);}
std::vector<FullResult> full_vjp(const NodeWeights& w,const std::vector<FullInput>& r,Index slots,const Options& o,const RecurrentConfig& c,int dense){return full_impl(w,r,slots,o,c,dense);}
}
