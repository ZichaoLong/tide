// Architecture §3/4: actual query batches with independent row/field VJP roots.
#include "weighted_query_vjp.h"
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
template<class Config> std::string signature(const NodeWeights& w,Row row,const Config& c,bool is_full){
  std::ostringstream s;s << (row.state?"state":"initial");
  if(row.state)for(const auto& [name,t]:row.state->slots)s << ':' << name;
  if constexpr(std::is_same_v<Config,KVConfig>){
    if(is_full||c.memory_read=="qry"){
      auto state=row.state?*row.state:w.kernel->initial(w);
      std::pair<Tensor,Tensor> visibility=detail::visible(state,row.time,c);
      const auto& [mask,age]=visibility;
      Tensor writes=state.slots.at("weights").index({mask}).detach();
      s << ":long=" << (writes>0).any().item<bool>();
      if(!is_full){Tensor short_weights=writes*at::exp(-c.beta_short*age.index({mask}).to(writes.scalar_type()));
        s << ":short=" << (short_weights>0).any().item<bool>();}
    }
  }
  tensors(row,[&](Tensor& t){s << '/' << t.sizes() << t.device() << int(t.scalar_type()) << t.requires_grad();});
  return s.str();
}
class QueryVjp final:public torch::autograd::Node {
 public:
  size_t weights=0,row_inputs=0,row_outputs=0,rows=0;
  std::vector<torch::autograd::SavedVariable> originals,leaves,outputs;
  std::vector<size_t> output_map;
  variable_list apply(variable_list&& bars) override {
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
    variable_list result(xs.size());
    for(const auto& [mask,used]:groups){
      variable_list roots,grads,inputs;std::vector<size_t> ids;
      for(auto row:used)for(auto j:mask){roots.push_back(ys[row*row_outputs+j]);grads.push_back(full[row*row_outputs+j]);}
      auto append=[&](size_t i){if(xs[i].requires_grad()&&should_compute_output(i)){ids.push_back(i);inputs.push_back(xs[i]);}};
      for(size_t i=0;i<weights;++i)append(i);
      for(auto row:used)for(size_t j=0;j<row_inputs;++j)append(weights+row*row_inputs+j);
      if(inputs.empty())continue;
      auto values=torch::autograd::grad(roots,inputs,grads,true,false,true);
      for(size_t i=0;i<ids.size();++i)if(values[i].defined()){
        auto& out=result[ids[i]];out=out.defined()?out+values[i]:values[i];
      }
    }
    return result;
  }
  void release_variables() override {originals.clear();leaves.clear();outputs.clear();}
  std::string name() const override {return "tide::weighted::BatchedQueryVjp";}
};
using Fields=std::vector<TensorFields>;
template<class Config> Fields bind_group(NodeWeights w,std::vector<Row> rows,const Config& c,const RegionalConfig& region,
    bool is_full,Index slots,const Options& options){
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
  for(auto& row:rows)tensors(row,leaf);
  node->row_inputs=(inputs.size()-node->weights)/rows.size();
  node->set_next_edges(torch::autograd::collect_next_edges(inputs));
  for(const auto& t:inputs)node->originals.emplace_back(t,false);
  for(const auto& t:leaves)node->leaves.emplace_back(t,false);
  Fields refs;
  {
    at::AutoGradMode enable(true);
    if(is_full){
      std::vector<FullInput> requests;
      for(const auto& row:rows)requests.push_back({&*row.state,row.time,ContentView{row.content},row.control});
      for(const auto& value:full(w,requests,slots,options,c))refs.push_back({{"value",value.value}});
    }else{
      std::vector<ReadInput> requests;
      for(const auto& row:rows)requests.push_back({row.state?&*row.state:nullptr,row.time,ContentView{row.content}});
      refs=describe(w,requests,c,region);
    }
  }
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
    bool is_full,Index slots,const Options& options){
  std::map<std::string,std::vector<size_t>> groups;Fields result(rows.size());
  for(size_t i=0;i<rows.size();++i)groups[signature(w,rows[i],c,is_full)].push_back(i);
  for(const auto& [key,ids]:groups){
    std::vector<Row> group;for(auto i:ids)group.push_back(rows[i]);
    auto values=bind_group(w,std::move(group),c,r,is_full,slots,options);
    for(size_t i=0;i<ids.size();++i)result[ids[i]]=std::move(values[i]);
  }
  return result;
}
template<class Config> Fields read_impl(const NodeWeights& w,const std::vector<ReadInput>& requests,const Config& c,const RegionalConfig& r){
  std::vector<Row> rows;
  for(const auto& q:requests)rows.push_back({q.state?std::make_optional(*q.state):std::nullopt,q.time,q.content.value,{}});
  return bind_queries(w,rows,c,r,false,0,Options{});
}
template<class Config> std::vector<FullResult> full_impl(const NodeWeights& w,const std::vector<FullInput>& requests,Index slots,const Options& options,const Config& c){
  std::vector<Row> rows;std::vector<FullResult> result;
  for(const auto& q:requests)rows.push_back({*q.comparison,q.time,q.content.value,q.control});
  for(const auto& fields:bind_queries(w,rows,c,RegionalConfig{},true,slots,options)){
    FullResult row{fields.at("value"),{}};
    for(Index j=0;j<slots;++j)row.emitted.push_back({j,row.value});result.push_back(std::move(row));
  }
  return result;
}
}
std::vector<TensorFields> read_vjp(const NodeWeights& w,const std::vector<ReadInput>& r,const KVConfig& c,const RegionalConfig& region){return read_impl(w,r,c,region);}
std::vector<TensorFields> read_vjp(const NodeWeights& w,const std::vector<ReadInput>& r,const RecurrentConfig& c,const RegionalConfig& region){return read_impl(w,r,c,region);}
std::vector<FullResult> full_vjp(const NodeWeights& w,const std::vector<FullInput>& r,Index slots,const Options& o,const KVConfig& c){return full_impl(w,r,slots,o,c);}
std::vector<FullResult> full_vjp(const NodeWeights& w,const std::vector<FullInput>& r,Index slots,const Options& o,const RecurrentConfig& c){return full_impl(w,r,slots,o,c);}
}
