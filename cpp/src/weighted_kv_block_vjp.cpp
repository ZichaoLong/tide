// Column leaves and block adjoints; outer roots retain original causal support.
#include "weighted_kv_block.h"
#include <ATen/record_function.h>
#include "tide/autograd_census.h"
#include "tide/operator_profile.h"
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <set>

namespace tide::weighted::block {
namespace {
using torch::autograd::variable_list;
enum class Part {parameter,old,writes,query};
struct Entry {Part part;std::vector<size_t> ids;std::string name;size_t owner=0;};
struct Layout {
  std::vector<Entry> entries;
  std::map<std::string,size_t> parameters;
  variable_list inputs;
  size_t add(const Tensor& t){inputs.push_back(t);return inputs.size()-1;}
  Layout(const NodeWeights& w,const Plan& p){
    std::map<const c10::TensorImpl*,size_t> aliases;
    auto param=[&](const std::string& name,const Tensor& t){
      auto id=t.unsafeGetTensorImpl();
      if(!aliases.count(id)){aliases[id]=entries.size();entries.push_back({Part::parameter,{add(t)},name});}
      parameters[name]=aliases.at(id);
    };
    param("$bias",w.bias);
    const std::string prefix=p.full?"full_program.":"read_program.";
    for(const auto& [name,t]:w.extra)if(name.find("kernel.")==0||name.find(prefix)==0)param(name,t);
    for(size_t b=0;b<p.banks.size();++b)for(const auto& name:{"keys","values","summaries","weights"})
      entries.push_back({Part::old,{add(p.banks[b]->old.slots.at(name))},name,b});
    for(const auto& name:{"values","summaries","contents"}){
      Entry entry{Part::writes,{},name};
      for(const auto& b:p.banks){const auto& ts=entry.name=="values"?b->values:entry.name=="summaries"?b->summaries:b->contents;
        for(const auto& t:ts)entry.ids.push_back(add(t));}
      entries.push_back(std::move(entry));
    }
    for(const auto& name:{"payload","control"}){
      if(std::string(name)=="control"&&!p.full)continue;Entry entry{Part::query,{},name};
      for(const auto& r:p.rows)entry.ids.push_back(add(entry.name=="payload"?r.payload:r.control));entries.push_back(std::move(entry));
    }
    for(const auto& b:p.banks)for(const auto& t:b->guards)if(t.defined())add(t);
    for(const auto& v:p.views){add(v->snapshot.value);for(const auto& [name,t]:v->snapshot.slots)add(t);}
  }
};
std::pair<variable_list,TensorFields> numerical(NodeWeights w,const KVConfig& c,const RegionalConfig& r,
    const Plan& p,const Layout& layout,bool private_leaves){
  variable_list leaves;
  {
    at::AutoGradMode mode(!private_leaves&&at::GradMode::is_enabled());
    for(const auto& entry:layout.entries){variable_list rows;bool grad=false;
      for(auto id:entry.ids){rows.push_back(layout.inputs[id]);grad|=layout.inputs[id].requires_grad();}
      Tensor raw=entry.part==Part::parameter||entry.part==Part::old?rows[0]:at::stack(rows);
      leaves.push_back(private_leaves?raw.detach().set_requires_grad(grad):raw);
    }
  }
  w.bias=leaves[layout.parameters.at("$bias")];
  for(auto& [name,t]:w.extra){auto it=layout.parameters.find(name);if(it!=layout.parameters.end())t=leaves[it->second];}
  std::vector<TensorFields> old(p.banks.size());TensorFields columns;
  for(size_t i=0;i<layout.entries.size();++i){const auto& e=layout.entries[i];
    if(e.part==Part::old)old[e.owner][e.name]=leaves[i];
    if(e.part==Part::writes||e.part==Part::query)columns[e.name]=leaves[i];}
  return {leaves,compute(w,c,r,p,old,columns.at("values"),columns.at("summaries"),columns.at("contents"),
                        columns.at("payload"),p.full?columns.at("control"):Tensor{})};
}
class Query final:public torch::autograd::Node {
 public:
  Plan plan;std::vector<Entry> layout;
  std::vector<torch::autograd::SavedVariable> originals,leaves,refs;
  std::vector<size_t> output_map;
  variable_list apply(variable_list&& bars)override{
    RECORD_FUNCTION("tide::stage::query_block_backward", std::vector<c10::IValue>());
    op_profile::Scope profile(op_profile::QueryAdjoint);
    if(at::GradMode::is_enabled())throw std::invalid_argument("KV block query supports first-order VJP only");
    if(originals.empty())throw std::runtime_error("KV block graph released; use retain_graph");
    variable_list inputs,xs,ys;
    for(auto& t:originals)inputs.push_back(t.unpack());for(auto& t:leaves)xs.push_back(t.unpack());for(auto& t:refs)ys.push_back(t.unpack());
    const size_t width=ys.size(),count=plan.rows.size();variable_list all(count*width),result(inputs.size());
    for(size_t i=0;i<bars.size();++i)all[output_map[i]]=bars[i];
    std::map<std::vector<size_t>,std::vector<size_t>> groups;
    for(size_t i=0;i<count;++i){std::vector<size_t> mask;
      for(size_t f=0;f<width;++f)if(all[i*width+f].defined())mask.push_back(f);if(!mask.empty())groups[mask].push_back(i);}
    for(const auto& [mask,rows]:groups){
      std::set<size_t> used(rows.begin(),rows.end());variable_list roots,cotangents,active;std::vector<size_t> active_ids;
      for(auto f:mask)if(ys[f].requires_grad()){
        variable_list cs;auto zero=at::zeros_like(ys[f][0]);
        for(size_t i=0;i<count;++i)cs.push_back(used.count(i)?all[i*width+f]:zero);
        roots.push_back(ys[f]);cotangents.push_back(at::stack(cs));
      }
      for(size_t i=0;i<xs.size();++i)if(xs[i].requires_grad()){active.push_back(xs[i]);active_ids.push_back(i);}
      if(roots.empty()||active.empty())continue;
      variable_list grads;{
        RECORD_FUNCTION("tide::stage::query_block_gradient", std::vector<c10::IValue>());
        op_profile::Scope gradient(op_profile::QueryGradient);
        grads=torch::autograd::grad(roots,active,cotangents,true,false,true);
      }
      std::map<size_t,Index> needed;for(auto i:rows)needed[plan.owner[i]]=std::max(needed[plan.owner[i]],plan.views[i]->end);
      auto add=[&](size_t id,const Tensor& grad){if(inputs[id].requires_grad()&&should_compute_output(id))
        result[id]=result[id].defined()?result[id]+grad:grad;};
      for(size_t j=0;j<active_ids.size();++j){if(!grads[j].defined())continue;
        const auto& e=layout[active_ids[j]];const auto& grad=grads[j];
        if(e.part==Part::parameter){add(e.ids[0],grad);continue;}
        if(e.part==Part::old){if(needed.count(e.owner))add(e.ids[0],grad);continue;}
        auto pieces=grad.unbind();
        if(e.part==Part::query){for(auto i:rows)add(e.ids[i],pieces[i]);continue;}
        size_t offset=0;
        for(size_t b=0;b<plan.banks.size();++b){const auto& bank=*plan.banks[b];
          Index end=needed.count(b)?needed.at(b)-bank.old.slots.at("times").numel():0;
          end=std::max(Index(0),std::min(end,static_cast<Index>(bank.values.size())));
          for(Index i=0;i<end;++i)add(e.ids[offset+i],pieces[offset+i]);offset+=bank.values.size();}
      }
    }
    return result;
  }
  void release_variables()override{originals.clear();leaves.clear();refs.clear();plan=Plan{};layout.clear();}
  std::string name()const override{return "tide::weighted::BlockQueryVjp";}
};
}
std::vector<TensorFields> evaluate(const NodeWeights& w,const KVConfig& c,const RegionalConfig& r,const Plan& p){
  RECORD_FUNCTION("tide::stage::query_block_bind", std::vector<c10::IValue>());
  op_profile::Scope profile(op_profile::QueryBind);const bool grad=at::GradMode::is_enabled();Layout layout(w,p);
  variable_list leaves;TensorFields columns;
  if(grad){at::AutoGradMode enable(true);std::tie(leaves,columns)=numerical(w,c,r,p,layout,true);}
  else std::tie(leaves,columns)=numerical(w,c,r,p,layout,false);
  std::shared_ptr<Query> node;
  if(grad){node=std::make_shared<Query>();node->plan=p;node->layout=layout.entries;
    node->set_next_edges(torch::autograd::collect_next_edges(layout.inputs));
    for(const auto& t:layout.inputs)node->originals.emplace_back(t,false);
    for(const auto& t:leaves)node->leaves.emplace_back(t,false);
    variable_list refs;for(const auto& [name,t]:columns){node->refs.emplace_back(t,false);refs.push_back(t);}
    if(autograd_census::enabled()){autograd_census::graph(p.full?"private/query.block.full":"private/query.block.read",refs);
      autograd_census::saved(layout.inputs);autograd_census::saved(leaves);autograd_census::saved(refs);}
  }
  profile.phase(op_profile::QueryPublish);at::NoGradGuard guard;std::vector<TensorFields> result(p.rows.size());
  size_t f=0;
  for(const auto& [name,ref]:columns){auto rows=ref.unbind();
    for(size_t i=0;i<rows.size();++i){auto value=rows[i].detach();
      if(grad&&ref.requires_grad()){node->output_map.push_back(i*columns.size()+f);torch::autograd::set_history(value,node);}
      result[i][name]=value;}++f;}
  return result;
}
}
