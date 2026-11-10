// KV §2: actual prefix snapshots and deterministic bulk first-order adjoints.
// Only untraced execution; the traced independent-state root chain is retained.
#include "weighted_kv_sequence_vjp.h"
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <array>
#include <algorithm>
#include <stdexcept>

namespace tide::weighted {
namespace {
using torch::autograd::variable_list;
const std::array<std::string,4> fields{"keys","values","summaries","weights"};
struct Write { size_t event,value,summary,content; };
struct Owner {
  std::array<size_t,4> old;
  std::vector<Write> writes;
  std::vector<std::pair<Index,Index>> spans;
};
std::pair<Tensor,Index> reduce(const variable_list& bars,const Owner& owner,Index size,const Tensor& ref){
  variable_list values;std::vector<std::array<Index,3>> ranges;Index offset=0,last=-1;
  for(size_t i=0;i<bars.size();++i)if(bars[i].defined()){
    auto [start,end]=owner.spans[i];ranges.push_back({offset,start,end});
    values.push_back(bars[i]);offset+=bars[i].size(0);last=std::max(last,end);
  }
  if(values.empty())return {Tensor{},-1};
  auto shape=ref.sizes().vec();shape[0]=1;values.push_back(at::zeros(shape,ref.options()));
  auto packed=at::cat(values);const Index zero=packed.size(0)-1;
  Index width=1;for(size_t i=1;i<shape.size();++i)width*=shape[i];
  const Index chunk=std::max<Index>(1,(16*1024*1024)/(ranges.size()*(8+width*ref.element_size())));
  variable_list pieces;
  for(Index begin=0;begin<size;begin+=chunk){
    auto end=std::min(size,begin+chunk);std::vector<Index> index;
    index.reserve((end-begin)*ranges.size());
    for(Index i=begin;i<end;++i)for(const auto& r:ranges)
      index.push_back(r[1]<=i&&i<r[2]?r[0]+i-r[1]:zero);
    auto ids=at::tensor(index,ref.options().dtype(at::kLong));
    std::vector<Index> dims{end-begin,static_cast<Index>(ranges.size())};
    dims.insert(dims.end(),shape.begin()+1,shape.end());
    pieces.push_back(packed.index_select(0,ids).reshape(dims).sum(1));
  }
  return {pieces.empty()?ref.slice(0,0,0):at::cat(pieces),last};
}
class SequenceVjp final:public torch::autograd::Node {
 public:
  std::vector<torch::autograd::SavedVariable> originals;
  std::vector<Owner> owners;
  std::vector<size_t> output_map;
  size_t total=0;
  variable_list apply(variable_list&& incoming) override {
    if(at::GradMode::is_enabled())throw std::invalid_argument("KV sequence supports first-order VJP only");
    if(originals.empty())throw std::runtime_error("KV sequence graph released; use retain_graph");
    variable_list inputs;for(auto& t:originals)inputs.push_back(t.unpack());
    variable_list bars(4*total),result(inputs.size());
    for(size_t i=0;i<incoming.size();++i)bars[output_map[i]]=incoming[i];
    size_t offset=0;
    for(const auto& owner:owners){
      auto n=inputs[owner.old[0]].size(0);std::array<Tensor,4> values;std::array<Index,4> last;
      for(size_t field=0;field<4;++field){
        auto start=bars.begin()+field*total+offset;
        auto pair=reduce(variable_list(start,start+owner.writes.size()),owner,n+owner.writes.size(),inputs[owner.old[field]]);
        values[field]=pair.first;last[field]=pair.second;
        if(pair.first.defined()&&inputs[owner.old[field]].requires_grad())result[owner.old[field]]=pair.first.slice(0,0,n);
      }
      std::vector<size_t> active;variable_list contents,masses;
      for(size_t j=0;j<owner.writes.size();++j){
        const auto& w=owner.writes[j];Index index=n+j;
        if(inputs[w.value].requires_grad()&&(index<last[0]||index<last[1])){
          variable_list halves;for(size_t f=0;f<2;++f)halves.push_back(index<last[f]?values[f][index]:at::zeros({inputs[w.value].numel()/2},inputs[w.value].options()));
          result[w.value]=at::cat(halves);
        }
        if(inputs[w.summary].requires_grad()&&index<last[2])result[w.summary]=values[2][index];
        if(inputs[w.content].requires_grad()&&index<last[3]){
          active.push_back(w.content);contents.push_back(inputs[w.content]);masses.push_back(values[3][index]);
        }
      }
      if(!active.empty()){
        auto h=at::stack(contents),mass=at::stack(masses)*(h.select(1,0)<=1);
        auto grads=at::cat({mass.unsqueeze(1),at::zeros_like(h.slice(1,1))},1);
        for(size_t j=0;j<active.size();++j)result[active[j]]=grads[j];
      }
      offset+=owner.writes.size();
    }
    for(size_t i=0;i<result.size();++i)if(!should_compute_output(i))result[i]=Tensor{};
    return result;
  }
  void release_variables() override{originals.clear();}
  std::string name() const override{return "tide::weighted::KVSequenceVjp";}
};
}
bool bind_kv_hidden_sequence(const NodeWeights& w,const std::vector<State>& old,
    const PackedSequence& batch,std::vector<State>& numeric,std::vector<State>& previous){
  if(batch.prepared.empty())return false;
  std::vector<std::vector<Index>> positive(old.size());
  for(size_t i=0;i<old.size();++i)for(Index j=batch.offsets[i];j<batch.offsets[i+1];++j)
    if(numeric[j].last_time==batch.times[j]){
      const auto& p=batch.prepared[j];
      if(p.size()!=2||!p.count("values")||!p.count("summaries"))return false;
      positive[i].push_back(j);
    }
  auto node=std::make_shared<SequenceVjp>();variable_list inputs{w.decay,w.weight,w.bias,w.read};
  for(const auto& [name,t]:w.extra)inputs.push_back(t);
  // Same original-version guard surface as the generic step boundary.
  for(const auto& state:old){inputs.push_back(state.value);for(const auto& [name,t]:state.slots)inputs.push_back(t);}
  for(const auto& view:batch.views){inputs.push_back(view.value);
    for(const auto& s:view.sources){inputs.push_back(s.atom.value);inputs.push_back(s.scale);}
    for(const auto& s:view.contributions)inputs.push_back(s.value);}
  for(const auto& p:batch.prepared)for(const auto& [name,t]:p)inputs.push_back(t);
  auto add=[&](const Tensor& t){inputs.push_back(t);return inputs.size()-1;};
  for(size_t i=0;i<old.size();++i){
    Owner owner;for(size_t f=0;f<4;++f)owner.old[f]=add(old[i].slots.at(fields[f]));
    Index n=old[i].slots.at("times").size(0),count=0;
    for(auto j:positive[i]){
      const auto& p=batch.prepared[j];
      owner.writes.push_back({static_cast<size_t>(j),add(p.at("values")),add(p.at("summaries")),add(batch.views[j].value)});
      Index end=n+(++count);owner.spans.push_back({end-numeric[j].slots.at("times").size(0),end});
    }
    node->total+=owner.writes.size();node->owners.push_back(std::move(owner));
  }
  node->set_next_edges(torch::autograd::collect_next_edges(inputs));
  for(const auto& t:inputs)node->originals.emplace_back(t,false);
  size_t output=0;
  for(size_t f=0;f<4;++f){
    variable_list values;std::vector<bool> connected;
    for(const auto& owner:node->owners){
      bool flag=inputs[owner.old[f]].requires_grad();
      for(const auto& row:owner.writes){
        flag=flag||inputs[f==2?row.summary:f==3?row.content:row.value].requires_grad();
        connected.push_back(flag);values.push_back(numeric[row.event].slots.at(fields[f]));
      }
    }
    if(values.empty())continue;
    Tensor joined;{at::NoGradGuard guard;joined=at::cat(values);}
    Index begin=0;size_t index=0;
    for(const auto& owner:node->owners)for(const auto& row:owner.writes){
      auto end=begin+values[index].size(0);auto value=joined.slice(0,begin,end);
      if(connected[index]){node->output_map.push_back(output);torch::autograd::set_history(value,node);}
      numeric[row.event].slots[fields[f]]=value;begin=end;++index;++output;
    }
  }
  previous.resize(numeric.size());
  for(size_t i=0;i<old.size();++i){
    auto current=old[i];size_t k=0;
    for(Index j=batch.offsets[i];j<batch.offsets[i+1];++j){
      previous[j]=current;
      if(k<positive[i].size()&&positive[i][k]==j){numeric[j].value=current.value;current=numeric[j];++k;}
      else numeric[j]=current;
    }
  }
  return true;
}
}
