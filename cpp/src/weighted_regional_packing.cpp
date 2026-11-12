// Core-5 regional row layout and its linear transpose; no sending mathematics.
#include "weighted_regional_packing.h"
#include <torch/csrc/autograd/functions/utils.h>
#include <algorithm>
#include <set>

namespace tide::weighted::columns {
namespace {
void require(bool condition,const char* message){if(!condition)throw std::invalid_argument(message);}
Tensor ids(const std::vector<Index>& values,const at::Device& device){
  return at::tensor(values,at::TensorOptions().dtype(at::kLong)).to(device);
}
bool aligned(const RegionalPackingGroup& g){
  if(static_cast<Index>(g.participants.size())!=g.candidates)return false;
  for(auto n:g.participants)if(g.domains.at(n)!=g.rows)return false;
  return true;
}
const Tensor& validate_sources(const RegionalPackingGroup& g,const RegionalFields& sources){
  require(!g.participants.empty(),"regional packing group is empty");
  for(auto n:g.participants)require(sources.count(n),"regional descriptor source missing");
  const auto& first=sources.at(g.participants[0]);require(!first.empty(),"regional descriptors need fields");
  const auto& ref=first.begin()->second;require(ref.is_floating_point(),"regional descriptors must be floating");
  for(auto n:g.participants){
    const auto& fields=sources.at(n);require(fields.size()==first.size(),"regional descriptor fields differ");
    for(const auto& [name,t]:fields){
      require(first.count(name),"regional descriptor fields differ");
      require(t.dim()>0&&t.size(0)==static_cast<Index>(g.domains.at(n).size())
        &&t.sizes().slice(1)==first.at(name).sizes().slice(1)
        &&t.device()==ref.device()&&t.scalar_type()==ref.scalar_type(),"regional descriptor metadata mismatch");
    }
  }
  return ref;
}
class UnpackVjp final:public torch::autograd::Node {
 public:
  RegionalPacking packing;
  Tensor prototype;
  std::vector<Index> output_nodes;
  explicit UnpackVjp(const RegionalPacking& p):packing(p){}
  torch::autograd::variable_list apply(torch::autograd::variable_list&& bars) override {
    std::map<Index,Tensor> inputs;
    for(size_t i=0;i<bars.size();++i)inputs[output_nodes[i]]=bars[i];
    auto result=packing.controls_pullback(inputs,prototype).values;
    for(size_t i=0;i<result.size();++i)if(!should_compute_output(i))result[i]=Tensor{};
    return result;
  }
  std::string name() const override{return "tide::weighted::RegionalUnpackVjp";}
};
}

RegionalPacking::RegionalPacking(const std::map<Index,std::vector<Index>>& input):domains(input){
  std::map<Index,std::vector<std::pair<Index,Index>>> frames;
  for(const auto& [node,rows]:domains){
    require(node>=0,"regional node ID must be nonnegative int64");
    for(size_t i=0;i<rows.size();++i){
      require(rows[i]>=0&&(!i||rows[i-1]<rows[i]),"regional row domains must be sorted, unique and nonnegative");
      frames[rows[i]].emplace_back(node,i);
    }
    inverse[node].resize(rows.size());
  }
  std::map<Index,std::vector<Index>> counts;
  for(const auto& [row,entries]:frames)counts[entries.size()].push_back(row);
  Index group_offset=0;
  for(const auto& [count,rows]:counts){
    RegionalPackingGroup g;g.candidates=count;g.rows=rows;std::set<Index> members;
    for(auto row:rows)for(const auto& [node,local]:frames.at(row))members.insert(node);
    g.participants={members.begin(),members.end()};std::map<Index,Index> offsets;Index offset=0;
    for(auto node:g.participants){g.domains[node]=domains.at(node);offsets[node]=offset;offset+=domains.at(node).size();}
    for(auto row:rows)for(const auto& [node,local]:frames.at(row)){
      inverse[node][local]=group_offset+g.node_ids.size();
      g.node_ids.push_back(node);g.local_rows.push_back(local);g.gather.push_back(offsets.at(node)+local);
    }
    group_offset+=g.node_ids.size();groups.push_back(std::move(g));
  }
}

PackedRegionalColumns RegionalPackingGroup::pack(const RegionalFields& sources) const {
  const auto& ref=validate_sources(*this,sources);PackedRegionalColumns out;
  auto direct=aligned(*this);auto index=direct?Tensor{}:ids(gather,ref.device());
  for(const auto& [name,first]:sources.at(participants[0])){
    std::vector<Tensor> values;for(auto node:participants)values.push_back(sources.at(node).at(name));
    if(direct)out.fields[name]=values.size()==1?values[0].unsqueeze(1):at::stack(values,1);
    else{
      std::vector<Index> shape{static_cast<Index>(rows.size()),candidates};
      shape.insert(shape.end(),first.sizes().begin()+1,first.sizes().end());
      out.fields[name]=at::cat(values).index_select(0,index).reshape(shape);
    }
  }
  out.nodes=ids(node_ids,ref.device()).reshape({static_cast<Index>(rows.size()),candidates});
  return out;
}

RegionalFields RegionalPackingGroup::pullback(const RegionalFields& sources,const TensorFields& bars,const Tensor& support) const {
  const auto& ref=validate_sources(*this,sources);const auto& names=sources.at(participants[0]);
  for(const auto& [name,t]:bars)require(names.count(name),"unknown regional packed field adjoint");
  std::vector<bool> active(node_ids.size(),true);
  if(support.defined()){
    const auto r=static_cast<Index>(rows.size());
    require(support.scalar_type()==at::kBool&&((support.dim()==1&&support.size(0)==r)
      ||support.sizes()==at::IntArrayRef({r,candidates})),"regional support must be bool R or R,N");
    auto host=support.to(at::kCPU).contiguous();auto p=host.const_data_ptr<bool>();
    for(size_t i=0;i<active.size();++i)active[i]=p[support.dim()==1?i/candidates:i];
  }
  RegionalFields out;
  for(auto node:participants)for(const auto& [name,t]:names)out[node][name]=Tensor{};
  for(const auto& [name,bar]:bars){
    if(!bar.defined())continue;
    auto shape=names.at(name).sizes().vec();shape[0]=candidates;shape.insert(shape.begin(),rows.size());
    require(bar.sizes()==shape&&bar.device()==ref.device()&&bar.scalar_type()==ref.scalar_type(),"regional field adjoint metadata mismatch");
    shape.erase(shape.begin());shape[0]=node_ids.size();auto flat=bar.reshape(shape);
    for(auto node:participants){
      std::vector<Index> positions,target;
      for(size_t i=0;i<node_ids.size();++i)if(node_ids[i]==node&&active[i]){positions.push_back(i);target.push_back(local_rows[i]);}
      if(positions.empty())continue;
      out[node][name]=at::zeros_like(sources.at(node).at(name)).index_copy(0,ids(target,bar.device()),flat.index_select(0,ids(positions,bar.device())));
    }
  }
  return out;
}

RegionalControlAdjoints RegionalPacking::controls_pullback(const std::map<Index,Tensor>& bars,const Tensor& like) const {
  require(like.defined()&&like.is_floating_point(),"regional adjoint reference must be floating");
  std::map<Index,Index> offsets;std::vector<Tensor> values;Index offset=0;
  for(const auto& [node,bar]:bars){
    require(domains.count(node),"unknown regional node adjoint");if(!bar.defined())continue;
    require(bar.sizes()==at::IntArrayRef({static_cast<Index>(domains.at(node).size())})
      &&bar.device()==like.device()&&bar.scalar_type()==like.scalar_type(),"regional node adjoint metadata mismatch");
    offsets[node]=offset;offset+=bar.numel();values.push_back(bar);
  }
  auto bank=values.empty()?Tensor{}:at::cat(values);RegionalControlAdjoints out;
  for(const auto& g:groups){
    std::vector<Index> positions,source,supported(g.rows.size(),0);
    for(size_t i=0;i<g.node_ids.size();++i)if(offsets.count(g.node_ids[i])){
      positions.push_back(i);source.push_back(offsets.at(g.node_ids[i])+g.local_rows[i]);supported[i/g.candidates]=1;
    }
    out.row_support.push_back(ids(supported,like.device()).to(at::kBool));
    if(positions.empty()){out.values.push_back(Tensor{});continue;}
    out.values.push_back(at::zeros({static_cast<Index>(g.node_ids.size())},like.options())
      .index_copy(0,ids(positions,like.device()),bank.index_select(0,ids(source,like.device())))
      .reshape({static_cast<Index>(g.rows.size()),g.candidates}));
  }
  return out;
}

std::map<Index,Tensor> RegionalPacking::unpack(const std::vector<Tensor>& values,const Tensor& like) const {
  require(values.size()==groups.size(),"regional control group count differs");
  auto ref=values.empty()?like:values[0];
  if(!domains.empty())require(ref.defined()&&ref.is_floating_point(),"regional controls need a floating reference");
  for(size_t i=0;i<groups.size();++i)require(values[i].sizes()==at::IntArrayRef({static_cast<Index>(groups[i].rows.size()),groups[i].candidates})
    &&values[i].device()==ref.device()&&values[i].scalar_type()==ref.scalar_type(),"regional control metadata mismatch");
  std::map<Index,Tensor> outputs;
  if(groups.empty()){for(const auto& [node,domain]:domains)outputs[node]=at::empty({0},ref.options());return outputs;}
  if(groups.size()==1&&aligned(groups[0])&&domains.size()==groups[0].participants.size()){
    auto parts=values[0].unbind(1);
    for(size_t i=0;i<parts.size();++i)outputs[groups[0].participants[i]]=parts[i];
    return outputs;
  }
  {
    at::NoGradGuard guard;std::vector<Tensor> flat;for(const auto& value:values)flat.push_back(value.reshape({-1}));
    auto bank=flat.size()==1?flat[0]:at::cat(flat);
    for(const auto& [node,index]:inverse)outputs[node]=bank.index_select(0,ids(index,ref.device()));
  }
  std::set<Index> differentiable;
  if(at::GradMode::is_enabled())for(size_t i=0;i<groups.size();++i)if(values[i].requires_grad())
    differentiable.insert(groups[i].participants.begin(),groups[i].participants.end());
  if(!differentiable.empty()){
    auto node=std::make_shared<UnpackVjp>(*this);node->prototype=at::empty({0},ref.options());
    node->set_next_edges(torch::autograd::collect_next_edges(values));
    for(auto& [id,value]:outputs)if(differentiable.count(id)){node->output_nodes.push_back(id);torch::autograd::set_history(value,node);}
  }
  return outputs;
}
}
