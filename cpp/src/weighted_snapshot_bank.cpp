// Reuse actual overlapping storage, never infer sharing from numerical equality.
#include "weighted_snapshot_bank.h"
#include "weighted_packed.h"
#include "tide/autograd_census.h"
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <algorithm>
#include <array>
#include <optional>

namespace tide::weighted::query {
namespace {
using torch::autograd::variable_list;
struct Bank {
  std::vector<std::array<Index,3>> spans; // source, absolute offset, row count
  std::vector<Index> lengths,tail;
  Tensor index;
  Index size=0,unit=1,width=1;
  Tensor values(const variable_list& rows)const{
    variable_list parts;
    for(const auto& s:spans)parts.push_back(rows[s[0]].as_strided({s[2],unit},{unit,1},s[1]));
    parts.push_back(at::zeros({1,unit},rows[0].options()));
    std::vector<Index> shape{size+1};shape.insert(shape.end(),tail.begin(),tail.end());
    return at::cat(parts).reshape(shape);
  }
  Tensor gather(const Tensor& values)const{
    std::vector<Index> shape=index.sizes().vec();
    shape.insert(shape.end(),values.sizes().begin()+1,values.sizes().end());
    return values.index_select(0,index.flatten()).reshape(shape);
  }
};
std::optional<Bank> plan(const variable_list& rows){
  if(rows.size()<2||rows[0].dim()==0)return {};
  Bank bank;const auto& ref=rows[0];bank.tail=ref.sizes().vec();bank.tail.erase(bank.tail.begin());
  for(auto n:bank.tail)bank.unit*=n;
  if(!bank.unit)return {};
  std::map<const c10::StorageImpl*,size_t> group_ids;
  std::vector<std::vector<std::array<Index,3>>> groups;
  Index total=0;
  for(size_t i=0;i<rows.size();++i){
    const auto& t=rows[i];auto shape=t.sizes().vec();
    if(shape.empty())return {};shape.erase(shape.begin());
    if(shape!=bank.tail||t.scalar_type()!=ref.scalar_type()||t.device()!=ref.device()||
        !t.is_contiguous()||t.is_conj()||t.is_neg()||t.storage_offset()%bank.unit)return {};
    auto start=t.storage_offset()/bank.unit;
    auto storage=t.storage().unsafeGetStorageImpl();
    if(!group_ids.count(storage)){group_ids[storage]=groups.size();groups.emplace_back();}
    groups[group_ids.at(storage)].push_back({start,start+t.size(0),static_cast<Index>(i)});
    bank.lengths.push_back(t.size(0));bank.width=std::max(bank.width,t.size(0));total+=t.size(0);
  }
  std::vector<Index> offsets(rows.size());
  for(auto& ranges:groups){
    std::sort(ranges.begin(),ranges.end());
    for(size_t begin=0;begin<ranges.size();){
      auto start=ranges[begin][0],end=ranges[begin][1];size_t stop=begin+1;
      while(stop<ranges.size()&&ranges[stop][0]<=end){end=std::max(end,ranges[stop][1]);++stop;}
      bank.spans.push_back({ranges[begin][2],start*bank.unit,end-start});
      for(auto j=begin;j<stop;++j)offsets[ranges[j][2]]=bank.size+ranges[j][0]-start;
      bank.size+=end-start;begin=stop;
    }
  }
  if(bank.size>=total)return {};
  std::vector<Index> index;index.reserve(rows.size()*bank.width);
  for(size_t i=0;i<rows.size();++i)for(Index j=0;j<bank.width;++j)
    index.push_back(j<bank.lengths[i]?offsets[i]+j:bank.size);
  bank.index=at::tensor(index,ref.options().dtype(at::kLong)).reshape({static_cast<Index>(rows.size()),bank.width});
  return bank;
}
class BankVjp final:public torch::autograd::Node {
 public:
  Bank bank;
  bool projected=false;
  Tensor safe,visible;
  std::vector<torch::autograd::SavedVariable> originals;
  variable_list apply(variable_list&& bars)override{
    if(at::GradMode::is_enabled())throw std::invalid_argument("KV snapshot bank supports first-order VJP only");
    if(originals.empty())throw std::runtime_error("KV bank graph released; use retain_graph");
    variable_list inputs,result(originals.size());for(auto& x:originals)inputs.push_back(x.unpack());
    if(!bars[0].defined())return result;
    auto grad=projected?at::where(visible.unsqueeze(-1),bars[0],at::zeros_like(bars[0])):bars[0];
    size_t first=projected?1:0;Tensor row_grads=grad;
    if(projected){
      const auto& weight=inputs[0];auto flat=grad.reshape({-1,weight.size(0)});
      if(should_compute_output(0))result[0]=at::mm(flat.t(),bank.gather(safe).reshape({-1,weight.size(1)}));
      bool used=false;for(size_t i=first;i<inputs.size();++i)used|=should_compute_output(i);
      row_grads=used?at::mm(flat,weight).reshape({grad.size(0),grad.size(1),weight.size(1)}):Tensor{};
    }
    for(size_t i=first;i<inputs.size();++i)if(should_compute_output(i))
      result[i]=row_grads[i-first].slice(0,0,bank.lengths[i-first]);
    return result;
  }
  void release_variables()override{originals.clear();safe=Tensor{};visible=Tensor{};bank.index=Tensor{};}
  std::string name()const override{return projected?"tide::weighted::SnapshotProjectVjp":"tide::weighted::SnapshotGatherVjp";}
};
void attach_bank_vjp(Tensor& output,const variable_list& inputs,std::shared_ptr<BankVjp> node){
  bool grad=false;for(const auto& t:inputs)grad|=t.requires_grad();
  if(!at::GradMode::is_enabled()||!grad)return;
  node->set_next_edges(torch::autograd::collect_next_edges(inputs));
  for(const auto& t:inputs)node->originals.emplace_back(t,false);
  if(autograd_census::enabled()){
    autograd_census::saved(inputs);
    if(node->projected)autograd_census::saved({node->safe,node->visible,node->bank.index});
  }
  torch::autograd::set_history(output,node);
}
}
Tensor bank_gather(const std::vector<Tensor>& rows){
  auto p=plan(rows);if(!p)return {};
  auto node=std::make_shared<BankVjp>();node->bank=std::move(*p);Tensor output;
  {at::NoGradGuard guard;output=node->bank.gather(node->bank.values(rows));}
  work::packing::projection("kv_unique_gather",node->bank.size);
  attach_bank_vjp(output,rows,node);return output;
}
Tensor bank_project(const std::vector<Tensor>& rows,const Tensor& weight,const Tensor& visible){
  auto p=plan(rows);if(!p)return {};
  auto node=std::make_shared<BankVjp>();node->bank=std::move(*p);node->projected=true;node->visible=visible;
  Tensor output;
  {
    at::NoGradGuard guard;auto values=node->bank.values(rows);
    // Integer visibility union, no atomic floating-point gradient accumulation.
    auto used=at::zeros({node->bank.size+1},visible.options().dtype(at::kLong));
    used.index_add_(0,node->bank.index.flatten(),visible.to(at::kLong).flatten());
    node->safe=at::where((used>0).unsqueeze(-1),values,at::zeros_like(values));
    output=node->bank.gather(packed::project(node->safe,weight));
    output=at::where(visible.unsqueeze(-1),output,at::zeros_like(output));
  }
  work::packing::projection("kv_unique_projection",node->bank.size);
  variable_list inputs{weight};inputs.insert(inputs.end(),rows.begin(),rows.end());
  attach_bank_vjp(output,inputs,node);return output;
}
}
