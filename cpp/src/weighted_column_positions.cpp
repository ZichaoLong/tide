// KV §2–4: common integer origins, bounded separate phases and literal fallback.
#include "weighted_column_queries.h"
#include "weighted_column_query_ops.h"
#include "tide/packing_work.h"
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <cmath>
#include <limits>
namespace tide::weighted::columns {
namespace {
using detail::require;
Tensor rotate_value(const Tensor& value,const Tensor& offsets,const std::vector<std::vector<double>>& frequencies,
                    double scale,bool transpose=false){
  auto coordinates=detail::time_values(offsets,value)/scale;if(transpose)coordinates=-coordinates;
  std::vector<Tensor> heads;for(Index h=0;h<value.size(1);++h)heads.push_back(rotate_queries(value.select(1,h),coordinates,frequencies[h]));
  return at::stack(heads,1);
}
class Rotation final:public torch::autograd::Node {
 public:
  std::vector<torch::autograd::SavedVariable> saved;
  std::vector<std::vector<double>> frequencies;double scale=1;
  torch::autograd::variable_list apply(torch::autograd::variable_list&& bars)override{
    if(at::GradMode::is_enabled())throw std::invalid_argument("column rotary positions support first-order VJP only");
    if(saved.empty())throw std::runtime_error("column rotary graph released");
    auto value=saved[0].unpack(),offsets=saved[1].unpack();
    if(!bars[0].defined())return {Tensor{}};
    return {rotate_value(bars[0],offsets,frequencies,scale,true)};
  }
  void release_variables()override{saved.clear();}
  std::string name()const override{return "tide::weighted::ColumnRotaryVjp";}
};
Tensor rotation(const Tensor& value,const Tensor& offsets,const std::vector<std::vector<double>>& frequencies,double scale){
  const bool grad=at::GradMode::is_enabled()&&value.requires_grad();at::NoGradGuard guard;
  auto result=rotate_value(value,offsets,frequencies,scale);
  if(grad){auto node=std::make_shared<Rotation>();node->scale=scale;node->frequencies=frequencies;
    node->set_next_edges(torch::autograd::collect_next_edges(value));
    node->saved.emplace_back(value,false);node->saved.emplace_back(offsets,false);torch::autograd::set_history(result,node);}
  return result;
}
struct Positions {Tensor query,key;std::string reason;};
Positions rebase(const PositionQuery& r){
  const auto dtype=r.query.scalar_type();
  if(dtype!=at::kDouble&&dtype!=at::kFloat)return {{}, {}, "half"};
  const double epsilon=dtype==at::kDouble?std::numeric_limits<double>::epsilon():std::numeric_limits<float>::epsilon();
  const Index integer_limit=Index{1}<<(dtype==at::kDouble?53:24);
  const double phase_limit=1/(8*std::sqrt(epsilon));double frequency=0;
  for(const auto& head:r.frequencies)for(auto f:head)frequency=std::max(frequency,std::abs(f));
  const auto& q=r.query_layout;const auto& k=r.bank_layout;
  require(q.samples()==k.samples(),"position query/bank sample domains differ");
  Index qw=1,kw=1;std::vector<Index> origins;
  for(Index b=0;b<q.samples();++b){const auto a=q.offsets[b],z=q.offsets[b+1],ka=k.offsets[b],kz=k.offsets[b+1];
    qw=std::max(qw,z-a);kw=std::max(kw,kz-ka);
    Index first=std::numeric_limits<Index>::max(),last=0;
    if(a<z){first=std::min(first,q.times[a]);last=std::max(last,q.times[z-1]);}
    if(ka<kz){first=std::min(first,k.times[ka]);last=std::max(last,k.times[kz-1]);}
    if(first==std::numeric_limits<Index>::max())first=last;
    const auto span=last-first;
    if(span>integer_limit)return {{}, {}, "integer_span"};
    const double phase=static_cast<double>(span)/r.scale*frequency;
    // Bounds a phase arithmetic error proxy, not output/route equivalence.
    if(!std::isfinite(phase)||phase>phase_limit)return {{}, {}, "phase_span"};
    origins.push_back(first);
  }
  std::vector<Index> qt(q.samples()*qw),kt(k.samples()*kw);
  for(Index b=0;b<q.samples();++b){
    for(Index j=q.offsets[b];j<q.offsets[b+1];++j)qt[b*qw+j-q.offsets[b]]=q.times[j]-origins[b];
    for(Index j=k.offsets[b];j<k.offsets[b+1];++j)kt[b*kw+j-k.offsets[b]]=k.times[j]-origins[b];
  }
  auto options=at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  return {at::tensor(qt,options).reshape({q.samples(),qw}),at::tensor(kt,options).reshape({k.samples(),kw}),"rebased"};
}
Tensor relative(const PositionQuery& r){
  auto displacement=-detail::time_values(r.age,r.query)/r.scale;std::vector<Tensor> logits;
  for(Index h=0;h<r.query.size(1);++h){auto key=rotate_queries(r.key.select(1,h).unsqueeze(1).expand({-1,r.query.size(2),-1,-1}),
      displacement,r.frequencies[h]);auto head=r.query.select(1,h).unsqueeze(2);
    if(r.query.scalar_type()==at::kHalf){head=head.to(at::kFloat);key=key.to(at::kFloat);}
    logits.push_back((head*key).sum(-1)/std::sqrt(double(r.query.size(-1))));
  }
  return at::stack(logits,1);
}
}
std::vector<Tensor> position_logits_many(const std::vector<PositionQuery>& requests){
  std::vector<Tensor> result(requests.size());std::vector<size_t> selected;std::vector<std::vector<Tensor>> operands;
  for(size_t i=0;i<requests.size();++i){const auto& r=requests[i];auto q=r.query,k=r.key;
    require(q.dim()==4&&k.dim()==4&&q.size(0)==k.size(0)&&q.size(1)==k.size(1)&&q.size(3)==k.size(3)
      &&q.scalar_type()==k.scalar_type()&&q.device()==k.device()&&static_cast<Index>(r.frequencies.size())==q.size(1)
      &&r.age.scalar_type()==at::kLong&&r.age.sizes()==at::IntArrayRef({q.size(0),q.size(2),k.size(2)}),
      "invalid positioned column query metadata");
    bool rotated=false;for(const auto& f:r.frequencies)rotated|=!f.empty();std::string reason;
    if(rotated){auto positions=rebase(r);reason=positions.reason;
      if(!positions.query.defined()){
        result[i]=relative(r);work::packing::bucket(("kv_rope_relative_"+reason).c_str(),q.size(2),1);continue;}
      q=rotation(q,positions.query.to(q.device()),r.frequencies,r.scale);
      k=rotation(k,positions.key.to(k.device()),r.frequencies,r.scale);
    }else{reason="content_distance";if(q.scalar_type()==at::kHalf){q=q.to(at::kFloat);k=k.to(at::kFloat);}}
    work::packing::bucket(("kv_rope_"+reason).c_str(),q.size(2),1);selected.push_back(i);
    operands.push_back({q.flatten(0,1),k.flatten(0,1).transpose(-1,-2)});
  }
  auto values=contract_nodes(Contraction::matmul,operands);
  for(size_t j=0;j<selected.size();++j){auto i=selected[j];const auto& q=requests[i].query;
    result[i]=values[j].reshape({q.size(0),q.size(1),q.size(2),requests[i].key.size(2)})/std::sqrt(double(q.size(3)));}
  for(size_t i=0;i<requests.size();++i){const auto& r=requests[i];bool distance=false;for(auto s:r.slopes)distance|=s!=0;
    if(distance){auto displacement=-detail::time_values(r.age,r.query)/r.scale;
      auto slopes=at::tensor(r.slopes,result[i].options()).reshape({1,-1,1,1});result[i]=result[i]+slopes*displacement.unsqueeze(1);}}
  return result;
}
Tensor position_logits(const PositionQuery& request){return position_logits_many({request}).front();}
}
