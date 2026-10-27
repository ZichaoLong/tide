// Recurrent Mem §2/3: int64 lazy decay and actual absolute-time convolution queries.
#include "weighted_query_batch.h"

namespace tide::weighted::query {
using namespace detail;
using packed::project;
namespace {
Tensor slot(const States& states,const std::string& name){
  std::vector<Tensor> rows;for(const auto& s:states)rows.push_back(s.slots.at(name));return at::stack(rows);
}
Tensor convolve(const NodeWeights& w,const States& states,const Times& times,const Tensor& raw,const RecurrentConfig& c){
  std::map<Index,std::vector<Index>> groups;std::vector<Tensor> result(states.size());
  const auto& weights=weight(w,"kernel.conv.weight");
  for(size_t i=0;i<states.size();++i){
    auto cache=recurrent::cache(states[i]);validate_cache(cache,c.conv_width(),(c.conv_size-1)*c.conv_stride);
    require(times[i]>=0&&at::isfinite(raw[i]).all().item<bool>(),"invalid convolution query time/value");
    groups[cache.times.numel()].push_back(i);
  }
  for(const auto& [length,rows]:groups){
    auto ids=at::tensor(rows,raw.options().dtype(at::kLong));
    auto base=weight(w,"kernel.conv.bias")+weights[0]*raw.index_select(0,ids);
    if(length&&c.conv_size>1){
      std::vector<Tensor> stamps,values;std::vector<Index> clock;
      for(auto i:rows){stamps.push_back(states[i].slots.at("conv_times"));values.push_back(states[i].slots.at("conv_values"));clock.push_back(times[i]);}
      auto old=at::stack(stamps);
      auto taps=at::tensor(clock,old.options()).unsqueeze(-1)-c.conv_stride*at::arange(1,c.conv_size,old.options());
      auto matches=(taps.unsqueeze(-1)==old.unsqueeze(1)).to(raw.scalar_type());
      auto history=at::matmul(matches,at::stack(values));
      base=base+(weights.slice(0,1)*history).sum(1);
    }
    for(size_t j=0;j<rows.size();++j)result[rows[j]]=base[j];
  }
  return at::stack(result);
}
}
std::array<Tensor,3> effective(const States& states,const Times& times,const RecurrentConfig& c){
  for(auto t:times)require(t>=0,"negative recurrent query time");
  auto clock=slot(states,"time"),H=slot(states,"H"),Z=slot(states,"Z"),W=slot(states,"W");
  auto delta=(at::tensor(times,clock.options())-clock).clamp_min(0).to(H.scalar_type());
  auto rates=c.basic()?std::vector<double>{c.lambda_short,0.}:c.matrix_rates;
  if(rates.empty())rates.resize(c.heads,0.);
  auto matrix=at::exp(-delta.unsqueeze(-1)*at::tensor(rates,H.options()));
  auto summary=at::exp(-delta.unsqueeze(-1)*at::tensor({c.lambda_short,0.},Z.options()));
  return {matrix.unsqueeze(-1).unsqueeze(-1)*H,summary.unsqueeze(-1)*Z,summary*W};
}
Features features(const NodeWeights& w,const States& states,const Times& times,const Tensor& u,const RecurrentConfig& c){
  auto projection=project(normalize(u,c.epsilon),weight(w,"kernel.projection"));
  Features f;f.raw=projection;const Index inner=c.heads*c.value_width,bc=c.groups*c.key_width;
  if(c.memory_family=="ssd"){
    f.gate=projection.slice(1,0,inner);f.raw=projection.slice(1,inner,inner+c.conv_width());
    f.delta=at::softplus(projection.slice(1,inner+c.conv_width())+weight(w,"kernel.delta_bias"));
  }
  auto all=at::silu(convolve(w,states,times,f.raw,c));
  if(c.memory_family=="gated_delta"){
    all=all.reshape({-1,c.heads,2*c.key_width+c.value_width});
    f.query=all.slice(-1,0,c.key_width);f.key=all.slice(-1,c.key_width,2*c.key_width);f.value=all.slice(-1,2*c.key_width);
    f.query=f.query/at::sqrt(c.qk_epsilon+f.query.square().sum(-1,true))/std::sqrt(c.key_width);
    f.key=f.key/at::sqrt(c.qk_epsilon+f.key.square().sum(-1,true));
  }else{
    f.value=all.slice(1,0,inner).reshape({-1,c.heads,c.value_width});
    f.key=all.slice(1,inner,inner+bc).reshape({-1,c.groups,c.key_width}).repeat_interleave(c.heads/c.groups,1);
    f.query=all.slice(1,inner+bc).reshape({-1,c.groups,c.key_width}).repeat_interleave(c.heads/c.groups,1);
  }
  return f;
}
std::array<Tensor,4> memory(const NodeWeights& w,const States& states,const Times& times,const Tensor& u,const RecurrentConfig& c){
  auto s=effective(states,times,c);auto values=s[1]/s[2].clamp_min(1).unsqueeze(-1);
  auto a=values.select(1,0),b=values.select(1,1);
  if(c.memory_read=="state"){
    if(c.basic()){
      auto q=project(normalize(u,c.epsilon),weight(w,"kernel.read_query"));
      auto reads=at::matmul(s[0],q.unsqueeze(1).unsqueeze(-1));
      values=at::matmul(weight(w,"kernel.read_out"),reads).squeeze(-1);
      a=values.select(1,0);b=values.select(1,1);
    }else{
      auto f=features(w,states,times,u,c);
      auto H=effective(states,times,c)[0]; // Preserve the scalar function's separate query branch.
      auto read=at::matmul(H,f.query.unsqueeze(-1)).squeeze(-1);
      b=project(read.flatten(1),weight(w,"kernel.read_out"));
    }
  }
  return {a,b,s[2].select(1,0),s[2].select(1,1)};
}
}
