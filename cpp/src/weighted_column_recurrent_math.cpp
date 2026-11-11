// Recurrent equations and exact-time convolution; no per-event floating fields.
#include "weighted_column_recurrent.h"
namespace tide::weighted::columns::rec {
using detail::weight;using detail::require;
namespace {
Tensor convolve(const NodeWeights& w,const RecurrentConfig& c,const Layout& queries,const Tensor& raw,
                const Layout& cache,const Tensor& values){
  const auto& weights=weight(w,"kernel.conv.weight");
  auto result=weight(w,"kernel.conv.bias")+weights[0]*raw;
  if(c.conv_size==1||cache.times.empty())return result;
  std::vector<Index> ids;
  for(Index b=0;b<queries.samples();++b)for(Index j=queries.offsets[b];j<queries.offsets[b+1];++j)
    for(Index tap=1;tap<c.conv_size;++tap){
      auto t=queries.times[j]-c.conv_stride*tap;
      auto first=cache.times.begin()+cache.offsets[b],last=cache.times.begin()+cache.offsets[b+1];
      auto found=std::lower_bound(first,last,t);
      ids.push_back(t>=0&&found!=last&&*found==t?found-cache.times.begin():cache.times.size());
    }
  auto history=gather(values,indices(ids,raw.device()).reshape({raw.size(0),c.conv_size-1}));
  return result+(weights.slice(0,1)*history).sum(1);
}
}
std::pair<Layout,Tensor> cache_bank(const NodeWeights& w,const RecurrentConfig& c,const Layout& writes,
                                 const Tensor& x,const Tensor& gain,const RecurrentState& old){
  if(c.basic())return {};
  auto raw=project(x,weight(w,"kernel.projection"));
  if(c.memory_family=="ssd")raw=raw.slice(1,c.heads*c.value_width,c.heads*c.value_width+c.conv_width());
  auto values=at::cat({old.cache_values,gain.unsqueeze(-1)*raw});
  Layout layout;layout.offsets={0};std::vector<Index> order;
  for(Index b=0;b<writes.samples();++b){
    for(Index j=old.cache_layout.offsets[b];j<old.cache_layout.offsets[b+1];++j){
      layout.times.push_back(old.cache_layout.times[j]);order.push_back(j);}
    for(Index j=writes.offsets[b];j<writes.offsets[b+1];++j){
      layout.times.push_back(writes.times[j]);order.push_back(old.cache_layout.times.size()+j);}
    layout.offsets.push_back(layout.times.size());
  }
  return {layout,values.index_select(0,indices(order,values.device()))};
}
Features features(const NodeWeights& w,const RecurrentConfig& c,const Layout& layout,const Tensor& u,
                  const Layout& cache,const Tensor& values){
  auto raw=project(normalize(u,c.epsilon),weight(w,"kernel.projection"));Features f;
  const Index inner=c.heads*c.value_width,bc=c.groups*c.key_width;
  if(c.memory_family=="ssd"){
    f.gate=raw.slice(1,0,inner);
    f.delta=at::softplus(raw.slice(1,inner+c.conv_width())+weight(w,"kernel.delta_bias"));
    raw=raw.slice(1,inner,inner+c.conv_width());
  }
  auto value=at::silu(convolve(w,c,layout,raw,cache,values));
  if(c.memory_family=="gated_delta"){
    value=value.reshape({-1,c.heads,2*c.key_width+c.value_width});
    f.query=detail::epsilon_l2(value.slice(-1,0,c.key_width),c.qk_epsilon)/std::sqrt(double(c.key_width));
    f.key=detail::epsilon_l2(value.slice(-1,c.key_width,2*c.key_width),c.qk_epsilon);
    f.value=value.slice(-1,2*c.key_width);
  }else{
    f.value=value.slice(1,0,inner).reshape({-1,c.heads,c.value_width});
    f.key=value.slice(1,inner,inner+bc).reshape({-1,c.groups,c.key_width}).repeat_interleave(c.heads/c.groups,1);
    f.query=value.slice(1,inner+bc).reshape({-1,c.groups,c.key_width}).repeat_interleave(c.heads/c.groups,1);
  }
  return f;
}
Coefficients coefficients(const NodeWeights& w,const RecurrentConfig& c,const Layout& layout,const Tensor& u,
                          const Tensor& x,const Tensor& gain,const Tensor& gap,const Layout& cache,const Tensor& values){
  if(c.basic()){
    auto k=project(x,weight(w,"kernel.key"));k=k/k.norm(2,-1,true).clamp_min(1);
    auto value=project(x,weight(w,"kernel.value"));
    auto retain=at::exp(-gain*at::softplus(at::matmul(x,weight(w,"kernel.retain"))+weight(w,"kernel.retain_bias")));
    auto decay=detail::passive_decay(gap.unsqueeze(-1),{c.lambda_short,0.},x)*retain.unsqueeze(-1);
    auto g=gain;const bool matrix=c.memory_family=="delta";Tensor a;
    if(matrix){
      g=g*at::sigmoid(at::matmul(x,weight(w,"kernel.write"))+weight(w,"kernel.write_bias"));
      a=(at::eye(c.key_width,x.options())-g.reshape({-1,1,1})*k.unsqueeze(-1)*k.unsqueeze(-2)).unsqueeze(1);
      a=a*decay.unsqueeze(-1).unsqueeze(-1);
    }else a=decay.unsqueeze(-1).unsqueeze(-1);
    auto b=(g.reshape({-1,1,1})*value.unsqueeze(-1)*k.unsqueeze(-2)).unsqueeze(1).expand({-1,2,-1,-1});
    return {a,b,matrix};
  }
  auto f=features(w,c,layout,u,cache,values);auto rates=c.matrix_rates;if(rates.empty())rates.resize(c.heads,0.);
  auto decay=detail::passive_decay(gap.unsqueeze(-1),rates,x);
  if(c.memory_family=="gated_delta"){
    auto delta=at::softplus(project(x,weight(w,"kernel.delta"))+weight(w,"kernel.delta_bias"));
    auto retain=at::exp(-gain.unsqueeze(-1)*weight(w,"kernel.log_rate").exp()*delta);
    auto g=gain.unsqueeze(-1)*at::sigmoid(project(x,weight(w,"kernel.write"))+weight(w,"kernel.write_bias"));
    auto a=at::eye(c.key_width,x.options())-g.unsqueeze(-1).unsqueeze(-1)*f.key.unsqueeze(-1)*f.key.unsqueeze(-2);
    return {a*(decay*retain).unsqueeze(-1).unsqueeze(-1),
            g.unsqueeze(-1).unsqueeze(-1)*f.value.unsqueeze(-1)*f.key.unsqueeze(-2),true};
  }
  auto step=gain.unsqueeze(-1)*f.delta;
  return {(decay*at::exp(-weight(w,"kernel.log_rate").exp()*step)).unsqueeze(-1).unsqueeze(-1),
          step.unsqueeze(-1).unsqueeze(-1)*f.value.unsqueeze(-1)*f.key.unsqueeze(-2),false};
}
Snapshot snapshot(const RecurrentWindow& window,const std::vector<Index>& rows,const std::string& mode,const RecurrentState& initial){
  Snapshot result;result.layout=window.layout.select(rows);std::vector<Index> selected;
  for(Index b=0;b<result.layout.samples();++b)for(Index j=result.layout.offsets[b];j<result.layout.offsets[b+1];++j){
    Index id=b;
    if(mode!="content"){
      auto first=window.write_layout.times.begin()+window.write_layout.offsets[b];
      auto last=window.write_layout.times.begin()+window.write_layout.offsets[b+1];
      auto bound=mode=="proposal"?std::upper_bound(first,last,result.layout.times[j]):std::lower_bound(first,last,result.layout.times[j]);
      if(bound!=first)id=result.layout.samples()+(bound-window.write_layout.times.begin())-1;
    }
    selected.push_back(id);result.times.push_back(mode=="content"?std::max(Index{0},initial.last_times[b]):window.times[id]);
  }
  const auto& source=mode=="content"?initial.fields:window.bank;
  for(const auto& [n,t]:source)result.fields[n]=t.index_select(0,indices(selected,t.device()));
  result.cache_layout=mode=="content"?initial.cache_layout:window.cache_layout;
  result.cache_values=mode=="content"?initial.cache_values:window.cache_values;return result;
}
std::array<Tensor,3> effective(const Snapshot& state,const RecurrentConfig& c){
  const auto &h=state.fields.at("H"),&z=state.fields.at("Z"),&w=state.fields.at("W");
  std::vector<Index> gaps;for(size_t j=0;j<state.times.size();++j)gaps.push_back(std::max(Index{0},state.layout.times[j]-state.times[j]));
  auto gap=indices(gaps,h.device()).unsqueeze(-1);
  auto rates=c.basic()?std::vector<double>{c.lambda_short,0.}:c.matrix_rates;
  if(rates.empty())rates.resize(c.heads,0.);
  auto hd=detail::passive_decay(gap,rates,h).unsqueeze(-1).unsqueeze(-1);
  auto zd=detail::passive_decay(gap,{c.lambda_short,0.},z);return {hd*h,zd.unsqueeze(-1)*z,zd*w};
}
}
