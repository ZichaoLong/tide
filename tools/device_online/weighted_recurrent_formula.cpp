// Literal textbook formulas with independent per-row autograd, no RowGraph,
// configured memory module, candidate decay, raw update, or convolution helpers.
#include "weighted_recurrent_formula.h"
#include <ATen/core/grad_mode.h>
#include <cmath>
namespace tide::device_online::recurrent_check {
using at::Tensor;
namespace {
Tensor normalized(const Tensor& x,double eps){return x/(eps+x.square().mean(-1,true)).sqrt();}
struct Local {
  const weighted::RecurrentConfig& c;const Fields& p;const State& s;int64_t time;
  Tensor u,mass,H,Z,W;
  Tensor linear(const std::string& name,const Tensor& x){auto out=at::mv(p.at(name+".matrix"),x);return p.count(name+".bias")?out+p.at(name+".bias"):out;}
  Tensor constant(const std::string& name){return linear(name,at::ones({1},at::kFloat));}
  Local(const weighted::RecurrentConfig& cfg,const Fields& parameters,const State& state,int64_t t,const Tensor& content,const Tensor& receive)
    :c(cfg),p(parameters),s(state),time(t),u(content),mass(receive){
    const auto age=std::max<int64_t>(0,time-s.slots.at("time").item<int64_t>());
    auto rates=c.basic()?std::vector<double>{c.lambda_short,0}:c.matrix_rates;if(rates.empty())rates.resize(c.heads,0);
    auto delta=at::scalar_tensor(age,at::kLong).to(at::kFloat),matrix_rates=at::tensor(rates,at::kDouble).to(at::kFloat);
    auto summary=(at::tensor({-float(c.lambda_short),0.f},at::kFloat)*delta).exp();
    H=(-matrix_rates*delta).exp().view({-1,1,1})*s.slots.at("H");Z=summary.unsqueeze(1)*s.slots.at("Z");W=summary*s.slots.at("W");
  }
  struct Features {Tensor raw,gate,value,key,query,delta;};
  Features features(){
    Features f;auto projected=linear("projection",normalized(u,c.epsilon));const auto hv=c.heads*c.value_width;
    if(c.memory_family=="gated_delta")f.raw=projected;
    else {f.gate=projected.slice(0,0,hv);f.raw=projected.slice(0,hv,hv+c.conv_width());
      f.delta=at::softplus(projected.slice(0,hv+c.conv_width())+constant("delta_bias"));}
    auto conv=p.at("conv.bias")+p.at("conv.weight")[0]*f.raw;
    auto stamps=s.slots.at("conv_times"),cache=s.slots.at("conv_values");
    if(c.conv_size>1&&stamps.numel()){
      auto match=at::zeros({c.conv_size-1,stamps.numel()},at::kFloat);
      for(int64_t tap=1;tap<c.conv_size;++tap)for(int64_t j=0;j<stamps.numel();++j)
        if(time>=tap*c.conv_stride&&stamps[j].item<int64_t>()==time-tap*c.conv_stride)match[tap-1][j]=1;
      conv=conv+(p.at("conv.weight").slice(0,1)*at::mm(match,cache)).sum(0);
    }
    conv=at::silu(conv);
    if(c.memory_family=="gated_delta"){
      conv=conv.reshape({c.heads,-1});auto q=conv.slice(1,0,c.key_width),k=conv.slice(1,c.key_width,2*c.key_width);
      f.query=q/(c.qk_epsilon+q.square().sum(-1,true)).sqrt()/std::sqrt(c.key_width);
      f.key=k/(c.qk_epsilon+k.square().sum(-1,true)).sqrt();f.value=conv.slice(1,2*c.key_width);
    }else{
      f.value=conv.slice(0,0,hv).reshape({c.heads,c.value_width});std::vector<Tensor> keys,queries;
      for(int64_t h=0;h<c.heads;++h){const auto group=h/(c.heads/c.groups),off=hv+group*c.key_width;
        keys.push_back(conv.slice(0,off,off+c.key_width));queries.push_back(conv.slice(0,off+c.groups*c.key_width,off+(c.groups+1)*c.key_width));}
      f.key=at::stack(keys);f.query=at::stack(queries);
    }return f;
  }
  Tensor matrix_read(const Tensor& query){return at::matmul(H,query.unsqueeze(-1)).squeeze(-1);}
  Fields update(){
    auto x=normalized(u,c.epsilon),weight=mass.clamp_max(1);Tensor matrix,raw;
    if(c.basic()){
      auto key=linear("key",x);key=key/key.norm().clamp_min(1);auto value=linear("value",x);
      auto retained=(-weight*at::softplus(linear("retain",x))).exp()*H;
      if(c.memory_family=="affine")matrix=retained+weight*value.unsqueeze(-1)*key;
      else {auto beta=weight*linear("write",x).sigmoid();matrix=retained+beta*(value-at::matmul(retained,key)).unsqueeze(-1)*key;}
    }else{
      auto f=features();raw=f.raw;auto delta=c.memory_family=="gated_delta"?at::softplus(linear("delta",x)):f.delta;
      auto step=weight*delta,retained=(-constant("log_rate").exp()*step).exp().view({-1,1,1})*H;
      if(c.memory_family=="gated_delta"){
        auto beta=weight*linear("write",x).sigmoid();auto residual=f.value-at::matmul(retained,f.key.unsqueeze(-1)).squeeze(-1);
        matrix=retained+beta.view({-1,1,1})*residual.unsqueeze(-1)*f.key.unsqueeze(1);
      }else matrix=retained+step.view({-1,1,1})*f.value.unsqueeze(-1)*f.key.unsqueeze(1);
    }
    Fields result{{"value",s.value},{"H",matrix},{"Z",Z+weight*linear("summary",x)},{"W",W+weight}};
    if(!c.basic()){
      auto stamps=s.slots.at("conv_times"),mask=stamps>=std::max<int64_t>(0,time-(c.conv_size-1)*c.conv_stride);
      auto ids=at::nonzero(mask).flatten();result["conv_values"]=at::cat({s.slots.at("conv_values").index_select(0,ids),(weight*raw).unsqueeze(0)});
    }return result;
  }
  Fields memory(){
    Tensor sm,lm;if(c.memory_read=="sum"){auto reads=Z/W.clamp_min(1).unsqueeze(1);sm=reads[0];lm=reads[1];}
    else if(c.basic()){auto q=linear("read.query",normalized(u,c.epsilon));sm=linear("read.out.0",at::mv(H[0],q));lm=linear("read.out.1",at::mv(H[1],q));}
    else {sm=Z[0]/W[0].clamp_min(1);lm=linear("read.out",matrix_read(features().query).flatten());}
    return {{"short_memory",sm},{"long_memory",lm},{"short_total",W[0].reshape({1})},{"long_total",W[1].reshape({1})}};
  }
  Fields full(){
    Tensor mix;auto x=normalized(u,c.epsilon);
    if(c.basic()){auto query=linear("full.query",x);mix=linear("full.out.0",at::mv(H[0],query))+linear("full.out.1",at::mv(H[1],query));}
    else {auto f=features();auto read=matrix_read(f.query);auto rho=constant("full.rho");
      if(c.memory_family=="gated_delta")mix=linear("full.out",normalized(read,c.output_epsilon).flatten()*rho*at::silu(linear("full.output_gate",x)));
      else {auto y=(read+constant("full.direct").unsqueeze(1)*f.value).flatten()*at::silu(f.gate);mix=linear("full.out",rho*normalized(y,c.output_epsilon));}
    }
    auto residual=u+c.alpha_mem*mix,n=normalized(residual,c.epsilon);
    return {{"G",residual+c.alpha_ff*linear("full.down",at::silu(linear("full.gate",n))*linear("full.up",n))}};
  }
};
}
Formula formula(const weighted::RecurrentConfig& c,WeightedRecurrentAction action,const WeightedRawLayout& l,
    const std::vector<State>& states,const Tensor& times,const Tensor& mass,const Tensor& content,const Tensor& owners,const Tensor& present,const WeightedRecurrentParameters& weights){
  at::AutoGradMode enabled(true);const auto rows=states.size();const auto banks=weights.at("summary.matrix").size(0);Formula f;auto old=states;
  std::vector<Fields> params(banks);std::vector<Tensor> masses,inputs;
  for(size_t r=0;r<rows;++r){for(const auto& field:l.fields()){
      auto& value=field.name=="value"?old[r].value:old[r].slots.at(field.name);value=value.to(at::kFloat).clone().set_requires_grad(true);
      f.leaves.push_back(value);f.labels.push_back({"raw."+field.name,int64_t(r),false});}
    masses.push_back(mass[r].reshape({1}).clone().set_requires_grad(true));f.leaves.push_back(masses.back());f.labels.push_back({"mass",int64_t(r),false});
    inputs.push_back(content[r].clone().set_requires_grad(true));f.leaves.push_back(inputs.back());f.labels.push_back({"content",int64_t(r),false});}
  for(const auto& [name,value]:weights)for(int64_t b=0;b<banks;++b){auto v=value[b].clone().set_requires_grad(true);params[b][name]=v;f.leaves.push_back(v);f.labels.push_back({name,b,true});}
  f.metadata=at::zeros({int64_t(rows),l.integer_width()},at::kLong);
  for(size_t r=0;r<rows;++r){const bool use=present[r].item<bool>();const bool update=action==WeightedRecurrentAction::Update;
    Fields output;auto& state=old[r];int64_t last=state.last_time,observed=state.observations;auto stamps=c.basic()?at::empty({0},at::kLong):state.slots.at("conv_times");
    if(update&&(!use||mass[r].item<float>()<=0))for(const auto& field:l.fields())output[field.name]=field.name=="value"?state.value:state.slots.at(field.name);
    else if(!use){if(action==WeightedRecurrentAction::Full)output["G"]=at::zeros({c.width},at::kFloat);
      else output={{"short_memory",at::zeros({c.summary_width},at::kFloat)},{"long_memory",at::zeros({c.summary_width},at::kFloat)},
        {"short_total",at::zeros({1},at::kFloat)},{"long_total",at::zeros({1},at::kFloat)}};
    }else{
      const auto time=times[r].item<int64_t>();Local math(c,params[owners[r].item<int64_t>()],state,time,inputs[r],masses[r]);
      output=update?math.update():(action==WeightedRecurrentAction::Memory?math.memory():math.full());
      if(update){last=time;++observed;if(!c.basic()){auto ids=at::nonzero(stamps>=std::max<int64_t>(0,time-l.horizon())).flatten();stamps=at::cat({stamps.index_select(0,ids),at::tensor({time},at::kLong)});}}
    }
    f.metadata[r][0]=last;f.metadata[r][1]=observed;f.metadata[r][2]=stamps.numel();f.metadata[r][3]=std::max<int64_t>(0,last);
    if(stamps.numel())f.metadata[r].slice(0,4,4+stamps.numel()).copy_(stamps);
    for(auto& [name,value]:output)value=value.flatten();f.fields.push_back(std::move(output));
  }return f;
}
}
