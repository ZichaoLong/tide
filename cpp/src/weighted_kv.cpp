// KV §1–4, original-time sequence state and exact zero-receive preservation.
#include "weighted_detail.h"
#include <limits>

namespace tide::weighted {
using namespace detail;
Tensor normalize(const Tensor& x,double eps){return x/at::sqrt(eps+x.square().mean(-1,true));}
void KVConfig::validate() const {
  require(width>0&&summary_width>0&&adapter_rank>0&&heads>0&&width%heads==0&&ffn_width>0&&window>=0,"invalid weighted KV sizes");
  for(auto x:{beta_short,epsilon,position_scale})require(std::isfinite(x)&&x>0,"invalid KV positive scalar");
  for(auto x:{alpha_att,alpha_ff})require(std::isfinite(x)&&x>=0,"invalid KV residual scale");
  require(std::isfinite(position_origin)&&(memory_read=="sum"||memory_read=="qry"),"invalid KV position/read mode");
  require(full_frequencies.empty()||full_frequencies.size()==static_cast<size_t>(heads),"invalid full frequency head count");
  require(distance_slopes.empty()||distance_slopes.size()==static_cast<size_t>(heads),"invalid distance head count");
  for(const auto& row:full_frequencies){
    require(2*row.size()<=static_cast<size_t>(width/heads),"invalid full rotary width");
    for(auto x:row)require(std::isfinite(x),"nonfinite rotary frequency");
  }
  require(2*read_frequencies.size()<=static_cast<size_t>(summary_width)&&
          (memory_read=="qry"||read_frequencies.empty()),"invalid read rotary width/mode");
  for(auto x:read_frequencies)require(std::isfinite(x),"nonfinite read rotary frequency");
  for(auto x:distance_slopes)require(std::isfinite(x)&&x>=0,"invalid distance slope");
}
Tensor current(const NodeWeights& w,const Tensor& u,const KVConfig& c){
  return at::matmul(normalize(u,c.epsilon),weight(w,"kernel.summary").t());
}
std::array<Tensor,4> memory(const NodeWeights& w,const State& s,Index time,const Tensor& u,const KVConfig& c){
  auto [mask,ages]=visible(s,time,c);auto age=ages.index({mask}).to(u.scalar_type());
  auto writes=s.slots.at("weights").index({mask});
  std::array<Tensor,2> weights{writes*at::exp(-c.beta_short*age),writes};
  std::array<Tensor,4> result;Tensor logits,values;
  if(c.memory_read=="qry"){
    auto query=at::matmul(weight(w,"kernel.read_q"),normalize(u,c.epsilon));
    auto keys=at::matmul(s.slots.at("keys").index({mask}),weight(w,"kernel.read_k").t());
    keys=rotate(keys,-age/c.position_scale,c.read_frequencies);
    logits=at::matmul(keys,query)/std::sqrt(c.summary_width);
    values=at::matmul(s.slots.at("values").index({mask}),weight(w,"kernel.read_v").t());
  }
  for(int a=0;a<2;++a){
    result[a+2]=weights[a].sum();
    result[a]=c.memory_read=="sum"?(weights[a].unsqueeze(-1)*s.slots.at("summaries").index({mask})).sum(0)/result[a+2].clamp_min(1)
                                 :weighted_read(logits,values,weights[a],result[a+2]);
  }
  return result;
}
Tensor content(const NodeWeights& w,const State& s,Index time,const Tensor& u,const KVConfig& c){
  auto [mask,age]=visible(s,time,c);
  auto weights=s.slots.at("weights").index({mask}),total=weights.sum();
  auto keys=s.slots.at("keys").index({mask}).reshape({-1,c.heads,c.width/c.heads});
  auto values=s.slots.at("values").index({mask}).reshape_as(keys);
  auto queries=at::matmul(weight(w,"full_program.content.query"),normalize(u,c.epsilon)).reshape({c.heads,-1});
  auto displacement=-age.index({mask}).to(u.scalar_type())/c.position_scale;
  std::vector<Tensor> reads;const std::vector<double> empty;
  for(Index h=0;h<c.heads;++h){
    auto key=rotate(keys.select(1,h),displacement,c.full_frequencies.empty()?empty:c.full_frequencies[h]);
    auto logits=at::matmul(key,queries[h])/std::sqrt(c.width/c.heads);
    if(!c.distance_slopes.empty())logits=logits+c.distance_slopes[h]*displacement;
    reads.push_back(weighted_read(logits,values.select(1,h),weights,total));
  }
  auto x=u+c.alpha_att*at::matmul(weight(w,"full_program.content.out"),at::cat(reads));
  auto normal=normalize(x,c.epsilon);
  return x+c.alpha_ff*at::matmul(weight(w,"full_program.content.down"),
          at::silu(at::matmul(weight(w,"full_program.content.gate"),normal))*at::matmul(weight(w,"full_program.content.up"),normal));
}
class KV final:public StateKernel {
 public:
  explicit KV(KVConfig c):c_(std::move(c)){c_.validate();}
  State initial(const NodeWeights& w) const override {
    const auto o=w.bias.options();
    return {at::zeros_like(w.bias),-1,0,{{"times",at::empty({0},o.dtype(at::kLong))},
             {"keys",at::empty({0,c_.width},o)},{"values",at::empty({0,c_.width},o)},
             {"summaries",at::empty({0,c_.summary_width},o)},{"weights",at::empty({0},o)}}};
  }
  State step(const NodeWeights& w,const State& old,const ContentView& h,Index time) const override {
    auto mass=h.value[0],u=h.value.slice(0,1);
    if(mass.item<double>()<=0)return old;
    require(time>old.last_time&&old.observations<std::numeric_limits<Index>::max(),"invalid KV write time/count");
    auto [mask,age]=visible(old,time,c_);
    auto kv=at::matmul(weight(w,"kernel.base")+at::matmul(weight(w,"kernel.adapter_u"),weight(w,"kernel.adapter_v")),normalize(u,c_.epsilon));
    std::map<std::string,Tensor> entries{{"keys",kv.slice(0,0,c_.width)},{"values",kv.slice(0,c_.width)},
      {"summaries",current(w,u,c_)},{"weights",mass.clamp_max(1)},
      {"times",at::full({},time,h.value.options().dtype(at::kLong))}};
    State next{old.value,time,old.observations+1};
    for(const auto& [name,value]:entries)next.slots[name]=at::cat({old.slots.at(name).index({mask}),value.unsqueeze(0)});
    return next;
  }
  bool exact_sequence() const override{return true;}
  std::set<std::string> integer_slots() const override{return {"times"};}
  void validate_policy(const Node& n,Index) const override {
    require(!n.clear&&!n.identity&&n.next_state=="adopt-v1"&&n.state_clock==StateClock{},
            "weighted KV requires proposal retention and the original logical clock");
  }
  void validate_weights(const NodeWeights& w) const override {
    require(w.bias.numel()==c_.width+1,"weighted payload width mismatch");
    shape(w,"kernel.base",{2*c_.width,c_.width});shape(w,"kernel.adapter_u",{2*c_.width,c_.adapter_rank});
    shape(w,"kernel.adapter_v",{c_.adapter_rank,c_.width});shape(w,"kernel.summary",{c_.summary_width,c_.width});
    if(c_.memory_read=="qry")for(const auto& name:{"kernel.read_q","kernel.read_k","kernel.read_v"})shape(w,name,{c_.summary_width,c_.width});
  }
  void validate_state(const NodeWeights&,const State& s) const override {
    require(s.slots.size()==5&&s.slots.count("times")&&s.slots.count("keys")&&s.slots.count("values")
            &&s.slots.count("summaries")&&s.slots.count("weights"),"invalid KV slot domain");
    const auto& times=s.slots.at("times");auto n=times.numel();
    require(times.scalar_type()==at::kLong&&times.dim()==1&&s.slots.at("keys").sizes()==at::IntArrayRef({n,c_.width})
            &&s.slots.at("values").sizes()==at::IntArrayRef({n,c_.width})
            &&s.slots.at("summaries").sizes()==at::IntArrayRef({n,c_.summary_width})
            &&s.slots.at("weights").sizes()==at::IntArrayRef({n}),"invalid KV slot shape/dtype");
    require((times>=0).all().item<bool>()&&(times<=s.last_time).all().item<bool>()
            &&(n<2||(times.slice(0,1)>times.slice(0,0,n-1)).all().item<bool>())
            &&s.observations>=n&&(!c_.window||n<=c_.window)&&s.value.count_nonzero().item<Index>()==0,
            "invalid KV timestamp/metadata state");
    require((s.slots.at("weights")>=0).all().item<bool>()&&(s.slots.at("weights")<=1).all().item<bool>(),"invalid KV write weights");
  }
 private:KVConfig c_;
};
std::shared_ptr<const StateKernel> make_kv(const KVConfig& c){return std::make_shared<KV>(c);}
void validate_transport(const Graph& g,const Model& m){
  std::vector<bool> weighted;
  for(const auto& w:m.nodes)weighted.push_back(dynamic_cast<const KV*>(w.kernel.get())!=nullptr);
  auto unit=[](const Tensor& scale){
    require(!scale.requires_grad()&&at::equal(scale,at::ones_like(scale)),"weighted payload transport requires fixed unit scales");
  };
  for(size_t p=0;p<g.inputs.size();++p)if(weighted[g.inputs[p]])unit(m.input_scale[p]);
  for(size_t p=0;p<g.outputs.size();++p)if(weighted[g.outputs[p]])unit(m.output_scale[p]);
  for(size_t e=0;e<g.edges.size();++e){
    if(weighted[g.edges[e].source])unit(m.edge_scale[e]);
    if(weighted[g.edges[e].target])unit(m.agg_scale[e]);
  }
}
}  // namespace tide::weighted
