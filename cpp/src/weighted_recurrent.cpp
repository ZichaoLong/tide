// 0770593 recurrent-memory: exact raw state, common summaries and dispatch.
#include "weighted_recurrent_detail.h"
#include <limits>

namespace tide::weighted {
using namespace detail;
void RecurrentConfig::validate() const {
  require(basic()||memory_family=="gated_delta"||memory_family=="ssd","invalid recurrent family");
  require(width>0&&summary_width>0&&key_width>0&&value_width>0&&ffn_width>0&&heads>0&&groups>0
          &&heads%groups==0&&conv_size>0&&conv_stride>0,"invalid recurrent dimensions");
  const auto max=std::numeric_limits<Index>::max();
  require(conv_size-1<=(max-1)/conv_stride&&key_width<=max/2&&value_width<=max-2*key_width
    &&heads<=max/(2*key_width+value_width)&&groups<=max/(2*key_width),"recurrent capacity overflow");
  require(value_width<=max/key_width&&matrices()<=max/(value_width*key_width),"recurrent matrix capacity overflow");
  if(!basic())require(conv_width()<=max-heads*value_width&&heads<=max-heads*value_width-conv_width(),"recurrent projection capacity overflow");
  require(basic()?(heads==1&&groups==1&&conv_size==1&&conv_stride==1&&matrix_rates.empty())
                 :(matrix_rates.empty()||matrix_rates.size()==static_cast<size_t>(heads)),"inapplicable recurrent head/clock fields");
  require(memory_family=="ssd"||groups==1,"groups apply only to SSD");
  require(memory_read=="sum"||memory_read=="state","recurrent read must be sum/state");
  for(auto x:{lambda_short,epsilon,qk_epsilon,output_epsilon})require(std::isfinite(x)&&x>0,"invalid recurrent positive scalar");
  for(auto x:{alpha_mem,alpha_ff})require(std::isfinite(x)&&x>=0,"invalid recurrent residual scale");
  for(auto x:matrix_rates)require(std::isfinite(x)&&x>=0,"invalid recurrent matrix rate");
  require(memory_family=="gated_delta"||qk_epsilon==1e-6,"qk epsilon applies only to gated_delta");
  require(!basic()||output_epsilon==1e-5,"output epsilon applies only to gated_delta/ssd");
}
namespace recurrent {
std::array<Tensor,3> effective(const State& s,Index time,const RecurrentConfig& c){
  require(time>=0,"negative recurrent query time");
  auto delta=(at::full({},time,s.slots.at("time").options())-s.slots.at("time")).clamp_min(0).to(s.value.scalar_type());
  auto rates=c.basic()?std::vector<double>{c.lambda_short,0.}:c.matrix_rates;
  if(rates.empty())rates.resize(c.heads,0.);
  auto matrix_decay=at::exp(-at::tensor(rates,s.value.options())*delta);
  auto decay=at::exp(-at::tensor({c.lambda_short,0.},s.value.options())*delta);
  return {matrix_decay.reshape({-1,1,1})*s.slots.at("H"),decay.unsqueeze(-1)*s.slots.at("Z"),decay*s.slots.at("W")};
}
Tensor finish(const NodeWeights& w,const Tensor& u,const Tensor& mix,const RecurrentConfig& c){
  auto x=u+c.alpha_mem*mix,normal=normalize(x,c.epsilon);
  return x+c.alpha_ff*at::matmul(weight(w,"full_program.content.down"),
    at::silu(at::matmul(weight(w,"full_program.content.gate"),normal))*at::matmul(weight(w,"full_program.content.up"),normal));
}
}
Tensor current(const NodeWeights& w,const Tensor& u,const RecurrentConfig& c){
  return at::matmul(weight(w,"kernel.summary"),normalize(u,c.epsilon));
}
std::array<Tensor,4> memory(const NodeWeights& w,const State& s,Index time,const Tensor& u,const RecurrentConfig& c){
  auto state=recurrent::effective(s,time,c);auto values=state[1]/state[2].clamp_min(1).unsqueeze(-1);
  Tensor a=values[0],b=values[1];
  if(c.memory_read=="state"){
    if(c.basic()){values=recurrent::basic_read(w,u,state[0],c);a=values[0];b=values[1];}
    else {
      auto read=c.memory_family=="gated_delta"?recurrent::gdn_read(w,s,time,u,c):recurrent::ssd_read(w,s,time,u,c);
      b=at::matmul(weight(w,"kernel.read_out"),read.flatten());
    }
  }
  return {a,b,state[2][0],state[2][1]};
}
Tensor content(const NodeWeights& w,const State& s,Index time,const Tensor& u,const RecurrentConfig& c){
  auto mix=c.basic()?recurrent::basic_mix(w,u,recurrent::effective(s,time,c)[0],c)
          :(c.memory_family=="gated_delta"?recurrent::gdn_mix(w,s,time,u,c):recurrent::ssd_mix(w,s,time,u,c));
  return recurrent::finish(w,u,mix,c);
}
void validate_content_weights(const NodeWeights& w,const RecurrentConfig& c){
  for(const std::string name:{"gate","up"})shape(w,"full_program.content."+name,{c.ffn_width,c.width});
  shape(w,"full_program.content.down",{c.width,c.ffn_width});
  if(c.basic()){
    shape(w,"full_program.content.query",{c.key_width,c.width});shape(w,"full_program.content.out",{2,c.width,c.value_width});
  }else {
    shape(w,"full_program.content.out",{c.width,c.heads*c.value_width});
    if(c.memory_family=="gated_delta"){
      shape(w,"full_program.content.output_gate",{c.heads*c.value_width,c.width});
      shape(w,"full_program.content.rho",{c.heads,c.value_width});
    }else {shape(w,"full_program.content.direct",{c.heads});shape(w,"full_program.content.rho",{c.heads*c.value_width});}
  }
}
namespace {
class Recurrent final:public MemoryKernel {
 public:
  explicit Recurrent(RecurrentConfig c):c_(std::move(c)){c_.validate();}
  State initial(const NodeWeights& w) const override {
    auto o=w.bias.options();State s{at::zeros_like(w.bias),-1,0,{
      {"time",at::zeros({},o.dtype(at::kLong))},{"H",at::zeros({c_.matrices(),c_.value_width,c_.key_width},o)},
      {"Z",at::zeros({2,c_.summary_width},o)},{"W",at::zeros({2},o)}}};
    if(!c_.basic()){
      s.slots["conv_times"]=at::empty({0},o.dtype(at::kLong));s.slots["conv_values"]=at::empty({0,c_.conv_width()},o);
    }
    return s;
  }
  State step(const NodeWeights& w,const State& old,const ContentView& h,Index time) const override {
    auto mass=h.value[0],u=h.value.slice(0,1);if(mass.item<double>()<=0)return old;
    require(time>old.last_time&&old.observations<std::numeric_limits<Index>::max(),"invalid recurrent write time/count");
    auto values=recurrent::effective(old,time,c_);auto receive=mass.clamp_max(1);Tensor matrix,raw;
    if(c_.basic())matrix=recurrent::basic_update(w,u,receive,values[0],c_);
    else {
      auto result=c_.memory_family=="gated_delta"?recurrent::gdn_update(w,old,time,u,receive,values[0],c_)
                                                  :recurrent::ssd_update(w,old,time,u,receive,values[0],c_);
      matrix=result.first;raw=result.second;
    }
    State next{old.value,time,old.observations+1,{{"time",at::full({},time,old.slots.at("time").options())},
      {"H",matrix},{"Z",values[1]+receive*current(w,u,c_)},{"W",values[2]+receive}}};
    if(!c_.basic()){
      auto cache=conv_write(recurrent::cache(old),time,receive,raw,c_.conv_size,c_.conv_stride);
      next.slots["conv_times"]=cache.times;next.slots["conv_values"]=cache.values;
    }
    return next;
  }
  bool exact_sequence() const override{return true;}
  std::set<std::string> integer_slots() const override{return c_.basic()?std::set<std::string>{"time"}:std::set<std::string>{"time","conv_times"};}
  void validate_policy(const Node& n,Index) const override {
    require(!n.clear&&!n.identity&&n.next_state=="adopt-v1"&&n.state_clock==StateClock{},"weighted recurrence requires proposal retention and logical clock");
  }
  void validate_weights(const NodeWeights& w) const override {
    require(w.bias.numel()==c_.width+1,"weighted recurrent payload width mismatch");
    shape(w,"kernel.summary",{c_.summary_width,c_.width});
    if(c_.basic())recurrent::validate_basic(w,c_);
    else {
      shape(w,"kernel.conv.weight",{c_.conv_size,c_.conv_width()});shape(w,"kernel.conv.bias",{c_.conv_width()});
      if(c_.memory_read=="state")shape(w,"kernel.read_out",{c_.summary_width,c_.heads*c_.value_width});
      if(c_.memory_family=="gated_delta")recurrent::validate_gdn(w,c_);else recurrent::validate_ssd(w,c_);
    }
  }
  void validate_state(const NodeWeights&,const State& s) const override {
    require(s.slots.size()==(c_.basic()?4:6)&&s.slots.count("time")&&s.slots.count("H")&&s.slots.count("Z")&&s.slots.count("W"),"invalid recurrent slot domain");
    const auto& time=s.slots.at("time");
    require(time.dim()==0&&time.scalar_type()==at::kLong&&s.slots.at("H").sizes()==at::IntArrayRef({c_.matrices(),c_.value_width,c_.key_width})
      &&s.slots.at("Z").sizes()==at::IntArrayRef({2,c_.summary_width})&&s.slots.at("W").sizes()==at::IntArrayRef({2}),"invalid recurrent state layout");
    require(time.item<Index>()==std::max<Index>(0,s.last_time)&&(s.slots.at("W")>=0).all().item<bool>()
      &&s.value.count_nonzero().item<Index>()==0&&((s.observations==0)==(s.last_time==-1)),"invalid recurrent timestamp/metadata state");
    if(!c_.basic()){
      require(s.slots.count("conv_times")&&s.slots.count("conv_values"),"missing convolution cache");
      auto b=recurrent::cache(s);validate_cache(b,c_.conv_width(),(c_.conv_size-1)*c_.conv_stride);
      require((b.times<=s.last_time).all().item<bool>()&&b.times.numel()<=s.observations,"invalid recurrent convolution clock/count");
    }
  }
 private:RecurrentConfig c_;
};
}
std::shared_ptr<const StateKernel> make_recurrent(const RecurrentConfig& c){return std::make_shared<Recurrent>(c);}
}
