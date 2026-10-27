// 0770593 architecture §2–4, allocation §2–3. See docs/weighted-message.md.
#include "weighted_recurrent_detail.h"
#include "weighted_scoring.h"
#include "weighted_aggregate.h"
#include "weighted_mean.h"
#include "weighted_source.h"
#include "weighted_sharing.h"
#include "weighted_query_batch.h"
#include "weighted_query_vjp.h"
#include "weighted_placement.h"
#include "tide/aggregate.h"
#include "tide/full.h"

namespace tide::weighted {
using namespace detail;
// Named linkage preserves installed-function RTTI across independently linked
// native/client shared libraries with the same source/ABI identity.
namespace installed {
class Aggregate final:public AggregateKernel {
 public:
  Aggregate(SourceClasses classes,double alpha,double epsilon):classes_(std::move(classes)),alpha_(alpha),epsilon_(epsilon){}
  bool joint_batch() const override{return true;}
  bool batched_autograd() const override{return true;}
  std::string sharing_key(const NodeWeights& w,const AggregateInput& r) const override {
    return aggregate_key(w,r,classes_,alpha_,epsilon_);
  }
  std::vector<AggregateResult> batch_grad(const NodeWeights& w,const std::vector<AggregateInput>& requests) const override {
    return aggregate_batch_vjp(w,requests,classes_,alpha_,epsilon_);
  }
  std::vector<AggregateResult> batch(const NodeWeights& w,const std::vector<AggregateInput>& requests) const override {
    return aggregate_batch(w,requests,classes_,alpha_,epsilon_);
  }
  AggregateResult step(const NodeWeights& w,const AggregateInput& r) const override {
    std::vector<Tensor> masses,values;
    for(const auto& source:r.sources){
      const auto& x=source.atom.value;auto mass=x[0].clamp_min(0),z=x.slice(0,1);
      if(alpha_){
        auto prefix="aggregate_program.transforms."+std::to_string(classes_.classify(source.atom));
        auto [a,b]=source_matrices(w,prefix);
        auto hidden=at::silu(at::matmul(a,normalize(z,epsilon_))+weight(w,prefix+".embedding"));
        z=z+alpha_*at::matmul(b,hidden);
      }
      masses.push_back(mass);values.push_back(z);
    }
    auto mass=at::stack(masses).sum();
    auto u=mass.item<double>()>0?weighted_mean(at::stack(masses),at::stack(values)):at::zeros_like(values[0]);
    return {at::cat({mass.reshape({1}),u}),{}};
  }
  void validate_weights(const NodeWeights& w,Index) const override {
    // Shapes are checked per class; one class may bind several physical atoms.
    for(const auto& [name,value]:w.extra)if(name.find("aggregate_program.transforms.")==0&&name.substr(name.size()-2)==".a"){
      auto prefix=name.substr(0,name.size()-2);const auto d=w.bias.numel()-1;
      validate_source(w,prefix,d);
    }
  }
  SourceClasses classes()const{return classes_;}
  double alpha()const{return alpha_;}
  double epsilon()const{return epsilon_;}
 private:SourceClasses classes_;double alpha_,epsilon_;
};
template<class Config> class Describe final:public ReadKernel {
 public:
  Describe(Config c,RegionalConfig r):c_(std::move(c)),r_(std::move(r)){}
  bool structured() const override{return true;}
  bool joint_batch() const override{return true;}
  bool batched_autograd() const override{return true;}
  std::vector<TensorFields> fields_batch_grad(const NodeWeights& w,const std::vector<ReadInput>& requests) const override {
    return query::read_vjp(w,requests,c_,r_);
  }
  std::vector<TensorFields> fields_batch(const NodeWeights& w,const std::vector<ReadInput>& requests) const override {
    return query::describe(w,requests,c_,r_);
  }
  FieldLayout field_layout(const NodeWeights&) const override {
    FieldLayout result{{"w",{}},{"mu",{}},{"score",{}}};
    if(r_.regional())result["evidence"]={*r_.region_control_width};
    if(r_.region_score=="att"){result["x"]={*r_.selection_width};result["key"]={*r_.selection_width};}
    return result;
  }
  TensorFields fields_step(const NodeWeights& w,const ReadInput& r) const override {
    auto initial=r.state?State{}:w.kernel->initial(w);const auto& state=r.state?*r.state:initial;
    auto mass=r.content.value[0],u=r.content.value.slice(0,1);
    auto mem=memory(w,state,r.time,u,c_);auto p=current(w,u,c_);
    Groups groups{normalize(u,c_.epsilon),at::cat({mem[0],mem[1]}),at::stack({mass.log1p(),mem[2].log1p(),mem[3].log1p()})};
    auto features=at::cat({groups[0],groups[2].slice(0,0,1),mem[0],mem[1],groups[2].slice(0,1),p*mem[0],p*mem[1]});
    auto received=mass.clamp_max(1);
    Tensor mu;
    if(r_.regional())mu=(r_.gamma_max*received).clamp_max(1);
    else {
      auto gamma=at::full({},r_.fixed_gain.value_or(1),mass.options());
      if(r_.send_gain=="context")gamma=1+(r_.gamma_max-1)*head(w.extra,"read_program.gain",features).sigmoid();
      else if(r_.gain_logit)gamma=1+(r_.gamma_max-1)*weight(w,"read_program.gain_logit").sigmoid();
      mu=(gamma*mass).clamp_max(1);
    }
    TensorFields result{{"w",received},{"mu",mu},{"score",local_score(w,r_,features,groups,c_.epsilon)}};
    if(r_.regional())result["evidence"]=region_evidence(w,r_,features,groups,c_.epsilon);
    if(r_.region_score=="att"){
      result["x"]=at::matmul(weight(w,"read_program.current_weight"),groups[0])+weight(w,"read_program.current_bias");
      result["key"]=memory_key(w,r_,groups,c_.epsilon);
    }
    return result;
  }
  void validate_weights(const NodeWeights& w) const override {
    validate_describe(w,r_,c_.width,c_.summary_width);
  }
  Config configuration()const{return c_;}
  RegionalConfig regional()const{return r_;}
 private:Config c_;RegionalConfig r_;
};
template<class Config> class Full final:public FullKernel {
 public:
  explicit Full(Config c):c_(std::move(c)){}
  bool joint_batch() const override{return true;}
  bool batched_autograd() const override{return true;}
  std::vector<FullResult> batch_grad(const NodeWeights& w,const std::vector<FullInput>& requests,Index slots,const Options& options) const override {
    return query::full_vjp(w,requests,slots,options,c_);
  }
  std::vector<FullResult> batch(const NodeWeights& w,const std::vector<FullInput>& requests,Index slots,const Options& options) const override {
    return query::full(w,requests,slots,options,c_);
  }
  FullResult step(const NodeWeights& w,const FullInput& r,Index slots,const Options& options) const override {
    require(options.mode=="hard","weighted Full uses exact VJP; legacy Emit mode must be hard");
    auto g=content(w,*r.comparison,r.time,r.content.value.slice(0,1),c_);
    auto payload=at::cat({r.control.reshape({1}),g});FullResult result{payload,{}};
    for(Index slot=0;slot<slots;++slot)result.emitted.push_back({slot,payload});
    return result;
  }
  void validate_weights(const NodeWeights& w,Index) const override {
    validate_content_weights(w,c_);
  }
  Config configuration()const{return c_;}
 private:Config c_;
};
}
using installed::Aggregate;
using installed::Describe;
using installed::Full;
std::shared_ptr<const StateKernel> make_kv(const KVConfig&);
std::optional<KVConfig> kv_configuration(const StateKernel&);
std::optional<RecurrentConfig> recurrent_configuration(const StateKernel&);
namespace {
template<class Config> NodeBinding inspect(const NodeWeights& w,const Config& c,const Aggregate& aggregate){
  const auto* read=dynamic_cast<const Describe<Config>*>(w.read_kernel.get());
  const auto* full=dynamic_cast<const Full<Config>*>(w.full_kernel.get());
  require(read&&full&&same_configuration(c,read->configuration())&&same_configuration(c,full->configuration())
    &&aggregate.epsilon()==c.epsilon,"weighted functions bind inconsistent configurations");
  c.validate();auto regional=read->regional();regional.validate();
  return {c,std::move(regional),aggregate.classes(),aggregate.alpha()};
}
}
NodeBinding node_binding(const NodeWeights& w){
  const auto* aggregate=dynamic_cast<const Aggregate*>(w.aggregate_kernel.get());
  require(w.kernel&&aggregate,"node binding requires installed weighted memory and Aggregate");
  if(auto c=kv_configuration(*w.kernel))return inspect(w,*c,*aggregate);
  if(auto c=recurrent_configuration(*w.kernel))return inspect(w,*c,*aggregate);
  throw std::invalid_argument("node binding requires installed weighted memory");
}
bool is_describe(const ReadKernel& kernel){
  return dynamic_cast<const Describe<KVConfig>*>(&kernel)||dynamic_cast<const Describe<RecurrentConfig>*>(&kernel);
}
void configure_node(NodeWeights& w,const KVConfig& c,const RegionalConfig& r,const SourceClasses& classes,double alpha){
  c.validate();r.validate();require(std::isfinite(alpha)&&alpha>=0,"invalid source transform scale");
  w.kernel=make_kv(c);w.aggregate_kernel=std::make_shared<Aggregate>(classes,alpha,c.epsilon);
  w.read_kernel=std::make_shared<Describe<KVConfig>>(c,r);w.full_kernel=std::make_shared<Full<KVConfig>>(c);
}
std::shared_ptr<const StateKernel> make_recurrent(const RecurrentConfig&);
void configure_node(NodeWeights& w,const RecurrentConfig& c,const RegionalConfig& r,const SourceClasses& classes,double alpha){
  c.validate();r.validate();require(std::isfinite(alpha)&&alpha>=0,"invalid source transform scale");
  w.kernel=make_recurrent(c);w.aggregate_kernel=std::make_shared<Aggregate>(classes,alpha,c.epsilon);
  w.read_kernel=std::make_shared<Describe<RecurrentConfig>>(c,r);w.full_kernel=std::make_shared<Full<RecurrentConfig>>(c);
}
}  // namespace tide::weighted
