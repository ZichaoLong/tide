// 0770593 architecture §2–4, allocation §2–3. See docs/weighted-message.md.
#include "weighted_recurrent_detail.h"
#include "weighted_scoring.h"
#include "weighted_aggregate.h"
#include "weighted_mean.h"
#include "weighted_query_batch.h"
#include "weighted_query_vjp.h"
#include "tide/aggregate.h"
#include "tide/full.h"

namespace tide::weighted {
using namespace detail;
namespace {
class Aggregate final:public AggregateKernel {
 public:
  Aggregate(SourceClasses classes,double alpha,double epsilon):classes_(std::move(classes)),alpha_(alpha),epsilon_(epsilon){}
  bool joint_batch() const override{return true;}
  bool batched_autograd() const override{return true;}
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
        auto hidden=at::silu(at::matmul(weight(w,prefix+".a"),normalize(z,epsilon_))+weight(w,prefix+".embedding"));
        z=z+alpha_*at::matmul(weight(w,prefix+".b"),hidden);
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
      require(value.dim()==2&&value.size(1)==d,"invalid source A shape");
      shape(w,prefix+".b",{d,value.size(0)});shape(w,prefix+".embedding",{value.size(0)});
    }
  }
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
 private:Config c_;
};
}
std::shared_ptr<const StateKernel> make_kv(const KVConfig&);
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
