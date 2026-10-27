// 0770593 architecture §2–4, allocation §2–3. See docs/weighted-message.md.
#include "weighted_recurrent_detail.h"
#include "weighted_regularizers.h"
#include "tide/aggregate.h"
#include "tide/full.h"

namespace tide::weighted {
using namespace detail;
namespace {
class Aggregate final:public AggregateKernel {
 public:
  Aggregate(SourceClasses classes,double alpha,double epsilon):classes_(std::move(classes)),alpha_(alpha),epsilon_(epsilon){}
  AggregateResult step(const NodeWeights& w,const AggregateInput& r) const override {
    std::vector<Tensor> masses,terms;
    for(const auto& source:r.sources){
      const auto& x=source.atom.value;auto mass=x[0].clamp_min(0),z=x.slice(0,1);
      if(alpha_){
        auto prefix="aggregate_program.transforms."+std::to_string(classes_.classify(source.atom));
        auto hidden=at::silu(at::matmul(weight(w,prefix+".a"),normalize(z,epsilon_))+weight(w,prefix+".embedding"));
        z=z+alpha_*at::matmul(weight(w,prefix+".b"),hidden);
      }
      masses.push_back(mass);terms.push_back(mass*z);
    }
    auto mass=at::stack(masses).sum();
    auto u=mass.item<double>()>0?at::stack(terms).sum(0)/mass:at::zeros_like(terms[0]);
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
  FieldLayout field_layout(const NodeWeights&) const override {
    FieldLayout result{{"w",{}},{"mu",{}},{"score",{}}};
    if(r_.regional())result["evidence"]={*r_.region_control_width};return result;
  }
  TensorFields fields_step(const NodeWeights& w,const ReadInput& r) const override {
    auto initial=r.state?State{}:w.kernel->initial(w);const auto& state=r.state?*r.state:initial;
    auto mass=r.content.value[0],u=r.content.value.slice(0,1);
    auto mem=memory(w,state,r.time,u,c_);auto p=current(w,u,c_);
    auto features=at::cat({normalize(u,c_.epsilon),mass.log1p().reshape({1}),mem[0],mem[1],
                          mem[2].log1p().reshape({1}),mem[3].log1p().reshape({1}),p*mem[0],p*mem[1]});
    auto received=mass.clamp_max(1);
    Tensor mu;
    if(r_.regional())mu=(r_.gamma_max*received).clamp_max(1);
    else {
      auto gamma=at::full({},r_.fixed_gain.value_or(1),mass.options());
      if(r_.send_gain=="context")gamma=1+(r_.gamma_max-1)*head(w.extra,"read_program.gain",features).sigmoid();
      else if(r_.gain_logit)gamma=1+(r_.gamma_max-1)*weight(w,"read_program.gain_logit").sigmoid();
      mu=(gamma*mass).clamp_max(1);
    }
    TensorFields result{{"w",received},{"mu",mu},{"score",head(w.extra,"read_program.score",features)}};
    if(r_.regional())result["evidence"]=r_.region_control_mode=="context"
      ?normalize(at::matmul(weight(w,"read_program.evidence_weight"),features)+weight(w,"read_program.evidence_bias"),c_.epsilon)
      :at::zeros({*r_.region_control_width},features.options());
    return result;
  }
  void validate_weights(const NodeWeights& w) const override {
    const auto f=c_.width+4*c_.summary_width+3;
    shape(w,"read_program.score.matrix",{r_.hidden_width,f});shape(w,"read_program.score.hidden_bias",{r_.hidden_width});
    shape(w,"read_program.score.vector",{r_.hidden_width});shape(w,"read_program.score.bias",{});
    if(r_.region_control_mode=="context"){
      shape(w,"read_program.evidence_weight",{*r_.region_control_width,f});shape(w,"read_program.evidence_bias",{*r_.region_control_width});
    }
    if(r_.send_gain=="context"){
      shape(w,"read_program.gain.matrix",{r_.hidden_width,f});shape(w,"read_program.gain.hidden_bias",{r_.hidden_width});
      shape(w,"read_program.gain.vector",{r_.hidden_width});shape(w,"read_program.gain.bias",{});
    } else if(r_.gain_logit)shape(w,"read_program.gain_logit",{});
  }
 private:Config c_;RegionalConfig r_;
};
template<class Config> class Full final:public FullKernel {
 public:
  explicit Full(Config c):c_(std::move(c)){}
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
