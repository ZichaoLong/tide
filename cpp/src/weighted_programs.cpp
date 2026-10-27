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
    return {{"w",{}},{"mu",{}},{"score",{}},{"evidence",{r_.region_control_width}}};
  }
  TensorFields fields_step(const NodeWeights& w,const ReadInput& r) const override {
    auto initial=r.state?State{}:w.kernel->initial(w);const auto& state=r.state?*r.state:initial;
    auto mass=r.content.value[0],u=r.content.value.slice(0,1);
    auto mem=memory(w,state,r.time,u,c_);auto p=current(w,u,c_);
    auto features=at::cat({normalize(u,c_.epsilon),mass.log1p().reshape({1}),mem[0],mem[1],
                          mem[2].log1p().reshape({1}),mem[3].log1p().reshape({1}),p*mem[0],p*mem[1]});
    auto evidence=r_.region_control_mode=="context"
      ?normalize(at::matmul(weight(w,"read_program.evidence_weight"),features)+weight(w,"read_program.evidence_bias"),c_.epsilon)
      :at::zeros({r_.region_control_width},features.options());
    auto received=mass.clamp_max(1);
    return {{"w",received},{"mu",(r_.gamma_max*received).clamp_max(1)},
            {"score",head(w.extra,"read_program.score",features)},{"evidence",evidence}};
  }
  void validate_weights(const NodeWeights& w) const override {
    const auto f=c_.width+4*c_.summary_width+3;
    shape(w,"read_program.score.matrix",{r_.hidden_width,f});shape(w,"read_program.score.hidden_bias",{r_.hidden_width});
    shape(w,"read_program.score.vector",{r_.hidden_width});shape(w,"read_program.score.bias",{});
    if(r_.region_control_mode=="context"){
      shape(w,"read_program.evidence_weight",{r_.region_control_width,f});shape(w,"read_program.evidence_bias",{r_.region_control_width});
    }
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
class RegionalTotal final:public RegionKernel {
 public:
  explicit RegionalTotal(RegionalConfig c):c_(std::move(c)){c_.validate();}
  Selection step(const RegionWeights& w,const RegionInput& r) const override {
    std::vector<Tensor> received,caps,scores,evidence;
    for(const auto& d:r.candidates){
      received.push_back(d.fields.at("w"));caps.push_back(d.fields.at("mu"));
      scores.push_back(d.fields.at("score"));evidence.push_back(d.fields.at("evidence"));
    }
    auto recv=at::stack(received),mu=at::stack(caps),sum=recv.sum(),mass=mu.sum();
    auto low=sum.clamp_max(1),high=mass.clamp_max(c_.k);
    auto fraction=at::full({},c_.total_fraction,sum.options());
    if(c_.region_control_mode=="context"){
      auto z=(recv.unsqueeze(-1)*at::stack(evidence)).sum(0)/sum.clamp_min(1);
      auto features=at::cat({z,sum.log1p().reshape({1}),mass.log1p().reshape({1})});
      fraction=head(w.extra,"head",features).sigmoid();
    }
    auto total=low+fraction*(high-low);
    auto allocation=allocate(at::stack(scores),mu,total,total,at::full({},c_.allocation_temperature,sum.options()),c_.regularizer,c_.ent_alpha);
    Selection result;result.history.last_time=r.time;
    for(size_t i=0;i<r.candidates.size();++i){
      auto node=r.candidates[i].node;auto c=allocation.controls[i];result.controls[node]=c;
      if(c.item<double>()>0)result.active.insert(node);
    }
    return result;
  }
  void validate_weights(const RegionWeights& w,const RegionLayout& layout) const override {
    require(c_.k<=static_cast<Index>(layout.members.size())&&layout.spec.budget==static_cast<Index>(layout.members.size())
            &&layout.spec.observe_all,"weighted region requires K=size, k<=K and observe-all");
    if(c_.region_control_mode=="context"){
      require(w.extra.at("head.matrix").sizes()==at::IntArrayRef({c_.hidden_width,c_.region_control_width+2})
              &&w.extra.at("head.hidden_bias").sizes()==at::IntArrayRef({c_.hidden_width})
              &&w.extra.at("head.vector").sizes()==at::IntArrayRef({c_.hidden_width})&&w.extra.at("head.bias").dim()==0,
              "invalid weighted region head shape");
    }
  }
  void validate_history(const History& h,const RegionLayout&) const override {
    require(h.scalars.empty()&&h.node_maps.empty()&&h.tensors.empty(),"weighted history is a singleton");
  }
 private:RegionalConfig c_;
};
}
void RegionalConfig::validate() const {
  Regularizer rule(regularizer,ent_alpha);
  require(hidden_width>0&&region_control_width>0&&k>0&&std::isfinite(gamma_max)&&gamma_max>=1
          &&std::isfinite(allocation_temperature)&&allocation_temperature>0,"invalid weighted region config");
  require(region_control_mode=="context"||region_control_mode=="fixed","invalid weighted region control mode");
  require(std::isfinite(total_fraction)&&total_fraction>=0&&total_fraction<=1,"invalid weighted fixed total fraction");
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
void configure_region(RegionWeights& w,const RegionalConfig& r){w.kernel=std::make_shared<RegionalTotal>(r);}
}  // namespace tide::weighted
