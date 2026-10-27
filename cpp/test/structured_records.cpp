// Core-5 product/clock interfaces, independent literal snapshot and VJP anchors.
#include "tide/kernel.h"
#include "tide/read.h"
#include "tide/region.h"
#include "tide/source_class.h"
#include "tide/tensor_fields.h"
#include "tide/frontier.h"
#include "tide/stream.h"
#include <torch/csrc/autograd/autograd.h>
#include <stdexcept>

namespace {
using namespace tide;
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
void close(const Tensor& a, const Tensor& b) {
  require(a.defined() == b.defined(), "structured field connectivity");
  if (a.defined()) require(at::allclose(a,b,a.scalar_type()==at::kDouble?1e-8:1e-5,
                                         a.scalar_type()==at::kDouble?1e-10:1e-6), "structured field formula");
}
class TypedState : public StateKernel {
 public:
  State initial(const NodeWeights& w) const override {
    return {at::zeros_like(w.bias), -1, 0, {{"times", at::empty({0}, w.bias.options().dtype(at::kLong))}}};
  }
  State step(const NodeWeights&, const State& old, const ContentView& h, Index time) const override {
    return {.5*old.value+h.value, time, old.observations+1,
            {{"times", at::cat({old.slots.at("times"), at::full({1},time,h.value.options().dtype(at::kLong))})}}};
  }
  std::set<std::string> integer_slots() const override { return {"times"}; }
  bool exact_sequence() const override { return true; }
  void validate_weights(const NodeWeights&) const override {}
  void validate_state(const NodeWeights&, const State& s) const override {
    require(s.slots.size()==1 && s.slots.count("times") && s.slots.at("times").numel()==s.observations, "typed state layout");
  }
};
class FieldsRead : public ReadKernel {
 public:
  bool structured() const override { return true; }
  FieldLayout field_layout(const NodeWeights&) const override { return {{"score",{}},{"memory",{2}},{"evidence",{2}}}; }
  TensorFields fields_step(const NodeWeights& w, const ReadInput& r) const override {
    auto memory = r.state ? r.state->value : w.extra.at("bound_initial");
    return {{"score",(memory*w.read).sum()},{"memory",memory},{"evidence",r.content.value*w.extra.at("gain")}};
  }
  void validate_weights(const NodeWeights&) const override {}
};
class FieldsSelect : public RegionKernel {
 public:
  Selection step(const RegionWeights&, const RegionInput& r) const override {
    Selection result; result.history.last_time=r.time;
    for (const auto& c:r.candidates) {
      require(!c.descriptor.defined(), "structured descriptor was also scalar");
      result.active.insert(c.node);
      result.controls[c.node]=at::sigmoid(c.fields.at("score")+c.fields.at("evidence").sum());
    }
    return result;
  }
  void validate_weights(const RegionWeights&,const RegionLayout&) const override {}
  void validate_history(const History&,const RegionLayout&) const override {}
};
}

void check_structured_records(const at::TensorOptions& options) {
  using namespace tide;
  const Index clock=(Index{1}<<54)+17;
  for (const std::string mode:{"content","old","proposal"}) for (bool packed:{false,true}) {
    Graph g; g.nodes={{0}}; g.nodes[0].memory="typed-state-test-v1"; g.nodes[0].readout="fields-test-v1";
    g.regions={{1,true,true,mode,"fields-test-v1"}};g.inputs={0};g.outputs={0};g.compile();
    auto leaf=[](Tensor x){return x.set_requires_grad(true);};
    auto initial=leaf(at::full({2},.7,options)), x=leaf(at::full({2},.6,options));
    auto bound=leaf(at::full({2},.1,options)), gain=leaf(at::full({},.3,options));
    Model m;m.nodes={{at::zeros({2},options),at::zeros({2,2},options),at::zeros({2},options),at::ones({2},options)}};
    auto& w=m.nodes[0];w.kernel=std::make_shared<TypedState>();w.read_kernel=std::make_shared<FieldsRead>();
    w.extra={{"bound_initial",bound},{"gain",gain}};
    m.regions={RegionWeights{{},std::make_shared<FieldsSelect>()}};
    m.input_scale=m.output_scale={at::ones({},options)};
    Continuation q;q.identity=g.identity;q.cut=clock;q.states[{0,0}]=w.kernel->initial(w);q.states[{0,0}].value=initial;
    Options config;config.packed=packed;
    const std::vector<External> rows{{0,0,0,clock,x},{0,0,1,clock+2,x}};
    auto out=packed?Frontier(g,m,config).run(q,rows,clock+3,clock+3):Streaming(g,m,config).run(q,rows,clock+3,clock+3);
    const auto& fields=out.trace[0].descriptor_fields;
    auto expected=mode=="content"?bound:mode=="old"?initial:.5*initial+x;
    close(fields.at("memory"),expected);close(out.trace[0].comparison,.5*initial+x);
    auto grad=torch::autograd::grad({fields.at("score")},{initial,x,bound,gain},{},true,false,true);
    require(!grad[3].defined(),"unused evidence parameter acquired a zero gradient");
    close(grad[0],mode=="content"?Tensor():at::full_like(initial,mode=="old"?1:.5));
    close(grad[1],mode=="proposal"?at::ones_like(x):Tensor());
    close(grad[2],mode=="content"?at::ones_like(bound):Tensor());
    const auto& stamps=out.continuation.states.at({0,0}).slots.at("times");
    require(stamps.scalar_type()==at::kLong && stamps[0].item<Index>()==clock && stamps[1].item<Index>()==clock+2,"clock precision lost");
    auto storage=pack_fields({{"memory",fields.at("memory")},{"evidence",fields.at("evidence")},{"times",stamps}});
    auto restored=storage.decode();close(restored.at("memory"),expected);
    require(at::equal(restored.at("times"),stamps),"typed codec lost metadata");
    auto local=torch::autograd::grad({restored.at("memory").sum()},{gain},{},true,false,true);
    require(!local[0].defined(),"field packing connected unused parameter");
    storage.roots.reset();require(!storage.decode().at("memory").requires_grad(),"decoded checkpoint kept old graph");
    auto tail=Streaming(g,m,config).run(out.continuation,{},clock+4,clock+4);
    require(at::equal(tail.continuation.states.at({0,0}).slots.at("times"),stamps),"idle continuation changed metadata");
  }
  Graph g;g.nodes={{0},{1}};g.edges={{0,1,1},{0,1,1}};g.regions={{1},{1}};g.inputs={1};g.outputs={1};g.compile();
  SourceClasses classes{{0,0},{0}};classes.validate(g);
  std::vector<Atom> atoms{{0,1,1,1,0,0,at::ones({2},options)},{0,1,1,1,1,0,at::ones({2},options)},
                          {0,1,1,0,0,0,at::ones({2},options)}};
  auto total=at::zeros({2},options);
  for(const auto& a:atoms){require(classes.classify(a)==0,"source classification");total=total+3*a.value;}
  close(total,at::full({2},9,options));
}
