#include "training_test.h"
#include "../../cpp/bench/streaming.h"
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <tuple>

namespace tide::device_online::test {
void train_forward_compare(const Result& actual,const Result& expected,const Graph& graph,bool conditioned_controls) {
  if(!conditioned_controls){tide_bench::compare(actual,expected,true,at::kFloat);return;}
  train_require(actual.trace.size()==expected.trace.size(),"conditioned control trace count mismatch");
  // A comparison-only copy. Candidate storage and the candidate's next advance
  // never receive any reference values. Only already-checked controls below
  // are removed from the generic componentwise comparison.
  auto checked=actual;
  std::map<std::tuple<Index,Index,Index>,std::vector<size_t>> frames;
  for(size_t i=0;i<actual.trace.size();++i) {
    const auto& a=actual.trace[i];const auto& b=expected.trace[i];
    train_require(std::tie(a.batch,a.node,a.time)==std::tie(b.batch,b.node,b.time),"conditioned control event mismatch");
    train_require(a.node>=0&&size_t(a.node)<graph.nodes.size(),"conditioned control node outside graph");
    for(const auto& pair:{std::make_pair(a.descriptor,b.descriptor),std::make_pair(a.control,b.control)})
      train_require(pair.first.defined()&&pair.second.defined()&&pair.first.dim()==0&&pair.second.dim()==0
        &&pair.first.scalar_type()==at::kFloat&&pair.second.scalar_type()==at::kFloat,
        "conditioned control requires scalar FP32 scores and probabilities");
    frames[{a.batch,graph.nodes[a.node].region,a.time}].push_back(i);
  }
  int conditioned=0;double maximum=0;
  for(const auto& [_,indices]:frames) {
    std::vector<Tensor> sa,sb,ca,cb;
    for(auto i:indices) {
      sa.push_back(actual.trace[i].descriptor);sb.push_back(expected.trace[i].descriptor);
      ca.push_back(actual.trace[i].control);cb.push_back(expected.trace[i].control);
    }
    const auto a=at::stack(sa).to(at::kDouble),b=at::stack(sb).to(at::kDouble);
    const auto u=at::stack(ca).to(at::kDouble),v=at::stack(cb).to(at::kDouble);
    if(at::allclose(u,v,1e-5,1e-6))continue;
    train_require(at::isfinite(a).all().item<bool>()&&at::isfinite(b).all().item<bool>()
      &&at::allclose(a,b,1e-5,1e-6),"conditioned control scores fail original tolerance");
    const auto pa=at::softmax(a,0),pb=at::softmax(b,0);
    train_require(at::allclose(u,pa,1e-5,1e-6)&&at::allclose(v,pb,1e-5,1e-6),
      "control is not complete-candidate softmax of its own scores");
    // Softmax is invariant to a common shift. Its Jacobian row has L1 norm
    // 2*p*(1-p)<=1/2. Centering score differences therefore gives a global
    // bound range(delta)/4, plus each implementation's original local tolerance.
    const auto delta=a-b;
    const auto propagated=(delta.max()-delta.min()).item<double>()*.25;
    const auto bound=propagated+2e-6+1e-5*(pa.abs()+pb.abs());
    train_require((u-v).abs().le(bound).all().item<bool>(),"control exceeds propagated score-error bound");
    maximum=std::max(maximum,(u-v).abs().max().item<double>());++conditioned;
    for(auto i:indices)checked.trace[i].control=expected.trace[i].control;
  }
  tide_bench::compare(checked,expected,true,at::kFloat);
  if(conditioned)std::cout<<"training-controls policy=conditioned strict_mismatch_frames="<<conditioned
    <<" maximum_abs="<<maximum<<" scores_and_other_tensors=strict routes=exact\n";
}
void train_control_checks() {
  at::NoGradGuard guard;Graph g;g.nodes={{0},{0}};
  Result expected;expected.continuation.identity="control-numerics-fixture";
  expected.trace.resize(2);
  auto scores=at::tensor({20.f,22.67f});auto probabilities=at::softmax(scores,0);
  for(Index i=0;i<2;++i) {
    auto& event=expected.trace[i];event.batch=0;event.node=i;event.time=0;event.active=i==1;
    event.descriptor=scores[i];event.control=probabilities[i];
  }
  auto actual=expected;auto perturbed=scores.clone();perturbed[0].add_(.00005f);
  auto changed=at::softmax(perturbed,0);
  for(Index i=0;i<2;++i){actual.trace[i].descriptor=perturbed[i];actual.trace[i].control=changed[i];}
  train_reject([&]{train_forward_compare(actual,expected,g,false);},"strict control comparison did not catch sensitive example");
  train_forward_compare(actual,expected,g,true);
  auto wrong=actual;wrong.trace[0].control=at::full({},.2f);
  train_reject([&]{train_forward_compare(wrong,expected,g,true);},"wrong softmax accepted");
  wrong=actual;wrong.trace[0].active=!wrong.trace[0].active;
  train_reject([&]{train_forward_compare(wrong,expected,g,true);},"route mismatch hidden by control policy");
  wrong=actual;wrong.trace[0].descriptor=at::full({},21.f);
  train_reject([&]{train_forward_compare(wrong,expected,g,true);},"score outside original tolerance accepted");
  wrong=actual;wrong.trace.pop_back();
  train_reject([&]{train_forward_compare(wrong,expected,g,true);},"missing candidate accepted");
  std::cout<<"training-control-comparison: passed strict_failure_preserved=true softmax_score_route_domain_guards=true\n";
}
} // namespace tide::device_online::test
