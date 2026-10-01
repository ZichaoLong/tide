#include "bounded_check.h"
#include "peer_transport.h"
#include <tide/lh_full.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>
#include <limits>
#include <set>

namespace accelerator_scale::bounded {
void check_inactive_full(pdg_scale::Config c,const pdg_scale::Topology& topology,at::Device device,Index devices) {
  if(c.memory!="add")return;
  c.steps=1;c.batch=1;at::AutoGradMode grad(true);
  auto reference=fixture(c,topology,false),actual=fixture(c,topology,true);
  reference.graph.compile();configure_model(reference.graph,reference.model);
  auto ids=at::zeros({1,1},at::TensorOptions().dtype(at::kLong));
  Index unused=-1;
  {
    at::NoGradGuard guard;
    auto probe=oracle(reference,c,topology,ids);std::set<Index> selected;
    for(const auto& e:probe.result.trace)if(e.active)selected.insert(e.node);
    const double maximum=c.runtime.dtype==at::kHalf?65504.:std::numeric_limits<float>::max();
    for(const auto& e:probe.result.trace) {
      auto& weights=reference.model.nodes[e.node];
      if(selected.count(e.node)||!weights.extra.count("lh_norm_weight"))continue;
      auto& norm=weights.extra.at("lh_norm_weight");auto saved=norm.clone();norm.fill_(maximum*.75);
      const bool overflow=!at::isfinite(lh_full_fresh(weights,e.proposal)).all().item<bool>();
      norm.copy_(saved);
      if(overflow){unused=e.node;break;}
    }
    if(unused<0)throw std::runtime_error("inactive Full guard fixture did not exercise unused overflow");
    reference.model.nodes[unused].extra.at("lh_norm_weight").fill_(maximum*.75);
    actual.model.nodes[unused].extra.at("lh_norm_weight").fill_(maximum*.75);
  }
  auto expected=oracle(reference,c,topology,ids);
  auto placement=place(actual,device,devices,"locality",true);
  placement.scoring={"model","model",at::kFloat};
  std::unique_ptr<PeerTransport> peer;
  if(device.type()==c10::DeviceType::PrivateUse1&&devices>1)
    peer=std::make_unique<PeerTransport>(placement.devices);
  Program program(actual,topology,placement,Limits{1,1,int64_t(4)<<30,true,true});
  std::vector<Tensor> inputs{at::zeros({1,c.width},actual.embedding.options()).set_requires_grad(true)};
  auto window=program.run(ids.to(actual.embedding.device()),inputs);synchronize(placement);
  close(window.logits[0].data,expected.logits[0],c.check_rtol,c.check_atol,"inactive Full output");
  auto value=root(window.logits[0].data,window.logits[0].dependencies);
  auto u=at::ones_like(expected.logits[0])/16.;
  auto ag=export_gradients(value,program.vjp(value,u.to(value.data.options()),false,inputs));
  auto bg=torch::autograd::grad({expected.logits[0]},expected.leaves,{u},false,false,true);
  for(size_t i=0;i<ag.size();++i)close(ag[i],bg[i],c.check_rtol,c.check_atol,"inactive Full VJP "+std::to_string(i));
  std::cout<<"PASS inactive finite Full cannot contaminate output/VJP or None/zero owner="<<unused<<'\n'<<std::flush;
}
} // namespace accelerator_scale::bounded
