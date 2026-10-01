#include "flow_config.h"
#include "peer_transport.h"
#include <tide/lh_full.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>
#include <limits>

namespace accelerator_scale::flows {
// A finite unused parameter must not poison an independent executed branch.
// The first Add region receives identical inputs, so stable ties select input0.
void check_inactive_full(const Config& c,const Topology& topology,at::Device device) {
  if(c.model.memory!="add" || topology.body.inputs.size()<2 || topology.body.regions[0].budget!=1)return;
  at::AutoGradMode grad(true);auto cfg=c;cfg.model.steps=1;cfg.model.grad=true;
  auto r=seeded(cfg,topology,false),a=seeded(cfg,topology,true);
  const auto unused=topology.body.inputs[1];const auto maximum=c.model.runtime.dtype==at::kHalf?65504.:double(std::numeric_limits<float>::max());
  {at::NoGradGuard guard;for(auto* f:{&r,&a}) {
    f->values.model.nodes[unused].extra.at("lh_norm_weight").fill_(maximum*.75);
    f->values.embedding[0].zero_();f->values.embedding[0][0].fill_(.5);
  }}
  auto rc=cfg;rc.devices=1;auto cpu=placed(r,rc,at::Device(at::kCPU));auto p=placed(a,cfg,device);
  auto ids=(at::arange(c.model.batch)*3).remainder(c.model.vocab).to(at::kLong).unsqueeze(0);
  Options opts;opts.trace=true;opts.packed=false;opts.workers=1;
  auto expected=host_window(cfg.model,r,cpu,"streaming",opts,ids);
  bool witnessed=false;
  for(const auto& e:expected.result.trace)if(e.node==unused && e.batch==0) {
    if(e.active)throw std::runtime_error("inactive Full fixture unexpectedly selected poisoned owner");
    at::NoGradGuard no_grad;
    witnessed=!at::isfinite(lh_full_fresh(r.values.model.nodes[unused],e.proposal)).all().item<bool>();
  }
  if(!witnessed)throw std::runtime_error("inactive Full fixture did not exercise overflow");
  std::unique_ptr<PeerTransport> peer;
  if(device.type()==c10::DeviceType::PrivateUse1 && c.devices>1)peer=std::make_unique<PeerTransport>(p.devices);
  bounded::Limits limits{1,c.model.batch,int64_t(16)<<30,true,true};
  bounded::Program program(a.values,schedule(a,limits,true),p,limits);auto window=program.run(ids.to(a.values.embedding.device()));
  const auto rtol=c.model.runtime.dtype==at::kHalf?.02:1e-5,atol=c.model.runtime.dtype==at::kHalf?.004:1e-6;
  if(!at::isfinite(window.logits[0].data).all().item<bool>() || !at::allclose(window.logits[0].data.to(at::kCPU),expected.logits[0],rtol,atol))
    throw std::runtime_error("inactive Full contaminated executed output");
  auto root=bounded::root(window.logits[0].data,window.logits[0].dependencies);
  auto ag=bounded::export_gradients(root,program.vjp(root,at::ones_like(root.data)/16.,false));
  auto bg=torch::autograd::grad({expected.logits[0]},r.values.owners,{at::ones_like(expected.logits[0])/16.},false,false,true);
  for(size_t i=0;i<bg.size();++i) {
    if(bg[i].defined()!=ag[i].defined())throw std::runtime_error("inactive Full changed None/zero owner");
    if(!bg[i].defined())continue;
    auto x=ag[i].detach().to(at::kCPU).to(at::kDouble),y=bg[i].detach().to(at::kDouble);
    if(!at::isfinite(x).all().item<bool>() || !at::allclose(x,y,c.model.runtime.dtype==at::kHalf?.02:1e-5,c.model.runtime.dtype==at::kHalf?.004:1e-6))
      throw std::runtime_error("inactive Full contaminated executed VJP");
  }
  std::cout<<"PASS inactive finite Full weights cannot contaminate active values/VJPs\n"<<std::flush;
}
}  // namespace accelerator_scale::flows
