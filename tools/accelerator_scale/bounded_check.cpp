#include "bounded_check.h"
#include "execution.h"
#include "peer_transport.h"
#include "../../cpp/bench/streaming.h"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <iostream>
#include <set>
#include <cmath>

namespace accelerator_scale::bounded {
void close(const Tensor& a,const Tensor& b,double rtol,double atol,const std::string& name) {
  if(a.defined()!=b.defined())throw std::runtime_error("bounded None/zero mismatch: "+name);
  if(!a.defined())return;
  auto x=a.detach().to(at::kCPU).to(at::kFloat),y=b.detach().to(at::kCPU).to(at::kFloat);
  if(x.sizes()!=y.sizes() || !at::isfinite(x).all().item<bool>() || !at::allclose(x,y,rtol,atol))
    throw std::runtime_error("bounded value mismatch: "+name+" max_error="+std::to_string((x-y).abs().max().item<double>())
      +" reference_max="+std::to_string(y.abs().max().item<double>())
      +" normalized_violation="+std::to_string(((x-y).abs()/(atol+rtol*y.abs())).max().item<double>()));
}
pdg_scale::Fixture fixture(pdg_scale::Config c,const pdg_scale::Topology& t,bool candidate) {
  portable_torch::seed_runtime(at::Device(at::kCPU),c.runtime.seed);
  c.emission=candidate?"row":"slot";auto f=pdg_scale::fixture(c,t);
  if(!candidate) {
    std::map<const c10::StorageImpl*,Tensor> owners;
    for(const auto& p:f.owners)owners[p.storage().unsafeGetStorageImpl()]=p;
    for(auto& w:f.model.nodes)for(auto& [name,p]:w.extra) {
      const auto it=owners.find(p.storage().unsafeGetStorageImpl());
      if(it!=owners.end() && !p.is_same(it->second))p=it->second.as_strided(p.sizes(),p.strides(),p.storage_offset());
    }
  }
  Scoring scoring;scoring.dtype=at::kFloat;configure_scoring(f,scoring);return f;
}
Oracle oracle(pdg_scale::Fixture& f,const pdg_scale::Config& c,const pdg_scale::Topology& t,const Tensor& ids) {
  Options options;options.trace=true;options.workers=1;options.packed=false;
  Execution cursor(f.graph,f.model,options,c.batch,Placement{});Oracle out;out.leaves=f.owners;
  for(Index token=0;token<c.steps;++token) {
    auto x=f.embedding.index_select(0,ids[token]);
    auto input=at::zeros_like(x).set_requires_grad(true);out.inputs.push_back(input);out.leaves.push_back(input);x=x+input;
    std::vector<External> external;
    for(Index b=0;b<c.batch;++b)external.push_back({b,0,token,token*(t.layers+1),x[b]});
    auto r=cursor.advance(external,(token+1)*(t.layers+1),(token+1)*(t.layers+1));
    out.result.trace.insert(out.result.trace.end(),r.trace.begin(),r.trace.end());
    out.result.messages.insert(out.result.messages.end(),r.messages.begin(),r.messages.end());
    out.result.outputs.insert(out.result.outputs.end(),r.outputs.begin(),r.outputs.end());
    std::vector<Tensor> h(c.batch,at::zeros({c.width},x.options()));for(const auto& o:r.outputs)h[o.batch]=o.value;
    out.logits.push_back(at::linear(at::stack(h),f.head));
  }
  out.result.continuation=cursor.snapshot();return out;
}
void check_forward(pdg_scale::Config c,const pdg_scale::Topology& topology,at::Device device,Index devices) {
  if(c.width>64 || c.batch>8 || c.steps>6)throw std::invalid_argument("bounded parity requires D<=64 B<=8 T<=6");
  const double rtol=c.check_rtol,atol=c.check_atol;
  for(bool grad:{false,true}) {
    at::AutoGradMode enabled(grad);auto ref=fixture(c,topology,false),actual=fixture(c,topology,true);
    auto placement=place(actual,device,devices,"locality",true);
    placement.scoring={"model","model",at::kFloat};
    auto ids=(at::arange(c.steps).unsqueeze(1)*7+at::arange(c.batch).unsqueeze(0)*3).remainder(c.vocab).to(at::kLong);
    auto expected=oracle(ref,c,topology,ids);
    // Exercise the replay adapter's explicit local-VJP tape at isolated public
    // roots as well as at the complete training loss checked during capture.
    std::unique_ptr<PeerTransport> peer;
    if(!device.is_cpu() && devices>1) {
      std::vector<at::Device> targets;
      for(Index i=0;i<devices;++i)targets.emplace_back(device.type(),device.index()+i);
      peer=std::make_unique<PeerTransport>(targets);
    }
    Program p(actual,topology,placement,Limits{c.steps,c.batch,int64_t(4)<<30,true,true});
    std::vector<Tensor> input;
    for(Index i=0;i<c.steps;++i)input.push_back(at::zeros({c.batch,c.width},actual.embedding.options()).set_requires_grad(true));
    auto window=p.run(ids.to(actual.embedding.device()),input);synchronize(placement);
    auto result=export_result(p,window);
    tide_bench::compare(result,expected.result,true,c.runtime.dtype,std::nullopt,rtol,atol);
    for(Index i=0;i<c.steps;++i)close(window.logits[i].data,expected.logits[i],rtol,atol,"logits");
    std::cout<<"PASS bounded complete observables grad="<<grad<<" memory="<<c.memory<<'\n'<<std::flush;
    if(!grad)continue;
    std::vector<std::pair<Value,Tensor>> roots;
    for(const auto& e:expected.result.trace)if(e.batch==0 && e.active && !e.emitted.empty()) {
      auto it=std::find_if(window.events.begin(),window.events.end(),[&](const auto& v){return v.node==e.node && v.time==e.time;});
      roots.push_back({row(it->fresh,0),e.full});roots.push_back({row(it->emitted.front(),0),e.emitted.front().value});
      roots.push_back({row(it->descriptor,0),e.descriptor});roots.push_back({row(it->control,0),e.control});break;
    }
    if(!expected.result.continuation.pending.empty()) {
      const auto& a=expected.result.continuation.pending.front();
      auto it=std::find_if(window.pending.begin(),window.pending.end(),[&](const auto& m){return m.node==a.node && m.time==a.time && m.kind==a.kind && m.source==a.source && m.position==a.position;});
      roots.push_back({row(it->value,a.batch),a.value});
    }
    const auto& first=*expected.result.continuation.states.begin();const auto b=first.first.first,n=first.first.second;
    const auto& s=window.states[n];roots.push_back({row(s.value,b),first.second.value});
    for(const auto& [name,value]:first.second.slots) {
      auto tensor=(name=="key"?s.key:name=="value"?s.cache_value:s.bias)[b].index_select(0,at::nonzero(s.valid[b]).reshape({-1}));
      auto dep=name=="log_bias"?p.empty_dependencies(tensor.device()):s.cache_dependencies;
      roots.push_back({{tensor,dep[b].unsqueeze(0)},value});
    }
    for(Index i=0;i<c.steps;++i)roots.push_back({root(window.logits[i].data,window.logits[i].dependencies),expected.logits[i]});
    for(size_t i=0;i<roots.size();++i)for(Index direction=0;direction<3;++direction) {
      const auto& [a,b]=roots[i];
      if(!b.requires_grad()) {
        if(a.dependencies.any().item<bool>())throw std::runtime_error("bounded disconnected public root became connected");
        continue;
      }
      // Two output-independent directions and a connected-zero probe. Share
      // exact binary fractions across devices/dtypes. A radial norm-squared
      // loss has cancellation-sensitive upstream arithmetic, so it is not an
      // isolated Jacobian comparison (complete losses are checked separately).
      auto index=at::arange(b.numel(),at::TensorOptions().dtype(at::kLong)).reshape(b.sizes());
      auto cotangent=((index*(direction+1)+Index(i)).remainder(7+4*direction)-(3+2*direction)).to(at::kFloat)/8.;
      if(direction==2)cotangent=at::zeros_like(cotangent);
      cotangent=cotangent.to(b.scalar_type());
      auto ag=p.vjp(a,cotangent.to(a.data.options()),true,input);
      auto bg=torch::autograd::grad({b},expected.leaves,{cotangent.to(b.options())},true,false,true);
      ag=export_gradients(a,std::move(ag));
      for(size_t j=0;j<ag.size();++j)close(ag[j],bg[j],rtol,atol,"root "+std::to_string(i)
        +" direction "+std::to_string(direction)+" leaf "+std::to_string(j));
    }
    std::cout<<"PASS bounded independent isolated VJPs and None/zero roots="<<roots.size()
      <<" directions=2+zero explicit_peer="<<bool(peer)<<'\n'<<std::flush;
  }
}
}  // namespace accelerator_scale::bounded
int main(int argc,char** argv) {
  bool npu=false;int code=0;
  try {
    bool replay=false,training=false,peer_check=false;int64_t devices=1;std::vector<char*> common{argv[0]};
    double atol=-1.,rtol=-1.;
    std::set<std::string> seen;
    for(int i=1;i<argc;++i) {
      const std::string flag=argv[i];
      if(flag=="--check-atol" || flag=="--check-rtol") {
        if(!seen.insert(flag).second || ++i==argc)throw std::invalid_argument("duplicate/missing "+flag);
        const std::string value=argv[i];size_t used=0;const auto x=std::stod(value,&used);
        if(used!=value.size() || !std::isfinite(x) || x<=0)throw std::invalid_argument("positive finite tolerance required");
        if(flag=="--check-atol")atol=x;else rtol=x;continue;
      }
      if(flag!="--replay" && flag!="--training" && flag!="--devices" && flag!="--peer-check"){common.push_back(argv[i]);continue;}
      if(!seen.insert(flag).second || ++i==argc)throw std::invalid_argument("duplicate/missing "+flag);
      const std::string value=argv[i];
      if(value.empty() || value.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("integer required: "+flag);
      auto n=std::stoll(value);
      if(flag=="--devices") {if(n<1 || n>16)throw std::invalid_argument("devices requires1..16");devices=n;}
      else {if(n>1)throw std::invalid_argument("boolean requires0|1");if(flag=="--replay")replay=n;else if(flag=="--training")training=n;else peer_check=n;}
    }
    auto c=pdg_scale::parse(common.size(),common.data());
    if(c.runtime.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(c.runtime.dtype!=at::kHalf && (atol>=0 || rtol>=0))throw std::invalid_argument("custom tolerance requires FP16");
    c.check_atol=c.runtime.dtype==at::kHalf?(atol>=0?atol:1e-3):1e-6;
    c.check_rtol=c.runtime.dtype==at::kHalf?(rtol>=0?rtol:2e-2):1e-5;
    c.runtime.allow_npu_float16=c.runtime.dtype==at::kHalf;
    auto device=portable_torch::resolve_device(c.runtime);npu=!device.is_cpu();
    if(device.type()!=c10::DeviceType::PrivateUse1 && !device.is_cpu())throw std::invalid_argument("bounded consumer supports CPU/NPU");
    if(c.runtime.dtype!=at::kFloat && c.runtime.dtype!=at::kHalf)throw std::invalid_argument("bounded consumer supports FP32/FP16 payload");
    if(replay && device.is_cpu())throw std::invalid_argument("replay requires NPU");
    if(peer_check && (device.is_cpu() || devices!=2 || training))throw std::invalid_argument("peer check requires two NPUs");
    at::set_num_threads(1);at::set_num_interop_threads(1);
    auto topology=pdg_scale::read_topology(c.topology);
    auto contexts=accelerator_scale::initialize_devices(device,devices);
    if(peer_check)accelerator_scale::bounded::check_peer(device,c.runtime.dtype);
    else if(training)accelerator_scale::bounded::check_training(c,topology,device,devices,replay);
    else if(replay)accelerator_scale::bounded::check_replay(c,topology,device,devices);
    else {
      accelerator_scale::bounded::check_forward(c,topology,device,devices);
      accelerator_scale::bounded::check_inactive_full(c,topology,device,devices);
    }
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';code=1;}
  if(npu)try{accelerator_scale::finalize();}catch(const std::exception& e){std::cerr<<e.what()<<'\n';code=1;}
  return code;
}
