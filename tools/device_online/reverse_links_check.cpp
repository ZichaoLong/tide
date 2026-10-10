#include "device_backend.h"
#include "reverse_links.h"
#include "content_fixture.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <array>
#include <iostream>
#include <map>

namespace {
using namespace tide;
using namespace tide::device_online;
using Key=std::array<Index,3>;
void require(bool on,const char* message){if(!on)throw std::runtime_error(message);}
void compare(const ReverseTape& t,const ReverseLinks& links) {
  const auto& g=*t.graph;auto events=t.state.metadata.cpu(),fm=t.fiber_meta.cpu();
  auto pm=t.pending.coordinates.cpu(),om=t.outputs.coordinates.cpu(),pv=t.pending.valid.cpu(),ov=t.outputs.valid.cpu();
  const auto count=t.state.count.cpu().item<Index>(),fibers=t.fiber_count.cpu().item<Index>();
  const auto inputs=Index(g.inputs.size()),edges=Index(g.edges.size()),total=links.messages.size(0);
  std::map<Key,Index> by_event;std::vector<Index> stage;
  for(Index i=0;i<count;++i){by_event[{events[i][0].item<Index>(),events[i][1].item<Index>(),events[i][2].item<Index>()}]=i;
    const auto s=events[i][12].item<Index>();if(s==Index(stage.size()))stage.push_back(i);}
  require(links.stages.cpu().item<Index>()==Index(stage.size()),"reverse stage count mismatch");stage.push_back(count);
  auto offsets=links.stage_offsets.cpu();for(size_t i=0;i<stage.size();++i)require(offsets[i].item<Index>()==stage[i],"reverse stage boundary mismatch");
  std::vector<Index> edge_slots(edges),output_slots(g.outputs.size());
  for(size_t j=0;j<g.outgoing_ports.bindings.size();++j){const auto binding=g.outgoing_ports.bindings[j];
    (binding.kind?edge_slots:output_slots)[binding.id]=j;}
  auto expected=at::full({total,4},-1,at::kLong),valid=at::zeros({total},at::kBool);
  std::vector<std::vector<Index>> producers(events.size(0)),consumers(events.size(0)),scales(std::max<Index>(1,links.parameters));
  for(Index i=0;i<total;++i) {
    const bool fiber=i<links.fibers,pending=i>=links.fibers&&i<links.fibers+links.pending;
    const auto local=fiber?i:pending?i-links.fibers:i-links.fibers-links.pending;
    if(!(fiber?i<fibers:pending?pv[local].item<bool>():ov[local].item<bool>()))continue;
    auto row=(fiber?fm:pending?pm:om)[local];const auto b=row[0].item<Index>(),n=row[1].item<Index>(),time=row[2].item<Index>();
    const auto kind=row[3].item<Index>(),id=row[4].item<Index>();Index target=-1,source=-1,aggregate=-1,delivery=-1;
    if(fiber){target=by_event.at({b,n,time});aggregate=kind==0?id:inputs+id;}
    if(fiber||pending) {
      if(kind==1&&time-g.edges[id].delay>=t.cut){source=by_event.at({b,g.edges[id].source,time-g.edges[id].delay});delivery=inputs+edges+edge_slots[id];}
    } else {source=by_event.at({b,n,time});delivery=inputs+edges+output_slots[id];}
    expected[i].copy_(at::tensor({target,source,aggregate,delivery},at::kLong));valid[i].fill_(true);
    if(source>=0)producers[source].push_back(i);if(target>=0)consumers[target].push_back(i);
    if(aggregate>=0)scales[aggregate].push_back(2*i);if(delivery>=0)scales[delivery].push_back(2*i+1);
  }
  require(at::equal(links.messages.cpu(),expected)&&at::equal(links.valid.cpu(),valid),"physical reverse dependency mismatch");
  auto check=[](const Tensor& head,const Tensor& next,const std::vector<std::vector<Index>>& groups){
    auto h=head.cpu(),n=next.cpu();for(size_t i=0;i<groups.size();++i){std::vector<Index> actual;
      for(Index j=h[i].item<Index>();j>=0;j=n[j].item<Index>()){if(actual.size()>=size_t(next.numel()))throw std::runtime_error("reverse list cycle");actual.push_back(j);}
      require(actual==groups[i],"reverse contributor order/identity mismatch");}};
  check(links.producer_head,links.producer_next,producers);check(links.consumer_head,links.consumer_next,consumers);
  check(links.scale_head,links.scale_next,scales);
  auto weights=links.scales.cpu();
  if(inputs+edges)require(at::equal(weights.narrow(0,0,inputs+edges),t.source_scales.cpu()),"Aggregate parameter identity changed");
  if(edges+Index(g.outputs.size()))require(at::equal(weights.narrow(0,inputs+edges,edges+g.outputs.size()),
    t.delivery_scales.cpu().narrow(0,0,edges+g.outputs.size()).view({-1})),"delivery parameter identity changed");
}
void precision(test::Fixture& f,at::ScalarType dtype) {
  test::model_dtype(f.model,dtype);
  for(auto& [_,s]:f.initial.states){s.value=s.value.to(dtype);for(auto& [__,v]:s.slots)v=v.to(dtype);}
  for(auto& x:f.input)x.value=x.value.to(dtype);
}
int windows(at::Device device,int shape,int variant,bool prefill,at::ScalarType dtype) {
  at::NoGradGuard guard;auto f=test::fixture(shape,variant);
  for(size_t n=0;n<f.graph.nodes.size();++n)if(!f.graph.nodes[n].identity){f.graph.nodes[n].full="tanh";f.model.nodes[n].weight.mul_(.125);}
  // Present zero messages keep their physical connections and edge identity.
  if(!f.model.edge_scale.empty())f.model.edge_scale[0].zero_();
  precision(f,dtype);
  f.graph.compile();f.initial.identity=f.graph.identity;ContentLimits limits;limits.prefill=prefill;limits.trace=512;
  ContentFlow flow(f.graph,f.model,f.initial,device,limits);Index previous=f.initial.cut;int cases=0;
  for(Index delta:{2,6,11,11}) {
    const auto stop=f.initial.cut+delta;std::vector<External> input;for(const auto& x:f.input)if(x.time>=previous&&x.time<stop)input.push_back(x);
    flow.advance_device(input,stop);auto tape=flow.reverse_tape();require(tape.cut==previous&&tape.stop==stop,"wrong reverse continuation boundary");
    auto error=at::zeros({1},tape.state.count.options().dtype(at::kInt));DeviceProgram p(device);
    auto links=append_reverse_links(p,tape,error,16*1024*1024);p.finish();portable_torch::synchronize(device);p.run();
    require(error.cpu().item<int>()==0,"valid actual forward dependencies refused");compare(tape,links);previous=stop;++cases;
  }
  return cases;
}
void refusals(at::Device device,at::ScalarType dtype) {
  at::NoGradGuard guard;auto f=test::fixture(0,0);precision(f,dtype);ContentFlow flow(f.graph,f.model,f.initial,device);flow.advance_device(f.input,11);
  auto tape=flow.reverse_tape();auto error=at::zeros({1},tape.state.count.options().dtype(at::kInt));
  // Clone metadata before corrupting it: these probes must not alter the live owner.
  auto malformed=tape;malformed.state.metadata=tape.state.metadata.clone();malformed.state.count=tape.state.count.clone();
  malformed.state.metadata[0][12].fill_(1);DeviceProgram p(device);auto out=append_reverse_links(p,malformed,error,16*1024*1024);p.finish();
  portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==2&&!out.valid.cpu().any().item<bool>(),"malformed reverse stage accepted");
  bool refused=false;try{DeviceProgram small(device);append_reverse_links(small,tape,error,1);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"reverse-link budget refusal missing");
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("reverse-link gate requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;
    const auto device=portable_torch::resolve_device(args);if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("reverse-link gate requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    for(int shape=0;shape<4;++shape)for(int variant=0;variant<2;++variant)for(bool prefill:{false,true})cases+=windows(device,shape,variant,prefill,args.dtype);
    refusals(device,args.dtype);std::cout<<"device-reverse-links: passed windows="<<cases<<" scope=device_message_stage_links_not_graph_training\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
