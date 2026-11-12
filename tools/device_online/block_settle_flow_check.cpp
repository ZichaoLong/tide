#include "block_settle_formula.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <iostream>
#include <functional>
namespace {
using namespace tide::device_online;
using namespace tide::device_online::block_flow_check;
using at::Tensor;
void require(bool x,const std::string& message){if(!x)throw std::runtime_error(message);}
void close(const Tensor& x,const Tensor& y,const std::string& name){
  if(x.sizes()!=y.sizes()||!at::allclose(x.cpu(),y,6e-5,6e-6)){
    std::cerr<<name<<" candidate="<<x.cpu()<<" formula="<<y<<'\n';throw std::runtime_error("block graph numerical difference: "+name);}}
WeightedRowAdjoint root(at::IntArrayRef shape,at::IntArrayRef mask,at::Device device,float shift=.3){
  return {values(shape,shift).to(device),at::zeros(mask,at::TensorOptions().device(device).dtype(at::kBool))};}
BlockSettleFlowRoots roots(const BlockSettleFlowTape& t,at::Device device){
  BlockSettleFlowRoots r;r.edges={root({Frames,3},{Frames,3},device),root({Frames,3,2},{Frames,3},device)};
  r.outputs={root({Frames,2},{Frames,2},device),root({Frames,2,2},{Frames,2},device)};
  for(const auto& stage:t.regions){const auto& x=*stage.local;const auto n=x.sequence.nodes,rows=Frames*n,fw=x.sequence.layout.floating_width(),fields=int64_t(x.sequence.layout.fields().size());
    BlockKVLocalRoots a;a.received=root({rows,3},{rows},device);
    a.messages.controls=root({Frames,n},{Frames,n},device);a.messages.full=root({Frames,n,2},{Frames,n},device);
    a.messages.read.memory=root({rows,2,2},{rows,2},device);a.messages.read.total=root({rows,2},{rows,2},device);
    for(const auto& [name,v]:x.messages.selection.describe.fields)a.messages.read.describe[name]=root(v.sizes(),{rows},device);
    auto old=root({rows,fw},{rows,fields},device,.4),proposal=root({rows,fw},{rows,fields},device,.5),final=root({Batch*n,fw},{Batch*n,fields},device,.6);
    a.raw={old.value,old.connected,proposal.value,proposal.connected,final.value,final.connected};r.regions.push_back(std::move(a));}
  return r;
}
void select(BlockSettleFlowRoots& r,int mode){auto set=[&](const Tensor& flags,int kind){flags.fill_(mode==8||mode==kind);};
  for(auto* f:{&r.outputs.strength,&r.outputs.content})set(f->connected,0);
  for(auto* f:{&r.edges.strength,&r.edges.content})set(f->connected,1);
  for(auto& a:r.regions){set(a.received.connected,2);set(a.messages.controls.connected,3);set(a.messages.full.connected,4);
    set(a.messages.read.memory.connected,5);set(a.messages.read.total.connected,5);
    for(auto& [name,x]:a.messages.read.describe)set(x.connected,6);
    for(auto x:{a.raw.old_connected,a.raw.proposal_connected,a.raw.final_connected})set(x,7);}
}
void compare(const Fixture& f,const Formula& ref,const BlockSettleFlowTape& t,const BlockSettleFlowGradient& g,
    const ParameterVjp& parameters,const BlockSettleFlowRoots& roots){
  at::AutoGradMode enabled(true);std::vector<Tensor> ys,bars;
  auto add=[&](const Tensor& y,const Tensor& bar,bool connected){if(connected&&y.requires_grad()){ys.push_back(y);bars.push_back(bar.cpu());}};
  auto row=[&](const Tensor& y,const WeightedRowAdjoint& root,int64_t at){add(y,root.value.reshape({root.connected.numel(),-1})[at].reshape(y.sizes()),root.connected.reshape({-1})[at].cpu().item<bool>());};
  close(t.nodes.payload,at::stack(ref.messages).reshape({Frames,Nodes,3}),"node messages");
  require(at::equal(t.nodes.present.cpu(),ref.present)&&at::equal(t.nodes.times.cpu(),ref.times),"node existence/exact times");
  const auto& graph=f.graph.graph();
  auto ports=[&](const std::vector<int64_t>& src,const std::vector<int64_t>& delay,const BlockPhysicalColumns& actual,const BlockPhysicalRoots& rs){
    for(int64_t frame=0;frame<Frames;++frame)for(size_t p=0;p<src.size();++p){const auto node=src[p];const int64_t flat=frame*src.size()+p;const bool present=ref.present[frame][node].item<bool>();
      close(actual.payload[frame][p],present?ref.messages[frame*Nodes+node]:at::zeros({3},at::kFloat),"physical payload");
      require(actual.present[frame][p].cpu().item<bool>()==present,"physical presence");
      require(actual.times[frame][p].cpu().item<int64_t>()==(present?ref.times[frame][node].item<int64_t>()+delay[p]:0),"physical exact clock");
      if(present){row(ref.messages[frame*Nodes+node][0],rs.strength,flat);row(ref.messages[frame*Nodes+node].slice(0,1),rs.content,flat);}}
  };
  ports({0,1,0},{1,1,1},t.edges,roots.edges);ports({2,2},{0,0},t.outputs,roots.outputs);
  for(int64_t r=0;r<2;++r){const auto& a=*t.regions[r].local;const auto& rs=roots.regions[r];const auto rows=Frames*a.sequence.nodes;
    close(a.aggregate.output,at::stack(ref.received[r]),"received");close(a.messages.selection.memory_values(),at::stack(ref.memory[r]),"Mem");
    close(a.messages.selection.memory_totals(),at::stack(ref.total[r]),"totals");close(a.messages.selection.controls.reshape({rows}),at::stack(ref.controls[r]),"controls");
    close(a.messages.full.output,at::stack(ref.full[r]),"Full");
    for(int64_t i=0;i<rows;++i){row(ref.received[r][i],rs.received,i);row(ref.controls[r][i],rs.messages.controls,i);row(ref.full[r][i],rs.messages.full,i);
      for(int channel=0;channel<2;++channel){row(ref.memory[r][i][channel],rs.messages.read.memory,i*2+channel);row(ref.total[r][i][channel],rs.messages.read.total,i*2+channel);}
      for(const auto& [name,x]:rs.messages.read.describe){close(a.messages.selection.describe.fields.at(name)[i],ref.describe[r][i].at(name),"Describe."+name);row(ref.describe[r][i].at(name),x,i);}}
    const std::vector<const std::vector<tide::State>*> states{&ref.old[r],&ref.proposal[r],&ref.final[r]};
    const std::vector<WeightedRawState> actual{a.sequence.old,a.sequence.proposal,a.sequence.final};
    const std::vector<Tensor> values{rs.raw.old,rs.raw.proposal,rs.raw.final},flags{rs.raw.old_connected,rs.raw.proposal_connected,rs.raw.final_connected};
    for(int part=0;part<3;++part){const auto integer=actual[part].integer.cpu(),floating=actual[part].floating.cpu();
      for(size_t i=0;i<states[part]->size();++i){const auto& s=(*states[part])[i];const auto length=s.slots.at("times").numel();
        require(integer[i][0].item<int64_t>()==s.last_time&&integer[i][1].item<int64_t>()==s.observations&&integer[i][2].item<int64_t>()==length&&
          at::equal(integer[i].slice(0,4,4+length),s.slots.at("times")),"raw state int64 exactness");
        int field=0;for(const auto& f:a.sequence.layout.fields()){const auto y=f.name=="value"?s.value:s.slots.at(f.name);
          close(floating[i].narrow(0,f.offset,y.numel()).reshape(y.sizes()),y,"raw."+f.name);
          add(y,values[part][i].narrow(0,f.offset,y.numel()).reshape(y.sizes()),flags[part][i][field++].cpu().item<bool>());}}}
  }
  auto expected=ys.empty()?std::vector<Tensor>(ref.leaves.size()):torch::autograd::grad(ys,ref.leaves,bars,true,false,true);
  std::map<std::string,size_t> canonical;for(size_t i=0;i<parameters.owners.size();++i)canonical[parameters.owners[i].canonical]=i;
  for(size_t j=0;j<ref.labels.size();++j){const auto& label=ref.labels[j];Tensor value,flag;
    if(label.kind=="input"){value=g.inputs.reshape({Frames*3,3})[label.row];flag=g.input_connected.reshape({Frames*3})[label.row];}
    else if(label.kind=="parameter"){const auto index=canonical.at(label.name);const auto offset=parameters.offsets[index];require(offset>=0,"missing mapped parameter");
      value=parameters.values.narrow(0,offset,ref.leaves[j].numel()).reshape(ref.leaves[j].sizes());flag=parameters.connected[index];}
    else {const auto& stage=*t.regions[label.region].local;const auto& part=stage.sequence.layout.field(label.name);int field=0;
      for(const auto& f:stage.sequence.layout.fields()){if(f.name==label.name)break;++field;}
      const auto& dg=g.regions[label.region];value=(label.kind=="bound"?dg.bound_initial:dg.sequence.initial)[label.row].narrow(0,part.offset,ref.leaves[j].numel()).reshape(ref.leaves[j].sizes());
      flag=(label.kind=="bound"?dg.bound_initial_connected:dg.sequence.initial_connected)[label.row][field];}
    const auto name=label.kind+":"+label.name+":"+std::to_string(label.region)+":"+std::to_string(label.row);
    require(flag.cpu().item<bool>()==expected[j].defined(),"None/zero differs "+name);
    if(expected[j].defined())close(value,expected[j],name);else require(value.cpu().eq(0).all().item<bool>(),"nonzero disconnected "+name);
  }
}
void check(at::Device device,const std::string& mode){
  at::NoGradGuard no_grad;Fixture f;for(auto& b:f.bindings)b.read_mode=mode;const auto ref=formula(f);const auto x=device_input(f,device);const auto bindings=device_bindings(f,device);
  DeviceProgram program(device);program.limit_workspace(1<<29);TaskServices services(program,device,1<<29);
  auto tape=append_block_settle_flow(program,services,f.graph,bindings,x,1<<29);
  auto root=roots(tape,device);auto gradient=append_block_settle_flow_vjp(program,services,f.graph,tape,root,1<<29);
  auto parameters=append_block_settle_parameter_vjp(program,gradient,f.registry,f.roles,tape.dispatcher->error(),1<<27);
  services.append_stop();services.finish();program.finish();
  for(int kind=mode=="+"?0:8;kind<10;++kind){select(root,kind);tape.dispatcher->epoch().fill_(kind+1);portable_torch::synchronize(device);
    services.submit();program.run();services.wait(10000);
    require(!tape.dispatcher->error().cpu().item<int>(),"block graph refused, code="+std::to_string(tape.dispatcher->error().cpu().item<int>()));
    try{compare(f,ref,tape,gradient,parameters,root);}catch(...){std::cerr<<"read="<<mode<<" roots="<<kind<<'\n';throw;}}
  services.close();program.close();
}
void rejected_bindings(at::Device device){
  at::NoGradGuard no_grad;
  Fixture f;auto x=device_input(f,device);auto b=device_bindings(f,device);
  const std::vector<std::function<void(tide::Graph&)>> changes{
    [](auto& g){g.nodes[0].clear=true;}, [](auto& g){g.nodes[0].identity=true;},
    [](auto& g){g.nodes[0].next_state="retain-v1";}, [](auto& g){g.nodes[0].emission="round_robin";},
    [](auto& g){g.nodes[0].emit_period=2;}, [](auto& g){g.nodes[0].emit_phases={0};},
    [](auto& g){g.nodes[0].state_clock={2,0,1};}, [](auto& g){g.regions[0].observe_all=false;}};
  for(const auto& change:changes){auto graph=f.graph.graph();change(graph);bool rejected=false;
    try{tide::SettleGraph spec(graph,f.graph.ranks(),"ports",f.graph.stride());DeviceProgram p(device);TaskServices services(p,device,1<<29);
      append_block_settle_flow(p,services,spec,b,x,1<<29);
    }catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"concrete block accepted incompatible state/emission/observation binding");}
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("block graph check requires explicit accelerator FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    rejected_bindings(device);
    for(const auto& mode:{"+","-","0"})check(device,mode);
    std::cout<<"block-settle-flow: passed actual_full_graph=true independent_scalar_formula=true roots=10 read_snapshots=3 "
      <<"B=2 L=2 duplicate_edges=true duplicate_ports=true shared_parameters=true exact_int64=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
