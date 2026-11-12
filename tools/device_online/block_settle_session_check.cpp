#include "block_settle_formula.h"
#include "block_settle_session.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <filesystem>
#include <cmath>
#include <iostream>
#include <unistd.h>
namespace {
using namespace tide::device_online;using namespace tide::device_online::block_flow_check;using at::Tensor;
void require(bool x,const std::string& what){if(!x)throw std::runtime_error(what);}
void close(const Tensor& x,const Tensor& y,const std::string& name){if(x.sizes()!=y.sizes()||!at::allclose(x.cpu(),y,8e-5,8e-6)){
  std::cerr<<name<<" actual="<<x.cpu()<<" expected="<<y<<'\n';throw std::runtime_error("resident block owner numeric difference "+name);}}
Fixture fixture(){Fixture f;
  for(size_t r=0;r<2;++r){WeightedRawLayout layout(f.bindings[r].memory,16);
    f.input.initial[r]=encode_weighted_raw_state(layout,f.states[r]);f.input.bound_initial[r]=encode_weighted_raw_state(layout,f.bound[r]);}
  return f;
}
struct Adam {std::map<std::string,Tensor> first,second;std::map<std::string,int64_t> step;};
BlockSettleFlowRoots roots(const BlockSettleWindow& w,bool final){
  const auto& t=*w.tape;BlockSettleFlowRoots r;const auto opts=t.outputs.payload.options();
  r.outputs.strength={at::full({Frames,2},.07,opts),at::ones({Frames,2},opts.dtype(at::kBool))};
  r.outputs.content={at::full({Frames,2,2},.13,opts),at::ones({Frames,2},opts.dtype(at::kBool))};
  if(final){for(const auto& stage:t.regions){BlockKVLocalRoots local;const auto& sequence=stage.local->sequence;
      local.raw.final=at::full_like(sequence.final.floating,.017);local.raw.final_connected=at::ones({sequence.final.floating.size(0),int64_t(sequence.layout.fields().size())},opts.dtype(at::kBool));
      r.regions.push_back(std::move(local));}}
  return r;
}
std::vector<Tensor> reference_gradients(const std::vector<Formula>& formulas){
  at::AutoGradMode yes(true);std::vector<Tensor> output,bars;
  for(const auto& ref:formulas)for(int64_t frame=0;frame<Frames;++frame)if(ref.present[frame][2].item<bool>())for(int port=0;port<2;++port){
    const auto y=ref.messages[frame*Nodes+2];output.push_back(y);bars.push_back(at::tensor({.07f,.13f,.13f}));}
  for(const auto& region:formulas.back().final)for(const auto& state:region){output.push_back(state.value);bars.push_back(at::full_like(state.value,.017));
    for(const auto& [name,x]:state.slots)if(name!="times"){output.push_back(x);bars.push_back(at::full_like(x,.017));}}
  return torch::autograd::grad(output,formulas.back().leaves,bars,true,false,true);
}
void compare_gradients(const Formula& ref,const std::vector<Tensor>& expected,const BlockSettleAdjoints& actual,
    const std::vector<BlockSettleWindow>& windows){
  std::map<std::string,size_t> owners;for(size_t j=0;j<actual.parameters.owners.size();++j)owners[actual.parameters.owners[j].canonical]=j;
  for(size_t j=0;j<ref.labels.size();++j){const auto& label=ref.labels[j];Tensor value,flag;
    if(label.kind=="input"){const auto& g=actual.windows[label.region];value=g.inputs.reshape({Frames*3,3})[label.row];flag=g.input_connected.reshape({Frames*3})[label.row];}
    else if(label.kind=="parameter"){const auto index=owners.at(label.name);const auto offset=actual.parameters.offsets[index];
      value=actual.parameters.values.narrow(0,offset,ref.leaves[j].numel()).reshape(ref.leaves[j].sizes());flag=actual.parameters.connected[index];}
    else {const auto& layout=windows.front().tape->regions[label.region].local->sequence.layout;const auto& f=layout.field(label.name);int field=0;
      for(const auto& p:layout.fields()){if(p.name==label.name)break;++field;}
      const auto& g=actual.windows.front().regions[label.region];
      value=(label.kind=="bound"?actual.bound_initial[label.region].value:g.sequence.initial)[label.row].narrow(0,f.offset,ref.leaves[j].numel()).reshape(ref.leaves[j].sizes());
      flag=(label.kind=="bound"?actual.bound_initial[label.region].connected:g.sequence.initial_connected)[label.row][field];}
    const auto name=label.kind+":"+label.name+":"+std::to_string(label.region)+":"+std::to_string(label.row);
    require(flag.cpu().item<bool>()==expected[j].defined(),"retained block None/zero differs "+name);
    if(expected[j].defined())close(value,expected[j],name);else require(value.cpu().eq(0).all().item<bool>(),"retained disconnected gradient nonzero");}
}
void update(Fixture& f,const Formula& ref,const std::vector<Tensor>& gradients,Adam& state,const tide::OptimizerGroup& group){
  for(size_t j=0;j<ref.labels.size();++j)if(ref.labels[j].kind=="parameter"&&gradients[j].defined()){
    const auto& name=ref.labels[j].name;const auto g=gradients[j];const auto value=f.registry.value(name);
    if(!state.first.count(name)){state.first[name]=at::zeros_like(value);state.second[name]=at::zeros_like(value);}
    state.first[name]=float(group.beta1)*state.first[name]+float(1-group.beta1)*g;
    state.second[name]=float(group.beta2)*state.second[name]+float(1-group.beta2)*g.square();const auto step=++state.step[name];
    const auto m=state.first[name]/float(1-std::pow(group.beta1,step)),v=state.second[name]/float(1-std::pow(group.beta2,step));
    value.copy_(float(1-group.lr*group.weight_decay)*value-float(group.lr)*m/(v.sqrt()+float(group.eps)));
  }
  f.states=ref.final;
  for(auto& region:f.states)for(auto& state:region){state.value=state.value.detach();
    for(auto& [name,x]:state.slots)x=x.detach();}
  for(size_t r=0;r<2;++r)f.input.initial[r]=encode_weighted_raw_state(WeightedRawLayout(f.bindings[r].memory,16),f.states[r]);
}
void compare_checkpoint(const BlockSettleCheckpoint& actual,const Fixture& f,const Adam& state,const ParameterVjp& layout){
  for(size_t i=0;i<layout.owners.size();++i){const auto& p=layout.owners[i];close(actual.parameters.at(p.canonical),f.registry.value(p.canonical),"AdamW parameter");
    const auto steps=state.step.count(p.canonical)?state.step.at(p.canonical):0;require(actual.slots.steps[i].item<int64_t>()==steps,"AdamW exact independent owner steps");
    if(steps){close(actual.slots.first.narrow(0,layout.offsets[i],p.value.numel()).reshape(p.value.sizes()),state.first.at(p.canonical),"AdamW first");
      close(actual.slots.second.narrow(0,layout.offsets[i],p.value.numel()).reshape(p.value.sizes()),state.second.at(p.canonical),"AdamW second");}}
  for(size_t r=0;r<2;++r){auto encoded=encode_weighted_raw_state(WeightedRawLayout(f.bindings[r].memory,16),f.states[r]);
    require(at::equal(actual.state[r].integer,encoded.integer),"checkpoint raw int64 differs");close(actual.state[r].floating,encoded.floating,"checkpoint exact raw fields");}
}
void exact(const BlockSettleCheckpoint& a,const BlockSettleCheckpoint& b){
  require(a.identity==b.identity&&a.generation==b.generation&&a.next_token==b.next_token&&a.aliases==b.aliases,"restored block metadata differs");
  for(const auto& [name,x]:a.parameters)require(at::equal(x,b.parameters.at(name)),"restored block parameter differs");
  const std::vector<Tensor> x{a.slots.values,a.slots.first,a.slots.second,a.slots.maximum,a.slots.steps,a.slots.corrections,a.last_positions,
      a.boundary.node_present,a.boundary.region_times,a.boundary.input_counts,a.boundary.input_times},
    y{b.slots.values,b.slots.first,b.slots.second,b.slots.maximum,b.slots.steps,b.slots.corrections,b.last_positions,
      b.boundary.node_present,b.boundary.region_times,b.boundary.input_counts,b.boundary.input_times};
  for(size_t j=0;j<x.size();++j)require(at::equal(x[j],y[j]),"restored optimizer/cut differs");
  for(size_t r=0;r<a.state.size();++r)for(const auto& pair:std::vector<std::pair<Tensor,Tensor>>{{a.state[r].floating,b.state[r].floating},{a.state[r].integer,b.state[r].integer},
      {a.bound_initial[r].floating,b.bound_initial[r].floating},{a.bound_initial[r].integer,b.bound_initial[r].integer}})require(at::equal(pair.first,pair.second),"restored raw/bound initial differs");
}
void complete_boundary(const BlockSettleCheckpoint& actual,const Fixture& f,const Formula& last,int step){
  const auto counts=f.input.present.to(at::kLong).sum(1)*(2*(step+1))-1;
  require(at::equal(actual.boundary.input_counts,counts),"independent input occurrence ledger differs");
  require(actual.boundary.node_present.all().item<bool>(),"nonempty initial node presence lost");
  auto input_times=at::full_like(counts,-1),region_times=at::full({Batch,2},-1,at::kLong);
  for(int64_t b=0;b<Batch;++b)for(int64_t l=0;l<Length;++l){const auto frame=b*Length+l;
    for(int64_t p=0;p<3;++p)if(f.input.present[b][l][p].item<bool>())
      input_times[b][p]=f.input.positions[b][l].item<int64_t>()*f.graph.stride()+f.graph.rank(f.graph.graph().inputs[p]);
    for(int64_t n=0;n<Nodes;++n)if(last.times[frame][n].item<int64_t>()>0)
      region_times[b][f.graph.graph().nodes[n].region]=last.times[frame][n];
  }
  require(at::equal(actual.boundary.input_times,input_times),"independent input int64 clock differs");
  require(at::equal(actual.boundary.region_times,region_times),"independent regional query history differs");
}
std::unique_ptr<BlockSettleSession> session(const Fixture& f,at::Device device,const tide::OptimizerGroup& group,const BlockSettleCheckpoint* saved=nullptr){
  BlockSettleLimits limits;limits.forward_bytes=1LL<<29;limits.backward_bytes=1LL<<30;limits.total_bytes=3LL<<30;
  return std::make_unique<BlockSettleSession>(f.graph,f.bindings,f.registry,f.roles,f.input.initial,f.input.bound_initial,device,
    DeviceOptimizerKind::adamw,std::vector<tide::OptimizerGroup>{group},limits,saved);
}
void check(at::Device device,const std::string& checkpoint_path,bool restore_only){
  at::NoGradGuard no_grad;auto f=fixture();const auto original=f; // Tensor registry cloned below: consumer originals stay immutable.
  tide::OptimizerGroup group;group.lr=.003;group.weight_decay=.02;group.eps=.1;
  for(const auto& p:f.registry.owners())group.parameters.push_back(p.canonical);
  auto consumer=fixture();std::optional<BlockSettleCheckpoint> saved;
  if(restore_only)saved=load_block_settle_checkpoint(checkpoint_path);
  auto owner=session(consumer,device,group,saved?&*saved:nullptr);
  Adam adam;std::unique_ptr<BlockSettleSession> restored;BlockSettleCheckpoint first;
  for(int step=0;step<3;++step){std::vector<Formula> reference;std::vector<BlockSettleWindow> windows;std::vector<BlockSettleFlowRoots> cotangents;
    for(int window=0;window<2;++window){f.input.positions=original.input.positions+(step*2+window)*20;
      reference.push_back(formula(f,window?&reference.back():nullptr));
      if(restore_only&&step==0)continue;
      auto w=owner->advance(f.input.positions,f.input.payload,f.input.present);windows.push_back(w);cotangents.push_back(roots(w,window==1));
      if(restored)restored->advance(f.input.positions,f.input.payload,f.input.present);
    }
    const auto gradients=reference_gradients(reference);BlockSettleAdjoints actual;
    if(!(restore_only&&step==0)){actual=owner->backward(cotangents);compare_gradients(reference.back(),gradients,actual,windows);owner->step();
      if(restored){auto r=restored->backward(cotangents);restored->step();exact(owner->checkpoint(),restored->checkpoint());}}
    update(f,reference.back(),gradients,adam,group);
    if(restore_only&&step==0)continue;
    compare_checkpoint(owner->checkpoint(),f,adam,actual.parameters);
    complete_boundary(owner->checkpoint(),f,reference.back(),step);
    if(step==0){first=owner->checkpoint();save_block_settle_checkpoint(first,checkpoint_path);auto loaded=load_block_settle_checkpoint(checkpoint_path);exact(first,loaded);
      bool save_refused=false,load_refused=false;
      try{save_block_settle_checkpoint(first,checkpoint_path,1);}catch(const std::invalid_argument&){save_refused=true;}
      try{load_block_settle_checkpoint(checkpoint_path,1);}catch(const std::invalid_argument&){load_refused=true;}
      require(save_refused&&load_refused,"checkpoint byte budget was ignored");exact(first,load_block_settle_checkpoint(checkpoint_path));
      auto corrupt=loaded;corrupt.boundary.region_times=loaded.boundary.region_times.clone();
      corrupt.boundary.region_times[0][0]=(loaded.last_positions[0].item<int64_t>()+1)*consumer.graph.stride()+consumer.graph.ranks()[0];
      bool cut_refused=false;try{session(consumer,device,group,&corrupt);}catch(const std::invalid_argument&){cut_refused=true;}
      require(cut_refused,"checkpoint accepted regional query history beyond its complete cut");
      restored=session(consumer,device,group,&loaded);}
  }
  if(restore_only)exact(owner->checkpoint(),load_block_settle_checkpoint(checkpoint_path+".expected"));
  else save_block_settle_checkpoint(owner->checkpoint(),checkpoint_path+".expected");
  owner->close();if(restored)restored->close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{std::string path;bool read=false;std::vector<char*> args{argv[0]};for(int i=1;i<argc;++i){const std::string x=argv[i];
      if(x=="--checkpoint-path"){if(++i==argc)throw std::invalid_argument("checkpoint path missing");path=argv[i];}
      else if(x=="--restore-check")read=true;else args.push_back(argv[i]);}
    const bool temporary=path.empty();if(temporary)path="/tmp/tide-block-session-"+std::to_string(::getpid())+".pt";
    const auto options=portable_torch::parse_cli(args.size(),args.data(),true);if(options.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(options.device_spec=="auto"||options.dtype!=at::kFloat)throw std::invalid_argument("block session check requires explicit accelerator FP32");
    const auto device=portable_torch::resolve_device(options);at::set_num_threads(1);at::set_num_interop_threads(1);check(device,path,read);
    if(temporary){std::filesystem::remove(path);std::filesystem::remove(path+".expected");}
    std::cout<<"block-settle-session: passed two_window_bptt=true independent_formula_vjp=true AdamW_steps=3 "
      <<"exact_raw_checkpoint=true fresh_owner=true fresh_process="<<read<<'\n';return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
