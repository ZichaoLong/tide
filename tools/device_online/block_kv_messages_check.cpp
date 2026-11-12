#include "block_kv_message_formula.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace {
using namespace tide::device_online;
using namespace tide::device_online::block_message_check;
using block_selection_check::Frames;using block_selection_check::Nodes;using block_selection_check::Rows;
using at::Tensor;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void close(const Tensor& actual,const Tensor& expected){
  require(actual.sizes()==expected.sizes(),"KV message shape differs");
  if(!at::allclose(actual.cpu(),expected,4e-5,4e-6)){
    std::cerr<<actual.cpu()<<" versus "<<expected<<'\n';throw std::runtime_error("KV message numeric difference");}
}
std::vector<WeightedRowAdjoint*> fields(BlockKVMessageRoots& r){
  return {&r.controls,&r.full,&r.message_strength,&r.message_content,&r.projection};
}
BlockKVMessageRoots make_roots(at::Device device){
  BlockKVMessageRoots r;const auto opts=at::TensorOptions().device(device).dtype(at::kFloat);
  auto fs=fields(r);for(int i=0;i<5;++i)*fs[i]={at::zeros(i==0||i==2?std::vector<int64_t>{Frames,Nodes}:
    std::vector<int64_t>{Frames,Nodes,2},opts),at::zeros({Frames,Nodes},opts.dtype(at::kBool))};return r;
}
void set_roots(BlockKVMessageRoots& roots,int mode){
  auto fs=fields(roots);
  for(int j=0;j<5;++j){auto& r=*fs[j];r.connected.zero_();r.value.fill_(NAN);
    const bool selected=mode==5||mode==j;
    if(!selected)continue;
    for(int64_t row=0;row<Rows;++row){r.connected.reshape({Rows})[row]=true;
      if(row>=9||j!=0&&(row==3||row==5||row==7))continue;
      auto value=r.value.reshape({Rows,-1})[row];value.copy_(block_selection_check::values(value.sizes(),.7+j*.13+row*.03));
      if(row==4&&mode==1)value.zero_();
    }
  }
}
void copy_roots(BlockKVMessageRoots& dst,BlockKVMessageRoots& src){auto a=fields(dst),b=fields(src);
  for(size_t i=0;i<a.size();++i){a[i]->value.copy_(b[i]->value);a[i]->connected.copy_(b[i]->connected);}}
void compare(const Fixture& f,const Formula& ref,const BlockKVMessageTape& t,const BlockKVMessageGradient& g,
    const ParameterVjp& parameters,BlockKVMessageRoots& roots,bool subset=false){
  at::AutoGradMode enabled(true);close(t.selection.controls,ref.selection.controls);
  close(t.full.output,at::stack(ref.full));close(t.messages.strength,at::stack(ref.message_strength).reshape({Frames,Nodes}));
  close(t.messages.content,at::stack(ref.message_content).reshape({Frames,Nodes,2}));
  if(t.messages.projection.defined())close(t.messages.projection,at::stack(ref.projection).reshape({Frames,Nodes,2}));
  auto expected_times=at::where(f.selection.input.present,f.selection.input.times,at::zeros_like(f.selection.input.times));
  require(t.messages.times.scalar_type()==at::kLong&&at::equal(t.messages.times.cpu(),expected_times),"KV message int64 times differ");
  require(at::equal(t.messages.present.cpu(),ref.presence),"KV message existence differs");
  require(t.messages.present[1][1].cpu().item<bool>()&&t.messages.content[1][1].cpu().eq(0).all().item<bool>(),"zero-valued present message lost");
  std::vector<Tensor> outputs,bars;
  std::vector<std::vector<Tensor>> ys{ref.selection.control_roots,ref.full,ref.message_strength,ref.message_content,ref.projection};
  auto rs=fields(roots);
  for(int k=0;k<5;++k)for(int64_t row=0;row<Rows;++row){
    if(rs[k]->connected.defined()&&rs[k]->connected.reshape({Rows})[row].item<bool>()&&ys[k][row].requires_grad()){
      outputs.push_back(ys[k][row]);bars.push_back(rs[k]->value.reshape({Rows,-1})[row].reshape(ys[k][row].sizes()));}
  }
  auto expected=outputs.empty()?std::vector<Tensor>(ref.leaves.size()):torch::autograd::grad(outputs,ref.leaves,bars,true,false,true);
  std::map<std::string,size_t> canonical;for(size_t i=0;i<parameters.owners.size();++i)canonical[parameters.owners[i].canonical]=i;
  for(size_t j=0;j<ref.labels.size();++j){const auto& label=ref.labels[j];Tensor value,connected;
    if(label.kind=="raw"||label.kind=="proposal"){
      const bool proposal=label.kind=="proposal";const auto& layout=proposal?f.proposal_layout:f.selection.raw_layout;
      const auto& part=layout.field(label.name);size_t field=0;for(const auto& x:layout.fields()){if(x.name==label.name)break;++field;}
      value=(proposal?g.full.raw:g.read.raw)[label.row].slice(0,part.offset,part.offset+ref.leaves[j].numel()).reshape(ref.leaves[j].sizes());
      connected=(proposal?g.full.raw_connected:g.read.raw_connected)[label.row][field];
    }else if(label.kind=="mass"){
      value=g.read.mass.reshape({Rows})[label.row];connected=g.read.mass_connected.reshape({Rows})[label.row];
    }else if(label.kind=="content"){
      value=g.content.value.reshape({Rows,2})[label.row];connected=g.content.connected.reshape({Rows})[label.row];
    }else{
      const auto name=f.registry.canonical_name(parameter_name(label));const auto index=canonical.at(name);const auto offset=parameters.offsets[index];
      if(subset&&offset<0){require(!parameters.connected[index].cpu().item<bool>(),"unselected trainable owner connected");continue;}
      require(offset>=0,"missing canonical message parameter contribution");
      value=parameters.values.narrow(0,offset,ref.leaves[j].numel()).reshape(ref.leaves[j].sizes());connected=parameters.connected[index];
    }
    try{require(connected.cpu().item<bool>()==expected[j].defined(),"KV message None/connected-zero differs");
      if(expected[j].defined())close(value,expected[j]);else require(value.cpu().eq(0).all().item<bool>(),"KV message disconnected bar nonzero");
    }catch(...){std::cerr<<label.kind<<'.'<<label.name<<" row="<<label.row<<'\n';throw;}
  }
}
template<class F> void refusal(at::Device device,F&& call,const char* message){
  DeviceProgram p(device);bool rejected=false;try{call(p);}catch(const std::invalid_argument&){rejected=true;}
  p.close();require(rejected,message);
}
void boundary_checks(at::Device device){
  at::NoGradGuard no_grad;Fixture f;
  // The omitted Full-owner argument must use the Read bindings, independently
  // of the main fixture's deliberately different legal Full owner table.
  f.full_owners=f.selection.input.node_owners.clone();const auto ref=formula(f);
  auto x=device_input(f,device);x.full_owners=at::Tensor{};auto weights=device_parameters(f,device);
  auto error=at::zeros({1},x.read.times.options().dtype(at::kInt));BlockKVMessageRoots empty;
  DeviceProgram plain(device);plain.limit_workspace(256*1024*1024);
  auto t=append_block_kv_messages(plain,f.selection.plan,f.selection.kv,f.selection.regional,x,weights,error,256*1024*1024);
  require(!t.messages.projection.defined()&&!t.projection_index,"default message block created cG");
  auto g=append_block_kv_messages_vjp(plain,t,empty,error,256*1024*1024);
  auto canonical=append_block_kv_message_parameter_vjp(plain,g,f.registry,f.roles,error,32*1024*1024);plain.finish();
  portable_torch::synchronize(device);plain.run();require(!error.cpu().item<int>(),"default/undefined-root message block failed");
  compare(f,ref,t,g,canonical,empty);
  BlockKVMessageRoots projected;projected.projection={at::ones_like(t.messages.content),at::ones_like(t.messages.present)};
  refusal(device,[&](DeviceProgram& p){append_block_kv_messages_vjp(p,t,projected,error,256*1024*1024);},"unrequested projection root accepted");
  for(int kind=0;kind<2;++kind){BlockKVMessageRoots half;
    if(kind==0)half.controls.value=at::zeros_like(t.messages.strength);
    else half.message_content.connected=at::ones_like(t.messages.present);
    refusal(device,[&](DeviceProgram& p){append_block_kv_messages_vjp(p,t,half,error,256*1024*1024);},"half-defined message root accepted");
  }
  auto bad_name=f.roles;bad_name.at("full.out")[0]="unregistered";
  refusal(device,[&](DeviceProgram& p){append_block_kv_message_parameter_vjp(p,g,f.registry,bad_name,error,32*1024*1024);},"unknown canonical name accepted");
  auto bad_role=f.roles;bad_role["not.a.role"]={"full.out.0"};
  refusal(device,[&](DeviceProgram& p){append_block_kv_message_parameter_vjp(p,g,f.registry,bad_role,error,32*1024*1024);},"unknown numerical role accepted");
  auto bad_count=f.roles;bad_count.at("full.out").pop_back();
  refusal(device,[&](DeviceProgram& p){append_block_kv_message_parameter_vjp(p,g,f.registry,bad_count,error,32*1024*1024);},"partial role bank mapping accepted");
  tide::ParameterRegistry bad_shape;bad_shape.add("bad.0",at::zeros({1},at::kFloat));bad_shape.add("bad.1",at::zeros({1},at::kFloat));
  BlockKVParameterRoles shape_roles{{"full.out",{"bad.0","bad.1"}}};
  refusal(device,[&](DeviceProgram& p){append_block_kv_message_parameter_vjp(p,g,bad_shape,shape_roles,error,32*1024*1024);},"canonical owner shape mismatch accepted");
  refusal(device,[&](DeviceProgram& p){append_block_kv_messages(p,f.selection.plan,f.selection.kv,f.selection.regional,x,weights,error,1);},"message forward budget omitted");
  refusal(device,[&](DeviceProgram& p){append_block_kv_messages_vjp(p,t,empty,error,1);},"message VJP budget omitted");
  refusal(device,[&](DeviceProgram& p){append_block_kv_message_parameter_vjp(p,g,f.registry,f.roles,error,1);},"message parameter budget omitted");
  plain.close();
}
void check(at::Device device){
  at::NoGradGuard no_grad;Fixture f;const auto ref=formula(f);auto x=device_input(f,device);auto weights=device_parameters(f,device);
  require(f.registry.canonical_name("read.query.0")==f.registry.canonical_name("full.query.0"),"shared Read/Full fixture alias missing");
  require(f.registry.canonical_name("read.query.0")!=f.registry.canonical_name("full.out.0"),"equal independent fixture parameters collapsed");
  require(f.full_owners[0][0].item<int64_t>()!=f.selection.input.node_owners[0][0].item<int64_t>(),"independent Full owner fixture missing");
  auto error=at::zeros({1},x.read.times.options().dtype(at::kInt));auto roots=make_roots(device),host_roots=make_roots(at::kCPU);
  DeviceProgram p(device);p.limit_workspace(256*1024*1024);
  auto t=append_block_kv_messages(p,f.selection.plan,f.selection.kv,f.selection.regional,x,weights,error,256*1024*1024,true);
  auto g=append_block_kv_messages_vjp(p,t,roots,error,256*1024*1024);
  auto canonical=append_block_kv_message_parameter_vjp(p,g,f.registry,f.roles,error,32*1024*1024);p.finish();
  for(int mode=0;mode<7;++mode){set_roots(host_roots,mode);copy_roots(roots,host_roots);
    portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"KV message block refused valid input");
    try{compare(f,ref,t,g,canonical,host_roots);}catch(...){std::cerr<<"KV message root mode="<<mode<<'\n';throw;}
  }
  set_roots(host_roots,5);copy_roots(roots,host_roots);portable_torch::synchronize(device);p.run();
  require(!error.cpu().item<int>(),"KV message final forward refused");
  {
    // A physical trainable subset must include every use of each selected
    // shared owner; selecting only one alias would deliberately omit a partial.
    BlockKVParameterRoles subset{{"read.query",f.roles.at("read.query")},{"full.query",f.roles.at("full.query")}};
    DeviceProgram reduction(device);auto selected=append_block_kv_message_parameter_vjp(reduction,g,f.registry,subset,error,32*1024*1024);reduction.finish();
    require(std::count_if(selected.offsets.begin(),selected.offsets.end(),[](int64_t x){return x>=0;})==2,"trainable subset added other owners");
    portable_torch::synchronize(device);reduction.run();require(!error.cpu().item<int>(),"message trainable subset reduction failed");
    compare(f,ref,t,g,selected,host_roots,true);reduction.close();
  }
  DeviceProgram reverse(device);reverse.limit_workspace(256*1024*1024);
  auto saved=append_block_kv_messages_vjp(reverse,t,roots,error,256*1024*1024);
  auto saved_parameters=append_block_kv_message_parameter_vjp(reverse,saved,f.registry,f.roles,error,32*1024*1024);reverse.finish();
  for(auto a:{x.read.read.floating,x.proposal.floating,x.read.mass,x.read.content})a.fill_(NAN);
  for(auto a:{x.read.read.integer,x.proposal.integer,x.read.times,x.read.node_owners,x.full_owners})a.fill_(-99);x.read.present.zero_();
  for(auto& [_,a]:parameter_roles(weights))a.fill_(NAN);weights.read.region.context_owners.fill_(-99);
  portable_torch::synchronize(device);reverse.run();require(!error.cpu().item<int>(),"KV message saved reverse reread source");
  compare(f,ref,t,saved,saved_parameters,host_roots);reverse.close();p.close();
  boundary_checks(device);
  std::cout<<"block-kv-messages: passed separate_read_proposal=true roots=7 shared_roles=true "
    <<"message_pair_unscaled=true explicit_cG=true default_no_projection=true undefined_roots=true "
    <<"owner_bindings=2 trainable_subset=true refusal_checks=10 int64_times=true saved_reverse=true full_executor=false\n";
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("KV message block requires explicit accelerator/FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);check(device);return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
