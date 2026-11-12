#include "block_selection_formula.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>

namespace {
using namespace tide::device_online;
using namespace tide::device_online::block_selection_check;
using at::Tensor;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void close(const Tensor& actual,const Tensor& expected){
  require(actual.sizes()==expected.sizes(),"block read-selection shape differs");
  if(!at::allclose(actual.cpu(),expected,3e-5,3e-6)){
    std::cerr<<actual.cpu()<<" versus "<<expected<<'\n';throw std::runtime_error("block read-selection numeric difference");}
}
BlockReadSelectionInput move(const BlockReadSelectionInput& x,at::Device d){
  return {copy_weighted_raw_state(x.read,d),x.mass.to(d),x.content.to(d),x.present.to(d),x.times.to(d),x.node_owners.to(d)};
}
BlockReadSelectionParameters move(const BlockReadSelectionParameters& w,at::Device d){
  BlockReadSelectionParameters out;out.query={w.query.query.to(d),w.query.key.to(d),w.query.value.to(d)};
  for(const auto& [name,value]:w.describe)out.describe[name]=value.to(d);
  auto h=w.region.context;out.region.context={h.matrix.to(d),h.hidden_bias.to(d),h.projection.to(d),h.bias.to(d)};
  out.region.context_owners=w.region.context_owners.to(d);return out;
}
void compare(const Fixture& f,const Formula& ref,const BlockReadSelectionTape& t,const BlockReadSelectionGradient& g,
    const Tensor& roots,const Tensor& bar){
  at::AutoGradMode enabled(true);close(t.query.memory,ref.memory);close(t.query.total,ref.total);close(t.controls,ref.controls);
  require(at::equal(t.active.cpu(),ref.controls>0),"block per-node activity differs");
  require(at::equal(t.region.history_present.cpu(),t.lengths.cpu()>0),"block region history presence differs");
  for(int64_t row=0;row<Rows;++row)for(const auto& [name,expected]:ref.descriptions[row])close(t.describe.fields.at(name)[row],expected);
  for(int64_t frame=0;frame<Frames;++frame)for(int64_t region=0;region<2;++region){
    const auto row=frame*2+region;const auto n=t.lengths[row].cpu().item<int64_t>();
    require(t.region.history_times[row].cpu().item<int64_t>()==(n?Clock+frame*3:-1),"block lost original int64 clock");
  }
  std::vector<Tensor> outputs,bars;
  for(int64_t row=0;row<Rows;++row)if(roots.reshape({Rows})[row].item<bool>()&&ref.control_roots[row].requires_grad()){
    outputs.push_back(ref.control_roots[row]);bars.push_back(bar.reshape({Rows})[row]);}
  auto expected=outputs.empty()?std::vector<Tensor>(ref.leaves.size()):torch::autograd::grad(outputs,ref.leaves,bars,true,false,true);
  const std::vector<std::string> qnames{"query","key","value"},hnames{"matrix","hidden_bias","projection","bias"};
  const auto qvalues=roles(g.query),qflags=std::vector<Tensor>{g.query_connected,g.key_connected,g.value_connected};
  for(size_t j=0;j<ref.leaves.size();++j){
    const auto& label=ref.labels[j];Tensor value,connected;
    if(label.kind=="raw"){
      const auto& part=f.raw_layout.field(label.name);auto index=0;
      for(const auto& p:f.raw_layout.fields()){if(p.name==label.name)break;++index;}
      value=g.raw[label.row].slice(0,part.offset,part.offset+ref.leaves[j].numel()).reshape(ref.leaves[j].sizes());connected=g.raw_connected[label.row][index];
    }else if(label.kind=="mass"){value=g.mass.reshape({Rows})[label.row];connected=g.mass_connected.reshape({Rows})[label.row];}
    else if(label.kind=="content"){value=g.content.reshape({Rows,2})[label.row];connected=g.content_connected.reshape({Rows})[label.row];}
    else if(label.kind=="describe"){const auto& x=g.describe.at(label.name);value=x.value[label.row];connected=x.connected[label.row];}
    else if(label.kind=="query"){
      const auto index=std::find(qnames.begin(),qnames.end(),label.name)-qnames.begin();value=qvalues[index][label.row];connected=qflags[index][label.row];
    }else {
      require(g.region_context.has_value(),"missing block context gradient");
      const auto index=std::find(hnames.begin(),hnames.end(),label.name)-hnames.begin();
      value=roles(g.region_context->parameters)[index][label.row];connected=g.region_context->owner_connected[label.row];
    }
    try{
      require(connected.cpu().item<bool>()==expected[j].defined(),"block None/connected-zero differs");
      if(expected[j].defined())close(value,expected[j]);else require(value.cpu().eq(0).all().item<bool>(),"block disconnected gradient is nonzero");
    }catch(...){std::cerr<<"root="<<label.kind<<'.'<<label.name<<" row="<<label.row<<'\n';throw;}
  }
}
void check(at::Device device){
  at::NoGradGuard no_grad;Fixture f;auto ref=formula(f);auto x=move(f.input,device);auto weights=move(f.weights,device);
  auto error=at::zeros({1},x.times.options().dtype(at::kInt)),roots=at::zeros_like(x.present),bar=at::zeros_like(x.mass);
  DeviceProgram p(device);p.limit_workspace(128*1024*1024);
  auto tape=append_block_read_selection(p,f.plan,f.kv,f.regional,x,weights,error,128*1024*1024);
  auto gradient=append_block_read_selection_vjp(p,tape,bar,roots,error,128*1024*1024);p.finish();
  Tensor host_roots,host_bar;
  double seconds=0;
  for(int mode=0;mode<3;++mode){
    host_roots=at::zeros({Frames,Nodes},at::kBool);host_bar=at::full({Frames,Nodes},NAN,at::kFloat);
    if(mode==0){host_roots.fill_(true);host_bar.copy_(values({Frames,Nodes},.9));}
    else if(mode==1){host_roots[0][0]=true;host_bar[0][0]=.7;host_roots[3].fill_(true);}
    else {host_roots[1][1]=true;host_bar[1][1]=0;host_roots[2][0]=true;host_bar[2][0]=-.3;}
    roots.copy_(host_roots);bar.copy_(host_bar);portable_torch::synchronize(device);
    auto begin=std::chrono::steady_clock::now();p.run();seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    require(!error.cpu().item<int>(),"block read-selection rejected valid input");compare(f,ref,tape,gradient,host_roots,host_bar);
  }
  // Separate reverse owner must use the saved numerical invocation, not the
  // subsequently overwritten input coordinates, raw state or parameter banks.
  DeviceProgram reverse(device);reverse.limit_workspace(128*1024*1024);
  auto saved=append_block_read_selection_vjp(reverse,tape,bar,roots,error,128*1024*1024);reverse.finish();
  for(auto tensor:{x.read.floating,x.mass,x.content})tensor.fill_(NAN);
  for(auto tensor:{x.read.integer,x.times,x.node_owners})tensor.fill_(-99);x.present.zero_();
  for(auto tensor:roles(weights.query))tensor.fill_(NAN);for(auto& [_,tensor]:weights.describe)tensor.fill_(NAN);
  for(auto tensor:roles(weights.region.context))tensor.fill_(NAN);weights.region.context_owners.fill_(-99);
  portable_torch::synchronize(device);reverse.run();require(!error.cpu().item<int>(),"block reverse reread mutable source");
  compare(f,ref,tape,saved,host_roots,host_bar);reverse.close();p.close();
  DeviceProgram refusal(device);bool denied=false;
  try{append_block_read_selection(refusal,f.plan,f.kv,f.regional,x,weights,error,1);}catch(const std::invalid_argument&){denied=true;}
  require(denied,"block read-selection omitted tensor-budget admission");refusal.close();
  std::cout<<"block-read-selection: passed frames=4 nodes=3 complete_regions=2 roots=3 shared_banks=2 "
    <<"independent_formula_vjp=true saved_versions=true full_executor=false three_replay_seconds="<<seconds<<'\n';
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("block Read-selection requires explicit accelerator/FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);check(device);return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
