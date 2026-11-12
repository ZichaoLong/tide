#include "block_kv_sequence.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <iostream>

namespace {
using namespace tide::device_online;
using at::Tensor;
using tide::State;
constexpr int64_t B=2,L=3,N=2,R=B*L*N,C=4,Banks=2,Clock=(int64_t{1}<<54)+17;
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
void close(const Tensor& a,const Tensor& b){require(a.sizes()==b.sizes()&&at::allclose(a.cpu(),b,5e-5,5e-6),"KV sequence numeric difference");}
Tensor values(at::IntArrayRef shape,float shift){int64_t n=1;for(auto s:shape)n*=s;return (at::arange(n,at::kFloat)*.13+shift).sin().reshape(shape)*.2;}
std::vector<Tensor> roles(const WeightedKVParameters& w){return {w.base,w.left,w.right,w.summary};}
struct Label {std::string kind,field;int64_t row;};
struct Formula {
  std::vector<Tensor> leaves;std::vector<Label> labels;std::vector<State> old,proposal,final;
};
struct Fixture {
  tide::weighted::KVConfig config;WeightedRawLayout layout;
  std::vector<State> states;WeightedRawState initial;Tensor content,times,owners,present;WeightedKVParameters parameters;
  Fixture():config([](){tide::weighted::KVConfig c;c.width=2;c.summary_width=2;c.epsilon=.13;c.window=8;return c;}()),layout(config,C){
    for(int64_t owner=0;owner<B*N;++owner){State s;s.last_time=Clock+(owner/N)*100+owner%N-2;s.observations=1;
      s.value=at::zeros({3},at::kFloat);s.slots["times"]=at::tensor({s.last_time},at::kLong);
      for(const auto& f:layout.fields())if(f.name!="value"){
        auto shape=f.shape;shape.insert(shape.begin(),1);s.slots[f.name]=values(shape,.3+owner*.03);
        if(f.name=="weights")s.slots[f.name].fill_(.3);}
      states.push_back(s);
    }
    initial=encode_weighted_raw_state(layout,states);content=values({R,3},.2);times=at::zeros({B,L,N},at::kLong);
    owners=at::zeros_like(times);present=at::ones({B,L,N},at::kBool);
    for(int64_t row=0;row<R;++row){const auto b=row/(L*N),l=row/N%L,n=row%N;
      times.reshape({R})[row]=Clock+b*100+n+std::vector<int64_t>{0,3,12}[l];owners.reshape({R})[row]=n;
      content[row][0]=row%4==1?0.f:.4f;
      if(row%5==4){present.reshape({R})[row]=false;content[row].fill_(NAN);times.reshape({R})[row]=-99;owners.reshape({R})[row]=-99;}}
    parameters={values({Banks,4,2},.2),values({Banks,4,2},.3),values({Banks,2,2},.4),values({Banks,2,2},.5)};
  }
};
Formula formula(const Fixture& f){
  at::AutoGradMode yes(true);Formula out;out.old.resize(R);out.proposal.resize(R);out.final.resize(B*N);
  auto leaf=[&](Tensor x,std::string kind,std::string field,int64_t row){x=x.clone().set_requires_grad(true);out.leaves.push_back(x);out.labels.push_back({kind,field,row});return x;};
  std::vector<Tensor> h;for(int64_t row=0;row<R;++row)h.push_back(leaf(f.content[row],"content","",row));
  std::vector<std::vector<Tensor>> w(Banks);auto inputs=roles(f.parameters);
  for(int64_t b=0;b<Banks;++b)for(int j=0;j<4;++j)w[b].push_back(leaf(inputs[j][b],"parameter",std::to_string(j),b));
  for(int64_t owner=0;owner<B*N;++owner){State s=f.states[owner];
    s.value=leaf(s.value,"initial","value",owner);
    for(auto& [name,value]:s.slots)if(name!="times")value=leaf(value,"initial",name,owner);
    for(int64_t l=0;l<L;++l){const auto row=((owner/N)*L+l)*N+owner%N;out.old[row]=s;
      if(f.present.reshape({R})[row].item<bool>()&&h[row][0].detach().item<float>()>0){
        auto u=h[row].slice(0,1),normal=u/(u.square().mean()+f.config.epsilon).sqrt();
        const auto b=f.owners.reshape({R})[row].item<int64_t>(),time=f.times.reshape({R})[row].item<int64_t>();
        auto kv=at::mv(w[b][0],normal)+at::mv(w[b][1],at::mv(w[b][2],normal));auto summary=at::mv(w[b][3],normal);
        auto keep=at::nonzero((time-s.slots["times"])<f.config.window).flatten();
        s.slots["times"]=at::cat({s.slots["times"].index_select(0,keep),at::tensor({time},at::kLong)});
        for(const auto& name:std::vector<std::string>{"keys","values","summaries","weights"}){
          auto value=name=="keys"?kv.slice(0,0,2):name=="values"?kv.slice(0,2,4):name=="summaries"?summary:h[row][0].clamp_max(1);
          s.slots[name]=at::cat({s.slots[name].index_select(0,keep),value.unsqueeze(0)});}
        s.last_time=time;++s.observations;
      }
      out.proposal[row]=s;
    }out.final[owner]=s;
  }
  return out;
}
void compare(const Fixture& f,const Formula& ref,const BlockKVSequenceTape& t,const BlockKVSequenceGradient& g,
    const BlockKVSequenceRoots& roots){
  at::AutoGradMode yes(true);const std::vector<const std::vector<State>*> states{&ref.old,&ref.proposal,&ref.final};
  const std::vector<WeightedRawState> actual{t.old,t.proposal,t.final};
  const std::vector<Tensor> bars{roots.old.cpu(),roots.proposal.cpu(),roots.final.cpu()},flags{roots.old_connected.cpu(),roots.proposal_connected.cpu(),roots.final_connected.cpu()};
  std::vector<Tensor> outputs,cotangents;
  for(int part=0;part<3;++part){auto values=actual[part].floating.cpu(),integer=actual[part].integer.cpu();
    for(size_t row=0;row<states[part]->size();++row){const auto& s=(*states[part])[row];
      require(integer[row][0].item<int64_t>()==s.last_time&&integer[row][1].item<int64_t>()==s.observations,"sequence lost exact clock/count");
      const auto size=s.slots.at("times").numel();require(integer[row][2].item<int64_t>()==size&&at::equal(integer[row].slice(0,4,4+size),s.slots.at("times")),"sequence cache times differ");
      int64_t field=0;for(const auto& p:f.layout.fields()){
        const auto expected=p.name=="value"?s.value:s.slots.at(p.name);close(values[row].slice(0,p.offset,p.offset+expected.numel()).reshape(expected.sizes()),expected);
        if(flags[part][row][field].item<bool>()){outputs.push_back(expected);cotangents.push_back(bars[part][row].slice(0,p.offset,p.offset+expected.numel()).reshape(expected.sizes()));}++field;
      }
    }
  }
  const auto gradient=torch::autograd::grad(outputs,ref.leaves,cotangents,true,false,true);auto wr=roles(g.update.parameters);
  for(size_t j=0;j<gradient.size();++j){const auto& label=ref.labels[j];Tensor value,connected;
    if(label.kind=="content"){value=g.update.content[label.row];connected=g.update.content_connected[label.row];}
    else if(label.kind=="parameter"){const int role=std::stoi(label.field);value=wr[role][label.row];connected=(role==3?g.update.summary_connected:g.update.kv_connected)[label.row];}
    else {const auto& part=f.layout.field(label.field);int64_t field=0;for(const auto& p:f.layout.fields()){if(p.name==label.field)break;++field;}
      value=g.initial[label.row].slice(0,part.offset,part.offset+ref.leaves[j].numel()).reshape(ref.leaves[j].sizes());connected=g.initial_connected[label.row][field];}
    require(connected.cpu().item<bool>()==gradient[j].defined(),"sequence None/zero root differs");
    if(gradient[j].defined())close(value,gradient[j]);else require(value.cpu().eq(0).all().item<bool>(),"disconnected sequence gradient nonzero");
  }
}
void check(at::Device device){
  at::NoGradGuard no_grad;Fixture f;auto reference=formula(f);const auto initial=copy_weighted_raw_state(f.initial,device);
  auto content=f.content.to(device),times=f.times.to(device),owners=f.owners.to(device),present=f.present.to(device);
  WeightedKVParameters w;auto wr=roles(f.parameters);w={wr[0].to(device),wr[1].to(device),wr[2].to(device),wr[3].to(device)};
  auto error=at::zeros({1},times.options().dtype(at::kInt));DeviceProgram p(device);p.limit_workspace(1<<28);
  auto t=append_block_kv_sequence(p,f.config,initial,content,times,owners,present,w,error,1<<28);
  const auto count=int64_t(f.layout.fields().size());
  BlockKVSequenceRoots roots{at::zeros_like(t.old.floating),at::ones({R,count},present.options()),
    at::zeros_like(t.proposal.floating),at::ones({R,count},present.options()),at::zeros_like(t.final.floating),at::ones({B*N,count},present.options())};
  roots.old.copy_(values(roots.old.sizes(),.3));roots.proposal.copy_(values(roots.proposal.sizes(),.5));roots.final.zero_();
  auto g=append_block_kv_sequence_vjp(p,t,roots,error,1<<28);p.finish();portable_torch::synchronize(device);p.run();
  require(!error.cpu().item<int>(),"valid KV sequence failed");compare(f,reference,t,g,roots);
  DeviceProgram reverse(device);reverse.limit_workspace(1<<28);
  auto saved=append_block_kv_sequence_vjp(reverse,t,roots,error,1<<28);reverse.finish();
  initial.floating.fill_(NAN);initial.integer.fill_(-99);content.fill_(NAN);times.fill_(-99);owners.fill_(-99);present.zero_();
  for(auto x:roles(w))x.fill_(NAN);portable_torch::synchronize(device);reverse.run();
  require(!error.cpu().item<int>(),"sequence saved reverse reread mutable source");compare(f,reference,t,saved,roots);reverse.close();p.close();
  std::cout<<"block-kv-sequence: passed B=2 L=3 N=2 actual_update=true independent_formula_vjp=true int64=true two_axis_isolation=true saved_reverse=true full_executor=false\n";
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("KV sequence check requires explicit accelerator FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);check(device);return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
