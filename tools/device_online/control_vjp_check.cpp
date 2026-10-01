#include "control_vjp.h"
#include "control_forward.h"
#include "full_vjp_fixture.h"
#include "tide/ops.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/custom_function.h>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>

namespace {
using namespace tide;using namespace tide::device_online;
constexpr Index capacity=53,first=2,nodes=8,samples=2;
void forward(at::Device device,Index width,at::ScalarType dtype) {
  at::NoGradGuard guard;Graph g;g.nodes={{0},{0},{0}};g.nodes[1].identity=true;g.nodes[2].full="identity";
  g.regions={{1,false,true,"content"}};g.inputs={0,1,2};g.outputs={0,1,2};g.compile();Model m;
  for(int n=0;n<3;++n)m.nodes.push_back({at::zeros({width},dtype),at::eye(width,at::TensorOptions().dtype(dtype)),
    at::zeros({width},dtype),at::ones({width},dtype)});
  m.input_scale={at::ones({},dtype),at::ones({},dtype),at::ones({},dtype)};m.output_scale=m.input_scale;
  ContentProfile profile(g,m,device);constexpr Index rows=9;
  const auto f=at::TensorOptions().dtype(dtype),l=at::TensorOptions().dtype(at::kLong);
  auto coords=at::zeros({rows,4},l),valid=at::ones({rows},at::kBool);
  auto h=(at::arange(rows*width,at::kFloat).reshape({rows,width}).remainder(19)*.021-.143).to(dtype);
  auto fresh=(at::arange(rows*width,at::kFloat).reshape({rows,width}).remainder(13)*.039+.027).to(dtype);
  auto controls=at::arange(rows,at::kFloat)*.093125f+.1171875f;
  coords.select(1,1).copy_(at::arange(rows,l).remainder(3));
  // Valid identity nodes bypass poisoned content/controls. Invalid rows bypass
  // even poisoned coordinates; the candidate retains their copied payload.
  for(Index row=0;row<rows;++row) {
    if(row%3==1){h[row].fill_(NAN);controls[row].fill_(NAN);}
    if(row%3==2)fresh[row].copy_(h[row]); // Ordinary identity Full still goes through Emit.
  }
  coords[rows-1].fill_(std::numeric_limits<Index>::max());valid[rows-1].fill_(false);h[rows-1].fill_(NAN);fresh[rows-1].fill_(NAN);
  ActionBatch actions{coords.to(device),fresh.to(device),valid.to(device)};
  auto content=h.to(device),p=controls.to(device),error=at::zeros({1},f.device(device).dtype(at::kInt));
  CannProgram program(device);auto result=append_control_forward(program,profile,actions,content,p,error);program.finish();
  for(int replay=0;replay<3;++replay) {
    if(replay==1){controls[0].fill_(.891234f);p.copy_(controls);valid[3].fill_(false);actions.valid.copy_(valid);}
    if(replay==2)error.fill_(3);
    auto expected=fresh.clone();
    if(replay!=2)for(Index row=0;row<rows;++row)if(valid[row].item<bool>()&&row%3!=1)
      expected[row].copy_(h[row]+controls[row].to(dtype)*(fresh[row]-h[row]));
    portable_torch::synchronize(device);program.run();auto actual=result.values.cpu();
    if(actual.scalar_type()!=dtype||!at::allclose(actual,expected,dtype==at::kHalf?0:1e-5,dtype==at::kHalf?0:1e-6,true))
      throw std::runtime_error("selected Emit forward rounding or bypass differs");
    if(error.cpu().item<int>()!=(replay==2?3:0))throw std::runtime_error("Emit forward changed error state");
  }
}
Tensor rounded(const Tensor& x){return x+(x.detach().to(at::kHalf).to(x.scalar_type())-x.detach());}
class HalfHst : public torch::autograd::Function<HalfHst> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor h,Tensor g,Tensor p,double zeta) {
    ctx->set_materialize_grads(false);ctx->save_for_backward({rounded(g-h)});ctx->saved_data["zeta"]=zeta;return g.clone();
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list u) {
    if(!u[0].defined())return {Tensor(),Tensor(),Tensor(),Tensor()};
    return {at::zeros_like(u[0]),u[0],(u[0]*ctx->get_saved_variables()[0]).sum()*ctx->saved_data["zeta"].toDouble(),Tensor()};
  }
};
struct Fixture {Graph graph;Tensor meta,values,raw,read,gradient,on;Index width;int mode,roots;double zeta;bool half;};
Fixture fixture(Index width,int mode,int roots,double zeta,at::ScalarType dtype) {
  const float nan=std::numeric_limits<float>::quiet_NaN();Fixture f;f.width=width;f.mode=mode;f.roots=roots;f.zeta=zeta;
  f.half=dtype==at::kHalf;
  f.meta=at::zeros({capacity,13},at::kLong);f.values=at::full({capacity,5*width+2},nan,at::kFloat);
  f.raw=at::full({capacity,width},nan,at::kFloat);f.read=at::full({nodes,width},nan,at::kFloat);
  f.gradient=at::full({capacity,width},nan,at::kFloat);f.on=at::zeros({capacity},at::kBool);
  for(Index r=0;r<4;++r)f.graph.regions.push_back({1,false,true,r%3==0?"content":r%3==1?"old":"proposal"});
  for(Index n=0;n<nodes;++n) {
    Node spec;spec.region=n/2;spec.identity=n==6;spec.readout=n%2?"norm-fp32-v1":"linear-v1";f.graph.nodes.push_back(spec);
    if(n==2||n==4||n==5)f.graph.nodes.back().full="identity";
    if(n%2==0&&!spec.identity)f.read[n].copy_(at::arange(width,at::kFloat).remainder(7)*.007-.019);
  }
  f.graph.compile();
  for(Index row=0;row<48;++row) {
    const Index b=row/24,n=(row/3)%nodes,time=(Index(1)<<55)+row%3;
    const bool active=n%2==0||row%3==1;
    f.meta[first+row].narrow(0,0,4).copy_(at::tensor({b,n,time,Index(active)},at::kLong));
    for(Index field=0;field<3;++field)f.values[first+row].narrow(0,field*width,width).copy_(
      at::arange(width,at::kFloat).remainder(11)*.003+.01*(field+1)-.002*row);
    if(row==4)f.values[first+row].narrow(0,0,3*width).zero_(); // Differentiable zero norm.
    f.raw[first+row].copy_(row%4?at::arange(width,at::kFloat).remainder(9)*.017-.021:
      f.values[first+row].narrow(0,0,width)); // Exact g==h still connects controls.
    if(f.graph.nodes[n].full=="identity")f.raw[first+row].copy_(f.values[first+row].narrow(0,0,width));
    const bool on=roots&&active&&(row%4!=1);
    if(on){f.on[row].fill_(true);f.gradient[row].copy_(roots==2?at::zeros({width},at::kFloat):
      at::arange(width,at::kFloat).remainder(5)*.03125-.0625);}
  }
  if(f.half) {
    f.read=f.read.to(at::kHalf);f.raw=f.raw.to(at::kHalf).to(at::kFloat);
    f.values=f.values.to(at::kHalf).to(at::kFloat);f.gradient.mul_(256.f);
  }
  f.on[capacity-1].fill_(true); // Stale padding never becomes a root.
  return f;
}
using Groups=std::map<std::array<Index,3>,std::vector<Index>>;
Groups groups(const Fixture& f,Index count) {
  Groups out;for(Index row=0;row<count;++row) {
    auto e=f.meta[first+row];const auto n=e[1].item<Index>();
    out[{e[0].item<Index>(),f.graph.nodes[n].region,e[2].item<Index>()}].push_back(row);
  }return out;
}
Index read_mode(const Fixture& f,Index node) {
  const auto& m=f.graph.regions[f.graph.nodes[node].region].read_mode;return m=="content"?0:m=="old"?1:2;
}
void probabilities(Fixture& f,Index count) {
  for(const auto& [_,rows]:groups(f,count)) {
    std::vector<Tensor> scores;
    for(auto row:rows) {
      const auto node=f.meta[first+row][1].item<Index>();auto x=f.values[first+row].narrow(0,read_mode(f,node)*f.width,f.width);
      auto score=f.graph.nodes[node].identity?at::zeros({},at::kFloat):node%2?x.norm():(x*f.read[node]).sum();
      f.values[first+row][5*f.width].copy_(score);scores.push_back(score);
    }
    auto p=at::softmax(at::stack(scores),0);for(size_t i=0;i<rows.size();++i)f.values[first+rows[i]][5*f.width+1].copy_(p[i]);
  }
}
void compare(const Fixture& f,Index count,const ControlVjp& out,at::ScalarType dtype) {
  at::AutoGradMode enabled(true);std::vector<Tensor> events,fresh,weights,leaves,loss;
  auto leaf=[&](const Tensor& x){auto y=x.detach().to(dtype).clone().set_requires_grad(true);leaves.push_back(y);return y;};
  for(Index row=0;row<capacity;++row)for(Index field=0;field<5;++field)
    events.push_back(leaf(row<count?f.values[first+row].narrow(0,field*f.width,f.width):at::zeros({f.width},at::kFloat)));
  for(Index row=0;row<capacity;++row)fresh.push_back(leaf(row<count?f.raw[first+row]:at::zeros({f.width},at::kFloat)));
  for(Index n=0;n<nodes;++n)weights.push_back(leaf(f.read[n]));
  for(const auto& [_,rows]:groups(f,count)) {
    std::vector<Tensor> scores;
    for(auto row:rows) {
      const auto n=f.meta[first+row][1].item<Index>();auto x=events[row*5+read_mode(f,n)];
      scores.push_back(f.graph.nodes[n].identity?at::zeros({},x.options()):n%2?x.norm():(x*weights[n]).sum());
    }
    auto p=at::softmax(at::stack(scores),0);
    for(size_t i=0;i<rows.size();++i)if(f.on[rows[i]].item<bool>()) {
      const auto row=rows[i],n=f.meta[first+row][1].item<Index>();auto h=events[row*5],g=fresh[row];
      auto y=f.graph.nodes[n].identity?g:f.half?
        (f.mode==1?HalfHst::apply(h,g,rounded(p[i]),f.zeta):rounded(h+rounded(rounded(p[i])*rounded(g-h)))):
        emit(h,g,p[i],f.mode==1?"hst":"softp",f.zeta);
      loss.push_back((y*f.gradient[row].to(dtype)).sum());
    }
  }
  std::vector<Tensor> grads(leaves.size());if(!loss.empty())grads=torch::autograd::grad({at::stack(loss).sum()},leaves,{},false,false,true);
  auto values=out.events.cpu(),flags=out.connected.cpu(),raw=out.fresh.cpu(),read=out.read.cpu(),rc=out.read_connected.cpu();
  for(const auto& x:{values,raw,read})if(x.scalar_type()!=at::kFloat)throw std::runtime_error("control adjoints must stay FP32");
  for(Index row=0;row<capacity;++row) {
    for(Index field=0;field<5;++field)test::full_same_precision(values[row][field],flags[row][field],grads[row*5+field],"control event",f.half);
    test::full_same_precision(raw[row],at::scalar_tensor(row<count&&f.on[row].item<bool>(),at::kBool),grads[capacity*5+row],"fresh Full",f.half);
  }
  for(Index n=0;n<nodes;++n)test::full_same_precision(read[n],rc[n],grads[capacity*6+n],"Read owner",f.half);
}
void check(at::Device device,Index width,int mode,int roots,double zeta,at::ScalarType dtype) {
  at::NoGradGuard guard;auto f=fixture(width,mode,roots,zeta,dtype);
  StateTape t;t.metadata=f.meta.to(device);t.values=f.values.to(device);t.samples=samples;
  ControlTape c{f.read.to(device),f.raw.to(device),mode,zeta};
  auto count=at::zeros({1},t.metadata.options()),range=at::tensor({first,first},at::kLong).to(device);
  auto gradient=f.gradient.to(device),on=f.on.to(device),error=at::zeros({1},gradient.options().dtype(at::kInt));
  CannProgram p(device);auto out=append_control_vjp(p,f.graph,t,c,count,range,gradient,on,error,32*1024*1024);p.finish();
  for(Index size:{48,17,0}) {
    probabilities(f,size);t.values.copy_(f.values);count.fill_(size);range[1].fill_(first+size);
    portable_torch::synchronize(device);p.run();
    if(error.cpu().item<int>())throw std::runtime_error("valid control tape refused");
    compare(f,size,out,at::kFloat);compare(f,size,out,at::kDouble);
  }
  count.fill_(capacity);portable_torch::synchronize(device);p.run();
  if(error.cpu().item<int>()!=2||out.connected.cpu().any().item<bool>())throw std::runtime_error("invalid control tape exposed adjoints");
  bool refused=false;try{CannProgram bad(device);append_control_vjp(bad,f.graph,t,c,count,range,gradient,on,error,1);}
  catch(const std::invalid_argument&){refused=true;}if(!refused)throw std::runtime_error("control budget ignored");
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("control VJP gate requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;
    const auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    for(Index width:{1,7,257})forward(device,width,args.dtype);
    for(Index width:{1,7,257})for(int mode:{1,2})for(int roots:{0,1,2})for(double zeta:{0.,.375}) {
      try{check(device,width,mode,roots,zeta,args.dtype);++cases;}catch(...){std::cerr<<"control VJP width="<<width<<" mode="<<mode<<" roots="<<roots<<" zeta="<<zeta<<'\n';throw;}
    }
    std::cout<<"device-control-vjp: passed cases="<<cases<<" replays="<<cases*3<<" forward_cases=3 forward_replays=9 CPU=FP32_FP64 complete_frames=true None_zero=true\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
