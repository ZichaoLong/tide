#include "weighted_pool.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <iostream>
namespace {
using namespace tide::device_online;using at::Tensor;
constexpr int64_t R=6,C=4;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void close(const Tensor& a,const Tensor& b){require(a.sizes()==b.sizes(),"pool shape mismatch");
  if(!at::allclose(a.cpu(),b,2e-5,2e-6)){std::cerr<<a.cpu()<<" versus "<<b<<'\n';throw std::runtime_error("pool numeric mismatch");}}
Tensor values(at::IntArrayRef shape,float shift){int64_t n=1;for(auto d:shape)n*=d;return (at::arange(n,at::kFloat)*.13+shift).sin().reshape(shape);}
struct Formula {std::vector<Tensor> leaves,output,total;};
Formula reference(const Tensor& weights,const Tensor& values,const Tensor& logits,const Tensor& lengths,const Tensor& present,bool attention){
  at::AutoGradMode enabled(true);Formula f;
  for(int64_t r=0;r<R;++r){auto w=weights[r].clone().set_requires_grad(true),v=values[r].clone().set_requires_grad(true),l=logits[r].clone().set_requires_grad(true);
    f.leaves.insert(f.leaves.end(),{w,v,l});if(!present[r].item<bool>()){f.output.push_back(at::zeros({values.size(2)},at::kFloat));f.total.push_back(at::zeros({},at::kFloat));continue;}
    const auto n=lengths[r].item<int64_t>();w=w.slice(0,0,n);v=v.slice(0,0,n);l=l.slice(0,0,n);auto total=w.sum();f.total.push_back(total);
    Tensor output;if(attention){auto indices=at::nonzero(w>0).flatten();
      output=indices.numel()?total.clamp_max(1)*at::matmul((l.index_select(0,indices)+w.index_select(0,indices).log()).softmax(0),v.index_select(0,indices))
        :at::zeros({values.size(2)},at::kFloat);
    }else output=(w.unsqueeze(1)*v).sum(0)/total.clamp_min(1);
    f.output.push_back(output);
  }return f;
}
void compare(const Formula& f,const Tensor& bars,const Tensor& stats,const Tensor& roots,const WeightedPoolTape& t,const WeightedPoolGradient& g){
  at::AutoGradMode enabled(true);std::vector<Tensor> outputs,cotangents;
  for(int64_t r=0;r<R;++r){close(t.output[r],f.output[r]);close(t.total[r],f.total[r]);
    if(roots[r][0].item<bool>()&&f.output[r].requires_grad()){outputs.push_back(f.output[r]);cotangents.push_back(bars[r]);}
    if(roots[r][1].item<bool>()&&f.total[r].requires_grad()){outputs.push_back(f.total[r]);cotangents.push_back(stats[r]);}}
  auto expected=torch::autograd::grad(outputs,f.leaves,cotangents,true,false,true);auto flags=g.connected.cpu();
  for(int64_t r=0;r<R;++r)for(int j=0;j<3;++j){const auto& e=expected[r*3+j];const auto x=j==0?g.weights:(j==1?g.values:g.logits);
    require(flags[r][j].item<bool>()==e.defined(),"pool independent None/zero root differs");
    if(e.defined())close(x[r],e);else if(x.defined())require(x[r].cpu().eq(0).all().item<bool>(),"pool absent derivative not zero");}
  require(g.logits.defined()==t.attention,"sum pool fabricated logits role");
}
void check(at::Device device,bool attention,int64_t width){
  at::NoGradGuard ng;auto opt=at::TensorOptions().device(device).dtype(at::kFloat);
  auto w=at::zeros({R,C},opt),v=at::zeros({R,C,width},opt),l=at::zeros_like(w),n=at::zeros({R},opt.dtype(at::kLong));
  auto present=at::zeros({R},opt.dtype(at::kBool)),bars=at::zeros({R,width},opt),stats=at::zeros({R},opt),roots=at::zeros({R,2},present.options()),error=at::zeros({1},opt.dtype(at::kInt));
  DeviceProgram p(device);p.limit_workspace(64*1024*1024);auto tape=append_weighted_pool(p,w,v,attention?l:Tensor{},n,present,error);
  auto g=append_weighted_pool_vjp(p,tape,bars,stats,roots,error);p.finish();
  for(int pass=0;pass<2;++pass){auto weights=at::tensor(std::vector<float>{.2,0,.3,0,.4,0,.6,0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0},at::kFloat).reshape({R,C});
    auto inputs=values({R,C,width},.3+pass*.2),scores=values({R,C},.1+pass*.1),lengths=at::tensor(std::vector<int64_t>{3,3,3,pass?2:3,0,-99},at::kLong);
    auto active=at::tensor({1,1,1,1,1,0},at::kInt).to(at::kBool);weights[5].fill_(NAN);inputs[5].fill_(NAN);scores[5].fill_(NAN);
    if(attention){inputs[0][1].fill_(NAN);scores[0][1].fill_(NAN);}
    if(pass)weights[3][1]=.7;
    auto ref=reference(weights,inputs,scores,lengths,active,attention);w.copy_(weights);v.copy_(inputs);l.copy_(scores);n.copy_(lengths);present.copy_(active);
    for(int test=0;test<3;++test){auto flags=at::ones({R,2},at::kBool),dy=values({R,width},.8),dw=values({R},.6);dy[2].zero_();dw[2]=0;
      if(test<2){flags.select(1,1-test).fill_(false);if(test==0)dw.fill_(NAN);else dy.fill_(NAN);}
      dy[5].fill_(NAN);dw[5]=NAN;bars.copy_(dy);stats.copy_(dw);roots.copy_(flags);
      portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"pool rejected valid rows");compare(ref,dy,dw,flags,tape,g);
      if(pass==1&&test==2){DeviceProgram reverse(device);auto saved=append_weighted_pool_vjp(reverse,tape,bars,stats,roots,error);reverse.finish();
        w.fill_(NAN);v.fill_(NAN);l.fill_(NAN);n.fill_(-99);present.zero_();portable_torch::synchronize(device);reverse.run();
        require(!error.cpu().item<int>(),"pool saved reverse reread mutable sources");compare(ref,dy,dw,flags,tape,saved);reverse.close();}
    }
  }
  present.fill_(true);portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==29,"pool malformed actual length accepted");p.close();
  DeviceProgram denied(device);bool refused=false;try{append_weighted_pool(denied,w,v,attention?l:Tensor{},n,present,error,1);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"pool tensor budget omitted");denied.close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("weighted pool requires explicit accelerator/FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    for(bool attention:{false,true})for(int64_t width:{3,129})check(device,attention,width);
    std::cout<<"weighted-pool: passed configurations=4 replay_rows=144 independent_vector_total_roots=true all_vjps=true saved_versions=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
