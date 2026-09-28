#include "dispatch.h"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#if PORTABLE_TORCH_ENABLE_NPU
#include <torch_npu/torch_npu.h>
#endif

namespace {
using namespace accelerator_scale;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void test(at::Device device) {
  // Independent host tuple oracle: affected-count priority, score/node ties,
  // negative and near-tie scores, zero counts and int64 values above 2^53.
  const Index rows=12,width=17,large=Index(1)<<60;
  auto scores=at::empty({rows,width},at::TensorOptions().dtype(at::kFloat));
  auto selected=at::empty({rows,width},at::TensorOptions().dtype(at::kLong));
  auto affected=at::empty_like(selected);auto expected=at::empty_like(selected);
  auto s=scores.accessor<float,2>();auto a=selected.accessor<Index,2>();
  auto b=affected.accessor<Index,2>();auto wanted=expected.accessor<Index,2>();
  for(Index row=0;row<rows;++row) {
    std::vector<std::tuple<Index,Index,float,Index>> order;
    for(Index node=0;node<width;++node) {
      s[row][node]=row==0?0.f:float((node*7+row)%5)-2.f;
      a[row][node]=row==0?0:(node+row)%3+(row>5?large:0);
      b[row][node]=row==0?0:(node*3+row)%4+(row>8?large:0);
      if(row==1){a[row][node]=b[row][node]=0;s[row][node]=node==1?std::nextafter(1.f,2.f):1.f;}
      order.emplace_back(a[row][node],-b[row][node],-s[row][node],node);
    }
    std::sort(order.begin(),order.end());
    for(Index i=0;i<width;++i)wanted[row][i]=std::get<3>(order[i]);
  }
  require(at::equal(rank_candidates(scores.to(device),selected.to(device),affected.to(device)).to(at::kCPU),expected),"lexicographic ranking/tie/int64 mismatch");
  auto invalid=scores.clone();invalid[0][0]=std::numeric_limits<float>::quiet_NaN();bool rejected=false;
  try{rank_candidates(invalid.to(device),selected.to(device),affected.to(device));}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"nonfinite descriptor accepted");
  auto x=at::arange(12,at::TensorOptions().dtype(at::kFloat).device(device)).set_requires_grad(true);
  TensorEventQueue queue(device);std::vector<Atom> oracle;
  for(Index i=0;i<12;++i) {
    // Parallel edges remain distinct; large positions/times cannot be cast to float.
    Atom atom{i%2,(i*3)%5,large+3*(i%3),1,i,large+i,x[i]};
    queue.push(atom);oracle.push_back(atom);
  }
  Continuation snapshot;queue.export_pending(snapshot);
  require(snapshot.pending.size()==12,"queue snapshot inventory");
  std::vector<Atom> arrived;require(!queue.pop(large,arrived),"stop is exclusive");
  std::vector<Tensor> used;
  for(Index time=large;time<=large+6;time+=3) {
    require(queue.pop(time+1,arrived),"missing ready time");
    std::vector<Atom> expected_atoms;for(const auto& a:oracle)if(a.time==time)expected_atoms.push_back(a);
    std::sort(expected_atoms.begin(),expected_atoms.end(),[](const auto& a,const auto& b){return a.key()<b.key();});
    require(arrived.size()==expected_atoms.size(),"queue candidate count");
    for(size_t i=0;i<arrived.size();++i) {
      require(arrived[i].key()==expected_atoms[i].key(),"queue exact order/key");
      used.push_back(arrived[i].value);
    }
  }
  require(!queue.pop(large+10,arrived),"queue failed to drain");
  at::stack(used).sum().backward();require(at::equal(x.grad(),at::ones_like(x)),"queue changed payload VJP");
  queue.export_pending(snapshot);require(snapshot.pending.empty(),"stale pending");
  std::cout<<"analytic ranking ties/int64, queue time/order/parallel-edge identity and payload VJP: passed\n";
}
}
int main(int argc,char** argv) {
  bool npu=false;int code=0;
  try {
    portable_torch::RuntimeOptions options;options.device_spec=argc>1?argv[1]:"cpu";
    options.dtype=at::kFloat;
    const auto device=portable_torch::resolve_device(options);npu=device.type()==c10::DeviceType::PrivateUse1;
    at::set_num_threads(1);at::set_num_interop_threads(1);test(device);
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';code=1;}
#if PORTABLE_TORCH_ENABLE_NPU
  if(npu)torch_npu::finalize_npu();
#endif
  return code;
}
