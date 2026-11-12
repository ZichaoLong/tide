#include "block_dispatch.h"
#include "device_launch_tide_block_probe.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <functional>
#include <iostream>
#include <limits>

namespace {
using namespace tide::device_online;
using I=int64_t;
uint8_t* address(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
void probe(DeviceProgram& p,const BlockDispatcher& q,const at::Tensor& staged,
           const at::Tensor& output,const at::Tensor& present,I command){
  const auto ids=q.selection(),meta=q.metadata(),emitted=q.emitted(),error=q.error();
  const auto rows=ids.numel(),slots=output.size(0),width=output.size(1);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_probe)(1,stream,
    address(ids),address(meta),address(staged),address(emitted),address(output),address(present),address(error),
    command,rows,slots,width),"resident block numerical probe");},
    {ids,meta,staged,emitted,output,present,error});
}
void check(at::Device device){
  at::NoGradGuard guard;
  const I base=(I(1)<<55)+17,slots=8,rows=4,width=2;
  BlockDispatchPlan plan;
  plan.groups=2;plan.batch_capacity=rows;plan.journal_capacity=slots;plan.tensor_bytes=1<<20;
  plan.tasks={{{0,0,0,base,0}},{{0,1,0,base,0}},{{1,0,1,base+7,0}},{{1,1,1,base+7,0}},
              {{0,0,0,base+14,0}},{{1,0,2,base+14,0}},{{0,0,3,base+21,0}},{{0,1,3,base+21,0}}};
  plan.dependencies={{{0,2,1}},{{0,2,1}},{{1,3,1}},{{2,4,1}},{{3,4,1}},{{4,5,0}},{{5,6,1}}};
  BlockDispatcher q(plan,device);
  auto short_plan=plan;short_plan.tensor_bytes=q.retained_bytes()-1;bool refused=false;
  try{BlockDispatcher insufficient(short_plan,device);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"real block metadata budget shortfall was accepted");
  const auto floats=at::TensorOptions().device(device).dtype(at::kFloat);
  auto source=at::zeros({slots+1,width},floats),gathered=at::zeros({rows,width},floats),staged=at::zeros_like(gathered);
  auto output=at::zeros({slots,width},floats),present=at::zeros({slots},floats.dtype(at::kBool));
  auto zero=at::zeros({1},floats.dtype(at::kInt));
  DeviceProgram p(device);p.limit_workspace(1<<26);
  q.append_reset(p);p.zero(output);p.zero(present);
  const auto head=p.label(),finish=p.label(),commit=p.label(),group0=p.label(),group1=p.label();
  p.mark(head);q.append_choose(p);p.branch(q.branch(),{commit,group0,group1});
  for(I group=0;group<2;++group){
    p.mark(group==0?group0:group1);
    // A whole chosen block reaches one batch matmul; the row count is padded
    // to declared capacity and never converted to a host scalar in the loop.
    auto matrix=at::eye(width,floats).mul(group==0?1:-1).reshape({1,width,width});
    p.index_select(source,0,q.selection(),gathered);
    p.batch_matmul(gathered.reshape({1,rows,width}),matrix,staged.reshape({1,rows,width}));
    probe(p,q,staged,output,present,0);p.branch(zero,{commit});
  }
  p.mark(commit);q.append_commit(p);probe(p,q,staged,output,present,1);
  p.branch(q.branch(),{finish,head});p.mark(finish);p.finish();
  for(I round=0;round<4;++round){
    auto values=at::arange((slots+1)*width,at::kFloat).reshape({slots+1,width});
    if(round==1)values.neg_(); // A different numerical result changes the device schedule.
    std::vector<uint8_t> seed(slots,0);if(round!=2){seed[0]=seed[1]=seed[5]=seed[7]=1;}
    auto seeds=at::zeros({slots},at::kBool);for(I v=0;v<slots;++v)seeds[v]=bool(seed[v]);
    q.seeds().copy_(seeds);q.epoch().fill_(round+1);source.copy_(values);
    // Reference is a scalar recursive DAG evaluation, independent of grouping.
    std::vector<I> status(slots,-1);std::vector<uint8_t> live(slots,0);
    auto expected=at::zeros({slots,width},at::kFloat);
    std::function<void(I)> visit=[&](I v){
      if(status[v]>=0)return;bool enabled=seed[v];
      for(const auto& e:plan.dependencies)if(e[1]==v){visit(e[0]);enabled|=e[2]&&live[e[0]];}
      status[v]=enabled?2:3;if(!enabled)return;
      expected[v].copy_(values[v].mul(plan.tasks[v][0]==0?1:-1));
      live[v]=expected[v][0].item<float>()>=0;
    };
    for(I v=0;v<slots;++v)visit(v);
    portable_torch::synchronize(device);p.run();
    require(q.error().cpu().item<int32_t>()==0,"device block dispatcher refused valid DAG");
    require(at::equal(q.state().cpu().select(1,0),at::tensor(status,at::kLong)),"device closure/status differs");
    require(at::equal(output.cpu(),expected),"vendor block numerical values differ");
    auto actual=present.cpu();for(I v=0;v<slots;++v)require(actual[v].item<bool>()==bool(live[v]),"zero/absent message distinction lost");
    const auto used=q.ledger().cpu()[2].item<I>();I count=0;for(auto s:status)count+=s==2;
    require(used==count,"actual journal contains missing or stale tasks");
    require(q.tasks().cpu()[0][3].item<I>()==base,"logical time passed through floating conversion");
  }
  // An actual numerical guard must leave the first transaction unpublished.
  q.epoch().fill_(5);source[0][0]=std::numeric_limits<float>::quiet_NaN();
  portable_torch::synchronize(device);p.run();
  require(q.error().cpu().item<int32_t>()==74&&q.ledger().cpu()[2].item<I>()==0,
          "numerical failure published a scheduler prefix");
  require(output.cpu().eq(0).all().item<bool>(),"numerical failure published staged payload");
  p.close();
  std::cout<<"resident-block: passed device_ready=true device_presence=true batched_vendor=true exact_int64=true"
    <<" host_loop=false scope=metadata_and_numerical_stage_boundary\n";
}
}
int main(int argc,char** argv){
  portable_torch::RuntimeSession runtime;
  try{
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("block check requires explicit resident device and float32");
    auto device=portable_torch::resolve_device(args);
    if(device.type()!=resident_device_type)throw std::invalid_argument("block check requires resident device");
    at::set_num_threads(1);at::set_num_interop_threads(1);check(device);runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
