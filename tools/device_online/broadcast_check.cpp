#include "broadcast_router.h"
#include "queue_transaction.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>

namespace {
using namespace tide::device_online;
using I=int64_t;
void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
void check(at::Device device,at::ScalarType dtype) {
  at::NoGradGuard guard;
  constexpr I nodes=4,samples=2,rows=6,width=7,capacity=24;
  const auto opts=at::TensorOptions().device(device).dtype(dtype),longs=opts.dtype(at::kLong);
  const std::vector<std::vector<Wire>> graphs{
    {},{{2,1,3},{0,2,2},{0,2,2},{1,0,4},{2,2,1},{3,0,7}},
    {{0,1,1},{1,2,7},{2,3,2}},{{3,3,1},{0,0,2},{1,1,3},{2,2,4}}};
  I cases=0;
  for(const auto& graph:graphs) {
    BroadcastRouter router(nodes,samples,graph,capacity,device);
    QueueTransaction queue(capacity,width,nodes,samples,opts);
    ActionBatch action{at::zeros({rows,4},longs),at::zeros({rows,width},opts),at::zeros({rows},opts.dtype(at::kBool))};
    auto scales=at::zeros({std::max<I>(1,graph.size()),1},opts),consumed=at::ones({capacity},opts.dtype(at::kInt));
    CannProgram program(device);auto incoming=router.append_stage(program,action,scales,queue.error());
    queue.append_stage(program,consumed,incoming);program.finish();
    for(I round=0;round<8;++round) {
      std::vector<I> coordinates;std::vector<uint8_t> selected;
      for(I i=0;i<rows;++i) {
        coordinates.insert(coordinates.end(),{i%samples,(i+round)%nodes,(I(1)<<55)+round*3+i,round*rows+i});
        selected.push_back(round!=7&&(i+round)%4!=0);
      }
      auto values=(at::arange(rows*width,at::kFloat).reshape({rows,width})+round).to(dtype);values[0].zero_();
      auto edge_scales=at::zeros({std::max<I>(1,graph.size()),1},dtype);
      for(I e=0;e<I(graph.size());++e)edge_scales[e].fill_(e%3==0?0:e%3==1?0.5:-1);
      std::vector<I> expected_coordinates;std::vector<at::Tensor> expected_values;
      for(I i=0;i<rows;++i)if(selected[i])for(I e=0;e<I(graph.size());++e) {
        const auto [source,target,delay]=graph[e];
        if(source!=coordinates[i*4+1])continue;
        expected_coordinates.insert(expected_coordinates.end(),{coordinates[i*4],target,coordinates[i*4+2]+delay,1,e,coordinates[i*4+3]});
        expected_values.push_back(values[i]*edge_scales[e]);
      }
      // Poison inactive actions: they must not become candidates or touch a
      // node's numerical path merely to be multiplied by zero afterward.
      for(I i=0;i<rows;++i)if(!selected[i]) {
        for(I j=0;j<4;++j)coordinates[i*4+j]=-99;
        values[i].fill_(std::numeric_limits<double>::quiet_NaN());
      }
      action.coordinates.copy_(at::tensor(coordinates,at::kLong).reshape({rows,4}));
      action.values.copy_(values);action.valid.copy_(at::tensor(selected,at::kByte).to(at::kBool));scales.copy_(edge_scales);
      portable_torch::synchronize(device);program.run();
      require(queue.error().cpu().item<int>()==0,"broadcast rejected legal inputs");
      auto got=queue.atoms();const I count=expected_values.size();
      require(queue.stats().cpu()[0].item<I>()==count,"broadcast message count differs");
      require(at::equal(got.valid.cpu(),at::arange(capacity,at::kLong)<count),"broadcast validity differs");
      if(count) {
        require(at::equal(got.coordinates.cpu().narrow(0,0,count),at::tensor(expected_coordinates,at::kLong).reshape({count,6})),"broadcast identity,delay or stable physical edge order differs");
        require(at::equal(got.values.cpu().narrow(0,0,count),at::stack(expected_values)),"broadcast gathered/scaled payload differs");
      }
      require(got.values.cpu().narrow(0,count,capacity-count).eq(0).all().item<bool>(),"broadcast padding retained invalid data");
      ++cases;
    }
    if(!graph.empty()) {
      // Produce beyond int64 maximum: preserve the complete old queue. A seal
      // is not a license to discard messages that arrive in a future window.
      action.coordinates.zero_();action.valid.zero_();action.valid[0].fill_(true);
      action.coordinates[0][1].fill_(std::get<0>(graph[0]));
      action.coordinates[0][2].fill_(std::numeric_limits<I>::max());
      auto before=queue.atoms();auto values=before.values.cpu(),coords=before.coordinates.cpu(),valid=before.valid.cpu(),stats=queue.stats().cpu();
      portable_torch::synchronize(device);program.run();
      require(queue.error().cpu().item<int>()==3,"actual arrival overflow was not refused");
      require(at::equal(values,before.values.cpu())&&at::equal(coords,before.coordinates.cpu())
          &&at::equal(valid,before.valid.cpu())&&at::equal(stats,queue.stats().cpu()),"failed routing committed queue changes");
      ++cases;
    }
    program.close();
  }
  // Distinguish routing-buffer capacity from queue capacity, and check invalid
  // active indices before any access to the topology table.
  for(int invalid=0;invalid<2;++invalid) {
    BroadcastRouter router(nodes,samples,{{0,1,1},{0,1,2}},1,device);
    QueueTransaction queue(capacity,width,nodes,samples,opts);
    ActionBatch action{at::zeros({1,4},longs),at::ones({1,width},opts),at::ones({1},opts.dtype(at::kBool))};
    if(invalid)action.coordinates[0][1].fill_(nodes);
    auto scales=at::ones({2,1},opts),consumed=at::zeros({capacity},opts.dtype(at::kInt));
    CannProgram program(device);auto incoming=router.append_stage(program,action,scales,queue.error());
    queue.append_stage(program,consumed,incoming);program.finish();portable_torch::synchronize(device);program.run();
    require(queue.error().cpu().item<int>()==(invalid?2:1),"routing preflight failure classification differs");
    require(queue.stats().cpu()[0].item<I>()==0&&!queue.atoms().valid.cpu().any().item<bool>(),"failed routing created messages");
    ++cases;program.close();
  }
  std::cout<<"device-broadcast: passed cases="<<cases<<" exact_int64=true physical_parallel_edges=true"
    <<" actual_arrival_overflow=true packed_delivery=true scope=broadcast_routing_and_queue_transaction\n";
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("broadcast check requires explicit NPU and FP32/FP16");
    args.allow_npu_float16=true;auto device=portable_torch::resolve_device(args);
    if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("broadcast check requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);check(device,args.dtype);return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
