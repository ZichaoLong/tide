#include "ready_batch.h"
#include "queue_transaction.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <map>

namespace {
using namespace tide::device_online;
using I=int64_t;
using Key=std::array<I,3>;
void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
void integers(const at::Tensor& actual,const std::vector<I>& values,const char* message) {
  auto expected=at::tensor(values,at::kLong);
  require(at::equal(actual.cpu().reshape({-1}).narrow(0,0,values.size()),expected),message);
}
void compare(const ReadyBatch& got,const AtomBatch& input,const at::Tensor& ready,
             const std::vector<I>& owners) {
  const auto coords=input.coordinates.accessor<I,2>();
  std::vector<I> order;
  for(I i=0;i<input.valid.numel();++i)if(ready[i].item<bool>())order.push_back(i);
  std::stable_sort(order.begin(),order.end(),[&](I a,I b) {
    for(I j=0;j<6;++j)if(coords[a][j]!=coords[b][j])return coords[a][j]<coords[b][j];
    return false;
  });
  std::vector<Key> fibers;
  std::vector<I> fiber_offsets;
  std::map<Key,std::vector<I>> frame_map;
  for(I i=0;i<I(order.size());++i) {
    const I row=order[i];Key fiber{coords[row][0],coords[row][1],coords[row][2]};
    if(fibers.empty()||fibers.back()!=fiber) {
      frame_map[{fiber[0],owners[fiber[1]],fiber[2]}].push_back(fibers.size());
      fibers.push_back(fiber);fiber_offsets.push_back(i);
    }
  }
  fiber_offsets.push_back(order.size());
  std::vector<I> frames,frame_offsets,frame_fibers,fiber_rows(fibers.size()*4);
  I frame=0;
  for(const auto& [key,rows]:frame_map) {
    frame_offsets.push_back(frame_fibers.size());frames.insert(frames.end(),key.begin(),key.end());
    for(auto row:rows) {
      frame_fibers.push_back(row);
      for(I j=0;j<3;++j)fiber_rows[row*4+j]=fibers[row][j];fiber_rows[row*4+3]=frame;
    }
    ++frame;
  }
  frame_offsets.push_back(frame_fibers.size());
  integers(got.counts,{I(order.size()),I(fibers.size()),I(frame_map.size())},"ready counts differ");
  require(at::equal(got.consumed.cpu().to(at::kBool),ready),"ready mask differs");
  require(got.branch.cpu().item<int>()==int(!order.empty()),"ready branch differs");
  require(at::equal(got.atoms.valid.cpu(),at::arange(input.valid.numel(),at::kLong)<I(order.size())),"ready validity differs");
  if(!order.empty()) {
    auto indices=at::tensor(order,at::kLong);
    require(at::equal(got.atoms.coordinates.cpu().narrow(0,0,order.size()),input.coordinates.index_select(0,indices)),"canonical atom order differs");
    require(at::equal(got.atoms.values.cpu().narrow(0,0,order.size()),input.values.index_select(0,indices)),"packed atom values differ");
    integers(got.fiber_offsets,fiber_offsets,"fiber offsets differ");integers(got.fibers,fiber_rows,"fiber/frame ownership differs");
    integers(got.frame_offsets,frame_offsets,"frame offsets differ");integers(got.frame_fibers,frame_fibers,"frame candidates differ");
    integers(got.frames,frames,"frame keys differ");
    require(got.atoms.values.cpu().narrow(0,order.size(),input.valid.numel()-order.size()).eq(0).all().item<bool>(),"invalid payload leaked into ready padding");
  }
}
void check(at::Device device,at::ScalarType dtype) {
  at::NoGradGuard guard;
  constexpr I capacity=32,width=7,samples=2,regions=3;
  const std::vector<I> owners{1,0,1,2,0};const I nodes=owners.size();
  const std::vector<std::vector<Wire>> graphs{
    {},{{0,1,2},{1,0,3},{0,1,2},{2,3,1},{3,2,2}},{{4,1,3},{1,2,2},{2,3,7}},{{2,2,1},{0,4,4},{4,0,1}}};
  const auto opts=at::TensorOptions().device(device).dtype(dtype),longs=opts.dtype(at::kLong);
  I cases=0;
  for(const auto& graph:graphs)for(bool prefill:{false,true}) {
    DeviceReady planner(owners,regions,graph,samples,device,prefill);
    QueueClosure reference(owners,regions,graph,samples,at::Device(at::kCPU));
    AtomBatch queue{at::zeros({capacity,6},longs),at::zeros({capacity,width},opts),at::zeros({capacity},opts.dtype(at::kBool))};
    auto stop=at::zeros({1},longs),error=at::zeros({1},opts.dtype(at::kInt));
    CannProgram program(device);auto packed=planner.append_stage(program,queue,stop,error);program.finish();
    for(I round=0;round<6;++round) {
      I base=round==3?(I(1)<<55)+19:round==4?std::numeric_limits<I>::max()-32:0;
      std::vector<I> c;
      for(I i=0;i<capacity;++i) {
        // Repeated exact tags exercise stable ties; other rows contain several
        // physical sources in one fiber and several nodes in one frame.
        I j=i/2;c.insert(c.end(),{j%2,(j/2)%nodes,base+(j/4+round)%7,i%2,j%3,j%4});
      }
      for(I field=0;field<6;++field)c[6+field]=c[field];
      auto coords=at::tensor(c,at::kLong).reshape({capacity,6});
      auto values=(at::arange(capacity*width,at::kFloat).reshape({capacity,width})-20).to(dtype);
      auto valid=at::arange(capacity,at::kLong).remainder(5)!=round%5;
      if(round==5)valid.zero_();
      values.index_put_({~valid},std::numeric_limits<double>::quiet_NaN());
      coords.index_put_({~valid},-7);
      const auto cpu_stop=at::full({1},round==2?0:base+6,at::kLong);
      AtomBatch input{coords,values,valid};auto expected=reference.ready(input,cpu_stop,prefill);
      queue.coordinates.copy_(coords);queue.values.copy_(values);queue.valid.copy_(valid);stop.copy_(cpu_stop);
      portable_torch::synchronize(device);program.run();
      require(error.cpu().item<int>()==0,"ready stage refused valid inputs");compare(packed,input,expected,owners);++cases;
    }
    program.close();
  }
  // One submission drains only ready work; device loop discovers each next
  // frontier after queue replacement. This is a no-emission scheduling probe,
  // not a claim of complete graph-module execution.
  for(bool prefill:{false,true}) {
    const auto graph=graphs[1];
    DeviceReady planner(owners,regions,graph,samples,device,prefill);
    QueueClosure reference(owners,regions,graph,samples,at::Device(at::kCPU));
    QueueTransaction queue(capacity,width,nodes,samples,opts);
    AtomBatch empty{at::zeros({1,6},longs),at::zeros({1,width},opts),at::zeros({1},opts.dtype(at::kBool))};
    auto stop=at::full({1},4,longs),count=at::zeros({1},longs),one=at::ones({1},longs);
    auto budget=at::full({1},capacity,longs),budget_error=at::full({1},4,opts.dtype(at::kInt));
    auto predicate=at::zeros({1},opts.dtype(at::kBool)),index=at::zeros({1},opts.dtype(at::kInt));
    CannProgram program(device);auto head=program.label(),test=program.label(),body=program.label(),exhausted=program.label(),end=program.label();
    program.mark(head);auto packed=planner.append_stage(program,queue.atoms(),stop,queue.error());
    program.branch(packed.branch,{end,test});program.mark(test);program.less(count,budget,predicate);program.cast_index(predicate,index);
    program.branch(index,{exhausted,body});program.mark(body);queue.append_stage(program,packed.consumed,empty);
    program.add(count,one);program.branch(index,{head});program.mark(exhausted);program.add(queue.error(),budget_error);
    program.mark(end);program.finish();
    std::vector<I> coords;
    for(I i=0;i<capacity;++i)coords.insert(coords.end(),{i%2,(i/2)%nodes,i%7,1,i,0});
    auto input=at::tensor(coords,at::kLong).reshape({capacity,6});auto values=at::arange(capacity*width,at::kFloat).reshape({capacity,width}).to(dtype);
    queue.atoms().coordinates.copy_(input);queue.atoms().values.copy_(values);queue.atoms().valid.fill_(true);queue.stats().fill_(capacity);
    PackedQueue cpu(capacity,width,opts.device(at::kCPU));cpu.append({input,values,at::ones({capacity},at::kBool)});
    for(I boundary:{4,7}) {
      I calls=0;auto cpu_stop=at::full({1},boundary,at::kLong);
      while(true){auto mask=reference.ready(cpu.atoms(),cpu_stop,prefill);if(!mask.any().item<bool>())break;
        cpu.replace(mask,{at::zeros({1,6},at::kLong),at::zeros({1,width},dtype),at::zeros({1},at::kBool)});++calls;}
      stop.copy_(cpu_stop);count.zero_();portable_torch::synchronize(device);program.run();
      require(queue.error().cpu().item<int>()==0&&count.cpu().item<I>()==calls,"device loop frontier count differs");
      auto live=cpu.atoms().valid;
      require(at::equal(queue.atoms().valid.cpu(),live)&&at::equal(queue.atoms().coordinates.cpu().index({live}),cpu.atoms().coordinates.index({live}))
          &&at::equal(queue.atoms().values.cpu().index({live}),cpu.atoms().values.index({live})),"device loop lost pending continuation");++cases;
    }
    // Budget exhaustion is a failure with unconsumed state retained.
    queue.atoms().coordinates.copy_(input);queue.atoms().values.copy_(values);queue.atoms().valid.fill_(true);
    budget.zero_();count.zero_();portable_torch::synchronize(device);program.run();
    require(queue.error().cpu().item<int>()==4&&queue.atoms().valid.cpu().all().item<bool>(),"loop budget silently discarded work");
    ++cases;program.close();
  }
  std::cout<<"device-ready: passed cases="<<cases<<" exact_int64=true stable_ties=true complete_fibers=true complete_frames=true"
    <<" device_loop_continuation=true scope=ready_packing_and_no_emission_drain\n";
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("ready check requires explicit NPU and FP32/FP16");
    args.allow_npu_float16=true;auto device=portable_torch::resolve_device(args);
    if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("ready check requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);check(device,args.dtype);runtime.close();return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
