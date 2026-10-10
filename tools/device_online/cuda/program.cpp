#include "program_internal.h"
#include <c10/cuda/CUDAStream.h>
#include <atomic>
#include <iostream>
#include <unordered_set>

namespace tide::device_online {
using cuda_backend::check;
namespace {std::atomic<bool> quarantined{false};}
DeviceProgram::Impl::Impl(at::Device d):device(d) {
  validate_kernel_device(d);
}
void DeviceProgram::Impl::initialize() {
  c10::cuda::CUDAGuard guard(device);
  cuda_backend::check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create CUDA control stream");
  cuda_backend::check(cudaEventCreateWithFlags(&input_ready,cudaEventDisableTiming),"create CUDA input boundary");
  cuda_backend::check(cudaEventCreateWithFlags(&complete,cudaEventDisableTiming),"create CUDA completion boundary");
  error=at::zeros({1},at::TensorOptions().device(device).dtype(at::kInt));owners.push_back(error);
}
void DeviceProgram::Impl::check() const {
  if(std::this_thread::get_id()!=thread)throw std::logic_error("CUDA program requires its constructing thread");
  if(quarantined.load())throw std::logic_error("CUDA resources quarantined; terminate worker");
  resource.check();
}
void DeviceProgram::Impl::building() const {
  check();if(finished||closed||failed)throw std::logic_error("CUDA program is not open for construction");
}
void DeviceProgram::Impl::buffer(const at::Tensor& t) const {
  if(!t.defined()||t.device()!=device||t.requires_grad()||!t.is_contiguous()||t.numel()<1)
    throw std::invalid_argument("CUDA program requires matching contiguous nonempty no-grad buffers");
  cuda_backend::type(t.scalar_type());
}
void DeviceProgram::Impl::append(std::function<void(cudaStream_t)> fn,const std::vector<at::Tensor>& buffers) {
  building();for(const auto& t:buffers){buffer(t);owners.push_back(t);}
  instructions.push_back({});operations.push_back(std::move(fn));indexes.push_back({});
}
void DeviceProgram::Impl::release() {
  if(closed)return;
  check();c10::cuda::CUDAGuard guard(device);
  if(in_flight){
    if(completion_recorded)cuda_backend::wait_event(complete,10000);
    else cuda_backend::wait_stream(stream,10000);
    in_flight=false;
  }
  if(executable){cuda_backend::check(cudaGraphExecDestroy(executable),"destroy CUDA executable");executable=nullptr;}
  if(graph){cuda_backend::check(cudaGraphDestroy(graph),"destroy CUDA graph");graph=nullptr;}
  if(blas){cuda_backend::check_blas(cublasDestroy(blas),"destroy cuBLAS handle");blas=nullptr;}
  if(input_ready){cuda_backend::check(cudaEventDestroy(input_ready),"destroy CUDA input event");input_ready=nullptr;}
  if(complete){cuda_backend::check(cudaEventDestroy(complete),"destroy CUDA completion event");complete=nullptr;}
  if(stream){cuda_backend::check(cudaStreamDestroy(stream),"destroy CUDA stream");stream=nullptr;}
  operations.clear();owners.clear();indexes.clear();workspace=at::Tensor();error=at::Tensor();closed=true;resource.close();
}
DeviceProgram::DeviceProgram(at::Device d) {
  if(quarantined.load())throw std::logic_error("CUDA resources quarantined; terminate worker");
  impl_=std::make_unique<Impl>(d);
  try {impl_->initialize();}catch(...) {
    try{impl_->release();}catch(...){quarantined.store(true);impl_->resource.quarantine();(void)impl_.release();}
    throw;
  }
}
DeviceProgram::~DeviceProgram() {
  if(!impl_)return;
  try{impl_->release();}catch(const std::exception& e){
    std::cerr<<"CUDA resources quarantined until process exit: "<<e.what()<<'\n';
    quarantined.store(true);impl_->resource.quarantine();(void)impl_.release();
  }
}
size_t DeviceProgram::label(){impl_->building();return impl_->labels++;}
void DeviceProgram::mark(size_t label) {
  auto& p=*impl_;p.building();if(label>=p.labels)throw std::invalid_argument("unknown control label");
  p.instructions.push_back({ControlInstruction::Mark,label,{}});p.operations.push_back({});p.indexes.push_back({});
}
void DeviceProgram::branch(const at::Tensor& index,const std::vector<size_t>& targets) {
  auto& p=*impl_;p.building();p.buffer(index);
  if(index.scalar_type()!=at::kInt||index.sizes()!=at::IntArrayRef{1}||targets.empty())
    throw std::invalid_argument("CUDA branch requires int32[1] and nonempty targets");
  for(auto t:targets)if(t>=p.labels)throw std::invalid_argument("unknown control target");
  p.instructions.push_back({ControlInstruction::Branch,0,targets});p.operations.push_back({});
  p.indexes.push_back(index);p.owners.push_back(index);
}
void DeviceProgram::kernel(std::function<void(void*)> fn,const std::vector<at::Tensor>& buffers) {
  if(!fn)throw std::invalid_argument("empty CUDA kernel submission");
  auto& p=*impl_;p.building();
  for(const auto& t:buffers){if(!t.defined()||t.device()!=p.device||t.requires_grad())throw std::invalid_argument("CUDA kernel ownership requires matching device without autograd");p.owners.push_back(t);}
  p.instructions.push_back({});p.indexes.push_back({});
  p.operations.push_back([fn](cudaStream_t s){fn(s);check(cudaGetLastError(),"submit CUDA kernel");});
}
void DeviceProgram::finish() {
  auto& p=*impl_;p.building();c10::cuda::CUDAGuard guard(p.device);
  const auto blocks=lower_control(p.instructions,p.labels);
  if(blocks.empty())throw std::invalid_argument("empty CUDA control program");
  try {
    check(cudaGraphCreate(&p.graph,0),"create conditional CUDA graph");
    cudaGraphConditionalHandle loop,pc;
    check(cudaGraphConditionalHandleCreate(&loop,p.graph,1,cudaGraphCondAssignDefault),"create loop condition");
    check(cudaGraphConditionalHandleCreate(&pc,p.graph,0,cudaGraphCondAssignDefault),"create block selector");
    cudaGraphNodeParams param{};param.type=cudaGraphNodeTypeConditional;
    param.conditional.handle=loop;param.conditional.type=cudaGraphCondTypeWhile;param.conditional.size=1;
    cudaGraphNode_t while_node;check(cudaGraphAddNode(&while_node,p.graph,nullptr,0,&param),"add device WHILE");
    auto body=param.conditional.phGraph_out[0];
    param={};param.type=cudaGraphNodeTypeConditional;param.conditional.handle=pc;
    param.conditional.type=cudaGraphCondTypeSwitch;param.conditional.size=blocks.size();
    cudaGraphNode_t switch_node;check(cudaGraphAddNode(&switch_node,body,nullptr,0,&param),"add device SWITCH");
    auto cases=param.conditional.phGraph_out;
    for(size_t i=0;i<blocks.size();++i) {
      const auto& b=blocks[i];auto map=at::tensor(b.targets,at::kInt).to(p.device);p.owners.push_back(map);
      check(cudaStreamBeginCaptureToGraph(p.stream,cases[i],nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal),"capture CUDA basic block");
      try {
      for(auto op:b.operations)p.operations[op](p.stream);
      const auto index=b.branch==std::numeric_limits<size_t>::max()?at::Tensor():p.indexes[b.branch];
      // A one-target branch ignores its input, matching the CANN contract.
      cuda_backend::route(p.stream,index.defined()&&b.targets.size()>1?index.const_data_ptr<int32_t>():nullptr,
        map.const_data_ptr<int32_t>(),b.targets.size(),blocks.size(),loop,pc,p.error.data_ptr<int32_t>());
      cudaGraph_t captured;check(cudaStreamEndCapture(p.stream,&captured),"finish CUDA basic block");
      if(captured!=cases[i])throw std::runtime_error("CUDA capture changed conditional body graph");
      } catch(...) {
        cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
        if(cudaStreamIsCapturing(p.stream,&status)==cudaSuccess&&status!=cudaStreamCaptureStatusNone){cudaGraph_t abandoned=nullptr;(void)cudaStreamEndCapture(p.stream,&abandoned);}
        throw;
      }
    }
    check(cudaGraphInstantiate(&p.executable,p.graph,0),"instantiate resident CUDA graph");p.finished=true;
  }catch(...){p.failed=true;throw;}
}
void DeviceProgram::order_after(const DeviceProgram& prior){impl_->building();impl_->preceding=prior.impl_->complete;}
void DeviceProgram::submit() {
  auto& p=*impl_;p.check();if(!p.finished||p.closed||p.failed||p.in_flight)throw std::logic_error("CUDA program is not executable");
  c10::cuda::CUDAGuard guard(p.device);
  try {
    p.in_flight=true;p.completion_recorded=false;
    check(cudaEventRecord(p.input_ready,c10::cuda::getCurrentCUDAStream(p.device.index()).stream()),"record input boundary");
    check(cudaStreamWaitEvent(p.stream,p.input_ready,0),"order CUDA input boundary");
    if(p.preceding)check(cudaStreamWaitEvent(p.stream,p.preceding,0),"order preceding CUDA program");
    check(cudaMemsetAsync(p.error.data_ptr(),0,sizeof(int32_t),p.stream),"reset CUDA runtime error");
    check(cudaGraphLaunch(p.executable,p.stream),"launch resident CUDA graph");
    check(cudaEventRecord(p.complete,p.stream),"record CUDA completion");
    p.completion_recorded=true;
  }catch(...){p.failed=true;throw;}
}
void DeviceProgram::wait(int32_t ms) {
  auto& p=*impl_;p.check();if(p.closed||!p.in_flight||ms<=0)throw std::logic_error("CUDA program has no waitable submission");
  c10::cuda::CUDAGuard guard(p.device);
  try{if(p.completion_recorded)cuda_backend::wait_event(p.complete,ms);else cuda_backend::wait_stream(p.stream,ms);p.in_flight=false;
    if(p.error.cpu().item<int32_t>())throw std::runtime_error("CUDA resident branch/index bounds failed");
  }catch(...){p.failed=true;throw;}
}
void DeviceProgram::run(int32_t ms){if(ms<=0)throw std::invalid_argument("positive timeout required");submit();wait(ms);}
void DeviceProgram::close(){impl_->release();}
void DeviceProgram::limit_workspace(int64_t bytes) {
  auto& p=*impl_;p.building();if(bytes<0||p.workspace.defined())throw std::invalid_argument("set workspace limit before matrix operations");
  p.workspace_limit=bytes;
}
int64_t DeviceProgram::workspace_bytes() const{return impl_->workspace.defined()?impl_->workspace.nbytes():0;}
int64_t DeviceProgram::retained_tensor_bytes() const {
  std::unordered_set<const c10::StorageImpl*> seen;int64_t bytes=0;
  for(const auto& t:impl_->owners)if(seen.insert(t.storage().unsafeGetStorageImpl()).second) {
    auto n=t.storage().nbytes();if(n>uint64_t(std::numeric_limits<int64_t>::max()-bytes))throw std::overflow_error("CUDA retained storage byte overflow");bytes+=n;
  }
  return bytes;
}
}
