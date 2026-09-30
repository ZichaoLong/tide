#include "cann_program.h"
#include "cann_api.h"
#include "portable_torch/runtime.hpp"
#include <c10/core/DeviceGuard.h>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
struct CannProgram::Impl {
  at::Device device;
  portable_torch::RuntimeResource resource;
  CannApi api;
  void *stream = nullptr, *launch = nullptr, *model = nullptr;
  void* target_list = nullptr;
  std::vector<void*> labels, lists, descriptors, scalars, arrays;
  std::vector<bool> marked;
  std::vector<at::Tensor> owners, workspaces;
  std::vector<std::function<void()>> commands;
  bool bound = false, finished = false, closed = false, failed = false, in_flight = false;
  explicit Impl(at::Device d) : device(d) {}
  void initialize() {
    c10::DeviceGuard guard(device);
    CannApi::check(api.create_stream(&stream, 0, 4), "create persistent control stream");
    CannApi::check(api.create_stream(&launch, 0, 0), "create control launch stream");
    CannApi::check(api.begin_model(&model, 0), "begin runtime control model");
    CannApi::check(api.bind_stream(model, stream, 0), "bind persistent control stream"); bound = true;
  }
  void building() const {
    resource.check();
    if (finished || closed || failed) throw std::logic_error("control program is not open for construction");
  }
  void validate(const at::Tensor& value, at::ScalarType dtype) const {
    if (!value.defined() || value.device() != device || value.scalar_type() != dtype
        || value.sizes() != at::IntArrayRef{1} || !value.is_contiguous() || value.requires_grad())
      throw std::invalid_argument("control buffer requires matching device/dtype and contiguous [1] without autograd");
  }
  void* tensor(const at::Tensor& value, at::ScalarType dtype) {
    building();
    if (!value.defined() || value.device()!=device || value.scalar_type()!=dtype
        || !value.is_contiguous() || value.requires_grad() || value.numel()<1)
      throw std::invalid_argument("CANN buffer requires matching device/dtype, contiguous nonempty storage and no autograd");
    const int type = dtype==at::kLong?9:dtype==at::kBool?12:dtype==at::kInt?3:
                     dtype==at::kFloat?0:dtype==at::kHalf?1:dtype==at::kByte?4:-1;
    if(type<0)throw std::invalid_argument("CANN buffer dtype is unavailable");
    auto sizes=value.sizes(),strides=value.strides();
    auto handle = api.create_tensor(sizes.data(), sizes.size(), type, strides.data(), 0, 2,
                                   sizes.data(), sizes.size(), value.data_ptr());
    if (!handle) throw std::runtime_error("create control tensor descriptor failed");
    descriptors.push_back(handle); owners.push_back(value); return handle;
  }
  template<class... Args> void op(const char* name, Args... args) {
    building(); c10::DeviceGuard guard(device);
    using Prepare = int (*)(Args..., uint64_t*, void**);
    using Execute = int (*)(void*, uint64_t, void*, void*);
    const auto prepare = CannApi::symbol<Prepare>(api.operators, std::string(name)+"GetWorkspaceSize");
    const auto execute = CannApi::symbol<Execute>(api.operators, name);
    uint64_t size = 0; void* executor = nullptr;
    CannApi::check(prepare(args..., &size, &executor), name);
    if (size > uint64_t(std::numeric_limits<int64_t>::max())) throw std::overflow_error("control workspace size");
    auto workspace = at::empty({std::max<int64_t>(1, size)}, at::TensorOptions().device(device).dtype(at::kByte));
    workspaces.push_back(workspace); // Keep it through all runtime-model executions.
    commands.push_back([this, execute, workspace, size, executor, name] {
      CannApi::check(execute(workspace.data_ptr(), size, executor, stream), name);
    });
  }
  void release() {
    if (closed) return;
    c10::DeviceGuard guard(device);
    // Bound streams must remain attached until executions have finished.
    // A timeout is NOT completion. Stop before freeing anything in that case.
    if (in_flight) {
      CannApi::check(api.sync_stream(launch, 10000), "drain control model before release");
      in_flight = false;
    }
    if (bound) { CannApi::check(api.unbind_stream(model, stream), "unbind control stream"); bound = false; }
    if (model) { CannApi::check(api.destroy_model(model), "destroy control model"); model = nullptr; }
    auto destroy = [](auto& items, auto fn, const char* name) {
      while (!items.empty()) { CannApi::check(fn(items.back()), name); items.pop_back(); }
    };
    destroy(lists, api.destroy_list, "destroy control target list");
    destroy(labels, api.destroy_label, "destroy control label");
    destroy(descriptors, api.destroy_tensor, "destroy control tensor descriptor");
    destroy(scalars, api.destroy_scalar, "destroy control scalar");
    destroy(arrays, api.destroy_int_array, "destroy control shape array");
    if (stream) { CannApi::check(api.destroy_stream(stream), "destroy control stream"); stream = nullptr; }
    if (launch) { CannApi::check(api.destroy_stream(launch), "destroy control launch stream"); launch = nullptr; }
    commands.clear(); owners.clear(); workspaces.clear(); closed = true;
    resource.close();
  }
};
CannProgram::CannProgram(at::Device device) : CannProgram(device,{}) {}
CannProgram::CannProgram(at::Device device,const std::function<void(CannApi&)>& configure_api) {
  if (device.type() != c10::DeviceType::PrivateUse1)
    throw std::invalid_argument("CANN device control requires an explicit NPU");
  impl_ = std::make_unique<Impl>(device);
  try { if(configure_api)configure_api(impl_->api);impl_->initialize(); }
  catch (...) {
    try { impl_->release(); }
    catch (...) { impl_->resource.quarantine();(void)impl_.release(); }
    throw;
  }
}
CannProgram::~CannProgram() {
  try { impl_->release(); }
  catch (const std::exception& error) {
    // Keep runtime handles AND tensor owners alive, even beyond static teardown.
    // Only process exit can reclaim resources after an unconfirmed completion.
    std::cerr << "CANN control resources quarantined until process exit: " << error.what() << '\n';
    impl_->resource.quarantine();
    (void)impl_.release();
  }
}
size_t CannProgram::label() {
  auto& p = *impl_; p.building(); c10::DeviceGuard guard(p.device);
  if(p.labels.size()>=size_t(std::numeric_limits<int32_t>::max()))throw std::overflow_error("too many control labels");
  void* handle = nullptr; CannApi::check(p.api.create_label(&handle), "create control label");
  p.labels.push_back(handle); p.marked.push_back(false); return p.labels.size()-1;
}
void CannProgram::mark(size_t index) {
  auto& p = *impl_; p.building(); c10::DeviceGuard guard(p.device);
  if (index >= p.labels.size() || p.marked[index]) throw std::invalid_argument("invalid or repeated control label");
  p.commands.push_back([&p, index] {
    CannApi::check(p.api.set_label(p.labels[index], p.stream), "mark control label");
  });
  p.marked[index] = true;
}
void CannProgram::branch(const at::Tensor& index, const std::vector<size_t>& targets) {
  auto& p = *impl_; p.building(); p.validate(index, at::kInt); c10::DeviceGuard guard(p.device);
  if (targets.empty() || targets.size() > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("invalid control target count");
  std::vector<int32_t> mapping;
  for (auto target : targets) {
    if (target >= p.labels.size()) throw std::invalid_argument("unknown control target");
    mapping.push_back(static_cast<int32_t>(target));
  }
  // A label may only occur in one CANN target list. Share a global list and
  // translate each branch's local index on-device; repeated destinations work.
  auto table=at::tensor(mapping,at::kInt).to(p.device), selected=table;
  if(mapping.size()>1) {
    selected=at::empty({1},table.options());
    p.op("aclnnIndexSelect",p.tensor(table,at::kInt),int64_t(0),p.tensor(index,at::kInt),p.tensor(selected,at::kInt));
  }
  p.owners.push_back(index);p.owners.push_back(selected);
  p.commands.push_back([&p, selected] {
    CannApi::check(p.api.jump(selected.data_ptr(), p.labels.size(), p.target_list, p.stream), "device control branch");
  });
}
void CannProgram::add(const at::Tensor& target, const at::Tensor& increment) {
  auto& p = *impl_; p.building(); int64_t one = 1;
  auto scalar = p.api.create_scalar(&one, 9);
  if (!scalar) throw std::runtime_error("create control scalar failed");
  p.scalars.push_back(scalar);
  p.op("aclnnInplaceAdd", p.tensor(target, target.scalar_type()), p.tensor(increment, target.scalar_type()), scalar);
}
void CannProgram::copy(const at::Tensor& target, const at::Tensor& source) {
  auto& p=*impl_;p.building();
  if(target.sizes()!=source.sizes()||target.scalar_type()!=source.scalar_type())
    throw std::invalid_argument("CANN copy requires identical shapes and dtypes");
  p.op("aclnnInplaceCopy",p.tensor(target,target.scalar_type()),p.tensor(source,source.scalar_type()));
}
void CannProgram::multiply(const at::Tensor& a,const at::Tensor& b,const at::Tensor& out) {
  auto& p=*impl_;const auto dtype=a.scalar_type();
  p.op("aclnnMul",p.tensor(a,dtype),p.tensor(b,dtype),p.tensor(out,dtype));
}
void CannProgram::softmax(const at::Tensor& input,int64_t axis,const at::Tensor& out) {
  auto& p=*impl_;
  p.op("aclnnSoftmax",p.tensor(input,input.scalar_type()),axis,p.tensor(out,input.scalar_type()));
}
void CannProgram::sigmoid(const at::Tensor& input,const at::Tensor& out) {
  auto& p=*impl_;
  p.op("aclnnSigmoid",p.tensor(input,input.scalar_type()),p.tensor(out,input.scalar_type()));
}
void CannProgram::tanh(const at::Tensor& input,const at::Tensor& out) {
  auto& p=*impl_;
  p.op("aclnnTanh",p.tensor(input,input.scalar_type()),p.tensor(out,input.scalar_type()));
}
void CannProgram::relu(const at::Tensor& input,const at::Tensor& out) {
  auto& p=*impl_;p.op("aclnnRelu",p.tensor(input,input.scalar_type()),p.tensor(out,input.scalar_type()));
}
void CannProgram::silu(const at::Tensor& input,const at::Tensor& out) {
  auto& p=*impl_;p.op("aclnnSilu",p.tensor(input,input.scalar_type()),p.tensor(out,input.scalar_type()));
}
void CannProgram::rms_norm(const at::Tensor& input,double epsilon,const at::Tensor& out) {
  auto& p=*impl_;p.building();
  if(input.dim()!=2||input.size(1)<1||!(epsilon>0))throw std::invalid_argument("CANN RMS norm requires nonempty rows and positive epsilon");
  auto weight=at::ones({input.size(1)},input.options()),rstd=at::empty({input.size(0),1},input.options().dtype(at::kFloat));
  p.op("aclnnRmsNorm",p.tensor(input,input.scalar_type()),p.tensor(weight,weight.scalar_type()),epsilon,
       p.tensor(out,input.scalar_type()),p.tensor(rstd,at::kFloat));
}
void CannProgram::layer_norm(const at::Tensor& input,double epsilon,const at::Tensor& out) {
  auto& p=*impl_;p.building();
  if(input.dim()!=2||input.size(1)<1||!(epsilon>0))throw std::invalid_argument("CANN layer norm requires nonempty rows and positive epsilon");
  const int64_t width=input.size(1);auto shape=p.api.create_int_array(&width,1);
  if(!shape)throw std::runtime_error("create normalization shape failed");p.arrays.push_back(shape);
  auto mean=at::empty({input.size(0),1},input.options().dtype(at::kFloat)),rstd=at::empty_like(mean);
  p.op("aclnnLayerNorm",p.tensor(input,input.scalar_type()),shape,static_cast<void*>(nullptr),static_cast<void*>(nullptr),epsilon,
       p.tensor(out,input.scalar_type()),p.tensor(mean,at::kFloat),p.tensor(rstd,at::kFloat));
}
void CannProgram::batch_matmul(const at::Tensor& a,const at::Tensor& b,const at::Tensor& out) {
  auto& p=*impl_;
  // CANN cubeMathType=0 is KEEP_DTYPE. Never silently enable HF32/FP16.
  p.op("aclnnBatchMatMul",p.tensor(a,a.scalar_type()),p.tensor(b,a.scalar_type()),
       p.tensor(out,a.scalar_type()),int8_t(0));
}
void CannProgram::index_copy(const at::Tensor& target,int64_t axis,const at::Tensor& indices,const at::Tensor& source) {
  auto& p=*impl_;
  p.op("aclnnInplaceIndexCopy",p.tensor(target,target.scalar_type()),axis,p.tensor(indices,at::kLong),p.tensor(source,target.scalar_type()));
}
void CannProgram::equal(const at::Tensor& a,const at::Tensor& b,const at::Tensor& out) {
  auto& p=*impl_;
  p.op("aclnnEqTensor",p.tensor(a,a.scalar_type()),p.tensor(b,a.scalar_type()),p.tensor(out,at::kBool));
}
void CannProgram::index_select(const at::Tensor& a,int64_t axis,const at::Tensor& index,const at::Tensor& out) {
  auto& p=*impl_;
  p.op("aclnnIndexSelect",p.tensor(a,a.scalar_type()),axis,p.tensor(index,at::kLong),p.tensor(out,a.scalar_type()));
}
void CannProgram::kernel(std::function<void(void*)> submit,const std::vector<at::Tensor>& buffers) {
  auto& p=*impl_;p.building();
  if(!submit)throw std::invalid_argument("empty CANN kernel submission");
  for(const auto& buffer:buffers) {
    if(!buffer.defined()||buffer.device()!=p.device||buffer.requires_grad())
      throw std::invalid_argument("kernel buffers require matching device and no autograd");
    p.owners.push_back(buffer);
  }
  p.commands.push_back([&p,submit]{submit(p.stream);});
}
void CannProgram::less(const at::Tensor& a, const at::Tensor& b, const at::Tensor& out) {
  auto& p = *impl_; p.op("aclnnLtTensor", p.tensor(a, at::kLong), p.tensor(b, at::kLong), p.tensor(out, at::kBool));
}
void CannProgram::logical_and(const at::Tensor& a, const at::Tensor& b, const at::Tensor& out) {
  auto& p = *impl_; p.op("aclnnLogicalAnd", p.tensor(a, at::kBool), p.tensor(b, at::kBool), p.tensor(out, at::kBool));
}
void CannProgram::cast_index(const at::Tensor& in, const at::Tensor& out) {
  auto& p = *impl_; p.op("aclnnCast", p.tensor(in, at::kBool), int(3), p.tensor(out, at::kInt));
}
void CannProgram::finish() {
  auto& p = *impl_; p.building(); c10::DeviceGuard guard(p.device);
  for (auto marked : p.marked) if (!marked) throw std::logic_error("control label has no target position");
  // CANN requires target-list creation BEFORE the first label is marked.
  // Defer all task emission, preserving source order, until lists are complete.
  try {
    if(!p.labels.empty()) {
      CannApi::check(p.api.create_list(p.labels.data(),p.labels.size(),&p.target_list),"create control target list");
      p.lists.push_back(p.target_list);
    }
    for (const auto& command : p.commands) command();
    CannApi::check(p.api.end_task(p.model, p.stream), "end control task");
    CannApi::check(p.api.end_model(p.model, nullptr), "finish control model"); p.finished = true;
  } catch (...) { p.failed = true; throw; }
}
void CannProgram::run(int32_t timeout_ms) {
  auto& p = *impl_;
  if (!p.finished || p.closed || p.failed || timeout_ms <= 0) throw std::logic_error("control program is not executable");
  p.resource.check();
  c10::DeviceGuard guard(p.device);
  try {
    p.in_flight = true; // Even a failed asynchronous submission needs a drain.
    CannApi::check(p.api.execute(p.model, p.launch), "execute control model");
    CannApi::check(p.api.sync_stream(p.launch, timeout_ms), "wait for control boundary");
    p.in_flight = false;
  } catch (...) { p.failed = true; throw; }
}
void CannProgram::close() {
  try {impl_->release();}
  catch (...) {impl_->failed=true;throw;}
}
int64_t CannProgram::workspace_bytes() const {
  int64_t result = 0;
  for (const auto& workspace : impl_->workspaces) result += workspace.nbytes();
  return result;
}
}  // namespace tide::device_online
