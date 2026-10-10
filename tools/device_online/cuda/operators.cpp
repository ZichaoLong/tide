#include "program_internal.h"
#include <cmath>
#include <climits>
#include <set>

namespace tide::device_online {
using namespace cuda_backend;
namespace {
bool floating(const at::Tensor& t){return t.scalar_type()==at::kFloat||t.scalar_type()==at::kHalf;}
void same(const at::Tensor& a,const at::Tensor& b) {
  if(a.sizes()!=b.sizes()||a.scalar_type()!=b.scalar_type())throw std::invalid_argument("CUDA operation tensor metadata mismatch");
}
int64_t axis_of(const at::Tensor& x,int64_t axis) {
  if(axis<0)axis+=x.dim();
  if(axis<0||axis>=x.dim())throw std::invalid_argument("invalid CUDA operation axis");return axis;
}
std::pair<int64_t,int64_t> sides(const at::Tensor& t,int64_t axis) {
  int64_t outer=1,inner=1;
  for(int64_t d=0;d<axis;++d)outer*=t.size(d);
  for(int64_t d=axis+1;d<t.dim();++d)inner*=t.size(d);
  return {outer,inner}; // Validated nonempty contiguous tensor bounds products.
}
Layout layout(const at::Tensor& a,const at::Tensor& b,const at::Tensor& out) {
  Layout l;l.rank=out.dim();l.count=out.numel();
  if(l.rank>8||a.dim()>l.rank||(b.defined()&&b.dim()>l.rank))throw std::invalid_argument("CUDA broadcast rank exceeds profile");
  for(int d=0;d<l.rank;++d) {
    l.size[d]=out.size(d);
    auto stride=[&](const at::Tensor& t) {
      auto j=d-(l.rank-t.dim());if(j<0)return int64_t(0);
      if(t.size(j)!=1&&t.size(j)!=l.size[d])throw std::invalid_argument("invalid CUDA broadcast extent");
      return t.size(j)==1?int64_t(0):t.stride(j);
    };
    l.a[d]=stride(a);if(b.defined())l.b[d]=stride(b);
  }
  return l;
}
}
void DeviceProgram::Impl::element(Op op,const at::Tensor& a,const at::Tensor& b,const at::Tensor& out) {
  buffer(a);buffer(out);if(b.defined())buffer(b);
  const bool compare=op==Equal||op==Less||op==And;
  if(b.defined()&&a.scalar_type()!=b.scalar_type())throw std::invalid_argument("CUDA operands require the same dtype");
  if(compare&&out.scalar_type()!=at::kBool)throw std::invalid_argument("CUDA comparison requires bool output");
  if(!compare&&op!=Cast&&a.scalar_type()!=out.scalar_type())throw std::invalid_argument("CUDA output dtype differs");
  if(op==Less&&a.scalar_type()!=at::kLong)throw std::invalid_argument("CUDA logical ordering requires int64");
  if(op==And&&a.scalar_type()!=at::kBool)throw std::invalid_argument("CUDA logical and requires bool");
  if(op!=Copy&&op!=Add&&op!=Equal&&op!=Less&&op!=And&&op!=Cast&&!floating(a))
    throw std::invalid_argument("CUDA numerical operation requires floating operands");
  const auto l=layout(a,b,out);const auto at=type(a.scalar_type()),bt=b.defined()?type(b.scalar_type()):at,ot=type(out.scalar_type());
  std::vector<at::Tensor> buffers{a,out};if(b.defined())buffers.push_back(b);
  append([=](cudaStream_t s){elementwise(s,op,a.const_data_ptr(),at,b.defined()?b.const_data_ptr():nullptr,bt,out.data_ptr(),ot,l);},buffers);
}
void DeviceProgram::add(const at::Tensor& a,const at::Tensor& b){impl_->element(Add,a,b,a);}
void DeviceProgram::copy(const at::Tensor& out,const at::Tensor& in){same(in,out);impl_->element(Copy,in,{},out);}
void DeviceProgram::cast(const at::Tensor& in,const at::Tensor& out) {
  if(!floating(in)||!floating(out)||in.sizes()!=out.sizes())throw std::invalid_argument("CUDA cast is FP32/FP16 only");
  impl_->element(Cast,in,{},out);
}
void DeviceProgram::zero(const at::Tensor& out) {
  impl_->append([out](cudaStream_t s){check(cudaMemsetAsync(out.data_ptr(),0,out.nbytes(),s),"zero CUDA buffer");},{out});
}
#define BINARY(name,op) void DeviceProgram::name(const at::Tensor& a,const at::Tensor& b,const at::Tensor& out){impl_->element(op,a,b,out);}
BINARY(multiply,Mul) BINARY(divide,Div) BINARY(equal,Equal) BINARY(less,Less) BINARY(logical_and,And)
BINARY(tanh_backward,TanhBackward) BINARY(silu_backward,SiluBackward) BINARY(relu_backward,ReluBackward)
#undef BINARY
#define UNARY(name,op) void DeviceProgram::name(const at::Tensor& a,const at::Tensor& out){same(a,out);impl_->element(op,a,{},out);}
UNARY(softplus,Softplus) UNARY(sigmoid,Sigmoid) UNARY(tanh,Tanh) UNARY(relu,Relu) UNARY(silu,Silu)
#undef UNARY
void DeviceProgram::cast_index(const at::Tensor& in,const at::Tensor& out) {
  if(in.scalar_type()!=at::kBool||out.scalar_type()!=at::kInt||in.sizes()!=out.sizes())
    throw std::invalid_argument("CUDA branch conversion requires bool to int32");
  impl_->element(Cast,in,{},out);
}
void DeviceProgram::sum(const at::Tensor& in,int64_t axis,bool keep,const at::Tensor& out) {
  impl_->buffer(in);impl_->buffer(out);axis=axis_of(in,axis);
  auto shape=in.sizes().vec();if(keep)shape[axis]=1;else shape.erase(shape.begin()+axis);
  if(!floating(in)||out.scalar_type()!=in.scalar_type()||out.sizes()!=at::IntArrayRef(shape))throw std::invalid_argument("CUDA sum metadata mismatch");
  auto [outer,inner]=sides(in,axis);
  impl_->append([=](cudaStream_t s){reduce(s,in.const_data_ptr(),type(in.scalar_type()),out.data_ptr(),outer,in.size(axis),inner,false);},{in,out});
}
void DeviceProgram::softmax(const at::Tensor& in,int64_t axis,const at::Tensor& out) {
  same(in,out);impl_->buffer(in);axis=axis_of(in,axis);
  if(!floating(in))throw std::invalid_argument("CUDA softmax requires floating input");
  auto [outer,inner]=sides(in,axis);
  impl_->append([=](cudaStream_t s){reduce(s,in.const_data_ptr(),type(in.scalar_type()),out.data_ptr(),outer,in.size(axis),inner,true);},{in,out});
}
void DeviceProgram::rms_norm(const at::Tensor& in,double epsilon,const at::Tensor& out,const at::Tensor& rstd) {
  same(in,out);impl_->buffer(in);
  if(!floating(in)||in.dim()<1||!std::isfinite(epsilon)||epsilon<=0)throw std::invalid_argument("invalid CUDA RMS norm");
  auto rows=in.numel()/in.size(-1);std::vector<at::Tensor> buffers{in,out};
  if(rstd.defined()){impl_->buffer(rstd);if(rstd.scalar_type()!=at::kFloat||rstd.numel()!=rows)throw std::invalid_argument("invalid CUDA rstd buffer");buffers.push_back(rstd);}
  impl_->append([=](cudaStream_t s){normalize(s,in.const_data_ptr(),type(in.scalar_type()),out.data_ptr(),rstd.defined()?rstd.data_ptr<float>():nullptr,rows,in.size(-1),epsilon,true);},buffers);
}
void DeviceProgram::layer_norm(const at::Tensor& in,double epsilon,const at::Tensor& out,const at::Tensor& rstd) {
  same(in,out);impl_->buffer(in);
  if(!floating(in)||in.dim()<1||!std::isfinite(epsilon)||epsilon<=0)throw std::invalid_argument("invalid CUDA layer norm");
  auto rows=in.numel()/in.size(-1);std::vector<at::Tensor> buffers{in,out};
  if(rstd.defined()){impl_->buffer(rstd);if(rstd.scalar_type()!=at::kFloat||rstd.numel()!=rows)throw std::invalid_argument("invalid CUDA rstd buffer");buffers.push_back(rstd);}
  impl_->append([=](cudaStream_t s){normalize(s,in.const_data_ptr(),type(in.scalar_type()),out.data_ptr(),rstd.defined()?rstd.data_ptr<float>():nullptr,rows,in.size(-1),epsilon,false);},buffers);
}
void DeviceProgram::batch_matmul(const at::Tensor& a,const at::Tensor& b,const at::Tensor& out) {
  auto& p=*impl_;p.building();p.buffer(a);p.buffer(b);p.buffer(out);
  if(!floating(a)||a.dim()!=3||b.dim()!=3||out.dim()!=3||a.scalar_type()!=b.scalar_type()||out.scalar_type()!=a.scalar_type()
      ||a.size(0)!=b.size(0)||out.size(0)!=a.size(0)||a.size(2)!=b.size(1)||out.size(1)!=a.size(1)||out.size(2)!=b.size(2))
    throw std::invalid_argument("CUDA BMM requires matching contiguous 3D batches");
  for(auto n:{a.size(0),a.size(1),a.size(2),b.size(2)})if(n>INT_MAX)throw std::overflow_error("CUDA BMM dimension exceeds cuBLAS int32");
  c10::cuda::CUDAGuard guard(p.device);
  if(!p.blas) {
    constexpr int64_t bytes=4*1024*1024;
    if(bytes>p.workspace_limit){p.failed=true;throw std::invalid_argument("cuBLAS workspace exceeds declared capacity");}
    check_blas(cublasCreate(&p.blas),"create cuBLAS handle");
    check_blas(cublasSetStream(p.blas,p.stream),"set cuBLAS control stream");
    check_blas(cublasSetMathMode(p.blas,CUBLAS_PEDANTIC_MATH),"disable TF32 and reduced precision");
    p.workspace=at::empty({bytes},a.options().dtype(at::kByte));
    check_blas(cublasSetWorkspace(p.blas,p.workspace.data_ptr(),bytes),"set bounded cuBLAS workspace");
  }
  auto handle=p.blas;
  p.append([=](cudaStream_t){
    const float alpha=1,beta=0;auto dtype=a.scalar_type()==at::kHalf?CUDA_R_16F:CUDA_R_32F;
    check_blas(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,b.size(2),a.size(1),a.size(2),
      &alpha,b.const_data_ptr(),dtype,b.size(2),b.stride(0),a.const_data_ptr(),dtype,a.size(2),a.stride(0),
      &beta,out.data_ptr(),dtype,out.size(2),out.stride(0),a.size(0),CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT),"capture CUDA BMM");
  },{a,b,out});
}
void DeviceProgram::permute(const at::Tensor& in,const std::vector<int64_t>& axes,const at::Tensor& out) {
  impl_->buffer(in);impl_->buffer(out);Layout l;l.rank=in.dim();l.count=out.numel();std::set<int64_t> seen;
  if(l.rank>8||axes.size()!=size_t(l.rank)||out.dim()!=l.rank||in.scalar_type()!=out.scalar_type())throw std::invalid_argument("invalid CUDA permute rank/dtype");
  for(int d=0;d<l.rank;++d){auto axis=axis_of(in,axes[d]);if(!seen.insert(axis).second||out.size(d)!=in.size(axis))throw std::invalid_argument("invalid CUDA permutation");l.size[d]=out.size(d);l.a[d]=in.stride(axis);}
  if(in.is_alias_of(out))throw std::invalid_argument("CUDA permutation requires separate output storage");
  impl_->append([=](cudaStream_t s){cuda_backend::permute(s,in.const_data_ptr(),type(in.scalar_type()),out.data_ptr(),l);},{in,out});
}
void DeviceProgram::index_select(const at::Tensor& in,int64_t axis,const at::Tensor& ids,const at::Tensor& out) {
  auto& p=*impl_;p.buffer(in);p.buffer(ids);p.buffer(out);axis=axis_of(in,axis);auto shape=in.sizes().vec();shape[axis]=ids.numel();
  if(ids.dim()!=1||ids.scalar_type()!=at::kLong||out.sizes()!=at::IntArrayRef(shape)||in.scalar_type()!=out.scalar_type()||in.is_alias_of(out))throw std::invalid_argument("invalid CUDA gather metadata");
  auto [outer,inner]=sides(in,axis);auto error=p.error;
  p.append([=](cudaStream_t s){index(s,in.const_data_ptr(),type(in.scalar_type()),ids.const_data_ptr<int64_t>(),out.data_ptr(),outer,in.size(axis),out.size(axis),inner,false,error.data_ptr<int32_t>());},{in,ids,out});
}
void DeviceProgram::index_copy(const at::Tensor& out,int64_t axis,const at::Tensor& ids,const at::Tensor& in) {
  auto& p=*impl_;p.buffer(in);p.buffer(ids);p.buffer(out);axis=axis_of(out,axis);auto shape=out.sizes().vec();shape[axis]=ids.numel();
  if(ids.dim()!=1||ids.scalar_type()!=at::kLong||in.sizes()!=at::IntArrayRef(shape)||in.scalar_type()!=out.scalar_type()||in.is_alias_of(out))throw std::invalid_argument("invalid CUDA scatter metadata");
  auto [outer,inner]=sides(out,axis);auto error=p.error;
  p.append([=](cudaStream_t s){index(s,in.const_data_ptr(),type(in.scalar_type()),ids.const_data_ptr<int64_t>(),out.data_ptr(),outer,in.size(axis),out.size(axis),inner,true,error.data_ptr<int32_t>());},{in,ids,out});
}
}
