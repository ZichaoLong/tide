#pragma once
// Semantic-source adapter, not an Ascend runtime or a vector performance claim.
// One logical worker per CUDA block executes the existing ordered scalar/tile
// program. Dense matrix operations use cuBLAS in the separate program backend.
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <type_traits>
#ifdef TIDE_KERNEL_CPU
#include <stdexcept>
#include <c10/util/Half.h>
using half=c10::Half;
#define __device__
#define __global__
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif
#define __aicore__ __device__
#define __gm__
using GM_ADDR=uint8_t*;
#define KERNEL_TASK_TYPE_DEFAULT(...)
constexpr int PIPE_V=0,PIPE_ALL=1,EVENT_ID0=0;

namespace tide_cuda {
constexpr size_t scratch_bytes=32768;
#ifdef TIDE_KERNEL_CPU
inline thread_local uint32_t worker=0,workers=1;
alignas(16) inline thread_local uint8_t scratch[scratch_bytes];
inline uint8_t* arena(){return scratch;}
[[noreturn]] inline void fail(){throw std::runtime_error("CUDA semantic adapter buffer/primitive contract");}
#else
__device__ inline uint8_t* arena(){extern __shared__ __align__(16) uint8_t scratch[];return scratch;}
__device__ inline void fail(){asm volatile("trap;");}
#endif
__device__ inline void reset_arena(){*reinterpret_cast<size_t*>(arena())=16;}
__device__ inline uint8_t* allocate(size_t bytes) {
  auto& cursor=*reinterpret_cast<size_t*>(arena());bytes=(bytes+15)/16*16;
  if(bytes>scratch_bytes-cursor){fail();return nullptr;}auto* result=arena()+cursor;cursor+=bytes;return result;
}
namespace scalar {
__device__ inline uint32_t GetBlockIdx(){
#ifdef TIDE_KERNEL_CPU
  return worker;
#else
  return blockIdx.x;
#endif
}
__device__ inline uint32_t GetBlockNum(){
#ifdef TIDE_KERNEL_CPU
  return workers;
#else
  return gridDim.x;
#endif
}
enum class QuePosition {VECIN,VECOUT,VECCALC};
enum class CacheLine {ENTIRE_DATA_CACHE};
enum class HardEvent {S_V,V_S};
enum class RoundMode {CAST_NONE,CAST_RINT};
enum class CMPMODE {LE};
template<class T> struct LocalTensor {
  T* data;size_t count;
  __device__ T GetValue(size_t i) const {if(i>=count)fail();return data[i];}
  __device__ void SetValue(size_t i,T x) const {if(i>=count)fail();data[i]=x;}
  __device__ LocalTensor operator[](size_t i) const {if(i>count)fail();return {data+i,count-i};}
  template<class U> __device__ LocalTensor<U> ReinterpretCast() const{return {reinterpret_cast<U*>(data),count*sizeof(T)/sizeof(U)};}
};
template<class T> struct GlobalTensor {
  T* data=nullptr;
  __device__ void SetGlobalBuffer(T* p){data=p;}
  __device__ GlobalTensor operator[](size_t i) const{return {data+i};}
};
struct Buffer {
  uint8_t* data=nullptr;size_t bytes=0;
  template<class T> __device__ LocalTensor<T> Get(){return {reinterpret_cast<T*>(data),bytes/sizeof(T)};}
};
template<QuePosition P> struct TBuf:Buffer {};
template<QuePosition P,int N> struct TQue:Buffer {
  template<class T> __device__ LocalTensor<T> AllocTensor(){return this->template Get<T>();}
  template<class T> __device__ LocalTensor<T> DeQue(){return this->template Get<T>();}
  template<class T> __device__ void EnQue(LocalTensor<T>){}
  template<class T> __device__ void FreeTensor(LocalTensor<T>){}
};
struct TPipe {
  template<class B> __device__ void InitBuffer(B& buffer,size_t bytes){buffer.data=allocate(bytes);buffer.bytes=bytes;}
  template<class B> __device__ void InitBuffer(B& buffer,int count,size_t bytes){if(count!=1)fail();InitBuffer(buffer,bytes);}
};
template<int P> __device__ void PipeBarrier(){}
template<HardEvent E> __device__ void SetFlag(int){}
template<HardEvent E> __device__ void WaitFlag(int){}
template<class T,CacheLine C> __device__ void DataCacheCleanAndInvalid(GlobalTensor<T>){}
template<class A,class B> __device__ B GetScalarBitcodeValue(A x) {
  static_assert(sizeof(A)==sizeof(B));B out;
  auto* a=reinterpret_cast<const uint8_t*>(&x);auto* b=reinterpret_cast<uint8_t*>(&out);
  for(size_t i=0;i<sizeof(A);++i)b[i]=a[i];return out;
}
struct DataCopyExtParams {uint16_t blocks;uint32_t bytes,source_stride,destination_stride,padding;};
template<class T> struct DataCopyPadExtParams {bool pad;uint8_t left,right;T value;};
template<class T> __device__ void DataCopyPad(LocalTensor<T> to,GlobalTensor<T> from,DataCopyExtParams p,DataCopyPadExtParams<T> pad) {
  if(p.blocks!=1||p.bytes%sizeof(T)||p.source_stride||p.destination_stride||p.padding)fail();
  auto n=p.bytes/sizeof(T);for(size_t i=0;i<pad.left;++i)to.SetValue(i,pad.value);
  for(size_t i=0;i<n;++i)to.SetValue(i+pad.left,from.data[i]);
  if(pad.pad)for(size_t i=0;i<pad.right;++i)to.SetValue(pad.left+n+i,pad.value);
}
template<class T> __device__ void DataCopyPad(GlobalTensor<T> to,LocalTensor<T> from,DataCopyExtParams p) {
  if(p.blocks!=1||p.bytes%sizeof(T)||p.source_stride||p.destination_stride||p.padding)fail();
  for(size_t i=0;i<p.bytes/sizeof(T);++i)to.data[i]=from.GetValue(i);
}
template<class A,class B> __device__ void Cast(LocalTensor<A> to,LocalTensor<B> from,RoundMode,uint32_t n) {
  for(uint32_t i=0;i<n;++i)to.SetValue(i,A(from.GetValue(i)));
}
template<class T> __device__ void Duplicate(LocalTensor<T> to,T x,uint32_t n){for(uint32_t i=0;i<n;++i)to.SetValue(i,x);}
#define TIDE_VECTOR_BINARY(name,expression) \
template<class T> __device__ void name(LocalTensor<T> to,LocalTensor<T> a,LocalTensor<T> b,uint32_t n){ \
 for(uint32_t i=0;i<n;++i){T x=a.GetValue(i),y=b.GetValue(i);to.SetValue(i,expression);}}
TIDE_VECTOR_BINARY(Add,x+y) TIDE_VECTOR_BINARY(Sub,x-y) TIDE_VECTOR_BINARY(Mul,x*y) TIDE_VECTOR_BINARY(Div,x/y)
TIDE_VECTOR_BINARY(Max,((x!=x)?x:((y!=y)?y:(x>y?x:y))))
#undef TIDE_VECTOR_BINARY
template<class T> __device__ void Muls(LocalTensor<T> to,LocalTensor<T> a,T b,uint32_t n){for(uint32_t i=0;i<n;++i)to.SetValue(i,a.GetValue(i)*b);}
template<class T> __device__ void Adds(LocalTensor<T> to,LocalTensor<T> a,T b,uint32_t n){for(uint32_t i=0;i<n;++i)to.SetValue(i,a.GetValue(i)+b);}
template<class T> __device__ void Exp(LocalTensor<T> to,LocalTensor<T> a,uint32_t n){for(uint32_t i=0;i<n;++i)to.SetValue(i,expf(float(a.GetValue(i))));}
template<class T> __device__ void Sqrt(LocalTensor<T> to,LocalTensor<T> a,uint32_t n){for(uint32_t i=0;i<n;++i)to.SetValue(i,sqrtf(float(a.GetValue(i))));}
template<class T> __device__ void Abs(LocalTensor<T> to,LocalTensor<T> a,uint32_t n){for(uint32_t i=0;i<n;++i)to.SetValue(i,fabsf(float(a.GetValue(i))));}
template<class T> __device__ void ReduceSum(LocalTensor<T> to,LocalTensor<T> a,LocalTensor<T>,int32_t n){T x=0;for(int32_t i=0;i<n;++i)x+=a.GetValue(i);to.SetValue(0,x);}
template<class T> __device__ void ReduceMax(LocalTensor<T> to,LocalTensor<T> a,LocalTensor<T>,int32_t n){
  T x=a.GetValue(0);for(int32_t i=1;i<n;++i){auto y=a.GetValue(i);x=x!=x?x:(y!=y?y:(x>y?x:y));}to.SetValue(0,x);
}
template<class T> __device__ void Compares(LocalTensor<uint8_t> out,LocalTensor<T> a,T b,CMPMODE mode,uint32_t n) {
  if(mode!=CMPMODE::LE)fail();for(uint32_t i=0;i<(n+7)/8;++i)out.SetValue(i,0);
  for(uint32_t i=0;i<n;++i)if(a.GetValue(i)<=b)out.SetValue(i/8,out.GetValue(i/8)|(1u<<(i%8)));
}
}
}
namespace AscendC=tide_cuda::scalar;
