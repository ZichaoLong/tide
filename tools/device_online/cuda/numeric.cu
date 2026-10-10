#include "numeric.h"
#include <cuda_fp16.h>
#include <cmath>

namespace tide::device_online::cuda_backend {
namespace {
__device__ float get_float(const void* p,Type t,int64_t i) {
  return t==F16?__half2float(static_cast<const __half*>(p)[i]):static_cast<const float*>(p)[i];
}
__device__ int64_t get_int(const void* p,Type t,int64_t i) {
  if(t==I64)return static_cast<const int64_t*>(p)[i];
  if(t==I32)return static_cast<const int32_t*>(p)[i];
  return static_cast<const uint8_t*>(p)[i];
}
__device__ void put_float(void* p,Type t,int64_t i,float x) {
  if(t==F16)static_cast<__half*>(p)[i]=__float2half_rn(x);else static_cast<float*>(p)[i]=x;
}
__device__ void put_int(void* p,Type t,int64_t i,int64_t x) {
  if(t==I64)static_cast<int64_t*>(p)[i]=x;
  else if(t==I32)static_cast<int32_t*>(p)[i]=x;
  else static_cast<uint8_t*>(p)[i]=x;
}
__global__ void elements(Op op,const void* a,Type at,const void* b,Type bt,void* out,Type ot,Layout l) {
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<l.count;i+=int64_t(gridDim.x)*blockDim.x) {
    int64_t coordinate=i,ai=0,bi=0;
    for(int d=l.rank-1;d>=0;--d){auto c=coordinate%l.size[d];coordinate/=l.size[d];ai+=c*l.a[d];bi+=c*l.b[d];}
    if(at>=I64) {
      int64_t x=get_int(a,at,ai),y=b?get_int(b,bt,bi):0,z=x;
      if(op==Add)z=x+y;else if(op==Equal)z=x==y;else if(op==Less)z=x<y;else if(op==And)z=bool(x)&&bool(y);
      put_int(out,ot,i,z);continue;
    }
    float x=get_float(a,at,ai),y=b?get_float(b,bt,bi):0,z=x;
    if(op==Add)z=x+y;else if(op==Mul)z=x*y;else if(op==Div)z=x/y;
    else if(op==Softplus)z=x>20?x:log1pf(expf(x));
    else if(op==Sigmoid)z=1.f/(1.f+expf(-x));else if(op==Tanh)z=tanhf(x);
    else if(op==Relu)z=x!=x?x:(x>0?x:0.f);else if(op==Silu)z=x/(1.f+expf(-x));
    else if(op==TanhBackward)z=x*(1.f-y*y);
    else if(op==SiluBackward){float s=1.f/(1.f+expf(-y));z=x*s*(1.f+y*(1.f-s));}
    else if(op==ReluBackward)z=y<=0?0.f:x;
    else if(op==Equal){put_int(out,ot,i,x==y);continue;}
    put_float(out,ot,i,z);
  }
}
__global__ void reduction(const void* in,Type type,void* out,int64_t outer,int64_t axis,int64_t inner,bool softmax) {
  for(int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;row<outer*inner;row+=int64_t(gridDim.x)*blockDim.x) {
    int64_t base=(row/inner)*axis*inner+row%inner;
    float sum=0,maximum=-INFINITY;
    if(softmax)for(int64_t j=0;j<axis;++j)maximum=fmaxf(maximum,get_float(in,type,base+j*inner));
    for(int64_t j=0;j<axis;++j){float x=get_float(in,type,base+j*inner);sum+=softmax?expf(x-maximum):x;}
    if(softmax)for(int64_t j=0;j<axis;++j)put_float(out,type,base+j*inner,expf(get_float(in,type,base+j*inner)-maximum)/sum);
    else put_float(out,type,row,sum);
  }
}
__global__ void norm_rows(const void* in,Type type,void* out,float* rstd,int64_t rows,int64_t width,float eps,bool rms) {
  for(int64_t r=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;r<rows;r+=int64_t(gridDim.x)*blockDim.x) {
    float mean=0,squares=0;
    if(!rms){for(int64_t j=0;j<width;++j)mean+=get_float(in,type,r*width+j);mean/=float(width);}
    for(int64_t j=0;j<width;++j){float x=get_float(in,type,r*width+j)-mean;squares+=x*x;}
    float inv=rsqrtf(squares/float(width)+eps);if(rstd)rstd[r]=inv;
    for(int64_t j=0;j<width;++j)put_float(out,type,r*width+j,(get_float(in,type,r*width+j)-mean)*inv);
  }
}
__device__ void copy_one(const void* in,void* out,Type type,int64_t src,int64_t dst) {
  if(type>=I64)put_int(out,type,dst,get_int(in,type,src));else put_float(out,type,dst,get_float(in,type,src));
}
__global__ void permutation(const void* in,Type type,void* out,Layout l) {
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<l.count;i+=int64_t(gridDim.x)*blockDim.x) {
    int64_t coordinate=i,source=0;
    for(int d=l.rank-1;d>=0;--d){source+=(coordinate%l.size[d])*l.a[d];coordinate/=l.size[d];}
    copy_one(in,out,type,source,i);
  }
}
__global__ void indexing(const void* in,Type type,const int64_t* ids,void* out,int64_t outer,
    int64_t source_axis,int64_t target_axis,int64_t inner,bool copy,int32_t* error) {
  const int64_t count=outer*(copy?source_axis:target_axis)*inner;
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x) {
    auto column=i%inner,row=(i/inner)%(copy?source_axis:target_axis),group=i/inner/(copy?source_axis:target_axis);
    auto mapped=ids[row];
    if(mapped<0||mapped>=(copy?target_axis:source_axis)){atomicExch(error,1);continue;}
    if(copy){
      // Stable last writer for duplicates; no conflicting scatter stores.
      bool last=true;for(int64_t j=row+1;j<source_axis;++j)if(ids[j]==mapped){last=false;break;}
      if(last)copy_one(in,out,type,i,(group*target_axis+mapped)*inner+column);
    } else copy_one(in,out,type,(group*source_axis+mapped)*inner+column,i);
  }
}
__global__ void branch(const int32_t* index,const int32_t* mapping,int32_t targets,int32_t blocks,
    cudaGraphConditionalHandle loop,cudaGraphConditionalHandle pc,int32_t* error) {
  int32_t i=index?*index:0;
  if(*error||i<0||i>=targets){*error=1;cudaGraphSetConditional(loop,0);return;}
  int32_t next=mapping[i];
  if(next<0||next>blocks){*error=1;cudaGraphSetConditional(loop,0);return;}
  cudaGraphSetConditional(pc,next);cudaGraphSetConditional(loop,next<blocks);
}
int grid(int64_t n){return int(n/256+bool(n%256)>65535?65535:n/256+bool(n%256));}
}
void elementwise(cudaStream_t s,Op op,const void* a,Type at,const void* b,Type bt,void* out,Type ot,Layout l) {
  elements<<<grid(l.count),256,0,s>>>(op,a,at,b,bt,out,ot,l);
}
void reduce(cudaStream_t s,const void* in,Type t,void* out,int64_t outer,int64_t axis,int64_t inner,bool softmax) {
  reduction<<<grid(outer*inner),256,0,s>>>(in,t,out,outer,axis,inner,softmax);
}
void normalize(cudaStream_t s,const void* in,Type t,void* out,float* rstd,int64_t rows,int64_t width,double eps,bool rms) {
  norm_rows<<<grid(rows),256,0,s>>>(in,t,out,rstd,rows,width,float(eps),rms);
}
void permute(cudaStream_t s,const void* in,Type t,void* out,Layout l){permutation<<<grid(l.count),256,0,s>>>(in,t,out,l);}
void index(cudaStream_t s,const void* in,Type t,const int64_t* ids,void* out,int64_t o,int64_t a,int64_t b,int64_t n,bool copy,int32_t* e) {
  indexing<<<grid(o*(copy?a:b)*n),256,0,s>>>(in,t,ids,out,o,a,b,n,copy,e);
}
void route(cudaStream_t s,const int32_t* i,const int32_t* m,int32_t n,int32_t b,cudaGraphConditionalHandle l,
    cudaGraphConditionalHandle pc,int32_t* error){branch<<<1,1,0,s>>>(i,m,n,b,l,pc,error);}
}
