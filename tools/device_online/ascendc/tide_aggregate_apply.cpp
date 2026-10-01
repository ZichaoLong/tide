#include "fiber_vector.h"
namespace {
using I=int64_t;
// Fold normalized source contributions in the independently prepared canonical
// Aggregate order. Physical rows stay distinct, including zero contributions.
template<class T>
__aicore__ inline void apply(GM_ADDR fibers,GM_ADDR offsets,GM_ADDR counts,
    GM_ADDR order,GM_ADDR kinds,GM_ADDR coefficients,GM_ADDR weighted,GM_ADDR content,GM_ADDR error,
    int64_t width,int64_t vectorized) {
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)fibers);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto f=(__gm__ I*)fibers,o=(__gm__ I*)offsets,c=(__gm__ I*)counts,k=(__gm__ I*)kinds,rows=(__gm__ I*)order;
  auto coe=(__gm__ float*)coefficients;auto x=(__gm__ T*)weighted,h=(__gm__ T*)content;
  if(!vectorized) {
    if(AscendC::GetBlockIdx()!=0)return;
    for(I event=0;event<c[1];++event)if(k[f[event*4+1]])for(I j=0;j<width;++j) {
      float total=0.f;
      for(I pos=o[event];pos<o[event+1];++pos) {
        const I a=rows[pos];const float value=float(x[a*width+j])*coe[a];x[a*width+j]=T(value);
        total=pos==o[event]?value:total+value;
      }
      h[event*width+j]=T(total);
    }
    AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);return;
  }
  tide_device::FiberVector op;op.init();auto value=op.x(),total=op.y();const I tiles=(width+255)/256;
  for(I task=AscendC::GetBlockIdx();task<c[1]*tiles;task+=AscendC::GetBlockNum()) {
    const I event=task/tiles,start=(task%tiles)*256;if(!k[f[event*4+1]])continue;
    const uint32_t size=width-start<256?width-start:256;
    for(I pos=o[event];pos<o[event+1];++pos) {
      const I a=rows[pos];const float coefficient=coe[a];op.load(value,x,a*width+start,size);
      AscendC::Muls(value,value,coefficient,size);AscendC::PipeBarrier<PIPE_V>();
      if(pos==o[event])AscendC::Muls(total,value,1.f,size);else AscendC::Add(total,total,value,size);
      AscendC::PipeBarrier<PIPE_V>();op.save(value,x,a*width+start,size);
    }
    op.save(total,h,event*width+start,size);
  }
}
} // namespace
extern "C" __global__ __aicore__ void tide_aggregate_apply(GM_ADDR fibers,GM_ADDR offsets,GM_ADDR counts,
    GM_ADDR order,GM_ADDR kinds,GM_ADDR coefficients,GM_ADDR weighted,GM_ADDR content,GM_ADDR error,
    int64_t width,int64_t vectorized,int64_t fp16) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(fp16)apply<half>(fibers,offsets,counts,order,kinds,coefficients,weighted,content,error,width,vectorized);
  else apply<float>(fibers,offsets,counts,order,kinds,coefficients,weighted,content,error,width,vectorized);
}
