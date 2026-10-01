#include "fiber_vector.h"
namespace {
using I=int64_t;
template<class T>
__aicore__ inline void sum(GM_ADDR values,GM_ADDR offsets,GM_ADDR lengths,GM_ADDR keys,GM_ADDR order,
    GM_ADDR scales,GM_ADDR content,GM_ADDR weighted,GM_ADDR error,I width) {
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)keys);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto o=(__gm__ I*)offsets,k=(__gm__ I*)keys,ordered=(__gm__ I*)order;
  auto x=(__gm__ T*)values,w=(__gm__ T*)scales,h=(__gm__ T*)content,z=(__gm__ T*)weighted;
  tide_device::FiberVector op;op.init();auto value=op.x(),total=op.y();
  const I count=((__gm__ I*)lengths)[1],tiles=(width+255)/256;
  for(I task=AscendC::GetBlockIdx();task<count*tiles;task+=AscendC::GetBlockNum()) {
    const I row=task/tiles,start=(task%tiles)*256;
    const uint32_t size=width-start<256?width-start:256;
    for(I pos=o[row];pos<o[row+1];++pos) {
      const I a=ordered[pos];const float scale=float(w[k[a]]);
      op.load(value,x,a*width+start,size);AscendC::Muls(value,value,scale,size);AscendC::PipeBarrier<PIPE_V>();
      // Accumulate before the independent payload storage conversion. All
      // terms and the accumulator use FP32; only declared payloads are rounded.
      if(pos==o[row])AscendC::Muls(total,value,1.f,size);else AscendC::Add(total,total,value,size);
      AscendC::PipeBarrier<PIPE_V>();op.save(value,z,a*width+start,size);
    }
    op.save(total,h,row*width+start,size);
  }
}
}
extern "C" __global__ __aicore__ void tide_vector_sum(GM_ADDR values,GM_ADDR offsets,
    GM_ADDR lengths,GM_ADDR keys,GM_ADDR order,GM_ADDR scales,GM_ADDR content,GM_ADDR weighted,GM_ADDR error,int64_t width,int64_t fp16) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(fp16)sum<half>(values,offsets,lengths,keys,order,scales,content,weighted,error,width);
  else sum<float>(values,offsets,lengths,keys,order,scales,content,weighted,error,width);
}
