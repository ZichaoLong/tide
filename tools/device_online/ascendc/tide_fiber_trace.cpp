#include "fiber_vector.h"
namespace {using I=int64_t;}
// Optional exact cache observability. Both old and proposed rows are retained;
// comparison/clear views are chosen from recorded selection at export only.
extern "C" __global__ __aicore__ void tide_fiber_trace(GM_ADDR events,GM_ADDR counts,GM_ADDR fibers,
    GM_ADDR old_key,GM_ADDR old_value,GM_ADDR old_bias,GM_ADDR key,GM_ADDR value,GM_ADDR bias,
    GM_ADDR meta,GM_ADDR payload,GM_ADDR count,GM_ADDR error,int64_t width,int64_t capacity,int64_t trace_rows) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)events);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;auto e=(__gm__ I*)events,c=(__gm__ I*)counts,f=(__gm__ I*)fibers;
  auto out=(__gm__ I*)meta,n=(__gm__ I*)count;auto v=(__gm__ float*)payload;n[0]=0;
  I total=0;
  for(I i=0;i<c[1]&&status[0]==0;++i) {
    const I size=e[i*7+3]+e[i*7+4];
    if(size>trace_rows-total){status[0]=12;break;}total+=size;
  }
  if(status[0])return;
  tide_device::FiberVector op;op.init();auto x=op.x();
  for(I i=0;i<c[1];++i)for(I kind=0;kind<2;++kind) {
    const I length=e[i*7+3+kind],owner=e[i*7+1],fiber=e[i*7];
    auto k=(__gm__ float*)(kind?key:old_key),val=(__gm__ float*)(kind?value:old_value),b=(__gm__ float*)(kind?bias:old_bias);
    for(I row=0;row<length;++row) {
      const I dst=n[0]++,src=owner*capacity+row;
      out[dst*5]=f[fiber*4];out[dst*5+1]=f[fiber*4+1];out[dst*5+2]=f[fiber*4+2];out[dst*5+3]=kind;out[dst*5+4]=row;
      for(I start=0;start<width;start+=256) {
        const uint32_t size=width-start<256?width-start:256;
        op.load(x,k,src*width+start,size);op.save(x,v,dst*(2*width+1)+start,size);
        op.load(x,val,src*width+start,size);op.save(x,v,dst*(2*width+1)+width+start,size);
      }
      op.load(x,b,src,1);op.save(x,v,dst*(2*width+1)+2*width,1);
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
