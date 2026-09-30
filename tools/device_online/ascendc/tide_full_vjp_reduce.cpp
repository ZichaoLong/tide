#include "fiber_vector.h"
namespace {using I=int64_t;}
// Every unique parameter-owner tile has one writer. Chunk rows retain their
// stable order, and repeated nodes/samples do not use conflicting scatter.
extern "C" __global__ __aicore__ void tide_full_vjp_reduce(GM_ADDR owners,GM_ADDR owner_count,GM_ADDR parameters,
    GM_ADDR weight_partials,GM_ADDR bias_partials,GM_ADDR weights,GM_ADDR biases,GM_ADDR error,
    int64_t width,int64_t chunk) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)owners);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  const I count=((__gm__ I*)owner_count)[0],matrix=width*width,mt=(matrix+255)/256,bt=(width+255)/256;
  tide_device::FiberVector vector;vector.init();auto total=vector.x(),part=vector.y();
  for(I task=AscendC::GetBlockIdx();task<count*(mt+bt);task+=AscendC::GetBlockNum()) {
    const I owner=task/(mt+bt),tile=task%(mt+bt),node=((__gm__ I*)owners)[owner];
    const bool bias=tile>=mt;const I size_per_owner=bias?width:matrix,start=(bias?tile-mt:tile)*256;
    const uint32_t size=size_per_owner-start<256?size_per_owner-start:256;
    auto output=(__gm__ float*)(bias?biases:weights),input=(__gm__ float*)(bias?bias_partials:weight_partials);
    vector.load(total,output,node*size_per_owner+start,size);
    for(I row=0;row<chunk;++row)if(((__gm__ I*)parameters)[row]==node) {
      vector.load(part,input,row*size_per_owner+start,size);AscendC::Add(total,total,part,size);AscendC::PipeBarrier<PIPE_V>();
    }
    vector.save(total,output,node*size_per_owner+start,size);
  }
}
