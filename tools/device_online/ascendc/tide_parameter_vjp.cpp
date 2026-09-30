#include "fiber_vector.h"
namespace {
using I=int64_t;
__aicore__ inline bool connected(I bank,I row,__gm__ uint8_t* full,__gm__ uint8_t* decay,__gm__ uint8_t* retention,__gm__ uint8_t* scales) {
  return (bank<2||bank>=5)?full[row]:bank==2?decay[row]:bank==3?retention[row]:scales[row];
}
}
extern "C" __global__ __aicore__ void tide_parameter_vjp(GM_ADDR owners,GM_ADDR references,GM_ADDR tiles,
    GM_ADDR weights,GM_ADDR bias,GM_ADDR decay,GM_ADDR retention,GM_ADDR scales,GM_ADDR lh_weights,GM_ADDR lh_biases,GM_ADDR gate,GM_ADDR up,GM_ADDR down,GM_ADDR full_connected,GM_ADDR decay_connected,
    GM_ADDR retention_connected,GM_ADDR scale_connected,GM_ADDR output,GM_ADDR output_connected,GM_ADDR error,int64_t count,int64_t tasks) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)owners);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto owner=(__gm__ I*)owners,ref=(__gm__ I*)references,offset=(__gm__ I*)tiles;
  auto fc=(__gm__ uint8_t*)full_connected,dc=(__gm__ uint8_t*)decay_connected,rc=(__gm__ uint8_t*)retention_connected,sc=(__gm__ uint8_t*)scale_connected;
  // Byte connection flags have exactly one writer, independent of numerical
  // feature tiling; no vector task reads the output flags in this kernel.
  if(AscendC::GetBlockIdx()==0)for(I i=0;i<count;++i) {
    bool on=false;for(I j=owner[i*4];j<owner[i*4+1];++j)on|=connected(ref[j*3],ref[j*3+1],fc,dc,rc,sc);
    ((__gm__ uint8_t*)output_connected)[i]=on;
  }
  tide_device::FiberVector vector;vector.init();auto total=vector.x(),part=vector.y();
  for(I task=AscendC::GetBlockIdx();task<tasks;task+=AscendC::GetBlockNum()) {
    I lo=0,hi=count;while(lo+1<hi){I mid=lo+(hi-lo)/2;if(offset[mid]<=task)lo=mid;else hi=mid;}
    const I i=lo,start=(task-offset[i])*256,remaining=owner[i*4+3]-start;
    const uint32_t size=remaining<256?remaining:256;
    AscendC::Duplicate(total,0.f,size);AscendC::PipeBarrier<PIPE_V>();
    for(I j=owner[i*4];j<owner[i*4+1];++j) {
      const I bank=ref[j*3];if(!connected(bank,ref[j*3+1],fc,dc,rc,sc))continue;
      auto source=(__gm__ float*)(bank==0?weights:bank==1?bias:bank==2?decay:bank==3?retention:bank==4?scales:bank==5?lh_weights:bank==6?lh_biases:bank==7?gate:bank==8?up:down);
      vector.load(part,source,ref[j*3+2]+start,size);AscendC::Add(total,total,part,size);AscendC::PipeBarrier<PIPE_V>();
    }
    vector.save(total,(__gm__ float*)output,owner[i*4+2]+start,size);
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
