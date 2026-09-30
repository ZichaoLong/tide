#include "fiber_vector.h"
extern "C" __global__ __aicore__ void tide_window_bridge_values(GM_ADDR mapping,GM_ADDR earlier_valid,
    GM_ADDR messages,GM_ADDR message_connected,GM_ADDR initial,GM_ADDR initial_connected,GM_ADDR local_pending,GM_ADDR local_pending_connected,
    GM_ADDR local_final,GM_ADDR local_final_connected,GM_ADDR pending,GM_ADDR final,GM_ADDR error,int64_t rows,int64_t states,int64_t width) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)mapping);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  const I tiles=(width+255)/256;tide_device::FiberVector vector;vector.init();auto sum=vector.x(),other=vector.y();
  for(I task=AscendC::GetBlockIdx();task<(rows+states)*tiles;task+=AscendC::GetBlockNum()) {
    const I row=task/tiles,start=(task%tiles)*256;const bool is_pending=row<rows;const I i=is_pending?row:row-rows;
    const uint32_t size=width-start<256?width-start:256;
    AscendC::Duplicate(sum,0.f,size);AscendC::PipeBarrier<PIPE_V>();
    if(!is_pending||((__gm__ uint8_t*)earlier_valid)[i]) {
      auto local=(__gm__ float*)(is_pending?local_pending:local_final);
      if(((__gm__ uint8_t*)(is_pending?local_pending_connected:local_final_connected))[i])vector.load(sum,local,i*width+start,size);
      const I next=is_pending?((__gm__ I*)mapping)[i]:i;
      if(((__gm__ uint8_t*)(is_pending?message_connected:initial_connected))[next]) {
        vector.load(other,(__gm__ float*)(is_pending?messages:initial),next*width+start,size);
        AscendC::Add(sum,sum,other,size);AscendC::PipeBarrier<PIPE_V>();
      }
    }
    vector.save(sum,(__gm__ float*)(is_pending?pending:final),i*width+start,size);
  }
}
