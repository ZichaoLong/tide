#include "block_settle_flow_math.h"
extern "C" __global__ __aicore__ void tide_block_flow_publish(GM_ADDR nodes,GM_ADDR strength,GM_ADDR content,
    GM_ADDR present,GM_ADDR times,GM_ADDR output,GM_ADDR live,GM_ADDR clock,GM_ADDR error,
    int64_t command,int64_t frames,int64_t count,int64_t all_nodes,int64_t width){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(command==0)tide_block_settle_flow_device::emit((__gm__ uint8_t*)present,(__gm__ uint8_t*)live,(__gm__ int32_t*)error,frames*count);
  else if(command==1)tide_block_settle_flow_device::publish((__gm__ int64_t*)nodes,(__gm__ float*)strength,(__gm__ float*)content,
    (__gm__ uint8_t*)present,(__gm__ int64_t*)times,(__gm__ float*)output,(__gm__ uint8_t*)live,(__gm__ int64_t*)clock,
    (__gm__ int32_t*)error,frames,count,all_nodes,width-1);
  else if(command==2)tide_block_settle_flow_device::ports((__gm__ int64_t*)nodes,(__gm__ int64_t*)strength,(__gm__ float*)content,
    (__gm__ uint8_t*)present,(__gm__ int64_t*)times,(__gm__ float*)output,(__gm__ uint8_t*)live,(__gm__ int64_t*)clock,
    (__gm__ int32_t*)error,frames,count,all_nodes,width);
  else ((__gm__ int32_t*)error)[0]=79;
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
