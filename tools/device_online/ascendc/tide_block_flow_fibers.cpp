#include "block_settle_flow_math.h"
extern "C" __global__ __aicore__ void tide_block_flow_fibers(GM_ADDR atoms,GM_ADDR counts,GM_ADDR bank,
    GM_ADDR positions,GM_ADDR external,GM_ADDR external_live,GM_ADDR messages,GM_ADDR message_live,
    GM_ADDR payload,GM_ADDR lengths,GM_ADDR source_owners,GM_ADDR node_owners,GM_ADDR times,GM_ADDR mapping,GM_ADDR error,
    int64_t batch,int64_t length,int64_t nodes,int64_t capacity,int64_t width,int64_t input_ports,int64_t all_nodes,int64_t rank,int64_t stride){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_settle_flow_device::fibers((__gm__ int64_t*)atoms,(__gm__ int64_t*)counts,(__gm__ int64_t*)bank,
    (__gm__ int64_t*)positions,(__gm__ float*)external,(__gm__ uint8_t*)external_live,
    (__gm__ float*)messages,(__gm__ uint8_t*)message_live,(__gm__ float*)payload,(__gm__ int64_t*)lengths,
    (__gm__ int64_t*)source_owners,(__gm__ int64_t*)node_owners,(__gm__ int64_t*)times,(__gm__ int64_t*)mapping,
    (__gm__ int32_t*)error,batch,length,nodes,capacity,width,input_ports,all_nodes,rank,stride);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
