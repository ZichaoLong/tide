#include "block_kv_bank_reverse_math.h"
extern "C" __global__ __aicore__ void tide_block_kv_bank_snapshot_vjp(GM_ADDR fields,GM_ADDR written,GM_ADDR mapping,GM_ADDR meta,GM_ADDR bar,GM_ADDR roots,GM_ADDR initial,GM_ADDR initial_roots,GM_ADDR writes,GM_ADDR write_roots,GM_ADDR error,int64_t batch,int64_t length,int64_t nodes,int64_t capacity,int64_t entries,int64_t width,int64_t write_width,int64_t queries,int64_t prefix,int64_t raw){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_block_kv_bank_reverse_device::snapshot((__gm__ int64_t*)fields,(__gm__ uint8_t*)written,(__gm__ int64_t*)mapping,(__gm__ int64_t*)meta,(__gm__ float*)bar,(__gm__ uint8_t*)roots,(__gm__ float*)initial,(__gm__ uint8_t*)initial_roots,(__gm__ float*)writes,(__gm__ uint8_t*)write_roots,(__gm__ int32_t*)error,batch,length,nodes,capacity,entries,width,write_width,queries,prefix,raw);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
