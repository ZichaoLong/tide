#include "weighted_raw_update_math.h"
extern "C" __global__ __aicore__ void tide_weighted_raw_reverse(GM_ADDR fields,GM_ADDR old_meta,GM_ADDR meta,GM_ADDR proposal,GM_ADDR mapping,GM_ADDR receive,GM_ADDR written,GM_ADDR bar,GM_ADDR roots,GM_ADDR old_bar,GM_ADDR proposal_bar,GM_ADDR mass_bar,GM_ADDR old_roots,GM_ADDR proposal_roots,GM_ADDR mass_roots,GM_ADDR error,int64_t rows,int64_t width,int64_t capacity,int64_t count,int64_t convolution){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using I=int64_t;AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)fields);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0)tide_weighted_raw_device::reverse((__gm__ I*)fields,(__gm__ I*)old_meta,(__gm__ I*)meta,(__gm__ float*)proposal,(__gm__ I*)mapping,(__gm__ float*)receive,(__gm__ uint8_t*)written,(__gm__ float*)bar,(__gm__ uint8_t*)roots,(__gm__ float*)old_bar,(__gm__ float*)proposal_bar,(__gm__ float*)mass_bar,(__gm__ uint8_t*)old_roots,(__gm__ uint8_t*)proposal_roots,(__gm__ uint8_t*)mass_roots,(__gm__ int32_t*)error,rows,width,capacity,count,convolution);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
