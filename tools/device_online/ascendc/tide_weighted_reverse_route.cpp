#include "weighted_reverse_route_math.h"
extern "C" __global__ __aicore__ void tide_weighted_reverse_route(GM_ADDR coordinates,GM_ADDR valid,GM_ADDR indices,GM_ADDR lengths,
    GM_ADDR gradients,GM_ADDR roots,GM_ADDR edges,GM_ADDR times,GM_ADDR present,GM_ADDR bars,GM_ADDR flags,
    GM_ADDR boundary,GM_ADDR boundary_present,GM_ADDR boundary_bars,GM_ADDR boundary_flags,GM_ADDR stage,GM_ADDR cut,GM_ADDR error,
    int64_t rows,int64_t slots,int64_t atoms,int64_t width,int64_t edge_count,int64_t stages,int64_t nodes,int64_t samples,int64_t boundary_count){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  tide_weighted_reverse_device::route((__gm__ I*)coordinates,(__gm__ uint8_t*)valid,(__gm__ I*)indices,(__gm__ I*)lengths,
    (__gm__ float*)gradients,(__gm__ uint8_t*)roots,(__gm__ I*)edges,(__gm__ I*)times,(__gm__ uint8_t*)present,
    (__gm__ float*)bars,(__gm__ uint8_t*)flags,(__gm__ I*)boundary,(__gm__ uint8_t*)boundary_present,
    (__gm__ float*)boundary_bars,(__gm__ uint8_t*)boundary_flags,(__gm__ I*)stage,(__gm__ I*)cut,(__gm__ int32_t*)error,
    rows,slots,atoms,width,edge_count,stages,nodes,samples,boundary_count);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
