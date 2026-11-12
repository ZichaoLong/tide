#include "weighted_allocation_math.h"
// This is a stable partition of actual owners, not a host-precomputed route.
extern "C" __global__ __aicore__ void tide_weighted_bank_plan(GM_ADDR owners,GM_ADDR roots,
    GM_ADDR permutation,GM_ADDR offsets,GM_ADDR cursors,GM_ADDR row_connected,
    GM_ADDR owner_connected,GM_ADDR error,int64_t rows,int64_t banks,int64_t reverse){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_device;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)owners);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0){
    auto ids=(__gm__ I*)owners;auto off=(__gm__ I*)offsets;auto cursor=(__gm__ I*)cursors;
    auto present=(__gm__ uint8_t*)row_connected;
    auto used=(__gm__ uint8_t*)owner_connected;auto err=(__gm__ int32_t*)error;
    for(I b=0;b<banks;++b){off[b]=0;cursor[b]=0;used[b]=0;}off[banks]=0;
    for(I r=0;r<rows;++r){present[r]=0;((__gm__ I*)permutation)[r]=0;}
    if(!err[0]){
      for(I r=0;r<rows;++r){
        const I b=ids[r];if(b< -1||b>=banks){err[0]=2;break;}
        if(b>=0&&(!reverse||((__gm__ uint8_t*)roots)[r])){present[r]=1;++off[b+1];}
      }
      if(!err[0]){
        for(I b=0;b<banks;++b){const I count=off[b+1];off[b+1]+=off[b];cursor[b]=off[b];used[b]=count>0;}
        for(I r=0;r<rows;++r)if(present[r])((__gm__ I*)permutation)[cursor[ids[r]]++]=r;
      }
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
