#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_state_shard_selection(GM_ADDR rows,GM_ADDR counts,
    GM_ADDR active,GM_ADDR controls,GM_ADDR local_active,GM_ADDR local_controls,GM_ADDR error,int64_t capacity,int64_t global_rows) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)counts);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto index=(__gm__ I*)rows;const I n=((__gm__ I*)counts)[1];auto status=(__gm__ int32_t*)error;
  auto on=(__gm__ uint8_t*)active,out=(__gm__ uint8_t*)local_active;
  auto control=(__gm__ float*)controls,value=(__gm__ float*)local_controls;
  for(I i=0;i<capacity;++i){out[i]=0;value[i]=0;}
  if(status[0]==0&&(n<0||n>capacity))status[0]=2;
  for(I i=0;i<n&&status[0]==0;++i) {
    const I r=index[i];if(r<0||r>=global_rows){status[0]=2;break;}
    out[i]=on[r];value[i]=control[r];
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
