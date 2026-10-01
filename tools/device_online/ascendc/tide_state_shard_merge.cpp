#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_state_shard_merge(GM_ADDR fibers,GM_ADDR counts,GM_ADDR source_rows,GM_ADDR local_counts,
    GM_ADDR local_meta,GM_ADDR global_meta,GM_ADDR actions,GM_ADDR event_count,GM_ADDR error,int64_t capacity,int64_t diagnostics,int64_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)counts);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;auto f=(__gm__ I*)fibers,out=(__gm__ I*)global_meta;
  const I n=((__gm__ I*)counts)[1];
  if(status[0]==0&&(n<0||n>capacity))status[0]=2;
  if(status[0]==0&&mode==1) {
    auto a=(__gm__ I*)actions,total=(__gm__ I*)event_count;
    if(total[0]<0||n>I(0x7fffffffffffffff)-total[0])status[0]=5;
    else {for(I i=0;i<n;++i){a[i*4]=f[i*4];a[i*4+1]=f[i*4+1];a[i*4+2]=a[i*4+3]=f[i*4+2];}total[0]+=n;}
  }else if(status[0]==0&&diagnostics) {
    const I local=((__gm__ I*)local_counts)[1];auto src=(__gm__ I*)source_rows,meta=(__gm__ I*)local_meta;
    if(local<0||local>capacity)status[0]=2;
    for(I i=0;i<local&&status[0]==0;++i) {
      const I r=src[i];if(r<0||r>=n){status[0]=2;break;}
      for(I j=0;j<13;++j)out[r*13+j]=meta[i*13+j];out[r*13+1]=f[r*4+1];
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
