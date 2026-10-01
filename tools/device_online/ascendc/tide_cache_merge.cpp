#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_cache_merge(GM_ADDR lengths,GM_ADDR other_lengths,
    GM_ADDR left_key,GM_ADDR left_value,GM_ADDR left_key_on,GM_ADDR left_value_on,
    GM_ADDR right_key,GM_ADDR right_value,GM_ADDR right_key_on,GM_ADDR right_value_on,
    GM_ADDR key,GM_ADDR value,GM_ADDR key_on,GM_ADDR value_on,GM_ADDR error,
    int64_t owners,int64_t capacity,int64_t width,int64_t combine,int64_t phase) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)lengths);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;if(status[0])return;
  auto len=(__gm__ I*)lengths;
  if(phase==0) {
    if(AscendC::GetBlockIdx()!=0)return;
    for(I o=0;o<owners;++o) {
      if(len[o]<0||len[o]>capacity||(combine&&len[o]!=((__gm__ I*)other_lengths)[o])){status[0]=2;break;}
      ((__gm__ uint8_t*)key_on)[o]=((__gm__ uint8_t*)left_key_on)[o]||(combine&&((__gm__ uint8_t*)right_key_on)[o]);
      ((__gm__ uint8_t*)value_on)[o]=((__gm__ uint8_t*)left_value_on)[o]||(combine&&((__gm__ uint8_t*)right_value_on)[o]);
    }
  }else {
    tide_device::FiberVector op;op.init();auto x=op.x(),y=op.y();const I tiles=(width+255)/256;
    for(I task=AscendC::GetBlockIdx();task<owners*tiles;task+=AscendC::GetBlockNum()) {
      const I o=task/tiles,start=(task%tiles)*256;const uint32_t size=width-start<256?width-start:256;
      for(I j=0;j<capacity;++j)for(I which=0;which<2;++which) {
        const I at=(o*capacity+j)*width+start;
        AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();
        if(j<len[o]&&((__gm__ uint8_t*)(which?left_value_on:left_key_on))[o])op.load(x,(__gm__ float*)(which?left_value:left_key),at,size);
        if(j<len[o]&&combine&&((__gm__ uint8_t*)(which?right_value_on:right_key_on))[o]) {
          op.load(y,(__gm__ float*)(which?right_value:right_key),at,size);AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();
        }
        op.save(x,(__gm__ float*)(which?value:key),at,size);
      }
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
