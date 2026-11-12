#include "kernel_operator.h"
// Test-only numerical boundary: ordinary vendor BMM computes staged values.
// This task derives presence and publishes only after scheduler commit.
extern "C" __global__ __aicore__ void tide_block_probe(
    GM_ADDR selected, GM_ADDR meta, GM_ADDR staged, GM_ADDR emitted,
    GM_ADDR output, GM_ADDR present, GM_ADDR error,
    int64_t command, int64_t rows, int64_t slots, int64_t width) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto err=(__gm__ int32_t*)error;auto ids=(__gm__ int64_t*)selected,m=(__gm__ int64_t*)meta;
  auto x=(__gm__ float*)staged,y=(__gm__ float*)output;
  auto send=(__gm__ uint8_t*)emitted,seen=(__gm__ uint8_t*)present;
  if(!err[0]){
    if(m[0]<0||m[0]>rows){err[0]=70;}
    else if(command==0){
      for(int64_t r=0;r<m[0];++r){
        for(int64_t j=0;j<width;++j){const float v=x[r*width+j];
          if(v!=v||v>3.4028234e38f||v< -3.4028234e38f)err[0]=74;}
        send[r]=x[r*width]>=0; // A zero-valued message is present.
      }
    }else if(command==1){
      for(int64_t r=0;r<m[0];++r)if(ids[r]<0||ids[r]>=slots)err[0]=70;
      if(!err[0])for(int64_t r=0;r<m[0];++r){
        seen[ids[r]]=send[r];for(int64_t j=0;j<width;++j)y[ids[r]*width+j]=x[r*width+j];
      }
    }else err[0]=70;
  }
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
