#include "kernel_operator.h"
#include "../optimizer_layout.h"
extern "C" __global__ __aicore__ void tide_optimizer_plan(GM_ADDR table,GM_ADDR options,GM_ADDR connected,
    GM_ADDR steps,GM_ADDR corrections,GM_ADDR next_steps,GM_ADDR next_corrections,GM_ADDR tile_errors,GM_ADDR error,
    int64_t count,int64_t tasks,int64_t kind,int64_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_device;using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)table);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;
  if(status[0])return;
  if(mode==1){for(I i=0;i<tasks;++i)if(((__gm__ int32_t*)tile_errors)[i*16]){status[0]=optimizer_finite_error;break;}}
  else {
    auto t=(__gm__ I*)table,s=(__gm__ I*)steps,ns=(__gm__ I*)next_steps;
    auto c=(__gm__ float*)corrections,nc=(__gm__ float*)next_corrections,opt=(__gm__ float*)options;
    for(I i=0;i<count;++i)if(t[i*OWNER_FIELDS+GROUP]>=0&&((__gm__ uint8_t*)connected)[i]) {
      if(t[i*OWNER_FIELDS+OFFSET]<0||s[i]<0||s[i]==I(0x7fffffffffffffffLL)){status[0]=optimizer_step_error;break;}
      ns[i]=s[i]+1;
      if(kind){const I g=t[i*OWNER_FIELDS+GROUP]*OPTION_COUNT;
        bool valid=true;
        for(I j=0;j<2;++j){const float value=c[i*2+j];
          valid&=(AscendC::GetScalarBitcodeValue<float,uint32_t>(value)&0x7f800000)!=0x7f800000&&value>=0.f&&value<=1.f;}
        if(!valid){status[0]=optimizer_finite_error;break;}
        // Stable 1-beta^step recurrence avoids subtracting two near-one FP32
        // numbers. Complements were computed from the declared host doubles.
        nc[i*2]=opt[g+B1C]+opt[g+B1]*c[i*2];nc[i*2+1]=opt[g+B2C]+opt[g+B2]*c[i*2+1];}
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
