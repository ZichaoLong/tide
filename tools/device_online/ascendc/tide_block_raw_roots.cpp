#include "weighted_allocation_math.h"
extern "C" __global__ __aicore__ void tide_block_raw_roots(GM_ADDR fields,GM_ADDR metadata,
    GM_ADDR mapping,GM_ADDR bar,GM_ADDR roots,GM_ADDR output,GM_ADDR connected,GM_ADDR error,
    int64_t rows,int64_t targets,int64_t width,int64_t capacity,int64_t count){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  using namespace tide_weighted_device;
  auto fs=(__gm__ I*)fields;auto mi=(__gm__ I*)metadata;auto map=(__gm__ I*)mapping;
  auto g=(__gm__ float*)bar;auto flags=(__gm__ uint8_t*)roots;
  auto result=(__gm__ float*)output;auto used=(__gm__ uint8_t*)connected;auto err=(__gm__ int32_t*)error;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer(err);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(!err[0])for(I r=0;r<rows;++r){const I target=map[r],n=mi[r*(4+capacity)+2];
    if(target<0||target>=targets||n<0||n>capacity){err[0]=78;break;}
    for(I f=0;f<count;++f)if(flags[r*count+f]){const I off=fs[4*f],size=fs[4*f+2]*(fs[4*f+3]?n:1);
      used[target*count+f]=1;
      for(I j=0;j<size;++j){const float x=g[r*width+off+j],y=result[target*width+off+j]+x;
        if(!finite(x)||!finite(y)){err[0]=78;break;}result[target*width+off+j]=y;}
      if(err[0])break;
    }if(err[0])break;
  }
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
