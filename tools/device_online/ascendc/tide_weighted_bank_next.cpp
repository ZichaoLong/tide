#include "weighted_allocation_math.h"
// One fixed-size numerical body visits stable (bank, local-row-offset) tiles.
// The cursor is int64[2]; neither bank counts nor tail lengths reach the host.
extern "C" __global__ __aicore__ void tide_weighted_bank_next(GM_ADDR offsets,GM_ADDR cursor,
    GM_ADDR branch,GM_ADDR error,int64_t rows,int64_t banks,int64_t capacity,int64_t reset){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_device;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)cursor);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(AscendC::GetBlockIdx()==0){
    auto index=(__gm__ I*)cursor;auto selected=(__gm__ int32_t*)branch;auto err=(__gm__ int32_t*)error;
    selected[0]=0;
    if(reset){index[0]=-1;index[1]=0;}
    else if(!err[0]){
      const auto off=(__gm__ I*)offsets;I b=index[0],start=index[1];
      if(rows<1||banks<1||capacity<1||capacity>rows||off[0]!=0||b< -1||b>banks||start<0
          ||((b==-1||b==banks)&&start!=0)){err[0]=2;}
      else if(b==banks){} // Repeated completion is harmless; reset starts replay.
      else{
        if(b>=0){
          if(off[b]<0||off[b+1]<off[b]||off[b+1]>rows){err[0]=2;}
          else{const I count=off[b+1]-off[b];
            if(start>=count||start%capacity)err[0]=2;
            else if(count-start>capacity){index[1]=start+capacity;selected[0]=1;}
          }
        }
        if(!err[0]&&!selected[0]){
          ++b;
          while(b<banks){
            if(off[b]<0||off[b+1]<off[b]||off[b+1]>rows){err[0]=2;break;}
            if(off[b+1]>off[b])break;
            ++b;
          }
          if(!err[0]){index[0]=b;index[1]=0;selected[0]=b<banks;}
        }
      }
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
