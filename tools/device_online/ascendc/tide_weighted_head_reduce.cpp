#include "weighted_allocation_math.h"
namespace {
using namespace tide_weighted_device;
__aicore__ inline bool reduce(__gm__ I* ids,__gm__ uint8_t* roots,__gm__ float* x,__gm__ float* out,
    I rows,I width,I owner){
  bool good=true;
  for(I j=0;j<width;++j){float value=0;
    for(I row=0;row<rows;++row)if(roots[row]&&ids[row]==owner)value+=x[row*width+j];
    out[owner*width+j]=value;if(!finite(value))good=false;}
  return good;
}
}
extern "C" __global__ __aicore__ void tide_weighted_head_reduce(GM_ADDR owners,GM_ADDR roots,GM_ADDR dx,
    GM_ADDR da,GM_ADDR db,GM_ADDR dc,GM_ADDR dd,GM_ADDR oa,GM_ADDR ob,GM_ADDR oc,GM_ADDR od,
    GM_ADDR connected,GM_ADDR row_errors,GM_ADDR error,int64_t rows,int64_t banks,int64_t input_width,
    int64_t aw,int64_t bw,int64_t cw,int64_t dw){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)owners);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto ids=(__gm__ I*)owners;auto flags=(__gm__ uint8_t*)roots;
  for(I bank=AscendC::GetBlockIdx();bank<banks;bank+=AscendC::GetBlockNum()){
    auto e=(__gm__ int32_t*)row_errors+bank;*e=0;bool present=false;
    for(I row=0;row<rows;++row)if(flags[row]&&ids[row]==bank)present=true;
    ((__gm__ uint8_t*)connected)[bank]=present;
    bool good=reduce(ids,flags,(__gm__ float*)da,(__gm__ float*)oa,rows,aw,bank);
    good=reduce(ids,flags,(__gm__ float*)db,(__gm__ float*)ob,rows,bw,bank)&&good;
    good=reduce(ids,flags,(__gm__ float*)dc,(__gm__ float*)oc,rows,cw,bank)&&good;
    good=reduce(ids,flags,(__gm__ float*)dd,(__gm__ float*)od,rows,dw,bank)&&good;
    if(!good)*e=23;
  }
  // One worker owns the compact buffers in the registered host wrapper.
  if(AscendC::GetBlockIdx()==0)for(I row=0;row<rows;++row)for(I j=0;j<input_width;++j){
    auto value=(__gm__ float*)dx+row*input_width+j;
    if(!flags[row])*value=0;else if(!finite(*value))((__gm__ int32_t*)row_errors)[0]=23;
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
