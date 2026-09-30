#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_content_outputs(GM_ADDR actions,GM_ADDR values,
    GM_ADDR active,GM_ADDR output_nodes,GM_ADDR scales,GM_ADDR coordinates,GM_ADDR payloads,
    GM_ADDR valid,GM_ADDR error,int64_t rows,int64_t ports,int64_t capacity,int64_t width) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)coordinates);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;auto ac=(__gm__ I*)actions,owners=(__gm__ I*)output_nodes,out=(__gm__ I*)coordinates;
  auto live=(__gm__ uint8_t*)valid,on=(__gm__ uint8_t*)active;
  auto x=(__gm__ float*)values,y=(__gm__ float*)payloads,s=(__gm__ float*)scales;
  for(I i=0;i<capacity;++i)live[i]=0;
  I count=0;
  for(I i=0;i<rows&&status[0]==0;++i)if(on[i])for(I port=0;port<ports;++port)if(owners[port]==ac[i*4+1]) {
    if(count==capacity){status[0]=1;break;}
    auto c=out+count*6;c[0]=ac[i*4];c[1]=ac[i*4+1];c[2]=ac[i*4+2];c[3]=0;c[4]=port;c[5]=ac[i*4+2];
    for(I j=0;j<width;++j)y[count*width+j]=x[i*width+j]*s[port];
    live[count]=1;++count;
  }
  if(status[0])for(I i=0;i<capacity;++i)live[i]=0;
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
