#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_emission_chunk(GM_ADDR meta,GM_ADDR count,
    GM_ADDR cursor,GM_ADDR source_order,GM_ADDR parameter_order,GM_ADDR destination_order,
    GM_ADDR branch,GM_ADDR chunks,GM_ADDR error,int64_t capacity,int64_t parameters,int64_t chunk) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)cursor);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto m=(__gm__ I*)meta,n=(__gm__ I*)count,pos=(__gm__ I*)cursor;
  auto s=(__gm__ I*)source_order,p=(__gm__ I*)parameter_order,d=(__gm__ I*)destination_order;
  auto go=(__gm__ int32_t*)branch,status=(__gm__ int32_t*)error;go[0]=0;
  for(I j=0;j<chunk;++j){s[j]=capacity;p[j]=parameters;d[j]=capacity+1+j;}
  if(status[0]==0&&(n[0]<0||n[0]>capacity||pos[0]<0||pos[0]>n[0]))status[0]=2;
  I used=0;
  while(status[0]==0&&pos[0]<n[0]&&used<chunk) {
    I row=pos[0]++,param=m[row*6+5];if(param==-1)continue;
    if(param<0||param>=parameters){status[0]=2;break;}
    s[used]=row;p[used]=param;d[used]=row;++used;
  }
  if(status[0]==0&&used) {
    auto batches=(__gm__ I*)chunks;
    if(batches[0]==I(0x7fffffffffffffff))status[0]=5;
    else {++batches[0];go[0]=1;}
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
