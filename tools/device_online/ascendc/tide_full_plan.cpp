#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_full_plan(GM_ADDR coordinates,GM_ADDR active,GM_ADDR kinds,
    GM_ADDR cursor,GM_ADDR source_order,GM_ADDR parameter_order,GM_ADDR destination_order,
    GM_ADDR branch,GM_ADDR chunks,GM_ADDR error,int64_t rows,int64_t nodes,int64_t chunk) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)cursor);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;
  auto c=(__gm__ I*)coordinates,k=(__gm__ I*)kinds,pos=(__gm__ I*)cursor;
  auto s=(__gm__ I*)source_order,p=(__gm__ I*)parameter_order,d=(__gm__ I*)destination_order;
  auto live=(__gm__ uint8_t*)active;go[0]=0;
  for(I i=0;i<chunk;++i){s[i]=rows;p[i]=nodes;d[i]=rows+i;}
  I count=0;
  if(status[0]==0&&(pos[0]<0||pos[0]>rows))status[0]=2;
  while(status[0]==0&&pos[0]<rows&&count<chunk) {
    I row=pos[0]++;if(!live[row])continue;
    I node=c[row*4+1];if(node<0||node>=nodes){status[0]=2;break;}
    if(k[node]==0)continue;
    s[count]=row;p[count]=node;d[count]=row;++count;
  }
  if(status[0]==0&&count>0) {
    auto batches=(__gm__ I*)chunks;
    if(batches[0]==I(0x7fffffffffffffff))status[0]=5;
    else {go[0]=1;++batches[0];}
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
