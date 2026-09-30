#include "kernel_operator.h"
namespace {
using I=int64_t;
__aicore__ inline int route(__gm__ I* action,__gm__ uint8_t* active,__gm__ I* offsets,
    __gm__ I* edges,__gm__ I* target,__gm__ I* delay,__gm__ I* coords,__gm__ uint8_t* valid,
    __gm__ I* value_order,__gm__ I* scale_order,I rows,I capacity,I nodes,I samples,I edge_count) {
  I count=0;
  // Preflight the complete proposal, including real arrival overflow. Unlike
  // closure lower bounds, actual emitted coordinates must never saturate.
  for(I i=0;i<rows;++i)if(active[i]) {
    auto p=action+i*4;
    if(p[0]<0||p[0]>=samples||p[1]<0||p[1]>=nodes||p[2]<0||p[3]<0)return 2;
    I begin=offsets[p[1]],end=offsets[p[1]+1],degree=end-begin;
    if(degree>capacity-count)return 1;
    count+=degree;
    for(I j=begin;j<end;++j)if(delay[edges[j]]>I(0x7fffffffffffffff)-p[2])return 3;
  }
  I cursor=0;
  for(I i=0;i<rows;++i)if(active[i]) {
    auto p=action+i*4;
    for(I j=offsets[p[1]];j<offsets[p[1]+1];++j) {
      I edge=edges[j];auto out=coords+cursor*6;
      out[0]=p[0];out[1]=target[edge];out[2]=p[2]+delay[edge];
      out[3]=1;out[4]=edge;out[5]=p[3];valid[cursor]=1;
      value_order[cursor]=i;scale_order[cursor]=edge;++cursor;
    }
  }
  for(I i=cursor;i<capacity;++i) {
    valid[i]=0;value_order[i]=rows;scale_order[i]=edge_count>0?edge_count:1;
    for(I j=0;j<6;++j)coords[i*6+j]=0;
  }
  return 0;
}
}
extern "C" __global__ __aicore__ void tide_broadcast_route(GM_ADDR action,GM_ADDR active,
    GM_ADDR offsets,GM_ADDR edges,GM_ADDR targets,GM_ADDR delays,GM_ADDR coords,GM_ADDR valid,
    GM_ADDR value_order,GM_ADDR scale_order,GM_ADDR branch,GM_ADDR error,
    int64_t rows,int64_t capacity,int64_t nodes,int64_t samples,int64_t edge_count) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int64_t> cache;cache.SetGlobalBuffer((__gm__ int64_t*)coords);
  AscendC::DataCacheCleanAndInvalid<int64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;
  go[0]=0;for(I i=0;i<capacity;++i)((__gm__ uint8_t*)valid)[i]=0;
  if(status[0]==0) {
    status[0]=route((__gm__ I*)action,(__gm__ uint8_t*)active,(__gm__ I*)offsets,(__gm__ I*)edges,
      (__gm__ I*)targets,(__gm__ I*)delays,(__gm__ I*)coords,(__gm__ uint8_t*)valid,
      (__gm__ I*)value_order,(__gm__ I*)scale_order,rows,capacity,nodes,samples,edge_count);
    go[0]=status[0]==0?1:0;
  }
  AscendC::DataCacheCleanAndInvalid<int64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
