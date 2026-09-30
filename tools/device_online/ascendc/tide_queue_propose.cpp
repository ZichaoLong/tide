#include "kernel_operator.h"
namespace {
using I=int64_t;
__aicore__ inline bool coordinate(__gm__ I* p,I nodes,I samples) {
  return p[0]>=0&&p[0]<samples&&p[1]>=0&&p[1]<nodes&&p[2]>=0
      &&p[3]>=0&&p[3]<=1&&p[4]>=0&&p[5]>=0;
}
// Metadata is small compared with payloads. Scalar AIV proposal is independent
// of payload width; actual placement is one bulk gather after device commit.
__aicore__ inline int propose(__gm__ I* old,__gm__ uint8_t* live,__gm__ int32_t* consumed,
    __gm__ I* in,__gm__ uint8_t* present,__gm__ I* coords,__gm__ uint8_t* valid,
    __gm__ I* order,__gm__ I* old_stats,__gm__ I* stats,I capacity,I arrivals,I nodes,I samples) {
  I count=0;
  for(I i=0;i<capacity;++i) {
    if(consumed[i]<0||consumed[i]>1)return 2;
    if(live[i]&&!coordinate(old+i*6,nodes,samples))return 2;
    if(live[i]&&!consumed[i])++count;
  }
  for(I i=0;i<arrivals;++i)if(present[i]) {
    if(!coordinate(in+i*6,nodes,samples))return 2;
    ++count;
  }
  if(count>capacity)return 1;
  I cursor=0;
  for(I i=0;i<capacity+arrivals;++i) {
    bool keep=i<capacity?(live[i]&&!consumed[i]):present[i-capacity];
    if(!keep)continue;
    auto p=i<capacity?old+i*6:in+(i-capacity)*6;
    for(I j=0;j<6;++j)coords[cursor*6+j]=p[j];
    order[cursor]=i;valid[cursor]=1;++cursor;
  }
  for(I i=cursor;i<capacity;++i) {
    order[i]=capacity+arrivals;valid[i]=0;
    for(I j=0;j<6;++j)coords[i*6+j]=0;
  }
  stats[0]=count;stats[1]=old_stats[1]>count?old_stats[1]:count;
  return 0;
}
}
extern "C" __global__ __aicore__ void tide_queue_propose(GM_ADDR old,GM_ADDR live,
    GM_ADDR consumed,GM_ADDR incoming,GM_ADDR present,GM_ADDR coords,GM_ADDR valid,GM_ADDR order,
    GM_ADDR old_stats,GM_ADDR stats,GM_ADDR branch,GM_ADDR error,
    int64_t capacity,int64_t arrivals,int64_t nodes,int64_t samples) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int64_t> cache;cache.SetGlobalBuffer((__gm__ int64_t*)coords);
  AscendC::DataCacheCleanAndInvalid<int64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;go[0]=0;
  if(status[0]==0) {
    status[0]=propose((__gm__ I*)old,(__gm__ uint8_t*)live,(__gm__ int32_t*)consumed,
      (__gm__ I*)incoming,(__gm__ uint8_t*)present,(__gm__ I*)coords,(__gm__ uint8_t*)valid,
      (__gm__ I*)order,(__gm__ I*)old_stats,(__gm__ I*)stats,capacity,arrivals,nodes,samples);
    go[0]=status[0]==0?1:0;
  }
  AscendC::DataCacheCleanAndInvalid<int64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
