#include "kernel_operator.h"
#include "event_sequence.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_attention_tile(GM_ADDR events,GM_ADDR tokens,GM_ADDR ids,
    GM_ADDR bias,GM_ADDR cursor,GM_ADDR indices,GM_ADDR valid,GM_ADDR additive,GM_ADDR branch,
    GM_ADDR work,GM_ADDR error,int64_t chunk,int64_t heads,int64_t kv_heads,int64_t capacity,
    int64_t owners,int64_t tile,int64_t fiber,int64_t event_rows,int64_t fiber_bias_rows) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)cursor);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto e=(__gm__ I*)events,t=(__gm__ I*)tokens,id=(__gm__ I*)ids,pos=(__gm__ I*)cursor;
  auto index=(__gm__ I*)indices,n=(__gm__ I*)valid,stats=(__gm__ I*)work;
  auto a=(__gm__ float*)additive,b=(__gm__ float*)bias;auto status=(__gm__ int32_t*)error;
  auto go=(__gm__ int32_t*)branch;go[0]=0;
  if(status[0]==0) {
    I maximum=0;
    for(I row=0;row<chunk;++row)if(id[row]>=0){const I event=fiber?t[id[row]*4+1]:id[row];
      if(e[event*7+4]>maximum)maximum=e[event*7+4];}
    if(pos[0]<maximum) {
      I real=0;
      for(I row=0;row<chunk;++row) {
        const I event=id[row]<0?-1:fiber?t[id[row]*4+1]:id[row];
        const I left=event<0?0:e[event*7+4]-pos[0];n[row]=left<0?0:left<tile?left:tile;real+=n[row]*heads;
        for(I head=0;head<heads;++head)for(I k=0;k<tile;++k) {
          I source=(owners*capacity+event_rows)*kv_heads;float add=0.f;
          if(k<n[row]) {const I cache_row=event_rows?
              tide_device::event_key_row(e,event,pos[0]+k,capacity,owners):e[event*7+1]*capacity+pos[0]+k;
            source=cache_row*kv_heads+head/(heads/kv_heads);
            if(fiber)add=b[fiber_bias_rows?event*capacity+pos[0]+k:cache_row];}
          index[(row*heads+head)*tile+k]=source;if(head==0)a[row*tile+k]=add;
        }
      }
      const I padded=chunk*heads*tile-real,limit=I(0x7fffffffffffffff);
      if(stats[0]==limit||real>limit-stats[1]||padded>limit-stats[2])status[0]=5;
      else {++stats[0];stats[1]+=real;stats[2]+=padded;go[0]=1;pos[0]+=maximum-pos[0]<tile?maximum-pos[0]:tile;}
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
