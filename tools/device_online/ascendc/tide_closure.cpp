#include "kernel_operator.h"
namespace {
using I=int64_t;
__aicore__ inline I min_i(I a,I b){return a<b?a:b;}
// Initial scalar-pipeline implementation inside one AIV kernel. Metadata only;
// no CPU/AiCPU callback and no numerical route prepass. Parallel tiling is a
// subsequent optimization and must keep the same exact int64 contract.
__aicore__ inline int close_queue(__gm__ I* c,__gm__ uint8_t* valid,__gm__ I* owner,
    __gm__ I* dist,__gm__ I* work,__gm__ I* stop,__gm__ int32_t* ready,__gm__ int32_t* branch,
    I capacity,I nodes,I regions,I samples,I prefill) {
  I end=stop[0];if(end<0)return 2;
  I size=samples*regions;
  for(I j=0;j<size*2+samples;++j)work[j]=end;
  for(I i=0;i<capacity;++i)if(valid[i]) {
    auto p=c+i*6;
    if(p[0]<0||p[0]>=samples||p[1]<0||p[1]>=nodes||p[2]<0||p[3]<0||p[3]>1||p[4]<0||p[5]<0)return 2;
    I b=p[0],r=owner[p[1]],t=p[2];if(r<0||r>=regions)return 2;
    if(t<end) {work[b*regions+r]=min_i(work[b*regions+r],t);work[size*2+b]=min_i(work[size*2+b],t);}
  }
  if(prefill)for(I b=0;b<samples;++b)for(I target=0;target<regions;++target) {
    I bound=end;
    for(I source=0;source<regions;++source) {
      I seed=work[b*regions+source],delay=dist[source*regions+target];
      if(delay<1)return 2;
      bound=min_i(bound,seed+min_i(delay,end-seed));
    }
    work[size+b*regions+target]=bound;
  }
  int any=0;
  for(I i=0;i<capacity;++i)if(valid[i]) {
    auto p=c+i*6;I b=p[0],r=owner[p[1]],t=p[2];
    bool yes=t<end&&(prefill?t<work[size+b*regions+r]:t==work[size*2+b]);
    ready[i]=yes?1:0;any|=ready[i];
  }
  branch[0]=any;return 0;
}
}
extern "C" __global__ __aicore__ void tide_closure(GM_ADDR coordinates,GM_ADDR valid,GM_ADDR owners,
    GM_ADDR distances,GM_ADDR workspace,GM_ADDR stop,GM_ADDR ready,GM_ADDR branch,GM_ADDR error,
    int64_t capacity,int64_t nodes,int64_t regions,int64_t samples,int64_t prefill) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int64_t> cache;cache.SetGlobalBuffer((__gm__ int64_t*)workspace);
  AscendC::DataCacheCleanAndInvalid<int64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto out=(__gm__ int32_t*)ready,go=(__gm__ int32_t*)branch,status=(__gm__ int32_t*)error;
  go[0]=0;for(int64_t i=0;i<capacity;++i)out[i]=0;
  if(status[0]==0)status[0]=close_queue((__gm__ int64_t*)coordinates,(__gm__ uint8_t*)valid,
    (__gm__ int64_t*)owners,(__gm__ int64_t*)distances,(__gm__ int64_t*)workspace,
    (__gm__ int64_t*)stop,out,go,capacity,nodes,regions,samples,prefill);
  if(status[0]){go[0]=0;for(int64_t i=0;i<capacity;++i)out[i]=0;}
  AscendC::DataCacheCleanAndInvalid<int64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
