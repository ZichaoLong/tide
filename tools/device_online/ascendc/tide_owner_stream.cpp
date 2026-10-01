#include "fiber_vector.h"
// mode0 packs actual connected payloads; mode1 accumulates one ordered partial;
// mode2 publishes a master to its local payload aliases. No peer address is read.
extern "C" __global__ __aicore__ void tide_owner_stream(GM_ADDR descriptors,GM_ADDR groups,GM_ADDR writes,
    GM_ADDR packet,GM_ADDR connected,GM_ADDR cursor,GM_ADDR branch,GM_ADDR error,
    int64_t count,int64_t total,int64_t capacity,int64_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)descriptors);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto d=(__gm__ I*)descriptors,g=(__gm__ I*)groups,w=(__gm__ I*)writes;
  auto on=(__gm__ uint8_t*)connected;const I begin=((__gm__ I*)cursor)[0];
  const bool bad=((__gm__ int32_t*)error)[0]!=0;
  const I active=capacity<total-begin?capacity:total-begin;
  if(AscendC::GetBlockIdx()==0) {
    if(mode==0) {
      ((__gm__ int32_t*)branch)[0]=!bad&&capacity<total-begin;
      if(!bad)for(I i=0;i<count;++i)on[i]=*(__gm__ uint8_t*)(uint64_t)d[i*4+3];
    } else if(!bad&&mode==1)for(I i=0;i<count;++i) {
      if(!on[i]||d[i*4]>=begin+active||d[i*4]+d[i*4+1]<=begin)continue;
      for(I j=g[i*2];j<g[i*2+1];++j)if(w[j*7+6])*(__gm__ uint8_t*)(uint64_t)w[j*7+6]=1;
    }
  }
  if(bad) {
    AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
    return;
  }
  tide_device::FiberVector op;op.init();auto x=op.x(),old=op.y();
  for(I tile=AscendC::GetBlockIdx();tile<(active+255)/256;tile+=AscendC::GetBlockNum()) {
    I pos=tile*256;const I end=pos+256<active?pos+256:active;
    while(pos<end) {
      const I absolute=begin+pos;I lo=0,hi=count;
      while(lo+1<hi){const I mid=lo+(hi-lo)/2;if(d[mid*4]<=absolute)lo=mid;else hi=mid;}
      const auto row=d+lo*4;const I offset=absolute-row[0];
      const I size=row[1]-offset<end-pos?row[1]-offset:end-pos;
      const bool live=mode==0?*(__gm__ uint8_t*)(uint64_t)row[3]:on[lo];
      if(mode==0) {
        if(live)op.load(x,(__gm__ float*)(uint64_t)row[2],offset,size);
        else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
        op.save(x,(__gm__ float*)packet,pos,size);
      } else if(live)for(I j=g[lo*2];j<g[lo*2+1];++j) {
        const auto target=w+j*7;I done=0;
        while(done<size) {
          const I at=offset+done,r=at/target[2],col=at%target[2];
          const I n=target[2]-col<size-done?target[2]-col:size-done;
          op.load(x,(__gm__ float*)packet,pos+done,n);
          if(mode==1) {
            op.load(old,(__gm__ float*)(uint64_t)target[0],r*target[3]+col,n);
            AscendC::Add(x,old,x,n);AscendC::PipeBarrier<PIPE_V>();
          }
          if(target[4])op.save(x,(__gm__ half*)(uint64_t)target[0],r*target[3]+col,n);
          else {
            if(target[5]) {
              auto half_value=old.ReinterpretCast<half>();
              AscendC::Cast(half_value,x,AscendC::RoundMode::CAST_RINT,n);AscendC::PipeBarrier<PIPE_V>();
              AscendC::Cast(x,half_value,AscendC::RoundMode::CAST_NONE,n);AscendC::PipeBarrier<PIPE_V>();
            }
            op.save(x,(__gm__ float*)(uint64_t)target[0],r*target[3]+col,n);
          }
          done+=n;
        }
      }
      pos+=size;
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
