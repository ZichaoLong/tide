#include "optimizer_vector.h"
#include "../optimizer_layout.h"
extern "C" __global__ __aicore__ void tide_optimizer_values(GM_ADDR table,GM_ADDR tiles,GM_ADDR options,GM_ADDR flags,
    GM_ADDR gradients,GM_ADDR connected,GM_ADDR values,GM_ADDR first,GM_ADDR second,GM_ADDR maximum,GM_ADDR steps,GM_ADDR corrections,
    GM_ADDR next_values,GM_ADDR next_first,GM_ADDR next_second,GM_ADDR next_maximum,GM_ADDR tile_errors,GM_ADDR error,
    int64_t count,int64_t tasks,int64_t kind) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_device;using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)table);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto t=(__gm__ I*)table,offsets=(__gm__ I*)tiles,fl=(__gm__ I*)flags;auto opt=(__gm__ float*)options;
  OptimizerVector vector;vector.init();auto w=vector.at(0),g=vector.at(1),m=vector.at(2),v=vector.at(3),mx=vector.at(4),tmp=vector.at(5),den=vector.at(6);
  for(I task=AscendC::GetBlockIdx();task<tasks;task+=AscendC::GetBlockNum()) {
    ((__gm__ int32_t*)tile_errors)[task*16]=0;
    I lo=0,hi=count;while(lo+1<hi){I mid=lo+(hi-lo)/2;if(offsets[mid]<=task)lo=mid;else hi=mid;}
    const I i=lo;if(!((__gm__ uint8_t*)connected)[i])continue;
    const I start=(task-offsets[i])*256,offset=t[i*OWNER_FIELDS+OFFSET]+start,remaining=t[i*OWNER_FIELDS+SIZE]-start;
    const uint32_t size=remaining<256?remaining:256;
    const I group=t[i*OWNER_FIELDS+GROUP],o=group*OPTION_COUNT,f=group*FLAG_COUNT;
    const float lr=opt[o+LR],wd=opt[o+WD],momentum=opt[o+MOM],damp=opt[o+DAMP];
    vector.load(w,(__gm__ float*)values,offset,size);vector.load(g,(__gm__ float*)gradients,offset,size);
    bool finite=vector.finite(w,size)&&vector.finite(g,size);
    if(fl[f+MAXIMIZE]){AscendC::Muls(g,g,-1.f,size);AscendC::PipeBarrier<PIPE_V>();}
    if(!kind) {
      if(wd!=0){AscendC::Muls(tmp,w,wd,size);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(g,g,tmp,size);AscendC::PipeBarrier<PIPE_V>();}
      if(momentum!=0) {
        if(((__gm__ I*)steps)[i]==0){AscendC::Muls(m,g,1.f,size);AscendC::PipeBarrier<PIPE_V>();}
        else {vector.load(m,(__gm__ float*)first,offset,size);finite=finite&&vector.finite(m,size);
          AscendC::Muls(m,m,momentum,size);AscendC::Muls(tmp,g,damp,size);AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add(m,m,tmp,size);AscendC::PipeBarrier<PIPE_V>();}
        finite=finite&&vector.finite(m,size);vector.save(m,(__gm__ float*)next_first,offset,size);
        if(fl[f+NESTEROV]){AscendC::Muls(tmp,m,momentum,size);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(g,g,tmp,size);}
        else AscendC::Muls(g,m,1.f,size);
        AscendC::PipeBarrier<PIPE_V>();
      }
      AscendC::Muls(g,g,-lr,size);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(w,w,g,size);AscendC::PipeBarrier<PIPE_V>();
    }else {
      const float b1=opt[o+B1],b1c=opt[o+B1C],b2=opt[o+B2],b2c=opt[o+B2C],eps=opt[o+EPS],decay=opt[o+DECAY];
      vector.load(m,(__gm__ float*)first,offset,size);vector.load(v,(__gm__ float*)second,offset,size);
      finite=finite&&vector.finite(m,size)&&vector.finite(v,size);
      AscendC::Muls(m,m,b1,size);AscendC::Muls(tmp,g,b1c,size);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(m,m,tmp,size);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(tmp,g,g,size);AscendC::Muls(v,v,b2,size);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls(tmp,tmp,b2c,size);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(v,v,tmp,size);AscendC::PipeBarrier<PIPE_V>();
      finite=finite&&vector.finite(m,size)&&vector.finite(v,size);
      vector.save(m,(__gm__ float*)next_first,offset,size);vector.save(v,(__gm__ float*)next_second,offset,size);
      if(fl[f+AMSGRAD]) {
        vector.load(mx,(__gm__ float*)maximum,offset,size);finite=finite&&vector.finite(mx,size);
        AscendC::Max(v,v,mx,size);AscendC::PipeBarrier<PIPE_V>();vector.save(v,(__gm__ float*)next_maximum,offset,size);
      }
      const float c1=((__gm__ float*)corrections)[i*2],c2=((__gm__ float*)corrections)[i*2+1];
      AscendC::Sqrt(den,v,size);AscendC::Duplicate(tmp,c2,size);AscendC::PipeBarrier<PIPE_V>();AscendC::Sqrt(tmp,tmp,size);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Div(den,den,tmp,size);AscendC::PipeBarrier<PIPE_V>();AscendC::Adds(den,den,eps,size);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Div(tmp,m,den,size);AscendC::Muls(w,w,decay,size);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls(tmp,tmp,-lr/c1,size);AscendC::PipeBarrier<PIPE_V>();AscendC::Add(w,w,tmp,size);AscendC::PipeBarrier<PIPE_V>();
    }
    finite=finite&&vector.finite(w,size);
    if(t[i*OWNER_FIELDS+PAYLOAD_HALF]) {
      // Check the actual rounding boundary, not abs(w)<=65504: FP32 masters
      // such as 65512 still round to finite half. Refuse before any owner/slot
      // commit if publication would produce Inf/NaN. Keep the master unrounded.
      auto rounded=den.ReinterpretCast<half>();
      AscendC::Cast(rounded,w,AscendC::RoundMode::CAST_RINT,size);AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(tmp,rounded,AscendC::RoundMode::CAST_NONE,size);AscendC::PipeBarrier<PIPE_V>();
      finite=finite&&vector.finite(tmp,size);
    }
    vector.save(w,(__gm__ float*)next_values,offset,size);
    if(!finite)((__gm__ int32_t*)tile_errors)[task*16]=1;
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
