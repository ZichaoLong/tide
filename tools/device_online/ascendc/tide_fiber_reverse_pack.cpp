#include "fiber_vector.h"
namespace {using I=int64_t;
__aicore__ inline void load(tide_device::FiberVector& op,AscendC::LocalTensor<float> x,
    GM_ADDR data,I offset,uint32_t size,I fp16) {
  if(fp16)op.load(x,(__gm__ half*)data,offset,size);else op.load(x,(__gm__ float*)data,offset,size);
}
__aicore__ inline void save(tide_device::FiberVector& op,AscendC::LocalTensor<float> x,
    GM_ADDR data,I offset,uint32_t size,I fp16) {
  if(fp16)op.save(x,(__gm__ half*)data,offset,size);else op.save(x,(__gm__ float*)data,offset,size);
}
}
extern "C" __global__ __aicore__ void tide_fiber_reverse_pack(GM_ADDR plan,GM_ADDR flags,GM_ADDR events,GM_ADDR config,
    GM_ADDR tokens,GM_ADDR fiber_meta,GM_ADDR sources,GM_ADDR messages,GM_ADDR scales,GM_ADDR fiber_values,
    GM_ADDR journal,GM_ADDR ticks,GM_ADDR qkv,GM_ADDR qkv_bias,GM_ADDR projection,GM_ADDR kinds,GM_ADDR domains,GM_ADDR weights,
    GM_ADDR proposal,GM_ADDR carry_key,GM_ADDR carry_value,GM_ADDR carry_bias,
    GM_ADDR x,GM_ADDR slots,GM_ADDR counts,GM_ADDR key,GM_ADDR value,GM_ADDR bias,GM_ADDR lengths,GM_ADDR old_lengths,GM_ADDR elapsed,
    GM_ADDR wqkv,GM_ADDR bqkv,GM_ADDR wo,GM_ADDR pool_kinds,GM_ADDR pool_lengths,GM_ADDR pool_weights,
    GM_ADDR root,GM_ADDR on,GM_ADDR key_root,GM_ADDR value_root,GM_ADDR bias_root,GM_ADDR key_on,GM_ADDR value_on,GM_ADDR bias_on,
    GM_ADDR error,int64_t chunk,int64_t source_capacity,int64_t width,int64_t heads,int64_t capacity,int64_t domain,int64_t inputs,int64_t mode,int64_t fp16) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)plan);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;if(status[0])return;
  auto p=(__gm__ I*)plan,e=(__gm__ I*)events,cfg=(__gm__ I*)config,t=(__gm__ I*)tokens,m=(__gm__ I*)fiber_meta,s=(__gm__ I*)sources;
  auto f=(__gm__ uint8_t*)flags;const I d=width/heads,wt=(width+255)/256,dt=(d+255)/256;
  if(mode==0) {
    if(AscendC::GetBlockIdx()!=0)return;
    for(I i=0;i<chunk;++i) {
      const I event=p[i*10],param=p[i*10+2],n=event<0?0:p[i*10+9];
      if(n>source_capacity){status[0]=2;break;}
      const bool adopt=event>=0&&(cfg[param*2]||e[event*13+3]);
      ((__gm__ I*)counts)[i]=n;((__gm__ I*)lengths)[i]=event<0?0:p[i*10+7];
      ((__gm__ I*)old_lengths)[i]=event<0?0:p[i*10+5];((__gm__ I*)elapsed)[i]=event<0?0:((__gm__ I*)ticks)[event];
      ((__gm__ I*)pool_kinds)[i]=event<0?0:((__gm__ I*)kinds)[param];
      ((__gm__ I*)pool_lengths)[i]=event<0?0:((__gm__ I*)domains)[param];
      ((__gm__ uint8_t*)on)[i]=f[i*4];((__gm__ uint8_t*)key_on)[i]=adopt&&f[i*4+1];
      ((__gm__ uint8_t*)value_on)[i]=adopt&&f[i*4+2];((__gm__ uint8_t*)bias_on)[i]=adopt&&f[i*4+3];
      for(I j=0;j<source_capacity;++j) {
        I slot=-1;if(j<n){const I row=t[p[i*10+8]+j],source=m[row*6+4]+(m[row*6+3]?inputs:0);slot=s[source*2+1];}
        ((__gm__ I*)slots)[i*source_capacity+j]=slot;
      }
    }
  }else {
    tide_device::FiberVector op;op.init();auto a=op.x();
    auto prop=(__gm__ uint8_t*)on,ko=(__gm__ uint8_t*)key_on,vo=(__gm__ uint8_t*)value_on;
    for(I task=AscendC::GetBlockIdx();task<chunk*source_capacity*wt;task+=AscendC::GetBlockNum()) {
      const I start=(task%wt)*256,j=(task/wt)%source_capacity,i=task/wt/source_capacity;
      const uint32_t size=width-start<256?width-start:256;
      if((prop[i]||ko[i]||vo[i])&&j<((__gm__ I*)counts)[i]) {
        const I row=t[p[i*10+8]+j],scale=((__gm__ I*)messages)[row*4+2];const float factor=((__gm__ float*)scales)[scale];
        op.load(a,(__gm__ float*)fiber_values,row*width+start,size);AscendC::Muls(a,a,factor,size);AscendC::PipeBarrier<PIPE_V>();
      }else {AscendC::Duplicate(a,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
      // The physical source product is rounded before QKV, just as forward.
      save(op,a,x,(i*source_capacity+j)*width+start,size,fp16);
    }
    for(I task=AscendC::GetBlockIdx();task<chunk*wt;task+=AscendC::GetBlockNum()) {
      const I i=task/wt,start=(task%wt)*256;const uint32_t size=width-start<256?width-start:256;
      if(prop[i])op.load(a,(__gm__ float*)proposal,p[i*10+3]*width+start,size);
      else {AscendC::Duplicate(a,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
      op.save(a,(__gm__ float*)root,i*width+start,size);
    }
    for(I task=AscendC::GetBlockIdx();task<chunk*width*4*wt;task+=AscendC::GetBlockNum()) {
      const I start=(task%wt)*256,kind=(task/wt)%4,row=(task/wt/4)%width,i=task/wt/4/width,param=p[i*10+2];
      const uint32_t size=width-start<256?width-start:256;
      if(prop[i]||(kind==1&&ko[i])||(kind==2&&vo[i]))
        load(op,a,kind==3?projection:qkv,(param*width+row)*(kind==3?width:3*width)+(kind==3?0:kind*width)+start,size,fp16);
      else {AscendC::Duplicate(a,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
      save(op,a,kind==3?wo:wqkv,(i*width+row)*(kind==3?width:3*width)+(kind==3?0:kind*width)+start,size,fp16);
    }
    const I bt=(3*width+255)/256,pt=(domain+255)/256;
    for(I task=AscendC::GetBlockIdx();task<chunk*(bt+pt);task+=AscendC::GetBlockNum()) {
      const I i=task/(bt+pt),tile=task%(bt+pt),param=p[i*10+2];const bool pool=tile>=bt;
      const I start=(pool?tile-bt:tile)*256,size0=(pool?domain:3*width)-start;
      const uint32_t size=size0<256?size0:256;
      if(prop[i]&&(!pool||((__gm__ I*)pool_kinds)[i]>=2))
        load(op,a,pool?weights:qkv_bias,param*(pool?domain:3*width)+start,size,pool?0:fp16);
      else {AscendC::Duplicate(a,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
      save(op,a,pool?pool_weights:bqkv,i*(pool?domain:3*width)+start,size,pool?0:fp16);
    }
    for(I task=AscendC::GetBlockIdx();task<chunk*heads*capacity*dt;task+=AscendC::GetBlockNum()) {
      const I start=(task%dt)*256,k=(task/dt)%capacity,h=(task/dt/capacity)%heads,i=task/dt/capacity/heads;
      const uint32_t size=d-start<256?d-start:256;const I event=p[i*10],param=p[i*10+2],owner=p[i*10+1];
      const bool clear=event>=0&&cfg[param*2+1]&&e[event*13+3];
      for(I which=0;which<2;++which) {
        if(prop[i]&&k<p[i*10+7])op.load(a,(__gm__ float*)journal,(p[i*10+6]+k)*(2*width+1)+which*width+h*d+start,size);
        else {AscendC::Duplicate(a,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
        save(op,a,which?value:key,((i*heads+h)*capacity+k)*d+start,size,fp16);
        if(!clear&&(which?vo[i]:ko[i])&&k<p[i*10+7])
          op.load(a,(__gm__ float*)(which?carry_value:carry_key),(owner*capacity+k)*width+h*d+start,size);
        else {AscendC::Duplicate(a,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
        op.save(a,(__gm__ float*)(which?value_root:key_root),((i*heads+h)*capacity+k)*d+start,size);
      }
    }
    if(AscendC::GetBlockIdx()==0)for(I i=0;i<chunk;++i) {
      const I event=p[i*10],param=p[i*10+2],owner=p[i*10+1];
      const bool clear=event>=0&&cfg[param*2+1]&&e[event*13+3];
      for(I k=0;k<capacity;++k) {
        const bool valid=event>=0&&k<p[i*10+7];
        const float log_bias=valid&&prop[i]?((__gm__ float*)journal)[(p[i*10+6]+k)*(2*width+1)+2*width]:0.f;
        if(fp16)((__gm__ half*)bias)[i*capacity+k]=half(log_bias);else ((__gm__ float*)bias)[i*capacity+k]=log_bias;
        ((__gm__ float*)bias_root)[i*capacity+k]=valid&&!clear&&((__gm__ uint8_t*)bias_on)[i]?
          ((__gm__ float*)carry_bias)[owner*capacity+k]:0.f;
      }
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
