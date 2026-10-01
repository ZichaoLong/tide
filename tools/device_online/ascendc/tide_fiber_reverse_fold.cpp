#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_fiber_reverse_fold(GM_ADDR plan,GM_ADDR flags,GM_ADDR events,GM_ADDR config,
    GM_ADDR tokens,GM_ADDR links,GM_ADDR scales,GM_ADDR raw,GM_ADDR node_offsets,GM_ADDR source_counts,
    GM_ADDR input_gradient,GM_ADDR input_connected,GM_ADDR key_gradient,GM_ADDR value_gradient,GM_ADDR bias_gradient,GM_ADDR cache_connected,
    GM_ADDR qkv_gradient,GM_ADDR qkv_bias_gradient,GM_ADDR projection_gradient,GM_ADDR projection_bias_gradient,GM_ADDR decay_gradient,GM_ADDR pool_gradient,
    GM_ADDR parameter_connected,GM_ADDR messages,GM_ADDR messages_connected,GM_ADDR scalar_partials,
    GM_ADDR carry_key,GM_ADDR carry_value,GM_ADDR carry_bias,GM_ADDR carry_key_on,GM_ADDR carry_value_on,GM_ADDR carry_bias_on,
    GM_ADDR parameters,GM_ADDR parameters_on,GM_ADDR table,GM_ADDR tile_offsets,GM_ADDR table_count,GM_ADDR error,
    int64_t chunk,int64_t source_capacity,int64_t width,int64_t heads,int64_t capacity,int64_t domain,int64_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)plan);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto p=(__gm__ I*)plan,e=(__gm__ I*)events,cfg=(__gm__ I*)config,t=(__gm__ I*)tokens;
  auto f=(__gm__ uint8_t*)flags,pc=(__gm__ uint8_t*)parameter_connected,cc=(__gm__ uint8_t*)cache_connected;
  auto tab=(__gm__ I*)table,off=(__gm__ I*)tile_offsets,count=(__gm__ I*)table_count;
  const I d=width/heads,wt=(width+255)/256,dt=(d+255)/256;
  if(mode==0) {
    if(AscendC::GetBlockIdx()!=0)return;count[0]=count[1]=0;off[0]=0;
    for(I i=0;i<chunk;++i)if(p[i*10]>=0) {
      const I event=p[i*10],owner=p[i*10+1],param=p[i*10+2],node=e[event*13+1];
      const bool adopt=cfg[param*2]||e[event*13+3];
      for(I which=0;which<3;++which)((__gm__ uint8_t*)(which==0?carry_key_on:which==1?carry_value_on:carry_bias_on))[owner]=
        cc[i*3+which]||(!adopt&&f[i*4+1+which]);
      if(((__gm__ uint8_t*)input_connected)[i])for(I j=0;j<p[i*10+9];++j)
        ((__gm__ uint8_t*)messages_connected)[t[p[i*10+8]+j]]=1;
      bool leader=true;for(I j=0;j<i;++j)if(p[j*10]>=0&&e[p[j*10]*13+1]==node)leader=false;
      if(!leader)continue;
      I position=((__gm__ I*)node_offsets)[node];
      for(I kind=0;kind<6;++kind) {
        const I size=kind==0?3*width*width:kind==1?3*width:kind==2?width*width:kind==3?width:kind==4?1:((__gm__ I*)source_counts)[node];
        bool used=false;for(I j=i;j<chunk;++j)if(p[j*10]>=0&&e[p[j*10]*13+1]==node&&pc[j*6+kind])used=true;
        if(used) {
          ((__gm__ uint8_t*)parameters_on)[node*6+kind]=1;
          if(size) {
            const I row=count[0]++;tab[row*5]=node;tab[row*5+1]=kind;tab[row*5+2]=i;tab[row*5+3]=position;tab[row*5+4]=size;
            count[1]+=(size+255)/256;off[count[0]]=count[1];
          }
        }
        position+=size;
      }
    }
  }else {
    tide_device::FiberVector op;op.init();auto x=op.x(),y=op.y();
    for(I task=AscendC::GetBlockIdx();task<chunk*source_capacity*wt;task+=AscendC::GetBlockNum()) {
      const I start=(task%wt)*256,j=(task/wt)%source_capacity,i=task/wt/source_capacity;
      if(p[i*10]<0||j>=p[i*10+9]||!((__gm__ uint8_t*)input_connected)[i])continue;
      const I row=t[p[i*10+8]+j],at=row*width+start,src=(i*source_capacity+j)*width+start;
      const uint32_t size=width-start<256?width-start:256;const float scale=((__gm__ float*)scales)[((__gm__ I*)links)[row*4+2]];
      op.load(x,(__gm__ float*)input_gradient,src,size);AscendC::Muls(x,x,scale,size);AscendC::PipeBarrier<PIPE_V>();
      op.load(y,(__gm__ float*)messages,at,size);AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();op.save(x,(__gm__ float*)messages,at,size);
      op.load(x,(__gm__ float*)input_gradient,src,size);op.load(y,(__gm__ float*)raw,at,size);
      AscendC::Mul(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();op.load(y,(__gm__ float*)scalar_partials,at,size);
      AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();op.save(x,(__gm__ float*)scalar_partials,at,size);
    }
    for(I task=AscendC::GetBlockIdx();task<chunk*heads*capacity*dt;task+=AscendC::GetBlockNum()) {
      const I start=(task%dt)*256,k=(task/dt)%capacity,h=(task/dt/capacity)%heads,i=task/dt/capacity/heads;
      const I event=p[i*10],param=p[i*10+2],owner=p[i*10+1];if(event<0)continue;
      const uint32_t size=d-start<256?d-start:256;const I at=(owner*capacity+k)*width+h*d+start;
      const bool adopt=cfg[param*2]||e[event*13+3];
      for(I which=0;which<2;++which) {
        auto dst=(__gm__ float*)(which?carry_value:carry_key);
        if(k<p[i*10+5]) {
          op.load(x,(__gm__ float*)(which?value_gradient:key_gradient),((i*heads+h)*capacity+k)*d+start,size);
          if(!adopt&&f[i*4+1+which]){op.load(y,dst,at,size);AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();}
        }else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
        op.save(x,dst,at,size);
      }
    }
    if(AscendC::GetBlockIdx()==0)for(I i=0;i<chunk;++i)if(p[i*10]>=0) {
      const I event=p[i*10],param=p[i*10+2],owner=p[i*10+1];const bool adopt=cfg[param*2]||e[event*13+3];
      for(I k=0;k<capacity;++k) {
        const I at=owner*capacity+k;float result=0;
        if(k<p[i*10+5]){result=((__gm__ float*)bias_gradient)[i*capacity+k];if(!adopt&&f[i*4+3])result+=((__gm__ float*)carry_bias)[at];}
        ((__gm__ float*)carry_bias)[at]=result;
      }
    }
    // Work is proportional to used node/parameter rows in this batch, rather
    // than scanning every node's dense matrices on each reverse iteration.
    for(I task=AscendC::GetBlockIdx();task<count[1];task+=AscendC::GetBlockNum()) {
      I lo=0,hi=count[0];while(lo+1<hi){const I mid=lo+(hi-lo)/2;if(off[mid]<=task)lo=mid;else hi=mid;}
      auto row=tab+lo*5;const I start=(task-off[lo])*256,kind=row[1];const uint32_t size=row[4]-start<256?row[4]-start:256;
      op.load(x,(__gm__ float*)parameters,row[3]+start,size);
      auto src=(__gm__ float*)(kind==0?qkv_gradient:kind==1?qkv_bias_gradient:kind==2?projection_gradient:kind==3?projection_bias_gradient:kind==4?decay_gradient:pool_gradient);
      const I stride=kind==0?3*width*width:kind==1?3*width:kind==2?width*width:kind==3?width:kind==4?1:domain;
      for(I i=row[2];i<chunk;++i)if(p[i*10]>=0&&e[p[i*10]*13+1]==row[0]&&pc[i*6+kind]) {
        op.load(y,src,i*stride+start,size);AscendC::Add(x,x,y,size);AscendC::PipeBarrier<PIPE_V>();
      }
      op.save(x,(__gm__ float*)parameters,row[3]+start,size);
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
