#include "fiber_vector.h"
namespace {using I=int64_t;
__aicore__ inline void load(tide_device::FiberVector& op,AscendC::LocalTensor<float> x,
    GM_ADDR data,I offset,uint32_t size,I fp16) {
  if(fp16)op.load(x,(__gm__ half*)data,offset,size);else op.load(x,(__gm__ float*)data,offset,size);
}
}
extern "C" __global__ __aicore__ void tide_fiber_vjp_pack(GM_ADDR plan,GM_ADDR rows,GM_ADDR qkv,GM_ADDR qkv_bias,
    GM_ADDR projection,GM_ADDR proposal,GM_ADDR proposal_on,GM_ADDR key_on,GM_ADDR value_on,
    GM_ADDR key,GM_ADDR value,GM_ADDR bias,GM_ADDR lengths,GM_ADDR old_lengths,GM_ADDR coefficients,
    GM_ADDR pooled_root,GM_ADDR source_gradient,GM_ADDR cache_key_gradient,GM_ADDR cache_value_gradient,
    GM_ADDR input,GM_ADDR weight,GM_ADDR additive,GM_ADDR projected,GM_ADDR query,
    GM_ADDR keys,GM_ADDR values,GM_ADDR biases,GM_ADDR lens,GM_ADDR cotangent,GM_ADDR query_root,GM_ADDR connected,
    GM_ADDR gradient,GM_ADDR error,int64_t batch,int64_t sources,int64_t width,int64_t heads,int64_t capacity,
    int64_t chunk,int64_t mode,int64_t fp16,float query_scale) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)plan);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto ids=(__gm__ I*)plan;const I d=width/heads,wt=(width+255)/256,dt=(d+255)/256;
  auto on=(__gm__ uint8_t*)proposal_on,ko=(__gm__ uint8_t*)key_on,vo=(__gm__ uint8_t*)value_on;
  tide_device::FiberVector op;op.init();auto x=op.x(),y=op.y();
  if(mode==0||mode==2) {
    // Projection operands: only Q is required to recompute an attention query.
    // The source VJP uses Q/K/V segments selected by structural connections.
    for(I task=AscendC::GetBlockIdx();task<chunk*wt;task+=AscendC::GetBlockNum()) {
      const I i=task/wt,start=(task%wt)*256,b=ids[i*3],r=ids[i*3+1];
      const uint32_t size=width-start<256?width-start:256;
      if(b>=0)load(op,x,rows,(b*sources+r)*width+start,size,fp16);
      else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
      op.save(x,(__gm__ float*)input,i*width+start,size);
      if(mode==0) {
        if(b>=0&&on[b])load(op,x,qkv_bias,b*3*width+start,size,fp16);
        else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
        op.save(x,(__gm__ float*)additive,i*width+start,size);
      }else {
        const bool active=b>=0&&on[b];
        if(!active){AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
        else op.load(x,(__gm__ float*)source_gradient,(b*sources+r)*width+start,size);
        op.save(x,(__gm__ float*)gradient,i*3*width+start,size);
      }
    }
    if(mode==2)for(I task=AscendC::GetBlockIdx();task<chunk*heads*dt;task+=AscendC::GetBlockNum()) {
      const I start=(task%dt)*256,h=(task/dt)%heads,i=task/dt/heads,b=ids[i*3],r=ids[i*3+1];
      const uint32_t size=d-start<256?d-start:256;
      for(I which=0;which<2;++which) {
        if(b>=0&&(on[b]||(which?vo[b]:ko[b])))
          op.load(x,(__gm__ float*)(which?cache_value_gradient:cache_key_gradient),
            ((b*heads+h)*capacity+((__gm__ I*)old_lengths)[b]+r)*d+start,size);
        else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
        op.save(x,(__gm__ float*)gradient,(i*3+1+which)*width+h*d+start,size);
      }
    }
    const I segments=mode==0?1:3;
    for(I task=AscendC::GetBlockIdx();task<chunk*width*segments*wt;task+=AscendC::GetBlockNum()) {
      const I start=(task%wt)*256,segment=(task/wt)%segments,r=(task/wt/segments)%width,i=task/wt/segments/width,b=ids[i*3];
      const uint32_t size=width-start<256?width-start:256;
      const bool active=b>=0&&(segment==0?on[b]:segment==1?(on[b]||ko[b]):(on[b]||vo[b]));
      if(active)load(op,x,qkv,(b*width+r)*3*width+segment*width+start,size,fp16);
      else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
      op.save(x,(__gm__ float*)weight,(i*width+r)*segments*width+segment*width+start,size);
    }
  }else {
    if(AscendC::GetBlockIdx()==0)for(I i=0;i<chunk;++i) {
      const I b=ids[i*3];((__gm__ uint8_t*)connected)[i]=b>=0;
      ((__gm__ I*)lens)[i]=b<0?0:((__gm__ I*)lengths)[b];
      for(I k=0;k<capacity;++k)((__gm__ float*)biases)[i*capacity+k]=b<0||k>=((__gm__ I*)lengths)[b]?0.f:
        fp16?float(((__gm__ half*)bias)[b*capacity+k]):((__gm__ float*)bias)[b*capacity+k];
    }
    for(I task=AscendC::GetBlockIdx();task<chunk*wt;task+=AscendC::GetBlockNum()) {
      const I i=task/wt,start=(task%wt)*256,b=ids[i*3],r=ids[i*3+1];
      const uint32_t size=width-start<256?width-start:256;
      op.load(x,(__gm__ float*)projected,i*width+start,size);
      if(fp16){AscendC::Muls(x,x,query_scale,size);AscendC::PipeBarrier<PIPE_V>();}
      op.save(x,(__gm__ float*)query,i*width+start,size);
      if(b>=0){op.load(x,(__gm__ float*)pooled_root,b*width+start,size);
        op.save(x,(__gm__ float*)query_root,i*width+start,size);
        const float coefficient=((__gm__ float*)coefficients)[b*sources+r];
        AscendC::Muls(x,x,coefficient,size);AscendC::PipeBarrier<PIPE_V>();}
      else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();op.save(x,(__gm__ float*)query_root,i*width+start,size);}
      op.save(x,(__gm__ float*)cotangent,i*width+start,size);
    }
    for(I task=AscendC::GetBlockIdx();task<chunk*heads*capacity*dt;task+=AscendC::GetBlockNum()) {
      const I start=(task%dt)*256,k=(task/dt)%capacity,h=(task/dt/capacity)%heads,i=task/dt/capacity/heads,b=ids[i*3];
      const uint32_t size=d-start<256?d-start:256;
      for(I which=0;which<2;++which) {
        if(b>=0&&k<((__gm__ I*)lengths)[b])load(op,x,which?value:key,((b*heads+h)*capacity+k)*d+start,size,fp16);
        else {AscendC::Duplicate(x,0.f,size);AscendC::PipeBarrier<PIPE_V>();}
        op.save(x,(__gm__ float*)(which?values:keys),((i*heads+h)*capacity+k)*d+start,size);
      }
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
