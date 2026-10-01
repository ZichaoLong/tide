#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_graph_reverse_meta(GM_ADDR metadata,GM_ADDR offsets,GM_ADDR stages,
    GM_ADDR messages,GM_ADDR valid,GM_ADDR producer_head,GM_ADDR producer_next,GM_ADDR consumer_head,GM_ADDR consumer_next,
    GM_ADDR scale_head,GM_ADDR scale_next,GM_ADDR pending_connected,GM_ADDR output_connected,GM_ADDR final_connected,
    GM_ADDR connected,GM_ADDR carry_connected,GM_ADDR stage_meta,GM_ADDR stage_count,GM_ADDR full_connected,GM_ADDR cot_connected,
    GM_ADDR cursor,GM_ADDR range,GM_ADDR branch,GM_ADDR full_h,GM_ADDR full_c,GM_ADDR full_parameters,
    GM_ADDR state_h,GM_ADDR state_decay,GM_ADDR state_retention,GM_ADDR decay_accum,GM_ADDR retention_accum,
    GM_ADDR weight_connected,GM_ADDR decay_connected,GM_ADDR retention_connected,GM_ADDR scalar_connected,
    GM_ADDR processed,GM_ADDR error,int64_t capacity,int64_t fibers,int64_t pending,int64_t outputs,
    int64_t nodes,int64_t samples,int64_t parameters,int64_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)metadata);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;
  auto e=(__gm__ I*)metadata,off=(__gm__ I*)offsets,pos=(__gm__ I*)cursor,r=(__gm__ I*)range;
  auto n=(__gm__ I*)stage_count,meta=(__gm__ I*)stage_meta;
  auto ph=(__gm__ I*)producer_head,pn=(__gm__ I*)producer_next,ch=(__gm__ I*)consumer_head,cn=(__gm__ I*)consumer_next;
  auto on=(__gm__ uint8_t*)connected,carry=(__gm__ uint8_t*)carry_connected,fc=(__gm__ uint8_t*)full_connected,cc=(__gm__ uint8_t*)cot_connected;
  if(mode==0){pos[0]=((__gm__ I*)stages)[0]-1;n[0]=0;r[0]=0;r[1]=0;go[0]=0;}
  if(mode==1)go[0]=0;
  if(status[0])return;
  if(mode==0) {
    for(I i=0;i<fibers+pending+outputs;++i) {
      const bool live=((__gm__ uint8_t*)valid)[i];
      on[i]=live&&(i<fibers?false:i<fibers+pending?((__gm__ uint8_t*)pending_connected)[i-fibers]:
        ((__gm__ uint8_t*)output_connected)[i-fibers-pending]);
    }
    for(I key=0;key<samples*nodes;++key)carry[key]=((__gm__ uint8_t*)final_connected)[key];
  } else if(mode==1) {
    if(pos[0]<0)return;
    r[0]=off[pos[0]];r[1]=off[pos[0]+1];n[0]=r[1]-r[0];
    for(I i=0;i<capacity;++i)fc[i]=0;
    for(I i=0;i<n[0];++i) {
      const I event=r[0]+i;for(I j=0;j<13;++j)meta[i*13+j]=e[event*13+j];
      for(I m=ph[event];m>=0;m=pn[m])if(on[m]){fc[i]=1;break;}
    }
    go[0]=1;
  } else if(mode==2) {
    for(I i=0;i<capacity;++i)for(I j=0;j<5;++j)cc[i*5+j]=0;
    for(I i=0;i<n[0];++i){cc[i*5]=((__gm__ uint8_t*)full_h)[i];cc[i*5+3]=((__gm__ uint8_t*)full_c)[i];}
  } else if(mode==3) {
    for(I i=0;i<n[0];++i)for(I m=ch[r[0]+i];m>=0;m=cn[m])on[m]|=((__gm__ uint8_t*)state_h)[i];
    for(I node=0;node<nodes;++node)((__gm__ uint8_t*)weight_connected)[node]|=((__gm__ uint8_t*)full_parameters)[node];
    for(I key=0;key<samples*nodes;++key){((__gm__ uint8_t*)decay_accum)[key]|=((__gm__ uint8_t*)state_decay)[key];
      ((__gm__ uint8_t*)retention_accum)[key]|=((__gm__ uint8_t*)state_retention)[key];}
    --pos[0];++((__gm__ I*)processed)[0];
  } else if(mode==4) {
    for(I node=0;node<nodes;++node)for(I b=0;b<samples;++b) {
      ((__gm__ uint8_t*)decay_connected)[node]|=((__gm__ uint8_t*)decay_accum)[b*nodes+node];
      ((__gm__ uint8_t*)retention_connected)[node]|=((__gm__ uint8_t*)retention_accum)[b*nodes+node];
    }
    auto head=(__gm__ I*)scale_head,next=(__gm__ I*)scale_next;
    for(I param=0;param<parameters;++param)for(I j=head[param];j>=0;j=next[j])if(on[j/2]) {
      ((__gm__ uint8_t*)scalar_connected)[param]=1;break;
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
