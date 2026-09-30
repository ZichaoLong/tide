#include "kernel_operator.h"
namespace {
using I=int64_t;
__aicore__ inline bool finite(float f){return (AscendC::GetScalarBitcodeValue<float,uint32_t>(f)&0x7f800000)!=0x7f800000;}
__aicore__ inline int select(__gm__ I* fibers,__gm__ I* frames,__gm__ I* offsets,__gm__ I* members,
    __gm__ I* lengths,__gm__ I* owner,__gm__ I* policy,__gm__ float* scores,__gm__ I* counts,
    __gm__ uint8_t* seen,__gm__ I* clocks,__gm__ uint8_t* present,__gm__ float* padded,
    __gm__ I* order,__gm__ uint8_t* active,I capacity,I nodes,I regions,I samples) {
  I nf=lengths[1],ng=lengths[2];if(nf<0||nf>capacity||ng<0||ng>nf||offsets[0]!=0||offsets[ng]!=nf)return 2;
  for(I i=0;i<capacity*nodes;++i)padded[i]=AscendC::GetScalarBitcodeValue<uint32_t,float>(0xff800000u);
  for(I i=0;i<capacity;++i)order[i]=capacity*nodes;
  for(I frame=0;frame<ng;++frame) {
    auto g=frames+frame*3;I b=g[0],r=g[1],t=g[2];
    if(b<0||b>=samples||r<0||r>=regions||t<0)return 2;
    I first=offsets[frame],end=offsets[frame+1];if(first<0||end<=first||end>nf||end-first>nodes)return 2;
    I history=b*regions+r;if(present[history]&&t<=clocks[history])return 2;
    I previous=-1;
    for(I j=first;j<end;++j) {
      I f=members[j];if(f<0||f>=nf)return 2;auto q=fibers+f*4;I n=q[1];
      if(q[0]!=b||n<=previous||n>=nodes||owner[n]!=r||q[2]!=t||q[3]!=frame)return 2;
      previous=n;if(seen[b*nodes+n]&&counts[b*nodes+n]<0)return 2;
      if(!finite(scores[f]))return 6;
      padded[frame*nodes+j-first]=scores[f];order[f]=frame*nodes+j-first;
    }
    auto p=policy+r*3;I budget=p[0];
    for(I selected=0;selected<budget&&selected<end-first;++selected) {
      I best=-1,best_count=0,best_node=0;float best_score=0;
      for(I j=first;j<end;++j) {
        I f=members[j];if(active[f]||(p[2]&&scores[f]<=0))continue;
        I n=fibers[f*4+1],c=p[1]&&seen[b*nodes+n]?counts[b*nodes+n]:0;
        if(best<0||c<best_count||(c==best_count&&(scores[f]>best_score||(scores[f]==best_score&&n<best_node)))) {
          best=f;best_count=c;best_node=n;best_score=scores[f];
        }
      }
      if(best<0)break;
      I key=b*nodes+best_node,old=seen[key]?counts[key]:0;
      if(old==I(0x7fffffffffffffff))return 5;
      counts[key]=old+1;seen[key]=1;active[best]=1;
    }
    clocks[history]=t;present[history]=1;
  }
  // Empty physical frames have one finite sentinel, avoiding all--inf softmax.
  // They are never selected as a logical fiber or included in a real denominator.
  for(I frame=ng;frame<capacity;++frame)padded[frame*nodes]=0;
  return 0;
}
}
extern "C" __global__ __aicore__ void tide_frame_select(GM_ADDR fibers,GM_ADDR frames,
    GM_ADDR offsets,GM_ADDR members,GM_ADDR lengths,GM_ADDR owner,GM_ADDR policy,GM_ADDR scores,
    GM_ADDR counts,GM_ADDR seen,GM_ADDR clocks,GM_ADDR present,GM_ADDR padded,GM_ADDR order,
    GM_ADDR active,GM_ADDR branch,GM_ADDR error,int64_t capacity,int64_t nodes,int64_t regions,int64_t samples) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int64_t> cache;cache.SetGlobalBuffer((__gm__ I*)counts);
  AscendC::DataCacheCleanAndInvalid<int64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;go[0]=0;
  for(I i=0;i<capacity;++i)((__gm__ uint8_t*)active)[i]=0;
  if(status[0]==0) {
    status[0]=select((__gm__ I*)fibers,(__gm__ I*)frames,(__gm__ I*)offsets,(__gm__ I*)members,
      (__gm__ I*)lengths,(__gm__ I*)owner,(__gm__ I*)policy,(__gm__ float*)scores,(__gm__ I*)counts,
      (__gm__ uint8_t*)seen,(__gm__ I*)clocks,(__gm__ uint8_t*)present,(__gm__ float*)padded,
      (__gm__ I*)order,(__gm__ uint8_t*)active,capacity,nodes,regions,samples);
    go[0]=status[0]==0&&((__gm__ I*)lengths)[2]>0?1:0;
  }
  if(status[0])for(I i=0;i<capacity;++i)((__gm__ uint8_t*)active)[i]=0;
  AscendC::DataCacheCleanAndInvalid<int64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
