#include "kernel_operator.h"
namespace {
using I=int64_t;
__aicore__ inline bool less(__gm__ I* a,__gm__ I* b,I fields) {
  for(I j=0;j<fields;++j)if(a[j]!=b[j])return a[j]<b[j];
  return false;
}
__aicore__ inline bool same(__gm__ I* a,__gm__ I* b,I fields) {
  for(I j=0;j<fields;++j)if(a[j]!=b[j])return false;
  return true;
}
// Stable insertion order on exact keys. This first implementation keeps the
// metadata work in one AIV task; parallel merge/scan is a later optimization.
__aicore__ inline void pack(__gm__ I* input,__gm__ int32_t* ready,__gm__ I* owner,
    __gm__ I* order,__gm__ I* coords,__gm__ uint8_t* valid,__gm__ I* offsets,
    __gm__ I* fibers,__gm__ I* frame_offsets,__gm__ I* frame_fibers,__gm__ I* frames,
    __gm__ I* counts,I capacity) {
  I atoms=0;
  for(I i=0;i<capacity;++i)if(ready[i]) {
    I j=atoms;
    while(j>0&&less(input+i*6,input+order[j-1]*6,6)){order[j]=order[j-1];--j;}
    order[j]=i;++atoms;
  }
  I nf=0;
  for(I i=0;i<atoms;++i) {
    for(I j=0;j<6;++j)coords[i*6+j]=input[order[i]*6+j];valid[i]=1;
    if(i==0||!same(coords+i*6,coords+(i-1)*6,3)) {
      offsets[nf]=i;for(I j=0;j<3;++j)fibers[nf*4+j]=coords[i*6+j];++nf;
    }
  }
  for(I i=atoms;i<capacity;++i) {order[i]=capacity;valid[i]=0;for(I j=0;j<6;++j)coords[i*6+j]=0;}
  for(I i=nf;i<=capacity;++i)offsets[i]=atoms;
  // Frame traversal is (sample,region,time,node). Keep fiber order separately
  // for state sequences; never split the selector's candidate denominator.
  for(I i=0;i<nf;++i) {
    auto a=fibers+i*4;I j=i;
    while(j>0) {
      auto b=fibers+frame_fibers[j-1]*4;
      bool before=a[0]<b[0]||(a[0]==b[0]&&(owner[a[1]]<owner[b[1]]
        ||(owner[a[1]]==owner[b[1]]&&(a[2]<b[2]||(a[2]==b[2]&&a[1]<b[1])))));
      if(!before)break;
      frame_fibers[j]=frame_fibers[j-1];--j;
    }
    frame_fibers[j]=i;
  }
  I nframes=0;
  for(I i=0;i<nf;++i) {
    auto f=fibers+frame_fibers[i]*4;
    bool fresh=nframes==0;
    if(!fresh){auto prev=frames+(nframes-1)*3;fresh=prev[0]!=f[0]||prev[1]!=owner[f[1]]||prev[2]!=f[2];}
    if(fresh){auto row=frames+nframes*3;row[0]=f[0];row[1]=owner[f[1]];row[2]=f[2];frame_offsets[nframes]=i;++nframes;}
    f[3]=nframes-1;
  }
  for(I i=nframes;i<=capacity;++i)frame_offsets[i]=nf;
  counts[0]=atoms;counts[1]=nf;counts[2]=nframes;
}
}
extern "C" __global__ __aicore__ void tide_ready_pack(GM_ADDR input,GM_ADDR live,
    GM_ADDR ready,GM_ADDR owner,GM_ADDR order,GM_ADDR coords,GM_ADDR valid,GM_ADDR offsets,
    GM_ADDR fibers,GM_ADDR frame_offsets,GM_ADDR frame_fibers,GM_ADDR frames,GM_ADDR counts,
    GM_ADDR branch,GM_ADDR error,GM_ADDR causal,GM_ADDR first,int64_t capacity,int64_t regions,int64_t samples) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<int64_t> cache;cache.SetGlobalBuffer((__gm__ I*)coords);
  AscendC::DataCacheCleanAndInvalid<int64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;auto n=(__gm__ I*)counts;
  go[0]=0;for(I i=0;i<3;++i)n[i]=0;
  for(I i=0;i<capacity;++i)((__gm__ uint8_t*)valid)[i]=0;
  // Coordinates and complete region-time membership were checked by closure
  // immediately before this task. No user-supplied arbitrary ready mask here.
  if(status[0]==0) {
    // Contract-relative fallback: a state-dependent Read whose next state
    // depends on selection may expose only its first complete region frame.
    // Other regions retain their complete certified multi-time prefixes.
    auto c=(__gm__ I*)input,r=(__gm__ I*)owner,policy=(__gm__ I*)causal,t=(__gm__ I*)first;
    auto selected=(__gm__ int32_t*)ready;
    for(I i=0;i<samples*regions;++i)t[i]=-1;
    for(I i=0;i<capacity;++i)if(selected[i]) {
      I region=r[c[i*6+1]],key=c[i*6]*regions+region,time=c[i*6+2];
      if(policy[region]&&(t[key]<0||time<t[key]))t[key]=time;
    }
    for(I i=0;i<capacity;++i)if(selected[i]) {
      I region=r[c[i*6+1]],key=c[i*6]*regions+region;
      if(policy[region]&&c[i*6+2]!=t[key])selected[i]=0;
    }
    pack((__gm__ I*)input,(__gm__ int32_t*)ready,(__gm__ I*)owner,(__gm__ I*)order,
      (__gm__ I*)coords,(__gm__ uint8_t*)valid,(__gm__ I*)offsets,(__gm__ I*)fibers,
      (__gm__ I*)frame_offsets,(__gm__ I*)frame_fibers,(__gm__ I*)frames,n,capacity);
    go[0]=n[0]>0?1:0;
  }
  AscendC::DataCacheCleanAndInvalid<int64_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
