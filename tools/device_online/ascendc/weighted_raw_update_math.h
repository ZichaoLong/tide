#pragma once
// KV §1.3 and fixed-time convolution §2: exact raw retention and append.
#include "weighted_allocation_math.h"
namespace tide_weighted_raw_device {
using namespace tide_weighted_device;
constexpr I max_time=I(0x7fffffffffffffffLL);
// fields: offset, width, ragged, role(value=0, ordinary=1, KV weight=2, W=3).
__aicore__ inline void plan(__gm__ I* fields,__gm__ I* old_meta,__gm__ float* old,
    __gm__ I* times,__gm__ float* mass,__gm__ uint8_t* present,__gm__ float* proposal,
    __gm__ I* meta,__gm__ I* mapping,__gm__ float* receive,__gm__ uint8_t* written,
    __gm__ int32_t* error,I rows,I width,I capacity,I count,I kv,I convolution,I horizon){
  const I iw=4+capacity;if(error[0])return;
  for(I r=0;r<rows;++r){const I m=r*iw,f=r*width,n=old_meta[m+2],last=old_meta[m],observed=old_meta[m+1];
    if(last< -1||observed<0||n<0||n>capacity||n>observed||
      (kv?(old_meta[m+3]!=0||(horizon&&n>horizon)):
        (old_meta[m+3]!=(last<0?0:last)||((observed==0)!=(last==-1))||n>(convolution?horizon+1:0)))){error[0]=24;return;}
    for(I i=0;i<n;++i)if(old_meta[m+4+i]<0||old_meta[m+4+i]>last||(i&&old_meta[m+4+i]<=old_meta[m+3+i])){error[0]=24;return;}
    for(I j=0;j<count;++j){const I off=fields[4*j],size=fields[4*j+1],ragged=fields[4*j+2],role=fields[4*j+3];
      for(I i=0;i<(ragged?n:1)*size;++i){const float x=old[f+off+i];
        if(!finite(x)||(role==0&&x!=0)||(role==2&&(x<0||x>1))||(role==3&&x<0)){error[0]=24;return;}}
    }
    for(I i=0;i<iw;++i)meta[m+i]=old_meta[m+i];
    for(I i=0;i<capacity;++i)mapping[r*capacity+i]=i<n?i:-2;
    written[r]=0;receive[2*r]=0;receive[2*r+1]=0;
    if(!present[r])continue;
    if(times[r]<0||!finite(mass[r])){error[0]=24;return;}
    if(mass[r]<=0)continue;
    if(times[r]<=last||observed==max_time){error[0]=24;return;}
    const float w=mass[r]<1?mass[r]:1;receive[2*r]=w;receive[2*r+1]=mass[r]<=1?1:0;
    for(I j=0;j<count;++j){const I off=fields[4*j],size=fields[4*j+1],role=fields[4*j+3];
      if(role==0||role==2)continue;
      for(I i=0;i<size;++i){const float x=proposal[f+off+i];
        if(!finite(x)||(role==3&&x<0)){error[0]=24;return;}}
    }
    I used=0;if(kv||convolution){
      for(I i=0;i<n;++i){const I stamp=old_meta[m+4+i],age=times[r]-stamp;
        const bool keep=stamp<=times[r]&&(kv?(!horizon||age<horizon):age<=horizon);
        if(keep){if(used>=capacity){error[0]=25;return;}mapping[r*capacity+used]=i;meta[m+4+used++]=stamp;}}
      if(used==capacity){error[0]=25;return;}mapping[r*capacity+used]=-1;meta[m+4+used++]=times[r];
      for(I i=used;i<capacity;++i){mapping[r*capacity+i]=-2;meta[m+4+i]=0;}
    }
    meta[m]=times[r];meta[m+1]=observed+1;meta[m+2]=used;meta[m+3]=kv?0:times[r];written[r]=1;
  }
}
__aicore__ inline void apply(__gm__ I* fields,__gm__ float* old,__gm__ float* proposal,
    __gm__ I* mapping,__gm__ float* receive,__gm__ uint8_t* written,__gm__ float* output,
    __gm__ int32_t* error,I rows,I width,I capacity,I count,I convolution){
  if(error[0])return;
  for(I r=0;r<rows;++r){const I base=r*width;for(I i=0;i<width;++i)output[base+i]=old[base+i];
    if(!written[r])continue;
    for(I j=0;j<count;++j){const I off=fields[4*j],size=fields[4*j+1],ragged=fields[4*j+2],role=fields[4*j+3];
      if(role==0)continue;
      for(I i=0;i<(ragged?capacity:1);++i){const I source=ragged?mapping[r*capacity+i]:-1;
        for(I d=0;d<size;++d){float value=0;
          if(source>=0)value=old[base+off+source*size+d];
          else if(source==-1)value=role==2?receive[2*r]:proposal[base+off+d]*(ragged&&convolution?receive[2*r]:1);
          output[base+off+i*size+d]=value;}
      }
    }
  }
}
__aicore__ inline void reverse(__gm__ I* fields,__gm__ I* old_meta,__gm__ I* meta,
    __gm__ float* proposal,__gm__ I* mapping,__gm__ float* receive,__gm__ uint8_t* written,
    __gm__ float* bar,__gm__ uint8_t* roots,__gm__ float* old_bar,__gm__ float* proposal_bar,
    __gm__ float* mass_bar,__gm__ uint8_t* old_roots,__gm__ uint8_t* proposal_roots,
    __gm__ uint8_t* mass_roots,__gm__ int32_t* error,I rows,I width,I capacity,I count,I convolution){
  if(error[0])return;const I iw=4+capacity;
  // No poison read from absent independent field roots or padding.
  for(I r=0;r<rows;++r)for(I j=0;j<count;++j)if(roots[r*count+j]){
    const I size=fields[4*j+1]*(fields[4*j+2]?meta[r*iw+2]:1);
    for(I d=0;d<size;++d)if(!finite(bar[r*width+fields[4*j]+d])){error[0]=26;return;}}
  for(I r=0;r<rows;++r){const I base=r*width;mass_bar[r]=0;mass_roots[r]=0;
    for(I d=0;d<width;++d){old_bar[base+d]=0;proposal_bar[base+d]=0;}
    for(I j=0;j<count;++j){const I root=r*count+j;old_roots[root]=0;proposal_roots[root]=0;
      if(!roots[root])continue;
      const I off=fields[4*j],size=fields[4*j+1],ragged=fields[4*j+2],role=fields[4*j+3];
      if(!written[r]||role==0){old_roots[root]=1;const I n=ragged?old_meta[r*iw+2]:1;
        for(I d=0;d<n*size;++d)old_bar[base+off+d]=bar[base+off+d];continue;}
      if(!ragged){proposal_roots[root]=1;for(I d=0;d<size;++d)proposal_bar[base+off+d]=bar[base+off+d];continue;}
      // Indexing an empty/fully evicted old tensor remains a connected zero.
      old_roots[root]=1;proposal_roots[root]=role!=2;
      if(role==2||convolution)mass_roots[r]=1;
      for(I i=0;i<meta[r*iw+2];++i){const I source=mapping[r*capacity+i];
        for(I d=0;d<size;++d){const float g=bar[base+off+i*size+d];
          if(source>=0)old_bar[base+off+source*size+d]=g;
          else if(source==-1){
            if(role==2)mass_bar[r]+=g*receive[2*r+1];
            else {proposal_bar[base+off+d]=g*(convolution?receive[2*r]:1);
              if(convolution)mass_bar[r]+=g*proposal[base+off+d]*receive[2*r+1];}
          }
        }
      }
    }
    if(!finite(mass_bar[r])){error[0]=26;return;}
  }
}
} // namespace tide_weighted_raw_device
