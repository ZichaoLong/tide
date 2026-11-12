#pragma once
#include "weighted_allocation_math.h"
namespace tide_block_kv_bank_device {
using namespace tide_weighted_device;
// Numerical entries are owner-major. Snapshots remain [B,L,N], with exact
// physical source ids; nonpositive writes occupy no raw sequence membership.
__aicore__ inline void plan(__gm__ I* fields,__gm__ float* initial,__gm__ I* initial_i,
    __gm__ float* write,__gm__ I* write_i,__gm__ I* query_times,
    __gm__ float* keys,__gm__ float* values,__gm__ float* summaries,__gm__ float* weights,
    __gm__ I* stamps,__gm__ uint8_t* valid,__gm__ uint8_t* written,
    __gm__ I* old_map,__gm__ I* proposal_map,__gm__ I* old_i,__gm__ I* proposal_i,
    __gm__ int32_t* error,I batch,I length,I nodes,I capacity,I d,I ds,I width,I write_width,I horizon,I bound){
  if(error[0])return;const I entries=capacity+(bound?0:length),iw=capacity+4;
  for(I r=0;r<batch*length*nodes;++r)written[r]=0;
  for(I b=0;b<batch;++b)for(I v=0;v<nodes;++v){
    const I owner=b*nodes+v,base=owner*iw,n=initial_i[base+2],last=initial_i[base];
    if(last< -1||initial_i[base+1]<0||n<0||n>capacity||n>initial_i[base+1]||initial_i[base+3]!=0||(horizon&&n>horizon)){
      error[0]=78;return;}
    for(I j=0;j<d+1;++j)if(initial[owner*width+j]!=0){error[0]=78;return;}
    for(I e=0;e<entries;++e){const I index=owner*entries+e;
      const I step=e-capacity,row=(b*length+step)*nodes+v;
      const bool use=e<capacity?e<n:write_i[row*5+1]!=0;
      valid[index]=use;stamps[index]=0;
      if(e>=capacity){written[row]=use;
        if(use&&(write_i[row*5+1]!=1||write_i[row*5+2]!=1||write_i[row*5+3]!=0||
            write_i[row*5]!=query_times[row]||write_i[row*5+4]!=query_times[row])){error[0]=78;return;}}
      if(use){const I stamp=e<capacity?initial_i[base+4+e]:query_times[row];
        if(stamp<0||(e<capacity&&(stamp>last||(e&&stamp<=initial_i[base+3+e])))){error[0]=78;return;}
        stamps[index]=stamp;}
      for(I f=1;f<5;++f){const I off=fields[4*f],wo=fields[4*f+1],size=fields[4*f+2];
        auto target=f==1?keys:f==2?values:f==3?summaries:weights;
        for(I k=0;k<size;++k){const float x=!use?0:e<capacity?initial[owner*width+off+e*size+k]:write[row*write_width+wo+k];
          if(!finite(x)||(f==4&&(x<0||x>1))){error[0]=78;return;}target[index*size+k]=x;}
      }
    }
    for(I l=0;l<length;++l){const I row=(b*length+l)*nodes+v,m=row*iw,prior=row-nodes;
      for(I j=0;j<iw;++j)old_i[m+j]=l?proposal_i[prior*iw+j]:initial_i[base+j];
      for(I j=0;j<capacity;++j)old_map[row*capacity+j]=l?proposal_map[prior*capacity+j]:(j<n?owner*entries+j:-1);
      for(I j=0;j<iw;++j)proposal_i[m+j]=old_i[m+j];
      for(I j=0;j<capacity;++j)proposal_map[row*capacity+j]=old_map[row*capacity+j];
      if(!written[row])continue;
      const I time=query_times[row],previous=old_i[m],observed=old_i[m+1],previous_n=old_i[m+2];
      if(time<0||time<=previous||observed==I(0x7fffffffffffffffLL)){error[0]=78;return;}
      I used=0;
      for(I j=0;j<previous_n;++j){const I stamp=old_i[m+4+j];
        if(!horizon||time-stamp<horizon){if(used==capacity){error[0]=79;return;}
          proposal_map[row*capacity+used]=old_map[row*capacity+j];proposal_i[m+4+used++]=stamp;}}
      if(used==capacity){error[0]=79;return;}
      proposal_map[row*capacity+used]=owner*entries+capacity+l;proposal_i[m+4+used++]=time;
      for(I j=used;j<capacity;++j){proposal_map[row*capacity+j]=-1;proposal_i[m+4+j]=0;}
      proposal_i[m]=time;proposal_i[m+1]=observed+1;proposal_i[m+2]=used;proposal_i[m+3]=0;
    }
  }
}
// Decode actual banks only. Before the first positive write all opaque padding
// belongs to the imported raw state and is preserved, including unused NaNs.
__aicore__ inline void decode(__gm__ I* fields,__gm__ float* initial,__gm__ I* initial_i,
    __gm__ float* keys,__gm__ float* values,__gm__ float* summaries,__gm__ float* weights,
    __gm__ I* mapping,__gm__ I* metadata,__gm__ float* raw,__gm__ int32_t* error,
    I batch,I length,I nodes,I capacity,I entries,I width){
  if(error[0])return;const I iw=capacity+4;
  for(I b=0;b<batch;++b)for(I l=0;l<length;++l)for(I v=0;v<nodes;++v){
    const I owner=b*nodes+v,row=(b*length+l)*nodes+v,n=metadata[row*iw+2];
    if(n<0||n>capacity){error[0]=80;return;}
    const bool unchanged=metadata[row*iw+1]==initial_i[owner*iw+1];
    for(I j=0;j<width;++j)raw[row*width+j]=unchanged?initial[owner*width+j]:0;
    for(I j=0;j<fields[2];++j)raw[row*width+j]=initial[owner*width+j];
    if(unchanged)continue;
    for(I i=0;i<n;++i){const I entry=mapping[row*capacity+i];
      if(entry<owner*entries||entry>=(owner+1)*entries){error[0]=80;return;}
      for(I f=1;f<5;++f){const I offset=fields[4*f],size=fields[4*f+2];auto source=f==1?keys:f==2?values:f==3?summaries:weights;
        for(I k=0;k<size;++k)raw[row*width+offset+i*size+k]=source[entry*size+k];}
    }
  }
}
}
