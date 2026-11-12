#pragma once
#include "weighted_allocation_math.h"
namespace tide_block_kv_sequence_device {
using namespace tide_weighted_device;
// fields: state offset, one-entry write offset, item width, ragged.
__aicore__ inline void forward(__gm__ I* fields,__gm__ float* initial,__gm__ I* initial_i,
    __gm__ float* write,__gm__ I* write_i,__gm__ I* times,
    __gm__ float* old,__gm__ I* old_i,__gm__ float* proposal,__gm__ I* proposal_i,
    __gm__ float* final,__gm__ I* final_i,__gm__ I* mapping,__gm__ int32_t* error,
    I batch,I length,I nodes,I width,I write_width,I capacity,I count,I horizon){
  if(error[0])return;const I rows=batch*length*nodes,iw=4+capacity;
  for(I b=0;b<batch;++b)for(I v=0;v<nodes;++v){
    const I owner=b*nodes+v,base=owner*iw,n=initial_i[base+2],last=initial_i[base];
    if(last< -1||initial_i[base+1]<0||n<0||n>capacity||n>initial_i[base+1]||
      initial_i[base+3]!=0||(horizon&&n>horizon)){error[0]=75;return;}
    for(I j=0;j<n;++j)if(initial_i[base+4+j]<0||initial_i[base+4+j]>last||
      (j&&initial_i[base+4+j]<=initial_i[base+3+j])){error[0]=75;return;}
    for(I f=0;f<count;++f){const I offset=fields[4*f],size=fields[4*f+2],ragged=fields[4*f+3];
      for(I j=0;j<size*(ragged?n:1);++j){const float x=initial[owner*width+offset+j];
        if(!finite(x)||(!ragged&&x!=0)||(f==count-1&&(x<0||x>1))){error[0]=75;return;}}
    }
    for(I l=0;l<length;++l){const I row=(b*length+l)*nodes+v,m=row*iw,fw=row*width;
      const I prior=row-nodes;
      for(I j=0;j<width;++j)old[fw+j]=l?proposal[prior*width+j]:initial[owner*width+j];
      for(I j=0;j<iw;++j)old_i[m+j]=l?proposal_i[prior*iw+j]:initial_i[base+j];
      for(I j=0;j<capacity;++j)mapping[row*capacity+j]=l?mapping[(rows+prior)*capacity+j]:(j<n?j:-1);
      for(I j=0;j<width;++j)proposal[fw+j]=old[fw+j];
      for(I j=0;j<iw;++j)proposal_i[m+j]=old_i[m+j];
      for(I j=0;j<capacity;++j)mapping[(rows+row)*capacity+j]=mapping[row*capacity+j];
      if(write_i[row*5+1]==0)continue;
      const I time=times[row],previous=old_i[m],observed=old_i[m+1],previous_n=old_i[m+2];
      if(write_i[row*5+1]!=1||write_i[row*5+2]!=1||write_i[row*5]!=time||
        write_i[row*5+4]!=time||time<0||time<=previous||observed==I(0x7fffffffffffffffLL)){
        error[0]=75;return;}
      I used=0;
      for(I j=0;j<previous_n;++j){const I stamp=old_i[m+4+j];
        if(!horizon||time-stamp<horizon){
          if(used==capacity){error[0]=76;return;}
          mapping[(rows+row)*capacity+used]=mapping[row*capacity+j];proposal_i[m+4+used++]=stamp;
        }
      }
      if(used==capacity){error[0]=76;return;}
      mapping[(rows+row)*capacity+used]=capacity+row;proposal_i[m+4+used++]=time;
      for(I j=used;j<capacity;++j){mapping[(rows+row)*capacity+j]=-1;proposal_i[m+4+j]=0;}
      proposal_i[m]=time;proposal_i[m+1]=observed+1;proposal_i[m+2]=used;proposal_i[m+3]=0;
      for(I f=0;f<count;++f){const I off=fields[4*f],wo=fields[4*f+1],size=fields[4*f+2];
        if(!fields[4*f+3])continue;
        for(I j=0;j<capacity;++j){const I source=mapping[(rows+row)*capacity+j];
          for(I k=0;k<size;++k){float x=0;
            if(source>=capacity)x=write[(source-capacity)*write_width+wo+k];
            else if(source>=0)x=initial[owner*width+off+source*size+k];
            if(!finite(x)||(f==count-1&&(x<0||x>1))){error[0]=75;return;}
            proposal[fw+off+j*size+k]=x;
          }
        }
      }
    }
    const I row=(b*length+length-1)*nodes+v;
    for(I j=0;j<width;++j)final[owner*width+j]=proposal[row*width+j];
    for(I j=0;j<iw;++j)final_i[base+j]=proposal_i[row*iw+j];
  }
}
// One output field at one snapshot has all retained numerical contributions;
// structural old/write roots survive slices that remove every old element.
__aicore__ inline void root(__gm__ I* fields,__gm__ I* write_i,__gm__ I* meta,__gm__ I* mapping,
    __gm__ float* bar,__gm__ uint8_t* roots,__gm__ float* initial_bar,__gm__ uint8_t* initial_roots,
    __gm__ float* write_bar,__gm__ uint8_t* write_roots,__gm__ int32_t* error,
    I owner,I row,I limit,I width,I write_width,I capacity,I count,I nodes,I start){
  const I n=meta[2];if(n<0||n>capacity){error[0]=77;return;}
  for(I f=0;f<count;++f)if(roots[f]){const I off=fields[4*f],wo=fields[4*f+1],size=fields[4*f+2],ragged=fields[4*f+3];
    initial_roots[owner*count+f]=1;
    if(ragged)for(I t=0;t<limit;++t){const I prior=start+t*nodes;
      if(write_i[prior*5+1])write_roots[prior*count+f]=1;}
    for(I j=0;j<(ragged?n:1);++j){const I source=ragged?mapping[j]:0;
      if(source<0){error[0]=77;return;}
      for(I k=0;k<size;++k){const float g=bar[off+j*size+k];if(!finite(g)){error[0]=77;return;}
        if(ragged&&source>=capacity)write_bar[(source-capacity)*write_width+wo+k]+=g;
        else initial_bar[owner*width+off+(ragged?source*size:0)+k]+=g;
      }
    }
  }
}
__aicore__ inline void reverse(__gm__ I* fields,__gm__ I* write_i,__gm__ I* old_i,__gm__ I* proposal_i,
    __gm__ I* mapping,__gm__ float* old_bar,__gm__ uint8_t* old_roots,
    __gm__ float* proposal_bar,__gm__ uint8_t* proposal_roots,
    __gm__ float* final_bar,__gm__ uint8_t* final_roots,
    __gm__ float* initial_bar,__gm__ uint8_t* initial_roots,
    __gm__ float* write_bar,__gm__ uint8_t* write_roots,__gm__ int32_t* error,
    I batch,I length,I nodes,I width,I write_width,I capacity,I count){
  if(error[0])return;const I owners=batch*nodes,rows=owners*length,iw=4+capacity;
  for(I j=0;j<owners*width;++j)initial_bar[j]=0;
  for(I j=0;j<owners*count;++j)initial_roots[j]=0;
  for(I j=0;j<rows*write_width;++j)write_bar[j]=0;
  for(I j=0;j<rows*count;++j)write_roots[j]=0;
  for(I b=0;b<batch;++b)for(I v=0;v<nodes;++v){const I owner=b*nodes+v,start=b*length*nodes+v;
    for(I l=0;l<length;++l){const I row=start+l*nodes;
      root(fields,write_i,old_i+row*iw,mapping+row*capacity,old_bar+row*width,old_roots+row*count,
        initial_bar,initial_roots,write_bar,write_roots,error,owner,row,l,width,write_width,capacity,count,nodes,start);
      if(error[0])return;
      root(fields,write_i,proposal_i+row*iw,mapping+(rows+row)*capacity,proposal_bar+row*width,proposal_roots+row*count,
        initial_bar,initial_roots,write_bar,write_roots,error,owner,row,l+1,width,write_width,capacity,count,nodes,start);
      if(error[0])return;
    }
    const I row=start+(length-1)*nodes;
    root(fields,write_i,proposal_i+row*iw,mapping+(rows+row)*capacity,final_bar+owner*width,final_roots+owner*count,
      initial_bar,initial_roots,write_bar,write_roots,error,owner,row,length,width,write_width,capacity,count,nodes,start);
    if(error[0])return;
  }
  for(I j=0;j<owners*width;++j)if(!finite(initial_bar[j])){error[0]=77;return;}
  for(I j=0;j<rows*write_width;++j)if(!finite(write_bar[j])){error[0]=77;return;}
}
}
