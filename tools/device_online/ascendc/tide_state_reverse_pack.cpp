#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_state_reverse_pack(GM_ADDR metadata,GM_ADDR count,GM_ADDR atoms,GM_ADDR atom_count,
    GM_ADDR links,GM_ADDR mapping,GM_ADDR local_meta,GM_ADDR local_count,GM_ADDR local_atoms,GM_ADDR local_atom_count,
    GM_ADDR event_rows,GM_ADDR atom_rows,GM_ADDR inverse,GM_ADDR local_links,GM_ADDR valid,GM_ADDR head,GM_ADDR next,GM_ADDR tail,
    GM_ADDR scale_rows,GM_ADDR error,int64_t rows,int64_t atom_capacity,int64_t events,int64_t fibers,int64_t nodes,int64_t local_nodes,int64_t scales) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)metadata);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;auto m=(__gm__ I*)metadata,a=(__gm__ I*)atoms,map=(__gm__ I*)mapping,l=(__gm__ I*)links;
  auto out=(__gm__ I*)local_meta,oa=(__gm__ I*)local_atoms,ol=(__gm__ I*)local_links,er=(__gm__ I*)event_rows,ar=(__gm__ I*)atom_rows;
  auto inv=(__gm__ I*)inverse,h=(__gm__ I*)head,ne=(__gm__ I*)next,last=(__gm__ I*)tail,sr=(__gm__ I*)scale_rows;
  auto live=(__gm__ uint8_t*)valid;auto ec=(__gm__ I*)local_count,ac=(__gm__ I*)local_atom_count;
  ec[0]=ac[0]=0;const I n=((__gm__ I*)count)[0],na=((__gm__ I*)atom_count)[0];
  for(I i=0;i<rows;++i)inv[i]=-1;
  for(I i=0;i<events;++i){er[i]=rows;h[i]=last[i]=-1;for(I j=0;j<13;++j)out[i*13+j]=0;}
  for(I i=0;i<fibers;++i){ar[i]=atom_capacity;sr[i]=scales;live[i]=0;ne[i]=-1;
    for(I j=0;j<6;++j)oa[i*6+j]=0;for(I j=0;j<4;++j)ol[i*4+j]=-1;}
  if(status[0]==0&&(n<0||n>rows||na<0||na>atom_capacity))status[0]=2;
  for(I i=0;i<n&&status[0]==0;++i) {
    const I node=m[i*13+1];if(node<0||node>=nodes){status[0]=2;break;}
    const I local=map[node];if(local==-1)continue;if(local<0||local>=local_nodes){status[0]=2;break;}
    if(ec[0]>=events){status[0]=1;break;}const I row=ec[0]++;inv[i]=row;er[row]=i;
    for(I j=0;j<13;++j)out[row*13+j]=m[i*13+j];out[row*13+1]=local;
  }
  for(I i=0;i<na&&status[0]==0;++i) {
    const I node=a[i*6+1];if(node<0||node>=nodes){status[0]=2;break;}
    const I local=map[node];if(local==-1)continue;if(local<0||local>=local_nodes){status[0]=2;break;}
    const I consumer=l[i*4],scale=l[i*4+2];
    if(consumer<0||consumer>=n||inv[consumer]<0||scale<0||scale>=scales){status[0]=2;break;}
    if(a[i*6]!=m[consumer*13]||node!=m[consumer*13+1]||a[i*6+2]!=m[consumer*13+2]){status[0]=2;break;}
    if(ac[0]>=fibers){status[0]=1;break;}const I row=ac[0]++,event=inv[consumer];ar[row]=i;sr[row]=scale;live[row]=1;
    for(I j=0;j<6;++j)oa[row*6+j]=a[i*6+j];oa[row*6+1]=local;
    ol[row*4]=event;ol[row*4+2]=row; // Local per-atom scale, not a new parameter identity.
    if(last[event]<0)h[event]=row;else ne[last[event]]=row;last[event]=row;
  }
  if(status[0]) {
    ec[0]=ac[0]=0;
    for(I i=0;i<events;++i){er[i]=rows;h[i]=-1;}
    for(I i=0;i<fibers;++i){ar[i]=atom_capacity;sr[i]=scales;live[i]=0;ne[i]=-1;}
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
