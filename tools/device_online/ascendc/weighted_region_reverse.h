#pragma once
#include "weighted_region_prepare.h"
namespace tide_weighted_region_device {
// Chain the declared sending map; never differentiate its forward iterations or
// reconnect score/threshold roots absent from a hard branch.
__aicore__ inline int reverse_finish(I n,I width,I qwidth,I mode,bool context,bool attention,
    bool query_received,I k,float fixed,float alpha_local,float alpha_query,float inverse_sqrt,
    __gm__ float* stats,__gm__ float* head,__gm__ float* query,__gm__ float* key,
    __gm__ float* gv,__gm__ float* gs,__gm__ uint8_t* connected,
    __gm__ float* gw,__gm__ float* gm,__gm__ float* gscore,__gm__ float* gkey,
    __gm__ float* gh,__gm__ float* gq,__gm__ uint8_t* fields,
    __gm__ uint8_t* head_root,__gm__ uint8_t* query_root,Math& math){
  const bool allocated=mode==Total||mode==Threshold||mode==Constrained;
  const bool total=mode==Total||mode==ZeroTotal;
  const bool score_root=connected[0],cap_root=connected[1];
  bool receive_root=!allocated&&connected[2],mu_root=cap_root;
  bool control_root=false;float gr=0,gcap=0;
  for(I j=0;j<n;++j){gm[j]=gv[width+j];gw[j]=allocated?0:gv[2*width+j];}
  if(total){
    const bool bound_root=allocated?(connected[2]||connected[3]):connected[3];
    const float db=allocated?gs[0]+gs[1]:gs[0];
    const float rho=context?math.sigmoid(head[0]):fixed;
    if(bound_root){
      gr=db*(1-rho)*(stats[ReceivedMass]<=1?1.f:0.f);
      gcap=db*rho*(stats[CapMass]<=float(k)?1.f:0.f);
      receive_root=true;mu_root=true;control_root=context;
      if(context)gh[0]=db*(minimum(float(k),stats[CapMass])-minimum(1,stats[ReceivedMass]))*rho*(1-rho);
    }
  }else if(allocated){
    if(connected[2]){
      if(mode==Threshold){gr=gs[0]*(stats[ReceivedMass]<=1?1.f:0.f);receive_root=true;}
      else {gcap=gs[0]*(stats[CapMass]<=1?1.f:0.f);mu_root=true;}
    }
    if(context&&connected[5]){control_root=true;gh[0]=gs[3];}
  }
  *head_root=control_root;*query_root=attention&&score_root;
  if(control_root){receive_root=true;mu_root=true;fields[Evidence]=1;}
  if(attention&&score_root){
    if(query_received)receive_root=true;else mu_root=true;
    fields[Current]=1;fields[Key]=1;
    for(I j=0;j<n;++j){
      gscore[j]=alpha_local*gv[j];const float coefficient=alpha_query*inverse_sqrt*gv[j];
      for(I d=0;d<qwidth;++d){gq[d]+=coefficient*key[j*qwidth+d];gkey[j*qwidth+d]=coefficient*query[d];}
    }
  }else if(score_root)for(I j=0;j<n;++j)gscore[j]=gv[j];
  fields[Received]=receive_root;fields[Caps]=mu_root;fields[Score]=score_root;
  for(I j=0;j<n;++j){gw[j]+=gr;gm[j]+=gcap;
    if(!finite(gw[j])||!finite(gm[j])||!finite(gscore[j]))return 23;
    if(attention)for(I d=0;d<qwidth;++d)if(!finite(gkey[j*qwidth+d]))return 23;}
  if(!finite(gh[0]))return 23;
  if(attention)for(I d=0;d<qwidth;++d)if(!finite(gq[d]))return 23;
  return 0;
}
__aicore__ inline int reverse_prepare(I n,I ewidth,I qwidth,bool context_root,bool query_root,
    bool query_received,__gm__ float* w,__gm__ float* mu,__gm__ float* evidence,__gm__ float* current,
    __gm__ float* stats,__gm__ float* ctx,__gm__ float* qry,__gm__ float* gc,__gm__ float* gq,
    __gm__ float* gw,__gm__ float* gm,__gm__ float* ge,__gm__ float* gx){
  const float received=stats[ReceivedMass],mass=stats[CapMass],qmass=stats[QueryMass];
  const float cden=maximum(1,received),qden=maximum(1,qmass);
  for(I j=0;j<n;++j){
    if(context_root){
      gw[j]+=gc[ewidth]/(1+received);gm[j]+=gc[ewidth+1]/(1+mass);
      for(I d=0;d<ewidth;++d){
        gw[j]+=gc[d]*(evidence[j*ewidth+d]-(received>=1?ctx[d]:0))/cden;
        ge[j*ewidth+d]=w[j]*gc[d]/cden;if(!finite(ge[j*ewidth+d]))return 23;
      }
    }
    if(query_root){float delta=0;const float weight=query_received?w[j]:mu[j];
      for(I d=0;d<qwidth;++d){
        delta+=gq[d]*(current[j*qwidth+d]-(qmass>=1?qry[d]:0))/qden;
        gx[j*qwidth+d]=weight*gq[d]/qden;if(!finite(gx[j*qwidth+d]))return 23;
      }
      if(query_received)gw[j]+=delta;else gm[j]+=delta;
    }
    if(!finite(gw[j])||!finite(gm[j]))return 23;
  }
  return 0;
}
}
