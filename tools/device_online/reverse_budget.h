#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
// Construction-time physical maxima, not observed active row counts. Every
// complete logical fiber and cache remains intact when these maxima shrink.
struct ReverseBatchPlan {
  int64_t owner_rows,query_rows,key_rows,requested_rows,tensor_bytes,budget;
};
namespace reverse_budget {
template<class Fits> int64_t largest(int64_t maximum,Fits fits) {
  int64_t lo=0,hi=std::max<int64_t>(0,maximum);
  while(lo<hi) {
    const auto mid=lo+(hi-lo)/2+1;
    if(fits(mid))lo=mid;else hi=mid-1;
  }
  return lo;
}
inline int64_t bytes(long double value) {
  if(!std::isfinite(value)||value<0||value>std::numeric_limits<int64_t>::max())
    throw std::invalid_argument("reverse reservation extent overflow");
  return static_cast<int64_t>(std::ceil(value));
}
inline long double attention(int64_t q,int64_t h,int64_t d,int64_t kh,int64_t k,int64_t tile,bool half) {
  return 4.L*(2.L*q*kh*k*d+1.L*q*k+12.L*q*h*d+8.L*q*h*tile*d+10.L*q*h*tile
    +1.L*q*tile+8.L*q*h)+32.L*q+1024+2.L*half*(1.L*q*h*d+1.L*q*h*tile*(d+1.L));
}
inline int64_t key_rows(int64_t q,int64_t h,int64_t d,int64_t kh,int64_t k,int64_t tile,bool half,int64_t budget) {
  return largest(std::min(k,tile),[&](int64_t t){return attention(q,h,d,kh,k,t,half)<=budget;});
}
inline long double fiber(int64_t b,int64_t s,int64_t w,int64_t k,int64_t domain,int64_t c,bool half) {
  return 4.L*(4.L*b*s*w+8.L*b*w*w+16.L*b*w+4.L*b*k*w+4.L*b*k+8.L*b*domain+4.L*b*s
    +12.L*c*w*w+32.L*c*w+4.L*c*k*w+4.L*c*k)+64.L*b+64.L*c+4096
    +2.L*half*(c*(3.L*w+1.L*w*w+2.L*k*w)+3.L*b*w)+4.L*b;
}
inline long double fiber_pack(int64_t b,int64_t s,int64_t w,int64_t k,int64_t domain,int64_t nodes) {
  return 4.L*b*(1.L*s*w+4.L*k*w+4.L*k+4.L*w*w+4.L*w+domain)+512.L*b+8.L*b*s+8.L*nodes+4096;
}
inline long double event_pack(int64_t c,int64_t w,int64_t kh,int64_t d,int64_t k,bool half) {
  const long double cols=w+2.L*kh*d;
  return 4.L*c*(4.L*w*cols+4.L*w*w+8.L*w+4.L*kh*k*d)+128.L*c
    +2.L*half*c*(2.L*w+w*cols+cols+2.L*kh*k*d);
}
inline void geometry(int64_t owners,int64_t w,int64_t h,int64_t k,int64_t maximum,int64_t budget) {
  if(owners<1||w<1||h<1||w%h||k<1||maximum<1||budget<1)
    throw std::invalid_argument("invalid reverse batch reservation geometry");
}
} // namespace reverse_budget

inline ReverseBatchPlan plan_fiber_vjp(int64_t b,int64_t s,int64_t w,int64_t h,int64_t k,
    int64_t domain,int64_t maximum,int64_t tile,bool half,int64_t budget) {
  using namespace reverse_budget;
  geometry(b,w,h,k,maximum,budget);
  if(s<1||s>k||domain<1||tile<1||tile>256)throw std::invalid_argument("invalid fiber reverse reservation geometry");
  const auto cap=static_cast<int64_t>(std::min<long double>(maximum,1.L*b*s));
  const auto c=largest(cap,[&](int64_t q){return fiber(b,s,w,k,domain,q,half)<=budget/2
    &&attention(q,h,w/h,h,k,1,half)<=budget/2;});
  if(!c)throw std::invalid_argument("one complete fiber adjoint row exceeds tensor budget");
  const auto t=key_rows(c,h,w/h,h,k,tile,half,budget/2);
  return {b,c,t,maximum,bytes(fiber(b,s,w,k,domain,c,half)+attention(c,h,w/h,h,k,t,half)),budget};
}
inline ReverseBatchPlan plan_fiber_reverse(int64_t owners,int64_t s,int64_t w,int64_t h,int64_t k,
    int64_t domain,int64_t nodes,int64_t maximum,bool half,int64_t budget) {
  using namespace reverse_budget;
  geometry(owners,w,h,k,maximum,budget);
  if(s<1||s>k||domain<1||nodes<1)throw std::invalid_argument("invalid fiber reverse reservation geometry");
  // Preserve the existing disjoint reservations: half for packed inputs, half
  // for the local VJP (itself split between projection and tiled attention).
  const auto b=largest(std::min(owners,maximum),[&](int64_t c){return
    fiber_pack(c,s,w,k,domain,nodes)<=budget/2 && fiber(c,s,w,k,domain,1,half)<=budget/2/2
    &&attention(1,h,w/h,h,k,1,half)<=budget/2/2;});
  if(!b)throw std::invalid_argument("one complete fiber reverse owner exceeds tensor budget");
  auto out=plan_fiber_vjp(b,s,w,h,k,domain,maximum,std::min<int64_t>(64,k),half,budget/2);
  out.tensor_bytes=bytes(fiber_pack(b,s,w,k,domain,nodes)+out.tensor_bytes);out.budget=budget;return out;
}
inline ReverseBatchPlan plan_event_reverse(int64_t owners,int64_t w,int64_t h,int64_t kh,int64_t k,
    int64_t maximum,bool half,int64_t budget) {
  using namespace reverse_budget;
  geometry(owners,w,h,k,maximum,budget);
  if(kh<1||h%kh)throw std::invalid_argument("invalid event reverse head geometry");
  const auto c=largest(std::min(owners,maximum),[&](int64_t q){return event_pack(q,w,kh,w/h,k,half)<=budget/2
    &&attention(q,h,w/h,kh,k,1,half)<=budget/2;});
  if(!c)throw std::invalid_argument("one complete event reverse owner exceeds tensor budget");
  const auto t=key_rows(c,h,w/h,kh,k,std::min<int64_t>(64,k),half,budget/2);
  return {c,c,t,maximum,bytes(event_pack(c,w,kh,w/h,k,half)+attention(c,h,w/h,kh,k,t,half)),budget};
}
} // namespace tide::device_online
