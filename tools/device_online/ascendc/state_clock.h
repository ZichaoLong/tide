#pragma once
// Policies are immutable int64 (period, first phase, phase count) node rows.
// The conversion never grows a nonnegative time: product/sum cannot overflow.
namespace tide_device {
__aicore__ inline int64_t local_time_unchecked(int64_t time,__gm__ int64_t* policy,int64_t node) {
  if(time==-1)return -1;
  const auto period=policy[node*3],first=policy[node*3+1],count=policy[node*3+2];
  return (time/period)*count+(time%period-first);
}
__aicore__ inline bool local_time(int64_t time,__gm__ int64_t* policy,int64_t node,int64_t& local) {
  const auto period=policy[node*3],first=policy[node*3+1],count=policy[node*3+2];
  if(time< -1||period<=0||first<0||first>=period||count<=0||count>period-first)return false;
  if(time!=-1&&(time%period<first||time%period-first>=count))return false;
  local=local_time_unchecked(time,policy,node);return true;
}
}
