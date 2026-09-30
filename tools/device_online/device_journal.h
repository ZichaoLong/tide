#pragma once
#include "cann_program.h"

namespace tide::device_online {
// Optional bounded debug record. Its capacity is per call, separate from the
// reclaimable event queue. Preflight all logs before committing any live owner.
struct JournalProposal {at::Tensor meta,values,count;};
class DeviceJournal {
 public:
  DeviceJournal(int64_t capacity,int64_t metadata_columns,int64_t width,at::Device);
  JournalProposal propose(CannProgram&,const at::Tensor& meta,const at::Tensor& values,
                          const at::Tensor& count,const at::Tensor& error) const;
  void commit(CannProgram&,const JournalProposal&,const at::Tensor& error) const;
  at::Tensor meta,values,count;
};
} // namespace tide::device_online
