#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace tide::device_online {
// Static lowering only: values and branch indexes are never supplied here.
struct ControlInstruction {
  enum Kind { Operation, Mark, Branch } kind=Operation;
  size_t label=0;
  std::vector<size_t> targets;
};
struct ControlBlock {
  std::vector<size_t> operations;
  size_t branch=std::numeric_limits<size_t>::max();
  std::vector<int32_t> targets; // Includes implicit fallthrough/exit.
};
std::vector<ControlBlock> lower_control(const std::vector<ControlInstruction>&,size_t labels);
}
