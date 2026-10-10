#include "control_flow.h"
#include <map>
#include <set>
#include <stdexcept>

namespace tide::device_online {
std::vector<ControlBlock> lower_control(const std::vector<ControlInstruction>& code,size_t labels) {
  const size_t none=std::numeric_limits<size_t>::max();
  std::vector<size_t> positions(labels,none);
  std::set<size_t> cuts{0,code.size()};
  for (size_t i=0;i<code.size();++i) {
    const auto& op=code[i];
    if (op.kind==ControlInstruction::Mark) {
      if (op.label>=labels || positions[op.label]!=none) throw std::invalid_argument("invalid or repeated control label");
      positions[op.label]=i;cuts.insert(i);
    } else if (op.kind==ControlInstruction::Branch) {
      if (op.targets.empty()) throw std::invalid_argument("empty control branch");
      for (auto t:op.targets) if (t>=labels) throw std::invalid_argument("unknown control target");
      cuts.insert(i+1);
    }
  }
  for (auto p:positions) if (p==none) throw std::invalid_argument("control label has no target position");
  std::vector<size_t> starts(cuts.begin(),cuts.end());
  if (starts.size()>size_t(std::numeric_limits<int32_t>::max())) throw std::overflow_error("too many control blocks");
  std::map<size_t,int32_t> index;
  for (size_t i=0;i<starts.size();++i) index[starts[i]]=i;
  std::vector<ControlBlock> blocks(starts.size()-1);
  for (size_t b=0;b<blocks.size();++b) {
    auto& out=blocks[b];
    for (size_t i=starts[b];i<starts[b+1];++i) {
      if (code[i].kind==ControlInstruction::Operation) out.operations.push_back(i);
      else if (code[i].kind==ControlInstruction::Branch) {
        out.branch=i;
        for (auto t:code[i].targets) out.targets.push_back(index.at(positions[t]));
      }
    }
    if (out.branch==none) out.targets.push_back(b+1);
  }
  return blocks;
}
}
