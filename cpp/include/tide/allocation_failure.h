#pragma once
// Failure diagnostics for allocation §6/S.2; never a partial Allocation result.
#include "tide/types.h"
#include <map>
#include <optional>
#include <stdexcept>

namespace tide::weighted {
class AllocationFailure : public std::runtime_error {
 public:
  Index row;
  std::map<std::string,Tensor> tensors;
  std::map<std::string,std::string> details;
  std::optional<Index> time;
  std::vector<Index> nodes,members;
  AllocationFailure(const std::string& message,Index index,
      const std::map<std::string,Tensor>& values,std::map<std::string,std::string> metadata)
      :std::runtime_error(message),row(index),details(std::move(metadata)) {
    // Snapshot only the failing row. No file I/O, CPU transfer or autograd graph.
    for(const auto& [name,value]:values)if(value.defined())tensors[name]=value.detach().clone();
  }
};
}
