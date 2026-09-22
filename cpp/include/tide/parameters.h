#pragma once

#include "tide/types.h"
#include <map>
#include <string>
#include <vector>

namespace tide {

// One physical Parameter object can be exposed under several names.  The
// TensorImpl identity is deliberately used here: two distinct tensors that
// happen to overlap storage are not declared shared parameters.
struct ParameterOwner {
  std::string canonical;
  std::vector<std::string> aliases;
  Tensor value;
};

class ParameterRegistry {
 public:
  ParameterRegistry() = default;

  // Registering a name twice is always an error, including when the Tensor is
  // the same alias.  Use a distinct name for every public path.
  void add(const std::string& name, const Tensor& value);

  // Add all trainable tensors exposed by the native Model.  `prefix` is useful
  // when one shared registry owns parameters from multiple graphs.  With the
  // default filter, non-gradient identity-node buffers are omitted.
  void add_model(const Model& model, const std::string& prefix = "",
                bool trainable_only = true);

  bool contains(const std::string& name) const;
  Tensor value(const std::string& name) const;
  std::string canonical_name(const std::string& name) const;
  std::vector<std::string> names() const;
  std::vector<ParameterOwner> owners() const;
  std::vector<std::vector<std::string>> alias_partitions() const;

 private:
  std::map<std::string, Tensor> named_;
};

}  // namespace tide
