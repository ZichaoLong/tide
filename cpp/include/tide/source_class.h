#pragma once
#include "tide/types.h"
#include <stdexcept>

namespace tide {
// Weighted-message architecture §2.1. Equal classes share parameters, not atoms.
// SourceDomain's exclusive-source rule is a different, unchanged contract.
struct SourceClasses {
  std::vector<Index> edges, inputs;
  void validate(const Graph& graph) const {
    if (edges.size() != graph.edges.size() || inputs.size() != graph.inputs.size())
      throw std::invalid_argument("source class physical domain mismatch");
    for (const auto& values : {edges, inputs}) for (auto label : values)
      if (label < 0) throw std::invalid_argument("source classes must be nonnegative");
  }
  Index classify(const Atom& atom) const {
    if (atom.kind != 0 && atom.kind != 1) throw std::invalid_argument("invalid source atom kind");
    const auto& values = atom.kind == 0 ? inputs : edges;
    if (atom.source < 0 || atom.source >= static_cast<Index>(values.size()))
      throw std::invalid_argument("source atom outside class domain");
    return values[atom.source];
  }
};
}  // namespace tide
