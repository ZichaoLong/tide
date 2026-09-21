#pragma once
#include "tide/types.h"

namespace tide {
Tensor aggregate(const Model&, const std::vector<Atom>&);
Tensor emit(const Tensor& h, const Tensor& g, const Tensor& p,
            const std::string& mode, double zeta);
Tensor full(const NodeWeights&, const Tensor& comparison, const Tensor& h,
            const Tensor& p, const Options&, bool identity = false);
void validate_model(const Graph&, const Model&);
struct ValidatedInput { std::vector<Atom> atoms; std::map<Owner, Owner> ledger_updates; };
ValidatedInput validate_external(const Graph&, const Model&, const Continuation&,
                                 const std::vector<External>&, Index stop, Index seal);
std::vector<Atom> validate_window(const Graph&, const Model&, Continuation&,
                                 const std::vector<External>&, Index stop, Index seal);
}  // namespace tide
