#pragma once
#include "tide/kernel.h"

namespace tide {
// Bind independently computed packed values to batched step graphs, preserving
// structural disconnection for each row and each state slot.
std::vector<State> state_batch_vjp(const NodeWeights&, const std::vector<State>& old,
    const ContentViews&, const std::vector<Index>& times, const std::vector<State>& numeric);
void state_sequence_vjp(const NodeWeights&, const std::vector<State>& old,
    const PackedSequence&, std::vector<State>& numeric, std::vector<State>& previous);
}
