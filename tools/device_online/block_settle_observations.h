#pragma once
#include "block_settle_session.h"
#include "tide/resident_blocks.h"
namespace tide::device_online {
BlockWindow block_settle_observations(const BlockSettleWindow&,ResidentToken);
BlockSettleFlowRoots block_settle_roots(const BlockWindow&,const BlockRoots&,const BlockSettleFlowTape&);
BlockGradients block_settle_gradients(const BlockSettleAdjoints&,const BlockSettleFlowTape&);
}
