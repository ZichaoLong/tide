#pragma once
// Pinned 0770593 allocation §6/S.2; docs/resident-weighted-allocation.md.
// Separate identities: forward bounded solve, exact-map VJP, no HST surrogate.
namespace tide::device_online::weighted_allocation {
enum Branch {Absent=-1,Zero=0,Caps=1,Slack=2,Lower=3,Upper=4,Equality=5};
enum Rule {Quad=0,Bernoulli=1,Entropy=2};
enum Bound {Lo=0,Hi=1,Temperature=2,Shift=3,BoundFields=4};
enum Slope {Score=0,Cap=1,Tau=2,SlopeFields=3};
enum Connection {Scores=0,Capacities=1,LowerBound=2,UpperBound=3,Temp=4,ScoreShift=5,Connections=6};
// Other resident errors keep their meaning. These values are local numerics.
enum Error {Domain=20,SupportUnderflow=21,Residual=22,GradientOverflow=23};
}
