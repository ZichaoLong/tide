#pragma once
// 0770593 allocation §2–4/S.1: complete SelStep composition, never a scheduler.
namespace tide::device_online::weighted_region {
enum Mode {Total=0,ZeroTotal=1,Threshold=2,HardThreshold=3,Constrained=4,
           Hard=5,Direct=6,Unit=7,Pooled=8,Proportional=9};
enum Statistic {ReceivedMass=0,CapMass=1,QueryMass=2,Statistics=3};
enum Field {Received=0,Caps=1,Score=2,Evidence=3,Current=4,Key=5,Fields=6};
}
