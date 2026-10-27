#pragma once
// 0770593 allocation §4.2–4.4, §6.3/S.3; docs/resident-weighted-sending.md.
namespace tide::device_online::weighted_sending {
enum Kind {Direct=0,Unit=1,Pooled=2,Proportional=3,SortedFill=4,Hard=5};
enum Base {CapBase=0,ReceivedBase=1,UniformBase=2};
enum Parameter {Total=0,Temperature=1,Parameters=2};
enum Detail {Share=0,LogRatio=1,Centered=2,Cap=3,Details=4};
// Sorted maps reuse Share/LogRatio for saturated/before-partial indicators.
enum Statistic {Tau=0,Mass=1,Scale=2,Statistics=3};
enum Connection {Scores=0,Caps=1,Received=2,Bound=3,Temp=4,Connections=5};
}
