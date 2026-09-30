# Stable projected source order — 2026-09-30

Clean `832a881` fixes native Aggregate InputOrigin ties by retaining physical
source order, matching the independent Python reference. Projection changes the
Aggregate source view; it does not merge logical slots or replace physical edge
identity. The [manifest](native-origin-order-20260930.json) fingerprints the clean
builds and existing raw records.

A 32-edge regression with equal projected keys fails against the old `0e66d89`
CPU library (exit 2, `equal projected keys must retain physical source order`).
The preserved `origins-oldcore-regression01` wrapper passed by observing that
expected failure; the old library did not pass the regression.

Corrected clean CPU and standalone NPU builds passed. CPU verification comprises
8 CTests, the Aggregate component in FP32 and FP64, and 362 focused Python/native
source-origin, source-domain, port-layout and Settle tests. The standalone NPU
FP32 Aggregate component also passed on the recorded CANN 9.0.0 / Torch 2.10.0
stack. This is targeted regression evidence, not a new complete CPU regression,
throughput comparison, device-resident scheduling or training qualification.
