# Road map (October 2026 to May 2027)

Targets: the **RTAS 2027 Brief Presentation (due February 11, 2027)**, and an **RTCSA 2027** full paper
(deadline not posted; past years point to about March). The RTAS full-paper deadline, November 5, 2026,
is too soon.

| When | Step | Caveat it answers |
|---|---|---|
| Oct 21 to Nov 8 | Real-model results: Qwen2.5-0.5B here, then 1.5B and 7B on a lab GPU | Toy model |
| Nov 1 to Nov 22 | Write up the two-level guarantee (proof, assumptions, non-preemptive reads) | Analysis |
| Nov 16 to Jan 10 | Token scheduler inside LeanStream on the Jetson Orin, with Renyuan | Simulated flash timing |
| Dec 1 to Jan 17 | Price rule driven by LeanStream's predictor; free-running generation | Idealized predictor, teacher forcing |
| Jan 11 to Feb 5 | Evaluation at reading and speech rates, with real I/O interference (camera, app updates) | Workload realism |
| Feb 11, 2027 | RTAS brief presentation due | |
| about March 2027 | RTCSA full paper (estimate) | |
| May 17 to 20 | CPS-IoT Week, Boulder | |

## Next code tasks

- [x] Layer-level reservation scheduler with tested guarantee
- [x] Token-level reservation against consumption deadlines, with tests
- [x] Stall-free policies with quality measured on the model (fixed, greedy, steady, price, unified)
- [x] RAM residency by frequency, damage and hybrid
- [ ] Qwen2.5-0.5B: law, residency, stall-free (running)
- [ ] Unified price from predicted (not true) neuron values
- [ ] Free-running generation under the token scheduler
- [ ] Flash traces measured on a phone or Jetson instead of the two-state model
- [ ] Bound the waste from non-preemptive in-flight reads; add read admission control
