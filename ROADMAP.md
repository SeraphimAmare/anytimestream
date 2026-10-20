# Road map (October 2026 to May 2027)

The date that matters is the **RTAS 2027 Brief Presentation, due February 11, 2027**. It needs
real-model numbers by January.

| When | Step | Caveat it answers | Done when |
|---|---|---|---|
| Oct 20 | Agree on direction with Prof. Yao | | Direction approved; per-token vs per-response deadline decided |
| Oct 21 to Nov 30 | Rerun on Qwen2.5-0.5B, then 1.5B and 7B (lab GPU) | Toy model | `build_budget_table.py` and `experiments/scripts/*` run on all three; fragility pattern compared |
| Nov 1 to Dec 15 | Learned predictor and free-running generation | Idealized predictor, teacher forcing | Split still beats equal shares and the ends rule when generated tokens feed back |
| Dec 1 to Jan 10 | Jetson Orin timing and LeanStream hookup (with Renyuan) | Uncalibrated simulator | Measured m_l, flash latency and κ; scheduler cutoffs drive LeanStream's cutoff flag |
| Jan 4 to Feb 5 | Full evaluation and write-up | | Quality and miss rate against equal shares, the ends rule and TEAL's split, with paired CIs |
| Feb 11, 2027 | RTAS Brief Presentation due | | Submitted |
| Late Apr to early May | OSCAR poster at Mason (2027 date not posted) | | Poster presented |
| May 17 to 20 | CPS-IoT Week, Boulder | | Brief presented |

## Next code tasks

- [x] Reservation scheduler with tested guarantee (`anytimestream/scheduler.py`)
- [x] Budget table format and builder (`anytimestream/budget_table.py`, `scripts/build_budget_table.py`)
- [ ] Run the builder on Qwen2.5-0.5B and compare its fragility profile with the toy model
- [ ] Hook `ReservationScheduler` into `experiments/kit/anytime_sim.py` in place of its `Slack` policy
- [ ] Add per-layer mandatory bounds `m_l` measured on device (replace the scalar `m`)
- [ ] Learned predictor from the pre-attention state; free-running evaluation
- [ ] Compare against the protect-first-and-last rule at every budget and model size
