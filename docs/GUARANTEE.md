# The stall-free guarantee

## Model

A response is a stream of tokens t = 0, 1, 2, ... A consumer (a reader, or a text-to-speech engine)
needs token t at its due time

    d_t = s + S + M + t P

where s is the start, S >= 0 a start-up allowance, P the consumption period, and M the mandatory
time bound below. A **stall** is a token finished after its due time.

Each token passes through L layers. Layer l of a token has:

- **Mandatory work**: attention plus the FFN neurons resident in RAM, with worst-case time m_l. It
  reads nothing from flash, so m_l is a property of the compute path alone. Write
  M = sum_l m_l + h, where h bounds the LM head.
- **Optional work**: FFN neurons whose weights must come from flash. Reads are issued
  asynchronously. A neuron is computed (at most kappa seconds) only after its weights arrive.

Assumptions:

- **A1:** every layer's actual mandatory time is at most m_l, and the head's is at most h.
- **A2:** M <= P.
- **A3:** the runtime can stop *waiting* for a read at any time. In-flight reads may be
  non-preemptive on the device (LeanStream: "inflight reads are non-preemptive"). They may keep the
  storage busy, but they never block the compute path, since mandatory work does not read flash.

## Level 1: layers within a token

Give token t the relative deadline D_t >= M, starting at a_t. Layer l starts at s_l, with s_0 = a_t,
and gets the cutoff

    c_l = s_l + m_l + sigma_l * w_l / W_l,
    sigma_l = max(0, a_t + D_t - h - s_l - sum_{j >= l} m_j)

where w_l >= 0 are the budget-table weights and W_l = sum_{j >= l} w_j. In layer l, an arrived
neuron is computed only if it finishes by c_l, and reads not arrived by c_l are abandoned.

**Lemma 1.** Under A1 and A3, every layer ends by its cutoff, and the token ends by a_t + D_t.

*Proof.* Induct on l with the invariant s_l + sum_{j >= l} m_j <= a_t + D_t - h.

- **Base:** it holds at l = 0 because D_t >= M.
- **Waiting and compute:** mandatory work ends by s_l + m_l <= c_l, since sigma_l >= 0. Optional
  work waits only for arrivals before c_l, and computes a neuron only if that ends by c_l. So the
  layer ends at e_l <= c_l.
- **Step:** w_l / W_l <= 1, so c_l <= s_l + m_l + sigma_l = a_t + D_t - h - sum_{j > l} m_j. Hence the
  invariant holds for s_{l+1} = e_l.
- **End:** after layer L-1, the head finishes by a_t + D_t.

The proof never uses arrival times, so it holds for any flash behavior, including reads that never
arrive. □

## Level 2: tokens within a stream

Token t starts at a_t, the end of token t-1, with a_0 = s. It receives

    D_t = M + min(o_t, d_t - a_t - M)

where o_t >= 0 is any policy's desired optional time (fixed, greedy, price, unified).

**Theorem.** Under A1 to A3, no token stalls, for any policy o_t and any flash behavior.

*Proof.* We show by induction that d_t - a_t >= M, which makes D_t >= M valid for Lemma 1.

- **Base:** t = 0 gives d_0 - a_0 = S + M >= M.
- **No stall:** by Lemma 1, token t ends at e_t <= a_t + D_t <= d_t, so it does not stall.
- **Step:** then a_{t+1} = e_t <= d_t, so d_{t+1} - a_{t+1} >= d_{t+1} - d_t = P >= M by A2. □

## What the guarantee does not cover

- **Waste from non-preemptive reads.** A read in flight at a cutoff still occupies the device, and
  may delay the next layer's reads by up to (queue depth) x (longest read). This lowers quality,
  never timeliness. Read admission (issue a read only if it is likely to land before the cutoff)
  limits the waste; this is on the road map.
- **The bound m_l.** It must come from measurement on the device, such as a high percentile of
  mandatory time under interference with a safety margin, or from WCET analysis of the kernels. If
  A1 is violated by x seconds, that token is late by at most x. Later tokens are back on time as
  long as x <= P - M, because the next token shrinks its optional time to make up the difference.
- **Quality.** The theorem is about time only. Quality is what the policies (fixed, price, unified)
  and the RAM residency choice optimize, and it is measured, not proved.

Tests: `tests/test_scheduler.py` covers Lemma 1 on 5,000 random cases with heavy-tailed and lost
reads. `tests/test_stream.py` covers the theorem on 40 random streams of 200 tokens, composed with
Lemma 1.
