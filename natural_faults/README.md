# Natural faults: blind-rater labels and per-clip scores (paper Sec. 5.4)

Faults that generators make on their own, with no injected error, judged by blind raters.

| Generator | Clips | Modes compared | Raters |
|---|---|---|---|
| `adapter_batch1`: our two-person audio adapter (a separate lip-sync generation submission; only evaluated here) | 39 inputs × 2 modes = 78 | `single` (one track, no person labels) vs. `duo` (two tracks with explicit assignment), seed 1234 | R-N1 (a project member), R-N2, R-N3 |
| `adapter_batch2`: the same adapter on 39 further inputs (none shared with the other batches); pre-registered | 39 inputs × 2 modes = 78 | as batch 1 | R-A1, R-A2, R-A3 |
| `ltx2`: LTX-2.3 (takes one mixed track; who speaks is set by the prompt) | 39 inputs × 2 modes = 78 | `unnamed` (prompt does not name the speaker) vs. `named`, seed 42 | R-M1, R-M2, R-M3 |

In every clip only one person should speak; `speaker_side` says which. Raters saw the clip with the speaker's own voice,
did not know the mode, did not see other raters' answers or any metric score, and were not authors of the paper.
Rater identities are not released (codes only). For the LTX-2 batch and adapter batch 2, the rating protocol, analysis rules and predictions were committed
before any rater answered.

## Files

- `labels_and_scores.csv`: one row per clip.
  - `rater1..3`: answers of the raters listed in `rater_codes`. 1 = correct; 2 = the listener also moves the mouth; 3 = the speaker does not move the mouth; 23 = wrong person speaks; 4 = mouth on the wrong face / extra face; 5 = cannot see.
  - Scores (larger = more anomalous): `S` (negated per-person Sync-C*), `L_Bind`, `T_offset` (|offset|), `per_person_D` (per-person SyncNet minimum distance, mean of both people), `mixed_negC` (negated mixed-track SyncNet confidence).
  - `SLT_gated`: SLT's pre-registered primary continuous reading, i.e. the largest part percentile among the paper's untouched real clips, with the offset part dropped when SyncNet confidence is below the real-footage floor (on these generated clips it always is).
  - `SLT_ungated`: the same without the gate (pre-registered secondary reading).
- `natural_faults_auc.py`: recomputes the majority labels, Fleiss κ and every AUC from the CSV. It needs only numpy; edit the block at the top and run it. Point estimates match the paper. Bootstrap intervals can differ from the paper's in the second decimal, because the resampling order differs.

No video, audio or model weights are included.
