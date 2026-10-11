# Minimum reporting checklist for lip sync in two-person video

From the paper *Blind Spots in Lip-Sync Evaluation of Two-Person Video and the Speaker–Listener–Timing (SLT) Scheme* (Figure 3).

- [ ] State how tracks are paired with people; report per-person Sync-C* (and the per-person distance D for generated video).
- [ ] Report the estimated audio–video offset and the share of clips with a usable sync signal; with other footage of the same person, separate an existing offset from an added one.
- [ ] Report a listener score during the other person's speech (e.g. Bind), together with the speaker's own mouth motion (Spk).
- [ ] Add a listener-liveliness or two-sided check, so that a frozen picture is not rewarded.
- [ ] Report face-detection coverage and effective sample size; compare systems on common inputs.
- [ ] Give a per-fault detection matrix; keep fixed-threshold detection, threshold-free AUC and paired worse-share apart.
- [ ] Without per-clip labels: render unperturbed and controlled-perturbation versions of each input and report AUC plus a manipulation check; fit fixed thresholds on same-source data and verify them on held-out data. This is not a substitute for validation on natural faults.
