#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""《Blind Spots in Lip-Sync Evaluation of Two-Person Video and the Speaker–Listener–Timing (SLT) Scheme》公开代码合成一个文件
（github.com/LuckisonDu/two-person-lipsync-slt 的 wvaq/ 12 个模块；除 config 的 4 行路径改成读本文件开头的变量外，原文一字未改），
外加独立的「check」模式：别人给一段双人视频 + 声音，就能得到 SLT 结果；不用设环境变量，也不用我们算好的中间结果。

用法（不带参数运行会打印完整用法）：
    python 口型论文复现.py check --video X.mp4 --audio-left L.wav --audio-right R.wav [--out 报告目录]
    python 口型论文复现.py check --video X.mp4 --audio MIX.wav         只有混音：只能看音画错位
    python 口型论文复现.py check --demo [--inject swap|shift@8|lis_talk_t@1.0|...]   下载一条论文片（约 100 MB）演示 / 人为注错
    python 口型论文复现.py --list                                        论文 8 步
    python 口型论文复现.py <步骤> [参数...] [--data-root D] [--work-dir W] [--syncnet-dir S]
        参数和原来 python -m wvaq.<步骤> 完全一样；路径用命令行给，默认在本文件旁边：
        ./data/seamless_interaction、./wvaq_work、./third_party/syncnet_python（第一次用 SyncNet 时自动下载代码和权重）。

为什么这样合并：各步一开头就导入很重的库（mediapipe、insightface、torch、SyncNet）。直接拼成一份，跑第 7 步也得装全这些库。
所以每个模块的代码放进下面各自的一个函数 module_<模块名>() 里（函数体就是原文件，只是整体缩进 4 格，VS Code 照常上色、跳转）；
运行时取出函数体原文执行，跑哪一步才加载哪一步用到的模块；模块之间的 from wvaq import ... 照常生效。
注意：这些函数不是给你直接调用的。check / demo / inject 的新代码全在文件末尾「新增」一节，只调用这些模块里的函数。
"""

# ============================================================================================
# 在这里改（直接在 VS Code 里点「运行」时用）  Edit here (used when you just press Run)
# ============================================================================================
VIDEO = None          # 你的双人视频，例 "talk.mp4"；None = 用论文演示片（第一次运行自动下载约 100 MB）
AUDIO_LEFT = None     # 左边那个人自己的音轨，例 "left.wav"（和 AUDIO_RIGHT 一起给）
AUDIO_RIGHT = None    # 右边那个人自己的音轨，例 "right.wav"
AUDIO_MIX = None      # 只有一条混音时填这个（只能看音画错位）；都不填 = 用视频自带的声音
INJECT = None         # 人为加一种论文里的错误再检查，例 "swap"（两人声音对调）、"shift@8"（声音晚 667 ms）、
                      # "lis_talk_t@1.0"（听者跟着动嘴）、"spk_mute_t@1.0"（说话人不动嘴）、"wrong_spk_t@1.0"（说错人）；None = 不加
OUT = None            # 报告放哪；None = ./wvaq_check/<视频名>

STEP = "check"        # 跑什么："check" = 检查上面的视频；或论文 8 步之一：prepare_clips, extract_landmarks, landmark_metrics,
                      # make_variants, syncnet_batch, sync_cstar, slt, auc_faults（复现论文用，要先下 Seamless 数据）
DATA_ROOT = "data/seamless_interaction"       # 论文步骤用：Seamless 原始数据目录（相对本文件）
WORK_DIR = "wvaq_work"                        # 论文步骤用：输出目录（相对本文件）
SYNCNET_DIR = "third_party/syncnet_python"    # SyncNet 代码与权重（第一次运行自动下载到这里）
# ============================================================================================

import sys, os, types, importlib.abc, importlib.util

STEPS = ['prepare_clips', 'extract_landmarks', 'landmark_metrics', 'make_variants', 'syncnet_batch', 'syncnet_score', 'sync_cstar', 'slt', 'auc_faults']
MODULES = {}   # 模块名 -> 装着该模块原文的函数

# ====================================================================================================
# wvaq/config.py —— 所有路径（环境变量）（34 行）
# ====================================================================================================
def module_config():
    """All filesystem locations used by the pipeline.

    Every path is taken from an environment variable so that nothing machine-specific
    is hard-coded in the scripts. Set at least WVAQ_DATA_ROOT and WVAQ_WORK_DIR.

      WVAQ_DATA_ROOT        root of the Seamless Interaction download (the directory that
                            contains e.g. improvised/train/0000/0000/<file_id>.{mp4,wav,json})
      WVAQ_WORK_DIR         where all intermediate files and results are written
      WVAQ_SYNCNET_DIR      a clone of https://github.com/joonson/syncnet_python
                            (we import SyncNetModel.py from it)
      WVAQ_SYNCNET_WEIGHTS  syncnet_v2.model (default: $WVAQ_SYNCNET_DIR/data/syncnet_v2.model)
    """
    import os

    # 本合成文件里改过的唯一一处：路径不读环境变量，直接读文件开头「在这里改」的 DATA_ROOT / WORK_DIR / SYNCNET_DIR（相对本文件）
    import __main__
    RELEASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def _here(x): return x if os.path.isabs(x) else os.path.join(RELEASE_DIR, x)
    DATA_ROOT = _here(getattr(__main__, "DATA_ROOT", "data/seamless_interaction"))
    WORK_DIR = _here(getattr(__main__, "WORK_DIR", "wvaq_work"))
    SYNCNET_DIR = _here(getattr(__main__, "SYNCNET_DIR", "third_party/syncnet_python"))
    SYNCNET_WEIGHTS = os.path.join(SYNCNET_DIR, "data", "syncnet_v2.model")

    CLIP_LIST = os.path.join(RELEASE_DIR, "clips", "clips.json")       # the 147 clips used in the paper

    CLIPS_DIR = os.path.join(WORK_DIR, "clips")            # <id>.mp4 (silent), <id>_s1.wav (left), <id>_s2.wav (right), <id>.wav (mix)
    LANDMARK_DIR = os.path.join(WORK_DIR, "landmarks")     # <id>.npz
    VARIANT_DIR = os.path.join(WORK_DIR, "variants")       # v/*.mp4, a/*.wav, manifest.json
    SYNCNET_RESULTS = os.path.join(WORK_DIR, "syncnet", "results.jsonl")
    DMAT_DIR = os.path.join(WORK_DIR, "syncnet", "dmat")   # per-item offset x time distance matrices
    RESULTS_DIR = os.path.join(WORK_DIR, "results")


    def clip_paths(cid, clips_dir=None):
        """Video and the two per-person audio tracks of one clip."""
        d = clips_dir or CLIPS_DIR
        return {"video": os.path.join(d, f"{cid}.mp4"), "s1": os.path.join(d, f"{cid}_s1.wav"), "s2": os.path.join(d, f"{cid}_s2.wav"), "mix": os.path.join(d, f"{cid}.wav")}

MODULES['config'] = module_config

# ====================================================================================================
# wvaq/faults.py —— 错误定义和关键点指标（库）（119 行）
# ====================================================================================================
def module_faults():
    """Fault injection and landmark-based metrics (library).

    Clips are side-by-side recordings of a real two-person conversation: each person occupies one half
    (352 px wide) with their own mic track, and the halves never overlap. A fault is therefore
    injected by re-ordering the frames of one half (or by editing the audio); no seam is created.
    The same functions are used to (a) re-order the per-frame landmark sequences when scoring the
    landmark metrics (landmark_metrics.py) and (b) render the faulted videos for the image-based metrics
    (make_variants.py), so the variants that people look at and the sequences that the metrics measure are
    produced by exactly one implementation.

    Fault names are "<kind>@<p>". Suffix "_c" on the kind: source frames are taken from the longest
    contiguous run only. Suffix "_t": speaking turns are first closed over word gaps of <= 6 frames
    (0.5 s), and the source is the longest contiguous run (replayed back and forth).
    See README.md for the list of faults.
    """
    import wave
    import numpy as np


    def env12(p, n):
        """RMS energy of a 16 kHz mono wav on the 12 fps video grid (n frames)."""
        w = wave.open(p); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768; h = w.getframerate() // 12
        return np.array([np.sqrt(np.mean(x[t*h:(t+1)*h]**2)) if (t+1)*h <= len(x) else 0.0 for t in range(n)])


    def voice_activity(e1, e2):
        """Per-frame activity of the two tracks: energy > 0.25 x the 95th percentile of both tracks pooled."""
        thr = 0.25 * max(np.percentile(np.r_[e1, e2], 95), 1e-4)
        return e1 > thr, e2 > thr


    def act(m):
        """Mouth activity: |frame-to-frame change of the mouth opening ratio|, smoothed over 3 frames, x100."""
        return np.convolve(np.nan_to_num(np.abs(np.diff(m, prepend=m[0])), nan=0.0), np.ones(3) / 3, "same") * 100


    def mad(x): x = x[np.isfinite(x)]; return float(np.median(np.abs(x - np.median(x)))) if len(x) >= 3 else np.nan


    def metrics(mar, lips, nose, fw, a1, a2):
        """Landmark metrics of one clip.
        mar (2,T), lips (2,T,40,2), nose (2,T,2), fw (2,T): the two people's sequences (already re-ordered by the fault);
        a1 / a2: per-frame voice activity of the two audio tracks.
        Spk = mouth activity of the speaker during his own solo speech; Lis = mouth activity of the listener during the
        other person's solo speech; Bind = Lis / Spk. Other columns are reference metrics (see README)."""
        on = [a1 & ~a2, a2 & ~a1]; sil = ~a1 & ~a2; A = [act(mar[0]), act(mar[1])]; r = {k: [] for k in ("Spk", "Lis", "LisOpen", "E2b", "E2c_h")}
        for p in (0, 1):
            q = 1 - p
            if on[p].sum() < 3: continue
            dl = lips[q][1:] - lips[q][:-1]; dn = (nose[q][1:] - nose[q][:-1])[:, None, :]; dh = np.r_[np.nan, np.linalg.norm(dl - dn, axis=-1).mean(-1) / fw[q][1:] * 100]
            r["Spk"].append(A[p][on[p]].mean()); r["Lis"].append(A[q][on[p]].mean()); r["LisOpen"].append(np.nanmean(mar[q][on[p]])); r["E2b"].append(mad(mar[q][on[p]])); r["E2c_h"].append(np.nanmean(dh[on[p]]))
        if not r["Spk"]: return None
        o = {k: float(np.nanmean(v)) for k, v in r.items()}; o["Bind"] = o["Lis"] / max(o["Spk"], 1e-6); o["negSpk"] = -o["Spk"]
        o["SLS"] = float(np.nanmean([mad(mar[0][sil]), mad(mar[1][sil])])) if sil.sum() >= 3 else np.nan
        return o


    def turn_mask(m, gap=6):
        """Close word gaps of <= gap frames so that one speaking turn becomes one contiguous run."""
        m = m.copy(); w = np.where(m)[0]
        for a_, b_ in zip(w[:-1], w[1:]):
            if 1 < b_ - a_ <= gap + 1: m[a_:b_] = True
        return m


    def reindex(n, on_self, on_other, kind, p, a_self=None, contig=False):
        """Frame-index map for one person.
        kind='lis_talk': the frames where the OTHER person speaks alone are replaced by frames where this person speaks alone.
        kind='spk_mute': the frames where this person speaks alone are replaced by quiet frames of this person
                         (frames at least 3 frames away from any own voice activity).
        kind='lis_quiet': cut-only control: the listening frames are replaced by another quiet stretch of this person.
        p: fraction of the target frames that is replaced (the first p of them).
        contig=True: the source frames are only the longest contiguous run (joining non-adjacent frames would create jumps that
        are themselves measured as mouth motion). contig='turn': in addition, turns are first closed over gaps <= 6 frames."""
        idx = np.arange(n)
        if contig == "turn":
            on_self, on_other = turn_mask(on_self), turn_mask(on_other)
        tgt = np.where(on_other)[0] if kind in ("lis_talk", "lis_quiet") else np.where(on_self)[0]
        quiet = ~(np.convolve(a_self.astype(float), np.ones(7), "same") > 0) if a_self is not None else on_other
        src = np.where(on_self)[0] if kind == "lis_talk" else np.where(quiet)[0]
        if contig and len(src):
            runs = np.split(src, np.where(np.diff(src) > 1)[0] + 1); src = max(runs, key=len)
        if len(tgt) < 3 or len(src) < 3: return idx
        k = int(round(len(tgt) * p)); pp = (list(src) + list(src[-2:0:-1])) * 50        # play back and forth to fill
        idx[tgt[:k]] = pp[:k]; return idx


    def person_index(n, on, a1, a2, v):
        """A video-side fault -> per-person frame-index maps [ix_left, ix_right].
        lis_talk@p / spk_mute@p / lis_quiet@p: see reindex.
        wrong_spk@1.0: both at once = the person who should speak keeps still and the one who should listen moves the mouth.
        vlead@k: both halves are advanced by k frames = audio delayed by k frames relative to the video.
        one_lag@k: the left person's video is delayed by k frames, the right person is unchanged.
        rot_spk@1.0: inside each of the left person's turns, the frames are circularly shifted by half the turn.
        rev_spk@1.0: each speaker's frames are reversed inside each of his own turns.
        frz_lis@1.0: the listener is frozen on the first frame of each of the other person's turns."""
        kind, p = v.split("@"); p = float(p); contig = {"_c": True, "_t": "turn"}.get(kind[-2:], False); kind = kind[:-2] if contig else kind
        if kind == "vlead": ix = np.minimum(np.arange(n) + int(p), n - 1); return [ix, ix.copy()]
        if kind == "one_lag":
            return [np.maximum(np.arange(n) - int(p), 0), np.arange(n)]
        if kind == "rot_spk":
            ix = np.arange(n); w = np.where(turn_mask(on[0]))[0]
            for r in (np.split(w, np.where(np.diff(w) > 1)[0] + 1) if len(w) else []):
                if len(r) >= 3: ix[r] = np.roll(r, len(r) // 2)
            return [ix, np.arange(n)]
        if kind in ("rev_spk", "frz_lis"):
            out = []
            for me in (0, 1):
                ix = np.arange(n); m = turn_mask(on[me]) if kind == "rev_spk" else turn_mask(on[1 - me]); w = np.where(m)[0]
                for r in (np.split(w, np.where(np.diff(w) > 1)[0] + 1) if len(w) else []):
                    if len(r) >= 3: ix[r] = r[::-1] if kind == "rev_spk" else r[0]
                out.append(ix)
            return out
        if kind == "wrong_spk":
            out = []
            for me, (os_, oo, am) in enumerate(((on[0], on[1], a1), (on[1], on[0], a2))):
                l, m = reindex(n, os_, oo, "lis_talk", 1.0, am, contig), reindex(n, os_, oo, "spk_mute", 1.0, am, contig); out.append(np.where(l != np.arange(n), l, m))
            return out
        return [reindex(n, on[0], on[1], kind, p, a1, contig), reindex(n, on[1], on[0], kind, p, a2, contig)]

MODULES['faults'] = module_faults

# ====================================================================================================
# wvaq/sync_cstar.py —— 第 6 步：Sync-C*（78 行）
# ====================================================================================================
def module_sync_cstar():
    #!/usr/bin/env python3
    """Step 6: Sync-C* (per-person SyncNet restricted to each person's own speaking periods).

    Sync-C* (AnyTalker, arXiv 2511.23475, Sec. 5, Eq. 7) = (L1 * Sync-C_L1 + L2 * Sync-C_L2) / (L1 + L2): each character is
    scored only over the periods in which they speak, weighted by period length. The paper does not specify which
    audio and which face are fed to SyncNet; we use the per-person protocol: left face x s1, right face x s2 (the two
    tracks actually used by the item).
    Computation: syncnet_batch.py stores per person the offset x time distance matrix (25 fps time steps). Speaking
    periods = the person's own track is active on the 25 fps grid (energy > 0.25 x the 95th percentile of both tracks
    pooled, the same rule as faults.voice_activity). Distances are averaged over those time steps only, then
    conf = median_o - min_o, D = min_o, |o_hat| = |argmin|. A person with < 10 active steps (0.4 s) is dropped; the two
    people are weighted by the number of active steps.

    The main block reports, on the real clips, the detection rate of each fault with the alarm line at the 95th
    percentile of the original clips (conversation bootstrap x2000, seed 0) and the paired comparison.
    Output: <results>/sync_cstar.json.
    Usage: python -m wvaq.sync_cstar [--manifest FILE] [--dmat DIR] [--out FILE]
    """
    import argparse, json, os, wave, numpy as np
    from wvaq import config


    def env25(p, n):
        w = wave.open(p); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768; h = w.getframerate() // 25
        return np.array([np.sqrt(np.mean(x[t*h:(t+1)*h]**2)) if (t+1)*h <= len(x) else 0.0 for t in range(n)])


    def star(item, dmdir):
        """Sync-C* of one manifest item -> {"negC": -Sync-C*, "D": Sync-D*, "absOff": |offset|} or None."""
        f = os.path.join(dmdir, item["key"].replace("|", "__").replace("@", "_") + ".npz")
        if not os.path.exists(f): return None
        z = np.load(f)
        if "L" not in z or "R" not in z: return None
        n = min(z["L"].shape[1], z["R"].shape[1]); e1, e2 = env25(item["s1"], n), env25(item["s2"], n)
        thr = 0.25 * max(np.percentile(np.r_[e1, e2], 95), 1e-4); acc = []
        for side, e in (("L", e1), ("R", e2)):
            m = e > thr
            if m.sum() < 10: continue
            d = z[side].astype(np.float64)[:, :n][:, m].mean(1); vs = (len(d) - 1) // 2; mi = int(np.argmin(d))
            acc.append((m.sum(), np.median(d) - d.min(), d.min(), abs(vs - mi)))
        if not acc: return None
        w = np.array([a[0] for a in acc], float)
        return {"negC": -float(np.average([a[1] for a in acc], weights=w)), "D": float(np.average([a[2] for a in acc], weights=w)), "absOff": float(np.average([a[3] for a in acc], weights=w))}


    def conv(c): return "_".join(c.split("_")[1:4])


    if __name__ == "__main__":
        ap = argparse.ArgumentParser()
        ap.add_argument("--manifest", default=os.path.join(config.VARIANT_DIR, "manifest.json"))
        ap.add_argument("--dmat", default=config.DMAT_DIR)
        ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "sync_cstar.json"))
        A = ap.parse_args(); os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
        OUT = {"real": {}, "paired": {}, "_thr": {}}
        RS = {}
        for it in json.load(open(A.manifest)):
            r = star(it, A.dmat)
            if r: RS[it["key"]] = r
        clips = sorted(set(k.split("|")[0] for k in RS if k.split("|")[1] == "none")); CV = sorted(set(map(conv, clips)))
        rng = np.random.default_rng(0); VARS = sorted(set(k.split("|")[1] for k in RS))

        def boot(prs):
            by = {g: [d for gg, d in prs if gg == g] for g in CV}
            bs = [np.mean([d for g in rng.choice(CV, len(CV)) for d in by.get(g, [])] or [np.nan]) for _ in range(2000)]
            return [float(np.mean([d for _, d in prs])), *map(float, np.nanpercentile(bs, [2.5, 97.5])), len(prs)]
        for k in ("negC", "D", "absOff"):
            thr = float(np.percentile([RS[f"{c}|none"][k] for c in clips], 95)); OUT["_thr"][k] = thr
            for v in VARS:
                prs = [(conv(c), RS[f"{c}|{v}"][k] > thr) for c in clips if f"{c}|{v}" in RS]
                if prs: OUT["real"][f"{v}|{k}"] = boot(prs)
                pp = [(conv(c), 1.0 if RS[f"{c}|{v}"][k] > RS[f"{c}|none"][k] else 0.5 if RS[f"{c}|{v}"][k] == RS[f"{c}|none"][k] else 0.0) for c in clips if f"{c}|{v}" in RS]
                if pp and v != "none": OUT["paired"][f"{v}|{k}"] = boot(pp)
        json.dump(OUT, open(A.out, "w"), indent=1)
        print("real clips n =", len(clips), "alarm lines", {k: round(v, 2) for k, v in OUT["_thr"].items()})
        for v in VARS:
            print(f"{v:18s}", " ".join(f"{k}:{OUT['real'][f'{v}|{k}'][0]*100:3.0f}%[{OUT['real'][f'{v}|{k}'][1]*100:.0f},{OUT['real'][f'{v}|{k}'][2]*100:.0f}]" if f"{v}|{k}" in OUT["real"] else f"{k}: -" for k in ("negC", "D", "absOff")),
                  " paired C*:", f"{OUT['paired'][f'{v}|negC'][0]*100:.0f}%" if f"{v}|negC" in OUT["paired"] else "-")

MODULES['sync_cstar'] = module_sync_cstar

# ====================================================================================================
# wvaq/slt.py —— 第 7 步：SLT 三项、报警线、检出率（133 行）
# ====================================================================================================
def module_slt():
    #!/usr/bin/env python3
    """Step 7: SLT - combined speaker / listener / timing decision.

    Three parts, all oriented "larger = worse":
      S (speaker)  = -Sync-C*  (per person, own speaking periods only; sync_cstar.star)
      L (listener) = Bind      (listener mouth activity / speaker mouth activity; landmark_metrics.py)
      T (timing)   = mean over the two people of the per-person SyncNet offset |o_hat|  (syncnet_batch per_left / per_right)
    Decision: alarm if any part exceeds its own line. The three lines are the same quantile q of the original clips;
    q is found by bisection on the original clips so that the combined false-alarm rate is <= 5 %
    (at P95 each, the combined false-alarm rate is about 14 %, so each line must be stricter).
    Report: detection rate per fault (conversation bootstrap x2000, seed 0), each part alone at its own P95, the lowest
    combined detection rate over the full-strength faults, and a held-out check: 500 times, lines calibrated on half of
    the conversations (again to 5 % combined false alarms) and evaluated on the other half.

    Output: <results>/slt.json (fields: n_clips, n_conv, q, lines [S, L, T], combo, single_P95, combo_min_full,
    single_min_full, heldout).
    Usage:
      python -m wvaq.slt                                   # calibrate lines and report
      python -m wvaq.slt --apply-lines slt.json [--out decisions.json]   # apply given lines to every manifest item
    """
    import argparse, json, os, numpy as np
    from wvaq import config
    from wvaq.sync_cstar import star

    VARS = ["none", "swap", "wrong_spk_t@1.0", "lis_talk_t@1.0", "lis_talk_t@0.33", "spk_mute_t@1.0", "lis_quiet_t@1.0", "shift@2", "shift@8", "vlead@2", "vlead@8"]
    FULL = ["swap", "wrong_spk_t@1.0", "lis_talk_t@1.0", "spk_mute_t@1.0", "shift@8", "vlead@8"]          # full-strength faults


    def load_jsonl(p):
        """SyncNet results: one JSON object per line; successful rows win over error rows, later rows over earlier ones."""
        R = {}
        for l in open(p):
            try: r = json.loads(l)
            except Exception: continue
            if "error" in r and r["key"] in R: continue
            R[r["key"]] = r
        return R


    def off(r):
        if not r or not (r.get("per_left") and r.get("per_right")): return np.nan
        return (abs(r["per_left"]["offset"]) + abs(r["per_right"]["offset"])) / 2


    def conv(c): return "_".join(c.split("_")[1:4])


    class Parts:
        """Loads the three inputs and returns the [S, L, T] vector of one (clip, fault)."""
        def __init__(self, landmarks_json, syncnet_jsonl, manifest, dmat):
            self.L = json.load(open(landmarks_json)); self.SR = load_jsonl(syncnet_jsonl)
            self.M = {it["key"]: it for it in json.load(open(manifest))}; self.CS = {}
            for k, it in self.M.items():
                r = star(it, dmat)
                if r: self.CS[k] = r

        def vals(self, c, v):
            k = f"{c}|{v}"; b = (self.L.get(c) or {}).get(v) or {}
            return [self.CS[k]["negC"] if k in self.CS else np.nan, b.get("Bind", np.nan), off(self.SR.get(k))]


    def setup(P):
        """Clips used for calibration: SyncNet-scored clips whose original has Sync-C* and landmark metrics."""
        clips = sorted(c for c in set(k.split("|")[0] for k in P.SR) if f"{c}|none" in P.CS and c in P.L)
        X = {v: np.array([P.vals(c, v) for c in clips]) for v in VARS}; G = np.array([conv(c) for c in clips])
        return clips, X, G


    def lines(X, q, rows=None):
        base = X["none"] if rows is None else X["none"][rows]; return np.nanpercentile(base, q, axis=0)


    def alarm(a, thr): return np.any(np.nan_to_num(a, nan=-np.inf) > thr, axis=1)


    def calib(X, rows=None):
        lo, hi = 90.0, 100.0
        for _ in range(40):
            q = (lo + hi) / 2; fa = alarm(X["none"] if rows is None else X["none"][rows], lines(X, q, rows)).mean()
            lo, hi = (lo, q) if fa < 0.05 else (q, hi)
        return hi


    def main(A, P):
        clips, X, G = setup(P); CV = sorted(set(G)); rng = np.random.default_rng(0)

        def boot(flags):
            by = {g: flags[G == g] for g in CV}; bs = [np.mean(np.concatenate([by[g] for g in rng.choice(CV, len(CV))])) for _ in range(2000)]
            return [float(flags.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]
        q = calib(X); thr = lines(X, q); OUT = {"n_clips": len(clips), "n_conv": len(CV), "q": q, "lines": thr.tolist(), "combo": {}, "single_P95": {}}
        for v in VARS: OUT["combo"][v] = boot(alarm(X[v], thr))
        for i, nm in enumerate(["speaker_SyncCstar", "listener_Bind", "timing_absOff"]):
            t = np.nanpercentile(X["none"][:, i], 95); OUT["single_P95"][nm] = {v: boot(np.nan_to_num(X[v][:, i], nan=-np.inf) > t) for v in VARS}
        OUT["combo_min_full"] = min(OUT["combo"][v][0] for v in FULL)
        OUT["single_min_full"] = {nm: min(d[v][0] for v in FULL) for nm, d in OUT["single_P95"].items()}
        # held-out conversations: calibrate on half of the conversations (combined false alarm 5 %), test on the other half
        fa, det = [], {v: [] for v in VARS[1:]}
        for _ in range(500):
            cal = set(rng.choice(CV, len(CV) // 2, replace=False)); tr = np.array([g in cal for g in G]); te = ~tr
            qq = calib(X, np.where(tr)[0]); t2 = lines(X, qq, np.where(tr)[0])
            fa.append(alarm(X["none"][te], t2).mean())
            for v in VARS[1:]: det[v].append(alarm(X[v][te], t2).mean())
        OUT["heldout"] = {"false_alarm": [float(np.mean(fa)), *map(float, np.percentile(fa, [2.5, 97.5]))], **{v: float(np.mean(d)) for v, d in det.items()}}
        json.dump(OUT, open(A.out, "w"), indent=1)
        print(f"{len(clips)} clips / {len(CV)} conversations; per-line quantile q = {q:.2f}; lines S/L/T = {[round(x, 3) for x in thr]}")
        for v in VARS: print(f"  {v:16s} combined {OUT['combo'][v][0]*100:4.0f}% [{OUT['combo'][v][1]*100:.0f},{OUT['combo'][v][2]*100:.0f}]  alone: " +
                             " ".join(f"{nm.split('_')[0]} {OUT['single_P95'][nm][v][0]*100:3.0f}" for nm in OUT["single_P95"]))
        print("lowest detection over full-strength faults: combined", round(OUT["combo_min_full"] * 100), " alone", {k: round(v * 100) for k, v in OUT["single_min_full"].items()})
        print("held-out conversations: combined false alarm", [round(x * 100, 1) for x in OUT["heldout"]["false_alarm"]], " full-strength detection", {v: round(OUT['heldout'][v] * 100) for v in FULL})


    def apply_lines(A, P):
        thr = np.array(json.load(open(A.apply_lines))["lines"], float); out = {}
        for k in sorted(P.M):
            c, v = k.split("|"); x = np.array(P.vals(c, v), float)
            out[k] = {"S": x[0], "L": x[1], "T": x[2], "alarm": bool(alarm(x[None], thr)[0]), "parts_over_line": [nm for nm, xi, t in zip("SLT", x, thr) if np.isfinite(xi) and xi > t]}
            print(f"{k:48s} S {x[0]:7.3f}  L {x[1]:7.3f}  T {x[2]:5.2f}  -> {'ALARM ' + ','.join(out[k]['parts_over_line']) if out[k]['alarm'] else 'ok'}")
        json.dump({"lines": thr.tolist(), "items": {k: {kk: (None if isinstance(vv, float) and not np.isfinite(vv) else vv) for kk, vv in r.items()} for k, r in out.items()}}, open(A.out, "w"), indent=1)


    if __name__ == "__main__":
        ap = argparse.ArgumentParser()
        ap.add_argument("--landmark-metrics", default=os.path.join(config.RESULTS_DIR, "landmark_metrics.json"))
        ap.add_argument("--syncnet", default=config.SYNCNET_RESULTS)
        ap.add_argument("--manifest", default=os.path.join(config.VARIANT_DIR, "manifest.json"))
        ap.add_argument("--dmat", default=config.DMAT_DIR)
        ap.add_argument("--apply-lines", default="")
        ap.add_argument("--out", default="")
        A = ap.parse_args()
        A.out = A.out or os.path.join(config.RESULTS_DIR, "slt_decisions.json" if A.apply_lines else "slt.json")
        os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
        P = Parts(A.landmark_metrics, A.syncnet, A.manifest, A.dmat)
        (apply_lines if A.apply_lines else main)(A, P)

MODULES['slt'] = module_slt

# ====================================================================================================
# wvaq/auc_faults.py —— 第 8 步：每种错误的 AUC（83 行）
# ====================================================================================================
def module_auc_faults():
    #!/usr/bin/env python3
    """Step 8: threshold-free AUC per fault.

    Same clips as slt.py (the SyncNet set). For each clip the three SLT parts S (-Sync-C*), L (Bind) and T (mean per-person
    |o_hat|) are computed; the continuous SLT score is the maximum over the three parts of each part's percentile among the
    original clips (no alarm line needed). For each fault, AUC of faulted vs original clips (same clips, paired set),
    with a conversation bootstrap (2000 resamples). Faults are reported in two groups: the six types seen while designing
    SLT (plus their weaker versions) and five fault types defined in advance and never used for design (held-out types).
    As a reference, the AUC of conventional mixed-audio SyncNet (-Sync-C on the larger face track) is also reported (MIX).

    Output: <results>/auc_faults.json.
    Usage: python -m wvaq.auc_faults [same input options as slt.py]
    """
    import argparse, json, os, numpy as np
    from wvaq import config
    from wvaq.slt import Parts, setup, conv

    DESIGN = ["swap", "wrong_spk_t@1.0", "lis_talk_t@1.0", "spk_mute_t@1.0", "shift@8", "vlead@8", "lis_talk_t@0.33", "shift@2", "vlead@2"]
    NEW = ["rev_spk_t@1.0", "frz_lis_t@1.0", "one_lag_t@8", "voice_drop_t@1.0", "rot_spk_t@1.0"]


    def auc(pos, neg):
        pos, neg = np.asarray(pos, float), np.asarray(neg, float); pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
        if not len(pos) or not len(neg): return np.nan
        gt = (pos[:, None] > neg[None, :]).mean(); eq = (pos[:, None] == neg[None, :]).mean(); return float(gt + 0.5 * eq)


    def big(r):
        """Conventional SyncNet: -Sync-C of the mixed track on the side with the larger median face width."""
        if not r or not r.get("face_w") or r["face_w"]["L"] is None: return np.nan
        s = r["std_L"] if r["face_w"]["L"] >= r["face_w"]["R"] else r["std_R"]; return -s["conf"] if s else np.nan


    if __name__ == "__main__":
        ap = argparse.ArgumentParser()
        ap.add_argument("--landmark-metrics", default=os.path.join(config.RESULTS_DIR, "landmark_metrics.json"))
        ap.add_argument("--syncnet", default=config.SYNCNET_RESULTS)
        ap.add_argument("--manifest", default=os.path.join(config.VARIANT_DIR, "manifest.json"))
        ap.add_argument("--dmat", default=config.DMAT_DIR)
        ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "auc_faults.json"))
        A = ap.parse_args(); os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
        P = Parts(A.landmark_metrics, A.syncnet, A.manifest, A.dmat)
        CL, X, _ = setup(P)
        S = {c: X["none"][i] for i, c in enumerate(CL)}                     # original clips, three parts
        FAM = {v: {c: np.array(P.vals(c, v), float) for c in CL} for v in DESIGN}
        for v in NEW: FAM[v] = {c: np.array(P.vals(c, v), float) for c in CL if f"{c}|{v}" in P.M}
        base = np.array(list(S.values()))

        def pct(x, j):  # percentile among the original clips (larger = worse)
            b = base[:, j]; b = b[np.isfinite(b)]
            return np.nan if not np.isfinite(x) else float((b < x).mean() + 0.5 * (b == x).mean())

        def slt(x): return np.nanmax([pct(x[j], j) for j in range(3)]) if np.any(np.isfinite(x)) else np.nan
        rng = np.random.default_rng(0); OUT = {"n_base": len(S), "faults": {}}
        for v, d in FAM.items():
            cs = [c for c in d if c in S]; G = np.array([conv(c) for c in cs]); CV = sorted(set(G))
            def score(fn, idx): return auc([fn(d[c]) for c in idx], [fn(S[c]) for c in idx])
            fns = {"SLT": slt, "S": lambda x: x[0], "L": lambda x: x[1], "T": lambda x: x[2]}
            r = {}
            for nm, fn in fns.items():
                pt = score(fn, cs); bs = []
                by = {g: [c for c in cs if conv(c) == g] for g in CV}
                for _ in range(2000):
                    idx = [c for g in rng.choice(CV, len(CV)) for c in by[g]]; bs.append(score(fn, idx))
                r[nm] = [pt, *np.nanpercentile(bs, [2.5, 97.5]).tolist(), len(cs)]
            OUT["faults"][v] = {"held_out_type": v in NEW, **r}
        for v, r in OUT["faults"].items():
            print(f"{'held-out' if r['held_out_type'] else 'design  '} {v:18s} " + "  ".join(f"{k} {r[k][0]:.2f} [{r[k][1]:.2f},{r[k][2]:.2f}]" for k in ("SLT", "S", "L", "T")) + f"  n={r['SLT'][3]}")

        # reference: conventional mixed-audio SyncNet (mixed track x larger face track), -C, same clips
        TD = dict(P.SR)
        for c in CL: TD[f"{c}|swap"] = TD.get(f"{c}|none")      # swapping s1 / s2 leaves video and mixed track unchanged: use the original's score
        MN = {c: big(TD.get(f"{c}|none")) for c in CL}
        for v in OUT["faults"]:
            cs = [c for c in CL if np.isfinite(MN[c]) and np.isfinite(big(TD.get(f"{c}|{v}")))]
            if not cs: continue
            Pv = {c: big(TD.get(f"{c}|{v}")) for c in cs}; G = np.array([conv(c) for c in cs]); CV = sorted(set(G)); by = {g: [c for c in cs if conv(c) == g] for g in CV}
            pt = auc([Pv[c] for c in cs], [MN[c] for c in cs]); bs = []
            for _ in range(2000):
                idx = [c for g in rng.choice(CV, len(CV)) for c in by[g]]; bs.append(auc([Pv[c] for c in idx], [MN[c] for c in idx]))
            OUT["faults"][v]["MIX"] = [pt, *np.nanpercentile(bs, [2.5, 97.5]).tolist(), len(cs)]
        json.dump(OUT, open(A.out, "w"), indent=1)
        print("MIX:", {v: round(r["MIX"][0], 2) for v, r in OUT["faults"].items() if "MIX" in r})

MODULES['auc_faults'] = module_auc_faults

# ====================================================================================================
# wvaq/landmark_metrics.py —— 第 3 步：Bind / Lis / Spk 等关键点指标（90 行）
# ====================================================================================================
def module_landmark_metrics():
    #!/usr/bin/env python3
    """Step 3: score every fault with the landmark metrics (Bind / Lis / Spk and reference metrics).

    For each clip and each fault, the per-frame landmark sequences (or the voice-activity masks, for
    audio-side faults) are modified with the functions in faults.py and the metrics are recomputed.
    All metrics are oriented "larger = worse" (Spk enters as negSpk).

    Two read-outs are printed (confidence intervals: conversation-level bootstrap, 2000 resamples, seed 0;
    clips cut from the same conversation are not independent):
      1. paired: a fault is detected on a clip if its score is worse than the clip's own original score;
      2. absolute: alarm line = 95th percentile of the original clips (5 % false alarms on real clips).

    Output: <out>.json (per clip: "orig" and one entry per fault) and <out>_summary.json.
    Usage: python -m wvaq.landmark_metrics [--landmarks DIR] [--clips DIR] [--out FILE]
    """
    import argparse, json, os, glob, numpy as np
    from wvaq import config
    from wvaq.faults import env12, metrics, person_index, turn_mask

    VARS = ["none", "swap"] + [f"lis_talk@{p}" for p in (0.33, 0.66, 1.0)] + [f"spk_mute@{p}" for p in (0.33, 0.66, 1.0)] + [f"shift@{k}" for k in (2, 4, 8)] + ["wrong_spk@1.0", "vlead@2", "vlead@8"] + ["lis_talk_c@1.0", "lis_talk_c@0.33", "spk_mute_c@1.0", "wrong_spk_c@1.0", "lis_talk_t@1.0", "lis_talk_t@0.33", "spk_mute_t@1.0", "wrong_spk_t@1.0", "lis_quiet_t@1.0", "rev_spk_t@1.0", "frz_lis_t@1.0", "one_lag_t@8", "voice_drop_t@1.0", "rot_spk_t@1.0"]
    KS = ("Lis", "Bind", "LisOpen", "E2b", "SLS", "E2c_h", "negSpk")


    def score_clip(npz, s1, s2, vars_=VARS):
        d = np.load(npz); mar, lips, nose, fw = d["mar"], d["lips"], d["nose"], d["facew"]; n = mar.shape[1]
        e1, e2 = env12(s1, n), env12(s2, n); thr = 0.25 * max(np.percentile(np.r_[e1, e2], 95), 1e-4); a1, a2 = e1 > thr, e2 > thr; on = [a1 & ~a2, a2 & ~a1]
        base = metrics(mar, lips, nose, fw, a1, a2)
        if base is None: return None
        R = {"orig": base}
        for v in vars_:
            if v == "none": m_ = metrics(mar, lips, nose, fw, a1, a2)
            elif v == "swap": m_ = metrics(mar, lips, nose, fw, a2, a1)
            elif v.startswith("voice_drop"):   # the left person's own track is muted in each of his turns -> by audio he no longer speaks
                m_ = metrics(mar, lips, nose, fw, a1 & ~turn_mask(on[0]), a2)
            elif v.startswith("shift"):
                k = int(v.split("@")[1]); sh = lambda a: np.r_[np.zeros(k, bool), a[:-k]]; m_ = metrics(mar, lips, nose, fw, sh(a1), sh(a2))
            else:
                ix = person_index(n, on, a1, a2, v)
                m_ = metrics(np.stack([mar[0][ix[0]], mar[1][ix[1]]]), np.stack([lips[0][ix[0]], lips[1][ix[1]]]), np.stack([nose[0][ix[0]], nose[1][ix[1]]]), np.stack([fw[0][ix[0]], fw[1][ix[1]]]), a1, a2)
            R[v] = m_
        return R


    def conv(c): return "_".join(c.split("_")[1:4])


    if __name__ == "__main__":
        ap = argparse.ArgumentParser()
        ap.add_argument("--landmarks", default=config.LANDMARK_DIR)
        ap.add_argument("--clips", default=config.CLIPS_DIR)
        ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "landmark_metrics.json"))
        A = ap.parse_args(); os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
        R = {}
        for f in sorted(glob.glob(f"{A.landmarks}/*.npz")):
            c = os.path.basename(f)[:-4]; p = config.clip_paths(c, A.clips)
            r = score_clip(f, p["s1"], p["s2"])
            if r is not None: R[c] = r
        json.dump(R, open(A.out, "w"), indent=1); cs = list(R)
        CV = sorted(set(conv(c) for c in cs)); rng = np.random.default_rng(0)
        print(f"{len(cs)} clips / {len(CV)} conversations. Cell = detection rate (clips where the faulted score is worse than the original / scorable clips) [95% CI, conversation bootstrap x2000].")
        print(f"{'fault':16s} " + " ".join(f"{k:>20s}" for k in KS)); SUM = {}
        for v in VARS:
            row = []
            for k in KS:
                prs = [(conv(c), R[c][v][k] > R[c]["orig"][k] + 1e-9) for c in cs if R[c][v] is not None and np.isfinite(R[c][v][k]) and np.isfinite(R[c]["orig"][k])]
                if not prs: row.append("-"); continue
                by = {g: [d for gg, d in prs if gg == g] for g in CV}; bs = []
                for _ in range(2000):
                    x = [d for g in rng.choice(CV, len(CV)) for d in by[g]]; bs.append(np.mean(x) if x else np.nan)
                r = float(np.mean([d for _, d in prs])); lo, hi = np.nanpercentile(bs, [2.5, 97.5]); SUM[f"{v}|{k}"] = {"rate": r, "lo": float(lo), "hi": float(hi), "n": len(prs)}
                row.append(f"{r*100:3.0f}% [{lo*100:3.0f},{hi*100:3.0f}] n={len(prs)}")
            print(f"{v:16s} " + " ".join(f"{x:>20s}" for x in row))
        # absolute read-out: alarm line = 95th percentile of the original clips (5 % false alarms on real clips).
        # Under the paired read-out a metric that changes at random already reaches 50 %.
        THR = {k: float(np.nanpercentile([R[c]["orig"][k] for c in cs], 95)) for k in KS}
        print("\nAbsolute read-out: detected if above the 95th percentile of the original clips. Lines: " + " ".join(f"{k}={THR[k]:.3g}" for k in KS))
        print(f"{'fault':16s} " + " ".join(f"{k:>20s}" for k in KS))
        for v in VARS:
            row = []
            for k in KS:
                prs = [(conv(c), R[c][v][k] > THR[k]) for c in cs if R[c][v] is not None and np.isfinite(R[c][v][k])]
                if not prs: row.append("-"); continue
                by = {g: [d for gg, d in prs if gg == g] for g in CV}; bs = []
                for _ in range(2000):
                    x = [d for g in rng.choice(CV, len(CV)) for d in by[g]]; bs.append(np.mean(x) if x else np.nan)
                r = float(np.mean([d for _, d in prs])); lo, hi = np.nanpercentile(bs, [2.5, 97.5]); SUM[f"abs|{v}|{k}"] = {"rate": r, "lo": float(lo), "hi": float(hi), "n": len(prs)}
                row.append(f"{r*100:3.0f}% [{lo*100:3.0f},{hi*100:3.0f}] n={len(prs)}")
            print(f"{v:16s} " + " ".join(f"{x:>20s}" for x in row))
        SUM["_thr"] = THR
        json.dump(SUM, open(A.out.replace(".json", "_summary.json"), "w"), indent=1)

MODULES['landmark_metrics'] = module_landmark_metrics

# ====================================================================================================
# wvaq/make_variants.py —— 第 4 步：生成带错误的视频 / 音频（100 行）
# ====================================================================================================
def module_make_variants():
    #!/usr/bin/env python3
    """Step 4: render faulted variants (video / audio files) for the image-based metrics (SyncNet).

    The fault definitions are imported from faults.py, i.e. the same code that re-orders the landmark
    sequences in landmark_metrics.py. Each person occupies one 352 px half, so a video fault re-orders the
    frames of one half without creating a seam.

    Outputs (under --out):
      v/<clip>__<name>.mp4                 video faults (name = kind for the @1.0 versions of lis_talk / spk_mute /
                                           wrong_spk, otherwise <kind>_<p>)
      a/<clip>__mix.wav                    sample-wise sum of the two own-mic tracks
      a/<clip>__{s1,s2,mix}_sh{2,8}.wav    audio delayed by 2 / 8 frames (shift@2 / shift@8)
      a/<clip>__{s1,mix}_vdrop.wav         voice_drop_t@1.0: left track muted in each of the left person's turns
      manifest.json                        [{"key": "<clip>|<fault>", "video", "s1", "s2", "mix"}, ...] for syncnet_batch.py
    Audio-only faults: none / swap (s1 and s2 exchanged, mix unchanged) / shift@2 / shift@8 / voice_drop_t@1.0.

    --n N: take N clips round-robin over conversations (as many per conversation as possible). The paper's
    SyncNet set is --n 100 on the 147 clips (listed in clips/syncnet100.txt).

    --clip-list FILE: use exactly these clips instead (e.g. clips/syncnet100.txt).

    Usage: python -m wvaq.make_variants [--n 100 | --clip-list FILE] [--vars a,b,...] [--no-base]
    """
    import argparse, json, os, glob, wave, subprocess, collections, numpy as np, cv2
    from wvaq import config
    from wvaq.faults import env12, person_index, turn_mask

    # faults used by SLT (design set) and the held-out fault types
    DEFAULT_VARS = ["wrong_spk_t@1.0", "lis_talk_t@1.0", "lis_talk_t@0.33", "spk_mute_t@1.0", "lis_quiet_t@1.0", "vlead@2", "vlead@8",
                    "rev_spk_t@1.0", "frz_lis_t@1.0", "one_lag_t@8", "voice_drop_t@1.0", "rot_spk_t@1.0"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", default=config.LANDMARK_DIR)
    ap.add_argument("--clips", default=config.CLIPS_DIR)
    ap.add_argument("--out", default=config.VARIANT_DIR)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--clip-list", default="", help="text file with one clip id per line; overrides --n")
    ap.add_argument("--vars", default=",".join(DEFAULT_VARS), help="comma-separated faults to render (besides the audio-only base set)")
    ap.add_argument("--no-base", action="store_true", help="do not emit none / swap / shift@2 / shift@8")
    ap.add_argument("--manifest-name", default="manifest.json")
    A = ap.parse_args()
    S, OUT, N = A.landmarks, A.out, A.n
    VIDEO_VARS = [v for v in A.vars.split(",") if v]; WITH_BASE = not A.no_base
    os.makedirs(f"{OUT}/v", exist_ok=True); os.makedirs(f"{OUT}/a", exist_ok=True)
    cs = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{S}/*.npz"))
    if A.clip_list:
        keep = {l.strip() for l in open(A.clip_list) if l.strip()}; cs = [c for c in cs if c in keep]
    elif N:
        by = collections.defaultdict(list)
        for c in cs: by["_".join(c.split("_")[1:4])].append(c)
        pick = []; i = 0
        while len(pick) < N and any(len(v) > i for v in by.values()):
            pick += [v[i] for v in by.values() if len(v) > i]; i += 1
        cs = sorted(pick[:N])


    def rd(p):
        w = wave.open(p); assert w.getframerate() == 16000 and w.getnchannels() == 1; return np.frombuffer(w.readframes(w.getnframes()), np.int16)


    def wr(p, x):
        w = wave.open(p, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(x.astype(np.int16).tobytes()); w.close()


    def enc(p, frames):
        h, w_ = frames[0].shape[:2]
        pr = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w_}x{h}", "-r", "12", "-i", "-", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", p], stdin=subprocess.PIPE)
        for f in frames: pr.stdin.write(f.tobytes())
        pr.stdin.close(); pr.wait()


    man = []
    for c in cs:
        d = np.load(f"{S}/{c}.npz"); n = d["mar"].shape[1]; P = config.clip_paths(c, A.clips); s1, s2, video = P["s1"], P["s2"], P["video"]
        e1, e2 = env12(s1, n), env12(s2, n); thr = 0.25 * max(np.percentile(np.r_[e1, e2], 95), 1e-4); a1, a2 = e1 > thr, e2 > thr; on = [a1 & ~a2, a2 & ~a1]
        cap = cv2.VideoCapture(video); fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fr.append(f)
        fr = fr[:n]; hw = fr[0].shape[1] // 2; x1, x2 = rd(s1), rd(s2); L = min(len(x1), len(x2)); x1, x2 = x1[:L], x2[:L]; mix = np.clip(x1.astype(np.int32) + x2, -32768, 32767)
        wr(f"{OUT}/a/{c}__mix.wav", mix)
        base = {"video": video, "s1": s1, "s2": s2, "mix": f"{OUT}/a/{c}__mix.wav"}
        if WITH_BASE: man.append({"key": f"{c}|none", **base}); man.append({"key": f"{c}|swap", **base, "s1": s2, "s2": s1})
        for k in ((2, 8) if WITH_BASE else ()):
            z = np.zeros(k * 16000 // 12, np.int16); sh = {}
            for nm, x in (("s1", x1), ("s2", x2), ("mix", mix)): p = f"{OUT}/a/{c}__{nm}_sh{k}.wav"; wr(p, np.r_[z, x][:L]); sh[nm] = p
            man.append({"key": f"{c}|shift@{k}", "video": video, **sh})
        if "voice_drop_t@1.0" in VIDEO_VARS:   # audio only: s1 muted in each of the left person's turns, mix recomputed, original video
            tm = turn_mask(on[0]); y1 = x1.copy(); h = 16000 // 12
            for t in np.where(tm)[0]: y1[t*h:(t+1)*h] = 0
            p1, pm = f"{OUT}/a/{c}__s1_vdrop.wav", f"{OUT}/a/{c}__mix_vdrop.wav"; wr(p1, y1); wr(pm, np.clip(y1.astype(np.int32) + x2, -32768, 32767))
            man.append({"key": f"{c}|voice_drop_t@1.0", "video": video, "s1": p1, "s2": s2, "mix": pm})
        for v in [x for x in VIDEO_VARS if not x.startswith("voice_drop")]:
            ix = person_index(n, on, a1, a2, v); kind, p_ = v.split("@")
            name = kind if (kind in ("lis_talk", "spk_mute", "wrong_spk") and p_ == "1.0") else f"{kind}_{p_}"
            out = [np.hstack([fr[ix[0][t]][:, :hw], fr[ix[1][t]][:, hw:]]) for t in range(len(fr))]; p = f"{OUT}/v/{c}__{name}.mp4"
            enc(p, out)
            man.append({"key": f"{c}|{v}", **base, "video": p})
    json.dump(man, open(f"{OUT}/{A.manifest_name}", "w"), indent=1)
    print(len(cs), "clips,", len(man), "items")

MODULES['make_variants'] = module_make_variants

# ====================================================================================================
# wvaq/prepare_clips.py —— 第 1 步：切出双人片段（171 行）
# ====================================================================================================
def module_prepare_clips():
    #!/usr/bin/env python3
    """Step 1: cut side-by-side two-person clips from Seamless Interaction recordings.

    Seamless Interaction records each participant of a conversation with their own camera and
    their own mic, and ships official VAD and word-level transcripts. We put the two
    participants of one conversation side by side, so who is speaking and every mouth shape are real,
    and the audio is available per person (own mic).

    Window search (mode "search"):
      Inside one conversation, find 4.083 s windows (49 frames at 12 fps) where in the first part one
      person speaks (VAD coverage >= 60 %) while the other is mostly silent (<= 15 %), and after a
      switch point (1 to 3 s) the roles are reversed.
    Video:
      each person's full frame is scaled to height 1280 (lanczos), then cropped to width 352 centred on
      the face at the middle of the window; the two halves are stacked horizontally into
      704x1280, 12 fps, 49 frames, no audio.
      Face requirements: exactly one face, face width 40..260 px, face-centre drift over first / middle /
      last frame smaller than one face width, and the crop must fully contain the face.
    Audio:
      the two own-mic wav files are cut at the same time, 16 kHz mono ->
      <id>_s1.wav (left), <id>_s2.wav (right), <id>.wav (mix).
    Person boxes ([row0, col0, row1, col1]): left [0,0,1280,352], right [0,352,1280,704].

    Mode "list" (default): recreate exactly the clips in clips/clips.json (same start times and crops)
    instead of searching. Mode "search" runs the original window search on whatever conversations are
    present under --roots.

    Usage:
      python -m wvaq.prepare_clips                         # recreate the 147 listed clips
      python -m wvaq.prepare_clips --search --max-per-conv 12
    """
    import argparse, glob, json, os, re, subprocess, collections
    import numpy as np, cv2
    from wvaq import config

    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default=config.DATA_ROOT, help="comma-separated Seamless roots (both participants of a conversation may be under different roots)")
    ap.add_argument("--out", default=config.CLIPS_DIR)
    ap.add_argument("--search", action="store_true", help="search windows instead of recreating the listed clips")
    ap.add_argument("--clip-list", default=config.CLIP_LIST)
    ap.add_argument("--skip-convs-from", default="", help="clips_manifest.json: skip conversations that appear in it (search mode)")
    ap.add_argument("--max-per-conv", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help=">0: stop after this many clips")
    A = ap.parse_args()
    SEC, NF, FPS, W2, H = 49 / 12, 49, 12, 352, 1280
    os.makedirs(A.out, exist_ok=True)
    ROOTS = [r for r in A.roots.split(",") if r]


    def vadmask(base, dur):
        m = np.zeros(int(dur * 100) + 1, bool)
        for s in json.load(open(base + ".json"))["metadata:vad"]: m[int(s["start"] * 100):int(s["end"] * 100)] = True
        return m


    def words(base, t0, t1):
        out = []
        for seg in json.load(open(base + ".json"))["metadata:transcript"]:
            for w in seg.get("words", []):
                if w.get("start") is not None and t0 <= w["start"] < t1: out.append(w["word"])   # a few words have no timestamp
        return " ".join(out)


    def dur(p):
        # some upstream segments are empty shells (ffprobe reports N/A) -> return 0 so the conversation is skipped
        try: return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p], capture_output=True, text=True).stdout)
        except ValueError: return 0.0


    def cut(cid, pa, pb, t0, crop_x):
        """Write <cid>.mp4 / _s1.wav / _s2.wav / .wav with the exact ffmpeg settings used for the paper."""
        fcx = ";".join(f"[{i}:v]scale=-2:{H}:flags=lanczos,crop={W2}:{H}:{crop_x[i]}:0,fps={FPS}[v{i}]" for i in range(2))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{SEC + 0.2:.3f}", "-i", pa + ".mp4", "-ss", f"{t0:.3f}", "-t", f"{SEC + 0.2:.3f}", "-i", pb + ".mp4",
                        "-filter_complex", fcx + ";[v0][v1]hstack=inputs=2[v]", "-map", "[v]", "-frames:v", str(NF), "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
                        f"{A.out}/{cid}.mp4"], check=True)
        for k, base in (("s1", pa), ("s2", pb)):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{SEC:.4f}", "-i", base + ".wav", "-ar", "16000", "-ac", "1", f"{A.out}/{cid}_{k}.wav"], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{A.out}/{cid}_s1.wav", "-i", f"{A.out}/{cid}_s2.wav", "-filter_complex", "amix=inputs=2:normalize=0", f"{A.out}/{cid}.wav"], check=True)


    def record(cid, conv, pa, pb, t0, s, first, face_w, crop_x):
        return dict(clip_id=cid, conv=conv, left=os.path.basename(pa), right=os.path.basename(pb), t0=round(t0, 3), switch=float(s), first=first,
                    face_w=face_w, crop_x=crop_x, bbox={"person1": [0, 0, H, W2], "person2": [0, W2, H, 2 * W2]},
                    text_left=words(pa, t0, t0 + SEC), text_right=words(pb, t0, t0 + SEC))


    def find_source(rel):
        """Locate a Seamless file id (relative path without extension) under any root; fall back to a search by file id."""
        for r in ROOTS:
            if os.path.exists(os.path.join(r, rel) + ".json"): return os.path.join(r, rel)
        for r in ROOTS:
            hit = glob.glob(f"{r}/**/{os.path.basename(rel)}.json", recursive=True)
            if hit: return hit[0][:-5]
        return None


    items = []
    if not A.search:
        for c in json.load(open(A.clip_list)):
            if A.limit and len(items) >= A.limit: break
            pa, pb = find_source(c["left_source"]), find_source(c["right_source"])
            if pa is None or pb is None: print(f"  missing source for {c['clip_id']}", flush=True); continue
            cut(c["clip_id"], pa, pb, c["t0"], c["crop_x"])
            items.append(record(c["clip_id"], c["conversation"], pa, pb, c["t0"], c["switch"], c["first"], c["face_w"], c["crop_x"]))
            print(f"  {c['clip_id']}", flush=True)
        json.dump(items, open(f"{A.out}/clips_manifest.json", "w"), ensure_ascii=False, indent=1)
        print(f"done: {len(items)} clips -> {A.out}")
        raise SystemExit(0)

    # ---------------- search mode ----------------
    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
    g = collections.defaultdict(list)
    SKIP = {x["conv"] for x in json.load(open(A.skip_convs_from))} if A.skip_convs_from else set()
    for f in sorted(f for r in ROOTS for f in glob.glob(f"{r}/*/train/*/*/*.json")):
        m = re.match(r"(V\d+_S\d+_I\d+)_(P\w+)\.json", os.path.basename(f))
        if m: g[m.group(1)].append(f[:-5])
    pairs = sorted((k, sorted(v, key=os.path.basename)) for k, v in g.items() if len(v) == 2 and k not in SKIP)
    print(f"{len(pairs)} conversations with both participants ({len(SKIP)} skipped)", flush=True)


    def frame(mp4, t):
        cap = cv2.VideoCapture(mp4); cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read(); cap.release()
        if not ok: return None
        s = H / f.shape[0]; return cv2.resize(f, (int(round(f.shape[1] * s)), H), interpolation=cv2.INTER_LANCZOS4)


    def face(mp4, t):
        f = frame(mp4, t)
        if f is None: return None
        fs = [z for z in fa.get(f) if z.det_score > 0.6]
        if len(fs) != 1: return None
        b = fs[0].bbox; return (b[0] + b[2]) / 2, b[2] - b[0], f.shape[1]


    rej = collections.Counter()
    for conv, (pa, pb) in pairs:
        if A.limit and len(items) >= A.limit: break
        D = min(dur(pa + ".wav"), dur(pb + ".wav"))   # window range from the two wav files
        if min(D, dur(pa + ".mp4"), dur(pb + ".mp4")) < SEC + 1.0: rej["empty or too short (conversation skipped)"] += 1; print(f"  skip {conv}: duration {D:.1f}s", flush=True); continue
        ma, mb = vadmask(pa, D), vadmask(pb, D)
        got, t0 = 0, 0.5
        while t0 + SEC < D - 0.5 and got < A.max_per_conv:
            best = None
            for s in np.arange(1.0, 3.01, 0.25):
                for first, (mx, my) in (("left", (ma, mb)), ("right", (mb, ma))):
                    i0, i1, i2, i3 = int(t0 * 100), int((t0 + s) * 100), int((t0 + s + 0.1) * 100), int((t0 + SEC) * 100)
                    if mx[i0:i1].mean() >= 0.6 and my[i0:i1].mean() <= 0.15 and my[i2:i3].mean() >= 0.6 and mx[i2:i3].mean() <= 0.15:
                        best = (s, first); break
                if best: break
            if not best: t0 += 0.5; continue
            s, first = best
            fc = []
            for base in (pa, pb):
                ff = [face(base + ".mp4", t0 + dt) for dt in (0.1, SEC / 2, SEC - 0.1)]
                if any(x is None for x in ff): fc = None; rej["not exactly one face"] += 1; break
                cx = [x[0] for x in ff]; fw = ff[1][1]; Wfull = ff[1][2]
                if not (40 <= fw <= 260) or max(cx) - min(cx) > fw: fc = None; rej["face width or drift out of range"] += 1; break
                x0 = int(np.clip(ff[1][0] - W2 / 2, 0, Wfull - W2))
                if not all(x0 + 5 <= c - fw / 2 and c + fw / 2 <= x0 + W2 - 5 for c in cx): fc = None; rej["crop does not contain the face"] += 1; break
                fc.append((x0, fw))
            if not fc: t0 += 1.0; continue
            cid = f"sbs_{conv}_{t0:.1f}".replace(".", "p")
            cut(cid, pa, pb, t0, [fc[0][0], fc[1][0]])
            items.append(record(cid, conv, pa, pb, t0, s, first, [round(float(fc[0][1]), 1), round(float(fc[1][1]), 1)], [fc[0][0], fc[1][0]]))
            print(f"  {cid} first={first} switch {s:.2f}s face width {items[-1]['face_w']}", flush=True)
            got += 1; t0 += SEC + 1.0
            json.dump(items, open(f"{A.out}/clips_manifest.partial.json", "w"), ensure_ascii=False, indent=1)
            if A.limit and len(items) >= A.limit: break
    json.dump(items, open(f"{A.out}/clips_manifest.json", "w"), ensure_ascii=False, indent=1)
    print(f"done: {len(items)} clips -> {A.out}; rejected windows: {dict(rej)}")

MODULES['prepare_clips'] = module_prepare_clips

# ====================================================================================================
# wvaq/extract_landmarks.py —— 第 2 步：嘴部关键点（59 行）
# ====================================================================================================
def module_extract_landmarks():
    #!/usr/bin/env python3
    """Step 2: per-frame mouth landmarks of both people in every clip.

    Frames are sampled on the 12 fps grid (frame k = t = k/12 s; 49 frames). On each frame, insightface
    (buffalo_l, detection only) keeps the two largest faces and orders them left to right; each face is
    cropped with a margin of 0.6 x face width and passed to MediaPipe FaceMesh (refined landmarks).
    Saved per clip (<id>.npz):
      mar   (2,T)        inner-lip distance / mouth-corner distance
      lips  (2,T,40,2)   the 40 FaceMesh lip landmarks, full-frame pixel coordinates
      nose  (2,T,2)      nose tip
      facew (2,T)        detection box width
    Processing time per clip is written to _times.json.

    Usage: python -m wvaq.extract_landmarks [--clips DIR] [--out DIR] [--frames 49]
    """
    import argparse, os, json, time, glob, numpy as np, cv2
    import mediapipe as mp
    from insightface.app import FaceAnalysis
    from wvaq import config

    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=config.CLIPS_DIR)
    ap.add_argument("--out", default=config.LANDMARK_DIR)
    ap.add_argument("--frames", type=int, default=49)
    A = ap.parse_args(); OUT = A.out; os.makedirs(OUT, exist_ok=True)
    LIPS = sorted({i for e in mp.solutions.face_mesh.FACEMESH_LIPS for i in e})          # 40 points
    FA = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"]); FA.prepare(ctx_id=-1, det_size=(640, 640))
    FM1 = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3)
    times = json.load(open(f"{OUT}/_times.json")) if os.path.exists(f"{OUT}/_times.json") else {}
    cids = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{A.clips}/*.mp4"))
    for c in cids:
        p = config.clip_paths(c, A.clips)["video"]; NT = A.frames
        o = f"{OUT}/{c}.npz"
        if os.path.exists(o): continue
        t0 = time.time(); cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS) or 12.0; fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fr.append(f)
        idx = [min(int(round(k / 12.0 * fps)), len(fr) - 1) for k in range(NT)]
        mar = np.full((2, NT), np.nan, np.float32); lips = np.full((2, NT, len(LIPS), 2), np.nan, np.float32)
        nose = np.full((2, NT, 2), np.nan, np.float32); fw = np.full((2, NT), np.nan, np.float32)
        for k, i in enumerate(idx):
            f = fr[i]; H, W = f.shape[:2]
            fs = sorted([z for z in FA.get(f) if z.det_score > 0.5], key=lambda z: -(z.bbox[2] - z.bbox[0]))[:2]
            if len(fs) != 2: continue
            for pi, z in enumerate(sorted(fs, key=lambda z: (z.bbox[0] + z.bbox[2]) / 2)):
                x0, y0, x1, y1 = z.bbox; m = 0.6 * (x1 - x0)
                a, b, c2, d = int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(W, x1 + m)), int(min(H, y1 + m))
                cr = f[b:d, a:c2]; r = FM1.process(cv2.cvtColor(cr, cv2.COLOR_BGR2RGB)); fw[pi, k] = x1 - x0
                if not r.multi_face_landmarks: continue
                L = r.multi_face_landmarks[0]; h, w = cr.shape[:2]
                P = lambda j: np.array([L.landmark[j].x * w + a, L.landmark[j].y * h + b])
                mar[pi, k] = np.linalg.norm(P(13) - P(14)) / max(np.linalg.norm(P(78) - P(308)), 1e-3)
                lips[pi, k] = np.stack([P(j) for j in LIPS]); nose[pi, k] = P(1)
        np.savez_compressed(o, mar=mar, lips=lips, nose=nose, facew=fw, fps=fps, n_src_frames=len(fr), wh=np.array([fr[0].shape[1], fr[0].shape[0]]))
        times[c] = round(time.time() - t0, 2); json.dump(times, open(f"{OUT}/_times.json", "w"))
        print(f"{c:36s} fps {fps:.0f} both faces measured {int(np.isfinite(mar).all(0).sum())}/{NT}  {times[c]}s", flush=True)
    print("DONE")

MODULES['extract_landmarks'] = module_extract_landmarks

# ====================================================================================================
# wvaq/syncnet_score.py —— SyncNet 打分（155 行）
# ====================================================================================================
def module_syncnet_score():
    #!/usr/bin/env python3
    """SyncNet lip-sync scoring (Sync-C confidence / Sync-D minimum distance / audio-video offset).

    Follows the official syncnet_python (MIT, joonson):
      - video at 25 fps, audio at 16 kHz (the network was trained at these rates)
      - face crop geometry as in run_pipeline.crop_video: box size = max(w,h)/2, centre and size median-filtered
        (window 13), crop_scale = 0.4, padding value 110, resized to 224
      - scoring as in SyncNetInstance.evaluate: 5-frame lip windows x 20-frame MFCC windows, offsets searched in
        +-vshift, conf = median(dist) - min(dist)
    Differences from the official code: faces are detected with insightface (buffalo_l) instead of S3FD, and the
    crops are not re-encoded with XVID. Values are therefore not bit-identical to the official tool; --selftest runs
    the official example clip plus two negative controls.

    Usage:
      python -m wvaq.syncnet_score --video clip.mp4 [--audio speech.wav]
      python -m wvaq.syncnet_score --selftest
    """
    import argparse, os, subprocess, sys, tempfile
    import cv2, numpy as np, torch
    from scipy import signal
    import python_speech_features
    from wvaq import config

    SN = config.SYNCNET_DIR
    sys.path.insert(0, SN)
    from SyncNetModel import S

    _M = {}
    DEV = os.environ.get("SN_DEVICE", "cpu")   # SN_DEVICE=cuda puts the network on the GPU (face detection stays on CPU)


    def model():
        if "m" not in _M:
            m = S(num_layers_in_fc_layers=1024)
            sd = torch.load(config.SYNCNET_WEIGHTS, map_location="cpu")
            own = m.state_dict()
            # the official weights were saved by an old PyTorch without BatchNorm num_batches_tracked (the official
            # loadParameters also only copies the keys present in the file) - only check that real parameters are present
            miss = [k for k in own if k not in sd and not k.endswith("num_batches_tracked")]
            assert not miss, f"SyncNet weights miss keys {miss[:3]} (incomplete file? expected 54,573,114 bytes)"
            m.load_state_dict({k: sd[k] for k in own if k in sd}, strict=False); m.eval(); _M["m"] = m.to(DEV)
        return _M["m"]


    _F = {}


    def detector():
        if "f" not in _F:
            from insightface.app import FaceAnalysis
            if os.environ.get("ORT_THREADS"):   # fixed onnxruntime thread count (useful in CPU-limited containers); unchanged when unset
                import onnxruntime as ort; _IS = ort.InferenceSession
                def _is(*a, **k):
                    so = k.get("sess_options") or ort.SessionOptions(); so.intra_op_num_threads = so.inter_op_num_threads = int(os.environ["ORT_THREADS"]); k["sess_options"] = so; return _IS(*a, **k)
                ort.InferenceSession = _is
            fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"])
            fa.prepare(ctx_id=-1, det_size=(640, 640)); _F["f"] = fa
        return _F["f"]


    def read_25fps(video, t0=None, dur=None):
        """Video (any frame rate) -> 25 fps frames + 16 kHz mono audio."""
        with tempfile.TemporaryDirectory() as td:
            ss = ["-ss", f"{t0:.3f}"] if t0 is not None else []
            tt = ["-t", f"{dur:.3f}"] if dur is not None else []
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *ss, "-i", video, *tt, "-vf", "fps=25", "-an",
                            os.path.join(td, "%06d.png")], check=True)
            r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *ss, "-i", video, *tt, "-vn", "-ac", "1", "-ar", "16000",
                                "-f", "s16le", "-"], capture_output=True)
            fr = [cv2.imread(os.path.join(td, f)) for f in sorted(os.listdir(td))]
        return fr, np.frombuffer(r.stdout, np.int16)


    def crops(frames, crop_scale=0.4, pick="largest"):
        """As run_pipeline.crop_video. Frames without a face reuse the nearest valid box; returns None if more than 30 %
        of the frames have no face. pick: largest (single person) / left / right (two people: the two largest faces,
        split by x centre)."""
        fa = detector(); boxes = []
        for f in frames:
            fs = sorted(fa.get(f), key=lambda z: -(z.bbox[2] - z.bbox[0]) * (z.bbox[3] - z.bbox[1]))
            if pick in ("left", "right"):
                two = sorted(fs[:2], key=lambda z: (z.bbox[0] + z.bbox[2]) / 2)
                boxes.append(None if len(two) < 2 else two[0 if pick == "left" else 1].bbox)
            else:
                boxes.append(None if not fs else fs[0].bbox)
        ok = [i for i, b in enumerate(boxes) if b is not None]
        if len(ok) < 0.7 * len(frames): return None
        boxes = [b if b is not None else boxes[min(ok, key=lambda j: abs(j - i))] for i, b in enumerate(boxes)]
        s = signal.medfilt([max(b[3] - b[1], b[2] - b[0]) / 2 for b in boxes], kernel_size=13)
        y = signal.medfilt([(b[1] + b[3]) / 2 for b in boxes], kernel_size=13)
        x = signal.medfilt([(b[0] + b[2]) / 2 for b in boxes], kernel_size=13)
        out = []
        for i, f in enumerate(frames):
            bs = s[i]; bsi = int(bs * (1 + 2 * crop_scale))
            fp = np.pad(f, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
            my, mx = y[i] + bsi, x[i] + bsi
            face = fp[int(my - bs):int(my + bs * (1 + 2 * crop_scale)), int(mx - bs * (1 + crop_scale)):int(mx + bs * (1 + crop_scale))]
            out.append(cv2.resize(face, (224, 224)))
        return out


    def calc_pdist(feat1, feat2, vshift=15):
        win = vshift * 2 + 1
        f2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
        return [torch.nn.functional.pairwise_distance(feat1[[i], :].repeat(win, 1), f2p[i:i + win, :]) for i in range(len(feat1))]


    @torch.no_grad()
    def score(frames25, audio16k, vshift=15):
        """dict(offset in frames, conf=Sync-C, min_dist=Sync-D, n_frames) or None (not enough faces)."""
        return score_crops(crops(frames25), audio16k, vshift)


    @torch.no_grad()
    def score_crops(cr, audio16k, vshift=15, keep=False):
        """Same as score, on already cropped faces (one detection pass can be scored against several audio tracks).
        keep=True also returns the offset x time distance matrix dmat (n_offsets x n_times, float16), used to recompute
        time-restricted variants such as Sync-C* (speaking periods only)."""
        if cr is None: return None
        im = torch.from_numpy(np.stack(cr, axis=3)[None].transpose(0, 3, 4, 1, 2).astype(np.float32))   # (1,3,T,224,224) BGR 0..255, as the official code
        mf = np.stack([np.array(i) for i in zip(*python_speech_features.mfcc(audio16k, 16000))])
        cc = torch.from_numpy(mf[None, None].astype(np.float32))
        n = min(len(cr), len(audio16k) // 640) - 5
        if n < 2 * vshift + 1: vshift = max(1, n // 3)
        m = model(); im_f, cc_f = [], []
        for i in range(0, n, 20):
            idx = range(i, min(n, i + 20))
            im_f.append(m.forward_lip(torch.cat([im[:, :, v:v + 5] for v in idx], 0).to(DEV)).cpu())
            cc_f.append(m.forward_aud(torch.cat([cc[:, :, :, v * 4:v * 4 + 20] for v in idx], 0).to(DEV)).cpu())
        full = torch.stack(calc_pdist(torch.cat(im_f), torch.cat(cc_f), vshift), 1); d = full.mean(1)
        mn, mi = torch.min(d, 0)
        out = dict(offset=int(vshift - mi), conf=float(torch.median(d) - mn), min_dist=float(mn), n_frames=len(cr))
        if keep: out["dmat"] = full.numpy().astype(np.float16); out["vshift"] = vshift
        return out


    if __name__ == "__main__":
        ap = argparse.ArgumentParser()
        ap.add_argument("--video"); ap.add_argument("--audio"); ap.add_argument("--t0", type=float); ap.add_argument("--dur", type=float)
        ap.add_argument("--selftest", action="store_true")
        a = ap.parse_args()
        if a.selftest:
            ex = os.path.join(SN, "data", "example.avi")
            if not os.path.exists(ex):
                subprocess.run(["curl", "-sL", "-m", "120", "-o", ex, "http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/example.avi"], check=True)
            fr, au = read_25fps(ex)
            r = score(fr, au); print(f"official example (official README: offset 3, conf ~10.0): {r}")
            r2 = score(fr, np.roll(au, 16000)); print(f"negative control 1, audio shifted by 1 s: {r2}")
            r3 = score(fr, np.zeros_like(au) + (np.random.RandomState(0).randn(len(au)) * 300).astype(np.int16)); print(f"negative control 2, white noise: {r3}")
            sys.exit(0)
        fr, au = read_25fps(a.video, a.t0, a.dur)
        if a.audio:
            r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", a.audio, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True)
            au = np.frombuffer(r.stdout, np.int16)
        print(score(fr, au))

MODULES['syncnet_score'] = module_syncnet_score

# ====================================================================================================
# wvaq/syncnet_batch.py —— 第 5 步：分人 SyncNet + 距离矩阵（88 行）
# ====================================================================================================
def module_syncnet_batch():
    #!/usr/bin/env python3
    """Step 5: batch SyncNet on two-person clips (CPU).

    For every manifest item ({"key", "video", "s1", "s2", "mix"}) the following scores are written:
      per_left / per_right   per-person use: left face x s1 track, right face x s2 track (offset, conf, min_dist);
                             the offset x time distance matrices are saved to <dmat>/<key>.npz (L, R) for Sync-C*
      std_L / std_R, face_w  conventional use with a single face track: mixed audio x left face track and x right
                             face track; the conventional score is the side with the larger median face width
      std (--with-largest)   conventional use with the per-frame largest face (may switch between the two people)
    Faces are detected once per frame and shared by all crops; cropping and scoring reuse syncnet_score.py.
    Note: clips are 12 fps; reading at 25 fps (as the network expects) repeats frames, so values are comparable only
    within this protocol.

    Results are appended line by line to the .jsonl output (a re-run skips finished keys; failed items are retried).
    Usage: python -m wvaq.syncnet_batch [--manifest FILE] [--out FILE.jsonl] [--dmat DIR] [--workers 2] [--slice i/n]
    """
    import argparse, json, os, subprocess
    from multiprocessing import Pool
    import numpy as np
    from wvaq import config

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(config.VARIANT_DIR, "manifest.json"))
    ap.add_argument("--out", default=config.SYNCNET_RESULTS)
    ap.add_argument("--dmat", default=config.DMAT_DIR)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--slice", default="", help="i/n: only process items with index %% n == i")
    ap.add_argument("--with-largest", action="store_true")
    A = ap.parse_args()


    def pcm16k(p):
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", p, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True)
        return np.frombuffer(r.stdout, np.int16)


    def one(it):
        import torch; torch.set_num_threads(3)
        from wvaq import syncnet_score as S
        fr, mix = S.read_25fps(it["video"])
        if it.get("mix"): mix = pcm16k(it["mix"])          # the clips are silent; the mixed track is given as a wav
        fa = S.detector(); dets = [sorted(fa.get(f), key=lambda z: -(z.bbox[2] - z.bbox[0]) * (z.bbox[3] - z.bbox[1])) for f in fr]

        class Cached:                      # lets S.crops reuse the detections above
            def __init__(s): s.i = 0
            def get(s, f): r = dets[s.i]; s.i += 1; return r

        def crop(pick):
            S._F["f"] = Cached(); c = S.crops(fr, pick=pick); S._F["f"] = fa; return c
        out = {"key": it["key"], "n25": len(fr)}
        per = [S.score_crops(crop(side), pcm16k(it[s]), keep=True) for side, s in (("left", "s1"), ("right", "s2"))]
        mats = {}
        for nm, r in zip(("L", "R"), per):
            if r: mats[nm] = r.pop("dmat"); r.pop("vshift", None)
        np.savez_compressed(os.path.join(A.dmat, it["key"].replace("|", "__").replace("@", "_") + ".npz"), **mats)
        out["per_left"], out["per_right"] = per
        # conventional use with one face track for the whole clip: mixed audio x left track and x right track
        out["std_L"], out["std_R"] = (S.score_crops(crop(side), mix) for side in ("left", "right"))
        two = [sorted(d[:2], key=lambda q: (q.bbox[0] + q.bbox[2]) / 2) for d in dets]; two = [t for t in two if len(t) == 2]
        out["face_w"] = {"L": float(np.median([t[0].bbox[2] - t[0].bbox[0] for t in two])) if two else None, "R": float(np.median([t[1].bbox[2] - t[1].bbox[0] for t in two])) if two else None}
        if A.with_largest: out["std"] = S.score_crops(crop("largest"), mix)
        return out


    def one_safe(it):
        # one broken item (e.g. truncated to a few frames) must not stop the batch: record an error line, treated as missing
        try: return one(it)
        except Exception as e: return {"key": it["key"], "error": f"{type(e).__name__}: {str(e)[:200]}"}


    if __name__ == "__main__":
        os.makedirs(A.dmat, exist_ok=True); os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
        items = json.load(open(A.manifest)); res = {}
        if os.path.exists(A.out):
            for l in open(A.out):
                try:
                    r = json.loads(l)
                    if "error" not in r: res[r["key"]] = r
                except Exception: pass
        sl = [int(x) for x in A.slice.split("/")] if A.slice else None
        todo = [it for k, it in enumerate(items) if it["key"] not in res and (sl is None or k % sl[1] == sl[0])]
        print(f"{len(items)} items, {len(res)} done before, {len(todo)} to do", flush=True)
        with Pool(A.workers) as p, open(A.out, "a") as fo:
            for r in p.imap_unordered(one_safe, todo):
                res[r["key"]] = r; fo.write(json.dumps(r) + "\n"); fo.flush()
                if "error" in r: print(f"{r['key']:40s} skipped: {r['error']}", flush=True); continue
                s = r.get("per_left")
                print(f"{r['key']:40s} per_left conf {'-' if s is None else round(s['conf'], 2)}", flush=True)

MODULES['syncnet_batch'] = module_syncnet_batch

# ====================================================================================================
# wvaq/__init__.py —— 包标记（1 行）
# ====================================================================================================
def module_init():
    """WVAQ: auditing lip-sync metrics for two-person video by fault injection."""

MODULES['__init__'] = module_init


# ====================================================================================================
# 下面是装载与运行的部分（不是原代码）
# ====================================================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
import inspect, textwrap

def _src(name):
    """取出 module_<名>() 的函数体 = 原文件原文（去掉 def 行和统一的 4 格缩进）。"""
    lines = inspect.getsource(MODULES[name]).splitlines(True)
    return textwrap.dedent(''.join(lines[1:]))

class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """让 `import wvaq.<模块>` 读本文件里的原文，而不是磁盘上的 wvaq/ 目录。"""
    def find_spec(self, name, path=None, target=None):
        if name == 'wvaq':
            return importlib.util.spec_from_loader(name, self, origin=os.path.join(HERE, 'wvaq', '__init__.py'), is_package=True)
        if name.startswith('wvaq.') and name[5:] in MODULES:
            return importlib.util.spec_from_loader(name, self, origin=os.path.join(HERE, 'wvaq', name[5:] + '.py'))
        return None
    def create_module(self, spec): return None
    def exec_module(self, mod):
        short = '__init__' if mod.__name__ == 'wvaq' else mod.__name__[5:]
        mod.__file__ = os.path.join(HERE, 'wvaq', short + '.py')
        if mod.__name__ == 'wvaq': mod.__path__ = []
        exec(compile(_src(short), mod.__file__, 'exec'), mod.__dict__)

sys.meta_path.insert(0, _Finder())



# ====================================================================================================
# 新增（不是论文原代码）：路径参数、自动准备 SyncNet、check / --demo / --inject
# New code (not part of the 12 original modules above): paths from the command line, SyncNet auto-setup,
# the stand-alone "check" mode, the downloadable demo clip and fault injection. The module bodies above are
# not modified; everything below only calls them.
# ====================================================================================================
USAGE = """\
WVAQ / SLT two-person lip-sync check  ·  双人口型检查（论文《Who Is Talking?》）

Check your own two-person video  检查自己的双人视频:
  python 口型论文复现.py check --video talk.mp4 --audio-left left.wav --audio-right right.wav [--out report_dir]
  python 口型论文复现.py check --video talk.mp4 --audio mix.wav          (only one mixed track: timing only / 只有混音：只能看音画错位)
  python 口型论文复现.py check --video talk.mp4                          (uses the video's own sound as the mixed track)

Demo (downloads one paper clip, about 100 MB, from Seamless Interaction, CC BY-NC 4.0)  演示:
  python 口型论文复现.py check --demo                       untouched clip: expected no alarm  原片：应不报警
  python 口型论文复现.py check --demo --inject swap         voices swapped: expected alarm (L and/or S)  两人声音对调：应报警
  python 口型论文复现.py check --demo --inject shift@8      audio 667 ms late: expected T alarm  声音推后 8 帧：T 报警
  python 口型论文复现.py check --demo --inject lis_talk_t@1.0   listener mouths along: expected L alarm  听者跟着动嘴：L 报警
  --inject works on any input that has two per-person tracks.  --inject 也可用在自己的视频上（要有两条分人音轨）
  Faults: {faults}

Reproduce the paper (8 steps; paths are options, no environment variables)  复现论文:
  python 口型论文复现.py --list
  python 口型论文复现.py <step> [step options] [--data-root DIR] [--work-dir DIR] [--syncnet-dir DIR]
  defaults: --data-root ./data/seamless_interaction  --work-dir ./wvaq_work  --syncnet-dir ./third_party/syncnet_python
  (all relative to this file)

Input rules 输入要求: two people visible, roughly left and right; per-person tracks = each person's own microphone.
The video is converted to 12 fps and cut into consecutive 49-frame windows (4.08 s) starting at frame 0, as in the paper.
"""

INJECT_FAULTS = ["swap", "shift@2", "shift@8", "vlead@2", "vlead@8", "lis_talk_t@1.0", "lis_talk_t@0.33", "spk_mute_t@1.0",
                 "wrong_spk_t@1.0", "lis_quiet_t@1.0", "rev_spk_t@1.0", "frz_lis_t@1.0", "one_lag_t@8", "voice_drop_t@1.0", "rot_spk_t@1.0"]
AUDIO_BASE_FAULTS = ("swap", "shift@2", "shift@8")          # rendered by make_variants' base set

# ---- frozen SLT lines of the paper (design set, 100 untouched clips / 27 conversations; results/slt.json) ----
SLT_Q = 98.98989898990294
SLT_LINES = (-1.8262606534073285, 0.875407364660035, 7.500000000001954)      # S, L, T
REF_UNTOUCHED = [  # (S, L, T) of the 100 untouched SyncNet-set clips (paper design set; per-clip values behind slt.json)
    (-3.362990, 0.252071, 2.5), (-7.457928, 0.201658, 2.0), (-3.940569, 0.802476, 4.5), (-4.387085, 0.189917, 1.0),
    (-2.591627, 0.336892, 0.5), (-5.708206, 0.312922, 0.5), (-5.405873, 0.494447, 1.0), (-4.840204, 0.333982, 1.0),
    (-4.921917, 0.350415, 1.5), (-5.408691, 0.221612, 1.0), (-5.922067, 0.253443, 1.0), (-4.615723, 0.236963, 1.0),
    (-5.782114, 0.258850, 1.0), (-5.391701, 0.246378, 4.0), (-4.738163, 0.381678, 4.0), (-2.228261, 0.134983, 0.0),
    (-3.137733, 0.428780, 0.0), (-6.174240, 0.698802, 0.0), (-4.573873, 0.228819, 0.0), (-4.460720, 0.193643, 1.5),
    (-3.727004, 0.444909, 1.0), (-4.722733, 0.441407, 1.0), (-4.493627, 0.702229, 1.5), (-4.098475, 0.176999, 1.0),
    (-4.129862, 0.101693, 1.0), (-3.675884, 0.237985, 1.5), (-4.082976, 0.050592, 0.5), (-2.369862, 0.007163, 0.5),
    (-5.366898, 0.100106, 0.5), (-5.645638, 0.662211, 5.0), (-5.417374, 0.293042, 0.0), (-7.520151, 0.301504, 0.0),
    (-6.537654, 0.088302, 0.5), (-2.006577, 0.351815, 0.5), (-6.006583, 0.372263, 3.5), (-7.159967, 0.121381, 3.0),
    (-6.548460, 0.229295, 3.0), (-5.130222, 0.059251, 3.5), (-5.912760, 0.227585, 3.0), (-6.693115, 0.058827, 1.5),
    (-5.629785, 0.247142, 0.5), (-3.392425, 0.947870, 3.5), (-5.727256, 0.291951, 0.5), (-5.206359, 0.109698, 3.0),
    (-6.184939, 0.379568, 3.0), (-4.031545, 0.188504, 3.0), (-4.835824, 0.040916, 3.0), (-4.473906, 0.087417, 3.0),
    (-5.225317, 0.183643, 1.0), (-1.375115, 0.373512, 1.5), (-3.667387, 0.427122, 1.0), (-6.521413, 0.062244, 0.0),
    (-5.820660, 0.361371, 8.0), (-5.812240, 0.259277, 1.0), (-5.110677, 0.208345, 0.5), (-6.265755, 0.875407, 0.5),
    (-6.904616, 0.315411, 0.5), (-7.220785, 0.325463, 0.5), (-1.826261, 0.584504, 1.0), (-3.026774, 0.288381, 1.5),
    (-7.989014, 0.156583, 2.0), (-3.387814, 0.031393, 0.5), (-4.534162, 0.072625, 1.0), (-5.264336, 0.212678, 0.5),
    (-2.802700, 0.042475, 1.0), (-6.588584, 0.157599, 1.0), (-5.450871, 0.266327, 1.5), (-7.137891, 0.076561, 1.5),
    (-4.424902, 0.233700, 1.5), (-6.426079, 0.404046, 1.5), (-5.105261, 0.116699, 2.0), (-7.914149, 0.098177, 0.5),
    (-7.792019, 0.119529, 1.0), (-3.124118, 0.306237, 0.5), (-4.130371, 0.059860, 0.5), (-6.400936, 0.154149, 0.5),
    (-5.435571, 0.480195, 1.0), (-4.515643, 0.795228, 0.0), (-5.613777, 0.213343, 0.5), (-5.830156, 0.241732, 0.5),
    (-8.121373, 0.145883, 7.5), (-5.657837, 0.081807, 1.0), (-4.682478, 0.016585, 1.5), (-3.024197, 0.473802, 0.5),
    (-4.259577, 0.516985, 1.0), (-5.057106, 0.076589, 1.5), (-5.075721, 0.602517, 2.5), (-6.546444, 0.192676, 2.5),
    (-5.998236, 0.465320, 2.5), (-5.236213, 0.363785, 3.0), (-3.135028, 0.247598, 3.0), (-5.776404, 0.443778, 7.0),
    (-3.952615, 0.239490, 6.0), (-5.053546, 0.297727, 6.5), (-5.798615, 0.202650, 6.0), (-5.068875, 0.409579, 6.0),
    (-6.885276, 0.214444, 0.5), (-4.772786, 0.090516, 1.5), (-6.050824, 0.137865, 0.5), (-6.327745, 0.083568, 0.5),
]

CAVEAT_EN = ("The lines were set on split-screen Seamless Interaction conversations (faces about 100 px wide, own-mic tracks, "
             "12 fps, 4.08 s windows). On other sources (same-shot footage, other resolutions, generated video) they must be "
             "recalibrated: see the paper, Sec. 'Discussion and Limitations' (Layout, resolution and source affect calibration) "
             "and the supplement 'How many clean clips recalibration needs' (about 40 or more clean clips from several conversations; "
             "real-clip lines flagged 44% of unperturbed generated clips).")
CAVEAT_ZH = ("报警线是在 Seamless 分屏真人对话上定的（脸宽约 100 像素、每人自己的麦、12 fps、4.08 秒一段）。换别的来源"
             "（同画面拍摄、别的分辨率、生成视频）要重新定线：见论文「Discussion and Limitations」中「Layout, resolution and source affect "
             "calibration」一段与补充材料「How many clean clips recalibration needs」（约 40 条以上、来自多段对话的干净片；"
             "真人片的线会把 44% 未加改动的生成片判成报警）。")
LICENCE_NOTE = ("Licences: SyncNet code (joonson/syncnet_python) is MIT; the SyncNet weights (syncnet_v2.model, VGG Oxford) come "
                "without a stated licence - treat them as research use only. insightface buffalo_l models: non-commercial research "
                "use only. MediaPipe: Apache-2.0. Seamless Interaction (demo clip): CC BY-NC 4.0, non-commercial.  "
                "许可：SyncNet 代码 MIT；SyncNet 权重没写许可，只当研究用；insightface 模型只许非商业研究；演示片 CC BY-NC 4.0 不许商用。")
HINT = {
    "S": ("说话人的嘴和自己的声音对不上：可能声音分给了错的人、说话人没动嘴，或口型不同步",
          "speaker's mouth does not follow his/her own voice: voice given to the wrong person, speaker not moving, or poor sync"),
    "L": ("听者在对方说话时跟着动嘴（或两人声音对调）",
          "listener moves the mouth while the other person speaks (or the two voices are swapped)"),
    "T": ("音画整体错位：声音和画面有时间偏移",
          "audio-video offset: the sound is shifted in time against the picture"),
}
PIP_REQ = ("numpy==1.26.4 scipy==1.17.1 opencv-contrib-python==4.11.0.86 insightface==1.0.1 onnxruntime==1.29.0 "
           "mediapipe==0.10.21 torch==2.10.0 python_speech_features==0.6")
SYNCNET_GIT = "https://github.com/joonson/syncnet_python"
SYNCNET_WEIGHT_URL = "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model"   # as its download_model.sh
SYNCNET_WEIGHT_BYTES = 54573114

# the demo clip: one untouched clip of the paper's 100-clip SyncNet set with typical S/L/T (none of the three near a line)
# the demo clip: one untouched clip of the paper's 100-clip SyncNet set with typical S/L/T (none of the three near a line),
# record copied from clips/clips.json so that this file works on its own
DEMO_CLIP = {'clip_id': 'sbs_V00_S0175_I00000128_31p0', 'conversation': 'V00_S0175_I00000128', 'left': 'V00_S0175_I00000128_P0008', 'right': 'V00_S0175_I00000128_P0250', 'left_source': 'naturalistic/train/0003/0001/V00_S0175_I00000128_P0008', 'right_source': 'naturalistic/train/0003/0001/V00_S0175_I00000128_P0250', 't0': 31.0, 't_end': 35.083, 'switch': 2.75, 'first': 'right', 'face_w': [115.4, 95.7], 'crop_x': [201, 202], 'syncnet100': True}
DEMO_PAPER_SLT = {  # the paper's stored (S, L, T) for this clip, untouched and per design fault (None = not measurable)
    'none': (-5.408691, 0.221612, 1.0),
    'swap': (-2.838721, 4.512393, 10.5),
    'wrong_spk_t@1.0': (-2.130371, 3.338319, 5.0),
    'lis_talk_t@1.0': (-5.322998, 1.290243, 1.0),
    'lis_talk_t@0.33': (-5.516211, 0.638797, 1.0),
    'spk_mute_t@1.0': (-2.006445, 0.573389, 7.5),
    'lis_quiet_t@1.0': (-5.289111, 0.328155, None),
    'shift@2': (-5.220892, 0.180428, 5.0),
    'shift@8': (-1.682007, 0.34263, 12.0),
    'vlead@2': (-4.855957, 0.18757, 5.5),
    'vlead@8': (-2.933301, 0.346386, 12.0),
}
HF_DATASET = "https://huggingface.co/datasets/facebook/seamless-interaction/resolve/main/"


def _say(*a): print(*a, flush=True)


def _pop_opt(argv, name):
    """Remove `--name value` / `--name=value` from argv wherever it is; return the value or None."""
    out = None; i = 0
    while i < len(argv):
        if argv[i] == name and i + 1 < len(argv): out = argv[i + 1]; del argv[i:i + 2]; continue
        if argv[i].startswith(name + "="): out = argv[i].split("=", 1)[1]; del argv[i]; continue
        i += 1
    return out


def _need(mods):
    """Clear message instead of a traceback when a package or ffmpeg is missing."""
    import shutil
    miss = []
    for m, pipname in mods:
        try: __import__(m)
        except ImportError: miss.append(pipname)
    tools = [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
    if miss or tools:
        if miss:
            _say(f"Missing Python packages / 缺少 Python 包: {', '.join(miss)}  (Python {sys.version.split()[0]} at {sys.executable})")
            _say("Install (tested with Python 3.12, CPU) / 安装：\n"
                 f"  python3.12 -m venv venv && . venv/bin/activate && pip install {PIP_REQ}\n"
                 "  (or: pip install -r requirements.txt)  then run this file with that Python.")
        if tools: _say(f"Missing on PATH / PATH 上没有: {', '.join(tools)}  -> install ffmpeg (e.g. apt install ffmpeg)")
        sys.exit(3)


ALL_PKGS = [("numpy", "numpy"), ("scipy", "scipy"), ("cv2", "opencv-contrib-python"), ("onnxruntime", "onnxruntime"),
            ("insightface", "insightface"), ("mediapipe", "mediapipe"), ("torch", "torch"), ("python_speech_features", "python_speech_features")]


def _download(url, dst, expect=None):
    import urllib.request, shutil
    os.makedirs(os.path.dirname(dst), exist_ok=True); tmp = dst + ".part"
    _say(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f: shutil.copyfileobj(r, f, 1 << 20)
    if expect and os.path.getsize(tmp) != expect:
        raise SystemExit(f"download of {url}: got {os.path.getsize(tmp)} bytes, expected {expect}")
    os.replace(tmp, dst)


def _ensure_syncnet():
    """SyncNet code + weights into ./third_party/syncnet_python on first use (same source as its download_model.sh)."""
    import subprocess, shutil
    d = SYNCNET_DIR if os.path.isabs(SYNCNET_DIR) else os.path.join(HERE, SYNCNET_DIR)
    w = os.path.join(d, "data", "syncnet_v2.model")
    first = False
    if not os.path.exists(os.path.join(d, "SyncNetModel.py")):
        first = True; _say(f"First use: fetching SyncNet code {SYNCNET_GIT} -> {d}")
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if shutil.which("git"):
            subprocess.run(["git", "clone", "--depth", "1", SYNCNET_GIT, d], check=True)
        else:   # no git: GitHub zip archive
            import zipfile, io, urllib.request
            z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(SYNCNET_GIT.replace("github.com", "codeload.github.com") + "/zip/refs/heads/master", timeout=120).read()))
            tmp = d + ".unzip"; z.extractall(tmp); os.replace(os.path.join(tmp, z.namelist()[0].split("/")[0]), d); shutil.rmtree(tmp, ignore_errors=True)
    if not os.path.exists(w) or os.path.getsize(w) != SYNCNET_WEIGHT_BYTES:
        first = True; _say(f"First use: fetching SyncNet weights -> {w}")
        _download(SYNCNET_WEIGHT_URL, w, SYNCNET_WEIGHT_BYTES)
    if first:
        lic = os.path.join(d, "LICENSE.md")
        if os.path.exists(lic): _say("SyncNet licence (code) / SyncNet 代码许可:\n  " + "\n  ".join(open(lic).read().strip().splitlines()[:3]) + "\n  ...")
        _say("  " + LICENCE_NOTE)


def _run_module_main(name, argv):
    """Run one original module as `python -m wvaq.<name> argv...` inside this process (fresh namespace every call)."""
    import wvaq  # noqa: F401
    old = sys.argv; sys.argv = [f"wvaq.{name}"] + list(argv)
    try:
        g = {"__name__": "__main__", "__file__": os.path.join(HERE, "wvaq", name + ".py"), "__package__": "wvaq"}
        exec(compile(_src(name), g["__file__"], "exec"), g)
    except SystemExit as e:
        if e.code not in (0, None): raise
    finally: sys.argv = old


# ---------------------------------------- input preparation ----------------------------------------
def _probe(p):
    import subprocess, json as _j
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", p], capture_output=True, text=True)
    if r.returncode: raise SystemExit(f"cannot read {p}: {r.stderr.strip()[:300]}")
    return _j.loads(r.stdout)


def _pcm(p):
    """Any audio (or the sound of a video) -> 16 kHz mono int16, as syncnet_batch.pcm16k."""
    import subprocess, numpy as np
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", p, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.int16)


def _wav(p, x):
    import wave
    w = wave.open(p, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(x.astype("<i2").tobytes()); w.close()


def _prepare(video, a_left, a_right, a_mix, wd, max_windows=0):
    """12 fps lossless copy of the video, consecutive 49-frame windows from frame 0 (w000.mp4, ...) and the matching
    16 kHz tracks w000_s1.wav (left person), w000_s2.wav (right), w000.wav (mix). Lossless (x264 qp 0, yuv420p), so a
    12 fps yuv420p input keeps its exact pixels, and a window that starts at frame 0 of a paper clip is that clip."""
    import subprocess, numpy as np
    os.makedirs(wd, exist_ok=True)
    info = _probe(video); vs = [s for s in info["streams"] if s.get("codec_type") == "video"]
    if not vs: raise SystemExit(f"{video}: no video stream")
    W, H = int(vs[0]["width"]), int(vs[0]["height"]); fr = vs[0].get("r_frame_rate", "0/1")
    fps_in = float(fr.split("/")[0]) / max(float(fr.split("/")[1]), 1)
    os.makedirs(os.path.join(wd, "src"), exist_ok=True); full = os.path.join(wd, "src", "full12.mp4")   # not *.mp4 in wd (extract_landmarks globs wd)
    vf = "fps=12" + (",scale=trunc(iw/2)*2:trunc(ih/2)*2" if (W % 2 or H % 2) else "")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", video, "-vf", vf, "-an", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", full], check=True)
    n12 = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", full],
                             capture_output=True, text=True).stdout.strip() or 0)
    if n12 < 25: raise SystemExit(f"{video}: only {n12} frames at 12 fps (need at least 25, ideally 49 = 4.08 s)")
    nw, L = (n12 // 49, 49) if n12 >= 49 else (1, n12)
    if max_windows: nw = min(nw, max_windows)
    tracks = {}
    if a_left: tracks["s1"], tracks["s2"] = _pcm(a_left), _pcm(a_right)
    else:
        mx = _pcm(a_mix if a_mix else video)
        if len(mx) == 0: raise SystemExit("no audio: give --audio-left/--audio-right, or --audio, or a video with sound")
        tracks["mix"] = mx
    if "s1" in tracks:
        n = max(len(tracks["s1"]), len(tracks["s2"]))
        x1, x2 = (np.r_[t, np.zeros(n - len(t), np.int16)] for t in (tracks["s1"], tracks["s2"]))
        tracks["mix"] = np.clip(x1.astype(np.int32) + x2, -32768, 32767)   # as make_variants (sample-wise sum)
    dur_a = min(len(t) for t in tracks.values()) / 16000
    if dur_a < n12 / 12 - 0.5: _say(f"  note: audio ({dur_a:.2f} s) is shorter than the video ({n12/12:.2f} s); missing audio counts as silence")
    wins = []
    for k in range(nw):
        s = k * L; cid = f"w{k:03d}"; out = os.path.join(wd, cid + ".mp4")
        ss = ["-ss", f"{(s - 0.5) / 12:.6f}"] if s else []     # frame-accurate: first frame with pts >= (s-0.5)/12 is frame s
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *ss, "-i", full, "-frames:v", str(L), "-an", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", out], check=True)
        a0, a1 = int(round(s * 16000 / 12)), int(round((s + L) * 16000 / 12))
        for nm, suf in (("s1", "_s1"), ("s2", "_s2"), ("mix", "")):
            if nm in tracks:
                x = tracks[nm][a0:a1]; _wav(os.path.join(wd, cid + suf + ".wav"), np.r_[x, np.zeros(max(0, a1 - a0 - len(x)), np.int16)])
        wins.append({"id": cid, "start_frame": s, "frames": L, "t_start_s": round(s / 12, 3), "t_end_s": round((s + L) / 12, 3)})
    return {"fps_in": round(fps_in, 3), "size": [W, H], "frames_12fps": n12, "windows": wins, "per_person": "s1" in tracks,
            "dropped_tail_frames": n12 - nw * L}


# ---------------------------------------- scoring ----------------------------------------
def _score_windows(prep, wd):
    """Per window: landmarks (extract_landmarks) -> Bind (landmark_metrics.score_clip); per-person SyncNet with distance
    matrices (syncnet_batch.one) -> Sync-C* (sync_cstar.star) and |offset| (slt.off). Same functions as the paper."""
    import numpy as np, json as _j
    lm, dm = os.path.join(wd, "landmarks"), os.path.join(wd, "dmat"); os.makedirs(dm, exist_ok=True)
    L = prep["windows"][0]["frames"]
    _say(f"[1/2] mouth landmarks of both people (insightface + MediaPipe), {len(prep['windows'])} window(s)")
    _run_module_main("extract_landmarks", ["--clips", wd, "--out", lm, "--frames", str(L)])
    from wvaq.landmark_metrics import score_clip
    from wvaq.sync_cstar import star
    from wvaq.slt import off
    from wvaq.faults import env12, voice_activity
    import wvaq
    old = sys.argv; sys.argv = ["wvaq.syncnet_batch", "--dmat", dm]
    try:
        import importlib; sb = importlib.import_module("wvaq.syncnet_batch")
    finally: sys.argv = old
    sb.A.dmat = dm
    _say(f"[2/2] SyncNet per person (CPU)")
    out = []
    for w in prep["windows"]:
        c = w["id"]; P = {"video": os.path.join(wd, c + ".mp4"), "mix": os.path.join(wd, c + ".wav")}
        r = dict(w)
        npz = os.path.join(lm, c + ".npz"); z = np.load(npz); r["both_faces_frames"] = int(np.isfinite(z["mar"]).all(0).sum())
        if prep["per_person"]:
            P["s1"], P["s2"] = os.path.join(wd, c + "_s1.wav"), os.path.join(wd, c + "_s2.wav")
            e1, e2 = env12(P["s1"], L), env12(P["s2"], L); a1, a2 = voice_activity(e1, e2)
            r["voice_frames"] = {"left": int(a1.sum()), "right": int(a2.sum()), "left_only": int((a1 & ~a2).sum()), "right_only": int((a2 & ~a1).sum())}
            lmr = score_clip(npz, P["s1"], P["s2"], vars_=[])
            o = lmr["orig"] if lmr else None
            r["Spk"], r["Lis"] = (o["Spk"], o["Lis"]) if o else (None, None)
            Lv = o["Bind"] if o else float("nan")
        else:
            P["s1"] = P["s2"] = P["mix"]; Lv = float("nan")
        it = {"key": f"{c}|none", **P}
        sn = sb.one_safe(it)
        if "error" in sn: _say(f"  {c}: SyncNet failed: {sn['error']}"); sn = {}
        r["syncnet"] = {k: sn.get(k) for k in ("per_left", "per_right", "std_L", "std_R", "face_w")}
        if prep["per_person"]:
            cs = star(it, dm); Sv = cs["negC"] if cs else float("nan"); Tv = off(sn)
            r.update(S=Sv, L=Lv, T=Tv)
        else:
            fw = sn.get("face_w") or {}; side = "L" if (fw.get("L") or 0) >= (fw.get("R") or 0) else "R"
            conv = sn.get(f"std_{side}")
            r.update(S=None, L=None, T_mix=off(sn), mixed_syncnet={"face": "left" if side == "L" else "right", **(conv or {})})
        out.append(r)
    return out


def _pct(x, i):
    import numpy as np
    ref = np.array([v[i] for v in REF_UNTOUCHED], float)
    return float(100 * np.mean(ref <= x))


def _decide(rows, per_person):
    import math
    for r in rows:
        if per_person:
            r["percentile"], r["alarm"] = {}, {}
            for i, k in enumerate("SLT"):
                x = r.get(k)
                ok = x is not None and isinstance(x, float) and math.isfinite(x)
                r["percentile"][k] = round(_pct(x, i), 1) if ok else None
                r["alarm"][k] = bool(ok and x > SLT_LINES[i])
                if not ok: r[k] = None
            r["alarm_any"] = any(r["alarm"].values())
            ps = [p for p in r["percentile"].values() if p is not None]
            r["slt_score_pct"] = max(ps) if ps else None
            r["hints"] = [{"part": k, "zh": HINT[k][0], "en": HINT[k][1]} for k in "SLT" if r["alarm"][k]]
            r["measurable"] = {k: r[k] is not None for k in "SLT"}
        else:
            x = r.get("T_mix")
            r["T_mix_percentile_vs_per_person_T"] = round(_pct(x, 2), 1) if x is not None and math.isfinite(x) else None
            r["T_mix_over_T_line_indicative"] = bool(x is not None and math.isfinite(x) and x > SLT_LINES[2])
    return rows


def _fmt(v, nd=3):
    return "  -  " if v is None else f"{v:.{nd}f}"


def _report(rows, prep, args, outdir, extra=None):
    import json as _j, numpy as np
    per = prep["per_person"]
    rep = {"tool": "WVAQ SLT check (口型论文复现.py check)", "input": {"video": os.path.abspath(args.video),
           "audio_left": args.audio_left and os.path.abspath(args.audio_left), "audio_right": args.audio_right and os.path.abspath(args.audio_right),
           "audio_mix": args.audio and os.path.abspath(args.audio)}, **(extra or {}),
           "mode": "per_person_tracks" if per else "mixed_track_only",
           "video": {k: prep[k] for k in ("fps_in", "size", "frames_12fps", "dropped_tail_frames")},
           "lines": {"S": SLT_LINES[0], "L": SLT_LINES[1], "T": SLT_LINES[2], "common_quantile_q": SLT_Q,
                     "set_on": "100 untouched split-screen Seamless Interaction clips (27 conversations); combined false alarm 3 % there"},
           "parts": {"S": "speaker = -Sync-C* (per person, own speaking periods, own track)", "L": "listener = Bind (listener / speaker mouth motion)",
                     "T": "timing = mean per-person |SyncNet offset| (25 fps frames)"},
           "windows": rows, "caveat_en": CAVEAT_EN, "caveat_zh": CAVEAT_ZH, "licences": LICENCE_NOTE}
    clean = lambda o: (None if isinstance(o, float) and not np.isfinite(o) else o)
    def walk(o):
        if isinstance(o, dict): return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list): return [walk(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): o = o.item()
        return clean(o)
    rep = walk(rep)
    _j.dump(rep, open(os.path.join(outdir, "report.json"), "w"), ensure_ascii=False, indent=1)
    T = ["WVAQ two-person lip-sync check (SLT)  ·  双人口型检查",
         f"video 视频: {rep['input']['video']}  ({prep['size'][0]}x{prep['size'][1]}, {prep['fps_in']} fps -> 12 fps, {prep['frames_12fps']} frames)"]
    if extra and extra.get("injected_fault"): T.append(f"injected fault 人为注入的错误: {extra['injected_fault']}  (source: {extra.get('injected_from')})")
    if extra and extra.get("paper_stored_SLT_for_this_clip"):
        ps = extra["paper_stored_SLT_for_this_clip"]
        T.append(f"demo clip 演示片 {extra['demo_clip']}: the paper's stored S, L, T for '{ps['fault']}' 论文存的值 = {ps['S_L_T']}"
                 + ("" if ps["S_L_T"] else " (not in the paper's design set 论文没测这一种)"))
    if per: T.append(f"tracks 音轨: left 左 = {rep['input']['audio_left']} | right 右 = {rep['input']['audio_right']}")
    else: T.append("tracks 音轨: one mixed track only / 只有一条混音 -> S and L need per-person tracks; only the offset is shown  S、L 需要分人音轨，这里只给音画错位")
    T.append(f"lines 报警线 (q={SLT_Q:.2f}): S > {SLT_LINES[0]:.3f}  L > {SLT_LINES[1]:.3f}  T > {SLT_LINES[2]:.1f}   (pct = percentile among 100 untouched paper clips 在 100 条论文原片中的百分位)")
    if prep["dropped_tail_frames"]: T.append(f"note: last {prep['dropped_tail_frames']} frames (< one 49-frame window) not checked  最后不足一段的帧没查")
    for r in rows:
        T.append("")
        T.append(f"window 第 {int(r['id'][1:]) + 1} 段  {r['t_start_s']:.2f}-{r['t_end_s']:.2f} s   both faces found 两张脸都找到: {r['both_faces_frames']}/{r['frames']} frames")
        if per:
            vf = r.get("voice_frames", {})
            T.append(f"  speaking frames 说话帧 (12 fps): left 左 {vf.get('left')} (alone 独说 {vf.get('left_only')}), right 右 {vf.get('right')} (alone {vf.get('right_only')})")
            for k, nm in (("S", "speaker  说话人 -Sync-C*"), ("L", "listener 听者   Bind   "), ("T", "timing   音画   |offset|")):
                x = r[k]; p = r["percentile"][k]
                st = "not measurable 量不到" if x is None else ("ALARM 报警" if r["alarm"][k] else "ok 正常")
                T.append(f"  {k} {nm} {_fmt(x, 3 if k != 'T' else 1):>8s}   pct {('-' if p is None else f'{p:5.1f}'):>5s}   {st}")
            if r["alarm_any"]:
                T.append("  => ALARM 报警: " + "; ".join(f"[{h['part']}] {h['zh']} / {h['en']}" for h in r["hints"]))
            else:
                T.append("  => no alarm 未报警" + ("" if all(r["measurable"].values()) else " (some parts not measurable 有项量不到)"))
        else:
            ms = r.get("mixed_syncnet") or {}
            T.append(f"  per-face offset vs mix 每张脸对混音的偏移: left {_fmt((r['syncnet'].get('per_left') or {}).get('offset'), 0)}, right {_fmt((r['syncnet'].get('per_right') or {}).get('offset'), 0)} frames (25 fps)")
            T.append(f"  T_mix (mean |offset|) {_fmt(r['T_mix'], 1)}   pct vs per-person T {r['T_mix_percentile_vs_per_person_T']}   "
                     + ("over the T line (indicative only) 超过 T 线（仅供参考）" if r["T_mix_over_T_line_indicative"] else "below the T line 低于 T 线"))
            T.append(f"  conventional mixed-track SyncNet ({ms.get('face')} face) conf {_fmt(ms.get('conf'), 2)}, offset {ms.get('offset')}  (cannot see who speaks 看不出谁在说)")
    T += ["", "Caveat 注意: " + CAVEAT_EN, "注意：" + CAVEAT_ZH, "", LICENCE_NOTE]
    open(os.path.join(outdir, "report.txt"), "w").write("\n".join(T) + "\n")
    _say("\n" + "\n".join(T))
    _say(f"\nreport -> {os.path.join(outdir, 'report.txt')} and report.json")
    return rep


# ---------------------------------------- demo download ----------------------------------------
class _HttpFile:
    """Read-only seekable file over HTTP range requests (enough for tarfile to walk the headers of one shard)."""
    def __init__(s, url):
        import urllib.request
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=120)
        s.url, s.size, s.pos, s.buf, s.b0 = r.geturl(), int(r.headers["Content-Length"]), 0, b"", -1
    def _get(s, a, b):
        import urllib.request
        return urllib.request.urlopen(urllib.request.Request(s.url, headers={"Range": f"bytes={a}-{b - 1}"}), timeout=300).read()
    def tell(s): return s.pos
    def seek(s, off, wh=0): s.pos = off if wh == 0 else s.pos + off if wh == 1 else s.size + off; return s.pos
    def read(s, n=-1):
        n = s.size - s.pos if n is None or n < 0 else min(n, s.size - s.pos)
        if n <= 0: return b""
        if not (s.b0 <= s.pos and s.pos + n <= s.b0 + len(s.buf)):
            s.b0 = s.pos; s.buf = s._get(s.pos, min(s.size, s.pos + max(n, 1 << 16)))
        o = s.pos - s.b0; d = s.buf[o:o + n]; s.pos += len(d); return d


def _demo_fetch(root):
    """Download the two participants (mp4 + wav + json) of the demo clip from the official Hugging Face copy of
    Seamless Interaction (public, not gated; files sit inside ~1 GB tar shards, so only the needed members are read
    with HTTP range requests, about 100 MB). Returns the clip record for prepare_clips."""
    import tarfile, json as _j, csv
    rec = dict(DEMO_CLIP)
    need = []
    for rel in (rec["left_source"], rec["right_source"]):
        for ext in (".json", ".wav", ".mp4"):
            p = os.path.join(root, rel + ext)
            if not (os.path.exists(p) and os.path.getsize(p) > 0): need.append((rel, ext))
    if need:
        _say("Demo data: Seamless Interaction (Meta), https://huggingface.co/datasets/facebook/seamless-interaction\n"
             "  licence CC BY-NC 4.0 (attribution, non-commercial). Downloaded to your disk for this demo only; this repository does not\n"
             "  redistribute it.  演示数据来自 Meta 的 Seamless Interaction，CC BY-NC 4.0（署名、不许商用），只下载到你本机，本仓不分发。")
        shards = sorted({os.path.dirname(rel) + ".tar" for rel, _ in need})
        for sh in shards:
            want = {os.path.basename(rel) + ext: os.path.join(root, rel + ext) for rel, ext in need if os.path.dirname(rel) + ".tar" == sh}
            _say(f"  reading tar index of {sh} (range requests) ...")
            f = _HttpFile(HF_DATASET + sh); tf = tarfile.open(fileobj=f, mode="r:")
            for m in tf:
                if m.name in want:
                    dst = want.pop(m.name); os.makedirs(os.path.dirname(dst), exist_ok=True)
                    _say(f"  {m.name}  {m.size / 1e6:.1f} MB")
                    with open(dst + ".part", "wb") as o:
                        a = m.offset_data
                        while a < m.offset_data + m.size:
                            b = min(m.offset_data + m.size, a + (8 << 20)); o.write(f._get(a, b)); a = b
                    os.replace(dst + ".part", dst)
                if not want: break
            if want: raise SystemExit(f"not found in {sh}: {sorted(want)}")
    return rec


# ---------------------------------------- fault injection ----------------------------------------
def _inject(prep, wd, fault, outdir):
    """Apply one paper fault to every window with the original make_variants (it imports faults.py), then join the faulted
    windows into one clip in <out>/injected_<fault>/ (video.mp4, left.wav, right.wav). Returns the three paths."""
    import json as _j, subprocess, shutil, numpy as np, wave
    if not prep["per_person"]: raise SystemExit("--inject needs two per-person tracks (--audio-left / --audio-right)")
    lm = os.path.join(wd, "landmarks")
    _say(f"[inject] landmarks of the input (make_variants needs the voice/turn layout) ...")
    _run_module_main("extract_landmarks", ["--clips", wd, "--out", lm, "--frames", str(prep["windows"][0]["frames"])])
    vd = os.path.join(wd, "variants"); lst = os.path.join(wd, "clip_list.txt")
    open(lst, "w").write("\n".join(w["id"] for w in prep["windows"]) + "\n")
    base = fault in AUDIO_BASE_FAULTS
    _run_module_main("make_variants", ["--landmarks", lm, "--clips", wd, "--out", vd, "--clip-list", lst, "--vars", "" if base else fault] + ([] if base else ["--no-base"]))
    man = {it["key"]: it for it in _j.load(open(os.path.join(vd, "manifest.json")))}
    items = [man[f"{w['id']}|{fault}"] for w in prep["windows"]]
    od = os.path.join(outdir, "injected_" + fault.replace("@", "_")); os.makedirs(od, exist_ok=True)
    paths = {"video": os.path.join(od, "video.mp4"), "left": os.path.join(od, "left.wav"), "right": os.path.join(od, "right.wav")}
    if len(items) == 1:
        shutil.copyfile(items[0]["video"], paths["video"])
    else:
        cat = os.path.join(wd, "concat.txt"); open(cat, "w").write("".join(f"file '{os.path.abspath(it['video'])}'\n" for it in items))
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", cat, "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", paths["video"]], check=True)
    for nm, key in (("left", "s1"), ("right", "s2")):
        xs = []
        for it in items:
            w = wave.open(it[key]); xs.append(np.frombuffer(w.readframes(w.getnframes()), np.int16))
        _wav(paths[nm], np.concatenate(xs))
    _say(f"[inject] {fault}: faulted clip -> {od}")
    return paths


# ---------------------------------------- check main ----------------------------------------
def _check_main(argv):
    import argparse, tempfile, shutil, time
    ap = argparse.ArgumentParser(prog="口型论文复现.py check", description="SLT check of one two-person video (see the usage text: python 口型论文复现.py)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="faults for --inject: " + ", ".join(INJECT_FAULTS) +
                                 "\nexpected on the demo: untouched -> no alarm; swap -> L and/or S; shift@8 -> T; lis_talk_t@1.0 -> L")
    ap.add_argument("--video"); ap.add_argument("--audio-left"); ap.add_argument("--audio-right")
    ap.add_argument("--audio", help="one mixed track (when per-person tracks are not available)")
    ap.add_argument("--out", help="report directory (default ./wvaq_check/<video name>)")
    ap.add_argument("--demo", action="store_true", help="download one paper clip (~100 MB, Seamless Interaction, CC BY-NC 4.0) and check it")
    ap.add_argument("--demo-dir", default=os.path.join(HERE, "wvaq_demo"))
    ap.add_argument("--inject", choices=INJECT_FAULTS, help="apply one of the paper's faults to the input before checking")
    ap.add_argument("--max-windows", type=int, default=0, help="check only the first N windows (0 = all)")
    ap.add_argument("--keep-work", action="store_true", help="keep intermediate files (<out>/_work)")
    A = ap.parse_args(argv)
    _need(ALL_PKGS)
    extra = {}
    if A.demo:
        root = os.path.join(A.demo_dir, "seamless"); rec = _demo_fetch(root)
        cdir = os.path.join(A.demo_dir, "clip"); cid = rec["clip_id"]
        if not os.path.exists(os.path.join(cdir, cid + "_s2.wav")):
            _say(f"cutting the paper clip {cid} (t0 {rec['t0']} s, crop {rec['crop_x']}) with prepare_clips ...")
            import json as _j
            os.makedirs(cdir, exist_ok=True); cl = os.path.join(cdir, "clip_record.json"); _j.dump([rec], open(cl, "w"))
            _run_module_main("prepare_clips", ["--roots", root, "--out", cdir, "--clip-list", cl])
        A.video, A.audio_left, A.audio_right = (os.path.join(cdir, cid + s) for s in (".mp4", "_s1.wav", "_s2.wav"))
        extra = {"demo_clip": cid, "paper_stored_SLT_for_this_clip": {"fault": A.inject or "none", "S_L_T": DEMO_PAPER_SLT.get(A.inject or "none")}}
        A.out = A.out or os.path.join(HERE, "wvaq_check", "demo" + ("_" + A.inject.replace("@", "_") if A.inject else ""))
    if not A.video: ap.print_help(); _say("\n" + USAGE.format(faults=", ".join(INJECT_FAULTS))); sys.exit(2)
    if bool(A.audio_left) != bool(A.audio_right): raise SystemExit("give both --audio-left and --audio-right (or only --audio)")
    for p in (A.video, A.audio_left, A.audio_right, A.audio):
        if p and not os.path.exists(p): raise SystemExit(f"not found: {p}")
    _ensure_syncnet()
    out = os.path.abspath(A.out or os.path.join("wvaq_check", os.path.splitext(os.path.basename(A.video))[0] + ("_" + A.inject.replace("@", "_") if A.inject else "")))
    os.makedirs(out, exist_ok=True); wd = os.path.join(out, "_work")
    if os.path.exists(wd): shutil.rmtree(wd)
    t0 = time.time()
    _say(f"preparing input (12 fps, 49-frame windows) -> {wd}")
    prep = _prepare(A.video, A.audio_left, A.audio_right, A.audio, wd, A.max_windows)
    if A.inject:
        p = _inject(prep, wd, A.inject, out)
        extra.update(injected_fault=A.inject, injected_from=os.path.abspath(A.video))
        A.video, A.audio_left, A.audio_right, A.audio = p["video"], p["left"], p["right"], None
        shutil.rmtree(wd); prep = _prepare(A.video, A.audio_left, A.audio_right, None, wd, A.max_windows)
    rows = _decide(_score_windows(prep, wd), prep["per_person"])
    rep = _report(rows, prep, A, out, extra)
    if not A.keep_work: shutil.rmtree(wd, ignore_errors=True)
    _say(f"done in {time.time() - t0:.0f} s")
    return rep


def _use_local_venv():
    """当前 Python 缺包时，若本文件旁边有 .venv（python -m venv .venv 建的，装好 requirements.txt），自动换它重跑。"""
    for p in (os.path.join(HERE, ".venv", "bin", "python"), os.path.join(HERE, ".venv", "Scripts", "python.exe")):
        if os.path.exists(p) and os.path.realpath(sys.executable) != os.path.realpath(p):
            try:
                import numpy, cv2, mediapipe, insightface, torch  # noqa: F401
                return
            except ImportError:
                _say(f"当前 Python 缺包，改用 {p} 运行 / switching to {p}\n")
                os.execv(p, [p, os.path.abspath(__file__)] + sys.argv[1:])


if __name__ == "__main__":
    _use_local_venv()
    # 不读命令行参数：全部照文件开头「在这里改」的设置运行
    def _p(x): return x if os.path.isabs(x) else os.path.join(HERE, x)
    if STEP == "check":
        opts = ["--video", _p(VIDEO)] if VIDEO else ["--demo"]
        if AUDIO_LEFT and AUDIO_RIGHT: opts += ["--audio-left", _p(AUDIO_LEFT), "--audio-right", _p(AUDIO_RIGHT)]
        elif AUDIO_MIX: opts += ["--audio", _p(AUDIO_MIX)]
        if INJECT: opts += ["--inject", INJECT]
        if OUT: opts += ["--out", _p(OUT)]
        _say(f"STEP=check  VIDEO={VIDEO or '演示片 demo'}  INJECT={INJECT}\n")
        _check_main(opts); sys.exit(0)
    if STEP not in STEPS:
        _say(f"STEP = {STEP!r} 不认识；可用 check 或 {', '.join(STEPS)}"); sys.exit(2)
    step = STEP
    _need({"prepare_clips": ALL_PKGS[:5], "extract_landmarks": ALL_PKGS[:6], "landmark_metrics": ALL_PKGS[:1], "make_variants": ALL_PKGS[:3],
           "syncnet_batch": ALL_PKGS, "syncnet_score": ALL_PKGS, "sync_cstar": ALL_PKGS[:1], "slt": ALL_PKGS[:1], "auc_faults": ALL_PKGS[:1]}[step])
    if step in ("syncnet_batch", "syncnet_score"): _ensure_syncnet()
    _say(f"STEP={step}\nDATA_ROOT={_p(DATA_ROOT)}\nWORK_DIR={_p(WORK_DIR)}\nSYNCNET_DIR={_p(SYNCNET_DIR)}\n")
    sys.argv = [f"wvaq.{step}"]
    import wvaq  # noqa: F401  先装好包本身
    g = {"__name__": "__main__", "__file__": os.path.join(HERE, "wvaq", step + ".py"), "__package__": "wvaq"}
    exec(compile(_src(step), g["__file__"], "exec"), g)
