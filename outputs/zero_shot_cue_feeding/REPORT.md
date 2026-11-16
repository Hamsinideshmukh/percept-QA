# Cue feeding test (untrained Qwen3.5-2B, classical-CV cues, routed by tagger v2) — 2.9 h

EGO classifier quality on training labels (5-fold CV balanced accuracy): {'speed': np.float64(64.8), 'turn': np.float64(70.9)}  (gate: 65.0)
Cue records: {'movers_found': 212, 'light_candidates': 332, 'vp_sources': {'motion': 498, 'lanes': 42, 'default': 60}, 'errors': 0}

Each run adds ONE cue (or all) only where the tagger routes it; other questions keep the base answer. Paired exact McNemar.

| Run | Questions that got the cue | Base acc | With cue | Change | wins/losses | p | words base→cue |
|---|---|---|---|---|---|---|---|
| time | 1000 | 50.0 | 47.3 | -2.7 | 109/136 | 0.0965 | 14.4→10.3 |
| ego | 593 | 45.9 | 40.1 | -5.7 | 49/83 | 0.0039 | 16.2→24.6 |
| movers | 247 | 49.0 | 38.1 | -10.9 | 15/42 | 0.0005 | 15.8→21.1 |
| light_zoom | 45 | 68.9 | 68.9 | +0.0 | 2/2 | 1.0 | 10.6→10.2 |
| far_zoom | 129 | 61.2 | 62.0 | +0.8 | 7/6 | 1.0 | 10.7→10.3 |
| all_cues | 1000 | 50.0 | 50.8 | +0.8 | 125/117 | 0.6528 | 14.4→10.0 |

Base accuracy: official 500 = 64.0, dev = 36.0

## Per question class (n >= 10)
- all_cues on others_motion: +15.8 (n=19, 5/2, p=0.4531)
- all_cues on where: +15.4 (n=26, 5/1, p=0.2188)
- all_cues on decision: +5.8 (n=224, 38/25, p=0.1299)
- all_cues on presence: +3.4 (n=58, 6/4, p=0.7539)
- all_cues on identity: +2.5 (n=40, 6/5, p=1.0)
- all_cues on light_state: +2.0 (n=50, 3/2, p=1.0)
- all_cues on why: +0.6 (n=357, 44/42, p=0.9142)
- all_cues on hazard: -4.0 (n=124, 13/18, p=0.4731)
- all_cues on layout: -5.3 (n=19, 1/2, p=1.0)
- all_cues on counting: -11.7 (n=60, 4/11, p=0.1185)
- all_cues on scene_other: -18.2 (n=11, 0/2, p=0.5)
- all_cues on own_action: -25.0 (n=12, 0/3, p=0.25)
- ego on decision: -4.9 (n=224, 14/25, p=0.1081)
- ego on why: -5.3 (n=357, 35/54, p=0.0558)
- ego on own_action: -33.3 (n=12, 0/4, p=0.125)
- far_zoom on others_motion: +7.7 (n=13, 1/0, p=1.0)
- far_zoom on presence: +3.4 (n=58, 4/2, p=0.6875)
- far_zoom on identity: +0.0 (n=40, 2/2, p=1.0)
- far_zoom on decision: -12.5 (n=16, 0/2, p=0.5)
- light_zoom on light_state: -2.7 (n=37, 1/2, p=1.0)
- movers on decision: -5.9 (n=85, 4/9, p=0.2668)
- movers on hazard: -9.1 (n=33, 3/6, p=0.5078)
- movers on why: -17.4 (n=121, 6/27, p=0.0003)
- time on others_motion: +15.8 (n=19, 5/2, p=0.4531)
- time on where: +15.4 (n=26, 5/1, p=0.2188)
- time on presence: +1.7 (n=58, 6/5, p=1.0)
- time on decision: +0.4 (n=224, 29/28, p=1.0)
- time on identity: +0.0 (n=40, 6/6, p=1.0)
- time on light_state: -2.0 (n=50, 2/3, p=1.0)
- time on hazard: -3.2 (n=124, 13/17, p=0.5847)
- time on layout: -5.3 (n=19, 1/2, p=1.0)
- time on why: -5.3 (n=357, 38/57, p=0.0642)
- time on counting: -11.7 (n=60, 4/11, p=0.1185)
- time on own_action: -16.7 (n=12, 0/2, p=0.5)
- time on scene_other: -18.2 (n=11, 0/2, p=0.5)