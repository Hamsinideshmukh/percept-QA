# Notebook B: fine-tuning WITH cues, 9.3 h
fine-tuning: {'mode': 'cues', 'updates': 1814, 'of': 1814, 'minutes': 486.8, 'loss_first': [0.707617416381836], 'loss_last': [0.3345928955078125], 'cues_given_in_training': {'LIGHT_ZOOM': 3606, 'FAR_ZOOM': 5112, 'EGO': 5370, 'TIME': 10879}}
cue estimator gates: {'stop': 84.5, 'trend': 85.4, 'turn': 69.9}

ft_cues: official 62.4 (with 1 reference only: 55.4) | dev 62.2 | words 16.4 | errors 0
ft_cues_nocue: official 63.6 (with 1 reference only: 56.2) | dev 61.6 | words 16.5 | errors 0
ft_cues_wrongego: official 61.4 (with 1 reference only: 54.0) | dev 56.4 | words 16.6 | errors 0

(for comparison: zero-shot official 64.0 / dev 36.0; plain fine-tuning official 62.6 / dev 59.0)

cues ON vs cues OFF, all 1000: -0.3 (helped 38, hurt 41, p=0.8221)
right vs WRONG car-state sentence, the 470 questions that got one: +7.2 (helped 61, hurt 27, p=0.0004)

per question type (cues ON accuracy, and change vs cues OFF):
  why            n=357  55.7  +1.1 (helped 23, hurt 19, p=0.6440)
  decision       n=224  66.5  -1.3 (helped 8, hurt 11, p=0.6476)
  hazard         n=124  57.3  +0.0 (helped 2, hurt 2, p=1.0000)
  counting       n= 60  61.7  -1.7 (helped 1, hurt 2, p=1.0000)
  presence       n= 58  72.4  -1.7 (helped 0, hurt 1, p=1.0000)
  light_state    n= 50  68.0  +0.0 (helped 1, hurt 1, p=1.0000)
  identity       n= 40  67.5  -2.5 (helped 0, hurt 1, p=1.0000)
  where          n= 26  69.2  -11.5 (helped 0, hurt 3, p=0.2500)
  others_motion  n= 19  84.2  -5.3 (helped 0, hurt 1, p=1.0000)
  layout         n= 19  73.7  +10.5 (helped 2, hurt 0, p=0.5000)
  own_action     n= 12  83.3  +8.3 (helped 1, hurt 0, p=1.0000)
  scene_other    n= 11  54.5  +0.0 (helped 0, hurt 0, p=1.0000)