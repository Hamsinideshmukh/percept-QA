# Judge probe — 0.32 h
Built from reference A, judged against reference B only. n questions = 176

| Variant | n | judged correct % | mean score | words |
|---|---|---|---|---|
| full_A | 176 | 94.9 | 0.85 | 19.6 |
| action_only | 176 | 72.7 | 0.68 | 5.5 |
| reason_only | 176 | 90.3 | 0.81 | 12.9 |
| right_action_wrong_reason | 176 | 70.5 | 0.63 | 19.4 |
| wrong_action_right_reason | 176 | 79.0 | 0.71 | 19.4 |
| swapped_motion_verb_right_reason | 23 | 69.6 | 0.59 | 21.2 |
| other_clip_answer | 176 | 46.6 | 0.46 | 19.3 |

## By class (judged correct %)
cname                             decision   why
variant                                         
full_A                                89.8  96.9
action_only                           73.5  72.4
reason_only                           89.8  90.6
right_action_wrong_reason             55.1  76.4
wrong_action_right_reason             75.5  80.3
swapped_motion_verb_right_reason     100.0  68.2
other_clip_answer                     44.9  47.2

## 8 examples

Q: Can you turn right at this intersection?
  ref B: No, because the lane markings only show left and straight as the possible options.
  [1] full_A: No, in this intersection I can only turn left
  [1] action_only: No
  [1] reason_only: in this intersection I can only turn left
  [0] right_action_wrong_reason: No because it passed a pedestrian on the left of the road. It decelerates to follow a cyclist ahead
  [1] wrong_action_right_reason: The car repositions itself to the left of the lane because in this intersection I can only turn left
  [0] other_clip_answer: The car repositions itself to the left of the lane, because it passed a pedestrian on the left of the road. It decelerates to follow a cyclist ahead.

Q: What is the current action and its justification? Answer in the form "action, justification".
  ref B: We are maintaining speed and slightly adjusting position in our lane as the road ahead is clear.
  [1] full_A: Continue straight, clear road ahead
  [0] action_only: Continue straight
  [1] reason_only: clear road ahead
  [1] right_action_wrong_reason: Continue straight because merged into its lane
  [1] wrong_action_right_reason: The car drives slowly and follows the bus in front because clear road ahead
  [1] other_clip_answer: The car drives slowly and follows the bus in front, as it merged into its lane.

Q: What is the current action and its justification? Answer in the form "action, justification".
  ref B: Stay straight and follow lane, no obstacles or traffic signals.
  [1] full_A: The car maintains its speed and stays on it lane , because the lane is free of traffic and the left lane is a bus lane
  [1] action_only: The car maintains its speed and stays on it lane
  [1] reason_only: the lane is free of traffic and the left lane is a bus lane
  [1] right_action_wrong_reason: The car maintains its speed and stays on it lane because is not safe to overtake the cyclist ahead
  [1] wrong_action_right_reason: I am maintaining my speed and following the road and the cyclist ahead because the lane is free of traffic and the left lane is a bus lane
  [1] other_clip_answer: I am maintaining my speed and following the road and the cyclist ahead, as it is not safe to overtake the cyclist ahead.

Q: What is the current action and its justification? Answer in the form "action, justification".
  ref B: Slow down slightly, zebra crossing ahead.
  [1] full_A: The car maintains its speed and its lane , as it is reaching a zebra crossing with no pedestrian
  [0] action_only: The car maintains its speed and its lane
  [1] reason_only: is reaching a zebra crossing with no pedestrian
  [0] right_action_wrong_reason: The car maintains its speed and its lane because if below the speed limit speed up to the limit and maintain speed
  [1] wrong_action_right_reason: If doing the speed limit there is no need to speed up because is reaching a zebra crossing with no pedestrian
  [0] other_clip_answer: If doing the speed limit there is no need to speed up, if below the speed limit speed up to the limit and maintain speed.

Q: Should you reduce your speed? Why?
  ref B: No. We should maintain our speed because there is a safe distance between us and the motorcycle in front,.
  [1] full_A: No , as the road is clear
  [1] action_only: No
  [1] reason_only: road is clear
  [0] right_action_wrong_reason: No because hitting the pedestrian crossing on the zebra crossing
  [0] wrong_action_right_reason: Yes. because road is clear
  [0] other_clip_answer: Yes. To avoid hitting the pedestrian crossing on the zebra crossing.

Q: What is the current action and its justification? Answer in the form "action, justification".
  ref B: Changing to the lane on the right, the lane marking says that we need to be in that lane to go straight in the upcoming intersection.
  [1] full_A: The car moves to the right lane , because the left lane is reserved for turning left at the next junction
  [1] action_only: The car moves to the right lane
  [1] reason_only: the left lane is reserved for turning left at the next junction
  [1] right_action_wrong_reason: The car moves to the right lane because the pedestrian has finished crossing
  [1] wrong_action_right_reason: I can start because the left lane is reserved for turning left at the next junction
  [0] other_clip_answer: I can start because the pedestrian has finished crossing.

Q: What is the current action and its justification? Answer in the form "action, justification".
  ref B: The car starts at the intersection, as the traffic lights turn from red & amber to green and the road is clear.
  [1] full_A: Starting from a stop, the light is changing from red+yellow to green, there are no vehicles in front and a pedestrian finished the crossing
  [1] action_only: Starting from a stop
  [1] reason_only: the light is changing from red+yellow to green, there are no vehicles in front and a pedestrian finished the crossing
  [1] right_action_wrong_reason: Starting from a stop because overtake the cyclist on the left
  [1] wrong_action_right_reason: The car is clearing the intersection while moving to the right lane because the light is changing from red+yellow to green, there are no vehicles in front and a pedestrian finished the crossing
  [1] other_clip_answer: The car is clearing the intersection while moving to the right lane, in order to overtake the cyclist on the left.

Q: What is the current action and its justification? Answer in the form "action, justification".
  ref B: The car moves to the left of the lane and follows the cyclist, as it has finished overtaking a pedestrian on the left of the road.
  [1] full_A: The car repositions itself to the left of the lane , because it passed a pedestrian on the left of the road. It decelerates to follow a cyclist ahead
  [1] action_only: The car repositions itself to the left of the lane
  [1] reason_only: it passed a pedestrian on the left of the road. It decelerates to follow a cyclist ahead
  [0] right_action_wrong_reason: The car repositions itself to the left of the lane because it is red
  [1] wrong_action_right_reason: We are stopping in front of a traffic light because it passed a pedestrian on the left of the road. It decelerates to follow a cyclist ahead
  [0] other_clip_answer: We are stopping in front of a traffic light because it is red.