import numpy as np

# 1: Background, 2: Head, 3: Torso, 4: Right arm, 5: Left arm, 6: Right forearm,
# 7: Left forearm, 8: Right hand, 9: Left hand, 10: Right thigh, 11: Left thigh
# 12: Right shank, 13: Left shank, 14: Right feet, 15: Left feet

r_chan = [0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
g_chan = [0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
b_chan = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def human_seg_combine_channel(human_seg_split_map):
    r_chan_seg = np.add.reduce(human_seg_split_map * np.array(r_chan), 2)
    g_chan_seg = np.add.reduce(human_seg_split_map * np.array(g_chan), 2)
    b_chan_seg = np.add.reduce(human_seg_split_map * np.array(b_chan), 2)
    return np.stack([b_chan_seg, g_chan_seg, r_chan_seg], axis=-1).astype(np.uint8)

def human_seg_combine_argmax(human_seg_argmax_map):
    onehot = np.stack([(human_seg_argmax_map == i).astype(np.uint8) for i in range(15)], axis=-1)
    return human_seg_combine_channel(onehot)
