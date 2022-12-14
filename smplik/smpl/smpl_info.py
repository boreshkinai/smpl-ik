# https://github.com/vchoutas/smplx/blob/master/smplx/joint_names.py
# SMPL
SMPL_JOINTS_45 = {
    'pelvis': 0,
    'left_hip': 1,
    'right_hip': 2,
    'spine1': 3,
    'left_knee': 4,
    'right_knee': 5,
    'spine2': 6,
    'left_ankle': 7,
    'right_ankle': 8,
    'spine3': 9,
    'left_foot': 10,
    'right_foot': 11,
    'neck': 12,
    'left_collar': 13,
    'right_collar': 14,
    'head': 15,
    'left_shoulder': 16,
    'right_shoulder': 17,
    'left_elbow': 18,
    'right_elbow': 19,
    'left_wrist': 20,
    'right_wrist': 21,
    'left_hand': 22,
    'right_hand': 23,
    'nose': 24,
    'right_eye': 25,
    'left_eye': 26,
    'right_ear': 27,
    'left_ear': 28,
    'left_big_toe': 29,
    'left_small_toe': 30,
    'left_heel': 31,
    'right_big_toe': 32,
    'right_small_toe': 33,
    'right_heel': 34,
    'left_thumb': 35,
    'left_index': 36,
    'left_middle': 37,
    'left_ring': 38,
    'left_pinky': 39,
    'right_thumb': 40,
    'right_index': 41,
    'right_middle': 42,
    'right_ring': 43,
    'right_pinky': 44
}
SMPL_POSES_NAMES = list(SMPL_JOINTS_45.keys())[:1+23]
SMPL_NUM_BETAS = 10
SMPL_SHAPE_NAMES = [f"Beta{i}" for i in range(SMPL_NUM_BETAS)]
SMPL_JOINT_NAMES = list(SMPL_JOINTS_45.keys())

# SMPLH
SMPLH_JOINTS_73 = {
    'pelvis': 0,
    'left_hip': 1,
    'right_hip': 2,
    'spine1': 3,
    'left_knee': 4,
    'right_knee': 5,
    'spine2': 6,
    'left_ankle': 7,
    'right_ankle': 8,
    'spine3': 9,
    'left_foot': 10,
    'right_foot': 11,
    'neck': 12,
    'left_collar': 13,
    'right_collar': 14,
    'head': 15,
    'left_shoulder': 16,
    'right_shoulder': 17,
    'left_elbow': 18,
    'right_elbow': 19,
    'left_wrist': 20,
    'right_wrist': 21,
    'left_index1': 22,
    'left_index2': 23,
    'left_index3': 24,
    'left_middle1': 25,
    'left_middle2': 26,
    'left_middle3': 27,
    'left_pinky1': 28,
    'left_pinky2': 29,
    'left_pinky3': 30,
    'left_ring1': 31,
    'left_ring2': 32,
    'left_ring3': 33,
    'left_thumb1': 34,
    'left_thumb2': 35,
    'left_thumb3': 36,
    'right_index1': 37,
    'right_index2': 38,
    'right_index3': 39,
    'right_middle1': 40,
    'right_middle2': 41,
    'right_middle3': 42,
    'right_pinky1': 43,
    'right_pinky2': 44,
    'right_pinky3': 45,
    'right_ring1': 46,
    'right_ring2': 47,
    'right_ring3': 48,
    'right_thumb1': 49,
    'right_thumb2': 50,
    'right_thumb3': 51,
    'nose': 52,
    'right_eye': 53,
    'left_eye': 54,
    'right_ear': 55,
    'left_ear': 56,
    'left_big_toe': 57,
    'left_small_toe': 58,
    'left_heel': 59,
    'right_big_toe': 60,
    'right_small_toe': 61,
    'right_heel': 62,
    'left_thumb': 63,
    'left_index': 64,
    'left_middle': 65,
    'left_ring': 66,
    'left_pinky': 67,
    'right_thumb': 68,
    'right_index': 69,
    'right_middle': 70,
    'right_ring': 71,
    'right_pinky': 72
}
SMPLH_POSES_NAMES = list(SMPLH_JOINTS_73.keys())[:1+21+2*15]
SMPLH_NUM_BETAS = 16
SMPLH_SHAPE_NAMES = [f"Beta{i}" for i in range(SMPLH_NUM_BETAS)]
SMPLH_JOINT_NAMES = list(SMPLH_JOINTS_73.keys())

# SMPLX
SMPLX_JOINTS_144 = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw',
    'left_eye_smplhf',
    'right_eye_smplhf',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
    'right_eye_brow1',
    'right_eye_brow2',
    'right_eye_brow3',
    'right_eye_brow4',
    'right_eye_brow5',
    'left_eye_brow5',
    'left_eye_brow4',
    'left_eye_brow3',
    'left_eye_brow2',
    'left_eye_brow1',
    'nose1',
    'nose2',
    'nose3',
    'nose4',
    'right_nose_2',
    'right_nose_1',
    'nose_middle',
    'left_nose_1',
    'left_nose_2',
    'right_eye1',
    'right_eye2',
    'right_eye3',
    'right_eye4',
    'right_eye5',
    'right_eye6',
    'left_eye4',
    'left_eye3',
    'left_eye2',
    'left_eye1',
    'left_eye6',
    'left_eye5',
    'right_mouth_1',
    'right_mouth_2',
    'right_mouth_3',
    'mouth_top',
    'left_mouth_3',
    'left_mouth_2',
    'left_mouth_1',
    'left_mouth_5',  # 59 in OpenPose output
    'left_mouth_4',  # 58 in OpenPose output
    'mouth_bottom',
    'right_mouth_4',
    'right_mouth_5',
    'right_lip_1',
    'right_lip_2',
    'lip_top',
    'left_lip_2',
    'left_lip_1',
    'left_lip_3',
    'lip_bottom',
    'right_lip_3',
    # Face contour
    'right_contour_1',
    'right_contour_2',
    'right_contour_3',
    'right_contour_4',
    'right_contour_5',
    'right_contour_6',
    'right_contour_7',
    'right_contour_8',
    'contour_middle',
    'left_contour_8',
    'left_contour_7',
    'left_contour_6',
    'left_contour_5',
    'left_contour_4',
    'left_contour_3',
    'left_contour_2',
    'left_contour_1',
]
SMPLX_POSES_NAMES = SMPLX_JOINTS_144[:1+21+1+2+2*15]
SMPLX_NUM_BETAS = 16
SMPLX_SHAPE_NAMES = [f"Beta{i}" for i in range(SMPLX_NUM_BETAS)]
SMPLX_JOINT_NAMES = SMPLX_JOINTS_144
