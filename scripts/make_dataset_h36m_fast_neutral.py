import argparse
import numpy as np
import pandas as pd
import json
import os
import torch
import torch.nn as nn

from pytorch3d.transforms import axis_angle_to_quaternion
from tqdm import tqdm

from smplx import SMPL


class SMPLR(nn.Module):

    def __init__(self, smpl_model_path, use_gender=False):
        super(SMPLR, self).__init__()
        self.model_m = os.path.join(smpl_model_path, 'basicModel_m_lbs_10_207_0_v1.0.0.pkl')
        self.model_f = os.path.join(smpl_model_path, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        self.model_neutral = os.path.join(smpl_model_path, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
        if use_gender:
            self.smpl_female = SMPL(model_path=self.model_f, gender='female', create_transl=False)
            self.smpl_male = SMPL(model_path=self.model_m, gender='male', create_transl=False)
            self.smpl_neutral = SMPL(model_path=self.model_neutral, gender='neutral', create_transl=False)
            self.smpls = {'f': self.smpl_female, 'm': self.smpl_male, 'n': self.smpl_neutral}
        else:
            self.smpl_neutral = SMPL(model_path=self.model_neutral, gender='neutral', create_transl=False)
            self.smpls = {'n': self.smpl_neutral}

    def forward(self, betas, global_orient, body_pose, gender='n'):
        if isinstance(body_pose, np.ndarray):
            betas, global_orient, body_pose = torch.from_numpy(betas).float(), torch.from_numpy(global_orient).float(), torch.from_numpy(body_pose).float()
        if len(body_pose.shape)==1:
            betas, global_orient, body_pose = betas.unsqueeze(0), global_orient.unsqueeze(0), body_pose.unsqueeze(0)
        outputs = self.smpls[gender](betas=betas, global_orient=global_orient, body_pose=body_pose)
        return outputs.joints[:, :len(SMPL_JOINT_NAMES)].reshape(-1, len(SMPL_JOINT_NAMES)*3)

    def forward_batch(self, betas, global_orient, body_pose, gender='n', batch_size=1024):
        outputs_batch = []
        for i in tqdm(range(0, len(body_pose), batch_size), desc=f"smpl {gender}", leave=True):
            outputs_batch.append(self.forward(betas=betas[i:i+batch_size],
                                              global_orient=global_orient[i:i+batch_size],
                                              body_pose=body_pose[i:i+batch_size],
                                              gender=gender)
                                )
        return torch.cat(outputs_batch)

    
# 1. Make dataset_settings.json
# 2. Make split.json
# 3. Make csv

# from protores.smpl.smpl_info import SMPLX_SHAPE_NAMES, SMPLX_POSE_NAMES, SMPLX_JOINT_NAMES
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
NUM_JOINTS = 24
SMPL_POSES_NAMES = list(SMPL_JOINTS_45.keys())[:NUM_JOINTS]
SMPL_NUM_BETAS = 10
SMPL_SHAPE_NAMES = [f"Beta{i}" for i in range(SMPL_NUM_BETAS)]
SMPL_JOINT_NAMES = list(SMPL_JOINTS_45.keys())[:NUM_JOINTS]


# ---------------------------------

def make_dataset_settings_json(dset_path='/home/vikram.voleti/GitHubRepos/smpl-ik/datasets/Human3.6M_smpl/'):
    # 1. Make 'dataset_settings.json'
    # In /home/vikram.voleti/GitHubRepos/smpl-ik/datasets/master_v1_raycast_fps30
    # Load protores' dataset_settings.json
    # with open('/home/vikram.voleti/GitHubRepos/smpl-ik/datasets/master_v1_raycast_fps30/dataset_settings.json', 'r') as f:
    #     a = json.load(f)
    a = {}


    # Check json
    # >>> a.keys()
    # dict_keys(['description', 'rate', 'features', 'skeleton'])

    # 1.1. Make 'description'
    # 1.2. Make 'rate'ription'
    # 1.3. Make 'features'
    # 1.4. Make 'skeleton'

    # 1.1. Make 'description'
    # >>> a['description']
    # 'Mixamo + Asset store clips'
    a['description'] = 'Human3.6M_SMPL'

    # 1.2. Make 'rate'
    # >>> a['rate']
    # 30.0
    a['rate'] = 50

    # 1.3. Make 'features'
    # >>> type(a['features'])
    # <class 'dict'>
    # >>> a['features'].keys()
    # dict_keys(['BonePositions', 'BoneRotations', 'TimeStamp', 'FootContact', 'Raycast'])

    a['features'] = {}

    # 1.3.1. Make 'BonePositions'
    # >>> type(a['features']['BonePositions'])
    # <class 'dict'>
    # >>> a['features']['BonePositions'].keys()
    # dict_keys(['settings', 'fields', 'types'])
    # >>> a['features']['BonePositions']['settings']
    # {'ReferenceFrame': 2}
    # >>> a['features']['BonePositions']['fields']
    # [{'Label': 'Hips', 'Type': 2}, {'Label': 'Spine0', 'Type': 2}, {'Label': 'Spine1', 'Type': 2}, {'Label': 'Chest', 'Type': 2}, {'Label': 'ClavicleLeft', 'Type': 2}, {'Label': 'BicepLeft', 'Type': 2}, {'Label': 'ForarmLeft', 'Type': 2}, {'Label': 'HandLeft', 'Type': 2}, {'Label': 'Index0Left', 'Type': 2}, {'Label': 'Index1Left', 'Type': 2}, {'Label': 'Index2Left', 'Type': 2}, {'Label': 'Index2LeftEnd', 'Type': 2}, {'Label': 'Middle0Left', 'Type': 2}, {'Label': 'Middle1Left', 'Type': 2}, {'Label': 'Middle2Left', 'Type': 2}, {'Label': 'Middle2LeftEnd', 'Type': 2}, {'Label': 'Pinky0Left', 'Type': 2}, {'Label': 'Pinky1Left', 'Type': 2}, {'Label': 'Pinky2Left', 'Type': 2}, {'Label': 'Pinky2LeftEnd', 'Type': 2}, {'Label': 'Ring0Left', 'Type': 2}, {'Label': 'Ring1Left', 'Type': 2}, {'Label': 'Ring2Left', 'Type': 2}, {'Label': 'Ring2LeftEnd', 'Type': 2}, {'Label': 'Thumb0Left', 'Type': 2}, {'Label': 'Thumb1Left', 'Type': 2}, {'Label': 'Thumb2Left', 'Type': 2}, {'Label': 'Thumb2LeftEnd', 'Type': 2}, {'Label': 'ClavicleRight', 'Type': 2}, {'Label': 'BicepRight', 'Type': 2}, {'Label': 'ForarmRight', 'Type': 2}, {'Label': 'HandRight', 'Type': 2}, {'Label': 'Index0Right', 'Type': 2}, {'Label': 'Index1Right', 'Type': 2}, {'Label': 'Index2Right', 'Type': 2}, {'Label': 'Index2RightEnd', 'Type': 2}, {'Label': 'Middle0Right', 'Type': 2}, {'Label': 'Middle1Right', 'Type': 2}, {'Label': 'Middle2Right', 'Type': 2}, {'Label': 'Middle2RightEnd', 'Type': 2}, {'Label': 'Pinky0Right', 'Type': 2}, {'Label': 'Pinky1Right', 'Type': 2}, {'Label': 'Pinky2Right', 'Type': 2}, {'Label': 'Pinky2RightEnd', 'Type': 2}, {'Label': 'Ring0Right', 'Type': 2}, {'Label': 'Ring1Right', 'Type': 2}, {'Label': 'Ring2Right', 'Type': 2}, {'Label': 'Ring2RightEnd', 'Type': 2}, {'Label': 'Thumb0Right', 'Type': 2}, {'Label': 'Thumb1Right', 'Type': 2}, {'Label': 'Thumb2Right', 'Type': 2}, {'Label': 'Thumb2RightEnd', 'Type': 2}, {'Label': 'Neck', 'Type': 2}, {'Label': 'Head', 'Type': 2}, {'Label': 'HeadEnd', 'Type': 2}, {'Label': 'ThighLeft', 'Type': 2}, {'Label': 'CalfLeft', 'Type': 2}, {'Label': 'FootLeft', 'Type': 2}, {'Label': 'ToeLeft', 'Type': 2}, {'Label': 'ToeLeftEnd', 'Type': 2}, {'Label': 'ThighRight', 'Type': 2}, {'Label': 'CalfRight', 'Type': 2}, {'Label': 'FootRight', 'Type': 2}, {'Label': 'ToeRight', 'Type': 2}, {'Label': 'ToeRightEnd', 'Type': 2}]
    # >>> a['features']['BonePositions']['types']
    # ['Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3', 'Vector3']
    # Replace 'Label' in 'fields':
    a['features']['BonePositions'] = {}
    a['features']['BonePositions']['settings'] = {'ReferenceFrame': 2}
    a['features']['BonePositions']['fields'] = []
    a['features']['BonePositions']['types'] = []
    for key in SMPL_JOINT_NAMES:
        a['features']['BonePositions']['fields'].append({'Label': key, 'Type': 3}) # DataTypes = {"Scalar": 1, "Vector2": 2, "Vector3": 3, "Quaternion": 4} (protores/data/augmentation/types.py)
        a['features']['BonePositions']['types'].append('Vector3')


    # 1.3.2. Make 'BoneRotations'
    # >>> type(a['features']['BoneRotations'])
    # <class 'dict'>
    # >>> a['features']['BoneRotations'].keys()
    # dict_keys(['settings', 'fields', 'types'])
    # >>> a['features']['BoneRotations']['settings']
    # {'RootJointFrame': 2, 'ChildJointsFrame': 1}
    # >>> a['features']['BoneRotations']['fields']
    # [{'Label': 'Hips', 'Type': 4}, {'Label': 'Spine0', 'Type': 4}, {'Label': 'Spine1', 'Type': 4}, {'Label': 'Chest', 'Type': 4}, {'Label': 'ClavicleLeft', 'Type': 4}, {'Label': 'BicepLeft', 'Type': 4}, {'Label': 'ForarmLeft', 'Type': 4}, {'Label': 'HandLeft', 'Type': 4}, {'Label': 'Index0Left', 'Type': 4}, {'Label': 'Index1Left', 'Type': 4}, {'Label': 'Index2Left', 'Type': 4}, {'Label': 'Index2LeftEnd', 'Type': 4}, {'Label': 'Middle0Left', 'Type': 4}, {'Label': 'Middle1Left', 'Type': 4}, {'Label': 'Middle2Left', 'Type': 4}, {'Label': 'Middle2LeftEnd', 'Type': 4}, {'Label': 'Pinky0Left', 'Type': 4}, {'Label': 'Pinky1Left', 'Type': 4}, {'Label': 'Pinky2Left', 'Type': 4}, {'Label': 'Pinky2LeftEnd', 'Type': 4}, {'Label': 'Ring0Left', 'Type': 4}, {'Label': 'Ring1Left', 'Type': 4}, {'Label': 'Ring2Left', 'Type': 4}, {'Label': 'Ring2LeftEnd', 'Type': 4}, {'Label': 'Thumb0Left', 'Type': 4}, {'Label': 'Thumb1Left', 'Type': 4}, {'Label': 'Thumb2Left', 'Type': 4}, {'Label': 'Thumb2LeftEnd', 'Type': 4}, {'Label': 'ClavicleRight', 'Type': 4}, {'Label': 'BicepRight', 'Type': 4}, {'Label': 'ForarmRight', 'Type': 4}, {'Label': 'HandRight', 'Type': 4}, {'Label': 'Index0Right', 'Type': 4}, {'Label': 'Index1Right', 'Type': 4}, {'Label': 'Index2Right', 'Type': 4}, {'Label': 'Index2RightEnd', 'Type': 4}, {'Label': 'Middle0Right', 'Type': 4}, {'Label': 'Middle1Right', 'Type': 4}, {'Label': 'Middle2Right', 'Type': 4}, {'Label': 'Middle2RightEnd', 'Type': 4}, {'Label': 'Pinky0Right', 'Type': 4}, {'Label': 'Pinky1Right', 'Type': 4}, {'Label': 'Pinky2Right', 'Type': 4}, {'Label': 'Pinky2RightEnd', 'Type': 4}, {'Label': 'Ring0Right', 'Type': 4}, {'Label': 'Ring1Right', 'Type': 4}, {'Label': 'Ring2Right', 'Type': 4}, {'Label': 'Ring2RightEnd', 'Type': 4}, {'Label': 'Thumb0Right', 'Type': 4}, {'Label': 'Thumb1Right', 'Type': 4}, {'Label': 'Thumb2Right', 'Type': 4}, {'Label': 'Thumb2RightEnd', 'Type': 4}, {'Label': 'Neck', 'Type': 4}, {'Label': 'Head', 'Type': 4}, {'Label': 'HeadEnd', 'Type': 4}, {'Label': 'ThighLeft', 'Type': 4}, {'Label': 'CalfLeft', 'Type': 4}, {'Label': 'FootLeft', 'Type': 4}, {'Label': 'ToeLeft', 'Type': 4}, {'Label': 'ToeLeftEnd', 'Type': 4}, {'Label': 'ThighRight', 'Type': 4}, {'Label': 'CalfRight', 'Type': 4}, {'Label': 'FootRight', 'Type': 4}, {'Label': 'ToeRight', 'Type': 4}, {'Label': 'ToeRightEnd', 'Type': 4}]
    # >>> a['features']['BoneRotations']['types']
    # ['Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion', 'Quaternion']
    # >>> len(a['features']['BoneRotations']['types'])
    # 65
    a['features']['BoneRotations'] = {}
    a['features']['BoneRotations']['settings'] = {'RootJointFrame': 2, 'ChildJointsFrame': 1}
    a['features']['BoneRotations']['fields'] = []
    a['features']['BoneRotations']['types'] = []
    for key in SMPL_POSES_NAMES:
        a['features']['BoneRotations']['fields'].append({'Label': key, 'Type': 4}) # DataTypes = {"Scalar": 1, "Vector2": 2, "Vector3": 3, "Quaternion": 4} (protores/data/augmentation/types.py)
        a['features']['BoneRotations']['types'].append('Quaternion')


    # 1.3.2. 'TimeStamp'
    # >>> a['features']['TimeStamp']
    # {'settings': {}, 'fields': [{'Label': 'Time', 'Type': 0}], 'types': ['Scalar']}
    # Remove 'TimeStamp'
    # b = a['features'].pop('TimeStamp')

    # 1.3.3. Make 'FootContact'
    # >>> a['features']['FootContact']
    # {'settings': {'FootThickness': 0.04, 'VelocityThreshold': 0.02, 'FloorHeight': 0.0, 'Joints': ['ToeLeft', 'ToeLeftEnd', 'ToeRight', 'ToeRightEnd']}, 'fields': [{'Label': 'ToeLeft', 'Type': 0}, {'Label': 'ToeLeftEnd', 'Type': 0}, {'Label': 'ToeRight', 'Type': 0}, {'Label': 'ToeRightEnd', 'Type': 0}], 'types': ['Scalar', 'Scalar', 'Scalar', 'Scalar']}
    # Remove 'FootContact'
    # b = a['features'].pop('FootContact')

    # 1.3.4. Remove 'Raycast'
    # b = a['features'].pop('Raycast')

    # 1.3.5. Add Betas
    a['features']['Betas'] = {}
    a['features']['Betas']['settings'] = {}
    a['features']['Betas']['fields'] = []
    a['features']['Betas']['types'] = []
    for key in SMPL_SHAPE_NAMES:
        a['features']['Betas']['fields'].append({'Label': key, 'Type': 1})
        a['features']['Betas']['types'].append('Scalar')


    # 1.3.8. Add gender
    a['features']['Gender'] = {}
    a['features']['Gender']['settings'] = {}
    a['features']['Gender']['fields'] = []
    a['features']['Gender']['types'] = []
    a['features']['Gender']['fields'].append({'Label': "Gender", 'Type': 1})
    a['features']['Gender']['types'].append('Scalar')

    # 1.3.7. Add trans
    a['features']['RootPosition'] = {}
    a['features']['RootPosition']['settings'] = {}
    a['features']['RootPosition']['fields'] = []
    a['features']['RootPosition']['types'] = []
    a['features']['RootPosition']['fields'].append({'Label': "RootPosition", 'Type': 3})
    a['features']['RootPosition']['types'].append('Vector3')

    # # 1.3.6. Add camera instrinsic parameters
    # a['features']['Cam'] = {}
    # a['features']['Cam']['settings'] = {}
    # a['features']['Cam']['fields'] = []
    # a['features']['Cam']['types'] = []
    # a['features']['Cam']['fields'].append({'Label': "Cam", 'Type': 3})
    # a['features']['Cam']['types'].append('Vector3')

    # # 1.3.8. Add image_name
    # a['features']['Image'] = {}
    # a['features']['Image']['settings'] = {}
    # a['features']['Image']['fields'] = []
    # a['features']['Image']['types'] = []
    # a['features']['Image']['fields'].append({'Label': "Image", 'Type': 0})
    # a['features']['Image']['types'].append('String')

    # 1.4. Make 'skeleton'
    # Retain skeleton as is for now
    # TODO : Change skeleton acc to betas in SMPL model

    # Save dataset_settings.json
    with open(os.path.join(dset_path, 'dataset_settings.json'), 'w') as f:
        json.dump(a, f, indent=2)


# ---------------------------------

def make_split_json(dset_path='/home/vikram.voleti/GitHubRepos/smpl-ik/datasets/Human3.6M_smpl/'):
    # 2. Make split.json
    # In /home/vikram.voleti/GitHubRepos/smpl-ik/datasets/master_v1_raycast_fps30
    # Load protores' split.json
    # with open('/home/vikram.voleti/GitHubRepos/smpl-ik/datasets/master_v1_raycast_fps30/split.json', 'r') as f:
    #     a = json.load(f)
    # Check json
    # >>> a.keys()
    # dict_keys(['training_files', 'validation_files'])
    # >>> len(a['training_files']), len(a['validation_files'])
    # 2799, 311
    # Make split.json
    a = {}
    a['training_files'] = ['csv/Human3.6M_train.csv']
    a['validation_files'] = ['csv/Human3.6M_test.csv']
    # Save split.json
    with open(os.path.join(dset_path, 'split.json'), 'w') as f:
        json.dump(a, f, indent=2)

# ---------------------------------

# 3. Make csv

def make_csv(dset_path='/home/vikram.voleti/GitHubRepos/smpl-ik/datasets/Human3.6M_smpl',
             ROMP_dset_path='/home/vikram.voleti/Datasets/H3.6M_ROMP',
             split='train',
             annots=None,
             smplr=None, 
             smpl_model_path='tools/smpl/models',
             smpl_randidx=1,
             augment_beta_std=None,
             augment_gender=False):

    subject_gender = {'S1':1, 'S5':1, 'S6':0, 'S7':1, 'S8':0, 'S9':0, 'S11':0}

    if smplr is None:
        print("Load SMPL models")
        smplr = SMPLR(smpl_model_path, use_gender=True)

    if annots is None:
        print("Load annotations")
        annots = np.load(os.path.join(ROMP_dset_path, 'annots.npz'), allow_pickle=True)['annots'][()]

    print("Load split files")
    split_imgs = pd.read_csv(os.path.join(ROMP_dset_path, f"h36m_{split}.txt"), header=None)

    # Header
    header_positions = []
    for key in SMPL_JOINT_NAMES:
        header_positions.append('_'.join(["BonePositions", key, 'X']))
        header_positions.append('_'.join(["BonePositions", key, 'Y']))
        header_positions.append('_'.join(["BonePositions", key, 'Z']))
    # BoneRotations (in Quaternions)
    header_quaternions = []
    for key in SMPL_POSES_NAMES:
        header_quaternions.append('_'.join(["BoneRotations", key, 'W']))
        header_quaternions.append('_'.join(["BoneRotations", key, 'X']))
        header_quaternions.append('_'.join(["BoneRotations", key, 'Y']))
        header_quaternions.append('_'.join(["BoneRotations", key, 'Z']))
    # Betas
    header_betas = []
    for key in SMPL_SHAPE_NAMES:
        header_betas.append('_'.join([key, 'V']))
    # Gender
    header_gender = ["Gender_V"]
    # RootPosition
    header_root_position = ["RootPosition_X", "RootPosition_Y", "RootPosition_Z"]
    # Cam
    header_cam = ["Cam_X", "Cam_Y", "Cam_Z"]

    print("Load split annotations")
    angles = []
    gender = []
    betas = []
    cam = []
    root_position = []
    for i, img in tqdm(split_imgs.iterrows(), total=len(split_imgs)):
        img_name = img[0]

        sample = annots[img_name]
        angles.append(sample['poses'][smpl_randidx])

        subject_id = img_name.split('_')[0]
        gender.append(['m','f'][subject_gender[subject_id]])

        betas.append(sample['betas'])
        
        root_position.append(sample['trans'][smpl_randidx])
        cam.append(sample['cam'][smpl_randidx])
    
    print("Get quaternions")
    angles = np.stack(angles).astype('float32')
    betas = np.stack(betas).astype('float32')
    quaternions = axis_angle_to_quaternion(torch.tensor(angles).float().reshape(-1, 3))
    quaternions = quaternions.reshape(-1, 4*24).numpy()
    
    if augment_beta_std is not None:
        betas_std = np.std(betas, axis=0, keepdims=True)
        betas_noise = augment_beta_std * np.random.normal(size=betas.shape)
        betas = betas + betas_noise
    
    print("Get SMPL joints, male")
    male_joints = smplr.forward_batch(betas=betas, 
                                      global_orient=angles[:, :3], 
                                      body_pose=angles[:, 3:], 
                                      gender='m', batch_size=10*1024)
    
    print("Get SMPL joints, female")
    female_joints = smplr.forward_batch(betas=betas, 
                                        global_orient=angles[:, :3], 
                                        body_pose=angles[:, 3:], 
                                        gender='f', batch_size=10*1024)
    print("Get SMPL joints, neutral")
    neutral_joints = smplr.forward_batch(betas=betas, 
                                        global_orient=angles[:, :3], 
                                        body_pose=angles[:, 3:], 
                                        gender='n', batch_size=10*1024)
    
    print("Build Dataframe")
    df_q = pd.DataFrame(male_joints.numpy(), columns=header_positions)
    
    df_q[header_quaternions] = quaternions
    df_q[header_betas] = betas
    df_q[header_gender] = gender
    df_q[header_root_position] = root_position
    df_q[header_cam] = cam
    # Fix the female subject joints
    if augment_gender:
        df_q[header_gender] = 0
        df_q_f = df_q.copy()
        df_q_n = df_q.copy()
        
        df_q_f[header_positions] = female_joints
        df_q_f[header_gender] = 1
        df_q = df_q.append(df_q_f)
        
        df_q_n[header_positions] = neutral_joints
        df_q_n[header_gender] = 2
        df_q = df_q.append(df_q_n)
        
        df_q[header_gender] = df_q[header_gender].astype(int)
        
        
        print("********************")
        print(df_q.Gender_V.unique())
        print("********************")
    else:
        df_q.loc[df_q.Gender_V == 'f', header_positions] = female_joints[df_q.Gender_V == 'f']
        df_q[header_gender] = (df_q[header_gender] == 'f').astype(int)
    
    print("Save csv")
    os.makedirs(os.path.join(dset_path, 'csv'), exist_ok=True)
    df_q.to_csv(os.path.join(dset_path, 'csv', f"Human3.6M_{split}.csv"), index=False)

    
def main(dset_path, ROMP_dset_path, smpl_model_path, augment_beta_std, augment_gender):
    os.makedirs(os.path.join(dset_path, 'csv'), exist_ok=True)

    # 1. Make dataset_settings.json
    make_dataset_settings_json(dset_path)

    # 2. Make split.json
    make_split_json(dset_path)

    # 3. Make csv
    print("Load annotations")
    annots = np.load(os.path.join(ROMP_dset_path, 'annots.npz'), allow_pickle=True)['annots'][()]
    print("Load SMPL models")
    smplr = SMPLR(smpl_model_path, use_gender=True)
    print("Process train split")
    make_csv(dset_path=dset_path, ROMP_dset_path=ROMP_dset_path, 
             split='train', annots=annots, smplr=smplr,
             augment_beta_std=augment_beta_std, augment_gender=augment_gender)
    print("Process test split")
    make_csv(dset_path=dset_path, ROMP_dset_path=ROMP_dset_path, 
             split='test', annots=annots, smplr=smplr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True, description="Make dataset")
    parser.add_argument('--dset_path', type=str, default="/home/vikram.voleti/GitHubRepos/smpl-ik/datasets/Human3.6M_smpl", help='Path to new dataset storage')
    parser.add_argument('--ROMP_dset_path', type=str, default="/home/vikram.voleti/Datasets/H3.6M_ROMP", help='Path to annots.npz')
    parser.add_argument('--smpl_model_path', type=str, default="tools/smpl/models", help='Path to model .pkl files')
    parser.add_argument('--augment_beta_std', type=float, default=None, help='The standard deviation of SMPL beta augmentation. No augmentation if None')
    parser.add_argument('--augment_gender', type=bool, default=False, help='Whether to augment gender')
    args = parser.parse_args()

    main(args.dset_path, args.ROMP_dset_path, args.smpl_model_path, args.augment_beta_std, args.augment_gender)

# python tools/ROMP_dataset/make_dataset_h36m_fast_neutral.py --dset_path=/workspace/smpl-ik/datasets/h36m_smpl_beta_augmentation_1_fmn_v1 --ROMP_dset_path=/home/vikram.voleti/Datasets/ROMP/h36m/ --augment_beta_std=1.0 --augment_gender=True

# BonePositions(24*3=72), BoneRotations(24*4=96), RootPosition(3), RootRotation(4), Beta(10), Gender
