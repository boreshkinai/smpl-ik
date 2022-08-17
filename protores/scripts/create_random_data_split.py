import argparse
import json
import glob
import os
import copy
import pandas as pd
import numpy as np
import torch
import random
from protores.data.dataset.typed_table import TypedColumnDataset, FlatTypedColumnDataset
from protores.geometry.skeleton import Skeleton
from protores.data.augmentation import MirrorSkeleton, RandomRotation
from protores.geometry.rotations import compute_ortho6d_from_rotation_matrix, compute_rotation_matrix_from_quaternion
from protores.geometry.quaternions import compute_quaternions_from_rotation_matrices
from protores.models.optional_lookat_model import TYPE_VALUES


def build_path(path: str, filename: str) -> str:
    return os.path.normpath(os.path.join(path, filename))


def create_split_file(dataset_path, output_path, id=0, file_extension=".csv", validation_proportion=0.1, test_proportion=0.1):
    assert 0.0 < validation_proportion < 1.0, "Validation proportion must be in the exclusive range ]0;1["
    assert 0.0 < test_proportion < 1.0, "Test proportion must be in the exclusive range ]0;1["

    json_data = {"training_files": [], "validation_files": [], "test_files": []}
    split_filename = os.path.join(output_path, 'split_{}.json'.format(id))

    all_files = [x[len(dataset_path):].strip("/").strip("\\").replace("\\", "/") for x in sorted(glob.glob(dataset_path + '/**/*' + file_extension, recursive=True))]
    all_files = [x for x in all_files if not x.startswith("Split")]
    print("Found %i (new) files" % len(all_files))

    test_count = int(round(test_proportion * len(all_files)))
    validation_count = int(round(validation_proportion * len(all_files)))
    training_count = len(all_files) - test_count - validation_count

    print("Training set will have %i files, validation %i, and test %i" % (training_count, validation_count, test_count))

    random.shuffle(all_files)

    json_data["training_files"].extend(all_files[:training_count])
    json_data["validation_files"].extend(all_files[training_count:training_count+validation_count])
    json_data["test_files"].extend(all_files[training_count+validation_count:])

    # Save info
    with open(split_filename, 'w') as outfile:
        json.dump(json_data, outfile, indent=4)

    training = [build_path(dataset_path, x) for x in json_data['training_files']]
    validation = [build_path(dataset_path, x) for x in json_data['validation_files']]
    test = [build_path(dataset_path, x) for x in json_data['test_files']]
    return {'Training': training, 'Validation': validation, 'Test': test, 'SplitFile': split_filename,
            'Settings': os.path.join(dataset_path, "dataset_settings.json")}


def load_dataframe(datapath, drop_duplicates=True):
    config_path = os.path.join(datapath, "dataset_settings.json")
    files = sorted(glob.glob(datapath + '/**/*.csv', recursive=True))
    feather_cache = os.path.join(datapath, "cache.feather")

    assert os.path.exists(config_path)
    with open(config_path, "r") as f:
        config = json.load(f)

    print("\t Loading DataFrame")

    if os.path.isfile(feather_cache) == True:
        df = pd.read_feather(feather_cache)
    else:
        all_sequences = []
        for f in files:
            sequence = pd.read_csv(f).reset_index(drop=True)
            all_sequences.append(sequence)
        df = pd.concat(all_sequences).reset_index(drop=True)
        if drop_duplicates:
            subset = list(df)
            df = df.drop_duplicates(subset=subset).reset_index(drop=True)
        df = df.astype('float32')
        df.to_feather(feather_cache)
    print("\t Loaded DataFrame")

    is_valid = True
    if len(df.index) == 0:
        is_valid = False
    assert is_valid

    return df, config


def augment_dataset(subset, skeleton):
    mirroring = MirrorSkeleton(skeleton, features=['BonePositions', 'BoneRotations'])
    fullbatch = subset.data

    # Mirror
    mirroring.init(subset, [x[0] for x in subset.features()])
    half_batch = fullbatch[:fullbatch.shape[0] // 2, :]  # Mirror first half
    comparison_batch = half_batch.detach().clone()
    while np.sum(
            comparison_batch.cpu().numpy() != half_batch.cpu().numpy()) == 0:  # Make sure we actually mirror something.
        mirroring.transform(half_batch)
    fullbatch[:fullbatch.shape[0] // 2, :] = half_batch

    # Rotate
    rotating = RandomRotation(axis=[0, 1, 0], features=['BonePositions', 'BoneRotations'])
    rotating.init(subset, [x[0] for x in subset.features()])
    rotating.transform(fullbatch)

    # New dataframe from augmented tensor
    df = pd.DataFrame(fullbatch.cpu().numpy(), index=subset.df.index, columns=subset.df.columns)

    # Re-shuffle to mix mirrored and non-mirrored data
    df = df.sample(frac=1.0, replace=False).reset_index(drop=True)
    subset.reset_internals(df)

    return subset


def save_packed_referenced_random_data(packed_inputs, frame_indices, outfile):
    # Lists of lists
    ids = packed_inputs['ids']
    types = packed_inputs['types']
    data = packed_inputs['data']

    ids = np.asarray(ids, dtype=int)
    types = np.asarray(types, dtype=int)
    for data_i in range(len(data)):
        for effector_i in range(len(data[data_i])):
            if types[data_i, effector_i] == TYPE_VALUES['position']:
                data[data_i][effector_i] = np.concatenate([data[data_i][effector_i], np.asarray([0.0, 0.0, 0.0, 0.0])])
            elif types[data_i, effector_i] == TYPE_VALUES['rotation']:
                data[data_i][effector_i] = np.concatenate([data[data_i][effector_i], np.asarray([0.0, 0.0, 0.0])])  # Rotation + Tolerance
            elif types[data_i, effector_i] == TYPE_VALUES['lookat']:
                raise ValueError("Lookat effectors not yet supported by the evaluation framework.")
            else:
                raise ValueError("Effector type {} not yet supported".format(types[data_i, effector_i]))
        data[data_i] = np.stack(data[data_i])

    data = np.stack(data)

    # Convert all quaternions into ortho 6D, in a single batch, so store indices.
    rotation_indices = (types == TYPE_VALUES['rotation']).nonzero()
    rotation_indices = np.stack([rotation_indices[0], rotation_indices[1]], axis=1)
    if rotation_indices.shape[0] > 0: # Do nothing if we don't have any rotation
        # Quats -> Matrices -> Ortho6D
        rotations = np.stack([data[idx[0], idx[1], :4] for idx in rotation_indices])
        rotations = torch.Tensor(rotations)
        rotations = compute_rotation_matrix_from_quaternion(rotations)
        rotations = compute_ortho6d_from_rotation_matrix(rotations).detach().cpu().numpy()

        # Back into that numpy array
        for i in range(len(rotation_indices)):
            data[rotation_indices[i][0], rotation_indices[i][1], :6] = rotations[i]

    # Define the output dataframe
    n_effectors = ids.shape[1]
    output_df = pd.DataFrame()
    loc = 0
    data_dims_id = ['X', 'Y', 'Z', 'A', 'B', 'C', 'T']

    # Populate df
    for eff_i in range(n_effectors):
        output_df.insert(loc, "effector{}_type".format(eff_i), types[:, eff_i])
        loc += 1
        output_df.insert(loc, "effector{}_id".format(eff_i), ids[:, eff_i])
        loc += 1
        output_df.insert(loc, "effector{}_weight".format(eff_i), np.ones_like(types[:, eff_i], dtype=np.float32))
        loc += 1
        for dim in range(data.shape[-1]):
            output_df.insert(loc, "effector{0}_data_{1}".format(eff_i, data_dims_id[dim]), data[:, eff_i, dim])
            loc += 1

    output_df['Frame'] = frame_indices
    output_df.set_index('Frame', inplace=True)

    # Save df
    output_df.to_csv(outfile, index=True)


def pick_and_remove_from_dictionary(d, key):
    val = random.choice(d[key])
    d[key].remove(val)
    if len(d[key]) == 0:
        del d[key]
    return val


def create_randomly_sampled_bonegroups_data(subset, skeleton, n_effectors):
    output_dict = {}
    frame_indices = np.asarray(list(subset.df.index))

    # Params
    bone_groups = {
        "lower_body": ["Hips", "Chest", "ThighLeft", "ThighRight"],
        "upper_body": ["Neck", "Head", "BicepLeft", "BicepRight"],
        "left_arm": ["ForarmLeft", "HandLeft"],
        "right_arm": ["ForarmRight", "HandRight"],
        "left_leg": ["CalfLeft", "FootLeft", "ToeLeft"],
        "right_leg": ["CalfRight", "FootRight", "ToeRight"]
    }
    main_group_names = ["left_arm", "right_arm", "left_leg", "right_leg"]
    transform_types = ['BonePositions', 'BoneRotations']
    minimum_positions_count = len(main_group_names)  # One per main group

    # Local quats to global quats:
    all_joints = [skeleton.index_bones[i] for i in range(skeleton.nb_joints)]
    rotation_features_idx = subset.get_feature_indices(['BoneRotations'], all_joints)
    rotations = subset.data[:, rotation_features_idx]
    nframes = rotations.shape[0]
    rotations = rotations.view(nframes, skeleton.nb_joints, 4)[:, :, [3, 0, 1, 2]]  # fix quat order
    rotations = compute_rotation_matrix_from_quaternion(rotations.view(-1, 4)).view(nframes, -1, 3, 3)
    _, rotations = skeleton.forward(rotations)  # to global
    rotations = compute_quaternions_from_rotation_matrices(rotations.view(-1, 3, 3)).view(nframes, -1, 4)

    all_ids = []
    all_types = []
    all_data = []

    # We want one effector per main bone group
    assert n_effectors >= minimum_positions_count
    second_sample_count = n_effectors - minimum_positions_count

    for e in range(n_effectors):
        output_dict["effector{}_type".format(e)] = []
        output_dict["effector{}_id".format(e)] = []
        output_dict["effector{}_data".format(e)] = []

    # Sample randomly at each frame
    for frame in range(subset.df.shape[0]):
        remaining_position_bones = copy.deepcopy(bone_groups)
        remaining_rotation_bones = copy.deepcopy(bone_groups)
        frame_position_bones = []
        frame_rotation_bones = []

        # FIRST SAMPLE PASS : 4 positional effectors, from the main groups.
        for main_group in main_group_names:
            chosen_bone = pick_and_remove_from_dictionary(remaining_position_bones, main_group)
            frame_position_bones.append(chosen_bone)

        # SECOND SAMPLE PASS : the rest of effectors,
        if second_sample_count > 0:
            additional_effector_types = random.choices(transform_types, k=second_sample_count)

            for effector_type in additional_effector_types:

                # Overwrite in cases where the bank of possible effectors of a given type is depleted.
                if len(list(remaining_position_bones.keys())) == 0:
                    effector_type = "BoneRotations"
                if len(list(remaining_rotation_bones.keys())) == 0:
                    effector_type = "BonePositions"

                if effector_type == "BonePositions":
                    chosen_group = random.choice(list(remaining_position_bones.keys()))
                    chosen_bone = pick_and_remove_from_dictionary(remaining_position_bones, chosen_group)
                    frame_position_bones.append(chosen_bone)

                if effector_type == "BoneRotations":
                    chosen_group = random.choice(list(remaining_rotation_bones.keys()))
                    chosen_bone = pick_and_remove_from_dictionary(remaining_rotation_bones, chosen_group)
                    frame_rotation_bones.append(chosen_bone)

        # We now have a list of positional effectors and an optionally empty list of rotational effectors.
        n_positions = len(frame_position_bones)
        n_rotations = len(frame_rotation_bones)
        assert n_positions + n_rotations == n_effectors
        bones = frame_position_bones + frame_rotation_bones
        features = ["BonePositions" for _ in range(n_positions)] + ["BoneRotations" for _ in range(n_rotations)]
        effector_types = [TYPE_VALUES['position']] * n_positions + [TYPE_VALUES['rotation']] * n_rotations
        effector_ids = [skeleton.bone_indexes[k] for k in frame_position_bones + frame_rotation_bones]
        effector_data = []

        for e in range(n_effectors):
            e_type = effector_types[e]
            e_id = effector_ids[e]
            e_idx = subset.get_feature_indices(features[e], bones[e])
            if e_type == TYPE_VALUES['position']:
                e_data = subset.df.values[frame, e_idx]
            elif e_type == TYPE_VALUES['rotation']:
                e_data = rotations[frame, e_id].cpu().numpy()
            effector_data.append(e_data)

            output_dict["effector{}_type".format(e)].append(e_type)
            output_dict["effector{}_id".format(e)].append(e_id)
            output_dict["effector{}_data".format(e)].append(e_data)

        all_ids.append(effector_ids)
        all_types.append(effector_types)
        all_data.append(effector_data)

    # Define the output dataframe
    output_df = pd.DataFrame()
    loc = 0
    for k, v in output_dict.items():
        output_df.insert(loc, k, v)
        loc += 1

    output_df['Frame'] = frame_indices
    output_df.set_index('Frame', inplace=True)

    packed_inputs = {'ids': all_ids, 'types': all_types, 'data': all_data}

    return output_df, packed_inputs, frame_indices


def create_randomly_sampled_dataframe(subset, skeleton, n_effectors, effectors,
                                      no_rotations=False, no_positions=False, combined_mode=False):
    output_dict = {}
    frame_indices = np.asarray(list(subset.df.index))

    assert not(no_rotations and no_positions)
    assert not(combined_mode and (no_rotations or no_positions))

    # Local quats to global quats:
    all_joints = [skeleton.index_bones[i] for i in range(skeleton.nb_joints)]
    rotation_features_idx = subset.get_feature_indices(['BoneRotations'], all_joints)
    rotations = subset.data[:, rotation_features_idx]
    nframes = rotations.shape[0]
    rotations = rotations.view(nframes, skeleton.nb_joints, 4)[:, :, [3, 0, 1, 2]]  # fix quat order
    rotations = compute_rotation_matrix_from_quaternion(rotations.view(-1, 4)).view(nframes, -1, 3, 3)
    _, rotations = skeleton.forward(rotations)  # to global
    rotations = compute_quaternions_from_rotation_matrices(rotations.view(-1, 3, 3)).view(nframes, -1, 4)
    effective_num_effector = n_effectors
    if no_positions:
        effective_num_effector = n_effectors + 1
    elif combined_mode:
        effective_num_effector = n_effectors * 2

    all_ids = []
    all_types = []
    all_data = []
    for e in range(effective_num_effector):
        output_dict["effector{}_type".format(e)] = []
        output_dict["effector{}_id".format(e)] = []
        output_dict["effector{}_data".format(e)] = []

    # Sample randomly at each frame
    for frame in range(subset.df.shape[0]):
        if combined_mode or no_rotations:
            n_pos_effectors = n_effectors
        elif no_positions:
            n_pos_effectors = 1
        else:
            # Sample at least one position effector (type = 0)
            n_pos_effectors = np.random.randint(1, effective_num_effector + 1)

        if no_positions:
            # WARNING: this assumes root/hips at index 0
            pos_effectors = [effectors[0]]
        else:
            pos_effectors = list(np.random.choice(effectors, n_pos_effectors, replace=False))

        if combined_mode:
            n_rot_effectors = n_effectors
        else:
            n_rot_effectors = effective_num_effector - n_pos_effectors
        rot_effectors = list(np.random.choice(effectors, n_rot_effectors, replace=False))

        bones = [str(s) for s in (pos_effectors + rot_effectors)]
        features = ["BonePositions" for _ in range(n_pos_effectors)] + ["BoneRotations" for _ in range(n_rot_effectors)]

        effector_types = [TYPE_VALUES['position']] * n_pos_effectors + [TYPE_VALUES['rotation']] * n_rot_effectors
        effector_ids = [skeleton.bone_indexes[k] for k in pos_effectors + rot_effectors]
        effector_data = []

        for effector in range(len(effector_types)):
            e_type = effector_types[effector]
            e_id = effector_ids[effector]
            e_idx = subset.get_feature_indices(features[effector], bones[effector])
            if e_type == TYPE_VALUES['position']:
                e_data = subset.df.values[frame, e_idx]
            elif e_type == TYPE_VALUES['rotation']:
                e_data = rotations[frame, e_id].cpu().numpy()
            effector_data.append(e_data)

            output_dict["effector{}_type".format(effector)].append(e_type)
            output_dict["effector{}_id".format(effector)].append(e_id)
            output_dict["effector{}_data".format(effector)].append(e_data)

        all_ids.append(effector_ids)
        all_types.append(effector_types)
        all_data.append(effector_data)

    # Define the output dataframe
    output_df = pd.DataFrame()
    loc = 0
    for k, v in output_dict.items():
        output_df.insert(loc, k, v)
        loc += 1

    output_df['Frame'] = frame_indices
    output_df.set_index('Frame', inplace=True)

    packed_inputs = {'ids' : all_ids, 'types': all_types, 'data': all_data}

    return output_df, packed_inputs, frame_indices


def create_randomized_sets(subset, effectors, config, skeleton, seed, use_bone_groups=True,
                           frames_per_effector=5000, n_effector_min=3, n_effector_max=10, output_string="output_data",
                           no_rotations=False, no_positions=False, combined_mode=False):
    df = subset.df
    n_buckets = n_effector_max - n_effector_min + 1

    # Create buckets for [n_effector_min, ..., n_effector_max] effectors
    for i in range(n_buckets):
        bucket_df = df.sample(n=frames_per_effector, replace=True, random_state=seed)
        bucket_dataset = FlatTypedColumnDataset(bucket_df, config)
        if use_bone_groups:
            output_df, packed_inputs, frame_indices = create_randomly_sampled_bonegroups_data(bucket_dataset, skeleton,
                                                                                              n_effector_min + i)
        else:
            output_df, packed_inputs, frame_indices = create_randomly_sampled_dataframe(bucket_dataset, skeleton,
                                                                                        n_effector_min + i, effectors,
                                                                                        no_rotations, no_positions,
                                                                                        combined_mode)
        # Packed versions
        if no_rotations or no_positions or combined_mode:
            out_string = output_string+"_packed.csv"
        else:
            out_string = output_string+"_{}Effectors_packed.csv".format(i+n_effector_min)
        save_packed_referenced_random_data(packed_inputs, frame_indices, out_string)

        # Unpacked versions
        if no_rotations or no_positions or combined_mode:
            out_string = output_string+".csv"
        else:
            out_string = output_string+"_{}Effectors.csv".format(i+n_effector_min)
        output_df.to_csv(out_string, index=True)


def create_data_splits(datapath, id_seed=0, subsample_fraction=0.1,
                       six_pos_version=True, five_pos_version=True, bake_train_augmentation=False,
                       bake_test_augmentation=False, create_random_effector_files=False, create_ICLR_random_effector_files=True,
                       n_frames_per_random_effector_count=5000, output_path=".", output_name="datasplit"):

    # Enforce reproducibility given the same source
    random.seed(id_seed)
    np.random.seed(id_seed)
    torch.manual_seed(id_seed)

    config_path = os.path.join(datapath, "dataset_settings.json") # Should contain only Positions, Rotations, Skeleton
    with open(config_path, "r") as f:
        config = json.load(f)

    assert len(config['features'].keys()) == 2 and 'BoneRotations' in config['features'].keys() and 'BonePositions' in config['features'].keys()

    skeleton = Skeleton(config['skeleton'])

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Split based on sequence
    split = create_split_file(datapath, output_path, id_seed)
    train_set = TypedColumnDataset(split, subset="Training").df.drop(columns="Sequence", inplace=False)
    valid_set = TypedColumnDataset(split, subset="Validation").df.drop(columns="Sequence", inplace=False)
    test_set = TypedColumnDataset(split, subset="Test").df.drop(columns="Sequence", inplace=False)

    # Subsample + Shuffle : Sample DF without replacement
    train_set = train_set.sample(frac=subsample_fraction, replace=False, random_state=id_seed).reset_index(drop=True)
    valid_set = valid_set.sample(frac=subsample_fraction, replace=False, random_state=id_seed).reset_index(drop=True)
    test_set = test_set.sample(frac=subsample_fraction, replace=False, random_state=id_seed).reset_index(drop=True)

    # Flat datasets
    train_set = FlatTypedColumnDataset(train_set, config)
    valid_set = FlatTypedColumnDataset(valid_set, config)
    test_set = FlatTypedColumnDataset(test_set, config)

    # Bake data augmentation
    if bake_train_augmentation:
        train_set = augment_dataset(train_set, skeleton)
    if bake_test_augmentation:
        valid_set = augment_dataset(valid_set, skeleton)
        test_set = augment_dataset(test_set, skeleton)

    train_set.df.to_csv(os.path.join(output_path, output_name+"_training_fullPose.csv"), index=False)
    valid_set.df.to_csv(os.path.join(output_path, output_name + "_validation_fullPose.csv"), index=False)
    test_set.df.to_csv(os.path.join(output_path, output_name + "_test_fullPose.csv"), index=False)

    # Save only fixed 6 joints position
    if six_pos_version:
        effectors = ["Hips", "Neck", "HandLeft", "HandRight", "FootLeft", "FootRight"]
        joint_positions_idx = train_set.get_feature_indices(["BonePositions"], effectors)
        sixpoints_train_df = train_set.df.iloc[:, joint_positions_idx]
        sixpoints_valid_df = valid_set.df.iloc[:, joint_positions_idx]
        sixpoints_test_df = test_set.df.iloc[:, joint_positions_idx]
        sixpoints_train_df.to_csv(os.path.join(output_path, output_name + "_training_6PointsEffectors.csv"), index=False)
        sixpoints_valid_df.to_csv(os.path.join(output_path, output_name + "_validation_6PointsEffectors.csv"), index=False)
        sixpoints_test_df.to_csv(os.path.join(output_path, output_name + "_test_6PointsEffectors.csv"), index=False)

        if args.sixpoints_with_added_rotation:
            # Version with bone rotation as additional info
            bone = 'Neck'
            all_joints = [skeleton.index_bones[i] for i in range(skeleton.nb_joints)]
            rotation_idx = valid_set.get_feature_indices(["BoneRotations"], all_joints)
            valid_rotations = torch.Tensor(valid_set.df.iloc[:, rotation_idx].values)
            nframes = valid_rotations.shape[0]
            valid_rotations = valid_rotations.view(nframes, skeleton.nb_joints, 4)[:, :, [3, 0, 1, 2]]  # fix quat order
            valid_rotations = compute_rotation_matrix_from_quaternion(valid_rotations.contiguous().view(-1, 4)).view(nframes, -1, 3, 3)
            _, valid_rotations = skeleton.forward(valid_rotations)  # to global
            valid_rotations = compute_quaternions_from_rotation_matrices(valid_rotations.view(-1, 3, 3)).view(nframes, -1, 4)

            rotation_idx = test_set.get_feature_indices(["BoneRotations"], all_joints)
            test_rotations = torch.Tensor(test_set.df.iloc[:, rotation_idx].values)
            nframes = test_rotations.shape[0]
            test_rotations = test_rotations.view(nframes, skeleton.nb_joints, 4)[:, :, [3, 0, 1, 2]]  # fix quat order
            test_rotations = compute_rotation_matrix_from_quaternion(test_rotations.contiguous().view(-1, 4)).view(nframes, -1, 3, 3)
            _, test_rotations = skeleton.forward(test_rotations)  # to global
            test_rotations = compute_quaternions_from_rotation_matrices(test_rotations.view(-1, 3, 3)).view(nframes, -1, 4)

            hips_rotation_idx = train_set.get_feature_indices(["BoneRotations"], [bone])
            indices = joint_positions_idx + hips_rotation_idx
            sixpoints_train_df = train_set.df.iloc[:, indices]
            sixpoints_valid_df = valid_set.df.iloc[:, indices]
            sixpoints_test_df = test_set.df.iloc[:, indices]

            sixpoints_valid_df['BoneRotations_{}_X'.format(bone)] = valid_rotations[:, skeleton.bone_indexes[bone], 1]
            sixpoints_valid_df['BoneRotations_{}_Y'.format(bone)] = valid_rotations[:, skeleton.bone_indexes[bone], 2]
            sixpoints_valid_df['BoneRotations_{}_Z'.format(bone)] = valid_rotations[:, skeleton.bone_indexes[bone], 3]
            sixpoints_valid_df['BoneRotations_{}_W'.format(bone)] = valid_rotations[:, skeleton.bone_indexes[bone], 0]

            sixpoints_test_df['BoneRotations_{}_X'.format(bone)] = test_rotations[:, skeleton.bone_indexes[bone], 1]
            sixpoints_test_df['BoneRotations_{}_Y'.format(bone)] = test_rotations[:, skeleton.bone_indexes[bone], 2]
            sixpoints_test_df['BoneRotations_{}_Z'.format(bone)] = test_rotations[:, skeleton.bone_indexes[bone], 3]
            sixpoints_test_df['BoneRotations_{}_W'.format(bone)] = test_rotations[:, skeleton.bone_indexes[bone], 0]

            sixpoints_train_df.to_csv(os.path.join(output_path, output_name + "_training_6PointsEffectors_{}Rotation.csv".format(bone)), index=False)
            sixpoints_valid_df.to_csv(os.path.join(output_path, output_name + "_validation_6PointsEffectors_{}Rotation.csv".format(bone)),index=False)
            sixpoints_test_df.to_csv(os.path.join(output_path, output_name + "_test_6PointsEffectors_{}Rotation.csv".format(bone)), index=False)

    # Save only fixed 5 joints position - same as FinalIK
    if five_pos_version:
        effectors = ["Chest", "HandLeft", "HandRight", "FootLeft", "FootRight"]
        joint_positions_idx = train_set.get_feature_indices(["BonePositions"], effectors)
        fivepoints_train_df = train_set.df.iloc[:, joint_positions_idx]
        fivepoints_valid_df = valid_set.df.iloc[:, joint_positions_idx]
        fivepoints_test_df = test_set.df.iloc[:, joint_positions_idx]
        fivepoints_train_df.to_csv(os.path.join(output_path, output_name + "_training_5PointsEffectors.csv"), index=False)
        fivepoints_valid_df.to_csv(os.path.join(output_path, output_name + "_validation_5PointsEffectors.csv"), index=False)
        fivepoints_test_df.to_csv(os.path.join(output_path, output_name + "_test_5PointsEffectors.csv"), index=False)

    if create_random_effector_files:
        # Use UX effectors only
        ux_effectors = ['Hips', 'Neck', 'HandLeft', 'HandRight', 'FootLeft', 'FootRight', 'CalfLeft', 'CalfRight',
                     'ForarmLeft', 'ForarmRight', 'Head', 'ToeLeft', 'ToeRight', 'Chest', 'BicepLeft', 'BicepRight',
                     'ThighLeft', 'ThighRight']

        create_randomized_sets(valid_set, ux_effectors, config, skeleton, id_seed, use_bone_groups=True,
                               frames_per_effector=n_frames_per_random_effector_count,
                               n_effector_min=4, n_effector_max=12,
                               output_string=os.path.join(output_path, output_name + "_validation_randomized"))

        create_randomized_sets(test_set, ux_effectors, config, skeleton, id_seed, use_bone_groups=True,
                               frames_per_effector=n_frames_per_random_effector_count,
                               n_effector_min=4, n_effector_max=12,
                               output_string=os.path.join(output_path, output_name + "_test_randomized"))

    if create_ICLR_random_effector_files:
        # Or use all effectors
        all_effectors = skeleton.all_joints
        fractions = [1, 0.75, 0.5, 0.25, 0.05]

        for percentage in fractions:
            effector_count = int(skeleton.nb_joints * percentage)

            # Positions
            curr_output_name = output_name + "_{:.0%}_positions".format(percentage)
            create_randomized_sets(valid_set, all_effectors, config, skeleton, id_seed, use_bone_groups=False,
                                   frames_per_effector=n_frames_per_random_effector_count,
                                   n_effector_min=effector_count, n_effector_max=effector_count,
                                   output_string=os.path.join(output_path, curr_output_name + "_validation_randomized"),
                                   no_rotations=True, no_positions=False, combined_mode=False)

            create_randomized_sets(test_set, all_effectors, config, skeleton, id_seed, use_bone_groups=False,
                                   frames_per_effector=n_frames_per_random_effector_count,
                                   n_effector_min=effector_count, n_effector_max=effector_count,
                                   output_string=os.path.join(output_path, curr_output_name + "_test_randomized"),
                                   no_rotations=True, no_positions=False, combined_mode=False)

            # Rotations
            curr_output_name = output_name + "_{:.0%}_rotations".format(percentage)
            create_randomized_sets(valid_set, all_effectors, config, skeleton, id_seed, use_bone_groups=False,
                                   frames_per_effector=n_frames_per_random_effector_count,
                                   n_effector_min=effector_count, n_effector_max=effector_count,
                                   output_string=os.path.join(output_path, curr_output_name + "_validation_randomized"),
                                   no_rotations=False, no_positions=True, combined_mode=False)

            create_randomized_sets(test_set, all_effectors, config, skeleton, id_seed, use_bone_groups=False,
                                   frames_per_effector=n_frames_per_random_effector_count,
                                   n_effector_min=effector_count, n_effector_max=effector_count,
                                   output_string=os.path.join(output_path, curr_output_name + "_test_randomized"),
                                   no_rotations=False, no_positions=True, combined_mode=False)

            # Combined
            curr_output_name = output_name + "_{:.0%}_combined".format(percentage)
            create_randomized_sets(valid_set, all_effectors, config, skeleton, id_seed, use_bone_groups=False,
                                   frames_per_effector=n_frames_per_random_effector_count,
                                   n_effector_min=effector_count, n_effector_max=effector_count,
                                   output_string=os.path.join(output_path, curr_output_name + "_validation_randomized"),
                                   no_rotations=False, no_positions=False, combined_mode=True)

            create_randomized_sets(test_set, all_effectors, config, skeleton, id_seed, use_bone_groups=False,
                                   frames_per_effector=n_frames_per_random_effector_count,
                                   n_effector_min=effector_count, n_effector_max=effector_count,
                                   output_string=os.path.join(output_path, curr_output_name + "_test_randomized"),
                                   no_rotations=False, no_positions=False, combined_mode=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True, description="Split dataset into Train|Valid|Test")
    parser.add_argument('--datapath', type=str, default="./datasets/deeppose_paper_mixamo_sources", help='Path to dataset')
    parser.add_argument('--subsample_fraction', type=float, default=0.1, help='Subsampling fraction')
    parser.add_argument('--output_name', type=str, default="minimixamo", help='Output data name')
    parser.add_argument('--nb_splits', type=int, default=3, help='Number of splits to produce')
    parser.add_argument('--n_frames_per_random_effector_count', type=int, default=5000, help='Number of frames in each random effector file')
    parser.add_argument('--skip_sixpoints_files', dest='skip_sixpoints_files', default=False, action='store_true')
    parser.add_argument('--sixpoints_with_added_rotation', dest='sixpoints_with_added_rotation', default=False, action='store_true')
    parser.add_argument('--skip_fivepoints_files', dest='skip_fivepoints_files', default=False, action='store_true')
    parser.add_argument('--skip_test_augmentation', dest='skip_test_augmentation', default=False, action='store_true')
    parser.add_argument('--skip_train_augmentation', dest='skip_train_augmentation', default=False, action='store_true')
    parser.add_argument('--skip_random_effector_files', dest='skip_random_effector_files', default=False, action='store_true')
    parser.add_argument('--skip_ICLR_random_effector_files', dest='skip_ICLR_random_effector_files', default=False, action='store_true')
    args = parser.parse_args()

    # N Splits
    for i in range(1, args.nb_splits + 1):

        print("Working on Split {}...".format(i))
        create_data_splits(args.datapath, id_seed=i, subsample_fraction=args.subsample_fraction,
                           six_pos_version=not args.skip_sixpoints_files, five_pos_version=not args.skip_sixpoints_files,
                           bake_train_augmentation=not args.skip_train_augmentation,
                           bake_test_augmentation=not args.skip_test_augmentation,
                           create_random_effector_files=not args.skip_random_effector_files,
                           create_ICLR_random_effector_files=not args.skip_ICLR_random_effector_files,
                           n_frames_per_random_effector_count=args.n_frames_per_random_effector_count,
                           output_path= os.path.join(args.datapath, "Split{}".format(i)), output_name=args.output_name)

    print("Done.")
