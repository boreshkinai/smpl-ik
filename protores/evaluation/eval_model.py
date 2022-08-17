import os
import torch
import pandas as pd
import argparse
import ntpath
import copy
from protores.data.datasets import DatasetLoader

import matplotlib.pyplot as plt

from protores.evaluation.eval_utils import build_dataset, fixed_points_effector_batch, BatchedRandomEffectorDataPacked, \
    save_predictions, get_confidence_interval_margins, setup_model, benchmark_from_csv

plt.style.use('seaborn')
plt.rcParams.update({'font.sans-serif':'cambria'})
plt.rcParams.update({'font.family':'sans-serif'})
import numpy as np
VALID_BENCHMARKS = {"minimixamo": "deeppose_paper2021_minimixamo", "miniunity": "deeppose_paper2021_miniunity"}
VALID_MODES = ["normal", "random", "five_points"]
RANDOM_EFFECTOR_COUNTS = [6, 7, 8, 9, 10, 11, 12]
FRACTION_RANDOM_PERCENTAGES = [5, 25, 50, 75, 100]
FRACTION_RANDOM_MODES = ["positions", "rotations", "combined"]


def plot_fraction_random_losses(res, title_prefix="", plot_confidence=False):
    for loss in res[FRACTION_RANDOM_MODES[0]]["losses"].keys():
        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        for mode in FRACTION_RANDOM_MODES:
            if mode == "positions":
                legend = "Positional Inputs"
            elif mode == "rotations":
                legend = "Rotational Inputs"
            elif mode == "combined":
                legend = "Combined Inputs"
            else:
                legend = mode

            ys = np.asarray(res[mode]["losses"][loss])
            ax.plot(xs, ys, linewidth=1.1, label=legend)

            if plot_confidence:
                margin = np.asarray(res[mode]["margins"][loss])
                lower = ys - margin
                upper = ys + margin
                ax.fill_between(xs, lower, upper, alpha=0.25)

        ax.set_xlabel('Effector percentage')
        ax.set_ylabel('Loss')
        if loss == "pos":
            ax.set_title(title_prefix + r"$\mathcal{L}_{gpd-L2}$")
        elif loss == "rot":
            ax.set_title(title_prefix + r"$\mathcal{L}_{loc-geo}$")
        elif loss == "fk":
            ax.set_title(title_prefix + r"$\mathcal{L}_{ikd-L2}$")
        else:
            ax.set_title(loss)
        ax.set_yscale('log')
        ax.grid(which='minor', alpha=0.2)
        plt.xticks(xs, xs)
        plt.tight_layout()
        ax.legend()  # Add a legend.
        plt.show()


class BenchmarkEvaluator:
    def __init__(self, datasets_path, benchmark, random_effector_counts, device, verbose):
        
        self.datasets_path = datasets_path
        self.benchmark = benchmark
        self.device = device
        self.verbose = verbose
        self.random_effector_counts = random_effector_counts
        
        self.data_path = os.path.join(self.datasets_path, VALID_BENCHMARKS[self.benchmark])
        
        dataset_loader = DatasetLoader(self.datasets_path)
        dataset_loader.pull(VALID_BENCHMARKS[self.benchmark])

        self.clipgroup_path = os.path.join(self.data_path, 'clipgroup')
        self.data_config_path = os.path.join(self.data_path, 'dataset_settings.json')
        
        full_pose_validation_csv = os.path.join(self.clipgroup_path, self.benchmark + "_validation_fullPose.csv")
        full_pose_test_csv = os.path.join(self.clipgroup_path, self.benchmark+"_test_fullPose.csv")
                
        validation_set, _, _, _ = build_dataset(self.data_config_path, full_pose_validation_csv)
        test_set, skeleton, _, _ = build_dataset(self.data_config_path, full_pose_test_csv)
        
        self.data_splits = {'validation': validation_set, 'test': test_set}
        self.skeleton = skeleton
        
        self.six_effectors = ['Hips', 'Neck', 'HandLeft', 'HandRight', 'FootLeft', 'FootRight']
        self.init_sixpoint_data()
        self.five_effectors = ['Chest', 'HandLeft', 'HandRight', 'FootLeft', 'FootRight']
        self.init_fivepoint_data()
        self.init_random_data()
        
    def init_sixpoint_data(self):
        valid_file = os.path.join(self.clipgroup_path, self.benchmark + "_validation_6PointsEffectors.csv")
        test_file = os.path.join(self.clipgroup_path, self.benchmark + "_test_6PointsEffectors.csv")
        
        self.sixpoints_batch = {}
        self.sixpoints_batch['validation'] = fixed_points_effector_batch(valid_file, 
                                                                         self.six_effectors, self.skeleton, self.device)
        self.sixpoints_batch['test'] = fixed_points_effector_batch(test_file, 
                                                                   self.six_effectors, self.skeleton, self.device)
        
    def init_fivepoint_data(self):
        valid_file = os.path.join(self.clipgroup_path, self.benchmark + "_validation_5PointsEffectors.csv")
        test_file = os.path.join(self.clipgroup_path, self.benchmark + "_test_5PointsEffectors.csv")
        
        self.fivepoints_batch = {}
        self.fivepoints_batch['validation'] = fixed_points_effector_batch(valid_file, 
                                                                          self.five_effectors, self.skeleton, self.device)
        self.fivepoints_batch['test'] = fixed_points_effector_batch(test_file, 
                                                                    self.five_effectors, self.skeleton, self.device)
        
    def init_random_data(self):
        self.random_data = {}
        for effector_count in self.random_effector_counts:
            #  Retrieve the right csv, build the dataframe for that number of effectors
            random_effectors_file = os.path.join(self.clipgroup_path, 
                                                 "{0}_{1}_randomized_{2}Effectors_packed.csv".format(self.benchmark, 
                                                                                                     "validation", 
                                                                                                     str(effector_count)))
            random_effectors_df = pd.read_csv(random_effectors_file)
            random_effector_data = BatchedRandomEffectorDataPacked(random_effectors_df, self.device)
            fullpose_batch = self.data_splits['validation'][random_effector_data.frame_indices]
            self.random_data[f"validation_{effector_count}effectors_input_data"] = random_effector_data.batch
            self.random_data[f"validation_{effector_count}effectors_fullpose_batch"] = fullpose_batch
            
            #  Retrieve the right csv, build the dataframe for that number of effectors
            random_effectors_file = os.path.join(self.clipgroup_path, 
                                                 "{0}_{1}_randomized_{2}Effectors_packed.csv".format(self.benchmark, 
                                                                                                     "test", 
                                                                                                     str(effector_count)))
            random_effectors_df = pd.read_csv(random_effectors_file)
            random_effector_data = BatchedRandomEffectorDataPacked(random_effectors_df, self.device)
            fullpose_batch = self.data_splits['test'][random_effector_data.frame_indices]
            self.random_data[f"test_{effector_count}effectors_input_data"] = random_effector_data.batch
            self.random_data[f"test_{effector_count}effectors_fullpose_batch"] = fullpose_batch
    
    @torch.no_grad()
    def fixed_points_benchmark(self, model, fixepoints_batch, fullpose_dataset):
        metrics = {}
        # Target data from fullpose
        _, target_data = model.get_data_from_batch(fullpose_dataset[:], fixed_effector_setup=True)

        data = fixepoints_batch.copy()
        for k, v in data.items():
            data[k] = v.to(model.device)

        predictions = model(data)
        
        for k, v in predictions.items():
            if v is not None:
                predictions[k] = v.to('cpu')  

        model.test_fk_metric.reset()
        model.test_position_metric.reset()
        model.test_rotation_metric.reset()

        model.update_test_metrics(predictions, target_data)
        metrics["fk"] = model.test_fk_metric.compute()
        metrics["position"] = model.test_position_metric.compute()
        metrics["rotation"] = model.test_rotation_metric.compute()[0]
        model.test_fk_metric.reset()
        model.test_position_metric.reset()
        model.test_rotation_metric.reset()

        if self.verbose:
            for k, v in metrics.items():
                print("\t {0} : {1}".format(k, v))
        return metrics
        
    @torch.no_grad()
    def evaluate_sixpoint(self, model, split):
        model.eval()
        if self.verbose:
            print(f"|== {split} metrics sixpoint benchmark:")
        return self.fixed_points_benchmark(model, self.sixpoints_batch[split], self.data_splits[split])
        
    @torch.no_grad()
    def evaluate_fivepoint(self, model, split):
        model.eval()
        if self.verbose:
            print(f"|== {split} metrics fivepoint benchmark:")
        return self.fixed_points_benchmark(model, self.fivepoints_batch[split], self.data_splits[split])
    
    @torch.no_grad()
    def evaluate_random(self, model, split):
        model.eval()
        model.test_fk_metric.reset()
        model.test_position_metric.reset()
        model.test_rotation_metric.reset()
        if self.verbose:
            print(f"|== {split} metrics random benchmark:")
        metrics = {}
        for effector_count in self.random_effector_counts:
            if self.verbose:
                print("  Benchmarking for {} random effectors... ".format(effector_count))
                
            data = self.random_data[f"{split}_{effector_count}effectors_input_data"].copy()
            for k, v in data.items():
                data[k] = v.to(model.device)
            
            predictions = model.forward_packed(data)
            
            _, target_data = model.get_data_from_batch(self.random_data[f"{split}_{effector_count}effectors_fullpose_batch"], 
                                                       fixed_effector_setup=True)
                
            for k, v in predictions.items():
                if v is not None:
                    predictions[k] = v.to('cpu')     
                
            # Accumulate metrics
            model.update_test_metrics(predictions, target_data)
            
        metrics["fk"] = model.test_fk_metric.compute()
        metrics["position"] = model.test_position_metric.compute()
        metrics["rotation"] = model.test_rotation_metric.compute()[0]

        if self.verbose:
            for k, v in metrics.items():
                print("\t {0} : {1}".format(k, v))

        model.test_fk_metric.reset()
        model.test_position_metric.reset()
        model.test_rotation_metric.reset()
        
        return metrics


@torch.no_grad()
def fixed_points_benchmark(model, fixepoints_batch, fullpose_dataset, out_file, verbose=False):
    metrics = {}
    # Target data from fullpose
    _, target_data = model.get_data_from_batch(fullpose_dataset[:], fixed_effector_setup=True)
    
    data = fixepoints_batch.copy()
    for k, v in data.items():
        data[k] = v.to(model.device)
    
    predictions = model(data)
    
    model.test_fk_metric.reset()
    model.test_position_metric.reset()
    model.test_rotation_metric.reset()
    
    model.update_test_metrics(predictions, target_data)
    metrics["fk"] = model.test_fk_metric.compute()
    metrics["position"] = model.test_position_metric.compute()
    metrics["rotation"] = model.test_rotation_metric.compute()[0]
    model.test_fk_metric.reset()
    model.test_position_metric.reset()
    model.test_rotation_metric.reset()

    if verbose:
        print(" Metrics from model predictions for {} : ".format(ntpath.basename(out_file)))
        for k, v in metrics.items():
            print("\t {0} : {1}".format(k, v))

    # Save results as csv
    save_predictions(validation_set.skeleton, predictions, out_file)


@torch.no_grad()
def random_benchmark(model, clipgroup_path, benchmark, subset, target_dataset, out_file, verbose=False, frames_per_bucket=None):
    metrics = {}
    all_predictions = {}
    all_frame_indices = []
    
    model.test_fk_metric.reset()
    model.test_position_metric.reset()
    model.test_rotation_metric.reset()

    for effector_count in RANDOM_EFFECTOR_COUNTS:
        if verbose:
            print("  Benchmarking for {} random effectors... ".format(effector_count))

        #  Retrieve the right csv, build the dataframe for that number of effectors
        random_effectors_file = os.path.join(clipgroup_path, "{0}_{1}_randomized_{2}Effectors_packed.csv".format(benchmark, subset, str(effector_count)))
        random_effectors_df = pd.read_csv(random_effectors_file, nrows=frames_per_bucket)
        random_effector_data = BatchedRandomEffectorDataPacked(random_effectors_df, model.device)
        all_frame_indices += random_effector_data.frame_indices
        fullpose_batch = target_dataset[random_effector_data.frame_indices]

        # Get "input_data" for that row, mimicking what would be output from a normal batch
        input_data = random_effector_data.batch

        # Build a "normal" minibatch from the corresponding full pose
        _, target_data = model.get_data_from_batch(fullpose_batch, fixed_effector_setup=True)

        # Run the model in the input_data
        predictions = model.forward_packed(input_data)

        for k, v in predictions.items():
            if k in list(all_predictions.keys()):
                if all_predictions[k] is not None and v is not None:
                    all_predictions[k] = torch.cat([all_predictions[k], v])
            else:
                all_predictions[k] = v

        # Accumulate metrics
        model.update_test_metrics(predictions, target_data)

        metrics["fk"] = model.test_fk_metric.compute()
        metrics["position"] = model.test_position_metric.compute()
        metrics["rotation"] = model.test_rotation_metric.compute()[0]

        if verbose or frames_per_bucket is not None:
            for k, v in metrics.items():
                print("\t {0} : {1}".format(k, v))

        model.test_fk_metric.reset()
        model.test_position_metric.reset()
        model.test_rotation_metric.reset()

    if frames_per_bucket is None:
        save_predictions(target_dataset.skeleton, all_predictions, out_file, frame_indices=all_frame_indices)


@torch.no_grad()
def random_fraction_benchmark(model, clipgroup_path, benchmark, subset, target_dataset,
                              effector_percentage, effector_mode, out_file, verbose=False, frames_per_bucket=None):
    metrics = {}
    all_predictions = {}
    all_frame_indices = []

    model.test_fk_metric.reset()
    model.test_position_metric.reset()
    model.test_rotation_metric.reset()

    #  Retrieve the right csv, build the dataframe for that number of effectors
    random_effectors_file = os.path.join(clipgroup_path,
                                         "{0}_{1}%_{2}_{3}_randomized_packed.csv".format(benchmark, effector_percentage,
                                                                                        effector_mode, subset))
    random_effectors_df = pd.read_csv(random_effectors_file, nrows=frames_per_bucket)
    random_effector_data = BatchedRandomEffectorDataPacked(random_effectors_df, model.device)
    all_frame_indices += random_effector_data.frame_indices
    fullpose_batch = target_dataset[random_effector_data.frame_indices]

    # Get "input_data" for that row, mimicking what would be output from a normal batch
    input_data = random_effector_data.batch

    # Build a "normal" minibatch from the corresponding full pose
    _, target_data = model.get_data_from_batch(fullpose_batch, fixed_effector_setup=True)

    # Run the model in the input_data
    predictions = model.forward_packed(input_data)

    for k, v in predictions.items():
        if k in list(all_predictions.keys()):
            if all_predictions[k] is not None and v is not None:
                all_predictions[k] = torch.cat([all_predictions[k], v])
        else:
            all_predictions[k] = v

    # Accumulate metrics
    model.update_test_metrics(predictions, target_data)

    metrics["fk"] = model.test_fk_metric.compute()
    metrics["position"] = model.test_position_metric.compute()
    metrics["rotation"] = model.test_rotation_metric.compute()[0]

    if verbose or frames_per_bucket is not None:
        for k, v in metrics.items():
            print("\t {0} : {1}".format(k, v))

    model.test_fk_metric.reset()
    model.test_position_metric.reset()
    model.test_rotation_metric.reset()

    margins = get_confidence_interval_margins(predictions, target_data, model.skeleton)

    if frames_per_bucket is None:
        save_predictions(target_dataset.skeleton, all_predictions, out_file, frame_indices=all_frame_indices)

    return metrics, margins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True, description="Split dataset into Train|Valid|Test")
    parser.add_argument('--datasets_path', type=str, default="./datasets/", help='Path to datasets')
    parser.add_argument('--benchmark', type=str, default="miniMixamo", help="Benchmark to perform. Must be in ['miniMixamo', 'miniUnity']")
    parser.add_argument('--model_folder', type=str, default="", help="Folder of model to benchmark")
    parser.add_argument('--checkpoint_name', type=str, default="last.ckpt", help="Name of model checkpoint")
    parser.add_argument('--skip_sixpoints_benchmark', dest='skip_sixpoints_benchmark', default=False, action='store_true')
    parser.add_argument('--skip_fivepoints_benchmark', dest='skip_fivepoints_benchmark', default=False, action='store_true')
    parser.add_argument('--skip_random_benchmark', dest='skip_random_benchmark', default=False, action='store_true')
    parser.add_argument('--quick_random_benchmark', dest='quick_random_benchmark', default=False, action='store_true')
    parser.add_argument('--skip_fraction_random_benchmark', dest='skip_fraction_random_benchmark', default=False, action='store_true')
    parser.add_argument('--verbose', dest='verbose', default=False, action='store_true')
    parser.add_argument('--benchmark_final_ik', dest='benchmark_final_ik', default=False, action='store_true')
    parser.add_argument('--final_ik_folder', type=str, default="D:\Workspace\DeepPose\.training\logs\paper2021\FinalIK", help="Path of FinalIK predictions")
    parser.add_argument('--force_new_predictions', dest='force_new_predictions', default=False, action='store_true')
    parser.add_argument('--device', type=str, default="cpu", help="Compute device")
    args = parser.parse_args()

    results_folder = "{}_benchmark".format(args.benchmark)
    results_path = os.path.join(args.model_folder, results_folder)
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    data_path = os.path.join(args.datasets_path, VALID_BENCHMARKS[args.benchmark.lower()])

    dataset_loader = DatasetLoader(args.datasets_path)
    dataset_loader.pull(VALID_BENCHMARKS[args.benchmark.lower()])

    clipgroup_path = os.path.join(data_path, 'clipgroup')
    data_config_path = os.path.join(data_path, 'dataset_settings.json')

    full_pose_validation_csv = os.path.join(clipgroup_path, args.benchmark + "_validation_fullPose.csv")
    full_pose_test_csv = os.path.join(clipgroup_path, args.benchmark+"_test_fullPose.csv")

    validation_set, _, _, _ = build_dataset(data_config_path, full_pose_validation_csv)
    test_set, skeleton, data_config, _ = build_dataset(data_config_path, full_pose_test_csv)

    assert args.benchmark.lower() in VALID_BENCHMARKS.keys(), "Invalid benchmark. Accepted benchmarks are : {}".format(
        list(VALID_BENCHMARKS.keys()))

    model = setup_model(args.model_folder, args.checkpoint_name, skeleton, args.device)

    if not args.skip_sixpoints_benchmark:
        print("\n|= Six points benchmark")
        six_effectors = ['Hips', 'Neck', 'HandLeft', 'HandRight', 'FootLeft', 'FootRight']
        sixpoints_valid_file = os.path.join(clipgroup_path, args.benchmark + "_validation_6PointsEffectors.csv")
        sixpoints_test_file = os.path.join(clipgroup_path, args.benchmark + "_test_6PointsEffectors.csv")
        sixpoints_valid_resultpath = os.path.join(results_path, "sixpoints_validation_predictions.csv")
        sixpoints_test_resultpath = os.path.join(results_path, "sixpoints_test_predictions.csv")

        print("|== Validation metrics:")
        if not os.path.exists(sixpoints_valid_resultpath) or args.force_new_predictions:
            sixpoints_valid_batch = fixed_points_effector_batch(sixpoints_valid_file, six_effectors, skeleton, args.device)

            # Used for comparing with random packed data
            # bone = 'Neck'
            # sixpoints_neckrotation_valid_file = os.path.join(clipgroup_path, args.benchmark + "_validation_6PointsEffectors_{}Rotation.csv".format(bone))
            # sixpoints_neckrotation_valid_batch = fixed_points_effector_batch(sixpoints_neckrotation_valid_file, six_effectors, skeleton, args.device, with_bone_rotation=True, bone='Neck')
            # sixpoints_packed_validation_file = os.path.join(clipgroup_path, args.benchmark + "_validation_6PointsEffectors_{}Rotation_packed.csv".format(bone))
            # fixed_points_to_packed_format(sixpoints_neckrotation_valid_batch, 6, sixpoints_packed_validation_file)

            fixed_points_benchmark(model, sixpoints_valid_batch, validation_set, sixpoints_valid_resultpath, verbose=args.verbose)
        benchmark_from_csv(model, data_config_path, full_pose_validation_csv, sixpoints_valid_resultpath)

        print("|== Test metrics:")
        if not os.path.exists(sixpoints_test_resultpath) or args.force_new_predictions:
            sixpoints_test_batch = fixed_points_effector_batch(sixpoints_test_file, six_effectors, skeleton, args.device)
            fixed_points_benchmark(model, sixpoints_test_batch, test_set, sixpoints_test_resultpath, verbose=args.verbose)
        benchmark_from_csv(model, data_config_path, full_pose_test_csv, sixpoints_test_resultpath)

    if not args.skip_fivepoints_benchmark:
        print("\n|= Five points benchmark")
        five_effectors = ['Chest', 'HandLeft', 'HandRight', 'FootLeft', 'FootRight']
        fivepoints_valid_file = os.path.join(clipgroup_path, args.benchmark + "_validation_5PointsEffectors.csv")
        fivepoints_test_file = os.path.join(clipgroup_path, args.benchmark + "_test_5PointsEffectors.csv")
        fivepoints_valid_resultpath = os.path.join(results_path, "fivepoints_validation_predictions.csv")
        fivepoints_test_resultpath = os.path.join(results_path, "fivepoints_test_predictions.csv")

        print("|== Validation metrics:")
        if not os.path.exists(fivepoints_valid_resultpath) or args.force_new_predictions:
            fivepoints_valid_batch = fixed_points_effector_batch(fivepoints_valid_file, five_effectors, skeleton, args.device)
            fixed_points_benchmark(model, fivepoints_valid_batch, validation_set, fivepoints_valid_resultpath, verbose=args.verbose)
        benchmark_from_csv(model, data_config_path, full_pose_validation_csv, fivepoints_valid_resultpath)

        print("|== Test metrics:")
        if not os.path.exists(fivepoints_test_resultpath) or args.force_new_predictions:
            fivepoints_test_batch = fixed_points_effector_batch(fivepoints_test_file, five_effectors, skeleton, args.device)
            fixed_points_benchmark(model, fivepoints_test_batch, test_set, fivepoints_test_resultpath, verbose=args.verbose)
        benchmark_from_csv(model, data_config_path, full_pose_test_csv, fivepoints_test_resultpath)

    if not args.skip_random_benchmark:
        print("\n|= Random benchmark")
        random_valid_resultpath = os.path.join(results_path, "random_validation_predictions.csv")
        random_test_resultpath = os.path.join(results_path, "random_test_predictions.csv")

        print("|== Validation metrics:")
        if args.quick_random_benchmark:
            random_benchmark(model, clipgroup_path, args.benchmark, "validation", validation_set,
                             random_valid_resultpath, verbose=args.verbose, frames_per_bucket=100)
        else:
            if not os.path.exists(random_valid_resultpath) or args.force_new_predictions:
                random_benchmark(model, clipgroup_path, args.benchmark, "validation", validation_set, random_valid_resultpath, verbose=args.verbose)
            benchmark_from_csv(model, data_config_path, full_pose_validation_csv, random_valid_resultpath)

        print("|== Test metrics:")
        if args.quick_random_benchmark:
            random_benchmark(model, clipgroup_path, args.benchmark, "test", test_set, random_test_resultpath, verbose=args.verbose, frames_per_bucket=200)
        else:
            if not os.path.exists(random_test_resultpath) or args.force_new_predictions:
                random_benchmark(model, clipgroup_path, args.benchmark, "test", test_set, random_test_resultpath, verbose=args.verbose)
            benchmark_from_csv(model, data_config_path, full_pose_test_csv, random_test_resultpath)

    if not args.skip_fraction_random_benchmark:

        res = {}
        xs = [p for p in FRACTION_RANDOM_PERCENTAGES]
        for effector_mode in FRACTION_RANDOM_MODES:
            res[effector_mode] = {'losses': {
                                        'pos': [],
                                        'rot': [],
                                        'fk': []},
                                   'margins':{
                                       'pos': [],
                                       'rot': [],
                                       'fk': []}}
        val_res = res
        test_res = copy.deepcopy(res)

        for percentage in FRACTION_RANDOM_PERCENTAGES:
            for effector_mode in FRACTION_RANDOM_MODES:
                print(f"\n|= Random {percentage}% {effector_mode} Benchmark")
                random_valid_resultpath = os.path.join(results_path, f"random_{percentage}%_{effector_mode}_validation_predictions.csv")
                random_test_resultpath = os.path.join(results_path, f"random_{percentage}%_{effector_mode}_test_predictions.csv")

                print("|== Validation metrics:")
                if args.quick_random_benchmark:
                    benchmark, margins = random_fraction_benchmark(model, clipgroup_path, args.benchmark, "validation", validation_set,
                                              percentage, effector_mode, random_valid_resultpath, verbose=args.verbose,
                                              frames_per_bucket=100)
                else:
                    if not os.path.exists(random_valid_resultpath) or args.force_new_predictions:
                        benchmark, margins = random_fraction_benchmark(model, clipgroup_path, args.benchmark, "validation", validation_set,
                                                  percentage, effector_mode, random_valid_resultpath, verbose=args.verbose)
                    benchmark, margins = benchmark_from_csv(model, data_config_path, full_pose_validation_csv, random_valid_resultpath)

                val_res[effector_mode]['losses']['pos'].append(benchmark['position'])
                val_res[effector_mode]['margins']['pos'].append(margins['pos'])
                val_res[effector_mode]['losses']['rot'].append(benchmark['rotation'])
                val_res[effector_mode]['margins']['rot'].append(margins['rot'])
                val_res[effector_mode]['losses']['fk'].append(benchmark['fk'])
                val_res[effector_mode]['margins']['fk'].append(benchmark['fk'])


                print("|== Test metrics:")
                if args.quick_random_benchmark:
                    benchmark, margins = random_fraction_benchmark(model, clipgroup_path, args.benchmark, "test", test_set,
                                              percentage, effector_mode, random_test_resultpath, verbose=args.verbose,
                                              frames_per_bucket=100)
                else:
                    if not os.path.exists(random_test_resultpath) or args.force_new_predictions:
                        benchmark, margins = random_fraction_benchmark(model, clipgroup_path, args.benchmark, "test", test_set,
                                                  percentage, effector_mode, random_test_resultpath, verbose=args.verbose)
                    benchmark, margins = benchmark_from_csv(model, data_config_path, full_pose_test_csv, random_test_resultpath)

                test_res[effector_mode]['losses']['pos'].append(benchmark['position'])
                test_res[effector_mode]['margins']['pos'].append(margins['pos'])
                test_res[effector_mode]['losses']['rot'].append(benchmark['rotation'])
                test_res[effector_mode]['margins']['rot'].append(margins['rot'])
                test_res[effector_mode]['losses']['fk'].append(benchmark['fk'])
                test_res[effector_mode]['margins']['fk'].append(benchmark['fk'])

        # Plot:
        if True:
            plot_fraction_random_losses(val_res, args.benchmark + " : Validation : ", plot_confidence=True)
            plot_fraction_random_losses(test_res, args.benchmark + " : Test Set : ", plot_confidence=True)

    if args.benchmark_final_ik:
        print("|= Final IK : 5-points benchmark:")
        finalik_valid_predictions = os.path.join(args.final_ik_folder, f'{args.benchmark}_validation_5PointsEffectors_FinalIK.csv')
        finalik_test_predictions = os.path.join(args.final_ik_folder, f'{args.benchmark}_test_5PointsEffectors_FinalIK.csv')
        print("|= Validation metrics:")
        benchmark_from_csv(model, data_config_path, full_pose_validation_csv, finalik_valid_predictions)
        print("|= Test metrics:")
        benchmark_from_csv(model, data_config_path, full_pose_test_csv, finalik_test_predictions)

    print("Done.")
