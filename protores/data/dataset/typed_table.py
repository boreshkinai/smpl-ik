from itertools import chain

import torch
import re
import json
import glob
import os
import pandas as pd

from torch.utils.data import Dataset
from protores.data.augmentation.types import DataTypes
from sklearn.model_selection import ParameterGrid


class FlatTypedColumnDataset(Dataset):
    """
    No sequence information here. Just a flat array of frames. Built from a Pandas dataframe.
    No split file. No subset.
    """
    def __init__(self, dataframe, config):
        self.data = None
        self._calculated_features = None
        self.df = dataframe
        self.config = config
        self.reset_internals(self.df)

    def reset_internals(self, dataframe):
        self.df = dataframe
        self.is_valid = True
        if len(dataframe.index) == 0:
            self.is_valid = False
            return
        self.data = torch.FloatTensor(dataframe.values).pin_memory()
        self._calculated_features = []
        self._cache(dataframe.columns.values.tolist())

    def _cache(self, columns):
        self.auto_detect_features(columns)
        self._slices = self.slices()
        self._slice_indices = self.slice_indices()
        self._datatype = self.datatype()
        self._features = [n[0] for n in self.features()]
        self._columns = columns
        self._transforms = []

    def get_config(self, key):
        return self.config[key]

    def get_slice_indices(self, feature_name):
        idx = self._features.index(feature_name)
        return self._slice_indices[idx]

    def get_feature(self, feature_name):
        idx = self._features.index(feature_name)
        return self._datatype[idx], self._slices[idx], self._slice_indices[idx]

    def add_transform(self, transform):
        transform.init(self, [x[0] for x in self.features()])
        self._transforms.append(transform)

    def add_calculated_feature(self, func, input_features, output_feature, feature_type="Scalar"):
        if not isinstance(input_features, list):
            input_features = [input_features]

        slices = [self.get_feature(x)[1][0].unsqueeze(0) for x in input_features]
        # invoke the function so we can tell the size of the new feature
        calculated = func(slices)[0]
        self.config['features'][output_feature] = {
            'types': [feature_type] * (calculated.shape[0] // self._get_size_of_feature(feature_type))
        }

        default_tensor = calculated.cpu()
        tiled = default_tensor.repeat(self.data.shape[0], 1)
        self.data = torch.cat((self.data, tiled), 1).cpu()

        new_column_names = self._columns
        for i in range(calculated.shape[0] // self._get_size_of_feature(feature_type)):
            for suffix in self._get_suffix_of_feature(feature_type):
                new_column_names.append(output_feature + "_" + suffix)

        self._cache(new_column_names)

        # pack some parameters necessary to invoke the calculated feature
        # (list of input indices, output indices, function)
        input_slice_indices = [self.get_slice_indices(x) for x in input_features]
        output_slice_indices = self.get_slice_indices(output_feature)
        self._calculated_features.append((input_slice_indices, output_slice_indices, func))

    def slices(self):
        return [self.data.narrow(1, slice[0], slice[1] - slice[0]) for slice in self.slice_indices()]

    def slice_indices(self):
        startIdx = 0
        endIdx = 0
        slices = list()
        for feature in self.features():
            startIdx = endIdx
            for columnType in feature[1]['types']:
                endIdx += DataTypes[columnType]
            slices.append((startIdx, endIdx))
        return slices

    def auto_detect_datatype(self, prefix, columns):
        matches = [c for c in columns if c.startswith(prefix)]
        if (len(matches) == 1):
            return ["Scalar"] * int(len(matches) / self._get_size_of_feature("Scalar"))
        suffixes = set([s.split("_")[-1] for s in matches])
        if suffixes == set(self._get_suffix_of_feature("Vector2")) and len(matches) % 2 == 0:
            return ["Vector2"] * int(len(matches) / self._get_size_of_feature("Vector2"))
        if suffixes == set(self._get_suffix_of_feature("Vector3")) and len(matches) % 3 == 0:
            return ["Vector3"] * int(len(matches) / self._get_size_of_feature("Vector3"))
        if suffixes == set(self._get_suffix_of_feature("Quaternion")) and len(matches) % 4 == 0:
            return ["Quaternion"] * int(len(matches) / self._get_size_of_feature("Quaternion"))
        return ["Scalar"] * len(matches)

    def auto_detect_features(self, columns):
        feature_names = list(set([x.split("_")[0] for x in columns]))
        datatypes = [self.auto_detect_datatype(x, columns) for x in feature_names]
        config = {"features": {}}
        for feature in zip(feature_names, datatypes):
            config['features'][feature[0]] = {'types': feature[1]}
        return config

    def features(self):
        return [(k, v) for k, v in self.config['features'].items()]

    def datatype(self):
        return [v['types'][0] if len(set(v['types'])) == 1 else "Mixed" for k, v in self.config['features'].items()]

    def selector_index(self, regex):
        p = re.compile(regex)
        matches = [self._columns.index(x) for x in self._columns if p.match(x) is not None]
        if len(matches) == 0:
            raise Exception("no feature matches " + regex)
        return matches

    def get_feature_indices(self, prefix, substr):
        if type(prefix) == str and type(substr) == str:
            return self.selector_index(prefix + "_" + substr + "_")
        elif type(prefix) == list and type(substr) == list:
            possible_values = ParameterGrid({"prefix": prefix, "substr": substr})
            return list(chain.from_iterable(
                [self.selector_index(values["prefix"] + "_" + values["substr"] + "_") for values in
                 possible_values]))
        else:
            raise ValueError("Invalid type passed into get_feature_indices, only supports str and list[str]")

    def select_features(self, *argv):
        components = []
        for i in argv:
            if not isinstance(i, list):
                i = [i]
            components.append(i)
        returned_list = []
        for component in components:
            indices = []
            for feature in component:
                indices.extend(self.selector_index(feature))
            returned_list.append(indices)
        if len(returned_list) == 1:
            return torch.LongTensor(returned_list[0])
        return [torch.LongTensor(x) for x in returned_list]

    def __getitem__(self, index):
        is_batch = True
        if not isinstance(index, list):
            index = [index]
            is_batch = False

        batch = self.data[index]

        for input_indice_list, target_indices, func in self._calculated_features:
            input_tensors = [batch.narrow(1, i[0], i[1] - i[0]) for i in input_indice_list]
            result = func(input_tensors)
            batch.narrow(1, target_indices[0], target_indices[1] - target_indices[0]).copy_(result)

        for xfo in self._transforms:
            batch = xfo.transform(batch)

        if not is_batch:
            batch = torch.squeeze(batch, dim=0)

        return batch

    def __len__(self):
        return self.data.shape[0]

    def _get_size_of_feature(self, feature_type):
        if feature_type == "Scalar":
            return 1
        elif feature_type == "Vector2":
            return 2
        elif feature_type == "Vector3":
            return 3
        elif feature_type == "Quaternion":
            return 4
        else:
            assert False, "type not yet supported"

    def _get_suffix_of_feature(self, feature_type):
        if feature_type == "Scalar":
            return ["V"]
        elif feature_type == "Vector2":
            return ["X", "Y"]
        elif feature_type == "Vector3":
            return ["X", "Y", "Z"]
        elif feature_type == "Quaternion":
            return ["X", "Y", "Z", "W"]
        else:
            assert False, "type not yet supported"


class TypedColumnDataset(FlatTypedColumnDataset):
    def __init__(self, input_path_or_split_struct, subset="", drop_duplicates=True):

        if (isinstance(input_path_or_split_struct, dict)):
            config_path = input_path_or_split_struct['Settings']
            split_file = input_path_or_split_struct['SplitFile']
            self.files = input_path_or_split_struct[subset]
            feather_cache = os.path.join(os.path.dirname(split_file), subset + "_cache.feather")
        else:
            config_path = os.path.join(input_path_or_split_struct, "dataset_settings.json")
            if subset is not None and subset != "":
                input_path_or_split_struct = os.path.join(input_path_or_split_struct, subset)

            self.files = sorted(glob.glob(input_path_or_split_struct + '/**/*.csv', recursive=True))
            feather_cache = os.path.join(input_path_or_split_struct, "cache.feather")

        if (config_path is not None and os.path.isfile(config_path)):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            p = pd.read_csv(self.files[0]).reset_index(drop=True)
            config = self.auto_detect_features(p.columns.values.tolist())

        print("\t Loading ", subset, " DataFrame")
        df = pd.DataFrame()
        if os.path.isfile(feather_cache) == True:
            df = pd.read_feather(feather_cache)
        else:
            sequence_index = 0
            all_sequences = []
            for f in self.files:
                sequence = pd.read_csv(f).reset_index(drop=True)
                sequence["Sequence"] = sequence_index
                all_sequences.append(sequence)
                sequence_index += 1
            df = pd.concat(all_sequences).reset_index(drop=True)
            if drop_duplicates:
                subset = list(df)
                subset.remove("Sequence")
                df = df.drop_duplicates(subset=subset).reset_index(drop=True)
            df = df.astype('float32')
            df.to_feather(feather_cache)
        config["features"]["Sequence"] = {"types": ["Scalar"]}
        print("\t Loaded ", subset, " DataFrame")

        super().__init__(df, config)
        self.sequences = pd.DataFrame([[int(k), v.values] for k,v in df.groupby('Sequence').groups.items()], columns=['Sequence','Indices'])

    @classmethod
    def FromSplit(cls, split):
        return TypedColumnDataset(split, subset="Training"), TypedColumnDataset(split, subset="Validation")


class TypedColumnSequenceDataset(Dataset):
    def __init__(self, input_path_or_split_struct, subset=""):
        self.dataset = TypedColumnDataset(input_path_or_split_struct=input_path_or_split_struct, subset=subset, drop_duplicates=False)
        self.config = self.dataset.config
        self.is_valid = self.dataset.is_valid

    def get_config(self, key):
        return self.dataset.get_config(key)

    def get_slice_indices(self, feature_name):
        return self.dataset.get_slice_indices(feature_name)

    def get_feature(self, feature_name):
        return self.dataset.get_feature(feature_name)

    def add_transform(self, transform):
        return self.dataset.add_transform(transform)

    def add_calculated_feature(self, func, input_features, output_feature):
        return self.dataset.add_calculated_feature(func, input_features, output_feature)

    def slices(self):
        return self.dataset.slices()

    def slice_indices(self):
        return self.dataset.slice_indices()

    def features(self):
        return self.dataset.features()

    def datatype(self):
        return self.dataset.datatype()

    def selector_index(self, regex):
        return self.dataset.selector_index(regex)

    def get_feature_indices(self, prefix, substr):
        return self.dataset.get_feature_indices(prefix, substr)

    def select_features(self, *argv):
        return self.dataset.get_feature_indices(*argv);

    def __getitem__(self, index):
        indices = self.dataset.sequences["Indices"][index]
        sequence_data = self.dataset.__getitem__(indices)
        return sequence_data

    def __len__(self):
        return len(self.dataset.sequences)



