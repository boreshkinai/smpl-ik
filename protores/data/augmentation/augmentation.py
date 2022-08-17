import torch
from typing import Optional

from protores.data.augmentation.types import DataTypes


def feature(func):
    func.is_feature = True
    return func


def get_view_shape(dt, tensor):
    if dt not in DataTypes:
        raise Exception("datatype " + dt + " not supported")

    denom = DataTypes[dt]
    assert denom > 0, "invalid number of elements in datatype"
    return (-1, int(tensor.shape[1] / denom), denom)


class BaseAugmentation:
    def __init__(self, features=None) -> None:
        self.features = features

    def _get_callback(self, dt):
        if dt == "Scalar":
            return self.scalar
        elif dt == "Vector2":
            return self.vector2
        elif dt == "Vector3":
            return self.vector3
        elif dt == "Quaternion":
            return self.quaternion
        else:
            assert False, "type not yet supported"

    def init(self, dataset, features):
        self.dataset = dataset
        if self.features is None:
            self.features = features
        assert all(x in features for x in self.features), "one or more feature specified does not exist in the dataset"

        self.callstack = []
        for f in self.features:
            datatype, tensor, slice_indices = self.dataset.get_feature(f)
            self.callstack.append((get_view_shape(datatype, tensor), slice_indices, self._get_callback(datatype), f))

    def transform(self, flattened_batch):
        self.begin_batch(flattened_batch)
        for q in self.callstack:
            view = q[0]
            narrow = q[1]
            callback = q[2]
            feature_name = q[3]
            slice = flattened_batch.narrow(1, narrow[0], narrow[1] - narrow[0])
            reshaped = slice.view(view[0], view[1], view[2])
            if feature_name is None:
                modified = callback(reshaped)
            else:
                modified = callback(reshaped, feature_name=feature_name)
            #assert modified.shape == reshaped.shape, "augmentation should not change the shape of the returned tensors"
            reshaped.copy_(modified)
        return flattened_batch

    def begin_batch(self, batch):
        pass

    def compute(self, batch):
        pass

    def scalar(self, scalar_tensor: torch.Tensor, feature_name: Optional[str] = None) -> torch.Tensor:
        return scalar_tensor

    def vector2(self, vector2_tensor: torch.Tensor, feature_name: Optional[str] = None) -> torch.Tensor:
        return vector2_tensor

    def vector3(self, vector3_tensor: torch.Tensor, feature_name: Optional[str] = None) -> torch.Tensor:
        return vector3_tensor

    def quaternion(self, quaterion_tensor: torch.Tensor, feature_name: Optional[str] = None) -> torch.Tensor:
        return quaterion_tensor


class FeatureAugmentation(BaseAugmentation):
    def __init__(self):
        super().__init__(None)

    def init(self, dataset, features):
        self.dataset = dataset
        if self.features is None:
            self.features = features
        assert all(x in features for x in self.features), "one or more feature specified does not exist in the dataset"

        # methods matching a feature name
        object_methods = [getattr(self, method_name) for method_name in dir(self) if callable(getattr(self, method_name)) and method_name in features]
        object_methods = [o for o in object_methods if o.is_feature is True]
        if len(object_methods)==0:
            raise Exception("this FeatureAugmentation does not declare a method which matches a feature name in the dataset")

        # check that all @feature attributes match a known feature
        all_methods_with_feature_attribute = [method_name for method_name in dir(self) if callable(getattr(self, method_name)) and hasattr(getattr(self, method_name), "is_feature") ]
        assert all(x in features for x in all_methods_with_feature_attribute), "one or more method is decorated with the @feature attribute but the feature is not found in the dataset"

        self.callstack = []
        for f in object_methods:
            datatype, tensor, slice_indices = self.dataset.get_feature(f.__name__)
            self.callstack.append((get_view_shape(datatype, tensor), slice_indices, f, None))
