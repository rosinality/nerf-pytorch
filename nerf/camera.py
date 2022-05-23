import copy
import warnings

import numpy as np
import torch

BROADCAST_TYPES = (float, int, list, tuple, torch.Tensor, np.ndarray)


def to_tensor(input, dtype=torch.float32, device="cpu"):
    out = torch.as_tensor(input, dtype=dtype, device=device)

    if out.dim() == 0:
        out = out.view(1)

    return out


def to_broadcasted_tensor(*args, dtype=torch.float32, device="cpu"):
    args_1d = [to_tensor(c, dtype, device) for c in args]

    sizes = [c.shape[0] for c in args_1d]
    batch_size = max(sizes)

    args_nd = []
    for c in args_1d:
        if c.shape[0] != 1 and c.shape[0] != batch_size:
            raise ValueError(f"Got non-broadcastable sizes {sizes!r}")

        expand_sizes = (batch_size,) + (-1,) * len(c.shape[1:])
        args_nd.append(c.expand(*expand_sizes))

    if len(args) == 1:
        args_nd = args_nd[0]

    return args_nd


class TensorAccessor:
    def __init__(self, class_object, index):
        self.__dict__["class_object"] = class_object
        self.__dict__["index"] = index

    def __setattr__(self, name, value):
        v = getattr(self.class_object, name)

        if not torch.is_tensor(v):
            raise AttributeError(
                f"Can only set values on attributes which are tensors; got {type(v)!r}"
            )

        value = torch.as_tensor(value, device=v.device, dtype=v.dtype)
        value.requires_grad = v.requires_grad

        if v.dim() > 1 and value.dim() > 1 and value.shape[1:] != v.shape[1:]:
            raise ValueError(
                f"Expected value to have shape {v.shape!r}; got {value.shape!r}"
            )

        if (
            v.dim == 0
            and isinstance(self.index, slice)
            and len(value) != len(self.index)
        ):
            raise ValueError(
                f"Expected value to have len {len(self.index)!r}; got {len(value)!r}"
            )

        self.class_object.__dict__[name][self.index] = value

    def __getattr__(self, name):
        if hasattr(self.class_object, name):
            return self.class_object.__dict__[name][self.index]

        else:
            raise AttributeError(
                f"Attribute {name} not found on {self.class_object.__class__.__name__!r}"
            )


class TensorProperty:
    def __init__(self, dtype=torch.float32, device="cpu", **kwargs):
        super().__init__()

        self.device = device
        self.batch_size = 0

        if kwargs is not None:
            args_to_broadcast = {}

            for k, v in kwargs.items():
                if v is None or isinstance(v, (str, bool)):
                    setattr(self, k, v)

                elif isinstance(v, BROADCAST_TYPES):
                    args_to_broadcast[k] = v

                else:
                    warnings.warn(f"Arg {k} with type {type(v)!r} is not broadcastable")

            names = args_to_broadcast.keys()
            values = tuple(v for v in args_to_broadcast.values())

            if len(values) > 0:
                broadcasted_values = to_broadcasted_tensor(*values, device=device)

                for i, n in enumerate(names):
                    setattr(self, n, broadcasted_values[i])

                    if self.batch_size == 0:
                        self.batch_size = broadcasted_values[i].shape[0]

    def __len__(self):
        return self.batch_size

    def isempty(self):
        return self.batch_size == 0

    def __getitem__(self, index):
        if isinstance(index, int):
            if index >= len(self) or index < -len(self):
                raise IndexError("Index is out of range")

            else:
                index = slice(index, None, len(self))

        subset = self.__class__()

        for k, v in self.__dict__.items():
            if k in ("device", "batch_size"):
                continue

            v_sub = v[index]
            setattr(subset, k, v_sub)

        subset.device = self.device
        subset.batch_size = len(v_sub)

        return subset

    def to(self, device="cpu"):
        for k in dir(self):
            v = getattr(self, k)

            if k == "device":
                setattr(self, k, device)

            if hasattr(v, "to"):
                setattr(self, k, v.to(device))

        return self

    def clone(self):
        other = self.__class__()

        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                v_clone = v.clone()

            else:
                v_clone = copy.deepcopy(v)

            setattr(other, k, v_clone)

        return other


class PerspectiveCamera(TensorProperty):
    def __init__(
        self,
        R=None,
        T=None,
        focal_length=1,
        principal_point=((0.0, 0.0),),
        device="cpu",
        image_size=None,
    ):
        if R is None:
            R = torch.eye(3).unsqueeze(0)

        if T is None:
            T = torch.zeros(1, 3)

        kwargs = {"image_size": image_size} if image_size is not None else {}

        super().__init__(
            device=device,
            R=R,
            T=T,
            focal_length=focal_length,
            principal_point=principal_point,
            **kwargs,
        )

        if image_size is not None:
            if (self.image_size < 1).any():
                raise ValueError("image size should be larger than 0")

        else:
            self.image_size = None

    @property
    def K(self):
        intrinsic = self.R.new_zeros(self.R.shape[0], 3, 3)
        intrinsic[:, 2, 2] = 1
        intrinsic[:, 0, 0] = self.focal_length
        intrinsic[:, 1, 1] = self.focal_length
        intrinsic[:, 0, 2] = 0.5 * self.image_size[:, 1]
        intrinsic[:, 1, 2] = 0.5 * self.image_size[:, 0]

        return intrinsic

    def camera_to_world(self):
        bottom = torch.tensor((0, 0, 0, 1), dtype=torch.float32).unsqueeze(0)
        bottom = bottom.repeat(self.R.shape[0], 1, 1)
        poses = torch.cat((self.R, self.T.unsqueeze(-1)), -1)
        poses = torch.cat((poses, bottom), -2)

        return poses

    @staticmethod
    def from_compact_representation(mat):
        R = mat[:, :3, :3]
        T = mat[:, :3, 3]
        focal_length = mat[:, 2, 4]
        image_size = mat[:, :2, 4]

        return PerspectiveCamera(
            R=R, T=T, focal_length=focal_length, image_size=image_size
        )

    def compact_representation(self):
        hwf = torch.cat(
            (self.image_size, self.focal_length.unsqueeze(-1)), -1
        ).unsqueeze(-1)
        mat = torch.cat((self.R, self.T.unsqueeze(-1), hwf), -1)

        return mat
