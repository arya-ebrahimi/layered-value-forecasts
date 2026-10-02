import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_franka_example() -> dict:
    """Creates a random input example for the Franka policy."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the book and place it in the book holder",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """
    Converts inputs from the franka_raw dataset (single-arm Franka Panda, one external camera
    "ext1" + one wrist camera) to the model's expected format. Used for both training and inference.

    state: 7-dim end-effector pose + gripper: ee_pos(3) + axis-angle(3) + gripper width(1).
        NOT joint angles -- matches src/openpi/training/rlt/envs.py's `_make_state`
        (LIBERO/pi05_libero's own convention), which the real-robot bridge and Franka
        teleop/inference client already use, so this checkpoint deploys through that bridge
        unmodified. See examples/franka_raw/convert_franka_raw_to_lerobot.py for the exact
        conversion from the raw hdf5 (ee_pos_t/ee_pos_q).
    actions: 7-dim (6 arm delta dims, dims 3-4 always 0 for this task, + a discrete
        gripper trigger in the last dim: -1=close, 0=no-op, +1=open).
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Only one wrist camera exists; pad the second wrist slot with zeros.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for the pi0 model, not pi0-FAST.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the franka_raw action space. Inference only."""

    def __call__(self, data: dict) -> dict:
        # Actions are padded to the model action dimension; return only the 7 real dims
        # (6 arm delta dims + 1 discrete gripper trigger).
        return {"actions": np.asarray(data["actions"][:, :7])}
