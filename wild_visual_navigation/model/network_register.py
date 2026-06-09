#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation.model import *
import inspect
import torch


def create_registery():
    """Creates register of avialble classes to instantiate based on global scope.

    Returns:
        register (str: class): Contains all avialble classes from model module
        cfg_keys (str: str): Converts the classnames to lower_case to get correct cfg parameters.
    """

    # Finds all classes available
    register = {k: v for k, v in globals().items() if inspect.isclass(v)}

    # Changes the keys to access the configuration parameters
    # SomeModelNAME -> some_model_name_cfg
    cfg_keys = {}
    for key in register.keys():
        previous_large = False
        cfg_key = []
        for j, k in enumerate(key):
            if k.isupper() and not previous_large and j != 0:
                cfg_key.append("_")
            if k.isupper():
                cfg_key.append(k.lower())
                previous_large = True
            if k.islower():
                cfg_key.append(k)
                previous_large = False

        cfg_key = "".join(cfg_key)
        cfg_keys[key] = cfg_key + "_cfg"

    return register, cfg_keys


def get_model(model_cfg: dict) -> torch.nn.Module:
    """Returns the instantiated model

    Args:
        model_cfg (dict): Contains "name": (str) ClassName; "class_name_cfg": (Dict).
    Returns:
        model (nn.Module): Some torch module
    """
    name = model_cfg["name"]
    register, cfg_keys = create_registery()
    model = register[name](**model_cfg[cfg_keys[name]])
    return model

def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")


if __name__ == "__main__":
    from wild_visual_navigation import WVN_ROOT_DIR
    from wild_visual_navigation.utils import load_yaml
    from os.path import join

    exp = load_yaml(join(WVN_ROOT_DIR, "cfg/exp/exp.yaml"))
    get_model(exp["model"])
