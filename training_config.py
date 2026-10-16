"""Construct SFT settings across the verified old and new loss-selector APIs."""

import inspect


def make_sft_config(config_type, **settings):
    # TRL 0.22.2 delegates to the model's ordinary NLL and has no loss_type field.
    # Later versions can default to another loss; select NLL only when declared.
    if "loss_type" in inspect.signature(config_type).parameters:
        settings["loss_type"] = "nll"
    return config_type(**settings)
