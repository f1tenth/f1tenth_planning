def load_params(default_params: dict, new_params: dict | str = None) -> dict:
    """
    Update a dictionary of default parameters with new parameters. from a dict or yaml file.
    New parameters are from a dict or yaml file.

    Args:
        default_params (dict): default parameters
        new_params (dict or str, optional): new parameters dict or path to yaml file
    """
    if isinstance(new_params, str):
        import yaml

        with open(new_params, "r") as f:
            new_params = yaml.safe_load(f)

    default_params.update(new_params or {})
    return default_params