from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from gami_tree_reproduce.utils import get_project_paths

project_paths = get_project_paths()


data_folder = Path(project_paths["data_preprocessed"])
assets_conf_experiment = Path(project_paths["assets_conf_experiments"])

assets_ebm_effect = Path(project_paths["assets_effects"], "ebm")
assets_ebm_effect.mkdir(exist_ok=True, parents=True)
assets_gaminet_effect = Path(project_paths["assets_effects"], "gaminet")
assets_gaminet_effect.mkdir(exist_ok=True, parents=True)

assets_ebm_importance = Path(project_paths["assets_importance"], "ebm")
assets_ebm_importance.mkdir(exist_ok=True, parents=True)
assets_gaminet_importance = Path(project_paths["assets_importance"], "gaminet")
assets_gaminet_importance.mkdir(exist_ok=True, parents=True)

experiments = [folder for folder in assets_conf_experiment.iterdir() if folder.is_dir()]


def get_data(data_name: str, train_test_val: str = "train") -> pd.DataFrame:
    path = project_paths["data_preprocessed"]
    path = Path(path, data_name, data_name + "_" + train_test_val).with_suffix(".pq")

    return pd.read_parquet(path)


def get_metadata(foldername: str) -> tuple:
    data, response_model, inducer, _ = tuple(foldername.split("_"))

    with Path(project_paths["assets_conf_data"], data).with_suffix(".yaml").open() as f:
        data_config = yaml.safe_load(f)

    data_name = data + "_" + response_model
    return data_name, inducer, data_config


def get_ebm_importance(ebm_model) -> pd.DataFrame:
    """
    Generate a dataframe with information on feature importance

    Args:
        ebm_model : ebm_model from interpretML as stored in model-wrapper._model

    Returns:
        pd.DataFrame:
            |<feature_name1>|<importance_score1>|
            |---------------|-------------------|
            |<feature_name2>|<importance_score2>|
            |   ...         |   ...             |
    """

    ebm_importance = pd.DataFrame(ebm_model.explain_global().data())
    splits = ebm_importance.names.str.split("&")
    splits = splits.apply(
        lambda string_list: [
            "X" + str(int(string.replace("feature_", "")) + 1) for string in string_list
        ]
    )
    splits = splits.apply("&".join)

    ebm_importance.loc[:, "names"] = splits
    ebm_importance = ebm_importance.drop("type", axis=1)
    ebm_importance.columns = ["feature", "importance"]

    return ebm_importance


def get_gaminet_importance(gaminet_model) -> pd.DataFrame:
    """
    Generate a dataframe with information on feature importance

    Args:
        gaminet_model : gaminet_model from pytorch Gaminet version as stored in model-wrapper._model

    Returns:
        pd.DataFrame:
            |<feature_name1>|<importance_score1>|
            |---------------|-------------------|
            |<feature_name2>|<importance_score2>|
            |   ...         |   ...             |
    """

    global_explanation = gaminet_model.global_explain()
    importances = [
        float(global_explanation[key]["importance"]) for key in global_explanation
    ]
    features = list(global_explanation.keys())
    features = [feature.replace(" x ", "&") for feature in features]

    return pd.DataFrame({"feature": features, "importance": importances})


def get_ebm_effects(ebm_model) -> pd.DataFrame:
    """
    _Generate dataframe with information on calculated effects

    For each feature, say X1 or X1&X2(interaction) the frame consists of input-grids (grid values where effects are estimated on)
    and effects (grid values with estimated effects). For interactions effects will be a matrix which is stores as list of lists

    Args:
        ebm_model : ebm_model from pytorch Gaminet version as stored in model-wrapper._model

    Returns: pd.DataFrame
        feature_name:	|<feature_name1>|<feature_name2>|...
        ----------------|---------------|---------------|...
        grid:		    | [values]	    | [<values>]	|...
        effect:		    | [values]	    | [<values>]	|...
    """
    single_bounds = ebm_model.feature_bounds_.tolist()
    features = ebm_model.term_features_

    main_effect_grid_len = int(ebm_model.max_bins)
    interaction_effect_grid_len = int(ebm_model.max_interaction_bins)

    main_effects_grids = [
        np.linspace(lst[0], lst[1], main_effect_grid_len) for lst in single_bounds
    ]
    interaction_effects_grids_per_feature = [
        np.linspace(lst[0], lst[1], interaction_effect_grid_len)
        for lst in single_bounds
    ]  # cannot use from main effect since different grid granularity

    main_effects_name = ["X" + str(tupl[0] + 1) for tupl in features if len(tupl) == 1]
    interaction_effects_name = [
        "X" + str(tupl[0] + 1) + "&X" + str(tupl[1] + 1)
        for tupl in features
        if len(tupl) == 2
    ]

    interaction_effects_grids = [
        [
            interaction_effects_grids_per_feature[tupl[0]],
            interaction_effects_grids_per_feature[tupl[1]],
        ]
        for tupl in features
        if len(tupl) == 2
    ]
    main_effects = [
        score_array for score_array in ebm_model.term_scores_ if score_array.ndim == 1
    ]
    interaction_effect = [
        score_array.tolist()
        for score_array in ebm_model.term_scores_
        if score_array.ndim == 2
    ]

    assert (
        len(main_effects_grids)
        == len(main_effects)
        == len(main_effects_name)
        == ebm_model.n_features_in_
    )
    assert (
        len(interaction_effects_grids)
        == len(interaction_effect)
        == len(interaction_effects_name)
        == ebm_model.interactions
    )

    # convert to list of lists
    main_effects_grids = [grid_arr.tolist() for grid_arr in main_effects_grids]
    main_effects_grids = [[elem] for elem in main_effects_grids]
    interaction_effects_grids = [
        [gridlist[0].tolist(), gridlist[1].tolist()]
        for gridlist in interaction_effects_grids
    ]

    main_effects = [grid_arr.tolist() for grid_arr in main_effects]
    main_effects = [[elem] for elem in main_effects]

    return pd.DataFrame(
        {
            "feature": main_effects_name + interaction_effects_name,
            "grid": main_effects_grids + interaction_effects_grids,
            "effect": main_effects + interaction_effect,
        }
    )


def get_gaminet_effect(gaminet_model) -> pd.DataFrame:
    """
    _Generate dataframe with information on calculated  effects

    For each feature, say X1 or X1&X2(interaction) the frame consists of input-grid (grid values where effects are estimated on)
    and effects (grid values with estimated effects). For interactions effects will be a matrix which is stores as list of lists

    Args:
        gaminet_model : gaminet_model from pytorch Gaminet version as stored in model-wrapper._model

    Returns: pd.DataFrame
        feature_name:	|<feature_name1>|<feature_name2>|...
        ----------------|---------------|---------------|...
        grid:		    | [values]	    | [<values>]	|...
        effect:		    | [values]	    | [<values>]	|...
    """

    global_dict = gaminet_model.data_dict_global_
    interaction_dict = {
        key: value for key, value in global_dict.items() if " x " in key
    }
    main_dict = {key: value for key, value in global_dict.items() if " x " not in key}

    main_effects_names = list(main_dict)
    interaction_effects_names = list(interaction_dict)

    main_effects_grids = [value["inputs"].tolist() for value in main_dict.values()]
    interaction_effects_grids = [
        [value["input1"].flatten().tolist(), value["input2"].flatten().tolist()]
        for value in interaction_dict.values()
    ]

    main_effects = [value["outputs"].tolist() for value in main_dict.values()]
    interaction_effects = [
        value["outputs"].tolist() for value in interaction_dict.values()
    ]

    assert (
        len(main_effects_grids)
        == len(main_effects_names)
        == len(main_effects)
        == gaminet_model.nfeature_num_
    )
    assert (
        len(interaction_effects_grids)
        == len(interaction_effects_names)
        == len(interaction_effects)
        == gaminet_model.n_interactions_
    )

    # convert to list of lists
    # = [[elem] for elem in main_effects]
    main_effects_grids = [[elem] for elem in main_effects_grids]
    main_effects = [[elem] for elem in main_effects]

    return pd.DataFrame(
        {
            "feature": main_effects_names
            + [name.replace(" x ", "&") for name in interaction_effects_names],
            "grid": main_effects_grids + interaction_effects_grids,
            "effect": main_effects + interaction_effects,
        }
    )


# generate data
for experiment in experiments:
    ## read model and get feature importance data
    model = joblib.load(Path(experiment, "model.gz"))
    data_name, inducer_name, data_config = get_metadata(experiment.stem)

    if inducer_name == "ebm":
        df_importance = get_ebm_importance(model._model)
        df_effects = get_ebm_effects(model._model)
        df_importance.to_parquet(
            Path(assets_ebm_importance, data_name).with_suffix(".pq")
        )
        df_effects.to_parquet(Path(assets_ebm_effect, data_name).with_suffix(".pq"))

    elif inducer_name == "gaminet":
        df_importance = get_gaminet_importance(model._model)
        df_effects = get_gaminet_effect(model._model)
        df_importance.to_parquet(
            Path(assets_gaminet_importance, data_name).with_suffix(".pq")
        )
        df_effects.to_parquet(Path(assets_gaminet_effect, data_name).with_suffix(".pq"))
