from pathlib import Path

import joblib
import matplotlib as mpl
import pandas as pd
import yaml

from gami_tree_reproduce.utils import get_project_paths

mpl.use("Agg")
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
    splits = splits.apply(lambda lst: "&".join(lst))

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
    _Generate dataframe with information on calculated scores/effects

    For each feature, say X1 or X1&X2(interaction) the frame consists of input (grid values where effects are estimated on)
    and scores (grid values with estimated effects). For interactions scores will be a matrix which is stores as list of lists

    Args:
        ebm_model : ebm_model from pytorch Gaminet version as stored in model-wrapper._model

    Returns: pd.DataFrame
        feature_name:	|<feature_name1>|<feature_name2>|...
        ----------------|---------------|---------------|...
        input:		    | [values]	    | [<values>]	|...
        scores:		    | [values]	    | [<values>]	|...
    """
    terms = ebm_model.term_features_
    idx_single = [idx for idx, tupl in enumerate(terms) if len(tupl) == 1]
    idx_double = [idx for idx, tupl in enumerate(terms) if len(tupl) == 2]

    single_features = [
        term for term in terms if len(term) == 1
    ]  # for main effects, [(0,), (1,),...]
    double_features = [
        term for term in terms if len(term) == 2
    ]  # for interaction effects, [(0,3), (1,2), ...]

    scores = ebm_model.term_scores_
    single_scores = [scores[idx].tolist() for idx in idx_single]
    single_scores_dict = {
        "X" + str(idx + 1): score for idx, score in enumerate(single_scores)
    }

    double_scores = [scores[idx].tolist() for idx in idx_double]
    double_scores_dict = {
        "X" + str(tupl[0] + 1) + "&" + "X" + str(tupl[1] + 1): scores
        for tupl, scores in zip(double_features, double_scores, strict=True)
    }

    single_input_dict = {
        term_idx: ebm_model.histogram_edges_[term_idx].tolist()
        for term_idx in range(len(single_features))
    }
    double_input_dict = {
        "X" + str(tupl[0] + 1) + "&" + "X" + str(tupl[1] + 1): [
            single_input_dict[tupl[0]],
            single_input_dict[tupl[1]],
        ]
        for tupl in double_features
    }
    single_input_dict = {
        "X" + str(key + 1): value for key, value in single_input_dict.items()
    }

    single_dict = {
        feature: {
            "inputs": single_input_dict[feature],
            "scores": single_scores_dict[feature],
        }
        for feature in single_input_dict
    }
    double_dict = {
        feature: {
            "inputs": double_input_dict[feature],
            "scores": double_scores_dict[feature],
        }
        for feature in double_scores_dict
    }
    total = single_dict | double_dict

    return pd.DataFrame({"feature": key, **value} for key, value in total.items())


def get_gaminet_effect(gaminet_model) -> pd.DataFrame:
    """
    _Generate dataframe with information on calculated scores/effects

    For each feature, say X1 or X1&X2(interaction) the frame consists of input (grid values where effects are estimated on)
    and scores (grid values with estimated effects). For interactions scores will be a matrix which is stores as list of lists

    Args:
        gaminet_model : gaminet_model from pytorch Gaminet version as stored in model-wrapper._model

    Returns: pd.DataFrame
        feature_name:	|<feature_name1>|<feature_name2>|...
        ----------------|---------------|---------------|...
        input:		    | [values]	    | [<values>]	|...
        scores:		    | [values]	    | [<values>]	|...
    """

    global_dict = gaminet_model.data_dict_global_

    single_features = {
        feature: entry for feature, entry in global_dict.items() if "inputs" in entry
    }
    single_features = {
        feature: {
            "inputs": single_features[feature]["inputs"].tolist(),
            "scores": single_features[feature]["outputs"].tolist(),
        }
        for feature in single_features
    }

    ## entries in outputs is a matrix where row/column values correspond to input1/2 values
    double_features = {
        feature.replace(" x ", "&"): entry
        for feature, entry in global_dict.items()
        if "input1" in entry
    }
    double_features = {
        feature: {
            "inputs": [
                double_features[feature]["input1"].flatten().tolist(),
                double_features[feature]["input2"].flatten().tolist(),
            ],
            "scores": [double_features[feature]["outputs"].tolist()],
        }
        for feature in double_features
    }
    total = single_features | double_features

    return pd.DataFrame({"feature": key, **value} for key, value in total.items())


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
        df_effects[df_effects["feature"].str.contains("&")].to_parquet(
            Path(assets_ebm_effect, data_name + "_interact").with_suffix(".pq")
        )
        df_effects[~df_effects["feature"].str.contains("&")].to_parquet(
            Path(assets_ebm_effect, data_name + "_main").with_suffix(".pq")
        )

    elif inducer_name == "gaminet":
        df_importance = get_gaminet_importance(model._model)
        df_effects = get_gaminet_effect(model._model)
        df_importance.to_parquet(
            Path(assets_gaminet_importance, data_name).with_suffix(".pq")
        )
        df_effects[df_effects["feature"].str.contains("&")].to_parquet(
            Path(assets_gaminet_effect, data_name + "_interact").with_suffix(".pq")
        )
        df_effects[~df_effects["feature"].str.contains("&")].to_parquet(
            Path(assets_gaminet_effect, data_name + "_main").with_suffix(".pq")
        )
