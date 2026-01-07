import re
from pathlib import Path

from gami_tree_reproduce.plot.plot_utils import plot_response, save_fig
from gami_tree_reproduce.utils import ASSETS_SIM_CONF, DATA

datasets = list(Path(DATA).glob("sim*_mod*[cr].pq"))
msg = "It appears the are no .pq files in data folder. \n You might want to run `make create_sim_data`."
assert len(datasets) > 0, msg

pattern_c = re.compile(r"sim\d+_mod\d+c")
pattern_r = re.compile(r"sim\d+_mod\d+r")

datasets_c = [path for path in datasets if pattern_c.search(path.name)]
datasets_r = [path for path in datasets if pattern_r.search(path.name)]

simulation_settings = list(Path(ASSETS_SIM_CONF).glob("*.yaml"))
simulation_names = [s.stem for s in simulation_settings]

datasets_c_dict = {
    sim: [dataset for dataset in datasets_c if sim in str(dataset)]
    for sim in simulation_names
}
datasets_r_dict = {
    sim: [dataset for dataset in datasets_r if sim in str(dataset)]
    for sim in simulation_names
}


sim1_data = {
    "classification": datasets_c_dict["sim1"],
    "regression": datasets_r_dict["sim1"],
}
sim2_data = {
    "classification": datasets_c_dict["sim2"],
    "regression": datasets_r_dict["sim2"],
}
sim3_data = {
    "classification": datasets_c_dict["sim3"],
    "regression": datasets_r_dict["sim3"],
}
sim4_data = {
    "classification": datasets_c_dict["sim4"],
    "regression": datasets_r_dict["sim4"],
}


fig, ax = plot_response(
    sim1_data["classification"], subplots_kwargs={"constrained_layout": True}
)
fig.suptitle("Data Distribution for Simulation 1(Classification Task)")
save_fig(fig, "sim1_yc")

fig, ax = plot_response(
    sim2_data["classification"], subplots_kwargs={"constrained_layout": True}
)
fig.suptitle("Data Distribution for Simulation 2(Classification Task)")
save_fig(fig, "sim2_yc")

fig, ax = plot_response(
    sim3_data["classification"], subplots_kwargs={"constrained_layout": True}
)
fig.suptitle("Data Distribution for Simulation 3(Classification Task)")
save_fig(fig, "sim3_yc")

fig, ax = plot_response(
    sim4_data["classification"], subplots_kwargs={"constrained_layout": True}
)
fig.suptitle("Data Distribution for Simulation 4(Classification Task)")
save_fig(fig, "sim4_yc")

fig, ax = plot_response(
    sim1_data["regression"],
    plot_method_kwargs={"density": True},
    subplots_kwargs={"constrained_layout": True},
)
fig.suptitle("Data Distribution for Simulation 1(Regression Task)")
save_fig(fig, "sim1_yr")

fig, ax = plot_response(
    sim2_data["regression"],
    plot_method_kwargs={"density": True},
    subplots_kwargs={"constrained_layout": True},
)
fig.suptitle("Data Distribution for Simulation 2(Regression Task)")
save_fig(fig, "sim2_yr")

fig, ax = plot_response(
    sim3_data["regression"],
    plot_method_kwargs={"density": True},
    subplots_kwargs={"constrained_layout": True},
)
fig.suptitle("Data Distribution for Simulation 3(Regression Task)")
save_fig(fig, "sim3_yr")

fig, ax = plot_response(
    sim4_data["regression"],
    plot_method_kwargs={"density": True},
    subplots_kwargs={"constrained_layout": True},
)
fig.suptitle("Data Distribution for Simulation 1(Regression Task)")
save_fig(fig, "sim4_yr")
