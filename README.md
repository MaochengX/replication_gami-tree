[![CI](https://github.com/MaochengX/LMU-Gami-tree-rep/actions/workflows/continous-integration.yaml/badge.svg)](https://github.com/MaochengX/LMU-Gami-tree-rep/actions/workflows/continous-integration.yaml)

# Prelude

Hello there! We are Sven and Maocheng and like python programming and X-AI.<br>
In this repo we conduct a replication study using so-called Gami-Trees, an by construction inherent explainable and interpretable Machine Learning Algorithm which was proposed by [Hu et.al (2022)](https://arxiv.org/abs/2207.06950).


# Structure
- Currently Gami-Tree is located in a seperate [integration branch](https://github.com/MaochengX/replication_gami-tree/tree/GAMI_Tree_implement), we plan integrate into main when we have the sampe user API.
- We conducted multiple configurations which can be check in `assets/conf` and selecting a configuration file in either `.../data` when cheking the simulation data setup or `.../experiments` with metadata on our training experiments.
- The naming convention for plots is in most cases `sim<data_simulation>_mod<response_model><task>_config<configuration>` where the number for the simulation dataset and configuration can be checked in with entries in `assets/conf/...`.
- We tried to split the whole experimental setup into some independet steps: first the data needs to be generated, then preprocessed (train, validation test split): only then it makes sense to run `run_experiment.py` for example.
- We provide a preliminary Makefile to orchestrate the several steps for replication.
- Important parameters are stored in `conf/data/` for the data generation and `conf/inducer` for inducer parameters.
## Results
We consider the following response models proposed by [Hu et.al](https://arxiv.org/abs/2207.06950):

$$
g_1(x) = \sum_{j=1}^{5} x_j + \sum_{j=6}^{8} 0.5\,x_j^2 + \sum_{j=9}^{10} x_j \mathbb{1}(x_j>0) + \sum_{j=1}^{10}\sum_{k=j+1}^{10} 0.2\,x_jx_k
$$

$$
\begin{aligned}
g_2(x) = \sum_{j=1}^{5} x_j + \sum_{j=6}^{8} 0.5\,x_j^2 + \sum_{j=9}^{10} x_j \mathbb{1}(x_j>0) + 0.25\,x_1x_2 + \\
  0.25\,x_1x_3^2 + 0.25\,x_4^2x_5^2 + \exp(x_4x_6/3) + x_5x_6 \mathbb{1}(x_5>0)\mathbb{1}(x_6>0) + \\
  \text{clip}(x_7+x_8,-1,0) + \text{clip}(x_7x_9,-1,1) + \mathbb{1}(x_8>0)\mathbb{1}(x_9>0)
\end{aligned}
$$

$$
\begin{aligned}
g_3(x) = \sum_{j=1}^{5} x_j + \sum_{j=6}^{8} 0.5\,x_j^2 + \sum_{j=9}^{10} x_j \mathbb{1}(x_j>0) + \\
  0.25\,x_1^2 x_2^2 + 2 (x_3 - 0.5)_+ + (x_4 - 0.5)_+ + \\
  0.5 \sin(\pi x_5)\sin(\pi x_6) + 0.5 \sin(\pi(x_7 + x_8))
\end{aligned}
$$

$$
\begin{aligned}
g_4(x) = \sum_{j=1}^{5} x_j + \sum_{j=6}^{8} 0.5\,x_j^2 + \sum_{j=9}^{10} x_j \mathbb{1}(x_j>0) + \\
  x_1 x_2 + x_1 x_3 + x_2 x_3 + 0.5 x_1 x_2 x_3 + \\
  x_4 x_5 + x_4 x_6 + x_5 x_6 + 0.5 \mathbb{1}(x_4>0) x_5 x_6
\end{aligned}
$$
<!----------------------------------------------------------------------------------------->
<!----------------------------------------------------------------------------------------->
For each Response Model we observed the following Interaction Effects:
<!----------------------------------------------------------------------------------------->
### $g_1$
#### Interaction Effects ($\rho = 0, n=50K$)
##### Gami-Net
The only interaction chosen by Gami-Net was for the variables $x_{10}$ and $x_{14}$ altough $x_{14}$ was not part of the true underlying model:
<p align="center">
  <img src="assets/plots/effects/gaminet/sim2_mod1r_config1/interact_X10-X14.png" width="600">
</p>

##### EBM
<p align="center">
  <img src="assets/plots/effects/ebm/sim2_mod1r_config1/interact_all.png", width = "600">
</p>

#### Interaction Effects ($\rho = 0.5, n=50K$)
##### Gami-Net
No Interactions were selected.
##### EBM
<p align="center">
  <img src="assets/plots/effects/ebm/sim4_mod1r_config1/interact_all.png" width="600">
</p>
<!----------------------------------------------------------------------------------------->

### $g_2$
#### Interaction Effects ($\rho = 0, n=50K$)
##### Gami-Net
<p align="center">
  <img src="assets/plots/effects/gaminet/sim2_mod2r_config1/interact_all.png" width="600">
</p>

##### EBM
<p align="center">
  <img src="assets/plots/effects/ebm/sim2_mod2r_config1/interact_all.png" width="600">
</p>

#### Interaction Effects ($\rho = 0.5, n=50K$)
##### Gami-Net
<p align="center">
  <img src="assets/plots/effects/gaminet/sim4_mod2r_config1/interact_all.png" width="600">
</p>

##### EBM
<p align="center">
  <img src="assets/plots/effects/ebm/sim4_mod2r_config1/interact_all.png" width="600">
</p>

<!----------------------------------------------------------------------------------------->
### $g_3$
#### Interaction Effects ($\rho = 0, n=50K$)
##### Gami-Net
<p align="center">
  <img src="assets/plots/effects/gaminet/sim2_mod3r_config1/interact_all.png" width="600">
</p>

##### EBM
<p align="center">
  <img src="assets/plots/effects/ebm/sim2_mod3r_config1/interact_all.png" width="600">
</p>

#### Interaction Effects ($\rho = 0.5, n=50K$)
##### Gami-Net
<p align="center">
  <img src="assets/plots/effects/gaminet/sim4_mod3r_config1/interact_all.png" width="600">
</p>

##### EBM
<p align="center">
  <img src="assets/plots/effects/ebm/sim4_mod4r_config1/interact_all.png" width="600">
</p>


<!----------------------------------------------------------------------------------------->
### $g_4$
#### Interaction Effects ($\rho = 0, n=50K$)
##### Gami-Net
<p align="center">
  <img src="assets/plots/effects/gaminet/sim2_mod4r_config1/interact_all.png" width="600">
</p>

##### EBM
<p align="center">
  <img src="assets/plots/effects/ebm/sim4_mod4r_config1/interact_all.png" width="600">
</p>


#### Interaction Effects ($\rho = 0.5, n=50K$)
##### Gami-Net
<p align="center">
  <img src="assets/plots/effects/gaminet/sim4_mod4r_config1/interact_all.png" width="600">
</p>

##### EBM
<p align="center">
  <img src="assets/plots/effects/ebm/sim4_mod4r_config1/interact_all.png" width="600">
</p>





<!----------------------------------------------------------------------------------------->
<!----------------------------------------------------------------------------------------->
For each Response Model we observed the following Main Effects:
<!----------------------------------------------------------------------------------------->
### $g_1$

#### Main Effects ($\rho = 0, n=50K$)
##### Gami-Net
<p align="center">
  <img src="assets/plots/effects/gaminet/sim2_mod1r_config1/main_all.png" width="600">
</p>

##### EBM
<p align="center">
  <img src="assets/plots/effects/ebm/sim2_mod1r_config1/main_all.png", width = "600">
</p>

#### Main Effects ($\rho = 0.5, n=50K$)
##### Gami-Net
<p align="center">
  <img src="assets/plots/effects/gaminet/sim4_mod1r_config1/main_all.png" width="600">
</p>

##### EBM
<p align="center">
  <img src="assets/plots/effects/ebm/sim4_mod1r_config1/main_all.png" width="600">
</p>
<!----------------------------------------------------------------------------------------->

### $g_2$
#### Main Effects ($\rho = 0, n=50K$)
##### Gami-Net
<p align="center">
  <img src="assets/plots/effects/gaminet/sim2_mod2r_config1/main_all.png" width="600">
</p>

##### EBM
<p align="center">
  <img src="assets/plots/effects/ebm/sim2_mod2r_config1/main_all.png" width="600">
</p>

#### Main Effects ($\rho = 0.5, n=50K$)
##### Gami-Net
<p align="center">
  <img src="assets/plots/effects/gaminet/sim4_mod2r_config1/main_all.png" width="600">
</p>

##### EBM
<p align="center">
  <img src="assets/plots/effects/ebm/sim4_mod2r_config1/main_all.png" width="600">
</p>

<!----------------------------------------------------------------------------------------->
### $g_3$
#### Main Effects ($\rho = 0, n=50K$)
##### Gami-Net
<p align="center">
  <img src="assets/plots/effects/gaminet/sim2_mod3r_config1/main_all.png" width="600">
</p>

##### EBM
<p align="center">
  <img src="assets/plots/effects/ebm/sim2_mod3r_config1/main_all.png" width="600">
</p>

#### Main Effects ($\rho = 0.5, n=50K$)
##### Gami-Net
<p align="center">
  <img src="assets/plots/effects/gaminet/sim4_mod3r_config1/main_all.png" width="600">
</p>

##### EBM
<p align="center">
  <img src="assets/plots/effects/ebm/sim4_mod4r_config1/main_all.png" width="600">
</p>


<!----------------------------------------------------------------------------------------->
### $g_4$
#### Main Effects ($\rho = 0, n=50K$)
##### Gami-Net
<p align="center">
  <img src="assets/plots/effects/gaminet/sim2_mod4r_config1/main_all.png" width="600">
</p>

##### EBM
<p align="center">
  <img src="assets/plots/effects/ebm/sim4_mod4r_config1/main_all.png" width="600">
</p>


#### Main Effects ($\rho = 0.5, n=50K$)
##### Gami-Net
<p align="center">
  <img src="assets/plots/effects/gaminet/sim4_mod4r_config1/main_all.png" width="600">
</p>

##### EBM
<p align="center">
  <img src="assets/plots/effects/ebm/sim4_mod4r_config1/main_all.png" width="600">
</p>
