[![CI](https://github.com/MaochengX/LMU-Gami-tree-rep/actions/workflows/continous-integration.yaml/badge.svg)](https://github.com/MaochengX/LMU-Gami-tree-rep/actions/workflows/continous-integration.yaml)

# Prelude

Hello there! We are Sven and Maocheng and like python programming and X-AI.<br>
In this repo we conduct a replication study using so-called Gami-Trees, an by construction inherent explainable ML tool which was proposed by [Hu et.al (2022)](https://arxiv.org/abs/2207.06950).<br>
This branch is the implementation of GAMI-Tree.

## Features:
 - Environment and dependency management using [`uvx`](https://docs.astral.sh/uv/guides/tools/)
 - Code Style: Black using [`ruff`](https://docs.astral.sh/ruff/)


# :rocket: Experiment Setup
Our experiment is structured into a source code, configutation and assets folder. The overall workflow can be controlled from within the `Makefile` at project root.<br>
In principle the workflow can be split into two parts: one for data generation and one for the actual simulation study
