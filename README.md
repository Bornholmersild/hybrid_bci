# Deep Learning-Based Multimodal Biosignal Control of a Soft Hand Exoskeleton for Grasp Assistance
Real-time EEG and EMG decoding for control of a soft robotic hand exoskeleton using deep learning and musculoskeletal simulation.

## NOTE
This work is part of an ongoing master’s thesis project. Certain implementations, refinements, and code structure optimizations are still under development.

## Overview

This project presents a multimodal biosignal framework for real-time control of a soft hand exoskeleton. The framework investigates the individual contributions of EEG and EMG, as well as their decision-level fusion, for decoding hand motor intentions.

Three deep learning architectures are explored:

* LSTM ($N_1$)
* CNN + LSTM ($N_2$)
* CNN + LSTM + Attention ($N_3$)

The system integrates:

* Real-time EMG acquisition
* EEG/EMG preprocessing pipelines
* Neural decoding networks
* Decision-level fusion
* MuJoCo + MyoSuite digital twin visualization

## System Architecture

The framework combines:

1) Biosignal acquisition
2) Signal preprocessing
3) Deep neural decoding
4) Motion classification
6) Exoskeleton actuation
7) Digital twin visualization

## Deep Learning Models
$N_1$ — LSTM

Baseline temporal sequence model for biosignal decoding.

$N_2$ — CNN + LSTM

Sequential 1D CNN layers extract local temporal and cross-channel features before the LSTM models temporal dependencies.

$N_3$ — CNN + LSTM + Attention

Attention mechanism enhances temporal feature weighting and improves discriminative representation learning.

## Repository Structure


## Environment

Python version: 3.11
Install:
py -3.11 -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

## Remarks
Trigno implementation only works in windows

data_fusion_manager script with listen_for_terminal_input func only works in windows given command: msvcrt.kbhit()
