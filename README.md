
# Neuro-inspired Recurrent Networks with Multiplicative Coupling

## Overview
This repository contains code and data for our study on **Multiplicative feedback gating enables rapid learning and flexible computation
in recurrent neural circuits**.  
- **Accelerate learning** on working memory and decision benchmarks (Fig. 1).
- **Stabilize and persist dynamics** such as sequences and limit cycles (Fig. 2).
- **Enable cognitive flexibility** in biologically constrained thalamocortical models (Fig. 3).
- **Support attention and working memory** in cortico–thalamic–cortical circuits (Fig. 4).
- **Mediate motor task switching** via cerebellar–thalamic–cortical loops (Fig. 5).
- **Produce grid-like codes** in visuospatial navigation (Fig. 6).

---


## Dependencies
- Python **3.10**
- PyTorch
- scikit-learn
- numba




## Repository Structure

├── benchmark/                    # Fig. 1: fast learning via multiplicative gating  
│
├── sequence_limit_cycle/         # Fig. 2: robust/persistent dynamics  
│
├── switching_PFC_MD/             # Fig. 3: cognitive flexibility in PFC–MD models  
│
├── cortico_pulvinar_cortico/     # Fig. 4: WM & attention in cortico–thalamic–cortical loops  
│
├── motor_plan/                   # Fig. 5: cerebellar–thalamic–cortical motor switching  
│
└── grid_cell/                    # Fig. 6: entorhinal–hippocampal visuospatial navigation  




## Reproducing the results of the paper
You can produce the main figures of the paper by running the files in different folders.

Reproducing Figure1 

python benchmark/figure1_gym.py  
python benchmark/figure1_MachineLearning_task.py  
python benchmark/figure1_RL.py  
python benchmark/figure1_STDP.py  
python benchmark/figure1_SupervisedLearning.py  


Reproducing Figure2 

python sequence_limit_cycle/figure2_histogram_diffMDsize.py  
python sequence_limit_cycle/figure2_loss.py  
python sequence_limit_cycle/figure2_sequence.py  
python sequence_limit_cycle/figure2_state.py  



Reproducing Figure3  

python switching_PFC_MD/figure3_context_switching.py
python switching_PFC_MD/figure3_tactile.py


Reproducing Figure4  

python cortico_pulvinar_cortico/figure4_plot_angle.py  
python cortico_pulvinar_cortico/figure4_plot_decode.py  
python cortico_pulvinar_cortico/figure4_plot_decode_test.py  
python cortico_pulvinar_cortico/figure4_plot_state.py  



Reproducing Figure5  

python motor_plan/figure5_plot_speed.py  
python motor_plan/figure5_trajectory_perturbation.py  


Reproducing Figure6  

python grid_cell/plot_ratemaps.py  
python grid_cell/sequence.py  
python grid_cell/plot_speed.py   
python grid_cell/plot_velocity_trajectry.py   


















