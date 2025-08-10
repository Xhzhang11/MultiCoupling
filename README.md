
# Neuro-inspired Recurrent Networks with Multiplicative Coupling

## Overview
This repository contains code and data for our study on **Multiplicative feedback gating enables rapid learning and flexible computation
in recurrent neural circuits**.  
- **Accelerate learning** on working memory and decision benchmarks (Figure. 1).
- **Stabilize and persist dynamics** such as sequences and limit cycles (Figure. 2).
- **Enable cognitive flexibility** in biologically constrained thalamocortical models (Figure. 3).
- **Support attention and working memory** in cortico–thalamic–cortical circuits (Figure. 4).
- **Mediate motor task switching** via cerebellar–thalamic–cortical loops (Figure. 5).
- **Produce grid-like codes** in visuospatial navigation (Fig. 6).

---


## Dependencies
- Python **3.10**
- PyTorch
- scikit-learn
- numba




## Repository Structure

├── benchmark/                    # Figure1: fast learning via multiplicative gating  
│
├── sequence_limit_cycle/         # Figure2: robust/persistent dynamics  
│
├── switching_PFC_MD/             # Figure3: cognitive flexibility in PFC–MD models  
│
├── cortico_pulvinar_cortico/     # Figure4: WM & attention in cortico–thalamic–cortical loops  
│
├── motor_plan/                   # Figure5: cerebellar–thalamic–cortical motor switching  
│
└── grid_cell/                    # Figure6: entorhinal–hippocampal visuospatial navigation  




## Reproducing the results of the paper
You can produce the main figures of the paper by running the files in different folders.

#Reproducing Figure1  
python benchmark/figure1_gym.py  
python benchmark/figure1_MachineLearning_task.py  
python benchmark/figure1_RL.py  
python benchmark/figure1_STDP.py  
python benchmark/figure1_SupervisedLearning.py  


#Reproducing Figure2  
python sequence_limit_cycle/figure2_histogram_diffMDsize.py  
python sequence_limit_cycle/figure2_loss.py  
python sequence_limit_cycle/figure2_sequence.py  
python sequence_limit_cycle/figure2_state.py  



#Reproducing Figure3  
python switching_PFC_MD/figure3_context_switching.py  
python switching_PFC_MD/figure3_tactile.py


#Reproducing Figure4  
python cortico_pulvinar_cortico/figure4_plot_angle.py  
python cortico_pulvinar_cortico/figure4_plot_decode.py  
python cortico_pulvinar_cortico/figure4_plot_decode_test.py  
python cortico_pulvinar_cortico/figure4_plot_state.py  



#Reproducing Figure5  
python motor_plan/figure5_plot_speed.py  
python motor_plan/figure5_trajectory_perturbation.py  


#Reproducing Figure6  
python grid_cell/plot_ratemaps.py  
python grid_cell/sequence.py  
python grid_cell/plot_speed.py   
python grid_cell/plot_velocity_trajectry.py   


















