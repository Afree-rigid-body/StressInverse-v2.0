#*************************************************************************#
#                                                                         #
#  script INPUT_PARAMETERS                                                #
#                                                                         #
#  list of input parameters needed for the inversion                      #
#                                                                         #
#*************************************************************************#
import numpy as np
import os

### NOTE: do not remove r before strings (r'filename'), to safely use
#         backslashes in filenames

#--------------------------------------------------------------------------
# input file with focal mechnaisms
#--------------------------------------------------------------------------
# input_file = r'../Data/West_Bohemia_mechanisms.dat'  # M-eM-^NM-^_M-gM-(M-^KM-eM-:M-^O
# input_file = r'../Data/Terakawa_data.dat' # TerakawaM-fM-^VM-^GM-gM-+M- M-fM-^UM-0M-fM-M-.
# input_file = r'../Data/horizontal_rupture_plane.dat' # M-fM-5M-^KM-hM-/M-^UM-fM-0M-4M-eM-9M-3M-gM- M-4M-hM-#M-^BM-iM-^]M-"
input_file = r'../Data/Xfj_full_mechanisms_new.txt' # M-fM-^VM-0M-dM-8M-0M-fM-1M-^_M-fM-^UM-0M-fM-M-.
# input_file = r'../Data/fj_full_mechanisms_cluster1.dat' # M-fM-^VM-0M-dM-8M-0M-fM-1M-^_M-fM-^UM-0M-fM-M-. Cluster1 
# input_file = r'../Data/fj_full_mechanisms_cluster2.dat' # M-fM-^VM-0M-dM-8M-0M-fM-1M-^_M-fM-^UM-0M-fM-M-. Cluster2 
# input_file = r'../Data/fj_full_mechanisms_cluster3.dat' # M-fM-^VM-0M-dM-8M-0M-fM-1M-^_M-fM-^UM-0M-fM-M-. Cluster3 

#--------------------------------------------------------------------------
# output file with results
#--------------------------------------------------------------------------
# output_file = r'../Output/West_Bohemia_Output'
output_file = r'../Output_test/West_Bohemia_Output'


# ASCII file with calculated principal mechanisms
principal_mechanisms_file = r'../Output_test/West_Bohemia_principal_mechanisms'


#-------------------------------------------------------------------------
# accuracy of focal mechansisms
#--------------------------------------------------------------------------
# number of random noise realizations for estimating the accuracy of the
# solution
# N_noise_realizations = 100
N_noise_realizations = 100
# =============================================================================
# 
# estimate of noise in the focal mechanisms (in degrees)
# the standard deviation of the normal distribution of
# errors
mean_deviation = 5

#--------------------------------------------------------------------------
# figure files
#--------------------------------------------------------------------------
# shape_ratio_plot = r'../Figures/shape_ratio'
# stress_plot      = r'../Figures/stress_directions'
# P_T_plot         = r'../Figures/P_T_axes'
# Mohr_plot        = r'../Figures/Mohr_circles'
# faults_plot      = r'../Figures/faults'

shape_ratio_plot = r'../test_Figures/shape_ratio'
stress_plot      = r'../test_Figures/stress_directions'
P_T_plot         = r'../test_Figures/P_T_axes'
Mohr_plot        = r'../test_Figures/Mohr_circles'
faults_plot      = r'../test_Figures/faults'

# shape_ratio_plot = r'../cluster1_Figures/shape_ratio'
# stress_plot      = r'../cluster1_Figures/stress_directions'
# P_T_plot         = r'../cluster1_Figures/P_T_axes'
# Mohr_plot        = r'../cluster1_Figures/Mohr_circles'
# faults_plot      = r'../cluster1_Figures/faults'

# shape_ratio_plot = r'../cluster2_Figures/shape_ratio'
# stress_plot      = r'../cluster2_Figures/stress_directions'
# P_T_plot         = r'../cluster2_Figures/P_T_axes'
# Mohr_plot        = r'../cluster2_Figures/Mohr_circles'
# faults_plot      = r'../cluster2_Figures/faults'

# shape_ratio_plot = r'../cluster3_Figures/shape_ratio'
# stress_plot      = r'../cluster3_Figures/stress_directions'
# P_T_plot         = r'../cluster3_Figures/P_T_axes'
# Mohr_plot        = r'../cluster3_Figures/Mohr_circles'
# faults_plot      = r'../cluster3_Figures/faults'


#--------------------------------------------------------------------------
# advanced control parameters (usually not needed to be changed)
#--------------------------------------------------------------------------
# number of iterations of the stress inversion 
N_iterations = 6

# number of initial stres inversions with random choice of faults
N_realizations = 10

# axis of the histogram of the shape ratio
shape_ratio_min = 0
shape_ratio_max = 1
shape_ratio_step = 0.025

shape_ratio_axis = np.arange(shape_ratio_min+0.0125, shape_ratio_max, shape_ratio_step)
 
# interval for friction values
friction_min  = 0.40
friction_max  = 1.00
friction_step = 0.05


#--------------------------------------------------------------------------
# create output directories if needed
all_files = (output_file, shape_ratio_plot, stress_plot, P_T_plot, Mohr_plot, faults_plot)
for f in all_files:
    folder = os.path.dirname(f)
    if not os.path.exists(folder):
        os.makedirs(folder)
