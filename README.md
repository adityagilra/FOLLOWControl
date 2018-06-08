# FOLLOWControl  
  
Code for Gilra and Gerstner, ICML 2018 (to appear), preprint at [https://arxiv.org/abs/1712.10158](https://arxiv.org/abs/1712.10158).  
  
First, learn the inverse model using the FOLLOW learning scheme introduced earlier in [Gilra and Gerstner, eLife 2017;6:e28295](https://elifesciences.org/articles/28295) -- see also [https://github.com/adityagilra/FOLLOW](https://github.com/adityagilra/FOLLOW).  
  
Then use the inverse model to control an arm to reproduce a desired trajectory.  

## 1. Inverse model
Learn the inverse model using FOLLOW learning via motor babbling. The differential feedforward network architecture is used.  
`nohup python inverse_diff_ff_robot_nengo_ocl.py &> nohup.out &`  
Other similarly named script files are for other architectures explored in the paper.  
  
These scripts import sim_robot.py and arm*.py for simulating the 'true' arm dynamics.  
They save in separate files: the variables monitored during learning and the final weights.  
  
## 2. Inverse model for motor control
Load the pre-learned weights file and a desired trajectory and use it to control the true arm.  
First generate the desired trajectory (see settings for 'zigzag' and 'diamond' within the file):  
`python generate_arm_trajectory_v2.py`  
  
Then run the control simulation with the differential feedforward network architecture:  
`python control_inverse_diff_robot_nengo_ocl.py `  
This file loads in previous weights (filename set in above script file), and desired trajectory (filename set in above script file), into the differential feedforward network, and then builds some extra feedback architecture to control the arm. Finally simulates the network and the true arm and saves the simulation variables.  
  
Some other files exist for using the forward model for control. This direction didn't work out and was abandoned.  
  
## 3. Figures
The figures in the paper require multiple simulation runs and collating the results together. Have a look at the end of input_rec_transform_nengo_plot_figs.py for the functional calls that create the figures. Their arguments are the names of the data files that must be loaded and plotted. The filenames contain the architectures and parameters for that data and these should be matched with the name of the data file created by the scripts using the relevant architecture. The data files are not provided here as they are too large. They can be generated by running various simulations as per parameters in the filenames embedded in input_rec_transform_nengo_plot_figs.py .  
