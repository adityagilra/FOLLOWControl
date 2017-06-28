# FOLLOWControl
Control using forward and inverse models learned by FOLLOW.  
1. The FOLLOW learning scheme applied to the forward model is as per Gilra and Gerstner 2017 on arxiv ([https://arxiv.org/abs/1702.06463]).
2. The inverse model learning and then using it for control is as per Gilra and Gerstner 2017, NIPS submitted.  
3. and 4. are new collaborative projects with Camilo and Marin.

##1. Inverse model
This is workflow used for the NIPS submission: Gilra and Gerstner 2017.
a. Learn the inverse model:
(i) purely recurrent network:
nohup python inverse_rec_robot_nengo_ocl.py &> nohup.out &
OR
(ii) feedforward followed by recurrent network:
nohup python inverse_ff_rec_robot_nengo_ocl.py &> nohup.out &

These files use sim_robot.py and arm*.py for simulating the 'true' arm dynamics.
They save in separate files: the variables monitored during learning and the final weights.

b. Load the pre-learned weights file and a desired trajectory and use it to control the true arm.
First generate the desired trajectory (see settings for 'star' and 'diamond' within the file):
python generate_arm_trajectory.py

Then run the control simulation (this file only uses the purely recurrent network currently):
python control_inverse_robot_nengo_ocl.py
This file loads in previous weights (filename set in above file), and desired trajectory (filename set in above file), into the purely recurrent architecture currently, and then builds some extra feedback architecture to control the arm. Finally simulates the network and the true arm and saves the simulation variables.

##2. Forward predictive model
a. Learn the forward model:
(i) purely recurrent network:
nohup python input_general_robot_nengo_directu_ocl.py &> nohup.out &
OR
(ii) feedforward followed by recurrent network:
nohup python input_ff_rec_robot_nengo_directu_ocl.py &> nohup.out &

These files use sim_robot.py and arm*.py for simulating the 'true' arm dynamics.
They save in separate files: the variables monitored during learning and the final weights.

b. Load the pre-learned weights file and a desired trajectory and use it to control the true arm.
First generate the desired trajectory (see settings for 'star' and 'diamond' within the file):
python generate_arm_trajectory.py

Then run the control simulation (this file only uses the purely recurrent network currently):
python control_robot_nengo_ocl.py
This file loads in previous weights (filename set in above file), and desired trajectory (filename set in above file), into the purely recurrent architecture currently, and then builds some extra feedback architecture to control the arm. Finally simulates the network and the true arm and saves the simulation variables.

The current control architecture using my forward model does not work.
I have other configurations in mind to make this work.

##3. New project: With Marin + Camilo, we'll use forward model + RL for control.

##4. New project: With Camilo, we'll use hierarchical FOLLOW learning for hierarchical control.
