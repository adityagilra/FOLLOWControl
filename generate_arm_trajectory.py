#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import matplotlib as mpl
# must be called before any pylab import, matplotlib calls
mpl.use('QT4Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from plot_utils import *
import shelve,contextlib

## choose one arm / acrobot model out of the below
#bot  = 'arm_1link'
bot = 'arm_2link'
#bot = 'acrobot_2link'

## choose one task out of the below:
#task = 'write_f'
#task = 'star'
task = 'diamond'
#task = '3point'

#np.random.seed(1)
np.random.seed(2)                                           # seed 2 for reach1

if bot == 'arm_1link':
    from arm_1link_gravity import evolveFns,armXY,armAngles                
    N = 2
    setpoint = 0.1
    torques = np.array([div_torq,0.,-div_torq])
    def get_torques(action):
        return torques[action]
    action0 = 1                                             # index for zero torque at t=0
    len_actions = 3
    acrobot = False
elif bot == 'arm_2link':
    botname = 'robot2_todorov_gravity'
    from arm_2link_todorov_gravity import evolveFns,armXY,armAngles,armAnglesEnd
    N = 4                                                   # 2-link arm
    if task == 'write_f':
        # crude 'digital-cursive' "f" for follow
        crudeTargetTraj = np.array([(0.,-0.2),(0.05,-0.1),(0.2,0.1),(0.1,0.2),(0.,0.1),(0.05,-0.05),\
                                    (0.1,-0.15),(0.2,-0.1),(0.05,0.),(0.2,0.)]) /2. \
                                    + np.array((0.,-0.48))  # base-point shift
    elif task == 'star':
        # crude 'digital-star'
        crudeTargetTraj = np.array([(0.,0.),(0.18,0.22),(0.0,0.25),(0.13,0.02),(0.1,0.35),(0.,0.)])\
                                    + np.array((0.,-0.58))  # base-point shift
    elif task == 'diamond':
        # crude 'digital-diamond'
        crudeTargetTraj = np.array([(0.,0.),(0.28,0.08),(0.4,0.4),(0.12,0.35),(0.,0.)])\
                                    + np.array((0.,-0.58))  # base-point shift
    elif task == '3point':
        # 3-point testing
        crudeTargetTraj = np.array([(0.,-0.2),(0.1,-0.15),(0.2,0.-0.1)]) /2. \
                                    + np.array((0.,-0.48))  # base-point shift
    else:
        print('please choose a task')
        sys.exit(1)

    dtpts = 0.001                                       # s, travel-time between points in targetTraj
                                                        # seems important to keep this same as Nengo dt
                                                        #  for nodeIn function in control_inverse_...!
    intermediate_pts = int(1./dtpts)
    # interpolate the crude trajectory by adding intermediate points
    xlist,ylist = zip(*crudeTargetTraj)
    targetTrajX = np.interp( np.linspace(0,len(xlist)-1,len(xlist)*intermediate_pts),
                            range(len(xlist)), xlist )
    targetTrajY = np.interp( np.linspace(0,len(ylist)-1,len(ylist)*intermediate_pts),
                            range(len(ylist)), ylist )
    anglesOld = np.array((0.,0.))
    Imax = len(targetTrajX)
    Tmax = Imax*dtpts
    stateTraj = np.zeros((Imax,4))
    for i,(x,y) in enumerate(zip(targetTrajX,targetTrajY)):
        angles = armAnglesEnd((x,y))
        if i<100: print (i,angles,anglesOld)
        dangles = (angles - anglesOld) / dtpts
        stateTraj[i,:] = np.array((angles[0],angles[1],dangles[0],dangles[1]))
        anglesOld = angles
        
    print (stateTraj)
    print ("Total time =",Tmax,"s.")
    #targetTraj = list(zip(targetTrajX,targetTrajY))        # in python3, zip() returns an iterator, hence list()
    varFactors = (1./2.5,1./2.5,0.05,0.05,0.02,0.02)        # angleFactors, velocityFactors, torqueFactors
                                                            # for robot2_todorov_gravity
    acrobot = False
    robotdt = 0.01                                          # evolve with this time step
    animdt = 0.1                                            # time step for an action
elif bot == 'acrobot_2link':
    from acrobot_2link import evolveFns,armXY,armAngles                
    N = 4                                                   # 2-link arm
    div_torq = 2.
    max_dq = np.array((4*np.pi,9*np.pi))                    # with zero friction (B matrix)
    torques = np.array([(0.,div_torq),(0.,0.),(0.,-div_torq)])
    def get_torques(action):
        return torques[action]
    len_actions = 3
    setpointX = 1.
    setpointY = -1.
    acrobot = True
    robotdt = 0.05                                          # evolve with this time step
    animdt = 0.2                                            # time step for an action

trange = np.arange(0,Tmax,dtpts)
print ("Saving to",botname+"_traj_"+task+".shelve")
with contextlib.closing(
        shelve.open(botname+"_traj_"+task+".shelve", 'c', protocol=-1)
        ) as data_dict:
    data_dict['trange'] = trange
    data_dict['ratorOut'] = np.zeros(Imax)                  # true torque sent to arm is unknown
    data_dict['ratorOut2'] = np.zeros(Imax)                 # learner-network's output is irrelevant
    data_dict['varFactors'] = varFactors
    data_dict['rateEvolve'] = stateTraj*varFactors[:4]      # desired arm trajectory

plt.figure()
plt.scatter(targetTrajX,targetTrajY)
plt.figure()
plt.scatter(stateTraj[:,0],stateTraj[:,1])
plt.figure()
for i in range(4):
    plt.plot(trange,stateTraj[:,i])

plt.show()
