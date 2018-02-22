# -*- coding: utf-8 -*-
# (c) Sep 2015 Aditya Gilra, EPFL.

"""
control of arbitrary dynamical system using predictive general model
in Nengo simulator
written by Aditya Gilra (c) May 2017.
"""

import nengo
import nengo_ocl

import numpy as np
import input_rec_transform_nengo_plot as myplot
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d

#import pickle
# pickle constructs the object in memory, use shelve for direct to/from disk
import shelve, contextlib
import pandas as pd
from os.path import isfile
import os,sys


########################
### Constants/parameters
########################

###
### Overall parameter control ###
###
OCL = True                              # use nengo_ocl or nengo to simulate
if OCL: import nengo_ocl
errorLearning = True                    # error-based PES learning OR algorithmic
recurrentLearning = True                # now it's on both, so this is obsolete, leave it True
plastDecoders = False                   # whether to just have plastic decoders or plastic weights
inhibition = False#True and not plastDecoders # clip ratorOut weights to +ve only and have inh interneurons

learnIfNoInput = False                  # Learn only when input is off (so learning only on error current)
errorFeedback = True                    # Forcefeed the error into the network (used only if errorLearning)
learnFunction = True                    # whether to learn a non-linear function or a linear matrix
#robotType = 'V-REP'
robotType = 'pendulum'
reloadRobotSim = False
trialClamp = False                      # reset robot and network at the end of each trial during learning (or testing if testLearned)
#funcType = 'robot1_gravity'             # if learnFunction, then robot one-link system simulated by V-Rep/pendulum/etc dynamics
#funcType = 'robot1XY_gravity'           # if learnFunction, then robot in x-y one-link system simulated by V-Rep/pendulum/etc dynamics
#funcType = 'robot1_gravity_interpol'    # if learnFunction, then robot in x-y one-link system simulated by V-Rep/pendulum/etc dynamics
#funcType = 'robot1XY_gravity_interpol'  # if learnFunction, then robot in x-y one-link system simulated by V-Rep/pendulum/etc dynamics
#funcType = 'robot2_gravity_interpol'    # if learnFunction, then robot in x-y one-link system simulated by V-Rep/pendulum/etc dynamics
#funcType = 'robot2_todorov'             # if learnFunction, then robot two-link system simulated by pendulum dynamics
funcType = 'robot2_todorov_gravity'     # if learnFunction, then robot two-link system with gravity simulated by pendulum dynamics
#funcType = 'robot2XY_todorov_gravity'   # if learnFunction, then robot in x-y two-link system with gravity simulated by pendulum dynamics
#funcType = 'acrobot2_gravity'           # if learnFunction, then acrobot two-link system with gravity simulated by pendulum dynamics, clipping on q,dq
#funcType = 'acrobot2XY_gravity'         # if learnFunction, then acrobot two-link system with gravity simulated by pendulum dynamics, clipping on q,dq

#pathprefix = '/lcncluster/gilra/tmp/'
pathprefix = '../data/'
#Nin = 200
#Nexc = 500
#filtStr = '_filt0.02'
#endStr = '_50000.0s'
Nin = 3000
Nexc = 5000
filtStr = ''
endStr = '_10000.0s'
weightsLoadFileName = 'inverse_diff_ff_S2_d50c50_N'+str(Nin)+'_ocl_Nexc'+str(Nexc)+'_norefinptau_seeds2345'+filtStr+'_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights'+endStr+'_endweights.shelve'
#weightsLoadFileName = 'inverse_diff_ff_rec_50ms_ocl_Nexc3000_norefinptau_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_50000.0s_endweights.shelve'
#trajectoryFileName = 'inverse_100ms_ocl_Nexc5000_norefinptau_directu_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom3000.0_seed2by0.3RLSwing_10.0s_start'
#trajectoryFileName = 'inverse_100ms_ocl_Nexc5000_norefinptau_directu_seeds2345_weightErrorCutoff0.0_nodeerr_learn_rec_nocopycat_func_robot2_todorov_gravity_seed2by0.3amplVaryHeights_testFrom10000.0_seed2by0.3RLReach3_10.0s_start'
#trajectoryFileName = 'robot2_todorov_gravity_traj_v2_star'
trajectoryFileName = 'robot2_todorov_gravity_traj_v2_sdiamond'
#trajectoryFileName = 'robot2_todorov_gravity_traj_v2_zigzag'
weightsLoadFileName = pathprefix + weightsLoadFileName

###
### Nengo model params ###
###
seedR0 = 2              # seed set while defining the Nengo model
seedR1 = 3              # another seed for the first layer
                        # some seeds give just flat lines for Lorenz! Why?
seedR2 = 4              # another seed for the second layer
                        # this is just for reproducibility
                        # seed for the W file is in rate_evolve.py
                        # output is very sensitive to this seedR
                        # as I possibly don't have enough neurons
                        # to tile the input properly (obsolete -- for high dim)
seedR4 = 5              # for the nengonetexpect layer to generate reference signal
seedRin = 2
np.random.seed([seedRin])# this seed generates the inpfn below (and non-nengo anything random)

tau = 0.02              # second, synaptic tau
neuronType = nengo.neurons.LIF()
                        # use LIF neurons for all ensembles


# N is the number of state variables in the system, N//2 is number of inputs
# Nout is the number of observables from the system
if 'robot1_' in funcType:
    N = 2
    Nobs = 2
if 'robot1XY' in funcType:
    N = 2
    Nobs = 3
elif 'robot2_' in funcType:
    N = 4                                                   # coordinate and velocity (q,p) for each degree of freedom
    Nobs = 4
elif 'robot2XY' in funcType:                                # includes acrobot2XY
    N = 4
    Nobs = 6                                                # x1,y1,x2,y2,omega1,omega2
else:
    N = 2
    Nobs = 2

if robotType == 'V-REP':
    torqueFactor = 100.                                     # torqueFactor multiplies inpfn directly which goes to robot and network
    angleFactor = 1./np.pi                                  # scales the angle from the robot going into the network
    velocityFactor = 1./5.                                  # scales the velocity from the robot going into the network
else:
    if funcType == 'robot1_gravity':
        varFactors = (0.5,0.1,0.125)                        # xyFactors, velocityFactors, torqueFactors
        #varFactors = (0.15,0.15,0.125)                     # xyFactors, velocityFactors, torqueFactors
    elif funcType == 'robot1XY_gravity':
        varFactors = (2.5,2.5,0.05,0.02)                    # xyFactors, velocityFactors, torqueFactors for 1-link arm with gravity and damping
        #varFactors = (2.,2.,0.01,0.075)                     # xyFactors, velocityFactors, torqueFactors for 1-link arm with gravity (trialClamp = True)
    elif funcType == 'robot1_gravity_interpol':
        varFactors = (1./3.5,0.05,0.02)                     # angleFactors, velocityFactors, torqueFactors for 1-link arm with gravity and damping
        #varFactors = (2.,2.,0.01,0.075)                     # xyFactors, velocityFactors, torqueFactors for 1-link arm with gravity (trialClamp = True)
    elif funcType == 'robot1XY_gravity_interpol':
        varFactors = (2.5,2.5,0.05,0.02)                    # xyFactors, velocityFactors, torqueFactors for 1-link arm with gravity and damping
        #varFactors = (2.,2.,0.01,0.075)                     # xyFactors, velocityFactors, torqueFactors for 1-link arm with gravity (trialClamp = True)
    elif funcType == 'robot2_gravity_interpol':
        varFactors = (1./3.5,1./3.5,0.05,0.05,0.02,0.02)    # angleFactors, velocityFactors, torqueFactors for 1-link arm with gravity and damping
        #varFactors = (2.,2.,0.01,0.075)                     # xyFactors, velocityFactors, torqueFactors for 1-link arm with gravity (trialClamp = True)
    elif funcType == 'robot2_todorov':
        varFactors = (1.,1.,0.5,0.5,0.5,0.5)                # angleFactors, velocityFactors, torqueFactors
    elif funcType == 'robot2_todorov_gravity':
        #varFactors = (1./2.5,1./2.5,0.05,0.05,0.02,0.02)    # angleFactors, velocityFactors, torqueFactors
        varFactors = (1./2.5,1./2.5,0.15,0.15,0.1,0.1)    # angleFactors, velocityFactors, torqueFactors
    elif funcType == 'robot2XY_todorov_gravity':
        #varFactors = (1.,1.,1.,1.,0.15,0.15,0.125,0.125)    # xyFactors, velocityFactors, torqueFactors
        varFactors = (2.5,2.5,1.2,1.2,0.075,0.075,0.025,0.025)    # xyFactors, velocityFactors, torqueFactors
    elif funcType == 'acrobot2_gravity':
        varFactors = (0.55,0.4,0.12,0.075,0.05,0.05)        # angleFactors, velocityFactors, torqueFactors
    elif funcType == 'acrobot2XY_gravity':
        varFactors = (0.9,0.9,0.45,0.45,0.08,0.05,0.025,0.025)    # xyFactors, velocityFactors, torqueFactors

###
### recurrent and feedforward connection matrices ###
###
if errorLearning:                                       # PES plasticity on
    Tmax = 10.                                          # second - how long to run the simulation
    continueTmax = 10000.                               # if continueLearning, then start with weights from continueTmax
    reprRadius = 1.0                                    # neurons represent (-reprRadius,+reprRadius)
    reprRadiusIn = 1.0                                  # input is integrated in ratorOut, so keep it smaller than reprRadius
    # with zero bias, at reprRadius, if you want 50Hz, gain=1.685, if 100Hz, gain=3.033, if 400Hz, 40.5
    nrngain = 40.5
    if 'acrobot' in funcType: inputreduction = 0.5      # input reduction factor
    else: inputreduction = 0.3                          # input reduction factor
    Tperiod = 1.                                        # second

reprRadiusErr = 0.2                                     # with error feedback, error is quite small

###
### time params ###
###
rampT = 0.5                                             # second
dt = 0.001                                              # second
weightdt = Tmax/20.                                     # how often to probe/sample weights
Tclamp = 0.25                                           # time to clamp the ref, learner and inputs after each trial (Tperiod)

###
### time params ###
###

if errorLearning:
    errorAverage = False                    # whether to average error over the Tperiod scale
                                            # Nopes, this won't make it learn the intricate dynamics
    errorFeedbackGain = 0.                  # Feedback gain
    #errorFeedbackGain = 3.                  # Feedback gain
                                            # below a gain of ~5, exc rates go to max, weights become large
    weightErrorTau = 10*tau                 # filter the error to the PES weight update rule
    errorFeedbackTau = 2*tau                # synaptic tau for the error signal into layer2.ratorOut
    errorGainDecay = False                  # whether errorFeedbackGain should decay exponentially to zero
                                            # decaying gain gives large weights increase below some critical gain ~3
    errorGainDecayRate = 1./200.            # 1/tau for decay of errorFeedbackGain if errorGainDecay is True
    errorGainProportion = False             # scale gain proportionally to a long-time averaged |error|
    errorGainProportionTau = Tperiod        # time scale to average error for calculating feedback gain

###
### load desired trajectory ###
###
print('reading data from',trajectoryFileName+'.shelve')
# with ensures that the file is closed at the end / if error
with contextlib.closing(
        shelve.open(trajectoryFileName+'.shelve', 'r')
        ) as data_dict:
    trange = data_dict['trange']
    trueTorque = data_dict['ratorOut']      # true torque sent to arm and learning network (via nodeIn)
    varFactorsOld = data_dict['varFactors']
    rateEvolve = data_dict['rateEvolve']    # true arm trajectory
rateEvolve[:,2] = rateEvolve[:,2]*varFactors[2]/varFactorsOld[2]
rateEvolve[:,3] = rateEvolve[:,3]*varFactors[3]/varFactorsOld[3]
# shape of trueTorque is (5001,) -- how come not 2D?!
#trueTorque[:,0] = trueTorque[:,0]#*varFactors[4]/varFactorsOld[4]
#trueTorque[:,1] = trueTorque[:,1]*varFactors[5]/varFactorsOld[5]
# the true arm trajectory rateEvolve and the network trajectory y2 are already scaled to nengo ensemble radius
#  so can be used directly here (divide by varFactors to get real world values)
desiredStateFn = interp1d(trange,rateEvolve,axis=0,kind='linear',bounds_error=False,fill_value=0.)
#torqueFn = interp1d(trange,trueTorque,axis=0,kind='linear',bounds_error=False,fill_value=0.)

from sim_robot import sim_robot
robtrange,rateEvolveProbe,evolveFns,armAngles = \
    sim_robot(robotType,funcType,False,'')

if 'XY' in funcType: XY = True
else: XY = False
armState = np.zeros(Nobs)                                           # online updated every dt by below fn
def evolveState(u):
    ''' u is the torque represented in the network 
        u is related to the original torque \tilde{u} by u_\alpha = varFactors_\alpha \tilde{u}_\alpha 
        where varFactors_alpha = angleFactor | velocityFactor | torqueFactor
    '''
    utilde = u/varFactors[Nobs:]
    qdot,dqdot = evolveFns(armState[:Nobs-N//2],armState[Nobs-N//2:],utilde,XY,dt)
                                                                    # returns deltaposn if XY else deltaangles
    armState[:Nobs-N//2] += qdot*dt
    armState[Nobs-N//2:] += dqdot*dt
    return armState*varFactors[:Nobs]

#plt.figure()
#plt.plot(trange,desiredStateFn(trange),color='r') 
#plt.show()
#sys.exit()

dataFileName = trajectoryFileName+'_diff-ff_gain'+str(errorFeedbackGain)+'_control'
print('data will be saved to', dataFileName+'.shelve')

if __name__ == "__main__":
    #########################
    ### Create Nengo network
    #########################
    print('building model')
    mainModel = nengo.Network(label="Single layer network", seed=seedR0)
    with mainModel:
        nodeIn = nengo.Node(desiredStateFn)                             # reference state evolution
        nodeInD = nengo.Node(lambda t: desiredStateFn(t-0.05))         # delayed reference state evolution
        nodeInTarget = nengo.Node(lambda t: desiredStateFn(t-0.05))     # target for error feedback (even more delayed) reference state evolution
        # input layer from which feedforward weights to ratorOut are computed
        ratorIn = nengo.Ensemble( Nin, dimensions=Nobs, radius=reprRadiusIn,
                            bias=nengo.dists.Uniform(1-nrngain,1+nrngain), gain=np.ones(Nin)*nrngain,
                            neuron_type=nengo.neurons.LIF(), seed=seedR1, label='ratorIn' )
        ratorInD = nengo.Ensemble( Nin, dimensions=Nobs, radius=reprRadiusIn,
                            bias=nengo.dists.Uniform(1-nrngain,1+nrngain), gain=np.ones(Nin)*nrngain,
                            neuron_type=nengo.neurons.LIF(), seed=seedR1, label='ratorIn' )
        nengo.Connection(nodeIn, ratorIn, synapse=None)
        nengo.Connection(nodeInD, ratorInD, synapse=None)
                                                                # No filtering here as no filtering/delay in the plant/arm
        # layer with learning incorporated
        #intercepts = np.append(np.random.uniform(-0.2,0.2,size=Nexc//2),np.random.uniform(-1.,1.,size=Nexc//2))
        ratorOut = nengo.Ensemble( Nexc, dimensions=N//2, radius=reprRadius,
                            bias=nengo.dists.Uniform(1-nrngain,1+nrngain), gain=np.ones(Nexc)*nrngain,
                            neuron_type=nengo.neurons.LIF(), seed=seedR2, label='ratorOut')
        # don't use the same seeds across the connections,
        #  else they seem to be all evaluated at the same values of low-dim variables
        #  causing seed-dependent convergence issues possibly due to similar frozen noise across connections
                
        EtoE = nengo.Connection(ratorInD.neurons, ratorOut.neurons,
                                    transform=np.zeros((Nexc,Nin)), synapse=tau)

        # make InEtoE connection after EtoE, so that reprRadius from EtoE
        #  instead of reprRadiusIn from InEtoE is used to compute decoders for ratorOut
        InEtoE = nengo.Connection(ratorIn.neurons, ratorOut.neurons,
                                    transform=np.zeros((Nexc,Nin)), synapse=tau)
        
        error = nengo.Node( size_in=Nobs, output = lambda timeval,err: err )
        # Error = x_desired - x_true
        # important to probe only ratorOut2error as output, and not directly ratorOut
        #  [was needed like this in the learning script, harmless here]
        # 'output' reads out the output of the connection in nengo 2.2 on
        torqueOut_probe = nengo.Probe(ratorOut,synapse=tau)
        nodeIn_probe = nengo.Probe(nodeInTarget, synapse=None)          # actually x_desired

        ###
        ### Add the x_desired and x_true connections to the error ensemble ###
        ###
        stateDesired2error = nengo.Connection(nodeInTarget,error,\
                                                synapse=None,transform=-np.eye(Nobs))
                                                                        # minus desired state
        # connect torque to node for filtering with tau (very important)
        #  and then via arm evolve function to error
        networkTorque = nengo.Node( size_in=N//2, output = lambda t,u: u)
        torque2Node = nengo.Connection(ratorOut,networkTorque,synapse=tau)
        # looks as if torque goes in, but actually via evolveState(), so finally arm state goes in
        stateTrue2error = nengo.Connection(networkTorque,error,\
                                                synapse=None,function=lambda u:evolveState(u))
                                                                        # minus true state
        # 'output' reads out the output of the connection in nengo 2.2 on
        stateTrue_probe = nengo.Probe(stateTrue2error, 'output')

        ###
        ### feed the error back to force output to follow the input ###
        ###
        errorFeedbackConn = nengo.Connection(error,ratorIn,\
                                            synapse=errorFeedbackTau,\
                                            transform=-errorFeedbackGain)
        errorFeedbackConnD = nengo.Connection(error,ratorInD,\
                                            synapse=errorFeedbackTau+tau,\
                                            transform=-errorFeedbackGain)

    #################################
    ### Load pre-learned weights
    #################################

    if OCL:
        sim = nengo_ocl.Simulator(mainModel,dt)
    else:
        sim = nengo.Simulator(mainModel,dt)

    if isfile(weightsLoadFileName):
        print('loading previously learned weights from',weightsLoadFileName)
        with contextlib.closing(
                shelve.open(weightsLoadFileName, 'r', protocol=-1)
                ) as weights_dict:
            #sim.data[EtoE].weights = weights_dict['learnedWeights']    # can't be set, only read
            sim.signals[ sim.model.sig[EtoE]['weights'] ] \
                                = weights_dict['learnedWeights']                    # can be set if weights/decoders are plastic
            sim.signals[ sim.model.sig[InEtoE]['weights'] ] \
                                = weights_dict['learnedInWeights']                  # can be set if weights/decoders are plastic
    else:
        print('Not found pre-learned weights,',weightsLoadFileName)

    def save_data(endTag):
        print('shelving data',endTag)
        # with statement causes close() at the end, else must call close() explictly
        # 'c' opens for read and write, creating it if not existing
        # protocol = -1 uses the highest protocol (currently 2) which is binary,
        #  default protocol=0 is ascii and gives `ValueError: insecure string pickle` on loading
        with contextlib.closing(
                shelve.open(dataFileName+endTag+'.shelve', 'c', protocol=-1)
                ) as data_dict:
            data_dict['trange'] = sim.trange()
            data_dict['dt'] = dt
            data_dict['Tperiod'] = Tperiod
            data_dict['tau'] = tau
            data_dict['ratorOut'] = sim.data[nodeIn_probe]
            data_dict['torqueOut'] = sim.data[torqueOut_probe]
            data_dict['stateTrue'] = sim.data[stateTrue_probe]
            data_dict['trueTorque'] = trueTorque
            data_dict['varFactors'] = varFactors

    ###
    ### run the simulation, with online arm communication ###
    ###
    for i,ti in enumerate(trange):
        sim.run(dt,progress_bar=False)
        if i%1000==0: print ('time elapsed',ti)
    save_data('')

    ###
    ### run the plotting sequence ###
    ###
    print('plotting data')
    #myplot.plot_rec_nengo_all(dataFileName)
    plt.figure()
    plt.plot(trange,trueTorque,color='b')
    plt.plot(trange,sim.data[torqueOut_probe],color='r')
    plt.figure()
    plt.plot(trange,sim.data[nodeIn_probe],color='b')                   # desired state
    plt.plot(trange,sim.data[stateTrue_probe],color='r')                # true state
    plt.show()
