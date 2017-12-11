# -*- coding: utf-8 -*-
# (c) Sep 2015 Aditya Gilra, EPFL.

"""
learning of arbitrary feed-forward or recurrent transforms
in Nengo simulator
written by Aditya Gilra (c) Sep 2015.
"""

import nengo
import nengo_dl
import tensorflow as tf

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

gpunum = os.environ.get('CUDA_VISIBLE_DEVICES')
if gpunum is None: tf_device = None
else: tf_device = '/gpu:0'              # choose the first of the visible devices

########################
### Constants/parameters
########################

###
### Overall parameter control ###
###
noBP = False#True                              # use FOLLOW learning (True) or backprop (False) to learn ff+rec weights
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
initLearned = False and recurrentLearning and not inhibition
                                        # whether to start with learned weights (bidirectional/unclipped)
                                        # currently implemented only for recurrent learning
testLearned = False                     # whether to test the learning, uses weights from continueLearning, but doesn't save again.
testLearnedOn = '_seed2by0.3amplVaryHeights'
#testLearnedOn = '__'                    # doesn't load any weights if file not found! use with initLearned say.
                                        # the string of inputType and trialClamp used for learning the to-be-tested system 
saveSpikes = True                       # save spikes if testLearned and saveSpikes
continueLearning = False                # whether to load old weights and continue learning from there
                                        # doesn't work, maybe save error state, also confirm same encoders/decoders?
                                        # saving weights at the end is always enabled
zeroLowWeights = False                  # set to zero weights below a certain value
weightErrorCutoff = 0.                  # Do not pass any abs(error) for weight change below this value
randomInitWeights = False#True and not plastDecoders and not inhibition
                                        # start from random initial weights instead of zeros
                                        # works only for weights, not decoders as decoders are calculated from transform
randomWeightSD = 1e-4                   # this is a approx SD of weight distribution (~Gaussian)
                                        #  for the LinOsc for min error before error rises
weightRegularize = False                # include a weight decay term to regularize weights

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

# choose one of ReLU or SoftLIFRate, ReLU also works with FOLLOW
#neuron_type = 'ReLU'
neuron_type = 'SoftLIFRate'

if neuron_type=='SoftLIFRate':
    import nengo_extras
    # keep this as similar to the Nengo default LIF as possible!
    tau_rc = 0.02   # seconds, only for neural membrane potential
    if noBP:
        # usual nengo doesn't have SoftLIFRate, not sure if nengo_ocl has softLIFRate though!!
        neuron_type_obj = nengo_extras.SoftLIFRate(tau_rc=tau_rc)
    else:
        neuron_type_obj = nengo_dl.SoftLIFRate(tau_rc=tau_rc)
    #max_rates_dist = nengo.dists.Uniform(200, 400)
else:
    neuron_type_obj = nengo.RectifiedLinear()
    #max_rates_dist = nengo.dists.Uniform(0, 1)

###
### choose dynamics evolution matrix ###
###
#init_vec_idx = -1
init_vec_idx = 0        # first / largest response vector

#evolve = 'EI'          # eigenvalue evolution
#evolve = 'Q'           # Hennequin et al 2014
evolve = 'fixedW'       # fixed W: Schaub et al 2015 / 2D oscillator
#evolve = None           # no recurrent connections, W=zeros

evolve_dirn = 'arb'     # arbitrary normalized initial direction
#evolve_dirn = ''        # along a0, i.e. eigvec of response energy matrix Q
#evolve_dirn = 'eigW'    # along eigvec of W
#evolve_dirn = 'schurW'  # along schur mode of W

# choose between one of the input types
#inputType = 'inputOsc'
#inputType = 'rampLeave'
#inputType = 'rampLeaveDirnVary'
#inputType = 'kickStart'
#inputType = 'persistent'
#inputType = 'persconst'
#inputType = 'amplVary'
inputType = 'amplVaryHeights'
#inputType = 'amplDurnVary'
#inputType = 'nostim'
#inputType = 'RLSwing'
#inputType = 'RLReach1'
#inputType = 'RLReach2'
#inputType = 'RLReach3'
#inputType = 'ShootWriteF'

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
        varFactors = (1./2.5,1./2.5,0.05,0.05,0.02,0.02)    # angleFactors, velocityFactors, torqueFactors
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
    Tmax = 5000.                                       # second - how long to run the simulation
    continueTmax = 10000.                               # if continueLearning, then start with weights from continueTmax
    reprRadius = 1.0                                    # neurons represent (-reprRadius,+reprRadius)
    reprRadiusIn = 0.2                                  # input is integrated in ratorOut, so keep it smaller than reprRadius
    if recurrentLearning:                               # L2 recurrent learning
        #PES_learning_rate = 9e-1                        # learning rate with excPES_integralTau = Tperiod
        #                                                #  as deltaW actually becomes very small integrated over a cycle!
        if testLearned:
            PES_learning_rate_rec = 1e-20               # effectively no learning
            PES_learning_rate_FF = 1e-20                # effectively no learning
        else:
            if noBP:
                PES_learning_rate_FF = 1e-4#2e-3        # nengo 2.6.0 allows max 1e-4, while nengo 2.4.0 allows 2e-3
                PES_learning_rate_rec = 1e-4#2e-3       # nengo 2.6.0 allows max 1e-4, while nengo 2.4.0 allows 2e-3
            else:
                PES_learning_rate_FF = 1e-5#1e-4
                PES_learning_rate_rec = 1e-5#1e-4
        if 'acrobot' in funcType: inputreduction = 0.5  # input reduction factor
        else: inputreduction = 0.3                      # input reduction factor
        Nexc = 500                                      # number of excitatory neurons
        Npre = 200                                      # number of neurons per ff layer that feeds into ratorOut
        Tperiod = 1.                                    # second
        if plastDecoders:                               # only decoders are plastic
            Wdyn2 = np.zeros(shape=(N+N//2,N+N//2))
        else:                                           # weights are plastic, connection is now between neurons
            if randomInitWeights:
                Wdyn2 = np.random.normal(size=(Nexc,Nexc))*randomWeightSD
            else:
                Wdyn2 = np.zeros(shape=(Nexc,Nexc))
        #Wdyn2 = W
        #Wdyn2 = W+np.random.randn(2*N,2*N)*np.max(W)/5.
        Wtransfer = np.eye(N)

Nerror = 200*N                                          # number of error calculating neurons
reprRadiusErr = 0.2                                     # with error feedback, error is quite small

###
### time params ###
###
rampT = 0.5                                             # second
dt = 0.001                                              # second
weightdt = Tmax/20.                                     # how often to probe/sample weights
Tclamp = 0.25                                           # time to clamp the ref, learner and inputs after each trial (Tperiod)
Tnolearning = 4*Tperiod
                                                        # in last Tnolearning s, turn off learning & weight decay

###
### Generate inputs for L1 ###
###
zerosN = np.zeros(N)
zerosNby2 = np.zeros(N//2)
zeros2N = np.zeros(Nobs+N//2)
if inputType == 'rampLeave':
    ## ramp input along y0
    inpfn = lambda t: tau*B/rampT*reprRadius if (t%Tperiod) < rampT else zerosN
elif inputType == 'rampLeaveDirnVary':
    ## ramp input along random directions
    # generate unit random vectors on the surface of a sphere i.e. random directions
    # http://codereview.stackexchange.com/questions/77927/generate-random-unit-vectors-around-circle
    # incorrect to select uniformly from theta,phi: http://mathworld.wolfram.com/SpherePointPicking.html
    if N//2 > 1:                                        # normalized random directions for >1D
        Bt = np.random.normal(size=(N//2,int(Tmax/Tperiod)+1))
                                                        # randomly varying vectors for each Tperiod
        Bt = 4. * Bt/np.linalg.norm(Bt,axis=0)          # multiplied by 2 here after normalizing, later /2 in varFactors
                                                        # multi-dimensional Gaussian distribution goes as exp(-r^2)
                                                        # so normalizing by r gets rid of r dependence, uniform in theta, phi
                                                        # randomly varying vectors for each Tperiod
    else:
        Bt = 4. * np.random.uniform(-1.,1.,size=(N//2,int(Tmax/Tperiod)+1))
                                                        # uniform between (-1,1) for 1D
    if trialClamp:
        inpfn = lambda t: Bt[:,int(t/Tperiod)] * \
                    ( (2*(t%Tperiod)/rampT)*(t%Tperiod<rampT/2.) + 2*(1 - (t%Tperiod)/rampT)*(t%Tperiod>=rampT/2.) ) * \
                    (t%Tperiod<rampT)                   # triangle for rampT and then zero, assuming comparison returns 0 or 1
        #inpfn = lambda t: Bt[:,int(t/Tperiod)] * \
        #            ( (t%Tperiod)/rampT ) * (t%Tperiod<rampT)
        #                                                # ramp for rampT and then zero, assuming comparison returns 0 or 1
    else:
        inpfns = [ interp1d(np.linspace(0,Tmax,int(Tmax/Tperiod)+1),Bt[i,:],axis=0,kind='cubic',\
                            bounds_error=False,fill_value=0.) for i in range(N//2) ]
        inpfn = lambda t: np.array([ inpfns[i](t) for i in range(N//2) ])
                                                        # torque should not depend on reprRadius, unlike for other funcType-s.
elif inputType == 'kickStart':
    ## ramp input along y0 only once initially, for self sustaining func-s
    inpfn = lambda t: tau*B/rampT*reprRadius if t < rampT else zerosN
elif inputType == 'persistent':
    ## decaying ramp input along y0,
    inpfn = lambda t: exp(-(t%Tperiod)/Tperiod)*tau*B/rampT*reprRadius \
                        if (t%(Tperiod/5.)) < rampT else zerosN
                                                        # Repeat a ramp 5 times within Tperiod
                                                        #  with a decaying envelope of time const Tperiod
                                                        # This whole sequence is periodic with Tperiod
elif inputType == 'persconst':
    ## ramp input along y0 with a constant offset at other times
    constN = np.ones(N)*tau*3
    inpfn = lambda t: tau*B/rampT*reprRadius if (t%Tperiod) < rampT else constN
elif inputType == 'amplVary':
    ## random uniform 'white-noise'
    noiseN = np.random.uniform(-2*reprRadius,2*reprRadius,size=int(1200./rampT))
    inpfn = lambda t: (noiseN[int(t/rampT)]*tau*B/rampT)*reprRadius \
                        if t<(Tmax-Tnolearning) else \
                        (tau*B/rampT*reprRadius if (t%Tperiod) < rampT else zerosN)
elif inputType == 'amplVaryHeights':
    heights = np.random.normal(size=(N//2,int(Tmax/Tperiod)+1))
    heights = heights/np.linalg.norm(heights,axis=0)/inputreduction
    ## random uniform 'white-noise' with 50 ms steps interpolated
    ##  50ms is longer than spiking-network response-time, and assumed shorter than tau-s of the dynamical system.
    noisedt = 50e-3
    # cubic interpolation for long sim time takes up ~64GB RAM and then hangs, so linear or nearest interpolation.
    noiseN = np.random.uniform(-reprRadius/inputreduction,reprRadius/inputreduction,size=(N//2,int(Tmax/noisedt)+1))
    noisefunc = interp1d(np.linspace(0,Tmax,int(Tmax/noisedt)+1),noiseN,kind='linear',\
                                            bounds_error=False,fill_value=np.zeros(N//2),axis=1)
    heightsfunc = interp1d(np.linspace(0,Tmax,int(Tmax/Tperiod)+1),heights,kind='linear',\
                                            bounds_error=False,fill_value=np.zeros(N//2),axis=1)
    del noiseN
    if trialClamp:
        #inpfn = lambda t: (noisefunc(t) + heights[:,int(t/Tperiod)]*reprRadius) * ((t%Tperiod)<(Tperiod-Tclamp))
        inpfn = lambda t: (noisefunc(t) + heightsfunc(t)*reprRadius) * ((t%Tperiod)<(Tperiod-Tclamp))
    else:
        #inpfn = lambda t: noisefunc(t) + heights[:,int(t/Tperiod)]*reprRadius
        inpfn = lambda t: noisefunc(t) + heightsfunc(t)*reprRadius
elif inputType == 'amplDurnVary':
    ## random uniform 'white-noise', with duration of each value also random
    noiseN = np.random.uniform(-2*reprRadius,2*reprRadius,size=int(1200./rampT))
    durationN = np.random.uniform(rampT,Tperiod,size=int(1200./rampT))
    cumDurationN = np.cumsum(durationN)
    # searchsorted returns the index where t should be placed in sort order
    inpfn = lambda t: (noiseN[np.searchsorted(cumDurationN,t)]*tau*B*reprRadius/rampT) \
                        if t<(Tmax-Tnolearning) else \
                        (tau*B/rampT*reprRadius if (t%Tperiod) < rampT else zerosN)
elif inputType == 'inputOsc':
    ## oscillatory input in all input dimensions
    omegas = 2*np.pi*np.random.uniform(1,3,size=N)      # 1 to 3 Hz
    phis = 2*np.pi*np.random.uniform(size=N)
    inpfn = lambda t: np.cos(omegas*t+phis)
elif 'RL' in inputType:
    ## load the input learned via reinforcement learning
    import pickle
    if 'todorov' in funcType:
        bot,animdt = 'arm_2link',0.1
    else:
        bot,animdt = 'acrobot_2link',0.2
    if 'Swing' in inputType: task = 'swing'
    elif 'Reach1' in inputType: task = 'reach1'
    elif 'Reach2' in inputType: task = 'reach2'
    elif 'Reach3' in inputType: task = 'reach3'
    timetaken,torqueArray = pickle.load( open( bot+'_'+task+"_data.pickle", "rb" ) )
    torqueTmax = 50000.                     # maximum possible torque length during RL;
                                            #  only first few s are valid currently.
    inpfn = interp1d(np.arange(0.,torqueTmax,animdt),torqueArray,axis=0,\
                                bounds_error=False,fill_value=np.zeros(N//2),kind='linear')
elif 'Shoot' in inputType:
    ## load the input learned via reinforcement learning
    import pickle
    if 'todorov' in funcType:
        bot = 'arm_2link'
    else:
        bot = 'acrobot_2link'
    robotdt = 0.01
    if 'WriteF' in inputType: task = 'write_f'
    robotdts,torqueArray = pickle.load( open( bot+'_'+task+"_data.pickle", "rb" ) )
    torqueTmax = 50000.                     # maximum possible torque length during RL;
                                            #  only first few s are valid currently.
    inpfn = interp1d(np.arange(0.,torqueTmax,robotdt),torqueArray,axis=0,\
                                bounds_error=False,fill_value=np.zeros(N//2),kind='nearest')
else:
    inpfn = lambda t: 0.0*np.ones(N)*reprRadius*tau     # constant input, currently zero
    #inpfn = None                                        # zero input

if errorLearning:
    if not weightRegularize:
        excPES_weightsDecayRate = 0.        # no decay of PES plastic weights
    else:
        excPES_weightsDecayRate = 1./1e4    # 1/tau of synaptic weight decay for PES plasticity 
        #if excPES_weightsDecayRate != 0.: PES_learning_rate /= excPES_weightsDecayRate
                                            # no need to correct PES_learning_rate,
                                            #  it's fine in ElementwiseInc in builders/operator.py
    #excPES_integralTau = 1.                 # tau of integration of deltaW for PES plasticity 
    excPES_integralTau = None               # don't integrate deltaW for PES plasticity (default) 
    copycatLayer = False                    # whether to use odeint rate_evolve or another copycat layer
                                            #  for generating the expected response signal for error computation
    errorAverage = False                    # whether to average error over the Tperiod scale
                                            # Nopes, this won't make it learn the intricate dynamics
    #tauErrInt = tau*5                       # longer integration tau for err -- obsolete (commented below)
    errorFeedbackGain = 10.                 # Feedback gain
                                            # below a gain of ~5, exc rates go to max, weights become large
    weightErrorTau = 10*tau                 # filter the error to the PES weight update rule
    errorFeedbackTau = 1*tau                # synaptic tau for the error signal into layer2.ratorOut
    errorGainDecay = False                  # whether errorFeedbackGain should decay exponentially to zero
                                            # decaying gain gives large weights increase below some critical gain ~3
    errorGainDecayRate = 1./200.            # 1/tau for decay of errorFeedbackGain if errorGainDecay is True
    errorGainProportion = False             # scale gain proportionally to a long-time averaged |error|
    errorGainProportionTau = Tperiod        # time scale to average error for calculating feedback gain
if errorLearning and recurrentLearning:
    inhVSG_weightsDecayRate = 1./40.
else:
    inhVSG_weightsDecayRate = 1./2.         # 1/tau of synaptic weight decay for VSG plasticity
#inhVSG_weightsDecayRate = 0.               # no decay of inh VSG plastic weights

#pathprefix = '/lcncluster/gilra/tmp/'
pathprefix = '../data/'
inputStr = ('_trials' if trialClamp else '') + \
        ('_seed'+str(seedRin)+'by'+str(inputreduction)+inputType if inputType != 'rampLeave' else '')
baseFileName = pathprefix+'inverse_diff_ff_mod_N'+str(Npre)+'_50ms'+('_DLnoBP' if noBP else '_DL')+'_Nexc'+str(Nexc) + \
                    ('_softLIF' if neuron_type == 'SoftLIFRate' else '_relu') + \
                    '_seeds'+str(seedR0)+str(seedR1)+str(seedR2)+str(seedR4) + \
                    ('_inhibition' if inhibition else '') + \
                    ('_zeroLowWeights' if zeroLowWeights else '') + \
                    '_weightErrorCutoff'+str(weightErrorCutoff) + \
                    ('_randomInitWeights'+str(randomWeightSD) if randomInitWeights else '') + \
                    ('_weightRegularize'+str(excPES_weightsDecayRate) if weightRegularize else '') + \
                    '_nodeerr' + ('_plastDecoders' if plastDecoders else '') + \
                    (   (   '_learn' + \
                            ('_rec' if recurrentLearning else '_ff') + \
                            ('' if errorFeedback else '_noErrFB') \
                        ) if errorLearning else '_algo' ) + \
                    ('_initLearned' if initLearned else '') + \
                    ('_learnIfNoInput' if learnIfNoInput else '') + \
                    ('' if copycatLayer else '_nocopycat') + \
                    ('_func_'+funcType if learnFunction else '') + \
                    (testLearnedOn if (testLearned or continueLearning) else inputStr)
                        # filename to save simulation data
dataFileName = baseFileName + \
                    ('_continueFrom'+str(continueTmax)+inputStr if continueLearning else '') + \
                    ('_testFrom'+str(continueTmax)+inputStr if testLearned else '') + \
                    '_'+str(Tmax)+'s'
print('data will be saved to', dataFileName, '_<start|end|currentweights>.shelve')
if continueLearning or testLearned:
    weightsSaveFileName = baseFileName + '_'+str(continueTmax+Tmax)+'s_endweights.shelve'
    weightsLoadFileName = baseFileName + '_'+str(continueTmax)+'s_endweights.shelve'
else:
    weightsSaveFileName = baseFileName + '_'+str(Tmax)+'s_endweights.shelve'
    weightsLoadFileName = baseFileName + '_'+str(Tmax)+'s_endweights.shelve'    

###
### Get data from the vrep robotics simulation server, or reload older sim data
###
robDataFileName = pathprefix+'general_learn_data' + \
                    '_trials_seeds'+str(seedR0)+str(seedR1)+str(seedR2)+str(seedR4) + \
                    ('_func_'+funcType if learnFunction else '') + \
                    '_'+str(Tmax)+'s' + \
                    ('_by'+str(inputreduction)+inputType if inputType != 'rampLeave' else '')
                        # filename to save vrep robot simulation data
print('robot sim will be saved to',robDataFileName)

#trange = np.arange(0,Tmax,dt)
#plt.figure()
#plt.plot(trange,[inpfn(t) for t in trange])
#plt.show()
#sys.exit()

from sim_robot import sim_robot
robtrange,rateEvolveProbe,evolveFns,armAngles = \
    sim_robot(robotType,funcType,reloadRobotSim,robDataFileName,Tmax,inpfn,trialClamp,Tperiod,Tclamp,dt)

if initLearned:
    if 'XY' in funcType: XY = True
    else: XY = False
    def Wdesired(x):
        ''' x is the augmented variable represented in the network 
            it obeys \tau_s x_\alpha = -x_\alpha + Wdesired_\alpha(x) 
            x is related to the original augmented variable \tilde{x} by x_\alpha = varFactors_\alpha \tilde{x}_\alpha 
            where varFactors_alpha = angleFactor | velocityFactor | torqueFactor
            now, original augmented variable obeys \dot{\tilde{x}}=f(\tilde{x})
            so, we have Wdesired_\alpha(x) = \tau_s * varFactor_alpha * f_\alpha(\tilde{x}) + x
        '''
        # \tilde{x} (two zeroes at x[N:N+N//2] are ignored
        xtilde = x/varFactors
        if XY: angles = armAngles(xtilde[:N])
        else: angles = xtilde[:N//2]
        # f(\tilde{x}), \dot{u} part is not needed
        qdot,dqdot = evolveFns(angles,xtilde[Nobs-N//2:Nobs],xtilde[Nobs:],XY,dt)
                                                                        # returns deltaposn if XY else deltaangles
        # \tau_s * varFactors_alpha * f_\alpha(\tilde{x}) + x
        return np.append(np.append(qdot,dqdot),np.zeros(N//2))*varFactors*tau + x
                                                                        # integral on torque u also
                                                                        # VERY IMP to compensate for synaptic decay on torque
        #return np.append( np.append(qdot,dqdot)*tau*varFactors[:Nobs] + x[:Nobs], np.zeros(N//2) )
                                                                        # normal synaptic decay on torque u

    ##### For the reference, choose EITHER robot simulation rateEvolveProbe above
    #####  OR evolve Wdesired inverted / evolveFns using odeint as below -- both should be exactly same
    def matevolve2(y,t):
        ''' the reference y is only N-dim i.e. (q,dq), not 2N-dim, as inpfn is used directly as reference for torque u
        '''
        ## invert the nengo function transformation with angleFactor, tau, +x, etc. in Wdesired()
        ## -- some BUG, inversion is not working correctly
        #xfull = np.append(y,inpfn(t))*varFactors
        #return ( (Wdesired(xfull)[:N]/tau/varFactors[:N] - xfull[:N]) \
        #                        if (t%Tperiod)<(Tperiod-Tclamp) else -x/tau )
        # instead of above, directly use evolveFns()
        #########  DOESN'T WORK: should only use armAngles() with valid posn, not all posn-s are valid for an arm!!!  ###########
        if XY: angles = armAngles(y[:N])
        else: angles = y[:N//2]        
        # evolveFns returns deltaposn if XY else deltaangles
        if trialClamp:
            return ( evolveFns( angles, y[Nobs-N//2:Nobs], inpfn(t), XY, dt).flatten()\
                        if (t%Tperiod)<(Tperiod-Tclamp) else -y/dt )
        else:
            return evolveFns( angles, y[Nobs-N//2:Nobs], inpfn(t), XY, dt).flatten()

    ##### uncomment below to override rateEvolveProbe by matevolve2-computed Wdesired-inversion / evolveFns-evolution, as reference signal
    #trange = np.arange(0.0,Tmax,dt)
    #y = odeint(matevolve2,0.001*np.ones(N),trange,hmax=dt)  # set hmax=dt, to avoid adaptive step size
    #                                                        # some systems (van der pol) have origin as a fixed pt
    #                                                        # hence start just off-origin
    #rateEvolveProbe = y                                     # only copies pointer, not full array (no need to use np.copy() here)

###
### Reference evolution used when copycat layer is not used for reference ###
###

# scale the output of the robot simulation or odeint to cover the representation range of the network
# here I scale by angle/velocity factors, below at nodeIn I scale by torque factors.
rateEvolveProbe *= varFactors[:Nobs]
rateEvolveFn = interp1d(robtrange,rateEvolveProbe,axis=0,kind='linear',\
                        bounds_error=False,fill_value=np.zeros(Nobs))
                                                        # used for the error signal below
                                      

## this color cycle doesn't seem to work!?
##plt.gca().set_color_cycle(['red', 'green', 'blue', 'cyan','magenta','yellow','black'])
#plt.figure(facecolor='w')
#plt.plot(robtrange,rateEvolveProbe[:,0],label='$\\theta_0$')
#plt.plot(robtrange,rateEvolveProbe[:,1],label='$\\theta_1$')
#plt.plot(robtrange,rateEvolveProbe[:,2],label='$\omega_0$')
#plt.plot(robtrange,rateEvolveProbe[:,3],label='$\omega_1$')
#plt.legend()
#plt.show()
#sys.exit()
del robtrange,rateEvolveProbe                               # free some memory

if __name__ == "__main__":
    #########################
    ### Create Nengo network
    #########################
    print('building model')
    mainModel = nengo.Network(label="Single layer network", seed=seedR0)
    with mainModel:
        # don't train any weights via backprop, except those specified later
        if not noBP: nengo_dl.configure_settings(trainable=False)
        rateEvolve = nengo.Node(rateEvolveFn)                   # reference state evolution
        rateEvolveD = nengo.Node(lambda t: rateEvolveFn(t-0.025))# delayed reference state evolution
        nodeIn = nengo.Node( size_in=N//2, output = lambda timeval,currval: inpfn(timeval-0.05)*varFactors[Nobs:] )
                                                                # reference input torque evolution
                                                                # scale input to network by torque factors
        # input layer from which feedforward weights to ratorOut are computed
        ratorIn = nengo.Ensemble( Npre, dimensions=Nobs, radius=reprRadiusIn,
                            neuron_type=neuron_type_obj, seed=seedR1, label='ratorIn' )
        ratorInD = nengo.Ensemble( Npre, dimensions=Nobs, radius=reprRadiusIn,
                            neuron_type=neuron_type_obj, seed=seedR1, label='ratorIn' )
        nengo.Connection(rateEvolve, ratorIn, synapse=None)
        nengo.Connection(rateEvolveD, ratorInD, synapse=None)
                                                                # No filtering here as no filtering/delay in the plant/arm
        # layer with learning incorporated
        #intercepts = np.append(np.random.uniform(-0.2,0.2,size=Nexc//2),np.random.uniform(-1.,1.,size=Nexc//2))
        ratorOut = nengo.Ensemble( Nexc, dimensions=N//2, radius=reprRadius,
                            neuron_type=neuron_type_obj, seed=seedR2, label='ratorOut')
        # don't use the same seeds across the connections,
        #  else they seem to be all evaluated at the same values of low-dim variables
        #  causing seed-dependent convergence issues possibly due to similar frozen noise across connections
        
        if trialClamp:
            # clamp ratorOut at the end of each trial (Tperiod) for 100ms.
            #  Error clamped below during end of the trial for 100ms.
            clampValsZeros = np.zeros(Nexc)
            clampValsNegs = -100.*np.ones(Nexc)
            endTrialClamp = nengo.Node(lambda t: clampValsZeros if (t%Tperiod)<(Tperiod-Tclamp) else clampValsNegs)
            nengo.Connection(endTrialClamp,ratorOut.neurons,synapse=1e-3)
                                                                    # fast synapse for fast-reacting clamp
        
        EtoE = nengo.Connection(ratorInD.neurons, ratorOut.neurons,
                                    transform=np.zeros((Nexc,Npre)), synapse=tau)   # synapse is tau_syn for filtering

        # make InEtoE connection after EtoE, so that reprRadius from EtoE
        #  instead of reprRadiusIn from InEtoE is used to compute decoders for ratorOut
        InEtoE = nengo.Connection(ratorIn.neurons, ratorOut.neurons,
                                    transform=np.zeros((Nexc,Npre)), synapse=tau)
                                                                    # Wdyn2 same as for EtoE, but mean(InEtoE) = mean(EtoE)/20

        nodeIn_probe = nengo.Probe(nodeIn, synapse=None)

        if testLearned and saveSpikes:
            ratorOut_EspikesOut = nengo.Probe(ratorOut.neurons, 'output')
                                                                    # this becomes too big for shelve (ndarray.dump())
                                                                    #  for my Lorenz _end simulation of 100s
                                                                    #  gives SystemError: error return without exception set
                                                                    # use python3.3+ or break into smaller sizes
                                                                    # even with python3.4, TypeError: gdbm mappings have byte or string elements only

    ############################
    ### Learn ratorOut EtoE connection
    ############################
    with mainModel:
        if errorLearning:
            ###
            ### error ensemble, could be with error averaging, gets post connection ###
            ###
            error = nengo.Node( size_in=N//2, output = lambda timeval,err: err )
            if trialClamp:
                errorOff = nengo.Node( size_in=N//2, output = lambda timeval,err: \
                                            (err if timeval<(Tmax-Tnolearning) else zerosNby2) \
                                            if ((timeval%Tperiod)<Tperiod-Tclamp and (timeval>Tperiod)) \
                                            else zerosNby2 )
            else:
                errorOff = nengo.Node( size_in=N//2, output = lambda timeval,err: \
                                            (err if (timeval<(Tmax-Tnolearning) and (timeval>Tperiod)) \
                                            else zerosNby2) )
            error2errorOff = nengo.Connection(error,errorOff,synapse=None)
            if errorAverage:                                        # average the error over Tperiod time scale
                errorT = np.eye(Nobs)*(1-tau/Tperiod*dt/Tperiod)# neuralT' = tau*dynT + I
                                                                    # dynT=-1/Tperiod*dt/Tperiod
                                                                    # *dt/Tperiod converts integral to mean
                nengo.Connection(errorOff,errorOff,transform=errorT,synapse=tau)
            # Error = post - pre * desired_transform
            ratorOut2error = nengo.Connection(ratorOut,error,synapse=tau)
                                                            # post input to error ensemble (pre below)
            # important to probe only ratorOut2error as output, and not directly ratorOut, to accommodate randomDecodersType != ''
            # 'output' reads out the output of the connection in nengo 2.2
            ratorOut_probe = nengo.Probe(ratorOut2error, 'output')

            ###
            ### Add the relevant pre signal to the error ensemble ###
            ###
            if recurrentLearning:                           # L2 rec learning
                # Error = post - desired_output
                inpfn2error = nengo.Connection(nodeIn,error,synapse=None,transform=-np.eye(N//2))
                                                        # - desired output (post above)
                plasticConnEE = EtoE
                rateEvolve_probe = nengo.Probe(inpfn2error, 'output')

            if trialClamp:
                # if trialClamp just forcing error to zero doesn't help, as errorWt decays at long errorWeightTau,
                #  so force errorWt also to zero, so that learning is shutoff at the end of a trial
                errorWt = nengo.Node( size_in=N//2, output = lambda timeval,errWt: \
                                            errWt if ((timeval%Tperiod)<Tperiod-Tclamp and timeval<Tmax-Tnolearning) \
                                            else zerosNby2 )
                                                            # To Do: implement weightErrorCutoff only on errWt[0:N] above
            else:
                errorWt = nengo.Node( size_in=N//2, output = lambda timeval,errWt: \
                                            errWt*(np.abs(errWt)>weightErrorCutoff) if timeval<Tmax-Tnolearning \
                                            else zerosNby2 )
            nengo.Connection(errorOff,errorWt,synapse=weightErrorTau)
                                                            # error to errorWt ensemble, filter for weight learning
                                                            # rest of errorWt beyond Nobs will be zero by default (no connection)

            ### Feed the error back and learn EtoE, etc connections via FOLLOW, only if noBP i.e. no backprop, else backprop
            if noBP:
                ###
                ### Add the exc learning rules to the connection, and the error ensemble to the learning rule ###
                ###
                EtoERulesDict = { 'PES' : nengo.PES(learning_rate=PES_learning_rate_rec,
                                                pre_tau=tau) }#,
                                                #clipType=excClipType,
                                                #decay_rate_x_dt=excPES_weightsDecayRate*dt,
                                                #integral_tau=excPES_integralTau) }
                plasticConnEE.learning_rule_type = EtoERulesDict
                #plasticConnEE.learning_rule['PES'].learning_rate=0
                                                                # learning_rate has no effect
                                                                # set to zero, yet works fine!
                                                                # It works only if you set it
                                                                # in the constructor PES() above

                # feedforward learning rule
                InEtoERulesDict = { 'PES' : nengo.PES(learning_rate=PES_learning_rate_FF,
                                                pre_tau=tau) }#,
                                                #clipType=excClipType,
                                                #decay_rate_x_dt=excPES_weightsDecayRate*dt,
                                                #integral_tau=excPES_integralTau) }
                InEtoE.learning_rule_type = InEtoERulesDict

                error_conn = nengo.Connection(\
                        errorWt,plasticConnEE.learning_rule['PES'],synapse=dt)
                error_conn = nengo.Connection(\
                        errorWt,InEtoE.learning_rule['PES'],synapse=dt)

                ###
                ### feed the error back to force output to follow the input (for both recurrent and feedforward learning) ###
                ###
                if errorFeedback and not testLearned:
                    errorFeedbackConn = nengo.Connection(errorOff,ratorOut,\
                            synapse=errorFeedbackTau,\
                            transform=-errorFeedbackGain)#*(np.random.uniform(-0.1,0.1,size=(N,N))+np.eye(N)))
                                                            # PES with error unconnected, so only decay
            else:
                # make some connection weights trainable by backprop (ignored for follow)
                mainModel.config[EtoE].trainable = True
                mainModel.config[InEtoE].trainable = True
                ## make some ensemble.neurons biases trainable by backprop (ignored for follow)
                #mainModel.config[ratorIn.neurons].trainable = False
                #mainModel.config[ratorOut.neurons].trainable = False

            ###
            ### error and weight probes ###
            ###
            errorOn_p = nengo.Probe(error, synapse=None, label='errorOn')
            error_p = nengo.Probe(errorWt, synapse=None, label='error')
            if not noBP and Nexc<=4000:                                  # GPU mem is not large enough to probe large weight matrices
                #learnedInWeightsProbe = nengo.Probe(\
                #            InEtoE,'weights',sample_every=weightdt,label='InEEweights')
                learnedWeightsProbe = nengo.Probe(\
                            plasticConnEE,'weights',sample_every=weightdt,label='EEweights')

    #################################
    ### Build Nengo network
    #################################

    #if noBP:
    #    sim = nengo_ocl.Simulator(mainModel,dt)
    #else:
    sim = nengo_dl.Simulator(mainModel,dt,device=tf_device,
                    minibatch_size=None,
                    tensorboard='arm_tensorboard')
    Eencoders = sim.data[ratorOut].encoders
    
    #################################
    ### load previously learned weights, if requested and file exists
    #################################
    if errorLearning and (continueLearning or testLearned) and isfile(weightsLoadFileName):
        print('loading previously learned weights from',weightsLoadFileName)
        sim.load_params(weightsLoadFileName, include_global=True, include_local=False)
    else:
        print('Not loading any pre-learned weights.')

    def save_data(endTag):
        #print 'pickling data'
        #pickle.dump( data_dict, open( "/lcncluster/gilra/tmp/rec_learn_data.pickle", "wb" ) )
        print('shelving data',endTag)
        # with statement causes close() at the end, else must call close() explictly
        # 'c' opens for read and write, creating it if not existing
        # protocol = -1 uses the highest protocol (currently 2) which is binary,
        #  default protocol=0 is ascii and gives `ValueError: insecure string pickle` on loading
        with contextlib.closing(
                shelve.open(dataFileName+endTag+'.shelve', 'c', protocol=-1)
                ) as data_dict:
            data_dict['trange'] = sim.trange()
            data_dict['Tmax'] = Tmax
            data_dict['rampT'] = rampT
            data_dict['Tperiod'] = Tperiod
            data_dict['dt'] = dt
            data_dict['tau'] = tau
            data_dict['ratorOut'] = sim.data[nodeIn_probe]
            data_dict['ratorOut2'] = sim.data[ratorOut_probe]
            data_dict['errorLearning'] = errorLearning
            data_dict['varFactors'] = varFactors
            data_dict['spikingNeurons'] = False
            if testLearned and saveSpikes:
                data_dict['EspikesOut2'] = sim.data[ratorOut_EspikesOut]
            data_dict['rateEvolve'] = rateEvolveFn(sim.trange())
            if errorLearning:
                data_dict['recurrentLearning'] = recurrentLearning
                data_dict['error'] = sim.data[errorOn_p]
                data_dict['error_p'] = sim.data[error_p]
                data_dict['copycatLayer'] = copycatLayer
                if recurrentLearning:
                    data_dict['rateEvolveFiltered'] = sim.data[rateEvolve_probe]
                    if copycatLayer:
                        data_dict['yExpectRatorOut'] = sim.data[expectOut_probe]

    def save_weights_evolution():
        if Nexc>4000 or noBP: return                                     # GPU runs are unable to probe large weight matrices
        print('shelving weights evolution')
        # with statement causes close() at the end, else must call close() explictly
        # 'c' opens for read and write, creating it if not existing
        # protocol = -1 uses the highest protocol (currently 2) which is binary,
        #  default protocol=0 is ascii and gives `ValueError: insecure string pickle` on loading
        with contextlib.closing(
                shelve.open(dataFileName+'_weights.shelve', 'c', protocol=-1)
                ) as data_dict:
            data_dict['Tmax'] = Tmax
            data_dict['errorLearning'] = errorLearning
            if errorLearning:
                data_dict['recurrentLearning'] = recurrentLearning
                data_dict['learnedWeights'] = sim.data[learnedWeightsProbe]
                data_dict['learnedInWeights'] = sim.data[learnedInWeightsProbe]
                data_dict['copycatLayer'] = copycatLayer
                #if recurrentLearning and copycatLayer:
                #    data_dict['copycatWeights'] = EtoEweights
                #    data_dict['copycatWeightsPert'] = EtoEweightsPert

    optimizer = tf.train.MomentumOptimizer(learning_rate=PES_learning_rate_FF, momentum=0.9, use_nesterov=True)
    def get_bp_args(tstart,tend):
        trangehere = np.arange(tstart,tend,dt)
        # [numSamples,tsteps,varnum] -- Here, one training sample, for trangehere steps, with Nobs vars for input
        return { 'inputs' : {rateEvolve:np.array([[rateEvolveFn(t) for t in trangehere]]),
                            rateEvolveD:np.array([[rateEvolveFn(t-0.025) for t in trangehere]])},
                    'targets' : {ratorOut_probe:np.array([[inpfn(t-0.05)*varFactors[Nobs:] for t in trangehere]])},
                    'optimizer' : optimizer, 'n_epochs' : 1 }

    _,_,_,_,realtimeold = os.times()
    def sim_run_flush(tFlush,nFlush):
        '''
            Run simulation for nFlush*tFlush seconds,
            Flush probes every tFlush of simulation time,
              (only flush those that don't have 'weights' in their label names)
        '''        
        weighttimeidxold = 0
        #doubledLearningRate = False
        for num,duration in enumerate([tFlush]*nFlush):
            _,_,_,_,realtime = os.times()
            print("Finished till",sim.time,'s, in',realtime-realtimeold,'s')
            print("Simulation time",Tnolearning+num*tFlush,"s")
            sys.stdout.flush()
            # run simulation for tFlush duration
            if noBP:
                sim.run(duration,progress_bar=False)
            else:
                sim.train(**get_bp_args(Tnolearning+num*tFlush,Tnolearning+(num+1)*tFlush))

    if noBP:
        ###
        ### run the simulation, with flushing for learning simulations ###
        ###
        if errorLearning:
            sim.run(Tnolearning)
            save_data('_start')
            nFlush = int((Tmax-2*Tnolearning)/Tperiod)
            sim_run_flush(Tperiod,nFlush)                               # last Tperiod remains (not flushed)
            sim.run(Tnolearning)
            save_data('_end')
        else:
            sim.run(Tmax)
            save_data('')
        #save_weights_evolution()
    else:
        if errorLearning:
            # NOTE: sim.train() in sim_run_flush() does not generate simulation data, Nengo's sim.time remains at zero!
            # Thus Tnolearning periods at start and end are being run usually.
            # Now the test data at the end is from 4 to 8s which is same as training data -- to rectify.
            sim.run(Tnolearning)
            save_data('_start')
            tFlush = 10.
            nFlush = int(Tmax/tFlush)
            sim_run_flush(tFlush,nFlush)
            sim.run(Tnolearning)
            save_data('_end')
        else:
            # [numSamples,tsteps,varnum] -- Here, one training sample, for trangehere steps, with Nobs vars for input
            sim.run(Tmax)
            save_data('')

    ###
    ### save the final learned exc weights ###
    ###
    if errorLearning and not testLearned:
        sim.save_params(weightsSaveFileName, include_global=True, include_local=False)
        print('saved end weights to',weightsSaveFileName)

    ###
    ### run the plotting sequence ###
    ###
    print('plotting data')
    myplot.plot_rec_nengo_all(dataFileName)
