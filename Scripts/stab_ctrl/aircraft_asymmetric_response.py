import control as cl
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("DATA")
from Cit_par23 import Parameters
from Import import load_ac_data
from Import import dat
from transfer import halfAmpTime, oscPeriod, timeConstant 



def asymmetric_motion(trim_aileron_in_deg,trim_rudder_in_deg,t_start, t_final, eigenmotion,initial_sideslip_guess,dat,group='B37'):

    ##########READ DATA##################################
    #####################################################
    P=Parameters()

    all_data = load_ac_data(group)
    rudder_input_arr=all_data['Deflection of rudder'].flatten()*np.pi/180
    rudder_input_arr=rudder_input_arr[t_start:t_final]
    rudder_input_arr=rudder_input_arr+np.ones(len(rudder_input_arr))*trim_rudder_in_deg*np.pi/180 ###ADDED BC RUDDER INPUT IS REVERSED

    aileron_input_arr=all_data['Deflection of aileron (right wing?)'].flatten()*np.pi/180
    aileron_input_arr=aileron_input_arr[t_start:t_final]
    aileron_input_arr=aileron_input_arr-np.ones(len(aileron_input_arr))*trim_aileron_in_deg*np.pi/180  

    t_arr = np.transpose(all_data['T']).flatten()
    t_arr = t_arr[t_start:t_final]
    time_offset = t_arr[0]
    for i in range(len(t_arr)):
        t_arr[i] -= time_offset

    roll_rate = all_data['Body Roll Rate'].flatten()*np.pi/180
    roll_rate =roll_rate[t_start:t_final]

    yaw_rate = all_data['Body Yaw Rate'].flatten()*np.pi/180
    yaw_rate =yaw_rate[t_start:t_final]

    roll_angle = all_data['Roll Angle'].flatten()*np.pi/180
    roll_angle =roll_angle[t_start:t_final]

    aoa_array = all_data['Angle of attack'].flatten()*np.pi/180
    aoa_array = aoa_array[t_start:t_final]

    tas_array = all_data['True Airspeed'].flatten()/1.94384
    tas_array = tas_array[t_start:t_final]

    #######IMPORT V0, mub and CL from citPar23#############
    ########################################################
    P.calculated_parameters(dat,min=t_start,max=t_final,run_calc=False)
    V0=P.V0[0]
    mub=P.mub[0]
    CL=P.CL[0]
    aoa0=aoa_array[0]

        
    ##########################CREATE A,B,C,D MATRICES#######
    ########################################################


    Ca=np.array([[1,0,0,0],[0,1,0,0],
                          [0,0,1,0],
                          [0,0,0,1]])

    Da=np.array([[0,0],[0,0],
                           [0,0],
                           [0,0]])


    C1a=np.array([[(P.CYbdot-2*mub)*P.b/V0,0,0,0],        #Checked
                            [0,-0.5*P.b/V0 ,0,0],        #Checked
                            [0,0,-2*mub*P.KX2*(P.b/V0)**2,2*mub*P.KXZ*(P.b/V0)**2],   #Checked
                            [P.Cnbdot*P.b/V0,0,2*mub*P.KXZ*(P.b/V0)**2,-2*mub*P.KZ2*(P.b/V0)**2]])  #Checked

    C2a=np.array([[P.CYb,CL,P.CYp*P.b/(2*V0),(P.CYr-4*mub)*P.b/(2*V0)],        #Checked
                            [0,0,P.b/(2*V0),0],                     #Checked
                            [P.Clb,0,P.Clp*P.b/(2*V0),P.Clr*P.b/(2*V0)],  #Checked
                            [P.Cnb,0,P.Cnp*P.b/(2*V0),P.Cnr*P.b/(2*V0)]])     #Checked

    C3a=np.array([[P.CYda,P.CYdr],     #Checked
                  [0,0],               #Checked
                  [P.Clda, P.Cldr],    #Checked
                  [P.Cnda, P.Cndr]])   #Checked

    Aa=-np.matmul(np.linalg.inv(C1a),C2a)

    Ba=-np.matmul(np.linalg.inv(C1a),C3a)

    sys=cl.ss(Aa, Ba, Ca, Da)

    input_array=[[],[]]
    input_array[0]=aileron_input_arr
    input_array[1]=-rudder_input_arr
    
    initial_conditions = np.array([initial_sideslip_guess,roll_angle[0],roll_rate[0],yaw_rate[0]])
    #initial_conditions = np.array([0,0,0,0])
    
    
    t_arr, y = cl.forced_response(sys, t_arr, input_array, initial_conditions)
    eigenvalues=np.linalg.eigvals(Aa)
    #output array is = [sideslip,roll (phi), p (roll rate), r  (yaw rate)] , same as state vector

    #####PRESENTATION OF RESULTS###########
    #######################################

    fig, axs = plt.subplots(4, 2)

    axs[0, 0].plot(t_arr, y[0]*180/np.pi, 'tab:red')
    #axs[0, 0].set_title('Sideslip Angle (rad)')
    axs[0, 0].set(xlabel='time', ylabel='Sideslip Angle (deg)')

    axs[0, 1].plot(t_arr, y[1]*180/np.pi, 'tab:red')
    axs[0, 1].plot(t_arr, roll_angle*180/np.pi, 'tab:orange')
    #axs[0, 1].set_title('Roll Angle (rad)')
    axs[0, 1].set(xlabel='time', ylabel='Roll Angle (deg)')

    axs[1, 0].plot(t_arr, y[2]*180/np.pi, 'tab:red')
    axs[1, 0].plot(t_arr, roll_rate*180/np.pi, 'tab:orange')
    #axs[1, 0].set_title('Roll Rate (rad/s)')
    axs[1, 0].set(xlabel='time', ylabel='Roll Rate (deg/s)')

    axs[1, 1].plot(t_arr, y[3]*180/np.pi, 'tab:red')
    axs[1, 1].plot(t_arr, yaw_rate*180/np.pi, 'tab:orange')
    #axs[1, 1].set_title('Yaw Rate (rad/s)')
    axs[1, 1].set(xlabel='time', ylabel='Yaw Rate (deg/s)')

    axs[2, 0].plot(t_arr, np.ones(len(t_arr))*aoa0*180/np.pi, 'tab:red')
    axs[2, 0].plot(t_arr, aoa_array*180/np.pi, 'tab:orange')    
    axs[2, 0].set(xlabel='time', ylabel='Angle of Attack (deg)')

    axs[2, 1].plot(t_arr, np.ones(len(t_arr))*V0, 'tab:red')
    axs[2, 1].plot(t_arr, tas_array, 'tab:orange')
    axs[2, 1].set(xlabel='time', ylabel='TAS (m/s)')

    axs[3, 0].plot(t_arr, aileron_input_arr*180/np.pi, 'tab:blue')
    axs[3, 0].set(xlabel='time', ylabel='Aileron Input (deg)')

    axs[3, 1].plot(t_arr, -rudder_input_arr*180/np.pi, 'tab:blue')
    axs[3, 1].set(xlabel='time', ylabel='Rudder Input (deg)')


    plt.show()


    ####Calculate half time, time constant tau and period from simulation############
    #################################################################################
    #################################################################################       

    half_T_non_periodic_0=-0.693/np.real(eigenvalues[0]) 
    tau_non_periodic_0=-1/np.real(eigenvalues[0])

    period=2*np.pi/np.imag(eigenvalues[1]) 
    half_T=-0.693/np.real(eigenvalues[1]) 

    tau_spiral=1/np.real(eigenvalues[3])


    #######Calculate Experimental Time Parameters#######
    ####################################################
    if eigenmotion==1:
        print('Experimental response Half amplitude time is',halfAmpTime(t_arr,roll_rate), 'and experimental response oscillation period is', oscPeriod(t_arr,roll_rate))
        print('Simulated response Half amplitude time is',halfAmpTime(t_arr,y[2]), 'and Simulated response oscillation period is', oscPeriod(t_arr,y[2]))
        print('Dutch roll eigenvalue conjugate pair based on A matrix is', eigenvalues[1], 'and', eigenvalues[2],'.', 'Period based on A matrix is',period, 'and time to half amplitude is', half_T, '.')
    
    elif eigenmotion==2:
        print('Simulation Response Time constant tau is', timeConstant(t_arr,y[1]))
        print('Experimental Response Time constant tau is', timeConstant(t_arr,roll_angle))
        print('Aperiodic roll eigenvalue based on A matrix is', eigenvalues[0],'.', 'and its time constant is',tau_non_periodic_0, '.')

    else:
        print('Simulation Response Time constant tau is', timeConstant(t_arr,y[1]))
        print('Experimental Response Time constant tau is', timeConstant(t_arr,roll_angle))
        print('Spiral eigenvalue based on A matrix is', eigenvalues[3],'.', 'and its time constant is', tau_spiral,'.' )


    return y, P, aileron_input_arr, rudder_input_arr, t_arr, roll_rate, yaw_rate, roll_angle, eigenvalues

####INPUTS####


t_start=48650
t_final=50600
eigenmotion=3
initial_sideslip_guess=0 ##in radians
trim_aileron_in_deg= -0.427  ##in degree
trim_rudder_in_deg= 2.297    ##in degree

y, P, aileron_input_arr, rudder_input_arr, t_arr, roll_rate, yaw_rate, roll_angle, eigenvalues = asymmetric_motion(trim_aileron_in_deg, trim_rudder_in_deg, t_start, t_final, eigenmotion, initial_sideslip_guess,dat)

#NOTES:

#B37 DATA:
##FOR DUTCH ROLL; [46640:46880] and eigenmotion = 1      --> use initial sideslip guess of 0.0 rad, aileron trim of -0.363 deg and 2.18 deg rudder trim.
##FOR APERIODIC ROLL; [45370:45470] and eigenmotion =2   --> use initial sideslip guess of 0.00837758 rad, -0.37 deg aileron trim and 1.6371 deg rudder trim.
##FOR SPIRAL; [48650:49100] AND eigenmotion = 3          --> use initial sideslip guess of 0 rad, aileron trim of -0.427 deg and rudder trim of 2.297 deg

#REFRENCE DATA:
##REF tstart=3890*10 SPIRAL
##REF tstart=3512*10 DUTCH ROLL

#MODEL IMPOREMENT:
#Aircraft period is longer.
#Aircraft more stiff, it has higher inertia.
