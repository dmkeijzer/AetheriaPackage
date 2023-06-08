import control as cl
import numpy as np
import matplotlib.pyplot as plt
from Cit_par23 import Parameters
from Import import load_ac_data, load_csvs
#from transfer import truncateData
from transfer import halfAmpTime, oscPeriod, timeConstant 



def initialise_param(P,data,tstart,tend):

#ref data: short period: tstart=3421, tend=3431, phugoid: 3247 to 3397
    #t=np.arange(tstart/10, tend/10, 0.1)
    t = np.ravel(dat['T'])
    
    t = t[tstart:tend]
    # time_offset = t[0]
    # for i in range(len(t)):
    #     t[i] -= time_offset
    V=np.ravel(data['True Airspeed'][tstart:tend])/1.9438
    alpha=np.ravel(data['Angle of attack'][tstart:tend])*np.pi/180
    pitch=np.ravel(data['Pitch Angle'][tstart:tend])*np.pi/180
    pitch_rate=np.ravel(data['Body Pitch Rate'][tstart:tend])*np.pi/180
    altitude=np.ravel(data['Baro Corrected Altitude #1'][tstart:tend])
    #print("Pmuc", P.muc, len(P.muc))
    P2=Parameters()
    P2.calculated_parameters(dat,min=tstart,max=tend,run_calc=False)
    P.V0=P2.V0[0]
    P.muc=P2.muc[0]
    P.CX0=P2.CX0[0]
    P.CZ0=P2.CZ0[0]
    
    print("V0", P.V0)
    print("P.CZ0:", P.CZ0)
    print("P.CX0: ", P.CX0)
    print("Cma: ", P.Cma)
    print("Cmde: ", P.Cmde)
    return P, V, alpha, pitch, pitch_rate,t, altitude


# =============================================================================
# P.V0=P.inp.P1.airspeed
# P.CZ0=P.inp.P1.CZ0
# P.CX0=P.inp.P1.CX0
# P.Cma=P.inp.P1.Cma
# P.Cmde=P.inp.P1.Cmde
# =============================================================================

##print("V0", P.V0)
##print("muc", P.muc)
##print([[P.CXu/P.V0, P.CXa, P.CZ0, P.CXq*P.c/P.V0],
##                [P.CZu/P.V0, P.CZa, -P.CX0, (P.CZq+2*P.muc)*P.c/P.V0],
##                [0, 0, 0, P.c/P.V0],
##                [P.Cmu/P.V0, P.Cma, 0, P.Cmq*P.c/P.V0]])
def symmetric_motion(t,P, V, alpha, pitch,pitch_rate,tstart,tend):
    C1=np.array([[-2 * P.muc * P.c/P.V0**2, 0, 0, 0],
                [0, (P.CZadot-2*P.muc)*P.c/P.V0, 0, 0],
                [0, 0, -P.c/P.V0, 0],
                [0,P.Cmadot*P.c/P.V0, 0, -2* P.muc * (P.KY2)*(P.c/P.V0)**2]])
    C2=np.array([[P.CXu/P.V0, P.CXa, P.CZ0, P.CXq*P.c/P.V0],
                [-0.28209/P.V0, P.CZa, -P.CX0, (P.CZq+2*P.muc)*P.c/P.V0],
                [0, 0, 0, P.c/P.V0],
                [P.Cmu/P.V0, P.Cma, 0, P.Cmq*P.c/P.V0]])
    C3=np.array([[P.CXde],
                [P.CZde],
                [0],
                [P.Cmde]])
    As=-np.matmul(np.linalg.inv(C1), C2)
    Bs=-np.matmul(np.linalg.inv(C1), C3)
    Cs=np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
    Ds=np.array([[0], [0], [0], [0]])
    sys=cl.ss(As, Bs, Cs, Ds)

   #response input attempts - maybe doesnt make sense at all
    #u = np.zeros(len(t))
    u =np.ravel(dat['Deflection of elevator'][tstart:tend])*np.pi/180
    for i in range(0, len(u)):
       u[i]-=u[0]*np.pi/180
        
    #u[int(4200.9*10-tstart):int(4373*10-tstart)]=-0.00298
    
    #x0=[0, alpha[0], pitch[0], pitch_rate[0]]
    x0=[0, 0, 0, 0]
##    u1 = np.zeros(int(final_t*1/t_step))
##    u1[0:int(1/t_step)]=5
    #responses
    t, y = cl.forced_response(sys, t, u, x0)



    ##    t, y1 = cl.step_response(sys, t)# [[100], [0], [0], [0]], [[V], [alpha], [pitch], [0]])
##    t, y2 = cl.impulse_response(sys, t)
##    t, y3 = cl.initial_response(sys, t, [[P.V0], [P.alpha0], [0.05], [0]])
##    t, y4 = cl.input_output_response(sys, t, u1)
    eigen = np.linalg.eigvals(As)#*P.c/P.V0
    
    return y, eigen, As, u#, y1, y2, y3, y4


def symmetric_eigvals():
    C1=np.array([[-2 * muc * c/V0**2, 0, 0, 0],
                [0, (CZadot-2*muc)*c/V0, 0, 0],
                [0, 0, -c/V0, 0],
                [0,Cmadot*c/V0, 0, -2* muc * (KY2)*(c/V0)**2]])
    C2=np.array([[CXu/V0, CXa, CZ0, CXq*c/V0],
                [-0.28209/V0, CZa, -CX0, (CZq+2*muc)*c/V0],
                [0, 0, 0, c/V0],
                [Cmu/V0, Cma, 0, Cmq*c/V0]])

    As=-np.matmul(np.linalg.inv(C1), C2)

    eigen = np.linalg.eigvals(As)#*P.c/P.V0

#t = truncateData(dat, tstart, tend)[4]
# =============================================================================
# 
# t_manoeuver = t[0]
# P.V0=P.V0[int(t_manoeuver)]
# P.muc = P.muc[int(t_manoeuver)]
# P.CZ0 = P.CZ0[int(t_manoeuver)]
# P.CX0=P.CX0[int(t_manoeuver)]
# =============================================================================

#Forced response graphs
def plot_result_symmetric(P,data,tstart,tend,plot=True):
    #Inistialising parameters for plots
    P, V, alpha, pitch,pitch_rate,t, altitude=initialise_param(P,data,tstart,tend)
    #simulating
    y,eigen,As, u = symmetric_motion(t,P, V, alpha, pitch,pitch_rate,tstart,tend)

    #plotting
    if plot:
        figure1, axis1 = plt.subplots(2, 2, figsize=(22, 12))
        axis1[0, 0].plot(t, altitude)
        axis1[0, 0].set_title("Altitude")
        
        axis1[0, 1].plot(t, V)
        axis1[0, 1].set_title("Velocity")
        axis1[1, 0].plot(t, alpha)
        axis1[1, 0].set_title("Angle of attack")
        axis1[1, 1].plot(t, u)
        axis1[1, 1].set(xlabel='Time (s)', ylabel='Elevator deflection (rad)')
        #axis1[1, 1].legend()
        #axis1[1, 1].set_title("Elevator input")
        
        figure, axis = plt.subplots(2, 2)
        axis[0, 0].plot(t, y[0]+V[0], label='Simulation')
        #axis[0, 0].plot (t , truncateData(dat, tstart, tend)[0])
        axis[0, 0].plot(t, V, label='Flight data')
        axis[0, 0].set(xlabel='Time (s)', ylabel='Velocity (m/s)')
        axis[0, 0].legend()
        #axis[0, 0].set_title("Velocity")
        axis[0, 1].plot(t, y[1]+alpha[0], label='Simulation')
        #axis[0, 1].plot (t, truncateData(dat, tstart, tend)[3])
        axis[0, 1].plot(t, alpha, label='Flight data')
        axis[0, 1].set(xlabel='Time (s)', ylabel='Angle of attack (rad)')
        axis[0, 1].legend()
        #axis[0, 1].set_title("Angle of attack")
        axis[1, 0].plot(t, y[2]+pitch[0], label='Simulation')
        #axis[1 ,0].plot (t, truncateData(dat, tstart, tend)[2])
        axis[1, 0].plot(t, pitch, label='Flight data')
        axis[1, 0].set(xlabel='Time (s)', ylabel='Pitch (rad)')
        axis[1, 0].legend()
        #axis[1, 0].set_title("Pitch")
        axis[1, 1].plot(t, y[3]+pitch_rate[0], label='Simulation')
        #axis[1, 1].plot (t, truncateData(dat, tstart, tend)[1])
        axis[1, 1].plot(t, pitch_rate, label='Flight data')
        axis[1, 1].set(xlabel='Time (s)', ylabel='Pitch rate (rad/s)')
        axis[1, 1].legend()
        #axis[1, 1].set_title("Pitch Rate")
        #figure.suptitle("Forced response")
        plt.savefig("goodstuff.pdf")
        plt.show()
        
    return y,eigen,As, u

phugoid=True
group='B37'
csv_dataset=np.array([load_csvs(group,0),load_csvs(group,1),load_csvs(group,2),load_csvs(group,3),])
P=Parameters()
#P.calculated_parameters(csv_dataset, Measurement=True, Wf=True)
P.calculated_parameters(csv_dataset, Measurement=True, Wf=True)

#time stamps for phugoid: tstart=4160*10 and tend=4450*10
if phugoid:
    if group=='B37':
        tstart = 4160*10
        tend = 4450*10
    elif group=='ref':
        tstart = 3174*10
        tend = 3350*10
#time stamps for short period: tstart=4401*10 and tend=4404*10
else:
    if group=='B37':
        tstart=  4472*10#4450*10#4401*10
        tend=  4487*10
    elif group=='ref':
        tstart=3406*10
        tend=3481*10

dat=load_ac_data(group)
#plotting results
plot_result_symmetric(P,dat,tstart,tend) 

#u = plot_result_symmetric(P, dat, tstart, tend)[3]
#print(u[int(4200.9*10-tstart):int(4373*10-tstart)])

#print eigenvalues
# P, V, alpha, pitch,pitch_rate,t, altitude=initialise_param(P,dat,tstart,tend)
# print("the eigenvalues are:", symmetric_motion(t,P, V, alpha, pitch,pitch_rate,tstart,tend)[1])

###Step response graphs
##y1 = symmetric_motion(t)[3]
##figure, axis = plt.subplots(2, 2)
##axis[0, 0].plot(t, y1[0][0])
##axis[0, 0].set_title("Velocity")
##axis[0, 1].plot(t, y1[1][0])
##axis[0, 1].set_title("Angle of attack")
##axis[1, 0].plot(t, y1[2][0])
##axis[1, 0].set_title("Pitch")
##axis[1, 1].plot(t, y1[3][0])
##axis[1, 1].set_title("q")
##figure.suptitle("Step response")
##plt.show()


###Impulse response graphs
##y2 = symmetric_motion(t)[4]
##figure, axis = plt.subplots(2, 2)
##axis[0, 0].plot(t, y2[0][0])
##axis[0, 0].set_title("Velocity")
##axis[0, 1].plot(t, y2[1][0])
##axis[0, 1].set_title("Angle of attack")
##axis[1, 0].plot(t, y2[2][0])
##axis[1, 0].set_title("Pitch")
##axis[1, 1].plot(t, y2[3][0])
##axis[1, 1].set_title("q")
##figure.suptitle("Impulse response")
##plt.show()

###Inital response graphs
##y3 = symmetric_motion(t)[5]
##figure, axis = plt.subplots(2, 2)
##axis[0, 0].plot(t, y3[0])
##axis[0, 0].set_title("Velocity")
##axis[0, 1].plot(t, y3[1])
##axis[0, 1].set_title("Angle of attack")
##axis[1, 0].plot(t, y3[2])
##axis[1, 0].set_title("Pitch")
##axis[1, 1].plot(t, y3[3])
##axis[1, 1].set_title("q")
##figure.suptitle("Initial response")
##plt.show()

##
###Input output response graphs
##y4 = symmetric_motion(t)[6]
##figure, axis = plt.subplots(2, 2)
##axis[0, 0].plot(t, y4[0])
##axis[0, 0].set_title("Velocity")
##axis[0, 1].plot(t, y4[1])
##axis[0, 1].set_title("Angle of attack")
##axis[1, 0].plot(t, y4[2])
##axis[1, 0].set_title("Pitch")
##axis[1, 1].plot(t, y4[3])
##axis[1, 1].set_title("q")
##figure.suptitle("Input output response")
##plt.show()

#getting half_T and p from simulation data - i=0 for short period, i=2 for phugoid
def eigenmotion(phugoid, P,data,tstart,tend):
    #tau=0
    p=0
# =============================================================================
#     #Inistialising parameters for plots
#     P, V, alpha, pitch,pitch_rate,t, altitude=initialise_param(P,data,tstart,tend)
#     #simulating
#     y,eigen, As, u = symmetric_motion(t, P, V, alpha, pitch,pitch_rate,tstart,tend)
#     
# =============================================================================
    if phugoid:
        tstart = 4160*10
        tend = 4450*10
        P, V, alpha, pitch,pitch_rate,t, altitude=initialise_param(P,data,tstart,tend)
        y,eigen, As, u = symmetric_motion(t, P, V, alpha, pitch,pitch_rate,tstart,tend)
        p=2*np.pi/np.imag(eigen[2])
        half_T=-0.693/np.real(eigen[2])#*P.c/P.V0
    else:
        tstart=4440*10#4450*10#4401*10
        tend=4490*10
        P, V, alpha, pitch,pitch_rate,t, altitude=initialise_param(P,data,tstart,tend)
        y,eigen, As, u = symmetric_motion(t, P, V, alpha, pitch,pitch_rate,tstart,tend)
        p=2*np.pi/np.imag(eigen[0])
        half_T=-0.693/np.real(eigen[0])
    
    for j in range(0, len(eigen)):
        eigen[j]=eigen[j]*P.c/P.V0
        
    
# =============================================================================
#     if np.imag(eigen[i]) == 0:
#         half_T=-0.693/np.real(eigen[i])#*P.c/P.V0
#         #tau=-1/np.real(eigen[i])
#         
#     else:
#         p=2*np.pi/np.imag(eigen[i])
#         half_T=-0.693/np.real(eigen[i])#*P.c/P.V0
#         #delta=np.real(eigen[i])*p
# =============================================================================
    
    
        
        
    if phugoid:
        # tstart = 4205*10
        # tend = 4350*10
        tstart = 4160 * 10
        tend = 4450 * 10
        tim2 = np.linspace(0,145,1450)
        P, V, alpha, pitch,pitch_rate,t, altitude=initialise_param(P,data,tstart,tend)
        y,eigen, As, u = symmetric_motion(t, P, V, alpha, pitch,pitch_rate,tstart,tend)
        print('Experimental response Half amplitude time is',halfAmpTime(tim2,pitch_rate[450:-1000]), 'and experimental response oscillation period is', oscPeriod(tim2,pitch_rate[450:-1000]))
        print('Simulated response Half amplitude time is',halfAmpTime(tim2,y[3][450:-1000]), 'and Simulated response oscillation period is', oscPeriod(tim2,y[3][450:-1000]))
        #print('Phugoid eigenvalue conjugate pair based on A matrix is', eigen[2], 'and', eigen[3],'.', 'Period based on A matrix is',p, 'and time to half amplitude is', half_T, '.')
    
    else:
        tstart=4472*10#4450*10#4401*10
        tend=4487*10
        tim2 = np.linspace(0, 14, 140)
        P, V, alpha, pitch,pitch_rate,t, altitude=initialise_param(P,data,tstart,tend)
        y,eigen, As, u = symmetric_motion(t, P, V, alpha, pitch,pitch_rate,tstart,tend)
        print(y[3])
        # print('Experimental response Half amplitude time is',halfAmpTime(tim2,pitch_rate), 'and experimental response oscillation period is', oscPeriod(tim2,pitch_rate))
        print('Simulated response Half amplitude time is',halfAmpTime(tim2,y[3][9:-1]), 'and Simulated response oscillation period is', oscPeriod(tim2,y[3][9:-1]))
        print('Short period eigenvalue conjugate pair based on A matrix is', eigen[0], 'and', eigen[1],'.', 'Period based on A matrix is',p, 'and time to half amplitude is', half_T, '.')
    #getting half_t and p from experimental data
    exp_half_T_short =  halfAmpTime(t,pitch_rate)
    exp_p_short = oscPeriod(t,pitch_rate)
    
    exp_half_T_phu =  halfAmpTime(t,V)
    exp_p_phu = oscPeriod(t,V)
        
    return half_T, p, eigen, exp_half_T_short, exp_p_short, exp_half_T_phu, exp_p_phu
        

eigenmotion(True, P, dat, 1, 1)
                
