#Assuming F3=0:

import numpy as np
from sympy import *
init_printing(use_unicode=True)

k_mu=0.1
x1=0.57
x2=3.4
x3=6.8
xcg=3.5
y1=2.3
y2=5.4
y3=2.3
m=2510
g=9.8

A=np.matrix([[k_mu,k_mu,k_mu,-k_mu,-k_mu],
             [1,1,1,1,1],
             [(x1-xcg),(x2-xcg),(x3-xcg),(x1-xcg),(x2-xcg)],
             [y1,y2,y3,-y1,-y2]])

b=np.matrix([[0],
             [m*g],
             [0],
             [0]])


augmented_matrix=np.hstack((A,b))
augmented_matrix=Matrix(augmented_matrix)
augmented_matrix_reduced=augmented_matrix.rref()
###INSTRUCTIONS: Type 'augmented_matrix_reduced' in the shell to see the matrix.

###If you run the code for xcg =4, you see that you have four pivots
###so it initially seems like you can get first row equal to-1243.71910112359
###but for this F1<0 so no control possible at x_cg=4.
###Since first row is limiting, the problem comes from yaw torque-lift coupling.

###If xcg=3.685, then your first row equals to 0 and you can finally get
###a solution where forces are greater than 0.
###xcg=3.685 is also where we see the line become vertical.


###If you actually change the convergence direction to -1, and run the ACAI file,
###you get 0.57m. Below this is uncontrollable. This is when the RHS of the
###third column goes from positive to negative.

###NOTE: YOU can just ignore column 5 as it is a linear
###combination of the first five rows.



###To prove the maximum thrust part you need to show that no solution exists unless the Tmax is equal to a certain value.
###Running the python program at x=3.5;
###The limiter of maximum thrust is equation 4.
###In this equation, if all rotors are loaded to the limit of Tmax= 12299.0/(1+1)=6149.5
###Indeed, when we plug this Tmax into the program instead of 1000000000000, we get ACAI=0. Hence, we get negative ACAI when we plug a lower no
###And we get a positive ACAI when we plug in a higher number showing that indeed, the ACAI is correct. 


