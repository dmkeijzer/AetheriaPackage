
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns

beta = np.radians(20) # The angle at which the engines are tilted
b = 14
lf = 10
mtow = 3000*9.80665

for i in range(2):
    # Example 3D vectors
    if i == 0:
        vectors = np.array([[0, np.sin(beta), np.cos(beta)],    # F1
                            [0, -np.sin(beta), np.cos(beta)],    #F2
                            [0, np.sin(beta), np.cos(beta)],     #F3
                            [0, -np.sin(beta), np.cos(beta)],     #F4
                            [0, np.sin(beta), np.cos(beta)],     #F5
                            [0, -np.sin(beta), np.cos(beta)]])    #F6
        vector_lengths = np.ones(6)  # Lengths of the vectors

        vector_bases = np.array([[0.4*lf, b/4, 0], 
                                [0.4*lf, -b/4, 0],
                                [0.1*lf, b/2, 0],
                                [0.1*lf, -b/2, 0],
                                [-0.6*lf, b/4, 0],
                                [-0.6*lf, -b/4, 0]])  # Base positions of the vectors

        vector_labels = ['F1', 'F2', 'F3', "F4", "F5", "F6"]  # Labels for each vector
        # Create a figure and 3D axis
        palette = 'Set2'  # Choose any Seaborn color palette of your choice
        colors = sns.color_palette(palette, 6)
    if i == 1:
        vectors = np.array([[0, np.sin(beta), np.cos(beta)],    # F1
                            [0, -np.sin(beta), np.cos(beta)],    #F2
                            [0, np.sin(beta), np.cos(beta)],     #F3
                            [0, -np.sin(beta), np.cos(beta)],     #F4
                            [0, np.sin(beta), np.cos(beta)]])     #F5
        vector_lengths = np.ones(6)*30  # Lengths of the vectors
        vector_bases = np.array([[0.4*lf, b/4, 0], 
                                [0.4*lf, -b/4, 0],
                                [0.1*lf, b/2, 0],
                                [0.1*lf, -b/2, 0],
                                [-0.6*lf, b/4, 0]])

        vector_labels = ['F1', 'F2', 'F3', "F4", "F5"]  # Labels for each vector
        # Create a figure and 3D axis
        palette = 'Set2'  # Choose any Seaborn color palette of your choice
        colors = sns.color_palette(palette, 6)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if i == 0:
        ax.set_title("all answers egines operation")
    else: 
        ax.set_title("Engine 6 failed")

    # Plotting the vectors

    for vector, length, base, col, label in zip(vectors, vector_lengths, vector_bases, colors, vector_labels):
        ax.quiver(*base, *vector, length=length, color= col)
        ax.text(base[0] + vector[0], base[1] + vector[1], base[2] + vector[2], label)

    # Set plot limits
    max_range = 1.4*lf
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0,5])

    # Set labels for the axes
    ax.set_xlabel('X + towards nose')
    ax.set_ylabel('Y + towards left wing')
    ax.set_zlabel('Z + upwards')

    # Show the plot
    plt.show()

# Solving for F6 falling out
#------------------------------------------------------
# F1 F2 F3 F4 F5 F6
# [[Sum in forces in x]
# [Sum in forces in y]
# [Sum in forces in z]
# [Sum in moment about x ]
# [Sum in moment about y]
# [Sum in moment about z] ]
#------------------------------------------------------


        #           F1  F2 F3 F4 F5 F6
eom_mat = np.array([[0, 0, 0, 0, 0, 0],                # Fx
           [np.sin(beta), -np.sin(beta), np.sin(5), -np.sin(5), np.sin(5), -np.sin(5)], #Fy equation
           [np.cos(beta),np.cos(beta),np.cos(beta),np.cos(beta),np.cos(beta),np.cos(beta)], # Fz equation
           [b*np.cos(beta),-b*np.cos(beta),b*np.cos(beta),-b*np.cos(beta),b*np.cos(beta),-b*np.cos(beta)], # Mx equation
           [-0.4*lf*np.cos(5),-0.4*lf*np.cos(5),-0.1*lf*np.cos(5),-0.1*lf*np.cos(5),0.6*lf*np.cos(5),0.6*lf*np.cos(5)], # My equation
           [-0.1, 0.1, -0.1, 0.1, -0.1, 0.1]])              # Mz equation 10% of the forces was taken

res_vec = np.array([0,0, mtow, 0, 0, 0])

eom_reformed_mat =  np.matrix(eom_mat[1:,:-1]) # Picking out the correct rows and columns
res_vec_reformed = res_vec[1:].reshape(5,1) # Remove F6

print(f"{eom_reformed_mat} * F vector \n \n = {res_vec_reformed} \n")

# print(eom_reformed_mat)
# print(res_vec_reformed)

forces = eom_reformed_mat**-1@res_vec_reformed

print(f"Resultant forces  \n ______________________ \n  {forces}")

#---------------------------------------Plotting results --------------------------------------


forces = np.array(forces)
normalized_forces = (forces- np.min(forces)) / (np.max(forces) - np.min(forces))


vectors = np.array([[0, np.sin(beta), np.cos(beta)],    # F1
                    [0, -np.sin(beta), np.cos(beta)],    #F2
                    [0, np.sin(beta), np.cos(beta)],     #F3
                    [0, -np.sin(beta), np.cos(beta)],     #F4
                    [0, np.sin(beta), np.cos(beta)]])     #F5
# vector_lengths = [0, 0, 0.707, 1, 0.285]  # Lengths of the vectors
vector_lengths = normalized_forces  # Lengths of the vectors
vector_bases = np.array([[0.4*lf, b/4, 0], 
                        [0.4*lf, -b/4, 0],
                        [0.1*lf, b/2, 0],
                        [0.1*lf, -b/2, 0],
                        [-0.6*lf, b/4, 0]])

vector_labels = ['F1', 'F2', 'F3', "F4", "F5"]  # Labels for each vector

# Create a figure and 3D axis
palette = 'Set2'  # Choose any Seaborn color palette of your choice
colors = sns.color_palette(palette, 6)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("The solution")

# Plotting the vectors

for vector, length, base, col, label in zip(vectors, vector_lengths, vector_bases, colors, vector_labels):
    ax.quiver(*base, *vector, length=length, color= col, normalize= True)
    ax.text(base[0] + vector[0], base[1] + vector[1], base[2] + vector[2], label)

# Set plot limits
max_range = 1.4*lf
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([0,np.max(np.abs(vector_lengths))])

# Set labels for the axes
ax.set_xlabel('X + towards nose')
ax.set_ylabel('Y + towards left wing')
ax.set_zlabel('Z + upwards')

plt.show()
