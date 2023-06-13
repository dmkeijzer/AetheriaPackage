%https://arxiv.org/pdf/2106.15134.pdf --> states unstable and poles at 0
%https://arxiv.org/ftp/arxiv/papers/1908/1908.07401.pdf
%https://www.ijais.org/research/volume9/number5/moussid-2015-ijais-451411.pdf

g=9.81;
m=2500;

Ixx=600;
Ixz=200;
Iyy=600;
Izz=600;
K_prop=1;
T_prop=0.25;

kp=0.1;
rotation_dir_arr= [-1 1 -1 1 -1 1]; %%One indicates clockwise\
x_cg=5;
x_arr=[2 2 4 4 6 6];
y_arr=[2 2 4 4 6 6];

A=zeros(12);

A(1,4)=1;
A(2,5)=1;
A(3,6)=1;
A(9,12)=1;
A(8,11)=1;
A(7,10)=1;
A(4,8)=g; %%% Signs changed here due to 
A(5,7)=-g; %%% Bank angle is defined opposite to actual aircraft

J_matrix=[Ixx 0 -Ixz; 0 Iyy 0; -Ixz 0 Izz];

B=zeros([12 4]);

B(6,1)=1/m;

B(10:end, 2:end)=inv(J_matrix);

C=zeros([18 12]);

C(1:12,1:12)=eye(12); 

% u_vec_to_thrust converts the [T,L,M,N] vector to [F1,....,F6]

prop_thrust_to_uvec=[1 1 1 1 1 1;x_cg*ones([1 6])-x_arr ;y_arr;kp*rotation_dir_arr];

D=zeros([18 4]);
D(13:end,1:end)=pinv(prop_thrust_to_uvec);

sys=ss(A,B,C,D);

s=tf([1 0],1);
low_pass_filter=K_prop/(T_prop*s+1); 

quad_dynamics=tf(sys);  %%%% just to store
%%% size(ss(quad_dynamics).A)=26 by 26. Why is this?

sys=sys*low_pass_filter;

%%size(ss(sys).A) Why is this a 16 by 16 matrix?
sys_tf=tf(sys);  
sys_filtered= sys_tf; %%%just to store , no floating point error here


%%%%%%%%%%Input output relations with non-zero transfer functions
%U1=T --> z and z' states                          (1 PID controller)
%U2=L=roll --> y,y' and phi, phi'  and psi, psi'    (3 PID controllers)
%U3=M=pitch --> x, x' and theta, theta'           (2 PID controllers)
%U4=N=yaw --> y,y' and phi, phi' and psi, psi'    (3 PID controllers)
%Due to non-zero Ixz, we see roll and yaw coupling.

 %%%In this step, stupid gains are found leading to weird zeros
 %%%This was mainly due to I gain, which was removed making the controller
 %%%PD only. This is due to the order of the transfer function being 2: no
 %%%steady state error so no I controller needed. Without D controller,
 %%%there are eigenvalues that cannot be made negative or 0. 
gain1=pidtune(sys_tf(3,1), 'PD');
sys_tf(1:end,1)=sys_tf(1:end,1)*gain1;

gain2=pidtune(sys_tf(7,2), 'PD');
sys_tf(1:end,2)=sys_tf(1:end,2)*gain2;

gain3=pidtune(sys_tf(8,3), 'PD');
sys_tf(1:end,3)=sys_tf(1:end,3)*gain3;

gain4=pidtune(sys_tf(9,4), 'PD');
sys_tf(1:end,4)=sys_tf(1:end,4)*gain4;

%size(ss(sys_tf).A) this yields a 44 by 44 matrix. Why?
%sys_k=sys_tf*[gain1;gain2;gain3;gain4];

sys_feedback=zeros([4 18]);
sys_feedback(1,3)=1;
sys_feedback(2,7)=1;
sys_feedback(3,8)=1;
sys_feedback(4,9)=1;

sys_cl=feedback(sys_tf,sys_feedback); 
%%%For some reason this, does not work:
%%sys_tf(3,1)/(1+sys_tf(3,1)) != sys_cl(3,1)
%%Why?


%%%%Note: there is still some relationships with negative gain margins due
%%%%to lack of outer loop

%%%%%%%%%%%%%%%%%%%%%%%HENCE PROCEED TO CONSTRUCT 2nd LOOP%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%This loop will feedback x, and y in body reference frame and the new
%%%inputs are height, x,y and yaw angle


gain5=pidtune(sys_cl(2,2), 'PD');
sys_cl(1:end,2)=sys_cl(1:end,2)*gain5;

gain6=pidtune(sys_cl(1,3), 'PD');
sys_cl(1:end,3)=sys_cl(1:end,3)*gain6;

feedback_xy_loop=zeros([4 18]);
feedback_xy_loop(3,1)=1;
feedback_xy_loop(2,2)=1;

sys_cl2=feedback(sys_cl,feedback_xy_loop);


t=linspace(0,10,100);
[y,t] = step(sys_cl2(1:end,4),t);
plot(t,y(1:end,3,1))  %%%%plot values of 3rd output to 1st input

%%%%WHY DO IT GET NON-CAUSAL ERROR WHEN I TRY TO SIMULATE THE WHOLE
%%%%RESPONSE, but maybe I should just simulate them one at a time?
%%%Maybe giving a step input the the whole system is the problem
%[y,t] = step(sys_cl2,t);
%plot(t,y(1:end,3,1))  %%%%plot values of 3rd output to 1st input

%%%%Also, when I give the command 'unique(round(eig(sys_cl2),2))', why do I
%%%%get so many eigenvalues?

%%%%%%%%%%%%%%%%%%%%%%%HENCE PROCEED TO CONSTRUCT 3rd LOOP%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%This loop will convert the x and y given in external reference frame to 
%%%the one given in body frame via means of yaw angle feedback.
%%%Now, the inputs are height, x_external frame, y_external frame, and yaw
%%%angle.




