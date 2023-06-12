%https://arxiv.org/pdf/2106.15134.pdf --> states unstable and poles at 0
%https://arxiv.org/ftp/arxiv/papers/1908/1908.07401.pdf
%https://www.ijais.org/research/volume9/number5/moussid-2015-ijais-451411.pdf

g=9.81;
m=2500;

Ixx=1;
Ixz=2;
Iyy=4;
Izz=1;
K_prop=1;
T_prop=20;

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
A(4,8)=-g;
A(5,7)=g;

J_matrix=[Ixx 0 Ixz; 0 Iyy 0; Ixz 0 Izz];

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

% THIS WAS CAUSING ERROR SO YOU CAN EDIT THIS
% for i = 1:4 
%     for j= 1:12 
%     sys(j,i)=sys(j,i)*low_pass_filter;
%     end
% end
%sys=tf(sys);  --> This line was causing error.

%%%%%%%%%%Input output relations with non-zero transfer functions
%U1=T --> z and z' states                          (1 PID controller)
%U2=L=roll --> y,y' and phi, phi'  and psi, psi'    (3 PID controllers)
%U3=M=pitch --> x, x' and theta, theta'           (2 PID controllers)
%U4=N=yaw --> y,y' and phi, phi' and psi, psi'    (3 PID controllers)
%Due to non-zero Ixz, we see roll and yaw coupling.


sys_feedback=zeros([4 18]);
for i = 1:4
   for j= 1:12
    gain=pidtune(sys(j,i), 'P');
    Kp = gain.Kp;
    disp(Kp)
    sys_feedback(i,j)=Kp;
    end
end
sys_cl=feedback(sys,sys_feedback); 

%%Is it not possible to tune each transfer function separately?
%%What does pid autotune even do?

%%%When I typed [~, gainMargin] = margin(sys(3,1)), I got an error saying
%%%unstable system. This means that the input thrust to output z is still
%%%unstable even after autotuning. Why is this? Isn't there a gain that can
%%%stabilize this relationship or did autotune function fail?

%%The result of the autotuning is that you get 18 non-one gains. which
%%makes sense physically as there are 18 physical relationships between
%%T,L,M, and N and the state variables as explained in above comment.

 t=linspace(0,2000,1000000);
 [y,t] = step(sys,t);
 plot(t,y(1:end,3,1))  %%%%plot values of 3rd output to 1st input