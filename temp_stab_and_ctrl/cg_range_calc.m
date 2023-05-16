%clear all;clc;close all;



%%%%%%%%%%%%%%%%%%%%%INPUTS%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ret] = cg_range_calc(convergence_direction,rotor_dir,rotor_ku,x_cg,x_rotor_locations,y_rotor_locations,Rotors,rotor_Yita,ma,Sproj,Tfactor)
step_size=0.02;     %program works by approaching the cg 0.1m at a time, this is an absolute value of step size, LEAVE AS IS.  
ACAI=1;    %any positive number works, LEAVE AS IS. 
%convergence_direction=1; %put as +1 to find max cg, put as -1 to find min cg.
s2i = struct('anticlockwise', 1, 'clockwise', -1);
%rotor_dir=[s2i.anticlockwise s2i.anticlockwise s2i.clockwise s2i.clockwise s2i.anticlockwise s2i.clockwise];
%rotor_ku=[0.1 0.1 0.1 0.1 0.1 0.1];
%x_cg=2;   %Make any guess between the two wings. This is just to initialize. Exact value does not matter. 
%x_rotor_locations=[3.5 5 3.5 5 7 7];
%y_rotor_locations=[7 10 -7 -10 5 -5];
%Rotors=[1 2 3 4 5 6];
%rotor_Yita=[1  1   1   1   1   0]; %efficiency parameters of the rotors (set to 0 or 1 depending on failure)
%ma=1.535; 
%Sproj=10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alist = [];
while ACAI>0

    Jx=1;Jy=1;Jz=1; % could be altered but does not affect result
    array_shape=size(rotor_dir);
    array_size=array_shape(2);
    rotor_d=[];
    rotor_angle=[];
    for i = 1:1:array_size
        rotor_d(i)=sqrt(y_rotor_locations(i)^2+(x_cg-x_rotor_locations(i))^2);
        rotor_angle(i)=atan2((y_rotor_locations(i)),(x_rotor_locations(i)-x_cg));
    end
    
    % dot(x)=Ax+B(F-G)
    A=[zeros(4,4) eye(4);zeros(4,8);];
    nA=size(A);
    nA=nA(1);
    % mass of the multirotor helicopter
    % acceleration of gravity
    g0=9.8;%m/s^2
    % moment of inertia
    Jf=diag([-ma Jx Jy Jz]);
    % dot(x)=Ax+B(F-G)
    B=[zeros(4,4);inv(Jf);];
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Obtain Bf and Tg
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % set rotor failures 0: failed; 0.5: 50% effencify 
    % rotor_Yita(1)=1;
    % Rc=nchoosek(Rotors,1);
    % rc=6;
    % rotor_Gama(Rc(rc,:))=0;
    % set the control effectiveness matrix Bf
    sz=size(rotor_angle);
    sz_angle=sz(2);
    sz=size(rotor_dir);
    sz_dir=sz(2);
    sz=size(rotor_ku);
    sz_ku=sz(2);
    sz=size(rotor_d);
    sz_d=sz(2);
    sz=size(rotor_Yita);
    sz_Gama=sz(2);
    sz=sz_angle; 
    if sz_angle==sz_dir
        for i=1:1:sz
            bt(i)=1*rotor_Yita(i);%lift
            bl(i)=-rotor_d(i)*sin(rotor_angle(i))*rotor_Yita(i);% roll torque
            bm(i)=rotor_d(i)*cos(rotor_angle(i))*rotor_Yita(i);% pitch torque
            bn(i)=rotor_dir(i)*rotor_ku(i)*rotor_Yita(i);% yaw torque
        end
    else
        error('please confirm the angle and direction of the rotors');
    end
    % F=Bf*f
    Bf=[bt;bl;bm;bn;];
    % vector of gravity
    Tg=[ma*g0 0 0 0]';
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    give_up_yaw=0;
    % give up yaw??
    if give_up_yaw==1
        A=[zeros(3,3) eye(3);zeros(3,6);];
        Jf_1=diag([-ma Jx Jy]);
        B=[zeros(3,3);inv(Jf_1);];
        Bf=[bt;bl;bm;];
        Tg=[ma*g0 0 0]';
    end
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Controllability Test Procedures
    % Step 1. Check the rank of the controllability matrix C(A,B)
    Cab=[B A*B A^2*B A^3*B A^4*B A^5*B A^6*B A^7*B];
    n=rank(Cab);
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Step 2. Compute the ACAI
    % minimum lift of the rotors
    umin=0;
    % maximum lift of the rotors
    % umax = 6.125;
    umax= 9.80665*ma*Tfactor*(1+1/(array_size-1))*(1+1.225*4*Sproj/(ma*9.80665))/(array_size);
    % control constraint set
    Uset.umin=umin*ones(sz,1);
    Uset.umax=umax*ones(sz,1);
    % threshold value
    delta=1e-10;
    % compute the ACAI
    Tg=[ma*g0 0 0 0]';
    
    
    ACAI=acai(Bf,Uset.umin,Uset.umax,Tg);
    if ACAI<delta && ACAI>-delta
        ACAI=0;
    end

    %disp(x_cg);
    %disp(ACAI);

    %if n<nA | ACAI<=0
        %fprintf('uncontrollable \n \n \n ');
       
    %else
        %fprintf('controllable   \n \n \n  ');
    %end
    x_cg=x_cg+convergence_direction*step_size;

    alist(end+1) = x_cg;
    %disp(alist)

end
%disp(alist)
if size(alist,2) > 1
    ret = alist(end-1);
elseif convergence_direction == 1
    ret = 1e6;
elseif convergence_direction == -1
    ret = -1e6;
else
    fprintf('Error no xcg')
    
end
end