
% %%%%%%W1%%%%%
% 
% Clmax_airfoil=1.67;
% ma=3210;
% Sproj=16.4;
% n_engines=12;
ROC=2;
rho=1.225;
% A_disk=7.84/n_engines;
% Cma1= -0.0285;
% Cma2=  -0.029;
% Cm1= -0.41723 ; %%%%
% Cm2= -0.4198 ; %%%%
% CL1=0.9*Clmax_airfoil  ; %assuming wing is at CLmax during landing
% CL2=0.9*Clmax_airfoil  ; %assuming wing is at CLmax during landing
% CLa1=0.0888;
% CLa2=0.0898;
% V2_V1_ratio=1;
% downwash=0.1;
% x2=9;
% x1=0.5;
% c1=0.91;
% c2=1.89;
% S1=7;
% S2=7;


% %%%%%%L1%%%%%
% 
% Clmax_airfoil=1.67;
% ma=4021;
% Sproj=16.4;
% n_engines=32;%NOT 36?
% ROC=2;
% rho=1.225;
% A_disk=2.089/n_engines;
% Cma1= -0.0245 ;
% Cma2=  -0.0295;
% Cm1= -0.4073 ; %%%%
% Cm2= -0.4247 ; %%%%
% CL1=0.9*Clmax_airfoil  ; %assuming wing is at CLmax during landing
% CL2=0.9*Clmax_airfoil  ; %assuming wing is at CLmax during landing
% CLa1=0.0782;
% CLa2=0.0903;
% V2_V1_ratio=1;
% downwash=0.1;
% x2=8;
% x1=0.5;
% c1=0.8812989698764242;
% c2=1.0456555605008826;
% S1=4;
% S2=10;

% %%%%%%J1andJ3%%%%%
% %Where is J2?
% Clmax_airfoil=1.67;
% ma=2510;
% Sproj=16.4;
% n_engines=8;
% ROC=2;
% rho=1.225;
% A_disk=55.78/n_engines;
% Cma1= -0.0277 ;
% Cma2=  0   ;   %%%%Cma is 0 about the aerodynamic center
% Cm1= -0.4163 ; %%%%
% Cm2=  0  ; %%%% since symmetric airfoil and Cma=0
% CL1= 0.9*Clmax_airfoil  ; %assuming wing is at CLmax during landing
% CL2= 0.0; %
% %Based on formula from ADSEE 3, aircraft control slides, slide 17:
% %CL2_most_negative=-0.35*4^(1/3) but this gives a weird value for xcg_fwd
% %So I set CL2 to 0 because it gives a very low x_cg_fwd. 
% %https://www.fzt.haw-hamburg.de/pers/Scholz/HOOU/AircraftDesign_9_EmpennageGeneralDesign.pdf
% % see table 9.2 for aspect ratio
% CLa1=0.0865;
% CLa2= 2*pi*4/(2+sqrt(4+(4/0.95)^2))*pi/180  ;
% %formula used is based on ADSEE II, Lift & drag estimation slides, slide 8
% %formula used with beta=0, sweep = 0, 
% V2_V1_ratio=1;
% downwash=0.1;
% x2= 8.5;
% x1= 3;  %%% I Played around with this to get the best cg range
% c1= 1.37 ;
% c2= 0.9 ;  %%b= sqrt(A*S)=3.6--> c=S/b=3.25/3.6=0.9
% S1= 14  ;
% S2= 3.25  ; % The horizontal stabiliser should be smaller








load("J1_input.mat")
file_path = 'J3 Tfac vs cg range_m=2510.csv';
table_data = readtable(file_path);


x_np_nom=-Cma1+(CLa1*x1)/(c1)-Cma2*(1-downwash)*(S2/S1)*(c2/c1)*(V2_V1_ratio)^2+CLa2*(1-downwash)*(x2/c1)*(S2/S1)*(V2_V1_ratio)^2;
x_np_den=CLa1/c1+CLa2/c1*(1-downwash)*(S2/S1)*(V2_V1_ratio)^2*(c2/c1);
x_np=x_np_nom/x_np_den;

x_cg = (-Cm1-Cm2*(c2/c1)*(S2/S1)*V2_V1_ratio^2+CL1*x1/c1+CL2*(x2/c1)*(S2/S1)*V2_V1_ratio^2)/(CL1/c1 + (CL2/c1)*(S2/S1)*V2_V1_ratio^2);

Tfac = table2array(table_data(end,:));
Thrust = 9.80665*ma*Tfac*(1+1/(n_engines-1))*(1+1.225*ROC^2*Sproj/(ma*9.80665))/(n_engines);

Power_per_engine = Thrust*ROC+1.2*Thrust.*(-ROC/2+sqrt(ROC^2/4+Thrust/(2*rho*A_disk)));

Thrust_per_engine_per_weight= 9.80665*ma*(1+1/(n_engines-1))*(1+1.225*ROC^2*Sproj/(ma*9.80665))/(n_engines);
Power_corresponding=Thrust_per_engine_per_weight*ROC+1.2*Thrust_per_engine_per_weight*(-ROC/2+sqrt(ROC^2/4+Thrust_per_engine_per_weight/(2*rho*A_disk)));

Pfac=Power_per_engine/Power_corresponding;
% above formula is from: https://arc.aiaa.org/doi/pdf/10.2514/6.2022-1030

xcgmax = table2array(table_data(2,:));
xcgmin = table2array(table_data(1,:));

Tfac_array_length = size(Tfac, 2);

figure;
hold on
plot(ones(1, Tfac_array_length) * x_np, Pfac)
plot(ones(1, Tfac_array_length) * x_cg, Pfac)
plot(xcgmax, Pfac)
plot(xcgmin, Pfac)
ylabel("Power factor", "FontSize",18)
xlabel("x_{cg} locations", "FontSize",18)
title("CG range envelope for longitudinal and vertical criteria","FontSize",18)
legend("x_{cg}_{aft} longitudinal constraint", "x_{cg}_{fw} longitudinal constraint", "x_{cg}_{aft} vertical constraint", "x_{cg}_{fw} vertical constraint", "FontSize",12)
hold off; % Release the hold on the plot



