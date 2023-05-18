
ma=4021;
Sproj=16.4;
n_engines=32;
ROC=2;
rho=1.225;
A_disk=0.2;

Cma1=1;
Cma2=0.4;
Cm1=-0.4;
Cm2=1;
CL1=1;
CL2=3;
CLa1=1;
CLa2=0.11;
V2_V1_ratio=1;
downwash=0;
x2=1;
x1=1;
c1=1;
c2=1;
S1=1;
S2=1;

file_path = 'L1 Tfac vs cg range_m=4021.csv';
table_data = readtable(file_path);


x_np_nom=-Cma1+(CLa1*x1)/(c1)-Cma2*(1-downwash)*(S2/S1)*(c2/c1)*(V2_V1_ratio)^2+CLa2*(1-downwash)*(x2/c1)*(S2/S1)*(V2_V1_ratio)^2;
x_np_den=CLa1/c1+CLa2/c1*(1-downwash)*(S2/S1)*(V2_V1_ratio)^2*(c2/c1);
x_np=x_np_nom/x_np_den;

x_cg = (-Cm1-Cm2*(c2/c1)*(S2/S1)*V2_V1_ratio^2+CL1*x1/c1+CL2*(x2/c1)*(S2/S1)*V2_V1_ratio^2)/(CL1/c1 + (CL2/c1)*(S2/S1)*V2_V1_ratio^2);

Tfac = table2array(table_data(end,:));
Thrust = 9.80665*ma*Tfac*(1+1/(n_engines-1))*(1+1.225*ROC^2*Sproj/(ma*9.80665))/(n_engines);

Power_per_engine = Thrust*ROC+1.2*Thrust.*(-ROC/2+sqrt(ROC^2/4+Thrust/(2*rho*A_disk)));

% above formula is from: https://arc.aiaa.org/doi/pdf/10.2514/6.2022-1030

xcgmax = table2array(table_data(2,:));
xcgmin = table2array(table_data(1,:));

Tfac_array_length = size(Tfac, 2);

figure;
hold on
plot(ones(1, Tfac_array_length) * x_np, Power_per_engine)
plot(ones(1, Tfac_array_length) * x_cg, Power_per_engine)
plot(xcgmax, Power_per_engine)
plot(xcgmin, Power_per_engine)
ylabel("Engine power requirement", "FontSize",18)
xlabel("x_{cg} locations", "FontSize",18)
title("CG range envelope for longitudinal and vertical criteria","FontSize",18)
legend("x_{cg}_{aft} longitudinal constraint", "x_{cg}_{fw} longitudinal constraint", "x_{cg}_{aft} vertical constraint", "x_{cg}_{fw} vertical constraint", "FontSize",12)
hold off; % Release the hold on the plot



