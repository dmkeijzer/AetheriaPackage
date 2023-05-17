
Cma1=;
Cma2=;
Cm1=;
Cm2=;
CL1=;;
CL2=;
CLa1=;
CLa2=;
V2_V1_ratio=1;
downwash=0;
x2=;
x1=;
c1=;
c2=;
S1=;
S2=;

file_path = 'path_to_your_csv_file.csv';
table_data = readtable(file_path);


x_np_nom=-Cma1+(CLa1*x1)/(c1)-Cma2*(1-downwash)*(S2/S1)*(c2/c1)*(V2_V1_ratio)^2+CLa2*(1-downwash)*(x2/c1)*(S2/S1)*(V2_V1_ratio)^2;
x_np_den=CLa1/c1+CLa2/c1*(1-downwash)*(S2/S1)*(V2_V1_ratio)^2*(c2/c1);
x_np=x_np_nom/x_np_den;

x_cg = (-Cm1-Cm2*(c2/c1)*(S2/S1)*V2_V1_ratio^2+CL1*x1/c1+CL2*(x2/c1)*(S2/S1)*V2_V1_ratio^2)/(CL1/c1 + (CL2/c1)*(S2/S1)*V2_V1_ratio^2);

Tfac = table_data(end,:);
xcgmax = table_data(2,:);
xcgmin = table_data(1,:);

Tfac_array_length = size(Tfac, 2);

plot(ones(1, array_size)*x_np,Tfac);
plot(ones(1, array_size)*x_cg,Tfac);
plot(xcgmax,Tfac);
plot(xcgmin,Tfac);



