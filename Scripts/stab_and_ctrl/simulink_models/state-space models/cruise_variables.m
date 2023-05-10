rho = 1.11164;  % at 1000 m altitude
g = 9.80665;

aeroCoeff = struct;
aeroCoeff.Cl0 = 0;
aeroCoeff.Cl_beta = -0.051919818;
aeroCoeff.Cl_da = -0.0981676066;
aeroCoeff.Cl_dr = 0.1746396276;
aeroCoeff.Cl_p = -0.7461562431;
aeroCoeff.Cl_r = 0.1584677463;
aeroCoeff.Cm0 = 0;
aeroCoeff.Cm_alpha = -0.2084539580;
aeroCoeff.Cm_de = -3.8146754263;
aeroCoeff.Cm_q = -22.9669011845;
aeroCoeff.Cn0 = 0;
aeroCoeff.Cn_beta = 0.0540377505;
aeroCoeff.Cn_beta_dot = 0;
aeroCoeff.Cn_da = 0.0101277353;
aeroCoeff.Cn_dr = -0.4423152173;
aeroCoeff.Cn_p = -0.0208253137;
aeroCoeff.Cn_r = -0.0816662365;
aeroCoeff.Cx0 = 0;
aeroCoeff.Cx_alpha = 0.2466037589;
aeroCoeff.Cx_q = 0;
aeroCoeff.Cx_de = 0;
aeroCoeff.Cy0 = 0;
aeroCoeff.Cy_beta = -0.1443361205;
aeroCoeff.Cy_beta_dot = 0;
aeroCoeff.Cy_da = 0;
aeroCoeff.Cy_dr = 0.8693004869;
aeroCoeff.Cy_p = -0.0224336303;
aeroCoeff.Cy_r = 0.1468814604;
aeroCoeff.Cz0 = -0.5167063636;
aeroCoeff.Cz_alpha = -3.6783451837;
aeroCoeff.Cz_de = -0.0000000000;
aeroCoeff.Cz_q = 2.4294005730;
aeroCoeff.Cx_u = -0.1637449639;
aeroCoeff.Cz_u = -0.0248990234;
aeroCoeff.Cm_u = 0.0061871085;
aeroCoeff.Cx_alpha_dot = 0.0000000000;
aeroCoeff.Cz_alpha_dot = -3.5068180832;
aeroCoeff.Cm_alpha_dot =-9.6904377913;

aeroCoeff.CL = 0.5158389632834982;

initCond = struct;
initCond.x0 = [0 0 -1000];
initCond.v0 = 72.1867618534;
initCond.euler0 = [0 0 0];
initCond.w0 = [0 0 0];

vehicle = struct;
vehicle.Sref = 19.8213410712;
vehicle.cbar = 1.2651477965;
vehicle.bref = 8.2092971467;
vehicle.Ixx =8223.4152729346;
vehicle.Iyy =17849.3539449603;
vehicle.Izz =32689.1284825992;
vehicle.Ixz =1155.5542328861;
vehicle.m = 2722.3210820672;
%vehicle.Kxx = vehicle.Ixx / (vehicle.m * vehicle.bref^2);
%vehicle.Kyy = vehicle.Iyy / (vehicle.m * vehicle.cbar^2);
%vehicle.Kzz = vehicle.Izz / (vehicle.m * vehicle.bref^2);
%vehicle.Kxz = vehicle.Ixz / (vehicle.m * vehicle.bref^2);
vehicle.Kxx =  0.04482;
vehicle.Kyy = 4.09638;
vehicle.Kzz = 0.17818;
vehicle.Kxz = 0.00630;
vehicle.mu_c = vehicle.m / (rho * vehicle.Sref * vehicle.cbar);
vehicle.mu_b = vehicle.m / (rho * vehicle.Sref * vehicle.bref);
vehicle.aileron_max = deg2rad(30);
vehicle.aileron_min = -deg2rad(30);
vehicle.elevator_max = deg2rad(17.5);
vehicle.elevator_min = -deg2rad(17.5);
vehicle.rudder_max = deg2rad(25);
vehicle.rudder_min = -deg2rad(25);

symmSys = struct;
symmSys.P = [-2 * vehicle.mu_c * vehicle.cbar / initCond.v0 0 0 0;
            0 (aeroCoeff.Cz_alpha_dot - 2 * vehicle.mu_c) * vehicle.cbar / initCond.v0 0 0;
            0 0 -vehicle.cbar / initCond.v0 0;
            0 aeroCoeff.Cm_alpha_dot * vehicle.cbar / initCond.v0 0 -2 * vehicle.mu_c * vehicle.Kyy * vehicle.cbar / initCond.v0];

symmSys.Q = [-aeroCoeff.Cx_u -aeroCoeff.Cx_alpha -aeroCoeff.Cz0 0;
            -aeroCoeff.Cz_u -aeroCoeff.Cz_alpha aeroCoeff.Cx0 -(aeroCoeff.Cz_q + 2 * vehicle.mu_c);
            0 0 0 -1;
            -aeroCoeff.Cm_u -aeroCoeff.Cm_alpha 0 -aeroCoeff.Cm_q];

symmSys.R = [-aeroCoeff.Cx_de;
            -aeroCoeff.Cz_de;
            0;
            -aeroCoeff.Cm_de];

symmSys.A = inv(symmSys.P) * symmSys.Q;
symmSys.B = inv(symmSys.P) * symmSys.R;

asymmSys = struct;
asymmSys.P = [(aeroCoeff.Cy_beta_dot - 2 * vehicle.mu_b) * vehicle.bref / initCond.v0 0 0 0;
              0 -1/2 * vehicle.bref / initCond.v0 0 0;
              0 0 -4 * vehicle.mu_b * vehicle.Kxx * vehicle.bref / initCond.v0 4 * vehicle.mu_b * vehicle.Kxz * vehicle.bref / initCond.v0;
              aeroCoeff.Cn_beta_dot * vehicle.bref / initCond.v0 0 4 * vehicle.mu_b * vehicle.Kxz * vehicle.bref / initCond.v0 -4 * vehicle.mu_b * vehicle.Kzz * vehicle.bref / initCond.v0];

asymmSys.Q = [-aeroCoeff.Cy_beta -aeroCoeff.CL -aeroCoeff.Cy_p -(aeroCoeff.Cy_r - 4 * vehicle.mu_b);
              0 0 -1 0;
              -aeroCoeff.Cl_beta 0 -aeroCoeff.Cl_p -aeroCoeff.Cl_r;
              -aeroCoeff.Cn_beta 0 -aeroCoeff.Cn_p -aeroCoeff.Cn_r];

asymmSys.R = [-aeroCoeff.Cy_da -aeroCoeff.Cy_dr;
              0 0;
              -aeroCoeff.Cl_da -aeroCoeff.Cl_dr;
              -aeroCoeff.Cn_da -aeroCoeff.Cn_dr];

asymmSys.A = inv(asymmSys.P) * asymmSys.Q;
asymmSys.B = inv(asymmSys.P) * asymmSys.R;
