% Remember that in this coordinate system, x is forward, y is right and z
% is up?


x_cg = -2.6;
y_cg = 0;
x_rot = [-0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -6.5 -6.5 -6.5 -6.5 -6.5 -6.5];
y_rot = [-4 -3 -2 2 3 4 -4 -3 -2 2 3 4];
eta_rot = ones(size(x_rot));
w_rot = [-1 -1 -1 1 1 1 -1 -1 -1 1 1 1];
k_mu = 0.1;

m = 3100;
g = 9.80665;
Jx = 200;
Jy = 1000;
Jz = 1500;

Bf = [eta_rot;
    -times(eta_rot, y_rot - y_cg);
    times(eta_rot, x_rot - x_cg);
    times(eta_rot, w_rot) * k_mu];

G = [m * g; 0; 0; 0];
A = [zeros(4, 4) eye(4);
    zeros(4,8)];
Jf = diag([-m Jx Jy Jz]);
B = [zeros(4, 4);
    inv(Jf)];