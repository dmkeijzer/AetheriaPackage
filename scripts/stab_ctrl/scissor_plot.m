loopforShS = true;
ShSvec = 0:0.002:0.6;
i=1;
res = zeros(3,0);
load("../../input/StabCtrl/midterm/W1_input.mat")
while loopforShS
    
    x_np_nom=-Cma1+(CLa1*x1)/(c1)-Cma2*(1-downwash)*(ShSvec(i))*(c2/c1)*(V2_V1_ratio)^2+CLa2*(1-downwash)*(x2/c1)*(ShSvec(i))*(V2_V1_ratio)^2;
    x_np_den=CLa1/c1+CLa2/c1*(1-downwash)*(ShSvec(i))*(V2_V1_ratio)^2;
    x_np=x_np_nom/x_np_den;
    
    x_cg = (-Cm1-Cm2*(c2/c1)*(ShSvec(i))*V2_V1_ratio^2+CL1*x1/c1+CL2*(x2/c1)*(ShSvec(i))*V2_V1_ratio^2)/(CL1/c1 + (CL2/c1)*(ShSvec(i))*V2_V1_ratio^2);
    res = horzcat(res, [x_cg x_np ShSvec(i)]');
    i = i+1;
    if i > length(ShSvec)
        loopforShS = false;
    end
end

figure
plot(res(1,:), res(end,:), res(2,:), res(end,:))
xlabel("x_{cg} limit location [m]")
ylabel("S_h / S ratio [-]")
title("Scissor plot")
legend("Front cg limit", "Aft cg limit")