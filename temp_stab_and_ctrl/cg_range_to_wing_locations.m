matfiles = dir('*.mat');

for i = 1:length(matfiles)
    filename = matfiles(i).name;
    load(filename);
    condition = true;
    x1vec = 0:0.1:x2;
    i1 = 1;
    res = zeros(3,0);
    while condition
        x1 = x1vec(i1);
        x_np_nom=-Cma1+(CLa1*x1)/(c1)-Cma2*(1-downwash)*(S2/S1)*(c2/c1)*(V2_V1_ratio)^2+CLa2*(1-downwash)*(x2/c1)*(S2/S1)*(V2_V1_ratio)^2;
        x_np_den=CLa1/c1+CLa2/c1*(1-downwash)*(S2/S1)*(V2_V1_ratio)^2;
        x_np=x_np_nom/x_np_den;

        x_cg = (-Cm1-Cm2*(c2/c1)*(S2/S1)*V2_V1_ratio^2+CL1*x1/c1+CL2*(x2/c1)*(S2/S1)*V2_V1_ratio^2)/(CL1/c1 + (CL2/c1)*(S2/S1)*V2_V1_ratio^2);
        res=horzcat(res, [x_cg x_np x1]');
        i1=i1+1;
        if i1>length(x1vec)
            condition = false;
        end
    end
    savename = sprintf("%s_cg_range_frontwing_location.csv", filename(1:2));
    writematrix(res, savename)
    figure;
    hold on
    plot(res(1,:), res(3,:))
    plot(res(2,:), res(3,:))
    xlabel("Absolute cg location")
    ylabel("Absolute frontwing location")
    title("CG range vs frontwing location"+filename(1:2))
    legend("Front cg boundary", "Rear cg boundary")
    hold off
end

for i = 1:length(matfiles)
    filename = matfiles(i).name;
    load(filename);
    condition = true;
    x2vec = x1:0.1:12;
    i1 = 1;
    res = zeros(3,0);
    while condition
        x2 = x2vec(i1);
        x_np_nom=-Cma1+(CLa1*x1)/(c1)-Cma2*(1-downwash)*(S2/S1)*(c2/c1)*(V2_V1_ratio)^2+CLa2*(1-downwash)*(x2/c1)*(S2/S1)*(V2_V1_ratio)^2;
        x_np_den=CLa1/c1+CLa2/c1*(1-downwash)*(S2/S1)*(V2_V1_ratio)^2;
        x_np=x_np_nom/x_np_den;

        x_cg = (-Cm1-Cm2*(c2/c1)*(S2/S1)*V2_V1_ratio^2+CL1*x1/c1+CL2*(x2/c1)*(S2/S1)*V2_V1_ratio^2)/(CL1/c1 + (CL2/c1)*(S2/S1)*V2_V1_ratio^2);
        res=horzcat(res, [x_cg x_np x2]');
        i1=i1+1;
        if i1>length(x2vec)
            condition = false;
        end
    end
    savename = sprintf("%s_cg_range_rearwing_location.csv", filename(1:2));
    writematrix(res, savename)
    figure;
    hold on
    plot(res(1,:), res(3,:))
    plot(res(2,:), res(3,:))
    xlabel("Absolute cg location")
    ylabel("Absolute rearwing location")
    title("CG range vs rearwing location" + filename(1:2))
    legend("Front cg boundary", "Rear cg boundary")
    hold off
end