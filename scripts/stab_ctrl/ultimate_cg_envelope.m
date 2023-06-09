matfiles = dir('../../input/StabCtrl/midterm/*.mat');

%%
%%%%%%HOVER CTRL LOOOOP
for i=1:length(matfiles)
    filename = matfiles(i).name;
    load(filename)

    % combinations = dec2bin(0:2^n-1)-'0';
    % combinations = combinations(sum(combinations,2) == 6,:);
    % combinations(combinations==0) = -1;
    combinations = rotor_direction;

    n=size(x_rotor_loc,2);
    r_ku = ones(1,n) * 0.1;
    rotor_eta = ones(1,n);
    
    Tfaclim = 6;
    if filename(1:2) == "W1"
        Tfaclim = 3.5;
    elseif filename(1:2) == "L1"
        Tfaclim = 3;
    end
    Tfacvec = 1:0.05:Tfaclim;
    etabroken = 0;
    if filename(1:2) == "J3"
        etabroken = 0.5;
    end
    loopfortfac = true;
    i1=1;
    resultslog = zeros(n+3,0);
    while loopfortfac
        Tfac = Tfacvec(i1);
        loopforrotationcombinations = true;
        i2 = 1;
        cgranges = zeros(4,0);
        while loopforrotationcombinations
            rotor_direction = combinations(i2,:);
            i3=1;
            x_cg_fw = [];
            x_cg_r = [];
            loopforbrokenengine = true;
            if filename(1:2) == "L1"
                loopforbrokenengine =false;
                x_cg_fw(end+1) = cg_range_calc(-1, rotor_direction, r_ku, cg_fw_guess, x_rotor_loc, y_rotor_loc, rotor_eta, ma, Sproj,Tfac);
                x_cg_r(end+1) = cg_range_calc(1, rotor_direction, r_ku, cg_r_guess, x_rotor_loc, y_rotor_loc, rotor_eta, ma, Sproj,Tfac);
            end
    
            while loopforbrokenengine
                x_cg_fw(end+1) = cg_range_calc(-1, rotor_direction, r_ku, cg_fw_guess, x_rotor_loc, y_rotor_loc, rotor_eta, ma, Sproj,Tfac);
                x_cg_r(end+1) = cg_range_calc(1, rotor_direction, r_ku, cg_r_guess, x_rotor_loc, y_rotor_loc, rotor_eta, ma, Sproj,Tfac);
                rotor_eta = ones(1,n);
                rotor_eta(i3)=etabroken;
                i3=i3+1;
                if i3 > n
                    loopforbrokenengine = false;
                end
                i3=i3+1;    
            end
    
    
    
            updatevector = [max(x_cg_fw) min(x_cg_r) rotor_direction Tfac];
            cgranges = horzcat(cgranges, updatevector');
        
            i2 = i2 +1;
            
            if i2> size(combinations,1)
                loopforrotationcombinations = false;
            end
        end
        i1=i1+1;
        if i1>size(Tfacvec)
            loopfortfac = false;
        end
        

        cgranges(:,cgranges(1,:)==-1e6) = [];
        cgranges(:,cgranges(2,:)==1e6) = [];
        cgranges = cgranges(:,cgranges(2,:) - cgranges(1,:) == max(cgranges(2,:) - cgranges(1,:)));
        resultslog = horzcat(resultslog,cgranges);
        disp(resultslog)
        if size(resultslog, 2) > 0
            cg_fw_guess = resultslog(1,end)+0.01;
            cg_r_guess = resultslog(2,end)-0.01;
        end
        %disp(resultslog)
    end
    %disp(resultslog)
    outputname = sprintf("../../output/stab_ctrl/midterm/%s_Tfac_vs_cg_range.csv", filename(1:2));
    writematrix(resultslog, outputname)
end








%%
%%%%%CG ENVELOPE PLOTTER (PPE CALC)
ROC = 2;
rho = 1.225;
for i = 1:length(matfiles)
    filename = matfiles(i).name;
    loadname = sprintf("../../input/StabCtrl/midterm/%s_input.mat", filename(1:2));
    load(loadname)
    inputfilename = sprintf("../../output/StabCtrl/midterm_output/%s_Tfac_vs_cg_range.csv", filename(1:2));
    table_data = readtable(inputfilename);
    
    xcgmax = table2array(table_data(2,:));
    xcgmin = table2array(table_data(1,:));
    Tfac = table2array(table_data(end,:));
    
    Thrust = 9.80665*ma*Tfac*(1+1/(n_engines-1))*(1+1.225*ROC^2*Sproj/(ma*9.80665))/(n_engines);
    Power_per_engine = Thrust*ROC+1.2*Thrust.*(-ROC/2+sqrt(ROC^2/4+Thrust/(2*rho*A_disk)));
    
    outputfilename = sprintf("../../output/StabCtrl/midterm_output/%s_cg_range_Ppe.csv", filename(1:2));
    writematrix(vertcat(xcgmin, xcgmax, Power_per_engine),outputfilename)
end

%%
%%%%%%CG RANGE TO WING LOCATIONS
for i = 1:length(matfiles)
    filename = matfiles(i).name;
    loadname = sprintf("../../input/StabCtrl/midterm/%s_input.mat", filename(1:2));
    load(loadname)
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
    savename = sprintf("../../output/StabCtrl/midterm_output/%s_cg_range_frontwing_location.csv", filename(1:2));
    writematrix(res, savename)
    % figure;
    % hold on
    % plot(res(1,:), res(3,:))
    % plot(res(2,:), res(3,:))
    % xlabel("Absolute cg location")
    % ylabel("Absolute frontwing location")
    % title("CG range vs frontwing location "+filename(1:2))
    % legend("Front cg boundary", "Rear cg boundary")
    % hold off
end

for i = 1:length(matfiles)
    filename = matfiles(i).name;
    loadname = sprintf("../../input/StabCtrl/midterm/%s_input.mat", filename(1:2));
    load(loadname)
    condition = true;
    x2vec = x1:0.1:lfus;
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
    savename = sprintf("../../output/StabCtrl/midterm_output/%s_cg_range_rearwing_location.csv", filename(1:2));
    writematrix(res, savename)
    % figure;
    % hold on
    % plot(res(1,:), res(3,:))
    % plot(res(2,:), res(3,:))
    % xlabel("Absolute cg location")
    % ylabel("Absolute rearwing location")
    % title("CG range vs rearwing location " + filename(1:2))
    % legend("Front cg boundary", "Rear cg boundary")
    % hold off
end

%%
%%%%%%%CG ENVELOPE PLOTTER WITH WINGS
for i = 1:length(matfiles)
    filename = matfiles(i).name;
    loadname = sprintf("../../input/StabCtrl/midterm/%s_input.mat", filename(1:2));
    load(loadname)
    Ppename = sprintf("../../output/StabCtrl/midterm_output/%s_cg_range_Ppe.csv", filename(1:2));
    Fwname = sprintf("../../output/StabCtrl/midterm_output/%s_cg_range_frontwing_location.csv", filename(1:2));
    Rwname = sprintf("../../output/StabCtrl/midterm_output/%s_cg_range_rearwing_location.csv", filename(1:2));
    Ppetable = readtable(Ppename);
    Fwtable = readtable(Fwname);
    Rwtable = readtable(Rwname);

    xcgmin_hover = table2array(Ppetable(1,:));
    xcgmax_hover = table2array(Ppetable(2,:));
    Ppe = table2array(Ppetable(3,:));

    xcgmin_fw = table2array(Fwtable(1,:));
    xcgmax_fw = table2array(Fwtable(2,:));
    fwloc = table2array(Fwtable(3,:));

    xcgmin_rw = table2array(Rwtable(1,:));
    xcgmax_rw = table2array(Rwtable(2,:));
    rwloc = table2array(Rwtable(3,:));

    cgmargin_front = cg_exc_front - 0.2*(cg_exc_aft-cg_exc_front);
    cgmargin_aft = cg_exc_aft + 0.2*(cg_exc_aft-cg_exc_front);

    figure
    sgtitle("CG envelope for Power per engine and wings location - " + filename(1:2), "FontSize",14)
    
    ax1 = subplot(2,1,1);

    yyaxis left
    hold on
    plot(xcgmin_hover, Ppe, "LineWidth",4)
    plot(xcgmax_hover,Ppe, "LineWidth", 4)
    ylabel("Ppe [W]", "FontSize",14)
    hold off

    yyaxis right
    hold on
    plot(xcgmin_fw, fwloc, "LineWidth",4)
    plot(xcgmax_fw, fwloc,"LineWidth",4)
    ylabel("Frontwing location [m]","FontSize",14)
    xline(cg_exc_front, "LineStyle","--","Color",[1 0.5 0],"LineWidth",4)
    xline(cg_exc_aft, "LineStyle","--","Color",[1 0.5 0],HandleVisibility="off",LineWidth=4)
    xline(cgmargin_front, "LineStyle","-","Color",[0 0.5 0],"LineWidth",4)
    xline(cgmargin_aft, "LineStyle","-","Color",[0 0.5 0],HandleVisibility="off",LineWidth=4)
    hold off
    xlabel("x_{cg} absolute location [m]","FontSize",14)
    legend("x_{cg}_{min} hover", "x_{cg}_{max} hover", "x_{cg}_{min} longitudinal", "x_{cg}_{max} londitudinal", "x_{cg} range req", "x_{cg} range margin", "Color", "none","EdgeColor", "k", "LineWidth",2, "FontSize", 14 )

    ax2 = subplot(2,1,2);

    yyaxis left
    hold on
    plot(xcgmin_hover, Ppe, "LineWidth",4)
    plot(xcgmax_hover,Ppe, "LineWidth",4)
    ylabel("Ppe [W]" ,"FontSize",14)
    hold off

    yyaxis right
    hold on
    plot(xcgmin_rw, rwloc, "LineWidth",4)
    plot(xcgmax_rw, rwloc,"LineWidth",4)
    ylabel("Rearwing location [m]", "FontSize",14)
    xline(cg_exc_front, "LineStyle","--","Color",[1 0.5 0],"LineWidth",4)
    xline(cg_exc_aft, "LineStyle","--","Color",[1 0.5 0],HandleVisibility="off",LineWidth=4)
    xline(cgmargin_front, "LineStyle","-","Color",[0 0.5 0],"LineWidth",4)
    xline(cgmargin_aft, "LineStyle","-","Color",[0 0.5 0],HandleVisibility="off",LineWidth=4)
    hold off
    xlabel("x_{cg} absolute location [m]", "FontSize",14)
    legend("x_{cg}_{min} hover", "x_{cg}_{max} hover", "x_{cg}_{min} longitudinal", "x_{cg}_{max} londitudinal", "x_{cg} range req", "x_{cg} range margin", "Color", "none","EdgeColor", "k", "LineWidth",2, "FontSize", 14)
end
