xlewing = 4.28;
xlewingtip = 4.84-4.58;
pylon = 2.5;
xleinboard = 4.764 - 4.58;
x_rotor_loc = [xlewing+xleinboard-pylon xlewing+xleinboard-pylon xlewing+xlewingtip xlewing+xlewingtip 9.295 9.295];
y_rotor_loc = [3.615 -3.615 5.015 -5.015 2.458 -2.458];
rotor_direction = [-1 1 -1 1 1 -1];
ma = 2158;
Sproj = 1.2 * 12;
cg_fw_guess = 7;
cg_r_guess = 3;

combinations = rotor_direction;

n=size(x_rotor_loc,2);
r_ku = ones(1,n) * 0.1;
rotor_eta = ones(1,n);

Tfaclim = 4;
Tfacvec = 1:0.05:Tfaclim;
etabroken = 0.5;

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


        while loopforbrokenengine
            x_cg_fw(end+1) = cg_range_calc(-1, rotor_direction, r_ku, cg_fw_guess, x_rotor_loc, y_rotor_loc, rotor_eta, ma, Sproj,Tfac);
            x_cg_r(end+1) = cg_range_calc(1, rotor_direction, r_ku, cg_r_guess, x_rotor_loc, y_rotor_loc, rotor_eta, ma, Sproj,Tfac);
            rotor_eta = ones(1,n);
            rotor_eta(i3)=etabroken;
            i3=i3+1;
            if i3 > n
                loopforbrokenengine = false;
            end

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
    %disp(cgranges)
    cgranges = cgranges(:,cgranges(2,:) - cgranges(1,:) == max(cgranges(2,:) - cgranges(1,:)));
    
    resultslog = horzcat(resultslog,cgranges);
    disp(resultslog)
    if size(resultslog, 2) > 0
        cg_fw_guess = resultslog(1,end)+0.01;
        cg_r_guess = resultslog(2,end)-0.01;
    end
    %disp(resultslog)
end
disp(resultslog)
%outputname = sprintf("../../output/stab_ctrl/midterm/%s_Tfac_vs_cg_range.csv", filename(1:2));
%writematrix(resultslog, outputname)