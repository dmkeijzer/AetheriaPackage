%%
%%%% INPUTS

Sproj = 1.2 * 11.9754;
Svee = 2.8078;
b_vee = 2.107;
lv = 4.3;
rotor_direction = [-1 1 -1 1 1 -1];
cg_fw_guess = 6;
cg_r_guess = 3;
x_rotor_loc = [4.564 4.564 4.64 4.64 9.295 9.295];
y_rotor_loc = [3.615 -3.615 5.015 -5.015 2.458 -2.458];
ma = 2158;
cg_frontlim = 4.71796962205636;
cg_rearlim = 5.068263100185028;

pylonstep = 0.05;
Tfac_step = 0.05;



%%
%%%% RUN

Zg = ma*9.80665+1.225*11.1^2*Sproj;
Lg = 0.5 * 1.225 * (31/3.6)^2 * (0.5*Svee*sin(deg2rad(35)))* (b_vee/2 * sin(deg2rad(35)));
Mg = 0;
Ng = 0.5 * 1.225 * (31/3.6)^2 * (0.5*Svee*sin(deg2rad(35)))*lv;
Tg = [Zg Lg Mg Ng]; %T = mg + 11.1m/s drag, ROLL = 0.5 * 1.225 * (31/3.6)^2 * (0.5*Svee*sin(35))* (b_vee/2 * sin(35)), PITCH = 0, YAW = 0.5 * 1.225 * (31/3.6)^2 * (0.5*Svee*sin(35))*lv

n=size(x_rotor_loc,2);
r_ku = ones(1,n) * 0.1;
rotor_eta = ones(1,n);

loopforpylonsize = true;
log = zeros(0,2);
pylonsize=0;
while loopforpylonsize
    loopfortfac = true;
    Tfac = 1;
    while loopfortfac
        x_cg_fw = [];
        x_cg_r = [];
        loopforbrokenengine = true;
        i2 = 1;
        while loopforbrokenengine
            x_cg_fw(end+1) = cg_range_calc(-1, rotor_direction, r_ku, cg_fw_guess, x_rotor_loc, y_rotor_loc, rotor_eta, ma, Sproj,Tfac, Tg);
            x_cg_r(end+1) = cg_range_calc(1, rotor_direction, r_ku, cg_r_guess, x_rotor_loc, y_rotor_loc, rotor_eta, ma, Sproj,Tfac, Tg);
            rotor_eta = ones(1,n);
            rotor_eta(i2)=0.5;
            if i2 > n
                loopforbrokenengine = false;
            end
            i2=i2+1;
        end
        if max(x_cg_fw) <= cg_frontlim && min(x_cg_r)>= cg_rearlim && max(x_cg_fw) > -10  && min(x_cg_r)< 100
            loopfortfac = false;
        elseif Tfac > 6 - (Tfac_step/2)
            loopfortfac = false;
            Tfac = 10;
        else
            Tfac = Tfac + Tfac_step;
        end
    end
    log = vertcat(log, [Tfac, pylonsize]);
    pylonsize = pylonsize + pylonstep;
    x_rotor_loc(1:2) = x_rotor_loc(1:2) - pylonstep;
    if pylonsize > 4.5
        loopforpylonsize = false;
    end
end
