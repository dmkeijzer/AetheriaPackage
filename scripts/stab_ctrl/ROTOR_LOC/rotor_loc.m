%%
%%%% INPUTS

Sproj = 1.2 * 11.9754;
Svee = 
b_vee = 
lv = 
rotor_direction = 
cg_fw_guess = 6;
cg_r_guess = 3;
x_rotor_loc = 
y_rotor_loc = 
ma = 
cg_frontlim = 
cg_rearlim = 

pylonstep = 0.05;
Tfac_step = 0.1;



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
Tfac = 1;
while loopforpylonsize
    loopfortfac = true;
    while loopfortfac
        x_cg_fw = [];
        x_cg_r = [];
        loopforbrokenengine = true;
        while loopforbrokenengine
            x_cg_fw(end+1) = cg_range_calc(-1, rotor_direction, r_ku, cg_fw_guess, x_rotor_loc, y_rotor_loc, rotor_eta, ma, Sproj,Tfac, Tg);
            x_cg_r(end+1) = cg_range_calc(1, rotor_direction, r_ku, cg_r_guess, x_rotor_loc, y_rotor_loc, rotor_eta, ma, Sproj,Tfac, Tg);
            rotor_eta = ones(1,n);
            rotor_eta(i2)=etabroken;
            i2=i2+1;
            if i2 > n
                loopforbrokenengine = false;
            end
        end
        
        if max(x_cg_fw) <= cg_frontlim && min(x_cg_r)>= cg_rearlim
            loopfortfac = false;
        else
            Tfac = Tfac + Tfac_step;
        end
    end
    log = vertcat(log, [Tfac, pylonsize]);
    pylonsize = pylonsize + pylonstep;
    x_rotor_loc(INDEX) = x_rotor_loc(INDEX) + pylonstep;
end