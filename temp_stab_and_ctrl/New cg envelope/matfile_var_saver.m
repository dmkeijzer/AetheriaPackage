matfiles = dir('*.mat');
for i=1:length(matfiles)
    filename = matfiles(i).name;
    inputfile = filename(1:2);
    if filename(1) == "J"
        inputfile = "J1";
    end
    jsonname = sprintf('../../input/%s_constants.json', inputfile);
    json = fileread(jsonname);
    
    
    
    % Decode the JSON data
    data = jsondecode(json);
    
    Cma1vec = [-1.596 -1.596 -1.596 -1.466 1.644];
    Cma2vec = [0 0 0 -1.675 -1.664];
    Cmac1vec = [-0.0391 -0.0391 -0.0391 -0.0317 -0.0319];
    Cmac2vec = [0 0 0 -0.032 -0.032];
    xcgfwguess = [6 6 6 6 6];
    xcgrguess = [3 3 3 3 4];
    x1vec = [3 3 3 0.5 0.5];
    x2vec = [8.5 8.5 8.5 8 9];


    CL1 = 1.9377;
    if filename(1) == "J"
        CL2 = -0.35*data.A_h^(1/3);
    else
        CL2 = 1.816;
    end
    if filename(1) == "J"
        CLa1 = data.cl_alpha;
        CLa2 = 2*pi*data.A_h/(2+sqrt(data.A_h+(data.A_h/0.95)^2));
    else
        CLa1 = data.cl_alpha1;
        CLa2 = data.cl_alpha2;
    end
    Cm1 = Cmac1vec(i);
    Cm2 = Cmac2vec(i);
    Cma1 = Cma1vec(i);
    Cma2 = Cma2vec(i);
    if filename(1) == "J"
        S1 = data.S;
        S2 = data.S_h;
    else
        S1 = data.S1;
        S2 = data.S2;
    end
    Sproj = data.StotS * data.S;
    V2_V1_ratio = 1;
    downwash = 0.1;
    ma = data.mtom;
    x1 = x1vec(i);
    x2 = x2vec(i);
    if filename(1) == "J"
        c1 = data.mac;
        c2 = sqrt(data.S_h / data.A_h);
    else
        c1 = data.mac1;
        c2 = data.mac2;
    end
    if filename(1:2) == "J2"
        x_rotor_loc = data.x_rotor_loc';
        x_rotor_loc = [x_rotor_loc x_rotor_loc(end-1:end)];
        y_rotor_loc = data.y_rotor_loc';
        y_rotor_loc = [y_rotor_loc y_rotor_loc(end-1:end)];
        rotor_direction = data.rotor_direction';
        rotor_direction = [rotor_direction -rotor_direction(end-1:end)];
    else
        x_rotor_loc = data.x_rotor_loc';
        y_rotor_loc = data.y_rotor_loc';
        rotor_direction = data.rotor_direction';
    end
    lfus = data.l_fuselage;
    n_engines = size(x_rotor_loc,2);
    cg_fw_guess = xcgfwguess(i);
    cg_r_guess = xcgrguess(i);
    A_disk = data.diskarea / n_engines;
    savename = sprintf("%s_input.mat", filename(1:2));
    save(savename, "A_disk", "CL1", "CL2", "CLa1", "CLa2", "Cm1","Cm2", "Cma1", "Cma2", "S1", "S2", "Sproj", "V2_V1_ratio", "downwash", "ma", "x1", "x2", "c1", "c2", "x_rotor_loc","y_rotor_loc","rotor_direction", "n_engines", "cg_fw_guess", "cg_r_guess", "lfus", "-mat");
end

