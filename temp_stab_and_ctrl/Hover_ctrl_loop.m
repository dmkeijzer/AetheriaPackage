x_cg_fw = [];
x_cg_r = [];
condition = true;
i=1;
cgranges = zeros(3,0);
%%%%INPUTS%%%%

s2i = struct('anticlockwise', 1, 'clockwise', -1);
rotor_direction = [-1 1 -1 1 -1 1];

n=size(rotor_direction);
n=n(2);
r_ku = ones(1,n) * 0.1;

x_rotor_loc = [0.275 0.1375 -0.1375 -0.275 -0.1375 0.1375];

y_rotor_loc = [0 0.238 0.238 0 -0.238 -0.238];

Rotor = 1:n;

rotor_eta = ones(1,n);

mass = 1.535;

S_proj = 0;

cg_fw_gess = 0;

cg_r_guess = 0;

Tfac = 1.2;


%%%%LOOOOP%%%%
combinations = unique(perms(rotor_direction), 'rows');
othercondition = true;
i1=1;
while othercondition
    rotor_direction = combinations(i1,:);
    i=1;
    x_cg_fw = [];
    x_cg_r = [];
    condition = true;
    while condition
        x_cg_fw(end+1) = cg_range_calc(-1, rotor_direction, r_ku, cg_fw_gess, x_rotor_loc, y_rotor_loc, Rotor, rotor_eta, mass, S_proj,Tfac);
        x_cg_r(end+1) = cg_range_calc(1, rotor_direction, r_ku, cg_r_guess, x_rotor_loc, y_rotor_loc, Rotor, rotor_eta, mass, S_proj,Tfac);
        rotor_eta = ones(1,n);
        rotor_eta(i)=0;
        if i > n
            condition = false;
        end
        i=i+1;
    end
    updatevector = [max(x_cg_fw) min(x_cg_r) rotor_direction];
    cgranges = horzcat(cgranges, updatevector');
    % fprintf('The constraining fw cg is %d.\n', max(x_cg_fw))
    % fprintf('The costraining rear cg is %d.\n', min(x_cg_r))
    % fprintf('The rotor direction vector is %s.\n', mat2str(rotor_direction))

    i1 = i1 +1;
    
    if i1> size(combinations,1)
        othercondition = false;
    end
end
cgranges = cgranges(:,cgranges(2,:) - cgranges(1,:) == max(cgranges(2,:) - cgranges(1,:)));



