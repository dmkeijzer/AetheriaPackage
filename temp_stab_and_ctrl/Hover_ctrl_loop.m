x_cg_fw = [];
x_cg_r = [];
condition = true;
i=1;
cgranges = zeros(4,0);
%%%%INPUTS%%%%

% s2i = struct('anticlockwise', 1, 'clockwise', -1);
% %%%%%%%%%%UNCOMMENT DESIRED DESIGN%%%%%%%%%%%%
% 
% W1
% x_rotor_loc = [1.19 1.19 1.19 1.19 1.19 1.19 9.55 9.55 9.55 9.55 9.55 9.55];
% 
% y_rotor_loc = [4.2 2.99 1.78 -1.78 -2.99 -4.2 4.2 2.99 1.78 -1.78 -2.99 -4.2];
%
% rotor_direction = [-1 -1 -1 1 1 1 -1 -1 -1 1 1 1];
% 
% S_proj = 16.8;
% 
% cg_fw_gess = 5;
% 
% cg_r_guess = 5;


%%%%%%J1
x_rotor_loc = [0.57 0.57 3.4 3.4 6.8 6.8];

y_rotor_loc = [2.3 -2.3 5.4 -5.4 2.3 -2.3];

rotor_direction = [-1 1 -1 1 -1 1];

S_proj = 16.8;

cg_fw_gess = 6;

cg_r_guess = 3;


n=size(x_rotor_loc,2);
r_ku = ones(1,n) * 0.1;

Rotor = 1:n;

rotor_eta = ones(1,n);

mass = 2510;



Tfacvec = 1:0.05:6;



%%%%LOOOOP%%%%
% combinations = unique(perms(rotor_direction), 'rows');
% combinations = dec2bin(0:2^n-1)-'0';
% combinations = combinations(sum(combinations,2) == 6,:);
% combinations(combinations==0) = -1;
combinations = rotor_direction;


othercondition = true;
i1=1;
yetanothercondition = true;
i2=1;
resultslog = zeros(n+3,0);
while yetanothercondition
    Tfac = Tfacvec(i2);
    othercondition = true;
    i1 = 1;
    cgranges = zeros(4,0);
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
            i=i+1;
            if i > n
                condition = false;
            end
            
        end
        updatevector = [max(x_cg_fw) min(x_cg_r) rotor_direction Tfac];
        cgranges = horzcat(cgranges, updatevector');
        % fprintf('The constraining fw cg is %d.\n', max(x_cg_fw))
        % fprintf('The costraining rear cg is %d.\n', min(x_cg_r))
        % fprintf('The rotor direction vector is %s.\n', mat2str(rotor_direction))
    
        i1 = i1 +1;
        
        if i1> size(combinations,1)
            othercondition = false;
        end
    end
    i2=i2+1;
    if i2>size(Tfacvec)
        yetanothercondition = false;
    end
    
    %cgranges = cgranges(:, ~cgranges(1:) == -1e6);
    cgranges(:,cgranges(1,:)==-1e6) = [];
    %cgranges = cgranges(:, ~cgranges(2:) == 1e6);
    cgranges(:,cgranges(2,:)==1e6) = [];
    % hold on
    % scatter(cgranges(1,:), cgranges(end,:))
    % scatter(cgranges(2,:), cgranges(end,:))
    cgranges = cgranges(:,cgranges(2,:) - cgranges(1,:) == max(cgranges(2,:) - cgranges(1,:)));
    resultslog = horzcat(resultslog,cgranges);
end
% %cgranges = cgranges(:, ~cgranges(1:) == -1e6);
% cgranges(:,cgranges(1,:)==-1e6) = [];
% %cgranges = cgranges(:, ~cgranges(2:) == 1e6);
% cgranges(:,cgranges(2,:)==1e6) = [];
% cgranges = cgranges(:,cgranges(2,:) - cgranges(1,:) == max(cgranges(2,:) - cgranges(1,:)));
% resultslog(:,all(diff(resultslog))==0) = [];

disp(resultslog)
plot(resultslog(1,:),resultslog(end,:))
hold on
plot(resultslog(2,:),resultslog(end,:))
plot(resultslog(2,:)-resultslog(1,:),resultslog(end,:))
title({'Cg range vs thrust surplus for hover controllability - W1'},...
    'FontWeight','bold',...
    'FontSize',18);
xlabel({'x cg [m]'},'FontSize',18);
ylabel({'T factor'},'FontSize',18);





