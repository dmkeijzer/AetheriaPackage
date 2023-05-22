matfiles = dir('*.mat');
for i = 1:length(matfiles)
    filename = matfiles(i).name;
    Ppename = sprintf("%s_cg_range_Ppe.csv", filename(1:2));
    Fwname = sprintf("%s_cg_range_frontwing_location.csv", filename(1:2));
    Rwname = sprintf("%s_cg_range_rearwing_location.csv", filename(1:2));
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

    figure
    sgtitle("CG envelope for Power per engine and wings lcoation - " + filename(1:2))
    
    ax1 = subplot(2,1,1);

    yyaxis left
    hold on
    plot(xcgmin_hover, Ppe)
    plot(xcgmax_hover,Ppe)
    ylabel("Ppe")
    hold off

    yyaxis right
    hold on
    plot(xcgmin_fw, fwloc)
    plot(xcgmax_fw, fwloc)
    ylabel("Frontwing location")
    hold off

    legend("x_{cg}_{min} hover", "x_{cg}_{max} hover", "x_{cg}_{min} longitudinal", "x_{cg}_{max} londitudinal")

    ax2 = subplot(2,1,2);

    yyaxis left
    hold on
    plot(xcgmin_hover, Ppe)
    plot(xcgmax_hover,Ppe)
    ylabel("Ppe")
    hold off

    yyaxis right
    hold on
    plot(xcgmin_rw, rwloc)
    plot(xcgmax_rw, rwloc)
    ylabel("Rearwing location")
    hold off

    legend("x_{cg}_{min} hover", "x_{cg}_{max} hover", "x_{cg}_{min} longitudinal", "x_{cg}_{max} londitudinal")
end
