data = table();

turnoutParameters = [0 0.5 1 1.5 2:1:12];

for i = 1:length(turnoutParameters)
    system("python main.py 1500 -H -p2 0 -p1 " + turnoutParameters(i));
    p = get_mean_polarisation("output.dat");
    data.turnoutParameter(i) = turnoutParameters(i);
    data.activistParameter(i) = 0;
    data.Polarisation1(i) = p;
end

for i = 1:length(turnoutParameters)
    system("python main.py 1500 -H -p2 0 -p1 " + turnoutParameters(i));
    p = get_mean_polarisation("output.dat");
    data.Polarisation2(i) = p;
end

for i = 1:length(turnoutParameters)
    system("python main.py 1500 -H -p2 0 -p1 " + turnoutParameters(i));
    p = get_mean_polarisation("output.dat");
    data.Polarisation3(i) = p;
end