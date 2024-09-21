data = table();

turnoutParameters = [0 0.5 1 1.5 2:2:16 24];

for i = 1:length(turnoutParameters)
    system("python main.py 2000 -H --line -p2 0 -p1 " + turnoutParameters(i));
    p = get_mean_polarisation("output.dat");
    data.turnoutParameter(i) = turnoutParameters(i);
    data.activistParameter(i) = 0;
    data.Polarisation1(i) = p;
end

for i = 1:length(turnoutParameters)
    system("python main.py 2000 -H --line -p2 0 -p1 " + turnoutParameters(i));
    p = get_mean_polarisation("output.dat");
    data.Polarisation2(i) = p;
end

for i = 1:length(turnoutParameters)
    system("python main.py 2000 -H --line -p2 0 -p1 " + turnoutParameters(i));
    p = get_mean_polarisation("output.dat");
    data.Polarisation3(i) = p;
end