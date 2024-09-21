opendata = table();

activistParameters = [-0.4 -0.2 0 0.2 0.4 0.6 0.8 1 2 3 4:2:10];

for i = 1:length(activistParameters)
   system("python main.py 2000 --line -H -p1 0 -p2 " + activistParameters(i));
    p = get_mean_polarisation("output.dat");
    data.turnoutParameter(i) = NaN;
    data.activistParameter(i) = activistParameters(i);
    data.Polarisation1(i) = p;
end

for i = 1:length(activistParameters)
   system("python main.py 2000 --line -H -p1 0 -p2 " + activistParameters(i));
    p = get_mean_polarisation("output.dat");
    data.Polarisation2(i) = p;
end

for i = 1:length(activistParameters)
   system("python main.py 2000 --line -H -p1 0 -p2 " + activistParameters(i));
    p = get_mean_polarisation("output.dat");
    data.Polarisation3(i) = p;
end