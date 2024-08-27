function polarisation = get_mean_polarisation(filename)
output = importpartyoutput(filename);

% Removes comment and timestep lines and makes a numeric matrix
% ToDo: Remove unnecessary variables at the end
filteredOutput = output(output.Z == 0, :);
filteredNumeric = filteredOutput{:,2:4};

filteredOutput.Norm = vecnorm(filteredNumeric, 2, 2);

% Remove semicolon to make this verbose
groupsummary(filteredOutput, "Party", "mean", "Norm");
polarisation = mean(filteredOutput.Norm);

end