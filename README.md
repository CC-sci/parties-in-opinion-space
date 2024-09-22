This code models the motion of political parties trying to maximise their voters under a Hotelling-
Downs framework. It applies the techniques of statistical mechanics, since the situation is analogous to
an energy-minimisation problem. In particular, this code uses the Metropolis Monte Carlo algorithm.

The programme can be run from a terminal with ```python main.py```, or in ipython ```run main```. This will
produce plots of four parties moving along two policy axes. 

In addition, the following options are available. For a four-party equilibrium set p1 and p2 equal to 2 or
higher, for example.

usage: ```main.py [-h] [-1d] [-o OUTPUT] [-H] [-S] [-D] [-t] [-M] [-n NUMBER] [-r] [-p1 TURNOUTPARAMETER]
               [-p2 ACTIVISTPARAMETER] [-d {s,m,u}] [-s] [-v]
               [steps]```

### Optional Simulation Parameters
```
positional arguments:
  steps                 the number of steps the simulation runs for (default: 500)

options:
  -h, --help            show this help message and exit
  -1d, --line           run a one-dimensional Hotelling simulation (default: False)
  -o OUTPUT, --output OUTPUT
                        the name of the output file to be overwritten/created (default: output.dat)
  -H, --nohistogram     supress histograms of the parties' positions (default: False)
  -S, --scatter         show scatterplot of parties' past and final positions (default: False)
  -D, --distribution    display plots of the voter distribution (default: False)
  -t, --tab             output tab seperated files for each party (default: False)
  -M, --matpol          run MATLAB routine to calculate polarisation (default: False)
  -n NUMBER, --number NUMBER
                        set the number of parties, otherwise 2 if -1d or 4 (default: None)
  -r, --ranking         preserve the left-right ordering of the parties, requires --line (default:
                        False)
  -p1 TURNOUTPARAMETER, --turnoutparameter TURNOUTPARAMETER
                        scales effects of imperfect turnout, farther voters contribute less (default:
                        1.0)
  -p2 ACTIVISTPARAMETER, --activistparameter ACTIVISTPARAMETER
                        scales how effectively activists influence their parties, positive value gives
                        a tendency towards extremes, negative a central tendency (default: 0.5)
  -d {s,m,u}, --distributionchoice {s,m,u}
                        choose the voter distribution, either with a single central peak (s), a more
                        mildly peaked distribution (m) or the uniform distribution (u) (default: s)
  -s, --stats           print detailed results after the simulation (default: False)
  -v, --verbose         print status during execution (includes -s) (default: False)
