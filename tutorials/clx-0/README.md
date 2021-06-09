# Readme for linear acoustics

# Introduction
This directory contains the input and executable files for the tutorial [CLX-0: Linear Acoustics with quasi-absorbing BCs](http://mofem.eng.gla.ac.uk/mofem/html/tutorial_hemholtz_problem.html)

## Data (default)

- Wave number, k=90 
- order of approximations, order = 6
## Running code

```
mpirun -np 2 ./helmholtz -file_name part.h5m -order 8 -k 180
```

## Extra input

Extra input commands can be found in ./param_file.petsc