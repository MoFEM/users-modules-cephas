# Readme for linear elasticity

# Introduction
This directory contains the input and executable files for the tutorial [VEC-0: Linear elastic](http://mofem.eng.gla.ac.uk/mofem/html/tutorial_elastic_problem.html)

## Data (default)

- order of approximations, order = 2

## Running code
# 2D Code
```
./elastic_2d -file_name beam_2D.cub -order 2
```
# 3D Code
```
./elastic_3d -file_name beam_3D.cub -order 2
```

## Extra input

Extra input commands can be found in ./param_file.petsc