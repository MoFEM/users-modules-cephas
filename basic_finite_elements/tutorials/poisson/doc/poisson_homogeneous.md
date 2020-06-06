SCL-1: Poisson's equation (homogeneous BC){#basic_tutorials_poisson_homogeneous}
==========================================================

\note Prerequisite of this tutorial include \ref basic_tutorials_mesh_generation_2d
and \ref basic_tutorials_mesh_generation_3d (for the 3D extension
implementation) 

\note After finishing this tutorial, if you would like to replicate the program
and practice yourself in an existing module or in your own module, you may wish
to have a look at \ref how_to_add_new_module_and_program and \ref how_to_compile_program.

[TOC]

# Introduction {#basic_tutorials_poisson_homogeneous_introduction}

## Intended learning outcome

After this tutorial, you will learn:

- general structure of a program developed using MoFEM
- idea of Simple Interface in MoFEM and how to use it
- idea of Boundary element in MoFEM and how to use it
- process of implementing User Data Operators (UDOs) and how to  **push** them to the main program
- a way to handle homogeneous boundary condition in MoFEM
- utilisation of tools to convert outputs (MOAB) and visualise them (Paraview)

## The problem

In this first tutorial that actually solves something meaningful using MoFEM, we
will solve a simple Poisson's equation in 2D with homogeneous boundary condition
(zero boundary values). The formal definition of the problem as follows 

Find \f$ u \in \bf{R} \f$ such that
\f[
\begin{aligned}
-\nabla \cdot \nabla u(\mathbf{x}) &= f \quad {\rm in} \quad {\boldsymbol
\Omega} \\ u &= 0 \quad {\rm on} \quad \partial {\boldsymbol \Omega}
\end{aligned}
\f]



## Discretisation


# Implementation {#basic_tutorials_poisson_homogeneous_implementation}

The immediate question you may have regarding the implementation is how to
implement the matrix **K** and vector **f**. This is
done through the implementation of the so-called **User Data Operators**. UDOs
are the essential part of MoFEM and they are present in all finite element
problems implemented using MoFEM. UDO is normally called Operator (for short) by MoFEM developers.

As a common practice, typically all the implementation of UDOs for a specific
problem is put in a `*.hpp` file. This file will be included in the main `*.cpp`
file which contains all the necessary classes and functions. Detailed
explanation of the implementation of UDOs as well as classes/functions is presented below

## User Data Operators

As described in the previous, solving the Poisson problem with homogeneous would
require the computation and assembling of the matrix **K** and
vector **f**. These essential processes of computation and assembling
will be handled by four different UDOs which separately deal with the four
matrices and vectors. They are

1. Poisson2DHomogeneousOperators::OpDomainLhs is responsible to calculate and
   assemble the left-hand-side matrix **K** of domain element  
2. Poisson2DHomogeneousOperators::OpDomainRhs is responsible to calculate and
   assemble the right-hand-side vector **f** of domain element 

That was for the implementation of UDOs that are responsible for the calculation
and assembling of the matrices and vectors. Next part we will look at the
classes and functions and see how the developed UDOs are \b pushed to the main
program.

## The Poisson2DHomogeneous class


Also talk about the naming convention: camelBack and give reference to page for
naming convention in MoFEM.

### Poisson2DHomogeneous::Poisson2DHomogeneous()

This is the constructor



  
### Poisson2DHomogeneous::readMesh()
This is the first member function
  

### Poisson2DHomogeneous::setupProblem()

The next member function deals with the setting up of the problem

### Poisson2DHomogeneous::setIntegrationRules()

### Poisson2DHomogeneous::boundaryCondition()

### Poisson2DHomogeneous::assembleSystem()

### Poisson2DHomogeneous::solveSystem()

### Poisson2DHomogeneous::outputResults()

### Poisson2DHomogeneous::runAnalysis()

## The main function

Although, in the implementation, the `main` function is located at the end of
the source code, we deliberately describe it here first in order to give you a
big picture and a linear thinking about the program.

Start with interface of PETSc, MOAB and MoFEM ...

It may be confusing at this moment ... general ideas is that this part create
the intefaces that allow MoFEM to talk to and utilise functionalities from PETSc
(solvers) and MOAB (element topology management). 

Now you can just focus on the part that actually run the analysis

\code
    // Run the main analysis
    Poisson2DHomogeneous poisson_problem(mb_instance, core);
    CHKERR poisson_problem.runAnalysis();
\endcode


# Results {#basic_tutorials_poisson_homogeneous_results}

Body force is set to be \f$ f=5.0 \f$ 

## Run the program

## Output


Visualisation in Paraview

\anchor figure_poisson_homogeneous
\image html poisson_homogeneous.png "Figure 1: Poisson homogeneous visualisation." width = 900px


## Discussion

## Possible extension



# Plain program {#basic_tutorials_poisson_homogeneous_plain_program}



The plain program for both the implementation of the UDOs (\c *.hpp) and the
main program (\c *.cpp) are as follows

## Implementation of User Data Operators (*.hpp){#basic_tutorials_poisson_homogeneous_plain_program_udo}

\include poisson_2d_homogeneous.hpp

## Implementation of the main program (*.cpp) {#basic_tutorials_poisson_homogeneous_plain_program_main}

\include poisson_2d_homogeneous.cpp

