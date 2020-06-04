SCL-1: Poisson's equation (homogeneous boundary condition){#basic_tutorials_poisson_homogeneous}
==========================================================

\note Prerequisite of this tutorial include \ref basic_tutorials_mesh_generation_2d
and \ref basic_tutorials_mesh_generation_3d (for the 3D extension
implementation) 

\note After finishing this tutorial, if you would like to replicate the program
and practice yourself in an existing module or in your own module, you may wish
to have a look at \ref how_to_add_new_module_and_program 

[TOC]

# Introduction {#basic_tutorials_poisson_homogeneous_introduction}

## Intended learning outcome {#basic_tutorials_poisson_homogeneous_ilo}

After this tutorial, you will learn:

- general structure of a program developed using MoFEM
- idea of Simple Interface in MoFEM and how to use it
- idea of Boundary element in MoFEM and how to use it
- process of implementing User Data Operators (UDOs) and how to  **push** them to the main program
- a way to handle homogeneous boundary condition in MoFEM
- utilisation of tools to convert outputs (MOAB) and visualise them (Paraview)

## The problem {#basic_tutorials_poisson_homogeneous_the_problem}

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



## Discretisation {#basic_tutorials_poisson_homogeneous_discretisation}


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

## User Data Operators {#basic_tutorials_poisson_homogeneous_udo}

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

## The Poisson2DHomogeneous class {#basic_tutorials_poisson_homogeneous_main_class}


Also talk about the naming convention: camelBack and give reference to page for
naming convention in MoFEM.

 basic_tutorials_poisson_homogeneous_the_main_function The main() function

Although, in the implementation, the \c main() function is located at the end of
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

## The class functions {#basic_tutorials_poisson_homogeneous_the_class_functions}


### Poisson2DHomogeneous::runAnalysis() {#basic_tutorials_poisson_homogeneous_run_analysis}
  

### Poisson2DHomogeneous::readMesh() {#basic_tutorials_poisson_homogeneous_read_mesh}
  

### Poisson2DHomogeneous::setupProblem() {#basic_tutorials_poisson_homogeneous_setup_problem}

### Poisson2DHomogeneous::setIntegrationRules() {#basic_tutorials_poisson_homogeneous_set_integration_rules}

### Poisson2DHomogeneous::boundaryCondition() {#basic_tutorials_poisson_homogeneous_boundary_condition}

### Poisson2DHomogeneous::assembleSystem() {#basic_tutorials_poisson_homogeneous_assemble_system}

### Poisson2DHomogeneous::solveSystem() {#basic_tutorials_poisson_homogeneous_solve_system}

### Poisson2DHomogeneous::outputResults() {#basic_tutorials_poisson_homogeneous_output_results}


# Results {#basic_tutorials_poisson_homogeneous_results}



## Run the program {#sub_sbasic_tutorials_poisson_homogeneous_run_the_programection}

## Output {#basic_tutorials_poisson_homogeneous_output}

## Discussion {#basic_tutorials_poisson_homogeneous_discussion}

## Possible extension {#basic_tutorials_poisson_homogeneous_possible_extension}



# Plain program {#basic_tutorials_poisson_homogeneous_plain_program}



The plain program for both the implementation of the UDOs (\c *.hpp) and the
main program (\c *.cpp) are as follows

## Implementation of User Data Operators (*.hpp){#basic_tutorials_poisson_homogeneous_plain_program_udo}

\include poisson_2d_homogeneous.hpp

## Implementation of the main program (*.cpp) {#basic_tutorials_poisson_homogeneous_plain_program_main}

\include poisson_2d_homogeneous.cpp

