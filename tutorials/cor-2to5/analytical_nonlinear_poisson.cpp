/**
 * \file analytical_nonlinear_poisson.cpp
 * \ingroup mofem_simple_interface
 * \example analytical_nonlinear_poisson.cpp
 *
 * For more information and detailed explain of this
 * example see \ref poisson_tut4
 *
 *
 */

/* This file is part of MoFEM.
 * MoFEM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * MoFEM is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with MoFEM. If not, see <http://www.gnu.org/licenses/>. */

#include <BasicFiniteElements.hpp>

#include <PoissonOperators.hpp>

#include <AuxPoissonFunctions.hpp>

static char help[] = "...\n\n";

/**
 * \brief Function
 *
 * This is prescribed exact function. If this function is given by polynomial
 * order of "p" and order of approximation is "p" or higher, solution of
 * finite element method is exact (with machine precision).
 *
 *  \f[
 *  u = 1+x+2y+3z
 *  \f]
 *
 */
struct ExactFunction {
  double operator()(const double x, const double y, const double z) const {
    return 1 + x + y + pow(z, 3);
  }
};

/**
 * \brief Exact gradient
 */
struct ExactFunctionGrad {
  FTensor::Tensor1<double, 3> operator()(const double x, const double y,
                                         const double z) const {
    FTensor::Tensor1<double, 3> grad;
    grad(0) = 1;
    grad(1) = 1;
    grad(2) = 3 * z * z;
    return grad;
  }
};

/**
 * \brief Laplacian of function.
 *
 */
struct ExactLaplacianFunction {
  double operator()(const double x, const double y, const double z) const {
    return 0.4e1 + (double)(4 * x) + (double)(4 * y) + 0.4e1 * pow(z, 0.3e1) +
           0.3e1 *
               (0.6e1 * z * z + 0.6e1 * (double)x * z * z +
                0.6e1 * (double)y * z * z + 0.6e1 * pow(z, 0.5e1)) *
               z * z +
           0.6e1 *
               (0.2e1 + (double)(2 * x) + (double)(2 * y) +
                0.2e1 * pow(z, 0.3e1) + (double)(x * x) + (double)(2 * x * y) +
                0.2e1 * (double)x * pow(z, 0.3e1) + (double)(y * y) +
                0.2e1 * (double)y * pow(z, 0.3e1) + pow(z, 0.6e1)) *
               z;
  }
};

struct FunA {
  double operator()(const double u) { return 1 + u * u; }
};

struct DiffFunA {
  double operator()(const double u) { return 2 * u; }
};

struct VolRuleNonlinear {
  int operator()(int, int, int p) const { return 2 * (p + 1); }
};

int main(int argc, char *argv[]) {

  // Initialize PETSc
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);
  // Create MoAB database
  moab::Core moab_core;              // create database
  moab::Interface &moab = moab_core; // create interface to database

  try {

    // Get command line options
    int order = 3;                    // default approximation order
    PetscBool flg_test = PETSC_FALSE; // true check if error is numerical error
    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Poisson's problem options",
                             "none");
    // Set approximation order
    CHKERR PetscOptionsInt("-order", "approximation order", "", order, &order,
                           PETSC_NULL);
    // Set testing (used by CTest)
    CHKERR PetscOptionsBool("-test", "if true is ctest", "", flg_test,
                            &flg_test, PETSC_NULL);
    ierr = PetscOptionsEnd();
    CHKERRG(ierr)

    // Create MoFEM database and link it to MoAB
    MoFEM::Core mofem_core(moab);           // create database
    MoFEM::Interface &m_field = mofem_core; // create interface to database
    // Register DM Manager
    CHKERR DMRegister_MoFEM("DMMOFEM"); // register MoFEM DM in PETSc

    // Create vector to store approximation global error
    Vec global_error;
    CHKERR PoissonExample::AuxFunctions(m_field).createGhostVec(&global_error);

    // First we crate elements, implementation of elements is problem
    // independent, we do not know yet what fields are present in the problem,
    // or even we do not decided yet what approximation base or spaces we are
    // going to use. Implementation of element is free from those constrains and
    // can be used in different context.

    // Elements used by KSP & DM to assemble system of equations
    boost::shared_ptr<ForcesAndSourcesCore>
        domain_lhs_fe; ///< Volume element for the matrix
    boost::shared_ptr<ForcesAndSourcesCore>
        boundary_lhs_fe; ///< Boundary element for the matrix
    boost::shared_ptr<ForcesAndSourcesCore>
        domain_rhs_fe; ///< Volume element to assemble vector
    boost::shared_ptr<ForcesAndSourcesCore>
        boundary_rhs_fe; ///< Volume element to assemble vector
    boost::shared_ptr<ForcesAndSourcesCore>
        domain_error; ///< Volume element evaluate error
    boost::shared_ptr<ForcesAndSourcesCore>
        post_proc_volume; ///< Volume element to Post-process results
    boost::shared_ptr<ForcesAndSourcesCore> null; ///< Null element do nothing
    {
      // Add problem specific operators the generic finite elements to calculate
      // matrices and vectors.
      CHKERR PoissonExample::CreateFiniteElements(m_field)
          .createFEToAssembleMatrixAndVectorForNonlinearProblem(
              ExactFunction(), ExactLaplacianFunction(), FunA(), DiffFunA(),
              domain_lhs_fe, boundary_lhs_fe, domain_rhs_fe, boundary_rhs_fe,
              VolRuleNonlinear());
      // Add problem specific operators the generic finite elements to calculate
      // error on elements and global error in H1 norm
      CHKERR PoissonExample::CreateFiniteElements(m_field)
          .createFEToEvaluateError(ExactFunction(), ExactFunctionGrad(),
                                   global_error, domain_error);
      // Post-process results
      CHKERR PoissonExample::CreateFiniteElements(m_field)
          .creatFEToPostProcessResults(post_proc_volume);
    }

    // Get simple interface is simplified version enabling quick and
    // easy construction of problem.
    Simple *simple_interface;
    // Query interface and get pointer to Simple interface
    CHKERR m_field.getInterface(simple_interface);

    // Build problem with simple interface
    {

      // Get options for simple interface from command line
      CHKERR simple_interface->getOptions();
      // Load mesh file to database
      CHKERR simple_interface->loadFile();

      // Add field on domain and boundary. Field is declared by space and base
      // and rank. space can be H1. Hcurl, Hdiv and L2, base can be
      // AINSWORTH_LEGENDRE_BASE, DEMKOWICZ_JACOBI_BASE and more, where rank is
      // number of coefficients for dof.
      //
      // Simple interface assumes that there are four types of field; 1) defined
      // on body domain, 2) fields defined on body boundary, 3) skeleton field
      // defined on finite element skeleton. Finally data field is defined on
      // body domain. Data field is not solved but used for post-process or to
      // keep material parameters, etc. Here we using it to calculate
      // approximation error on elements.

      // Add domain filed "U" in space H1 and Legendre base, Ainsworth recipe is
      // used to construct base functions.
      CHKERR simple_interface->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE,
                                              1);
      // Add Lagrange multiplier field on body boundary
      CHKERR simple_interface->addBoundaryField("L", H1,
                                                AINSWORTH_LEGENDRE_BASE, 1);
      // Add error (data) field, we need only L2 norm. Later order is set to 0,
      // so this is piecewise discontinuous constant approx., i.e. 1 DOF for
      // element. You can use more DOFs and collate moments of error to drive
      // anisotropic h/p-adaptivity, however this is beyond this example.
      CHKERR simple_interface->addDataField("ERROR", L2,
                                            AINSWORTH_LEGENDRE_BASE, 1);

      // Set fields order domain and boundary fields.
      CHKERR simple_interface->setFieldOrder("U",
                                             order); // to approximate function
      CHKERR simple_interface->setFieldOrder("L",
                                             order); // to Lagrange multipliers
      CHKERR simple_interface->setFieldOrder(
          "ERROR", 0); // approximation order for error

      // Setup problem. At that point database is constructed, i.e. fields,
      // finite elements and problem data structures with local and global
      // indexing.
      CHKERR simple_interface->setUp();
    }

    // Get access to PETSC-MoFEM DM manager.
    // or more derails see
    // <http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/index.html>
    // Form that point internal MoFEM data structures are linked with PETSc
    // interface. If DM functions contains string MoFEM is is MoFEM specific DM
    // interface function, otherwise DM function part of standard PETSc
    // interface.

    DM dm;
    // Get dm
    CHKERR simple_interface->getDM(&dm);

    // Set KSP context for DM. At that point only elements are added to DM
    // operators. Calculations of matrices and vectors is executed by KSP
    // solver. This part of the code makes connection between implementation of
    // finite elements and data operators with finite element declarations in DM
    // manager. From that point DM takes responsibility for executing elements,
    // calculations of matrices and vectors, and solution of the problem.
    {
      // Set operators for SNES solver
      CHKERR DMMoFEMSNESSetJacobian(dm, simple_interface->getDomainFEName(),
                                    domain_lhs_fe, null, null);
      CHKERR DMMoFEMSNESSetJacobian(dm, simple_interface->getBoundaryFEName(),
                                    boundary_lhs_fe, null, null);
      // Set calculation of the right hand side vector for KSP solver
      CHKERR DMMoFEMSNESSetFunction(dm, simple_interface->getDomainFEName(),
                                    domain_rhs_fe, null, null);
      CHKERR DMMoFEMSNESSetFunction(dm, simple_interface->getBoundaryFEName(),
                                    boundary_rhs_fe, null, null);
    }

    // Solve problem, only PETEc interface here.
    {

      // Create the right hand side vector and vector of unknowns
      Vec F, D;
      CHKERR DMCreateGlobalVector(dm, &F);
      // Create unknown vector by creating duplicate copy of F vector. only
      // structure is duplicated no values.
      CHKERR VecDuplicate(F, &D);

      // Create solver and link it to DM
      SNES solver;
      CHKERR SNESCreate(PETSC_COMM_WORLD, &solver);
      CHKERR SNESSetFromOptions(solver);
      CHKERR SNESSetDM(solver, dm);
      // Set-up solver, is type of solver and pre-conditioners
      CHKERR SNESSetUp(solver);
      // At solution process, KSP solver using DM creates matrices, Calculate
      // values of the left hand side and the right hand side vector. then
      // solves system of equations. Results are stored in vector D.
      CHKERR SNESSolve(solver, F, D);

      // Scatter solution on the mesh. Stores unknown vector on field on the
      // mesh.
      CHKERR DMoFEMMeshToGlobalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

      // Clean data. Solver and vector are not needed any more.
      CHKERR SNESDestroy(&solver);
      CHKERR VecDestroy(&D);
      CHKERR VecDestroy(&F);
    }

    // Calculate error
    {
      // Loop over all elements in mesh, and run users operators on each
      // element.
      CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
                                      domain_error);
      CHKERR PoissonExample::AuxFunctions(m_field).assembleGhostVector(
          global_error);
      CHKERR PoissonExample::AuxFunctions(m_field).printError(global_error);
      if (flg_test == PETSC_TRUE) {
        CHKERR PoissonExample::AuxFunctions(m_field).testError(global_error);
      }
    }

    {
      // Loop over all elements in the mesh and for each execute
      // post_proc_volume element and operators on it.
      CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
                                      post_proc_volume);
      // Write results
      CHKERR boost::static_pointer_cast<PostProcVolumeOnRefinedMesh>(
          post_proc_volume)
          ->writeFile("out_vol.h5m");
    }

    // Destroy DM, no longer needed.
    CHKERR DMDestroy(&dm);

    // Destroy ghost vector
    CHKERR VecDestroy(&global_error);
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}