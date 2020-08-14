/**
 * \file heat_equation.cpp
 * \example heat_equation.cpp
 *
 * \brief Solve the time-dependent Heat Equation
 \f[
 \begin{aligned}
\frac{\partial u(\mathbf{x}, t)}{\partial t}-\Delta u(\mathbf{x}, t)
&=f(\mathbf{x}, t) & & \forall \mathbf{x} \in \Omega, t \in(0, T), \\
u(\mathbf{x}, 0) &=u_{0}(\mathbf{x}) & & \forall \mathbf{x} \in \Omega, \\
u(\mathbf{x}, t) &=g(\mathbf{x}, t) & & \forall \mathbf{x} \in \partial \Omega,
t \in(0, T). \end{aligned}
 \f]
 **/

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

#include <stdlib.h>
#include <cmath>
#include <BasicFiniteElements.hpp>
#include <heat_equation.hpp>

using namespace MoFEM;
using namespace HeatEquationOperators;

static char help[] = "...\n\n";

struct HeatEquation {
public:
  HeatEquation(MoFEM::Interface &m_field);

  // Declaration of the main function to run analysis
  MoFEMErrorCode runProgram();

private:
  // Declaration of other main functions called in runProgram()
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode initialCondition();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();

  // Function to calculate the Source term
  static double sourceTermFunction(const double x, const double y,
                                   const double z, const double t) {
    return 0.1 * pow(M_E, -M_PI * M_PI * t) * sin(1. * M_PI * x) *
           sin(2. * M_PI * y);
    // return 0;
  }

  // Function to calculate the Boundary term
  static double boundaryFunction(const double x, const double y, const double z,
                                 const double t) {
    return abs(0.1 * pow(M_E, -M_PI * M_PI * t) * sin(2. * M_PI * x) *
               sin(3. * M_PI * y));
    // return 0;
  }

  // initial value of u (initial condition)
  const double initU = 0.1;

  // Main interfaces
  MoFEM::Interface &mField;
  Simple *simpleInterface;

  // mpi parallel communicator
  MPI_Comm mpiComm;
  // Number of processors
  const int mpiRank;

  // Discrete Manager and time-stepping solver using SmartPetscObj
  SmartPetscObj<DM> dM;
  SmartPetscObj<TS> tsSolver;

  // Field name and approximation order
  std::string domainField;
  int oRder;
  MatrixDouble invJac;

  // Object to mark boundary entities for the assembling of domain elements
  boost::shared_ptr<std::vector<bool>> boundaryMarker;

  // MoFEM working Pipelines for stiff part function and tangent (Jacobian)
  boost::shared_ptr<FaceEle> stiffTangentLhsMatrixPipeline;
  boost::shared_ptr<FaceEle> stiffFunctionRhsVectorPipeline;
  boost::shared_ptr<EdgeEle> boundaryLhsMatrixPipeline;
  boost::shared_ptr<EdgeEle> boundaryRhsVectorPipeline;

  // Objects needed for solution updates in Newton's method
  boost::shared_ptr<DataAtGaussPoints> previousUpdate;
  boost::shared_ptr<VectorDouble> fieldValuePtr;
  boost::shared_ptr<MatrixDouble> fieldGradPtr;
  boost::shared_ptr<VectorDouble> fieldDotPtr;

  // Object needed for postprocessing
  boost::shared_ptr<PostProcFaceOnRefinedMesh> postProcFace;

  // Boundary entities marked for fieldsplit (block) solver - optional
  Range boundaryEntitiesForFieldsplit;
};

HeatEquation::HeatEquation(MoFEM::Interface &m_field)
    : domainField("U"), mField(m_field), mpiComm(mField.get_comm()),
      mpiRank(mField.get_comm_rank()) {
  stiffTangentLhsMatrixPipeline =
      boost::shared_ptr<FaceEle>(new FaceEle(mField));
  stiffFunctionRhsVectorPipeline =
      boost::shared_ptr<FaceEle>(new FaceEle(mField));
  boundaryLhsMatrixPipeline = boost::shared_ptr<EdgeEle>(new EdgeEle(mField));
  boundaryRhsVectorPipeline = boost::shared_ptr<EdgeEle>(new EdgeEle(mField));

  previousUpdate =
      boost::shared_ptr<DataAtGaussPoints>(new DataAtGaussPoints());
  fieldValuePtr = boost::shared_ptr<VectorDouble>(previousUpdate,
                                                  &previousUpdate->fieldValue);
  fieldGradPtr = boost::shared_ptr<MatrixDouble>(previousUpdate,
                                                 &previousUpdate->fieldGrad);
  fieldDotPtr = boost::shared_ptr<VectorDouble>(previousUpdate,
                                                &previousUpdate->fieldDot);
}

MoFEMErrorCode HeatEquation::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HeatEquation::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR simpleInterface->addDomainField(domainField, H1,
                                         AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simpleInterface->addBoundaryField(domainField, H1,
                                           AINSWORTH_LEGENDRE_BASE, 1);

  int oRder = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &oRder, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder(domainField, oRder);

  CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HeatEquation::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto stiff_tangent_rule = [](int, int, int p) -> int { return 2 * p; };
  auto stiff_function_rule = [](int, int, int p) -> int { return 2 * p; };
  stiffTangentLhsMatrixPipeline->getRuleHook = stiff_tangent_rule;
  stiffFunctionRhsVectorPipeline->getRuleHook = stiff_function_rule;

  auto boundary_lhs_rule = [](int, int, int p) -> int { return 2 * p; };
  auto boundary_rhs_rule = [](int, int, int p) -> int { return 2 * p; };
  boundaryLhsMatrixPipeline->getRuleHook = boundary_lhs_rule;
  boundaryRhsVectorPipeline->getRuleHook = boundary_rhs_rule;

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HeatEquation::initialCondition() {
  MoFEMFunctionBegin;

  // Get surface entities form blockset, set initial values in those
  // blocksets. To keep it simple, it is assumed that inital values are on
  // blockset 1
  if (mField.getInterface<MeshsetsManager>()->checkMeshset(1, BLOCKSET)) {
    Range inner_surface;
    CHKERR mField.getInterface<MeshsetsManager>()->getEntitiesByDimension(
        1, BLOCKSET, 2, inner_surface, true);
    if (!inner_surface.empty()) {
      Range inner_surface_verts;
      CHKERR mField.get_moab().get_connectivity(inner_surface,
                                                inner_surface_verts, false);
      CHKERR mField.getInterface<FieldBlas>()->setField(
          initU, MBVERTEX, inner_surface_verts, domainField);
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HeatEquation::boundaryCondition() {
  MoFEMFunctionBegin;

  auto get_ents_on_mesh_skin = [&]() {
    Range boundary_entities;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      std::string entity_name = it->getName();
      if (entity_name.compare(0, 18, "BOUNDARY_CONDITION") == 0) {
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
                                                   boundary_entities, true);
      }
    }
    // Add vertices
    Range boundary_vertices;
    CHKERR mField.get_moab().get_connectivity(boundary_entities,
                                              boundary_vertices, true);
    boundary_entities.merge(boundary_vertices);
    // cerr << boundary_entities;
    boundaryEntitiesForFieldsplit = boundary_entities;

    // Since Dirichlet b.c. are essential boundary conditions, remove DOFs from
    // the problem.
    // CHKERR mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
    //     simpleInterface->getProblemName(), domainField, boundary_entities);

    return boundary_entities;
  };

  auto mark_boundary_dofs = [&](Range &&skin_edges) {
    auto problem_manager = mField.getInterface<ProblemsManager>();
    auto marker_ptr = boost::make_shared<std::vector<bool>>();
    problem_manager->markDofs(simpleInterface->getProblemName(), ROW,
                              skin_edges, *marker_ptr);
    return marker_ptr;
  };

  // Get global local vector of marked DOFs. Is global, since is set for all
  // DOFs on processor. Is local since only DOFs on processor are in the
  // vector. To access DOFs use local indices.
  boundaryMarker = mark_boundary_dofs(get_ents_on_mesh_skin());

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HeatEquation::assembleSystem() {
  MoFEMFunctionBegin;

  { // Push operators to the Pipeline that is responsible for calculating the
    // Tangent (Jacobian) of function (LHS) of the stiff part

    // Add default operators to calculate inverse of Jacobian (needed for
    // implementation of 2D problem but not 3D ones)
    stiffTangentLhsMatrixPipeline->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(invJac));
    stiffTangentLhsMatrixPipeline->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(invJac));

    // Push operators for Jacobian of function (RHS) of the stiff part
    stiffTangentLhsMatrixPipeline->getOpPtrVector().push_back(
        new OpStiffTangentLHS(domainField, domainField, previousUpdate,
                              boundaryMarker));
  }

  { // Push operators to the Pipeline that is responsible for calculating the
    // function (RHS) of the stiff part

    // Add default operators to calculate inverse of Jacobian (needed for
    // implementation of 2D problem but not 3D ones)
    stiffFunctionRhsVectorPipeline->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(invJac));
    stiffFunctionRhsVectorPipeline->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(invJac));

    // Add default operator to calculate field values at integration points
    stiffFunctionRhsVectorPipeline->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues(domainField, fieldValuePtr));
    // Add default operator to calculate field gradient at integration points
    stiffFunctionRhsVectorPipeline->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<2>(domainField, fieldGradPtr));
    // Add default operator to calculate time derivative of field at
    // integration points
    stiffFunctionRhsVectorPipeline->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValuesDot(domainField, fieldDotPtr));

    stiffFunctionRhsVectorPipeline->getOpPtrVector().push_back(
        new OpStiffFunctionRhs(domainField, sourceTermFunction, previousUpdate,
                               boundaryMarker));
  }

  { // Push operators to the Pipeline that is responsible for calculating the
    // boundary condition

    boundaryLhsMatrixPipeline->getOpPtrVector().push_back(
        new OpBoundaryLhs(domainField, domainField));

    // Add default operator to calculate field values at integration points
    boundaryRhsVectorPipeline->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues(domainField, fieldValuePtr));
    boundaryRhsVectorPipeline->getOpPtrVector().push_back(
        new OpBoundaryRhs(domainField, boundaryFunction, previousUpdate));
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HeatEquation::solveSystem() {
  MoFEMFunctionBegin;

  // Create and set Time Stepping solver
  tsSolver = createTS(mField.get_comm());
  // Use fully implicit type (backward Euler) for Heat Equation
  CHKERR TSSetType(tsSolver, TSBEULER);
  // CHKERR TSSetType(tsSolver, TSARKIMEX);
  // CHKERR TSARKIMEXSetType(tsSolver, TSARKIMEXA2);

  // get Discrete Manager (SmartPetscObj)
  dM = simpleInterface->getDM();

  boost::shared_ptr<ForcesAndSourcesCore> null;

  // Add element to calculate Jacobian of function dF/du (LHS) of stiff part
  CHKERR DMMoFEMTSSetIJacobian(dM, simpleInterface->getDomainFEName(),
                               stiffTangentLhsMatrixPipeline, null, null);
  // and boundary term (LHS)
  CHKERR DMMoFEMTSSetIJacobian(dM, simpleInterface->getBoundaryFEName(),
                               boundaryLhsMatrixPipeline, null, null);

  // Add element to calculate function F (RHS) of stiff part
  CHKERR DMMoFEMTSSetIFunction(dM, simpleInterface->getDomainFEName(),
                               stiffFunctionRhsVectorPipeline, null, null);
  // and boundary term (RHS)
  CHKERR DMMoFEMTSSetIFunction(dM, simpleInterface->getBoundaryFEName(),
                               boundaryRhsVectorPipeline, null, null);

  // Add element to calculate function G (RHS) of slow (nonlinear) part
  // Note: G(t,y) = 0 in the heat equation with fully implicit scheme
  // CHKERR DMMoFEMTSSetRHSFunction(dM, simpleInterface->getDomainFEName(),
  //                                vol_ele_slow_rhs, null, null);

  { // Set output of the results

    // Create element for post-processing
    postProcFace = boost::shared_ptr<PostProcFaceOnRefinedMesh>(
        new PostProcFaceOnRefinedMesh(mField));
    // Generate post-processing mesh
    postProcFace->generateReferenceElementMesh();
    // Postprocess only field values
    postProcFace->addFieldValuesPostProc(domainField);

    // Add monitor to time solver
    boost::shared_ptr<Monitor> monitor_ptr(new Monitor(dM, postProcFace));
    CHKERR DMMoFEMTSSetMonitor(dM, tsSolver, simpleInterface->getDomainFEName(),
                               monitor_ptr, null, null);
  }

  // Create global solution vector
  SmartPetscObj<Vec> global_solution;
  CHKERR DMCreateGlobalVector_MoFEM(dM, global_solution);

  // Scatter result data on the mesh
  // CHKERR DMoFEMMeshToGlobalVector(dM, global_solution, INSERT_VALUES,
  //                                 SCATTER_REVERSE);
  CHKERR DMoFEMMeshToLocalVector(dM, global_solution, INSERT_VALUES,
                                 SCATTER_FORWARD);

  // Solve problem
  double max_time = 1;
  double max_steps = 1000;
  CHKERR TSSetDM(tsSolver, dM);
  CHKERR TSSetMaxTime(tsSolver, max_time);
  CHKERR TSSetMaxSteps(tsSolver, max_steps);
  CHKERR TSSetSolution(tsSolver, global_solution);
  CHKERR TSSetFromOptions(tsSolver);
  CHKERR TSSolve(tsSolver, global_solution);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HeatEquation::outputResults() {
  MoFEMFunctionBegin;

  // Processes to set output results are integrated in solveSystem()

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HeatEquation::runProgram() {
  MoFEMFunctionBegin;

  readMesh();
  setupProblem();
  setIntegrationRules();
  initialCondition();
  boundaryCondition();
  assembleSystem();
  solveSystem();
  // outputResults();

  MoFEMFunctionReturn(0);
}

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Error handling
  try {
    // Register MoFEM discrete manager in PETSc
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    // Create MOAB instance
    moab::Core mb_instance;              // mesh database
    moab::Interface &moab = mb_instance; // mesh database interface

    // Create MoFEM instance
    MoFEM::Core core(moab);           // finite element database
    MoFEM::Interface &m_field = core; // finite element interface

    // Run the main analysis
    HeatEquation heat_problem(m_field);
    CHKERR heat_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}