/**
 * \file poisson_2d_dis_galerkin.cpp
 * \example poisson_2d_dis_galerkin.cpp
 *
 * Example of implementation for discontinuous Galerkin.
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
#include <poisson_2d_dis_galerkin.hpp>
#include <poisson_2d_homogeneous.hpp>

using namespace MoFEM;
using namespace Poisson2DiscontGalerkinOperators;
// using namespace Poisson2DHomogeneousOperators;

using PostProcFaceEle = PostProcFaceOnRefinedMesh;

static char help[] = "...\n\n";

struct Poisson2DiscontGalerkin {
public:
  Poisson2DiscontGalerkin(MoFEM::Interface &m_field);

  // Declaration of the main function to run analysis
  MoFEMErrorCode runProgram();

private:
  // Declaration of other main functions called in runProgram()
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();

  // MoFEM interfaces
  MoFEM::Interface &mField;
  Simple *simpleInterface;

  // Field name and approximation order
  std::string domainField;
  int oRder;
  double pEnalty;
};

Poisson2DiscontGalerkin::Poisson2DiscontGalerkin(MoFEM::Interface &m_field)
    : domainField("U"), mField(m_field), oRder(4), pEnalty(1e6) {}

//! [Read mesh]
MoFEMErrorCode Poisson2DiscontGalerkin::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Setup problem]
MoFEMErrorCode Poisson2DiscontGalerkin::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &oRder, PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-penalty", &pEnalty,
                               PETSC_NULL);

  MOFEM_LOG("WORLD", Sev::inform) << "Set order: " << oRder;
  MOFEM_LOG("WORLD", Sev::inform) << "Set penalty: " << pEnalty;

  CHKERR simpleInterface->addDomainField(domainField, L2,
                                         AINSWORTH_LEGENDRE_BASE, 1);
  simpleInterface->getAddSkeletonFE() = true;
  simpleInterface->getAddBoundaryFE() = true;
  CHKERR simpleInterface->setFieldOrder(domainField, oRder);
  CHKERR simpleInterface->setUp();


  MoFEMFunctionReturn(0);
}
//! [Setup problem]

//! [Boundary condition]
MoFEMErrorCode Poisson2DiscontGalerkin::boundaryCondition() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}

//! [Boundary condition]

//! [Assemble system]
MoFEMErrorCode Poisson2DiscontGalerkin::assembleSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto side_fe_lhs = boost::make_shared<FaceSide>(mField);

  auto add_base_ops = [&](auto &pipeline) {
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
    pipeline.push_back(new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline.push_back(new OpSetInvJacL2ForFace(inv_jac_ptr));
  };

  add_base_ops(pipeline_mng->getOpDomainLhsPipeline());

  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new Poisson2DHomogeneousOperators::OpDomainLhsMatrixK(domainField,
                                                          domainField));

  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new Poisson2DHomogeneousOperators::OpDomainRhsVectorF(domainField));

  // Push operators to the Pipeline for Skeleton
  add_base_ops(side_fe_lhs->getOpPtrVector());
  side_fe_lhs->getOpPtrVector().push_back(
      new OpCalculateSideData(domainField, domainField));

  // Push operators to the Pipeline for Skeleton
  pipeline_mng->getOpSkeletonLhsPipeline().push_back(
      new OpDomainLhsPenalty(side_fe_lhs, pEnalty));

  // Push operators to the Pipeline for Boundary
  pipeline_mng->getOpBoundaryLhsPipeline().push_back(
      new OpL2BoundaryLhs(side_fe_lhs, pEnalty));

  MoFEMFunctionReturn(0);
}
//! [Assemble system]

//! [Set integration rules]
MoFEMErrorCode Poisson2DiscontGalerkin::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto rule_lhs = [](int, int, int p) -> int { return 2 * (p - 1); };
  auto rule_rhs = [](int, int, int p) -> int { return p; };

  auto rule_2 = [this](int, int, int) { return 2 * oRder; };

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(rule_lhs);
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule_rhs);

  CHKERR pipeline_mng->setSkeletonLhsIntegrationRule(rule_2);
  CHKERR pipeline_mng->setSkeletonRhsIntegrationRule(rule_2);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(rule_2);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(rule_2);

  MoFEMFunctionReturn(0);
}
//! [Set integration rules]

//! [Solve system]
MoFEMErrorCode Poisson2DiscontGalerkin::solveSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto ksp_solver = pipeline_mng->createKSP();
  CHKERR KSPSetFromOptions(ksp_solver);
  CHKERR KSPSetUp(ksp_solver);

  // Create RHS and solution vectors
  auto dm = simpleInterface->getDM();
  auto F = smartCreateDMVector(dm);
  auto D = smartVectorDuplicate(F);

  CHKERR KSPSolve(ksp_solver, F, D);

  // Scatter result data on the mesh
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}
//! [Solve system]

//! [Output results]
MoFEMErrorCode Poisson2DiscontGalerkin::outputResults() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getSkeletonRhsFE().reset();
  pipeline_mng->getSkeletonLhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();
  pipeline_mng->getBoundaryLhsFE().reset();

  auto post_proc_fe = boost::make_shared<PostProcFaceEle>(mField);
  post_proc_fe->generateReferenceElementMesh();
  post_proc_fe->addFieldValuesPostProc(domainField);
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_result.h5m");

  MoFEMFunctionReturn(0);
}
//! [Output results]

//! [Run program]
MoFEMErrorCode Poisson2DiscontGalerkin::runProgram() {
  MoFEMFunctionBegin;

  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR setIntegrationRules();
  CHKERR solveSystem();
  CHKERR outputResults();

  MoFEMFunctionReturn(0);
}
//! [Run program]

//! [Main]
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
    Poisson2DiscontGalerkin poisson_problem(m_field);
    CHKERR poisson_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}
//! [Main]