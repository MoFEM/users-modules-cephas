/**
 * \file poisson_2d_homogeneous.cpp
 * \example poisson_2d_homogeneous.cpp
 *
 * Solution of poisson equation. Direct implementation of User Data Operators
 * for teaching proposes.
 *
 * \note In practical application we suggest use form integrators to generalise
 * and simplify code. However, here we like to expose user to ways how to
 * implement data operator from scratch.
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

static const int nb_ref_levels =
    1; ///< if larger than zero set n-levels of random mesh refinements with
       ///< hanging nodes

constexpr auto field_name = "U";

#include <BasicFiniteElements.hpp>
#include <poisson_2d_homogeneous.hpp>
#include <random_mesh_refine.hpp>

using namespace MoFEM;
using namespace Poisson2DHomogeneousOperators;

using PostProcFaceEle = PostProcFaceOnRefinedMesh;

static char help[] = "...\n\n";

struct Poisson2DHomogeneous {
public:
  Poisson2DHomogeneous(MoFEM::Interface &m_field);

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
  int oRder;
};

Poisson2DHomogeneous::Poisson2DHomogeneous(MoFEM::Interface &m_field)
    : mField(m_field) {}

//! [Read mesh]
MoFEMErrorCode Poisson2DHomogeneous::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Setup problem]
MoFEMErrorCode Poisson2DHomogeneous::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR simpleInterface->addDomainField(field_name, H1,
                                         AINSWORTH_LEGENDRE_BASE, 1);

  int oRder = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &oRder, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder(field_name, oRder);

  // Refine random elements and create hanging nodes. This is only need if one
  // like to refine mesh.
  if (nb_ref_levels)
    CHKERR random_mesh_refine(mField);

  CHKERR simpleInterface->setUp();

  // Remove hanging nodes
  if(nb_ref_levels)
    CHKERR remove_hanging_dofs(mField);

  MoFEMFunctionReturn(0);
}
//! [Setup problem]

//! [Boundary condition]
MoFEMErrorCode Poisson2DHomogeneous::boundaryCondition() {
  MoFEMFunctionBegin;

  // Get boundary edges marked in block named "BOUNDARY_CONDITION"
  Range boundary_entities;
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
    std::string entity_name = it->getName();
    if (entity_name.compare(0, 18, "BOUNDARY_CONDITION") == 0) {
      CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
                                                 boundary_entities, true);
    }
  }
  // Add vertices to boundary entities
  Range boundary_vertices;
  CHKERR mField.get_moab().get_connectivity(boundary_entities,
                                            boundary_vertices, true);
  boundary_entities.merge(boundary_vertices);

  // Remove DOFs as homogeneous boundary condition is used
  CHKERR mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
      simpleInterface->getProblemName(), field_name, boundary_entities);

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Assemble system]
MoFEMErrorCode Poisson2DHomogeneous::assembleSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

  { // Push operators to the Pipeline that is responsible for calculating LHS

    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpCalculateHOJac<2>(jac_ptr));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpSetHOInvJacToScalarBases<2>(H1, inv_jac_ptr));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpSetHOWeights(det_ptr));

    if (nb_ref_levels) { // This part is advanced. Can be skipped for not
                         // refined meshes with
      // hanging nodes.
      // Force integration on last refinement level, and add to top elements
      // DOFs and based from underlying elements.
      pipeline_mng->getDomainLhsFE()->exeTestHook = test_bit_child;
      set_parent_dofs(mField, pipeline_mng->getDomainLhsFE(),
                      OpFaceEle::OPSPACE, QUIET, Sev::noisy);
      set_parent_dofs(mField, pipeline_mng->getDomainLhsFE(), OpFaceEle::OPROW,
                      QUIET, Sev::noisy);
      set_parent_dofs(mField, pipeline_mng->getDomainLhsFE(), OpFaceEle::OPCOL,
                      QUIET, Sev::noisy);
    }

    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpDomainLhsMatrixK(field_name, field_name));
  }

  { // Push operators to the Pipeline that is responsible for calculating RHS

    if (nb_ref_levels) { // This part is advanced. Can be skipped for not
                         // refined meshes with
      // hanging nodes.
      // Force integration on last refinement level, and add to top elements
      // DOFs and based from underlying elements.
      pipeline_mng->getDomainRhsFE()->exeTestHook = test_bit_child;
      set_parent_dofs(mField, pipeline_mng->getDomainRhsFE(),
                      OpFaceEle::OPSPACE, QUIET, Sev::noisy);
      set_parent_dofs(mField, pipeline_mng->getDomainRhsFE(), OpFaceEle::OPROW,
                      QUIET, Sev::noisy);
    }

    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpDomainRhsVectorF(field_name));
  }

  MoFEMFunctionReturn(0);
}
//! [Assemble system]

//! [Set integration rules]
MoFEMErrorCode Poisson2DHomogeneous::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto rule_lhs = [](int, int, int p) -> int { return 2 * (p - 1); };
  auto rule_rhs = [](int, int, int p) -> int { return p; };

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(rule_lhs);
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule_rhs);

  MoFEMFunctionReturn(0);
}
//! [Set integration rules]

//! [Solve system]
MoFEMErrorCode Poisson2DHomogeneous::solveSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto ksp_solver = pipeline_mng->createKSP();
  CHKERR KSPSetFromOptions(ksp_solver);
  CHKERR KSPSetUp(ksp_solver);

  // Create RHS and solution vectors
  auto dm = simpleInterface->getDM();
  auto F = smartCreateDMVector(dm);
  auto D = smartVectorDuplicate(F);

  // Solve the system
  CHKERR KSPSolve(ksp_solver, F, D);

  // Scatter result data on the mesh
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}
//! [Solve system]

//! [Output results]
MoFEMErrorCode Poisson2DHomogeneous::outputResults() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();

  auto post_proc_fe = boost::make_shared<PostProcFaceEle>(mField);
  post_proc_fe->generateReferenceElementMesh();

  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

  constexpr auto SPACE_DIM = 2; // dimension of problem

  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateHOJac<SPACE_DIM>(jac_ptr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpSetHOInvJacToScalarBases<SPACE_DIM>(H1, inv_jac_ptr));

  if (nb_ref_levels) { // This part is advanced. Can be skipped for not refined
                       // meshes with
    // hanging nodes.
    post_proc_fe->exeTestHook = test_bit_child;
    set_parent_dofs(mField, post_proc_fe, OpFaceEle::OPSPACE, QUIET,
                    Sev::noisy);
    set_parent_dofs(mField, post_proc_fe, OpFaceEle::OPROW, QUIET, Sev::noisy);
  }

  auto u_ptr = boost::make_shared<VectorDouble>();
  auto grad_u_ptr = boost::make_shared<MatrixDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(field_name, u_ptr));


  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateScalarFieldGradient<SPACE_DIM>(field_name, grad_u_ptr));

  using OpPPMap = OpPostProcMap<SPACE_DIM, SPACE_DIM>;

  post_proc_fe->getOpPtrVector().push_back(

      new OpPPMap(post_proc_fe->postProcMesh, post_proc_fe->mapGaussPts,

                  OpPPMap::DataMapVec{{"U", u_ptr}},

                  OpPPMap::DataMapMat{{"GRAD_U", grad_u_ptr}},

                  OpPPMap::DataMapMat{},

                  OpPPMap::DataMapMat{}

                  )

  );

  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_result.h5m");

  MoFEMFunctionReturn(0);
}
//! [Output results]

//! [Run program]
MoFEMErrorCode Poisson2DHomogeneous::runProgram() {
  MoFEMFunctionBegin;

  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();
  CHKERR setIntegrationRules();
  CHKERR assembleSystem();
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
    Poisson2DHomogeneous poisson_problem(m_field);
    CHKERR poisson_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}
//! [Main]