#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <poisson_2d_homogeneous.hpp>

using namespace MoFEM;
using namespace Poisson2DHomogeneousOperators;

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
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();

  // MoFEM interfaces
  MoFEM::Interface &mField;
  Simple *simpleInterface;

  // mpi parallel communicator
  MPI_Comm mpiComm;
  // Number of processors
  const int mpiRank;

  // Discrete Manager and linear KSP solver using SmartPetscObj
  SmartPetscObj<DM> dM;
  SmartPetscObj<KSP> kspSolver;

  // Field name and approximation order
  std::string domainField;
  int oRder;
  MatrixDouble invJac;

  // MoFEM working Pipelines for LHS and RHS
  boost::shared_ptr<FaceEle> pipelineLhs;
  boost::shared_ptr<FaceEle> pipelineRhs;

  // Object needed for postprocessing
  boost::shared_ptr<FaceEle> postProc;
};

Poisson2DHomogeneous::Poisson2DHomogeneous(MoFEM::Interface &m_field)
    : domainField("U"), mField(m_field), mpiComm(mField.get_comm()),
      mpiRank(mField.get_comm_rank()) {
  pipelineLhs = boost::shared_ptr<FaceEle>(new FaceEle(mField));
  pipelineRhs = boost::shared_ptr<FaceEle>(new FaceEle(mField));
}

MoFEMErrorCode Poisson2DHomogeneous::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson2DHomogeneous::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR simpleInterface->addDomainField(domainField, H1,
                                         AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);

  int oRder = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &oRder, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder(domainField, oRder);

  CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson2DHomogeneous::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto rule_lhs = [](int, int, int p) -> int { return 2 * (p - 1); };
  auto rule_rhs = [](int, int, int p) -> int { return p; };
  pipelineLhs->getRuleHook = rule_lhs;
  pipelineRhs->getRuleHook = rule_rhs;

  MoFEMFunctionReturn(0);
}

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
      simpleInterface->getProblemName(), domainField, boundary_entities);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson2DHomogeneous::assembleSystem() {
  MoFEMFunctionBegin;

  { // Push operators to the Pipeline that is responsible for calculating LHS

    pipelineLhs->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(invJac));
    pipelineLhs->getOpPtrVector().push_back(new OpSetInvJacH1ForFace(invJac));

    pipelineLhs->getOpPtrVector().push_back(
        new OpDomainLhsMatrixK(domainField, domainField));
  }

  { // Push operators to the Pipeline that is responsible for calculating LHS

    pipelineRhs->getOpPtrVector().push_back(
        new OpDomainRhsVectorF(domainField));
  }

  // get Discrete Manager (SmartPetscObj)
  dM = simpleInterface->getDM();

  { // Set operators for linear equations solver (KSP) from MoFEM Pipelines

    boost::shared_ptr<FaceEle> null;
    CHKERR DMMoFEMKSPSetComputeOperators(dM, simpleInterface->getDomainFEName(),
                                         pipelineLhs, null, null);
    CHKERR DMMoFEMKSPSetComputeRHS(dM, simpleInterface->getDomainFEName(),
                                   pipelineRhs, null, null);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson2DHomogeneous::solveSystem() {
  MoFEMFunctionBegin;

  // Create RHS and solution vectors
  SmartPetscObj<Vec> global_rhs, global_solution;
  CHKERR DMCreateGlobalVector_MoFEM(dM, global_rhs);
  global_solution = smartVectorDuplicate(global_rhs);

  // Setup KSP solver
  kspSolver = createKSP(mField.get_comm());
  CHKERR KSPSetFromOptions(kspSolver);
  CHKERR KSPSetDM(kspSolver, dM);
  CHKERR KSPSetUp(kspSolver);

  // Solve the system
  CHKERR KSPSolve(kspSolver, global_rhs, global_solution);
  // VecView(global_rhs, PETSC_VIEWER_STDOUT_SELF);

  // Scatter result data on the mesh
  CHKERR DMoFEMMeshToGlobalVector(dM, global_solution, INSERT_VALUES,
                                  SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson2DHomogeneous::outputResults() {
  MoFEMFunctionBegin;

  postProc = boost::shared_ptr<FaceEle>(new PostProcFaceOnRefinedMesh(mField));

  CHKERR boost::static_pointer_cast<PostProcFaceOnRefinedMesh>(postProc)
      ->generateReferenceElementMesh();
  CHKERR boost::static_pointer_cast<PostProcFaceOnRefinedMesh>(postProc)
      ->addFieldValuesPostProc(domainField);

  CHKERR DMoFEMLoopFiniteElements(dM, simpleInterface->getDomainFEName(),
                                  postProc);

  CHKERR boost::static_pointer_cast<PostProcFaceOnRefinedMesh>(postProc)
      ->writeFile("out_result.h5m");

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson2DHomogeneous::runProgram() {
  MoFEMFunctionBegin;

  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR setIntegrationRules();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR outputResults();

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
    Poisson2DHomogeneous poisson_problem(m_field);
    CHKERR poisson_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}