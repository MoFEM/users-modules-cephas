#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <poisson_2d_nonhomogeneous.hpp>

using namespace MoFEM;
using namespace Poisson2DNonhomogeneousOperators;

static char help[] = "...\n\n";

struct Poisson2DNonhomogeneous {
public:
  Poisson2DNonhomogeneous(moab::Core &mb_instance, MoFEM::Core &core);

  // Declaration of the main function to run analysis
  MoFEMErrorCode runWholeProgram();

private:
  // Declaration of other main functions called in runWholeProgram()
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();

  // Function to calculate the Source term
  static double sourceTermFunction(const double x, const double y,
                                   const double z) {
    return 200 * sin(x * 10.) * cos(y * 10.);
    // return 1;
  }
  // Function to calculate the Boundary term
  static double boundaryFunction(const double x, const double y,
                                 const double z) {
    return sin(x * 10.) * cos(y * 10.);
    // return 0;
  }

  // Main interfaces
  MoFEM::Interface &mField;
  moab::Interface &mOab;
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

  // Object to mark boundary entities for the assembling of domain elements
  boost::shared_ptr<std::vector<bool>> boundaryMarker;

  // MoFEM working Pipelines for LHS and RHS of domain and boundary
  boost::shared_ptr<FaceEle> domainPipelineLhs;
  boost::shared_ptr<FaceEle> domainPipelineRhs;
  boost::shared_ptr<EdgeEle> boundaryPipelineLhs;
  boost::shared_ptr<EdgeEle> boundaryPipelineRhs;

  // Object needed for postprocessing
  boost::shared_ptr<FaceEle> postProc;

  // Boundary entities marked for fieldsplit (block) solver - optional
  Range boundaryEntitiesForFieldsplit;
};

Poisson2DNonhomogeneous::Poisson2DNonhomogeneous(moab::Core &mb_instance,
                                                 MoFEM::Core &core)
    : domainField("U"), mOab(mb_instance), mField(core),
      mpiComm(mField.get_comm()), mpiRank(mField.get_comm_rank()) {
  domainPipelineLhs = boost::shared_ptr<FaceEle>(new FaceEle(mField));
  domainPipelineRhs = boost::shared_ptr<FaceEle>(new FaceEle(mField));
  boundaryPipelineLhs = boost::shared_ptr<EdgeEle>(new EdgeEle(mField));
  boundaryPipelineRhs = boost::shared_ptr<EdgeEle>(new EdgeEle(mField));
}

MoFEMErrorCode Poisson2DNonhomogeneous::runWholeProgram() {
  MoFEMFunctionBegin;

  readMesh();
  setupProblem();
  setIntegrationRules();
  boundaryCondition();
  assembleSystem();
  solveSystem();
  outputResults();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson2DNonhomogeneous::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson2DNonhomogeneous::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR simpleInterface->addDomainField(domainField, H1,
                                         AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  CHKERR simpleInterface->addBoundaryField(domainField, H1,
                                           AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);

  int oRder = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &oRder, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder(domainField, oRder);

  CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson2DNonhomogeneous::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto domain_rule_lhs = [](int, int, int p) -> int { return 2 * (p - 1); };
  auto domain_rule_rhs = [](int, int, int p) -> int { return 2 * (p - 1); };
  domainPipelineLhs->getRuleHook = domain_rule_lhs;
  domainPipelineRhs->getRuleHook = domain_rule_rhs;

  auto boundary_rule_lhs = [](int, int, int p) -> int { return 2 * p; };
  auto boundary_rule_rhs = [](int, int, int p) -> int { return 2 * p; };
  boundaryPipelineLhs->getRuleHook = boundary_rule_lhs;
  boundaryPipelineRhs->getRuleHook = boundary_rule_rhs;

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson2DNonhomogeneous::boundaryCondition() {
  MoFEMFunctionBegin;

  // Get boundary edges marked in block named "BOUNDARY_CONDITION"
  auto get_ents_on_mesh_skin = [&]() {
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

    // Store entities for fieldsplit (block) solver
    boundaryEntitiesForFieldsplit = boundary_entities;

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

MoFEMErrorCode Poisson2DNonhomogeneous::assembleSystem() {
  MoFEMFunctionBegin;

  { // Push operators to the Pipeline that is responsible for calculating LHS of
    // domain elements
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(invJac));
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(invJac));

    domainPipelineLhs->getOpPtrVector().push_back(
        new OpDomainLhs(domainField, domainField, boundaryMarker));
  }

  { // Push operators to the Pipeline that is responsible for calculating RHS of
    // domain elements
    domainPipelineRhs->getOpPtrVector().push_back(
        new OpDomainRhs(domainField, sourceTermFunction, boundaryMarker));
  }

  { // Push operators to the Pipeline that is responsible for calculating LHS of
    // boundary elements
    boundaryPipelineLhs->getOpPtrVector().push_back(
        new OpBoundaryLhs(domainField, domainField));
  }

  { // Push operators to the Pipeline that is responsible for calculating RHS of
    // boundary elements
    boundaryPipelineRhs->getOpPtrVector().push_back(
        new OpBoundaryRhs(domainField, boundaryFunction));
  }

  // get Discrete Manager (SmartPetscObj)
  dM = simpleInterface->getDM();

  { // Set operators for linear equations solver (KSP) from MoFEM Pipelines

    // Set operators for calculation of LHS and RHS of domain elements
    boost::shared_ptr<FaceEle> null_face;
    CHKERR DMMoFEMKSPSetComputeOperators(dM, simpleInterface->getDomainFEName(),
                                         domainPipelineLhs, null_face,
                                         null_face);
    CHKERR DMMoFEMKSPSetComputeRHS(dM, simpleInterface->getDomainFEName(),
                                   domainPipelineRhs, null_face, null_face);

    // Set operators for calculation of LHS and RHS of domain elements
    boost::shared_ptr<EdgeEle> null_edge;
    CHKERR DMMoFEMKSPSetComputeOperators(
        dM, simpleInterface->getBoundaryFEName(), boundaryPipelineLhs,
        null_edge, null_edge);
    CHKERR DMMoFEMKSPSetComputeRHS(dM, simpleInterface->getBoundaryFEName(),
                                   boundaryPipelineRhs, null_edge, null_edge);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson2DNonhomogeneous::solveSystem() {
  MoFEMFunctionBegin;

  // Create RHS and solution vectors
  SmartPetscObj<Vec> global_rhs, global_solution;
  CHKERR DMCreateGlobalVector_MoFEM(dM, global_rhs);
  global_solution = smartVectorDuplicate(global_rhs);

  // Setup KSP solver
  kspSolver = createKSP(mField.get_comm());
  CHKERR KSPSetFromOptions(kspSolver);

  // Setup fieldsplit (block) solver - optional: yes/no
  if (1) {
    PC pc;
    CHKERR KSPGetPC(kspSolver, &pc);
    PetscBool is_pcfs = PETSC_FALSE;
    PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &is_pcfs);

    // Set up FIELDSPLIT, only when user set -pc_type fieldsplit
    // Identify the index for boundary entities, remaining will be for domain
    // Then split the fields for boundary and domain for solving
    if (is_pcfs == PETSC_TRUE) {
      IS is_domain, is_boundary;
      const MoFEM::Problem *problem_ptr;
      CHKERR DMMoFEMGetProblemPtr(dM, &problem_ptr);
      CHKERR mField.getInterface<ISManager>()->isCreateProblemFieldAndRank(
          problem_ptr->getName(), ROW, domainField, 0, 1, &is_boundary,
          &boundaryEntitiesForFieldsplit);
      // CHKERR ISView(is_boundary, PETSC_VIEWER_STDOUT_SELF);

      CHKERR PCFieldSplitSetIS(pc, NULL, is_boundary);

      CHKERR ISDestroy(&is_boundary);
    }
  }

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

MoFEMErrorCode Poisson2DNonhomogeneous::outputResults() {
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
    MoFEM::Core core(moab); // finite element database
    // MoFEM::Interface &mField = core; // finite element interface

    // Run the main analysis
    Poisson2DNonhomogeneous poisson_problem(mb_instance, core);
    CHKERR poisson_problem.runWholeProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}