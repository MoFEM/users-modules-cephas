#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <nonlinear_poisson_2d.hpp>

using namespace MoFEM;
using namespace NonlinearPoissonOps;

static char help[] = "...\n\n";

struct NonlinearPoisson {
public:
  NonlinearPoisson(MoFEM::Interface &m_field);

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
  Simple *simpleInterface;

  // mpi parallel communicator
  MPI_Comm mpiComm;
  // Number of processors
  const int mpiRank;

  // Discrete Manager and nonliner SNES solver using SmartPetscObj
  SmartPetscObj<DM> dM;
  SmartPetscObj<SNES> snesSolver;

  // Field name and approximation order
  std::string domainField;
  int order;

  // Object to mark boundary entities for the assembling of domain elements
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;

  // MoFEM working Pipelines for LHS and RHS of domain and boundary
  boost::shared_ptr<FaceEle> domainTangentMatrixPipeline;
  boost::shared_ptr<FaceEle> domainResidualVectorPipeline;
  boost::shared_ptr<EdgeEle> boundaryTangentMatrixPipeline;
  boost::shared_ptr<EdgeEle> boundaryResidualVectorPipeline;

  // Objects needed for solution updates in Newton's method
  boost::shared_ptr<DataAtGaussPoints> previousUpdate;
  boost::shared_ptr<VectorDouble> fieldValuePtr;
  boost::shared_ptr<MatrixDouble> fieldGradPtr;

  // Object needed for postprocessing
  boost::shared_ptr<FaceEle> postProc;

  // Boundary entities marked for fieldsplit (block) solver - optional
  Range boundaryEntitiesForFieldsplit;
};

NonlinearPoisson::NonlinearPoisson(MoFEM::Interface &m_field)
    : domainField("U"), mField(m_field), mpiComm(mField.get_comm()),
      mpiRank(mField.get_comm_rank()) {
  domainTangentMatrixPipeline = boost::shared_ptr<FaceEle>(new FaceEle(mField));
  domainResidualVectorPipeline =
      boost::shared_ptr<FaceEle>(new FaceEle(mField));
  boundaryTangentMatrixPipeline =
      boost::shared_ptr<EdgeEle>(new EdgeEle(mField));
  boundaryResidualVectorPipeline =
      boost::shared_ptr<EdgeEle>(new EdgeEle(mField));

  previousUpdate =
      boost::shared_ptr<DataAtGaussPoints>(new DataAtGaussPoints());
  fieldValuePtr = boost::shared_ptr<VectorDouble>(previousUpdate,
                                                  &previousUpdate->fieldValue);
  fieldGradPtr = boost::shared_ptr<MatrixDouble>(previousUpdate,
                                                 &previousUpdate->fieldGrad);
}

MoFEMErrorCode NonlinearPoisson::runProgram() {
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

MoFEMErrorCode NonlinearPoisson::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NonlinearPoisson::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR simpleInterface->addDomainField(domainField, H1,
                                         AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  CHKERR simpleInterface->addBoundaryField(domainField, H1,
                                           AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);

  int order = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder(domainField, order);

  CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NonlinearPoisson::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto domain_rule_lhs = [](int, int, int p) -> int { return 2 * (p + 1); };
  auto domain_rule_rhs = [](int, int, int p) -> int { return 2 * (p + 1); };
  domainTangentMatrixPipeline->getRuleHook = domain_rule_lhs;
  domainResidualVectorPipeline->getRuleHook = domain_rule_rhs;

  auto boundary_rule_lhs = [](int, int, int p) -> int { return 2 * p + 1; };
  auto boundary_rule_rhs = [](int, int, int p) -> int { return 2 * p + 1; };
  boundaryTangentMatrixPipeline->getRuleHook = boundary_rule_lhs;
  boundaryResidualVectorPipeline->getRuleHook = boundary_rule_rhs;

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NonlinearPoisson::boundaryCondition() {
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
    auto marker_ptr = boost::make_shared<std::vector<unsigned char>>();
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

MoFEMErrorCode NonlinearPoisson::assembleSystem() {
  MoFEMFunctionBegin;

  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

  { // Push operators to the Pipeline that is responsible for calculating the
    // domain tangent matrix (LHS)

    // Add default operators to calculate inverse of Jacobian (needed for
    // implementation of 2D problem but not 3D ones)
    domainTangentMatrixPipeline->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(inv_jac_ptr));
    domainTangentMatrixPipeline->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(inv_jac_ptr));

    // Add default operator to calculate field values at integration points
    domainTangentMatrixPipeline->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues(domainField, fieldValuePtr));
    // Add default operator to calculate field gradient at integration points
    domainTangentMatrixPipeline->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<2>(domainField, fieldGradPtr));

    // Push operators for domain tangent matrix (LHS)
    domainTangentMatrixPipeline->getOpPtrVector().push_back(
        new OpDomainTangentMatrix(domainField, domainField, previousUpdate,
                                  boundaryMarker));
  }

  { // Push operators to the Pipeline that is responsible for calculating the
    // domain residual vector (RHS)

    // Add default operators to calculate inverse of Jacobian (needed for
    // implementation of 2D problem but not 3D ones)
    domainResidualVectorPipeline->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(inv_jac_ptr));
    domainResidualVectorPipeline->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(inv_jac_ptr));

    // Add default operator to calculate field values at integration points
    domainResidualVectorPipeline->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues(domainField, fieldValuePtr));
    // Add default operator to calculate field gradient at integration points
    domainResidualVectorPipeline->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<2>(domainField, fieldGradPtr));

    domainResidualVectorPipeline->getOpPtrVector().push_back(
        new OpDomainResidualVector(domainField, sourceTermFunction,
                                   previousUpdate, boundaryMarker));
  }

  { // Push operators to the Pipeline that is responsible for calculating the
    // boundary tangent matrix (LHS)

    boundaryTangentMatrixPipeline->getOpPtrVector().push_back(
        new OpBoundaryTangentMatrix(domainField, domainField));
  }

  { // Push operators to the Pipeline that is responsible for calculating
    // boundary residual vector (RHS)

    // Add default operator to calculate field values at integration points
    boundaryResidualVectorPipeline->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues(domainField, fieldValuePtr));

    boundaryResidualVectorPipeline->getOpPtrVector().push_back(
        new OpBoundaryResidualVector(domainField, boundaryFunction,
                                     previousUpdate));
  }

  // get Discrete Manager (SmartPetscObj)
  dM = simpleInterface->getDM();

  { // Set operators for nonlinear equations solver (SNES) from MoFEM Pipelines

    // Set operators for calculation of LHS and RHS of domain elements
    boost::shared_ptr<FaceEle> null_face;
    CHKERR DMMoFEMSNESSetJacobian(dM, simpleInterface->getDomainFEName(),
                                  domainTangentMatrixPipeline, null_face,
                                  null_face);
    CHKERR DMMoFEMSNESSetFunction(dM, simpleInterface->getDomainFEName(),
                                  domainResidualVectorPipeline, null_face,
                                  null_face);

    // Set operators for calculation of LHS and RHS of boundary elements
    boost::shared_ptr<EdgeEle> null_edge;
    CHKERR DMMoFEMSNESSetJacobian(dM, simpleInterface->getBoundaryFEName(),
                                  boundaryTangentMatrixPipeline, null_edge,
                                  null_edge);
    CHKERR DMMoFEMSNESSetFunction(dM, simpleInterface->getBoundaryFEName(),
                                  boundaryResidualVectorPipeline, null_edge,
                                  null_edge);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NonlinearPoisson::solveSystem() {
  MoFEMFunctionBegin;

  // Create RHS and solution vectors
  SmartPetscObj<Vec> global_rhs, global_solution;
  CHKERR DMCreateGlobalVector_MoFEM(dM, global_rhs);
  global_solution = smartVectorDuplicate(global_rhs);

  // Create nonlinear solver (SNES)
  snesSolver = createSNES(mField.get_comm());
  CHKERR SNESSetFromOptions(snesSolver);

  // Fieldsplit block solver: yes/no
  if (1) {
    KSP ksp_solver;
    CHKERR SNESGetKSP(snesSolver, &ksp_solver);
    PC pc;
    CHKERR KSPGetPC(ksp_solver, &pc);

    PetscBool is_pcfs = PETSC_FALSE;
    PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &is_pcfs);

    // Set up FIELDSPLIT, only when option used -pc_type fieldsplit
    if (is_pcfs == PETSC_TRUE) {
      IS is_boundary;
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

  CHKERR SNESSetDM(snesSolver, dM);
  CHKERR SNESSetUp(snesSolver);

  // Solve the system
  CHKERR SNESSolve(snesSolver, global_rhs, global_solution);
  // VecView(global_rhs, PETSC_VIEWER_STDOUT_SELF);

  // Scatter result data on the mesh
  CHKERR DMoFEMMeshToGlobalVector(dM, global_solution, INSERT_VALUES,
                                  SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NonlinearPoisson::outputResults() {
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
    MoFEM::Core core(moab);           // finite element database
    MoFEM::Interface &m_field = core; // finite element interface

    // Run the main analysis
    NonlinearPoisson poisson_problem(m_field);
    CHKERR poisson_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}