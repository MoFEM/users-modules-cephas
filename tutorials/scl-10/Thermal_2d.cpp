#include <MoFEM.hpp>
#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <Thermal_2d.hpp>

using namespace MoFEM;
using namespace Thermal2DOperators;

using DomainEle = FaceElementForcesAndSourcesCoreBase;
using DomainEleOp = DomainEle::UserDataOperator;

using EdgeElem = EdgeElementForcesAndSourcesCoreBase;
using EdgeEleOp = EdgeElem::UserDataOperator;

using OpDomainGradGrad = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpGradGrad<1, 1, 2>;
using OpBoundaryMass = FormsIntegrators<EdgeEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, 1>;
using OpBoundarySource = FormsIntegrators<EdgeEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;
using OpDomainMass = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, 1>;
static char help[] = "...\n\n";

struct Poisson2DNonhomogeneous {
public:
  Poisson2DNonhomogeneous(MoFEM::Interface &m_field);

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
    // return 10 * sin(x * 10.) * cos(y * 10.);
    // return -exp(-100. * (pow(x,2) + pow(y,2))) *
    //        (400. * M_PI *
    //             (x * cos(M_PI * y) * sin(M_PI * x) +
    //              y * cos(M_PI * x) * sin(M_PI * y)) +
    //         2. * (20000. * (pow(x,2) + pow(y,2)) - 200. - pow(M_PI,2)) *
    //             cos(M_PI * x) * cos(M_PI * y));
    return 0;
  }
  // Function to calculate the Boundary term
  static double boundaryFunction(const double x, const double y,
                                 const double z) {
    //return sin(x * 10.) * cos(y * 10.);
    return 1;
  }

  // Function to calculate the Boundary term
  static double boundaryFunction2(const double x, const double y,
                                 const double z) {
    //return sin(x * 10.) * cos(y * 10.);
    return 2;
  }
    // Function to calculate the Boundary term
  static double boundaryFlux(const double x, const double y,
                                 const double z) {
    //return sin(x * 10.) * cos(y * 10.);
    return 1;
  }
  // Main interfaces
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

  // Object to mark boundary entities for the assembling of domain elements
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker_1;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker_2;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker_3;
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

Poisson2DNonhomogeneous::Poisson2DNonhomogeneous(MoFEM::Interface &m_field)
    : domainField("TEMP"), mField(m_field), mpiComm(mField.get_comm()),
      mpiRank(mField.get_comm_rank()) {
  domainPipelineLhs = boost::shared_ptr<FaceEle>(new FaceEle(mField));
  domainPipelineRhs = boost::shared_ptr<FaceEle>(new FaceEle(mField));
  boundaryPipelineLhs = boost::shared_ptr<EdgeEle>(new EdgeEle(mField));
  boundaryPipelineRhs = boost::shared_ptr<EdgeEle>(new EdgeEle(mField));
}

MoFEMErrorCode Poisson2DNonhomogeneous::runProgram() {
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
  
  // BC Bottom Boundary
    auto get_ents_on_mesh_skin_1 = [&]() {
    Range boundary_entities;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      std::string entity_name = it->getName();
      if (entity_name.compare(0, 20, "BOUNDARY_CONDITION_1") == 0) {
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

  // BC Top Boundary
    auto get_ents_on_mesh_skin_2 = [&]() {
    Range boundary_entities;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      std::string entity_name = it->getName();
      if (entity_name.compare(0, 20, "BOUNDARY_CONDITION_2") == 0) {
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

  // BC for DOMAIN
    auto get_ents_on_mesh_skin = [&]() {
    Range boundary_entities;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      std::string entity_name = it->getName();
      Range boundary_entities_loop;
      if (entity_name.compare(0, 20, "BOUNDARY_CONDITION_1") == 0) {
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
                                                   boundary_entities_loop, true);
        boundary_entities.merge(boundary_entities_loop);                                           
      }
      if (entity_name.compare(0, 20, "BOUNDARY_CONDITION_2") == 0) {
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
                                                   boundary_entities_loop, true);
        boundary_entities.merge(boundary_entities_loop);                                             
      }
       if (entity_name.compare(0, 6, "FLUX_1") == 0) {
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
                                                   boundary_entities_loop, true);
        boundary_entities.merge(boundary_entities_loop);  
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
  //

    // BC Bottom Boundary
    auto get_ents_on_mesh_skin_3 = [&]() {
    Range boundary_entities;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      std::string entity_name = it->getName();
      if (entity_name.compare(0, 6, "FLUX_1") == 0) {
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
  boundaryMarker_1 = mark_boundary_dofs(get_ents_on_mesh_skin_1());
  boundaryMarker_2 = mark_boundary_dofs(get_ents_on_mesh_skin_2());
  boundaryMarker_3 = mark_boundary_dofs(get_ents_on_mesh_skin_3());
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson2DNonhomogeneous::assembleSystem() {
  MoFEMFunctionBegin;
  // make this an input
  double D = 1;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-D", &D, PETSC_NULL);
  auto D_mat = [D](const double, const double, const double) { return D; };
  auto q_unit = [](const double, const double, const double) { return 1; };
  { // Push operators to the Pipeline that is responsible for calculating LHS of
    // domain elements
    // Calculate the inverse Jacobian for both Domain and Bonduary
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(invJac));
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(invJac));

    // Push back the LHS  for conduction
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, true, boundaryMarker));
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpDomainGradGrad(domainField, domainField, D_mat));   
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpUnSetBc(domainField));
    // Push back the LHS for the definition of Neuman bcs
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, false, boundaryMarker_3));
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpDomainGradGrad(domainField, domainField, q_unit));   
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpUnSetBc(domainField));        
  }

  { // Push operators to the Pipeline that is responsible for calculating RHS of the source
    domainPipelineRhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, true, boundaryMarker));
    domainPipelineRhs->getOpPtrVector().push_back(
        new OpDomainRhs(domainField, sourceTermFunction));
    domainPipelineRhs->getOpPtrVector().push_back(
        new OpUnSetBc(domainField));
  }

  { // Push operators in boundary pipeline LHS for Dirichelet
    boundaryPipelineLhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, false, boundaryMarker_1));
    boundaryPipelineLhs->getOpPtrVector().push_back(
        new OpBoundaryLhs(domainField, domainField));
    boundaryPipelineLhs->getOpPtrVector().push_back(new OpUnSetBc(domainField));
  }

  { // Push operators in boundary pipeline RHS for Dirichelet
    boundaryPipelineRhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, false, boundaryMarker_1));
    boundaryPipelineRhs->getOpPtrVector().push_back(
        new OpBoundaryRhs(domainField, boundaryFunction));
    boundaryPipelineRhs->getOpPtrVector().push_back(new OpUnSetBc(domainField));
  }

  { // Push operators in boundary pipeline LHS for Dirichelet
    boundaryPipelineLhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, false, boundaryMarker_2));
    boundaryPipelineLhs->getOpPtrVector().push_back(
        new OpBoundaryLhs(domainField, domainField));
    boundaryPipelineLhs->getOpPtrVector().push_back(new OpUnSetBc(domainField));
  }

    { // Push operators in boundary pipeline RHS for Dirichelet
    boundaryPipelineRhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, false, boundaryMarker_2));
    boundaryPipelineRhs->getOpPtrVector().push_back(
        new OpBoundaryRhs(domainField, boundaryFunction2));
    boundaryPipelineRhs->getOpPtrVector().push_back(new OpUnSetBc(domainField));
  }

    { // Push operators to the Pipeline that is responsible for calculating RHS of
    // boundary elements
    boundaryPipelineRhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, false, boundaryMarker_3));
    boundaryPipelineRhs->getOpPtrVector().push_back(
        new OpBoundarySource(domainField, boundaryFlux));
    boundaryPipelineRhs->getOpPtrVector().push_back(new OpUnSetBc(domainField));
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
    MoFEM::Core core(moab);           // finite element database
    MoFEM::Interface &m_field = core; // finite element interface

    // Run the main analysis
    Poisson2DNonhomogeneous poisson_problem(m_field);
    CHKERR poisson_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}