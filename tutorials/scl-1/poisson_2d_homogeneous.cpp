#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <poisson_2d_homogeneous.hpp>

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
  std::string domainField;
  int oRder;
  PetscBool testL2Space;
  double pEnalty;
};

Poisson2DHomogeneous::Poisson2DHomogeneous(MoFEM::Interface &m_field)
    : domainField("U"), mField(m_field), testL2Space(PETSC_FALSE),
      pEnalty(1e6) {}

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

  int oRder = 2;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &oRder, PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-penalty", &pEnalty,
                               PETSC_NULL);
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-is_l2", &testL2Space,
                             PETSC_NULL);

  if (!testL2Space) {
    CHKERR simpleInterface->addDomainField(domainField, H1,
                                           AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  } else {
    CHKERR simpleInterface->addDomainField(domainField, L2,
                                           AINSWORTH_LEGENDRE_BASE, 1);
    simpleInterface->getAddSkeletonFE() = true;
    simpleInterface->getAddBoundaryFE() = true;
  }

  CHKERR simpleInterface->setFieldOrder(domainField, oRder);
  CHKERR simpleInterface->setUp();

  auto my_function = [&](boost::shared_ptr<FieldEntity> ent_ptr) {
    MoFEMFunctionBeginHot;
    auto field_data = ent_ptr->getEntFieldData();
    auto ent = ent_ptr->getEnt();
    double coords[3];
    CHKERR mField.get_moab().get_coords(&ent, 1, coords);

    for (auto &v : field_data)
      v = exp(-100. * (coords[1] * coords[1] + coords[0] * coords[0]));

    MoFEMFunctionReturnHot(0);
  };
  // CHKERR mField.getInterface<FieldBlas>()->fieldLambdaOnEntities(my_function,
                                                                //  domainField);

  // print field
  // for (_IT_GET_DOFS_FIELD_BY_NAME_FOR_LOOP_(mField, domainField, dof)) {
  //   if ((*dof)->getEntType() == MBTRI)
  //     cerr << (*dof)->getFieldData() << endl;
  // }

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
  // FIXME: TODO: this might need change for L2 space
  // if (testL2Space) {
  //   Range adj;
  //   CHKERR mField.get_moab().get_adjacencies(boundary_entities, 2, false, adj,
  //                                            moab::Interface::UNION);
  //   cout << adj << endl;
  //   boundary_entities.merge(adj);
  // }

  // Add vertices to boundary entities
  Range boundary_vertices;
  CHKERR mField.get_moab().get_connectivity(boundary_entities,
                                            boundary_vertices, true);
  boundary_entities.merge(boundary_vertices);

  // Remove DOFs as homogeneous boundary condition is used
  // if (!testL2Space)
    CHKERR mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
        simpleInterface->getProblemName(), domainField, boundary_entities);

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
  auto side_fe_rhs = boost::make_shared<FaceSide>(mField);
  auto side_fe_lhs = boost::make_shared<FaceSide>(mField);

  side_fe_rhs->getOpPtrVector().push_back(new OpCalculateSideData(domainField));
  side_fe_lhs->getOpPtrVector().push_back(
      new OpCalculateSideData(domainField, domainField));

  { // Push operators to the Pipeline that is responsible for calculating LHS

    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpCalculateHOJacForFace(jac_ptr));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpSetInvJacH1ForFace(inv_jac_ptr));
    // if (testL2Space)
    //   pipeline_mng->getOpDomainLhsPipeline().push_back(
    //       new OpSetInvJacL2ForFace(inv_jac_ptr));

    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpDomainLhsMatrixK(domainField, domainField));
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpDomainRhsVectorF(domainField));
  }
  if (testL2Space) { // Push operators to the Pipeline that is responsible for
                     // Skeleton
    
    pipeline_mng->getOpSkeletonLhsPipeline().push_back(
        new OpComputeJumpOnSkeleton(domainField, side_fe_lhs));
    pipeline_mng->getOpSkeletonLhsPipeline().push_back(
        new OpDomainLhsPenalty(domainField, domainField, pEnalty));

    pipeline_mng->getOpSkeletonRhsPipeline().push_back(
        new OpComputeJumpOnSkeleton(domainField, side_fe_rhs));
    pipeline_mng->getOpSkeletonRhsPipeline().push_back(
        new OpDomainRhsPenalty(domainField, pEnalty));

    // pipeline_mng->getOpBoundaryRhsPipeline().push_back(new OpBoundaryRhs(
    //     domainField, [&](double, double, double) { return pEnalty; }));
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(
        new OpBoundaryLhs(domainField, domainField, side_fe_lhs, pEnalty));
  }

  MoFEMFunctionReturn(0);
}
//! [Assemble system]

//! [Set integration rules]
MoFEMErrorCode Poisson2DHomogeneous::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto rule_lhs = [](int, int, int p) -> int { return 2 * (p - 1); };
  auto rule_rhs = [](int, int, int p) -> int { return p; };
  auto rule_2 = [](int, int, int){ return 2; };

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(rule_lhs);
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule_rhs);
  if (testL2Space) {
    CHKERR pipeline_mng->setSkeletonLhsIntegrationRule(rule_2);
    CHKERR pipeline_mng->setSkeletonRhsIntegrationRule(rule_2);
    CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(rule_2);
    CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(rule_2);
  }

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
  // CHKERR pipeline_mng->loopFiniteElements();

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
MoFEMErrorCode Poisson2DHomogeneous::runProgram() {
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
    Poisson2DHomogeneous poisson_problem(m_field);
    CHKERR poisson_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}
//! [Main]