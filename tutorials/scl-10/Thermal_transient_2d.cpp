#include <MoFEM.hpp>
#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <Thermal_2d.hpp>

using namespace MoFEM;
using namespace Thermal2DOperators;

using DomainEle = FaceElementForcesAndSourcesCoreBase;
using DomainEleOp = DomainEle::UserDataOperator;
using OpDomainSource = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;
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
inline double sqr(double x) { return x * x; }
struct Thermal2D {
public:
  Thermal2D(MoFEM::Interface &m_field);

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
    // return -exp(-100. * (sqr(x) + sqr(y))) *
    //        (400. * M_PI *
    //             (x * cos(M_PI * y) * sin(M_PI * x) +
    //              y * cos(M_PI * x) * sin(M_PI * y)) +
    //         2. * (20000. * (sqr(x) + sqr(y)) - 200. - sqr(M_PI)) *
    //             cos(M_PI * x) * cos(M_PI * y)); // this with D_mat = 1 return exp(-100. * (sqr(x) + sqr(y)))
    return 0;
  }
  // Function to impose Dirichelet boundary condition (Temperature)
  static double boundaryFunction(const double x, const double y,
                                 const double z) {
    return 1; 
  }

    // Function to impose Neuman boundary condition (Heat Flux)
  static double boundaryFlux(const double x, const double y,
                                 const double z) {
    
    return 5;
  }
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

Thermal2D::Thermal2D(MoFEM::Interface &m_field)
    : domainField("TEMP"), mField(m_field), mpiComm(mField.get_comm()),
      mpiRank(mField.get_comm_rank()) {
  domainPipelineLhs = boost::shared_ptr<FaceEle>(new FaceEle(mField));
  domainPipelineRhs = boost::shared_ptr<FaceEle>(new FaceEle(mField));
  boundaryPipelineLhs = boost::shared_ptr<EdgeEle>(new EdgeEle(mField));
  boundaryPipelineRhs = boost::shared_ptr<EdgeEle>(new EdgeEle(mField));
}

MoFEMErrorCode Thermal2D::runProgram() {
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

MoFEMErrorCode Thermal2D::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Thermal2D::setupProblem() {
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

MoFEMErrorCode Thermal2D::setIntegrationRules() {
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

MoFEMErrorCode Thermal2D::boundaryCondition() {
  MoFEMFunctionBegin;
  
  // BC Boundary
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

  //BC for DOMAIN
    auto get_ents_on_mesh_skin_2 = [&]() {
    Range boundary_entities;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      std::string entity_name = it->getName();
      Range boundary_entities_loop;
      if (entity_name.compare(0, 20, "BOUNDARY_CONDITION_1") == 0) {
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
  boundaryMarker_1 = mark_boundary_dofs(get_ents_on_mesh_skin_1());
  boundaryMarker_2 = mark_boundary_dofs(get_ents_on_mesh_skin_2());
  boundaryMarker_3 = mark_boundary_dofs(get_ents_on_mesh_skin_3());
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Thermal2D::assembleSystem() {
  MoFEMFunctionBegin;
  // Conductivity of the material
  double D = 1;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-D", &D, PETSC_NULL);
  // Density of the material
  double rho = 1;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-rho", &D, PETSC_NULL);
  // Heat capacity of the material
  double Cv = 1;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-Cv", &D, PETSC_NULL);

  auto D_mat = [D](const double, const double, const double) { return D; };
  auto transient_coeff = [rho](const double, const double, const double) { return rho; };
  auto heat_capacity = [Cv](const double, const double, const double) { return Cv; };
  auto q_unit = [](const double, const double, const double) { return 1; };
  { // Push operators to the Pipeline that is responsible for calculating LHS of
    // domain elements
    // Calculate the inverse Jacobian for both Domain and Boundary
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(invJac));
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(invJac));

    // Push back the LHS for steady conduction
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, true, boundaryMarker_2));
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpDomainGradGrad(domainField, domainField, D_mat));   
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpUnSetBc(domainField));
    // Push back the LHS for transient conduction   
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, true, boundaryMarker_2));
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpDomainMass(domainField, domainField, transient_coeff));   
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpUnSetBc(domainField));     
    //Push back the LHS for the definition of Neuman bcs
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, false, boundaryMarker_3));
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpDomainGradGrad(domainField, domainField, D_mat));   
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpUnSetBc(domainField));        
  }

  { // Push operators to the Pipeline that is responsible for calculating RHS of the source
    domainPipelineRhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, true, boundaryMarker_2));
    domainPipelineRhs->getOpPtrVector().push_back(
        new OpDomainSource(domainField, sourceTermFunction));
    domainPipelineRhs->getOpPtrVector().push_back(
        new OpUnSetBc(domainField));
  }

  { // Push operators in boundary pipeline LHS for Dirichelet
    boundaryPipelineLhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, false, boundaryMarker_1));
    boundaryPipelineLhs->getOpPtrVector().push_back(
        new OpBoundaryMass(domainField, domainField, q_unit));
    boundaryPipelineLhs->getOpPtrVector().push_back(new OpUnSetBc(domainField));
  }

  { // Push operators in boundary pipeline RHS for Dirichelet
    boundaryPipelineRhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, false, boundaryMarker_1));
    boundaryPipelineRhs->getOpPtrVector().push_back(
        new OpBoundarySource(domainField, boundaryFunction));
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

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Thermal2D::solveSystem() {
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

  // // Add element to calculate Jacobian of function dF/du (LHS) of stiff part
  // CHKERR DMMoFEMTSSetIJacobian(dM, simpleInterface->getDomainFEName(),
  //                              stiffTangentLhsMatrixPipeline, null, null);
  // // and boundary term (LHS)
  // CHKERR DMMoFEMTSSetIJacobian(dM, simpleInterface->getBoundaryFEName(),
  //                              boundaryLhsMatrixPipeline, null, null);

  // // Add element to calculate function F (RHS) of stiff part
  // CHKERR DMMoFEMTSSetIFunction(dM, simpleInterface->getDomainFEName(),
  //                              stiffFunctionRhsVectorPipeline, null, null);
  // // and boundary term (RHS)
  // CHKERR DMMoFEMTSSetIFunction(dM, simpleInterface->getBoundaryFEName(),
  //                              boundaryRhsVectorPipeline, null, null);

  // Add element to calculate function G (RHS) of slow (nonlinear) part
  // Note: G(t,y) = 0 in the heat equation with fully implicit scheme
  // CHKERR DMMoFEMTSSetRHSFunction(dM, simpleInterface->getDomainFEName(),
  //                                vol_ele_slow_rhs, null, null);

  { // Set output of the results

    // Create element for post-processing
    CHKERR boost::static_pointer_cast<PostProcFaceOnRefinedMesh>(postProc)
        ->generateReferenceElementMesh();
    CHKERR boost::static_pointer_cast<PostProcFaceOnRefinedMesh>(postProc)
        ->addFieldValuesPostProc(domainField);

    // Add monitor to time solver
    boost::shared_ptr<Monitor> monitor_ptr(new Monitor(dM, postProc));
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
  // double max_time = 1;
  // double max_steps = 1000;
  CHKERR TSSetDM(tsSolver, dM);
  // CHKERR TSSetMaxTime(tsSolver, max_time);
  // CHKERR TSSetMaxSteps(tsSolver, max_steps);
  CHKERR TSSetSolution(tsSolver, global_solution);
  CHKERR TSSetFromOptions(tsSolver);
  CHKERR TSSolve(tsSolver, global_solution);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Thermal2D::outputResults() {
  MoFEMFunctionBegin;

  // Processes to set output results are integrated in solveSystem()

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
    Thermal2D poisson_problem(m_field);
    CHKERR poisson_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}