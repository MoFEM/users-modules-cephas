#include <MoFEM.hpp>
#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <Thermal_2d.hpp>

using namespace MoFEM;

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
using OpHdivU = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMixDivTimesScalar<2>;
constexpr int SPACE_DIM = 2; //< Space dimension of problem, mesh

using OpK = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM, 0>;
using OpBodyForce = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 0>;
using OpInternalForce =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
        GAUSS>::OpGradTimesSymTensor<1, SPACE_DIM, SPACE_DIM>;


// These Youngs modulus and poisson ratio should be an input

constexpr double young_modulus = 50e3;
constexpr double poisson_ratio = 0.;
constexpr double coeff_expansion = 1e-5;
constexpr double bulk_modulus_K = young_modulus / (3 * (1 - 2 * poisson_ratio));
constexpr double shear_modulus_G = young_modulus / (2 * (1 + poisson_ratio));

using namespace Thermal2DOperators;

static char help[] = "...\n\n";
inline double sqr(double x) { return x * x; }
struct Thermal_Elasticity2D {
public:
  Thermal_Elasticity2D(MoFEM::Interface &m_field);

  // Declaration of the main function to run analysis
  MoFEMErrorCode runProgram();

private:
  // Declaration of other main functions called in runProgram()
  MoFEMErrorCode readMesh();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();

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
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker_1;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker_2;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker_3;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker_4;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker_5;
  // MoFEM working Pipelines for LHS and RHS of domain and boundary
  boost::shared_ptr<FaceEle> domainPipelineLhs;
  boost::shared_ptr<FaceEle> domainPipelineRhs;
  boost::shared_ptr<EdgeEle> boundaryPipelineLhs;
  boost::shared_ptr<EdgeEle> boundaryPipelineRhs;
  // Elasticity pointers
  boost::shared_ptr<MatrixDouble> matGradPtr;
  boost::shared_ptr<MatrixDouble> matStrainPtr;
  boost::shared_ptr<MatrixDouble> matStressPtr;
  boost::shared_ptr<MatrixDouble> matDPtr;
  boost::shared_ptr<MatrixDouble> thDPtr;
  boost::shared_ptr<MatrixDouble> bodyForceMatPtr;

  Range fluxBoundaryConditions_1;
  Range fluxBoundaryConditions_2;

  // Objects needed for solution updates in Newton's method (dynamics) and for calling also temperature field
  boost::shared_ptr<DataAtGaussPoints> previousUpdate;
  boost::shared_ptr<VectorDouble> fieldValuePtr;
  boost::shared_ptr<MatrixDouble> fieldGradPtr;
  boost::shared_ptr<VectorDouble> fieldDotPtr;

  // Object needed for postprocessing
  boost::shared_ptr<FaceEle> postProc;

  // Boundary entities marked for fieldsplit (block) solver - optional
  Range boundaryEntitiesForFieldsplit;
};

Thermal_Elasticity2D::Thermal_Elasticity2D(MoFEM::Interface &m_field)
    : domainField("TEMP"), mField(m_field), mpiComm(mField.get_comm()),
      mpiRank(mField.get_comm_rank()) {
  domainPipelineLhs = boost::shared_ptr<FaceEle>(new FaceEle(mField));
  domainPipelineRhs = boost::shared_ptr<FaceEle>(new FaceEle(mField));
  boundaryPipelineLhs = boost::shared_ptr<EdgeEle>(new EdgeEle(mField));
  boundaryPipelineRhs = boost::shared_ptr<EdgeEle>(new EdgeEle(mField));

  previousUpdate =
      boost::shared_ptr<DataAtGaussPoints>(new DataAtGaussPoints());
  fieldValuePtr = boost::shared_ptr<VectorDouble>(previousUpdate,
                                                  &previousUpdate->fieldValue);
}

//! [Create common data]
MoFEMErrorCode Thermal_Elasticity2D::createCommonData() {
  MoFEMFunctionBegin;

  // Push operator to get TEMP from Integration Points and pass at pointer 
  domainPipelineLhs->getOpPtrVector().push_back(
    new OpCalculateScalarFieldValues("TEMP", fieldValuePtr));


  auto set_matrial_stiffens = [&]() {
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    FTensor::Index<'l', SPACE_DIM> l;
    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
    MoFEMFunctionBegin;
    constexpr double A =
        (SPACE_DIM == 2) ? 2 * shear_modulus_G /
                               (bulk_modulus_K + (4. / 3.) * shear_modulus_G)
                         : 1;
    auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*matDPtr);
    t_D(i, j, k, l) = 2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.) +
                      A * (bulk_modulus_K - (2. / 3.) * shear_modulus_G) *
                          t_kd(i, j) * t_kd(k, l);
    MoFEMFunctionReturn(0);
  };

  auto set_thermal_strain = [&]() {
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    FTensor::Index<'l', SPACE_DIM> l;

    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
    MoFEMFunctionBegin;                    
    auto alpha = coeff_expansion; 
    constexpr double A =
        (SPACE_DIM == 2) ? 2 * shear_modulus_G /
                               (bulk_modulus_K + (4. / 3.) * shear_modulus_G)
                         : 1;
    auto t_DT = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*thDPtr);
    t_DT(i, j, k, l) = (2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.) +
                      A * (bulk_modulus_K - (2. / 3.) * shear_modulus_G) *
                          t_kd(i, j) * t_kd(k, l)) * alpha;
                          
    MoFEMFunctionReturn(0);
  };

  auto set_body_force = [&]() {
    FTensor::Index<'i', SPACE_DIM> i;
    MoFEMFunctionBegin;
    auto t_force = getFTensor1FromMat<SPACE_DIM, 0>(*bodyForceMatPtr);
    t_force(i) = 0;
    t_force(1) = -1;
    MoFEMFunctionReturn(0);
  };

  matGradPtr = boost::make_shared<MatrixDouble>();
  matStrainPtr = boost::make_shared<MatrixDouble>();
  matStressPtr = boost::make_shared<MatrixDouble>();
  matDPtr = boost::make_shared<MatrixDouble>();
  thDPtr = boost::make_shared<MatrixDouble>();
  bodyForceMatPtr = boost::make_shared<MatrixDouble>();

  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  matDPtr->resize(size_symm * size_symm, 1);
  thDPtr->resize(size_symm * size_symm, 1);
  bodyForceMatPtr->resize(SPACE_DIM, 1);

  CHKERR set_matrial_stiffens();
  CHKERR set_thermal_strain();
  CHKERR set_body_force();

  MoFEMFunctionReturn(0);
}
//! [Create common data]

MoFEMErrorCode Thermal_Elasticity2D::runProgram() {
  MoFEMFunctionBegin;

  readMesh();
  createCommonData();
  setupProblem();
  setIntegrationRules();
  boundaryCondition();
  assembleSystem();
  solveSystem();
  outputResults();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Thermal_Elasticity2D::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Thermal_Elasticity2D::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR simpleInterface->addDomainField(domainField, H1,
                                         AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  CHKERR simpleInterface->addBoundaryField(domainField, H1,
                                           AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  // Add fields for elasticity
  CHKERR simpleInterface->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, SPACE_DIM);
  CHKERR simpleInterface->addBoundaryField("U", H1, AINSWORTH_LEGENDRE_BASE, SPACE_DIM);

  int oRder = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &oRder, PETSC_NULL);

  CHKERR simpleInterface->setFieldOrder("TEMP", oRder-1);

  //Set order for displacement
  CHKERR simpleInterface->setFieldOrder("U", oRder);

  CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Thermal_Elasticity2D::setIntegrationRules() {
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

MoFEMErrorCode Thermal_Elasticity2D::boundaryCondition() {
  MoFEMFunctionBegin;
  // Start Elastic BC
auto fix_disp = [&](const std::string blockset_name) {
    Range fix_ents;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, blockset_name.length(), blockset_name) ==
          0) {
        CHKERR mField.get_moab().get_entities_by_handle(it->meshset, fix_ents,
                                                        true);
      }
    }
    return fix_ents;
  };

  auto remove_ents = [&](const Range &&ents, const int lo, const int hi) {
    auto prb_mng = mField.getInterface<ProblemsManager>();
    auto simple = mField.getInterface<Simple>();
    MoFEMFunctionBegin;
    Range verts;
    CHKERR mField.get_moab().get_connectivity(ents, verts, true);
    verts.merge(ents);
    if (SPACE_DIM == 3) {
      Range adj;
      CHKERR mField.get_moab().get_adjacencies(ents, 1, false, adj,
                                               moab::Interface::UNION);
      verts.merge(adj);
    };
    CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(verts);
    CHKERR prb_mng->removeDofsOnEntities(simple->getProblemName(), "U", verts,
                                         lo, hi);
    MoFEMFunctionReturn(0);
  };

  CHKERR remove_ents(fix_disp("FIX_X"), 0, 0);
  CHKERR remove_ents(fix_disp("FIX_Y"), 1, 1);
  CHKERR remove_ents(fix_disp("FIX_Z"), 2, 2);
  CHKERR remove_ents(fix_disp("FIX_ALL"), 0, 3);


  // End Elastic BC


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

  //   // Remove DOFs as homogeneous boundary condition is used
  // CHKERR mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
  //     simpleInterface->getProblemName(), domainField, get_ents_on_mesh_skin_1());

    auto get_ents_on_mesh_skin_5 = [&]() {
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

  //   // Remove DOFs as homogeneous boundary condition is used
  // CHKERR mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
  //     simpleInterface->getProblemName(), domainField, get_ents_on_mesh_skin_5());

    // BC NEUMANN 1
    auto get_ents_on_mesh_skin_2 = [&]() {
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

    // BC NEUMANN 2
    auto get_ents_on_mesh_skin_3 = [&]() {
    Range boundary_entities;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      std::string entity_name = it->getName();
      if (entity_name.compare(0, 6, "FLUX_2") == 0) {
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
    auto get_ents_on_mesh_skin_4 = [&]() {
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
      if (entity_name.compare(0, 6, "FLUX_2") == 0) {
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
  boundaryMarker_4 = mark_boundary_dofs(get_ents_on_mesh_skin_4());
  boundaryMarker_5 = mark_boundary_dofs(get_ents_on_mesh_skin_5());
  fluxBoundaryConditions_1 = get_ents_on_mesh_skin_2();
  fluxBoundaryConditions_2 = get_ents_on_mesh_skin_3();
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Thermal_Elasticity2D::assembleSystem() {
  MoFEMFunctionBegin;
  // make this an input
  double D = 1.;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-D", &D, PETSC_NULL);
  auto D_mat = [D](const double, const double, const double) { return D; };
  auto q_unit = [](const double, const double, const double) { return 1; };

  double source = 0.;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-source", &source, PETSC_NULL);
  auto sourceTermFunction =[source](const double, const double, const double) { return source; };
  
  double bc_temp1 = 0.;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-bc_temp1", &bc_temp1, PETSC_NULL);
  auto bc_1 =[bc_temp1](const double, const double, const double) { return bc_temp1; };

  double bc_temp2 = 0.;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-bc_temp2", &bc_temp2, PETSC_NULL);
  auto bc_2 =[bc_temp2](const double, const double, const double) { return bc_temp2; };

  double bc_flux1 = 0.;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-bc_flux1", &bc_flux1, PETSC_NULL);
  auto flux_1 =[&](const double, const double, const double) { 
    const auto fe_ent = boundaryPipelineRhs->getFEEntityHandle();
    if(fluxBoundaryConditions_1.find(fe_ent)!=fluxBoundaryConditions_1.end()) {
      return bc_flux1; 
    } else {
      return 0.;
    }
  };

  double bc_flux2 = 0.;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-bc_flux2", &bc_flux2, PETSC_NULL);
  auto flux_2 =[&](const double, const double, const double) { 
    const auto fe_ent = boundaryPipelineRhs->getFEEntityHandle();
    if(fluxBoundaryConditions_2.find(fe_ent)!=fluxBoundaryConditions_2.end()) {
      return bc_flux2; 
    } else {
      return 0.;
    }
  };

  { // Push operators to the Pipeline that is responsible for calculating LHS of
    // domain elements
    // Calculate the inverse Jacobian for both Domain and Boundary
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(invJac));
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(invJac));

    // Push back the LHS  for conduction
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, true, boundaryMarker_1));
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpDomainGradGrad(domainField, domainField, D_mat));   
    domainPipelineLhs->getOpPtrVector().push_back(
        new OpUnSetBc(domainField));
   }

  { // Push operators to the Pipeline that is responsible for calculating RHS of the source
    domainPipelineRhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, true, boundaryMarker_1));
    domainPipelineRhs->getOpPtrVector().push_back(
        new OpDomainSource(domainField, sourceTermFunction));
    domainPipelineRhs->getOpPtrVector().push_back(
        new OpUnSetBc(domainField));
  }

  // Start Code for non zero Dirichelet conditions
  { // Push operators in boundary pipeline LHS for Dirichelet
    boundaryPipelineLhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, false, boundaryMarker_1));
    boundaryPipelineLhs->getOpPtrVector().push_back(
        new OpBoundaryMass(domainField, domainField, q_unit));
    boundaryPipelineLhs->getOpPtrVector().push_back(new OpUnSetBc(domainField));
  }

      { // Push operators in boundary pipeline LHS for Dirichelet
    boundaryPipelineLhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, false, boundaryMarker_5));
    boundaryPipelineLhs->getOpPtrVector().push_back(
        new OpBoundaryMass(domainField, domainField, q_unit));
    boundaryPipelineLhs->getOpPtrVector().push_back(new OpUnSetBc(domainField));
  }

  { // Push operators in boundary pipeline RHS for Dirichelet
    boundaryPipelineRhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, false, boundaryMarker_1));
    boundaryPipelineRhs->getOpPtrVector().push_back(
        new OpBoundarySource(domainField, bc_1));
    boundaryPipelineRhs->getOpPtrVector().push_back(new OpUnSetBc(domainField));
  }

  { // Push operators in boundary pipeline RHS for Dirichelet
    boundaryPipelineRhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, false, boundaryMarker_5));
    boundaryPipelineRhs->getOpPtrVector().push_back(
        new OpBoundarySource(domainField, bc_2));
    boundaryPipelineRhs->getOpPtrVector().push_back(new OpUnSetBc(domainField));
  }
// End Code for non zero Dirichelet conditions

  { // Push operators to the Pipeline that is responsible for calculating RHS of
    // boundary elements
    boundaryPipelineRhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, false, boundaryMarker_2));
    boundaryPipelineRhs->getOpPtrVector().push_back(
        new OpBoundarySource(domainField, flux_1));
    boundaryPipelineRhs->getOpPtrVector().push_back(new OpUnSetBc(domainField));
  }

    { // Push operators to the Pipeline that is responsible for calculating RHS of
    // boundary elements
    boundaryPipelineRhs->getOpPtrVector().push_back(
        new OpSetBc(domainField, false, boundaryMarker_3));
    boundaryPipelineRhs->getOpPtrVector().push_back(
        new OpBoundarySource(domainField, flux_2));
    boundaryPipelineRhs->getOpPtrVector().push_back(new OpUnSetBc(domainField));
  }

  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  if (SPACE_DIM == 2) {
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpCalculateInvJacForFace(invJac));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpSetInvJacH1ForFace(invJac));
  }
  pipeline_mng->getOpDomainLhsPipeline().push_back(new OpK("U", "U", matDPtr));
  // Start coupling term

  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpKut("U", "TEMP", thDPtr, previousUpdate));
  // end coupling
  double set_body_force = 0.;
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-set_body_force", &set_body_force, PETSC_NULL);
  auto body_force =[&](const double, const double, const double) { return set_body_force; };     
  pipeline_mng->getOpDomainRhsPipeline().push_back(new OpBodyForce(
      "U", bodyForceMatPtr, body_force));  

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

MoFEMErrorCode Thermal_Elasticity2D::solveSystem() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  auto solver = pipeline_mng->createKSP();
  CHKERR KSPSetFromOptions(solver);
  CHKERR KSPSetUp(solver);

  auto dm = simpleInterface->getDM();
  auto D = smartCreateDMVector(dm);
  auto F = smartVectorDuplicate(D);

  CHKERR KSPSolve(solver, F, D);
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
  
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Thermal_Elasticity2D::outputResults() {
  MoFEMFunctionBegin;

PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  pipeline_mng->getDomainLhsFE().reset();
  auto post_proc_fe = boost::make_shared<PostProcFaceOnRefinedMesh>(mField);
  post_proc_fe->generateReferenceElementMesh();
  if (SPACE_DIM) {
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(invJac));
    post_proc_fe->getOpPtrVector().push_back(new OpSetInvJacH1ForFace(invJac));
  }
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
                                                               matGradPtr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpSymmetrizeTensor<SPACE_DIM>("U", matGradPtr, matStrainPtr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
          "U", matStrainPtr, matStressPtr, matDPtr));
  post_proc_fe->getOpPtrVector().push_back(new OpPostProcElastic<SPACE_DIM>(
      "U", post_proc_fe->postProcMesh, post_proc_fe->mapGaussPts, matStrainPtr,
      matStressPtr));
  post_proc_fe->addFieldValuesPostProc("U");
  post_proc_fe->addFieldValuesPostProc("TEMP");
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_result.h5m");

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
    Thermal_Elasticity2D poisson_problem(m_field);
    CHKERR poisson_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}