#include <stdlib.h>
#include <BasicFiniteElements.hpp>

using namespace MoFEM;
// using namespace Poisson2DNonhomogeneousOperators;

inline double sqr(double x) { return x * x; }

constexpr int SPACE_DIM = 2;
using PostProcFaceEle =
    PostProcBrokenMeshInMoab<FaceElementForcesAndSourcesCore>;

using DomainEle = PipelineManager::FaceEle;
using DomainEleOp = DomainEle::UserDataOperator;

using OpDomainGradGrad = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpGradGrad<1, 1, SPACE_DIM>;

using OpDomainSource = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

static char help[] = "...\n\n";

struct StandardPoisson {
public:
  StandardPoisson(MoFEM::Interface &m_field);

  // Declaration of the main function to run analysis
  MoFEMErrorCode runProgram();

private:
  // Declaration of other main functions called in runProgram()
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode checkError();
  MoFEMErrorCode outputResults();

  //! [Analytical function]
  static double analyticalFunction(const double x, const double y,
                                   const double z) {
    return exp(-100. * (sqr(x) + sqr(y))) * cos(M_PI * x) * cos(M_PI * y);
  }
  //! [Analytical function]

  //! [Analytical function gradient]
  static VectorDouble analyticalFunctionGrad(const double x, const double y,
                                             const double z) {
    VectorDouble res;
    res.resize(2);
    res[0] = -exp(-100. * (sqr(x) + sqr(y))) *
             (200. * x * cos(M_PI * x) + M_PI * sin(M_PI * x)) * cos(M_PI * y);
    res[1] = -exp(-100. * (sqr(x) + sqr(y))) *
             (200. * y * cos(M_PI * y) + M_PI * sin(M_PI * y)) * cos(M_PI * x);
    return res;
  }
  //! [Analytical function gradient]

  //! [Source function]
  static double sourceFunction(const double x, const double y, const double z) {
    return -exp(-100. * (sqr(x) + sqr(y))) *
           (400. * M_PI *
                (x * cos(M_PI * y) * sin(M_PI * x) +
                 y * cos(M_PI * x) * sin(M_PI * y)) +
            2. * (20000. * (sqr(x) + sqr(y)) - 200. - sqr(M_PI)) *
                cos(M_PI * x) * cos(M_PI * y));
  }
  //! [Source function]

  struct CommonData {
    boost::shared_ptr<VectorDouble> approxVals;
    boost::shared_ptr<MatrixDouble> approxValsGrad;
    SmartPetscObj<Vec> petscVec;

    enum VecElements { ERROR_L2_NORM = 0, ERROR_H1_SEMINORM, LAST_ELEMENT };
  };

  boost::shared_ptr<CommonData> commonDataPtr;

  struct OpError : public DomainEleOp {
    std::string domainField;
    boost::shared_ptr<CommonData> commonDataPtr;
    MoFEM::Interface &mField;
    OpError(std::string domain_field,
            boost::shared_ptr<CommonData> &common_data_ptr,
            MoFEM::Interface &m_field)
        : DomainEleOp(domain_field, OPROW), domainField(domain_field),
          commonDataPtr(common_data_ptr), mField(m_field) {
      std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
      doEntities[MBTRI] = doEntities[MBQUAD] = true;
    }
    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  // Main interfaces
  MoFEM::Interface &mField;
  Simple *simpleInterface;

  // Field name and approximation order
  std::string domainField;
  int oRder;

  // Object to mark boundary entities for the assembling of domain elements
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;

  // Boundary entities marked for fieldsplit (block) solver - optional
  Range boundaryEntitiesForFieldsplit;
};

StandardPoisson::StandardPoisson(MoFEM::Interface &m_field)
    : domainField("U"), mField(m_field) {}

MoFEMErrorCode StandardPoisson::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode StandardPoisson::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR simpleInterface->addDomainField(domainField, H1,
                                         AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simpleInterface->addBoundaryField(domainField, H1,
                                           AINSWORTH_LEGENDRE_BASE, 1);

  int oRder = 2;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &oRder, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder(domainField, oRder);

  CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode StandardPoisson::boundaryCondition() {
  MoFEMFunctionBegin;

  auto bc_mng = mField.getInterface<BcManager>();

  // Remove BCs from blockset name "BOUNDARY_CONDITION" or SETU, note that you
  // can use regular expression to put list of blocksets;
  CHKERR bc_mng->removeBlockDOFsOnEntities<BcScalarMeshsetType<BLOCKSET>>(
      simpleInterface->getProblemName(), "BOUNDARY", std::string(domainField),
      true);

  MoFEMFunctionReturn(0);
}

//! [Create common data]
MoFEMErrorCode StandardPoisson::createCommonData() {
  MoFEMFunctionBegin;
  commonDataPtr = boost::make_shared<CommonData>();
  PetscInt ghosts[2] = {0, 1};
  if (!mField.get_comm_rank())
    commonDataPtr->petscVec =
        createGhostVector(mField.get_comm(), 2, 2, 0, ghosts);
  else
    commonDataPtr->petscVec =
        createGhostVector(mField.get_comm(), 0, 2, 2, ghosts);
  commonDataPtr->approxVals = boost::make_shared<VectorDouble>();
  commonDataPtr->approxValsGrad = boost::make_shared<MatrixDouble>();
  MoFEMFunctionReturn(0);
}
//! [Create common data]

MoFEMErrorCode StandardPoisson::assembleSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  { // Push operators to the Pipeline that is responsible for calculating LHS of
    // domain elements
    CHKERR AddHOOps<2, 2, 2>::add(pipeline_mng->getOpDomainLhsPipeline(), {H1});
    auto unity = [](const double, const double, const double) { return 1; };
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpDomainGradGrad(domainField, domainField, unity));
  }

  { // Push operators to the Pipeline that is responsible for calculating RHS of
    // domain elements
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpDomainSource(domainField, sourceFunction));
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode StandardPoisson::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto domain_rule_lhs = [](int, int, int p) -> int { return 2 * (p - 1); };
  auto domain_rule_rhs = [](int, int, int p) -> int { return 2 * (p - 1); };
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(domain_rule_lhs);
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(domain_rule_rhs);

  auto boundary_rule_lhs = [](int, int, int p) -> int { return 2 * p; };
  auto boundary_rule_rhs = [](int, int, int p) -> int { return 2 * p; };
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(boundary_rule_lhs);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(boundary_rule_rhs);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode StandardPoisson::solveSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto ksp_solver = pipeline_mng->createKSP();
  CHKERR KSPSetFromOptions(ksp_solver);

  // Create RHS and solution vectors
  auto dm = simpleInterface->getDM();
  auto F = createDMVector(dm);
  auto D = vectorDuplicate(F);

  CHKERR KSPSetUp(ksp_solver);

  // Solve the system
  CHKERR KSPSolve(ksp_solver, F, D);

  // Scatter result data on the mesh
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}

//! [Check error]
MoFEMErrorCode StandardPoisson::checkError() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getOpDomainRhsPipeline().clear();

  CHKERR AddHOOps<2, 2, 2>::add(pipeline_mng->getOpDomainRhsPipeline(), {H1});

  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldValues(domainField, commonDataPtr->approxVals));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldGradient<2>(domainField,
                                            commonDataPtr->approxValsGrad));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpError(domainField, commonDataPtr, mField));

  CHKERR VecZeroEntries(commonDataPtr->petscVec);

  CHKERR pipeline_mng->loopFiniteElements();

  CHKERR VecAssemblyBegin(commonDataPtr->petscVec);
  CHKERR VecAssemblyEnd(commonDataPtr->petscVec);
  const double *array;
  CHKERR VecGetArrayRead(commonDataPtr->petscVec, &array);
  MOFEM_LOG("EXAMPLE", Sev::inform)
      << "Global error L2 norm: " << std::sqrt(array[0]);
  MOFEM_LOG("EXAMPLE", Sev::inform)
      << "Global error H1 seminorm: " << std::sqrt(array[1]);
  CHKERR VecRestoreArrayRead(commonDataPtr->petscVec, &array);

  MoFEMFunctionReturn(0);
}
//! [Check error]

MoFEMErrorCode StandardPoisson::outputResults() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getBoundaryLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();

  using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

  auto post_proc_fe = boost::make_shared<PostProcFaceEle>(mField);

  CHKERR AddHOOps<2, 2, 2>::add(post_proc_fe->getOpPtrVector(), {H1});

  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(domainField, commonDataPtr->approxVals));
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateScalarFieldGradient<2>(domainField,
                                            commonDataPtr->approxValsGrad));

  post_proc_fe->getOpPtrVector().push_back(new OpPPMap(
      post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),
      {{domainField, commonDataPtr->approxVals}},
      {{domainField + "_GRAD", commonDataPtr->approxValsGrad}}, {}, {}));
  pipeline_mng->getDomainRhsFE() = post_proc_fe;

  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_result.h5m");

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode StandardPoisson::runProgram() {
  MoFEMFunctionBegin;

  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();
  CHKERR createCommonData();
  CHKERR assembleSystem();
  CHKERR setIntegrationRules();
  CHKERR solveSystem();
  CHKERR checkError();
  CHKERR outputResults();

  MoFEMFunctionReturn(0);
}

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for example problem
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "EXAMPLE"));
  LogManager::setLog("EXAMPLE");
  MOFEM_LOG_TAG("EXAMPLE", "StandardPoisson");

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
    StandardPoisson poisson_problem(m_field);
    CHKERR poisson_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}

//! [OpError]
MoFEMErrorCode StandardPoisson::OpError::doWork(int side, EntityType type,
                                                EntData &data) {
  MoFEMFunctionBegin;
  const int nb_integration_pts = getGaussPts().size2();
  const double area = getMeasure();
  auto t_w = getFTensor0IntegrationWeight();
  auto t_val = getFTensor0FromVec(*(commonDataPtr->approxVals));
  auto t_val_grad = getFTensor1FromMat<2>(*(commonDataPtr->approxValsGrad));
  auto t_coords = getFTensor1CoordsAtGaussPts();
  FTensor::Tensor1<double, 2> t_diff;
  FTensor::Index<'i', 2> i;

  double error_l2 = 0;
  double error_h1 = 0;

  for (int gg = 0; gg != nb_integration_pts; ++gg) {
    const double alpha = t_w * area;

    double diff = t_val - StandardPoisson::analyticalFunction(
                              t_coords(0), t_coords(1), t_coords(2));
    error_l2 += alpha * sqr(diff);

    VectorDouble vec = StandardPoisson::analyticalFunctionGrad(
        t_coords(0), t_coords(1), t_coords(2));
    auto t_fun_grad = getFTensor1FromArray<2, 2>(vec);
    t_diff(i) = t_val_grad(i) - t_fun_grad(i);

    error_h1 += alpha * t_diff(i) * t_diff(i);

    ++t_w;
    ++t_val;
    ++t_val_grad;
    ++t_coords;
  }

  int index = CommonData::ERROR_L2_NORM;
  constexpr std::array<int, 2> indices = {CommonData::ERROR_L2_NORM,
                                          CommonData::ERROR_H1_SEMINORM};
  std::array<double, 2> values;
  values[0] = error_l2;
  values[1] = error_h1;
  CHKERR VecSetValues(commonDataPtr->petscVec, 2, indices.data(), values.data(),
                      ADD_VALUES);
  MoFEMFunctionReturn(0);
}
//! [OpError]