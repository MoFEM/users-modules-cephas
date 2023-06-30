#include <BasicFiniteElements.hpp>
#include <MoFEM.hpp>
using namespace MoFEM;

static char help[] = "...\n\n";

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = PipelineManager::FaceEle;
  using BoundaryEle = PipelineManager::EdgeEle;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = VolumeElementForcesAndSourcesCore;
  using BoundaryEle = PipelineManager::FaceEle;
};

constexpr int SPACE_DIM = 3;
constexpr int BASE_DIM = 3;
constexpr AssemblyType A = AssemblyType::PETSC; //< selected assembly type
constexpr IntegrationType I =
    IntegrationType::GAUSS;                     //< selected integration type
constexpr CoordinateTypes coord_type = CARTESIAN;

using EntData = EntitiesFieldData::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

using AssemblyDomainEleOp = FormsIntegrators<DomainEleOp>::Assembly<A>::OpBase;
using AssemblyBoundaryEleOp =
    FormsIntegrators<BoundaryEleOp>::Assembly<A>::OpBase;

//! [Domain Operators]
// OpCurlCurl - Once moved to Bilinear Forms in Core.
//! [Domain Operators]

//! [Boundary Operators]
using OpBoundarySourceRhs = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<3, 3>;
//! [Boundary Operators]

int order = 2; ///< Order
// material parameters
double mu = 1.;       ///< magnetic constant  N / A2
double epsilon = 0.1; ///< regularization paramater

using ScalFun =
    boost::function<double(const double, const double, const double)>;

struct OpCurlCurl : public AssemblyDomainEleOp {
  OpCurlCurl(const std::string row_field_name, const std::string col_field_name,
             ScalFun beta)
      : AssemblyDomainEleOp(row_field_name, col_field_name,
                            AssemblyDomainEleOp::OPROWCOL),
        betaCoeff(beta) {
    sYmm = true;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {

    MoFEMFunctionBegin;

    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    FTensor::Index<'l', SPACE_DIM> l;
    const double vol = getMeasure();
    size_t nb_base_functions = row_data.getN().size2() / 3;
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_curl_base = row_data.getFTensor2DiffN<3, SPACE_DIM>();
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto get_t_vec = [&](const int rr) {
      std::array<double *, SPACE_DIM> ptrs;
      for (auto l = 0; l != SPACE_DIM; ++l)
        ptrs[l] = &locMat(rr + l, l);
      return FTensor::Tensor1<FTensor::PackPtr<double *, SPACE_DIM>, SPACE_DIM>(
          ptrs);
    };

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double alpha =
          t_w * vol * betaCoeff(t_coords(0), t_coords(1), t_coords(2));

      FTensor::Tensor1<double, SPACE_DIM> t_row_curl;
      int rr = 0;
      for (; rr != nbRows / SPACE_DIM; ++rr) {
        t_row_curl(i) = FTensor::levi_civita(j, i, k) * t_row_curl_base(j, k);

        FTensor::Tensor1<double, SPACE_DIM> t_col_curl;

        auto t_col_curl_base = col_data.getFTensor2DiffN<3, SPACE_DIM>(gg, 0);

        auto t_vec = get_t_vec(3 * rr);

        int cc = 0;
        for (; cc != nbCols / SPACE_DIM; cc++) {
          t_col_curl(i) = FTensor::levi_civita(j, i, k) * t_col_curl_base(j, k);

          // locMat(rr, cc) += alpha * t_row_curl(i) * t_col_curl(i);
          t_vec(l) += alpha * t_row_curl(i) * t_col_curl(i);

          ++t_col_curl_base;
          ++t_vec;
        }
        ++t_row_curl_base;
      }
      for (; rr < nb_base_functions; ++rr)
        ++t_row_curl_base;
      ++t_w;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
  ScalFun betaCoeff;
};

struct OpStab : public AssemblyDomainEleOp {
  OpStab(const std::string row_field_name, const std::string col_field_name,
         ScalFun beta)
      : AssemblyDomainEleOp(row_field_name, col_field_name,
                            AssemblyDomainEleOp::OPROWCOL),
        betaCoeff(beta) {
    sYmm = true;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {

    MoFEMFunctionBegin;

    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'l', SPACE_DIM> l;
    const double vol = getMeasure();
    size_t nb_base_functions = row_data.getN().size2() / 3;
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor1N<3>();
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto get_t_vec = [&](const int rr) {
      std::array<double *, SPACE_DIM> ptrs;
      for (auto l = 0; l != SPACE_DIM; ++l)
        ptrs[l] = &locMat(rr + l, l);
      return FTensor::Tensor1<FTensor::PackPtr<double *, SPACE_DIM>, SPACE_DIM>(
          ptrs);
    };

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double alpha =
          t_w * vol * betaCoeff(t_coords(0), t_coords(1), t_coords(2));

      int rr = 0;
      for (; rr != nbRows / SPACE_DIM; ++rr) {

        auto t_col_base = col_data.getFTensor1N<3>();

        auto t_vec = get_t_vec(3 * rr);

        int cc = 0;
        for (; cc != nbCols / SPACE_DIM; cc++) {

          // locMat(rr, cc) += alpha * t_row_curl(i) * t_col_curl(i);
          t_vec(l) += alpha * t_row_base(i) * t_col_base(i);

          ++t_col_base;
          ++t_vec;
        }
        ++t_row_base;
      }
      for (; rr < nb_base_functions; ++rr)
        ++t_row_base;
      ++t_w;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
  ScalFun betaCoeff;
};

struct Magnetostatics {
public:
  Magnetostatics(MoFEM::Interface &m_field);

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

  // boundary condition ranges
  Range naturalBcRange;
};

Magnetostatics::Magnetostatics(MoFEM::Interface &m_field) : mField(m_field) {}

//! [Read mesh]
MoFEMErrorCode Magnetostatics::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Setup problem]
MoFEMErrorCode Magnetostatics::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR simpleInterface->addDomainField("MAGNETIC_POTENTIAL", HCURL,
                                         DEMKOWICZ_JACOBI_BASE, 1);
  CHKERR simpleInterface->addBoundaryField("MAGNETIC_POTENTIAL", HCURL,
                                           DEMKOWICZ_JACOBI_BASE, 1);

  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder("MAGNETIC_POTENTIAL", order);

  CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}
//! [Setup problem]

//! [Boundary condition]
MoFEMErrorCode Magnetostatics::boundaryCondition() {
  MoFEMFunctionBegin;

  auto natural_bc = [&]() {
    Range boundary_entities;
    auto bc_mng = mField.getInterface<BcManager>();
    CHKERR bc_mng->pushMarkDOFsOnEntities(simpleInterface->getProblemName(),
                                          "MAGNETIC_POTENTIAL",
                                          "MAGNETIC_POTENTIAL", 0, 0);

    auto &bc_map = bc_mng->getBcMapByBlockName();
    for (auto b : bc_map) {
      if (std::regex_match(b.first, std::regex("(.*)NATURAL(.*)"))) {
        boundary_entities.merge(*(b.second->getBcEntsPtr()));
      }
    }
    // Add vertices to boundary entities
    Range boundary_vertices;
    CHKERR mField.get_moab().get_connectivity(boundary_entities,
                                              boundary_vertices, true);
    boundary_entities.merge(boundary_vertices);

    return boundary_entities;
  };

  naturalBcRange = natural_bc();

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Assemble system]
MoFEMErrorCode Magnetostatics::assembleSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  // Push Domain operators to the Pipeline that is responsible for calculating
  // LHS
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pipeline_mng->getOpDomainLhsPipeline(), {HCURL});
  pipeline_mng->getOpDomainLhsPipeline().push_back(new OpCurlCurl(
      "MAGNETIC_POTENTIAL", "MAGNETIC_POTENTIAL",
      [](const double, const double, const double) { return 1.; }));
  pipeline_mng->getOpDomainLhsPipeline().push_back(new OpStab(
      "MAGNETIC_POTENTIAL", "MAGNETIC_POTENTIAL",
      [](const double, const double, const double) { return epsilon / mu; }));

  // Push Domain operators to the Pipeline that is responsible for calculating
  // RHS
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pipeline_mng->getOpDomainRhsPipeline(), {HCURL});

  boost::function<FTensor::Tensor1<double, 3>(const double, const double,
                                              const double)>
      natural_fun = [](double x, double y, double z) {
        const double r = sqrt(x * x + y * y);
        return FTensor::Tensor1<double, 3>{(-y / r), (x / r), 0.};
      };

  // Push Boundary operators to the Pipeline that is responsible for calculating
  // RHS
  pipeline_mng->getOpBoundaryRhsPipeline().push_back(
      new OpBoundarySourceRhs("MAGNETIC_POTENTIAL", natural_fun,
                              boost::make_shared<Range>(naturalBcRange)));

  MoFEMFunctionReturn(0);
}
//! [Assemble system]

//! [Set integration rules]
MoFEMErrorCode Magnetostatics::setIntegrationRules() {
  MoFEMFunctionBegin;
  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto rule_lhs = [](int, int, int p) -> int { return 2 * p + 1; };
  auto rule_rhs = [](int, int, int p) -> int { return 2 * p + 1; };

  CHKERR pipeline_mng->setDomainLhsIntegrationRule(rule_lhs);
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule_rhs);

  auto boundary_rule_lhs = [](int, int, int p) -> int { return 2 * p; };
  auto boundary_rule_rhs = [](int, int, int p) -> int { return 2 * p; };

  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(boundary_rule_lhs);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(boundary_rule_rhs);

  MoFEMFunctionReturn(0);
}
//! [Set integration rules]

//! [Solve system]
MoFEMErrorCode Magnetostatics::solveSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto ksp_solver = pipeline_mng->createKSP();
  CHKERR KSPSetFromOptions(ksp_solver);
  CHKERR KSPSetUp(ksp_solver);

  // Create RHS and solution vectors
  auto dm = simpleInterface->getDM();
  auto F = createDMVector(dm);
  auto D = vectorDuplicate(F);

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
MoFEMErrorCode Magnetostatics::outputResults() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();

  auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      post_proc_fe->getOpPtrVector(), {HCURL});

  auto field_val_ptr = boost::make_shared<MatrixDouble>();
  auto induction_ptr = boost::make_shared<MatrixDouble>();

  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateHVecVectorField<SPACE_DIM>("MAGNETIC_POTENTIAL",
                                                field_val_ptr));
  // Only implement for <3, 3>  OR <1, 2> (<BASE_DIM, SPACE_DIM>)
  post_proc_fe->getOpPtrVector().push_back(new OpCalculateHcurlVectorCurl<3, 3>(
      "MAGNETIC_POTENTIAL", induction_ptr));

  using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

  post_proc_fe->getOpPtrVector().push_back(

      new OpPPMap(
          post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

          OpPPMap::DataMapVec{},

          OpPPMap::DataMapMat{{"MAGNETIC_POTENTIAL", field_val_ptr},
                              {"MAGNETIC_INDUCTION_FIELD", induction_ptr}},

          OpPPMap::DataMapMat{},

          OpPPMap::DataMapMat{}

          )

  );

  // post_proc.getOpPtrVector().push_back(new OpPostProcessCurl(
  //     blockData, post_proc.getPostProcMesh(), post_proc.getMapGaussPts()));
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_result.h5m");

  MoFEMFunctionReturn(0);
}
//! [Output results]

//! [Run program]
MoFEMErrorCode Magnetostatics::runProgram() {
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
    Magnetostatics ex(m_field);
    CHKERR ex.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}
//! [Main]