/**
 * @file test_tangent.cpp
 * @example test_tangent.cpp
 * @brief Test operators in forms integrators
 * @date 2022-12-15
 *
 * @copyright Copyright (c) 2022
 *
 * TODO: Add more operators.
 *
 */

#include <MoFEM.hpp>
using namespace MoFEM;

static char help[] = "...\n\n";

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = PipelineManager::FaceEle;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = VolumeElementForcesAndSourcesCore;
};

//! [Define dimension]
constexpr int SPACE_DIM = EXECUTABLE_DIMENSION;
constexpr AssemblyType A = AssemblyType::PETSC; //< selected assembly type
constexpr IntegrationType I =
    IntegrationType::GAUSS; //< selected integration type

using EntData = EntitiesFieldData::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

using AssemblyDomainEleOp = FormsIntegrators<DomainEleOp>::Assembly<A>::OpBase;
using OpMassLL =
    FormsIntegrators<DomainEleOp>::Assembly<A>::BiLinearForm<I>::OpMass<1, 1>;
using OpSourceL =
    FormsIntegrators<DomainEleOp>::Assembly<A>::LinearForm<I>::OpSource<1, 1>;
using OpMassVV = FormsIntegrators<DomainEleOp>::Assembly<A>::BiLinearForm<
    I>::OpMass<3, SPACE_DIM>;
using OpSourceV = FormsIntegrators<DomainEleOp>::Assembly<A>::LinearForm<
    I>::OpSource<3, SPACE_DIM>;

constexpr bool debug = true;

struct OpRhs : public AssemblyDomainEleOp {

  OpRhs(const std::string field_name, boost::shared_ptr<VectorDouble> l_ptr,
        boost::shared_ptr<VectorDouble> l_dot_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        lPtr(l_ptr), lDotPtr(l_dot_ptr) {}

  MoFEMErrorCode iNtegrate(EntData &data);

private:
  boost::shared_ptr<VectorDouble> lPtr;
  boost::shared_ptr<VectorDouble> lDotPtr;
};

struct OpLhs : public AssemblyDomainEleOp {
  OpLhs(const std::string field_name)
      : AssemblyDomainEleOp(field_name, field_name,
                            AssemblyDomainEleOp::OPROWCOL) {}

  MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);
};

MoFEMErrorCode testOp(MoFEM::Interface &m_field);

MoFEMErrorCode initialiseFieldLevelSet(MoFEM::Interface &m_field);

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  try {

    // Create MoAB database
    moab::Core moab_core;
    moab::Interface &moab = moab_core;

    // Create MoFEM database and link it to MoAB
    MoFEM::Core mofem_core(moab);
    MoFEM::Interface &m_field = mofem_core;

    // Register DM Manager
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    // Add logging channel for example
    auto core_log = logging::core::get();
    core_log->add_sink(
        LogManager::createSink(LogManager::getStrmWorld(), "LevelSet"));
    LogManager::setLog("LevelSet");
    MOFEM_LOG_TAG("LevelSet", "LevelSet");

    // Simple interface
    auto simple = m_field.getInterface<Simple>();

    // get options from command line
    CHKERR simple->getOptions();
    // load mesh file
    CHKERR simple->loadFile();

    // Scalar fields and vector field is tested. Add more fields, i.e. vector
    // field if needed.
    CHKERR simple->addDomainField("L", H1, AINSWORTH_LEGENDRE_BASE, 1);
    CHKERR simple->addDomainField("V", HCURL, AINSWORTH_LEGENDRE_BASE, 1);

    // set fields order, i.e. for most first cases order is sufficient.
    CHKERR simple->setFieldOrder("L", 2);
    CHKERR simple->setFieldOrder("V", 2);

    // setup problem
    CHKERR simple->setUp();

    auto pip = m_field.getInterface<PipelineManager>(); // get interface to
                                                        // pipeline manager

    if (debug)
      CHKERR testOp(m_field);

    CHKERR initialiseFieldLevelSet(m_field);
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}

FTensor::Index<'i', SPACE_DIM> i;
FTensor::Index<'j', SPACE_DIM> j;
FTensor::Index<'k', SPACE_DIM> k;

inline auto get_axis() { return FTensor::Tensor1<double, 3>{0, 0, 1}; }

template <typename T> inline auto get_velocity(T &t_coords) {
  FTensor::Index<'I', 3> I;
  FTensor::Index<'J', 3> J;
  FTensor::Index<'K', 3> K;
  FTensor::Tensor1<double, 3> t_velocity;
  auto t_axis = get_axis();
  t_velocity(K) = (levi_civita(I, J, K) * t_coords(I)) * t_axis(J);
  return t_velocity;
}

double get_level_set(const double x, const double y, const double z) {
  constexpr double xc = 0.1;
  constexpr double yc = 0.;
  constexpr double zc = 0.;
  constexpr double r = 0.2;
  return std::sqrt(pow(x - xc, 2) + pow(y - yc, 2) + pow(z - zc, 2)) - r;
}

MoFEMErrorCode OpRhs::iNtegrate(EntData &data) {
  MoFEMFunctionBegin;

  const auto nb_int_points = getGaussPts().size1();
  const auto nb_dofs = data.getIndices().size();
  const auto nb_base_func = data.getN().size2();

  auto t_l = getFTensor0FromVec(*lPtr);
  auto t_l_dot = getFTensor0FromVec(*lDotPtr);

  auto t_coords = getFTensor1Coords();
  auto t_base = data.getFTensor0N();
  auto t_diff_base = data.getFTensor1DiffN<SPACE_DIM>();

  auto t_w = getFTensor0IntegrationWeight();
  for (auto gg = 0; gg != nb_int_points; ++gg) {
    auto t_velocity = get_velocity(t_coords);
    const auto alpha = t_w * getMeasure();

    auto res0 = alpha * t_l_dot;
    FTensor::Tensor1<double, SPACE_DIM> t_res1;
    t_res1(i) = (alpha * t_l) * t_velocity(i);

    ++t_w;
    ++t_coords;
    ++t_l;
    ++t_l_dot;

    auto &nf = this->locF;

    int rr = 0;
    for (; rr != nb_dofs; ++rr) {
      nf(rr) += res0 * t_base + t_res1(i) * t_diff_base(i);
      ++t_base;
      ++t_diff_base;
    }
    for (; rr < nb_base_func; ++rr) {
      ++t_base;
      ++t_diff_base;
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode OpLhs::iNtegrate(EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const auto nb_int_points = getGaussPts().size1();
  const auto nb_row_dofs = row_data.getIndices().size();
  const auto nb_col_dofs = col_data.getIndices().size();
  const auto nb_base_func = row_data.getN().size2();

  auto t_coords = getFTensor1Coords();
  auto t_row_base = row_data.getFTensor0N();
  auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();

  auto t_w = getFTensor0IntegrationWeight();
  for (auto gg = 0; gg != nb_int_points; ++gg) {
    auto t_velocity = get_velocity(t_coords);
    const auto alpha = t_w * getMeasure();
    const auto beta = alpha * getTSa();
    ++t_w;
    ++t_coords;

    auto &mat = this->locMat;

    int rr = 0;
    for (; rr != nb_row_dofs; ++rr) {
      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (int cc = 0; cc != nb_col_dofs; ++cc) {
        mat(rr, cc) += beta * t_row_base * t_col_base + (alpha * t_col_base) *
                                                            t_row_diff_base(i) *
                                                            t_velocity(i);

        ++t_col_base;
      }
      ++t_row_base;
      ++t_row_diff_base;
    }
    for (; rr < nb_base_func; ++rr) {
      ++t_row_base;
      ++t_row_diff_base;
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode testOp(MoFEM::Interface &m_field) {
  MoFEMFunctionBegin;

  // get operators tester
  auto simple = m_field.getInterface<Simple>();
  auto opt = m_field.getInterface<OperatorsTester>(); // get interface to
                                                      // OperatorsTester
  auto pip = m_field.getInterface<PipelineManager>(); // get interface to
                                                      // pipeline manager

  auto post_proc = [&](auto dm, auto f_res, auto out_name) {
    MoFEMFunctionBegin;
    auto post_proc_fe =
        boost::make_shared<PostProcBrokenMeshInMoab<DomainEle>>(m_field);

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    auto l_vec = boost::make_shared<VectorDouble>();
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("L", l_vec, f_res));

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(

            post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

            {{"L", l_vec}},

            {},

            {}, {})

    );

    CHKERR DMoFEMLoopFiniteElements(dm, simple->getDomainFEName(),
                                    post_proc_fe);
    post_proc_fe->writeFile(out_name);
    MoFEMFunctionReturn(0);
  };

  pip->getOpDomainLhsPipeline().clear();
  pip->getOpDomainRhsPipeline().clear();

  pip->setDomainLhsIntegrationRule([](int, int, int o) { return 2 * o; });
  pip->setDomainLhsIntegrationRule([](int, int, int o) { return 2 * o; });

  auto l_ptr = boost::make_shared<VectorDouble>();
  auto l_dot_ptr = boost::make_shared<VectorDouble>();

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpDomainRhsPipeline(), {H1, L2});
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpDomainLhsPipeline(), {H1, L2});

  pip->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldValues("L", l_ptr));
  pip->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldValuesDot("L", l_dot_ptr));
  pip->getOpDomainRhsPipeline().push_back(new OpRhs("L", l_ptr, l_dot_ptr));

  pip->getOpDomainLhsPipeline().push_back(new OpLhs("L"));

  constexpr double eps = 1e-6;

  auto x = opt->setRandomFields(simple->getDM(), {{"L", {-1, 1}}});
  auto dot_x = opt->setRandomFields(simple->getDM(), {{"L", {-1, 1}}});
  auto diff_x = opt->setRandomFields(simple->getDM(), {{"L", {-1, 1}}});

  auto diff_res = opt->checkCentralFiniteDifference(
      simple->getDM(), simple->getDomainFEName(), pip->getDomainRhsFE(),
      pip->getDomainLhsFE(), x, dot_x, SmartPetscObj<Vec>(), diff_x, 0, 1, eps);

  if (debug) {
    // Example how to plot direction in direction diff_x. If instead
    // directionalCentralFiniteDifference(...) diff_res is used, then error
    // on directive is plotted.
    CHKERR post_proc(simple->getDM(),
                     //
                     opt->directionalCentralFiniteDifference(
                         simple->getDM(), simple->getDomainFEName(),
                         pip->getDomainRhsFE(), x, dot_x, SmartPetscObj<Vec>(),
                         diff_res, 0, 1, eps),
                     //
                     "tangent_op_error.h5m");
  }

  // Calculate norm of difference between directive calculated from finite
  // difference, and tangent matrix.
  double fnorm;
  CHKERR VecNorm(diff_res, NORM_2, &fnorm);
  MOFEM_LOG_C("LevelSet", Sev::inform,
              "Test consistency of tangent matrix %3.4e", fnorm);

  constexpr double err = 1e-9;
  if (fnorm > err)
    SETERRQ1(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
             "Norm of directional derivative too large err = %3.4e", fnorm);

  MoFEMFunctionReturn(0);
};

MoFEMErrorCode initialiseFieldLevelSet(MoFEM::Interface &m_field) {
  MoFEMFunctionBegin;

  // get operators tester
  auto simple = m_field.getInterface<Simple>();

  auto sub_dm = createSmartDM(m_field.get_comm(), "DMMOFEM");
  CHKERR DMMoFEMCreateSubDM(sub_dm, simple->getDM(), "SUB_LEVEL");
  CHKERR DMMoFEMSetDestroyProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMSetSquareProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMAddElement(sub_dm, simple->getDomainFEName());
  CHKERR DMMoFEMAddSubFieldRow(sub_dm, "L");
  CHKERR DMSetUp(sub_dm);

  auto lhs_fe = boost::make_shared<DomainEle>(m_field);
  auto rhs_fe = boost::make_shared<DomainEle>(m_field);

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      lhs_fe->getOpPtrVector(), {H1});
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      rhs_fe->getOpPtrVector(), {H1});

  lhs_fe->getOpPtrVector().push_back(new OpMassLL("L", "L"));
  rhs_fe->getOpPtrVector().push_back(new OpSourceL("L", get_level_set));

  boost::shared_ptr<FEMethod> null_fe;
  CHKERR DMMoFEMKSPSetComputeOperators(sub_dm, simple->getDomainFEName(),
                                       lhs_fe, null_fe, null_fe);
  CHKERR DMMoFEMKSPSetComputeRHS(sub_dm, simple->getDomainFEName(), rhs_fe,
                                 null_fe, null_fe);

  auto ksp = MoFEM::createKSP(m_field.get_comm());
  CHKERR KSPSetDM(ksp, sub_dm);
  CHKERR KSPSetFromOptions(ksp);
  CHKERR KSPSetUp(ksp);

  auto L = smartCreateDMVector(sub_dm);
  auto F = smartVectorDuplicate(L);

  CHKERR KSPSolve(ksp, F, L);
  CHKERR VecGhostUpdateBegin(L, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(L, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(sub_dm, L, INSERT_VALUES, SCATTER_REVERSE);

  auto post_proc = [&](auto dm, auto out_name) {
    MoFEMFunctionBegin;
    auto post_proc_fe =
        boost::make_shared<PostProcBrokenMeshInMoab<DomainEle>>(m_field);

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    auto l_vec = boost::make_shared<VectorDouble>();
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("L", l_vec));

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(

            post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

            {{"L", l_vec}},

            {},

            {}, {})

    );

    CHKERR DMoFEMLoopFiniteElements(dm, simple->getDomainFEName(),
                                    post_proc_fe);
    post_proc_fe->writeFile(out_name);
    MoFEMFunctionReturn(0);
  };

  CHKERR post_proc(sub_dm, "inital_level_set.h5m");

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode initialiseFieldVelocity(MoFEM::Interface &m_field) {
  MoFEMFunctionBegin;

  // get operators tester
  auto simple = m_field.getInterface<Simple>();

  auto sub_dm = createSmartDM(m_field.get_comm(), "DMMOFEM");
  CHKERR DMMoFEMCreateSubDM(sub_dm, simple->getDM(), "SUB_VELOCITY");
  CHKERR DMMoFEMSetDestroyProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMSetSquareProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMAddElement(sub_dm, simple->getDomainFEName());
  CHKERR DMMoFEMAddSubFieldRow(sub_dm, "V");
  CHKERR DMSetUp(sub_dm);

  auto lhs_fe = boost::make_shared<DomainEle>(m_field);
  auto rhs_fe = boost::make_shared<DomainEle>(m_field);

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      lhs_fe->getOpPtrVector(), {HCURL});
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      rhs_fe->getOpPtrVector(), {HCURL});

  lhs_fe->getOpPtrVector().push_back(new OpMassLL("L", "L"));
  rhs_fe->getOpPtrVector().push_back(new OpSourceL("L", get_level_set));

  boost::shared_ptr<FEMethod> null_fe;
  CHKERR DMMoFEMKSPSetComputeOperators(sub_dm, simple->getDomainFEName(),
                                       lhs_fe, null_fe, null_fe);
  CHKERR DMMoFEMKSPSetComputeRHS(sub_dm, simple->getDomainFEName(), rhs_fe,
                                 null_fe, null_fe);

  auto ksp = MoFEM::createKSP(m_field.get_comm());
  CHKERR KSPSetDM(ksp, sub_dm);
  CHKERR KSPSetFromOptions(ksp);
  CHKERR KSPSetUp(ksp);

  auto L = smartCreateDMVector(sub_dm);
  auto F = smartVectorDuplicate(L);

  CHKERR KSPSolve(ksp, F, L);
  CHKERR VecGhostUpdateBegin(L, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(L, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(sub_dm, L, INSERT_VALUES, SCATTER_REVERSE);

  auto post_proc = [&](auto dm, auto out_name) {
    MoFEMFunctionBegin;
    auto post_proc_fe =
        boost::make_shared<PostProcBrokenMeshInMoab<DomainEle>>(m_field);

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    auto l_vec = boost::make_shared<VectorDouble>();
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("L", l_vec));

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(

            post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

            {{"L", l_vec}},

            {},

            {}, {})

    );

    CHKERR DMoFEMLoopFiniteElements(dm, simple->getDomainFEName(),
                                    post_proc_fe);
    post_proc_fe->writeFile(out_name);
    MoFEMFunctionReturn(0);
  };

  CHKERR post_proc(sub_dm, "inital_level_set.h5m");

  MoFEMFunctionReturn(0);
}