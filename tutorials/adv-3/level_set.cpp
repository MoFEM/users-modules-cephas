/**
 * @file level_set.cpp
 * @example level_set.cpp
 * @brief Implementation DG upwind method for advection/level set problem
 * @date 2022-12-15
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <MoFEM.hpp>
using namespace MoFEM;

static char help[] = "...\n\n";

//! [Define dimension]
constexpr int SPACE_DIM = EXECUTABLE_DIMENSION;
constexpr AssemblyType A = AssemblyType::PETSC; //< selected assembly type
constexpr IntegrationType I =
    IntegrationType::GAUSS; //< selected integration type

using EntData = EntitiesFieldData::EntData;
using DomainEle = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::DomainEle;
using DomianParentEle =
    PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::DomianParentEle;
using BoundaryEle =
    PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::BoundaryEle;
using FaceSideEle =
    PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::FaceSideEle;

using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using FaceSideEleOp = FaceSideEle::UserDataOperator;

using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

constexpr FieldSpace potential_velocity_space = SPACE_DIM == 2 ? H1 : HCURL;
constexpr size_t potential_velocity_field_dim = SPACE_DIM == 2 ? 1 : 3;

constexpr bool debug = false;
constexpr int nb_levels = 3; //< number of refinement levels

constexpr int start_bit =
    nb_levels + 1; //< first refinement level for computational mesh

constexpr int current_bit =
    2 * start_bit + 1; ///< dofs bit used to do calculations
constexpr int skeleton_bit = 2 * start_bit + 2; ///< skeleton elemets bit
constexpr int aggregate_bit =
    2 * start_bit + 3; ///< all bits for advection problem
constexpr int projection_bit =
    2 * start_bit + 4; //< bit from which data are projected
constexpr int aggregate_projection_bit =
    2 * start_bit + 5; ///< all bits for projection problem

struct LevelSet {

  LevelSet(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  using VecSideArray = std::array<VectorDouble, 2>;
  using MatSideArray = std::array<MatrixDouble, 2>;

  /**
   * @brief data structure carrying information on skeleton on both sides.
   *
   */
  struct SideData {
    // data for skeleton computation
    std::array<EntityHandle, 2> feSideHandle;
    std::array<VectorInt, 2>
        indicesRowSideMap; ///< indices on rows for left hand-side
    std::array<VectorInt, 2>
        indicesColSideMap; ///< indices on columns for left hand-side
    std::array<MatrixDouble, 2> rowBaseSideMap; // base functions on rows
    std::array<MatrixDouble, 2> colBaseSideMap; // base function  on columns
    std::array<int, 2> senseMap; // orientation of local element edge/face in
                                 // respect to global orientation of edge/face

    VecSideArray lVec;   //< Values of level set field
    MatSideArray velMat; //< Values of velocity field

    int currentFESide; ///< current side counter
  };

  /**
   * @brief advection velocity field
   *
   * \note in current implementation is assumed that advection field has zero
   * normal component.
   *
   * \note function define a vector velocity potential field, curl of potential
   * field gives velocity, thus velocity is divergence free.
   *
   * @tparam SPACE_DIM
   * @param x
   * @param y
   * @param z
   * @return auto
   */
  template <int SPACE_DIM>
  static double get_velocity_potential(double x, double y, double z);

  /**
   * @brief inital level set, i.e. advected filed
   *
   * @param x
   * @param y
   * @param z
   * @return double
   */
  static double get_level_set(const double x, const double y, const double z);

  /**
   * @brief read mesh
   *
   * @return MoFEMErrorCode
   */
  MoFEMErrorCode readMesh();

  /**
   * @brief create fields, and set approximation order
   *
   * @return MoFEMErrorCode
   */
  MoFEMErrorCode setupProblem();

  /**
   * @brief push operators to integrate operators on domain
   *
   * @return MoFEMErrorCode
   */
  MoFEMErrorCode pushOpDomain();

  /**
   * @brief evaluate error
   *
   * @return MoFEMErrorCode
   */
  std::tuple<double, Tag> evaluateError();

  /**
   * @brief Get operator calculating velocity on coarse mesh
   *
   * @param vel_ptr
   * @return DomainEleOp*
   */
  ForcesAndSourcesCore::UserDataOperator *
  getZeroLevelVelOp(boost::shared_ptr<MatrixDouble> vel_ptr);

  /**
   * @brief create side element to assemble data from sides
   *
   * @param side_data_ptr
   * @return boost::shared_ptr<FaceSideEle>
   */
  boost::shared_ptr<FaceSideEle>
  getSideFE(boost::shared_ptr<SideData> side_data_ptr);

  /**
   * @brief push operator to integrate on skeleton
   *
   * @return MoFEMErrorCode
   */
  MoFEMErrorCode pushOpSkeleton();

  /**
   * @brief test integration side elements
   *
   * Check consistency between volume and skeleton integral.
   *
   * @return MoFEMErrorCode
   */
  MoFEMErrorCode testSideFE();

  /**
   * @brief test consistency between tangent matrix and the right hand side
   * vectors
   *
   * @return MoFEMErrorCode
   */
  MoFEMErrorCode testOp();

  /**
   * @brief initialise field set
   *
   * @param level_fun
   * @return MoFEMErrorCode
   */
  MoFEMErrorCode initialiseFieldLevelSet(
      boost::function<double(double, double, double)> level_fun =
          get_level_set);

  /**
   * @brief initialise potential velocity field
   *
   * @param vel_fun
   * @return MoFEMErrorCode
   */
  MoFEMErrorCode initialiseFieldVelocity(
      boost::function<double(double, double, double)> vel_fun =
          get_velocity_potential<SPACE_DIM>);

  /**
   * @brief dg level set projection
   *
   * @param prj_bit
   * @param mesh_bit
   * @return MoFEMErrorCode
   */
  MoFEMErrorCode dgProjection(const int prj_bit = projection_bit);

  /**
   * @brief solve advection problem
   *
   * @return * MoFEMErrorCode
   */
  MoFEMErrorCode solveAdvection();

  /**
   * @brief Wrapper executing stages while mesh refinement
   */
  struct WrapperClass {
    WrapperClass() = default;
    virtual MoFEMErrorCode setBits(LevelSet &level_set, int l) = 0;
    virtual MoFEMErrorCode runCalcs(LevelSet &level_set, int l) = 0;
    virtual MoFEMErrorCode setAggregateBit(LevelSet &level_set, int l) = 0;
    virtual double getThreshold(const double max) = 0;
  };

  /**
   * @brief Used to execute inital mesh approximation while mesh refinement
   * 
   */
  struct WrapperClassInitalSolution : public WrapperClass {

    WrapperClassInitalSolution(boost::shared_ptr<double> max_ptr)
        : WrapperClass(), maxPtr(max_ptr) {}

    MoFEMErrorCode setBits(LevelSet &level_set, int l) {
      MoFEMFunctionBegin;
      auto simple = level_set.mField.getInterface<Simple>();
      simple->getBitRefLevel() =
          BitRefLevel().set(skeleton_bit) | BitRefLevel().set(aggregate_bit);
      simple->getBitRefLevelMask() = BitRefLevel().set();
      simple->reSetUp(true);
      MoFEMFunctionReturn(0);
    };

    MoFEMErrorCode runCalcs(LevelSet &level_set, int l) {
      MoFEMFunctionBegin;
      CHKERR level_set.initialiseFieldLevelSet();
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode setAggregateBit(LevelSet &level_set, int l) {
      auto bit_mng = level_set.mField.getInterface<BitRefManager>();
      auto set_bit = [](auto l) { return BitRefLevel().set(l); };
      MoFEMFunctionBegin;
      Range level;
      CHKERR bit_mng->getEntitiesByRefLevel(set_bit(start_bit + l),
                                            BitRefLevel().set(), level);
      CHKERR level_set.mField.getInterface<CommInterface>()
          ->synchroniseEntities(level);
      CHKERR bit_mng->setNthBitRefLevel(current_bit, false);
      CHKERR bit_mng->setNthBitRefLevel(level, current_bit, true);
      CHKERR bit_mng->setNthBitRefLevel(level, aggregate_bit, true);
      MoFEMFunctionReturn(0);
    }

    double getThreshold(const double max) {
      *maxPtr = std::max(*maxPtr, max);
      return 0.05 * (*maxPtr);
    }

    private:
      boost::shared_ptr<double> maxPtr;
  };

  /**
   * @brief Use peculated errors on all levels while mesh projection
   * 
   */
  struct WrapperClassErrorProjection : public WrapperClass {
      WrapperClassErrorProjection(boost::shared_ptr<double> max_ptr)
          : maxPtr(max_ptr) {}

      MoFEMErrorCode setBits(LevelSet &level_set, int l) { return 0; };
      MoFEMErrorCode runCalcs(LevelSet &level_set, int l) { return 0; }
      MoFEMErrorCode setAggregateBit(LevelSet &level_set, int l) {
      auto bit_mng = level_set.mField.getInterface<BitRefManager>();
      auto set_bit = [](auto l) { return BitRefLevel().set(l); };
      MoFEMFunctionBegin;
      Range level;
      CHKERR bit_mng->getEntitiesByRefLevel(set_bit(start_bit + l),
                                            BitRefLevel().set(), level);
      CHKERR level_set.mField.getInterface<CommInterface>()
          ->synchroniseEntities(level);
      CHKERR bit_mng->setNthBitRefLevel(current_bit, false);
      CHKERR bit_mng->setNthBitRefLevel(level, current_bit, true);
      CHKERR bit_mng->setNthBitRefLevel(level, aggregate_bit, true);
      MoFEMFunctionReturn(0);
    }
    double getThreshold(const double max) { return 0.05 * (*maxPtr); }

  private:
    boost::shared_ptr<double> maxPtr;

  };

  MoFEMErrorCode refineMesh(WrapperClass &&wp);

  struct OpRhsDomain;   ///< integrate volume operators on rhs
  struct OpLhsDomain;   ///< integrate volume operator on lhs
  struct OpRhsSkeleton; ///< integrate skeleton operators on rhs
  struct OpLhsSkeleton; ///< integrate skeleton operators on khs

  // Main interfaces
  MoFEM::Interface &mField;

  using AssemblyDomainEleOp =
      FormsIntegrators<DomainEleOp>::Assembly<A>::OpBase;
  using OpMassLL =
      FormsIntegrators<DomainEleOp>::Assembly<A>::BiLinearForm<I>::OpMass<1, 1>;
  using OpSourceL =
      FormsIntegrators<DomainEleOp>::Assembly<A>::LinearForm<I>::OpSource<1, 1>;
  using OpMassVV = FormsIntegrators<DomainEleOp>::Assembly<A>::BiLinearForm<
      I>::OpMass<potential_velocity_field_dim, potential_velocity_field_dim>;
  using OpSourceV = FormsIntegrators<DomainEleOp>::Assembly<A>::LinearForm<
      I>::OpSource<potential_velocity_field_dim, potential_velocity_field_dim>;
  using OpScalarFieldL = FormsIntegrators<DomainEleOp>::Assembly<A>::LinearForm<
      I>::OpBaseTimesScalar<1>;

  using AssemblyBoundaryEleOp =
      FormsIntegrators<BoundaryEleOp>::Assembly<A>::OpBase;

  enum ElementSide { LEFT_SIDE = 0, RIGHT_SIDE = 1 };

private:
  boost::shared_ptr<double> maxPtr;

};

template <>
double LevelSet::get_velocity_potential<2>(double x, double y, double z) {
  return (x * x - 0.25) * (y * y - 0.25);
}

double LevelSet::get_level_set(const double x, const double y, const double z) {
  constexpr double xc = 0.1;
  constexpr double yc = 0.;
  constexpr double zc = 0.;
  constexpr double r = 0.2;
  return std::sqrt(pow(x - xc, 2) + pow(y - yc, 2) + pow(z - zc, 2)) - r;
}

MoFEMErrorCode LevelSet::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();

  if constexpr (debug) {
    CHKERR testSideFE();
    CHKERR testOp();
  }

  CHKERR initialiseFieldVelocity();

  maxPtr = boost::make_shared<double>(0);
  CHKERR refineMesh(WrapperClassInitalSolution(maxPtr));

  auto simple = mField.getInterface<Simple>();
  simple->getBitRefLevel() = BitRefLevel().set(skeleton_bit) |
                             BitRefLevel().set(aggregate_bit);
  simple->getBitRefLevelMask() = BitRefLevel().set();
  simple->reSetUp(true);

  CHKERR solveAdvection();

  MoFEMFunctionReturn(0);
}

struct LevelSet::OpRhsDomain : public AssemblyDomainEleOp {

  OpRhsDomain(const std::string field_name,
              boost::shared_ptr<VectorDouble> l_ptr,
              boost::shared_ptr<VectorDouble> l_dot_ptr,
              boost::shared_ptr<MatrixDouble> vel_ptr);
  MoFEMErrorCode iNtegrate(EntData &data);

private:
  boost::shared_ptr<VectorDouble> lPtr;
  boost::shared_ptr<VectorDouble> lDotPtr;
  boost::shared_ptr<MatrixDouble> velPtr;
};

struct LevelSet::OpLhsDomain : public AssemblyDomainEleOp {
  OpLhsDomain(const std::string field_name,
              boost::shared_ptr<MatrixDouble> vel_ptr);
  MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);

private:
  boost::shared_ptr<MatrixDouble> velPtr;
};

struct LevelSet::OpRhsSkeleton : public BoundaryEleOp {

  OpRhsSkeleton(boost::shared_ptr<SideData> side_data_ptr,
                boost::shared_ptr<FaceSideEle> side_fe_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        EntitiesFieldData::EntData &data);

private:
  boost::shared_ptr<SideData> sideDataPtr;
  boost::shared_ptr<FaceSideEle>
      sideFEPtr; ///< pointer to element to get data on edge/face sides

  VectorDouble resSkelton;
};

struct LevelSet::OpLhsSkeleton : public BoundaryEleOp {
  OpLhsSkeleton(boost::shared_ptr<SideData> side_data_ptr,
                boost::shared_ptr<FaceSideEle> side_fe_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        EntitiesFieldData::EntData &data);

private:
  boost::shared_ptr<SideData> sideDataPtr;
  boost::shared_ptr<FaceSideEle>
      sideFEPtr; ///< pointer to element to get data on edge/face sides

  MatrixDouble matSkeleton;
};

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

    LevelSet level_set(m_field);
    CHKERR level_set.runProblem();
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}

MoFEMErrorCode LevelSet::readMesh() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  // get options from command line
  CHKERR simple->getOptions();

  // Only L2 field is set in this example. Two lines bellow forces simple
  // interface to creat lower dimension (edge) elements, despite that fact that
  // there is no field spanning on such elements. We need them for DG method.
  simple->getAddSkeletonFE() = true;
  simple->getAddBoundaryFE() = true;

  // load mesh file
  simple->getBitRefLevel() = BitRefLevel();
  CHKERR simple->loadFile();

  // Initialise bit ref levels
  auto set_problem_bit = [&]() {
    MoFEMFunctionBegin;
    auto bit0 = BitRefLevel().set(start_bit);
    BitRefLevel start_mask;
    for (auto s = 0; s != start_bit; ++s)
      start_mask[s] = true;

    auto bit_mng = mField.getInterface<BitRefManager>();

    Range level0;
    CHKERR bit_mng->getEntitiesByRefLevel(BitRefLevel().set(0),
                                          BitRefLevel().set(), level0);
                                          
    CHKERR bit_mng->setNthBitRefLevel(level0, current_bit, true);
    CHKERR bit_mng->setNthBitRefLevel(level0, aggregate_bit, true);
    CHKERR bit_mng->setNthBitRefLevel(level0, skeleton_bit, true);

    // Set bits to build adjacencies between parents and children. That is used
    // by simple interface.
    simple->getBitAdjEnt() = BitRefLevel().set();
    simple->getBitAdjParent() = BitRefLevel().set();
    simple->getBitRefLevel() = BitRefLevel().set(current_bit);
    simple->getBitRefLevelMask() = BitRefLevel().set();

#ifndef NDEBUG
    if constexpr (debug) {
      auto proc_str = boost::lexical_cast<std::string>(mField.get_comm_rank());
      CHKERR bit_mng->writeBitLevelByDim(
          BitRefLevel().set(0), BitRefLevel().set(), SPACE_DIM,
          (proc_str + "level_base.vtk").c_str(), "VTK", "");
    }
#endif

    MoFEMFunctionReturn(0);
  };

  CHKERR set_problem_bit();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode LevelSet::setupProblem() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  // Scalar fields and vector field is tested. Add more fields, i.e. vector
  // field if needed.
  CHKERR simple->addDomainField("L", L2, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple->addDomainField("V", potential_velocity_space,
                                AINSWORTH_LEGENDRE_BASE, 1);

  // set fields order, i.e. for most first cases order is sufficient.
  CHKERR simple->setFieldOrder("L", 4);
  CHKERR simple->setFieldOrder("V", 4);

  // setup problem
  CHKERR simple->setUp();

  MoFEMFunctionReturn(0);
}

FTensor::Index<'i', SPACE_DIM> i;
FTensor::Index<'j', SPACE_DIM> j;
FTensor::Index<'k', SPACE_DIM> k;

LevelSet::OpRhsDomain::OpRhsDomain(const std::string field_name,
                                   boost::shared_ptr<VectorDouble> l_ptr,
                                   boost::shared_ptr<VectorDouble> l_dot_ptr,
                                   boost::shared_ptr<MatrixDouble> vel_ptr)
    : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
      lPtr(l_ptr), lDotPtr(l_dot_ptr), velPtr(vel_ptr) {}

LevelSet::OpLhsDomain::OpLhsDomain(const std::string field_name,
                                   boost::shared_ptr<MatrixDouble> vel_ptr)
    : AssemblyDomainEleOp(field_name, field_name,
                          AssemblyDomainEleOp::OPROWCOL),
      velPtr(vel_ptr) {
  this->sYmm = false;
}

LevelSet::OpRhsSkeleton::OpRhsSkeleton(
    boost::shared_ptr<SideData> side_data_ptr,
    boost::shared_ptr<FaceSideEle> side_fe_ptr)
    : BoundaryEleOp(NOSPACE, BoundaryEleOp::OPSPACE),
      sideDataPtr(side_data_ptr), sideFEPtr(side_fe_ptr) {}

LevelSet::OpLhsSkeleton::OpLhsSkeleton(
    boost::shared_ptr<SideData> side_data_ptr,
    boost::shared_ptr<FaceSideEle> side_fe_ptr)
    : BoundaryEleOp(NOSPACE, BoundaryEleOp::OPSPACE),
      sideDataPtr(side_data_ptr), sideFEPtr(side_fe_ptr) {}

MoFEMErrorCode LevelSet::OpRhsDomain::iNtegrate(EntData &data) {
  MoFEMFunctionBegin;

  const auto nb_int_points = getGaussPts().size2();
  const auto nb_dofs = data.getIndices().size();
  const auto nb_base_func = data.getN().size2();

  auto t_l = getFTensor0FromVec(*lPtr);
  auto t_l_dot = getFTensor0FromVec(*lDotPtr);
  auto t_vel = getFTensor1FromMat<SPACE_DIM>(*velPtr);

  auto t_base = data.getFTensor0N();
  auto t_diff_base = data.getFTensor1DiffN<SPACE_DIM>();

  auto t_w = getFTensor0IntegrationWeight();
  for (auto gg = 0; gg != nb_int_points; ++gg) {
    const auto alpha = t_w * getMeasure();
    auto res0 = alpha * t_l_dot;
    FTensor::Tensor1<double, SPACE_DIM> t_res1;
    t_res1(i) = (alpha * t_l) * t_vel(i);
    ++t_w;
    ++t_l;
    ++t_l_dot;
    ++t_vel;

    auto &nf = this->locF;

    int rr = 0;
    for (; rr != nb_dofs; ++rr) {
      nf(rr) += res0 * t_base - t_res1(i) * t_diff_base(i);
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

MoFEMErrorCode LevelSet::OpLhsDomain::iNtegrate(EntData &row_data,
                                                EntData &col_data) {
  MoFEMFunctionBegin;

  const auto nb_int_points = getGaussPts().size2();
  const auto nb_base_func = row_data.getN().size2();
  const auto nb_row_dofs = row_data.getIndices().size();
  const auto nb_col_dofs = col_data.getIndices().size();

  auto t_vel = getFTensor1FromMat<SPACE_DIM>(*velPtr);

  auto t_row_base = row_data.getFTensor0N();
  auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();

  auto t_w = getFTensor0IntegrationWeight();
  for (auto gg = 0; gg != nb_int_points; ++gg) {
    const auto alpha = t_w * getMeasure();
    const auto beta = alpha * getTSa();
    ++t_w;

    auto &mat = this->locMat;

    int rr = 0;
    for (; rr != nb_row_dofs; ++rr) {
      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (int cc = 0; cc != nb_col_dofs; ++cc) {
        mat(rr, cc) +=
            (beta * t_row_base - alpha * (t_row_diff_base(i) * t_vel(i))) *
            t_col_base;
        ++t_col_base;
      }
      ++t_row_base;
      ++t_row_diff_base;
    }
    for (; rr < nb_base_func; ++rr) {
      ++t_row_base;
      ++t_row_diff_base;
    }

    ++t_vel;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
LevelSet::OpRhsSkeleton::doWork(int side, EntityType type,
                                EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  // Collect data from side domain elements
  CHKERR loopSideFaces("dFE", *sideFEPtr);
  const auto in_the_loop =
      sideFEPtr->nInTheLoop; // return number of elements on the side

  auto not_side = [](auto s) {
    return s == LEFT_SIDE ? RIGHT_SIDE : LEFT_SIDE;
  };

  auto get_ntensor = [](auto &base_mat) {
    return FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(
        &*base_mat.data().begin());
  };

  if (in_the_loop > 0) {

    // get normal of the face or edge
    auto t_normal = getFTensor1Normal();
    const auto nb_gauss_pts = getGaussPts().size2();

    for (auto s0 : {LEFT_SIDE, RIGHT_SIDE}) {

      // gent number of DOFs on the right side.
      const auto nb_rows = sideDataPtr->indicesRowSideMap[s0].size();

      if (nb_rows) {

        resSkelton.resize(nb_rows, false);
        resSkelton.clear();

        // get orientation of the local element edge
        const auto opposite_s0 = not_side(s0);
        const auto sense_row = sideDataPtr->senseMap[s0];
#ifndef NDEBUG
        const auto opposite_sense_row = sideDataPtr->senseMap[opposite_s0];
        if (sense_row * opposite_sense_row > 0)
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                  "Should be opposite sign");
#endif

        // iterate the side cols
        const auto nb_row_base_functions =
            sideDataPtr->rowBaseSideMap[s0].size2();

        auto t_w = getFTensor0IntegrationWeight();
        auto arr_t_l =
            make_array(getFTensor0FromVec(sideDataPtr->lVec[LEFT_SIDE]),
                       getFTensor0FromVec(sideDataPtr->lVec[RIGHT_SIDE]));
        auto arr_t_vel = make_array(
            getFTensor1FromMat<SPACE_DIM>(sideDataPtr->velMat[LEFT_SIDE]),
            getFTensor1FromMat<SPACE_DIM>(sideDataPtr->velMat[RIGHT_SIDE]));

        auto next = [&]() {
          for (auto &t_l : arr_t_l)
            ++t_l;
          for (auto &t_vel : arr_t_vel)
            ++t_vel;
        };

#ifndef NDEBUG
        if (nb_gauss_pts != sideDataPtr->rowBaseSideMap[s0].size1())
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                  "Inconsistent number of DOFs");
#endif

        auto t_row_base = get_ntensor(sideDataPtr->rowBaseSideMap[s0]);
        for (int gg = 0; gg != nb_gauss_pts; ++gg) {
          FTensor::Tensor1<double, SPACE_DIM> t_vel;
          t_vel(i) = (arr_t_vel[LEFT_SIDE](i) + arr_t_vel[RIGHT_SIDE](i)) / 2.;
          const auto dot = sense_row * (t_normal(i) * t_vel(i));
          const auto l_upwind_side = (dot > 0) ? s0 : opposite_s0;
          const auto l_upwind = arr_t_l[l_upwind_side];
          const auto res = t_w * dot * l_upwind;
          next();
          ++t_w;
          auto rr = 0;
          for (; rr != nb_rows; ++rr) {
            resSkelton[rr] += t_row_base * res;
            ++t_row_base;
          }
          for (; rr < nb_row_base_functions; ++rr) {
            ++t_row_base;
          }
        }
        // assemble local operator vector to global vector
        CHKERR ::VecSetValues(getTSf(),
                              sideDataPtr->indicesRowSideMap[s0].size(),
                              &*sideDataPtr->indicesRowSideMap[s0].begin(),
                              &*resSkelton.begin(), ADD_VALUES);
      }
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
LevelSet::OpLhsSkeleton::doWork(int side, EntityType type,
                                EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  // Collect data from side domain elements
  CHKERR loopSideFaces("dFE", *sideFEPtr);
  const auto in_the_loop =
      sideFEPtr->nInTheLoop; // return number of elements on the side

  auto not_side = [](auto s) {
    return s == LEFT_SIDE ? RIGHT_SIDE : LEFT_SIDE;
  };

  auto get_ntensor = [](auto &base_mat) {
    return FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(
        &*base_mat.data().begin());
  };

  if (in_the_loop > 0) {

    // get normal of the face or edge
    auto t_normal = getFTensor1Normal();
    const auto nb_gauss_pts = getGaussPts().size2();

    for (auto s0 : {LEFT_SIDE, RIGHT_SIDE}) {

      // gent number of DOFs on the right side.
      const auto nb_rows = sideDataPtr->indicesRowSideMap[s0].size();

      if (nb_rows) {

        // get orientation of the local element edge
        const auto opposite_s0 = not_side(s0);
        const auto sense_row = sideDataPtr->senseMap[s0];

        // iterate the side cols
        const auto nb_row_base_functions =
            sideDataPtr->rowBaseSideMap[s0].size2();

        for (auto s1 : {LEFT_SIDE, RIGHT_SIDE}) {

          // gent number of DOFs on the right side.
          const auto nb_cols = sideDataPtr->indicesColSideMap[s1].size();
          const auto sense_col = sideDataPtr->senseMap[s1];

          // resize local element matrix
          matSkeleton.resize(nb_rows, nb_cols, false);
          matSkeleton.clear();

          auto t_w = getFTensor0IntegrationWeight();
          auto arr_t_vel = make_array(
              getFTensor1FromMat<SPACE_DIM>(sideDataPtr->velMat[LEFT_SIDE]),
              getFTensor1FromMat<SPACE_DIM>(sideDataPtr->velMat[RIGHT_SIDE]));

          auto next = [&]() {
            for (auto &t_vel : arr_t_vel)
              ++t_vel;
          };

          auto t_row_base = get_ntensor(sideDataPtr->rowBaseSideMap[s0]);
          for (int gg = 0; gg != nb_gauss_pts; ++gg) {
            FTensor::Tensor1<double, SPACE_DIM> t_vel;
            t_vel(i) =
                (arr_t_vel[LEFT_SIDE](i) + arr_t_vel[RIGHT_SIDE](i)) / 2.;
            const auto dot = sense_row * (t_normal(i) * t_vel(i));
            const auto l_upwind_side = (dot > 0) ? s0 : opposite_s0;
            const auto sense_upwind = sideDataPtr->senseMap[l_upwind_side];
            auto res = t_w * dot; // * sense_row * sense_upwind;
            next();
            ++t_w;
            auto rr = 0;
            if (s1 == l_upwind_side) {
              for (; rr != nb_rows; ++rr) {
                auto get_ntensor = [](auto &base_mat, auto gg, auto bb) {
                  double *ptr = &base_mat(gg, bb);
                  return FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(ptr);
                };
                auto t_col_base =
                    get_ntensor(sideDataPtr->colBaseSideMap[s1], gg, 0);
                const auto res_row = res * t_row_base;
                ++t_row_base;
                // iterate columns
                for (size_t cc = 0; cc != nb_cols; ++cc) {
                  matSkeleton(rr, cc) += res_row * t_col_base;
                  ++t_col_base;
                }
              }
            }
            for (; rr < nb_row_base_functions; ++rr) {
              ++t_row_base;
            }
          }
          // assemble system
          CHKERR ::MatSetValues(getTSB(),
                                sideDataPtr->indicesRowSideMap[s0].size(),
                                &*sideDataPtr->indicesRowSideMap[s0].begin(),
                                sideDataPtr->indicesColSideMap[s1].size(),
                                &*sideDataPtr->indicesColSideMap[s1].begin(),
                                &*matSkeleton.data().begin(), ADD_VALUES);
        }
      }
    }
  }
  MoFEMFunctionReturn(0);
}

ForcesAndSourcesCore::UserDataOperator *
LevelSet::getZeroLevelVelOp(boost::shared_ptr<MatrixDouble> vel_ptr) {
  auto get_parent_vel_this = [&]() {
    auto parent_fe_ptr = boost::make_shared<DomianParentEle>(mField);
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        parent_fe_ptr->getOpPtrVector(), {potential_velocity_space});
    parent_fe_ptr->getOpPtrVector().push_back(
        new OpCalculateHcurlVectorCurl<potential_velocity_field_dim, SPACE_DIM>(
            "V", vel_ptr));
    return parent_fe_ptr;
  };

  auto get_parents_vel_fe_ptr = [&](auto this_fe_ptr) {
    std::vector<boost::shared_ptr<DomianParentEle>> parents_elems_ptr_vec;
    for (int l = 0; l <= nb_levels; ++l)
      parents_elems_ptr_vec.emplace_back(
          boost::make_shared<DomianParentEle>(mField));
    for (auto l = 1; l <= nb_levels; ++l) {
      parents_elems_ptr_vec[l - 1]->getOpPtrVector().push_back(
          new OpRunParent(parents_elems_ptr_vec[l], BitRefLevel().set(),
                          BitRefLevel().set(0).flip(), this_fe_ptr,
                          BitRefLevel().set(0), BitRefLevel().set()));
    }
    return parents_elems_ptr_vec[0];
  };

  auto this_fe_ptr = get_parent_vel_this();
  auto parent_fe_ptr = get_parents_vel_fe_ptr(this_fe_ptr);
  return new OpRunParent(parent_fe_ptr, BitRefLevel().set(),
                         BitRefLevel().set(0).flip(), this_fe_ptr,
                         BitRefLevel().set(0), BitRefLevel().set());
}

MoFEMErrorCode LevelSet::pushOpDomain() {
  MoFEMFunctionBegin;
  auto pip = mField.getInterface<PipelineManager>(); // get interface to
                                                     // pipeline manager

  pip->getOpDomainLhsPipeline().clear();
  pip->getOpDomainRhsPipeline().clear();

  pip->setDomainLhsIntegrationRule([](int, int, int o) { return 3 * o; });
  pip->setDomainRhsIntegrationRule([](int, int, int o) { return 3 * o; });

  pip->getDomainLhsFE()->exeTestHook = [&](FEMethod *fe_ptr) {
    return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
        current_bit);
  };
  pip->getDomainRhsFE()->exeTestHook = [&](FEMethod *fe_ptr) {
    return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
        current_bit);
  };

  auto l_ptr = boost::make_shared<VectorDouble>();
  auto l_dot_ptr = boost::make_shared<VectorDouble>();
  auto vel_ptr = boost::make_shared<MatrixDouble>();

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpDomainRhsPipeline(), {potential_velocity_space, L2});
  pip->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldValues("L", l_ptr));
  pip->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldValuesDot("L", l_dot_ptr));
  pip->getOpDomainRhsPipeline().push_back(getZeroLevelVelOp(vel_ptr));
  pip->getOpDomainRhsPipeline().push_back(
      new OpRhsDomain("L", l_ptr, l_dot_ptr, vel_ptr));

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpDomainLhsPipeline(), {potential_velocity_space, L2});
  pip->getOpDomainLhsPipeline().push_back(getZeroLevelVelOp(vel_ptr));
  pip->getOpDomainLhsPipeline().push_back(new OpLhsDomain("L", vel_ptr));

  MoFEMFunctionReturn(0);
}

boost::shared_ptr<FaceSideEle>
LevelSet::getSideFE(boost::shared_ptr<SideData> side_data_ptr) {

  auto simple = mField.getInterface<Simple>();

  auto l_ptr = boost::make_shared<VectorDouble>();
  auto vel_ptr = boost::make_shared<MatrixDouble>();

  struct OpSideData : public FaceSideEleOp {
    OpSideData(boost::shared_ptr<SideData> side_data_ptr)
        : FaceSideEleOp("L", "L", FaceSideEleOp::OPROWCOL),
          sideDataPtr(side_data_ptr) {
      std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
      for (auto t = moab::CN::TypeDimensionMap[SPACE_DIM].first;
           t <= moab::CN::TypeDimensionMap[SPACE_DIM].second; ++t)
        doEntities[t] = true;
      sYmm = false;
    }

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data) {
      MoFEMFunctionBegin;
      if ((CN::Dimension(row_type) == SPACE_DIM) &&
          (CN::Dimension(col_type) == SPACE_DIM)) {

        auto reset = [&](auto nb_in_loop) {
          sideDataPtr->feSideHandle[nb_in_loop] = 0;
          sideDataPtr->indicesRowSideMap[nb_in_loop].clear();
          sideDataPtr->indicesColSideMap[nb_in_loop].clear();
          sideDataPtr->rowBaseSideMap[nb_in_loop].clear();
          sideDataPtr->colBaseSideMap[nb_in_loop].clear();
          sideDataPtr->senseMap[nb_in_loop] = 0;
        };

        const auto nb_in_loop = getFEMethod()->nInTheLoop;
        if (nb_in_loop == 0)
          for (auto s : {0, 1})
            reset(s);

        sideDataPtr->currentFESide = nb_in_loop;
        sideDataPtr->senseMap[nb_in_loop] = getSkeletonSense();

      } else {
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Should not happen");
      }

      MoFEMFunctionReturn(0);
    };

  private:
    boost::shared_ptr<SideData> sideDataPtr;
  };

  struct OpSideDataOnParent : public DomainEleOp {

    OpSideDataOnParent(boost::shared_ptr<SideData> side_data_ptr,
                       boost::shared_ptr<VectorDouble> l_ptr,
                       boost::shared_ptr<MatrixDouble> vel_ptr)
        : DomainEleOp("L", "L", DomainEleOp::OPROWCOL),
          sideDataPtr(side_data_ptr), lPtr(l_ptr), velPtr(vel_ptr) {
      std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
      for (auto t = moab::CN::TypeDimensionMap[SPACE_DIM].first;
           t <= moab::CN::TypeDimensionMap[SPACE_DIM].second; ++t)
        doEntities[t] = true;
      sYmm = false;
    }

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data) {
      MoFEMFunctionBegin;

      if ((CN::Dimension(row_type) == SPACE_DIM) &&
          (CN::Dimension(col_type) == SPACE_DIM)) {
        const auto nb_in_loop = sideDataPtr->currentFESide;
        sideDataPtr->feSideHandle[nb_in_loop] = getFEEntityHandle();
        sideDataPtr->indicesRowSideMap[nb_in_loop] = row_data.getIndices();
        sideDataPtr->indicesColSideMap[nb_in_loop] = col_data.getIndices();
        sideDataPtr->rowBaseSideMap[nb_in_loop] = row_data.getN();
        sideDataPtr->colBaseSideMap[nb_in_loop] = col_data.getN();
        (sideDataPtr->lVec)[nb_in_loop] = *lPtr;
        (sideDataPtr->velMat)[nb_in_loop] = *velPtr;

#ifndef NDEBUG
        if ((sideDataPtr->lVec)[nb_in_loop].size() !=
            (sideDataPtr->velMat)[nb_in_loop].size2())
          SETERRQ2(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                   "Wrong number of integaration pts %d != %d",
                   (sideDataPtr->lVec)[nb_in_loop].size(),
                   (sideDataPtr->velMat)[nb_in_loop].size2());
        if ((sideDataPtr->velMat)[nb_in_loop].size1() != SPACE_DIM)
          SETERRQ1(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                   "Wrong size of velocity vector size = %d",
                   (sideDataPtr->velMat)[nb_in_loop].size1());
#endif

        if (!nb_in_loop) {
          (sideDataPtr->lVec)[1] = sideDataPtr->lVec[0];
          (sideDataPtr->velMat)[1] = (sideDataPtr->velMat)[0];
        } else {
#ifndef NDEBUG
          if (sideDataPtr->rowBaseSideMap[0].size1() !=
              sideDataPtr->rowBaseSideMap[1].size1()) {
            SETERRQ2(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                     "Wrong number of integration pt %d != %d",
                     sideDataPtr->rowBaseSideMap[0].size1(),
                     sideDataPtr->rowBaseSideMap[1].size1());
          }
          if (sideDataPtr->colBaseSideMap[0].size1() !=
              sideDataPtr->colBaseSideMap[1].size1()) {
            SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                    "Wrong number of integration pt");
          }
#endif
        }

      } else {
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Should not happen");
      }

      MoFEMFunctionReturn(0);
    };

  private:
    boost::shared_ptr<SideData> sideDataPtr;
    boost::shared_ptr<VectorDouble> lPtr;
    boost::shared_ptr<MatrixDouble> velPtr;
  };

  // Calculate fields on param mesh bit element
  auto get_parent_this = [&]() {
    auto parent_fe_ptr = boost::make_shared<DomianParentEle>(mField);
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        parent_fe_ptr->getOpPtrVector(), {potential_velocity_space, L2});
    parent_fe_ptr->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("L", l_ptr));
    parent_fe_ptr->getOpPtrVector().push_back(
        new OpSideDataOnParent(side_data_ptr, l_ptr, vel_ptr));
    return parent_fe_ptr;
  };

  auto get_parents_fe_ptr = [&](auto this_fe_ptr) {
    std::vector<boost::shared_ptr<DomianParentEle>> parents_elems_ptr_vec;
    for (int l = 0; l <= nb_levels; ++l)
      parents_elems_ptr_vec.emplace_back(
          boost::make_shared<DomianParentEle>(mField));
    for (auto l = 1; l <= nb_levels; ++l) {
      parents_elems_ptr_vec[l - 1]->getOpPtrVector().push_back(
          new OpRunParent(parents_elems_ptr_vec[l], BitRefLevel().set(),
                          BitRefLevel().set(current_bit).flip(), this_fe_ptr,
                          BitRefLevel().set(current_bit), BitRefLevel().set()));
    }
    return parents_elems_ptr_vec[0];
  };

  // Create aliased shared pointers, all elements are destroyed if side_fe_ptr
  // is destroyed
  auto get_side_fe_ptr = [&]() {
    auto side_fe_ptr = boost::make_shared<FaceSideEle>(mField);

    auto this_fe_ptr = get_parent_this();
    auto parent_fe_ptr = get_parents_fe_ptr(this_fe_ptr);

    side_fe_ptr->getOpPtrVector().push_back(new OpSideData(side_data_ptr));
    side_fe_ptr->getOpPtrVector().push_back(getZeroLevelVelOp(vel_ptr));
    side_fe_ptr->getOpPtrVector().push_back(
        new OpRunParent(parent_fe_ptr, BitRefLevel().set(),
                        BitRefLevel().set(current_bit).flip(), this_fe_ptr,
                        BitRefLevel().set(current_bit), BitRefLevel().set()));

    return side_fe_ptr;
  };

  return get_side_fe_ptr();
};

MoFEMErrorCode LevelSet::pushOpSkeleton() {
  MoFEMFunctionBegin;
  auto pip = mField.getInterface<PipelineManager>(); // get interface to

  pip->getOpSkeletonLhsPipeline().clear();
  pip->getOpSkeletonRhsPipeline().clear();

  pip->setSkeletonLhsIntegrationRule([](int, int, int o) { return 18; });
  pip->setSkeletonRhsIntegrationRule([](int, int, int o) { return 18; });

  pip->getSkeletonLhsFE()->exeTestHook = [&](FEMethod *fe_ptr) {
    return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
        skeleton_bit);
  };
  pip->getSkeletonRhsFE()->exeTestHook = [&](FEMethod *fe_ptr) {
    return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
        skeleton_bit);
  };

  auto side_data_ptr = boost::make_shared<SideData>();
  auto side_fe_ptr = getSideFE(side_data_ptr);

  pip->getOpSkeletonRhsPipeline().push_back(
      new OpRhsSkeleton(side_data_ptr, side_fe_ptr));
  pip->getOpSkeletonLhsPipeline().push_back(
      new OpLhsSkeleton(side_data_ptr, side_fe_ptr));

  MoFEMFunctionReturn(0);
}

std::tuple<double, Tag> LevelSet::evaluateError() {

  struct OpErrorSkel : BoundaryEleOp {

    OpErrorSkel(boost::shared_ptr<FaceSideEle> side_fe_ptr,
                boost::shared_ptr<SideData> side_data_ptr,
                SmartPetscObj<Vec> error_sum_ptr, Tag th_error)
        : BoundaryEleOp(NOSPACE, BoundaryEleOp::OPSPACE),
          sideFEPtr(side_fe_ptr), sideDataPtr(side_data_ptr),
          errorSumPtr(error_sum_ptr), thError(th_error) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      // Collect data from side domain elements
      CHKERR loopSideFaces("dFE", *sideFEPtr);
      const auto in_the_loop =
          sideFEPtr->nInTheLoop; // return number of elements on the side

      auto not_side = [](auto s) {
        return s == LEFT_SIDE ? RIGHT_SIDE : LEFT_SIDE;
      };

      auto nb_gauss_pts = getGaussPts().size2();

      for (auto s : {LEFT_SIDE, RIGHT_SIDE}) {

        auto arr_t_l =
            make_array(getFTensor0FromVec(sideDataPtr->lVec[LEFT_SIDE]),
                       getFTensor0FromVec(sideDataPtr->lVec[RIGHT_SIDE]));
        auto arr_t_vel = make_array(
            getFTensor1FromMat<SPACE_DIM>(sideDataPtr->velMat[LEFT_SIDE]),
            getFTensor1FromMat<SPACE_DIM>(sideDataPtr->velMat[RIGHT_SIDE]));

        auto next = [&]() {
          for (auto &t_l : arr_t_l)
            ++t_l;
          for (auto &t_vel : arr_t_vel)
            ++t_vel;
        };

        double e = 0;
        auto t_w = getFTensor0IntegrationWeight();
        for (int gg = 0; gg != nb_gauss_pts; ++gg) {
          e += t_w * getMeasure() *
               pow(arr_t_l[LEFT_SIDE] - arr_t_l[RIGHT_SIDE], 2);
          next();
          ++t_w;
        }
        e = std::sqrt(e);

        moab::Interface &moab =
            getNumeredEntFiniteElementPtr()->getBasicDataPtr()->moab;
        const void *tags_ptr[2];
        CHKERR moab.tag_get_by_ptr(thError, sideDataPtr->feSideHandle.data(), 2,
                                   tags_ptr);
        for (auto ff : {0, 1}) {
          *((double *)tags_ptr[ff]) += e;
        }
        CHKERR VecSetValue(errorSumPtr, 0, e, ADD_VALUES);
      };

      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<FaceSideEle> sideFEPtr;
    boost::shared_ptr<SideData> sideDataPtr;
    SmartPetscObj<Vec> errorSumPtr;
    Tag thError;
  };

  auto simple = mField.getInterface<Simple>();

  auto error_sum_ptr = createSmartVectorMPI(mField.get_comm(), PETSC_DECIDE, 1);
  Tag th_error;
  double def_val = 0;
  CHKERR mField.get_moab().tag_get_handle("Error", 1, MB_TYPE_DOUBLE, th_error,
                                          MB_TAG_CREAT | MB_TAG_SPARSE,
                                          &def_val);

  auto clear_tags = [&]() {
    MoFEMFunctionBegin;
    Range fe_ents;
    CHKERR mField.get_moab().get_entities_by_dimension(0, SPACE_DIM, fe_ents);
    double zero;
    CHKERR mField.get_moab().tag_clear_data(th_error, fe_ents, &zero);
    MoFEMFunctionReturn(0);
  };

  auto evaluate_error = [&]() {
    MoFEMFunctionBegin;
    auto skel_fe = boost::make_shared<BoundaryEle>(mField);
    skel_fe->getRuleHook = [](int, int, int o) { return 3 * o; };
    auto side_data_ptr = boost::make_shared<SideData>();
    auto side_fe_ptr = getSideFE(side_data_ptr);
    skel_fe->getOpPtrVector().push_back(
        new OpErrorSkel(side_fe_ptr, side_data_ptr, error_sum_ptr, th_error));
    auto simple = mField.getInterface<Simple>();

    skel_fe->exeTestHook = [&](FEMethod *fe_ptr) {
      return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
          skeleton_bit);
    };

    CHKERR DMoFEMLoopFiniteElements(simple->getDM(),
                                    simple->getSkeletonFEName(), skel_fe);

    MoFEMFunctionReturn(0);
  };

  auto assemble_and_sum = [](auto vec) {
    CHK_THROW_MESSAGE(VecAssemblyBegin(vec), "assemble");
    CHK_THROW_MESSAGE(VecAssemblyEnd(vec), "assemble");
    double sum;
    CHK_THROW_MESSAGE(VecSum(vec, &sum), "assemble");
    return sum;
  };

  auto propagate_error_to_parents = [&]() {
    MoFEMFunctionBegin;

    auto &moab = mField.get_moab();
    auto fe_ptr = boost::make_shared<FEMethod>();
    fe_ptr->exeTestHook = [&](FEMethod *fe_ptr) {
      return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
          current_bit);
    };

    fe_ptr->preProcessHook = []() { return 0; };
    fe_ptr->postProcessHook = []() { return 0; };
    fe_ptr->operatorHook = [&]() {
      MoFEMFunctionBegin;

      auto fe_ent = fe_ptr->numeredEntFiniteElementPtr->getEnt();
      auto parent = fe_ptr->numeredEntFiniteElementPtr->getParentEnt();
      auto th_parent = fe_ptr->numeredEntFiniteElementPtr->getBasicDataPtr()
                           ->th_RefParentHandle;

      double error;
      CHKERR moab.tag_get_data(th_error, &fe_ent, 1, &error);

      boost::function<MoFEMErrorCode(EntityHandle, double)> add_error =
          [&](auto fe_ent, auto error) {
            MoFEMFunctionBegin;
            double *e_ptr;
            CHKERR moab.tag_get_by_ptr(th_error, &fe_ent, 1,
                                       (const void **)&e_ptr);
            (*e_ptr) += error;

            EntityHandle parent;
            CHKERR moab.tag_get_data(th_parent, &fe_ent, 1, &parent);
            if (parent != fe_ent && parent)
              CHKERR add_error(parent, *e_ptr);

            MoFEMFunctionReturn(0);
          };

      CHKERR add_error(parent, error);

      MoFEMFunctionReturn(0);
    };

    CHKERR DMoFEMLoopFiniteElements(simple->getDM(), simple->getDomainFEName(),
                                    fe_ptr);

    MoFEMFunctionReturn(0);
  };

  CHK_THROW_MESSAGE(clear_tags(), "clear error tags");
  CHK_THROW_MESSAGE(evaluate_error(), "evaluate error");
  CHK_THROW_MESSAGE(propagate_error_to_parents(), "propagate error");

  return std::make_tuple(assemble_and_sum(error_sum_ptr), th_error);
}

/**
 * @brief test side element
 *
 * Check consistency between volume and skeleton integral
 *
 * @return MoFEMErrorCode
 */
MoFEMErrorCode LevelSet::testSideFE() {
  MoFEMFunctionBegin;

  /**
   * @brief calculate volume
   *
   */
  struct DivergenceVol : public DomainEleOp {
    DivergenceVol(boost::shared_ptr<VectorDouble> l_ptr,
                  boost::shared_ptr<MatrixDouble> vel_ptr,
                  SmartPetscObj<Vec> div_vec)
        : DomainEleOp("L", DomainEleOp::OPROW), lPtr(l_ptr), velPtr(vel_ptr),
          divVec(div_vec) {}
    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data) {
      MoFEMFunctionBegin;
      const auto nb_dofs = data.getIndices().size();
      if (nb_dofs) {
        const auto nb_gauss_pts = getGaussPts().size2();
        const auto t_w = getFTensor0IntegrationWeight();
        auto t_diff = data.getFTensor1DiffN<SPACE_DIM>();
        auto t_l = getFTensor0FromVec(*lPtr);
        auto t_vel = getFTensor1FromMat<SPACE_DIM>(*velPtr);
        double div = 0;
        for (auto gg = 0; gg != nb_gauss_pts; ++gg) {
          for (int rr = 0; rr != nb_dofs; ++rr) {
            div += getMeasure() * t_w * t_l * (t_diff(i) * t_vel(i));
            ++t_diff;
          }
          ++t_w;
          ++t_l;
          ++t_vel;
        }
        CHKERR VecSetValue(divVec, 0, div, ADD_VALUES);
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<VectorDouble> lPtr;
    boost::shared_ptr<MatrixDouble> velPtr;
    SmartPetscObj<Vec> divVec;
  };

  /**
   * @brief calculate skeleton integral
   *
   */
  struct DivergenceSkeleton : public BoundaryEleOp {
    DivergenceSkeleton(boost::shared_ptr<SideData> side_data_ptr,
                       boost::shared_ptr<FaceSideEle> side_fe_ptr,
                       SmartPetscObj<Vec> div_vec)
        : BoundaryEleOp(NOSPACE, BoundaryEleOp::OPSPACE),
          sideDataPtr(side_data_ptr), sideFEPtr(side_fe_ptr), divVec(div_vec) {}
    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data) {
      MoFEMFunctionBegin;

      auto get_ntensor = [](auto &base_mat) {
        return FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(
            &*base_mat.data().begin());
      };

      auto not_side = [](auto s) {
        return s == LEFT_SIDE ? RIGHT_SIDE : LEFT_SIDE;
      };

      // Collect data from side domain elements
      CHKERR loopSideFaces("dFE", *sideFEPtr);
      const auto in_the_loop =
          sideFEPtr->nInTheLoop; // return number of elements on the side

      auto t_normal = getFTensor1Normal();
      const auto nb_gauss_pts = getGaussPts().size2();
      for (auto s0 : {LEFT_SIDE, RIGHT_SIDE}) {
        const auto nb_dofs = sideDataPtr->indicesRowSideMap[s0].size();
        if (nb_dofs) {
          auto t_base = get_ntensor(sideDataPtr->rowBaseSideMap[s0]);
          auto nb_row_base_functions = sideDataPtr->rowBaseSideMap[s0].size2();
          auto side_sense = sideDataPtr->senseMap[s0];
          auto opposite_s0 = not_side(s0);

          auto arr_t_l =
              make_array(getFTensor0FromVec(sideDataPtr->lVec[LEFT_SIDE]),
                         getFTensor0FromVec(sideDataPtr->lVec[RIGHT_SIDE]));
          auto arr_t_vel = make_array(
              getFTensor1FromMat<SPACE_DIM>(sideDataPtr->velMat[LEFT_SIDE]),
              getFTensor1FromMat<SPACE_DIM>(sideDataPtr->velMat[RIGHT_SIDE]));

          auto next = [&]() {
            for (auto &t_l : arr_t_l)
              ++t_l;
            for (auto &t_vel : arr_t_vel)
              ++t_vel;
          };

          double div = 0;

          auto t_w = getFTensor0IntegrationWeight();
          for (int gg = 0; gg != nb_gauss_pts; ++gg) {
            FTensor::Tensor1<double, SPACE_DIM> t_vel;
            t_vel(i) =
                (arr_t_vel[LEFT_SIDE](i) + arr_t_vel[RIGHT_SIDE](i)) / 2.;
            const auto dot = (t_normal(i) * t_vel(i)) * side_sense;
            const auto l_upwind_side = (dot > 0) ? s0 : opposite_s0;
            const auto l_upwind =
                arr_t_l[l_upwind_side]; //< assume that field is continues,
                                        // initialisation field has to be smooth
                                        // and exactly approximated by approx
                                        // base
            auto res = t_w * l_upwind * dot;
            ++t_w;
            next();
            int rr = 0;
            for (; rr != nb_dofs; ++rr) {
              div += t_base * res;
              ++t_base;
            }
            for (; rr < nb_row_base_functions; ++rr) {
              ++t_base;
            }
          }
          CHKERR VecSetValue(divVec, 0, div, ADD_VALUES);
        }
        if (!in_the_loop)
          break;
      }

      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<SideData> sideDataPtr;
    boost::shared_ptr<FaceSideEle> sideFEPtr;
    boost::shared_ptr<MatrixDouble> velPtr;
    SmartPetscObj<Vec> divVec;
  };

  auto vol_fe = boost::make_shared<DomainEle>(mField);
  auto skel_fe = boost::make_shared<BoundaryEle>(mField);

  vol_fe->getRuleHook = [](int, int, int o) { return 3 * o; };
  skel_fe->getRuleHook = [](int, int, int o) { return 3 * o; };

  auto div_vol_vec = createSmartVectorMPI(mField.get_comm(), PETSC_DECIDE, 1);
  auto div_skel_vec = createSmartVectorMPI(mField.get_comm(), PETSC_DECIDE, 1);

  auto l_ptr = boost::make_shared<VectorDouble>();
  auto vel_ptr = boost::make_shared<MatrixDouble>();
  auto side_data_ptr = boost::make_shared<SideData>();
  auto side_fe_ptr = getSideFE(side_data_ptr);

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      vol_fe->getOpPtrVector(), {potential_velocity_space, L2});
  vol_fe->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues("L", l_ptr));
  vol_fe->getOpPtrVector().push_back(getZeroLevelVelOp(vel_ptr));
  vol_fe->getOpPtrVector().push_back(
      new DivergenceVol(l_ptr, vel_ptr, div_vol_vec));

  skel_fe->getOpPtrVector().push_back(
      new DivergenceSkeleton(side_data_ptr, side_fe_ptr, div_skel_vec));

  auto simple = mField.getInterface<Simple>();
  auto dm = simple->getDM();

  /**
   * Set up problem such that gradient of level set field is orthogonal to
   * velocity field. Then volume and skeleton integral should yield the same
   * value.
   */

  CHKERR initialiseFieldVelocity(
      [](double x, double y, double) { return x - y; });
  CHKERR initialiseFieldLevelSet(
      [](double x, double y, double) { return x - y; });

  vol_fe->exeTestHook = [&](FEMethod *fe_ptr) {
    return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
        current_bit);
  };
  skel_fe->exeTestHook = [&](FEMethod *fe_ptr) {
    return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
        skeleton_bit);
  };

  CHKERR DMoFEMLoopFiniteElements(dm, simple->getDomainFEName(), vol_fe);
  CHKERR DMoFEMLoopFiniteElements(dm, simple->getSkeletonFEName(), skel_fe);
  CHKERR DMoFEMLoopFiniteElements(dm, simple->getBoundaryFEName(), skel_fe);

  auto assemble_and_sum = [](auto vec) {
    CHK_THROW_MESSAGE(VecAssemblyBegin(vec), "assemble");
    CHK_THROW_MESSAGE(VecAssemblyEnd(vec), "assemble");
    double sum;
    CHK_THROW_MESSAGE(VecSum(vec, &sum), "assemble");
    return sum;
  };

  auto div_vol = assemble_and_sum(div_vol_vec);
  auto div_skel = assemble_and_sum(div_skel_vec);

  auto eps = std::abs((div_vol - div_skel) / (div_vol + div_skel));

  MOFEM_LOG("WORLD", Sev::inform) << "Testing divergence volume: " << div_vol;
  MOFEM_LOG("WORLD", Sev::inform) << "Testing divergence skeleton: " << div_skel
                                  << " relative difference: " << eps;

  constexpr double eps_err = 1e-6;
  if (eps > eps_err)
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "No consistency between skeleton integral and volume integral");

  MoFEMFunctionReturn(0);
};

MoFEMErrorCode LevelSet::testOp() {
  MoFEMFunctionBegin;

  // get operators tester
  auto simple = mField.getInterface<Simple>();
  auto opt = mField.getInterface<OperatorsTester>(); // get interface to
                                                     // OperatorsTester
  auto pip = mField.getInterface<PipelineManager>(); // get interface to
                                                     // pipeline manager

  CHKERR pushOpDomain();
  CHKERR pushOpSkeleton();

  auto post_proc = [&](auto dm, auto f_res, auto out_name) {
    MoFEMFunctionBegin;
    auto post_proc_fe =
        boost::make_shared<PostProcBrokenMeshInMoab<DomainEle>>(mField);

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

  constexpr double eps = 1e-4;

  auto x =
      opt->setRandomFields(simple->getDM(), {{"L", {-1, 1}}, {"V", {-1, 1}}});
  auto dot_x = opt->setRandomFields(simple->getDM(), {{"L", {-1, 1}}});
  auto diff_x = opt->setRandomFields(simple->getDM(), {{"L", {-1, 1}}});

  auto test_domain_ops = [&](auto fe_name, auto lhs_pipeline,
                             auto rhs_pipeline) {
    MoFEMFunctionBegin;

    auto diff_res = opt->checkCentralFiniteDifference(
        simple->getDM(), fe_name, rhs_pipeline, lhs_pipeline, x, dot_x,
        SmartPetscObj<Vec>(), diff_x, 0, 1, eps);

    if constexpr (debug) {
      // Example how to plot direction in direction diff_x. If instead
      // directionalCentralFiniteDifference(...) diff_res is used, then error
      // on directive is plotted.
      CHKERR post_proc(simple->getDM(), diff_res, "tangent_op_error.h5m");
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

  CHKERR test_domain_ops(simple->getDomainFEName(), pip->getDomainLhsFE(),
                         pip->getDomainRhsFE());
  CHKERR test_domain_ops(simple->getSkeletonFEName(), pip->getSkeletonLhsFE(),
                         pip->getSkeletonRhsFE());

  MoFEMFunctionReturn(0);
};

MoFEMErrorCode LevelSet::initialiseFieldLevelSet(
    boost::function<double(double, double, double)> level_fun) {
  MoFEMFunctionBegin;

  // get operators tester
  auto simple = mField.getInterface<Simple>();
  auto pip = mField.getInterface<PipelineManager>(); // get interface to
                                                     // pipeline manager
  auto prb_mng = mField.getInterface<ProblemsManager>();

  boost::shared_ptr<FEMethod> lhs_fe = boost::make_shared<DomainEle>(mField);
  boost::shared_ptr<FEMethod> rhs_fe = boost::make_shared<DomainEle>(mField);
  auto swap_fe = [&]() {
    lhs_fe.swap(pip->getDomainLhsFE());
    rhs_fe.swap(pip->getDomainRhsFE());
  };
  swap_fe();

  pip->setDomainLhsIntegrationRule([](int, int, int o) { return 3 * o; });
  pip->setDomainRhsIntegrationRule([](int, int, int o) { return 3 * o; });

  auto sub_dm = createSmartDM(mField.get_comm(), "DMMOFEM");
  CHKERR DMMoFEMCreateSubDM(sub_dm, simple->getDM(), "LEVELSET_POJECTION");
  CHKERR DMMoFEMSetDestroyProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMSetSquareProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMAddElement(sub_dm, simple->getDomainFEName());
  CHKERR DMMoFEMAddSubFieldRow(sub_dm, "L");
  CHKERR DMSetUp(sub_dm);

  BitRefLevel remove_mask = BitRefLevel().set(current_bit);
  remove_mask.flip(); // DOFs which are not on bit_domain_ele should be removed
  CHKERR prb_mng->removeDofsOnEntities("LEVELSET_POJECTION", "L",
                                       BitRefLevel().set(), remove_mask);
  auto test_bit_ele = [&](FEMethod *fe_ptr) {
    return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
        current_bit);
  };
  pip->getDomainLhsFE()->exeTestHook = test_bit_ele;
  pip->getDomainRhsFE()->exeTestHook = test_bit_ele;

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpDomainRhsPipeline(), {potential_velocity_space, L2});
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpDomainLhsPipeline(), {potential_velocity_space, L2});
  pip->getOpDomainLhsPipeline().push_back(new OpMassLL("L", "L"));
  pip->getOpDomainRhsPipeline().push_back(new OpSourceL("L", level_fun));

  CHKERR mField.getInterface<FieldBlas>()->setField(0, "L");

  auto ksp = pip->createKSP(sub_dm);
  CHKERR KSPSetDM(ksp, sub_dm);
  CHKERR KSPSetFromOptions(ksp);
  CHKERR KSPSetUp(ksp);

  auto L = smartCreateDMVector(sub_dm);
  auto F = smartVectorDuplicate(L);

  CHKERR KSPSolve(ksp, F, L);
  CHKERR VecGhostUpdateBegin(L, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(L, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(sub_dm, L, INSERT_VALUES, SCATTER_REVERSE);

  auto [error, th_error] = evaluateError();
  MOFEM_LOG("LevelSet", Sev::inform) << "Error indicator " << error;
#ifndef NDEBUG
  auto fe_meshset =
      mField.get_finite_element_meshset(simple->getDomainFEName());
  std::vector<Tag> tags{th_error};
  CHKERR mField.get_moab().write_file("error.h5m", "MOAB",
                                      "PARALLEL=WRITE_PART", &fe_meshset, 1,
                                      &*tags.begin(), tags.size());
#endif

  auto post_proc = [&](auto dm, auto out_name, auto th_error) {
    MoFEMFunctionBegin;
    auto post_proc_fe =
        boost::make_shared<PostProcBrokenMeshInMoab<DomainEle>>(mField);
    post_proc_fe->setTagsToTransfer({th_error});
    post_proc_fe->exeTestHook = test_bit_ele;

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    auto l_vec = boost::make_shared<VectorDouble>();
    auto l_grad_mat = boost::make_shared<MatrixDouble>();
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        post_proc_fe->getOpPtrVector(), {potential_velocity_space, L2});
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("L", l_vec));
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("L", l_grad_mat));

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(

            post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

            {{"L", l_vec}},

            {{"GradL", l_grad_mat}},

            {}, {})

    );

    CHKERR DMoFEMLoopFiniteElements(dm, simple->getDomainFEName(),
                                    post_proc_fe);
    post_proc_fe->writeFile(out_name);
    MoFEMFunctionReturn(0);
  };

  if constexpr (debug)
    CHKERR post_proc(sub_dm, "initial_level_set.h5m", th_error);

  swap_fe();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode LevelSet::initialiseFieldVelocity(
    boost::function<double(double, double, double)> vel_fun) {
  MoFEMFunctionBegin;

  // get operators tester
  auto simple = mField.getInterface<Simple>();
  auto pip = mField.getInterface<PipelineManager>(); // get interface to
                                                     // pipeline manager
  auto prb_mng = mField.getInterface<ProblemsManager>();

  boost::shared_ptr<FEMethod> lhs_fe = boost::make_shared<DomainEle>(mField);
  boost::shared_ptr<FEMethod> rhs_fe = boost::make_shared<DomainEle>(mField);
  auto swap_fe = [&]() {
    lhs_fe.swap(pip->getDomainLhsFE());
    rhs_fe.swap(pip->getDomainRhsFE());
  };
  swap_fe();

  pip->setDomainLhsIntegrationRule([](int, int, int o) { return 3 * o; });
  pip->setDomainRhsIntegrationRule([](int, int, int o) { return 3 * o; });

  auto sub_dm = createSmartDM(mField.get_comm(), "DMMOFEM");
  CHKERR DMMoFEMCreateSubDM(sub_dm, simple->getDM(), "VELOCITY_PROJECTION");
  CHKERR DMMoFEMSetDestroyProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMSetSquareProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMAddElement(sub_dm, simple->getDomainFEName());
  CHKERR DMMoFEMAddSubFieldRow(sub_dm, "V");
  CHKERR DMSetUp(sub_dm);

  // Velocities are calculated only on corse mesh
  BitRefLevel remove_mask = BitRefLevel().set(0);
  remove_mask.flip(); // DOFs which are not on bit_domain_ele should be removed
  CHKERR prb_mng->removeDofsOnEntities("VELOCITY_PROJECTION", "V",
                                       BitRefLevel().set(), remove_mask);

  auto test_bit = [&](FEMethod *fe_ptr) {
    return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(0);
  };
  pip->getDomainLhsFE()->exeTestHook = test_bit;
  pip->getDomainRhsFE()->exeTestHook = test_bit;

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpDomainLhsPipeline(), {potential_velocity_space, L2});
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpDomainRhsPipeline(), {potential_velocity_space, L2});

  pip->getOpDomainLhsPipeline().push_back(new OpMassVV("V", "V"));
  pip->getOpDomainRhsPipeline().push_back(new OpSourceV("V", vel_fun));

  auto ksp = pip->createKSP(sub_dm);
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
        boost::make_shared<PostProcBrokenMeshInMoab<DomainEle>>(mField);
    post_proc_fe->exeTestHook = test_bit;

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        post_proc_fe->getOpPtrVector(), {potential_velocity_space});

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    if constexpr (SPACE_DIM == 2) {
      auto l_vec = boost::make_shared<VectorDouble>();
      auto potential_vec = boost::make_shared<VectorDouble>();
      auto velocity_mat = boost::make_shared<MatrixDouble>();

      post_proc_fe->getOpPtrVector().push_back(
          new OpCalculateScalarFieldValues("V", potential_vec));
      post_proc_fe->getOpPtrVector().push_back(
          new OpCalculateHcurlVectorCurl<potential_velocity_field_dim,
                                         SPACE_DIM>("V", velocity_mat));

      post_proc_fe->getOpPtrVector().push_back(

          new OpPPMap(

              post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

              {{"VelocityPotential", potential_vec}},

              {{"Velocity", velocity_mat}},

              {}, {})

      );

    } else {
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
              "3d case not implemented");
    }

    CHKERR DMoFEMLoopFiniteElements(dm, simple->getDomainFEName(),
                                    post_proc_fe);
    post_proc_fe->writeFile(out_name);
    MoFEMFunctionReturn(0);
  };

  if constexpr (debug)
    CHKERR post_proc(sub_dm, "initial_velocity_potential.h5m");

  swap_fe();

  MoFEMFunctionReturn(0);
}

LevelSet *level_set_raw_ptr = nullptr;
MoFEM::TsCtx *ts_ctx;

MoFEMErrorCode LevelSet::solveAdvection() {
  MoFEMFunctionBegin;

  // get operators tester
  auto simple = mField.getInterface<Simple>();
  auto pip = mField.getInterface<PipelineManager>(); // get interface to
  auto prb_mng = mField.getInterface<ProblemsManager>();

  CHKERR pushOpDomain();
  CHKERR pushOpSkeleton();

  auto sub_dm = createSmartDM(mField.get_comm(), "DMMOFEM");
  CHKERR DMMoFEMCreateSubDM(sub_dm, simple->getDM(), "ADVECTION");
  CHKERR DMMoFEMSetDestroyProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMSetSquareProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMAddElement(sub_dm, simple->getDomainFEName());
  CHKERR DMMoFEMAddElement(sub_dm, simple->getSkeletonFEName());
  CHKERR DMMoFEMAddSubFieldRow(sub_dm, "L");
  CHKERR DMSetUp(sub_dm);

  BitRefLevel remove_mask = BitRefLevel().set(current_bit);
  remove_mask.flip(); // DOFs which are not on bit_domain_ele should be removed
  CHKERR prb_mng->removeDofsOnEntities("ADVECTION", "L", BitRefLevel().set(),
                                       remove_mask);

  auto add_post_proc_fe = [&]() {
    auto post_proc_fe = boost::make_shared<PostProcEle>(mField);

    Tag th_error;
    double def_val = 0;
    CHKERR mField.get_moab().tag_get_handle(
        "Error", 1, MB_TYPE_DOUBLE, th_error, MB_TAG_CREAT | MB_TAG_SPARSE,
        &def_val);
    post_proc_fe->setTagsToTransfer({th_error});

    post_proc_fe->exeTestHook = [&](FEMethod *fe_ptr) {
      return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
          current_bit);
    };

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;
    auto l_vec = boost::make_shared<VectorDouble>();
    auto vel_ptr = boost::make_shared<MatrixDouble>();

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        post_proc_fe->getOpPtrVector(), {H1});
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("L", l_vec));
    post_proc_fe->getOpPtrVector().push_back(getZeroLevelVelOp(vel_ptr));

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(

            post_proc_fe->getPostProcMesh(),

            post_proc_fe->getMapGaussPts(),

            {{"L", l_vec}},

            {{"V", vel_ptr}},

            {}, {})

    );
    return post_proc_fe;
  };

  auto post_proc_fe = add_post_proc_fe();

  auto set_time_monitor = [&](auto dm, auto ts) {
    auto monitor_ptr = boost::make_shared<FEMethod>();

    monitor_ptr->preProcessHook = []() { return 0; };
    monitor_ptr->operatorHook = []() { return 0; };
    monitor_ptr->postProcessHook = [&]() {
      MoFEMFunctionBegin;

      if (!post_proc_fe)
        SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                "Null pointer for post proc element");

      CHKERR DMoFEMLoopFiniteElements(dm, simple->getDomainFEName(),
                                      post_proc_fe);
      CHKERR post_proc_fe->writeFile(
          "level_set_" +
          boost::lexical_cast<std::string>(monitor_ptr->ts_step) + ".h5m");
      MoFEMFunctionReturn(0);
    };

    boost::shared_ptr<FEMethod> null;
    DMMoFEMTSSetMonitor(sub_dm, ts, simple->getDomainFEName(), monitor_ptr,
                        null, null);

    return monitor_ptr;
  };

  auto ts = pip->createTSIM(sub_dm);

  auto set_solution = [&](auto ts) {
    MoFEMFunctionBegin;
    auto D = smartCreateDMVector(sub_dm);
    CHKERR DMoFEMMeshToLocalVector(sub_dm, D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR TSSetSolution(ts, D);
    MoFEMFunctionReturn(0);
  };
  CHKERR set_solution(ts);

  auto monitor_pt = set_time_monitor(sub_dm, ts);
  CHKERR TSSetFromOptions(ts);

  auto B = smartCreateDMMatrix(sub_dm);
  CHKERR TSSetIJacobian(ts, B, B, TsSetIJacobian, nullptr);
  level_set_raw_ptr = this;

  CHKERR TSSetUp(ts);

  auto ts_pre_step = [](TS ts) {
    auto &m_field = level_set_raw_ptr->mField;
    auto simple = m_field.getInterface<Simple>();
    auto bit_mng = m_field.getInterface<BitRefManager>();
    MoFEMFunctionBegin;

    auto [error, th_error] = level_set_raw_ptr->evaluateError();
    MOFEM_LOG("LevelSet", Sev::inform) << "Error indicator " << error;

    auto get_norm = [&](auto x) {
      double nrm;
      CHKERR VecNorm(x, NORM_2, &nrm);
      return nrm;
    };

    auto set_solution = [&](auto ts) {
      MoFEMFunctionBegin;
      DM dm;
      CHKERR TSGetDM(ts, &dm);
      auto prb_ptr = getProblemPtr(dm);

      auto x = smartCreateDMVector(dm);
      CHKERR DMoFEMMeshToLocalVector(dm, x, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateBegin(x, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(x, INSERT_VALUES, SCATTER_FORWARD);

      MOFEM_LOG("LevelSet", Sev::inform)
          << "Problem " << prb_ptr->getName() << " solution vector norm "
          << get_norm(x);
      CHKERR TSSetSolution(ts, x);

      MoFEMFunctionReturn(0);
    };

    auto refine_and_project = [&](auto ts) {
      MoFEMFunctionBegin;

      CHKERR level_set_raw_ptr->refineMesh(
          WrapperClassErrorProjection(level_set_raw_ptr->maxPtr));
      simple->getBitRefLevel() = BitRefLevel().set(skeleton_bit) |
                                 BitRefLevel().set(aggregate_bit) |
                                 BitRefLevel().set(aggregate_projection_bit);

      simple->reSetUp(true);
      DM dm;
      CHKERR TSGetDM(ts, &dm);
      CHKERR DMSubDMSetUp_MoFEM(dm);

      BitRefLevel remove_mask = BitRefLevel().set(current_bit);
      remove_mask
          .flip(); // DOFs which are not on bit_domain_ele should be removed
      CHKERR level_set_raw_ptr->mField.getInterface<ProblemsManager>()
          ->removeDofsOnEntities("ADVECTION", "L", BitRefLevel().set(),
                                 remove_mask);

      MoFEMFunctionReturn(0);
    };

    auto ts_reset_theta = [&](auto ts) {
      MoFEMFunctionBegin;
      DM dm;
      CHKERR TSGetDM(ts, &dm);

      // FIXME: Look into vec-5 how to transfer internal theta method variables

      CHKERR TSReset(ts);
      CHKERR TSSetUp(ts);

      CHKERR level_set_raw_ptr->dgProjection(projection_bit);
      CHKERR set_solution(ts);

      auto B = smartCreateDMMatrix(dm);
      CHKERR TSSetIJacobian(ts, B, B, TsSetIJacobian, nullptr);

      MoFEMFunctionReturn(0);
    };

    CHKERR refine_and_project(ts);
    CHKERR ts_reset_theta(ts);

    MoFEMFunctionReturn(0);
  };

  auto ts_post_step = [](TS ts) { return 0; };

  CHKERR TSSetPreStep(ts, ts_pre_step);
  CHKERR TSSetPostStep(ts, ts_post_step);

  CHKERR TSSolve(ts, NULL);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode LevelSet::refineMesh(WrapperClass &&wp) {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  auto bit_mng = mField.getInterface<BitRefManager>();
  ParallelComm *pcomm =
      ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
  auto proc_str = boost::lexical_cast<std::string>(mField.get_comm_rank());

  auto set_bit = [](auto l) { return BitRefLevel().set(l); };

  auto save_range = [&](const std::string name, const Range &r) {
    MoFEMFunctionBegin;
    auto meshset_ptr = get_temp_meshset_ptr(mField.get_moab());
    CHKERR mField.get_moab().add_entities(*meshset_ptr, r);
    CHKERR mField.get_moab().write_file(name.c_str(), "VTK", "",
                                        meshset_ptr->get_ptr(), 1);
    MoFEMFunctionReturn(0);
  };

  // select domain elements to refine by threshold
  auto get_refined_elements_meshset = [&](auto bit, auto mask) {
    Range fe_ents;
    CHKERR bit_mng->getEntitiesByDimAndRefLevel(bit, mask, SPACE_DIM, fe_ents);

    Tag th_error;
    CHK_MOAB_THROW(mField.get_moab().tag_get_handle("Error", th_error),
                   "get error handle");
    std::vector<double> errors(fe_ents.size());
    CHK_MOAB_THROW(
        mField.get_moab().tag_get_data(th_error, fe_ents, &*errors.begin()),
        "get tag data");
    auto it = std::max_element(errors.begin(), errors.end());
    double max;
    MPI_Allreduce(&*it, &max, 1, MPI_DOUBLE, MPI_MAX, mField.get_comm());
    MOFEM_LOG("LevelSet", Sev::inform) << "Max error: " << max;
    auto threshold = wp.getThreshold(max);

    std::vector<EntityHandle> fe_to_refine;
    fe_to_refine.reserve(fe_ents.size());

    auto fe_it = fe_ents.begin();
    auto error_it = errors.begin();
    for (auto i = 0; i != fe_ents.size(); ++i) {
      if (*error_it > threshold) {
        fe_to_refine.push_back(*fe_it);
      }
      ++fe_it;
      ++error_it;
    }

    Range ents;
    ents.insert_list(fe_to_refine.begin(), fe_to_refine.end());
    CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(
        ents, nullptr, NOISY);

    auto get_neighbours_by_bridge_vertices = [&](auto &&ents) {
      Range verts;
      CHKERR mField.get_moab().get_connectivity(ents, verts, true);
      CHKERR mField.get_moab().get_adjacencies(verts, SPACE_DIM, false, ents,
                                               moab::Interface::UNION);
      CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(ents);
      return ents;
    };

    ents = get_neighbours_by_bridge_vertices(ents);

#ifndef NDEBUG
    if (debug) {
      auto meshset_ptr = get_temp_meshset_ptr(mField.get_moab());
      CHK_MOAB_THROW(mField.get_moab().add_entities(*meshset_ptr, ents),
                     "add entities to meshset");
      CHKERR mField.get_moab().write_file(
          (proc_str + "_fe_to_refine.vtk").c_str(), "VTK", "",
          meshset_ptr->get_ptr(), 1);
    }
#endif

    return ents;
  };

  // refine elements, and set bit ref level
  auto refine_mesh = [&](auto l, auto &&fe_to_refine) {
    Skinner skin(&mField.get_moab());
    MoFEMFunctionBegin;

    // get entities in "l-1" level
    Range level_ents;
    CHKERR bit_mng->getEntitiesByDimAndRefLevel(
        set_bit(start_bit + l - 1), BitRefLevel().set(), SPACE_DIM, level_ents);
    // select entities to refine
    fe_to_refine = intersect(level_ents, fe_to_refine);
    // select entities not to refine
    level_ents = subtract(level_ents, fe_to_refine);

    // for entities to refine get children, i.e. redlined entities
    Range fe_to_refine_children;
    bit_mng->updateRangeByChildren(fe_to_refine, fe_to_refine_children);
    // add entities to to level "l"
    fe_to_refine_children =
        fe_to_refine_children.subset_by_dimension(SPACE_DIM);
    level_ents.merge(fe_to_refine_children);

    auto fix_neighbour_level = [&](auto ll) {
      MoFEMFunctionBegin;
      // filter entities on level ll
      auto level_ll = level_ents;
      CHKERR bit_mng->filterEntitiesByRefLevel(set_bit(ll), BitRefLevel().set(),
                                               level_ll);
      // find skin of ll level
      Range skin_edges;
      CHKERR skin.find_skin(0, level_ll, false, skin_edges);
      // get parents of skin of level ll
      Range skin_parents;
      for (auto lll = 0; lll <= ll; ++lll) {
        CHKERR bit_mng->updateRangeByParent(skin_edges, skin_parents);
      }
      // filter parents on level ll - 1
      BitRefLevel bad_bit;
      for (auto lll = 0; lll <= ll - 2; ++lll) {
        bad_bit[lll] = true;
      }
      // get adjacents to parents
      Range skin_adj_ents;
      CHKERR mField.get_moab().get_adjacencies(skin_parents, SPACE_DIM, false,
                                               skin_adj_ents,
                                               moab::Interface::UNION);
      CHKERR bit_mng->filterEntitiesByRefLevel(bad_bit, BitRefLevel().set(),
                                               skin_adj_ents);
      skin_adj_ents = intersect(skin_adj_ents, level_ents);
      if (!skin_adj_ents.empty()) {
        level_ents = subtract(level_ents, skin_adj_ents);
        Range skin_adj_ents_children;
        bit_mng->updateRangeByChildren(skin_adj_ents, skin_adj_ents_children);
        level_ents.merge(skin_adj_ents_children);
      }
      MoFEMFunctionReturn(0);
    };

    CHKERR fix_neighbour_level(l);

    CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(
        level_ents);

    // get lower dimension entities for level "l"
    for (auto d = 0; d != SPACE_DIM; ++d) {
      if (d == 0) {
        CHKERR mField.get_moab().get_connectivity(
            level_ents.subset_by_dimension(SPACE_DIM), level_ents, true);
      } else {
        CHKERR mField.get_moab().get_adjacencies(
            level_ents.subset_by_dimension(SPACE_DIM), d, false, level_ents,
            moab::Interface::UNION);
      }
    }
    CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(
        level_ents);

    // set bit ref level to level entities
    CHKERR bit_mng->setNthBitRefLevel(start_bit + l, false);
    CHKERR bit_mng->setNthBitRefLevel(level_ents, start_bit + l, true);

#ifndef NDEBUG
    auto proc_str = boost::lexical_cast<std::string>(mField.get_comm_rank());
    CHKERR bit_mng->writeBitLevelByDim(
        set_bit(start_bit + l), BitRefLevel().set(), SPACE_DIM,
        (boost::lexical_cast<std::string>(l) + "_" + proc_str + "_ref_mesh.vtk")
            .c_str(),
        "VTK", "");
#endif

    MoFEMFunctionReturn(0);
  };

  // set skeleton
  auto set_skelton_bit = [&](auto l) {
    MoFEMFunctionBegin;

    // get entities of dim-1 on level "l"
    Range level_edges;
    CHKERR bit_mng->getEntitiesByDimAndRefLevel(set_bit(start_bit + l),
                                                BitRefLevel().set(),
                                                SPACE_DIM - 1, level_edges);

    // get parent of entities of level "l"
    Range level_edges_parents;
    CHKERR bit_mng->updateRangeByParent(level_edges, level_edges_parents);
    level_edges_parents =
        level_edges_parents.subset_by_dimension(SPACE_DIM - 1);
    CHKERR bit_mng->filterEntitiesByRefLevel(
        set_bit(start_bit + l), BitRefLevel().set(), level_edges_parents);

    // skeleton entities which do not have parents
    auto parent_skeleton = intersect(level_edges, level_edges_parents);
    auto skeleton = subtract(level_edges, level_edges_parents);

    // add adjacent domain entities
    CHKERR mField.get_moab().get_adjacencies(unite(parent_skeleton, skeleton),
                                             SPACE_DIM, false, skeleton,
                                             moab::Interface::UNION);

    // set levels
    CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(skeleton);
    CHKERR bit_mng->setNthBitRefLevel(skeleton_bit, false);
    CHKERR bit_mng->setNthBitRefLevel(skeleton, skeleton_bit, true);

#ifndef NDEBUG
    CHKERR bit_mng->writeBitLevel(
        set_bit(skeleton_bit), BitRefLevel().set(),
        (boost::lexical_cast<std::string>(l) + "_" + proc_str + "_skeleton.vtk")
            .c_str(),
        "VTK", "");
#endif
    MoFEMFunctionReturn(0);
  };

  // Reset bit sand set old current and aggregate bits as projection bits
  Range level0_current;
  CHKERR bit_mng->getEntitiesByRefLevel(BitRefLevel().set(current_bit),
                                        BitRefLevel().set(), level0_current);

  Range level0_aggregate;
  CHKERR bit_mng->getEntitiesByRefLevel(BitRefLevel().set(aggregate_bit),
                                        BitRefLevel().set(), level0_aggregate);

  BitRefLevel start_mask;
  for (auto s = 0; s != start_bit; ++s)
    start_mask[s] = true;
  CHKERR bit_mng->lambdaBitRefLevel(
      [&](EntityHandle ent, BitRefLevel &bit) { bit &= start_mask; });
  CHKERR bit_mng->setNthBitRefLevel(level0_current, projection_bit, true);
  CHKERR bit_mng->setNthBitRefLevel(level0_aggregate, aggregate_projection_bit,
                                    true);

  // Set zero bit ref level
  Range level0;
  CHKERR bit_mng->getEntitiesByRefLevel(set_bit(0), BitRefLevel().set(),
                                        level0);
  CHKERR bit_mng->setNthBitRefLevel(level0, start_bit, true);
  CHKERR bit_mng->setNthBitRefLevel(level0, current_bit, true);
  CHKERR bit_mng->setNthBitRefLevel(level0, aggregate_bit, true);
  CHKERR bit_mng->setNthBitRefLevel(level0, skeleton_bit, true);

  CHKERR wp.setBits(*this, 0);
  CHKERR wp.runCalcs(*this, 0);
  for (auto l = 0; l != nb_levels; ++l) {
    MOFEM_LOG("WORLD", Sev::inform) << "Process level: " << l;
    CHKERR refine_mesh(l + 1, get_refined_elements_meshset(
                                  set_bit(start_bit + l), BitRefLevel().set()));
    CHKERR set_skelton_bit(l + 1);
    CHKERR wp.setAggregateBit(*this, l + 1);
    CHKERR wp.setBits(*this, l + 1);
    CHKERR wp.runCalcs(*this, l + 1);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode LevelSet::dgProjection(const int projection_bit) {
  MoFEMFunctionBegin;

  // get operators tester
  auto simple = mField.getInterface<Simple>();
  auto pip = mField.getInterface<PipelineManager>(); // get interface to
  auto bit_mng = mField.getInterface<BitRefManager>();
  auto prb_mng = mField.getInterface<ProblemsManager>();

  auto lhs_fe = boost::make_shared<DomainEle>(mField);
  auto rhs_fe_prj = boost::make_shared<DomainEle>(mField);
  auto rhs_fe_current = boost::make_shared<DomainEle>(mField);

  lhs_fe->getRuleHook = [](int, int, int o) { return 3 * o; };
  rhs_fe_prj->getRuleHook = [](int, int, int o) { return 3 * o; };
  rhs_fe_current->getRuleHook = [](int, int, int o) { return 3 * o; };

  auto sub_dm = createSmartDM(mField.get_comm(), "DMMOFEM");
  CHKERR DMMoFEMCreateSubDM(sub_dm, simple->getDM(), "DG_PROJECTION");
  CHKERR DMMoFEMSetDestroyProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMSetSquareProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMAddElement(sub_dm, simple->getDomainFEName());
  CHKERR DMMoFEMAddSubFieldRow(sub_dm, "L");
  CHKERR DMSetUp(sub_dm);

  Range current_ents;
  CHKERR bit_mng->getEntitiesByDimAndRefLevel(BitRefLevel().set(current_bit),
                                              BitRefLevel().set(), SPACE_DIM,
                                              current_ents);
  Range prj_ents;
  CHKERR bit_mng->getEntitiesByDimAndRefLevel(BitRefLevel().set(projection_bit),
                                              BitRefLevel().set(), SPACE_DIM,
                                              prj_ents);
  for (auto l = 0; l != nb_levels; ++l) {
    CHKERR bit_mng->updateRangeByParent(prj_ents, prj_ents);
  }
  current_ents = subtract(current_ents, prj_ents);

  auto test_mesh_bit = [&](FEMethod *fe_ptr) {
    return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
        current_bit);
  };
  auto test_prj_bit = [&](FEMethod *fe_ptr) {
    return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
        projection_bit);
  };
  auto test_current_bit = [&](FEMethod *fe_ptr) {
    return current_ents.find(fe_ptr->getFEEntityHandle()) != current_ents.end();
  };

  lhs_fe->exeTestHook = test_mesh_bit;
  rhs_fe_prj->exeTestHook = test_prj_bit;
  rhs_fe_current->exeTestHook = test_current_bit;

  BitRefLevel remove_mask = BitRefLevel().set(current_bit);
  remove_mask.flip(); // DOFs which are not on bit_domain_ele should be removed
  CHKERR prb_mng->removeDofsOnEntities(
      "DG_PROJECTION", "L", BitRefLevel().set(), remove_mask, nullptr, 0,
      MAX_DOFS_ON_ENTITY, 0, 100, NOISY, true);

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      lhs_fe->getOpPtrVector(), {potential_velocity_space, L2});
  lhs_fe->getOpPtrVector().push_back(new OpMassLL("L", "L"));

  auto l_vec = boost::make_shared<VectorDouble>();

  // This assumes that projection mesh is refined, current mesh is coarsened.
  auto set_prj_from_child = [&](auto rhs_fe_prj) {
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        rhs_fe_prj->getOpPtrVector(), {potential_velocity_space, L2}); 

    // Evaluate field value on projection mesh      
    rhs_fe_prj->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("L", l_vec));

    // This element is used to assemble    
    auto get_parent_this = [&]() {
      auto fe_parent_this = boost::make_shared<DomianParentEle>(mField);
      fe_parent_this->getOpPtrVector().push_back(
          new OpScalarFieldL("L", l_vec));
      return fe_parent_this;
    };

    // Create levels of parent elements, until current element is reached, and
    // then assemble.
    auto get_parents_fe_ptr = [&](auto this_fe_ptr) {
      std::vector<boost::shared_ptr<DomianParentEle>> parents_elems_ptr_vec;
      for (int l = 0; l <= nb_levels; ++l)
        parents_elems_ptr_vec.emplace_back(
            boost::make_shared<DomianParentEle>(mField));
      for (auto l = 1; l <= nb_levels; ++l) {
        parents_elems_ptr_vec[l - 1]->getOpPtrVector().push_back(
            new OpRunParent(parents_elems_ptr_vec[l], BitRefLevel().set(),
                            BitRefLevel().set(current_bit).flip(), this_fe_ptr,
                            BitRefLevel().set(current_bit),
                            BitRefLevel().set()));
      }
      return parents_elems_ptr_vec[0];
    };

    auto this_fe_ptr = get_parent_this();
    auto parent_fe_ptr = get_parents_fe_ptr(this_fe_ptr);
    rhs_fe_prj->getOpPtrVector().push_back(
        new OpRunParent(parent_fe_ptr, BitRefLevel().set(),
                        BitRefLevel().set(current_bit).flip(), this_fe_ptr,
                        BitRefLevel().set(current_bit), BitRefLevel().set()));
  };

  // This assumed that current mesh is refined, and projection mesh is coarser
  auto set_prj_from_parent = [&](auto rhs_fe_current) {

    // Evaluate field on coarser element
    auto get_parent_this = [&]() {
      auto fe_parent_this = boost::make_shared<DomianParentEle>(mField);
      fe_parent_this->getOpPtrVector().push_back(
          new OpCalculateScalarFieldValues("L", l_vec));
      return fe_parent_this;
    };

    // Create stack of evaluation on parent elements
    auto get_parents_fe_ptr = [&](auto this_fe_ptr) {
      std::vector<boost::shared_ptr<DomianParentEle>> parents_elems_ptr_vec;
      for (int l = 0; l <= nb_levels; ++l)
        parents_elems_ptr_vec.emplace_back(
            boost::make_shared<DomianParentEle>(mField));
      for (auto l = 1; l <= nb_levels; ++l) {
        parents_elems_ptr_vec[l - 1]->getOpPtrVector().push_back(
            new OpRunParent(parents_elems_ptr_vec[l], BitRefLevel().set(),
                            BitRefLevel().set(projection_bit).flip(),
                            this_fe_ptr, BitRefLevel().set(projection_bit),
                            BitRefLevel().set()));
      }
      return parents_elems_ptr_vec[0];
    };

    auto this_fe_ptr = get_parent_this();
    auto parent_fe_ptr = get_parents_fe_ptr(this_fe_ptr);

    auto reset_op_ptr = new DomainEleOp(NOSPACE, DomainEleOp::OPSPACE);
    reset_op_ptr->doWorkRhsHook = [&](DataOperator *op_ptr, int, EntityType,
                                      EntData &) {
      l_vec->resize(static_cast<DomainEleOp *>(op_ptr)->getGaussPts().size2(),
                    false);
      l_vec->clear();
      return 0;
    };
    rhs_fe_current->getOpPtrVector().push_back(reset_op_ptr);
    rhs_fe_current->getOpPtrVector().push_back(
        new OpRunParent(parent_fe_ptr, BitRefLevel().set(),
                        BitRefLevel().set(projection_bit).flip(), this_fe_ptr,
                        BitRefLevel(), BitRefLevel()));

    // At the end assemble of current finite element
    rhs_fe_current->getOpPtrVector().push_back(new OpScalarFieldL("L", l_vec));
  };

  set_prj_from_child(rhs_fe_prj);
  set_prj_from_parent(rhs_fe_current);

  boost::shared_ptr<FEMethod> null_fe;
  smartGetDMKspCtx(sub_dm)->clearLoops();
  CHKERR DMMoFEMKSPSetComputeOperators(sub_dm, simple->getDomainFEName(),
                                       lhs_fe, null_fe, null_fe);
  CHKERR DMMoFEMKSPSetComputeRHS(sub_dm, simple->getDomainFEName(), rhs_fe_prj,
                                 null_fe, null_fe);
  CHKERR DMMoFEMKSPSetComputeRHS(sub_dm, simple->getDomainFEName(),
                                 rhs_fe_current, null_fe, null_fe);
  auto ksp = MoFEM::createKSP(mField.get_comm());
  CHKERR KSPSetDM(ksp, sub_dm);

  CHKERR KSPSetDM(ksp, sub_dm);
  CHKERR KSPSetFromOptions(ksp);
  CHKERR KSPSetUp(ksp);

  auto L = smartCreateDMVector(sub_dm);
  auto F = smartVectorDuplicate(L);

  CHKERR KSPSolve(ksp, F, L);
  CHKERR VecGhostUpdateBegin(L, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(L, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(sub_dm, L, INSERT_VALUES, SCATTER_REVERSE);

  auto [error, th_error] = evaluateError();
  MOFEM_LOG("LevelSet", Sev::inform) << "Error indicator " << error;

  auto post_proc = [&](auto dm, auto out_name, auto th_error) {
    MoFEMFunctionBegin;
    auto post_proc_fe =
        boost::make_shared<PostProcBrokenMeshInMoab<DomainEle>>(mField);
    post_proc_fe->setTagsToTransfer({th_error});
    post_proc_fe->exeTestHook = test_mesh_bit;

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    auto l_vec = boost::make_shared<VectorDouble>();
    auto l_grad_mat = boost::make_shared<MatrixDouble>();
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        post_proc_fe->getOpPtrVector(), {potential_velocity_space, L2});
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("L", l_vec));
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("L", l_grad_mat));

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(

            post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

            {{"L", l_vec}},

            {{"GradL", l_grad_mat}},

            {}, {})

    );

    CHKERR DMoFEMLoopFiniteElements(dm, simple->getDomainFEName(),
                                    post_proc_fe);
    post_proc_fe->writeFile(out_name);
    MoFEMFunctionReturn(0);
  };

  if constexpr (debug)
    CHKERR post_proc(sub_dm, "dg_projection.h5m", th_error);

  MoFEMFunctionReturn(0);
}