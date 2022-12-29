/**
 * @file level_set.cpp
 * @example level_set.cpp
 * @brief Example with level set method
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

//! [Define dimension]
constexpr int SPACE_DIM = EXECUTABLE_DIMENSION;
constexpr AssemblyType A = AssemblyType::PETSC; //< selected assembly type
constexpr IntegrationType I =
    IntegrationType::GAUSS; //< selected integration type

using EntData = EntitiesFieldData::EntData;
using DomainEle = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::DomainEle;
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

constexpr bool debug = true;

struct LevelSet {

  LevelSet(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();

  MoFEMErrorCode pushOpDomain();
  MoFEMErrorCode pushOpSkeleton();

  MoFEMErrorCode testOp();

  MoFEMErrorCode initialiseFieldLevelSet();
  MoFEMErrorCode initialiseFieldVelocity();
  MoFEMErrorCode solveLevelSet();

  struct OpRhsDomain;
  struct OpLhsDomain;
  struct OpRhsSkeleton;
  struct OpLhsSkeleton;

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

  using AssemblyBoundaryEleOp =
      FormsIntegrators<BoundaryEleOp>::Assembly<A>::OpBase;

  using VecSideArray = std::array<VectorDouble, 2>;
  using MatSideArray = std::array<MatrixDouble, 2>;

  enum ElementSide { LEFT_SIDE = 0, RIGHT_SIDE };

  struct SideData {
    // data for skeleton computation
    std::array<VectorInt, 2>
        indicesRowSideMap; ///< indices on rows for left hand-side
    std::array<VectorInt, 2>
        indicesColSideMap; ///< indices on columns for left hand-side
    std::array<MatrixDouble, 2> rowBaseSideMap; // base functions on rows
    std::array<MatrixDouble, 2> colBaseSideMap; // base function  on columns
    std::array<MatrixDouble, 2>
        rowDiffBaseSideMap; // derivative of base functions
    std::array<MatrixDouble, 2>
        colDiffBaseSideMap;        // derivative of base functions
    std::array<double, 2> areaMap; // area/volume of elements on the side
    std::array<int, 2> senseMap;   // orientation of local element edge/face in
                                   // respect to global orientation of edge/face

    VecSideArray lVec;
    MatSideArray velMat;
  };
};

MoFEMErrorCode LevelSet::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();

  if (debug)
    CHKERR testOp();
  CHKERR initialiseFieldVelocity();
  CHKERR initialiseFieldLevelSet();
  // CHKERR solveLevelSet();

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
  CHKERR simple->loadFile();
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
  CHKERR simple->setFieldOrder("L", 1);
  CHKERR simple->setFieldOrder("V", 1);

  // setup problem
  CHKERR simple->setUp();

  MoFEMFunctionReturn(0);
}

FTensor::Index<'i', SPACE_DIM> i;
FTensor::Index<'j', SPACE_DIM> j;
FTensor::Index<'k', SPACE_DIM> k;

template <int SPACE_DIM>
auto get_velocity_potential(double x, double y, double z);

template <> auto get_velocity_potential<2>(double x, double y, double z) {
  return (x * x - 0.25) * (y * y - 0.25);
}

double get_level_set(const double x, const double y, const double z) {
  constexpr double xc = 0.1;
  constexpr double yc = 0.;
  constexpr double zc = 0.;
  constexpr double r = 0.2;
  return std::sqrt(pow(x - xc, 2) + pow(y - yc, 2) + pow(z - zc, 2)) - r;
}

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
      sideDataPtr(side_data_ptr), sideFEPtr(side_fe_ptr) {
  sYmm = false;
}

MoFEMErrorCode LevelSet::OpRhsDomain::iNtegrate(EntData &data) {
  MoFEMFunctionBegin;

  const auto nb_int_points = getGaussPts().size1();
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

  const auto nb_int_points = getGaussPts().size1();
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

  // Collect data from side domian elements
  CHKERR loopSideFaces("dFE", *sideFEPtr);
  const auto in_the_loop =
      sideFEPtr->nInTheLoop; // return number of elements on the side

  if (in_the_loop > 0) {

    // get normal of the face or edge
    auto t_normal = getFTensor1Normal();
    t_normal.normalize();

    const auto nb_gauss_pts = getGaussPts().size2();

#ifndef NDEBUG
    if (sideDataPtr->senseMap[0] * sideDataPtr->senseMap[1] >= 0)
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "Both sided should have opposite sense");
#endif // NDEBUG

    for (auto s0 : {LEFT_SIDE, RIGHT_SIDE}) {

      // gent number of DOFs on the right side.
      const auto nb_rows = sideDataPtr->indicesRowSideMap[s0].size();

      if (nb_rows) {

        // get orientation of the local element edge
        const auto sense_row = sideDataPtr->senseMap[s0];

        auto t_w = getFTensor0IntegrationWeight();

        auto arr_t_l =
            make_array(getFTensor0FromVec(sideDataPtr->lVec[LEFT_SIDE]),
                       getFTensor0FromVec(sideDataPtr->lVec[RIGHT_SIDE]));
        auto arr_t_vel = make_array(
            getFTensor1FromMat<SPACE_DIM>(sideDataPtr->velMat[LEFT_SIDE]),
            getFTensor1FromMat<SPACE_DIM>(sideDataPtr->velMat[RIGHT_SIDE]));

        auto get_ntensor = [](auto &base_mat) {
          return FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(
              &*base_mat.data().begin());
        };
        auto t_row_base = get_ntensor(sideDataPtr->rowBaseSideMap[s0]);
        resSkelton.resize(nb_rows, false);
        resSkelton.clear();

        for (int gg = 0; gg != nb_gauss_pts; ++gg) {

          const auto alpha = getMeasure() * t_w;
          const auto dot = (t_normal(i) * (arr_t_vel[LEFT_SIDE](i) +
                                           arr_t_vel[RIGHT_SIDE](i))) *
                           (sense_row / 2);

          auto not_side = [](auto s) {
            return s == LEFT_SIDE ? RIGHT_SIDE : LEFT_SIDE;
          };

          auto get_side = [&]() {
            if (dot > 0)
              return s0;
            else
              return not_side(s0);
          };

          const auto res = alpha * dot * arr_t_l[get_side()];

          for (auto t_l : arr_t_l)
            ++t_l;
          for (auto t_vel : arr_t_vel)
            ++t_vel;

          for (auto rr = 0; rr != nb_rows; ++rr) {
            resSkelton[rr] = t_row_base * res;
            ++t_row_base;
          }
        }

        // assemble local operator vector to global vector
        CHKERR ::VecSetValues(getKSPf(),
                              sideDataPtr->indicesRowSideMap[0].size(),
                              &*sideDataPtr->indicesRowSideMap[0].begin(),
                              &*resSkelton.data().begin(), ADD_VALUES);
      }
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
LevelSet::OpLhsSkeleton::doWork(int side, EntityType type,
                                EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  // Collect data from side domian elements
  CHKERR loopSideFaces("dFE", *sideFEPtr);
  const auto in_the_loop =
      sideFEPtr->nInTheLoop; // return number of elements on the side

#ifndef NDEBUG
  const std::array<std::string, 2> ele_type_name = {"BOUNDARY", "SKELETON"};
  MOFEM_LOG("SELF", Sev::noisy)
      << "OpLhsSkeleton inTheLoop " << ele_type_name[in_the_loop];
  MOFEM_LOG("SELF", Sev::noisy)
      << "OpLhsSkeleton sense " << sideDataPtr->senseMap[0] << " "
      << sideDataPtr->senseMap[1];
#endif

  if (in_the_loop > 0) {

    // get normal of the face or edge
    auto t_normal = getFTensor1Normal();
    t_normal.normalize();

    // get number of integration points on face
    const size_t nb_integration_pts = getGaussPts().size2();

    auto get_ntensor = [](auto &base_mat) {
      return FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(
          &*base_mat.data().begin());
    };

    // iterate the sides rows
    for (auto s0 : {LEFT_SIDE, RIGHT_SIDE}) {

      // gent number of DOFs on the right side.
      const auto nb_rows = sideDataPtr->indicesRowSideMap[s0].size();

      if (nb_rows) {

        // get orientation of the local element edge
        const auto sense_row = sideDataPtr->senseMap[s0];

        // iterate the side cols
        const auto nb_row_base_functions =
            sideDataPtr->rowBaseSideMap[s0].size2();
        for (auto s1 : {LEFT_SIDE, RIGHT_SIDE}) {

          // get orientation of the edge
          const auto sense_col = sideDataPtr->senseMap[s1];

          // number of dofs, for homogenous approximation this value is
          // constant.
          const auto nb_cols = sideDataPtr->indicesColSideMap[s1].size();

          // resize local element matrix
          matSkeleton.resize(nb_rows, nb_cols, false);
          matSkeleton.clear();

          // get base functions, and integration weights
          auto t_row_base = get_ntensor(sideDataPtr->rowBaseSideMap[s0]);
          auto t_w = getFTensor0IntegrationWeight();

          auto arr_t_l =
              make_array(getFTensor0FromVec(sideDataPtr->lVec[LEFT_SIDE]),
                         getFTensor0FromVec(sideDataPtr->lVec[RIGHT_SIDE]));
          auto arr_t_vel = make_array(
              getFTensor1FromMat<SPACE_DIM>(sideDataPtr->velMat[LEFT_SIDE]),
              getFTensor1FromMat<SPACE_DIM>(sideDataPtr->velMat[RIGHT_SIDE]));

          // iterate integration points on face/edge
          for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

            const auto alpha = getMeasure() * t_w;
            const auto dot = (t_normal(i) * (arr_t_vel[LEFT_SIDE](i) +
                                             arr_t_vel[RIGHT_SIDE](i))) *
                             (sense_row / 2);

            auto not_side = [](auto s) {
              return s == LEFT_SIDE ? RIGHT_SIDE : LEFT_SIDE;
            };

            auto get_side = [&]() {
              if (dot > 0)
                return s0;
              else
                return not_side(s0);
            };

            const auto res = alpha * dot;

            size_t rr = 0;
            for (; rr != nb_rows; ++rr) {

              if (s1 == get_side()) {

                auto get_ntensor = [](auto &base_mat, auto gg, auto bb) {
                  double *ptr = &base_mat(gg, bb);
                  return FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(ptr);
                };

                const auto res_row = res * t_row_base;

                auto t_col_base =
                    get_ntensor(sideDataPtr->colBaseSideMap[s1], gg, 0);

                // iterate columns
                for (size_t cc = 0; cc != nb_cols; ++cc) {

                  matSkeleton(rr, cc) += res_row * t_col_base;

                  // move to next column base and element of matrix
                  ++t_col_base;
                }
              }

              // move to next row base
              ++t_row_base;
            }

            // this is obsolete for this particular example, we keep it for
            // generality. in case of multi-physics problems different fields
            // can chare hierarchical base but use different approx. order, so
            // is possible to have more base functions than DOFs on element.
            for (; rr < nb_row_base_functions; ++rr) {
              ++t_row_base;
            }

            ++t_w;
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

MoFEMErrorCode LevelSet::pushOpDomain() {
  MoFEMFunctionBegin;
  auto pip = mField.getInterface<PipelineManager>(); // get interface to
                                                     // pipeline manager

  pip->getOpDomainLhsPipeline().clear();
  pip->getOpDomainRhsPipeline().clear();

  pip->setDomainLhsIntegrationRule([](int, int, int o) { return 4 * o; });
  pip->setDomainRhsIntegrationRule([](int, int, int o) { return 4 * o; });

  auto l_ptr = boost::make_shared<VectorDouble>();
  auto l_dot_ptr = boost::make_shared<VectorDouble>();
  auto vel_ptr = boost::make_shared<MatrixDouble>();

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpDomainRhsPipeline(), {potential_velocity_space, L2});
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpDomainLhsPipeline(), {potential_velocity_space, L2});

  pip->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldValues("L", l_ptr));
  pip->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldValuesDot("L", l_dot_ptr));
  pip->getOpDomainRhsPipeline().push_back(
      new OpCalculateHcurlVectorCurl<potential_velocity_field_dim, SPACE_DIM>(
          "V", vel_ptr));
  pip->getOpDomainLhsPipeline().push_back(
      new OpCalculateHcurlVectorCurl<potential_velocity_field_dim, SPACE_DIM>(
          "V", vel_ptr));

  pip->getOpDomainRhsPipeline().push_back(
      new OpRhsDomain("L", l_ptr, l_dot_ptr, vel_ptr));
  pip->getOpDomainLhsPipeline().push_back(new OpLhsDomain("L", vel_ptr));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode LevelSet::pushOpSkeleton() {
  MoFEMFunctionBegin;
  auto pip = mField.getInterface<PipelineManager>(); // get interface to
                                                     // pipeline manager
  struct OpSwap : public FaceSideEleOp {
    OpSwap(boost::shared_ptr<VectorDouble> l_ptr,
           boost::shared_ptr<MatrixDouble> vel_ptr,
           boost::shared_ptr<SideData> side_data_ptr)
        : FaceSideEleOp(NOSPACE, BoundaryEleOp::OPSPACE), lPtr(l_ptr),
          velPtr(vel_ptr), sideDataPtr(side_data_ptr) {}
    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;
      const auto nb_in_loop = getFEMethod()->nInTheLoop;
      (sideDataPtr->lVec)[nb_in_loop].swap(*lPtr);
      (sideDataPtr->velMat)[nb_in_loop].swap(*velPtr);
      if (!nb_in_loop) {
        (sideDataPtr->lVec)[1].resize(0, false);
        (sideDataPtr->velMat)[1].resize(0, 0, false);
      }
      MoFEMFunctionReturn(0);
    };

  private:
    boost::shared_ptr<VectorDouble> lPtr;
    boost::shared_ptr<MatrixDouble> velPtr;
    boost::shared_ptr<SideData> sideDataPtr;
  };

  struct OpSideData : public FaceSideEleOp {
    OpSideData(boost::shared_ptr<SideData> side_data_ptr)
        : FaceSideEleOp("L", FaceSideEleOp::OPROWCOL),
          sideDataPtr(side_data_ptr) {
      std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
      for (auto t = moab::CN::TypeDimensionMap[SPACE_DIM].first;
           t <= moab::CN::TypeDimensionMap[SPACE_DIM].second; ++t)
        doEntities[t] = true;
    }
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data) {
      MoFEMFunctionBegin;

      if ((CN::Dimension(row_type) == SPACE_DIM) &&
          (CN::Dimension(col_type) == SPACE_DIM)) {
        const auto nb_in_loop = getFEMethod()->nInTheLoop;
        sideDataPtr->indicesRowSideMap[nb_in_loop] = row_data.getIndices();
        sideDataPtr->indicesColSideMap[nb_in_loop] = col_data.getIndices();
        sideDataPtr->rowBaseSideMap[nb_in_loop] = row_data.getN();
        sideDataPtr->colBaseSideMap[nb_in_loop] = col_data.getN();
        sideDataPtr->rowDiffBaseSideMap[nb_in_loop] = row_data.getDiffN();
        sideDataPtr->colDiffBaseSideMap[nb_in_loop] = col_data.getDiffN();
        sideDataPtr->areaMap[nb_in_loop] = getMeasure();
        sideDataPtr->senseMap[nb_in_loop] = getSkeletonSense();
        if (!nb_in_loop) {
          sideDataPtr->indicesRowSideMap[1].clear();
          sideDataPtr->indicesColSideMap[1].clear();
          sideDataPtr->rowBaseSideMap[1].clear();
          sideDataPtr->colBaseSideMap[1].clear();
          sideDataPtr->rowDiffBaseSideMap[1].clear();
          sideDataPtr->colDiffBaseSideMap[1].clear();
          sideDataPtr->areaMap[1] = 0;
          sideDataPtr->senseMap[1] = 0;
        }
      } else {
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Should not happen");
      }

      MoFEMFunctionReturn(0);
    };

  private:
    boost::shared_ptr<SideData> sideDataPtr;
  };

  pip->getOpDomainLhsPipeline().clear();
  pip->getOpDomainRhsPipeline().clear();

  pip->setSkeletonLhsIntegrationRule([](int, int, int o) { return 4 * o; });
  pip->setSkeletonRhsIntegrationRule([](int, int, int o) { return 4 * o; });

  auto l_ptr = boost::make_shared<VectorDouble>();
  auto l_dot_ptr = boost::make_shared<VectorDouble>();
  auto vel_ptr = boost::make_shared<MatrixDouble>();
  auto side_data_ptr = boost::make_shared<SideData>();

  auto side_fe_ptr = boost::make_shared<FaceSideEle>(mField);
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      side_fe_ptr->getOpPtrVector(), {potential_velocity_space, L2});
  side_fe_ptr->getOpPtrVector().push_back(new OpSideData(side_data_ptr));
  side_fe_ptr->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues("L", l_ptr));
  side_fe_ptr->getOpPtrVector().push_back(
      new OpCalculateHcurlVectorCurl<potential_velocity_field_dim, SPACE_DIM>(
          "V", vel_ptr));
  side_fe_ptr->getOpPtrVector().push_back(
      new OpSwap(l_ptr, vel_ptr, side_data_ptr));

  CHKERR AddHOOps<1, SPACE_DIM, SPACE_DIM>::add(pip->getOpSkeletonRhsPipeline(),
                                                {NOSPACE});
  CHKERR AddHOOps<1, SPACE_DIM, SPACE_DIM>::add(pip->getOpSkeletonLhsPipeline(),
                                                {NOSPACE});

  pip->getOpSkeletonRhsPipeline().push_back(
      new OpRhsSkeleton(side_data_ptr, side_fe_ptr));
  pip->getOpSkeletonLhsPipeline().push_back(
      new OpLhsSkeleton(side_data_ptr, side_fe_ptr));

  MoFEMFunctionReturn(0);
}

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

  constexpr double eps = 1e-6;

  auto x =
      opt->setRandomFields(simple->getDM(), {{"L", {-1, 1}}, {"V", {-1, 1}}});
  auto dot_x = opt->setRandomFields(simple->getDM(), {{"L", {-1, 1}}});
  auto diff_x = opt->setRandomFields(simple->getDM(), {{"L", {-1, 1}}});

  auto test_domain_ops = [&](auto fe_name, auto &lhs_pipeline,
                             auto &rhs_pipeline) {
    MoFEMFunctionBegin;

    auto diff_res = opt->checkCentralFiniteDifference(
        simple->getDM(), fe_name, rhs_pipeline, lhs_pipeline, x, dot_x,
        SmartPetscObj<Vec>(), diff_x, 0, 1, eps);

    if (debug) {
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

MoFEMErrorCode LevelSet::initialiseFieldLevelSet() {
  MoFEMFunctionBegin;

  // get operators tester
  auto simple = mField.getInterface<Simple>();
  auto pip = mField.getInterface<PipelineManager>(); // get interface to
                                                     // pipeline manager

  pip->getOpDomainLhsPipeline().clear();
  pip->getOpDomainRhsPipeline().clear();
  pip->setDomainLhsIntegrationRule([](int, int, int o) { return 4 * o; });
  pip->setDomainRhsIntegrationRule([](int, int, int o) { return 4 * o; });

  auto sub_dm = createSmartDM(mField.get_comm(), "DMMOFEM");
  CHKERR DMMoFEMCreateSubDM(sub_dm, simple->getDM(), "SUB_LEVEL");
  CHKERR DMMoFEMSetDestroyProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMSetSquareProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMAddElement(sub_dm, simple->getDomainFEName());
  CHKERR DMMoFEMAddSubFieldRow(sub_dm, "L");
  CHKERR DMSetUp(sub_dm);

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpDomainRhsPipeline(), {potential_velocity_space, L2});
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpDomainLhsPipeline(), {potential_velocity_space, L2});
  pip->getOpDomainLhsPipeline().push_back(new OpMassLL("L", "L"));
  pip->getOpDomainRhsPipeline().push_back(new OpSourceL("L", get_level_set));

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

  if constexpr (debug)
    CHKERR post_proc(sub_dm, "inital_level_set.h5m");

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode LevelSet::initialiseFieldVelocity() {
  MoFEMFunctionBegin;

  // get operators tester
  auto simple = mField.getInterface<Simple>();
  auto pip = mField.getInterface<PipelineManager>(); // get interface to
                                                     // pipeline manager

  pip->getOpDomainLhsPipeline().clear();
  pip->getOpDomainRhsPipeline().clear();
  pip->setDomainLhsIntegrationRule([](int, int, int o) { return 4 * o; });
  pip->setDomainRhsIntegrationRule([](int, int, int o) { return 4 * o; });

  auto sub_dm = createSmartDM(mField.get_comm(), "DMMOFEM");
  CHKERR DMMoFEMCreateSubDM(sub_dm, simple->getDM(), "SUB_VELOCITY");
  CHKERR DMMoFEMSetDestroyProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMSetSquareProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMAddElement(sub_dm, simple->getDomainFEName());
  CHKERR DMMoFEMAddSubFieldRow(sub_dm, "V");
  CHKERR DMSetUp(sub_dm);

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpDomainLhsPipeline(), {potential_velocity_space, L2});
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pip->getOpDomainRhsPipeline(), {potential_velocity_space, L2});

  pip->getOpDomainLhsPipeline().push_back(new OpMassVV("V", "V"));
  pip->getOpDomainRhsPipeline().push_back(
      new OpSourceV("V", get_velocity_potential<SPACE_DIM>));

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

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        post_proc_fe->getOpPtrVector(), {potential_velocity_space});

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    if constexpr (SPACE_DIM == 2) {
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

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode LevelSet::solveLevelSet() {
  MoFEMFunctionBegin;

  // get operators tester
  auto simple = mField.getInterface<Simple>();
  auto pip = mField.getInterface<PipelineManager>(); // get interface to
                                                     // pipeline manager

  CHKERR pushOpDomain();

  auto sub_dm = createSmartDM(mField.get_comm(), "DMMOFEM");
  CHKERR DMMoFEMCreateSubDM(sub_dm, simple->getDM(), "SUB_LEVEL");
  CHKERR DMMoFEMSetDestroyProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMSetSquareProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMAddElement(sub_dm, simple->getDomainFEName());
  CHKERR DMMoFEMAddSubFieldRow(sub_dm, "L");
  CHKERR DMSetUp(sub_dm);

  auto D = smartCreateDMVector(sub_dm);
  CHKERR DMoFEMMeshToLocalVector(sub_dm, D, INSERT_VALUES, SCATTER_FORWARD);

  auto add_post_proc_fe = [&]() {
    auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;
    auto l_vec = boost::make_shared<VectorDouble>();
    auto vel_ptr = boost::make_shared<MatrixDouble>();

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        post_proc_fe->getOpPtrVector(), {H1});
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("L", l_vec));
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateHcurlVectorCurl<potential_velocity_field_dim, SPACE_DIM>(
            "V", vel_ptr));

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(

            post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

            {{"L", l_vec}},

            {{"Velocity", vel_ptr}},

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
  CHKERR TSSetSolution(ts, D);
  auto monitor_ptr = set_time_monitor(sub_dm, ts);
  CHKERR TSSetSolution(ts, D);
  CHKERR TSSetFromOptions(ts);
  CHKERR TSSetUp(ts);
  CHKERR TSSolve(ts, NULL);

  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(sub_dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}