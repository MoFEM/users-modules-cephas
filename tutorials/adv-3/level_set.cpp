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

struct LevelSet {

  LevelSet(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();

  MoFEMErrorCode pushOpDomain();

  boost::shared_ptr<FaceSideEle> getSideFE();  
  MoFEMErrorCode pushOpSkeleton();

  MoFEMErrorCode testOp();

  MoFEMErrorCode initialiseFieldLevelSet(
      boost::function<double(double, double, double)> level_fun =
          get_level_set);
  MoFEMErrorCode initialiseFieldVelocity(
      boost::function<double(double, double, double)> vel_fun =
          get_velocity_potential<SPACE_DIM>);
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

  enum ElementSide { LEFT_SIDE = 0, RIGHT_SIDE = 1 };

  struct SideData {
    // data for skeleton computation
    std::array<VectorInt, 2>
        indicesRowSideMap; ///< indices on rows for left hand-side
    std::array<VectorInt, 2>
        indicesColSideMap; ///< indices on columns for left hand-side
    std::array<MatrixDouble, 2> rowBaseSideMap; // base functions on rows
    std::array<MatrixDouble, 2> colBaseSideMap; // base function  on columns
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

  if constexpr (debug)
    CHKERR testOp();
  // CHKERR initialiseFieldVelocity();
  // CHKERR initialiseFieldLevelSet();
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
  CHKERR simple->setFieldOrder("L", 2);
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

        auto t_row_base = get_ntensor(sideDataPtr->rowBaseSideMap[s0]);
        for (int gg = 0; gg != nb_gauss_pts; ++gg) {
          const auto dot =
              (sense_row / 2.) * (t_normal(i) * (arr_t_vel[LEFT_SIDE](i) +
                                                 arr_t_vel[RIGHT_SIDE](i)));
          const auto l_upwind = (dot > 0) ? arr_t_l[s0] : arr_t_l[opposite_s0];
          const auto alpha = t_w;
          const auto res = alpha * dot * l_upwind;
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
            const auto dot =
                (sense_row / 2.) * (t_normal(i) * (arr_t_vel[LEFT_SIDE](i) +
                                                   arr_t_vel[RIGHT_SIDE](i)));
            const auto l_upwind_side = (dot > 0) ? s0 : opposite_s0;
            const auto alpha = t_w;
            const auto res = alpha * dot;
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

MoFEMErrorCode LevelSet::pushOpDomain() {
  MoFEMFunctionBegin;
  auto pip = mField.getInterface<PipelineManager>(); // get interface to
                                                     // pipeline manager

  pip->getOpDomainLhsPipeline().clear();
  pip->getOpDomainRhsPipeline().clear();

  pip->setDomainLhsIntegrationRule([](int, int, int o) { return 3 * o; });
  pip->setDomainRhsIntegrationRule([](int, int, int o) { return 3 * o; });

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

boots::shared_ptr<FaceSideEle> getSideFE() {


};
 
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
      (sideDataPtr->lVec)[nb_in_loop] = *lPtr;
      (sideDataPtr->velMat)[nb_in_loop] = *velPtr;
      if (!nb_in_loop) {
        (sideDataPtr->lVec)[1] = sideDataPtr->lVec[0];
        (sideDataPtr->velMat)[1] = (sideDataPtr->velMat)[0];
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
        const auto nb_in_loop = getFEMethod()->nInTheLoop;
        sideDataPtr->indicesRowSideMap[nb_in_loop] = row_data.getIndices();
        sideDataPtr->indicesColSideMap[nb_in_loop] = col_data.getIndices();
        sideDataPtr->rowBaseSideMap[nb_in_loop] = row_data.getN();
        sideDataPtr->colBaseSideMap[nb_in_loop] = col_data.getN();
        sideDataPtr->areaMap[nb_in_loop] = getMeasure();
        sideDataPtr->senseMap[nb_in_loop] = getSkeletonSense();
        if (!nb_in_loop) {
          sideDataPtr->indicesRowSideMap[1].clear();
          sideDataPtr->indicesColSideMap[1].clear();
          sideDataPtr->rowBaseSideMap[1].clear();
          sideDataPtr->colBaseSideMap[1].clear();
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

  pip->getOpSkeletonLhsPipeline().clear();
  pip->getOpSkeletonRhsPipeline().clear();

  pip->setSkeletonLhsIntegrationRule([](int, int, int o) { return 3 * o; });
  pip->setSkeletonRhsIntegrationRule([](int, int, int o) { return 3 * o; });

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

  if constexpr (debug) {

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

    struct DivergenceSkeleton : public BoundaryEleOp {
      DivergenceSkeleton(boost::shared_ptr<SideData> side_data_ptr,
                         boost::shared_ptr<FaceSideEle> side_fe_ptr,
                         SmartPetscObj<Vec> div_vec)
          : BoundaryEleOp(NOSPACE, BoundaryEleOp::OPSPACE),
            sideDataPtr(side_data_ptr), sideFEPtr(side_fe_ptr),
            divVec(div_vec) {}
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
              const auto l_upwind =
                  (dot > 0) ? arr_t_l[s0] : arr_t_l[opposite_s0];
              for (int rr = 0; rr != nb_dofs; ++rr) {
                div += t_w * t_base * l_upwind * dot;
                ++t_base;
              }
              ++t_w;
              next();
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

    vol_fe->getRuleHook = [](int, int, int) { return 6; };
    skel_fe->getRuleHook = [](int, int, int) { return 6; };

    auto div_vol_vec = createSmartVectorMPI(mField.get_comm(), PETSC_DECIDE, 1);
    auto div_skel_vec =
        createSmartVectorMPI(mField.get_comm(), PETSC_DECIDE, 1);

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        vol_fe->getOpPtrVector(), {potential_velocity_space, L2});
    vol_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("L", l_ptr));
    vol_fe->getOpPtrVector().push_back(
        new OpCalculateHcurlVectorCurl<potential_velocity_field_dim, SPACE_DIM>(
            "V", vel_ptr));
    vol_fe->getOpPtrVector().push_back(
        new DivergenceVol(l_ptr, vel_ptr, div_vol_vec));

    CHKERR AddHOOps<1, SPACE_DIM, SPACE_DIM>::add(skel_fe->getOpPtrVector(),
                                                  {NOSPACE});
    skel_fe->getOpPtrVector().push_back(
        new DivergenceSkeleton(side_data_ptr, side_fe_ptr, div_skel_vec));

    auto simple = mField.getInterface<Simple>();
    auto dm = simple->getDM();

    CHKERR initialiseFieldVelocity(
        [](double x, double y, double z) { return x * x + y * y; });
    CHKERR initialiseFieldLevelSet(
        [](double x, double y, double) { return x * x + y * y; });

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
    MOFEM_LOG("WORLD", Sev::inform)
        << "Testing divergence skeleton: " << div_skel
        << " relative difference: " << eps;

    constexpr double eps_err = 1e-6;
    if (eps > eps_err)
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "No consistency between skeleton integral and volume integral");
  }

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

  boost::shared_ptr<FEMethod> lhs_fe = boost::make_shared<DomainEle>(mField);
  boost::shared_ptr<FEMethod> rhs_fe = boost::make_shared<DomainEle>(mField);
  auto swap_fe = [&]() {
    lhs_fe.swap(pip->getDomainLhsFE());
    rhs_fe.swap(pip->getDomainRhsFE());
  };
  swap_fe();

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
  pip->getOpDomainRhsPipeline().push_back(new OpSourceL("L", level_fun));

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
    CHKERR post_proc(sub_dm, "initial_level_set.h5m");

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

  boost::shared_ptr<FEMethod> lhs_fe = boost::make_shared<DomainEle>(mField);
  boost::shared_ptr<FEMethod> rhs_fe = boost::make_shared<DomainEle>(mField);
  auto swap_fe = [&]() {
    lhs_fe.swap(pip->getDomainLhsFE());
    rhs_fe.swap(pip->getDomainRhsFE());
  };
  swap_fe();

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

  swap_fe();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode LevelSet::solveLevelSet() {
  MoFEMFunctionBegin;

  // get operators tester
  auto simple = mField.getInterface<Simple>();
  auto pip = mField.getInterface<PipelineManager>(); // get interface to
                                                     // pipeline manager

  CHKERR pushOpDomain();
  CHKERR pushOpSkeleton();

  auto sub_dm = createSmartDM(mField.get_comm(), "DMMOFEM");
  CHKERR DMMoFEMCreateSubDM(sub_dm, simple->getDM(), "SUB_LEVEL");
  CHKERR DMMoFEMSetDestroyProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMSetSquareProblem(sub_dm, PETSC_TRUE);
  CHKERR DMMoFEMAddElement(sub_dm, simple->getDomainFEName());
  CHKERR DMMoFEMAddElement(sub_dm, simple->getSkeletonFEName());
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