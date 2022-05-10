/**
 * \file plate.cpp
 * \example plate.cpp
 *
 * Implementation Kirchhoff-Love plate using Discointinous Galerkin (DG-Nitsche
 * method)
 */

/* This file is part of MoFEM.
 * MoFEM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * MoFEM is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with MoFEM. If not, see <http://www.gnu.org/licenses/>. */

#include <MoFEM.hpp>

using namespace MoFEM;

static char help[] = "...\n\n";

#include <BasicFiniteElements.hpp>

constexpr int BASE_DIM = 1;  ///< dimension of base
constexpr int SPACE_DIM = 2; ///< dimension of space
constexpr int FIELD_DIM = 1; ///< dimension of approx. field

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = PipelineManager::FaceEle;
  using DomainEleOp = DomainEle::UserDataOperator;
  using BoundaryEle = PipelineManager::EdgeEle;
  using BoundaryEleOp = BoundaryEle::UserDataOperator;
  using PostProcEle = PostProcFaceOnRefinedMesh;

  using FaceSideEle = MoFEM::FaceElementForcesAndSourcesCoreOnSide;
  using FaceSideOp = FaceSideEle::UserDataOperator;
};

using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = ElementsAndOps<SPACE_DIM>::PostProcEle;

using FaceSideEle = ElementsAndOps<SPACE_DIM>::FaceSideEle;
using FaceSideOp = ElementsAndOps<SPACE_DIM>::FaceSideOp;

using DomainEleOp =
    DomainEle::UserDataOperator;            ///< Finire element operator type
using EntData = EntitiesFieldData::EntData; ///< Data on entities

using OpDomainGradGrad = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpGradGrad<BASE_DIM, FIELD_DIM, SPACE_DIM>;

using OpDomainPlateStiffness =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
        GAUSS>::OpGradGradSymTensorGradGrad<1, 1, SPACE_DIM, 0>;
using OpDomainPlateLoad = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<BASE_DIM, FIELD_DIM>;

// Kronecker delta
constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();

// material parameters
constexpr double lambda = 1;
constexpr double mu = 1;   ///< lame parameter
constexpr double t = 1; ///< plate stiffness

FTensor::Index<'i', SPACE_DIM> i;
FTensor::Index<'j', SPACE_DIM> j;
FTensor::Index<'k', SPACE_DIM> k;
FTensor::Index<'l', SPACE_DIM> l;

static double penalty = 1e6;
static double phi =
    -1; // 1 - symmetric Nitsche, 0 - nonsymmetric, -1 antisymmetrica
static double nitsche = 1;
static int order = 4; // approximation order

auto source = [](const double x, const double y, const double) {
  return cos(2 * x * M_PI) * sin(2 * y * M_PI);
};

/**
 * @brief get fourth-order constitutive tensor
 *
 */
auto plate_stiffness = []() {
  constexpr auto a = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  auto mat_D_ptr = boost::make_shared<MatrixDouble>(a * a, 1);
  auto t_D = getFTensor4DdgFromMat<2, 2, 0>(*(mat_D_ptr));
  constexpr double t3 = t * t * t;
  constexpr double A = mu * t3 / 12;
  constexpr double B = lambda * t3 / 12;
  t_D(i, j, k, l) =
      2 * B * ((t_kd(i, k) ^ t_kd(j, l)) / 4.) + A * t_kd(i, j) * t_kd(k, l);
  // t_D(i, j, k, l) = (t_kd(i, k) ^ t_kd(j, l)) / 4.;
  return mat_D_ptr;
};

enum ElementSide { LEFT_SIDE = 0, RIGHT_SIDE };

// data for skeleton computation
std::array<std::vector<VectorInt>, 2>
    indicesSideMap; ///< indices on rows for left hand-side
std::array<std::vector<MatrixDouble>, 2>
    diffBaseSideMap; // first derivative of base functions
std::array<std::vector<MatrixDouble>, 2>
    diff2BaseSideMap;          // second derivative of base functions
std::array<double, 2> areaMap; // area/volume of elements on the side
std::array<int, 2> senseMap; // orientaton of local element edge/face in respect
                             // to global orientation of edge/face

/**
 * @brief Operator tp collect data from elements on the side of Edge/Face
 *
 */
struct OpCalculateSideData : public FaceSideOp {

  OpCalculateSideData(std::string field_name, std::string col_field_name);

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
};

/**
 * @brief Operator the left hand side matrix
 *
 */
struct OpH1LhsSkeleton : public BoundaryEleOp {
public:
  /**
   * @brief Construct a new OpH1LhsSkeleton
   *
   * @param side_fe_ptr pointer to FE to evaluate side elements
   */
  OpH1LhsSkeleton(boost::shared_ptr<FaceSideEle> side_fe_ptr,
                  boost::shared_ptr<MatrixDouble> d_mat_ptr);

  MoFEMErrorCode doWork(int side, EntityType type,
                        EntitiesFieldData::EntData &data);

private:
  boost::shared_ptr<FaceSideEle>
      sideFEPtr;       ///< pointer to element to get data on edge/face sides
  MatrixDouble locMat; ///< local operator matrix
  boost::shared_ptr<MatrixDouble> dMatPtr;
};

struct Plate {

  Plate(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();

  MoFEM::Interface &mField;
};

//! [Read mesh]
MoFEMErrorCode Plate::readMesh() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  CHKERR simple->getOptions();
  CHKERR simple->loadFile();

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode Plate::setupProblem() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();

  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-penalty", &penalty,
                               PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-phi", &phi, PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-nitsche", &nitsche,
                               PETSC_NULL);

  MOFEM_LOG("WORLD", Sev::inform) << "Set order: " << order;
  MOFEM_LOG("WORLD", Sev::inform) << "Set penalty: " << penalty;
  MOFEM_LOG("WORLD", Sev::inform) << "Set phi: " << phi;
  MOFEM_LOG("WORLD", Sev::inform) << "Set nitche: " << nitsche;

  CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, FIELD_DIM);
  CHKERR simple->addSkeletonField("U", H1, AINSWORTH_LEGENDRE_BASE, FIELD_DIM);
  CHKERR simple->addBoundaryField("U", H1, AINSWORTH_LEGENDRE_BASE, FIELD_DIM);

  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setUp();

  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Boundary condition]
MoFEMErrorCode Plate::boundaryCondition() {
  MoFEMFunctionBegin;

  // get edges and vertices on body skin
  auto get_skin = [&]() {
    Range body_ents;
    MOAB_THROW(
        mField.get_moab().get_entities_by_dimension(0, SPACE_DIM, body_ents));
    Skinner skin(&mField.get_moab());
    Range skin_ents;
    MOAB_THROW(skin.find_skin(0, body_ents, false, skin_ents));
    Range verts;
    MOAB_THROW(mField.get_moab().get_connectivity(skin_ents, verts, true));
    skin_ents.merge(verts);
    return skin_ents;
  };

  // remove dofs on skin edges from priblem
  auto remove_dofs_from_problem = [&](Range &&skin) {
    MoFEMFunctionBegin;
    auto problem_mng = mField.getInterface<ProblemsManager>();
    auto simple = mField.getInterface<Simple>();
    CHKERR problem_mng->removeDofsOnEntities(simple->getProblemName(), "U",
                                             skin, 0, 1);
    MoFEMFunctionReturn(0);
  };

  //  it make plate simply supported
  CHKERR remove_dofs_from_problem(get_skin());

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Plate::assembleSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto rule_lhs = [](int, int, int p) -> int { return 2 * p; };
  auto rule_rhs = [](int, int, int p) -> int { return 2 * p; };
  auto rule_2 = [this](int, int, int) { return 2 * order; };

  CHKERR pipeline_mng->setDomainLhsIntegrationRule(rule_lhs);
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule_rhs);

  CHKERR pipeline_mng->setSkeletonLhsIntegrationRule(rule_2);
  CHKERR pipeline_mng->setSkeletonRhsIntegrationRule(rule_2);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(rule_2);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(rule_2);

  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
  auto base_mass_ptr = boost::make_shared<MatrixDouble>();
  auto data_l2_ptr = boost::make_shared<EntitiesFieldData>(MBENTITYSET);

  auto mat_D_ptr = plate_stiffness();

  auto push_ho_direcatives = [&](auto &pipeline) {
    pipeline.push_back(new OpBaseDerivativesMass<BASE_DIM>(
        base_mass_ptr, data_l2_ptr, AINSWORTH_LEGENDRE_BASE, L2));
    pipeline.push_back(new OpBaseDerivativesNext<BASE_DIM>(
        BaseDerivatives::SecondDerivative, base_mass_ptr, data_l2_ptr,
        AINSWORTH_LEGENDRE_BASE, H1));
  };

  /**
   * @brief calculate jacobian
   *
   */
  auto push_jacobian = [&](auto &pipeline) {
    pipeline.push_back(new OpSetHOWeightsOnFace());
    pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
    pipeline.push_back(
        new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
    // push first base direvatives tp physical element shape
    pipeline.push_back(new OpSetInvJacH1ForFace<1>(inv_jac_ptr));
    // push second base direvatives tp physical element shape
    pipeline.push_back(new OpSetInvJacH1ForFace<2>(inv_jac_ptr));
  };

  push_ho_direcatives(pipeline_mng->getOpDomainLhsPipeline());
  push_jacobian(pipeline_mng->getOpDomainLhsPipeline());

  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpDomainPlateStiffness("U", "U", mat_D_ptr));
  // pipeline_mng->getOpDomainLhsPipeline().push_back(new OpDomainGradGrad(
      // "U", "U", [](const double, const double, const double) { return 1; }));

  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpDomainPlateLoad("U", source));

  // Push operators to the Pipeline for Skeleton
  auto side_fe_ptr = boost::make_shared<FaceSideEle>(mField);
  push_ho_direcatives(side_fe_ptr->getOpPtrVector());
  push_jacobian(side_fe_ptr->getOpPtrVector());
  side_fe_ptr->getOpPtrVector().push_back(new OpCalculateSideData("U", "U"));

  // Push operators to the Pipeline for Skeleton
  pipeline_mng->getOpSkeletonLhsPipeline().push_back(
      new OpH1LhsSkeleton(side_fe_ptr, mat_D_ptr));

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve system]
MoFEMErrorCode Plate::solveSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  auto simple = mField.getInterface<Simple>();

  auto ksp_solver = pipeline_mng->createKSP();
  CHKERR KSPSetFromOptions(ksp_solver);
  CHKERR KSPSetUp(ksp_solver);

  // Create RHS and solution vectors
  auto dm = simple->getDM();
  auto F = smartCreateDMVector(dm);
  auto D = smartVectorDuplicate(F);

  CHKERR KSPSolve(ksp_solver, F, D);

  // Scatter result data on the mesh
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}
//! [Solve system]

//! [Output results]
MoFEMErrorCode Plate::outputResults() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getSkeletonRhsFE().reset();
  pipeline_mng->getSkeletonLhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();
  pipeline_mng->getBoundaryLhsFE().reset();

  auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
  post_proc_fe->generateReferenceElementMesh();
  post_proc_fe->addFieldValuesPostProc("U");
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_result.h5m");

  MoFEMFunctionReturn(0);
}
//! [Output results]

//! [Run program]
MoFEMErrorCode Plate::runProblem() {
  MoFEMFunctionBegin;

  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  // CHKERR checkResults();
  CHKERR outputResults();

  MoFEMFunctionReturn(0);
}
//! [Run program]

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(LogManager::createSink(LogManager::getStrmWorld(), "PL"));
  LogManager::setLog("PL");
  MOFEM_LOG_TAG("PL", "plate");

  try {

    //! [Register MoFEM discrete manager in PETSc]
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);
    //! [Register MoFEM discrete manager in PETSc

    //! [Create MoAB]
    moab::Core mb_instance;              ///< mesh database
    moab::Interface &moab = mb_instance; ///< mesh database interface
    //! [Create MoAB]

    //! [Create MoFEM]
    MoFEM::Core core(moab);           ///< finite element database
    MoFEM::Interface &m_field = core; ///< finite element database insterface
    //! [Create MoFEM]

    //! [Plate]
    Plate ex(m_field);
    CHKERR ex.runProblem();
    //! [Plate]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}

OpCalculateSideData::OpCalculateSideData(std::string row_field_name,
                                         std::string col_field_name)
    : FaceSideOp(row_field_name, col_field_name, FaceSideOp::OPROW) {}

MoFEMErrorCode OpCalculateSideData::doWork(int side, EntityType type,
                                           EntData &data) {
  MoFEMFunctionBeginHot;

  const auto nb_in_loop = getFEMethod()->nInTheLoop;

  auto clear = [](auto nb) {
    indicesSideMap[nb].clear();
    diffBaseSideMap[nb].clear();
    diff2BaseSideMap[nb].clear();
  };

  if (type == MBVERTEX) {
    areaMap[nb_in_loop] = getMeasure();
    senseMap[nb_in_loop] = getEdgeSense();
    if (!nb_in_loop) {
      clear(0);
      clear(1);
      areaMap[1] = 0;
      senseMap[1] = 0;
    }
  }

  const auto nb_dofs = data.getIndices().size();
  if (nb_dofs) {
    indicesSideMap[nb_in_loop].push_back(data.getIndices());
    diffBaseSideMap[nb_in_loop].push_back(
        data.getN(BaseDerivatives::FirstDerivative));
    diff2BaseSideMap[nb_in_loop].push_back(
        data.getN(BaseDerivatives::SecondDerivative));
  }

  MoFEMFunctionReturnHot(0);
}

template <typename T> inline auto get_ntensor(T &base_mat) {
  return FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(
      &*base_mat.data().begin());
};

template <typename T> inline auto get_ntensor(T &base_mat, int gg, int bb) {
  double *ptr = &base_mat(gg, bb);
  return FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(ptr);
};

template <typename T> inline auto get_diff_ntensor(T &base_mat) {
  double *ptr = &*base_mat.data().begin();
  return getFTensor1FromPtr<2>(ptr);
};

template <typename T>
inline auto get_diff_ntensor(T &base_mat, int gg, int bb) {
  double *ptr = &base_mat(gg, 2 * bb);
  return getFTensor1FromPtr<2>(ptr);
};

template <typename T> inline auto get_diff2_ntensor(T &base_mat) {
  double *ptr = &*base_mat.data().begin();
  return getFTensor2SymmetricLowerFromPtr<2>(ptr);
};

template <typename T>
inline auto get_diff2_ntensor(T &base_mat, int gg, int bb) {
  double *ptr = &base_mat(gg, 4 * bb);
  return getFTensor2SymmetricLowerFromPtr<2>(ptr);
};

/**
 * @brief Construct a new OpH1LhsSkeleton
 *
 * @param side_fe_ptr pointer to FE to evaluate side elements
 */
OpH1LhsSkeleton::OpH1LhsSkeleton(boost::shared_ptr<FaceSideEle> side_fe_ptr,
                                 boost::shared_ptr<MatrixDouble> mat_d_ptr)
    : BoundaryEleOp(NOSPACE, BoundaryEleOp::OPSPACE), sideFEPtr(side_fe_ptr),
      dMatPtr(mat_d_ptr) {}

MoFEMErrorCode OpH1LhsSkeleton::doWork(int side, EntityType type,
                                       EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  // Collect data from side domian elements
  CHKERR loopSideFaces("dFE", *sideFEPtr);
  const auto in_the_loop =
      sideFEPtr->nInTheLoop; // return number of elements on the side

  // calculate  penalty
  const double s = getMeasure() / (areaMap[0] + areaMap[1]);
  const double p = penalty * s;

  // get normal of the face or edge
  auto t_normal = getFTensor1Normal();
  t_normal.normalize();

  // Elastic stiffness tensor (4th rank tensor with minor and major
  // symmetry)
  auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*dMatPtr);

  // get number of integration points on face
  const size_t nb_integration_pts = getGaussPts().size2();

  // beta paramter is zero, when penalty method is used, also, takes value 1,
  // when boundary edge/face is evaluated, and 2 when is skeleton edge/face.
  const double beta = static_cast<double>(nitsche) / (in_the_loop + 1);

  auto integrate = [&](auto sense_row, auto &row_ind, auto &row_diff,
                       auto &row_diff2, auto sense_col, auto &col_ind,
                       auto &col_diff, auto &col_diff2) {
    MoFEMFunctionBeginHot;

    // number of dofs, for homogenous approximation this value is
    // constant.
    const auto nb_rows = row_ind.size();
    const auto nb_cols = col_ind.size();

    const auto nb_row_base_functions = row_diff.size2() / SPACE_DIM;

    if (nb_cols) {

      // resize local element matrix
      locMat.resize(nb_rows, nb_cols, false);
      locMat.clear();

      // get base functions, and integration weights
      auto t_diff_row_base = get_diff_ntensor(row_diff);
      auto t_diff2_row_base = get_diff2_ntensor(row_diff2);

      auto t_w = getFTensor0IntegrationWeight();

      // iterate integration points on face/edge
      for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

        // t_w is integration weight, and measure is element area, or
        // volume, depending if problem is in 2d/3d.
        const double alpha = getMeasure() * t_w;
        auto t_mat = locMat.data().begin();

        // iterate rows
        size_t rr = 0;
        for (; rr != nb_rows; ++rr) {

          FTensor::Tensor2_symmetric<double, SPACE_DIM> t_mv;
          t_mv(i, j) = t_D(i, j, k, l) * t_diff2_row_base(k, l);

          // calculate tetting function
          FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_vn_plus;
          t_vn_plus(i, j) = beta * (phi * t_mv(i, j) / p);
          FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_vn;
          t_vn(i, j) = (t_diff_row_base(j) * (t_normal(i) * sense_row)) -
                       t_vn_plus(i, j);

          // get base functions on columns
          auto t_diff_col_base = get_diff_ntensor(col_diff, gg, 0);
          auto t_diff2_col_base = get_diff2_ntensor(col_diff2, gg, 0);

          // iterate columns
          for (size_t cc = 0; cc != nb_cols; ++cc) {

            FTensor::Tensor2_symmetric<double, SPACE_DIM> t_mu;
            t_mu(i, j) = t_D(i, j, k, l) * t_diff2_col_base(k, l);

            // // calculate variance of tested function
            FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_un;
            t_un(i, j) = -p * ((t_diff_col_base(j) * (t_normal(i) * sense_col) -
                                beta * t_mu(i, j) / p));

            // assemble matrix
            *t_mat -= alpha * (t_vn(i, j) * t_un(i, j));
            *t_mat -= alpha * (t_vn_plus(i, j) * (beta * t_mu(i, j)));

            // move to next column base and element of matrix
            ++t_diff_col_base;
            ++t_diff2_col_base;
            ++t_mat;
          }

          // move to next row base
          ++t_diff_row_base;
          ++t_diff2_row_base;
        }

        // this is obsolete for this particular example, we keep it for
        // generality. in case of multi-physcis problems diffrent fields
        // can chare hierarchical base but use diffrent approx. order,
        // so is possible to have more base functions than DOFs on
        // element.
        for (; rr < nb_row_base_functions; ++rr) {
          ++t_diff_row_base;
          ++t_diff2_row_base;
        }

        ++t_w;
      }

      // assemble system
      CHKERR ::MatSetValues(getKSPB(), nb_rows, &*row_ind.begin(),
                            col_ind.size(), &*col_ind.begin(),
                            &*locMat.data().begin(), ADD_VALUES);
    }

    MoFEMFunctionReturnHot(0);
  };

  // iterate the sides rows
  for (auto s0 : {LEFT_SIDE, RIGHT_SIDE}) {

    const auto sense_row = senseMap[s0];

    for (auto x0 = 0; x0 != indicesSideMap[s0].size(); ++x0) {

        for (auto s1 : {LEFT_SIDE, RIGHT_SIDE}) {
          const auto sense_col = senseMap[s1];

          for (auto x1 = 0; x1 != indicesSideMap[s1].size(); ++x1) {

            CHKERR integrate(sense_row, indicesSideMap[s0][x0],
                             diffBaseSideMap[s0][x0], diff2BaseSideMap[s0][x0],

                             sense_col, indicesSideMap[s1][x1],
                             diffBaseSideMap[s1][x1], diff2BaseSideMap[s1][x1]

            );
          }
        }
      
    }
  }

  MoFEMFunctionReturn(0);
}