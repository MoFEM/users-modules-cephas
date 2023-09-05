/**
 * \file dynamic_first_order_con_law.cpp
 * \example dynamic_first_order_con_law.cpp
 *
 * Explicit first order conservation laws for solid dynamics
 *
 */

#include <MoFEM.hpp>
#include <MatrixFunction.hpp>

using namespace MoFEM;

// template <int DIM> struct ElementsAndOps {};

// template <> struct ElementsAndOps<2> {
//   using DomainEle = FaceElementForcesAndSourcesCore;
// };

// template <> struct ElementsAndOps<3> {
//   using DomainEle = VolumeElementForcesAndSourcesCore;
// };

template <typename T> inline double trace(FTensor::Tensor2<T, 2, 2> &t_stress) {
  constexpr double third = boost::math::constants::third<double>();
  return (t_stress(0, 0) + t_stress(1, 1));
};

template <typename T> inline double trace(FTensor::Tensor2<T, 3, 3> &t_stress) {
  constexpr double third = boost::math::constants::third<double>();
  return (t_stress(0, 0) + t_stress(1, 1) + t_stress(2, 2));
};

constexpr int SPACE_DIM =
    EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh

using EntData = EntitiesFieldData::EntData;
using DomainEle = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;
using PostProcFaceEle =
    PostProcBrokenMeshInMoab<FaceElementForcesAndSourcesCore>;

using BoundaryEle =
    PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using SetPtsData = FieldEvaluatorInterface::SetPtsData;

template <int DIM> struct PostProcEleByDim;

template <> struct PostProcEleByDim<2> {
  using PostProcEleDomain = PostProcBrokenMeshInMoabBaseCont<DomainEle>;
  using PostProcEleBdy = PostProcBrokenMeshInMoabBaseCont<BoundaryEle>;
  using SideEle = PipelineManager::ElementsAndOpsByDim<2>::FaceSideEle;
};

template <> struct PostProcEleByDim<3> {
  using PostProcEleDomain = PostProcBrokenMeshInMoabBaseCont<BoundaryEle>;
  using PostProcEleBdy = PostProcBrokenMeshInMoabBaseCont<BoundaryEle>;
  using SideEle = PipelineManager::ElementsAndOpsByDim<3>::FaceSideEle;
};

using PostProcEleDomain = PostProcEleByDim<SPACE_DIM>::PostProcEleDomain;
using SideEle = PostProcEleByDim<SPACE_DIM>::SideEle;
using PostProcEleBdy = PostProcEleByDim<SPACE_DIM>::PostProcEleBdy;

using OpMassV = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, SPACE_DIM>;
using OpMassF = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, SPACE_DIM * SPACE_DIM>;

using OpInertiaForce = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 1>;

using OpBodyForce = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 0>;

using DomainNaturalBC =
    NaturalBC<DomainEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
using OpBodyForceVector =
    DomainNaturalBC::OpFlux<NaturalMeshsetTypeVectorScaling<BLOCKSET>, 1,
                            SPACE_DIM>;

using OpGradTimesTensor2 =
    FormsIntegrators<DomainEleOp>::Assembly<AssemblyType::PETSC>::LinearForm<
        IntegrationType::GAUSS>::OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;

using OpRhsTestPiola =
    FormsIntegrators<DomainEleOp>::Assembly<AssemblyType::PETSC>::LinearForm<
        IntegrationType::GAUSS>::OpBaseTimesVector<1, SPACE_DIM * SPACE_DIM, 1>;

using OpGradTimesPiola =
    FormsIntegrators<DomainEleOp>::Assembly<AssemblyType::PETSC>::LinearForm<
        IntegrationType::GAUSS>::OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;

using BoundaryNaturalBC =
    NaturalBC<BoundaryEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
using OpForce = BoundaryNaturalBC::OpFlux<NaturalForceMeshsets, 1, SPACE_DIM>;

/** \brief Save field DOFS on vertices/tags
 */
template <int FIELD_DIM>
struct InitialiseDeformationGradient : public MoFEM::DofMethod {

  MoFEM::Interface &mField;
  std::string tagName;
  InitialiseDeformationGradient(MoFEM::Interface &m_field, std::string tag_name)
      : mField(m_field), tagName(tag_name) {}

  Tag tH;

  MoFEMErrorCode preProcess() {
    MoFEMFunctionBegin;
    if (!fieldPtr) {
      SETERRQ(mField.get_comm(), MOFEM_DATA_INCONSISTENCY,
              "Null pointer, probably field not found");
    }
    // if (fieldPtr->getSpace() != H1) {
    //   SETERRQ(mField.get_comm(), MOFEM_DATA_INCONSISTENCY,
    //           "Field must be in H1 space");
    // }
    std::vector<double> def_vals(fieldPtr->getNbOfCoeffs(), 0);
    rval = mField.get_moab().tag_get_handle(tagName.c_str(), tH);
    if (rval != MB_SUCCESS) {
      CHKERR mField.get_moab().tag_get_handle(
          tagName.c_str(), fieldPtr->getNbOfCoeffs(), MB_TYPE_DOUBLE, tH,
          MB_TAG_CREAT | MB_TAG_SPARSE, &def_vals[0]);
    }

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode operator()() {
    MoFEMFunctionBegin;
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
    // if (dofPtr->getEntType() != MBVERTEX)
    //   MoFEMFunctionReturnHot(0);
    EntityHandle ent = dofPtr->getEnt();
    int rank = dofPtr->getNbOfCoeffs();

    // double tag_val[rank];
    // CHKERR mField.get_moab().tag_get_data(tH, &ent, 1, tag_val);
    // auto t_F = getFTensor2FromPtr<FIELD_DIM,FIELD_DIM>(tag_val);
    // t_F(i, j) = t_kd(i, j);
    // CHKERR mField.get_moab().tag_set_data(tH, &ent, 1, &tag_val);

    Range nodes;
    CHKERR mField.get_moab().get_entities_by_type(0, dofPtr->getEntType(),
                                                  nodes);
    int length;
    CHKERR mField.get_moab().tag_get_length(tH, length);

    std::vector<double> check(/*nodes.size() */ length);
    CHKERR mField.get_moab().tag_get_data(tH, &ent, 1, &*check.begin());

    std::vector<double> data(/*nodes.size() */ length);
    CHKERR mField.get_moab().tag_get_data(tH, &ent, 1, &*data.begin());
    auto t_F = getFTensor2FromPtr<FIELD_DIM, FIELD_DIM>(&*data.begin());
    t_F(i, j) = t_kd(i, j);

    CHKERR mField.get_moab().tag_set_data(tH, &ent, 1, &*data.begin());

    MoFEMFunctionReturn(0);
  }
};

constexpr double rho = 1;
constexpr double omega = 1.;
constexpr double young_modulus = 1.;
constexpr double poisson_ratio = 0.;
double bulk_modulus_K = young_modulus / (3. * (1. - 2. * poisson_ratio));
double shear_modulus_G = young_modulus / (2. * (1. + poisson_ratio));
double mu = young_modulus / (2. * (1. + poisson_ratio));
double lamme_lambda = young_modulus * poisson_ratio /
                      ((1. + poisson_ratio) * (1. - 2. * poisson_ratio));

// #include <HenckyOps.hpp>
// using namespace HenckyOps;

static double *ts_time_ptr;
static double *ts_aa_ptr;

// Operator to Calculate F
template <int DIM_0, int DIM_1>
struct OpCalculateFStab : public ForcesAndSourcesCore::UserDataOperator {
  OpCalculateFStab(boost::shared_ptr<MatrixDouble> def_grad_ptr,
                   boost::shared_ptr<MatrixDouble> def_grad_stab_ptr,
                   boost::shared_ptr<MatrixDouble> def_grad_dot_ptr,
                   double tau_F_ptr, double xi_F_ptr,
                   boost::shared_ptr<MatrixDouble> grad_x_ptr,
                   boost::shared_ptr<MatrixDouble> grad_vel_ptr)
      : ForcesAndSourcesCore::UserDataOperator(NOSPACE, OPLAST),
        defGradPtr(def_grad_ptr), defGradStabPtr(def_grad_stab_ptr),
        defGradDotPtr(def_grad_dot_ptr), tauFPtr(tau_F_ptr), xiF(xi_F_ptr),
        gradxPtr(grad_x_ptr), gradVelPtr(grad_vel_ptr) {}

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;
    // Define Indicies
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;

    // Number of Gauss points
    const size_t nb_gauss_pts = getGaussPts().size2();

    defGradStabPtr->resize(DIM_0 * DIM_1, nb_gauss_pts, false);
    defGradStabPtr->clear();

    // Extract matrix from data matrix
    auto t_F = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*defGradPtr);
    auto t_Fstab = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*defGradStabPtr);
    auto t_F_dot = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*defGradDotPtr);

    // tau_F = alpha deltaT
    auto tau_F = tauFPtr;
    double xi_F = xiF;
    auto t_gradx = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*gradxPtr);
    auto t_gradVel = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*gradVelPtr);

    for (auto gg = 0; gg != nb_gauss_pts; ++gg) {
      // Stabilised Deformation Gradient
      t_Fstab(i, j) = t_F(i, j) + tau_F * (t_gradVel(i, j) - t_F_dot(i, j)) +
                      xi_F * (t_gradx(i, j) - t_F(i, j));

      // if(sqrt(t_gradx(i,j)*t_gradx(i,j)) > 1.e-28)
      //   cerr << t_gradx <<"\n";

      ++t_F;
      ++t_Fstab;
      ++t_gradVel;
      ++t_F_dot;

      ++t_gradx;
    }

    MoFEMFunctionReturn(0);
  }

private:
  double tauFPtr;
  double xiF;
  boost::shared_ptr<MatrixDouble> defGradPtr;
  boost::shared_ptr<MatrixDouble> defGradStabPtr;
  boost::shared_ptr<MatrixDouble> defGradDotPtr;
  boost::shared_ptr<MatrixDouble> gradxPtr;
  boost::shared_ptr<MatrixDouble> gradVelPtr;
};

// Operator to Calculate P
template <int DIM_0, int DIM_1>
struct OpCalculatePiola : public ForcesAndSourcesCore::UserDataOperator {
  OpCalculatePiola(double shear_modulus, double bulk_modulus, double m_u,
                   double lambda_lamme,
                   boost::shared_ptr<MatrixDouble> first_piola_ptr,
                   boost::shared_ptr<MatrixDouble> def_grad_ptr)
      : ForcesAndSourcesCore::UserDataOperator(NOSPACE, OPLAST),
        shearModulus(shear_modulus), bulkModulus(bulk_modulus), mU(m_u),
        lammeLambda(lambda_lamme), firstPiolaPtr(first_piola_ptr),
        defGradPtr(def_grad_ptr) {}

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;
    // Define Indicies
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;

    // Define Kronecker Delta
    constexpr auto t_kd = FTensor::Kronecker_Delta<double>();

    // Number of Gauss points
    const size_t nb_gauss_pts = getGaussPts().size2();

    // Resize Piola
    firstPiolaPtr->resize(DIM_0 * DIM_1, nb_gauss_pts, false); // ignatios check
    firstPiolaPtr->clear();

    // Extract matrix from data matrix
    auto t_P = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*firstPiolaPtr);
    auto t_F = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*defGradPtr);
    const double two_o_three = 2. / 3.;
    const double trace_t_dk = DIM_0;
    for (auto gg = 0; gg != nb_gauss_pts; ++gg) {

      t_P(i, j) = shearModulus * (t_F(i, j) + t_F(j, i) - 2. * t_kd(i, j) -
                                  two_o_three * trace(t_F) * t_kd(i, j) +
                                  two_o_three * trace_t_dk  * t_kd(i, j)) +
                  bulkModulus * trace(t_F) * t_kd(i, j) -
                  bulkModulus * trace_t_dk * t_kd(i, j);

      ++t_F;
      ++t_P;
    }

    MoFEMFunctionReturn(0);
  }

private:
  double shearModulus;
  double bulkModulus;
  double mU;
  double lammeLambda;
  boost::shared_ptr<MatrixDouble> firstPiolaPtr;
  boost::shared_ptr<MatrixDouble> defGradPtr;
};

template <int DIM>
struct OpCalculateDisplacement : public ForcesAndSourcesCore::UserDataOperator {
  OpCalculateDisplacement(boost::shared_ptr<MatrixDouble> spatial_pos_ptr,
                   boost::shared_ptr<MatrixDouble> reference_pos_ptr,
                   boost::shared_ptr<MatrixDouble> u_ptr)
      : ForcesAndSourcesCore::UserDataOperator(NOSPACE, OPLAST),
        xPtr(spatial_pos_ptr),
        XPtr(reference_pos_ptr),
        uPtr(u_ptr) {}

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;
    // Define Indicies
    FTensor::Index<'i', DIM> i;

    // Number of Gauss points
    const size_t nb_gauss_pts = getGaussPts().size2();

    uPtr->resize(DIM, nb_gauss_pts, false); // ignatios check
    uPtr->clear();

    // Extract matrix from data matrix
    auto t_x = getFTensor1FromMat<DIM>(*xPtr);
    auto t_X = getFTensor1FromMat<DIM>(*XPtr);
    auto t_u = getFTensor1FromMat<DIM>(*uPtr);
    for (auto gg = 0; gg != nb_gauss_pts; ++gg) {

      t_u(i) = t_x(i) - t_X(i);
      ++t_u;
      ++t_x;
      ++t_X;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> xPtr;
  boost::shared_ptr<MatrixDouble> XPtr;
  boost::shared_ptr<MatrixDouble> uPtr;
};


template <int DIM_0, int DIM_1>
struct OpCalculatePiolaIncompressibleNH
    : public ForcesAndSourcesCore::UserDataOperator {
  OpCalculatePiolaIncompressibleNH(
      double shear_modulus, double bulk_modulus, double m_u,
      double lambda_lamme, boost::shared_ptr<MatrixDouble> first_piola_ptr,
      boost::shared_ptr<MatrixDouble> def_grad_ptr,
      boost::shared_ptr<MatrixDouble> inv_def_grad_ptr,
      boost::shared_ptr<VectorDouble> det)
      : ForcesAndSourcesCore::UserDataOperator(NOSPACE, OPLAST),
        shearModulus(shear_modulus), bulkModulus(bulk_modulus), mU(m_u),
        lammeLambda(lambda_lamme), firstPiolaPtr(first_piola_ptr),
        defGradPtr(def_grad_ptr), invDefGradPtr(inv_def_grad_ptr), dEt(det) {}

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;
    // Define Indicies
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    FTensor::Index<'l', SPACE_DIM> l;

    // Define Kronecker Delta
    constexpr auto t_kd = FTensor::Kronecker_Delta<double>();

    // Number of Gauss points
    const size_t nb_gauss_pts = getGaussPts().size2();

    // Resize Piola
    firstPiolaPtr->resize(DIM_0 * DIM_1, nb_gauss_pts, false); // ignatios check
    firstPiolaPtr->clear();

    // Extract matrix from data matrix
    auto t_P = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*firstPiolaPtr);
    auto t_F = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*defGradPtr);
    auto t_inv_F = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*invDefGradPtr);
    auto t_det = getFTensor0FromVec<1>(*dEt);
    const double two_o_three = 2. / 3.;
    const double one_o_three = 1. / 3.;
    const double bulk_mod = bulkModulus;
    const double shear_mod = shearModulus;
    for (auto gg = 0; gg != nb_gauss_pts; ++gg) {

      // Nearly incompressible NH
      // volumetric part
      t_P(i, j) = bulk_mod * (t_det - 1.) * t_det * t_inv_F(j, i);
      // deviatoric part
      t_P(i, j) +=
          shear_mod * pow(t_det, two_o_three) *
          (t_F(i, j) - one_o_three * (t_F(l, k) * t_F(l, k)) * t_inv_F(j, i));

      ++t_F;
      ++t_P;
      ++t_inv_F;
      ++t_det;
    }

    MoFEMFunctionReturn(0);
  }

private:
  double shearModulus;
  double bulkModulus;
  double mU;
  double lammeLambda;
  boost::shared_ptr<MatrixDouble> firstPiolaPtr;
  boost::shared_ptr<MatrixDouble> defGradPtr;
  boost::shared_ptr<MatrixDouble> invDefGradPtr;
  boost::shared_ptr<VectorDouble> dEt;
};

template <int DIM_0, int DIM_1>
struct OpCalculateTranspose : public ForcesAndSourcesCore::UserDataOperator {
  OpCalculateTranspose(boost::shared_ptr<MatrixDouble> in_ptr,
                       boost::shared_ptr<MatrixDouble> out_ptr)
      : ForcesAndSourcesCore::UserDataOperator(NOSPACE, OPLAST), inMat(in_ptr),
        outMat(out_ptr) {}

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;
    // Define Indicies
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;

    constexpr auto t_kd = FTensor::Kronecker_Delta<double>();

    // Number of Gauss points
    const size_t nb_gauss_pts = getGaussPts().size2();

    // Resize Piola
    outMat->resize(DIM_0 * DIM_1, nb_gauss_pts, false);
    outMat->clear();

    // Extract matrix from data matrix
    auto t_out = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*outMat);
    auto t_in = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*inMat);
    const double two_o_three = 2. / 3.;
    for (auto gg = 0; gg != nb_gauss_pts; ++gg) {

      t_out(j, i) = t_in(i, j);
      // if((t_out( i, j) * t_out( i, j)) > 1.e-12 )
      //   cerr << "t_in " << t_in << "\n";
      ++t_out;
      ++t_in;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> inMat;
  boost::shared_ptr<MatrixDouble> outMat;
};

template <int DIM_0, int DIM_1>
struct OpCalculateDeformationGradient
    : public ForcesAndSourcesCore::UserDataOperator {
  OpCalculateDeformationGradient(
      boost::shared_ptr<MatrixDouble> def_grad_ptr,
      boost::shared_ptr<MatrixDouble> grad_tensor_ptr)
      : ForcesAndSourcesCore::UserDataOperator(NOSPACE, OPLAST),
        defGradPtr(def_grad_ptr), gradTensorPtr(grad_tensor_ptr) {}

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;
    // Define Indicies
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;

    // Define Kronecker Delta
    constexpr auto t_kd = FTensor::Kronecker_Delta<double>();

    // Number of Gauss points
    const size_t nb_gauss_pts = getGaussPts().size2();

    // Resize Piola
    defGradPtr->resize(DIM_0 * DIM_1, nb_gauss_pts, false);
    defGradPtr->clear();

    // Extract matrix from data matrix
    auto t_F = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*defGradPtr);
    auto t_H = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*gradTensorPtr);
    for (auto gg = 0; gg != nb_gauss_pts; ++gg) {

      t_F(i, j) = t_H(i, j) + t_kd(i, j);

      ++t_F;
      ++t_H;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> gradTensorPtr;
  boost::shared_ptr<MatrixDouble> defGradPtr;
};

struct Example;
struct TSPrePostProc {

  TSPrePostProc() = default;
  virtual ~TSPrePostProc() = default;

  /**
   * @brief Used to setup TS solver
   *
   * @param ts
   * @return MoFEMErrorCode
   */
  MoFEMErrorCode tsSetUp(TS ts);

  // SmartPetscObj<VecScatter> getScatter(Vec x, Vec y, enum FR fr);
  SmartPetscObj<Vec> getSubVector();
  SmartPetscObj<Vec> T;

  SmartPetscObj<DM> solverSubDM;
  SmartPetscObj<Vec> globSol;
  Example *fsRawPtr;
  static MoFEMErrorCode tsPostStage(TS ts, PetscReal stagetime,
                                    PetscInt stageindex, Vec *Y);
  static MoFEMErrorCode tsPostStep(TS ts);
  static MoFEMErrorCode tsPreStep(TS ts);

private:
  /**
   * @brief Pre process time step
   *
   * Refine mesh and update fields
   *
   * @param ts
   * @return MoFEMErrorCode
   */
  // static MoFEMErrorCode tsPreProc(TS ts);

  /**
   * @brief Post process time step.
   *
   * Currently that function do not make anything major
   *
   * @param ts
   * @return MoFEMErrorCode
   */

  SmartPetscObj<Vec> globRes; //< global residual
  SmartPetscObj<Mat> subB;    //< sub problem tangent matrix
  SmartPetscObj<KSP> subKSP;  //< sub problem KSP solver

  boost::shared_ptr<SnesCtx>
      snesCtxPtr; //< infernal data (context) for MoFEM SNES fuctions
  boost::shared_ptr<TsCtx>
      tsCtxPtr; //<  internal data (context) for MoFEM TS functions.
};

static boost::weak_ptr<TSPrePostProc> tsPrePostProc;

struct LinMomTimeScale : public MoFEM::TimeScale {
  using MoFEM::TimeScale::TimeScale;
  double getScale(const double time) {
    return sin(2. * M_PI * MoFEM::TimeScale::getScale(time));
  };
};

struct CommonData {
  SmartPetscObj<Mat> M;   ///< Mass matrix
  SmartPetscObj<KSP> ksp; ///< Linear solver
};

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();
  MoFEMErrorCode checkResults();
  friend struct TSPrePostProc;

  struct DynamicFirstOrderConsSinusTimeScale : public MoFEM::TimeScale {
    using MoFEM::TimeScale::TimeScale;
    double getScale(const double time) { return 0.001 * sin(0.1 * time); };
  };

  struct DynamicFirstOrderConsConstantTimeScale : public MoFEM::TimeScale {
    using MoFEM::TimeScale::TimeScale;
    double getScale(const double time) { return 0.001; };
  };
};

//! [Run problem]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR outputResults();
  CHKERR checkResults();
  MoFEMFunctionReturn(0);
}
//! [Run problem]

//! [Read mesh]
MoFEMErrorCode Example::readMesh() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();

  CHKERR simple->getOptions();
  CHKERR simple->loadFile();
  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  enum bases { AINSWORTH, DEMKOWICZ, LASBASETOPT };
  const char *list_bases[LASBASETOPT] = {"ainsworth", "demkowicz"};
  PetscInt choice_base_value = AINSWORTH;
  CHKERR PetscOptionsGetEList(PETSC_NULL, NULL, "-base", list_bases,
                              LASBASETOPT, &choice_base_value, PETSC_NULL);

  FieldApproximationBase base;
  switch (choice_base_value) {
  case AINSWORTH:
    base = AINSWORTH_LEGENDRE_BASE;
    MOFEM_LOG("WORLD", Sev::inform)
        << "Set AINSWORTH_LEGENDRE_BASE for displacements";
    break;
  case DEMKOWICZ:
    base = DEMKOWICZ_JACOBI_BASE;
    MOFEM_LOG("WORLD", Sev::inform)
        << "Set DEMKOWICZ_JACOBI_BASE for displacements";
    break;
  default:
    base = LASTBASE;
    break;
  }
  // Add field
  CHKERR simple->addDomainField("V", H1, base, SPACE_DIM);
  CHKERR simple->addBoundaryField("V", H1, base, SPACE_DIM);
  CHKERR simple->addDomainField("F", H1, base, SPACE_DIM * SPACE_DIM);
  CHKERR simple->addDataField("x_1", H1, base, SPACE_DIM);
  CHKERR simple->addDataField("x_2", H1, base, SPACE_DIM);
  CHKERR simple->addDataField("F_0", H1, base, SPACE_DIM * SPACE_DIM);
  CHKERR simple->addDataField("F_dot", H1, base, SPACE_DIM * SPACE_DIM);

  CHKERR simple->addDataField("GEOMETRY", H1, base, SPACE_DIM);
  int order = 2;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simple->setFieldOrder("V", order);
  CHKERR simple->setFieldOrder("F", order);
  CHKERR simple->setFieldOrder("F_0", order);
  CHKERR simple->setFieldOrder("F_dot", order);
  CHKERR simple->setFieldOrder("x_1", order);
  CHKERR simple->setFieldOrder("x_2", order);
  CHKERR simple->setFieldOrder("GEOMETRY", order);
  CHKERR simple->setUp();

  auto project_ho_geometry = [&]() {
    Projection10NodeCoordsOnField ent_method_x(mField, "x_1");
    CHKERR mField.loop_dofs("x_1", ent_method_x);
    Projection10NodeCoordsOnField ent_method_x_2(mField, "x_2");
    CHKERR mField.loop_dofs("x_2", ent_method_x_2);

    Projection10NodeCoordsOnField ent_method(mField, "GEOMETRY");
    return mField.loop_dofs("GEOMETRY", ent_method);
  };
  CHKERR project_ho_geometry();
  // InitialiseDeformationGradient<SPACE_DIM> ent_method_F(mField, "F");
  // mField.loop_dofs("F", ent_method_F);

  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Boundary condition]
MoFEMErrorCode Example::boundaryCondition() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  auto time_scale = boost::make_shared<TimeScale>();

  PetscBool sin_time_function = PETSC_FALSE;
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-sin_time_function",
                             &sin_time_function, PETSC_NULL);

  if (sin_time_function)
    time_scale = boost::make_shared<DynamicFirstOrderConsSinusTimeScale>();
  else
    time_scale = boost::make_shared<DynamicFirstOrderConsConstantTimeScale>();

  pipeline_mng->getBoundaryExplicitRhsFE().reset();
  CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
      pipeline_mng->getOpBoundaryExplicitRhsPipeline(), {NOSPACE}, "GEOMETRY");

  CHKERR BoundaryNaturalBC::AddFluxToPipeline<OpForce>::add(
      pipeline_mng->getOpBoundaryExplicitRhsPipeline(), mField, "V",
      {time_scale}, "FORCE", Sev::inform);

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * approx_order;
  };

  CHKERR pipeline_mng->setBoundaryExplicitRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainExplicitRhsIntegrationRule(integration_rule);

  CHKERR bc_mng->removeBlockDOFsOnEntities<DisplacementCubitBcData>(
      simple->getProblemName(), "V");

  auto get_pre_proc_hook = [&]() {
    return EssentialPreProc<DisplacementCubitBcData>(
        mField, pipeline_mng->getDomainExplicitRhsFE(), {time_scale});
  };
  pipeline_mng->getDomainExplicitRhsFE()->preProcessHook = get_pre_proc_hook();

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

typedef struct _RKTableau *RKTableau;
struct _RKTableau {
  char *name;
  PetscInt order; /* Classical approximation order of the method i */
  PetscInt s; /* Number of stages                                           */
  PetscBool FSAL;       /* flag to indicate if tableau is FSAL       */
  PetscInt pinterp;     /* Interpolation order     */
  PetscReal *A, *b, *c; /* Tableau */
  PetscReal *bembed;    /* Embedded formula of order one less (order-1)    */
  PetscReal *binterp;   /* Dense output formula   */
  PetscReal
      ccfl; /* Placeholder for CFL coefficient relative to forward Euler  */
};
typedef struct _RKTableauLink *RKTableauLink;
struct _RKTableauLink {
  struct _RKTableau tab;
  RKTableauLink next;
};

static RKTableauLink RKTableauList;
typedef struct {
  RKTableau tableau;
  Vec *Y;            /* States computed during the step */
  Vec *YdotRHS;      /* Function evaluations for the non-stiff part */
  PetscScalar *work; /* Scalar work */
  PetscReal stage_time;
  // TSStepStatus status;
} TS_RK;

MoFEMErrorCode TSPrePostProc::tsPostStage(TS ts, PetscReal stagetime,
                                          PetscInt stageindex, Vec *Y) {
  MoFEMFunctionBegin;
  // cerr << "tsPostStage " <<"\n";
  if (auto ptr = tsPrePostProc.lock()) {
    auto &m_field = ptr->fsRawPtr->mField;
    auto *simple = m_field.getInterface<Simple>();
    auto *pipeline_mng = m_field.getInterface<PipelineManager>();

    // int n_size;
    // CHKERR TSSSPGetNumStages(ts, &n_size);

    // CHKERR VecGetSize(ptr->T, &size);
    // CHKERR VecGhostUpdateBegin(ptr->T, INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR VecGhostUpdateEnd(ptr->T, INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR DMoFEMMeshToLocalVector(ptr->solverSubDM, ptr->T, INSERT_VALUES,
    // SCATTER_REVERSE);

    // CHKERR VecView(ptr->T, PETSC_VIEWER_STDOUT_WORLD);

    auto fb = m_field.getInterface<FieldBlas>();
    double dt;
    CHKERR TSGetTimeStep(ts, &dt);
    double time;
    CHKERR TSGetTime(ts, &time);
    // PetscPrintf(PETSC_COMM_WORLD, "Timestep %e time %e\n", dt, time);

    // TS_RK *rk = (TS_RK *)(getPetscObject(ts))->data;

    // int ns = rk->tableau->s;
    PetscInt num_stages;
    Vec *stage_solutions;

    CHKERR TSGetStages(ts, &num_stages, &stage_solutions);
    PetscPrintf(PETSC_COMM_WORLD, "Pseudo timestep %d time %e dt %e\n", num_stages,
                time, dt);
    // double pseudo_time_step;
    // CHKERR TSPseudoComputeTimeStep(ts, &pseudo_time_step);
    const double inv_num_step = (double)num_stages;
    CHKERR fb->fieldCopy(1., "x_1", "x_2");
    CHKERR fb->fieldAxpy(dt, "V", "x_2");
    CHKERR fb->fieldCopy(1., "x_2", "x_1");

    CHKERR fb->fieldCopy(-inv_num_step / dt, "F_0", "F_dot");
    CHKERR fb->fieldAxpy(inv_num_step / dt, "F", "F_dot");
    CHKERR fb->fieldCopy(1., "F", "F_0");

    // PetscPrintf(PETSC_COMM_WORLD, "Timestep %e time %e pseudo-time-step
    // %e\n", dt, time, pseudo_time_step);
    // v = (x_t+1 - x_t) / Δt
    // x_t+1 = Δt * v + x_t
    // cerr << "dt " << dt <<"\n";
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode TSPrePostProc::tsPostStep(TS ts) {
  MoFEMFunctionBegin;

  if (auto ptr = tsPrePostProc.lock()) {
    auto &m_field = ptr->fsRawPtr->mField;
    auto fb = m_field.getInterface<FieldBlas>();
    // find trajectory V and F
    //
    // x_t+1 = Δt * v + x_t
    //  CHKERR fb->fieldCopy(1., "x_2", "x_1");
    double dt;
    CHKERR TSGetTimeStep(ts, &dt);
    double time;
    CHKERR TSGetTime(ts, &time);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode TSPrePostProc::tsPreStep(TS ts) {
  MoFEMFunctionBegin;

  if (auto ptr = tsPrePostProc.lock()) {
    auto &m_field = ptr->fsRawPtr->mField;
    auto *simple = m_field.getInterface<Simple>();
    auto *pipeline_mng = m_field.getInterface<PipelineManager>();

    double dt;
    CHKERR TSGetTimeStep(ts, &dt);
    double time;
    CHKERR TSGetTime(ts, &time);
    int step_num;
    CHKERR TSGetStepNumber(ts, &step_num);
    // TSTrajectory tj;
    // CHKERR TSGetTrajectory(ts, &tj);

    // auto T = createDMVector(simple->getDM());
    // Vec TT = smartVectorDuplicate(T);
    // CHKERR TSGetSolution(ts, &TT);

    // CHKERR VecGhostUpdateBegin(TT,
    //                            ADD_VALUES, SCATTER_REVERSE);
    // CHKERR VecGhostUpdateEnd(TT,
    //                          ADD_VALUES, SCATTER_REVERSE);
    // CHKERR VecAssemblyBegin(TT);
    // CHKERR VecAssemblyEnd(TT);

    // CHKERR VecCopy(TT, T);

    // CHKERR TSTrajectoryGetVecs(tj, ts, step_num, &time, T,
    //                            pipeline_mng->getDomainExplicitRhsFE()->ts_u_t);

    // // if (*(pipeline_mng->getDomainExplicitRhsFE()
    // //           ->vecAssembleSwitch)) {

    // CHKERR
    // VecGhostUpdateBegin(pipeline_mng->getDomainExplicitRhsFE()->ts_u_t,
    //                            ADD_VALUES, SCATTER_REVERSE);
    // CHKERR VecGhostUpdateEnd(pipeline_mng->getDomainExplicitRhsFE()->ts_u_t,
    //                          ADD_VALUES, SCATTER_REVERSE);
    // CHKERR VecAssemblyBegin(pipeline_mng->getDomainExplicitRhsFE()->ts_u_t);
    // CHKERR VecAssemblyEnd(pipeline_mng->getDomainExplicitRhsFE()->ts_u_t);
  }
  MoFEMFunctionReturn(0);
}

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystem() {
  MoFEMFunctionBegin;
  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  auto get_body_force = [this](const double, const double, const double) {
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Tensor1<double, SPACE_DIM> t_source;
    t_source(i) = 0.;
    t_source(0) = 0.1;
    t_source(1) = 1.;
    return t_source;
  };

  auto rho_ptr = boost::make_shared<double>(rho);

  auto add_rho_block = [this, rho_ptr](auto &pip, auto block_name, Sev sev) {
    MoFEMFunctionBegin;

    struct OpMatRhoBlocks : public DomainEleOp {
      OpMatRhoBlocks(boost::shared_ptr<double> rho_ptr,
                     MoFEM::Interface &m_field, Sev sev,
                     std::vector<const CubitMeshSets *> meshset_vec_ptr)
          : DomainEleOp(NOSPACE, DomainEleOp::OPSPACE), rhoPtr(rho_ptr) {
        CHK_THROW_MESSAGE(extractRhoData(m_field, meshset_vec_ptr, sev),
                          "Can not get data from block");
      }

      MoFEMErrorCode doWork(int side, EntityType type,
                            EntitiesFieldData::EntData &data) {

        MoFEMFunctionBegin;

        for (auto &b : blockData) {
          if (b.blockEnts.find(getFEEntityHandle()) != b.blockEnts.end()) {
            *rhoPtr = b.rho;
            MoFEMFunctionReturnHot(0);
          }
        }

        *rhoPtr = rho;

        MoFEMFunctionReturn(0);
      }

    private:
      struct BlockData {
        double rho;
        Range blockEnts;
      };

      std::vector<BlockData> blockData;

      MoFEMErrorCode
      extractRhoData(MoFEM::Interface &m_field,
                     std::vector<const CubitMeshSets *> meshset_vec_ptr,
                     Sev sev) {
        MoFEMFunctionBegin;

        for (auto m : meshset_vec_ptr) {
          MOFEM_TAG_AND_LOG("WORLD", sev, "Mat Rho Block") << *m;
          std::vector<double> block_data;
          CHKERR m->getAttributes(block_data);
          if (block_data.size() < 1) {
            SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                    "Expected that block has two attributes");
          }
          auto get_block_ents = [&]() {
            Range ents;
            CHKERR
            m_field.get_moab().get_entities_by_handle(m->meshset, ents, true);
            return ents;
          };

          MOFEM_TAG_AND_LOG("WORLD", sev, "Mat Rho Block")
              << m->getName() << ": ro = " << block_data[0];

          blockData.push_back({block_data[0], get_block_ents()});
        }
        MOFEM_LOG_CHANNEL("WORLD");
        MOFEM_LOG_CHANNEL("WORLD");
        MoFEMFunctionReturn(0);
      }

      boost::shared_ptr<double> rhoPtr;
    };

    //   pip.push_back(new OpMatRhoBlocks(
    //       rho_ptr, mField, sev,

    //       // Get blockset using regular expression
    //       mField.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(std::regex(

    //           (boost::format("%s(.*)") % block_name).str()

    //               ))

    //           ));
    MoFEMFunctionReturn(0);
  };

  // Get pointer to U_tt shift in domain element
  auto get_rho = [rho_ptr](const double, const double, const double) {
    return *rho_ptr;
  };

  // specific time scaling
  auto get_time_scale = [this](const double time) {
    return sin(time * omega * M_PI);
  };

  auto apply_rhs = [&](auto &pip) {
    MoFEMFunctionBegin;

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1},
                                                          "GEOMETRY");


    // Calculate Gradient of velocity
    auto mat_v_grad_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        "V", mat_v_grad_ptr));

    auto mat_v_grad_trans_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateTranspose<SPACE_DIM, SPACE_DIM>(
        mat_v_grad_ptr, mat_v_grad_trans_ptr));

    auto gravity_vector_ptr = boost::make_shared<MatrixDouble>();
    gravity_vector_ptr->resize(SPACE_DIM, 1);
    auto set_body_force = [&]() {
      FTensor::Index<'i', SPACE_DIM> i;
      MoFEMFunctionBegin;
      auto t_force = getFTensor1FromMat<SPACE_DIM, 0>(*gravity_vector_ptr);
      double unit_weight = 1.;
      CHKERR PetscOptionsGetReal(PETSC_NULL, "", "-unit_weight", &unit_weight,
                                 PETSC_NULL);
      t_force(i) = 0;
      if (SPACE_DIM == 2) {
        t_force(1) = -unit_weight;
      } else if (SPACE_DIM == 3) {
        t_force(2) = unit_weight;
      }
      MoFEMFunctionReturn(0);
    };

    // CHKERR set_body_force();
    // pip.push_back(new OpBodyForce(
    //   "V", gravity_vector_ptr, [](double, double, double) { return 1.; }));

    // some operator that calculates F^st
    //  pip.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
    //      "x_2", mat_x_grad_ptr));

    //  // Calculate P
    //   auto mat_P_ptr = boost::make_shared<MatrixDouble>();
    //   pip.push_back(new OpCalculatePiola<SPACE_DIM,
    //   SPACE_DIM>(shear_modulus_G, bulk_modulus_K, mu, lamme_lambda,
    //   mat_P_ptr, mat_F_tensor_ptr));

    // Calculate unknown F
    auto mat_H_tensor_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateTensor2FieldValues<SPACE_DIM, SPACE_DIM>(
        "F", mat_H_tensor_ptr));

    //  // Calculate F
    double tau = 0.2;
    CHKERR PetscOptionsGetReal(PETSC_NULL, "", "-tau", &tau, PETSC_NULL);

    double xi = 0.;
    CHKERR PetscOptionsGetReal(PETSC_NULL, "", "-xi", &xi, PETSC_NULL);

    // Calculate P stab

    auto one = [&](const double, const double, const double) {
      return 3. * bulk_modulus_K;
    };
    auto minus_one = [](const double, const double, const double) {
      return -1.;
    };

    auto mat_dot_F_tensor_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateTensor2FieldValues<SPACE_DIM, SPACE_DIM>(
        "F_dot", mat_dot_F_tensor_ptr));

    // Calculate Gradient of Spatial Positions
    auto mat_x_grad_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        "x_2", mat_x_grad_ptr));

    auto mat_F_tensor_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateDeformationGradient<SPACE_DIM, SPACE_DIM>(
        mat_F_tensor_ptr, mat_H_tensor_ptr));

    auto mat_F_stab_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateFStab<SPACE_DIM, SPACE_DIM>(
          mat_F_tensor_ptr, mat_F_stab_ptr, mat_dot_F_tensor_ptr, tau, xi,
          mat_x_grad_ptr, mat_v_grad_ptr));

    PetscBool is_linear_elasticity = PETSC_TRUE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-is_linear_elasticity",
                               &is_linear_elasticity, PETSC_NULL);

    auto mat_P_stab_ptr = boost::make_shared<MatrixDouble>();
    if (is_linear_elasticity) {
      pip.push_back(new OpCalculatePiola<SPACE_DIM, SPACE_DIM>(
          shear_modulus_G, bulk_modulus_K, mu, lamme_lambda, mat_P_stab_ptr,
          mat_F_stab_ptr));
    } else {
      auto inv_F = boost::make_shared<MatrixDouble>();
      auto det_ptr = boost::make_shared<VectorDouble>();

      pip.push_back(
          new OpInvertMatrix<SPACE_DIM>(mat_F_stab_ptr, det_ptr, inv_F));

        // OpCalculatePiolaIncompressibleNH
      pip.push_back(new OpCalculatePiolaIncompressibleNH<SPACE_DIM, SPACE_DIM>(
          shear_modulus_G, bulk_modulus_K, mu, lamme_lambda, mat_P_stab_ptr,
          mat_F_stab_ptr, inv_F, det_ptr));  
    }

    pip.push_back(new OpGradTimesTensor2("V", mat_P_stab_ptr, minus_one));
    pip.push_back(new OpRhsTestPiola("F", mat_v_grad_ptr, one));

    MoFEMFunctionReturn(0);
  };

  CHKERR apply_rhs(pipeline_mng->getOpDomainExplicitRhsPipeline());

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * approx_order;
  };
  CHKERR pipeline_mng->setDomainExplicitRhsIntegrationRule(integration_rule);
  // CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);

  // auto get_bc_hook = [&]() {
  //   EssentialPreProc<DisplacementCubitBcData> hook(
  //       mField, pipeline_mng->getDomainExplicitRhsFE(),
  //       {boost::make_shared<TimeScale>()});
  //   return hook;
  // };

  // pipeline_mng->getDomainExplicitRhsFE()->preProcessHook = get_bc_hook();

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

/**
 * @brief Monitor solution
 *
 * This functions is called by TS solver at the end of each step. It is used
 * to output results to the hard drive.
 */

struct Monitor : public FEMethod {
  MoFEM::Interface &mField;
  Monitor(SmartPetscObj<DM> dm, MoFEM::Interface &m_field,
          boost::shared_ptr<PostProcEle> post_proc,
          boost::shared_ptr<PostProcFaceEle> post_proc_bdry,
          boost::shared_ptr<MatrixDouble> velocity_field_ptr,
          boost::shared_ptr<MatrixDouble> x2_field_ptr,
          boost::shared_ptr<MatrixDouble> geometry_field_ptr,
          std::array<double, 3> pass_field_eval_coords,
          boost::shared_ptr<SetPtsData> pass_field_eval_data)
      : dM(dm), mField(m_field), postProc(post_proc),
        postProcBdy(post_proc_bdry), velocityFieldPtr(velocity_field_ptr),
        x2FieldPtr(x2_field_ptr), geometryFieldPtr(geometry_field_ptr),
        fieldEvalCoords(pass_field_eval_coords),
        fieldEvalData(pass_field_eval_data){};
  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;

    // cerr << "wagawaga\n";
    auto *simple = mField.getInterface<Simple>();

    if (SPACE_DIM == 3) {
      CHKERR mField.getInterface<FieldEvaluatorInterface>()->evalFEAtThePoint3D(
          fieldEvalCoords.data(), 1e-12, simple->getProblemName(),
          simple->getDomainFEName(), fieldEvalData, mField.get_comm_rank(),
          mField.get_comm_rank(), nullptr, MF_EXIST, QUIET);
    } else {
      CHKERR mField.getInterface<FieldEvaluatorInterface>()->evalFEAtThePoint2D(
          fieldEvalCoords.data(), 1e-12, simple->getProblemName(),
          simple->getDomainFEName(), fieldEvalData, mField.get_comm_rank(),
          mField.get_comm_rank(), nullptr, MF_EXIST, QUIET);
    }

    if (velocityFieldPtr->size1()) {
      auto t_vel = getFTensor1FromMat<SPACE_DIM>(*velocityFieldPtr);
      auto t_x2_field = getFTensor1FromMat<SPACE_DIM>(*x2FieldPtr);
      auto t_geom = getFTensor1FromMat<SPACE_DIM>(*geometryFieldPtr);
      PetscPrintf(PETSC_COMM_WORLD, "Velocities x: %e y: %e z: %e\n", t_vel(0),
                  t_vel(1), t_vel(2));
      double u_x = t_x2_field(0) - t_geom(0);
      double u_y = t_x2_field(1) - t_geom(1);
      double u_z = t_x2_field(2) - t_geom(2);
      PetscPrintf(PETSC_COMM_WORLD, "Displacement x: %e y: %e z: %e\n",
                  u_x, u_y, u_z);
      // cerr << "Velocities x: " << t_p(0) << " y: " << t_p(1) << " z: " <<
      // t_p(2) <<"\n"; MOFEM_LOG("EXAMPLE", Sev::noisy)
      //     << "Velocities x: " << t_p(0) << " y: " << t_p(1) << " z: " <<
      //     t_p(2);
    }

    for (auto m : mField.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(
             std::regex((boost::format("%s(.*)") % "Data_Vertex").str()))) {
      Range ents;
      mField.get_moab().get_entities_by_dimension(m->getMeshset(), 0, ents,
                                                  true);
      auto print_vets = [](boost::shared_ptr<FieldEntity> ent_ptr) {
        MoFEMFunctionBegin;
        if (!(ent_ptr->getPStatus() & PSTATUS_NOT_OWNED)) {
          MOFEM_LOG("SYNC", Sev::inform)
              << "Velocities: " << ent_ptr->getEntFieldData()[0] << " "
              << ent_ptr->getEntFieldData()[1] << " "
              << ent_ptr->getEntFieldData()[2] << "\n";
        }
        MoFEMFunctionReturn(0);
      };
      CHKERR mField.getInterface<FieldBlas>()->fieldLambdaOnEntities(
          print_vets, "V", &ents);
      MOFEM_LOG_SEVERITY_SYNC(mField.get_comm(), Sev::inform);
    }

    PetscBool print_volume = PETSC_FALSE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-print_volume", &print_volume, PETSC_NULL);

    PetscBool print_skin = PETSC_FALSE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-print_skin", &print_skin, PETSC_NULL);

    constexpr int save_every_nth_step = 1;
    if (ts_step % save_every_nth_step == 0) {

      if (print_volume) {
        CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc/*,
                                    this->getCacheWeakPtr()*/);
        CHKERR postProc->writeFile(
            "out_step_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
      }

      if (print_skin) {
        CHKERR DMoFEMLoopFiniteElements(dM, "bFE", postProcBdy);
        CHKERR postProcBdy->writeFile(
            "out_boundary_" + boost::lexical_cast<std::string>(ts_step) +
            ".h5m");
      }
    }
    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcEle> postProc;
  boost::shared_ptr<PostProcFaceEle> postProcBdy;
  boost::shared_ptr<MatrixDouble> velocityFieldPtr;
  boost::shared_ptr<MatrixDouble> x2FieldPtr;
  boost::shared_ptr<MatrixDouble> geometryFieldPtr;
  std::array<double, 3> fieldEvalCoords;
  boost::shared_ptr<SetPtsData> fieldEvalData;
};

//! [Solve]
MoFEMErrorCode Example::solveSystem() {
  MoFEMFunctionBegin;
  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  auto dm = simple->getDM();

  auto calculate_stress_ops = [&](auto &pip) {
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1});
    // auto boundary_post_proc_fe = boost::make_shared<PostProcEle>(mField);

    auto v_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>("V", v_ptr));
    auto X_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("GEOMETRY", X_ptr));

    auto x_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>("x_1", x_ptr));

    // Calculate unknown F
    auto mat_H_tensor_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateTensor2FieldValues<SPACE_DIM, SPACE_DIM>(
        "F", mat_H_tensor_ptr));
  
  auto u_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateDisplacement<SPACE_DIM>(x_ptr, X_ptr, u_ptr));
    // Calculate P

    auto mat_F_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateDeformationGradient<SPACE_DIM, SPACE_DIM>(
        mat_F_ptr, mat_H_tensor_ptr));

    PetscBool is_linear_elasticity = PETSC_TRUE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-is_linear_elasticity",
                               &is_linear_elasticity, PETSC_NULL);

    auto mat_P_ptr = boost::make_shared<MatrixDouble>();
    if (is_linear_elasticity) {
      pip.push_back(new OpCalculatePiola<SPACE_DIM, SPACE_DIM>(
          shear_modulus_G, bulk_modulus_K, mu, lamme_lambda, mat_P_ptr,
          mat_F_ptr));
    } else {
      auto inv_F = boost::make_shared<MatrixDouble>();
      auto det_ptr = boost::make_shared<VectorDouble>();

      pip.push_back(
          new OpInvertMatrix<SPACE_DIM>(mat_F_ptr, det_ptr, inv_F));

      // OpCalculatePiolaIncompressibleNH
      pip.push_back(new OpCalculatePiolaIncompressibleNH<SPACE_DIM, SPACE_DIM>(
          shear_modulus_G, bulk_modulus_K, mu, lamme_lambda, mat_P_ptr,
          mat_F_ptr, inv_F, det_ptr));  
    }

    auto mat_v_grad_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        "V", mat_v_grad_ptr));

    return boost::make_tuple(v_ptr, X_ptr, x_ptr, mat_P_ptr, mat_F_ptr, u_ptr);
  };

  auto post_proc_boundary = [&]() {
    auto boundary_post_proc_fe = boost::make_shared<PostProcFaceEle>(mField);

    AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
        boundary_post_proc_fe->getOpPtrVector(), {}, "GEOMETRY");
    auto op_loop_side =
        new OpLoopSide<SideEle>(mField, simple->getDomainFEName(), SPACE_DIM);
    // push ops to side element, through op_loop_side operator
    auto [boundary_v_ptr, boundary_X_ptr, boundary_x_ptr, boundary_mat_P_ptr,
          boundary_mat_F_ptr, boundary_u_ptr] =
        calculate_stress_ops(op_loop_side->getOpPtrVector());
    boundary_post_proc_fe->getOpPtrVector().push_back(op_loop_side);

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    boundary_post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(

            boundary_post_proc_fe->getPostProcMesh(),
            boundary_post_proc_fe->getMapGaussPts(),

            OpPPMap::DataMapVec{},

            OpPPMap::DataMapMat{{"V", boundary_v_ptr},
                                {"GEOMETRY", boundary_X_ptr},
                                {"x", boundary_x_ptr}, {"U", boundary_u_ptr}},

            OpPPMap::DataMapMat{{"FIRST_PIOLA", boundary_mat_P_ptr},
                                {"F", boundary_mat_F_ptr}},

            OpPPMap::DataMapMat{}

            )

    );
    return boundary_post_proc_fe;
  };

  // Add monitor to time solver

  auto rho_ptr = boost::make_shared<double>(rho);
  auto get_rho = [rho_ptr](const double, const double, const double) {
    return *rho_ptr;
  };

  SmartPetscObj<Mat> M;   ///< Mass matrix
  SmartPetscObj<KSP> ksp; ///< Linear solver

  auto ts_pre_post_proc = boost::make_shared<TSPrePostProc>();
  tsPrePostProc = ts_pre_post_proc;

  CHKERR DMCreateMatrix_MoFEM(dm, M);
  CHKERR MatZeroEntries(M);

  boost::shared_ptr<DomainEle> vol_mass_ele(new DomainEle(mField));

  vol_mass_ele->B = M;

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * approx_order;
  };

  vol_mass_ele->getRuleHook = integration_rule;

  auto energy_consistency = [&](const double, const double, const double) {
    return 3. * bulk_modulus_K;
  };
  vol_mass_ele->getOpPtrVector().push_back(new OpMassV("V", "V"));
  vol_mass_ele->getOpPtrVector().push_back(
      new OpMassF("F", "F", energy_consistency));

  CHKERR DMoFEMLoopFiniteElements(dm, simple->getDomainFEName(), vol_mass_ele);
  CHKERR MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
  CHKERR MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

  auto lumpVec = createDMVector(simple->getDM());
  CHKERR MatGetRowSum(M, lumpVec);

  CHKERR MatZeroEntries(M);
  CHKERR MatDiagonalSet(M, lumpVec, INSERT_VALUES);

  // CHKERR VecView(lumpVec, PETSC_VIEWER_STDOUT_WORLD);

  // auto lumpVecCheck = createDMVector(simple->getDM());
  // CHKERR MatGetRowSum(M, lumpVecCheck);
  // CHKERR VecView(lumpVecCheck, PETSC_VIEWER_STDOUT_WORLD);

  // MatView(M,PETSC_VIEWER_STDOUT_WORLD);
  // Create and septup KSP (linear solver), we need this to calculate g(t,u) =
  // M^-1G(t,u)
  ksp = createKSP(mField.get_comm());
  CHKERR KSPSetOperators(ksp, M, M);
  CHKERR KSPSetFromOptions(ksp);
  CHKERR KSPSetUp(ksp);

  auto solve_boundary_for_g = [&]() {
    MoFEMFunctionBegin;
    if (*(pipeline_mng->getBoundaryExplicitRhsFE()->vecAssembleSwitch)) {

      CHKERR VecGhostUpdateBegin(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F,
                                 ADD_VALUES, SCATTER_REVERSE);
      CHKERR VecGhostUpdateEnd(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F,
                               ADD_VALUES, SCATTER_REVERSE);
      CHKERR VecAssemblyBegin(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F);
      CHKERR VecAssemblyEnd(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F);
      *(pipeline_mng->getBoundaryExplicitRhsFE()->vecAssembleSwitch) = false;

      auto D =
          smartVectorDuplicate(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F);
      // cerr << "ksp.use_count() " <<  ksp.use_count() <<"\n";
      // auto ptr_ksp = ksp();
      CHKERR KSPSolve(ksp, pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, D);
      CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecCopy(D, pipeline_mng->getBoundaryExplicitRhsFE()->ts_F);
    }

    MoFEMFunctionReturn(0);
  };

  pipeline_mng->getBoundaryExplicitRhsFE()->postProcessHook =
      solve_boundary_for_g;

  MoFEM::SmartPetscObj<TS> ts;
  ts = pipeline_mng->createTSEX(dm);

  // Field eval
  PetscBool field_eval_flag = PETSC_TRUE;
  boost::shared_ptr<MatrixDouble> velocity_field_ptr;
  boost::shared_ptr<MatrixDouble> geometry_field_ptr;
  boost::shared_ptr<MatrixDouble> spatial_position_field_ptr;
  boost::shared_ptr<SetPtsData> field_eval_data;

  std::array<double, 3> field_eval_coords = {0.5, 0.5, 5.};
  int dim = 3;
  CHKERR PetscOptionsGetRealArray(NULL, NULL, "-field_eval_coords",
                                  field_eval_coords.data(), &dim,
                                  &field_eval_flag);

  if (field_eval_flag) {
    field_eval_data =
        mField.getInterface<FieldEvaluatorInterface>()->getData<DomainEle>();
    if (SPACE_DIM == 3) {
      CHKERR mField.getInterface<FieldEvaluatorInterface>()->buildTree3D(
          field_eval_data, simple->getDomainFEName());
    } else {
      CHKERR mField.getInterface<FieldEvaluatorInterface>()->buildTree2D(
          field_eval_data, simple->getDomainFEName());
    }

    field_eval_data->setEvalPoints(field_eval_coords.data(), 1);

    auto no_rule = [](int, int, int) { return -1; };

    auto fe_ptr = field_eval_data->feMethodPtr.lock();
    fe_ptr->getRuleHook = no_rule;

    // field_ptr = boost::make_shared<MatrixDouble>();

    velocity_field_ptr = boost::make_shared<MatrixDouble>();
    geometry_field_ptr = boost::make_shared<MatrixDouble>();
    spatial_position_field_ptr = boost::make_shared<MatrixDouble>();
    fe_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("V", velocity_field_ptr));
    fe_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("GEOMETRY", geometry_field_ptr));
    fe_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("x_2", spatial_position_field_ptr));
  }

  auto post_proc_domain = [&]() {
    auto post_proc_fe_vol = boost::make_shared<PostProcEle>(mField);

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    auto [boundary_v_ptr, boundary_X_ptr, boundary_x_ptr, boundary_mat_P_ptr,
          boundary_mat_F_ptr, boundary_u_ptr] =
        calculate_stress_ops(post_proc_fe_vol->getOpPtrVector());

    post_proc_fe_vol->getOpPtrVector().push_back(

        new OpPPMap(

            post_proc_fe_vol->getPostProcMesh(),
            post_proc_fe_vol->getMapGaussPts(),

            {},

            {{"V", boundary_v_ptr},
             {"GEOMETRY", boundary_X_ptr},
             {"x", boundary_x_ptr}, {"U", boundary_u_ptr}},

            {{"FIRST_PIOLA", boundary_mat_P_ptr}, {"F", boundary_mat_F_ptr}},

            {}

            )

    );
    return post_proc_fe_vol;
  };

  boost::shared_ptr<FEMethod> null_fe;
  auto monitor_ptr = boost::make_shared<Monitor>(
      SmartPetscObj<DM>(dm, true), mField, post_proc_domain(),
      post_proc_boundary(), velocity_field_ptr, spatial_position_field_ptr,
      geometry_field_ptr, field_eval_coords, field_eval_data);

  CHKERR DMMoFEMTSSetMonitor(dm, ts, simple->getDomainFEName(), null_fe,
                             null_fe, monitor_ptr);

  double ftime = 1;
  // CHKERR TSSetMaxTime(ts, ftime);
  CHKERR TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);

  auto T = createDMVector(simple->getDM());
  CHKERR DMoFEMMeshToLocalVector(simple->getDM(), T, INSERT_VALUES,
                                 SCATTER_FORWARD);
  CHKERR TSSetSolution(ts, T);
  CHKERR TSSetFromOptions(ts);

  auto fb = mField.getInterface<FieldBlas>();

  CHKERR TSSetPostStage(ts, TSPrePostProc::tsPostStage);
  CHKERR TSSetPostStep(ts, TSPrePostProc::tsPostStep);
  CHKERR TSSetPreStep(ts, TSPrePostProc::tsPreStep);

  boost::shared_ptr<ForcesAndSourcesCore> null;

  if (auto ptr = tsPrePostProc.lock()) {
    // if (true) {

    ptr->fsRawPtr = this;

    //   ptr->T = createDMVector(dm);
    // ptr->T = T;
    // ptr->solverSubDM = simple->getDM();
    CHKERR TSSetUp(ts);
    // CHKERR TSSetSaveTrajectory(ts);
    CHKERR TSSolve(ts, NULL);

    CHKERR TSGetTime(ts, &ftime);

    //   PetscInt steps, snesfails, rejects, nonlimits, limits;
    // #if PETSC_VERSION_GE(3, 8, 0)
    //   CHKERR TSGetStepNumber(ts, &steps);
    // #else
    //   CHKERR TSGetTimeStepNumber(ts, &steps);
    // #endif
    //   // CHKERR TSGetSNESFailures(ts, &snesfails);
    //   // CHKERR TSGetStepRejections(ts, &rejects);
    //   // CHKERR TSGetSNESIterations(ts, &nonlimits);
    //   CHKERR TSGetKSPIterations(ts, &limits);
    //   MOFEM_LOG_C("EXAMPLE", Sev::inform,
    //               "steps %d (%d rejected, %d SNES fails), ftime %g, nonlimits
    //               "
    //               "%d, limits %d\n",
    //               steps, rejects, snesfails, ftime, nonlimits, limits);
  }

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::outputResults() {
  MoFEMFunctionBegin;
  PetscBool test_flg = PETSC_FALSE;
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-test", &test_flg, PETSC_NULL);
  if (test_flg) {
    auto *simple = mField.getInterface<Simple>();
    auto T = createDMVector(simple->getDM());
    CHKERR DMoFEMMeshToLocalVector(simple->getDM(), T, INSERT_VALUES,
                                   SCATTER_FORWARD);
    double nrm2;
    CHKERR VecNorm(T, NORM_2, &nrm2);
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Regression norm " << nrm2;
    constexpr double regression_value = 0.0194561;
    if (fabs(nrm2 - regression_value) > 1e-2)
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
              "Regression test failed; wrong norm value.");
  }
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

//! [Check]
MoFEMErrorCode Example::checkResults() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}
//! [Check]

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "EXAMPLE"));
  LogManager::setLog("EXAMPLE");
  MOFEM_LOG_TAG("EXAMPLE", "example");

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
    MoFEM::Interface &m_field = core; ///< finite element database interface
    //! [Create MoFEM]

    //! [Example]
    Example ex(m_field);
    CHKERR ex.runProblem();
    //! [Example]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
