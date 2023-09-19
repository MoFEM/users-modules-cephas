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


enum FR { F, R }; // F - forward, and reverse

/** \brief Save field DOFS on vertices/tags
 */

constexpr double omega = 1.;
constexpr double young_modulus = 1.;
constexpr double poisson_ratio = 0.;
double bulk_modulus_K = young_modulus / (3. * (1. - 2. * poisson_ratio));
double shear_modulus_G = young_modulus / (2. * (1. + poisson_ratio));
double mu = young_modulus / (2. * (1. + poisson_ratio));
double lamme_lambda = young_modulus * poisson_ratio /
                      ((1. + poisson_ratio) * (1. - 2. * poisson_ratio));

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
                                  two_o_three * trace_t_dk * t_kd(i, j)) +
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
        xPtr(spatial_pos_ptr), XPtr(reference_pos_ptr), uPtr(u_ptr) {}

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

  // SmartPetscObj<VecScatter> getScatter(Vec x, Vec y, enum FR fr);
  Example *fsRawPtr;
  static MoFEMErrorCode tsPostStage(TS ts, PetscReal stagetime,
                                    PetscInt stageindex, Vec *Y);
  static MoFEMErrorCode tsPostStep(TS ts);
  static MoFEMErrorCode tsPreStep(TS ts);
};

static boost::weak_ptr<TSPrePostProc> tsPrePostProc;

struct LinMomTimeScale : public MoFEM::TimeScale {
  using MoFEM::TimeScale::TimeScale;
  double getScale(const double time) {
    return sin(2. * M_PI * MoFEM::TimeScale::getScale(time));
  };
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

MoFEMErrorCode TSPrePostProc::tsPostStage(TS ts, PetscReal stagetime,
                                          PetscInt stageindex, Vec *Y) {
  MoFEMFunctionBegin;
  // cerr << "tsPostStage " <<"\n";
  if (auto ptr = tsPrePostProc.lock()) {
    auto &m_field = ptr->fsRawPtr->mField;
    auto *simple = m_field.getInterface<Simple>();
    auto *pipeline_mng = m_field.getInterface<PipelineManager>();

    auto fb = m_field.getInterface<FieldBlas>();
    double dt;
    CHKERR TSGetTimeStep(ts, &dt);
    double time;
    CHKERR TSGetTime(ts, &time);
    PetscInt num_stages;
    Vec *stage_solutions;

    CHKERR TSGetStages(ts, &num_stages, &stage_solutions);
    PetscPrintf(PETSC_COMM_WORLD, "Check timestep %d time %e dt %e\n",
                num_stages, time, dt);

    const double inv_num_step = (double)num_stages;
    CHKERR fb->fieldCopy(1., "x_1", "x_2");
    CHKERR fb->fieldAxpy(dt, "V", "x_2");
    CHKERR fb->fieldCopy(1., "x_2", "x_1");

    CHKERR fb->fieldCopy(-inv_num_step / dt, "F_0", "F_dot");
    CHKERR fb->fieldAxpy(inv_num_step / dt, "F", "F_dot");
    CHKERR fb->fieldCopy(1., "F", "F_0");
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode TSPrePostProc::tsPostStep(TS ts) {
  MoFEMFunctionBegin;

  if (auto ptr = tsPrePostProc.lock()) {
    auto &m_field = ptr->fsRawPtr->mField;
    auto fb = m_field.getInterface<FieldBlas>();
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

    auto gravity_vector_ptr = boost::make_shared<MatrixDouble>();
    gravity_vector_ptr->resize(SPACE_DIM, 1);
    auto set_body_force = [&]() {
      FTensor::Index<'i', SPACE_DIM> i;
      MoFEMFunctionBegin;
      auto t_force = getFTensor1FromMat<SPACE_DIM, 0>(*gravity_vector_ptr);
      double unit_weight = 0.;
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

    CHKERR set_body_force();
    pip.push_back(new OpBodyForce("V", gravity_vector_ptr,
                                  [](double, double, double) { return 1.; }));

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
          mField.get_comm_rank(), getCacheWeakPtr().lock(), MF_EXIST, QUIET);
    } else {
      CHKERR mField.getInterface<FieldEvaluatorInterface>()->evalFEAtThePoint2D(
          fieldEvalCoords.data(), 1e-12, simple->getProblemName(),
          simple->getDomainFEName(), fieldEvalData, mField.get_comm_rank(),
          mField.get_comm_rank(), getCacheWeakPtr().lock(), MF_EXIST, QUIET);
    }

    if (velocityFieldPtr->size1()) {
      auto t_vel = getFTensor1FromMat<SPACE_DIM>(*velocityFieldPtr);
      auto t_x2_field = getFTensor1FromMat<SPACE_DIM>(*x2FieldPtr);
      auto t_geom = getFTensor1FromMat<SPACE_DIM>(*geometryFieldPtr);

      double u_x = t_x2_field(0) - t_geom(0);
      double u_y = t_x2_field(1) - t_geom(1);
      double u_z = t_x2_field(2) - t_geom(2);

      MOFEM_LOG("SYNC", Sev::inform)
          << "Velocities x: " << t_vel(0) << " y: " << t_vel(1)
          << " z: " << t_vel(2) << "\n";
      MOFEM_LOG("SYNC", Sev::inform) << "Displacement x: " << u_x
                                     << " y: " << u_y << " z: " << u_z << "\n";
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
    }
    MOFEM_LOG_SEVERITY_SYNC(mField.get_comm(), Sev::inform);

    PetscBool print_volume = PETSC_FALSE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-print_volume", &print_volume,
                               PETSC_NULL);

    PetscBool print_skin = PETSC_FALSE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-print_skin", &print_skin,
                               PETSC_NULL);

    int save_every_nth_step = 1;
    CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-save_step",
                              &save_every_nth_step, PETSC_NULL);
    if (ts_step % save_every_nth_step == 0) {

      if (print_volume) {
        CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc);
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
  MoFEM::SmartPetscObj<TS> ts;
  ts = pipeline_mng->createTSEX(dm);

  auto calculate_stress_ops = [&](auto &pip) {
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1});

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

      pip.push_back(new OpInvertMatrix<SPACE_DIM>(mat_F_ptr, det_ptr, inv_F));

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
                                {"x", boundary_x_ptr},
                                {"U", boundary_u_ptr}},

            OpPPMap::DataMapMat{{"FIRST_PIOLA", boundary_mat_P_ptr},
                                {"F", boundary_mat_F_ptr}},

            OpPPMap::DataMapMat{}

            )

    );
    return boundary_post_proc_fe;
  };

  // Add monitor to time solver

  double rho = 1.;
  CHKERR PetscOptionsGetReal(PETSC_NULL, "", "-density", &rho, PETSC_NULL);
  auto get_rho = [rho](const double, const double, const double) {
    return rho;
  };

  ublas::vector<Vec> nested_vectors(2);
  ublas::vector<IS> nested_is(2);
  auto create_subdm = [&](const std::string &f) {
    DM subdm;
    CHKERR DMCreate(mField.get_comm(), &subdm);
    CHKERR DMSetType(subdm, "DMMOFEM");
    const std::string dm_name = "SUB_" + f + f;
    CHKERR DMMoFEMCreateSubDM(subdm, dm, dm_name.c_str());

    CHKERR DMMoFEMSetSquareProblem(subdm, PETSC_TRUE);
    CHKERR DMMoFEMSetDestroyProblem(subdm, PETSC_TRUE);

    CHKERR DMMoFEMAddSubFieldRow(subdm, f.c_str());
    CHKERR DMMoFEMAddSubFieldCol(subdm, f.c_str());
    CHKERR DMMoFEMAddElement(subdm, simple->getDomainFEName().c_str());

    if (f == "V")
      CHKERR DMMoFEMAddElement(subdm, simple->getBoundaryFEName().c_str());
      
    CHKERR DMSetUp(subdm);
    if (f == "V") {
      CHKERR DMMoFEMGetSubRowIS(subdm, &nested_is[0]);
    } else {
      CHKERR DMMoFEMGetSubRowIS(subdm, &nested_is[1]);
    }

    return SmartPetscObj<DM>(subdm);
  };

  SmartPetscObj<Mat> M_VV;   ///< Mass matrix
  SmartPetscObj<Mat> M_FF;   ///< Mass matrix

  auto dm_sub_VV = create_subdm("V");
  auto dm_sub_FF = create_subdm("F");
  // cerr << "CREATE!\n";
  auto ts_pre_post_proc = boost::make_shared<TSPrePostProc>();
  tsPrePostProc = ts_pre_post_proc;

  CHKERR DMCreateMatrix_MoFEM(dm_sub_VV, M_VV);
  CHKERR MatZeroEntries(M_VV);

  CHKERR DMCreateMatrix_MoFEM(dm_sub_FF, M_FF);
  CHKERR MatZeroEntries(M_FF);
  
  boost::shared_ptr<DomainEle> vol_mass_ele_VV(new DomainEle(mField));
  boost::shared_ptr<DomainEle> vol_mass_ele_FF(new DomainEle(mField));

  vol_mass_ele_VV->B = M_VV;
  vol_mass_ele_FF->B = M_FF;

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * approx_order;
  };

  vol_mass_ele_VV->getRuleHook = integration_rule;
  vol_mass_ele_FF->getRuleHook = integration_rule;

  vol_mass_ele_VV->getOpPtrVector().push_back(new OpMassV("V", "V", get_rho));
  vol_mass_ele_FF->getOpPtrVector().push_back(new OpMassF("F", "F"));

  CHKERR DMoFEMLoopFiniteElements(dm_sub_VV, simple->getDomainFEName(), vol_mass_ele_VV);
  CHKERR MatAssemblyBegin(M_VV, MAT_FINAL_ASSEMBLY);
  CHKERR MatAssemblyEnd(M_VV, MAT_FINAL_ASSEMBLY);

  CHKERR DMoFEMLoopFiniteElements(dm_sub_FF, simple->getDomainFEName(), vol_mass_ele_FF);
  CHKERR MatAssemblyBegin(M_FF, MAT_FINAL_ASSEMBLY);
  CHKERR MatAssemblyEnd(M_FF, MAT_FINAL_ASSEMBLY);

  // auto lumpVec_VV = createDMVector(dm_sub_VV);
  // CHKERR MatGetRowSum(M_VV, lumpVec_VV);

  // CHKERR MatZeroEntries(M_VV);
  // CHKERR MatDiagonalSet(M_VV, lumpVec_VV, INSERT_VALUES);

  // auto lumpVec_FF = createDMVector(dm_sub_FF);
  // CHKERR MatGetRowSum(M_FF, lumpVec_FF);

  // CHKERR MatZeroEntries(M_FF);
  // CHKERR MatDiagonalSet(M_FF, lumpVec_FF, INSERT_VALUES);

  boost::shared_ptr<FEMethod> nullFE;

  auto prb_mng = mField.getInterface<ProblemsManager>();
  std::vector<std::string> field_v{"V"};          
  std::vector<std::string> field_f{"F"};
  // cerr << "CREATE 2!\n";
  CHKERR prb_mng->buildSubProblem("SUB_VV", field_v, field_v,
                                  simple->getProblemName(), PETSC_TRUE);

  // partition problem
  CHKERR prb_mng->partitionFiniteElements("SUB_VV", true, 0,
                                          mField.get_comm_size());

  // set ghost nodes
  CHKERR prb_mng->partitionGhostDofsOnDistributedMesh("SUB_VV");
  // cerr << "CREATE 3!\n";

  CHKERR prb_mng->buildSubProblem("SUB_FF", field_f, field_f,
                                  simple->getProblemName(), PETSC_TRUE);

  // partition problem
  CHKERR prb_mng->partitionFiniteElements("SUB_FF", true, 0,
                                          mField.get_comm_size());
  CHKERR prb_mng->partitionGhostDofsOnDistributedMesh("SUB_FF");                                        
  // set ghost nodes

  CHKERR DMCreateGlobalVector_MoFEM(dm_sub_VV, &nested_vectors(0));
  CHKERR DMCreateGlobalVector_MoFEM(dm_sub_FF, &nested_vectors(1));

  auto ksp_VV = pipeline_mng->createKSP(dm_sub_VV);
  CHKERR DMMoFEMSetSquareProblem(dm_sub_VV, PETSC_TRUE);
  CHKERR KSPSetFromOptions(ksp_VV);
  CHKERR KSPSetOperators(ksp_VV, M_VV, M_VV);
  CHKERR DMMoFEMKSPSetComputeOperators(dm_sub_VV, simple->getDomainFEName(),
                                       vol_mass_ele_VV, nullFE, nullFE);
  CHKERR KSPSetUp(ksp_VV);

  auto ksp_FF = pipeline_mng->createKSP(dm_sub_FF);
  CHKERR DMMoFEMSetSquareProblem(dm_sub_FF, PETSC_TRUE);
  CHKERR KSPSetFromOptions(ksp_FF);
  CHKERR KSPSetOperators(ksp_FF, M_FF, M_FF);

  CHKERR DMMoFEMKSPSetComputeOperators(dm_sub_FF, simple->getDomainFEName(),
                                       vol_mass_ele_FF, nullFE, nullFE);
  
  CHKERR KSPSetUp(ksp_FF);

auto solve_boundary_for_g = [&]() {
  MoFEMFunctionBegin;
  if (*(pipeline_mng->getBoundaryExplicitRhsFE()->vecAssembleSwitch)) {

    CHKERR VecGhostUpdateBegin(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecAssemblyBegin(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F);
    CHKERR VecAssemblyEnd(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F);

    VecScatter scctx;
    CHKERR mField.getInterface<VecManager>()->vecScatterCreate(
        pipeline_mng->getBoundaryExplicitRhsFE()->ts_F,
        simple->getProblemName(), ROW, nested_vectors(0), "SUB_VV", ROW,
        &scctx);

    CHKERR VecScatterBegin(scctx, pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, nested_vectors(0), INSERT_VALUES,
                           SCATTER_FORWARD);
    CHKERR VecScatterEnd(scctx, pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, nested_vectors(0), INSERT_VALUES,
                         SCATTER_FORWARD);
    

    // CHKERR VecGhostUpdateBegin(nested_vectors(0), INSERT_VALUES,
    //                            SCATTER_FORWARD);
    // CHKERR VecGhostUpdateEnd(nested_vectors(0), INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR VecAssemblyBegin(nested_vectors(0));
    // CHKERR VecAssemblyEnd(nested_vectors(0));
    
    *(pipeline_mng->getBoundaryExplicitRhsFE()->vecAssembleSwitch) =
        false;

    auto D_VV = vectorDuplicate(nested_vectors(0));
    
    CHKERR KSPSolve(ksp_VV, nested_vectors(0), D_VV);
    CHKERR VecGhostUpdateBegin(D_VV, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D_VV, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecAssemblyBegin(D_VV);
    CHKERR VecAssemblyEnd(D_VV);
    CHKERR VecCopy(D_VV, nested_vectors(0));
    
    
    //BEGIN
  
  VecScatter scctx_2;
    CHKERR mField.getInterface<VecManager>()->vecScatterCreate(
        pipeline_mng->getBoundaryExplicitRhsFE()->ts_F,
        simple->getProblemName(), ROW, nested_vectors(1), "SUB_FF", ROW,
        &scctx_2);

    CHKERR VecScatterBegin(scctx_2, pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, nested_vectors(1), INSERT_VALUES,
                           SCATTER_FORWARD);
    CHKERR VecScatterEnd(scctx_2, pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, nested_vectors(1), INSERT_VALUES,
                         SCATTER_FORWARD);
    

    // CHKERR VecGhostUpdateBegin(nested_vectors(1), INSERT_VALUES,
    //                            SCATTER_FORWARD);
    // CHKERR VecGhostUpdateEnd(nested_vectors(1), INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR VecAssemblyBegin(nested_vectors(1));
    // CHKERR VecAssemblyEnd(nested_vectors(1));
    // cerr << "before (1)\n";
    // CHKERR VecView(nested_vectors(1),PETSC_VIEWER_STDOUT_WORLD);

    auto D_FF = vectorDuplicate(nested_vectors(1));

    CHKERR KSPSolve(ksp_FF, nested_vectors(1), D_FF);
    CHKERR VecGhostUpdateBegin(D_FF, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D_FF, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecAssemblyBegin(D_FF);
    CHKERR VecAssemblyEnd(D_FF);
    CHKERR VecCopy(D_FF, nested_vectors(1));
    
    // CHKERR VecScatterBegin(scctx, pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, nested_vectors(0), INSERT_VALUES,
    //                        SCATTER_REVERSE);
    // CHKERR VecScatterEnd(scctx, pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, nested_vectors(0), INSERT_VALUES,
    //                      SCATTER_REVERSE);
    
    // CHKERR VecScatterBegin(scctx, pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, nested_vectors(1), INSERT_VALUES,
    //                        SCATTER_REVERSE);
    // CHKERR VecScatterEnd(scctx, pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, nested_vectors(1), INSERT_VALUES,
    //                      SCATTER_REVERSE);
    

    // VecScatter scctx_3;
    // CHKERR mField.getInterface<VecManager>()->vecScatterCreate(
    //     nested_vectors(0), "SUB_VV", ROW,
    //     pipeline_mng->getBoundaryExplicitRhsFE()->ts_F,
    //     simple->getProblemName(), ROW, &scctx_3);

    // CHKERR VecScatterBegin(scctx, nested_vectors(0), pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, INSERT_VALUES,
    //                        SCATTER_REVERSE);
    // CHKERR VecScatterEnd(scctx, nested_vectors(0), pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, INSERT_VALUES,
    //                      SCATTER_REVERSE);

    // VecScatter scctx_4;
    // CHKERR mField.getInterface<VecManager>()->vecScatterCreate(
    //     nested_vectors(1), "SUB_FF", ROW,
    //     pipeline_mng->getBoundaryExplicitRhsFE()->ts_F,
    //     simple->getProblemName(), ROW, &scctx_4);

    // CHKERR VecScatterBegin(scctx_2, nested_vectors(1), pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, INSERT_VALUES,
    //                        SCATTER_REVERSE);
    // CHKERR VecScatterEnd(scctx_2, nested_vectors(1), pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, INSERT_VALUES,
    //                      SCATTER_REVERSE);                         


    CHKERR VecZeroEntries(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F);
    // Vec help_vec;
    VecScatter scctx_3;
    CHKERR mField.getInterface<VecManager>()->vecScatterCreate(
         nested_vectors(0), "SUB_VV", ROW, pipeline_mng->getBoundaryExplicitRhsFE()->ts_F,
        simple->getProblemName(), ROW,
        &scctx_3);

    CHKERR VecScatterBegin(scctx_3,
                           nested_vectors(0), pipeline_mng->getBoundaryExplicitRhsFE()->ts_F,
                           INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecScatterEnd(scctx_3,
                         nested_vectors(0), pipeline_mng->getBoundaryExplicitRhsFE()->ts_F,
                          INSERT_VALUES, SCATTER_FORWARD);

    VecScatter scctx_4;
    CHKERR mField.getInterface<VecManager>()->vecScatterCreate(
         nested_vectors(1), "SUB_FF", ROW, pipeline_mng->getBoundaryExplicitRhsFE()->ts_F,
        simple->getProblemName(), ROW,
        &scctx_4);

    CHKERR VecScatterBegin(scctx_4,
                           nested_vectors(1), pipeline_mng->getBoundaryExplicitRhsFE()->ts_F,
                           INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecScatterEnd(scctx_4,
                         nested_vectors(1), pipeline_mng->getBoundaryExplicitRhsFE()->ts_F,
                          INSERT_VALUES, SCATTER_FORWARD);

    // CHKERR VecCopy(help_vec, pipeline_mng->getBoundaryExplicitRhsFE()->ts_F);

    // CHKERR VecGhostUpdateBegin(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR VecGhostUpdateEnd(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F, INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR VecAssemblyBegin(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F);
    // CHKERR VecAssemblyEnd(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F);

    // cerr << "ts_F\n";
    // CHKERR VecView(pipeline_mng->getBoundaryExplicitRhsFE()->ts_F,PETSC_VIEWER_STDOUT_WORLD);
    // cerr << "(0)\n";
    // CHKERR VecView(nested_vectors(0),PETSC_VIEWER_STDOUT_WORLD);
    // cerr << "after (1)\n";
    // CHKERR VecView(nested_vectors(1),PETSC_VIEWER_STDOUT_WORLD);
    // cerr << "end\n";
    CHKERR VecScatterDestroy(&scctx);
    CHKERR VecScatterDestroy(&scctx_2);
    CHKERR VecScatterDestroy(&scctx_3);
    CHKERR VecScatterDestroy(&scctx_4);

  }

  MoFEMFunctionReturn(0);
};

pipeline_mng->getBoundaryExplicitRhsFE()->postProcessHook =
    solve_boundary_for_g;

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
    velocity_field_ptr = boost::make_shared<MatrixDouble>();
    geometry_field_ptr = boost::make_shared<MatrixDouble>();
    spatial_position_field_ptr = boost::make_shared<MatrixDouble>();
    fe_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("V", velocity_field_ptr));
    fe_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("GEOMETRY",
                                                    geometry_field_ptr));
    fe_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>(
            "x_2", spatial_position_field_ptr));
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
             {"x", boundary_x_ptr},
             {"U", boundary_u_ptr}},

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
    ptr->fsRawPtr = this;

    CHKERR TSSetUp(ts);
    CHKERR TSSolve(ts, NULL);
    CHKERR TSGetTime(ts, &ftime);

    CHKERR VecDestroy(&nested_vectors(0));
    CHKERR VecDestroy(&nested_vectors(1));
      
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