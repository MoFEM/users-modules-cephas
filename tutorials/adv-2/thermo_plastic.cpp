/**
 * \file thermo_plastic.cpp
 * \example thermo_plastic.cpp
 *
 * Thermo plasticity
 *
 */



#ifndef EXECUTABLE_DIMENSION
#define EXECUTABLE_DIMENSION 3
#endif

#include <MoFEM.hpp>
#include <MatrixFunction.hpp>
#include <IntegrationRules.hpp>

using namespace MoFEM;

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = PipelineManager::FaceEle;
  using BoundaryEle = PipelineManager::EdgeEle;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = PipelineManager::VolEle;
  using BoundaryEle = PipelineManager::FaceEle;
};

constexpr int SPACE_DIM =
    EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh

constexpr EntityType boundary_ent = SPACE_DIM == 3 ? MBTRI : MBEDGE;
using EntData = EntitiesFieldData::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

using AssemblyDomainEleOp =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::OpBase;

//! [Only used with Hooke equation (linear material model)]
using OpKCauchy = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM, 0>;
using OpInternalForceCauchy = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesSymTensor<1, SPACE_DIM, SPACE_DIM>;
//! [Only used with Hooke equation (linear material model)]

//! [Only used for dynamics]
using OpMass = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, SPACE_DIM>;
using OpInertiaForce = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 1>;
//! [Only used for dynamics]

//! [Only used with Hencky/nonlinear material]
using OpKPiola = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradTensorGrad<1, SPACE_DIM, SPACE_DIM, 1>;
using OpInternalForcePiola = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;
//! [Only used with Hencky/nonlinear material]

//! [Essential boundary conditions]
using OpBoundaryMass = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, SPACE_DIM>;
using OpBoundaryVec = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 0>;
using OpBoundaryInternal = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 1>;
//! [Essential boundary conditions]
using OpScaleL2 = MoFEM::OpScaleBaseBySpaceInverseOfMeasure<DomainEleOp>;

// Thermal operators
/**
 * @brief Integrate Lhs base of flux (1/k) base of flux (FLUX x FLUX)
 *
 */
using OpHdivHdiv = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<3, 3>;

/**
 * @brief Integrate Lhs div of base of flux time base of temperature (FLUX x T)
 * and transpose of it, i.e. (T x FLAX)
 *
 */
using OpHdivT = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMixDivTimesScalar<SPACE_DIM>;

/**
 * @brief Integrate Lhs base of temerature times (heat capacity) times base of
 * temperature (T x T)
 *
 */
using OpCapacity = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, 1>;

/**
 * @brief Integrating Rhs flux base (1/k) flux  (FLUX)
 */
using OpHdivFlux = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesVector<3, 3, 1>;

/**
 * @brief  Integrate Rhs div flux base times temperature (T)
 *
 */
using OpHDivTemp = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpMixDivTimesU<3, 1, 2>;

/**
 * @brief Integrate Rhs base of temperature time heat capacity times heat rate
 * (T)
 *
 */
using OpBaseDotT = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpBaseTimesScalar<1>;

/**
 * @brief Integrate Rhs base of temperature times divergence of flux (T)
 *
 */
using OpBaseDivFlux = OpBaseDotT;

using DomainNaturalBC =
    NaturalBC<DomainEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
using OpBodyForce = DomainNaturalBC::OpFlux<BLOCKSET, 1, SPACE_DIM>;
using OpHeatSource = DomainNaturalBC::OpFlux<BLOCKSET, 1, 1>;

using BoundaryNaturalBC =
    NaturalBC<BoundaryEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
using OpForce = BoundaryNaturalBC::OpFlux<BLOCKSET, 1, SPACE_DIM>;
using OpTemperatureBC =
    BoundaryNaturalBC::OpFlux<BLOCKSET, SPACE_DIM, SPACE_DIM>;

double scale = 1.;

double young_modulus = 206913;
double poisson_ratio = 0.29;
double coeff_expansion = 10e-6;
double ref_temp = 0.0;
double rho = 0;
double sigmaY = 450;
double H = 129;
double visH = 0;
double cn = 1;
double Qinf = 0; // 265;
double b_iso = 16.93;
double heat_conductivity =
    16.2; // Force / (time temerature )  or Power /
          // (length temperature) // Time unit is hour. force unit kN
double heat_capacity =
    5961.6; // length^2/(time^2 temerature) // length is millimeter time is hour
double omega_0 = 2e-3;
double omega_h = 2e-3;
double omega_inf = 0;
double fraction_of_dissipation = 0.9;
int order = 2;

int number_of_cycles_in_total_time = 6;
double amplitude_cycle = 0.5;
double amplitude_shift = 0.5;
double phase_shift = 0.8;

inline long double hardening(long double tau, double temp) {
  return H * tau * (1 - omega_h * temp) +
         Qinf * (1. - std::exp(-b_iso * tau)) * (1 - omega_inf * temp) +
         sigmaY * (1 - omega_0 * temp);
}

inline long double hardening_dtau(long double tau, double temp) {
  return H * (1 - omega_h * temp) +
         Qinf * b_iso * std::exp(-b_iso * tau) * (1 - omega_inf * temp);
}

inline long double hardening_dtemp(long double tau, double temp) {
  return -H * tau * omega_h - Qinf * (1. - std::exp(-b_iso * tau)) * omega_inf -
         sigmaY * omega_0;
}

#include <HenckyOps.hpp>
#include <PlasticOps.hpp>
using namespace PlasticOps;
#include <PlasticThermalOps.hpp>

using namespace HenckyOps;
struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  MoFEMErrorCode setupProblem();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode bC();
  MoFEMErrorCode OPs();
  MoFEMErrorCode tsSolve();

  boost::shared_ptr<PlasticThermalOps::CommonData> commonPlasticDataPtr;
  boost::shared_ptr<HenckyOps::CommonData> commonHenckyDataPtr;
  boost::shared_ptr<PostProcEle> postProcFe;
  boost::shared_ptr<DomainEle> reactionFe;
  boost::shared_ptr<DomainEle> feAxiatorRhs;
  boost::shared_ptr<DomainEle> feAxiatorLhs;
  boost::shared_ptr<DomainEle> feThermalRhs;
  boost::shared_ptr<DomainEle> feThermalLhs;

  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;

  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
  boost::shared_ptr<std::vector<unsigned char>> reactionMarker;

  struct BcTempFun {
    BcTempFun(double v, FEMethod &fe) : valTemp(v), fE(fe) {}
    double operator()(const double, const double, const double) {
      return -valTemp; // * fE.ts_t;
    }

  private:
    double valTemp;
    FEMethod &fE;
  };
  std::vector<BcTempFun> bcTemperatureFunctions;
};

//! [Run problem]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR setupProblem();
  CHKERR createCommonData();
  CHKERR bC();
  CHKERR OPs();
  CHKERR tsSolve();
  MoFEMFunctionReturn(0);
}
//! [Run problem]

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  // Add field
  constexpr FieldApproximationBase base = AINSWORTH_LEGENDRE_BASE;
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  // Mechanical fields
  CHKERR simple->addDomainField("U", H1, base, SPACE_DIM);
  CHKERR simple->addDomainField("TAU", L2, base, 1);
  CHKERR simple->addDomainField("EP", L2, base, size_symm);
  CHKERR simple->addBoundaryField("U", H1, base, SPACE_DIM);
  // Temperature
  const auto flux_space = (SPACE_DIM == 2) ? HCURL : HDIV;
  CHKERR simple->addDomainField("T", L2, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple->addDomainField("FLUX", flux_space, DEMKOWICZ_JACOBI_BASE, 1);
  CHKERR simple->addBoundaryField("FLUX", flux_space, DEMKOWICZ_JACOBI_BASE, 1);

  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setFieldOrder("TAU", order - 1);
  CHKERR simple->setFieldOrder("EP", order - 1);
  CHKERR simple->setFieldOrder("FLUX", order);
  CHKERR simple->setFieldOrder("T", order - 1);
  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;

  auto get_command_line_parameters = [&]() {
    MoFEMFunctionBegin;
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-scale", &scale, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-rho", &rho, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-young_modulus",
                                 &young_modulus, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-poisson_ratio",
                                 &poisson_ratio, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-coeff_expansion",
                                 &coeff_expansion, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-ref_temp", &ref_temp,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-hardening", &H, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-hardening_viscous", &visH,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-yield_stress", &sigmaY,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-cn", &cn, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-Qinf", &Qinf, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-b_iso", &b_iso, PETSC_NULL);

    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-capacity", &heat_capacity,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-conductivity",
                                 &heat_conductivity, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-omega_0", &omega_0,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-omega_h", &omega_h,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-omega_inf", &omega_inf,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-fraction_of_dissipation",
                                 &fraction_of_dissipation, PETSC_NULL);
    CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-number_of_cycles_in_total_time",
                              &number_of_cycles_in_total_time, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-amplitude_cycle",
                                 &amplitude_cycle, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-phase_shift", &phase_shift,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-amplitude_shift",
                                 &amplitude_shift, PETSC_NULL);
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Young modulus " << young_modulus;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Poisson ratio " << poisson_ratio;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Coeff_expansion " << coeff_expansion;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Reference_temperature  " << ref_temp;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Yield stress " << sigmaY;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Hardening " << H;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Viscous hardening " << visH;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Saturation yield stress " << Qinf;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Saturation exponent " << b_iso;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "cn " << cn;
    MOFEM_LOG("EXAMPLE", Sev::inform)
        << "fraction_of_dissipation " << fraction_of_dissipation;

    PetscBool is_scale = PETSC_TRUE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-is_scale", &is_scale,
                               PETSC_NULL);
    if (is_scale) {
      scale = scale / young_modulus;
      young_modulus *= scale;
      rho *= scale;
      sigmaY *= scale;
      H *= scale;
      Qinf *= scale;
      visH *= scale;
      heat_capacity *= scale;

      MOFEM_LOG("EXAMPLE", Sev::inform)
          << "Scaled Young modulus " << young_modulus;
      MOFEM_LOG("EXAMPLE", Sev::inform)
          << "Scaled Poisson ratio " << poisson_ratio;
      MOFEM_LOG("EXAMPLE", Sev::inform) << "Scaled Yield stress " << sigmaY;
      MOFEM_LOG("EXAMPLE", Sev::inform) << "Scaled Hardening " << H;
      MOFEM_LOG("EXAMPLE", Sev::inform) << "Scaled Viscous hardening " << visH;
      MOFEM_LOG("EXAMPLE", Sev::inform)
          << "Scaled Saturation yield stress " << Qinf;
      MOFEM_LOG("EXAMPLE", Sev::inform)
          << "Scaled heat capacity " << heat_capacity;
    }

    MoFEMFunctionReturn(0);
  };

  auto set_matrial_stiffness = [&]() {
    MoFEMFunctionBegin;
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    FTensor::Index<'l', SPACE_DIM> l;
    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
    const double bulk_modulus_K = young_modulus / (3 * (1 - 2 * poisson_ratio));
    const double shear_modulus_G = young_modulus / (2 * (1 + poisson_ratio));

    // Plane stress or when 1, plane strain or 3d
    const double A = (SPACE_DIM == 2)
                         ? 2 * shear_modulus_G /
                               (bulk_modulus_K + (4. / 3.) * shear_modulus_G)
                         : 1;

    auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(
        *commonPlasticDataPtr->mDPtr);
    auto t_D_axiator = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(
        *commonPlasticDataPtr->mDPtr_Axiator);
    auto t_D_deviator = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(
        *commonPlasticDataPtr->mDPtr_Deviator);

    constexpr double third = boost::math::constants::third<double>();
    t_D_axiator(i, j, k, l) = A *
                              (bulk_modulus_K - (2. / 3.) * shear_modulus_G) *
                              t_kd(i, j) * t_kd(k, l);
    t_D_deviator(i, j, k, l) =
        2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.);
    t_D(i, j, k, l) = t_D_axiator(i, j, k, l) + t_D_deviator(i, j, k, l);

    MoFEMFunctionReturn(0);
  };

  commonPlasticDataPtr = boost::make_shared<PlasticThermalOps::CommonData>();
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  commonPlasticDataPtr->mDPtr = boost::make_shared<MatrixDouble>();
  commonPlasticDataPtr->mDPtr->resize(size_symm * size_symm, 1);
  commonPlasticDataPtr->mDPtr_Axiator = boost::make_shared<MatrixDouble>();
  commonPlasticDataPtr->mDPtr_Axiator->resize(size_symm * size_symm, 1);
  commonPlasticDataPtr->mDPtr_Deviator = boost::make_shared<MatrixDouble>();
  commonPlasticDataPtr->mDPtr_Deviator->resize(size_symm * size_symm, 1);

  commonPlasticDataPtr->mGradPtr = boost::make_shared<MatrixDouble>();
  commonPlasticDataPtr->mStrainPtr = boost::make_shared<MatrixDouble>();
  commonPlasticDataPtr->mStressPtr = boost::make_shared<MatrixDouble>();

  CHKERR get_command_line_parameters();
  CHKERR set_matrial_stiffness();

  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Boundary condition]
MoFEMErrorCode Example::bC() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();
  auto prb_mng = mField.getInterface<ProblemsManager>();

  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_X",
                                           "U", 0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_Y",
                                           "U", 1, 1);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_Z",
                                           "U", 2, 2);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                           "REMOVE_ALL", "U", 0, 3);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                           "ZERO_FLUX", "FLUX", 0, 1);

  PetscBool zero_fix_skin_flux = PETSC_TRUE;
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-fix_skin_flux",
                             &zero_fix_skin_flux, PETSC_NULL);
  if (zero_fix_skin_flux) {
    Range faces;
    CHKERR mField.get_moab().get_entities_by_dimension(0, SPACE_DIM, faces);
    Skinner skin(&mField.get_moab());
    Range skin_ents;
    CHKERR skin.find_skin(0, faces, false, skin_ents);
    ParallelComm *pcomm =
        ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
    if (pcomm == NULL)
      SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
              "Communicator not created");
    CHKERR pcomm->filter_pstatus(skin_ents,
                                 PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                 PSTATUS_NOT, -1, &skin_ents);
    CHKERR prb_mng->removeDofsOnEntities(simple->getProblemName(), "FLUX",
                                         skin_ents, 0, 1);
  }

  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "FIX_X", "U",
                                        0, 0);
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "FIX_Y", "U",
                                        1, 1);
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "FIX_Z", "U",
                                        2, 2);
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "FIX_ALL",
                                        "U", 0, 3);

  auto &bc_map = bc_mng->getBcMapByBlockName();
  boundaryMarker = bc_mng->getMergedBlocksMarker(vector<string>{"FIX_"});

  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "REACTION",
                                        "U", 0, 3);
  for (auto b : bc_map) {
    MOFEM_LOG("EXAMPLE", Sev::verbose) << "Marker " << b.first;
  }

  // OK. We have problem with GMesh, it adding empty characters at the end of
  // block. So first block is search by regexp. popMarkDOFsOnEntities should
  // work with regexp.
  std::string reaction_block_set;
  for (auto bc : bc_map) {
    if (bc_mng->checkBlock(bc, "REACTION")) {
      reaction_block_set = bc.first;
      break;
    }
  }

  if (auto bc = bc_mng->popMarkDOFsOnEntities(reaction_block_set)) {
    reactionMarker = bc->getBcMarkersPtr();

    // Only take reaction from nodes
    Range nodes;
    CHKERR mField.get_moab().get_entities_by_type(0, MBVERTEX, nodes, true);
    CHKERR prb_mng->markDofs(simple->getProblemName(), ROW,
                             ProblemsManager::MarkOP::AND, nodes,
                             *reactionMarker);

  } else {
    MOFEM_LOG("EXAMPLE", Sev::warning) << "REACTION blockset does not exist";
  }

  if (!reactionMarker) {
    MOFEM_LOG("EXAMPLE", Sev::warning) << "REACTION blockset does not exist";
  }

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::OPs() {
  MoFEMFunctionBegin;
  auto pipeline_mng = mField.getInterface<PipelineManager>();
  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();

  auto integration_rule_deviator = [](int o_row, int o_col, int approx_order) {
    return 2 * (approx_order - 1);
  };
  auto integration_rule_bc = [](int, int, int approx_order) {
    return 2 * approx_order;
  };

  feAxiatorRhs = boost::make_shared<DomainEle>(mField);
  feAxiatorLhs = boost::make_shared<DomainEle>(mField);
  auto integration_rule_axiator = [](int, int, int approx_order) {
    return 2 * (approx_order - 1);
  };
  feAxiatorRhs->getRuleHook = integration_rule_axiator;
  feAxiatorLhs->getRuleHook = integration_rule_axiator;

  feThermalRhs = boost::make_shared<DomainEle>(mField);
  feThermalLhs = boost::make_shared<DomainEle>(mField);
  auto integration_rule_thermal = [](int, int, int approx_order) {
    return 2 * approx_order;
  };
  feThermalRhs->getRuleHook = integration_rule_thermal;
  feThermalLhs->getRuleHook = integration_rule_thermal;

  auto add_domain_base_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(new OpCalculateHOJac<SPACE_DIM>(jac_ptr));
    pipeline.push_back(
        new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline.push_back(
        new OpSetHOInvJacToScalarBases<SPACE_DIM>(H1, inv_jac_ptr));

    if (SPACE_DIM == 2) {
      pipeline.push_back(new OpMakeHdivFromHcurl());
      pipeline.push_back(new OpSetContravariantPiolaTransformOnFace2D(jac_ptr));
      pipeline.push_back(new OpSetInvJacHcurlFace(inv_jac_ptr));
      pipeline.push_back(new OpSetInvJacL2ForFace(inv_jac_ptr));
    }

    pipeline.push_back(new OpCalculateScalarFieldValuesDot(
        "TAU", commonPlasticDataPtr->getPlasticTauDotPtr()));
    pipeline.push_back(new OpCalculateTensor2SymmetricFieldValues<SPACE_DIM>(
        "EP", commonPlasticDataPtr->getPlasticStrainPtr()));
    pipeline.push_back(new OpCalculateTensor2SymmetricFieldValuesDot<SPACE_DIM>(
        "EP", commonPlasticDataPtr->getPlasticStrainDotPtr()));
    pipeline.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        "U", commonPlasticDataPtr->mGradPtr));
    pipeline.push_back(new OpCalculateScalarFieldValues(
        "TAU", commonPlasticDataPtr->getPlasticTauPtr()));

    pipeline.push_back(new OpCalculateScalarFieldValues(
        "T", commonPlasticDataPtr->getTempValPtr()));
    pipeline.push_back(new OpCalculateScalarFieldValuesDot(
        "T", commonPlasticDataPtr->getTempValDotPtr()));
    pipeline.push_back(new OpCalculateHdivVectorDivergence<3, SPACE_DIM>(
        "FLUX", commonPlasticDataPtr->getTempDivFluxPtr()));
    pipeline.push_back(new OpCalculateHVecVectorField<3>(
        "FLUX", commonPlasticDataPtr->getTempFluxValPtr()));

    MoFEMFunctionReturn(0);
  };

  auto add_domain_stress_ops = [&](auto &pipeline, auto m_D_ptr) {
    MoFEMFunctionBegin;

    pipeline.push_back(new OpSymmetrizeTensor<SPACE_DIM>(
        "U", commonPlasticDataPtr->mGradPtr, commonPlasticDataPtr->mStrainPtr));
    pipeline.push_back(new PlasticThermalOps::OpPlasticStressThermal(
        "U", commonPlasticDataPtr, m_D_ptr, 1));

    if (m_D_ptr != commonPlasticDataPtr->mDPtr_Axiator)
      pipeline.push_back(
          new OpCalculatePlasticSurface("U", commonPlasticDataPtr));

    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_lhs_mechanical = [&](auto &pipeline, auto m_D_ptr) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    pipeline.push_back(new OpKCauchy("U", "U", m_D_ptr));
    pipeline.push_back(new PlasticThermalOps::OpKCauchyThermoElasticity(
        "U", "T", commonPlasticDataPtr, m_D_ptr));
    pipeline.push_back(new OpCalculatePlasticInternalForceLhs_dEP(
        "U", "EP", commonPlasticDataPtr, m_D_ptr));

    pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_rhs_mechanical = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    CHKERR DomainNaturalBC::addFluxToPipeline(FluxOpType<OpBodyForce>(),
                                              pipeline, mField, "U",
                                              "BODY_FORCE", Sev::inform);
    CHKERR DomainNaturalBC::addScalingMethod(
        FluxOpType<OpBodyForce>(), pipeline, boost::make_shared<TimeScale>(),
        Sev::inform);

    // Calculate internal forces
    pipeline.push_back(
        new OpInternalForceCauchy("U", commonPlasticDataPtr->mStressPtr));

    pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_lhs_constrain = [&](auto &pipeline, auto m_D_ptr) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    pipeline.push_back(new OpCalculatePlasticFlowLhs_dU(
        "EP", "U", commonPlasticDataPtr, m_D_ptr));
    pipeline.push_back(new OpCalculateContrainsLhs_dU(
        "TAU", "U", commonPlasticDataPtr, m_D_ptr));

    pipeline.push_back(new OpCalculatePlasticFlowLhs_dEP(
        "EP", "EP", commonPlasticDataPtr, m_D_ptr));
    pipeline.push_back(
        new OpCalculatePlasticFlowLhs_dTAU("EP", "TAU", commonPlasticDataPtr));
    pipeline.push_back(new OpCalculateContrainsLhs_dEP(
        "TAU", "EP", commonPlasticDataPtr, m_D_ptr));
    pipeline.push_back(
        new OpCalculateContrainsLhs_dTAU("TAU", "TAU", commonPlasticDataPtr));

    pipeline.push_back(new PlasticThermalOps::OpCalculateContrainsLhs_dT(
        "TAU", "T", commonPlasticDataPtr));
    pipeline.push_back(new PlasticThermalOps::OpPlasticHeatProduction_dEP(
        "T", "EP", commonPlasticDataPtr));

    pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_rhs_constrain = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    pipeline.push_back(
        new OpCalculatePlasticFlowRhs("EP", commonPlasticDataPtr));
    pipeline.push_back(
        new OpCalculateContrainsRhs("TAU", commonPlasticDataPtr));

    pipeline.push_back(new PlasticThermalOps::OpPlasticHeatProduction(
        "T", commonPlasticDataPtr));

    MoFEMFunctionReturn(0);
  };

  auto add_boundary_ops_lhs_mechanical = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    auto &bc_map = mField.getInterface<BcManager>()->getBcMapByBlockName();
    for (auto bc : bc_map) {
      if (bc_mng->checkBlock(bc, "FIX_")) {
        pipeline.push_back(
            new OpSetBc("U", false, bc.second->getBcMarkersPtr()));
        pipeline.push_back(new OpBoundaryMass(
            "U", "U", [](double, double, double) { return 1.; },
            bc.second->getBcEntsPtr()));
        pipeline.push_back(new OpUnSetBc("U"));
      }
    }
    MoFEMFunctionReturn(0);
  };

  auto add_boundary_ops_rhs_mechanical = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    auto get_minus_time = [&](double, double, double) {
      auto *pipeline_mng = mField.getInterface<PipelineManager>();
      auto &fe_domain_rhs = pipeline_mng->getBoundaryRhsFE();
      return -fe_domain_rhs->ts_t;
    };

    auto time_scaled = [&](double, double, double) {
      auto *pipeline_mng = mField.getInterface<PipelineManager>();
      auto &fe_domain_rhs = pipeline_mng->getBoundaryRhsFE();
      if (number_of_cycles_in_total_time != 0) {
        return amplitude_cycle * sin((2 * fe_domain_rhs->ts_t - phase_shift) *
                                     number_of_cycles_in_total_time * M_PI) +
               amplitude_shift;
      } else {
        return fe_domain_rhs->ts_t;
      }
    };

    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    CHKERR BoundaryNaturalBC::addFluxToPipeline(FluxOpType<OpForce>(), pipeline,
                                                mField, "FLUX", "FORCE");
    CHKERR BoundaryNaturalBC::addScalingMethod(FluxOpType<OpForce>(), pipeline,
                                               boost::make_shared<TimeScale>(),
                                               Sev::inform);

    pipeline.push_back(new OpUnSetBc("U"));

    auto u_mat_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_mat_ptr));

    for (auto &bc : mField.getInterface<BcManager>()->getBcMapByBlockName()) {
      if (bc_mng->checkBlock(bc, "FIX_")) {
        pipeline.push_back(
            new OpSetBc("U", false, bc.second->getBcMarkersPtr()));
        auto attr_vec = boost::make_shared<MatrixDouble>(SPACE_DIM, 1);
        attr_vec->clear();
        if (bc.second->bcAttributes.size() < SPACE_DIM)
          SETERRQ1(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                   "Wrong size of boundary attributes vector. Set right block "
                   "size attributes. Size of attributes %d",
                   bc.second->bcAttributes.size());
        std::copy(&bc.second->bcAttributes[0],
                  &bc.second->bcAttributes[SPACE_DIM],
                  attr_vec->data().begin());

        pipeline.push_back(new OpBoundaryVec("U", attr_vec, time_scaled,
                                             bc.second->getBcEntsPtr()));
        pipeline.push_back(new OpBoundaryInternal(
            "U", u_mat_ptr, [](double, double, double) { return 1.; },
            bc.second->getBcEntsPtr()));

        pipeline.push_back(new OpUnSetBc("U"));
      }
    }

    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_lhs_thermal = [&](auto &pipeline, auto &fe) {
    MoFEMFunctionBegin;
    auto resistance = [](const double, const double, const double) {
      return (1. / heat_conductivity);
    };
    auto capacity = [&](const double, const double, const double) {
      return -heat_capacity * fe.ts_a;
    };
    auto unity = []() { return 1; };
    pipeline.push_back(new OpHdivHdiv("FLUX", "FLUX", resistance));
    pipeline.push_back(new OpHdivT("FLUX", "T", unity, true));
    pipeline.push_back(new OpCapacity("T", "T", capacity));
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_rhs_thermal = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    auto resistance = [](const double, const double, const double) {
      return (1. / heat_conductivity);
    };
    auto capacity = [&](const double, const double, const double) {
      return -heat_capacity;
    };
    auto unity = [](const double, const double, const double) { return 1; };
    pipeline.push_back(new OpHdivFlux(
        "FLUX", commonPlasticDataPtr->getTempFluxValPtr(), resistance));
    pipeline.push_back(
        new OpHDivTemp("FLUX", commonPlasticDataPtr->getTempValPtr(), unity));
    pipeline.push_back(new OpBaseDivFlux(
        "T", commonPlasticDataPtr->getTempDivFluxPtr(), unity));
    pipeline.push_back(new OpBaseDotT(
        "T", commonPlasticDataPtr->getTempValDotPtr(), capacity));

    CHKERR DomainNaturalBC::addFluxToPipeline(FluxOpType<OpHeatSource>(),
                                              pipeline, mField, "U",
                                              "HEAT_SOURCE", Sev::inform);
    CHKERR DomainNaturalBC::addScalingMethod(
        FluxOpType<OpHeatSource>(), pipeline, boost::make_shared<TimeScale>(),
        Sev::inform);

    MoFEMFunctionReturn(0);
  };

  auto add_boundary_ops_rhs_thermal = [&](auto &pipeline, auto &fe) {
    MoFEMFunctionBegin;

    if (SPACE_DIM == 2) {
      pipeline.push_back(new OpSetContravariantPiolaTransformOnEdge2D());
    } else if (SPACE_DIM == 3) {
      pipeline.push_back(new OpHOSetCovariantPiolaTransformOnFace3D(HDIV));
    }

    CHKERR BoundaryNaturalBC::addFluxToPipeline(
        FluxOpType<OpTemperatureBC>(), pipeline, mField, "FLUX", "TEMPERATURE");
    CHKERR BoundaryNaturalBC::addScalingMethod(
        FluxOpType<OpTemperatureBC>(), pipeline,
        boost::make_shared<TimeScale>(), Sev::inform);

    MoFEMFunctionReturn(0);
  };

  // Mechanics
  CHKERR add_domain_base_ops(pipeline_mng->getOpDomainLhsPipeline());
  CHKERR add_domain_stress_ops(pipeline_mng->getOpDomainLhsPipeline(),
                               commonPlasticDataPtr->mDPtr_Deviator);
  CHKERR add_domain_ops_lhs_mechanical(pipeline_mng->getOpDomainLhsPipeline(),
                                       commonPlasticDataPtr->mDPtr_Deviator);
  CHKERR add_domain_ops_lhs_constrain(pipeline_mng->getOpDomainLhsPipeline(),
                                      commonPlasticDataPtr->mDPtr_Deviator);
  CHKERR add_boundary_ops_lhs_mechanical(
      pipeline_mng->getOpBoundaryLhsPipeline());

  // Axiator
  CHKERR add_domain_base_ops(feAxiatorLhs->getOpPtrVector());
  CHKERR add_domain_ops_lhs_mechanical(feAxiatorLhs->getOpPtrVector(),
                                       commonPlasticDataPtr->mDPtr_Axiator);
  // Temperature
  CHKERR add_domain_base_ops(feThermalLhs->getOpPtrVector());
  CHKERR add_domain_ops_lhs_thermal(feThermalLhs->getOpPtrVector(),
                                    *feThermalLhs);

  // Mechanics
  CHKERR add_domain_base_ops(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_domain_stress_ops(pipeline_mng->getOpDomainRhsPipeline(),
                               commonPlasticDataPtr->mDPtr_Deviator);
  CHKERR add_domain_ops_rhs_mechanical(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_domain_ops_rhs_constrain(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_boundary_ops_rhs_mechanical(
      pipeline_mng->getOpBoundaryRhsPipeline());

  // Axiator
  CHKERR add_domain_base_ops(feAxiatorRhs->getOpPtrVector());
  CHKERR add_domain_stress_ops(feAxiatorRhs->getOpPtrVector(),
                               commonPlasticDataPtr->mDPtr_Axiator);
  CHKERR add_domain_ops_rhs_mechanical(feAxiatorRhs->getOpPtrVector());

  // Temperature
  CHKERR add_domain_base_ops(feThermalRhs->getOpPtrVector());
  CHKERR add_domain_ops_rhs_thermal(feThermalRhs->getOpPtrVector());
  CHKERR
  add_boundary_ops_rhs_thermal(pipeline_mng->getOpBoundaryRhsPipeline(),
                               *pipeline_mng->getBoundaryRhsFE());

  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule_deviator);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule_deviator);

  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule_bc);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule_bc);

  auto create_reaction_pipeline = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    if (reactionMarker) {

      auto det_ptr = boost::make_shared<VectorDouble>();
      auto jac_ptr = boost::make_shared<MatrixDouble>();
      auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
      pipeline.push_back(new OpCalculateHOJac<SPACE_DIM>(jac_ptr));
      pipeline.push_back(
          new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
      pipeline.push_back(
          new OpSetHOInvJacToScalarBases<SPACE_DIM>(H1, inv_jac_ptr));

      pipeline.push_back(
          new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
              "U", commonPlasticDataPtr->mGradPtr));
      pipeline.push_back(new OpCalculateTensor2SymmetricFieldValues<SPACE_DIM>(
          "EP", commonPlasticDataPtr->getPlasticStrainPtr()));
      pipeline.push_back(
          new OpSymmetrizeTensor<SPACE_DIM>("U", commonPlasticDataPtr->mGradPtr,
                                            commonPlasticDataPtr->mStrainPtr));
      pipeline.push_back(new PlasticThermalOps::OpPlasticStressThermal(
          "U", commonPlasticDataPtr, commonPlasticDataPtr->mDPtr, scale));
      pipeline.push_back(new OpSetBc("U", false, reactionMarker));

      // Calculate internal forece
      pipeline.push_back(
          new OpInternalForceCauchy("U", commonPlasticDataPtr->mStressPtr));

      pipeline.push_back(new OpUnSetBc("U"));
    }

    MoFEMFunctionReturn(0);
  };

  reactionFe = boost::make_shared<DomainEle>(mField);
  reactionFe->getRuleHook = integration_rule_deviator;

  CHKERR create_reaction_pipeline(reactionFe->getOpPtrVector());

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::tsSolve() {
  MoFEMFunctionBegin;

  Simple *simple = mField.getInterface<Simple>();
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  ISManager *is_manager = mField.getInterface<ISManager>();

  auto set_section_monitor = [&](auto solver) {
    MoFEMFunctionBegin;
    SNES snes;
    CHKERR TSGetSNES(solver, &snes);
    PetscViewerAndFormat *vf;
    CHKERR PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,
                                      PETSC_VIEWER_DEFAULT, &vf);
    CHKERR SNESMonitorSet(
        snes,
        (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal, void *))SNESMonitorFields,
        vf, (MoFEMErrorCode(*)(void **))PetscViewerAndFormatDestroy);
    MoFEMFunctionReturn(0);
  };

  auto create_post_process_element = [&]() {
    MoFEMFunctionBegin;
    postProcFe = boost::make_shared<PostProcEle>(mField);

    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateHOJac<SPACE_DIM>(jac_ptr));
    postProcFe->getOpPtrVector().push_back(
        new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
    postProcFe->getOpPtrVector().push_back(
        new OpSetHOInvJacToScalarBases<SPACE_DIM>(H1, inv_jac_ptr));

    postProcFe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
            "U", commonPlasticDataPtr->mGradPtr));
    postProcFe->getOpPtrVector().push_back(new OpCalculateScalarFieldValues(
        "TAU", commonPlasticDataPtr->getPlasticTauPtr()));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateTensor2SymmetricFieldValues<SPACE_DIM>(
            "EP", commonPlasticDataPtr->getPlasticStrainPtr()));
    postProcFe->getOpPtrVector().push_back(new OpCalculateScalarFieldValues(
        "T", commonPlasticDataPtr->getTempValPtr()));

    postProcFe->getOpPtrVector().push_back(new OpSymmetrizeTensor<SPACE_DIM>(
        "U", commonPlasticDataPtr->mGradPtr, commonPlasticDataPtr->mStrainPtr));
    postProcFe->getOpPtrVector().push_back(
        new PlasticThermalOps::OpPlasticStressThermal(
            "U", commonPlasticDataPtr, commonPlasticDataPtr->mDPtr, scale));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculatePlasticSurface("U", commonPlasticDataPtr));

    auto u_ptr = boost::make_shared<MatrixDouble>();
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    postProcFe->getOpPtrVector().push_back(

        new OpPPMap(

            postProcFe->getPostProcMesh(), postProcFe->getMapGaussPts(),

            {{"PLASTIC_SURFACE", commonPlasticDataPtr->getPlasticSurfacePtr()},
             {"PLASTIC_MULTIPLIER", commonPlasticDataPtr->getPlasticTauPtr()},
             {"T", commonPlasticDataPtr->getTempValPtr()}},

            {{"U", u_ptr}},

            {},

            {{"STRAIN", commonPlasticDataPtr->mStrainPtr},
             {"STRESS", commonPlasticDataPtr->mStressPtr}}

            )

    );

    MoFEMFunctionReturn(0);
  };

  auto scatter_create = [&](auto D, auto coeff) {
    SmartPetscObj<IS> is;
    CHKERR is_manager->isCreateProblemFieldAndRank(simple->getProblemName(),
                                                   ROW, "U", coeff, coeff, is);
    int loc_size;
    CHKERR ISGetLocalSize(is, &loc_size);
    Vec v;
    CHKERR VecCreateMPI(mField.get_comm(), loc_size, PETSC_DETERMINE, &v);
    VecScatter scatter;
    CHKERR VecScatterCreate(D, is, v, PETSC_NULL, &scatter);
    return std::make_tuple(SmartPetscObj<Vec>(v),
                           SmartPetscObj<VecScatter>(scatter));
  };

  auto set_time_monitor = [&](auto dm, auto solver) {
    MoFEMFunctionBegin;
    boost::shared_ptr<Monitor> monitor_ptr(new Monitor(
        dm, postProcFe, reactionFe, uXScatter, uYScatter, uZScatter));
    boost::shared_ptr<ForcesAndSourcesCore> null;
    CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                               monitor_ptr, null, null);
    MoFEMFunctionReturn(0);
  };

  auto dm = simple->getDM();
  auto D = smartCreateDMVector(dm);

  boost::shared_ptr<FEMethod> null;
  CHKERR DMMoFEMTSSetIJacobian(dm, simple->getDomainFEName(), feAxiatorLhs,
                               null, null);
  CHKERR DMMoFEMTSSetIFunction(dm, simple->getDomainFEName(), feAxiatorRhs,
                               null, null);
  CHKERR DMMoFEMTSSetIJacobian(dm, simple->getDomainFEName(), feThermalLhs,
                               null, null);
  CHKERR DMMoFEMTSSetIFunction(dm, simple->getDomainFEName(), feThermalRhs,
                               null, null);

  CHKERR create_post_process_element();
  uXScatter = scatter_create(D, 0);
  uYScatter = scatter_create(D, 1);
  if (SPACE_DIM == 3)
    uZScatter = scatter_create(D, 2);

  auto solver = pipeline_mng->createTSIM();

  CHKERR TSSetSolution(solver, D);
  CHKERR set_section_monitor(solver);
  CHKERR set_time_monitor(dm, solver);
  CHKERR TSSetSolution(solver, D);
  CHKERR TSSetFromOptions(solver);
  CHKERR TSSetUp(solver);
  CHKERR TSSolve(solver, NULL);

  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}
//! [Solve]

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
    MoFEM::Interface &m_field = core; ///< finite element database insterface
    //! [Create MoFEM]

    //! [Load mesh]
    Simple *simple = m_field.getInterface<Simple>();
    CHKERR simple->getOptions();
    CHKERR simple->loadFile();
    //! [Load mesh]

    //! [Example]
    Example ex(m_field);
    CHKERR ex.runProblem();
    //! [Example]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
