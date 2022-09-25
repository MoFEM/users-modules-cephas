/**
 * \file plastic.cpp
 * \example plastic.cpp
 *
 * Plasticity in 2d and 3d
 *
 */

#ifndef EXECUTABLE_DIMENSION
#define EXECUTABLE_DIMENSION 3
#endif

#include <MoFEM.hpp>
#include <MatrixFunction.hpp>
#include <IntegrationRules.hpp>

#ifdef __cplusplus
extern "C" {
#endif
#include <cblas.h>
#include <lapack_wrap.h>
// #include <gm_rule.h>
#include <quad.h>
#ifdef __cplusplus
}
#endif

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
constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;

constexpr AssemblyType A = AssemblyType::PETSC; //< selected assembly type
constexpr IntegrationType G =
    IntegrationType::GAUSS; //< selected integration type

using EntData = EntitiesFieldData::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

using AssemblyDomainEleOp = FormsIntegrators<DomainEleOp>::Assembly<A>::OpBase;

//! [Only used with Hooke equation (linear material model)]
using OpKCauchy = FormsIntegrators<DomainEleOp>::Assembly<A>::BiLinearForm<
    GAUSS>::OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM, 0>;
using OpInternalForceCauchy = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<G>::OpGradTimesSymTensor<1, SPACE_DIM, SPACE_DIM>;
//! [Only used with Hooke equation (linear material model)]

//! [Only used for dynamics]
using OpMass = FormsIntegrators<DomainEleOp>::Assembly<A>::BiLinearForm<
    GAUSS>::OpMass<1, SPACE_DIM>;
using OpInertiaForce = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<G>::OpBaseTimesVector<1, SPACE_DIM, 1>;
//! [Only used for dynamics]

//! [Only used with Hencky/nonlinear material]
using OpKPiola = FormsIntegrators<DomainEleOp>::Assembly<A>::BiLinearForm<
    GAUSS>::OpGradTensorGrad<1, SPACE_DIM, SPACE_DIM, 1>;
using OpInternalForcePiola = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<G>::OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;
//! [Only used with Hencky/nonlinear material]

//! [Essential boundary conditions]
using OpBoundaryMass = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::BiLinearForm<G>::OpMass<1, SPACE_DIM>;
using OpBoundaryVec = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<G>::OpBaseTimesVector<1, SPACE_DIM, 0>;
using OpBoundaryInternal = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<G>::OpBaseTimesVector<1, SPACE_DIM, 1>;
//! [Essential boundary conditions]
using OpScaleL2 = MoFEM::OpScaleBaseBySpaceInverseOfMeasure<DomainEleOp>;

using DomainNaturalBC = NaturalBC<DomainEleOp>::Assembly<A>::LinearForm<G>;
using OpBodyForce =
    DomainNaturalBC::OpFlux<NaturalMeshsetType<BLOCKSET>, 1, SPACE_DIM>;

using BoundaryNaturalBC = NaturalBC<BoundaryEleOp>::Assembly<A>::LinearForm<G>;
using OpForce =
    BoundaryNaturalBC::OpFlux<NaturalMeshsetType<BLOCKSET>, 1, SPACE_DIM>;

using OpEssentialLhs = EssentialBC<BoundaryEleOp>::Assembly<A>::BiLinearForm<
    GAUSS>::OpEssentialLhs<DisplacementCubitBcData, 1, SPACE_DIM>;
using OpEssentialRhs = EssentialBC<BoundaryEleOp>::Assembly<A>::LinearForm<
    GAUSS>::OpEssentialRhs<DisplacementCubitBcData, 1, SPACE_DIM>;

PetscBool is_large_strains = PETSC_TRUE;

double scale = 1.;

double young_modulus = 206913;
double poisson_ratio = 0.29;
double rho = 0;
double sigmaY = 450;
double H = 129;
double visH = 0;
double cn = 1;
double zeta = 5e-2;
double Qinf = 265;
double b_iso = 16.93;

int order = 2; ///< Order if fixed.
int geom_order = 2; ///< Order if fixed.

inline long double hardening(long double tau) {
  return H * tau + Qinf * (1. - std::exp(-b_iso * tau)) + sigmaY;
}

inline long double hardening_dtau(long double tau) {
  return H + Qinf * b_iso * std::exp(-b_iso * tau);
}

inline long double hardening_dtau2(long double tau) {
  return -(Qinf * (b_iso * b_iso)) * std::exp(-b_iso * tau);
}

#include <HenckyOps.hpp>
#include <PlasticOps.hpp>

using namespace PlasticOps;
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

  boost::shared_ptr<PlasticOps::CommonData> commonPlasticDataPtr;
  boost::shared_ptr<HenckyOps::CommonData> commonHenckyDataPtr;
  boost::shared_ptr<DomainEle> reactionFe;

  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;

  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
  boost::shared_ptr<std::vector<unsigned char>> reactionMarker;

  struct PlasticityTimeScale : public MoFEM::TimeScale {
    using MoFEM::TimeScale::TimeScale;
    double getScale(const double time) {
      return scale * MoFEM::TimeScale::getScale(time);
    };
  };
};

//! [Run problem]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR createCommonData();
  CHKERR setupProblem();
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

  Range domain_ents;
  CHKERR mField.get_moab().get_entities_by_dimension(0, SPACE_DIM, domain_ents,
                                                     true);
  auto get_ents_by_dim = [&](const auto dim) {
    if (dim == SPACE_DIM) {
      return domain_ents;
    } else {
      Range ents;
      if (dim == 0)
        CHKERR mField.get_moab().get_connectivity(domain_ents, ents, true);
      else
        CHKERR mField.get_moab().get_entities_by_dimension(0, dim, ents, true);
      return ents;
    }
  };

  auto get_base = [&]() {
    auto domain_ents = get_ents_by_dim(SPACE_DIM);
    if (domain_ents.empty())
      CHK_THROW_MESSAGE(MOFEM_NOT_FOUND, "Empty mesh");
    const auto type = type_from_handle(domain_ents[0]);
    switch (type) {
    case MBQUAD:
      return DEMKOWICZ_JACOBI_BASE;
    case MBHEX:
      return DEMKOWICZ_JACOBI_BASE;
    case MBTRI:
      return AINSWORTH_LEGENDRE_BASE;
    case MBTET:
      return AINSWORTH_LEGENDRE_BASE;
    default:
      CHK_THROW_MESSAGE(MOFEM_NOT_FOUND, "Element type not handled");
    }
    return NOBASE;
  };

  const auto base = get_base();
  MOFEM_LOG("WORLD", Sev::inform) << "Base " << ApproximationBaseNames[base];

  CHKERR simple->addDomainField("U", H1, base, SPACE_DIM);
  CHKERR simple->addDomainField("TAU", L2, base, 1);
  CHKERR simple->addDomainField("EP", L2, base, size_symm);
  CHKERR simple->addBoundaryField("U", H1, base, SPACE_DIM);

  CHKERR simple->addDataField("GEOMETRY", H1, base, SPACE_DIM);

  auto ents = get_ents_by_dim(0);
  ents.merge(get_ents_by_dim(1));
  // ents.merge(get_ents_by_dim(2));
  CHKERR simple->setFieldOrder("U", order, &ents);
  CHKERR simple->setFieldOrder("EP", order - 1);
  CHKERR simple->setFieldOrder("TAU", order - 2);

  CHKERR simple->setFieldOrder("GEOMETRY", geom_order);

  CHKERR simple->setUp();
  CHKERR simple->addFieldToEmptyFieldBlocks("U", "TAU");

  auto project_ho_geometry = [&]() {
    Projection10NodeCoordsOnField ent_method(mField, "GEOMETRY");
    return mField.loop_dofs("GEOMETRY", ent_method);
  };
  CHKERR project_ho_geometry();

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
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-hardening", &H, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-hardening_viscous", &visH,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-yield_stress", &sigmaY,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-cn", &cn, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-zeta", &zeta, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-Qinf", &Qinf, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-b_iso", &b_iso, PETSC_NULL);
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-large_strains",
                               &is_large_strains, PETSC_NULL);

    CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
    CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-geom_order", &geom_order,
                              PETSC_NULL);

    MOFEM_LOG("EXAMPLE", Sev::inform) << "Young modulus " << young_modulus;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Poisson ratio " << poisson_ratio;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Yield stress " << sigmaY;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Hardening " << H;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Viscous hardening " << visH;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Saturation yield stress " << Qinf;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Saturation exponent " << b_iso;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "cn " << cn;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "zeta " << zeta;

    MOFEM_LOG("EXAMPLE", Sev::inform) << "order " << order;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "geom order " << geom_order;

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

      MOFEM_LOG("EXAMPLE", Sev::inform)
          << "Scaled Young modulus " << young_modulus;
      MOFEM_LOG("EXAMPLE", Sev::inform)
          << "Scaled Poisson ratio " << poisson_ratio;
      MOFEM_LOG("EXAMPLE", Sev::inform) << "Scaled Yield stress " << sigmaY;
      MOFEM_LOG("EXAMPLE", Sev::inform) << "Scaled Hardening " << H;
      MOFEM_LOG("EXAMPLE", Sev::inform) << "Scaled Viscous hardening " << visH;
      MOFEM_LOG("EXAMPLE", Sev::inform)
          << "Scaled Saturation yield stress " << Qinf;
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
                         :
      1;

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

  auto make_d_mat = []() {
    return boost::make_shared<MatrixDouble>(size_symm * size_symm, 1);
  };

  commonPlasticDataPtr = boost::make_shared<PlasticOps::CommonData>();
  commonPlasticDataPtr->mDPtr = make_d_mat();
  commonPlasticDataPtr->mDPtr_Axiator = make_d_mat();
  commonPlasticDataPtr->mDPtr_Deviator = make_d_mat();

  commonPlasticDataPtr->mGradPtr = boost::make_shared<MatrixDouble>();
  commonPlasticDataPtr->mStrainPtr = boost::make_shared<MatrixDouble>();
  commonPlasticDataPtr->mStressPtr = boost::make_shared<MatrixDouble>();

  CHKERR get_command_line_parameters();
  CHKERR set_matrial_stiffness();

  if (is_large_strains) {
    commonHenckyDataPtr = boost::make_shared<HenckyOps::CommonData>();
    commonHenckyDataPtr->matGradPtr = commonPlasticDataPtr->mGradPtr;
    commonHenckyDataPtr->matDPtr = commonPlasticDataPtr->mDPtr;
    commonHenckyDataPtr->matLogCPlastic =
        commonPlasticDataPtr->getPlasticStrainPtr();
    commonPlasticDataPtr->mStrainPtr = commonHenckyDataPtr->getMatLogC();
    commonPlasticDataPtr->mStressPtr =
        commonHenckyDataPtr->getMatHenckyStress();
  }

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

  CHKERR bc_mng->pushMarkDOFsOnEntities<DisplacementCubitBcData>(
      simple->getProblemName(), "U");

  auto &bc_map = bc_mng->getBcMapByBlockName();
  boundaryMarker = bc_mng->getMergedBlocksMarker(vector<string>{"FIX_"});

  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "REACTION",
                                        "U", 0, 3);

  for (auto bc : bc_map)
    MOFEM_LOG("EXAMPLE", Sev::verbose) << "Marker " << bc.first;

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

  auto add_domain_base_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pipeline, {H1, L2},
                                                          "GEOMETRY");

    pipeline.push_back(new OpCalculateScalarFieldValuesDot(
        "TAU", commonPlasticDataPtr->getPlasticTauDotPtr()));
    pipeline.push_back(new OpCalculateScalarFieldValues(
        "TAU", commonPlasticDataPtr->getPlasticTauPtr()));
    pipeline.push_back(new OpCalculateTensor2SymmetricFieldValues<SPACE_DIM>(
        "EP", commonPlasticDataPtr->getPlasticStrainPtr()));
    pipeline.push_back(new OpCalculateTensor2SymmetricFieldValuesDot<SPACE_DIM>(
        "EP", commonPlasticDataPtr->getPlasticStrainDotPtr()));
    pipeline.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        "U", commonPlasticDataPtr->mGradPtr));

    MoFEMFunctionReturn(0);
  };

  auto add_domain_stress_ops = [&](auto &pipeline, auto m_D_ptr) {
    MoFEMFunctionBegin;

    if (is_large_strains) {

      if (commonPlasticDataPtr->mGradPtr != commonHenckyDataPtr->matGradPtr)
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                "Wrong pointer for grad");

      pipeline.push_back(
          new OpCalculateEigenVals<SPACE_DIM>("U", commonHenckyDataPtr));
      pipeline.push_back(
          new OpCalculateLogC<SPACE_DIM>("U", commonHenckyDataPtr));
      pipeline.push_back(
          new OpCalculateLogC_dC<SPACE_DIM>("U", commonHenckyDataPtr));
      pipeline.push_back(new OpCalculateHenckyPlasticStress<SPACE_DIM>(
          "U", commonHenckyDataPtr, m_D_ptr));
      pipeline.push_back(
          new OpCalculatePiolaStress<SPACE_DIM>("U", commonHenckyDataPtr));

    } else {
      pipeline.push_back(
          new OpSymmetrizeTensor<SPACE_DIM>("U", commonPlasticDataPtr->mGradPtr,
                                            commonPlasticDataPtr->mStrainPtr));
      pipeline.push_back(
          new OpPlasticStress("U", commonPlasticDataPtr, m_D_ptr, 1));
    }

    if (m_D_ptr != commonPlasticDataPtr->mDPtr_Axiator) {
      pipeline.push_back(
          new OpCalculatePlasticSurface("U", commonPlasticDataPtr));
      pipeline.push_back(
          new OpCalculatePlasticity("U", commonPlasticDataPtr, m_D_ptr));
    }

    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_lhs_mechanical = [&](auto &pipeline, auto m_D_ptr) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    if (is_large_strains) {
      pipeline.push_back(
          new OpHenckyTangent<SPACE_DIM>("U", commonHenckyDataPtr, m_D_ptr));
      pipeline.push_back(
          new OpKPiola("U", "U", commonHenckyDataPtr->getMatTangent()));
      pipeline.push_back(new OpCalculatePlasticInternalForceLhs_LogStrain_dEP(
          "U", "EP", commonPlasticDataPtr, commonHenckyDataPtr, m_D_ptr));
    } else {
      pipeline.push_back(new OpKCauchy("U", "U", m_D_ptr));
      pipeline.push_back(new OpCalculatePlasticInternalForceLhs_dEP(
          "U", "EP", commonPlasticDataPtr, m_D_ptr));
    }

    pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_rhs_mechanical = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    CHKERR DomainNaturalBC::AddFluxToPipeline<OpBodyForce>::add(
        pipeline, mField, "U", {boost::make_shared<PlasticityTimeScale>()},
        "BODY_FORCE", Sev::inform);

    // Calculate internal forces
    if (is_large_strains) {
      pipeline.push_back(new OpInternalForcePiola(
          "U", commonHenckyDataPtr->getMatFirstPiolaStress()));
    } else {
      pipeline.push_back(
          new OpInternalForceCauchy("U", commonPlasticDataPtr->mStressPtr));
    }

    pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_lhs_constrain = [&](auto &pipeline, auto m_D_ptr) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    if (is_large_strains) {
      pipeline.push_back(new OpCalculateConstraintsLhs_LogStrain_dU(
          "TAU", "U", commonPlasticDataPtr, commonHenckyDataPtr, m_D_ptr));
      pipeline.push_back(new OpCalculatePlasticFlowLhs_LogStrain_dU(
          "EP", "U", commonPlasticDataPtr, commonHenckyDataPtr, m_D_ptr));
    } else {
      pipeline.push_back(new OpCalculateConstraintsLhs_dU(
          "TAU", "U", commonPlasticDataPtr, m_D_ptr));
      pipeline.push_back(new OpCalculatePlasticFlowLhs_dU(
          "EP", "U", commonPlasticDataPtr, m_D_ptr));
    }

    pipeline.push_back(new OpCalculatePlasticFlowLhs_dEP(
        "EP", "EP", commonPlasticDataPtr, m_D_ptr));
    pipeline.push_back(new OpCalculatePlasticFlowLhs_dTAU(
        "EP", "TAU", commonPlasticDataPtr, m_D_ptr));
    pipeline.push_back(new OpCalculateConstraintsLhs_dEP(
        "TAU", "EP", commonPlasticDataPtr, m_D_ptr));
    pipeline.push_back(
        new OpCalculateConstraintsLhs_dTAU("TAU", "TAU", commonPlasticDataPtr));

    pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_rhs_constrain = [&](auto &pipeline, auto m_D_ptr) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    pipeline.push_back(
        new OpCalculateConstraintsRhs("TAU", commonPlasticDataPtr, m_D_ptr));
    pipeline.push_back(
        new OpCalculatePlasticFlowRhs("EP", commonPlasticDataPtr, m_D_ptr));

    pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_boundary_ops_lhs_mechanical = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    CHKERR EssentialBC<BoundaryEleOp>::Assembly<A>::BiLinearForm<G>::
        AddEssentialToPipeline<OpEssentialLhs>::add(
            mField, pipeline, simple->getProblemName(), "U");
    MoFEMFunctionReturn(0);
  };

  auto add_boundary_ops_rhs_mechanical = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
        pipeline, {NOSPACE}, "GEOMETRY");

    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));
    CHKERR BoundaryNaturalBC::AddFluxToPipeline<OpForce>::add(
        pipeline, mField, "U", {boost::make_shared<PlasticityTimeScale>()},
        "FORCE", Sev::inform);

    pipeline.push_back(new OpUnSetBc("U"));

    auto u_mat_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_mat_ptr));

    CHKERR EssentialBC<BoundaryEleOp>::Assembly<A>::LinearForm<G>::
        AddEssentialToPipeline<OpEssentialRhs>::add(
            mField, pipeline, simple->getProblemName(), "U", u_mat_ptr,
            {boost::make_shared<TimeScale>()}); // note displacements have no
                                                // scaling

    MoFEMFunctionReturn(0);
  };

  CHKERR add_domain_base_ops(pipeline_mng->getOpDomainLhsPipeline());
  CHKERR add_domain_stress_ops(pipeline_mng->getOpDomainLhsPipeline(),
                               commonPlasticDataPtr->mDPtr);
  CHKERR add_domain_ops_lhs_mechanical(pipeline_mng->getOpDomainLhsPipeline(),
                                       commonPlasticDataPtr->mDPtr);
  CHKERR add_domain_ops_lhs_constrain(pipeline_mng->getOpDomainLhsPipeline(),
                                      commonPlasticDataPtr->mDPtr);
  CHKERR
  add_boundary_ops_lhs_mechanical(pipeline_mng->getOpBoundaryLhsPipeline());

  CHKERR add_domain_base_ops(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_domain_stress_ops(pipeline_mng->getOpDomainRhsPipeline(),
                               commonPlasticDataPtr->mDPtr);
  CHKERR add_domain_ops_rhs_mechanical(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_domain_ops_rhs_constrain(pipeline_mng->getOpDomainRhsPipeline(),
                                      commonPlasticDataPtr->mDPtr);

  // Boundary
  CHKERR
  add_boundary_ops_rhs_mechanical(pipeline_mng->getOpBoundaryRhsPipeline());

  auto integration_rule_bc = [](int, int, int ao) { return 2 * ao; };

  auto vol_rule = [](int, int, int ao) { return 2 * ao + geom_order - 1; };

  CHKERR pipeline_mng->setDomainRhsIntegrationRule(vol_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(vol_rule);

  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule_bc);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule_bc);

  auto create_reaction_pipeline = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    if (reactionMarker) {

      CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pipeline, {H1, L2},
                                                            "GEOMETRY");

      pipeline.push_back(
          new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
              "U", commonPlasticDataPtr->mGradPtr));
      pipeline.push_back(new OpCalculateTensor2SymmetricFieldValues<SPACE_DIM>(
          "EP", commonPlasticDataPtr->getPlasticStrainPtr()));

      if (is_large_strains) {

        if (commonPlasticDataPtr->mGradPtr != commonHenckyDataPtr->matGradPtr)
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                  "Wrong pointer for grad");

        pipeline.push_back(
            new OpCalculateEigenVals<SPACE_DIM>("U", commonHenckyDataPtr));
        pipeline.push_back(
            new OpCalculateLogC<SPACE_DIM>("U", commonHenckyDataPtr));
        pipeline.push_back(
            new OpCalculateLogC_dC<SPACE_DIM>("U", commonHenckyDataPtr));
        pipeline.push_back(new OpCalculateHenckyPlasticStress<SPACE_DIM>(
            "U", commonHenckyDataPtr, commonPlasticDataPtr->mDPtr));
        pipeline.push_back(
            new OpCalculatePiolaStress<SPACE_DIM>("U", commonHenckyDataPtr));

      } else {
        pipeline.push_back(new OpSymmetrizeTensor<SPACE_DIM>(
            "U", commonPlasticDataPtr->mGradPtr,
            commonPlasticDataPtr->mStrainPtr));
        pipeline.push_back(new OpPlasticStress(
            "U", commonPlasticDataPtr, commonPlasticDataPtr->mDPtr, scale));
      }

      pipeline.push_back(new OpSetBc("U", false, reactionMarker));
      // Calculate internal force
      if (is_large_strains) {
        pipeline.push_back(new OpInternalForcePiola(
            "U", commonHenckyDataPtr->getMatFirstPiolaStress()));
      } else {
        pipeline.push_back(
            new OpInternalForceCauchy("U", commonPlasticDataPtr->mStressPtr));
      }
      pipeline.push_back(new OpUnSetBc("U"));
    }

    MoFEMFunctionReturn(0);
  };

  reactionFe = boost::make_shared<DomainEle>(mField);
  reactionFe->getRuleHook = vol_rule;

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

  auto snes_ctx_ptr = smartGetDMSnesCtx(simple->getDM());

  auto set_section_monitor = [&](auto solver) {
    MoFEMFunctionBegin;
    SNES snes;
    CHKERR TSGetSNES(solver, &snes);
    CHKERR SNESMonitorSet(snes,
                          (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal,
                                             void *))MoFEMSNESMonitorFields,
                          (void *)(snes_ctx_ptr.get()), nullptr);
    MoFEMFunctionReturn(0);
  };

  auto create_post_process_element = [&]() {
    auto pp_fe = boost::make_shared<PostProcEle>(mField);
    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

        
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        pp_fe->getOpPtrVector(), {H1, L2}, "GEOMETRY");

    auto x_ptr = boost::make_shared<MatrixDouble>();
    pp_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("GEOMETRY", x_ptr));
    auto u_ptr = boost::make_shared<MatrixDouble>();
    pp_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));

    pp_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
            "U", commonPlasticDataPtr->mGradPtr));
    pp_fe->getOpPtrVector().push_back(new OpCalculateScalarFieldValues(
        "TAU", commonPlasticDataPtr->getPlasticTauPtr()));
    pp_fe->getOpPtrVector().push_back(
        new OpCalculateTensor2SymmetricFieldValues<SPACE_DIM>(
            "EP", commonPlasticDataPtr->getPlasticStrainPtr()));

    if (is_large_strains) {

      if (commonPlasticDataPtr->mGradPtr != commonHenckyDataPtr->matGradPtr)
        CHK_THROW_MESSAGE(MOFEM_DATA_INCONSISTENCY, "Wrong pointer for grad");

      pp_fe->getOpPtrVector().push_back(
          new OpCalculateEigenVals<SPACE_DIM>("U", commonHenckyDataPtr));
      pp_fe->getOpPtrVector().push_back(
          new OpCalculateLogC<SPACE_DIM>("U", commonHenckyDataPtr));
      pp_fe->getOpPtrVector().push_back(
          new OpCalculateLogC_dC<SPACE_DIM>("U", commonHenckyDataPtr));
      pp_fe->getOpPtrVector().push_back(
          new OpCalculateHenckyPlasticStress<SPACE_DIM>(
              "U", commonHenckyDataPtr, commonPlasticDataPtr->mDPtr,
              1 /*scale*/));
      pp_fe->getOpPtrVector().push_back(
          new OpCalculatePiolaStress<SPACE_DIM>("U", commonHenckyDataPtr));

      pp_fe->getOpPtrVector().push_back(

          new OpPPMap(

              pp_fe->getPostProcMesh(), pp_fe->getMapGaussPts(),

              {},

              {{"U", u_ptr}, {"GEOMETRY", x_ptr}},

              {{"GRAD", commonPlasticDataPtr->mGradPtr},
               {"FIRST_PIOLA", commonHenckyDataPtr->getMatFirstPiolaStress()}},

              {}

              )

      );

    } else {
      pp_fe->getOpPtrVector().push_back(
          new OpSymmetrizeTensor<SPACE_DIM>("U", commonPlasticDataPtr->mGradPtr,
                                            commonPlasticDataPtr->mStrainPtr));
      pp_fe->getOpPtrVector().push_back(new OpPlasticStress(
          "U", commonPlasticDataPtr, commonPlasticDataPtr->mDPtr, 1 /*scale*/));

      pp_fe->getOpPtrVector().push_back(

          new OpPPMap(

              pp_fe->getPostProcMesh(), pp_fe->getMapGaussPts(),

              {},

              {{"U", u_ptr}, {"GEOMETRY", x_ptr}},

              {},

              {{"STRAIN", commonPlasticDataPtr->mStrainPtr},
               {"STRESS", commonPlasticDataPtr->mStressPtr}}

              )

      );
    }

    pp_fe->getOpPtrVector().push_back(
        new OpCalculatePlasticSurface("U", commonPlasticDataPtr));

    pp_fe->getOpPtrVector().push_back(

        new OpPPMap(

            pp_fe->getPostProcMesh(), pp_fe->getMapGaussPts(),

            {{"PLASTIC_SURFACE", commonPlasticDataPtr->getPlasticSurfacePtr()},
             {"PLASTIC_MULTIPLIER", commonPlasticDataPtr->getPlasticTauPtr()}},

            {},

            {},

            {{"PLASTIC_STRAIN", commonPlasticDataPtr->getPlasticStrainPtr()},
             {"PLASTIC_FLOW", commonPlasticDataPtr->getPlasticFlowPtr()}}

            )

    );

    return pp_fe;
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
    boost::shared_ptr<Monitor> monitor_ptr(
        new Monitor(dm, create_post_process_element(), reactionFe, uXScatter,
                    uYScatter, uZScatter));
    boost::shared_ptr<ForcesAndSourcesCore> null;
    CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                               monitor_ptr, null, null);
    MoFEMFunctionReturn(0);
  };

  auto set_fieldsplit_preconditioner = [&](auto solver) {
    MoFEMFunctionBeginHot;

    SNES snes;
    CHKERR TSGetSNES(solver, &snes);
    KSP ksp;
    CHKERR SNESGetKSP(snes, &ksp);
    PC pc;
    CHKERR KSPGetPC(ksp, &pc);
    PetscBool is_pcfs = PETSC_FALSE;
    PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &is_pcfs);

    // Setup fieldsplit (block) solver - optional: yes/no
    if (is_pcfs == PETSC_TRUE) {

      auto bc_mng = mField.getInterface<BcManager>();
      auto name_prb = simple->getProblemName();
      auto is_all_bc = bc_mng->getBlockIS(name_prb, "FIX_X", "U", 0, 0);
      is_all_bc = bc_mng->getBlockIS(name_prb, "FIX_Y", "U", 1, 1, is_all_bc);
      is_all_bc = bc_mng->getBlockIS(name_prb, "FIX_Z", "U", 2, 2, is_all_bc);
      is_all_bc = bc_mng->getBlockIS(name_prb, "FIX_ALL", "U", 0, 2, is_all_bc);

      int is_all_bc_size;
      CHKERR ISGetSize(is_all_bc, &is_all_bc_size);
      MOFEM_LOG("EXAMPLE", Sev::inform)
          << "Field split block size " << is_all_bc_size;

      CHKERR PCFieldSplitSetIS(pc, PETSC_NULL,
                               is_all_bc); // boundary block
    }

    MoFEMFunctionReturnHot(0);
  };

  auto dm = simple->getDM();
  auto D = smartCreateDMVector(dm);
  uXScatter = scatter_create(D, 0);
  uYScatter = scatter_create(D, 1);
  if constexpr (SPACE_DIM == 3)
    uZScatter = scatter_create(D, 2);

  auto solver = pipeline_mng->createTSIM();

  auto post_rhs = [&]() {
    MoFEMFunctionBegin;
    auto get_iter = [&]() {
      SNES snes;
      CHKERR TSGetSNES(solver, &snes);
      int iter;
      CHKERR SNESGetIterationNumber(snes, &iter);
      return iter;
    };

    MoFEMFunctionReturn(0);
  };

  mField.getInterface<PipelineManager>()->getDomainRhsFE()->postProcessHook =
      post_rhs;

  CHKERR TSSetSolution(solver, D);
  CHKERR set_section_monitor(solver);
  CHKERR set_time_monitor(dm, solver);
  CHKERR TSSetSolution(solver, D);
  CHKERR TSSetFromOptions(solver);
  CHKERR set_fieldsplit_preconditioner(solver);
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
