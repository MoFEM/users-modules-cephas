/**
 * \file plastic.cpp
 * \example plastic.cpp
 *
 * Plasticity in 2d and 3d
 *
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
  using DomainEleOp = DomainEle::UserDataOperator;
  using BoundaryEle = PipelineManager::EdgeEle;
  using BoundaryEleOp = BoundaryEle::UserDataOperator;
  using SkeletonEle = PipelineManager::EdgeEle;
  using SkeletonEleOp = SkeletonEle::UserDataOperator;
  using PostProcEle = PostProcFaceOnRefinedMesh;
  using PostProcEleSkeleton = PostProcEdgeOnRefinedMesh;

  using DomainSideEle = FaceElementForcesAndSourcesCoreOnSideSwitch<0>;
  using DomainSideEleOp = DomainSideEle::UserDataOperator;

  using OpSetPiolaTransformOnBoundary =
      OpSetContravariantPiolaTransformOnEdge2D;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = VolumeElementForcesAndSourcesCore;
  using DomainEleOp = DomainEle::UserDataOperator;
  using BoundaryEle = FaceElementForcesAndSourcesCore;
  using BoundaryEleOp = BoundaryEle::UserDataOperator;
  using SkeletonEle = FaceElementForcesAndSourcesCore;
  using SkeletonEleOp = SkeletonEle::UserDataOperator;
  using PostProcEle = PostProcVolumeOnRefinedMesh;
  using PostProcEleSkeleton = PostProcFaceOnRefinedMesh;

  using DomainSideEle = VolumeElementForcesAndSourcesCoreOnSide;
  using DomainSideEleOp = DomainSideEle::UserDataOperator;

  using OpSetPiolaTransformOnBoundary =
      OpHOSetContravariantPiolaTransformOnFace3D;
};

//< Space dimension of problem, mesh
constexpr int SPACE_DIM = EXECUTABLE_DIMENSION;


constexpr EntityType boundary_ent = SPACE_DIM == 3 ? MBTRI : MBEDGE;
using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = ElementsAndOps<SPACE_DIM>::DomainEleOp;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = ElementsAndOps<SPACE_DIM>::BoundaryEleOp;
using PostProcEle = ElementsAndOps<SPACE_DIM>::PostProcEle;

using SkeletonEle = ElementsAndOps<SPACE_DIM>::SkeletonEle;
using SkeletonEleOp = ElementsAndOps<SPACE_DIM>::SkeletonEleOp;
using DomainSideEle = ElementsAndOps<SPACE_DIM>::DomainSideEle;
using DomainSideEleOp = ElementsAndOps<SPACE_DIM>::DomainSideEleOp;

using PostProcEleSkeleton = ElementsAndOps<SPACE_DIM>::PostProcEleSkeleton;

using AssemblyDomainEleOp =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::OpBase;
using AssemblySkeletonEleOp =
    FormsIntegrators<SkeletonEleOp>::Assembly<PETSC>::OpBase;

using OpSetPiolaTransformOnBoundary =
    ElementsAndOps<SPACE_DIM>::OpSetPiolaTransformOnBoundary;

//! [Body force]
using OpBodyForce = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpSource<1, SPACE_DIM>;
//! [Body force]

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

PetscBool is_large_strains = PETSC_TRUE;
PetscBool is_with_ALE = PETSC_TRUE; 
PetscBool test_convection = PETSC_TRUE; 
PetscBool test_penalty = PETSC_TRUE;

double petsc_time = 0;
double scale = 1.;

double young_modulus = 206913;
double poisson_ratio = 0.29;
double rho = 0;
double sigmaY = 450;
double H = 1290;
double visH = 0;
double cn = 1;
double Qinf = 265;
double b_iso = 16.93;
int order = 2;

// for rotating body
double penalty = 1e5;
double density = 1.0;
array<double, 3> angular_velocity{0, 0, 1};

inline long double hardening(long double tau, double temp) {
  return H * tau + Qinf * (1. - std::exp(-b_iso * tau)) + sigmaY;
}

inline long double hardening_dtau(long double tau, double temp) {
  return H + Qinf * b_iso * std::exp(-b_iso * tau);
}

#include <HenckyOps.hpp>
#include <PlasticOps.hpp>
#include <OpPostProcElastic.hpp>
using namespace PlasticOps;
#include <AlePlasticOps.hpp>

using namespace HenckyOps;
using namespace AlePlasticOps;

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

  boost::shared_ptr<AlePlasticOps::CommonData> commonDataPtr;
  boost::shared_ptr<HenckyOps::CommonData> commonHenckyDataPtr;

  boost::shared_ptr<PostProcEle> postProcFe;
  boost::shared_ptr<PostProcEleSkeleton> postProcFeSkeletonFe;
  boost::shared_ptr<DomainEle> reactionFe;
  boost::shared_ptr<DomainEle> feAxiatorRhs;
  boost::shared_ptr<DomainEle> feAxiatorLhs;

  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;

  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;

  std::vector<FTensor::Tensor1<double, 3>> bodyForces;
  boost::shared_ptr<DomainSideEle> domainSideFe;
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
  // FieldApproximationBase base = DEMKOWICZ_JACOBI_BASE;
  // FieldApproximationBase base = AINSWORTH_LEGENDRE_BASE;

  // Select base
  enum bases { AINSWORTH, DEMKOWICZ, LASBASETOPT };
  const char *list_bases[LASBASETOPT] = {"ainsworth", "demkowicz"};
  PetscInt choice_base_value = AINSWORTH;
  CHKERR PetscOptionsGetEList(PETSC_NULL, NULL, "-base", list_bases,
                              LASBASETOPT, &choice_base_value, PETSC_NULL);

  FieldApproximationBase base;
  switch (choice_base_value) {
  case AINSWORTH:
    base = AINSWORTH_LEGENDRE_BASE;
    MOFEM_LOG("EXAMPLE", Sev::inform)
        << "Set AINSWORTH_LEGENDRE_BASE for displacements";
    break;
  case DEMKOWICZ:
    base = DEMKOWICZ_JACOBI_BASE;
    MOFEM_LOG("EXAMPLE", Sev::inform)
        << "Set DEMKOWICZ_JACOBI_BASE for displacents";
    break;
  default:
    base = LASTBASE;
    break;
  }

  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  CHKERR simple->addDomainField("U", H1, base, SPACE_DIM);
  CHKERR simple->addDomainField("TAU", L2, base, 1);
  CHKERR simple->addDomainField("EP", L2, base, size_symm);
  CHKERR simple->addBoundaryField("U", H1, base, SPACE_DIM);
  // for ALE
  if (is_with_ALE)
    CHKERR simple->addSkeletonField("U", H1, base, order);

  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setFieldOrder("TAU", order - 1);
  CHKERR simple->setFieldOrder("EP", order - 1);

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
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-hardening", &H, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-hardening_viscous", &visH,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-yield_stress", &sigmaY,
                                 PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-cn", &cn, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-Qinf", &Qinf, PETSC_NULL);
    CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-b_iso", &b_iso, PETSC_NULL);

    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-large_strains",
                               &is_large_strains, PETSC_NULL);
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-is_ale",
                               &is_with_ALE, PETSC_NULL);
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-test_penalty",
                               &test_penalty, PETSC_NULL);
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-test_convection",
                               &test_convection, PETSC_NULL);

    MOFEM_LOG("EXAMPLE", Sev::inform) << "Young modulus " << young_modulus;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Poisson ratio " << poisson_ratio;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Yield stress " << sigmaY;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Hardening " << H;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Viscous hardening " << visH;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Saturation yield stress " << Qinf;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Saturation exponent " << b_iso;
    MOFEM_LOG("EXAMPLE", Sev::inform) << "cn " << cn;

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
                         : 1;

    auto t_D =
        getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);
    auto t_D_axiator = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(
        *commonDataPtr->mDPtr_Axiator);
    auto t_D_deviator = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(
        *commonDataPtr->mDPtr_Deviator);

    constexpr double third = boost::math::constants::third<double>();
    t_D_axiator(i, j, k, l) = A *
                              (bulk_modulus_K - (2. / 3.) * shear_modulus_G) *
                              t_kd(i, j) * t_kd(k, l);
    t_D_deviator(i, j, k, l) =
        2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.);
    t_D(i, j, k, l) = t_D_axiator(i, j, k, l) + t_D_deviator(i, j, k, l);

    MoFEMFunctionReturn(0);
  };

  commonDataPtr = boost::make_shared<AlePlasticOps::CommonData>();
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  commonDataPtr->mDPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mDPtr->resize(size_symm * size_symm, 1);
  commonDataPtr->mDPtr_Axiator = boost::make_shared<MatrixDouble>();
  commonDataPtr->mDPtr_Axiator->resize(size_symm * size_symm, 1);
  commonDataPtr->mDPtr_Deviator = boost::make_shared<MatrixDouble>();
  commonDataPtr->mDPtr_Deviator->resize(size_symm * size_symm, 1);

  commonDataPtr->mGradPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mStrainPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mStressPtr = boost::make_shared<MatrixDouble>();

  CHKERR get_command_line_parameters();
  CHKERR set_matrial_stiffness();

  if (is_large_strains) {
    commonHenckyDataPtr = boost::make_shared<HenckyOps::CommonData>();
    commonHenckyDataPtr->matGradPtr = commonDataPtr->mGradPtr;
    commonHenckyDataPtr->matDPtr = commonDataPtr->mDPtr;
    commonHenckyDataPtr->matLogCPlastic = commonDataPtr->getPlasticStrainPtr();
    commonDataPtr->mStrainPtr = commonHenckyDataPtr->getMatLogC();
    commonDataPtr->mStressPtr = commonHenckyDataPtr->getMatHenckyStress();
  }

  // for ALE
  commonDataPtr->plasticTauJumpPtr = boost::make_shared<VectorDouble>();
  commonDataPtr->plasticStrainJumpPtr = boost::make_shared<MatrixDouble>();

  commonDataPtr->guidingVelocityPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->velocityDotNormalPtr = boost::make_shared<VectorDouble>();

  commonDataPtr->plasticGradTauPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->plasticGradStrainPtr = boost::make_shared<MatrixDouble>();

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

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::OPs() {
  MoFEMFunctionBegin;
  auto pipeline_mng = mField.getInterface<PipelineManager>();
  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();

  feAxiatorLhs = boost::make_shared<DomainEle>(mField);
  feAxiatorRhs = boost::make_shared<DomainEle>(mField);
  auto integration_rule_axiator = [](int, int, int approx_order) {
    return 2 * (approx_order - 1);
  };
  feAxiatorLhs->getRuleHook = integration_rule_axiator;
  feAxiatorRhs->getRuleHook = integration_rule_axiator;

  auto integration_rule_deviator = [](int o_row, int o_col, int approx_order) {
    return 2 * (approx_order - 1);
  };
  auto integration_rule_bc = [](int, int, int approx_order) {
    return 2 * approx_order;
  };

  auto add_skeleton_base_ops = [&](auto &pipeline) {
    MoFEMFunctionBeginHot;

    // pipeline.push_back(new OpSetPiolaTransformOnBoundary());

    // domainSideFe->getRuleHook = [](double, double, double) { return -1; };

    // if (SPACE_DIM == 2) {
    //   auto det_ptr = boost::make_shared<VectorDouble>();
    //   auto jac_ptr = boost::make_shared<MatrixDouble>();
    //   auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    //   domainSideFe->getOpPtrVector().push_back(
    //       new OpCalculateHOJacForFace(jac_ptr));
    //   domainSideFe->getOpPtrVector().push_back(
    //       new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
    //   // domainSideFe->getOpPtrVector().push_back(
    //   //     new OpSetInvJacH1ForFace(inv_jac_ptr));
    //   // domainSideFe->getOpPtrVector().push_back(
    //   //     new OpSetInvJacL2ForFace(inv_jac_ptr));
    // }

    domainSideFe = boost::make_shared<DomainSideEle>(mField);
    domainSideFe->getOpPtrVector().push_back(
        new OpVolumeSideCalculateEP("EP", commonDataPtr));
    domainSideFe->getOpPtrVector().push_back(
        new OpVolumeSideCalculateTAU("TAU", commonDataPtr));
    pipeline.push_back(
        new OpCalculateVelocityOnSkeleton("U", commonDataPtr));

    MoFEMFunctionReturnHot(0);
  };

  auto add_skeleton_rhs_ops = [&](auto &pipeline) {
    MoFEMFunctionBeginHot;
    // pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    auto domain_side_fe_ep = boost::make_shared<DomainSideEle>(mField);
    auto domain_side_fe_tau = boost::make_shared<DomainSideEle>(mField);
    domain_side_fe_ep->getOpPtrVector().push_back(
        new OpVolumeSideCalculateEP("EP", commonDataPtr));
    domain_side_fe_tau->getOpPtrVector().push_back(
        new OpVolumeSideCalculateTAU("TAU", commonDataPtr));

    pipeline.push_back(
        new OpCalculateJumpOnEPSkeleton("U", commonDataPtr, domain_side_fe_ep));
    pipeline.push_back(
        new OpCalculatePlasticFlowPenalty_Rhs("U", commonDataPtr));
    pipeline.push_back(new OpCalculateJumpOnTAUSkeleton("U", commonDataPtr,
                                                        domain_side_fe_tau));
    pipeline.push_back(
        new OpCalculateConstraintPenalty_Rhs("U", commonDataPtr));

    // pipeline.push_back(new OpUnSetBc("U"));

    MoFEMFunctionReturnHot(0);
  };

  auto add_skeleton_lhs_ops = [&](auto &pipeline) {
    MoFEMFunctionBeginHot;
    // pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    auto domain_side_fe_ep = boost::make_shared<DomainSideEle>(mField);
    auto domain_side_fe_tau = boost::make_shared<DomainSideEle>(mField);
    domain_side_fe_ep->getOpPtrVector().push_back(
        new OpVolumeSideCalculateEP("EP", commonDataPtr));
    domain_side_fe_ep->getOpPtrVector().push_back(
        new OpDomainSideGetColData("EP", "EP", commonDataPtr));
    domain_side_fe_tau->getOpPtrVector().push_back(
        new OpVolumeSideCalculateTAU("TAU", commonDataPtr));
    domain_side_fe_tau->getOpPtrVector().push_back(
        new OpDomainSideGetColData("TAU", "TAU", commonDataPtr));

    pipeline.push_back(
        new OpCalculateJumpOnEPSkeleton("U", commonDataPtr, domain_side_fe_ep));
    pipeline.push_back(
        new OpCalculatePlasticFlowPenaltyLhs_dEP("U", "U", commonDataPtr));
    pipeline.push_back(new OpCalculateJumpOnTAUSkeleton("U", commonDataPtr,
                                                        domain_side_fe_tau));
    pipeline.push_back(
        new OpCalculateConstraintPenaltyLhs_dTAU("U", "U", commonDataPtr));

    // pipeline.push_back(new OpUnSetBc("U"));

    MoFEMFunctionReturnHot(0);
  };

  auto add_domain_base_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    if (SPACE_DIM == 2) {
      auto det_ptr = boost::make_shared<VectorDouble>();
      auto jac_ptr = boost::make_shared<MatrixDouble>();
      auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
      pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
      pipeline.push_back(new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
      pipeline.push_back(new OpSetInvJacH1ForFace(inv_jac_ptr));
      pipeline.push_back(new OpSetInvJacL2ForFace(inv_jac_ptr));
    }

    pipeline.push_back(new OpCalculateScalarFieldValuesDot(
        "TAU", commonDataPtr->getPlasticTauDotPtr()));
    pipeline.push_back(new OpCalculateTensor2SymmetricFieldValues<SPACE_DIM>(
        "EP", commonDataPtr->getPlasticStrainPtr()));
    pipeline.push_back(new OpCalculateTensor2SymmetricFieldValuesDot<SPACE_DIM>(
        "EP", commonDataPtr->getPlasticStrainDotPtr()));
    pipeline.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        "U", commonDataPtr->mGradPtr));
    pipeline.push_back(new OpCalculateScalarFieldValues(
        "TAU", commonDataPtr->getPlasticTauPtr()));

    if (is_with_ALE && test_convection) {
      pipeline.push_back(
          new OpCalculateTensor2SymmetricFieldGradient<SPACE_DIM, SPACE_DIM>(
              "EP", commonDataPtr->plasticGradStrainPtr));
      pipeline.push_back(new OpCalculateScalarFieldGradient<SPACE_DIM>(
          "TAU", commonDataPtr->plasticGradTauPtr));
      pipeline.push_back(
          new OpCalculatePlasticConvRotatingFrame("TAU", commonDataPtr));
    }

    MoFEMFunctionReturn(0);
  };

  auto add_domain_stress_ops = [&](auto &pipeline, auto m_D_ptr) {
    MoFEMFunctionBegin;

    if (is_large_strains) {

      if (commonDataPtr->mGradPtr != commonHenckyDataPtr->matGradPtr)
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
      pipeline.push_back(new OpSymmetrizeTensor<SPACE_DIM>(
          "U", commonDataPtr->mGradPtr, commonDataPtr->mStrainPtr));
      pipeline.push_back(new OpPlasticStress("U", commonDataPtr, m_D_ptr, 1));
    }

    if (m_D_ptr != commonDataPtr->mDPtr_Axiator)
      pipeline.push_back(new OpCalculatePlasticSurface("U", commonDataPtr));

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
          "U", "EP", commonDataPtr, commonHenckyDataPtr, m_D_ptr));
    } else {
      pipeline.push_back(new OpKCauchy("U", "U", m_D_ptr));
      pipeline.push_back(new OpCalculatePlasticInternalForceLhs_dEP(
          "U", "EP", commonDataPtr, m_D_ptr));
    }

    pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_rhs_mechanical = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      const std::string block_name = "BODY_FORCE";
      if (it->getName().compare(0, block_name.size(), block_name) == 0) {
        std::vector<double> attr;
        CHKERR it->getAttributes(attr);
        if (attr.size() == 3) {
          bodyForces.push_back(
              FTensor::Tensor1<double, 3>{attr[0], attr[1], attr[2]});
        } else {
          SETERRQ1(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
                   "Should be three atributes in BODYFORCE blockset, but is %d",
                   attr.size());
        }
      }
    }

    auto get_body_force = [this](const double, const double, const double) {
      auto *pipeline_mng = mField.getInterface<PipelineManager>();
      FTensor::Index<'i', SPACE_DIM> i;
      FTensor::Tensor1<double, SPACE_DIM> t_source;
      t_source(i) = 0;
      auto fe_domain_rhs = pipeline_mng->getDomainRhsFE();
      const auto time = fe_domain_rhs->ts_t;
      // hardcoded gravity load
      for (auto &t_b : bodyForces)
        t_source(i) += (scale * t_b(i)) * time;
      return t_source;
    };

    pipeline.push_back(new OpBodyForce("U", get_body_force));

    // Calculate internal forece
    if (is_large_strains) {
      pipeline.push_back(new OpInternalForcePiola(
          "U", commonHenckyDataPtr->getMatFirstPiolaStress()));
    } else {
      pipeline.push_back(
          new OpInternalForceCauchy("U", commonDataPtr->mStressPtr));
    }

    pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_lhs_constrain = [&](auto &pipeline, auto m_D_ptr) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    if (is_large_strains) {
      pipeline.push_back(
          new OpHenckyTangent<SPACE_DIM>("U", commonHenckyDataPtr));
      pipeline.push_back(new OpCalculateContrainsLhs_LogStrain_dU(
          "TAU", "U", commonDataPtr, commonHenckyDataPtr, m_D_ptr));
      pipeline.push_back(new OpCalculatePlasticFlowLhs_LogStrain_dU(
          "EP", "U", commonDataPtr, commonHenckyDataPtr, m_D_ptr));
    } else {
      pipeline.push_back(
          new OpCalculatePlasticFlowLhs_dU("EP", "U", commonDataPtr, m_D_ptr));
      pipeline.push_back(
          new OpCalculateContrainsLhs_dU("TAU", "U", commonDataPtr, m_D_ptr));
    }

    pipeline.push_back(
        new OpCalculatePlasticFlowLhs_dEP("EP", "EP", commonDataPtr, m_D_ptr));

    if (is_with_ALE && test_convection) {
      pipeline.push_back(new OpCalculatePlasticFlowLhs_dEP_ALE(
          "EP", "EP", commonDataPtr, m_D_ptr,
          commonDataPtr->guidingVelocityPtr));
      pipeline.push_back(new OpCalculatePlasticFlowLhs_dTAU_ALE(
          "EP", "TAU", commonDataPtr, m_D_ptr,
          commonDataPtr->guidingVelocityPtr));
      pipeline.push_back(new OpCalculateConstrainsLhs_dTAU_ALE(
          "TAU", "TAU", commonDataPtr, m_D_ptr,
          commonDataPtr->guidingVelocityPtr));
    }

    pipeline.push_back(
        new OpCalculatePlasticFlowLhs_dTAU("EP", "TAU", commonDataPtr));
    pipeline.push_back(
        new OpCalculateContrainsLhs_dEP("TAU", "EP", commonDataPtr, m_D_ptr));
    pipeline.push_back(
        new OpCalculateContrainsLhs_dTAU("TAU", "TAU", commonDataPtr));

    pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_rhs_constrain = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    pipeline.push_back(new OpCalculatePlasticFlowRhs("EP", commonDataPtr));
    pipeline.push_back(new OpCalculateContrainsRhs("TAU", commonDataPtr));

    pipeline.push_back(new OpUnSetBc("U"));
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

    auto get_time = [&](double, double, double) {
      auto *pipeline_mng = mField.getInterface<PipelineManager>();
      auto &fe_domain_rhs = pipeline_mng->getBoundaryRhsFE();
      return fe_domain_rhs->ts_t;
    };

    auto get_time_scaled = [&](double, double, double) {
      auto *pipeline_mng = mField.getInterface<PipelineManager>();
      auto &fe_domain_rhs = pipeline_mng->getBoundaryRhsFE();
      // return fe_domain_rhs->ts_t * scale;
      return fe_domain_rhs->ts_t < 1.3 ? fe_domain_rhs->ts_t * scale
                                        : 1.3 * scale;
    };

    auto get_minus_time = [&](double, double, double) {
      auto *pipeline_mng = mField.getInterface<PipelineManager>();
      auto &fe_domain_rhs = pipeline_mng->getBoundaryRhsFE();
      return -fe_domain_rhs->ts_t;
    };

    auto time_scaled = [&](double, double, double) {
      auto *pipeline_mng = mField.getInterface<PipelineManager>();
      auto &fe_domain_rhs = pipeline_mng->getBoundaryRhsFE();
      return -fe_domain_rhs->ts_t;
    };

    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, 5, "FORCE") == 0) {
        Range force_edges;
        std::vector<double> attr_vec;
        CHKERR it->getMeshsetIdEntitiesByDimension(
            mField.get_moab(), SPACE_DIM - 1, force_edges, true);
        it->getAttributes(attr_vec);
        if (attr_vec.size() < SPACE_DIM)
          SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                  "Wrong size of boundary attributes vector. Set right block "
                  "size attributes.");
        auto force_vec_ptr = boost::make_shared<MatrixDouble>(SPACE_DIM, 1);
        std::copy(&attr_vec[0], &attr_vec[SPACE_DIM],
                  force_vec_ptr->data().begin());
        pipeline.push_back(
            new OpBoundaryVec("U", force_vec_ptr, get_time_scaled,
                              boost::make_shared<Range>(force_edges)));
      }
    }

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

  // Axiator
  CHKERR add_domain_base_ops(pipeline_mng->getOpDomainLhsPipeline());
  CHKERR add_domain_stress_ops(pipeline_mng->getOpDomainLhsPipeline(),
                               commonDataPtr->mDPtr_Deviator);
  CHKERR add_domain_ops_lhs_mechanical(pipeline_mng->getOpDomainLhsPipeline(),
                                       commonDataPtr->mDPtr_Deviator);
  CHKERR add_domain_ops_lhs_constrain(pipeline_mng->getOpDomainLhsPipeline(),
                                      commonDataPtr->mDPtr_Deviator);
  CHKERR add_boundary_ops_lhs_mechanical(
      pipeline_mng->getOpBoundaryLhsPipeline());

  CHKERR add_domain_base_ops(feAxiatorLhs->getOpPtrVector());
  CHKERR add_domain_stress_ops(feAxiatorLhs->getOpPtrVector(),
                               commonDataPtr->mDPtr_Axiator);
  CHKERR add_domain_ops_lhs_mechanical(feAxiatorLhs->getOpPtrVector(),
                                       commonDataPtr->mDPtr_Axiator);

  CHKERR add_domain_base_ops(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_domain_stress_ops(pipeline_mng->getOpDomainRhsPipeline(),
                               commonDataPtr->mDPtr_Deviator);

  CHKERR add_domain_ops_rhs_mechanical(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_domain_ops_rhs_constrain(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_boundary_ops_rhs_mechanical(
      pipeline_mng->getOpBoundaryRhsPipeline());

  CHKERR add_domain_base_ops(feAxiatorRhs->getOpPtrVector());
  CHKERR add_domain_stress_ops(feAxiatorRhs->getOpPtrVector(),
                               commonDataPtr->mDPtr_Axiator);
  CHKERR add_domain_ops_rhs_mechanical(feAxiatorRhs->getOpPtrVector());

  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule_deviator);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule_deviator);

  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule_bc);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule_bc);

  if (is_with_ALE) {
    CHKERR add_skeleton_base_ops(pipeline_mng->getOpSkeletonRhsPipeline());
    CHKERR add_skeleton_base_ops(pipeline_mng->getOpSkeletonLhsPipeline());
    if (test_penalty) {
      CHKERR add_skeleton_rhs_ops(pipeline_mng->getOpSkeletonRhsPipeline());
      CHKERR add_skeleton_lhs_ops(pipeline_mng->getOpSkeletonLhsPipeline());
    }
    CHKERR
    pipeline_mng->setSkeletonLhsIntegrationRule(integration_rule_bc);
    CHKERR
    pipeline_mng->setSkeletonRhsIntegrationRule(integration_rule_bc);
  }

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

  auto create_post_process_skeleton_element = [&]() {
    MoFEMFunctionBeginHot;
    if (!is_with_ALE)
      MoFEMFunctionReturnHot(0);
    // skeleton side element
    postProcFeSkeletonFe = boost::make_shared<PostProcEleSkeleton>(mField);
    postProcFeSkeletonFe->generateReferenceElementMesh();
    postProcFeSkeletonFe->addFieldValuesPostProc("U");
    if (!domainSideFe)
      MOFEM_LOG("WORLD", Sev::error) << "domainSideFe not initialised";
    
    auto &pipeline = postProcFeSkeletonFe->getOpPtrVector();
    pipeline.push_back(new OpCalculateVelocityOnSkeleton("U", commonDataPtr));
    pipeline.push_back(
        new OpCalculateJumpOnEPSkeleton("U", commonDataPtr, domainSideFe));
    pipeline.push_back(
        new OpCalculateJumpOnTAUSkeleton("U", commonDataPtr, domainSideFe));

    pipeline.push_back(new OpPostProcAleSkeleton(
        "U", postProcFeSkeletonFe->postProcMesh,
        postProcFeSkeletonFe->mapGaussPts, commonDataPtr));

    MoFEMFunctionReturnHot(0);
  };

  auto create_post_process_element = [&]() {
    MoFEMFunctionBegin;
    postProcFe = boost::make_shared<PostProcEle>(mField);
    postProcFe->generateReferenceElementMesh();
    if (SPACE_DIM == 2) {
      auto det_ptr = boost::make_shared<VectorDouble>();
      auto jac_ptr = boost::make_shared<MatrixDouble>();
      auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
      postProcFe->getOpPtrVector().push_back(
          new OpCalculateHOJacForFace(jac_ptr));
      postProcFe->getOpPtrVector().push_back(
          new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
      postProcFe->getOpPtrVector().push_back(
          new OpSetInvJacH1ForFace(inv_jac_ptr));
      postProcFe->getOpPtrVector().push_back(
          new OpSetInvJacL2ForFace(inv_jac_ptr));
    }

    postProcFe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
            "U", commonDataPtr->mGradPtr));
    postProcFe->getOpPtrVector().push_back(new OpCalculateScalarFieldValues(
        "TAU", commonDataPtr->getPlasticTauPtr()));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateTensor2SymmetricFieldValues<SPACE_DIM>(
            "EP", commonDataPtr->getPlasticStrainPtr()));

    if (is_large_strains) {

      if (commonDataPtr->mGradPtr != commonHenckyDataPtr->matGradPtr)
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                "Wrong pointer for grad");

      postProcFe->getOpPtrVector().push_back(
          new OpCalculateEigenVals<SPACE_DIM>("U", commonHenckyDataPtr));
      postProcFe->getOpPtrVector().push_back(
          new OpCalculateLogC<SPACE_DIM>("U", commonHenckyDataPtr));
      postProcFe->getOpPtrVector().push_back(
          new OpCalculateLogC_dC<SPACE_DIM>("U", commonHenckyDataPtr));
      postProcFe->getOpPtrVector().push_back(
          new OpCalculateHenckyPlasticStress<SPACE_DIM>(
              "U", commonHenckyDataPtr, commonDataPtr->mDPtr, scale));
      postProcFe->getOpPtrVector().push_back(
          new OpCalculatePiolaStress<SPACE_DIM>("U", commonHenckyDataPtr));
      postProcFe->getOpPtrVector().push_back(new OpPostProcHencky<SPACE_DIM>(
          "U", postProcFe->postProcMesh, postProcFe->mapGaussPts,
          commonHenckyDataPtr));

    } else {
      postProcFe->getOpPtrVector().push_back(new OpSymmetrizeTensor<SPACE_DIM>(
          "U", commonDataPtr->mGradPtr, commonDataPtr->mStrainPtr));
      postProcFe->getOpPtrVector().push_back(
          new OpPlasticStress("U", commonDataPtr, commonDataPtr->mDPtr, scale));
      postProcFe->getOpPtrVector().push_back(
          new Tutorial::OpPostProcElastic<SPACE_DIM>(
              "U", postProcFe->postProcMesh, postProcFe->mapGaussPts,
              commonDataPtr->mStrainPtr, commonDataPtr->mStressPtr));
    }

    postProcFe->getOpPtrVector().push_back(
        new OpCalculatePlasticSurface("U", commonDataPtr));
    postProcFe->getOpPtrVector().push_back(new OpPostProcPlastic(
        "U", postProcFe->postProcMesh, postProcFe->mapGaussPts, commonDataPtr));
    postProcFe->addFieldValuesPostProc("U");
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

    boost::shared_ptr<MonitorPostProcSkeleton> sk_monitor_ptr(
        new MonitorPostProcSkeleton(dm, postProcFeSkeletonFe));

    boost::shared_ptr<ForcesAndSourcesCore> null;
    CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                               monitor_ptr, null, null);
    if (is_with_ALE)
      CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                                 sk_monitor_ptr, null, null);
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

  boost::shared_ptr<FEMethod> null;
  CHKERR DMMoFEMTSSetIJacobian(dm, simple->getDomainFEName(), feAxiatorLhs,
                               null, null);
  CHKERR DMMoFEMTSSetIFunction(dm, simple->getDomainFEName(), feAxiatorRhs,
                               null, null);

  CHKERR create_post_process_element();
  CHKERR create_post_process_skeleton_element();

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
#ifndef NDEBUG
    if (m_field.get_comm_rank() == 1) {
      std::cout << "debug in parallel, attach the process and press any key "
                   "and ENTER to continue "
                << std::endl;
      string wait;
      cin >> wait;
    }
#endif

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
