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
  using PostProcEle = PostProcFaceOnRefinedMesh;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = VolumeElementForcesAndSourcesCore;
  using DomainEleOp = DomainEle::UserDataOperator;
  using BoundaryEle = FaceElementForcesAndSourcesCore;
  using BoundaryEleOp = BoundaryEle::UserDataOperator;
  using PostProcEle = PostProcVolumeOnRefinedMesh;
};

constexpr int SPACE_DIM =
    EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh

constexpr EntityType boundary_ent = SPACE_DIM == 3 ? MBTRI : MBEDGE;
using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = ElementsAndOps<SPACE_DIM>::DomainEleOp;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = ElementsAndOps<SPACE_DIM>::BoundaryEleOp;
using PostProcEle = ElementsAndOps<SPACE_DIM>::PostProcEle;

using AssemblyDomainEleOp =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::OpBase;

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

double scale = 1.;

double young_modulus = 206913;
double poisson_ratio = 0.29;
double rho = 0;
double sigmaY = 450;
double H = 129;
double visH = 0;
double cn = 1;
double Qinf = 265;
double b_iso = 16.93;
int order = 2;

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
  boost::shared_ptr<PostProcEle> postProcFe;
  boost::shared_ptr<DomainEle> reactionFe;
  boost::shared_ptr<DomainEle> feAxiatorRhs;
  boost::shared_ptr<DomainEle> feAxiatorLhs;

  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;

  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
  boost::shared_ptr<std::vector<unsigned char>> reactionMarker;

  std::vector<FTensor::Tensor1<double, 3>> bodyForces;
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
  CHKERR simple->addDomainField("U", H1, base, SPACE_DIM);
  CHKERR simple->addDomainField("TAU", L2, base, 1);
  CHKERR simple->addDomainField("EP", L2, base, size_symm);
  CHKERR simple->addBoundaryField("U", H1, base, SPACE_DIM);
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

  commonPlasticDataPtr = boost::make_shared<PlasticOps::CommonData>();
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

  auto add_domain_base_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    if (SPACE_DIM == 2) {
      auto det_ptr = boost::make_shared<VectorDouble>();
      auto jac_ptr = boost::make_shared<MatrixDouble>();
      auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
      pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
      pipeline.push_back(new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
      pipeline.push_back(new OpSetInvJacH1ForFace(inv_jac_ptr));
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

    if (m_D_ptr != commonPlasticDataPtr->mDPtr_Axiator)
      pipeline.push_back(
          new OpCalculatePlasticSurface("U", commonPlasticDataPtr));

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
          new OpInternalForceCauchy("U", commonPlasticDataPtr->mStressPtr));
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
          "TAU", "U", commonPlasticDataPtr, commonHenckyDataPtr, m_D_ptr));
      pipeline.push_back(new OpCalculatePlasticFlowLhs_LogStrain_dU(
          "EP", "U", commonPlasticDataPtr, commonHenckyDataPtr, m_D_ptr));
    } else {
      pipeline.push_back(new OpCalculatePlasticFlowLhs_dU(
          "EP", "U", commonPlasticDataPtr, m_D_ptr));
      pipeline.push_back(new OpCalculateContrainsLhs_dU(
          "TAU", "U", commonPlasticDataPtr, m_D_ptr));
    }

    pipeline.push_back(new OpCalculatePlasticFlowLhs_dEP(
        "EP", "EP", commonPlasticDataPtr, m_D_ptr));
    pipeline.push_back(
        new OpCalculatePlasticFlowLhs_dTAU("EP", "TAU", commonPlasticDataPtr));
    pipeline.push_back(new OpCalculateContrainsLhs_dEP(
        "TAU", "EP", commonPlasticDataPtr, m_D_ptr));
    pipeline.push_back(
        new OpCalculateContrainsLhs_dTAU("TAU", "TAU", commonPlasticDataPtr));

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

    MoFEMFunctionReturn(0);
  };

  auto add_boundary_ops_lhs_mechanical = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    auto &bc_map = mField.getInterface<BcManager>()->getBcMapByBlockName();
    for (auto bc : bc_map) {
      if (bc_mng->checkBlock(bc, "FIX_")){
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
      return fe_domain_rhs->ts_t * scale;
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
      if (bc_mng->checkBlock(bc, "FIX_"))  {
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
                               commonPlasticDataPtr->mDPtr_Deviator);
  CHKERR add_domain_ops_lhs_mechanical(pipeline_mng->getOpDomainLhsPipeline(),
                                       commonPlasticDataPtr->mDPtr_Deviator);
  CHKERR add_domain_ops_lhs_constrain(pipeline_mng->getOpDomainLhsPipeline(),
                                      commonPlasticDataPtr->mDPtr_Deviator);
  CHKERR add_boundary_ops_lhs_mechanical(
      pipeline_mng->getOpBoundaryLhsPipeline());

  CHKERR add_domain_base_ops(feAxiatorLhs->getOpPtrVector());
  CHKERR add_domain_stress_ops(feAxiatorLhs->getOpPtrVector(),
                               commonPlasticDataPtr->mDPtr_Axiator);
  CHKERR add_domain_ops_lhs_mechanical(feAxiatorLhs->getOpPtrVector(),
                                       commonPlasticDataPtr->mDPtr_Axiator);

  CHKERR add_domain_base_ops(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_domain_stress_ops(pipeline_mng->getOpDomainRhsPipeline(),
                               commonPlasticDataPtr->mDPtr_Deviator);
  CHKERR add_domain_ops_rhs_mechanical(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_domain_ops_rhs_constrain(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_boundary_ops_rhs_mechanical(
      pipeline_mng->getOpBoundaryRhsPipeline());

  CHKERR add_domain_base_ops(feAxiatorRhs->getOpPtrVector());
  CHKERR add_domain_stress_ops(feAxiatorRhs->getOpPtrVector(),
                               commonPlasticDataPtr->mDPtr_Axiator);
  CHKERR add_domain_ops_rhs_mechanical(feAxiatorRhs->getOpPtrVector());
  
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule_deviator);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule_deviator);

  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule_bc);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule_bc);

  auto create_reaction_pipeline = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    if (reactionMarker) {

      if (SPACE_DIM == 2) {
        auto det_ptr = boost::make_shared<VectorDouble>();
        auto jac_ptr = boost::make_shared<MatrixDouble>();
        auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
        pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
        pipeline.push_back(
            new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
        pipeline.push_back(new OpSetInvJacH1ForFace(inv_jac_ptr));
      }

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
        pipeline.push_back(new OpPlasticStress("U", commonPlasticDataPtr,
                                               commonPlasticDataPtr->mDPtr, 1));
      }

      pipeline.push_back(new OpSetBc("U", false, reactionMarker));
      // Calculate internal forece
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
    postProcFe->generateReferenceElementMesh();
    if (SPACE_DIM == 2) {
      auto det_ptr = boost::make_shared<VectorDouble>();
      auto jac_ptr = boost::make_shared<MatrixDouble>();
      auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
      postProcFe->getOpPtrVector().push_back(
          new OpCalculateHOJacForFace(jac_ptr));
      postProcFe->getOpPtrVector().push_back(
          new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
      postProcFe->getOpPtrVector().push_back(new OpSetInvJacH1ForFace(inv_jac_ptr));
    }

    postProcFe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
            "U", commonPlasticDataPtr->mGradPtr));
    postProcFe->getOpPtrVector().push_back(new OpCalculateScalarFieldValues(
        "TAU", commonPlasticDataPtr->getPlasticTauPtr()));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateTensor2SymmetricFieldValues<SPACE_DIM>(
            "EP", commonPlasticDataPtr->getPlasticStrainPtr()));

    if (is_large_strains) {

      if (commonPlasticDataPtr->mGradPtr != commonHenckyDataPtr->matGradPtr)
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
              "U", commonHenckyDataPtr, commonPlasticDataPtr->mDPtr, scale));
      postProcFe->getOpPtrVector().push_back(
          new OpCalculatePiolaStress<SPACE_DIM>("U", commonHenckyDataPtr));
      postProcFe->getOpPtrVector().push_back(new OpPostProcHencky<SPACE_DIM>(
          "U", postProcFe->postProcMesh, postProcFe->mapGaussPts,
          commonHenckyDataPtr));

    } else {
      postProcFe->getOpPtrVector().push_back(
          new OpSymmetrizeTensor<SPACE_DIM>("U", commonPlasticDataPtr->mGradPtr,
                                            commonPlasticDataPtr->mStrainPtr));
      postProcFe->getOpPtrVector().push_back(new OpPlasticStress(
          "U", commonPlasticDataPtr, commonPlasticDataPtr->mDPtr, scale));
      postProcFe->getOpPtrVector().push_back(
          new Tutorial::OpPostProcElastic<SPACE_DIM>(
              "U", postProcFe->postProcMesh, postProcFe->mapGaussPts,
              commonPlasticDataPtr->mStrainPtr,
              commonPlasticDataPtr->mStressPtr));
    }

    postProcFe->getOpPtrVector().push_back(
        new OpCalculatePlasticSurface("U", commonPlasticDataPtr));
    postProcFe->getOpPtrVector().push_back(
        new OpPostProcPlastic("U", postProcFe->postProcMesh,
                              postProcFe->mapGaussPts, commonPlasticDataPtr));
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

  boost::shared_ptr<FEMethod> null;
  CHKERR DMMoFEMTSSetIJacobian(dm, simple->getDomainFEName(), feAxiatorLhs,
                               null, null);
  CHKERR DMMoFEMTSSetIFunction(dm, simple->getDomainFEName(), feAxiatorRhs,
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
