/**
 * \file plastic.cpp
 * \example plastic.cpp
 *
 * Plane stress elastic problem
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
  using DomainEle = PipelineManager::FaceEle2D;
  using DomainEleOp = DomainEle::UserDataOperator;
  using BoundaryEle = PipelineManager::EdgeEle2D;
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
PetscBool is_dual_base = PETSC_TRUE;

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

#include <HenckyOps.hpp>
#include <PlasticOps.hpp>
#include <OpPostProcElastic.hpp>
#include <DualBase.hpp>

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

  MatrixDouble invJac;
  boost::shared_ptr<PlasticOps::CommonData> commonPlasticDataPtr;
  boost::shared_ptr<HenckyOps::CommonData> commonHenckyDataPtr;
  boost::shared_ptr<PostProcEle> postProcFe;
  boost::shared_ptr<DomainEle> reactionFe;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;

  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
  boost::shared_ptr<std::vector<unsigned char>> reactionMarker;

  struct BCs : boost::enable_shared_from_this<BCs> {
    Range bcEdges;
    std::vector<double> bcAttributes;
    std::vector<unsigned char> bcMarkers;
    inline auto getBcEdgesPtr() {
      return boost::shared_ptr<Range>(shared_from_this(), &bcEdges);
    }
    inline auto getBcMarkersPtr() {
      return boost::shared_ptr<std::vector<unsigned char>>(shared_from_this(),
                                                           &bcMarkers);
    }
  };

  std::vector<boost::shared_ptr<BCs>> bcVec;

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
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-dual_base", &is_dual_base,
                               PETSC_NULL);

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
    t_D(i, j, k, l) = 2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.) +
                      A * (bulk_modulus_K - (2. / 3.) * shear_modulus_G) *
                          t_kd(i, j) * t_kd(k, l);
    MoFEMFunctionReturn(0);
  };

  commonPlasticDataPtr = boost::make_shared<PlasticOps::CommonData>();
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  commonPlasticDataPtr->mDPtr = boost::make_shared<MatrixDouble>();
  commonPlasticDataPtr->mDPtr->resize(size_symm * size_symm, 1);

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

  auto prb_mng = mField.getInterface<ProblemsManager>();
  auto simple = mField.getInterface<Simple>();

  auto get_block_ents = [&](const std::string blockset_name) {
    Range remove_ents;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, blockset_name.length(), blockset_name) ==
          0) {
        CHKERR mField.get_moab().get_entities_by_handle(it->meshset,
                                                        remove_ents, true);
      }
    }
    return remove_ents;
  };

  auto get_adj_ents = [&](const Range &ents) {
    Range verts;
    CHKERR mField.get_moab().get_connectivity(ents, verts, true);
    if (SPACE_DIM == 3) {
      CHKERR mField.get_moab().get_adjacencies(ents, 1, false, verts,
                                               moab::Interface::UNION);
    }
    verts.merge(ents);
    CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(verts);
    return verts;
  };

  auto remove_dofs_on_ents = [&](const Range &&ents, const int lo,
                                 const int hi) {
    MoFEMFunctionBegin;
    CHKERR prb_mng->removeDofsOnEntities(simple->getProblemName(), "U", ents,
                                         lo, hi);
    MoFEMFunctionReturn(0);
  };

  auto mark_fix_dofs = [&](std::vector<unsigned char> &marked_u_dofs,
                           const auto lo, const auto hi) {
    return prb_mng->modifyMarkDofs(simple->getProblemName(), ROW, "U", lo, hi,
                                   ProblemsManager::MarkOP::OR, 1,
                                   marked_u_dofs);
  };

  auto fix_disp = [&](const std::string blockset_name, const bool fix_x,
                      const bool fix_y, const bool fix_z) {
    MoFEMFunctionBegin;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, blockset_name.length(), blockset_name) ==
          0) {

        bcVec.emplace_back(new BCs());

        CHKERR mField.get_moab().get_entities_by_handle(
            it->meshset, bcVec.back()->bcEdges, true);
        CHKERR it->getAttributes(bcVec.back()->bcAttributes);

        if (fix_x)
          CHKERR mark_fix_dofs(bcVec.back()->bcMarkers, 0, 0);
        if (fix_y)
          CHKERR mark_fix_dofs(bcVec.back()->bcMarkers, 1, 1);
        if (fix_z)
          CHKERR mark_fix_dofs(bcVec.back()->bcMarkers, 2, 2);

        CHKERR prb_mng->markDofs(
            simple->getProblemName(), ROW, ProblemsManager::AND,
            get_adj_ents(bcVec.back()->bcEdges), bcVec.back()->bcMarkers);
      }
    }
    MoFEMFunctionReturn(0);
  };

  auto mark_dofs_on_ents = [&](const Range &&ents) {
    std::vector<unsigned char> marked_ents;
    CHKERR prb_mng->markDofs(simple->getProblemName(), ROW, ProblemsManager::OR,
                             ents, marked_ents);

    return marked_ents;
  };

  CHKERR remove_dofs_on_ents(get_adj_ents(get_block_ents("REMOVE_X")), 0, 0);
  CHKERR remove_dofs_on_ents(get_adj_ents(get_block_ents("REMOVE_Y")), 1, 1);
  CHKERR remove_dofs_on_ents(get_adj_ents(get_block_ents("REMOVE_Z")), 2, 2);
  CHKERR remove_dofs_on_ents(get_adj_ents(get_block_ents("REMOVE_ALL")), 0, 3);

  CHKERR fix_disp("FIX_X", true, false, false);
  CHKERR fix_disp("FIX_Y", false, true, false);
  if (SPACE_DIM == 3)
    CHKERR fix_disp("FIX_Z", false, false, true);
  CHKERR fix_disp("FIX_ALL", true, true, true);

  boundaryMarker = boost::make_shared<std::vector<char unsigned>>();
  for (auto b : bcVec) {
    boundaryMarker->resize(b->bcMarkers.size(), 0);
    for (int i = 0; i != b->bcMarkers.size(); ++i) {
      (*boundaryMarker)[i] |= b->bcMarkers[i];
    }
  }
  if (boundaryMarker->empty()) {
    auto marker = mark_dofs_on_ents(Range());
    boundaryMarker = boost::make_shared<std::vector<unsigned char>>(
        marker.begin(), marker.end());
  }

  auto rec_marker = mark_dofs_on_ents(
      get_adj_ents(get_block_ents("REACTION")).subset_by_type(MBVERTEX));
  reactionMarker = boost::make_shared<std::vector<unsigned char>>(
      rec_marker.begin(), rec_marker.end());

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::OPs() {
  MoFEMFunctionBegin;
  auto pipeline_mng = mField.getInterface<PipelineManager>();
  auto simple = mField.getInterface<Simple>();

  auto add_domain_base_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    if (is_dual_base)
      pipeline.push_back(new OpScaleL2(L2));

    if (SPACE_DIM == 2) {
      pipeline.push_back(new OpCalculateInvJacForFace(invJac));
      pipeline.push_back(new OpSetInvJacH1ForFace(invJac));
    }

    pipeline.push_back(new OpCalculateScalarFieldValuesDot(
        "TAU", commonPlasticDataPtr->getPlasticTauDotPtr()));
    pipeline.push_back(new OpCalculateTensor2SymmetricFieldValuesDot<SPACE_DIM>(
        "EP", commonPlasticDataPtr->getPlasticStrainDotPtr()));

    MoFEMFunctionReturn(0);
  };

  auto add_domain_stress_ops = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    pipeline.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        "U", commonPlasticDataPtr->mGradPtr));
    pipeline.push_back(new OpCalculateTensor2SymmetricFieldValues<SPACE_DIM>(
        "EP", commonPlasticDataPtr->getPlasticStrainPtr()));
    pipeline.push_back(new OpCalculateScalarFieldValues(
        "TAU", commonPlasticDataPtr->getPlasticTauPtr()));

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
          "U", commonHenckyDataPtr));
      pipeline.push_back(
          new OpCalculatePiolaStress<SPACE_DIM>("U", commonHenckyDataPtr));

    } else {
      pipeline.push_back(
          new OpSymmetrizeTensor<SPACE_DIM>("U", commonPlasticDataPtr->mGradPtr,
                                            commonPlasticDataPtr->mStrainPtr));
      pipeline.push_back(new OpPlasticStress("U", commonPlasticDataPtr, 1));
    }

    pipeline.push_back(
        new OpCalculatePlasticSurface("U", commonPlasticDataPtr));

    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_lhs = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    if (is_dual_base)
      pipeline.push_back(
          new DualBaseOps::OpCalculateDualBase("TAU", commonPlasticDataPtr));

    if (is_large_strains) {
      pipeline.push_back(
          new OpHenckyTangent<SPACE_DIM>("U", commonHenckyDataPtr));
      pipeline.push_back(
          new OpKPiola("U", "U", commonHenckyDataPtr->getMatTangent()));
      pipeline.push_back(new OpCalculatePlasticInternalForceLhs_LogStrain_dEP(
          "U", "EP", commonPlasticDataPtr, commonHenckyDataPtr));
    } else {
      pipeline.push_back(new OpKCauchy("U", "U", commonPlasticDataPtr->mDPtr));
    }

    pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_rhs = [&](auto &pipeline) {
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

    if (is_dual_base) {
      pipeline.push_back(
          new DualBaseOps::OpCalculateDualBase("TAU", commonPlasticDataPtr));
      pipeline.push_back(
          new DualBaseOps::OpDualSwap("TAU", commonPlasticDataPtr));
    }

    if (is_dual_base) {
      pipeline.push_back(
          new DualBaseOps::OpDualSwap("TAU", commonPlasticDataPtr));
    }

    pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_lhs_constrain = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    if (is_dual_base) {
      pipeline.push_back(
          new DualBaseOps::OpCalculateDualBase("TAU", commonPlasticDataPtr));
    }

    if (is_large_strains) {
      pipeline.push_back(
          new OpHenckyTangent<SPACE_DIM>("U", commonHenckyDataPtr));
      pipeline.push_back(new OpCalculateContrainsLhs_LogStrain_dU(
          "TAU", "U", commonPlasticDataPtr, commonHenckyDataPtr));
      pipeline.push_back(new OpCalculatePlasticFlowLhs_LogStrain_dU(
          "EP", "U", commonPlasticDataPtr, commonHenckyDataPtr));
    } else {
      pipeline.push_back(
          new OpCalculatePlasticFlowLhs_dU("EP", "U", commonPlasticDataPtr));
      pipeline.push_back(
          new OpCalculateContrainsLhs_dU("TAU", "U", commonPlasticDataPtr));
    }

    pipeline.push_back(
        new OpCalculatePlasticFlowLhs_dEP("EP", "EP", commonPlasticDataPtr));
    pipeline.push_back(
        new OpCalculatePlasticFlowLhs_dTAU("EP", "TAU", commonPlasticDataPtr));
    pipeline.push_back(
        new OpCalculateContrainsLhs_dEP("TAU", "EP", commonPlasticDataPtr));
    pipeline.push_back(
        new OpCalculateContrainsLhs_dTAU("TAU", "TAU", commonPlasticDataPtr));

    if (is_dual_base) {
      pipeline.push_back(
          new DualBaseOps::OpDualSwap("TAU", commonPlasticDataPtr));
    }

    pipeline.push_back(new OpUnSetBc("U"));
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_rhs_constrain = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));

    if (is_dual_base) {
      pipeline.push_back(
          new DualBaseOps::OpCalculateDualBase("TAU", commonPlasticDataPtr));
      pipeline.push_back(
          new DualBaseOps::OpDualSwap("TAU", commonPlasticDataPtr));
    }

    pipeline.push_back(
        new OpCalculatePlasticFlowRhs("EP", commonPlasticDataPtr));
    pipeline.push_back(
        new OpCalculateContrainsRhs("TAU", commonPlasticDataPtr));

    if (is_dual_base) {
      pipeline.push_back(
          new DualBaseOps::OpDualSwap("TAU", commonPlasticDataPtr));
    }

    MoFEMFunctionReturn(0);
  };

  auto add_boundary_ops_lhs = [&](auto &pipeline) {
    MoFEMFunctionBegin;
    for (auto &bc : bcVec) {
      pipeline.push_back(new OpSetBc("U", false, bc->getBcMarkersPtr()));
      pipeline.push_back(new OpBoundaryMass(
          "U", "U", [](double, double, double) { return 1.; },
          bc->getBcEdgesPtr()));
      pipeline.push_back(new OpUnSetBc("U"));
    }
    MoFEMFunctionReturn(0);
  };

  auto add_boundary_ops_rhs = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    auto get_time = [&](double, double, double) {
      auto *pipeline_mng = mField.getInterface<PipelineManager>();
      auto &fe_domain_rhs = pipeline_mng->getDomainRhsFE();
      return fe_domain_rhs->ts_t;
    };

    auto get_time_scaled = [&](double, double, double) {
      auto *pipeline_mng = mField.getInterface<PipelineManager>();
      auto &fe_domain_rhs = pipeline_mng->getDomainRhsFE();
      return fe_domain_rhs->ts_t * scale;
    };

    auto get_minus_time = [&](double, double, double) {
      auto *pipeline_mng = mField.getInterface<PipelineManager>();
      auto &fe_domain_rhs = pipeline_mng->getDomainRhsFE();
      return -fe_domain_rhs->ts_t;
    };

    auto time_scaled = [&](double, double, double) {
      auto *pipeline_mng = mField.getInterface<PipelineManager>();
      auto &fe_domain_rhs = pipeline_mng->getDomainRhsFE();
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

    for (auto &bc : bcVec) {
      pipeline.push_back(new OpSetBc("U", false, bc->getBcMarkersPtr()));
      auto attr_vec = boost::make_shared<MatrixDouble>(SPACE_DIM, 1);
      attr_vec->clear();
      if (bc->bcAttributes.size() < SPACE_DIM)
        SETERRQ1(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                 "Wrong size of boundary attributes vector. Set right block "
                 "size attributes. Size of attributes %d",
                 bc->bcAttributes.size());
      std::copy(&bc->bcAttributes[0], &bc->bcAttributes[SPACE_DIM],
                attr_vec->data().begin());

      pipeline.push_back(
          new OpBoundaryVec("U", attr_vec, time_scaled, bc->getBcEdgesPtr()));
      pipeline.push_back(new OpBoundaryInternal(
          "U", u_mat_ptr, [](double, double, double) { return 1.; },
          bc->getBcEdgesPtr()));

      pipeline.push_back(new OpUnSetBc("U"));
    }

    MoFEMFunctionReturn(0);
  };

  CHKERR add_domain_base_ops(pipeline_mng->getOpDomainLhsPipeline());
  CHKERR add_domain_stress_ops(pipeline_mng->getOpDomainLhsPipeline());
  CHKERR add_domain_ops_lhs(pipeline_mng->getOpDomainLhsPipeline());
  CHKERR add_domain_ops_lhs_constrain(pipeline_mng->getOpDomainLhsPipeline());
  CHKERR add_boundary_ops_lhs(pipeline_mng->getOpBoundaryLhsPipeline());

  CHKERR add_domain_base_ops(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_domain_stress_ops(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_domain_ops_rhs(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_domain_ops_rhs_constrain(pipeline_mng->getOpDomainRhsPipeline());
  CHKERR add_boundary_ops_rhs(pipeline_mng->getOpBoundaryRhsPipeline());

  auto integration_rule_nc = [](int, int, int approx_order) { return -1; };

  auto set_gauss_rule_3d = [&](ForcesAndSourcesCore *fe_ptr, int, int,
                               int approx_order, int add) {
    MoFEMFunctionBegin;

    const int rule = 2 * (approx_order - 1) + add;

    if (rule < 0) {
      auto &gauss_pts = fe_ptr->gaussPts;
      gauss_pts.resize(4, 1);
      gauss_pts(0, 0) = gauss_pts(1, 0) = gauss_pts(2, 0) = 0.25;
      gauss_pts(3, 0) = 1;
      MoFEMFunctionReturnHot(0);
    }

    const auto order_num = IntRules::NCO::tetrahedron_nco_order_num(rule);
    MatrixDouble xyz(order_num, 3);
    VectorDouble w(order_num);
    IntRules::NCO::tetrahedron_nco_rule(rule, order_num, &*xyz.data().begin(),
                                        &*w.begin());

    double s = 0;
    auto &gauss_pts = fe_ptr->gaussPts;
    gauss_pts.resize(4, order_num);
    for (int gg = 0; gg != order_num; ++gg) {
      gauss_pts(3, gg) = w(gg);
      for (auto d : {0, 1, 2})
        gauss_pts(d, gg) = xyz(gg, d);
    }

    MoFEMFunctionReturn(0);
  };

  auto set_gauss_rule_2d = [&](ForcesAndSourcesCore *fe_ptr, int, int,
                               int approx_order, int add) {
    MoFEMFunctionBegin;

    const int rule = 2 * (approx_order - 1) + add;
    if (rule < 0) {
      auto &gauss_pts = fe_ptr->gaussPts;
      gauss_pts.resize(4, 1);
      gauss_pts(0, 0) = gauss_pts(1, 0) = gauss_pts(2, 0) = 0.25;
      gauss_pts(3, 0) = 1;
      MoFEMFunctionReturnHot(0);
    }
    const auto order_num = IntRules::NCO::triangle_nco_order_num(rule);
    MatrixDouble xyz(order_num, 2);
    VectorDouble w(order_num);
    IntRules::NCO::triangle_nco_rule(rule, order_num, &*xyz.data().begin(),
                                     &*w.begin());

    auto &gauss_pts = fe_ptr->gaussPts;
    gauss_pts.resize(3, order_num);
    for (int gg = 0; gg != order_num; ++gg) {
      gauss_pts(2, gg) = w(gg);
      for (auto d : {0, 1})
        gauss_pts(d, gg) = xyz(gg, d);
    }

    MoFEMFunctionReturn(0);
  };

  auto integration_rule_domain = [](int, int, int approx_order) {
    return 2 * approx_order;
  };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule_nc);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule_nc);

  if (SPACE_DIM == 3) {
    auto set = [&](ForcesAndSourcesCore *fe_ptr, int ro, int co, int ao) {
      return set_gauss_rule_3d(fe_ptr, ro, co, ao, 0);
    };
    boost::dynamic_pointer_cast<ForcesAndSourcesCore>(
        pipeline_mng->getDomainLhsFE())
        ->setRuleHook = set;
    boost::dynamic_pointer_cast<ForcesAndSourcesCore>(
        pipeline_mng->getDomainRhsFE())
        ->setRuleHook = set;
  } else {
    auto set = [&](ForcesAndSourcesCore *fe_ptr, int ro, int co, int ao) {
      return set_gauss_rule_2d(fe_ptr, ro, co, ao, 0);
    };
    boost::dynamic_pointer_cast<ForcesAndSourcesCore>(
        pipeline_mng->getDomainLhsFE())
        ->setRuleHook = set;
    boost::dynamic_pointer_cast<ForcesAndSourcesCore>(
        pipeline_mng->getDomainRhsFE())
        ->setRuleHook = set;
  }

  auto integration_rule_bc = [](int, int, int approx_order) {
    return 2 * approx_order;
  };
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule_bc);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule_bc);

  auto create_reaction_pipeline = [&](auto &pipeline) {
    MoFEMFunctionBegin;

    if (SPACE_DIM == 2) {
      pipeline.push_back(new OpCalculateInvJacForFace(invJac));
      pipeline.push_back(new OpSetInvJacH1ForFace(invJac));
    }

    if (is_dual_base)
      pipeline.push_back(new OpScaleL2(L2));

    pipeline.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
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
          "U", commonHenckyDataPtr, scale));
      pipeline.push_back(
          new OpCalculatePiolaStress<SPACE_DIM>("U", commonHenckyDataPtr));

    } else {
      pipeline.push_back(
          new OpSymmetrizeTensor<SPACE_DIM>("U", commonPlasticDataPtr->mGradPtr,
                                            commonPlasticDataPtr->mStrainPtr));
      pipeline.push_back(new OpPlasticStress("U", commonPlasticDataPtr, scale));
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
    MoFEMFunctionReturn(0);
  };

  reactionFe = boost::make_shared<DomainEle>(mField);
  reactionFe->getRuleHook = integration_rule_nc;
  if (SPACE_DIM == 3) {
    auto set = [&](ForcesAndSourcesCore *fe_ptr, int ro, int co, int ao) {
      return set_gauss_rule_3d(fe_ptr, ro, co, ao, 0);
    };
    reactionFe->setRuleHook = set;
  } else {
    auto set = [&](ForcesAndSourcesCore *fe_ptr, int ro, int co, int ao) {
      return set_gauss_rule_2d(fe_ptr, ro, co, ao, 0);
    };
    reactionFe->setRuleHook = set;
  }

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
    if (is_dual_base)
      postProcFe->getOpPtrVector().push_back(new OpScaleL2(L2));
    if (SPACE_DIM == 2) {
      postProcFe->getOpPtrVector().push_back(
          new OpCalculateInvJacForFace(invJac));
      postProcFe->getOpPtrVector().push_back(new OpSetInvJacH1ForFace(invJac));
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
              "U", commonHenckyDataPtr, scale));
      postProcFe->getOpPtrVector().push_back(
          new OpCalculatePiolaStress<SPACE_DIM>("U", commonHenckyDataPtr));
      postProcFe->getOpPtrVector().push_back(new OpPostProcHencky<SPACE_DIM>(
          "U", postProcFe->postProcMesh, postProcFe->mapGaussPts,
          commonHenckyDataPtr));

    } else {
      postProcFe->getOpPtrVector().push_back(
          new OpSymmetrizeTensor<SPACE_DIM>("U", commonPlasticDataPtr->mGradPtr,
                                            commonPlasticDataPtr->mStrainPtr));
      postProcFe->getOpPtrVector().push_back(
          new OpPlasticStress("U", commonPlasticDataPtr, scale));
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

  auto dm = simple->getDM();
  auto D = smartCreateDMVector(dm);
  CHKERR create_post_process_element();
  uXScatter = scatter_create(D, 0);
  uYScatter = scatter_create(D, 1);
  if (SPACE_DIM == 3)
    uZScatter = scatter_create(D, 2);

  auto solver = pipeline_mng->createTS();

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
