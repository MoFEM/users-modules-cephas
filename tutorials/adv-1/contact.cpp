/**
 * \file contact.cpp
 * \example contact.cpp
 *
 * Example of contact problem
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

//! [Operators used for contact]
using OpMixDivULhs = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMixDivTimesVec<SPACE_DIM>;
using OpLambdaGraULhs = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMixTensorTimesGrad<SPACE_DIM>;
using OpMixDivURhs = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpMixDivTimesU<SPACE_DIM, SPACE_DIM>;
using OpMixLambdaGradURhs = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpMixTensorTimesGradU<SPACE_DIM>;
using OpMixUTimesDivLambdaRhs = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpMixVecTimesDivLambda<SPACE_DIM>;
using OpMixUTimesLambdaRhs = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;
using OpSpringLhs = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, SPACE_DIM>;
using OpSpringRhs = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 1>;
//! [Operators used for contact]

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

// Only used with Hencky/nonlinear material
using OpKPiola = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradTensorGrad<1, SPACE_DIM, SPACE_DIM, 1>;
using OpInternalForcePiola = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;

constexpr bool is_quasi_static = true;
constexpr bool is_large_strains = true;

constexpr int order = 2;
constexpr double young_modulus = 100;
constexpr double poisson_ratio = 0.25;
constexpr double rho = 0;
constexpr double cn = 0.01;
constexpr double spring_stiffness = 0.1;

#include <OpPostProcElastic.hpp>
#include <ContactOps.hpp>
#include <HenckyOps.hpp>
#include <PostProcContact.hpp>
using namespace ContactOps;
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
  MoFEMErrorCode postProcess();
  MoFEMErrorCode checkResults();

  MatrixDouble invJac, jAc;
  boost::shared_ptr<ContactOps::CommonData> commonDataPtr;
  boost::shared_ptr<PostProcEle> postProcFe;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;

  template <int DIM> Range getEntsOnMeshSkin();
};

//! [Run problem]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR setupProblem();
  CHKERR createCommonData();
  CHKERR bC();
  CHKERR OPs();
  CHKERR tsSolve();
  CHKERR postProcess();
  CHKERR checkResults();
  MoFEMFunctionReturn(0);
}
//! [Run problem]

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  // Add field
  CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, SPACE_DIM);
  CHKERR simple->addBoundaryField("U", H1, AINSWORTH_LEGENDRE_BASE, SPACE_DIM);

  if (SPACE_DIM == 2) {
    CHKERR simple->addDomainField("SIGMA", HCURL, DEMKOWICZ_JACOBI_BASE, 2);
    CHKERR simple->addBoundaryField("SIGMA", HCURL, DEMKOWICZ_JACOBI_BASE, 2);
  } else {
    CHKERR simple->addDomainField("SIGMA", HDIV, DEMKOWICZ_JACOBI_BASE, 3);
    CHKERR simple->addBoundaryField("SIGMA", HDIV, DEMKOWICZ_JACOBI_BASE, 3);
  }

  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setFieldOrder("SIGMA", 0);

  auto skin_edges = getEntsOnMeshSkin<SPACE_DIM>();

  // filter not owned entities, those are not on boundary
  Range boundary_ents;
  ParallelComm *pcomm =
      ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
  if (pcomm == NULL) {
    pcomm = new ParallelComm(&mField.get_moab(), mField.get_comm());
  }

  CHKERR pcomm->filter_pstatus(skin_edges, PSTATUS_SHARED | PSTATUS_MULTISHARED,
                               PSTATUS_NOT, -1, &boundary_ents);

  CHKERR simple->setFieldOrder("SIGMA", order - 1, &boundary_ents);
  // Range adj_edges;
  // CHKERR mField.get_moab().get_adjacencies(boundary_ents, 1, false, adj_edges,
  //                                          moab::Interface::UNION);
  // adj_edges.merge(boundary_ents);
  // CHKERR simple->setFieldOrder("U", order, &adj_edges);

  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;

  auto set_matrial_stiffness = [&]() {
    MoFEMFunctionBegin;
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    FTensor::Index<'l', SPACE_DIM> l;
    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
    constexpr double bulk_modulus_K =
        young_modulus / (3 * (1 - 2 * poisson_ratio));
    constexpr double shear_modulus_G =
        young_modulus / (2 * (1 + poisson_ratio));
    constexpr double A =
        (SPACE_DIM == 2) ? 2 * shear_modulus_G /
                               (bulk_modulus_K + (4. / 3.) * shear_modulus_G)
                         : 1;
    auto t_D =
        getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);
    t_D(i, j, k, l) = 2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.) +
                      A * (bulk_modulus_K - (2. / 3.) * shear_modulus_G) *
                          t_kd(i, j) * t_kd(k, l);
    MoFEMFunctionReturn(0);
  };

  commonDataPtr = boost::make_shared<ContactOps::CommonData>();

  commonDataPtr->mGradPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mStrainPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mStressPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->contactStressPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->contactStressDivergencePtr =
      boost::make_shared<MatrixDouble>();
  commonDataPtr->contactTractionPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->contactDispPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->curlContactStressPtr = boost::make_shared<MatrixDouble>();

  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;

  commonDataPtr->mDPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mDPtr->resize(size_symm * size_symm, 1);

  jAc.resize(SPACE_DIM, SPACE_DIM, false);
  invJac.resize(SPACE_DIM, SPACE_DIM, false);

  CHKERR set_matrial_stiffness();
  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Boundary condition]
MoFEMErrorCode Example::bC() {
  MoFEMFunctionBegin;

  auto fix_disp = [&](const std::string blockset_name) {
    Range fix_ents;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, blockset_name.length(), blockset_name) ==
          0) {
        CHKERR mField.get_moab().get_entities_by_handle(it->meshset, fix_ents,
                                                        true);
      }
    }
    return fix_ents;
  };

  auto remove_ents = [&](const Range &&ents, const int lo, const int hi) {
    auto prb_mng = mField.getInterface<ProblemsManager>();
    auto simple = mField.getInterface<Simple>();
    MoFEMFunctionBegin;
    Range verts;
    CHKERR mField.get_moab().get_connectivity(ents, verts, true);
    verts.merge(ents);
    if (SPACE_DIM == 3) {
      Range adj;
      CHKERR mField.get_moab().get_adjacencies(ents, 1, false, adj,
                                               moab::Interface::UNION);
      verts.merge(adj);
    };
    CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(verts);
    CHKERR prb_mng->removeDofsOnEntities(simple->getProblemName(), "U", verts,
                                         lo, hi);
    MoFEMFunctionReturn(0);
  };

  Range boundary_ents;
  boundary_ents.merge(fix_disp("FIX_X"));
  boundary_ents.merge(fix_disp("FIX_Y"));
  boundary_ents.merge(fix_disp("FIX_Z"));
  boundary_ents.merge(fix_disp("FIX_ALL"));
  CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(boundary_ents);
  CHKERR mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
      mField.getInterface<Simple>()->getProblemName(), "SIGMA", boundary_ents,
      0, 2);

  CHKERR remove_ents(fix_disp("FIX_X"), 0, 0);
  CHKERR remove_ents(fix_disp("FIX_Y"), 1, 1);
  CHKERR remove_ents(fix_disp("FIX_Z"), 2, 2);
  CHKERR remove_ents(fix_disp("FIX_ALL"), 0, 3);

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::OPs() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  auto add_domain_base_ops = [&](auto &pipeline) {
    if (SPACE_DIM == 2) {
      pipeline.push_back(new OpCalculateJacForFace(jAc));
      pipeline.push_back(new OpCalculateInvJacForFace(invJac));
      pipeline.push_back(new OpSetInvJacH1ForFace(invJac));
      pipeline.push_back(new OpMakeHdivFromHcurl());
      pipeline.push_back(new OpSetContravariantPiolaTransformFace(jAc));
      pipeline.push_back(new OpSetInvJacHcurlFace(invJac));
    }
  };

  auto henky_common_data_ptr = boost::make_shared<HenckyOps::CommonData>();
  henky_common_data_ptr->matGradPtr = commonDataPtr->mGradPtr;
  henky_common_data_ptr->matDPtr = commonDataPtr->mDPtr;

  auto add_domain_ops_lhs = [&](auto &pipeline) {
    if (is_large_strains) {
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
              "U", commonDataPtr->mGradPtr));
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculateEigenVals<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculateLogC<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculateLogC_dC<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculateHenckyStress<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpCalculatePiolaStress<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpHenckyTangent<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpKPiola("U", "U", henky_common_data_ptr->getMatTangent()));
    } else {
      pipeline.push_back(new OpKCauchy("U", "U", commonDataPtr->mDPtr));
    }

    if (!is_quasi_static) {
      // Get pointer to U_tt shift in domain element
      auto get_rho = [this](const double, const double, const double) {
        auto *pipeline_mng = mField.getInterface<PipelineManager>();
        auto &fe_domain_lhs = pipeline_mng->getDomainLhsFE();
        return rho * fe_domain_lhs->ts_aa;
      };
      pipeline_mng->getOpDomainLhsPipeline().push_back(
          new OpMass("U", "U", get_rho));
    }

    pipeline.push_back(new OpMixDivULhs("SIGMA", "U", 1, true));
    pipeline.push_back(new OpLambdaGraULhs("SIGMA", "U", 1, true));
  };

  auto add_domain_ops_rhs = [&](auto &pipeline) {
    auto get_body_force = [this](const double, const double, const double) {
      auto *pipeline_mng = mField.getInterface<PipelineManager>();
      FTensor::Index<'i', SPACE_DIM> i;
      FTensor::Tensor1<double, SPACE_DIM> t_source;
      auto fe_domain_rhs = pipeline_mng->getDomainRhsFE();
      const auto time = fe_domain_rhs->ts_t;
      
      // hardcoded gravity load
      t_source(i) = 0;
      t_source(1) = 1.0 * time;
      return t_source;
    };

    pipeline.push_back(new OpBodyForce("U", get_body_force));
    pipeline.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        "U", commonDataPtr->mGradPtr));

    if (is_large_strains) {
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateEigenVals<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateLogC<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateLogC_dC<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateHenckyStress<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculatePiolaStress<SPACE_DIM>("U", henky_common_data_ptr));
      pipeline_mng->getOpDomainRhsPipeline().push_back(new OpInternalForcePiola(
          "U", henky_common_data_ptr->getMatFirstPiolaStress()));
    } else {
      pipeline.push_back(new OpSymmetrizeTensor<SPACE_DIM>(
          "U", commonDataPtr->mGradPtr, commonDataPtr->mStrainPtr));
      pipeline.push_back(new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
          "U", commonDataPtr->mStrainPtr, commonDataPtr->mStressPtr,
          commonDataPtr->mDPtr));
      pipeline.push_back(
          new OpInternalForceCauchy("U", commonDataPtr->mStressPtr));
    }

    pipeline.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>(
        "U", commonDataPtr->contactDispPtr));

    pipeline.push_back(new OpCalculateHVecTensorField<SPACE_DIM, SPACE_DIM>(
        "SIGMA", commonDataPtr->contactStressPtr));
    pipeline.push_back(
        new OpCalculateHVecTensorDivergence<SPACE_DIM, SPACE_DIM>(
            "SIGMA", commonDataPtr->contactStressDivergencePtr));

    pipeline.push_back(
        new OpMixDivURhs("SIGMA", commonDataPtr->contactDispPtr, 1));
    pipeline.push_back(
        new OpMixLambdaGradURhs("SIGMA", commonDataPtr->mGradPtr));
    pipeline.push_back(new OpMixUTimesDivLambdaRhs(
        "U", commonDataPtr->contactStressDivergencePtr));
    pipeline.push_back(
        new OpMixUTimesLambdaRhs("U", commonDataPtr->contactStressPtr));

    // only in case of dynamics
    if (!is_quasi_static) {
      auto mat_acceleration = boost::make_shared<MatrixDouble>();
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateVectorFieldValuesDotDot<SPACE_DIM>("U",
                                                            mat_acceleration));
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpInertiaForce("U", mat_acceleration, rho));
    }
  };

  auto add_boundary_base_ops = [&](auto &pipeline) {
    if (SPACE_DIM == 2)
      pipeline.push_back(new OpSetContravariantPiolaTransformOnEdge());
    pipeline.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>(
        "U", commonDataPtr->contactDispPtr));
    pipeline.push_back(new OpCalculateHVecTensorTrace<SPACE_DIM, BoundaryEleOp>(
        "SIGMA", commonDataPtr->contactTractionPtr));
  };

  auto add_boundary_ops_lhs = [&](auto &pipeline) {
    pipeline.push_back(
        new OpConstrainBoundaryLhs_dU("SIGMA", "U", commonDataPtr));
    pipeline.push_back(
        new OpConstrainBoundaryLhs_dTraction("SIGMA", "SIGMA", commonDataPtr));
    pipeline.push_back(new OpSpringLhs(
        "U", "U",

        [this](double, double, double) { return spring_stiffness; }

        ));
  };

  auto add_boundary_ops_rhs = [&](auto &pipeline) {
    pipeline.push_back(new OpConstrainBoundaryRhs("SIGMA", commonDataPtr));
    pipeline.push_back(
        new OpSpringRhs("U", commonDataPtr->contactDispPtr, spring_stiffness));
  };

  add_domain_base_ops(pipeline_mng->getOpDomainLhsPipeline());
  add_domain_base_ops(pipeline_mng->getOpDomainRhsPipeline());
  add_domain_ops_lhs(pipeline_mng->getOpDomainLhsPipeline());
  add_domain_ops_rhs(pipeline_mng->getOpDomainRhsPipeline());

  add_boundary_base_ops(pipeline_mng->getOpBoundaryLhsPipeline());
  add_boundary_base_ops(pipeline_mng->getOpBoundaryRhsPipeline());
  add_boundary_ops_lhs(pipeline_mng->getOpBoundaryLhsPipeline());
  add_boundary_ops_rhs(pipeline_mng->getOpBoundaryRhsPipeline());

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * order + 1;
  };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule);

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
        new Monitor(dm, commonDataPtr, uXScatter, uYScatter, uZScatter));
    boost::shared_ptr<ForcesAndSourcesCore> null;
    CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                               monitor_ptr, null, null);
    MoFEMFunctionReturn(0);
  };

  auto dm = simple->getDM();
  auto D = smartCreateDMVector(dm);
  uXScatter = scatter_create(D, 0);
  uYScatter = scatter_create(D, 1);
  if (SPACE_DIM == 3)
    uZScatter = scatter_create(D, 2);

  if (is_quasi_static) {
    auto solver = pipeline_mng->createTS();
    auto D = smartCreateDMVector(dm);
    CHKERR set_section_monitor(solver);
    CHKERR set_time_monitor(dm, solver);
    CHKERR TSSetSolution(solver, D);
    CHKERR TSSetFromOptions(solver);
    CHKERR TSSetUp(solver);
    CHKERR TSSolve(solver, NULL);
  } else {
    auto solver = pipeline_mng->createTS2();
    auto dm = simple->getDM();
    auto D = smartCreateDMVector(dm);
    auto DD = smartVectorDuplicate(D);
    CHKERR set_section_monitor(solver);
    CHKERR set_time_monitor(dm, solver);
    CHKERR TS2SetSolution(solver, D, DD);
    CHKERR TSSetFromOptions(solver);
    CHKERR TSSetUp(solver);
    CHKERR TSSolve(solver, NULL);
  }

  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::postProcess() { return 0; }
//! [Postprocess results]

//! [Check]
MoFEMErrorCode Example::checkResults() { return 0; }
//! [Check]

template <int DIM> Range Example::getEntsOnMeshSkin() {
  Range faces;
  EntityType type = MBTRI;
  if (DIM == 3)
    type = MBTET;
  CHKERR mField.get_moab().get_entities_by_type(0, type, faces);
  Skinner skin(&mField.get_moab());
  Range skin_ents;
  CHKERR skin.find_skin(0, faces, false, skin_ents);
  return skin_ents;
};

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
    CHKERR simple->loadFile("");
    //! [Load mesh]

    //! [Example]
    Example ex(m_field);
    CHKERR ex.runProblem();
    //! [Example]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
