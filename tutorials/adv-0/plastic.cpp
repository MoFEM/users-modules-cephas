/**
 * \file lesson7_plastic.cpp
 * \example lesson7_plastic.cpp
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

struct OpEdgeForceRhs : BoundaryEleOp {

  Range forceEdges;
  VectorDouble forceVec;

  OpEdgeForceRhs(const std::string field_name, const Range &force_edges,
                 const VectorDouble &force_vec);

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);
};

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

// Only used with Henky/nonlinear material
using OpKPiola = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradTensorGrad<1, SPACE_DIM, SPACE_DIM, 1>;
using OpInternalForcePiola = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;

constexpr double young_modulus = 1e1;
constexpr double poisson_ratio = 0.25;
constexpr double sigmaY = 1;
constexpr double H = 1e-2;
constexpr double cn = H;
constexpr int order = 2;

#include <PlasticOps.hpp>
#include <OpPostProcElastic.hpp>
#include <HenkyOps.hpp>

using namespace PlasticOps;
using namespace HenkyOps;
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

  MatrixDouble invJac;
  boost::shared_ptr<PlasticOps::CommonData> commonDataPtr;
  boost::shared_ptr<PostProcFaceOnRefinedMesh> postProcFe;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
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
  CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, 2);
  CHKERR simple->addDomainField("TAU", L2, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple->addDomainField("EP", L2, AINSWORTH_LEGENDRE_BASE, 3);
  CHKERR simple->addBoundaryField("U", H1, AINSWORTH_LEGENDRE_BASE, 2);
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

  commonDataPtr = boost::make_shared<PlasticOps::CommonData>();

  commonDataPtr->mGradPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mStrainPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mStressPtr = boost::make_shared<MatrixDouble>();

  commonDataPtr->plasticSurfacePtr = boost::make_shared<VectorDouble>();
  commonDataPtr->plasticFlowPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->plasticTauPtr = boost::make_shared<VectorDouble>();
  commonDataPtr->plasticTauDotPtr = boost::make_shared<VectorDouble>();
  commonDataPtr->plasticStrainPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->plasticStrainDotPtr = boost::make_shared<MatrixDouble>();

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

  auto remove_ents = [&](const Range &&ents, const bool fix_x,
                         const bool fix_y) {
    auto prb_mng = mField.getInterface<ProblemsManager>();
    auto simple = mField.getInterface<Simple>();
    MoFEMFunctionBegin;
    Range verts;
    CHKERR mField.get_moab().get_connectivity(ents, verts, true);
    verts.merge(ents);
    const int lo_coeff = fix_x ? 0 : 1;
    const int hi_coeff = fix_y ? 1 : 0;
    CHKERR prb_mng->removeDofsOnEntities(simple->getProblemName(), "U", verts,
                                         lo_coeff, hi_coeff);
    MoFEMFunctionReturn(0);
  };

  CHKERR remove_ents(fix_disp("FIX_X"), true, false);
  CHKERR remove_ents(fix_disp("FIX_Y"), false, true);
  CHKERR remove_ents(fix_disp("FIX_ALL"), true, true);

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::OPs() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();

  auto add_domain_base_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpCalculateInvJacForFace(invJac));
    pipeline.push_back(new OpSetInvJacH1ForFace(invJac));

    pipeline.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        "U", commonDataPtr->mGradPtr));
    pipeline.push_back(new OpSymmetrizeTensor<SPACE_DIM>(
        "U", commonDataPtr->mGradPtr, commonDataPtr->mStrainPtr));

    pipeline.push_back(new OpCalculateScalarFieldValues(
        "TAU", commonDataPtr->plasticTauPtr, MBTRI));
    pipeline.push_back(new OpCalculateScalarFieldValuesDot(
        "TAU", commonDataPtr->plasticTauDotPtr, MBTRI));

    pipeline.push_back(new OpCalculateTensor2SymmetricFieldValues<2>(
        "EP", commonDataPtr->plasticStrainPtr, MBTRI));
    pipeline.push_back(new OpCalculateTensor2SymmetricFieldValuesDot<2>(
        "EP", commonDataPtr->plasticStrainDotPtr, MBTRI));

    pipeline.push_back(new OpPlasticStress("U", commonDataPtr));
    pipeline.push_back(new OpCalculatePlasticSurface("U", commonDataPtr));
  };

  auto add_domain_ops_lhs = [&](auto &pipeline) {
    pipeline.push_back(new OpKCauchy("U", "U", commonDataPtr->mDPtr));

    pipeline.push_back(
        new OpCalculatePlasticInternalForceLhs_dEP("U", "EP", commonDataPtr));

    pipeline.push_back(
        new OpCalculatePlasticFlowLhs_dU("EP", "U", commonDataPtr));
    pipeline.push_back(
        new OpCalculatePlasticFlowLhs_dEP("EP", "EP", commonDataPtr));
    pipeline.push_back(
        new OpCalculatePlasticFlowLhs_dTAU("EP", "TAU", commonDataPtr));

    pipeline.push_back(
        new OpCalculateContrainsLhs_dU("TAU", "U", commonDataPtr));
    pipeline.push_back(
        new OpCalculateContrainsLhs_dEP("TAU", "EP", commonDataPtr));
    pipeline.push_back(
        new OpCalculateContrainsLhs_dTAU("TAU", "TAU", commonDataPtr));
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

    // Calculate internal forece
    pipeline.push_back(new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
        "U", commonDataPtr->mStrainPtr, commonDataPtr->mStressPtr,
        commonDataPtr->mDPtr));
    pipeline.push_back(
        new OpInternalForceCauchy("U", commonDataPtr->mStressPtr));

    pipeline.push_back(new OpCalculatePlasticFlowRhs("EP", commonDataPtr));
    pipeline.push_back(new OpCalculateContrainsRhs("TAU", commonDataPtr));
  };

  auto add_boundary_ops_rhs = [&](auto &pipeline) {
    MoFEMFunctionBeginHot;

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, 5, "FORCE") == 0) {
        Range my_edges;
        std::vector<double> force_vec;
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
                                                   my_edges, true);
        it->getAttributes(force_vec);
        pipeline.push_back(new OpEdgeForceRhs("U", my_edges, force_vec));
      }
    }

    MoFEMFunctionReturnHot(0);
  };

  add_domain_base_ops(pipeline_mng->getOpDomainLhsPipeline());
  add_domain_ops_lhs(pipeline_mng->getOpDomainLhsPipeline());
  add_domain_base_ops(pipeline_mng->getOpDomainRhsPipeline());
  add_domain_ops_rhs(pipeline_mng->getOpDomainRhsPipeline());
  add_boundary_ops_rhs(pipeline_mng->getOpBoundaryRhsPipeline());

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * approx_order;
  };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::tsSolve() {
  MoFEMFunctionBegin;

  Simple *simple = mField.getInterface<Simple>();
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  ISManager *is_manager = mField.getInterface<ISManager>();

  auto solver = pipeline_mng->createTS();

  auto dm = simple->getDM();
  auto D = smartCreateDMVector(dm);

  CHKERR TSSetSolution(solver, D);
  CHKERR TSSetFromOptions(solver);
  CHKERR TSSetUp(solver);

  auto set_section_monitor = [&]() {
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
    postProcFe = boost::make_shared<PostProcFaceOnRefinedMesh>(mField);
    postProcFe->generateReferenceElementMesh();

    postProcFe->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(invJac));
    postProcFe->getOpPtrVector().push_back(new OpSetInvJacH1ForFace(invJac));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<2, 2>("U", commonDataPtr->mGradPtr));
    postProcFe->getOpPtrVector().push_back(new OpSymmetrizeTensor<SPACE_DIM>(
        "U", commonDataPtr->mGradPtr, commonDataPtr->mStrainPtr));
    postProcFe->getOpPtrVector().push_back(
        new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
            "U", commonDataPtr->mStrainPtr, commonDataPtr->mStressPtr,
            commonDataPtr->mDPtr));

    postProcFe->getOpPtrVector().push_back(new OpCalculateScalarFieldValues(
        "TAU", commonDataPtr->plasticTauPtr, MBTRI));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateTensor2SymmetricFieldValues<2>(
            "EP", commonDataPtr->plasticStrainPtr, MBTRI));
    postProcFe->getOpPtrVector().push_back(
        new OpPlasticStress("U", commonDataPtr));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculatePlasticSurface("U", commonDataPtr));

    postProcFe->getOpPtrVector().push_back(
        new Tutorial::OpPostProcElastic<SPACE_DIM>(
            "U", postProcFe->postProcMesh, postProcFe->mapGaussPts,
            commonDataPtr->mStrainPtr, commonDataPtr->mStressPtr));

    postProcFe->getOpPtrVector().push_back(new OpPostProcPlastic(
        "U", postProcFe->postProcMesh, postProcFe->mapGaussPts, commonDataPtr));
    postProcFe->addFieldValuesPostProc("U");
    MoFEMFunctionReturn(0);
  };

  auto scatter_create = [&](auto coeff) {
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

  auto set_time_monitor = [&]() {
    MoFEMFunctionBegin;
    boost::shared_ptr<Monitor> monitor_ptr(
        new Monitor(dm, postProcFe, uXScatter, uYScatter));
    boost::shared_ptr<ForcesAndSourcesCore> null;
    CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                               monitor_ptr, null, null);
    MoFEMFunctionReturn(0);
  };

  CHKERR set_section_monitor();
  CHKERR create_post_process_element();
  uXScatter = scatter_create(0);
  uYScatter = scatter_create(1);
  CHKERR set_time_monitor();

  CHKERR TSSolve(solver, D);

  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::postProcess() {
  MoFEMFunctionBegin;

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

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

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

OpEdgeForceRhs::OpEdgeForceRhs(const std::string field_name,
                               const Range &force_edges,
                               const VectorDouble &force_vec)
    : BoundaryEleOp(field_name, OPROW), forceEdges(force_edges),
      forceVec(force_vec) {}

MoFEMErrorCode OpEdgeForceRhs::doWork(int side, EntityType type,
                                      DataForcesAndSourcesCore::EntData &data) {

  MoFEMFunctionBegin;
  const int nb_dofs = data.getIndices().size();
  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  EntityHandle ent = getNumeredEntFiniteElementPtr()->getEnt();
  if (forceEdges.find(ent) == forceEdges.end()) {
    MoFEMFunctionReturnHot(0);
  }

  std::array<double, MAX_DOFS_ON_ENTITY> nF;
  std::fill(&nF[0], &nF[nb_dofs], 0);

  FTensor::Tensor1<double, 2> t_force(forceVec(0), forceVec(1));
  const int nb_gauss_pts = data.getN().size1();
  auto t_w = getFTensor0IntegrationWeight();
  auto t_base = data.getFTensor0N();

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_nf(&nF[0], &nF[1]);
    for (int bb = 0; bb != nb_dofs / 2; ++bb) {
      t_nf(i) += (t_w * t_base * getMeasure()) * t_force(i);
      ++t_nf;
      ++t_base;
    }
    ++t_w;
  }

  if ((getDataCtx() & PetscData::CtxSetTime).any())
    for (int dd = 0; dd != nb_dofs; ++dd)
      nF[dd] *= getTStime();

  CHKERR VecSetValues(getKSPf(), data, nF.data(), ADD_VALUES);

  MoFEMFunctionReturn(0);
}