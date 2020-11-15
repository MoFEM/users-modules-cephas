/**
 * \file lesson8_contact.cpp
 * \example lesson8_contact.cpp
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

using OpK = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM, 0>;
using OpInternalForce = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesSymTensor<1, SPACE_DIM, SPACE_DIM>;
using OpBodyForce = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpSource<1, SPACE_DIM>;

constexpr int order = 2;
constexpr double young_modulus = 10;
constexpr double poisson_ratio = 0.25;
constexpr double cn = 0.1;
constexpr double spring_stiffness = 0;
constexor double stab = 1e-12;

double integral_1_lhs = 0;
double integral_1_rhs = 0;
double integral_2_lhs = 0;
double integral_2_rhs = 0;
double integral_3_lhs = 0;
double integral_3_rhs = 0;

boost::shared_ptr<BoundaryEle> debug_post_proc;
moab::Core mb_post_debug;
moab::Interface &moab_debug = mb_post_debug;

#include <ContactOps.hpp>
#include <OpPostProcElastic.hpp>
using namespace OpContactTools;

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
  boost::shared_ptr<OpContactTools::CommonData> commonDataPtr;
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
  CHKERR simple->setFieldOrder("U", order + 1, &boundary_ents);

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

  commonDataPtr = boost::make_shared<OpContactTools::CommonData>();

  commonDataPtr->mGradPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mStrainPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mStressPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->contactStressPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->contactStressDivergencePtr =
      boost::make_shared<MatrixDouble>();
  commonDataPtr->contactTractionPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->contactDispPtr = boost::make_shared<MatrixDouble>();
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;

  commonDataPtr->mDPtr = boost::make_shared<MatrixDouble>();
  commonDataPtr->mDPtr->resize(size_symm * size_symm, 1);

  jAc.resize(SPACE_DIM, SPACE_DIM, false);
  invJac.resize(SPACE_DIM, SPACE_DIM, false);

  debug_post_proc = boost::make_shared<BoundaryEle>(mField);
  if (SPACE_DIM == 2) 
    debug_post_proc->getOpPtrVector().push_back(
        new OpSetContrariantPiolaTransformOnEdge());
  debug_post_proc->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<SPACE_DIM>(
          "U", commonDataPtr->contactDispPtr));
  debug_post_proc->getOpPtrVector().push_back(
      new OpConstrainBoundaryTraction("SIGMA", commonDataPtr));
  debug_post_proc->getOpPtrVector().push_back(
      new OpPostProcDebug(mField, "U", commonDataPtr, &moab_debug));
  debug_post_proc->setRuleHook = [](int a, int b, int c) {
    return 2 * order + 1;
  };
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

    CHKERR prb_mng->removeDofsOnEntities(simple->getProblemName(), "U", verts,
                                         lo, hi);
    MoFEMFunctionReturn(0);
  };

  Range boundary_ents;
  boundary_ents.merge(fix_disp("FIX_X"));
  boundary_ents.merge(fix_disp("FIX_Y"));
  boundary_ents.merge(fix_disp("FIX_Z"));
  boundary_ents.merge(fix_disp("FIX_ALL"));
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

  auto add_domain_ops_lhs = [&](auto &pipeline) {
    pipeline.push_back(new OpK("U", "U", commonDataPtr->mDPtr));
    pipeline.push_back(
        new OpConstrainDomainLhs_dU("SIGMA", "U", commonDataPtr));
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
    pipeline.push_back(new OpSymmetrizeTensor<SPACE_DIM>(
        "U", commonDataPtr->mGradPtr, commonDataPtr->mStrainPtr));
    pipeline.push_back(new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
        "U", commonDataPtr->mStrainPtr, commonDataPtr->mStressPtr,
        commonDataPtr->mDPtr));
    pipeline.push_back(new OpInternalForce("U", commonDataPtr->mStressPtr));

    pipeline.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>(
        "U", commonDataPtr->contactDispPtr));

    pipeline.push_back(new OpCalculateHVecTensorField<SPACE_DIM, SPACE_DIM>(
        "SIGMA", commonDataPtr->contactStressPtr));
    pipeline.push_back(
        new OpCalculateHVecTensorDivergence<SPACE_DIM, SPACE_DIM>(
            "SIGMA", commonDataPtr->contactStressDivergencePtr));
    pipeline.push_back(new OpConstrainDomainRhs("SIGMA", commonDataPtr));
    pipeline.push_back(new OpInternalDomainContactRhs("U", commonDataPtr));
  };

  auto add_boundary_base_ops = [&](auto &pipeline) {
    if (SPACE_DIM == 2)
      pipeline.push_back(new OpSetContrariantPiolaTransformOnEdge());
    pipeline.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>(
        "U", commonDataPtr->contactDispPtr));
    pipeline.push_back(new OpConstrainBoundaryTraction("SIGMA", commonDataPtr));
  };

  auto add_boundary_ops_lhs = [&](auto &pipeline) {
    pipeline.push_back(
        new OpConstrainBoundaryLhs_dU("SIGMA", "U", commonDataPtr));
    pipeline.push_back(
        new OpConstrainBoundaryLhs_dTraction("SIGMA", "SIGMA", commonDataPtr));
    pipeline.push_back(new OpSpringLhs("U", "U"));
  };

  auto add_boundary_ops_rhs = [&](auto &pipeline) {
    pipeline.push_back(new OpConstrainBoundaryRhs("SIGMA", commonDataPtr));
    pipeline.push_back(new OpSpringRhs("U", commonDataPtr));
    // for tests, comment OpInternalDomainContactRhs
    //just for testing, does not assemble
    pipeline.push_back(new OpInternalBoundaryContactRhs("U", commonDataPtr));
  };

  add_domain_base_ops(pipeline_mng->getOpDomainLhsPipeline());
  add_domain_base_ops(pipeline_mng->getOpDomainRhsPipeline());
  add_domain_ops_lhs(pipeline_mng->getOpDomainLhsPipeline());
  add_domain_ops_rhs(pipeline_mng->getOpDomainRhsPipeline());

  add_boundary_base_ops(pipeline_mng->getOpBoundaryLhsPipeline());
  add_boundary_base_ops(pipeline_mng->getOpBoundaryRhsPipeline());
  // add_boundary_ops_lhs(pipeline_mng->getOpBoundaryLhsPipeline());
  // add_boundary_ops_rhs(pipeline_mng->getOpBoundaryRhsPipeline());

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
    postProcFe = boost::make_shared<PostProcEle>(mField);
    postProcFe->generateReferenceElementMesh();
    if (SPACE_DIM == 2) {
      postProcFe->getOpPtrVector().push_back(new OpCalculateJacForFace(jAc));
      postProcFe->getOpPtrVector().push_back(
          new OpCalculateInvJacForFace(invJac));
      postProcFe->getOpPtrVector().push_back(new OpSetInvJacH1ForFace(invJac));
      postProcFe->getOpPtrVector().push_back(new OpMakeHdivFromHcurl());
      postProcFe->getOpPtrVector().push_back(
          new OpSetContravariantPiolaTransformFace(jAc));
      postProcFe->getOpPtrVector().push_back(new OpSetInvJacHcurlFace(invJac));
    }

    postProcFe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
            "U", commonDataPtr->mGradPtr));
    postProcFe->getOpPtrVector().push_back(new OpSymmetrizeTensor<SPACE_DIM>(
        "U", commonDataPtr->mGradPtr, commonDataPtr->mStrainPtr));
    postProcFe->getOpPtrVector().push_back(
        new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
            "U", commonDataPtr->mStrainPtr, commonDataPtr->mStressPtr,
            commonDataPtr->mDPtr));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateHVecTensorDivergence<SPACE_DIM, SPACE_DIM>(
            "SIGMA", commonDataPtr->contactStressDivergencePtr));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateHVecTensorField<SPACE_DIM, SPACE_DIM>(
            "SIGMA", commonDataPtr->contactStressPtr));

    postProcFe->getOpPtrVector().push_back(
        new Tutorial::OpPostProcElastic<SPACE_DIM>(
            "U", postProcFe->postProcMesh, postProcFe->mapGaussPts,
            commonDataPtr->mStrainPtr, commonDataPtr->mStressPtr));

    postProcFe->getOpPtrVector().push_back(new OpPostProcContact<SPACE_DIM>(
        "SIGMA", postProcFe->postProcMesh, postProcFe->mapGaussPts,
        commonDataPtr));
    postProcFe->addFieldValuesPostProc("U", "DISPLACEMENT");
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
        new Monitor(dm, postProcFe, uXScatter, uYScatter, uZScatter));
    boost::shared_ptr<ForcesAndSourcesCore> null;
    CHKERR DMMoFEMTSSetMonitor(dm, solver, simple->getDomainFEName(),
                               monitor_ptr, null, null);
    MoFEMFunctionReturn(0);
  };

  CHKERR set_section_monitor();
  CHKERR create_post_process_element();
  uXScatter = scatter_create(0);
  uYScatter = scatter_create(1);
  if (SPACE_DIM == 3)
    uZScatter = scatter_create(2);
  CHKERR set_time_monitor();

  CHKERR TSSolve(solver, D);

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

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

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
