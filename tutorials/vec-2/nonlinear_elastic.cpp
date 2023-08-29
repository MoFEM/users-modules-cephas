/**
 * \file nonlinear_elastic.cpp
 * \example nonlinear_elastic.cpp
 *
 * Plane stress elastic dynamic problem
 *
 */

#include <MoFEM.hpp>
#include <MatrixFunction.hpp>

using namespace MoFEM;

constexpr int SPACE_DIM =
    EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh

using EntData = EntitiesFieldData::EntData;
using DomainEle = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::DomainEle;
using BoundaryEle =
    PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::BoundaryEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

using DomainNaturalBC =
    NaturalBC<DomainEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
using OpBodyForce =
    DomainNaturalBC::OpFlux<NaturalMeshsetType<BLOCKSET>, 1, SPACE_DIM>;

using BoundaryNaturalBC =
    NaturalBC<BoundaryEleOp>::Assembly<PETSC>::LinearForm<GAUSS>;
using OpForce = BoundaryNaturalBC::OpFlux<NaturalForceMeshsets, 1, SPACE_DIM>;

template <int DIM> struct PostProcEleByDim;

template <> struct PostProcEleByDim<2> {
  using PostProcEleDomain = PostProcBrokenMeshInMoabBase<DomainEle>;
  using PostProcEleBdy = PostProcBrokenMeshInMoabBase<BoundaryEle>;
  using SideEle = PipelineManager::ElementsAndOpsByDim<2>::FaceSideEle;
};

template <> struct PostProcEleByDim<3> {
  using PostProcEleDomain = PostProcBrokenMeshInMoabBase<BoundaryEle>;
  using PostProcEleBdy = PostProcBrokenMeshInMoabBase<BoundaryEle>;
  using SideEle = PipelineManager::ElementsAndOpsByDim<3>::FaceSideEle;
};

using SideEle = PostProcEleByDim<SPACE_DIM>::SideEle;
using PostProcEleBdy = PostProcEleByDim<SPACE_DIM>::PostProcEleBdy;

constexpr double young_modulus = 100;
constexpr double poisson_ratio = 0.3;

#include <HenckyOps.hpp>
using namespace HenckyOps;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;
  // SmartPetscObj<Vec> normUVec;

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode gettingNorms();
  MoFEMErrorCode outputResults();
  MoFEMErrorCode checkResults();
};

//! [Run problem]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR gettingNorms();
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
  CHKERR simple->addDomainField("U", H1, base, SPACE_DIM);
  CHKERR simple->addBoundaryField("U", H1, base, SPACE_DIM);
  int order = 2;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Boundary condition]
MoFEMErrorCode Example::boundaryCondition() {
  MoFEMFunctionBegin;
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();
  auto time_scale = boost::make_shared<TimeScale>();

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * (approx_order - 1);
  };

  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);

  CHKERR BoundaryNaturalBC::AddFluxToPipeline<OpForce>::add(
      pipeline_mng->getOpBoundaryRhsPipeline(), mField, "U", {time_scale},
      "FORCE", Sev::inform);

  //! [Define gravity vector]
  CHKERR DomainNaturalBC::AddFluxToPipeline<OpBodyForce>::add(
      pipeline_mng->getOpDomainRhsPipeline(), mField, "U", {time_scale},
      "BODY_FORCE", Sev::inform);

  // Essential BC
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_X",
                                           "U", 0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_Y",
                                           "U", 1, 1);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "REMOVE_Z",
                                           "U", 2, 2);
  CHKERR bc_mng->pushMarkDOFsOnEntities<DisplacementCubitBcData>(
      simple->getProblemName(), "U");

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystem() {
  MoFEMFunctionBegin;
  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  auto add_domain_ops_lhs = [&](auto &pip) {
    MoFEMFunctionBegin;
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1});
    CHKERR HenckyOps::opFactoryDomainLhs<SPACE_DIM, PETSC, GAUSS, DomainEleOp>(
        mField, pip, "U", "MAT_ELASTIC", Sev::inform);
    MoFEMFunctionReturn(0);
  };

  auto add_domain_ops_rhs = [&](auto &pip) {
    MoFEMFunctionBegin;
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1});
    CHKERR HenckyOps::opFactoryDomainRhs<SPACE_DIM, PETSC, GAUSS, DomainEleOp>(
        mField, pip, "U", "MAT_ELASTIC", Sev::inform);
    MoFEMFunctionReturn(0);
  };

  CHKERR add_domain_ops_lhs(pipeline_mng->getOpDomainLhsPipeline());
  CHKERR add_domain_ops_rhs(pipeline_mng->getOpDomainRhsPipeline());

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
  Monitor(SmartPetscObj<DM> dm, boost::shared_ptr<PostProcEleBdy> post_proc)
      : dM(dm), postProc(post_proc){};
  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;
    constexpr int save_every_nth_step = 1;
    if (ts_step % save_every_nth_step == 0) {
      CHKERR DMoFEMLoopFiniteElements(dM, "bFE", postProc);
      CHKERR postProc->writeFile(
          "out_step_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
    }
    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcEleBdy> postProc;
};

// struct MonitorNorms : public FEMethod {
//   MonitorNorms(SmartPetscObj<DM> dm, boost::shared_ptr<DomainEle>
//   norm_proc_fe,
//                SmartPetscObj<Vec> norm_u_vec)
//       : dM(dm), normProcFe(norm_proc_fe), normUVec(norm_u_vec){};
//   MoFEMErrorCode postProcess() {
//     MoFEMFunctionBegin;
//     constexpr int save_every_nth_step = 1;
//     if (ts_step % save_every_nth_step == 0) {
//       CHKERR VecZeroEntries(normUVec);
//       CHKERR DMoFEMLoopFiniteElements(dM, "dFE", normProcFe);
//       MOFEM_LOG("WORLD", Sev::inform) << "Norms: ";

//       CHKERR VecAssemblyBegin(normUVec);
//       CHKERR VecAssemblyEnd(normUVec);
//       double norm_u2;
//       CHKERR VecSum(normUVec, &norm_u2);
//       MOFEM_LOG("EXAMPLE", Sev::inform)
//           << "norm_u at " << ts_step << ": " << std::sqrt(norm_u2);
//     }
//     MoFEMFunctionReturn(0);
//   }

// private:
//   SmartPetscObj<DM> dM;
//   boost::shared_ptr<DomainEle> normProcFe;
//   SmartPetscObj<Vec> normUVec;
// };

//! [Solve]
MoFEMErrorCode Example::solveSystem() {
  MoFEMFunctionBegin;
  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  auto dm = simple->getDM();
  auto ts = pipeline_mng->createTSIM();

  // auto norm_vec = createSmartVectorMPI(mField.get_comm(), PETSC_DECIDE, 1);

  // Setup postprocessing
  auto create_post_proc_fe = [dm, this, simple]() {
    auto post_proc_ele_domain = [dm, this](auto &pip_domain) {
      CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip_domain, {H1});
      auto common_ptr = commonDataFactory<SPACE_DIM, GAUSS, DomainEleOp>(
          mField, pip_domain, "U", "MAT_ELASTIC", Sev::inform);
      return common_ptr;
    };

    auto post_proc_fe_bdy = boost::make_shared<
        typename PostProcEleByDim<SPACE_DIM>::PostProcEleBdy>(mField);
    auto u_ptr = boost::make_shared<MatrixDouble>();
    post_proc_fe_bdy->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));
    auto op_loop_side =
        new OpLoopSide<SideEle>(mField, simple->getDomainFEName(), SPACE_DIM);
    auto common_ptr = post_proc_ele_domain(op_loop_side->getOpPtrVector());
    post_proc_fe_bdy->getOpPtrVector().push_back(op_loop_side);

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    post_proc_fe_bdy->getOpPtrVector().push_back(

        new OpPPMap(

            post_proc_fe_bdy->getPostProcMesh(),
            post_proc_fe_bdy->getMapGaussPts(),

            {},

            {{"U", u_ptr}},

            {{"GRAD", common_ptr->matGradPtr},
             {"FIRST_PIOLA", common_ptr->getMatFirstPiolaStress()}},

            {}

            )

    );
    return post_proc_fe_bdy;
  };

  // auto create_post_proc_norm_fe = [dm, this]() {
  //   auto post_proc_fe = boost::make_shared<DomainEle>(mField);

  //   auto post_proc_rule_hook = [](int, int, int p) -> int { return 2 * p; };
  //   post_proc_fe->getRuleHook = post_proc_rule_hook;

  //   CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
  //       post_proc_fe->getOpPtrVector(), {H1});

  //   auto u_ptr = boost::make_shared<MatrixDouble>();
  //   post_proc_fe->getOpPtrVector().push_back(
  //       new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));

  //   post_proc_fe->getOpPtrVector().push_back(
  //       new OpCalcNormL2Tesnosr1<SPACE_DIM>("U", u_ptr, normUVec,
  //                                           mField.get_comm_rank()));

  //   return post_proc_fe;
  // };

  auto add_extra_finite_elements_to_ksp_solver_pipelines = [&]() {
    MoFEMFunctionBegin;

    auto pre_proc_ptr = boost::make_shared<FEMethod>();
    auto post_proc_rhs_ptr = boost::make_shared<FEMethod>();
    auto post_proc_lhs_ptr = boost::make_shared<FEMethod>();

    auto time_scale = boost::make_shared<TimeScale>();

    auto get_bc_hook_rhs = [this, pre_proc_ptr, time_scale]() {
      EssentialPreProc<DisplacementCubitBcData> hook(mField, pre_proc_ptr,
                                                     {time_scale}, false);
      return hook;
    };
    pre_proc_ptr->preProcessHook = get_bc_hook_rhs();

    auto get_post_proc_hook_rhs = [this, post_proc_rhs_ptr]() {
      MoFEMFunctionBegin;
      CHKERR EssentialPreProcReaction<DisplacementCubitBcData>(
          mField, post_proc_rhs_ptr, nullptr, Sev::verbose)();
      CHKERR EssentialPostProcRhs<DisplacementCubitBcData>(
          mField, post_proc_rhs_ptr, 1.)();
      MoFEMFunctionReturn(0);
    };
    auto get_post_proc_hook_lhs = [this, post_proc_lhs_ptr]() {
      return EssentialPostProcLhs<DisplacementCubitBcData>(
          mField, post_proc_lhs_ptr, 1.);
    };
    post_proc_rhs_ptr->postProcessHook = get_post_proc_hook_rhs;
    post_proc_lhs_ptr->postProcessHook = get_post_proc_hook_lhs();

    // This is low level pushing finite elements (pipelines) to solver
    auto ts_ctx_ptr = getDMTsCtx(simple->getDM());
    ts_ctx_ptr->getPreProcessIFunction().push_front(pre_proc_ptr);
    ts_ctx_ptr->getPreProcessIJacobian().push_front(pre_proc_ptr);
    ts_ctx_ptr->getPostProcessIFunction().push_back(post_proc_rhs_ptr);
    ts_ctx_ptr->getPostProcessIJacobian().push_back(post_proc_lhs_ptr);
    MoFEMFunctionReturn(0);
  };

  // Add extra finite elements to SNES solver pipelines to resolve essential
  // boundary conditions
  CHKERR add_extra_finite_elements_to_ksp_solver_pipelines();

  auto create_monitor_fe = [dm](auto &&post_proc_fe) {
    return boost::make_shared<Monitor>(dm, post_proc_fe);
  };

  // auto create_monitor_norm_fe = [dm, this](auto &&post_proc_fe) {
  //   return boost::make_shared<MonitorNorms>(dm, post_proc_fe, normUVec);
  // };

  // Set monitor which postprocessing results and saves them to the hard drive
  boost::shared_ptr<FEMethod> null_fe;
  auto monitor_ptr = create_monitor_fe(create_post_proc_fe());
  CHKERR DMMoFEMTSSetMonitor(dm, ts, simple->getDomainFEName(), null_fe,
                             null_fe, monitor_ptr);

  // normUVec = createSmartVectorMPI(mField.get_comm(), PETSC_DECIDE, 1);
  // auto monitor_norm_ptr = create_monitor_norm_fe(create_post_proc_norm_fe());
  // CHKERR DMMoFEMTSSetMonitor(dm, ts, simple->getDomainFEName(), null_fe,
  //                            null_fe, monitor_norm_ptr);

  // Set time solver
  double ftime = 1;
  CHKERR TSSetDuration(ts, PETSC_DEFAULT, ftime);
  CHKERR TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);

  auto D = createDMVector(simple->getDM());

  CHKERR TSSetSolution(ts, D);
  CHKERR TSSetFromOptions(ts);

  CHKERR TSSolve(ts, NULL);
  CHKERR TSGetTime(ts, &ftime);

  PetscInt steps, snesfails, rejects, nonlinits, linits;
  CHKERR TSGetStepNumber(ts, &steps);
  CHKERR TSGetSNESFailures(ts, &snesfails);
  CHKERR TSGetStepRejections(ts, &rejects);
  CHKERR TSGetSNESIterations(ts, &nonlinits);
  CHKERR TSGetKSPIterations(ts, &linits);
  MOFEM_LOG_C("EXAMPLE", Sev::inform,
              "steps %d (%d rejected, %d SNES fails), ftime %g, nonlinits "
              "%d, linits %d",
              steps, rejects, snesfails, ftime, nonlinits, linits);

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Getting norms]
MoFEMErrorCode Example::gettingNorms() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();
  auto dm = simple->getDM();

  auto post_proc_norm_fe = boost::make_shared<DomainEle>(mField);
  auto norms_vec = createSmartVectorMPI(mField.get_comm(), PETSC_DECIDE, 1);

  auto post_proc_norm_rule_hook = [](int, int, int p) -> int { return 2 * p; };
  post_proc_norm_fe->getRuleHook = post_proc_norm_rule_hook;

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      post_proc_norm_fe->getOpPtrVector(), {H1});

  auto u_ptr = boost::make_shared<MatrixDouble>();
  post_proc_norm_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));

  post_proc_norm_fe->getOpPtrVector().push_back(
      new OpCalcNormL2Tesnosr1<SPACE_DIM>("U", u_ptr, norms_vec,
                                          mField.get_comm_rank()));

  CHKERR DMoFEMLoopFiniteElements(dm, "dFE", post_proc_norm_fe);

  CHKERR VecAssemblyBegin(norms_vec);
  CHKERR VecAssemblyEnd(norms_vec);
  double norm_u2;
  CHKERR VecSum(norms_vec, &norm_u2);
  MOFEM_LOG("EXAMPLE", Sev::inform) << "norm_u: " << std::sqrt(norm_u2);

  MoFEMFunctionReturn(0);
}
//! [Getting norms]

//! [Postprocessing results]
MoFEMErrorCode Example::outputResults() {
  MoFEMFunctionBegin;
  PetscInt test_nb = 0;
  PetscBool test_flg = PETSC_FALSE;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-test", &test_nb, &test_flg);

  if (test_flg) {
    auto *simple = mField.getInterface<Simple>();
    auto T = createDMVector(simple->getDM());
    CHKERR DMoFEMMeshToLocalVector(simple->getDM(), T, INSERT_VALUES,
                                   SCATTER_FORWARD);
    double nrm2;
    CHKERR VecNorm(T, NORM_2, &nrm2);
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Regression norm " << nrm2;
    double regression_value = 0;
    switch (test_nb) {
    case 1:
      regression_value = 1.02789;
      break;
    case 2:
      regression_value = 1.8841e+00;
      break;
    case 3:
      regression_value = 1.8841e+00;
      break;

    default:
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "Wrong test number.");
      break;
    }
    if (fabs(nrm2 - regression_value) > 1e-2)
      SETERRQ2(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
               "Regression test field; wrong norm value. %6.4e != %6.4e", nrm2,
               regression_value);
  }

  // double norm_u2;
  // CHKERR VecSum(normUVec, &norm_u2);
  // MOFEM_LOG("EXAMPLE", Sev::inform) << "norm_u at the end: " <<
  // std::sqrt(norm_u2);
  MoFEMFunctionReturn(0);
}
//! [Postprocessing results]

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
