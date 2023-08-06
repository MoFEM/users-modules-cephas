/**
 * \file dynamic_elastic.cpp
 * \example dynamic_elastic.cpp
 *
 * Plane stress elastic dynamic problem
 *
 */

#include <MoFEM.hpp>
#include <MatrixFunction.hpp>
using namespace MoFEM;

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = FaceElementForcesAndSourcesCore;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = VolumeElementForcesAndSourcesCore;
};

constexpr int SPACE_DIM =
    EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh

using EntData = EntitiesFieldData::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

using OpMass = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, SPACE_DIM>;
using OpMassForTensor = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
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
using OpGradTimesTensor2 = FormsIntegrators<DomainEleOp>::Assembly<
          PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;

using OpGradTimesPiola = FormsIntegrators<DomainEleOp>::Assembly<
          PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;

using OpRhsTestPiola = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM * SPACE_DIM, 1>;

constexpr double rho = 1;
constexpr double omega = 2.4;
constexpr double young_modulus = 1;
constexpr double poisson_ratio = 0.25;

#include <HenckyOps.hpp>
using namespace HenckyOps;

static double *ts_time_ptr;
static double *ts_aa_ptr;


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
  MoFEMErrorCode tsSetUp(TS ts);

  // SmartPetscObj<VecScatter> getScatter(Vec x, Vec y, enum FR fr);
  SmartPetscObj<Vec> getSubVector();

  SmartPetscObj<DM> solverSubDM;
  SmartPetscObj<Vec> globSol;
  Example *fsRawPtr;
  static MoFEMErrorCode tsPreStage(TS ts, double a);
  static MoFEMErrorCode tsPostStep(TS ts);

private:
  /**
   * @brief Pre process time step
   *
   * Refine mesh and update fields
   *
   * @param ts
   * @return MoFEMErrorCode
   */
  // static MoFEMErrorCode tsPreProc(TS ts);

  /**
   * @brief Post process time step.
   *
   * Currently that function do not make anything major
   *
   * @param ts
   * @return MoFEMErrorCode
   */
  // static MoFEMErrorCode tsPostProc(TS ts);

  

  // static MoFEMErrorCode tsSetIFunction(TS ts, PetscReal t, Vec u, Vec u_t,
  //                                      Vec f,
  //                                      void *ctx); //< Wrapper for SNES Rhs
  // static MoFEMErrorCode tsSetIJacobian(TS ts, PetscReal t, Vec u, Vec u_t,
  //                                      PetscReal a, Mat A, Mat B,
  //                                      void *ctx); ///< Wrapper for SNES Lhs
  // static MoFEMErrorCode tsMonitor(TS ts, PetscInt step, PetscReal t, Vec u,
  //                                 void *ctx);      ///< Wrapper for TS monitor
  // static MoFEMErrorCode pcSetup(PC pc);
  // static MoFEMErrorCode pcApply(PC pc, Vec pc_f, Vec pc_x);

  SmartPetscObj<Vec> globRes; //< global residual
  SmartPetscObj<Mat> subB;    //< sub problem tangent matrix
  SmartPetscObj<KSP> subKSP;  //< sub problem KSP solver

  boost::shared_ptr<SnesCtx>
      snesCtxPtr; //< infernal data (context) for MoFEM SNES fuctions
  boost::shared_ptr<TsCtx>
      tsCtxPtr;   //<  internal data (context) for MoFEM TS functions.
};

static boost::weak_ptr<TSPrePostProc> tsPrePostProc;


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
  // Add field
  CHKERR simple->addDomainField("V", H1, AINSWORTH_LEGENDRE_BASE,
                                SPACE_DIM);
  CHKERR simple->addBoundaryField("V", H1, AINSWORTH_LEGENDRE_BASE, SPACE_DIM);
  CHKERR simple->addDomainField("F", H1, AINSWORTH_LEGENDRE_BASE,
                                SPACE_DIM * SPACE_DIM);
  CHKERR simple->addDataField("x_1", H1, AINSWORTH_LEGENDRE_BASE,
                              SPACE_DIM);
  CHKERR simple->addDataField("x_2", H1, AINSWORTH_LEGENDRE_BASE,
                              SPACE_DIM);
  CHKERR simple->addDataField("GEOMETRY", H1, AINSWORTH_LEGENDRE_BASE,
                              SPACE_DIM);                              
  int order = 2;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simple->setFieldOrder("V", order);
  CHKERR simple->setFieldOrder("F", order);
  CHKERR simple->setFieldOrder("x_1", order); 
  CHKERR simple->setFieldOrder("x_2", order); 
  CHKERR simple->setFieldOrder("GEOMETRY", order);
  CHKERR simple->setUp();

  auto project_ho_geometry = [&]() {
    Projection10NodeCoordsOnField ent_method(mField, "GEOMETRY");
    return mField.loop_dofs("GEOMETRY", ent_method);
    Projection10NodeCoordsOnField ent_method_x(mField, "x_1");
    return mField.loop_dofs("x_1", ent_method_x);
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

  CHKERR bc_mng->removeBlockDOFsOnEntities<DisplacementCubitBcData>(
      simple->getProblemName(), "U");

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

MoFEMErrorCode TSPrePostProc::tsPreStage(TS ts, double a) {
  MoFEMFunctionBegin;

if (auto ptr = tsPrePostProc.lock()) {
    auto &m_field = ptr->fsRawPtr->mField;
    auto fb = m_field.getInterface<FieldBlas>();
    double dt;
    TSGetTimeStep(ts, &dt);
    //x_t+1 = Δt * v + x_t 
    CHKERR fb->fieldCopy(1., "x_1", "x_2");
    CHKERR fb->fieldAxpy(dt, "V", "x_2");
}
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode TSPrePostProc::tsPostStep(TS ts) {
  MoFEMFunctionBegin;

if (auto ptr = tsPrePostProc.lock()) {
    auto &m_field = ptr->fsRawPtr->mField;
    auto fb = m_field.getInterface<FieldBlas>();
    //x_t+1 = Δt * v + x_t 
    CHKERR fb->fieldCopy(1., "x_2", "x_1");
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

  auto rho_ptr = boost::make_shared<double>(rho);

  auto add_rho_block = [this, rho_ptr](auto &pip, auto block_name, Sev sev) {
    MoFEMFunctionBegin;

    struct OpMatRhoBlocks : public DomainEleOp {
      OpMatRhoBlocks(boost::shared_ptr<double> rho_ptr,
                     MoFEM::Interface &m_field, Sev sev,
                     std::vector<const CubitMeshSets *> meshset_vec_ptr)
          : DomainEleOp(NOSPACE, DomainEleOp::OPSPACE), rhoPtr(rho_ptr) {
        CHK_THROW_MESSAGE(extractRhoData(m_field, meshset_vec_ptr, sev),
                          "Can not get data from block");
      }

      MoFEMErrorCode doWork(int side, EntityType type,
                            EntitiesFieldData::EntData &data) {

        MoFEMFunctionBegin;

        for (auto &b : blockData) {
          if (b.blockEnts.find(getFEEntityHandle()) != b.blockEnts.end()) {
            *rhoPtr = b.rho;
            MoFEMFunctionReturnHot(0);
          }
        }

        *rhoPtr = rho;

        MoFEMFunctionReturn(0);
      }

    private:
      struct BlockData {
        double rho;
        Range blockEnts;
      };

      std::vector<BlockData> blockData;

      MoFEMErrorCode
      extractRhoData(MoFEM::Interface &m_field,
                     std::vector<const CubitMeshSets *> meshset_vec_ptr,
                     Sev sev) {
        MoFEMFunctionBegin;

        for (auto m : meshset_vec_ptr) {
          MOFEM_TAG_AND_LOG("WORLD", sev, "Mat Rho Block") << *m;
          std::vector<double> block_data;
          CHKERR m->getAttributes(block_data);
          if (block_data.size() < 1) {
            SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                    "Expected that block has two attributes");
          }
          auto get_block_ents = [&]() {
            Range ents;
            CHKERR
            m_field.get_moab().get_entities_by_handle(m->meshset, ents, true);
            return ents;
          };

          MOFEM_TAG_AND_LOG("WORLD", sev, "Mat Rho Block")
              << m->getName() << ": ro = " << block_data[0];

          blockData.push_back({block_data[0], get_block_ents()});
        }
        MOFEM_LOG_CHANNEL("WORLD");
        MOFEM_LOG_CHANNEL("WORLD");
        MoFEMFunctionReturn(0);
      }

      boost::shared_ptr<double> rhoPtr;
    };

    pip.push_back(new OpMatRhoBlocks(
        rho_ptr, mField, sev,

        // Get blockset using regular expression
        mField.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(std::regex(

            (boost::format("%s(.*)") % block_name).str()

                ))

            ));
    MoFEMFunctionReturn(0);
  };

  // Get pointer to U_tt shift in domain element
  auto get_rho = [rho_ptr](const double, const double, const double) {
    return *rho_ptr;
  };

  // specific time scaling
  auto get_time_scale = [this](const double time) {
    return sin(time * omega * M_PI);
  };

  auto apply_lhs = [&](auto &pip) {
    MoFEMFunctionBegin;
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1},
                                                          "GEOMETRY");
    CHKERR HenckyOps::opFactoryDomainLhs<SPACE_DIM, PETSC, GAUSS, DomainEleOp>(
        mField, pip, "U", "MAT_ELASTIC", Sev::verbose);
    CHKERR add_rho_block(pip, "MAT_RHO", Sev::verbose);

    pip.push_back(new OpMass("V", "V", get_rho));
    pip.push_back(new OpMassForTensor("F", "F"));
    // static_cast<OpMass &>(pip.back()).feScalingFun =
    //     [](const FEMethod *fe_ptr) { return fe_ptr->ts_aa; };
    MoFEMFunctionReturn(0);
  };

  auto apply_rhs = [&](auto &pip) {
    MoFEMFunctionBegin;

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        pipeline_mng->getOpDomainRhsPipeline(), {H1}, "GEOMETRY");
    

    auto mat_F_tensor_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateTensor2FieldValues<SPACE_DIM, SPACE_DIM>(
        "F", mat_F_tensor_ptr));

    auto mat_dot_F_tensor_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(new OpCalculateTensor2FieldValuesDot<SPACE_DIM, SPACE_DIM>(
        "F", mat_dot_F_tensor_ptr));

    auto mat_v_grad_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(
      new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("V",
                                                               mat_v_grad_ptr));

    auto mat_x_grad_ptr = boost::make_shared<MatrixDouble>();
    pip.push_back(
      new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("x",
                                                               mat_x_grad_ptr));

    auto gravity_vector_ptr = boost::make_shared<MatrixDouble>();
    gravity_vector_ptr->resize(SPACE_DIM, 1);
    auto set_body_force = [&]() {
      FTensor::Index<'i', SPACE_DIM> i;
      MoFEMFunctionBegin;
      auto t_force = getFTensor1FromMat<SPACE_DIM, 0>(*gravity_vector_ptr);
      double unit_weight = 1.;
      CHKERR PetscOptionsGetReal(PETSC_NULL, "", "-unit_weight", &unit_weight,
                                 PETSC_NULL);
      t_force(i) = 0;
      if (SPACE_DIM == 2) {
        t_force(1) = -unit_weight;
      } else if (SPACE_DIM == 3) {
        t_force(2) = -unit_weight;
      }
      MoFEMFunctionReturn(0);
    };

    CHKERR set_body_force();
    pip.push_back(new OpBodyForce(
      "U", gravity_vector_ptr, [](double, double, double) { return 1.; }));


    //some operator that calculates F^st
    auto mat_P_stab_ptr = boost::make_shared<MatrixDouble>();
    //some operator that calculate P^st
    auto one = [](const double, const double, const double) { return 1; };
    pip.push_back(new OpGradTimesTensor2("V", mat_P_stab_ptr));
    pip.push_back(new OpGradTimesPiola("V", mat_P_stab_ptr, one));
    
    pip.push_back(new OpRhsTestPiola("F", mat_v_grad_ptr, one));

    // CHKERR add_rho_block(pip, "MAT_RHO", Sev::inform);

    // auto mat_acceleration_ptr = boost::make_shared<MatrixDouble>();
    // // Apply inertia
    // pip.push_back(new OpCalculateVectorFieldValuesDotDot<SPACE_DIM>(
    //     "U", mat_acceleration_ptr));
    // pip.push_back(new OpInertiaForce("U", mat_acceleration_ptr, get_rho));

    // CHKERR DomainNaturalBC::AddFluxToPipeline<OpBodyForceVector>::add(
    //     pip, mField, "U", {},
    //     {boost::make_shared<TimeScaleVector<SPACE_DIM>>("-time_vector_file",
    //                                                     true)},
    //     "BODY_FORCE", Sev::inform);

    MoFEMFunctionReturn(0);
  };

  CHKERR apply_lhs(pipeline_mng->getOpDomainLhsPipeline());
  CHKERR apply_rhs(pipeline_mng->getOpDomainRhsPipeline());

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * approx_order;
  };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);

  auto get_bc_hook = [&]() {
    EssentialPreProc<DisplacementCubitBcData> hook(
        mField, pipeline_mng->getDomainRhsFE(),
        {boost::make_shared<TimeScale>()});
    return hook;
  };

  pipeline_mng->getDomainRhsFE()->preProcessHook = get_bc_hook();

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
  Monitor(SmartPetscObj<DM> dm, boost::shared_ptr<PostProcEle> post_proc)
      : dM(dm), postProc(post_proc){};
  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;
    constexpr int save_every_nth_step = 1;
    if (ts_step % save_every_nth_step == 0) {
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc);
      CHKERR postProc->writeFile(
          "out_step_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
    }
    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcEle> postProc;
};

//! [Solve]
MoFEMErrorCode Example::solveSystem() {
  MoFEMFunctionBegin;
  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  auto dm = simple->getDM();
  MoFEM::SmartPetscObj<TS> ts;
  ts = pipeline_mng->createTSEX();

  // Setup postprocessing
  auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      post_proc_fe->getOpPtrVector(), {H1});
  auto common_ptr = commonDataFactory<SPACE_DIM, GAUSS, DomainEleOp>(
      mField, post_proc_fe->getOpPtrVector(), "U", "MAT_ELASTIC", Sev::inform);

  auto u_ptr = boost::make_shared<MatrixDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));
  auto X_ptr = boost::make_shared<MatrixDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<SPACE_DIM>("GEOMETRY", X_ptr));

  using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

  post_proc_fe->getOpPtrVector().push_back(

      new OpPPMap(

          post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

          {},

          {{"U", u_ptr}, {"GEOMETRY", X_ptr}},

          {{"GRAD", common_ptr->matGradPtr},
           {"FIRST_PIOLA", common_ptr->getMatFirstPiolaStress()}},

          {}

          )

  );

  // Add monitor to time solver
  boost::shared_ptr<FEMethod> null_fe;
  auto monitor_ptr = boost::make_shared<Monitor>(dm, post_proc_fe);
  CHKERR DMMoFEMTSSetMonitor(dm, ts, simple->getDomainFEName(), null_fe,
                             null_fe, monitor_ptr);

  double ftime = 1;
  // CHKERR TSSetMaxTime(ts, ftime);
  CHKERR TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);

  {
    // SmartPetscObj<Mat> M;   ///< Mass matrix
    // SmartPetscObj<KSP> ksp; ///< Linear solver
    // CHKERR DMCreateMatrix_MoFEM(dm, M);
    // CHKERR MatZeroEntries(M);
    // CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
    //                                 vol_mass_ele);
    // CHKERR MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    // CHKERR MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    // // Create and septup KSP (linear solver), we need this to calculate g(t,u) =
    // // M^-1G(t,u)
    // ksp = createKSP(m_field.get_comm());
    // CHKERR KSPSetOperators(ksp, M, M);
    // CHKERR KSPSetFromOptions(ksp);
    // CHKERR KSPSetUp(ksp);
  }

  auto T = createDMVector(simple->getDM());
  CHKERR DMoFEMMeshToLocalVector(simple->getDM(), T, INSERT_VALUES,
                                 SCATTER_FORWARD);
  // auto TT = vectorDuplicate(T);
  // CHKERR TS2SetSolution(ts, T, TT);
  CHKERR TSSetFromOptions(ts);

  auto fb = mField.getInterface<FieldBlas>();

  auto ts_pre_post_proc = boost::make_shared<TSPrePostProc>();
  tsPrePostProc = ts_pre_post_proc;

  CHKERR TSSetPreStage(ts, TSPrePostProc::tsPreStage);
  CHKERR TSSetPostStep(ts, TSPrePostProc::tsPostStep);

  CHKERR TSSolve(ts, NULL);
  CHKERR TSGetTime(ts, &ftime);

  PetscInt steps, snesfails, rejects, nonlinits, linits;
#if PETSC_VERSION_GE(3, 8, 0)
  CHKERR TSGetStepNumber(ts, &steps);
#else
  CHKERR TSGetTimeStepNumber(ts, &steps);
#endif
  CHKERR TSGetSNESFailures(ts, &snesfails);
  CHKERR TSGetStepRejections(ts, &rejects);
  CHKERR TSGetSNESIterations(ts, &nonlinits);
  CHKERR TSGetKSPIterations(ts, &linits);
  MOFEM_LOG_C("EXAMPLE", Sev::inform,
              "steps %d (%d rejected, %d SNES fails), ftime %g, nonlinits "
              "%d, linits %d\n",
              steps, rejects, snesfails, ftime, nonlinits, linits);

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
