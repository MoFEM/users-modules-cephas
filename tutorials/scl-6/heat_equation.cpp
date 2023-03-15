/**
 * \file heat_equation.cpp
 * \example heat_equation.cpp
 *
 * \brief Solve the time-dependent Heat Equation
 \f[
 \begin{aligned}
\frac{\partial u(\mathbf{x}, t)}{\partial t}-\Delta u(\mathbf{x}, t)
&=f(\mathbf{x}, t) & & \forall \mathbf{x} \in \Omega, t \in(0, T), \\
u(\mathbf{x}, 0) &=u_{0}(\mathbf{x}) & & \forall \mathbf{x} \in \Omega, \\
u(\mathbf{x}, t) &=g(\mathbf{x}, t) & & \forall \mathbf{x} \in \partial \Omega,
t \in(0, T). \end{aligned}
 \f]
 **/

#include <stdlib.h>
#include <cmath>
#include <BasicFiniteElements.hpp>

using namespace MoFEM;

static char help[] = "...\n\n";

template <int DIM> struct ElementsAndOps {};

//! [Define dimension]
constexpr int SPACE_DIM = 2; //< Space dimension of problem, mesh
//! [Define dimension]

using EntData = EntitiesFieldData::EntData;
using DomainEle = PipelineManager::FaceEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = PipelineManager::EdgeEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

using OpDomainMass = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, 1>;
using OpDomainGradGrad = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpGradGrad<1, 1, SPACE_DIM>;
using OpDomainTimesScalarField = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesScalar<1>;
using OpDomainGradTimesVec = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, 1, SPACE_DIM>;
using OpDomainSource = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

using OpBoundaryMass = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, 1>;
using OpBoundaryTimeScalarField = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesScalar<1>;
using OpBoundarySource = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

// Capacity
constexpr double c = 1;
constexpr double k = 1;
constexpr double init_u = 0.;

/**
 * @brief Monitor solution
 *
 * This functions is called by TS solver at the end of each step. It is used
 * to output results to the hard drive.
 */
struct Monitor : public FEMethod {

  Monitor(SmartPetscObj<DM> dm, boost::shared_ptr<PostProcEle> post_proc)
      : dM(dm), postProc(post_proc){};

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  static constexpr int saveEveryNthStep = 1;

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;
    if (ts_step % saveEveryNthStep == 0) {
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc);
      CHKERR postProc->writeFile(
          "out_level_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
    }
    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcEle> postProc;
};

struct HeatEquation {
public:
  HeatEquation(MoFEM::Interface &m_field);

  // Declaration of the main function to run analysis
  MoFEMErrorCode runProgram();

private:
  // Declaration of other main functions called in runProgram()
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode initialCondition();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();

  // Main interfaces
  MoFEM::Interface &mField;

  // Object to mark boundary entities for the assembling of domain elements
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
};

HeatEquation::HeatEquation(MoFEM::Interface &m_field) : mField(m_field) {}

MoFEMErrorCode HeatEquation::readMesh() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();
  CHKERR mField.getInterface(simple);
  CHKERR simple->getOptions();
  CHKERR simple->loadFile();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HeatEquation::setupProblem() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();
  CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple->addBoundaryField("U", H1, AINSWORTH_LEGENDRE_BASE, 1);

  int order = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simple->setFieldOrder("U", order);

  CHKERR simple->setUp();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HeatEquation::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto integration_rule = [](int o_row, int o_col, int approx_order) {
    return 2 * approx_order;
  };

  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HeatEquation::initialCondition() {
  MoFEMFunctionBegin;

  // Get surface entities form blockset, set initial values in those
  // blocksets. To keep it simple, it is assumed that inital values are on
  // blockset 1
  if (mField.getInterface<MeshsetsManager>()->checkMeshset(1, BLOCKSET)) {
    Range inner_surface;
    CHKERR mField.getInterface<MeshsetsManager>()->getEntitiesByDimension(
        1, BLOCKSET, 2, inner_surface, true);
    if (!inner_surface.empty()) {
      Range inner_surface_verts;
      CHKERR mField.get_moab().get_connectivity(inner_surface,
                                                inner_surface_verts, false);
      CHKERR mField.getInterface<FieldBlas>()->setField(
          init_u, MBVERTEX, inner_surface_verts, "U");
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HeatEquation::boundaryCondition() {
  MoFEMFunctionBegin;

  auto bc_mng = mField.getInterface<BcManager>();
  auto *simple = mField.getInterface<Simple>();
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "ESSENTIAL",
                                        "U", 0, 0);

  auto &bc_map = bc_mng->getBcMapByBlockName();
  boundaryMarker = boost::make_shared<std::vector<char unsigned>>();
  for (auto b : bc_map) {
    if (std::regex_match(b.first, std::regex("(.*)ESSENTIAL(.*)"))) {
      boundaryMarker->resize(b.second->bcMarkers.size(), 0);
      for (int i = 0; i != b.second->bcMarkers.size(); ++i) {
        (*boundaryMarker)[i] |= b.second->bcMarkers[i];
      }
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HeatEquation::assembleSystem() {
  MoFEMFunctionBegin;

  auto add_domain_lhs_ops = [&](auto &pipeline) {
    AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pipeline, {H1});
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));
    pipeline.push_back(new OpDomainGradGrad(
        "U", "U", [](double, double, double) -> double { return k; }));
    auto get_c = [this](const double, const double, const double) {
      auto pipeline_mng = mField.getInterface<PipelineManager>();
      auto &fe_domain_lhs = pipeline_mng->getDomainLhsFE();
      return c * fe_domain_lhs->ts_a;
    };
    pipeline.push_back(new OpDomainMass("U", "U", get_c));
    pipeline.push_back(new OpUnSetBc("U"));
  };

  auto add_domain_rhs_ops = [&](auto &pipeline) {
    AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pipeline, {H1});
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));
    auto grad_u_at_gauss_pts = boost::make_shared<MatrixDouble>();
    auto dot_u_at_gauss_pts = boost::make_shared<VectorDouble>();
    pipeline.push_back(new OpCalculateScalarFieldGradient<SPACE_DIM>(
        "U", grad_u_at_gauss_pts));
    pipeline.push_back(
        new OpCalculateScalarFieldValuesDot("U", dot_u_at_gauss_pts));
    pipeline.push_back(new OpDomainGradTimesVec(
        "U", grad_u_at_gauss_pts,
        [](double, double, double) -> double { return k; }));
    pipeline.push_back(new OpDomainTimesScalarField(
        "U", dot_u_at_gauss_pts,
        [](const double, const double, const double) { return c; }));
    auto source_term = [&](const double x, const double y, const double z) {
      auto pipeline_mng = mField.getInterface<PipelineManager>();
      auto &fe_domain_lhs = pipeline_mng->getDomainRhsFE();
      const auto t = fe_domain_lhs->ts_t;
      return 1e1 * pow(M_E, -M_PI * M_PI * t) * sin(1. * M_PI * x) *
             sin(2. * M_PI * y);
    };
    pipeline.push_back(new OpDomainSource("U", source_term));
    pipeline.push_back(new OpUnSetBc("U"));
  };

  auto add_boundary_lhs_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpSetBc("U", false, boundaryMarker));
    pipeline.push_back(new OpBoundaryMass(
        "U", "U", [](const double, const double, const double) { return c; }));
    pipeline.push_back(new OpUnSetBc("U"));
  };
  auto add_boundary_rhs_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpSetBc("U", false, boundaryMarker));
    auto u_at_gauss_pts = boost::make_shared<VectorDouble>();
    auto boundary_function = [&](const double x, const double y,
                                 const double z) {
      auto pipeline_mng = mField.getInterface<PipelineManager>();
      auto &fe_rhs = pipeline_mng->getBoundaryRhsFE();
      const auto t = fe_rhs->ts_t;
      return 0;
      // abs(0.1 * pow(M_E, -M_PI * M_PI * t) * sin(2. * M_PI * x) *
      //     sin(3. * M_PI * y));
    };
    pipeline.push_back(new OpCalculateScalarFieldValues("U", u_at_gauss_pts));
    pipeline.push_back(new OpBoundaryTimeScalarField(
        "U", u_at_gauss_pts,
        [](const double, const double, const double) { return c; }));
    pipeline.push_back(new OpBoundarySource("U", boundary_function));
    pipeline.push_back(new OpUnSetBc("U"));
  };

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  add_domain_lhs_ops(pipeline_mng->getOpDomainLhsPipeline());
  add_domain_rhs_ops(pipeline_mng->getOpDomainRhsPipeline());
  add_boundary_lhs_ops(pipeline_mng->getOpBoundaryLhsPipeline());
  add_boundary_rhs_ops(pipeline_mng->getOpBoundaryRhsPipeline());

  MoFEMFunctionReturn(0);
}

struct CalcJacobian {
  static PetscErrorCode set(TS ts, PetscReal t, Vec u, Vec u_t, PetscReal a,
                            Mat A, Mat B, void *ctx) {
    MoFEMFunctionBegin;
    if (a != lastA) {
      lastA = a;
      CHKERR TsSetIJacobian(ts, t, u, u_t, a, A, B, ctx);
    }
    MoFEMFunctionReturn(0);
  }

private:
  static double lastA;
};

double CalcJacobian::lastA = 0;

MoFEMErrorCode HeatEquation::solveSystem() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  auto create_post_process_element = [&]() {
    auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    post_proc_fe->getOpPtrVector().push_back(new OpCalculateHOJac<2>(jac_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpSetHOInvJacToScalarBases<2>(H1, inv_jac_ptr));

    auto u_ptr = boost::make_shared<VectorDouble>();
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("U", u_ptr));

    using OpPPMap = OpPostProcMapInMoab<2, 2>;

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(post_proc_fe->getPostProcMesh(),
                    post_proc_fe->getMapGaussPts(),

                    {{"U", u_ptr}},

                    {}, {}, {}

                    )

    );

    return post_proc_fe;
  };

  auto set_time_monitor = [&](auto dm, auto solver) {
    MoFEMFunctionBegin;
    boost::shared_ptr<Monitor> monitor_ptr(
        new Monitor(dm, create_post_process_element()));
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

    if (is_pcfs == PETSC_TRUE) {
      auto bc_mng = mField.getInterface<BcManager>();
      auto name_prb = simple->getProblemName();
      auto is_all_bc = bc_mng->getBlockIS(name_prb, "ESSENTIAL", "U", 0, 0);
      int is_all_bc_size;
      CHKERR ISGetSize(is_all_bc, &is_all_bc_size);
      MOFEM_LOG("EXAMPLE", Sev::inform)
          << "Field split block size " << is_all_bc_size;
      CHKERR PCFieldSplitSetIS(pc, PETSC_NULL,
                               is_all_bc); // boundary block
    }
    MoFEMFunctionReturnHot(0);
  };

  /**
   *  That to work, you have to create solver, as follows,
      \code
      auto solver = // pipeline_mng->createTSIM( simple->getDM());
      \endcode
      That is explicitly use use Simple DM to create solver for DM. Pipeline
      menage by default creat copy of DM, in case several solvers are used the
      same DM.

      Alternatively you can get dm directly from the solver, i.e.
      \code
      DM ts_dm;
      CHKERR TSGetDM(solver, &ts_dm);
      CHKERR DMTSSetIJacobian(
        ts_dm, CalcJacobian::set, smartGetDMTsCtx(ts_dm).get());
      \endcode
  */
  auto set_user_ts_jacobian = [&](auto dm) {
    MoFEMFunctionBegin;
    CHKERR DMTSSetIJacobian(dm, CalcJacobian::set, smartGetDMTsCtx(dm).get());
    MoFEMFunctionReturn(0);
  };

  auto dm = simple->getDM();
  auto D = smartCreateDMVector(dm);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_FORWARD);

  auto solver = pipeline_mng->createTSIM(
      simple->getDM()); // Note DM is set as argument. If DM is not, internal
                        // copy of pipeline DM is created.
  CHKERR set_user_ts_jacobian(dm);
  CHKERR set_time_monitor(dm, solver);
  CHKERR TSSetSolution(solver, D);
  CHKERR TSSetFromOptions(solver);
  CHKERR set_fieldsplit_preconditioner(solver);
  CHKERR TSSetUp(solver);

  CHKERR TSSolve(solver, D);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HeatEquation::outputResults() {
  MoFEMFunctionBegin;

  // Processes to set output results are integrated in solveSystem()

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HeatEquation::runProgram() {
  MoFEMFunctionBegin;

  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR setIntegrationRules();
  CHKERR initialCondition();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR outputResults();

  MoFEMFunctionReturn(0);
}

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "EXAMPLE"));
  LogManager::setLog("EXAMPLE");
  MOFEM_LOG_TAG("EXAMPLE", "example")

  // Error handling
  try {
    // Register MoFEM discrete manager in PETSc
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    // Create MOAB instance
    moab::Core mb_instance;              // mesh database
    moab::Interface &moab = mb_instance; // mesh database interface

    // Create MoFEM instance
    MoFEM::Core core(moab);           // finite element database
    MoFEM::Interface &m_field = core; // finite element interface

    // Run the main analysis
    HeatEquation heat_problem(m_field);
    CHKERR heat_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}