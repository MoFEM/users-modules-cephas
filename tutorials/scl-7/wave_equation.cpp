/**
 * \file wave_equation.cpp
 * \example wave_equation.cpp
 *
 * \brief Solve the time-dependent Wave Equation
 \f[
 \begin{aligned}
\frac{\partial u(\mathbf{x}, t)}{\partial t}-\Delta u(\mathbf{x}, t)
&=f(\mathbf{x}, t) & & \forall \mathbf{x} \in \Omega, t \in(0, T), \\
u(\mathbf{x}, 0) &=u_{0}(\mathbf{x}) & & \forall \mathbf{x} \in \Omega, \\
u(\mathbf{x}, t) &=g(\mathbf{x}, t) & & \forall \mathbf{x} \in \partial \Omega,
t \in(0, T). \end{aligned}
 \f]
 **/

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
using PostProcEle = PostProcFaceOnRefinedMesh;

using OpDomainMass = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, 1>;
using OpDomainGradGrad = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpGradGrad<1, 1, SPACE_DIM>;
using OpDomainTimesScalarField = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesScalarField<1>;
using OpDomainGradTimesVec = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, 1, SPACE_DIM>;

using OpBoundaryMass = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, 1>;
using OpBoundaryTimeScalarField = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesScalarField<1>;
using OpBoundarySource = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

constexpr double wave_speed2 = 1;

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

struct WaveEquation {
public:
  WaveEquation(MoFEM::Interface &m_field);

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

WaveEquation::WaveEquation(MoFEM::Interface &m_field) : mField(m_field) {}

MoFEMErrorCode WaveEquation::readMesh() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();
  CHKERR mField.getInterface(simple);
  CHKERR simple->getOptions();
  CHKERR simple->loadFile();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode WaveEquation::setupProblem() {
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

MoFEMErrorCode WaveEquation::setIntegrationRules() {
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

MoFEMErrorCode WaveEquation::initialCondition() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode WaveEquation::boundaryCondition() {
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

MoFEMErrorCode WaveEquation::assembleSystem() {
  MoFEMFunctionBegin;

  auto add_domain_base_ops = [&](auto &pipeline) {
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(new OpCalculateHOJac<2>(jac_ptr));
    pipeline.push_back(new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline.push_back(new OpSetHOInvJacToScalarBases<2>(H1, inv_jac_ptr));
    pipeline.push_back(new OpSetHOWeights(det_ptr));
  };

  auto add_domain_lhs_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));
    pipeline.push_back(new OpDomainGradGrad(
        "U", "U", [](double, double, double) -> double { return 1; }));
    auto get_c = [this](const double, const double, const double) {
      auto pipeline_mng = mField.getInterface<PipelineManager>();
      auto &fe_domain_lhs = pipeline_mng->getDomainLhsFE();
      return wave_speed2 * fe_domain_lhs->ts_aa;
    };
    pipeline.push_back(new OpDomainMass("U", "U", get_c));
    pipeline.push_back(new OpUnSetBc("U"));
  };

  auto add_domain_rhs_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));
    auto grad_u_at_gauss_pts = boost::make_shared<MatrixDouble>();
    auto dot2_u_at_gauss_pts = boost::make_shared<VectorDouble>();
    pipeline.push_back(new OpCalculateScalarFieldGradient<SPACE_DIM>(
        "U", grad_u_at_gauss_pts));
    pipeline.push_back(
        new OpCalculateScalarFieldValuesDotDot("U", dot2_u_at_gauss_pts));
    pipeline.push_back(new OpDomainGradTimesVec(
        "U", grad_u_at_gauss_pts,
        [](double, double, double) -> double { return 1; }));
    pipeline.push_back(new OpDomainTimesScalarField(
        "U", dot2_u_at_gauss_pts,
        [](const double, const double, const double) { return wave_speed2; }));
    pipeline.push_back(new OpUnSetBc("U"));
  };

  auto add_boundary_base_ops = [&](auto &pipeline) {};

  auto add_lhs_base_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpSetBc("U", false, boundaryMarker));
    pipeline.push_back(new OpBoundaryMass(
        "U", "U", [](const double, const double, const double) { return 1; }));
    pipeline.push_back(new OpUnSetBc("U"));
  };

  auto add_rhs_base_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpSetBc("U", false, boundaryMarker));
    auto u_at_gauss_pts = boost::make_shared<VectorDouble>();
    auto boundary_function = [&](const double x, const double y,
                                 const double z) {
      auto pipeline_mng = mField.getInterface<PipelineManager>();
      auto &fe_rhs = pipeline_mng->getBoundaryRhsFE();
      const double t = fe_rhs->ts_t;
      if ((t <= 0.5) && (x < 0.) && (y > -1. / 3) && (y < 1. / 3))
        return sin(4 * M_PI * t);
      else
        return 0.;
    };
    pipeline.push_back(new OpCalculateScalarFieldValues("U", u_at_gauss_pts));
    pipeline.push_back(new OpBoundaryTimeScalarField(
        "U", u_at_gauss_pts,
        [](const double, const double, const double) { return 1; }));
    pipeline.push_back(new OpBoundarySource("U", boundary_function));
    pipeline.push_back(new OpUnSetBc("U"));
  };

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  add_domain_base_ops(pipeline_mng->getOpDomainLhsPipeline());
  add_domain_base_ops(pipeline_mng->getOpDomainRhsPipeline());
  add_domain_lhs_ops(pipeline_mng->getOpDomainLhsPipeline());
  add_domain_rhs_ops(pipeline_mng->getOpDomainRhsPipeline());

  add_boundary_base_ops(pipeline_mng->getOpBoundaryLhsPipeline());
  add_boundary_base_ops(pipeline_mng->getOpBoundaryRhsPipeline());
  add_lhs_base_ops(pipeline_mng->getOpBoundaryLhsPipeline());
  add_rhs_base_ops(pipeline_mng->getOpBoundaryRhsPipeline());

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode WaveEquation::solveSystem() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  auto create_post_process_element = [&]() {
    auto post_froc_fe = boost::make_shared<PostProcEle>(mField);
    post_froc_fe->generateReferenceElementMesh();
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    post_froc_fe->getOpPtrVector().push_back(new OpCalculateHOJac<2>(jac_ptr));
    post_froc_fe->getOpPtrVector().push_back(
        new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
    post_froc_fe->getOpPtrVector().push_back(
        new OpSetHOInvJacToScalarBases<2>(H1, inv_jac_ptr));
    post_froc_fe->addFieldValuesPostProc("U");
    return post_froc_fe;
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

  auto dm = simple->getDM();
  auto D = smartCreateDMVector(dm);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_FORWARD);

  auto ts = pipeline_mng->createTSIM2();

  CHKERR TSSetSolution(ts, D);
  auto DD = smartVectorDuplicate(D);
  CHKERR TS2SetSolution(ts, D, DD);
  CHKERR set_time_monitor(dm, ts);
  CHKERR TSSetFromOptions(ts);
  CHKERR set_fieldsplit_preconditioner(ts);
  CHKERR TSSetUp(ts);
  CHKERR TSSolve(ts, NULL);

  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode WaveEquation::outputResults() {
  MoFEMFunctionBegin;

  // Processes to set output results are integrated in solveSystem()

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode WaveEquation::runProgram() {
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
    WaveEquation heat_problem(m_field);
    CHKERR heat_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}