/**
 * \file photon_diffusion.cpp
 * \example photon_diffusion.cpp
 *
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
constexpr int SPACE_DIM = 3; //< Space dimension of problem, mesh
//! [Define dimension]

using DomainEle = VolumeElementForcesAndSourcesCore;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = FaceElementForcesAndSourcesCore;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcVolumeOnRefinedMesh;

using OpDomainMass = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, 1>;
using OpDomainGradGrad = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpGradGrad<1, 1, SPACE_DIM>;
using OpDomainTimesScalarField = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesScalarField<1>;
using OpDomainGradTimesVec = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, 1, SPACE_DIM>;
using OpDomainSource = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

using OpBoundaryMass = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, 1>;
using OpBoundaryTimeScalarField = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesScalarField<1>;
using OpBoundarySource = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

constexpr double c = 30.; ///< speed of light

constexpr double mu_a = 0.09;
constexpr double mu_sp = 16.5;
constexpr double h = 0.5;   ///< convective heat coefficient
constexpr double flux = 1e3; ///< 0.5 mW mm^-2
constexpr double duration = 0.05;

constexpr double D = 1. / (3. * (mu_a + mu_sp));
constexpr double inv_c = 1. / c;

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

  static constexpr int saveEveryNthStep = 2;

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

struct PhotonDiffusion {
public:
  PhotonDiffusion(MoFEM::Interface &m_field);

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

  boost::shared_ptr<FEMethod> domianLhsFEPtr;
  boost::shared_ptr<FEMethod> boundaryLhsFEPtr;
  boost::shared_ptr<FEMethod> boundaryRhsFEPtr;
};

PhotonDiffusion::PhotonDiffusion(MoFEM::Interface &m_field) : mField(m_field) {}

MoFEMErrorCode PhotonDiffusion::readMesh() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();
  CHKERR mField.getInterface(simple);
  CHKERR simple->getOptions();
  CHKERR simple->loadFile();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::setupProblem() {
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

MoFEMErrorCode PhotonDiffusion::setIntegrationRules() {
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

MoFEMErrorCode PhotonDiffusion::initialCondition() {
  MoFEMFunctionBegin;

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::boundaryCondition() {
  MoFEMFunctionBegin;
  auto bc_mng = mField.getInterface<BcManager>();
  auto *simple = mField.getInterface<Simple>();
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "MIX", "U", 0,
                                        0, false);
  CHKERR bc_mng->pushMarkDOFsOnEntities(simple->getProblemName(), "SPOT", "U",
                                        0, 0, false);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::assembleSystem() {
  MoFEMFunctionBegin;

  auto bc_mng = mField.getInterface<BcManager>();
  auto &bc_map = bc_mng->getBcMapByBlockName();

  auto add_domain_base_ops = [&](auto &pipeline) {
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    auto det_ptr = boost::make_shared<VectorDouble>();
    pipeline.push_back(new OpCalculateHOJacVolume(jac_ptr));
    pipeline.push_back(new OpInvertMatrix<3>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline.push_back(new OpSetHOInvJacToScalarBases(H1, inv_jac_ptr));
    pipeline.push_back(new OpSetHOWeights(det_ptr));
  };

  auto add_domain_lhs_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpDomainGradGrad(
        "U", "U", [](double, double, double) -> double { return D; }));
    auto get_mass_coefficient = [&](const double, const double, const double) {
      return inv_c * domianLhsFEPtr->ts_a + mu_a;
    };
    pipeline.push_back(new OpDomainMass("U", "U", get_mass_coefficient));
  };

  auto add_domain_rhs_ops = [&](auto &pipeline) {
    auto grad_u_at_gauss_pts = boost::make_shared<MatrixDouble>();
    auto u_at_gauss_pts = boost::make_shared<VectorDouble>();
    auto dot_u_at_gauss_pts = boost::make_shared<VectorDouble>();
    pipeline.push_back(new OpCalculateScalarFieldGradient<SPACE_DIM>(
        "U", grad_u_at_gauss_pts));
    pipeline.push_back(new OpCalculateScalarFieldValues("U", u_at_gauss_pts));
    pipeline.push_back(
        new OpCalculateScalarFieldValuesDot("U", dot_u_at_gauss_pts));
    pipeline.push_back(new OpDomainGradTimesVec(
        "U", grad_u_at_gauss_pts,
        [](double, double, double) -> double { return D; }));
    pipeline.push_back(new OpDomainTimesScalarField(
        "U", dot_u_at_gauss_pts,
        [](const double, const double, const double) { return inv_c; }));
    pipeline.push_back(new OpDomainTimesScalarField(
        "U", u_at_gauss_pts,
        [](const double, const double, const double) { return mu_a; }));
    auto source_term = [&](const double, const double, const double) {
      return 0;
    };
    pipeline.push_back(new OpDomainSource("U", source_term));
  };

  auto add_boundary_base_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpSetHOWeigthsOnFace());
  };

  auto add_lhs_base_ops = [&](auto &pipeline) {
    for (auto b : bc_map) {
      if (std::regex_match(b.first, std::regex("(.*)MIX(.*)"))) {
        pipeline.push_back(new OpBoundaryMass(
            "U", "U",

            [](const double, const double, const double) { return h; },

            b.second->getBcEdgesPtr()));
      }
    }
    for (auto b : bc_map) {
      if (std::regex_match(b.first, std::regex("(.*)SPOT(.*)"))) {
        pipeline.push_back(new OpBoundaryMass(
            "U", "U",

            [&](const double, const double, const double) {
              if (boundaryRhsFEPtr->ts_t > duration)
                return h;
              else
                return 0.;
            },

            b.second->getBcEdgesPtr()));
      }
    }
  };

  auto add_rhs_base_ops = [&](auto &pipeline) {
    auto u_at_gauss_pts = boost::make_shared<VectorDouble>();
    pipeline.push_back(new OpCalculateScalarFieldValues("U", u_at_gauss_pts));
    for (auto b : bc_map) {
      if (std::regex_match(b.first, std::regex("(.*)MIX(.*)"))) {
        pipeline.push_back(new OpBoundaryTimeScalarField(
            "U", u_at_gauss_pts,

            [](const double, const double, const double) { return h; },

            b.second->getBcEdgesPtr()));
      }
    }
    for (auto b : bc_map) {
      if (std::regex_match(b.first, std::regex("(.*)SPOT(.*)"))) {
        pipeline.push_back(new OpBoundaryTimeScalarField(
            "U", u_at_gauss_pts,

            [&](const double, const double, const double) {
              if (boundaryRhsFEPtr->ts_t > duration)
                return h;
              else
                return 0.;
            },

            b.second->getBcEdgesPtr()));
      }
    }
    for (auto b : bc_map) {
      if (std::regex_match(b.first, std::regex("(.*)SPOT(.*)"))) {
        pipeline.push_back(new OpBoundarySource(
            "U",

            [&](const double, const double, const double) {
              if (boundaryRhsFEPtr->ts_t < duration)
                return -flux;
              else
                return 0.;
            },

            b.second->getBcEdgesPtr()));
      }
    }
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

  domianLhsFEPtr = pipeline_mng->getDomainLhsFE();
  boundaryLhsFEPtr = pipeline_mng->getBoundaryLhsFE();
  boundaryRhsFEPtr = pipeline_mng->getBoundaryRhsFE();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::solveSystem() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  auto create_post_process_element = [&]() {
    auto post_froc_fe = boost::make_shared<PostProcEle>(mField);
    post_froc_fe->generateReferenceElementMesh();
    post_froc_fe->addFieldValuesPostProc("U");
    post_froc_fe->addFieldValuesGradientPostProc("U");
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

  auto dm = simple->getDM();
  auto D = smartCreateDMVector(dm);

  // MOFEM_LOG("PHOTON", Sev::inform)
  //     << "reading vector in binary from vector.dat ...";
  // PetscViewer viewer;
  // PetscViewerBinaryOpen(PETSC_COMM_WORLD, "initial_vector.dat", FILE_MODE_READ,
  //                       &viewer);
  // VecLoad(D, viewer);

  // CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  auto solver = pipeline_mng->createTS();
  CHKERR TSSetSolution(solver, D);
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

MoFEMErrorCode PhotonDiffusion::outputResults() {
  MoFEMFunctionBegin;

  // Processes to set output results are integrated in solveSystem()

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::runProgram() {
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
      LogManager::createSink(LogManager::getStrmWorld(), "PHOTON"));
  LogManager::setLog("PHOTON");
  MOFEM_LOG_TAG("PHOTON", "photon")

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
    PhotonDiffusion heat_problem(m_field);
    CHKERR heat_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}