/**
 * \file inital_diffusion.cpp
 * \example inital_diffusion.cpp
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

double n = 1.44;             ///< refractive index of diffusive medium
double c = 30.;              ///< speed of light (cm/ns)
double v = c / n;            ///< phase velocity of light in medium (cm/ns)
double mu_a = 0.09;          ///< absorption coefficient (cm^-1)
double mu_sp = 16.5;         ///< scattering coefficient (cm^-1)
double flux_magnitude = 1e3; ///< impulse magnitude

double slab_thickness = 5.0;
double beam_radius = 2.5; //< spot radius
double beam_centre_x = 0.0;
double beam_centre_y = 0.0;

double initial_time = 0.1;

double D = 1. / (3. * (mu_a + mu_sp));

#include <boost/math/quadrature/gauss_kronrod.hpp>
using namespace boost::math::quadrature;

struct OpError : public DomainEleOp {
  OpError(boost::shared_ptr<VectorDouble> u_at_pts_ptr, double &l2_error)
      : DomainEleOp("U", OPROW), uAtPtsPtr(u_at_pts_ptr), l2Error(l2_error) {
    // Only will be executed once on vertices
    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
  }

  //   MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
  //     MoFEMFunctionBegin;
  //     SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "Not yet implemented");
  //     MogEMFunctionReturn(0);
  //   }

private:
  boost::shared_ptr<VectorDouble> uAtPtsPtr;
  double &l2Error;
};

struct PhotonDiffusion {
public:
  PhotonDiffusion(MoFEM::Interface &m_field);

  // Declaration of the main function to run analysis
  MoFEMErrorCode runProgram();

  /**
   * @brief Pulse is infinitely short.
   *
   * \note It is good approximation of pulse in femtosecond scale. To make it
   * longer one can apply third integral over time.
   *
   * \note Note analysis in photon_diffusion is shifted in time by initial_time.
   *
   * @param x
   * @param y
   * @param z
   * @return double
   */
  static double sourceFunction(const double x, const double y, const double z) {
    const double A = 4 * D * v * initial_time;
    const double T =
        (v / pow(M_PI * A, 3. / 2.)) * exp(-mu_a * v * initial_time);

    auto phi_pulse = [&](const double r_s, const double phi_s) {
      const double xs = r_s * cos(phi_s);
      const double ys = r_s * sin(phi_s);
      const double xp = x - xs - beam_centre_x;
      const double yp = y - ys - beam_centre_y;
      const double zp1 = z + slab_thickness / 2. - 1. / mu_sp;
      const double zp2 = z + slab_thickness / 2. + 1. / mu_sp;
      const double P1 = xp * xp + yp * yp + zp1 * zp1;
      const double P2 = xp * xp + yp * yp + zp2 * zp2;
      return r_s * (exp(-P1 / A) - exp(-P2 / A));
    };

    auto f = [&](const double r_s) {
      auto g = [&](const double phi_s) { return phi_pulse(r_s, phi_s); };
      return gauss_kronrod<double, 31>::integrate(
          g, 0, 2 * M_PI, 0, std::numeric_limits<float>::epsilon());
    };

    return T * flux_magnitude *
           gauss_kronrod<double, 31>::integrate(
               f, 0, beam_radius, 0, std::numeric_limits<float>::epsilon());
  };

private:
  // Declaration of other main functions called in runProgram()
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode initialCondition();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode checkResults();
  MoFEMErrorCode outputResults();

  // Main interfaces
  MoFEM::Interface &mField;

  // Object to mark boundary entities for the assembling of domain elements
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
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

  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-flux_magnitude",
                               &flux_magnitude, PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-slab_thickness",
                               &slab_thickness, PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-beam_radius", &beam_radius,
                               PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-beam_centre_x", &beam_centre_x,
                               PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-beam_centre_y", &beam_centre_y,
                               PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-initial_time", &initial_time,
                               PETSC_NULL);

  MOFEM_LOG("INITIAL", Sev::inform) << "Refractive index: " << n;
  MOFEM_LOG("INITIAL", Sev::inform) << "Speed of light (cm/ns): " << c;
  MOFEM_LOG("INITIAL", Sev::inform)
      << "Phase velocity in medium (cm/ns): " << v;
  MOFEM_LOG("INITIAL", Sev::inform)
      << "Absorption coefficient (cm^-1): " << mu_a;
  MOFEM_LOG("INITIAL", Sev::inform)
      << "Scattering coefficient (cm^-1): " << mu_sp;
  MOFEM_LOG("INITIAL", Sev::inform) << "Impulse magnitude: " << flux_magnitude;
  MOFEM_LOG("INITIAL", Sev::inform) << "Initial time (ns): " << initial_time;

  int order = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);

  MOFEM_LOG("INITIAL", Sev::inform) << "Approximation order: " << order;

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
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::assembleSystem() {
  MoFEMFunctionBegin;

  auto bc_mng = mField.getInterface<BcManager>();
  auto &bc_map = bc_mng->getBcMapByBlockName();

  auto add_domain_base_ops = [&](auto &pipeline) {
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto det_ptr = boost::make_shared<VectorDouble>();
    pipeline.push_back(new OpCalculateHOJacVolume(jac_ptr));
    pipeline.push_back(new OpInvertMatrix<3>(jac_ptr, det_ptr, nullptr));
    pipeline.push_back(new OpSetHOWeights(det_ptr));
  };

  auto add_domain_lhs_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpDomainMass(
        "U", "U", [](const double, const double, const double) { return 1; }));
  };

  auto add_domain_rhs_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpDomainSource("U", sourceFunction));
  };

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  add_domain_base_ops(pipeline_mng->getOpDomainLhsPipeline());
  add_domain_base_ops(pipeline_mng->getOpDomainRhsPipeline());
  add_domain_lhs_ops(pipeline_mng->getOpDomainLhsPipeline());
  add_domain_rhs_ops(pipeline_mng->getOpDomainRhsPipeline());

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::solveSystem() {
  MoFEMFunctionBegin;
  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  auto solver = pipeline_mng->createKSP();

  CHKERR KSPSetFromOptions(solver);
  //CHKERR KSPSetUp(solver);

  auto dm = simple->getDM();
  auto D = smartCreateDMVector(dm);
  auto F = smartVectorDuplicate(D);

  MOFEM_LOG("INITIAL", Sev::inform) << "Solver start";
  CHKERR KSPSolve(solver, F, D);
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MOFEM_LOG("INITIAL", Sev::inform)
      << "writing vector in binary to vector.dat ...";
  PetscViewer viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, "initial_vector.dat", FILE_MODE_WRITE,
                        &viewer);
  VecView(D, viewer);
  PetscViewerDestroy(&viewer);

  MOFEM_LOG("INITIAL", Sev::inform) << "Solver done";
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::checkResults() {
  MoFEMFunctionBegin;
  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getOpDomainRhsPipeline().clear();
  auto u_vals_at_gauss_pts = boost::make_shared<VectorDouble>();
  double l2_error = 0;
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateScalarFieldValues("U", u_vals_at_gauss_pts));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpError(u_vals_at_gauss_pts, l2_error));
  CHKERR pipeline_mng->loopFiniteElements();
  MOFEM_LOG("INITIAL", Sev::inform) << "L2 error " << l2_error;
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::outputResults() {
  MoFEMFunctionBegin;
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
  post_proc_fe->generateReferenceElementMesh();
  post_proc_fe->addFieldValuesPostProc("U");
  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_initial.h5m");
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
  // CHKERR checkResults();
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
      LogManager::createSink(LogManager::getStrmWorld(), "INITIAL"));
  LogManager::setLog("INITIAL");
  MOFEM_LOG_TAG("INITIAL", "initial_diffusion")

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