/**
 * \file inital_diffusion.cpp
 * \example inital_diffusion.cpp
 *
 **/

#include <stdlib.h>
#include <cmath>
#include <BasicFiniteElements.hpp>

#define BOOST_MATH_GAUSS_NO_COMPUTE_ON_DEMAND

using namespace MoFEM;

static char help[] = "...\n\n";

template <int DIM> struct ElementsAndOps {};

//! [Define dimension]
constexpr int SPACE_DIM = 3; //< Space dimension of problem, mesh
//! [Define dimension]

using DomainEle = VolumeElementForcesAndSourcesCore;
using DomainEleOp = DomainEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

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

const double n = 1.44;   ///< refractive index of diffusive medium
const double c = 30.;    ///< speed of light (cm/ns)
const double v = c / n;  ///< phase velocity of light in medium (cm/ns)

double mu_a;          ///< absorption coefficient (cm^-1)
double mu_sp;         ///< scattering coefficient (cm^-1)
double D;

double slab_thickness;
double beam_radius; //< spot radius
double beam_centre_x;
double beam_centre_y;
double flux_magnitude = 1e3; ///< impulse magnitude
double initial_time;

char out_file_name[255] = "init_file.dat";;
int numHoLevels = 1;

PetscBool output_volume = PETSC_FALSE;

#include <boost/math/quadrature/gauss_kronrod.hpp>
using namespace boost::math::quadrature;

struct OpError : public DomainEleOp {
  OpError(boost::shared_ptr<VectorDouble> u_at_pts_ptr, double &l2_error)
      : DomainEleOp("PHOTON_FLUENCE_RATE", OPROW), uAtPtsPtr(u_at_pts_ptr),
        l2Error(l2_error) {
    // Only will be executed once on vertices
    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
  }

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
    const double A = 4. * D * v * initial_time;
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
      return gauss_kronrod<double, 15>::integrate(
          g, 0, 2 * M_PI, 0, std::numeric_limits<float>::epsilon());
    };

    return T * flux_magnitude *
           gauss_kronrod<double, 15>::integrate(
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
  CHKERR simple->addDomainField("PHOTON_FLUENCE_RATE", H1,
                                AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple->addBoundaryField("PHOTON_FLUENCE_RATE", H1,
                                  AINSWORTH_LEGENDRE_BASE, 1);

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
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-mu_a", &mu_a, PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-mu_sp", &mu_sp, PETSC_NULL);
  CHKERR PetscOptionsGetScalar(PETSC_NULL, "", "-initial_time", &initial_time,
                               PETSC_NULL);

  CHKERR PetscOptionsGetString(PETSC_NULL, "", "-output_file", out_file_name,
                               255, PETSC_NULL);
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-output_volume", &output_volume,
                             PETSC_NULL);

  D = 1. / (3. * (mu_a + mu_sp));                    

  MOFEM_LOG("INITIAL", Sev::inform) << "Refractive index: " << n;
  MOFEM_LOG("INITIAL", Sev::inform) << "Speed of light (cm/ns): " << c;
  MOFEM_LOG("INITIAL", Sev::inform)
      << "Phase velocity in medium (cm/ns): " << v;
  MOFEM_LOG("INITIAL", Sev::inform)
      << "Absorption coefficient (cm^-1): " << mu_a;
  MOFEM_LOG("INITIAL", Sev::inform)
      << "Scattering coefficient (cm^-1): " << mu_sp;
  MOFEM_LOG("INITIAL", Sev::inform) << "Diffusion coefficient D : " << D;
  MOFEM_LOG("INITIAL", Sev::inform) << "Impulse magnitude: " << flux_magnitude;
  MOFEM_LOG("INITIAL", Sev::inform) << "Compute time (ns): " << initial_time;
  MOFEM_LOG("INITIAL", Sev::inform) << "Slab thickness: " << slab_thickness;

  int order = 2;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);

  MOFEM_LOG("INITIAL", Sev::inform) << "Approximation order: " << order;

  CHKERR simple->setFieldOrder("PHOTON_FLUENCE_RATE", order);

  // if (numHoLevels > 0) {

  //   Range ho_ents;
  //   for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
  //     if (it->getName().compare(0, 3, "CAM") == 0) {
  //       CHKERR mField.get_moab().get_entities_by_dimension(it->getMeshset(), 2,
  //                                                          ho_ents, true);
  //     }
  //   }

  //   EntityHandle meshset;
  //   CHKERR mField.get_moab().create_meshset(MESHSET_SET, meshset);
  //   CHKERR mField.get_moab().add_entities(meshset, ho_ents);
  //   std::string field_name;
  //   field_name = "out_test_" +
  //                boost::lexical_cast<std::string>(mField.get_comm_rank()) +
  //                ".vtk";
  //   CHKERR mField.get_moab().write_file(field_name.c_str(), "VTK", "", &meshset,
  //                                       1);
  //   CHKERR mField.get_moab().delete_entities(&meshset, 1);

  //   CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(ho_ents);

  //   CHKERR simple->setFieldOrder("PHOTON_FLUENCE_RATE", order + 1, &ho_ents);

  //   CHKERR mField.getInterface<CommInterface>()->synchroniseFieldEntities(
  //       "PHOTON_FLUENCE_RATE");
  // }

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

  auto *simple = mField.getInterface<Simple>();

  // Get boundary edges marked in block named "BOUNDARY_CONDITION"
  Range boundary_ents;
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
    std::string entity_name = it->getName();
    if (entity_name.compare(0, 3, "INT") == 0) {
      CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), 1,
                                                 boundary_ents, true);
    }
  }
  // Add vertices to boundary entities
  Range boundary_verts;
  CHKERR mField.get_moab().get_connectivity(boundary_ents, boundary_verts,
                                            true);
  boundary_ents.merge(boundary_verts);

  // Remove DOFs as homogeneous boundary condition is used
  CHKERR mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
      simple->getProblemName(), "PHOTON_FLUENCE_RATE", boundary_ents);

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
        "PHOTON_FLUENCE_RATE", "PHOTON_FLUENCE_RATE",
        [](const double, const double, const double) { return 1; }));
  };

  auto add_domain_rhs_ops = [&](auto &pipeline) {
    pipeline.push_back(
        new OpDomainSource("PHOTON_FLUENCE_RATE", sourceFunction));
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
  // CHKERR KSPSetUp(solver);

  auto dm = simple->getDM();
  auto X = smartCreateDMVector(dm);
  auto F = smartVectorDuplicate(X);

  MOFEM_LOG("INITIAL", Sev::inform) << "Solver start";
  CHKERR KSPSolve(solver, F, X);
  CHKERR VecGhostUpdateBegin(X, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(X, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, X, INSERT_VALUES, SCATTER_REVERSE);

  MOFEM_LOG("INITIAL", Sev::inform)
      << "writing vector in binary to " << out_file_name << " ...";
  PetscViewer viewer;
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, out_file_name, FILE_MODE_WRITE,
                        &viewer);
  VecView(X, viewer);
  PetscViewerDestroy(&viewer);

  MOFEM_LOG("INITIAL", Sev::inform) << "Solver done";
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PhotonDiffusion::outputResults() {
  MoFEMFunctionBegin;
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();
  auto post_proc_fe = boost::make_shared<PostProcEle>(mField);

  auto u_ptr = boost::make_shared<VectorDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues("U", u_ptr));

  using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

  post_proc_fe->getOpPtrVector().push_back(

      new OpPPMap(

          post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(), 
          
          {{"PHOTON_FLUENCE_RATE", u_ptr}},

          {},

          {},

          {})

  );     

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
  if (output_volume)
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