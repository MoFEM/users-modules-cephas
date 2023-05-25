/**
 * \file free_surface.cpp
 * \example free_surface.cpp
 *
 * Using PipelineManager interface calculate the divergence of base functions,
 * and integral of flux on the boundary. Since the h-div space is used, volume
 * integral and boundary integral should give the same result.
 *
 * Implementation based on \cite Lovric2019-qn
 */

#include <MoFEM.hpp>
#include <petsc/private/petscimpl.h>

using namespace MoFEM;

static char help[] = "...\n\n";

#ifdef PYTHON_INIT_SURFACE
#include <boost/python.hpp>
#include <boost/python/def.hpp>
namespace bp = boost::python;

struct SurfacePython {
  SurfacePython() = default;
  virtual ~SurfacePython() = default;

  MoFEMErrorCode surfaceInit(const std::string py_file) {
    MoFEMFunctionBegin;
    try {

      // create main module
      auto main_module = bp::import("__main__");
      mainNamespace = main_module.attr("__dict__");
      bp::exec_file(py_file.c_str(), mainNamespace, mainNamespace);
      // create a reference to python function
      surfaceFun = mainNamespace["surface"];

    } catch (bp::error_already_set const &) {
      // print all other errors to stderr
      PyErr_Print();
      CHK_THROW_MESSAGE(MOFEM_OPERATION_UNSUCCESSFUL, "Python error");
    }
    MoFEMFunctionReturn(0);
  };

  MoFEMErrorCode evalSurface(

      double x, double y, double z, double eta, double &s

  ) {
    MoFEMFunctionBegin;
    try {

      // call python function
      s = bp::extract<double>(surfaceFun(x, y, z, eta));

    } catch (bp::error_already_set const &) {
      // print all other errors to stderr
      PyErr_Print();
      CHK_THROW_MESSAGE(MOFEM_OPERATION_UNSUCCESSFUL, "Python error");
    }
    MoFEMFunctionReturn(0);
  }

private:
  bp::object mainNamespace;
  bp::object surfaceFun;
};

static boost::weak_ptr<SurfacePython> surfacePythonWeakPtr;

#endif

int coord_type = EXECUTABLE_COORD_TYPE;

constexpr int BASE_DIM = 1;
constexpr int SPACE_DIM = 2;
constexpr int U_FIELD_DIM = SPACE_DIM;

constexpr AssemblyType A = AssemblyType::PETSC; //< selected assembly type
constexpr IntegrationType I =
    IntegrationType::GAUSS;                     //< selected integration type

template <int DIM>
using ElementsAndOps = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>;

using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomianParentEle = ElementsAndOps<SPACE_DIM>::DomianParentEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryParentEle = ElementsAndOps<SPACE_DIM>::BoundaryParentEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using SideEle = ElementsAndOps<SPACE_DIM>::FaceSideEle;
using SideOp = SideEle::UserDataOperator;

using PostProcEleDomain = PostProcBrokenMeshInMoab<DomainEle>;
using PostProcEleDomainCont = PostProcBrokenMeshInMoabBaseCont<DomainEle>;
using PostProcEleBdyCont = PostProcBrokenMeshInMoabBaseCont<BoundaryEle>;

using EntData = EntitiesFieldData::EntData;

using AssemblyDomainEleOp = FormsIntegrators<DomainEleOp>::Assembly<A>::OpBase;
using AssemblyBoundaryEleOp =
    FormsIntegrators<BoundaryEleOp>::Assembly<A>::OpBase;

using OpDomainMassU = FormsIntegrators<DomainEleOp>::Assembly<A>::BiLinearForm<
    I>::OpMass<BASE_DIM, U_FIELD_DIM>;
using OpDomainMassH = FormsIntegrators<DomainEleOp>::Assembly<A>::BiLinearForm<
    I>::OpMass<BASE_DIM, 1>;
using OpDomainMassP = OpDomainMassH;
using OpDomainMassG = OpDomainMassH;
using OpBoundaryMassL = FormsIntegrators<BoundaryEleOp>::Assembly<
    A>::BiLinearForm<I>::OpMass<BASE_DIM, 1>;

using OpDomainAssembleVector = FormsIntegrators<DomainEleOp>::Assembly<
    A>::LinearForm<I>::OpBaseTimesVector<BASE_DIM, SPACE_DIM, 1>;
using OpDomainAssembleScalar = FormsIntegrators<DomainEleOp>::Assembly<
    A>::LinearForm<I>::OpBaseTimesScalar<BASE_DIM>;
using OpBoundaryAssembleScalar = FormsIntegrators<BoundaryEleOp>::Assembly<
    A>::LinearForm<I>::OpBaseTimesScalar<BASE_DIM>;

template <CoordinateTypes COORD_TYPE>
using OpMixScalarTimesDiv = FormsIntegrators<DomainEleOp>::Assembly<
    A>::BiLinearForm<I>::OpMixScalarTimesDiv<SPACE_DIM, COORD_TYPE>;

// Flux is applied by Lagrange Multiplie
using BoundaryNaturalBC = NaturalBC<BoundaryEleOp>::Assembly<A>::LinearForm<I>;
using OpFluidFlux =
    BoundaryNaturalBC::OpFlux<NaturalMeshsetType<BLOCKSET>, 1, 1>;

FTensor::Index<'i', SPACE_DIM> i;
FTensor::Index<'j', SPACE_DIM> j;
FTensor::Index<'k', SPACE_DIM> k;
FTensor::Index<'l', SPACE_DIM> l;

constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();

// mesh refinement
int order = 3;          ///< approximation order
int nb_levels = 4;      //< number of refinement levels
int refine_overlap = 4; //< mesh overlap while refine

constexpr bool debug = true;

auto get_start_bit = []() {
  return nb_levels + 1;
}; //< first refinement level for computational mesh
auto get_current_bit = []() {
  return 2 * get_start_bit() + 1;
}; ///< dofs bit used to do calculations
auto get_skin_parent_bit = []() { return 2 * get_start_bit() + 2; };
auto get_skin_child_bit = []() { return 2 * get_start_bit() + 3; };
auto get_projection_bit = []() { return 2 * get_start_bit() + 4; };
auto get_skin_projection_bit = []() { return 2 * get_start_bit() + 5; };

// FIXME: Set parameters from command line

// Physical parameters
constexpr double a0 = 980;
constexpr double rho_m = 0.998;
constexpr double mu_m = 0.010101 * 1e2;
constexpr double rho_p = 0.0012;
constexpr double mu_p = 0.000182 * 1e2;
constexpr double lambda = 73; ///< surface tension
constexpr double W = 0.25;

// Model parameters
constexpr double h = 0.01 / 8; // mesh size
constexpr double eta = h;
constexpr double eta2 = eta * eta;

// Numerical parameters
constexpr double md = 1e-2;
constexpr double eps = 1e-12;
constexpr double tol = std::numeric_limits<float>::epsilon();

constexpr double rho_ave = (rho_p + rho_m) / 2;
constexpr double rho_diff = (rho_p - rho_m) / 2;
constexpr double mu_ave = (mu_p + mu_m) / 2;
constexpr double mu_diff = (mu_p - mu_m) / 2;

const double kappa = (3. / (4. * std::sqrt(2. * W))) * (lambda / eta);

auto integration_rule = [](int, int, int) { return 2 * order + 1; };

auto cylindrical = [](const double r) {
  if (coord_type == CYLINDRICAL)
    return 2 * M_PI * r;
  else
    return 1.;
};

auto wetting_angle_sub_stepping = [](auto ts_step) {
  constexpr int sub_stepping = 16;
  return std::min(1., static_cast<double>(ts_step) / sub_stepping);
};

// cut off function

auto my_max = [](const double x) { return (x - 1 + std::abs(x + 1)) / 2; };
auto my_min = [](const double x) { return (x + 1 - std::abs(x - 1)) / 2; };
auto cut_off = [](const double h) { return my_max(my_min(h)); };
auto d_cut_off = [](const double h) {
  if (h >= -1 && h < 1)
    return 1.;
  else
    return 0.;
};

auto phase_function = [](const double h, const double diff, const double ave) {
  return diff * cut_off(h) + ave;
};

auto d_phase_function_h = [](const double h, const double diff) {
  return diff * d_cut_off(h);
};

// order function (free energy)

auto get_f = [](const double h) { return 4 * W * h * (h * h - 1); };
auto get_f_dh = [](const double h) { return 4 * W * (3 * h * h - 1); };

auto get_M0 = [](auto h) { return md; };
auto get_M0_dh = [](auto h) { return 0; };

auto get_M2 = [](auto h_tmp) {
  const double h = cut_off(h_tmp);
  return md * (1 - h * h);
};

auto get_M2_dh = [](auto h_tmp) {
  const double h = cut_off(h_tmp);
  return -md * 2 * h * d_cut_off(h_tmp);
};

auto get_M3 = [](auto h_tmp) {
  const double h = cut_off(h_tmp);
  const double h2 = h * h;
  const double h3 = h2 * h;
  if (h >= 0)
    return md * (2 * h3 - 3 * h2 + 1);
  else
    return md * (-2 * h3 - 3 * h2 + 1);
};

auto get_M3_dh = [](auto h_tmp) {
  const double h = cut_off(h_tmp);
  if (h >= 0)
    return md * (6 * h * (h - 1)) * d_cut_off(h_tmp);
  else
    return md * (-6 * h * (h + 1)) * d_cut_off(h_tmp);
};

auto get_M = [](auto h) { return get_M0(h); };
auto get_M_dh = [](auto h) { return get_M0_dh(h); };

auto get_D = [](const double A) {
  FTensor::Ddg<double, SPACE_DIM, SPACE_DIM> t_D;
  t_D(i, j, k, l) = A * ((t_kd(i, k) ^ t_kd(j, l)) / 4.);
  return t_D;
};

// some examples of initialisation functions

auto kernel_oscillation = [](double r, double y, double) {
  constexpr int n = 3;
  constexpr double R = 0.0125;
  constexpr double A = R * 0.2;
  const double theta = atan2(r, y);
  const double w = R + A * cos(n * theta);
  const double d = std::sqrt(r * r + y * y);
  return tanh((w - d) / (eta * std::sqrt(2)));
};

auto kernel_eye = [](double r, double y, double) {
  constexpr double y0 = 0.5;
  constexpr double R = 0.4;
  y -= y0;
  const double d = std::sqrt(r * r + y * y);
  return tanh((R - d) / (eta * std::sqrt(2)));
};

auto capillary_tube = [](double x, double y, double z) {
  constexpr double water_height = 0.;
  return tanh((water_height - y) / (eta * std::sqrt(2)));
  ;
};

auto bubble_device = [](double x, double y, double z) {
  return -tanh((-0.039 - x) / (eta * std::sqrt(2)));
};

/**
 * @brief Initialisation function
 *
 * @note If UMs are compiled with Python to initialise phase field "H"
 * surface.py function is used, which has to be present in execution folder.
 *
 */
auto init_h = [](double r, double y, double theta) {
#ifdef PYTHON_INIT_SURFACE
  double s = 1;
  if (auto ptr = surfacePythonWeakPtr.lock()) {
    CHK_THROW_MESSAGE(ptr->evalSurface(r, y, theta, eta, s),
                      "error eval python");
  }
  return s;
#else
  return bubble_device(r, y, theta);
  // return capillary_tube(r, y, theta);
  // return kernel_eye(r, y, theta);
#endif
};

auto wetting_angle = [](double water_level) { return water_level; };

auto bit = [](auto b) { return BitRefLevel().set(b); };
auto marker = [](auto b) { return BitRefLevel().set(BITREFLEVEL_SIZE - b); };
auto get_fe_bit = [](FEMethod *fe_ptr) {
  return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel();
};

auto get_global_size = [](int l_size) {
  int g_size;
  MPI_Allreduce(&l_size, &g_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return g_size;
};

auto save_range = [](moab::Interface &moab, const std::string name,
                     const Range r) {
  MoFEMFunctionBegin;
  if (get_global_size(r.size())) {
    auto out_meshset = get_temp_meshset_ptr(moab);
    CHKERR moab.add_entities(*out_meshset, r);
    CHKERR moab.write_file(name.c_str(), "MOAB", "PARALLEL=WRITE_PART",
                           out_meshset->get_ptr(), 1);
  }
  MoFEMFunctionReturn(0);
};

/**
 * @brief get entities of dofs in the problem - used for debugging
 *
 */
auto get_dofs_ents_by_field_name = [](auto dm, auto field_name) {
  auto prb_ptr = getProblemPtr(dm);
  std::vector<EntityHandle> ents_vec;

  MoFEM::Interface *m_field_ptr;
  CHKERR DMoFEMGetInterfacePtr(dm, &m_field_ptr);

  auto bit_number = m_field_ptr->get_field_bit_number(field_name);
  auto dofs = prb_ptr->numeredRowDofsPtr;
  auto lo_it = dofs->lower_bound(FieldEntity::getLoBitNumberUId(bit_number));
  auto hi_it = dofs->upper_bound(FieldEntity::getHiBitNumberUId(bit_number));
  ents_vec.reserve(std::distance(lo_it, hi_it));

  for (; lo_it != hi_it; ++lo_it) {
    ents_vec.push_back((*lo_it)->getEnt());
  }

  std::sort(ents_vec.begin(), ents_vec.end());
  auto it = std::unique(ents_vec.begin(), ents_vec.end());
  Range r;
  r.insert_list(ents_vec.begin(), it);
  return r;
};

/**
 * @brief get entities of dofs in the problem - used for debugging
 *
 */
auto get_dofs_ents_all = [](auto dm) {
  auto prb_ptr = getProblemPtr(dm);
  std::vector<EntityHandle> ents_vec;

  MoFEM::Interface *m_field_ptr;
  CHKERR DMoFEMGetInterfacePtr(dm, &m_field_ptr);

  auto dofs = prb_ptr->numeredRowDofsPtr;
  ents_vec.reserve(dofs->size());

  for (auto d : *dofs) {
    ents_vec.push_back(d->getEnt());
  }

  std::sort(ents_vec.begin(), ents_vec.end());
  auto it = std::unique(ents_vec.begin(), ents_vec.end());
  Range r;
  r.insert_list(ents_vec.begin(), it);
  return r;
};

#include <FreeSurfaceOps.hpp>
using namespace FreeSurfaceOps;

struct FreeSurface;

enum FR { F, R }; // F - forward, and reverse

/**
 * @brief Set of functions called by PETSc solver used to refine and update
 * mesh.
 *
 * @note Currently theta method is only handled by this code.
 *
 */
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

  SmartPetscObj<VecScatter> getScatter(Vec x, Vec y, enum FR fr);
  SmartPetscObj<Vec> getSubVector();

  SmartPetscObj<DM> solverSubDM;
  SmartPetscObj<Vec> globSol;
  FreeSurface *fsRawPtr;

private:
  /**
   * @brief Pre process time step
   *
   * Refine mesh and update fields
   *
   * @param ts
   * @return MoFEMErrorCode
   */
  static MoFEMErrorCode tsPreProc(TS ts);

  /**
   * @brief Post process time step.
   *
   * Currently that function do not make anything major
   *
   * @param ts
   * @return MoFEMErrorCode
   */
  static MoFEMErrorCode tsPostProc(TS ts);

  static MoFEMErrorCode tsPreStage(TS ts);

  static MoFEMErrorCode tsSetIFunction(TS ts, PetscReal t, Vec u, Vec u_t,
                                       Vec f,
                                       void *ctx); //< Wrapper for SNES Rhs
  static MoFEMErrorCode tsSetIJacobian(TS ts, PetscReal t, Vec u, Vec u_t,
                                       PetscReal a, Mat A, Mat B,
                                       void *ctx); ///< Wrapper for SNES Lhs
  static MoFEMErrorCode tsMonitor(TS ts, PetscInt step, PetscReal t, Vec u,
                                  void *ctx);      ///< Wrapper for TS monitor
  static MoFEMErrorCode pcSetup(PC pc);
  static MoFEMErrorCode pcApply(PC pc, Vec pc_f, Vec pc_x);

  SmartPetscObj<Vec> globRes; //< global residual
  SmartPetscObj<Mat> subB;    //< sub problem tangent matrix
  SmartPetscObj<KSP> subKSP;  //< sub problem KSP solver

  boost::shared_ptr<SnesCtx>
      snesCtxPtr; //< infernal data (context) for MoFEM SNES fuctions
  boost::shared_ptr<TsCtx>
      tsCtxPtr;   //<  internal data (context) for MoFEM TS functions.
};

static boost::weak_ptr<TSPrePostProc> tsPrePostProc;

struct FreeSurface {

  FreeSurface(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

  MoFEMErrorCode makeRefProblem();

  MoFEM::Interface &mField;

private:
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode projectData();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();

  MoFEMErrorCode allBoundaryWet(BitRefLevel bit, BitRefLevel mask);

  /// @brief Find entities on refinement levels
  /// @param overlap level of overlap
  /// @return
  std::vector<Range> findEntitiesCrossedByPhaseInterface(size_t overlap);

  /// @brief
  /// @param ents
  /// @param level
  /// @param mask
  /// @return
  Range findParentsToRefine(Range ents, BitRefLevel level, BitRefLevel mask);

  /// @brief
  /// @param overlap
  /// @return
  MoFEMErrorCode refineMesh(size_t overlap);

  /// @brief Create hierarchy of elements run on parents levels
  /// @param fe_top pipeline element to which hierarchy is attached
  /// @param op type of operator OPSPACE, OPROW, OPCOL or OPROWCOL
  /// @param get_elema lambda function to create element on hierarchy
  /// @param verbosity verbosity level
  /// @param sev severity level
  /// @return
  MoFEMErrorCode setParentDofs(
      boost::shared_ptr<FEMethod> fe_top, std::string field_name,
      ForcesAndSourcesCore::UserDataOperator::OpType op,
      BitRefLevel child_ent_bit,
      boost::function<boost::shared_ptr<ForcesAndSourcesCore>()> get_elem,
      int verbosity, LogManager::SeverityLevel sev);

  friend struct TSPrePostProc;
};

//! [Run programme]
MoFEMErrorCode FreeSurface::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  MoFEMFunctionReturn(0);
}
//! [Run programme]

//! [Read mesh]
MoFEMErrorCode FreeSurface::readMesh() {
  MoFEMFunctionBegin;
  MOFEM_LOG("FS", Sev::inform) << "Read mesh for problem";
  auto simple = mField.getInterface<Simple>();

  simple->getParentAdjacencies() = true;
  simple->getBitRefLevel() = BitRefLevel();

  CHKERR simple->getOptions();
  CHKERR simple->loadFile();

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode FreeSurface::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-nb_levels", &nb_levels,
                            PETSC_NULL);
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-refine_overlap", &refine_overlap,
                            PETSC_NULL);

  const char *coord_type_names[] = {"cartesian", "polar", "cylindrical",
                                    "spherical"};
  CHKERR PetscOptionsGetEList(PETSC_NULL, NULL, "-coords", coord_type_names,
                              LAST_COORDINATE_SYSTEM, &coord_type, PETSC_NULL);

  MOFEM_LOG("FS", Sev::inform) << "order = " << order;
  MOFEM_LOG("FS", Sev::inform) << "nb_levels = " << nb_levels;
  nb_levels += 1;

  auto simple = mField.getInterface<Simple>();
  auto bit_mng = mField.getInterface<BitRefManager>();

  // Fields on domain

  // Velocity field
  CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, U_FIELD_DIM);
  // Pressure field
  CHKERR simple->addDomainField("P", H1, AINSWORTH_LEGENDRE_BASE, 1);
  // Order/phase fild
  CHKERR simple->addDomainField("H", H1, AINSWORTH_LEGENDRE_BASE, 1);
  // Chemical potential
  CHKERR simple->addDomainField("G", H1, AINSWORTH_LEGENDRE_BASE, 1);

  // Field on boundary
  CHKERR simple->addBoundaryField("U", H1, AINSWORTH_LEGENDRE_BASE,
                                  U_FIELD_DIM);
  CHKERR simple->addBoundaryField("H", H1, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple->addBoundaryField("G", H1, AINSWORTH_LEGENDRE_BASE, 1);
  // Lagrange multiplier which constrains slip conditions
  CHKERR simple->addBoundaryField("L", H1, AINSWORTH_LEGENDRE_BASE, 1);

  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setFieldOrder("P", order - 1);
  CHKERR simple->setFieldOrder("H", order);
  CHKERR simple->setFieldOrder("G", order);
  CHKERR simple->setFieldOrder("L", order);

  // Initialise bit ref levels
  auto set_problem_bit = [&]() {
    MoFEMFunctionBegin;
    // Set bits to build adjacencies between parents and children. That is
    // used by simple interface.
    simple->getBitAdjEnt() = BitRefLevel().set();
    simple->getBitAdjParent() = BitRefLevel().set();
    simple->getBitRefLevel() = BitRefLevel().set();
    simple->getBitRefLevelMask() = BitRefLevel().set();
    MoFEMFunctionReturn(0);
  };

  CHKERR set_problem_bit();

  CHKERR simple->setUp();

  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Boundary condition]
MoFEMErrorCode FreeSurface::boundaryCondition() {
  MoFEMFunctionBegin;

#ifdef PYTHON_INIT_SURFACE
  auto get_py_surface_init = []() {
    auto py_surf_init = boost::make_shared<SurfacePython>();
    CHKERR py_surf_init->surfaceInit("surface.py");
    surfacePythonWeakPtr = py_surf_init;
    return py_surf_init;
  };
  auto py_surf_init = get_py_surface_init();
#endif

  auto simple = mField.getInterface<Simple>();
  auto pip_mng = mField.getInterface<PipelineManager>();
  auto bc_mng = mField.getInterface<BcManager>();
  auto bit_mng = mField.getInterface<BitRefManager>();
  auto dm = simple->getDM();

  using UDO = ForcesAndSourcesCore::UserDataOperator;

  auto reset_bits = [&]() {
    MoFEMFunctionBegin;
    BitRefLevel start_mask;
    for (auto s = 0; s != get_start_bit(); ++s)
      start_mask[s] = true;
    // reset bit ref levels
    CHKERR bit_mng->lambdaBitRefLevel(
        [&](EntityHandle ent, BitRefLevel &bit) { bit &= start_mask; });
    Range level0;
    CHKERR bit_mng->getEntitiesByRefLevel(bit(0), BitRefLevel().set(), level0);
    CHKERR bit_mng->setNthBitRefLevel(level0, get_current_bit(), true);
    CHKERR bit_mng->setNthBitRefLevel(level0, get_projection_bit(), true);
    MoFEMFunctionReturn(0);
  };

  auto add_parent_field = [&](auto fe, auto op, auto field) {
    return setParentDofs(
        fe, field, op, bit(get_skin_parent_bit()),

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new DomianParentEle(mField));
          return fe_parent;
        },

        QUIET, Sev::noisy);
  };

  auto h_ptr = boost::make_shared<VectorDouble>();
  auto grad_h_ptr = boost::make_shared<MatrixDouble>();
  auto g_ptr = boost::make_shared<VectorDouble>();
  auto grad_g_ptr = boost::make_shared<MatrixDouble>();

  auto set_generic = [&](auto fe) {
    MoFEMFunctionBegin;
    auto &pip = fe->getOpPtrVector();

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1});

    CHKERR setParentDofs(
        fe, "", UDO::OPSPACE, bit(get_skin_parent_bit()),

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new DomianParentEle(mField));
          AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
              fe_parent->getOpPtrVector(), {H1});
          return fe_parent;
        },

        QUIET, Sev::noisy);

    CHKERR add_parent_field(fe, UDO::OPROW, "H");
    pip.push_back(new OpCalculateScalarFieldValues("H", h_ptr));
    pip.push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_h_ptr));

    CHKERR add_parent_field(fe, UDO::OPROW, "G");
    pip.push_back(new OpCalculateScalarFieldValues("G", g_ptr));
    pip.push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("G", grad_g_ptr));

    MoFEMFunctionReturn(0);
  };

  auto post_proc = [&](auto exe_test) {
    MoFEMFunctionBegin;
    auto post_proc_fe = boost::make_shared<PostProcEleDomain>(mField);
    post_proc_fe->exeTestHook = exe_test;

    CHKERR set_generic(post_proc_fe);

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(post_proc_fe->getPostProcMesh(),
                    post_proc_fe->getMapGaussPts(),

                    {{"H", h_ptr}, {"G", g_ptr}},

                    {{"GRAD_H", grad_h_ptr}, {"GRAD_G", grad_g_ptr}},

                    {},

                    {}

                    )

    );

    CHKERR DMoFEMLoopFiniteElements(dm, "dFE", post_proc_fe);
    CHKERR post_proc_fe->writeFile("out_init.h5m");

    MoFEMFunctionReturn(0);
  };

  auto solve_init = [&](auto exe_test) {
    MoFEMFunctionBegin;

    pip_mng->getOpDomainRhsPipeline().clear();
    pip_mng->getOpDomainLhsPipeline().clear();

    auto set_domain_rhs = [&](auto fe) {
      MoFEMFunctionBegin;
      CHKERR set_generic(fe);
      auto &pip = fe->getOpPtrVector();

      CHKERR add_parent_field(fe, UDO::OPROW, "H");
      pip.push_back(new OpRhsH<true>("H", nullptr, nullptr, h_ptr, grad_h_ptr,
                                     grad_g_ptr));
      CHKERR add_parent_field(fe, UDO::OPROW, "G");
      pip.push_back(new OpRhsG<true>("G", h_ptr, grad_h_ptr, g_ptr));
      MoFEMFunctionReturn(0);
    };

    auto set_domain_lhs = [&](auto fe) {
      MoFEMFunctionBegin;

      CHKERR set_generic(fe);
      auto &pip = fe->getOpPtrVector();

      CHKERR add_parent_field(fe, UDO::OPROW, "H");
      CHKERR add_parent_field(fe, UDO::OPCOL, "H");
      pip.push_back(new OpLhsH_dH<true>("H", nullptr, h_ptr, grad_g_ptr));

      CHKERR add_parent_field(fe, UDO::OPCOL, "G");
      pip.push_back(new OpLhsH_dG<true>("H", "G", h_ptr));

      CHKERR add_parent_field(fe, UDO::OPROW, "G");
      pip.push_back(new OpLhsG_dG("G"));

      CHKERR add_parent_field(fe, UDO::OPCOL, "H");
      pip.push_back(new OpLhsG_dH<true>("G", "H", h_ptr));
      MoFEMFunctionReturn(0);
    };

    auto create_subdm = [&]() {
      auto level_ents_ptr = boost::make_shared<Range>();
      CHKERR mField.getInterface<BitRefManager>()->getEntitiesByRefLevel(
          bit(get_current_bit()), BitRefLevel().set(), *level_ents_ptr);

      DM subdm;
      CHKERR DMCreate(mField.get_comm(), &subdm);
      CHKERR DMSetType(subdm, "DMMOFEM");
      CHKERR DMMoFEMCreateSubDM(subdm, dm, "SUB_INIT");
      CHKERR DMMoFEMAddElement(subdm, simple->getDomainFEName());
      CHKERR DMMoFEMSetSquareProblem(subdm, PETSC_TRUE);
      CHKERR DMMoFEMSetDestroyProblem(subdm, PETSC_TRUE);

      for (auto f : {"H", "G"}) {
        CHKERR DMMoFEMAddSubFieldRow(subdm, f, level_ents_ptr);
        CHKERR DMMoFEMAddSubFieldCol(subdm, f, level_ents_ptr);
      }
      CHKERR DMSetUp(subdm);

      if constexpr (debug) {
        if (mField.get_comm_size() == 1) {
          auto dm_ents = get_dofs_ents_all(SmartPetscObj<DM>(subdm, true));
          CHKERR save_range(mField.get_moab(), "sub_dm_init_ents_verts.h5m",
                            dm_ents.subset_by_type(MBVERTEX));
          dm_ents = subtract(dm_ents, dm_ents.subset_by_type(MBVERTEX));
          CHKERR save_range(mField.get_moab(), "sub_dm_init_ents.h5m", dm_ents);
        }
      }

      return SmartPetscObj<DM>(subdm);
    };

    auto subdm = create_subdm();

    pip_mng->getDomainRhsFE().reset();
    pip_mng->getDomainLhsFE().reset();
    CHKERR pip_mng->setDomainRhsIntegrationRule(integration_rule);
    CHKERR pip_mng->setDomainLhsIntegrationRule(integration_rule);
    pip_mng->getDomainLhsFE()->exeTestHook = exe_test;
    pip_mng->getDomainRhsFE()->exeTestHook = exe_test;

    CHKERR set_domain_rhs(pip_mng->getCastDomainRhsFE());
    CHKERR set_domain_lhs(pip_mng->getCastDomainLhsFE());

    auto D = createDMVector(subdm);
    auto snes = pip_mng->createSNES(subdm);
    auto snes_ctx_ptr = getDMSnesCtx(subdm);

    auto set_section_monitor = [&](auto solver) {
      MoFEMFunctionBegin;
      CHKERR SNESMonitorSet(solver,
                            (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal,
                                               void *))MoFEMSNESMonitorFields,
                            (void *)(snes_ctx_ptr.get()), nullptr);

      MoFEMFunctionReturn(0);
    };

    auto print_section_field = [&]() {
      MoFEMFunctionBegin;

      auto section =
          mField.getInterface<ISManager>()->sectionCreate("SUB_INIT");
      PetscInt num_fields;
      CHKERR PetscSectionGetNumFields(section, &num_fields);
      for (int f = 0; f < num_fields; ++f) {
        const char *field_name;
        CHKERR PetscSectionGetFieldName(section, f, &field_name);
        MOFEM_LOG("FS", Sev::inform)
            << "Field " << f << " " << std::string(field_name);
      }

      MoFEMFunctionReturn(0);
    };

    CHKERR set_section_monitor(snes);
    CHKERR print_section_field();

    for (auto f : {"U", "P", "H", "G", "L"}) {
      CHKERR mField.getInterface<FieldBlas>()->setField(0, f);
    }

    CHKERR SNESSetFromOptions(snes);
    CHKERR SNESSolve(snes, PETSC_NULL, D);

    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(subdm, D, INSERT_VALUES, SCATTER_REVERSE);

    MoFEMFunctionReturn(0);
  };

  CHKERR reset_bits();
  CHKERR solve_init(
      [](FEMethod *fe_ptr) { return get_fe_bit(fe_ptr).test(0); });
  CHKERR allBoundaryWet(bit(0), BitRefLevel().set());
  CHKERR refineMesh(refine_overlap);

  for (auto f : {"U", "P", "H", "G", "L"}) {
    CHKERR mField.getInterface<FieldBlas>()->setField(0, f);
  }
  CHKERR solve_init([](FEMethod *fe_ptr) {
    return get_fe_bit(fe_ptr).test(get_start_bit() + nb_levels - 1);
  });
  CHKERR allBoundaryWet(get_start_bit() + nb_levels - 1, BitRefLevel().set());

  CHKERR post_proc([](FEMethod *fe_ptr) {
    return get_fe_bit(fe_ptr).test(get_start_bit() + nb_levels - 1);
  });

  pip_mng->getOpDomainRhsPipeline().clear();
  pip_mng->getOpDomainLhsPipeline().clear();

  // Remove DOFs where boundary conditions are set
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "SYM_X",
                                           "U", 0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "SYM_X",
                                           "L", 0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "SYM_Y",
                                           "U", 1, 1);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "SYM_Y",
                                           "L", 1, 1);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX", "U",
                                           0, SPACE_DIM);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX", "L",
                                           0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "ZERO",
                                           "L", 0, 0);

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Data projection]
MoFEMErrorCode FreeSurface::projectData() {
  MoFEMFunctionBegin;

  // FIXME: Functionality in this method should be improved (projection should
  // be done field by field), generalised, and move to become core
  // functionality.

  auto simple = mField.getInterface<Simple>();
  auto pip_mng = mField.getInterface<PipelineManager>();
  auto bc_mng = mField.getInterface<BcManager>();
  auto bit_mng = mField.getInterface<BitRefManager>();
  auto field_blas = mField.getInterface<FieldBlas>();

  // Store all existing elements pipelines, replace them by data projection
  // pipelines, to put them back when projection is done.
  auto fe_domain_lhs = pip_mng->getDomainLhsFE();
  auto fe_domain_rhs = pip_mng->getDomainRhsFE();
  auto fe_bdy_lhs = pip_mng->getBoundaryLhsFE();
  auto fe_bdy_rhs = pip_mng->getBoundaryRhsFE();

  pip_mng->getDomainLhsFE().reset();
  pip_mng->getDomainRhsFE().reset();
  pip_mng->getBoundaryLhsFE().reset();
  pip_mng->getBoundaryRhsFE().reset();

  using UDO = ForcesAndSourcesCore::UserDataOperator;

  // extract field data for domain parent element
  auto add_parent_field_domain = [&](auto fe, auto op, auto field, auto bit) {
    return setParentDofs(
        fe, field, op, bit,

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new DomianParentEle(mField));
          return fe_parent;
        },

        QUIET, Sev::noisy);
  };

  // extract field data for boundary parent element
  auto add_parent_field_bdy = [&](auto fe, auto op, auto field, auto bit) {
    return setParentDofs(
        fe, field, op, bit,

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new BoundaryParentEle(mField));
          return fe_parent;
        },

        QUIET, Sev::noisy);
  };

  // run this element on element with given bit, otherwise run other nested
  // element
  auto create_run_parent_op = [&](auto parent_fe_ptr, auto this_fe_ptr,
                                  auto fe_bit) {
    auto parent_mask = fe_bit;
    parent_mask.flip();
    return new OpRunParent(parent_fe_ptr, BitRefLevel().set(), parent_mask,
                           this_fe_ptr, fe_bit, BitRefLevel().set(), QUIET,
                           Sev::inform);
  };

  // create hierarchy of nested elements
  auto get_parents_vel_fe_ptr = [&](auto this_fe_ptr, auto fe_bit) {
    std::vector<boost::shared_ptr<DomianParentEle>> parents_elems_ptr_vec;
    for (int l = 0; l < nb_levels; ++l)
      parents_elems_ptr_vec.emplace_back(
          boost::make_shared<DomianParentEle>(mField));
    for (auto l = 1; l < nb_levels; ++l) {
      parents_elems_ptr_vec[l - 1]->getOpPtrVector().push_back(
          create_run_parent_op(parents_elems_ptr_vec[l], this_fe_ptr, fe_bit));
    }
    return parents_elems_ptr_vec[0];
  };

  // solve projection problem
  auto solve_projection = [&](auto exe_test) {
    MoFEMFunctionBegin;

    auto set_domain_rhs = [&](auto fe) {
      MoFEMFunctionBegin;
      auto &pip = fe->getOpPtrVector();

      auto u_ptr = boost::make_shared<MatrixDouble>();
      auto p_ptr = boost::make_shared<VectorDouble>();
      auto h_ptr = boost::make_shared<VectorDouble>();
      auto g_ptr = boost::make_shared<VectorDouble>();

      auto eval_fe_ptr = boost::make_shared<DomianParentEle>(mField);
      {
        auto &pip = eval_fe_ptr->getOpPtrVector();
        CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1});
        CHKERR setParentDofs(
            eval_fe_ptr, "", UDO::OPSPACE, bit(get_skin_projection_bit()),

            [&]() {
              boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
                  new DomianParentEle(mField));
              return fe_parent;
            },

            QUIET, Sev::noisy);
        // That can be done much smarter, by block, field by field. For
        // simplicity is like that.
        CHKERR add_parent_field_domain(eval_fe_ptr, UDO::OPROW, "U",
                                       bit(get_skin_projection_bit()));
        pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));
        CHKERR add_parent_field_domain(eval_fe_ptr, UDO::OPROW, "P",
                                       bit(get_skin_projection_bit()));
        pip.push_back(new OpCalculateScalarFieldValues("P", p_ptr));
        CHKERR add_parent_field_domain(eval_fe_ptr, UDO::OPROW, "H",
                                       bit(get_skin_projection_bit()));
        pip.push_back(new OpCalculateScalarFieldValues("H", h_ptr));
        CHKERR add_parent_field_domain(eval_fe_ptr, UDO::OPROW, "G",
                                       bit(get_skin_projection_bit()));
        pip.push_back(new OpCalculateScalarFieldValues("G", g_ptr));
      }
      auto parent_eval_fe_ptr =
          get_parents_vel_fe_ptr(eval_fe_ptr, bit(get_projection_bit()));
      pip.push_back(create_run_parent_op(parent_eval_fe_ptr, eval_fe_ptr,
                                         bit(get_projection_bit())));

      auto assemble_fe_ptr = boost::make_shared<DomianParentEle>(mField);
      {
        auto &pip = assemble_fe_ptr->getOpPtrVector();
        CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1});
        CHKERR setParentDofs(
            assemble_fe_ptr, "", UDO::OPSPACE, bit(get_skin_parent_bit()),

            [&]() {
              boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
                  new DomianParentEle(mField));
              return fe_parent;
            },

            QUIET, Sev::noisy);
        CHKERR add_parent_field_domain(assemble_fe_ptr, UDO::OPROW, "U",
                                       bit(get_skin_parent_bit()));
        pip.push_back(new OpDomainAssembleVector("U", u_ptr));
        CHKERR add_parent_field_domain(assemble_fe_ptr, UDO::OPROW, "P",
                                       bit(get_skin_parent_bit()));
        pip.push_back(new OpDomainAssembleScalar("P", p_ptr));
        CHKERR add_parent_field_domain(assemble_fe_ptr, UDO::OPROW, "H",
                                       bit(get_skin_parent_bit()));
        pip.push_back(new OpDomainAssembleScalar("H", h_ptr));
        CHKERR add_parent_field_domain(assemble_fe_ptr, UDO::OPROW, "G",
                                       bit(get_skin_parent_bit()));
        pip.push_back(new OpDomainAssembleScalar("G", g_ptr));
      }
      auto parent_assemble_fe_ptr =
          get_parents_vel_fe_ptr(assemble_fe_ptr, bit(get_current_bit()));
      pip.push_back(create_run_parent_op(
          parent_assemble_fe_ptr, assemble_fe_ptr, bit(get_current_bit())));

      MoFEMFunctionReturn(0);
    };

    auto set_domain_lhs = [&](auto fe) {
      MoFEMFunctionBegin;

      auto &pip = fe->getOpPtrVector();

      CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1});

      CHKERR setParentDofs(
          fe, "", UDO::OPSPACE, bit(get_skin_parent_bit()),

          [&]() {
            boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
                new DomianParentEle(mField));
            return fe_parent;
          },

          QUIET, Sev::noisy);

      // That can be done much smarter, by block, field by field. For simplicity
      // is like that.
      CHKERR add_parent_field_domain(fe, UDO::OPROW, "U",
                                     bit(get_skin_parent_bit()));
      CHKERR add_parent_field_domain(fe, UDO::OPCOL, "U",
                                     bit(get_skin_parent_bit()));
      pip.push_back(new OpDomainMassU("U", "U"));
      CHKERR add_parent_field_domain(fe, UDO::OPROW, "P",
                                     bit(get_skin_parent_bit()));
      CHKERR add_parent_field_domain(fe, UDO::OPCOL, "P",
                                     bit(get_skin_parent_bit()));
      pip.push_back(new OpDomainMassP("P", "P"));
      CHKERR add_parent_field_domain(fe, UDO::OPROW, "H",
                                     bit(get_skin_parent_bit()));
      CHKERR add_parent_field_domain(fe, UDO::OPCOL, "H",
                                     bit(get_skin_parent_bit()));
      pip.push_back(new OpDomainMassH("H", "H"));
      CHKERR add_parent_field_domain(fe, UDO::OPROW, "G",
                                     bit(get_skin_parent_bit()));
      CHKERR add_parent_field_domain(fe, UDO::OPCOL, "G",
                                     bit(get_skin_parent_bit()));
      pip.push_back(new OpDomainMassG("G", "G"));

      MoFEMFunctionReturn(0);
    };

    auto set_bdy_rhs = [&](auto fe) {
      MoFEMFunctionBegin;
      auto &pip = fe->getOpPtrVector();

      auto l_ptr = boost::make_shared<VectorDouble>();

      auto eval_fe_ptr = boost::make_shared<BoundaryParentEle>(mField);
      {
        auto &pip = eval_fe_ptr->getOpPtrVector();
        CHKERR setParentDofs(
            eval_fe_ptr, "", UDO::OPSPACE, bit(get_skin_projection_bit()),

            [&]() {
              boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
                  new BoundaryParentEle(mField));
              return fe_parent;
            },

            QUIET, Sev::noisy);
        // That can be done much smarter, by block, field by field. For
        // simplicity is like that.
        CHKERR add_parent_field_bdy(eval_fe_ptr, UDO::OPROW, "L",
                                    bit(get_skin_projection_bit()));
        pip.push_back(new OpCalculateScalarFieldValues("L", l_ptr));
      }
      auto parent_eval_fe_ptr =
          get_parents_vel_fe_ptr(eval_fe_ptr, bit(get_projection_bit()));
      pip.push_back(create_run_parent_op(parent_eval_fe_ptr, eval_fe_ptr,
                                         bit(get_projection_bit())));

      auto assemble_fe_ptr = boost::make_shared<BoundaryParentEle>(mField);
      {
        auto &pip = assemble_fe_ptr->getOpPtrVector();
        CHKERR setParentDofs(
            assemble_fe_ptr, "", UDO::OPSPACE, bit(get_skin_parent_bit()),

            [&]() {
              boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
                  new BoundaryParentEle(mField));
              return fe_parent;
            },

            QUIET, Sev::noisy);

        struct OpLSize : public BoundaryEleOp {
          OpLSize(boost::shared_ptr<VectorDouble> l_ptr)
              : BoundaryEleOp(NOSPACE, DomainEleOp::OPSPACE), lPtr(l_ptr) {}
          MoFEMErrorCode doWork(int, EntityType, EntData &) {
            MoFEMFunctionBegin;
            if (lPtr->size() != getGaussPts().size2()) {
              lPtr->resize(getGaussPts().size2());
              lPtr->clear();
            }
            MoFEMFunctionReturn(0);
          }

        private:
          boost::shared_ptr<VectorDouble> lPtr;
        };

        pip.push_back(new OpLSize(l_ptr));

        CHKERR add_parent_field_bdy(assemble_fe_ptr, UDO::OPROW, "L",
                                    bit(get_skin_parent_bit()));
        pip.push_back(new OpBoundaryAssembleScalar("L", l_ptr));
      }
      auto parent_assemble_fe_ptr =
          get_parents_vel_fe_ptr(assemble_fe_ptr, bit(get_current_bit()));
      pip.push_back(create_run_parent_op(
          parent_assemble_fe_ptr, assemble_fe_ptr, bit(get_current_bit())));

      MoFEMFunctionReturn(0);
    };

    auto set_bdy_lhs = [&](auto fe) {
      MoFEMFunctionBegin;

      auto &pip = fe->getOpPtrVector();

      CHKERR setParentDofs(
          fe, "", UDO::OPSPACE, bit(get_skin_parent_bit()),

          [&]() {
            boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
                new BoundaryParentEle(mField));
            return fe_parent;
          },

          QUIET, Sev::noisy);

      // That can be done much smarter, by block, field by field. For simplicity
      // is like that.
      CHKERR add_parent_field_bdy(fe, UDO::OPROW, "L",
                                  bit(get_skin_parent_bit()));
      CHKERR add_parent_field_bdy(fe, UDO::OPCOL, "L",
                                  bit(get_skin_parent_bit()));
      pip.push_back(new OpBoundaryMassL("L", "L"));

      MoFEMFunctionReturn(0);
    };

    auto create_subdm = [&]() -> SmartPetscObj<DM> {
      auto level_ents_ptr = boost::make_shared<Range>();
      CHKERR mField.getInterface<BitRefManager>()->getEntitiesByRefLevel(
          bit(get_current_bit()), BitRefLevel().set(), *level_ents_ptr);

      auto get_prj_ents = [&]() {
        Range prj_mesh;
        CHKERR bit_mng->getEntitiesByDimAndRefLevel(bit(get_projection_bit()),
                                                    BitRefLevel().set(),
                                                    SPACE_DIM, prj_mesh);
        auto common_ents = intersect(prj_mesh, *level_ents_ptr);
        prj_mesh = subtract(unite(*level_ents_ptr, prj_mesh), common_ents)
                       .subset_by_dimension(SPACE_DIM);

        return prj_mesh;
      };

      auto prj_ents = get_prj_ents();

      if (get_global_size(prj_ents.size())) {

        auto rebuild = [&]() {
          auto prb_mng = mField.getInterface<ProblemsManager>();
          MoFEMFunctionBegin;

          std::vector<std::string> fields{"U", "P", "H", "G", "L"};
          std::map<std::string, boost::shared_ptr<Range>> range_maps{

              {"U", level_ents_ptr},
              {"P", level_ents_ptr},
              {"H", level_ents_ptr},
              {"G", level_ents_ptr},
              {"L", level_ents_ptr}

          };

          CHKERR prb_mng->buildSubProblem("SUB_SOLVER", fields, fields,
                                          simple->getProblemName(), PETSC_TRUE,
                                          &range_maps, &range_maps);

          // partition problem
          CHKERR prb_mng->partitionFiniteElements("SUB_SOLVER", true, 0,
                                                  mField.get_comm_size());
          // set ghost nodes
          CHKERR prb_mng->partitionGhostDofsOnDistributedMesh("SUB_SOLVER");
          MoFEMFunctionReturn(0);
        };

        MOFEM_LOG("FS", Sev::verbose) << "Create projection problem";

        CHKERR rebuild();

        auto dm = simple->getDM();
        DM subdm;
        CHKERR DMCreate(mField.get_comm(), &subdm);
        CHKERR DMSetType(subdm, "DMMOFEM");
        CHKERR DMMoFEMCreateSubDM(subdm, dm, "SUB_SOLVER");
        return SmartPetscObj<DM>(subdm);
      }

      MOFEM_LOG("FS", Sev::inform) << "Nothing to project";

      return SmartPetscObj<DM>();
    };

    auto create_dummy_dm = [&]() {
      auto dummy_dm = createDM(mField.get_comm(), "DMMOFEM");
      CHK_THROW_MESSAGE(DMMoFEMCreateMoFEM(dummy_dm, &mField,
                                           simple->getProblemName().c_str(),
                                           BitRefLevel(), BitRefLevel()),
                        "create dummy dm");
      return dummy_dm;
    };

    auto subdm = create_subdm();
    if (subdm) {

      pip_mng->getDomainRhsFE().reset();
      pip_mng->getDomainLhsFE().reset();
      pip_mng->getBoundaryRhsFE().reset();
      pip_mng->getBoundaryLhsFE().reset();
      CHKERR pip_mng->setDomainRhsIntegrationRule(integration_rule);
      CHKERR pip_mng->setDomainLhsIntegrationRule(integration_rule);
      CHKERR pip_mng->setBoundaryRhsIntegrationRule(integration_rule);
      CHKERR pip_mng->setBoundaryLhsIntegrationRule(integration_rule);
      pip_mng->getDomainLhsFE()->exeTestHook = exe_test;
      pip_mng->getDomainRhsFE()->exeTestHook = [](FEMethod *fe_ptr) {
        return get_fe_bit(fe_ptr).test(nb_levels - 1);
      };
      pip_mng->getBoundaryLhsFE()->exeTestHook = exe_test;
      pip_mng->getBoundaryRhsFE()->exeTestHook = [](FEMethod *fe_ptr) {
        return get_fe_bit(fe_ptr).test(nb_levels - 1);
      };

      CHKERR set_domain_rhs(pip_mng->getCastDomainRhsFE());
      CHKERR set_domain_lhs(pip_mng->getCastDomainLhsFE());
      CHKERR set_bdy_rhs(pip_mng->getCastBoundaryRhsFE());
      CHKERR set_bdy_lhs(pip_mng->getCastBoundaryLhsFE());

      auto D = createDMVector(subdm);
      auto F = vectorDuplicate(D);

      auto ksp = pip_mng->createKSP(subdm);
      CHKERR KSPSetFromOptions(ksp);
      CHKERR KSPSetUp(ksp);

      // get vector norm
      auto get_norm = [&](auto x) {
        double nrm;
        CHKERR VecNorm(x, NORM_2, &nrm);
        return nrm;
      };

      /**
       * @brief Zero DOFs, used by FieldBlas
       */
      auto zero_dofs = [](boost::shared_ptr<FieldEntity> ent_ptr) {
        MoFEMFunctionBegin;
        for (auto &v : ent_ptr->getEntFieldData()) {
          v = 0;
        }
        MoFEMFunctionReturn(0);
      };

      /**
       * @brief cut-off values at nodes, i.e. abs("H") <= 1
       *
       */
      auto cut_off_dofs = [&]() {
        MoFEMFunctionBegin;

        Range current_verts;
        CHKERR bit_mng->getEntitiesByTypeAndRefLevel(bit(get_current_bit()),
                                                     BitRefLevel().set(),
                                                     MBVERTEX, current_verts);

        auto cut_off_verts = [&](boost::shared_ptr<FieldEntity> ent_ptr) {
          MoFEMFunctionBegin;
          for (auto &h : ent_ptr->getEntFieldData()) {
            h = cut_off(h);
          }
          MoFEMFunctionReturn(0);
        };

        auto field_blas = mField.getInterface<FieldBlas>();
        CHKERR field_blas->fieldLambdaOnEntities(cut_off_verts, "H",
                                                 &current_verts);
        MoFEMFunctionReturn(0);
      };

      auto solve = [&](auto S) {
        MoFEMFunctionBegin;
        CHKERR VecZeroEntries(S);
        CHKERR VecZeroEntries(F);
        CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
        CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);
        CHKERR KSPSolve(ksp, F, S);
        CHKERR VecGhostUpdateBegin(S, INSERT_VALUES, SCATTER_FORWARD);
        CHKERR VecGhostUpdateEnd(S, INSERT_VALUES, SCATTER_FORWARD);
        MoFEMFunctionReturn(0);
      };

      MOFEM_LOG("FS", Sev::inform) << "Solve projection";
      CHKERR solve(D);

      auto glob_x = createDMVector(simple->getDM());
      auto sub_x = createDMVector(subdm);
      auto dummy_dm = create_dummy_dm();

      /**
       * @brief get TSTheta data operators
       */
      auto apply_restrict = [&]() {
        auto get_is = [](auto v) {
          IS iy;
          auto create = [&]() {
            MoFEMFunctionBegin;
            int n, ystart;
            CHKERR VecGetLocalSize(v, &n);
            CHKERR VecGetOwnershipRange(v, &ystart, NULL);
            CHKERR ISCreateStride(PETSC_COMM_SELF, n, ystart, 1, &iy);
            MoFEMFunctionReturn(0);
          };
          CHK_THROW_MESSAGE(create(), "create is");
          return SmartPetscObj<IS>(iy);
        };

        auto iy = get_is(glob_x);
        auto s = createVecScatter(glob_x, PETSC_NULL, glob_x, iy);

        CHK_THROW_MESSAGE(
            DMSubDomainRestrict(simple->getDM(), s, PETSC_NULL, dummy_dm),
            "restrict");
        Vec X0, Xdot;
        CHK_THROW_MESSAGE(DMGetNamedGlobalVector(dummy_dm, "TSTheta_X0", &X0),
                          "get X0");
        CHK_THROW_MESSAGE(
            DMGetNamedGlobalVector(dummy_dm, "TSTheta_Xdot", &Xdot),
            "get Xdot");

        auto forward_ghost = [](auto g) {
          MoFEMFunctionBegin;
          CHKERR VecGhostUpdateBegin(g, INSERT_VALUES, SCATTER_FORWARD);
          CHKERR VecGhostUpdateEnd(g, INSERT_VALUES, SCATTER_FORWARD);
          MoFEMFunctionReturn(0);
        };

        CHK_THROW_MESSAGE(forward_ghost(X0), "");
        CHK_THROW_MESSAGE(forward_ghost(Xdot), "");

        if constexpr (debug) {
          MOFEM_LOG("FS", Sev::inform)
              << "Reverse restrict: X0 " << get_norm(X0) << " Xdot "
              << get_norm(Xdot);
        }

        return std::vector<Vec>{X0, Xdot};
      };

      auto ts_solver_vecs = apply_restrict();

      if (ts_solver_vecs.size()) {

        for (auto v : ts_solver_vecs) {
          MOFEM_LOG("FS", Sev::inform) << "Solve projection vector";

          CHKERR DMoFEMMeshToLocalVector(simple->getDM(), v, INSERT_VALUES,
                                         SCATTER_REVERSE);
          CHKERR solve(sub_x);

          for (auto f : {"U", "P", "H", "G", "L"}) {
            MOFEM_LOG("WORLD", Sev::verbose) << "Zero field " << f;
            CHKERR field_blas->fieldLambdaOnEntities(zero_dofs, f);
          }
          CHKERR DMoFEMMeshToLocalVector(subdm, sub_x, INSERT_VALUES,
                                         SCATTER_REVERSE);
          CHKERR DMoFEMMeshToLocalVector(simple->getDM(), v, INSERT_VALUES,
                                         SCATTER_FORWARD);

          MOFEM_LOG("FS", Sev::inform) << "Norm V " << get_norm(v);
        }

        CHKERR DMRestoreNamedGlobalVector(dummy_dm, "TSTheta_X0",
                                          &ts_solver_vecs[0]);
        CHKERR DMRestoreNamedGlobalVector(dummy_dm, "TSTheta_Xdot",
                                          &ts_solver_vecs[1]);
      }

      for (auto f : {"U", "P", "H", "G", "L"}) {
        MOFEM_LOG("WORLD", Sev::verbose) << "Zero field " << f;
        CHKERR field_blas->fieldLambdaOnEntities(zero_dofs, f);
      }
      CHKERR DMoFEMMeshToLocalVector(subdm, D, INSERT_VALUES, SCATTER_REVERSE);
      CHKERR cut_off_dofs();
    }

    MoFEMFunctionReturn(0);
  };

  // postprocessing (only for debugging proposes)
  auto post_proc = [&](auto exe_test) {
    MoFEMFunctionBegin;
    auto post_proc_fe = boost::make_shared<PostProcEleDomain>(mField);
    auto &pip = post_proc_fe->getOpPtrVector();
    post_proc_fe->exeTestHook = exe_test;

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1});

    CHKERR setParentDofs(
        post_proc_fe, "", UDO::OPSPACE, bit(get_skin_parent_bit()),

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new DomianParentEle(mField));
          AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
              fe_parent->getOpPtrVector(), {H1});
          return fe_parent;
        },

        QUIET, Sev::noisy);

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    auto u_ptr = boost::make_shared<MatrixDouble>();
    auto p_ptr = boost::make_shared<VectorDouble>();
    auto h_ptr = boost::make_shared<VectorDouble>();
    auto g_ptr = boost::make_shared<VectorDouble>();

    CHKERR add_parent_field_domain(post_proc_fe, UDO::OPROW, "U",
                                   bit(get_skin_parent_bit()));
    pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));
    CHKERR add_parent_field_domain(post_proc_fe, UDO::OPROW, "P",
                                   bit(get_skin_parent_bit()));
    pip.push_back(new OpCalculateScalarFieldValues("P", p_ptr));
    CHKERR add_parent_field_domain(post_proc_fe, UDO::OPROW, "H",
                                   bit(get_skin_parent_bit()));
    pip.push_back(new OpCalculateScalarFieldValues("H", h_ptr));
    CHKERR add_parent_field_domain(post_proc_fe, UDO::OPROW, "G",
                                   bit(get_skin_parent_bit()));
    pip.push_back(new OpCalculateScalarFieldValues("G", g_ptr));

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(post_proc_fe->getPostProcMesh(),
                    post_proc_fe->getMapGaussPts(),

                    {{"P", p_ptr}, {"H", h_ptr}, {"G", g_ptr}},

                    {{"U", u_ptr}},

                    {},

                    {}

                    )

    );

    auto dm = simple->getDM();
    CHKERR DMoFEMLoopFiniteElements(dm, "dFE", post_proc_fe);
    CHKERR post_proc_fe->writeFile("out_projection.h5m");

    MoFEMFunctionReturn(0);
  };

  CHKERR solve_projection([](FEMethod *fe_ptr) {
    return get_fe_bit(fe_ptr).test(get_current_bit());
  });

  if constexpr (debug) {
    CHKERR post_proc([](FEMethod *fe_ptr) {
      return get_fe_bit(fe_ptr).test(get_current_bit());
    });
  }

  fe_domain_lhs.swap(pip_mng->getDomainLhsFE());
  fe_domain_rhs.swap(pip_mng->getDomainRhsFE());
  fe_bdy_lhs.swap(pip_mng->getBoundaryLhsFE());
  fe_bdy_rhs.swap(pip_mng->getBoundaryRhsFE());

  MoFEMFunctionReturn(0);
}
//! [Data projection]

//! [Push operators to pip]
MoFEMErrorCode FreeSurface::assembleSystem() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();

  using UDO = ForcesAndSourcesCore::UserDataOperator;

  auto add_parent_field_domain = [&](auto fe, auto op, auto field) {
    return setParentDofs(
        fe, field, op, bit(get_skin_parent_bit()),

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new DomianParentEle(mField));
          return fe_parent;
        },

        QUIET, Sev::noisy);
  };

  auto add_parent_field_bdy = [&](auto fe, auto op, auto field) {
    return setParentDofs(
        fe, field, op, bit(get_skin_parent_bit()),

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new BoundaryParentEle(mField));
          return fe_parent;
        },

        QUIET, Sev::noisy);
  };

  auto test_bit_child = [](FEMethod *fe_ptr) {
    return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
        get_start_bit() + nb_levels - 1);
  };

  auto dot_u_ptr = boost::make_shared<MatrixDouble>();
  auto u_ptr = boost::make_shared<MatrixDouble>();
  auto grad_u_ptr = boost::make_shared<MatrixDouble>();
  auto dot_h_ptr = boost::make_shared<VectorDouble>();
  auto h_ptr = boost::make_shared<VectorDouble>();
  auto grad_h_ptr = boost::make_shared<MatrixDouble>();
  auto g_ptr = boost::make_shared<VectorDouble>();
  auto grad_g_ptr = boost::make_shared<MatrixDouble>();
  auto lambda_ptr = boost::make_shared<VectorDouble>();
  auto p_ptr = boost::make_shared<VectorDouble>();
  auto div_u_ptr = boost::make_shared<VectorDouble>();

  // Push element from reference configuration to current configuration in 3d
  // space
  auto set_domain_general = [&](auto fe) {
    MoFEMFunctionBegin;
    auto &pip = fe->getOpPtrVector();

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1});

    CHKERR setParentDofs(
        fe, "", UDO::OPSPACE, bit(get_skin_parent_bit()),

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new DomianParentEle(mField));
          AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
              fe_parent->getOpPtrVector(), {H1});
          return fe_parent;
        },

        QUIET, Sev::noisy);

    CHKERR add_parent_field_domain(fe, UDO::OPROW, "U");
    pip.push_back(
        new OpCalculateVectorFieldValuesDot<U_FIELD_DIM>("U", dot_u_ptr));
    pip.push_back(new OpCalculateVectorFieldValues<U_FIELD_DIM>("U", u_ptr));
    pip.push_back(new OpCalculateVectorFieldGradient<U_FIELD_DIM, SPACE_DIM>(
        "U", grad_u_ptr));

    switch (coord_type) {
    case CARTESIAN:
      pip.push_back(
          new OpCalculateDivergenceVectorFieldValues<SPACE_DIM, CARTESIAN>(
              "U", div_u_ptr));
      break;
    case CYLINDRICAL:
      pip.push_back(
          new OpCalculateDivergenceVectorFieldValues<SPACE_DIM, CYLINDRICAL>(
              "U", div_u_ptr));
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, MOFEM_NOT_IMPLEMENTED,
              "Coordinate system not implemented");
    }

    CHKERR add_parent_field_domain(fe, UDO::OPROW, "H");
    pip.push_back(new OpCalculateScalarFieldValuesDot("H", dot_h_ptr));
    pip.push_back(new OpCalculateScalarFieldValues("H", h_ptr));
    pip.push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_h_ptr));

    CHKERR add_parent_field_domain(fe, UDO::OPROW, "G");
    pip.push_back(new OpCalculateScalarFieldValues("G", g_ptr));
    pip.push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("G", grad_g_ptr));

    CHKERR add_parent_field_domain(fe, UDO::OPROW, "P");
    pip.push_back(new OpCalculateScalarFieldValues("P", p_ptr));
    MoFEMFunctionReturn(0);
  };

  auto set_domain_rhs = [&](auto fe) {
    MoFEMFunctionBegin;
    auto &pip = fe->getOpPtrVector();

    CHKERR set_domain_general(fe);

    CHKERR add_parent_field_domain(fe, UDO::OPROW, "U");
    pip.push_back(new OpRhsU("U", dot_u_ptr, u_ptr, grad_u_ptr, h_ptr,
                             grad_h_ptr, g_ptr, p_ptr));

    CHKERR add_parent_field_domain(fe, UDO::OPROW, "H");
    pip.push_back(new OpRhsH<false>("H", u_ptr, dot_h_ptr, h_ptr, grad_h_ptr,
                                    grad_g_ptr));

    CHKERR add_parent_field_domain(fe, UDO::OPROW, "G");
    pip.push_back(new OpRhsG<false>("G", h_ptr, grad_h_ptr, g_ptr));

    CHKERR add_parent_field_domain(fe, UDO::OPROW, "P");
    pip.push_back(new OpDomainAssembleScalar(
        "P", div_u_ptr, [](const double r, const double, const double) {
          return cylindrical(r);
        }));
    pip.push_back(new OpDomainAssembleScalar(
        "P", p_ptr, [](const double r, const double, const double) {
          return eps * cylindrical(r);
        }));
    MoFEMFunctionReturn(0);
  };

  auto set_domain_lhs = [&](auto fe) {
    MoFEMFunctionBegin;
    auto &pip = fe->getOpPtrVector();

    CHKERR set_domain_general(fe);

    CHKERR add_parent_field_domain(fe, UDO::OPROW, "U");
    {
      CHKERR add_parent_field_domain(fe, UDO::OPCOL, "U");
      pip.push_back(new OpLhsU_dU("U", u_ptr, grad_u_ptr, h_ptr));
      CHKERR add_parent_field_domain(fe, UDO::OPCOL, "H");
      pip.push_back(
          new OpLhsU_dH("U", "H", dot_u_ptr, u_ptr, grad_u_ptr, h_ptr, g_ptr));

      CHKERR add_parent_field_domain(fe, UDO::OPCOL, "G");
      pip.push_back(new OpLhsU_dG("U", "G", grad_h_ptr));
    }

    CHKERR add_parent_field_domain(fe, UDO::OPROW, "H");
    {
      CHKERR add_parent_field_domain(fe, UDO::OPCOL, "U");
      pip.push_back(new OpLhsH_dU("H", "U", grad_h_ptr));
      CHKERR add_parent_field_domain(fe, UDO::OPCOL, "H");
      pip.push_back(new OpLhsH_dH<false>("H", u_ptr, h_ptr, grad_g_ptr));
      CHKERR add_parent_field_domain(fe, UDO::OPCOL, "G");
      pip.push_back(new OpLhsH_dG<false>("H", "G", h_ptr));
    }

    CHKERR add_parent_field_domain(fe, UDO::OPROW, "G");
    {
      CHKERR add_parent_field_domain(fe, UDO::OPCOL, "H");
      pip.push_back(new OpLhsG_dH<false>("G", "H", h_ptr));
      CHKERR add_parent_field_domain(fe, UDO::OPCOL, "G");
      pip.push_back(new OpLhsG_dG("G"));
    }

    CHKERR add_parent_field_domain(fe, UDO::OPROW, "P");
    {
      CHKERR add_parent_field_domain(fe, UDO::OPCOL, "U");

      switch (coord_type) {
      case CARTESIAN:
        pip.push_back(new OpMixScalarTimesDiv<CARTESIAN>(
            "P", "U",
            [](const double r, const double, const double) {
              return cylindrical(r);
            },
            true, false));
        break;
      case CYLINDRICAL:
        pip.push_back(new OpMixScalarTimesDiv<CYLINDRICAL>(
            "P", "U",
            [](const double r, const double, const double) {
              return cylindrical(r);
            },
            true, false));
        break;
      default:
        SETERRQ(PETSC_COMM_WORLD, MOFEM_NOT_IMPLEMENTED,
                "Coordinate system not implemented");
      }

      CHKERR add_parent_field_domain(fe, UDO::OPCOL, "P");
      pip.push_back(new OpDomainMassP("P", "P", [](double r, double, double) {
        return eps * cylindrical(r);
      }));
    }

    MoFEMFunctionReturn(0);
  };

  auto get_block_name = [](auto name) {
    return boost::format("%s(.*)") % "WETTING_ANGLE";
  };

  auto get_blocks = [&](auto &&name) {
    return mField.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(
        std::regex(name.str()));
  };

  auto set_boundary_rhs = [&](auto fe) {
    MoFEMFunctionBegin;
    auto &pip = fe->getOpPtrVector();

    CHKERR setParentDofs(
        fe, "", UDO::OPSPACE, bit(get_skin_parent_bit()),

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new BoundaryParentEle(mField));
          return fe_parent;
        },

        QUIET, Sev::noisy);

    CHKERR add_parent_field_bdy(fe, UDO::OPROW, "U");
    pip.push_back(new OpCalculateVectorFieldValues<U_FIELD_DIM>("U", u_ptr));

    CHKERR add_parent_field_bdy(fe, UDO::OPROW, "L");
    pip.push_back(new OpCalculateScalarFieldValues("L", lambda_ptr));
    pip.push_back(new OpNormalConstrainRhs("L", u_ptr));

    CHKERR BoundaryNaturalBC::AddFluxToPipeline<OpFluidFlux>::add(
        pip, mField, "L", {}, "INFLUX",
        [](double r, double, double) { return cylindrical(r); }, Sev::inform);

    CHKERR add_parent_field_bdy(fe, UDO::OPROW, "U");
    pip.push_back(new OpNormalForceRhs("U", lambda_ptr));

    auto wetting_block = get_blocks(get_block_name("WETTING_ANGLE"));
    if (wetting_block.size()) {
      // push operators to the side element which is called from op_bdy_side
      auto op_bdy_side =
          new OpLoopSide<SideEle>(mField, simple->getDomainFEName(), SPACE_DIM);
      op_bdy_side->getSideFEPtr()->exeTestHook = test_bit_child;

      CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
          op_bdy_side->getOpPtrVector(), {H1});

      CHKERR setParentDofs(
          op_bdy_side->getSideFEPtr(), "", UDO::OPSPACE,
          bit(get_skin_parent_bit()),

          [&]() {
            boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
                new DomianParentEle(mField));
            AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
                fe_parent->getOpPtrVector(), {H1});
            return fe_parent;
          },

          QUIET, Sev::noisy);

      CHKERR add_parent_field_domain(op_bdy_side->getSideFEPtr(), UDO::OPROW,
                                     "H");
      op_bdy_side->getOpPtrVector().push_back(
          new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_h_ptr));
      // push bdy side op
      pip.push_back(op_bdy_side);

      // push operators for rhs wetting angle
      for (auto &b : wetting_block) {
        Range force_edges;
        std::vector<double> attr_vec;
        CHKERR b->getMeshsetIdEntitiesByDimension(
            mField.get_moab(), SPACE_DIM - 1, force_edges, true);
        b->getAttributes(attr_vec);
        if (attr_vec.size() != 1)
          SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
                  "Should be one attribute");
        MOFEM_LOG("FS", Sev::inform) << "Wetting angle: " << attr_vec.front();
        // need to find the attributes and pass to operator
        CHKERR add_parent_field_bdy(fe, UDO::OPROW, "G");
        pip.push_back(new OpWettingAngleRhs(
            "G", grad_h_ptr, boost::make_shared<Range>(force_edges),
            attr_vec.front()));
      }
    }

    MoFEMFunctionReturn(0);
  };

  auto set_boundary_lhs = [&](auto fe) {
    MoFEMFunctionBegin;
    auto &pip = fe->getOpPtrVector();

    CHKERR setParentDofs(
        fe, "", UDO::OPSPACE, bit(get_skin_parent_bit()),

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new BoundaryParentEle(mField));
          return fe_parent;
        },

        QUIET, Sev::noisy);

    CHKERR add_parent_field_bdy(fe, UDO::OPROW, "L");
    CHKERR add_parent_field_bdy(fe, UDO::OPCOL, "U");
    pip.push_back(new OpNormalConstrainLhs("L", "U"));

    auto wetting_block = get_blocks(get_block_name("WETTING_ANGLE"));
    if (wetting_block.size()) {
      auto col_ind_ptr = boost::make_shared<std::vector<VectorInt>>();
      auto col_diff_base_ptr = boost::make_shared<std::vector<MatrixDouble>>();

      // push operators to the side element which is called from op_bdy_side
      auto op_bdy_side =
          new OpLoopSide<SideEle>(mField, simple->getDomainFEName(), SPACE_DIM);
      op_bdy_side->getSideFEPtr()->exeTestHook = test_bit_child;

      CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
          op_bdy_side->getOpPtrVector(), {H1});

      CHKERR setParentDofs(
          op_bdy_side->getSideFEPtr(), "", UDO::OPSPACE,
          bit(get_skin_parent_bit()),

          [&]() {
            boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
                new DomianParentEle(mField));
            AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
                fe_parent->getOpPtrVector(), {H1});
            return fe_parent;
          },

          QUIET, Sev::noisy);

      CHKERR add_parent_field_domain(op_bdy_side->getSideFEPtr(), UDO::OPROW,
                                     "H");
      op_bdy_side->getOpPtrVector().push_back(
          new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_h_ptr));
      CHKERR add_parent_field_domain(op_bdy_side->getSideFEPtr(), UDO::OPCOL,
                                     "H");
      op_bdy_side->getOpPtrVector().push_back(
          new OpLoopSideGetDataForSideEle("H", col_ind_ptr, col_diff_base_ptr));

      // push bdy side op
      pip.push_back(op_bdy_side);

      // push operators for lhs wetting angle
      for (auto &b : wetting_block) {
        Range force_edges;
        std::vector<double> attr_vec;
        CHKERR b->getMeshsetIdEntitiesByDimension(
            mField.get_moab(), SPACE_DIM - 1, force_edges, true);
        b->getAttributes(attr_vec);
        if (attr_vec.size() != 1)
          SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
                  "Should be one attribute");
        MOFEM_LOG("FS", Sev::inform)
            << "wetting angle edges size " << force_edges.size();

        CHKERR add_parent_field_bdy(fe, UDO::OPROW, "G");
        pip.push_back(new OpWettingAngleLhs(
            "G", grad_h_ptr, col_ind_ptr, col_diff_base_ptr,
            boost::make_shared<Range>(force_edges), attr_vec.front()));
      }
    }

    MoFEMFunctionReturn(0);
  };

  auto *pip_mng = mField.getInterface<PipelineManager>();

  CHKERR set_domain_rhs(pip_mng->getCastDomainRhsFE());
  CHKERR set_domain_lhs(pip_mng->getCastDomainLhsFE());
  CHKERR set_boundary_rhs(pip_mng->getCastBoundaryRhsFE());
  CHKERR set_boundary_lhs(pip_mng->getCastBoundaryLhsFE());

  CHKERR pip_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pip_mng->setDomainLhsIntegrationRule(integration_rule);
  CHKERR pip_mng->setBoundaryRhsIntegrationRule(integration_rule);
  CHKERR pip_mng->setBoundaryLhsIntegrationRule(integration_rule);

  pip_mng->getDomainLhsFE()->exeTestHook = test_bit_child;
  pip_mng->getDomainRhsFE()->exeTestHook = test_bit_child;
  pip_mng->getBoundaryLhsFE()->exeTestHook = test_bit_child;
  pip_mng->getBoundaryRhsFE()->exeTestHook = test_bit_child;

  MoFEMFunctionReturn(0);
}
//! [Push operators to pip]

/**
 * @brief Monitor solution
 *
 * This functions is called by TS kso at the end of each step. It is used
 */
struct Monitor : public FEMethod {
  Monitor(
      SmartPetscObj<DM> dm, boost::shared_ptr<moab::Core> post_proc_mesh,
      boost::shared_ptr<PostProcEleDomainCont> post_proc,
      boost::shared_ptr<PostProcEleBdyCont> post_proc_edge,
      std::pair<boost::shared_ptr<BoundaryEle>, boost::shared_ptr<VectorDouble>>
          p)
      : dM(dm), postProcMesh(post_proc_mesh), postProc(post_proc),
        postProcEdge(post_proc_edge), liftFE(p.first), liftVec(p.second) {}
  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;

    MOFEM_LOG("FS", Sev::verbose) << "Monitor";
    constexpr int save_every_nth_step = 1;
    if (ts_step % save_every_nth_step == 0) {
      MOFEM_LOG("FS", Sev::verbose) << "Mesh pre proc";
      MoFEM::Interface *m_field_ptr;
      CHKERR DMoFEMGetInterfacePtr(dM, &m_field_ptr);
      auto post_proc_begin =
          boost::make_shared<PostProcBrokenMeshInMoabBaseBegin>(*m_field_ptr,
                                                                postProcMesh);
      auto post_proc_end = boost::make_shared<PostProcBrokenMeshInMoabBaseEnd>(
          *m_field_ptr, postProcMesh);
      CHKERR DMoFEMPreProcessFiniteElements(dM, post_proc_begin->getFEMethod());
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc,
                                      this->getCacheWeakPtr());
      CHKERR DMoFEMLoopFiniteElements(dM, "bFE", postProcEdge,
                                      this->getCacheWeakPtr());
      CHKERR DMoFEMPostProcessFiniteElements(dM, post_proc_end->getFEMethod());
      CHKERR post_proc_end->writeFile(
          "out_step_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
      MOFEM_LOG("FS", Sev::verbose) << "Mesh pre proc done";
    }

    liftVec->resize(SPACE_DIM, false);
    liftVec->clear();
    CHKERR DMoFEMLoopFiniteElements(dM, "bFE", liftFE, this->getCacheWeakPtr());
    MPI_Allreduce(MPI_IN_PLACE, &(*liftVec)[0], SPACE_DIM, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MOFEM_LOG("FS", Sev::inform)
        << "Step " << ts_step << " time " << ts_t
        << " lift vec x: " << (*liftVec)[0] << " y: " << (*liftVec)[1];

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<moab::Core> postProcMesh;
  boost::shared_ptr<PostProcEleDomainCont> postProc;
  boost::shared_ptr<PostProcEleBdyCont> postProcEdge;
  boost::shared_ptr<BoundaryEle> liftFE;
  boost::shared_ptr<VectorDouble> liftVec;
};

//! [Solve]
MoFEMErrorCode FreeSurface::solveSystem() {
  MoFEMFunctionBegin;

  using UDO = ForcesAndSourcesCore::UserDataOperator;

  auto simple = mField.getInterface<Simple>();
  auto pip_mng = mField.getInterface<PipelineManager>();

  auto create_solver_dm = [&](auto dm) -> SmartPetscObj<DM> {
    DM subdm;

    auto setup_subdm = [&](auto dm) {
      MoFEMFunctionBegin;
      auto simple = mField.getInterface<Simple>();
      auto bit_mng = mField.getInterface<BitRefManager>();
      auto dm = simple->getDM();
      auto level_ents_ptr = boost::make_shared<Range>();
      CHKERR bit_mng->getEntitiesByRefLevel(
          bit(get_current_bit()), BitRefLevel().set(), *level_ents_ptr);
      CHKERR DMCreate(mField.get_comm(), &subdm);
      CHKERR DMSetType(subdm, "DMMOFEM");
      CHKERR DMMoFEMCreateSubDM(subdm, dm, "SUB_SOLVER");
      CHKERR DMMoFEMAddElement(subdm, simple->getDomainFEName());
      CHKERR DMMoFEMAddElement(subdm, simple->getBoundaryFEName());
      CHKERR DMMoFEMSetSquareProblem(subdm, PETSC_TRUE);
      CHKERR DMMoFEMSetDestroyProblem(subdm, PETSC_FALSE);
      for (auto f : {"U", "P", "H", "G", "L"}) {
        CHKERR DMMoFEMAddSubFieldRow(subdm, f, level_ents_ptr);
        CHKERR DMMoFEMAddSubFieldCol(subdm, f, level_ents_ptr);
      }
      CHKERR DMSetUp(subdm);
      MoFEMFunctionReturn(0);
    };

    CHK_THROW_MESSAGE(setup_subdm(dm), "create subdm");

    return SmartPetscObj<DM>(subdm);
  };

  auto get_fe_post_proc = [&](auto post_proc_mesh) {
    auto add_parent_field_domain = [&](auto fe, auto op, auto field) {
      return setParentDofs(
          fe, field, op, bit(get_skin_parent_bit()),

          [&]() {
            boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
                new DomianParentEle(mField));
            return fe_parent;
          },

          QUIET, Sev::noisy);
    };

    auto post_proc_fe =
        boost::make_shared<PostProcEleDomainCont>(mField, post_proc_mesh);
    post_proc_fe->exeTestHook = [](FEMethod *fe_ptr) {
      return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
          get_start_bit() + nb_levels - 1);
    };

    auto u_ptr = boost::make_shared<MatrixDouble>();
    auto grad_u_ptr = boost::make_shared<MatrixDouble>();
    auto h_ptr = boost::make_shared<VectorDouble>();
    auto grad_h_ptr = boost::make_shared<MatrixDouble>();
    auto p_ptr = boost::make_shared<VectorDouble>();
    auto g_ptr = boost::make_shared<VectorDouble>();
    auto grad_g_ptr = boost::make_shared<MatrixDouble>();

    AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        post_proc_fe->getOpPtrVector(), {H1});

    CHKERR setParentDofs(
        post_proc_fe, "", UDO::OPSPACE, bit(get_skin_parent_bit()),

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new DomianParentEle(mField));
          AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
              fe_parent->getOpPtrVector(), {H1});
          return fe_parent;
        },

        QUIET, Sev::noisy);

    CHKERR add_parent_field_domain(post_proc_fe, UDO::OPROW, "U");
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<U_FIELD_DIM>("U", u_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<U_FIELD_DIM, SPACE_DIM>("U",
                                                                   grad_u_ptr));

    CHKERR add_parent_field_domain(post_proc_fe, UDO::OPROW, "H");
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("H", h_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_h_ptr));

    CHKERR add_parent_field_domain(post_proc_fe, UDO::OPROW, "P");
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("P", p_ptr));

    CHKERR add_parent_field_domain(post_proc_fe, UDO::OPROW, "G");
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("G", g_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("G", grad_g_ptr));

    using OpPPMap = OpPostProcMapInMoab<2, 2>;

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(
            post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

            {{"H", h_ptr}, {"P", p_ptr}, {"G", g_ptr}},

            {{"U", u_ptr}, {"H_GRAD", grad_h_ptr}, {"G_GRAD", grad_g_ptr}},

            {{"GRAD_U", grad_u_ptr}},

            {}

            )

    );

    return post_proc_fe;
  };

  auto get_bdy_post_proc_fe = [&](auto post_proc_mesh) {
    auto add_parent_field_bdy = [&](auto fe, auto op, auto field) {
      return setParentDofs(
          fe, field, op, bit(get_skin_parent_bit()),

          [&]() {
            boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
                new BoundaryParentEle(mField));
            return fe_parent;
          },

          QUIET, Sev::noisy);
    };

    auto post_proc_fe =
        boost::make_shared<PostProcEleBdyCont>(mField, post_proc_mesh);
    post_proc_fe->exeTestHook = [](FEMethod *fe_ptr) {
      return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
          get_start_bit() + nb_levels - 1);
    };

    CHKERR setParentDofs(
        post_proc_fe, "", UDO::OPSPACE, bit(get_skin_parent_bit()),

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new BoundaryParentEle(mField));
          return fe_parent;
        },

        QUIET, Sev::noisy);

    struct OpGetNormal : public BoundaryEleOp {

      OpGetNormal(boost::shared_ptr<VectorDouble> l_ptr,
                  boost::shared_ptr<MatrixDouble> n_ptr)
          : BoundaryEleOp(NOSPACE, BoundaryEleOp::OPSPACE), ptrL(l_ptr),
            ptrNormal(n_ptr) {}

      MoFEMErrorCode doWork(int side, EntityType type,
                            EntitiesFieldData::EntData &data) {
        MoFEMFunctionBegin;
        auto t_l = getFTensor0FromVec(*ptrL);
        auto t_n_fe = getFTensor1NormalsAtGaussPts();
        ptrNormal->resize(SPACE_DIM, getGaussPts().size2(), false);
        auto t_n = getFTensor1FromMat<SPACE_DIM>(*ptrNormal);
        for (auto gg = 0; gg != getGaussPts().size2(); ++gg) {
          t_n(i) = t_n_fe(i) * t_l / std::sqrt(t_n_fe(i) * t_n_fe(i));
          ++t_n_fe;
          ++t_l;
          ++t_n;
        }
        MoFEMFunctionReturn(0);
      };

    protected:
      boost::shared_ptr<VectorDouble> ptrL;
      boost::shared_ptr<MatrixDouble> ptrNormal;
    };

    auto u_ptr = boost::make_shared<MatrixDouble>();
    auto p_ptr = boost::make_shared<VectorDouble>();
    auto lambda_ptr = boost::make_shared<VectorDouble>();
    auto normal_l_ptr = boost::make_shared<MatrixDouble>();

    CHKERR add_parent_field_bdy(post_proc_fe, UDO::OPROW, "U");
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<U_FIELD_DIM>("U", u_ptr));

    CHKERR add_parent_field_bdy(post_proc_fe, UDO::OPROW, "L");
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("L", lambda_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpGetNormal(lambda_ptr, normal_l_ptr));

    CHKERR add_parent_field_bdy(post_proc_fe, UDO::OPROW, "P");
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("P", p_ptr));

    auto op_ptr = new BoundaryEleOp(NOSPACE, BoundaryEleOp::OPSPACE);
    op_ptr->doWorkRhsHook = [&](DataOperator *base_op_ptr, int side,
                                EntityType type,
                                EntitiesFieldData::EntData &data) {
      MoFEMFunctionBegin;
      auto op_ptr = static_cast<BoundaryEleOp *>(base_op_ptr);
      MoFEMFunctionReturn(0);
    };

    using OpPPMap = OpPostProcMapInMoab<2, 2>;

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(post_proc_fe->getPostProcMesh(),
                    post_proc_fe->getMapGaussPts(),

                    OpPPMap::DataMapVec{{"P", p_ptr}},

                    OpPPMap::DataMapMat{{"U", u_ptr}, {"L", normal_l_ptr}},

                    OpPPMap::DataMapMat(),

                    OpPPMap::DataMapMat()

                        )

    );

    return post_proc_fe;
  };

  auto get_lift_fe = [&]() {
    auto add_parent_field_bdy = [&](auto fe, auto op, auto field) {
      return setParentDofs(
          fe, field, op, bit(get_skin_parent_bit()),

          [&]() {
            boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
                new BoundaryParentEle(mField));
            return fe_parent;
          },

          QUIET, Sev::noisy);
    };

    auto fe = boost::make_shared<BoundaryEle>(mField);
    fe->exeTestHook = [](FEMethod *fe_ptr) {
      return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
          get_start_bit() + nb_levels - 1);
    };

    auto lift_ptr = boost::make_shared<VectorDouble>();
    auto p_ptr = boost::make_shared<VectorDouble>();
    auto ents_ptr = boost::make_shared<Range>();

    CHKERR setParentDofs(
        fe, "", UDO::OPSPACE, bit(get_skin_parent_bit()),

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new BoundaryParentEle(mField));
          return fe_parent;
        },

        QUIET, Sev::noisy);

    std::vector<const CubitMeshSets *> vec_ptr;
    CHKERR mField.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(
        std::regex("LIFT"), vec_ptr);
    for (auto m_ptr : vec_ptr) {
      auto meshset = m_ptr->getMeshset();
      Range ents;
      CHKERR mField.get_moab().get_entities_by_dimension(meshset, SPACE_DIM - 1,
                                                         ents, true);
      ents_ptr->merge(ents);
    }

    MOFEM_LOG("FS", Sev::noisy) << "Lift ents " << (*ents_ptr);

    CHKERR add_parent_field_bdy(fe, UDO::OPROW, "P");
    fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("L", p_ptr));
    fe->getOpPtrVector().push_back(
        new OpCalculateLift("L", p_ptr, lift_ptr, ents_ptr));

    return std::make_pair(fe, lift_ptr);
  };

  auto set_post_proc_monitor = [&](auto ts) {
    MoFEMFunctionBegin;
    DM dm;
    CHKERR TSGetDM(ts, &dm);
    boost::shared_ptr<FEMethod> null_fe;
    auto post_proc_mesh = boost::make_shared<moab::Core>();
    auto monitor_ptr = boost::make_shared<Monitor>(
        SmartPetscObj<DM>(dm, true), post_proc_mesh,
        get_fe_post_proc(post_proc_mesh), get_bdy_post_proc_fe(post_proc_mesh),
        get_lift_fe());
    CHKERR DMMoFEMTSSetMonitor(dm, ts, simple->getDomainFEName(), null_fe,
                               null_fe, monitor_ptr);
    MoFEMFunctionReturn(0);
  };

  auto dm = simple->getDM();
  auto ts = createTS(mField.get_comm());
  CHKERR TSSetDM(ts, dm);

  auto ts_pre_post_proc = boost::make_shared<TSPrePostProc>();
  tsPrePostProc = ts_pre_post_proc;

  if (auto ptr = tsPrePostProc.lock()) {

    ptr->fsRawPtr = this;
    ptr->solverSubDM = create_solver_dm(simple->getDM());
    ptr->globSol = createDMVector(dm);
    CHKERR DMoFEMMeshToLocalVector(dm, ptr->globSol, INSERT_VALUES,
                                   SCATTER_FORWARD);
    CHKERR VecAssemblyBegin(ptr->globSol);
    CHKERR VecAssemblyEnd(ptr->globSol);

    auto sub_ts = pip_mng->createTSIM(ptr->solverSubDM);

    CHKERR set_post_proc_monitor(sub_ts);

    // Add monitor to time solver
    double ftime = 1;
    // CHKERR TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);
    // CHKERR TSSetSolution(ts, ptr->globSol);
    CHKERR TSSetFromOptions(ts);
    CHKERR ptr->tsSetUp(ts);
    CHKERR TSSetUp(ts);

    auto print_fields_in_section = [&]() {
      MoFEMFunctionBegin;
      auto section = mField.getInterface<ISManager>()->sectionCreate(
          simple->getProblemName());
      PetscInt num_fields;
      CHKERR PetscSectionGetNumFields(section, &num_fields);
      for (int f = 0; f < num_fields; ++f) {
        const char *field_name;
        CHKERR PetscSectionGetFieldName(section, f, &field_name);
        MOFEM_LOG("FS", Sev::inform)
            << "Field " << f << " " << std::string(field_name);
      }
      MoFEMFunctionReturn(0);
    };

    CHKERR print_fields_in_section();

    CHKERR TSSolve(ts, ptr->globSol);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode FreeSurface::allBoundaryWet(BitRefLevel bit, BitRefLevel mask) {
  MoFEMFunctionBegin;

  PetscBool all_boundary_wet = PETSC_FALSE;
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-all_boundary_wet",
                             &all_boundary_wet, PETSC_NULL);
  if (all_boundary_wet) {

    moab::Skinner skinner(&mField.get_moab());
    ParallelComm *pcomm =
        ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);

    auto get_bit_ents = [&](BitRefLevel bit, BitRefLevel mask) {
      Range bit_skin;
      for (auto m : mField.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(
               std::regex(

                   (boost::format("%s(.*)") % "ALL_WET").str()

                       ))

      ) {
        auto meshset = m->getMeshset();
        Range wet_ents;
        CHKERR mField.get_moab().get_entities_by_dimension(
            meshset, SPACE_DIM - 1, wet_ents, true);
        bit_skin.merge(wet_ents);
      }

      CHKERR mField.getInterface<BitRefManager>()->filterEntitiesByRefLevel(
          bit, mask, bit_skin);

      return bit_skin;
    };

    auto skin_ents = get_bit_ents(bit, mask);
    Range skin_verts;
    CHKERR mField.get_moab().get_connectivity(skin_ents, skin_verts, true);
    // FIXME: Add faces in edges
    skin_ents.merge(skin_verts);

    auto field_blas = mField.getInterface<FieldBlas>();
    CHKERR field_blas->setField(-1, MBVERTEX, skin_verts, "H");
  }

  MoFEMFunctionReturn(0);
}

int main(int argc, char *argv[]) {

#ifdef PYTHON_INIT_SURFACE
  Py_Initialize();
#endif

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(LogManager::createSink(LogManager::getStrmWorld(), "FS"));
  LogManager::setLog("FS");
  MOFEM_LOG_TAG("FS", "free surface");

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

    //! [FreeSurface]
    FreeSurface ex(m_field);
    CHKERR ex.runProblem();
    //! [FreeSurface]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();

#ifdef PYTHON_INIT_SURFACE
  if (Py_FinalizeEx() < 0) {
    exit(120);
  }
#endif
}

std::vector<Range>
FreeSurface::findEntitiesCrossedByPhaseInterface(size_t overlap) {

  auto &moab = mField.get_moab();
  auto bit_mng = mField.getInterface<BitRefManager>();
  auto comm_mng = mField.getInterface<CommInterface>();

  Range vertices;
  CHK_THROW_MESSAGE(bit_mng->getEntitiesByTypeAndRefLevel(
                        bit(0), BitRefLevel().set(), MBVERTEX, vertices),
                    "can not get vertices on bit 0");

  auto &dofs_mi = mField.get_dofs()->get<Unique_mi_tag>();
  auto field_bit_number = mField.get_field_bit_number("H");

  Range plus_range, minus_range;
  std::vector<EntityHandle> plus, minus;

  // get vertices on level 0 on plus and minus side
  for (auto p = vertices.pair_begin(); p != vertices.pair_end(); ++p) {

    const auto f = p->first;
    const auto s = p->second;

    // Lowest Dof UId for given field (field bit number) on entity f
    const auto lo_uid = DofEntity::getLoFieldEntityUId(field_bit_number, f);
    const auto hi_uid = DofEntity::getHiFieldEntityUId(field_bit_number, s);
    auto it = dofs_mi.lower_bound(lo_uid);
    const auto hi_it = dofs_mi.upper_bound(hi_uid);

    plus.clear();
    minus.clear();
    plus.reserve(std::distance(it, hi_it));
    minus.reserve(std::distance(it, hi_it));

    for (; it != hi_it; ++it) {
      const auto v = (*it)->getFieldData();
      if (v > 0)
        plus.push_back((*it)->getEnt());
      else
        minus.push_back((*it)->getEnt());
    }

    plus_range.insert_list(plus.begin(), plus.end());
    minus_range.insert_list(minus.begin(), minus.end());
  }

  MOFEM_LOG_CHANNEL("SYNC");
  MOFEM_TAG_AND_LOG("SYNC", Sev::noisy, "FS")
      << "Plus range " << plus_range << endl;
  MOFEM_TAG_AND_LOG("SYNC", Sev::noisy, "FS")
      << "Minus range " << minus_range << endl;
  MOFEM_LOG_SEVERITY_SYNC(mField.get_comm(), Sev::noisy);

  auto get_elems = [&](auto &ents, auto bit, auto mask) {
    Range adj;
    CHK_MOAB_THROW(moab.get_adjacencies(ents, SPACE_DIM, false, adj,
                                        moab::Interface::UNION),
                   "can not get adjacencies");
    CHK_THROW_MESSAGE(bit_mng->filterEntitiesByRefLevel(bit, mask, adj),
                      "can not filter elements with bit 0");
    return adj;
  };

  CHKERR comm_mng->synchroniseEntities(plus_range);
  CHKERR comm_mng->synchroniseEntities(minus_range);

  std::vector<Range> ele_plus(nb_levels), ele_minus(nb_levels);
  ele_plus[0] = get_elems(plus_range, bit(0), BitRefLevel().set());
  ele_minus[0] = get_elems(minus_range, bit(0), BitRefLevel().set());
  auto common = intersect(ele_plus[0], ele_minus[0]);
  ele_plus[0] = subtract(ele_plus[0], common);
  ele_minus[0] = subtract(ele_minus[0], common);

  auto get_children = [&](auto &p, auto &c) {
    MoFEMFunctionBegin;
    CHKERR bit_mng->updateRangeByChildren(p, c);
    c = c.subset_by_dimension(SPACE_DIM);
    MoFEMFunctionReturn(0);
  };

  for (auto l = 1; l != nb_levels; ++l) {
    CHK_THROW_MESSAGE(get_children(ele_plus[l - 1], ele_plus[l]),
                      "get children");
    CHK_THROW_MESSAGE(get_children(ele_minus[l - 1], ele_minus[l]),
                      "get children");
  }

  auto get_level = [&](auto &p, auto &m, auto z, auto bit, auto mask) {
    Range l;
    CHK_THROW_MESSAGE(
        bit_mng->getEntitiesByDimAndRefLevel(bit, mask, SPACE_DIM, l),
        "can not get vertices on bit");
    l = subtract(l, p);
    l = subtract(l, m);
    for (auto f = 0; f != z; ++f) {
      Range conn;
      CHK_MOAB_THROW(moab.get_connectivity(l, conn, true), "");
      CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(conn);
      l = get_elems(conn, bit, mask);
    }
    return l;
  };

  std::vector<Range> vec_levels(nb_levels);
  for (auto l = nb_levels - 1; l >= 0; --l) {
    vec_levels[l] = get_level(ele_plus[l], ele_minus[l], 2 * overlap, bit(l),
                              BitRefLevel().set());
  }

  if constexpr (debug) {
    if (mField.get_comm_size() == 1) {
      for (auto l = 0; l != nb_levels; ++l) {
        std::string name = (boost::format("out_r%d.h5m") % l).str();
        CHK_THROW_MESSAGE(save_range(mField.get_moab(), name, vec_levels[l]),
                          "save mesh");
      }
    }
  }

  return vec_levels;
}

MoFEMErrorCode FreeSurface::refineMesh(size_t overlap) {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  auto bit_mng = mField.getInterface<BitRefManager>();
  auto prb_mng = mField.getInterface<ProblemsManager>();

  BitRefLevel start_mask;
  for (auto s = 0; s != get_start_bit(); ++s)
    start_mask[s] = true;

  // store prev_level
  Range prev_level;
  CHKERR bit_mng->getEntitiesByRefLevel(bit(get_current_bit()),
                                        BitRefLevel().set(), prev_level);
  Range prev_level_skin;
  CHKERR bit_mng->getEntitiesByRefLevel(bit(get_skin_parent_bit()),
                                        BitRefLevel().set(), prev_level_skin);
  // reset bit ref levels
  CHKERR bit_mng->lambdaBitRefLevel(
      [&](EntityHandle ent, BitRefLevel &bit) { bit &= start_mask; });
  CHKERR bit_mng->setNthBitRefLevel(prev_level, get_projection_bit(), true);
  CHKERR bit_mng->setNthBitRefLevel(prev_level_skin, get_skin_projection_bit(),
                                    true);

  // set refinement levels
  auto set_levels = [&](auto &&
                            vec_levels /*entities are refined on each level*/) {
    MoFEMFunctionBegin;

    // start with zero level, which is the coarsest mesh
    Range level0;
    CHKERR bit_mng->getEntitiesByRefLevel(bit(0), BitRefLevel().set(), level0);
    CHKERR bit_mng->setNthBitRefLevel(level0, get_start_bit(), true);

    // get lower dimension entities
    auto get_adj = [&](auto ents) {
      Range conn;
      CHK_MOAB_THROW(mField.get_moab().get_connectivity(ents, conn, true),
                     "get conn");
      for (auto d = 1; d != SPACE_DIM; ++d) {
        CHK_MOAB_THROW(mField.get_moab().get_adjacencies(
                           ents.subset_by_dimension(SPACE_DIM), d, false, conn,
                           moab::Interface::UNION),
                       "get adj");
      }
      ents.merge(conn);
      return ents;
    };

    // set bit levels
    for (auto l = 1; l != nb_levels; ++l) {
      Range level_prev;
      CHKERR bit_mng->getEntitiesByDimAndRefLevel(bit(get_start_bit() + l - 1),
                                                  BitRefLevel().set(),
                                                  SPACE_DIM, level_prev);
      Range parents;
      CHKERR bit_mng->updateRangeByParent(vec_levels[l], parents);
      // subtract entities from previous level, which are refined, so should be
      // not there
      level_prev = subtract(level_prev, parents);
      // and instead add their children
      level_prev.merge(vec_levels[l]);
      // set bit to each level
      CHKERR bit_mng->setNthBitRefLevel(level_prev, get_start_bit() + l, true);
    }

    // set bit levels to lower dimension entities
    for (auto l = 1; l != nb_levels; ++l) {
      Range level;
      CHKERR bit_mng->getEntitiesByDimAndRefLevel(
          bit(get_start_bit() + l), BitRefLevel().set(), SPACE_DIM, level);
      level = get_adj(level);
      CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(level);
      CHKERR bit_mng->setNthBitRefLevel(level, get_start_bit() + l, true);
    }

    MoFEMFunctionReturn(0);
  };

  // resolve skin between refined levels
  auto set_skins = [&]() {
    MoFEMFunctionBegin;

    moab::Skinner skinner(&mField.get_moab());
    ParallelComm *pcomm =
        ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);

    // get skin of bit level
    auto get_bit_skin = [&](BitRefLevel bit, BitRefLevel mask) {
      Range bit_ents;
      CHK_THROW_MESSAGE(
          mField.getInterface<BitRefManager>()->getEntitiesByDimAndRefLevel(
              bit, mask, SPACE_DIM, bit_ents),
          "can't get bit level");
      Range bit_skin;
      CHK_MOAB_THROW(skinner.find_skin(0, bit_ents, false, bit_skin),
                     "can't get skin");
      return bit_skin;
    };

    auto get_level_skin = [&]() {
      Range skin;
      BitRefLevel bit_prev;
      for (auto l = 1; l != nb_levels; ++l) {
        auto skin_level_mesh = get_bit_skin(bit(l), BitRefLevel().set());
        // filter (remove) all entities which are on partitions boarders
        CHKERR pcomm->filter_pstatus(skin_level_mesh,
                                     PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                     PSTATUS_NOT, -1, nullptr);
        auto skin_level =
            get_bit_skin(bit(get_start_bit() + l), BitRefLevel().set());
        skin_level = subtract(skin_level,
                              skin_level_mesh); // get only internal skins, not
                                                // on the body boundary
        // get lower dimension adjacencies (FIXME: add edges if 3D)
        Range skin_level_verts;
        CHKERR mField.get_moab().get_connectivity(skin_level, skin_level_verts,
                                                  true);
        skin_level.merge(skin_level_verts);

        // remove previous level
        bit_prev.set(l - 1);
        Range level_prev;
        CHKERR bit_mng->getEntitiesByRefLevel(bit_prev, BitRefLevel().set(),
                                              level_prev);
        skin.merge(subtract(skin_level, level_prev));
      }

      return skin;
    };

    auto resolve_shared = [&](auto &&skin) {
      Range tmp_skin = skin;

      map<int, Range> map_procs;
      CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(
          tmp_skin, &map_procs);

      Range from_other_procs; // entities which also exist on other processors
      for (auto &m : map_procs) {
        if (m.first != mField.get_comm_rank()) {
          from_other_procs.merge(m.second);
        }
      }

      auto common = intersect(
          skin, from_other_procs); // entities which are on internal skin
      skin.merge(from_other_procs);

      // entities which are on processors boards, and several processors are not
      // true skin.
      if (!common.empty()) {
        // skin is internal exist on other procs
        skin = subtract(skin, common);
      }

      return skin;
    };

    // get parents of entities
    auto get_parent_level_skin = [&](auto skin) {
      Range skin_parents;
      CHKERR bit_mng->updateRangeByParent(
          skin.subset_by_dimension(SPACE_DIM - 1), skin_parents);
      Range skin_parent_verts;
      CHKERR mField.get_moab().get_connectivity(skin_parents, skin_parent_verts,
                                                true);
      skin_parents.merge(skin_parent_verts);
      CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(
          skin_parents);
      return skin_parents;
    };

    auto child_skin = resolve_shared(get_level_skin());
    auto parent_skin = get_parent_level_skin(child_skin);

    child_skin = subtract(child_skin, parent_skin);
    CHKERR bit_mng->setNthBitRefLevel(child_skin, get_skin_child_bit(), true);
    CHKERR bit_mng->setNthBitRefLevel(parent_skin, get_skin_parent_bit(), true);

    MoFEMFunctionReturn(0);
  };

  // take last level, remove childs on boarder, and set bit
  auto set_current = [&]() {
    MoFEMFunctionBegin;
    Range last_level;
    CHKERR bit_mng->getEntitiesByRefLevel(bit(get_start_bit() + nb_levels - 1),
                                          BitRefLevel().set(), last_level);
    Range skin_child;
    CHKERR bit_mng->getEntitiesByRefLevel(bit(get_skin_child_bit()),
                                          BitRefLevel().set(), skin_child);

    last_level = subtract(last_level, skin_child);
    CHKERR bit_mng->setNthBitRefLevel(last_level, get_current_bit(), true);
    MoFEMFunctionReturn(0);
  };

  // set bits to levels
  CHKERR set_levels(findEntitiesCrossedByPhaseInterface(overlap));
  // set bits to skin
  CHKERR set_skins();
  // set current level bit
  CHKERR set_current();

  if constexpr (debug) {
    if (mField.get_comm_size() == 1) {
      for (auto l = 0; l != nb_levels; ++l) {
        std::string name = (boost::format("out_level%d.h5m") % l).str();
        CHKERR bit_mng->writeBitLevel(BitRefLevel().set(get_start_bit() + l),
                                      BitRefLevel().set(), name.c_str(), "MOAB",
                                      "PARALLEL=WRITE_PART");
      }
      CHKERR bit_mng->writeBitLevel(BitRefLevel().set(get_current_bit()),
                                    BitRefLevel().set(), "current_bit.h5m",
                                    "MOAB", "PARALLEL=WRITE_PART");
      CHKERR bit_mng->writeBitLevel(BitRefLevel().set(get_projection_bit()),
                                    BitRefLevel().set(), "projection_bit.h5m",
                                    "MOAB", "PARALLEL=WRITE_PART");

      CHKERR bit_mng->writeBitLevel(BitRefLevel().set(get_skin_child_bit()),
                                    BitRefLevel().set(), "skin_child_bit.h5m",
                                    "MOAB", "PARALLEL=WRITE_PART");
      CHKERR bit_mng->writeBitLevel(BitRefLevel().set(get_skin_parent_bit()),
                                    BitRefLevel().set(), "skin_parent_bit.h5m",
                                    "MOAB", "PARALLEL=WRITE_PART");
    }
  }

  MoFEMFunctionReturn(0);
};

MoFEMErrorCode FreeSurface::setParentDofs(
    boost::shared_ptr<FEMethod> fe_top, std::string field_name,
    ForcesAndSourcesCore::UserDataOperator::OpType op,
    BitRefLevel child_ent_bit,
    boost::function<boost::shared_ptr<ForcesAndSourcesCore>()> get_elem,
    int verbosity, LogManager::SeverityLevel sev) {
  MoFEMFunctionBegin;

  /**
   * @brief Collect data from parent elements to child
   */
  boost::function<void(boost::shared_ptr<ForcesAndSourcesCore>, int)>
      add_parent_level =
          [&](boost::shared_ptr<ForcesAndSourcesCore> parent_fe_pt, int level) {
            // Evaluate if not last parent element
            if (level > 0) {

              // Create domain parent FE
              auto fe_ptr_current = get_elem();

              // Call next level
              add_parent_level(
                  boost::dynamic_pointer_cast<ForcesAndSourcesCore>(
                      fe_ptr_current),
                  level - 1);

              // Add data to curent fe level
              if (op == ForcesAndSourcesCore::UserDataOperator::OPSPACE) {

                // Only base
                parent_fe_pt->getOpPtrVector().push_back(

                    new OpAddParentEntData(

                        H1, op, fe_ptr_current,

                        BitRefLevel().set(), BitRefLevel().set(),

                        child_ent_bit, BitRefLevel().set(),

                        verbosity, sev));

              } else {

                // Filed data
                parent_fe_pt->getOpPtrVector().push_back(

                    new OpAddParentEntData(

                        field_name, op, fe_ptr_current,

                        BitRefLevel().set(), BitRefLevel().set(),

                        child_ent_bit, BitRefLevel().set(),

                        verbosity, sev));
              }
            }
          };

  add_parent_level(boost::dynamic_pointer_cast<ForcesAndSourcesCore>(fe_top),
                   nb_levels);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode TSPrePostProc::tsPreProc(TS ts) {
  MoFEMFunctionBegin;

  if (auto ptr = tsPrePostProc.lock()) {
    MOFEM_LOG("FS", Sev::inform) << "Run step pre proc";

    auto &m_field = ptr->fsRawPtr->mField;
    auto simple = m_field.getInterface<Simple>();
    auto bit_mng = m_field.getInterface<BitRefManager>();
    auto bc_mng = m_field.getInterface<BcManager>();
    auto field_blas = m_field.getInterface<FieldBlas>();
    auto opt = m_field.getInterface<OperatorsTester>();
    auto pip_mng = m_field.getInterface<PipelineManager>();
    auto prb_mng = m_field.getInterface<ProblemsManager>();

    // get vector norm
    auto get_norm = [&](auto x) {
      double nrm;
      CHKERR VecNorm(x, NORM_2, &nrm);
      return nrm;
    };

    // refine problem and project data, including theta data
    auto refine_problem = [&]() {
      MoFEMFunctionBegin;
      MOFEM_LOG("FS", Sev::inform) << "Refine problem";
      CHKERR ptr->fsRawPtr->refineMesh(refine_overlap);
      CHKERR ptr->fsRawPtr->projectData();
      CHKERR ptr->fsRawPtr->allBoundaryWet(get_current_bit(),
                                           BitRefLevel().set());
      MoFEMFunctionReturn(0);
    };

    // set new jacobin operator, since problem and thus tangent matrix size has
    // changed
    auto set_jacobian_operators = [&]() {
      MoFEMFunctionBegin;
      ptr->subB = createDMMatrix(ptr->solverSubDM);
      CHKERR KSPReset(ptr->subKSP);
      MoFEMFunctionReturn(0);
    };

    // set new solution
    auto set_solution = [&]() {
      MoFEMFunctionBegin;
      MOFEM_LOG("FS", Sev::inform) << "Set solution";

      PetscObjectState state;

      // Record the state, and set it again. This is to fool PETSc that solution
      // vector is not updated. Otherwise PETSc will treat every step as a first
      // step.

      // globSol is updated as result mesh refinement -  this is not really set
      // a new solution.

      CHKERR PetscObjectStateGet(getPetscObject(ptr->globSol.get()), &state);
      CHKERR DMoFEMMeshToLocalVector(simple->getDM(), ptr->globSol,
                                     INSERT_VALUES, SCATTER_FORWARD);
      CHKERR PetscObjectStateSet(getPetscObject(ptr->globSol.get()), state);
      MOFEM_LOG("FS", Sev::verbose)
          << "Set solution, vector norm " << get_norm(ptr->globSol);
      MoFEMFunctionReturn(0);
    };

    PetscBool is_theta;
    PetscObjectTypeCompare((PetscObject)ts, TSTHETA, &is_theta);
    if (is_theta) {

      CHKERR refine_problem();         // refine problem
      CHKERR set_jacobian_operators(); // set new jacobian
      CHKERR set_solution();           // set solution

    } else {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_NOT_IMPLEMENTED,
              "Sorry, only TSTheta handling is implemented");
    }

    // Need barriers, somme functions in TS solver need are called collectively
    // and requite the same state of variables
    PetscBarrier((PetscObject)ts);

    MOFEM_LOG_CHANNEL("SYNC");
    MOFEM_TAG_AND_LOG("SYNC", Sev::verbose, "FS") << "PreProc done";
    MOFEM_LOG_SEVERITY_SYNC(m_field.get_comm(), Sev::verbose);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode TSPrePostProc::tsPostProc(TS ts) {
  if (auto ptr = tsPrePostProc.lock()) {
    auto &m_field = ptr->fsRawPtr->mField;
    MOFEM_LOG_CHANNEL("SYNC");
    MOFEM_TAG_AND_LOG("SYNC", Sev::verbose, "FS") << "PostProc done";
    MOFEM_LOG_SEVERITY_SYNC(m_field.get_comm(), Sev::verbose);
  }
  return 0;
}

MoFEMErrorCode TSPrePostProc::tsSetIFunction(TS ts, PetscReal t, Vec u, Vec u_t,
                                             Vec f, void *ctx) {
  MoFEMFunctionBegin;
  if (auto ptr = tsPrePostProc.lock()) {
    auto sub_u = ptr->getSubVector();
    auto sub_u_t = vectorDuplicate(sub_u);
    auto sub_f = vectorDuplicate(sub_u);
    auto scatter = ptr->getScatter(sub_u, u, R);

    auto apply_scatter_and_update = [&](auto x, auto sub_x) {
      MoFEMFunctionBegin;
      CHKERR VecScatterBegin(scatter, x, sub_x, INSERT_VALUES, SCATTER_REVERSE);
      CHKERR VecScatterEnd(scatter, x, sub_x, INSERT_VALUES, SCATTER_REVERSE);
      CHKERR VecGhostUpdateBegin(sub_x, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(sub_x, INSERT_VALUES, SCATTER_FORWARD);
      MoFEMFunctionReturn(0);
    };

    CHKERR apply_scatter_and_update(u, sub_u);
    CHKERR apply_scatter_and_update(u_t, sub_u_t);

    CHKERR TsSetIFunction(ts, t, sub_u, sub_u_t, sub_f, ptr->tsCtxPtr.get());
    CHKERR VecScatterBegin(scatter, sub_f, f, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecScatterEnd(scatter, sub_f, f, INSERT_VALUES, SCATTER_FORWARD);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode TSPrePostProc::tsSetIJacobian(TS ts, PetscReal t, Vec u, Vec u_t,
                                             PetscReal a, Mat A, Mat B,
                                             void *ctx) {
  MoFEMFunctionBegin;
  if (auto ptr = tsPrePostProc.lock()) {
    auto sub_u = ptr->getSubVector();
    auto sub_u_t = vectorDuplicate(sub_u);
    auto scatter = ptr->getScatter(sub_u, u, R);

    auto apply_scatter_and_update = [&](auto x, auto sub_x) {
      MoFEMFunctionBegin;
      CHKERR VecScatterBegin(scatter, x, sub_x, INSERT_VALUES, SCATTER_REVERSE);
      CHKERR VecScatterEnd(scatter, x, sub_x, INSERT_VALUES, SCATTER_REVERSE);
      CHKERR VecGhostUpdateBegin(sub_x, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(sub_x, INSERT_VALUES, SCATTER_FORWARD);
      MoFEMFunctionReturn(0);
    };

    CHKERR apply_scatter_and_update(u, sub_u);
    CHKERR apply_scatter_and_update(u_t, sub_u_t);

    CHKERR TsSetIJacobian(ts, t, sub_u, sub_u_t, a, ptr->subB, ptr->subB,
                          ptr->tsCtxPtr.get());
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode TSPrePostProc::tsMonitor(TS ts, PetscInt step, PetscReal t,
                                        Vec u, void *ctx) {
  MoFEMFunctionBegin;
  if (auto ptr = tsPrePostProc.lock()) {
    auto get_norm = [&](auto x) {
      double nrm;
      CHKERR VecNorm(x, NORM_2, &nrm);
      return nrm;
    };

    auto sub_u = ptr->getSubVector();
    auto scatter = ptr->getScatter(sub_u, u, R);
    CHKERR VecScatterBegin(scatter, u, sub_u, INSERT_VALUES, SCATTER_REVERSE);
    CHKERR VecScatterEnd(scatter, u, sub_u, INSERT_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateBegin(sub_u, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(sub_u, INSERT_VALUES, SCATTER_FORWARD);

    MOFEM_LOG("FS", Sev::verbose)
        << "u norm " << get_norm(u) << " u sub nom " << get_norm(sub_u);

    CHKERR TsMonitorSet(ts, step, t, sub_u, ptr->tsCtxPtr.get());
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode TSPrePostProc::pcSetup(PC pc) {
  MoFEMFunctionBegin;
  if (auto ptr = tsPrePostProc.lock()) {
    MOFEM_LOG("FS", Sev::verbose) << "SetUP sub PC";
    ptr->subKSP = createKSP(ptr->fsRawPtr->mField.get_comm());
    CHKERR KSPSetFromOptions(ptr->subKSP);
  }
  MoFEMFunctionReturn(0);
};

MoFEMErrorCode TSPrePostProc::pcApply(PC pc, Vec pc_f, Vec pc_x) {
  MoFEMFunctionBegin;
  if (auto ptr = tsPrePostProc.lock()) {
    auto sub_x = ptr->getSubVector();
    auto sub_f = vectorDuplicate(sub_x);
    auto scatter = ptr->getScatter(sub_x, pc_x, R);
    CHKERR VecScatterBegin(scatter, pc_f, sub_f, INSERT_VALUES,
                           SCATTER_REVERSE);
    CHKERR VecScatterEnd(scatter, pc_f, sub_f, INSERT_VALUES, SCATTER_REVERSE);
    CHKERR KSPSetOperators(ptr->subKSP, ptr->subB, ptr->subB);
    MOFEM_LOG("FS", Sev::verbose) << "PCShell solve";
    CHKERR KSPSolve(ptr->subKSP, sub_f, sub_x);
    MOFEM_LOG("FS", Sev::verbose) << "PCShell solve <- done";
    CHKERR VecScatterBegin(scatter, sub_x, pc_x, INSERT_VALUES,
                           SCATTER_FORWARD);
    CHKERR VecScatterEnd(scatter, sub_x, pc_x, INSERT_VALUES, SCATTER_FORWARD);
  }
  MoFEMFunctionReturn(0);
};

SmartPetscObj<VecScatter> TSPrePostProc::getScatter(Vec x, Vec y, enum FR fr) {
  if (auto ptr = tsPrePostProc.lock()) {
    auto prb_ptr = ptr->fsRawPtr->mField.get_problem("SUB_SOLVER");
    if (auto sub_data = prb_ptr->getSubData()) {
      auto is = sub_data->getSmartColIs();
      VecScatter s;
      if (fr == R) {
        CHK_THROW_MESSAGE(VecScatterCreate(x, PETSC_NULL, y, is, &s),
                          "crate scatter");
      } else {
        CHK_THROW_MESSAGE(VecScatterCreate(x, is, y, PETSC_NULL, &s),
                          "crate scatter");
      }
      return SmartPetscObj<VecScatter>(s);
    }
  }
  CHK_THROW_MESSAGE(MOFEM_DATA_INCONSISTENCY, "No prb pinter");
  return SmartPetscObj<VecScatter>();
}

SmartPetscObj<Vec> TSPrePostProc::getSubVector() {
  return createDMVector(solverSubDM);
}

MoFEMErrorCode TSPrePostProc::tsSetUp(TS ts) {

  auto &m_field = fsRawPtr->mField;
  auto simple = m_field.getInterface<Simple>();

  MoFEMFunctionBegin;

  auto dm = simple->getDM();

  globRes = vectorDuplicate(globSol);
  CHKERR TSSetIFunction(ts, globRes, tsSetIFunction, nullptr);
  CHKERR TSSetIJacobian(ts, PETSC_NULL, PETSC_NULL, tsSetIJacobian, nullptr);
  CHKERR TSMonitorSet(ts, tsMonitor, fsRawPtr, PETSC_NULL);

  SNES snes;
  CHKERR TSGetSNES(ts, &snes);
  auto snes_ctx_ptr = getDMSnesCtx(dm);

  auto set_section_monitor = [&](auto snes) {
    MoFEMFunctionBegin;
    CHKERR SNESMonitorSet(snes,
                          (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal,
                                             void *))MoFEMSNESMonitorFields,
                          (void *)(snes_ctx_ptr.get()), nullptr);
    MoFEMFunctionReturn(0);
  };

  CHKERR set_section_monitor(snes);

  auto ksp = createKSP(m_field.get_comm());
  CHKERR KSPSetType(ksp, KSPPREONLY); // Run KSP internally in ShellPC
  auto sub_pc = createPC(fsRawPtr->mField.get_comm());
  CHKERR PCSetType(sub_pc, PCSHELL);
  CHKERR PCShellSetSetUp(sub_pc, pcSetup);
  CHKERR PCShellSetApply(sub_pc, pcApply);
  CHKERR KSPSetPC(ksp, sub_pc);
  CHKERR SNESSetKSP(snes, ksp);

  snesCtxPtr = getDMSnesCtx(solverSubDM);
  tsCtxPtr = getDMTsCtx(solverSubDM);

  CHKERR TSSetPreStep(ts, tsPreProc);
  CHKERR TSSetPostStep(ts, tsPostProc);

  MoFEMFunctionReturn(0);
}