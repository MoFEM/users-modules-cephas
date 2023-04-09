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

using namespace MoFEM;

static char help[] = "...\n\n";

#include <BasicFiniteElements.hpp>

constexpr int BASE_DIM = 1;
constexpr int SPACE_DIM = 2;
constexpr int U_FIELD_DIM = SPACE_DIM;
constexpr CoordinateTypes coord_type =
    EXECUTABLE_COORD_TYPE; ///< select coordinate system <CARTESIAN,
                           ///< CYLINDRICAL>;

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

using AssemblyDomainEleOp =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::OpBase;
using AssemblyBoundaryEleOp =
    FormsIntegrators<BoundaryEleOp>::Assembly<PETSC>::OpBase;

using OpDomainMassU = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<BASE_DIM, U_FIELD_DIM>;
using OpDomainMassH = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<BASE_DIM, 1>;
using OpDomainMassP = OpDomainMassH;

using OpDomainSourceU = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<BASE_DIM, U_FIELD_DIM>;
using OpDomainSourceH = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<BASE_DIM, 1>;

using OpBaseTimesScalar = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesScalar<1, 1>;

using OpMixScalarTimesDiv = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMixScalarTimesDiv<SPACE_DIM, coord_type>;

FTensor::Index<'i', SPACE_DIM> i;
FTensor::Index<'j', SPACE_DIM> j;
FTensor::Index<'k', SPACE_DIM> k;
FTensor::Index<'l', SPACE_DIM> l;

constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();

// mesh refinement
constexpr int order = 3; ///< approximation order

constexpr bool debug = true;
constexpr int nb_levels = 4; //< number of refinement levels
constexpr int start_bit =
    nb_levels + 1; //< first refinement level for computational mesh
constexpr int current_bit =
    2 * start_bit + 1; ///< dofs bit used to do calculations

// Physical parameters
constexpr double a0 = 980;
constexpr double rho_m = 0.998;
constexpr double mu_m = 1.0101;
constexpr double rho_p = 0.0012;
constexpr double mu_p = 0.0182;
constexpr double lambda = 73;
constexpr double W = 0.25;
constexpr double cos_alpha = 70; // wetting angle

template <int T> constexpr int powof2() {
  if constexpr (T == 0)
    return 1;
  else
    return powof2<T - 1>() * 2;
};

// Model parameters
constexpr double h = 0.025 / (nb_levels); // mesh size
constexpr double eta = h;
constexpr double eta2 = eta * eta;

// Numerical parameters
constexpr double md = 1e-3;
constexpr double eps = 1e-12;
constexpr double tol = std::numeric_limits<float>::epsilon();

constexpr double rho_ave = (rho_p + rho_m) / 2;
constexpr double rho_diff = (rho_p - rho_m) / 2;
constexpr double mu_ave = (mu_p + mu_m) / 2;
constexpr double mu_diff = (mu_p - mu_m) / 2;

const double kappa = (3. / (4. * std::sqrt(2. * W))) * (lambda / eta);

auto integration_rule = [](int, int, int) { return 2 * order; };

auto cylindrical = [](const double r) {
  // When we move to C++17 add if constexpr()
  if constexpr (coord_type == CYLINDRICAL)
    return 2 * M_PI * r;
  else
    return 1.;
};

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

auto init_h = [](double r, double y, double theta) {
  return capillary_tube(r, y, theta);
  // return kernel_eye(r, y, theta);
};

auto wetting_angle = [](double water_level) { return water_level; };

auto bit = [](auto b) { return BitRefLevel().set(b); };
auto marker = [](auto b) { return BitRefLevel().set(BITREFLEVEL_SIZE - b); };

auto save_range = [](moab::Interface &moab, const std::string name,
                     const Range r) {
  MoFEMFunctionBegin;
  if (r.size()) {
    auto out_meshset = get_temp_meshset_ptr(moab);
    CHKERR moab.add_entities(*out_meshset, r);
    CHKERR moab.write_file(name.c_str(), "VTK", "", out_meshset->get_ptr(), 1);
  }
  MoFEMFunctionReturn(0);
};

auto get_dofs_ents = [](auto dm, auto field_name) {
  auto prb_ptr = getProblemPtr(dm);
  std::vector<EntityHandle> ents_vec;

  MoFEM::Interface *m_field_ptr;
  CHKERR DMoFEMGetInterfacePtr(dm, &m_field_ptr);

  auto bit_number = m_field_ptr->get_field_bit_number(field_name);
  auto dofs = prb_ptr->numeredRowDofsPtr;
  auto lo_it = dofs->lower_bound(FieldEntity::getLoBitNumberUId(bit_number));
  auto hi_it = dofs->upper_bound(FieldEntity::getHiBitNumberUId(bit_number));
  ents_vec.reserve(std::distance(lo_it, hi_it));

  for(; lo_it!=hi_it;++lo_it) {
    ents_vec.push_back((*lo_it)->getEnt());
  }

  std::sort(ents_vec.begin(), ents_vec.end());
  auto it = std::unique(ents_vec.begin(), ents_vec.end());
  Range r;
  r.insert_list(ents_vec.begin(), it);
  return r;
};

#include <FreeSurfaceOps.hpp>
using namespace FreeSurfaceOps;

struct FreeSurface {

  FreeSurface(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

  MoFEMErrorCode makeRefProblem();

  MoFEM::Interface &mField;

private:
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();

  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;

  /// @brief
  /// @param level
  /// @param mask
  /// @return
  std::vector<Range> findEntitiesCrossedByPhaseInterface();

  /// @brief
  /// @param ents
  /// @param level
  /// @param mask
  /// @return
  Range findParentsToRefine(Range ents, BitRefLevel level, BitRefLevel mask);

  /// @brief
  /// @param vec_levels
  /// @return
  std::vector<Range> findChildren(std::vector<Range> vec_levels);

  /// @brief
  /// @return
  MoFEMErrorCode refineMesh();

  /// @brief
  /// @param fe_top
  /// @param op
  /// @param get_elem
  /// @param verbosity
  /// @param sev
  /// @return
  MoFEMErrorCode setParentDofs(
      boost::shared_ptr<FEMethod> fe_top, std::string field_name,
      ForcesAndSourcesCore::UserDataOperator::OpType op,
      boost::function<boost::shared_ptr<ForcesAndSourcesCore>()> get_elem,
      int verbosity, LogManager::SeverityLevel sev);

  /// @brief
  /// @return
  MoFEMErrorCode rebuildProblem();

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
  MOFEM_LOG("FS", Sev::inform)
      << "Read mesh for problem in " << EXECUTABLE_COORD_TYPE;
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

  std::vector<Range> vec_levels;
  for (auto l = 0; l != nb_levels; ++l) {
    vec_levels.push_back(Range());
    CHKERR bit_mng->getEntitiesByRefLevel(bit(l), BitRefLevel().set(),
                                          vec_levels.back());
  }
  auto vec_children = findChildren(vec_levels);

  Range ents;
  for (auto l = 0; l != nb_levels; ++l) {
    ents.merge(subtract(vec_levels[l], vec_children[l]));
  }
  CHKERR simple->setFieldOrder("U", order, &ents);
  CHKERR simple->setFieldOrder("P", order - 1, &ents);
  CHKERR simple->setFieldOrder("H", order, &ents);
  CHKERR simple->setFieldOrder("G", order, &ents);
  CHKERR simple->setFieldOrder("L", order, &ents);

  // Range level0;
  // CHKERR bit_mng->getEntitiesByRefLevel(bit(0), BitRefLevel().set(), level0);
  // CHKERR simple->setFieldOrder("U", order, &level0);
  // CHKERR simple->setFieldOrder("P", order - 1, &level0);
  // CHKERR simple->setFieldOrder("H", order, &level0);
  // CHKERR simple->setFieldOrder("G", order, &level0);
  // CHKERR simple->setFieldOrder("L", order, &level0);

  // Range levelN;
  // CHKERR bit_mng->getEntitiesByTypeAndRefLevel(BitRefLevel().set(),
  //                                              bit(0).flip(), MBVERTEX, levelN);
  // CHKERR simple->setFieldOrder("U", 1, &levelN);
  // CHKERR simple->setFieldOrder("P", 1, &levelN);
  // CHKERR simple->setFieldOrder("H", 1, &levelN);
  // CHKERR simple->setFieldOrder("G", 1, &levelN);
  // CHKERR simple->setFieldOrder("L", 1, &levelN);

  // Initialise bit ref levels
  auto set_problem_bit = [&]() {
    MoFEMFunctionBegin;
    auto bit0 = BitRefLevel().set(start_bit);
    BitRefLevel start_mask;
    for (auto s = 0; s != start_bit; ++s)
      start_mask[s] = true;

    auto bit_mng = mField.getInterface<BitRefManager>();

    Range level0;
    CHKERR bit_mng->getEntitiesByRefLevel(BitRefLevel().set(0),
                                          BitRefLevel().set(), level0);
    CHKERR bit_mng->setNthBitRefLevel(level0, current_bit, true);

    // Set bits to build adjacencies between parents and children. That is
    // used by simple interface.
    simple->getBitAdjEnt() = BitRefLevel().set();
    simple->getBitAdjParent() = BitRefLevel().set();
    simple->getBitRefLevel() = BitRefLevel().set(current_bit);
    simple->getBitRefLevelMask() = BitRefLevel().set();

#ifndef NDEBUG
    if constexpr (debug) {
      auto proc_str = boost::lexical_cast<std::string>(mField.get_comm_rank());
      CHKERR bit_mng->writeBitLevelByDim(
          BitRefLevel().set(0), BitRefLevel().set(), SPACE_DIM,
          (proc_str + "level_base.vtk").c_str(), "VTK", "");
    }
#endif

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

  using UDO = ForcesAndSourcesCore::UserDataOperator;

  auto add_parent_field = [&](auto fe, auto op, auto field) {
    return setParentDofs(
        fe, field, op,

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new DomianParentEle(mField));
          return fe_parent;
        },

        QUIET, Sev::noisy);
  };

  int exe_bit = 0;
  auto test_bit_child = [&](FEMethod *fe_ptr) {
    return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(exe_bit);
  };

  auto simple = mField.getInterface<Simple>();
  auto pip_mng = mField.getInterface<PipelineManager>();
  auto bc_mng = mField.getInterface<BcManager>();
  auto dm = simple->getDM();

  auto h_ptr = boost::make_shared<VectorDouble>();
  auto grad_h_ptr = boost::make_shared<MatrixDouble>();
  auto g_ptr = boost::make_shared<VectorDouble>();
  auto grad_g_ptr = boost::make_shared<MatrixDouble>();

  auto set_generic = [&](auto fe) {
    MoFEMFunctionBegin;
    auto &pip = fe->getOpPtrVector();

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1});

    CHKERR setParentDofs(
        fe, "", UDO::OPSPACE,

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

  auto post_proc = [&]() {
    MoFEMFunctionBegin;
    auto post_proc_fe = boost::make_shared<PostProcEleDomain>(mField);
    post_proc_fe->exeTestHook = test_bit_child;

    CHKERR set_generic(post_proc_fe);

    using OpPPMap = OpPostProcMapInMoab<2, 2>;

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

  auto solve_init = [&]() {
    MoFEMFunctionBegin;

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
      DM subdm;
      CHKERR DMCreate(mField.get_comm(), &subdm);
      CHKERR DMSetType(subdm, "DMMOFEM");
      CHKERR DMMoFEMCreateSubDM(subdm, dm, "SUB");
      CHKERR DMMoFEMAddElement(subdm, simple->getDomainFEName());
      CHKERR DMMoFEMSetSquareProblem(subdm, PETSC_TRUE);
      CHKERR DMMoFEMSetDestroyProblem(subdm, PETSC_TRUE);
      CHKERR DMMoFEMAddSubFieldRow(subdm, "H");
      CHKERR DMMoFEMAddSubFieldRow(subdm, "G");
      CHKERR DMMoFEMAddSubFieldCol(subdm, "H");
      CHKERR DMMoFEMAddSubFieldCol(subdm, "G");
      CHKERR DMSetUp(subdm);
      return SmartPetscObj<DM>(subdm);
    };

    auto subdm = create_subdm();

    pip_mng->getDomainRhsFE().reset();
    pip_mng->getDomainLhsFE().reset();
    CHKERR pip_mng->setDomainRhsIntegrationRule(integration_rule);
    CHKERR pip_mng->setDomainLhsIntegrationRule(integration_rule);
    pip_mng->getDomainLhsFE()->exeTestHook = test_bit_child;
    pip_mng->getDomainRhsFE()->exeTestHook = test_bit_child;

    CHKERR set_domain_rhs(pip_mng->getCastDomainRhsFE());
    CHKERR set_domain_lhs(pip_mng->getCastDomainLhsFE());

    auto D = smartCreateDMVector(subdm);
    auto snes = pip_mng->createSNES(subdm);
    auto snes_ctx_ptr = smartGetDMSnesCtx(subdm);

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

      auto section = mField.getInterface<ISManager>()->sectionCreate("SUB");
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

    CHKERR SNESSetFromOptions(snes);
    CHKERR SNESSolve(snes, PETSC_NULL, D);

    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(subdm, D, INSERT_VALUES, SCATTER_REVERSE);

    MoFEMFunctionReturn(0);
  };

  CHKERR solve_init();
  CHKERR post_proc();

  CHKERR refineMesh();
  CHKERR rebuildProblem();

  exe_bit = start_bit + nb_levels - 1;
  CHKERR solve_init();
  CHKERR post_proc();

  // Clear pipelines
  pip_mng->getOpDomainRhsPipeline().clear();
  pip_mng->getOpDomainLhsPipeline().clear();

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pip]
MoFEMErrorCode FreeSurface::assembleSystem() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();

  using UDO = ForcesAndSourcesCore::UserDataOperator;

  auto add_parent_field_domain = [&](auto fe, auto op, auto field) {
    return setParentDofs(
        fe, field, op,

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new DomianParentEle(mField));
          return fe_parent;
        },

        QUIET, Sev::noisy);
  };

  auto add_parent_field_bdy = [&](auto fe, auto op, auto field) {
    return setParentDofs(
        fe, field, op,

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new BoundaryParentEle(mField));
          return fe_parent;
        },

        QUIET, Sev::noisy);
  };

  auto test_bit_child = [](FEMethod *fe_ptr) {
    return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
        start_bit + nb_levels - 1);
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
        fe, "", UDO::OPSPACE,

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
    pip.push_back(
        new OpCalculateDivergenceVectorFieldValues<SPACE_DIM, coord_type>(
            "U", div_u_ptr));

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
    pip.push_back(new OpBaseTimesScalar(
        "P", div_u_ptr, [](const double r, const double, const double) {
          return cylindrical(r);
        }));
    pip.push_back(new OpBaseTimesScalar(
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
      pip.push_back(new OpMixScalarTimesDiv(
          "P", "U",
          [](const double r, const double, const double) {
            return cylindrical(r);
          },
          true, false));
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
        fe, "", UDO::OPSPACE,

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

    CHKERR add_parent_field_bdy(fe, UDO::OPROW, "U");
    pip.push_back(new OpNormalForceRhs("U", lambda_ptr));

    // push operators to the side element which is called from op_bdy_side
    auto op_bdy_side =
        new OpLoopSide<SideEle>(mField, simple->getDomainFEName(), SPACE_DIM);
    op_bdy_side->getSideFEPtr()->exeTestHook = test_bit_child;

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        op_bdy_side->getOpPtrVector(), {H1});

    CHKERR setParentDofs(
        op_bdy_side->getSideFEPtr(), "", UDO::OPSPACE,

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
    for (auto &b : get_blocks(get_block_name("WETTING_ANGLE"))) {
      Range force_edges;
      std::vector<double> attr_vec;
      CHKERR b->getMeshsetIdEntitiesByDimension(
          mField.get_moab(), SPACE_DIM - 1, force_edges, true);
      b->getAttributes(attr_vec);
      if (attr_vec.size() != 1)
        SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA, "Should be one attribute");
      // need to find the attributes and pass to operator
      CHKERR add_parent_field_bdy(fe, UDO::OPROW, "G");
      pip.push_back(new OpWettingAngleRhs(
          "G", grad_h_ptr, boost::make_shared<Range>(force_edges),
          attr_vec.front()));
    }

    MoFEMFunctionReturn(0);
  };

  auto set_boundary_lhs = [&](auto fe) {
    MoFEMFunctionBegin;
    auto &pip = fe->getOpPtrVector();

    CHKERR setParentDofs(
        fe, "", UDO::OPSPACE,

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new BoundaryParentEle(mField));
          return fe_parent;
        },

        QUIET, Sev::noisy);

    CHKERR add_parent_field_bdy(fe, UDO::OPROW, "L");
    CHKERR add_parent_field_bdy(fe, UDO::OPCOL, "U");
    pip.push_back(new OpNormalConstrainLhs("L", "U"));

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
    for (auto &b : get_blocks(get_block_name("WETTING_ANGLE"))) {
      Range force_edges;
      std::vector<double> attr_vec;
      CHKERR b->getMeshsetIdEntitiesByDimension(
          mField.get_moab(), SPACE_DIM - 1, force_edges, true);
      b->getAttributes(attr_vec);
      if (attr_vec.size() != 1)
        SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA, "Should be one attribute");
      MOFEM_LOG("FS", Sev::inform)
          << "wetting angle edges size " << force_edges.size();

      CHKERR add_parent_field_bdy(fe, UDO::OPROW, "G");
      pip.push_back(new OpWettingAngleLhs(
          "G", grad_h_ptr, col_ind_ptr, col_diff_base_ptr,
          boost::make_shared<Range>(force_edges), attr_vec.front()));
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
 * This functions is called by TS solver at the end of each step. It is used
 * to output results to the hard drive.
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
    constexpr int save_every_nth_step = 1;
    if (ts_step % save_every_nth_step == 0) {
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

struct TSPrePostProc {
  static MoFEMErrorCode tsPreProc(TS ts);
  static MoFEMErrorCode tsPostProc(TS ts);
  static FreeSurface *fsRawPtr;
};

FreeSurface *TSPrePostProc::fsRawPtr = nullptr;

//! [Solve]
MoFEMErrorCode FreeSurface::solveSystem() {
  MoFEMFunctionBegin;

  using UDO = ForcesAndSourcesCore::UserDataOperator;

  auto *simple = mField.getInterface<Simple>();
  auto *pip_mng = mField.getInterface<PipelineManager>();
  auto dm = simple->getDM();
  auto snes_ctx_ptr = smartGetDMSnesCtx(dm);

  auto get_fe_post_proc = [&](auto post_proc_mesh) {

    auto add_parent_field_domain = [&](auto fe, auto op, auto field) {
      return setParentDofs(
          fe, field, op,

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
          start_bit + nb_levels - 1);
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
        post_proc_fe, "", UDO::OPSPACE,

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
          fe, field, op,

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
          start_bit + nb_levels - 1);
    };

    CHKERR setParentDofs(
        post_proc_fe, "", UDO::OPSPACE,

        [&]() {
          boost::shared_ptr<ForcesAndSourcesCore> fe_parent(
              new BoundaryParentEle(mField));
          return fe_parent;
        },

        QUIET, Sev::noisy);

    auto u_ptr = boost::make_shared<MatrixDouble>();
    auto p_ptr = boost::make_shared<VectorDouble>();
    auto lambda_ptr = boost::make_shared<VectorDouble>();

    CHKERR add_parent_field_bdy(post_proc_fe, UDO::OPROW, "U");
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<U_FIELD_DIM>("U", u_ptr));

    CHKERR add_parent_field_bdy(post_proc_fe, UDO::OPROW, "L");
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("L", lambda_ptr));

    CHKERR add_parent_field_bdy(post_proc_fe, UDO::OPROW, "P");
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("P", p_ptr));

    using OpPPMap = OpPostProcMapInMoab<2, 2>;

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(post_proc_fe->getPostProcMesh(),
                    post_proc_fe->getMapGaussPts(),

                    OpPPMap::DataMapVec{{"L", lambda_ptr}, {"P", p_ptr}},

                    OpPPMap::DataMapMat{{"U", u_ptr}},

                    OpPPMap::DataMapMat(),

                    OpPPMap::DataMapMat()

                        )

    );

    return post_proc_fe;
  };

  auto get_lift_fe = [&]() {
    auto add_parent_field_bdy = [&](auto fe, auto op, auto field) {
      return setParentDofs(
          fe, field, op,

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
          start_bit + nb_levels - 1);
    };

    auto lift_ptr = boost::make_shared<VectorDouble>();
    auto p_ptr = boost::make_shared<VectorDouble>();
    auto ents_ptr = boost::make_shared<Range>();

    CHKERR setParentDofs(
        fe, "", UDO::OPSPACE,

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
        new OpCalculateScalarFieldValues("P", p_ptr));
    fe->getOpPtrVector().push_back(
        new OpCalculateLift("P", p_ptr, lift_ptr, ents_ptr));

    return std::make_pair(fe, lift_ptr);
  };

  auto set_ts = [&](auto solver) {
    MoFEMFunctionBegin;
    SNES snes;
    CHKERR TSGetSNES(solver, &snes);
    KSP ksp;
    CHKERR SNESGetKSP(snes, &ksp);
    MoFEMFunctionReturn(0);
  };

  auto set_section_monitor = [&](auto solver) {
    MoFEMFunctionBegin;
    SNES snes;
    CHKERR TSGetSNES(solver, &snes);
    CHKERR SNESMonitorSet(snes,
                          (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal,
                                             void *))MoFEMSNESMonitorFields,
                          (void *)(snes_ctx_ptr.get()), nullptr);

    MoFEMFunctionReturn(0);
  };

  auto ts = pip_mng->createTSIM();
  // CHKERR TSSetType(ts, TSALPHA);

  auto set_post_proc_monitor = [&](auto dm) {
    MoFEMFunctionBegin;
    boost::shared_ptr<FEMethod> null_fe;
    auto post_proc_mesh = boost::make_shared<moab::Core>();
    auto monitor_ptr = boost::make_shared<Monitor>(
        dm, post_proc_mesh, get_fe_post_proc(post_proc_mesh),
        get_bdy_post_proc_fe(post_proc_mesh), get_lift_fe());
    CHKERR DMMoFEMTSSetMonitor(dm, ts, simple->getDomainFEName(), null_fe,
                               null_fe, monitor_ptr);
    MoFEMFunctionReturn(0);
  };
  CHKERR set_post_proc_monitor(dm);

  // Add monitor to time solver
  double ftime = 1;
  // CHKERR TSSetMaxTime(ts, ftime);
  CHKERR TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);

  auto T = smartCreateDMVector(simple->getDM());
  CHKERR DMoFEMMeshToLocalVector(simple->getDM(), T, INSERT_VALUES,
                                 SCATTER_FORWARD);
  CHKERR TSSetSolution(ts, T);
  CHKERR TSSetFromOptions(ts);
  CHKERR set_ts(ts);
  CHKERR set_section_monitor(ts);
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

  TSPrePostProc::fsRawPtr = this;
  CHKERR TSSetPreStep(ts, TSPrePostProc::tsPreProc);
  CHKERR TSSetPostStep(ts, TSPrePostProc::tsPostProc);

  CHKERR TSSolve(ts, NULL);
  CHKERR TSGetTime(ts, &ftime);

  MoFEMFunctionReturn(0);
}

int main(int argc, char *argv[]) {

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
}

std::vector<Range> FreeSurface::findEntitiesCrossedByPhaseInterface() {

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
    vec_levels[l] = get_level(ele_plus[l], ele_minus[l], 2 * 3, bit(l),
                              BitRefLevel().set());
  }

  if constexpr (debug) {
    for (auto l = 0; l != nb_levels; ++l) {
      std::string name = (boost::format("out_r%d.vtk") % l).str();
      CHK_THROW_MESSAGE(save_range(mField.get_moab(), name, vec_levels[l]),
                        "save mesh");
    }
  }

  return vec_levels;
}

std::vector<Range> FreeSurface::findChildren(std::vector<Range> vec_levels) {

  auto bit_mng = mField.getInterface<BitRefManager>();

  std::vector<Range> vec_children(nb_levels);

  // remove same dimension children
  for (auto l = nb_levels - 1; l > 0; --l) {

    for (auto d = 1; d <= SPACE_DIM; ++d) {
      // Range r;
      // bit_mng->getEntitiesByDimAndRefLevel(bit(l), BitRefLevel().set(), d, r);
      // vec_children[l].merge(r);
      Range c;
      CHK_THROW_MESSAGE(bit_mng->updateRangeByChildren(
                            vec_levels[l - 1].subset_by_dimension(d), c),
                        "get children");
      vec_children[l].merge(c.subset_by_dimension(d));
      // vec_children[l] = subtract(r, c.subset_by_dimension(d));
    }
  }

  if constexpr (debug) {
    for (auto l = 0; l != nb_levels; ++l) {
      std::string name = (boost::format("out_children%d.vtk") % l).str();
      CHK_THROW_MESSAGE(save_range(mField.get_moab(), name, vec_children[l]),
                        "save mesh");
    }
  }

  return vec_children;
}

MoFEMErrorCode FreeSurface::refineMesh() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  auto bit_mng = mField.getInterface<BitRefManager>();
  auto prb_mng = mField.getInterface<ProblemsManager>();

  BitRefLevel start_mask;
  for (auto s = 0; s != start_bit; ++s)
    start_mask[s] = true;

  // reset bit ref levels
  CHKERR bit_mng->lambdaBitRefLevel(
      [&](EntityHandle ent, BitRefLevel &bit) { bit &= start_mask; });

  auto vec_levels = findEntitiesCrossedByPhaseInterface();

  Range level0;
  CHKERR bit_mng->getEntitiesByRefLevel(bit(0), BitRefLevel().set(), level0);
  CHKERR bit_mng->setNthBitRefLevel(level0, current_bit, true);
  CHKERR bit_mng->setNthBitRefLevel(level0, start_bit, true);

  auto get_adj = [&](auto ents) {
    Range conn;
    CHK_MOAB_THROW(mField.get_moab().get_connectivity(ents, conn, true),
                   "get conn");
    for (auto d = 1; d != SPACE_DIM; ++d) {
      CHK_MOAB_THROW(mField.get_moab().get_adjacencies(
                         ents.subset_by_dimension(SPACE_DIM), d, false, ents,
                         moab::Interface::UNION),
                     "get adj");
    }
    ents.merge(conn);        
    return ents;
  };

  for (auto l = 1; l != nb_levels; ++l) {
    Range level_prev;
    CHKERR bit_mng->getEntitiesByDimAndRefLevel(
        bit(start_bit + l - 1), BitRefLevel().set(), SPACE_DIM, level_prev);
    Range parents;
    CHKERR bit_mng->updateRangeByParent(vec_levels[l], parents);
    level_prev = subtract(level_prev, parents);
    level_prev.merge(vec_levels[l]);
    CHKERR bit_mng->setNthBitRefLevel(level_prev, start_bit + l, true);
  }

  for (auto l = 1; l != nb_levels; ++l) {
    Range level;
    CHKERR bit_mng->getEntitiesByRefLevel(bit(start_bit + l),
                                          BitRefLevel().set(), level);
    level = get_adj(level);
    CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(level);
    CHKERR bit_mng->setNthBitRefLevel(level, start_bit + l, true);
    CHKERR bit_mng->setNthBitRefLevel(level, current_bit, true);
  }

  if constexpr (debug) {

    for (auto l = 0; l != nb_levels; ++l) {
      std::string name = (boost::format("out_level%d.vtk") % l).str();
      CHKERR bit_mng->writeBitLevel(BitRefLevel().set(start_bit + l),
                                    BitRefLevel().set(), name.c_str(), "VTK",
                                    "");
    }

    CHKERR bit_mng->writeBitLevel(BitRefLevel().set(current_bit),
                                  BitRefLevel().set(), "current_bit.vtk", "VTK",
                                  "");
  }

  MoFEMFunctionReturn(0);
};

MoFEMErrorCode FreeSurface::setParentDofs(
    boost::shared_ptr<FEMethod> fe_top, std::string field_name,
    ForcesAndSourcesCore::UserDataOperator::OpType op,
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

                        BitRefLevel().set(), BitRefLevel().set(),

                        verbosity, sev));

              } else {

                // Filed data
                parent_fe_pt->getOpPtrVector().push_back(

                    new OpAddParentEntData(

                        field_name, op, fe_ptr_current,

                        BitRefLevel().set(), BitRefLevel().set(),

                        BitRefLevel().set(), BitRefLevel().set(),

                        verbosity, sev));
              }
            }
          };

  add_parent_level(boost::dynamic_pointer_cast<ForcesAndSourcesCore>(fe_top),
                   nb_levels);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode FreeSurface::rebuildProblem() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  auto bit_mng = mField.getInterface<BitRefManager>();
  auto prb_mng = mField.getInterface<ProblemsManager>();
  auto pip_mng = mField.getInterface<PipelineManager>();

  simple->getBitRefLevel() = BitRefLevel().set(current_bit);
  simple->getBitRefLevelMask() = BitRefLevel().set();
  simple->reSetUp(true);

  BitRefLevel mask;
  moab::Skinner skinner(&mField.get_moab());
  ParallelComm *pcomm =
      ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);

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
      CHKERR pcomm->filter_pstatus(skin_level_mesh,
                                   PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                   PSTATUS_NOT, -1, nullptr);
      auto skin_level = get_bit_skin(bit(start_bit + l), BitRefLevel().set());
      skin_level = subtract(skin_level, skin_level_mesh);
      Range skin_level_verts;
      CHKERR mField.get_moab().get_connectivity(skin_level, skin_level_verts,
                                                true);
      skin_level.merge(skin_level_verts);

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

    Range from_other_procs;
    for (auto &m : map_procs) {
      if (m.first != mField.get_comm_rank()) {
        from_other_procs.merge(m.second);
      }
    }

    auto common = intersect(skin, from_other_procs);
    skin.merge(from_other_procs);

    if (!common.empty()) {
      // skin is internal exist on other procs
      skin = subtract(skin, common);
    }

    return  skin;
  };

  auto skin = resolve_shared(get_level_skin());

  if constexpr (debug) {
    CHKERR save_range(
        mField.get_moab(),
        (boost::format("skin_%d.vtk") % mField.get_comm_rank()).str(), skin);
  }

  for (auto f : {"U", "P", "H", "G", "L"}) {
    CHKERR prb_mng->removeDofsOnEntities(simple->getProblemName(), f, skin);
  }

  // std::vector<Range> vec_levels_all;
  // for (auto l = 0; l != nb_levels; ++l) {
  //   vec_levels_all.push_back(Range());
  //   CHKERR bit_mng->getEntitiesByRefLevel(bit(l), BitRefLevel().set(),
  //                                         vec_levels_all.back());
  // }
  // auto vec_children = findChildren(vec_levels_all);
  // Range children_to_remove;
  // for(auto &v : vec_children) {
  //   children_to_remove.merge(v);
  // }
  // for (auto f : {"U", "P", "H", "G" , "L"}) {
  //   CHKERR prb_mng->removeDofsOnEntities(simple->getProblemName(), f,
  //                                        children_to_remove);
  // }

  // Range last_level;
  // CHKERR bit_mng->getEntitiesByRefLevel(bit(start_bit + nb_levels - 1),
  //                                       BitRefLevel().set(), last_level);
  // Range prev_level;
  // CHKERR bit_mng->getEntitiesByRefLevel(bit(start_bit + nb_levels - 2),
  //                                       BitRefLevel().set(), prev_level);
  // CHKERR prb_mng->removeDofsOnEntities(simple->getProblemName(), "P",
  //                                      subtract(last_level, prev_level));

  auto r_p = get_dofs_ents(simple->getDM(), "P");
  auto r_u = get_dofs_ents(simple->getDM(), "U");

  if (mField.get_comm_rank() == 0) {
    CHKERR save_range(mField.get_moab(), "ents_p.vtk", r_p);
    CHKERR save_range(mField.get_moab(), "ents_u.vtk", r_u);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode TSPrePostProc::tsPreProc(TS ts) {
  auto &m_field = fsRawPtr->mField;
  auto simple = m_field.getInterface<Simple>();
  auto bit_mng = m_field.getInterface<BitRefManager>();
  auto bc_mng = m_field.getInterface<BcManager>();
  auto field_blas = m_field.getInterface<FieldBlas>();
  auto opt = m_field.getInterface<OperatorsTester>();
  auto pip_mng = m_field.getInterface<PipelineManager>();
  MoFEMFunctionBegin;

  MOFEM_LOG("FS", Sev::inform) << "Run step pre proc";

  auto get_norm = [&](auto x) {
    double nrm;
    CHKERR VecNorm(x, NORM_2, &nrm);
    return nrm;
  };

  auto reset_solution = [&](auto ts) {
    MoFEMFunctionBegin;
    MOFEM_LOG("FS", Sev::inform) << "Reset fields";
    DM dm;
    CHKERR TSGetDM(ts, &dm);
    auto x = smartCreateDMVector(dm);
    CHKERR DMoFEMMeshToLocalVector(dm, x, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateBegin(x, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(x, INSERT_VALUES, SCATTER_FORWARD);
    for (auto f : {"U", "P", "H", "G", "L"}) {
      CHKERR field_blas->setField(0, f);
    }
    CHKERR DMoFEMMeshToLocalVector(dm, x, INSERT_VALUES, SCATTER_REVERSE);
    MoFEMFunctionReturn(0);
  };

  auto set_solution = [&](auto ts) {
    MoFEMFunctionBegin;
    DM dm;
    CHKERR TSGetDM(ts, &dm);
    auto prb_ptr = getProblemPtr(dm);
    auto x = smartCreateDMVector(dm);
    CHKERR DMoFEMMeshToLocalVector(dm, x, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateBegin(x, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(x, INSERT_VALUES, SCATTER_FORWARD);
    MOFEM_LOG("FS", Sev::inform) << "Set solution, vector norm " << get_norm(x);
    CHKERR TSSetSolution(ts, x);
    MoFEMFunctionReturn(0);
  };

  auto refine_problem = [&](auto ts) {
    MoFEMFunctionBegin;
    MOFEM_LOG("FS", Sev::inform) << "Refine problem";

    CHKERR fsRawPtr->refineMesh();
    CHKERR fsRawPtr->rebuildProblem();

    // Remove DOFs where boundary conditions are set

    CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                             "SYMMETRY", "U", 0, 0);
    CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                             "SYMMETRY", "L", 0, 0);
    CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX",
                                             "U", 0, SPACE_DIM);
    CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX",
                                             "L", 0, 0);
    CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "ZERO",
                                             "L", 0, 0);

    MoFEMFunctionReturn(0);
  };

  auto ts_reset_theta = [&](auto ts) {
    MoFEMFunctionBegin;
    MOFEM_LOG("FS", Sev::inform) << "Reset time solver";

    DM dm;
    CHKERR TSGetDM(ts, &dm);
    CHKERR TSReset(ts);
    CHKERR TSSetUp(ts);
    CHKERR set_solution(ts);
    auto B = smartCreateDMMatrix(dm);
    CHKERR TSSetIJacobian(ts, B, B, TsSetIJacobian, nullptr);
    MoFEMFunctionReturn(0);
  };

  CHKERR refine_problem(ts);
  CHKERR reset_solution(ts);
  CHKERR ts_reset_theta(ts);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode TSPrePostProc::tsPostProc(TS ts) { return 0; }