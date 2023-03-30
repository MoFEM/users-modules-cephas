/**
 * \file free_surface.cpp
 * \example free_surface.cpp
 *
 * Using PipelineManager interface calculate the divergence of base functions,
 * and integral of flux on the boundary. Since the h-div space is used, volume
 * integral and boundary integral should give the same result.
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

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = PipelineManager::FaceEle;
  using DomianParentEle = FaceElementForcesAndSourcesCoreOnChildParent;
  using BoundaryEle = PipelineManager::EdgeEle;
  using BoundaryParentEle = EdgeElementForcesAndSourcesCoreOnChildParent;
  //using PostProcEle = PostProcFaceOnRefinedMesh;
  using SideEle = FaceElementForcesAndSourcesCoreOnSide;
  using SideOp = SideEle::UserDataOperator;
  using PostProcEdgeEle = PostProcEdgeOnRefinedMesh;
};

using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomianParentEle = ElementsAndOps<SPACE_DIM>::DomianParentEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using SideEle = ElementsAndOps<SPACE_DIM>::SideEle;
using EntData = DataForcesAndSourcesCore::EntData;
using BoundaryParentEle = ElementsAndOps<SPACE_DIM>::BoundaryParentEle;

using EntData = EntitiesFieldData::EntData;

using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;
using PostProcEdgeEle = PostProcBrokenMeshInMoab<BoundaryEle>;

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
constexpr double h = 0.025; // mesh size
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

auto cappilary_tube = [](double x, double y, double z) {
  constexpr double water_height = 0.;
  return tanh((water_height - y) / (eta * std::sqrt(2)));
  ;
};

auto init_h = [](double r, double y, double theta) {
  return cappilary_tube(r, y, theta);
  // return kernel_eye(r, y, theta);
};

auto wetting_angle = [](double water_level) { return water_level; };

auto bit = [](auto b) { return BitRefLevel().set(b); };
auto marker = [](auto b) { return BitRefLevel().set(BITREFLEVEL_SIZE - b); };

auto save_range = [](moab::Interface &moab, const std::string name,
                     const Range r) {
  MoFEMFunctionBegin;
  EntityHandle out_meshset;
  CHKERR moab.create_meshset(MESHSET_SET, out_meshset);
  CHKERR moab.add_entities(out_meshset, r);
  CHKERR moab.write_file(name.c_str(), "VTK", "", &out_meshset, 1);
  CHKERR moab.delete_entities(&out_meshset, 1);
  MoFEMFunctionReturn(0);
};

auto get_dofs_ents = [](auto dm) {
  auto prb_ptr = getProblemPtr(dm);
  std::vector<EntityHandle> ents_vec;
  ents_vec.reserve(prb_ptr->numeredRowDofsPtr->size());
  for (auto dof : *prb_ptr->numeredRowDofsPtr) {
    ents_vec.push_back(dof->getEnt());
  }
  std::sort(ents_vec.begin(), ents_vec.end());
  auto it = std::unique(ents_vec.begin(), ents_vec.end());
  Range r;
  r.insert_list(ents_vec.begin(), it);
  return r;
};

template <typename PARENT> struct ExtractParentType { using Prent = PARENT; };

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

  boost::shared_ptr<FEMethod> domianLhsFEPtr;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
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

  CHKERR simple->getOptions();
  CHKERR simple->loadFile();

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode FreeSurface::setupProblem() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();

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

  CHKERR simple->setUp();

  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Boundary condition]
MoFEMErrorCode FreeSurface::boundaryCondition() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  auto pipeline_mng = mField.getInterface<PipelineManager>();
  auto bc_mng = mField.getInterface<BcManager>();
  auto dm = simple->getDM();

  auto h_ptr = boost::make_shared<VectorDouble>();
  auto grad_h_ptr = boost::make_shared<MatrixDouble>();
  auto g_ptr = boost::make_shared<VectorDouble>();
  auto grad_g_ptr = boost::make_shared<MatrixDouble>();

  auto set_generic = [&](auto &pipeline, auto &fe) {
    MoFEMFunctionBegin;
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pipeline, {H1});
    pipeline.push_back(new OpCalculateScalarFieldValues("H", h_ptr));
    pipeline.push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_h_ptr));

    pipeline.push_back(new OpCalculateScalarFieldValues("G", g_ptr));
    pipeline.push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("G", grad_g_ptr));
    MoFEMFunctionReturn(0);
  };

  auto post_proc = [&]() {
    MoFEMFunctionBegin;
    auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
    CHKERR set_generic(post_proc_fe->getOpPtrVector(), post_proc_fe);

    using OpPPMap = OpPostProcMapInMoab<2, 2>;

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(
            post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

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

    auto set_domain_rhs = [&](auto &pipeline, auto &fe) {
      set_generic(pipeline, fe);
      pipeline.push_back(new OpRhsH<true>("H", nullptr, nullptr, h_ptr,
                                          grad_h_ptr, grad_g_ptr));
      pipeline.push_back(new OpRhsG<true>("G", h_ptr, grad_h_ptr, g_ptr));
    };

    auto set_domain_lhs = [&](auto &pipeline, auto &fe) {
      set_generic(pipeline, fe);
      pipeline.push_back(new OpLhsH_dH<true>("H", nullptr, h_ptr, grad_g_ptr));
      pipeline.push_back(new OpLhsH_dG<true>("H", "G", h_ptr));
      pipeline.push_back(new OpLhsG_dG("G"));
      pipeline.push_back(new OpLhsG_dH<true>("G", "H", h_ptr));
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

    auto prb_ents = get_dofs_ents(subdm);

    pipeline_mng->getDomainRhsFE().reset();
    pipeline_mng->getDomainLhsFE().reset();
    CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
    CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);

    set_domain_rhs(pipeline_mng->getOpDomainRhsPipeline(),
                   pipeline_mng->getDomainRhsFE());
    set_domain_lhs(pipeline_mng->getOpDomainLhsPipeline(),
                   pipeline_mng->getDomainLhsFE());

    auto D = smartCreateDMVector(subdm);
    auto snes = pipeline_mng->createSNES(subdm);
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

  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "SYMETRY",
                                           "U", 0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "SYMETRY",
                                           "L", 0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX", "U",
                                           0, SPACE_DIM);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX", "L",
                                           0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "ZERO",
                                           "L", 0, 0);

  // Clear pipelines
  pipeline_mng->getOpDomainRhsPipeline().clear();
  pipeline_mng->getOpDomainLhsPipeline().clear();

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode FreeSurface::assembleSystem() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();

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
  auto set_domain_general = [&](auto &pipeline, auto &fe) {
    MoFEMFunctionBegin;
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pipeline, {H1});

    pipeline.push_back(
        new OpCalculateVectorFieldValuesDot<U_FIELD_DIM>("U", dot_u_ptr));
    pipeline.push_back(
        new OpCalculateVectorFieldValues<U_FIELD_DIM>("U", u_ptr));
    pipeline.push_back(
        new OpCalculateVectorFieldGradient<U_FIELD_DIM, SPACE_DIM>("U",
                                                                   grad_u_ptr));
    pipeline.push_back(
        new OpCalculateDivergenceVectorFieldValues<SPACE_DIM, coord_type>(
            "U", div_u_ptr));

    pipeline.push_back(new OpCalculateScalarFieldValuesDot("H", dot_h_ptr));
    pipeline.push_back(new OpCalculateScalarFieldValues("H", h_ptr));
    pipeline.push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_h_ptr));

    pipeline.push_back(new OpCalculateScalarFieldValues("G", g_ptr));
    pipeline.push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("G", grad_g_ptr));

    pipeline.push_back(new OpCalculateScalarFieldValues("P", p_ptr));
    MoFEMFunctionReturn(0);
  };

  auto set_domain_rhs = [&](auto &pipeline, auto &fe) {
    MoFEMFunctionBegin;
    CHKERR set_domain_general(pipeline, fe);
    pipeline.push_back(new OpRhsU("U", dot_u_ptr, u_ptr, grad_u_ptr, h_ptr,
                                  grad_h_ptr, g_ptr, p_ptr));
    pipeline.push_back(new OpRhsH<false>("H", u_ptr, dot_h_ptr, h_ptr,
                                         grad_h_ptr, grad_g_ptr));
    pipeline.push_back(new OpRhsG<false>("G", h_ptr, grad_h_ptr, g_ptr));
    pipeline.push_back(new OpBaseTimesScalar(
        "P", div_u_ptr, [](const double r, const double, const double) {
          return cylindrical(r);
        }));
    pipeline.push_back(new OpBaseTimesScalar(
        "P", p_ptr, [](const double r, const double, const double) {
          return eps * cylindrical(r);
        }));
    MoFEMFunctionReturn(0);
  };

  auto set_domain_lhs = [&](auto &pipeline, auto &fe) {
    MoFEMFunctionBegin;
    CHKERR set_domain_general(pipeline, fe);
    pipeline.push_back(new OpLhsU_dU("U", u_ptr, grad_u_ptr, h_ptr));
    pipeline.push_back(
        new OpLhsU_dH("U", "H", dot_u_ptr, u_ptr, grad_u_ptr, h_ptr, g_ptr));
    pipeline.push_back(new OpLhsU_dG("U", "G", grad_h_ptr));

    pipeline.push_back(new OpLhsH_dU("H", "U", grad_h_ptr));
    pipeline.push_back(new OpLhsH_dH<false>("H", u_ptr, h_ptr, grad_g_ptr));
    pipeline.push_back(new OpLhsH_dG<false>("H", "G", h_ptr));

    pipeline.push_back(new OpLhsG_dH<false>("G", "H", h_ptr));
    pipeline.push_back(new OpLhsG_dG("G"));

    pipeline.push_back(new OpMixScalarTimesDiv(
        "P", "U",
        [](const double r, const double, const double) {
          return cylindrical(r);
        },
        true, false));
    pipeline.push_back(
        new OpDomainMassP("P", "P", [](double r, double, double) {
          return eps * cylindrical(r);
        }));
    MoFEMFunctionReturn(0);
  };

  auto set_boundary_rhs = [&](auto &pipeline, auto &fe) {
    MoFEMFunctionBegin;
    pipeline.push_back(
        new OpCalculateVectorFieldValues<U_FIELD_DIM>("U", u_ptr));
    pipeline.push_back(new OpCalculateScalarFieldValues("L", lambda_ptr));
    pipeline.push_back(new OpNormalConstrainRhs("L", u_ptr));
    pipeline.push_back(new OpNormalForceRhs("U", lambda_ptr));

    // push operators to the side element which is called from op_bdy_side
    auto op_bdy_side =
        new OpLoopSide<SideEle>(mField, simple->getDomainFEName(), SPACE_DIM);
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        op_bdy_side->getOpPtrVector(), {H1});
    op_bdy_side->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_h_ptr));
    // push bdy side op
    pipeline.push_back(op_bdy_side);

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      std::string block_name = "WETTING_ANGLE";
      if (it->getName().compare(0, block_name.length(), block_name) == 0) {
        Range force_edges;
        std::vector<double> attr_vec;
        CHKERR it->getMeshsetIdEntitiesByDimension(
            mField.get_moab(), SPACE_DIM - 1, force_edges, true);
        it->getAttributes(attr_vec);
        if (attr_vec.size() != 1)
          SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
                  "Should be one attribute");
        // need to find the attributes and pass to operator
        pipeline.push_back(new OpWettingAngleRhs(
            "G", grad_h_ptr, boost::make_shared<Range>(force_edges),
            attr_vec.front()));
      }
    }
    MoFEMFunctionReturn(0);
  };

  auto set_boundary_lhs = [&](auto &pipeline, auto &fe) {
    MoFEMFunctionBegin;
    pipeline.push_back(new OpNormalConstrainLhs("L", "U"));

    auto col_ind_ptr = boost::make_shared<std::vector<VectorInt>>();
    auto col_diff_base_ptr = boost::make_shared<std::vector<MatrixDouble>>();

    // push operators to the side element which is called from op_bdy_side
    auto op_bdy_side =
        new OpLoopSide<SideEle>(mField, simple->getDomainFEName(), SPACE_DIM);
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        op_bdy_side->getOpPtrVector(), {H1});
    op_bdy_side->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_h_ptr));
    op_bdy_side->getOpPtrVector().push_back(
        new OpLoopSideGetDataForSideEle("H", col_ind_ptr, col_diff_base_ptr));

    // push bdy side op
    pipeline.push_back(op_bdy_side);

    // push operators for lhs wetting angle
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, 13, "WETTING_ANGLE") == 0) {
        Range force_edges;
        std::vector<double> attr_vec;
        CHKERR it->getMeshsetIdEntitiesByDimension(
            mField.get_moab(), SPACE_DIM - 1, force_edges, true);
        it->getAttributes(attr_vec);
        if (attr_vec.size() != 1)
          SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
                  "Should be one attribute");
        MOFEM_LOG("FS", Sev::inform)
            << "wetting angle edges size " << force_edges.size();
        pipeline.push_back(new OpWettingAngleLhs(
            "G", grad_h_ptr, col_ind_ptr, col_diff_base_ptr,
            boost::make_shared<Range>(force_edges), attr_vec.front()));
      }
    }
    MoFEMFunctionReturn(0);
  };

  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);

  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule);

  set_domain_rhs(pipeline_mng->getOpDomainRhsPipeline(),
                 pipeline_mng->getDomainRhsFE());
  set_domain_lhs(pipeline_mng->getOpDomainLhsPipeline(),
                 pipeline_mng->getDomainLhsFE());

  CHKERR set_boundary_rhs(pipeline_mng->getOpBoundaryRhsPipeline(),
                          pipeline_mng->getBoundaryRhsFE());
  CHKERR set_boundary_lhs(pipeline_mng->getOpBoundaryLhsPipeline(),
                          pipeline_mng->getBoundaryLhsFE());

  domianLhsFEPtr = pipeline_mng->getDomainLhsFE();

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
  Monitor(
      SmartPetscObj<DM> dm, boost::shared_ptr<PostProcEle> post_proc,
      boost::shared_ptr<PostProcEdgeEle> post_proc_edge,
      std::pair<boost::shared_ptr<BoundaryEle>, boost::shared_ptr<VectorDouble>>
          p)
      : dM(dm), postProc(post_proc), postProcEdge(post_proc_edge),
        liftFE(p.first), liftVec(p.second) {}
  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;
    constexpr int save_every_nth_step = 1;
    if (ts_step % save_every_nth_step == 0) {
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc,
                                      this->getCacheWeakPtr());
      CHKERR postProc->writeFile(
          "out_step_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");

      // CHKERR DMoFEMLoopFiniteElements(dM, "bFE", postProcEdge,
      //                                 this->getCacheWeakPtr());
      // CHKERR postProcEdge->writeFile(
      //     "out_step_bdy_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
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
  boost::shared_ptr<PostProcEle> postProc;
  boost::shared_ptr<PostProcEdgeEle> postProcEdge;
  boost::shared_ptr<BoundaryEle> liftFE;
  boost::shared_ptr<VectorDouble> liftVec;
};

//! [Solve]
MoFEMErrorCode FreeSurface::solveSystem() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  auto dm = simple->getDM();
  auto snes_ctx_ptr = smartGetDMSnesCtx(dm);

  auto get_fe_post_proc = [&]() {
    auto post_proc_fe = boost::make_shared<PostProcEle>(mField);

    auto u_ptr = boost::make_shared<MatrixDouble>();
    auto grad_u_ptr = boost::make_shared<MatrixDouble>();
    auto h_ptr = boost::make_shared<VectorDouble>();
    auto grad_h_ptr = boost::make_shared<MatrixDouble>();
    auto p_ptr = boost::make_shared<VectorDouble>();
    auto g_ptr = boost::make_shared<VectorDouble>();
    auto grad_g_ptr = boost::make_shared<MatrixDouble>();

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        post_proc_fe->getOpPtrVector(), {H1});

    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<U_FIELD_DIM>("U", u_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<U_FIELD_DIM, SPACE_DIM>("U",
                                                                   grad_u_ptr));

    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("H", h_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_h_ptr));

    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("P", p_ptr));
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

  auto get_bdy_post_proc_fe = [&]() {
    auto post_proc_fe = boost::make_shared<PostProcEdgeEle>(mField);

    auto u_ptr = boost::make_shared<MatrixDouble>();
    auto p_ptr = boost::make_shared<VectorDouble>();
    auto lambda_ptr = boost::make_shared<VectorDouble>();

    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<U_FIELD_DIM>("U", u_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("L", lambda_ptr));
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
    auto fe = boost::make_shared<BoundaryEle>(mField);
    auto lift_ptr = boost::make_shared<VectorDouble>();
    auto p_ptr = boost::make_shared<VectorDouble>();
    auto ents_ptr = boost::make_shared<Range>();

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

  auto ts = pipeline_mng->createTSIM();
  CHKERR TSSetType(ts, TSALPHA);

  auto set_post_proc_monitor = [&](auto dm) {
    MoFEMFunctionBegin;
    boost::shared_ptr<FEMethod> null_fe;
    auto monitor_ptr = boost::make_shared<Monitor>(
        dm, get_fe_post_proc(), get_bdy_post_proc_fe(), get_lift_fe());
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
    MoFEM::Interface &m_field = core; ///< finite element database insterface
    //! [Create MoFEM]

    //! [FreeSurface]
    FreeSurface ex(m_field);
    CHKERR ex.runProblem();
    //! [FreeSurface]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
