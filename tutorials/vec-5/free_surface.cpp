/**
 * \file free_surface.cpp
 * \example free_surface.cpp
 *
 * Using PipelineManager interface calculate the divergence of base functions,
 * and integral of flux on the boundary. Since the h-div space is used, volume
 * integral and boundary integral should give the same result.
 */

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

#include <MoFEM.hpp>

using namespace MoFEM;

static char help[] = "...\n\n";

#include <BasicFiniteElements.hpp>

constexpr int BASE_DIM = 1;
constexpr int SPACE_DIM = 2;
constexpr int U_FIELD_DIM = SPACE_DIM;
constexpr CoordinateTypes coord_type =
    CARTESIAN; ///< select coordinate system <CARTESIAN, CYLINDRICAL>;

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = PipelineManager::FaceEle;
  using DomianParentEle = FaceElementForcesAndSourcesCoreOnChildParent;
  using DomainEleOp = DomainEle::UserDataOperator;
  using BoundaryEle = PipelineManager::EdgeEle;
  using BoundaryEleOp = BoundaryEle::UserDataOperator;
  using PostProcEle = PostProcFaceOnRefinedMesh;
};

using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomianParentEle = ElementsAndOps<SPACE_DIM>::DomianParentEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;

using EntData = EntitiesFieldData::EntData;

using PostProcEle = ElementsAndOps<SPACE_DIM>::PostProcEle;
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

using OpBaseTimesScalarField = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesScalarField<1, 1>;

using OpMixScalarTimesDiv = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMixScalarTimesDiv<SPACE_DIM, coord_type>;

FTensor::Index<'i', SPACE_DIM> i;
FTensor::Index<'j', SPACE_DIM> j;
FTensor::Index<'k', SPACE_DIM> k;
FTensor::Index<'l', SPACE_DIM> l;

constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();

// Physical parameters
constexpr double a0 = 0.98;
constexpr double rho_m = 0.998;
constexpr double mu_m = 0.0101;
constexpr double rho_p = 0.0012;
constexpr double mu_p = 0.000182;
constexpr double lambda = 7.4;
constexpr double W = 0.25;

// Model parameters
constexpr double h = 0.1; // mesh size
constexpr double eta = h;
constexpr double eta2 = eta * eta;

// Numerical parameteres
constexpr double md = 1e-2;
constexpr double eps = 1e-12;
constexpr double tol = std::numeric_limits<float>::epsilon();

constexpr double rho_ave = (rho_p + rho_m) / 2;
constexpr double rho_diff = (rho_p - rho_m) / 2;
constexpr double mu_ave = (mu_p + mu_m) / 2;
constexpr double mu_diff = (mu_p - mu_m) / 2;

const double kappa = (3. / (4. * std::sqrt(2. * W))) * (lambda / eta);

auto integration_rule = [](int, int, int approx_order) {
  return 2 * approx_order;
};

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

auto init_h = [](double r, double y, double theta) {
  return kernel_eye(r, y, theta);
};

static constexpr int max_nb_levels = 2;
static constexpr int bit_shift = 10;

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

/**
 * @brief set levels of projection operators, which project field data from
 * parent entities, to child, up to to level, i.e. last mesh refinement.
 *
 */
auto set_parent_dofs = [](auto &m_field, auto &fe_top, auto op,
                          auto field_name = std::string()) {
  MoFEMFunctionBegin;

  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
  auto det_ptr = boost::make_shared<VectorDouble>();

  BitRefLevel bit_marker(0);
  int nb_ref_levels = max_nb_levels;
  for (auto l = 1; l <= max_nb_levels; ++l)
    bit_marker |= marker(l);

  boost::function<void(boost::shared_ptr<ForcesAndSourcesCore>, int)>
      add_parent_level =
          [&](boost::shared_ptr<ForcesAndSourcesCore> parent_fe_pt, int level) {
            
            if (level > 0) {

              auto fe_ptr_current = boost::shared_ptr<ForcesAndSourcesCore>(
                  new DomianParentEle(m_field));

              if (op == DomainEleOp::OPSPACE) {
                fe_ptr_current->getOpPtrVector().push_back(
                    new OpCalculateHOJacForFace(jac_ptr));
                fe_ptr_current->getOpPtrVector().push_back(
                    new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
                fe_ptr_current->getOpPtrVector().push_back(
                    new OpSetInvJacH1ForFace(inv_jac_ptr));
              }

              add_parent_level(
                  boost::dynamic_pointer_cast<ForcesAndSourcesCore>(
                      fe_ptr_current),
                  level - 1);

              if (op == DomainEleOp::OPSPACE) {

                parent_fe_pt->getOpPtrVector().push_back(

                    new OpAddParentEntData(

                        H1, DomainEleOp::OPSPACE, fe_ptr_current,

                        BitRefLevel().set(), bit(0).flip(),

                        bit_marker, BitRefLevel().set(),

                        QUIET, Sev::noisy));

              } else {

                parent_fe_pt->getOpPtrVector().push_back(

                    new OpAddParentEntData(

                        field_name, op, fe_ptr_current,

                        BitRefLevel().set(), bit(0).flip(),

                        bit_marker, BitRefLevel().set(),

                        QUIET, Sev::noisy));
              }
            }
          };

  add_parent_level(boost::dynamic_pointer_cast<ForcesAndSourcesCore>(fe_top),
                   max_nb_levels);

  MoFEMFunctionReturn(0);
};

#include <FreeSurfaceOps.hpp>
using namespace FreeSurfaceOps;

struct FreeSurface {

  FreeSurface(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();

  MoFEM::Interface &mField;

  boost::shared_ptr<FEMethod> domianLhsFEPtr;

  Range findEntitiesCrossedByPhaseInterface();
  Range getBitSkin(BitRefLevel bit, BitRefLevel mask);
  Range findEntitiesOnNextLevel(const Range &ents);

  MoFEMErrorCode setBitLevels();

  Range bodySkinBit0;
  Range bodySkinBitAll;
};

//! [Run programme]
MoFEMErrorCode FreeSurface::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();
  // CHKERR assembleSystem();
  // CHKERR solveSystem();
  MoFEMFunctionReturn(0);
}
//! [Run programme]

//! [Read mesh]
MoFEMErrorCode FreeSurface::readMesh() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  simple->getBitRefLevel().reset();

  CHKERR simple->getOptions();
  CHKERR simple->loadFile();

  simple->getBitRefLevel() = bit(0);

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
  // Lagrange multiplier which constrains slip conditions
  CHKERR simple->addBoundaryField("L", H1, AINSWORTH_LEGENDRE_BASE, 1);

  constexpr int order = 1;
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setFieldOrder("P", order - 1);
  CHKERR simple->setFieldOrder("H", order);
  CHKERR simple->setFieldOrder("G", order);
  CHKERR simple->setFieldOrder("L", order);

  // Simple interface will resolve adjacency to DOFs of parent of the element.
  // Using that information MAtrixManager  allocate appropriately size of
  // matrix.
  simple->getParentAdjacencies() = true;
  BitRefLevel bit_marker;
  for (auto l = 1; l <= max_nb_levels; ++l)
    bit_marker |= marker(l);
  simple->getBitAdjEnt() = bit_marker;
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
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
    pipeline.push_back(
        new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline.push_back(new OpSetInvJacH1ForFace(inv_jac_ptr));

    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPSPACE, std::string());
    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW, "H");

    pipeline.push_back(new OpCalculateScalarFieldValues("H", h_ptr));
    pipeline.push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_h_ptr));

    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW, "G");
    pipeline.push_back(new OpCalculateScalarFieldValues("G", g_ptr));
    pipeline.push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("G", grad_g_ptr));
  };

  auto post_proc = [&](auto b) {
    MoFEMFunctionBegin;
    auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
    post_proc_fe->generateReferenceElementMesh();

    post_proc_fe->exeTestHook = [&](FEMethod *fe_ptr) {
      return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(b);
    };

    set_generic(post_proc_fe->getOpPtrVector(), post_proc_fe);

    post_proc_fe->getOpPtrVector().push_back(

        new OpPostProcMap<2, 2>(
            post_proc_fe->postProcMesh, post_proc_fe->mapGaussPts,

            OpPostProcMap<2, 2>::DataMapVec{{"H", h_ptr}, {"G", g_ptr}},

            OpPostProcMap<2, 2>::DataMapMat{{"GRAD_H", grad_h_ptr},
                                            {"GRAD_G", grad_g_ptr}},

            OpPostProcMap<2, 2>::DataMapMat{}

            )

    );

    CHKERR DMoFEMLoopFiniteElements(dm, "dFE", post_proc_fe);
    CHKERR post_proc_fe->writeFile("out_init.h5m");

    MoFEMFunctionReturn(0);
  };

  auto solve_init = [&](auto b) {
    MoFEMFunctionBegin;

    auto set_domain_rhs = [&](auto &pipeline, auto &fe) {
      set_generic(pipeline, fe);
      CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW, "H");
      pipeline.push_back(new OpRhsH<true>("H", nullptr, nullptr, h_ptr,
                                          grad_h_ptr, grad_g_ptr));
      CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW, "G");
      pipeline.push_back(new OpRhsG<true>("G", h_ptr, grad_h_ptr, g_ptr));
    };

    auto set_domain_lhs = [&](auto &pipeline, auto &fe) {
      set_generic(pipeline, fe);
      CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW, "H");
      CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPCOL, "H");
      pipeline.push_back(new OpLhsH_dH<true>("H", nullptr, h_ptr, grad_g_ptr));
      CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPCOL, "G");
      pipeline.push_back(new OpLhsH_dG<true>("H", "G", h_ptr));
      CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW, "G");
      pipeline.push_back(new OpLhsG_dG("G"));
      CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPCOL, "H");
      pipeline.push_back(new OpLhsG_dH<true>("G", "H", h_ptr));
    };

    auto create_subdm = [&]() {
      DM subdm;
      CHKERR DMCreate(mField.get_comm(), &subdm);
      CHKERR DMSetType(subdm, "DMMOFEM");
      CHKERR DMMoFEMCreateSubDM(subdm, dm, "SUB");
      CHKERR DMMoFEMAddElement(subdm, simple->getDomainFEName().c_str());
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
    // CHKERR save_range(mField.get_moab(), "prb_verts_dofs.vtk",
    //                   prb_ents.subset_by_dimension(0));
    // CHKERR save_range(mField.get_moab(), "prb_edges_dofs.vtk",
    //                   prb_ents.subset_by_dimension(1));
    // CHKERR save_range(mField.get_moab(), "prb_faces_dofs.vtk",
    //                   prb_ents.subset_by_dimension(2));

    pipeline_mng->getDomainRhsFE().reset();
    pipeline_mng->getDomainLhsFE().reset();
    CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
    CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);

    auto exe_test_hook = [&](FEMethod *fe_ptr) {
      return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(b);
    };
    pipeline_mng->getDomainRhsFE()->exeTestHook = exe_test_hook;
    pipeline_mng->getDomainLhsFE()->exeTestHook = exe_test_hook;

    set_domain_rhs(pipeline_mng->getOpDomainRhsPipeline(),
                   pipeline_mng->getDomainRhsFE());
    set_domain_lhs(pipeline_mng->getOpDomainLhsPipeline(),
                   pipeline_mng->getDomainLhsFE());

    auto D = smartCreateDMVector(subdm);
    auto snes = pipeline_mng->createSNES(subdm);

    auto set_section_monitor = [&](auto solver) {
      MoFEMFunctionBegin;
      PetscViewerAndFormat *vf;
      CHKERR PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,
                                        PETSC_VIEWER_DEFAULT, &vf);
      CHKERR SNESMonitorSet(
          solver,
          (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal,
                             void *))SNESMonitorFields,
          vf, (MoFEMErrorCode(*)(void **))PetscViewerAndFormatDestroy);
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

    CHKERR SNESSetFromOptions(snes);
    CHKERR SNESSolve(snes, PETSC_NULL, D);

    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(subdm, D, INSERT_VALUES, SCATTER_REVERSE);

    MoFEMFunctionReturn(0);
  };

  bodySkinBit0 = getBitSkin(bit(0), BitRefLevel().set());
  bodySkinBitAll = bodySkinBit0;
  for (auto l = 0; l != max_nb_levels; ++l)
    CHKERR mField.getInterface<BitRefManager>()->updateRangeByChildren(
        bodySkinBitAll, bodySkinBitAll);

  CHKERR solve_init(0);
  CHKERR post_proc(0);

  CHKERR setBitLevels();
  simple->getParentAdjacencies() = true;

  BitRefLevel bit_marker;
  for (auto l = 1; l <= max_nb_levels; ++l)
    bit_marker |= marker(l);
  simple->getBitAdjEnt() = bit_marker;

  BitRefLevel bit_level;
  for (auto l = 0; l <= max_nb_levels; ++l)
    bit_level |= bit(bit_shift + l);
  simple->getBitRefLevel() = bit_level;
  simple->getBitRefLevelMask() = BitRefLevel().set();

  simple->reSetUp(false);

  for (auto field : {"U", "P", "H", "G", "L"}) {
    CHKERR
    mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
        simple->getProblemName(), field, BitRefLevel().set(),
        (bit(bit_shift + max_nb_levels) | bit_marker).flip());
    for (auto l = 1; l <= max_nb_levels; ++l) {
      CHKERR
      mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
          simple->getProblemName(), field, marker(l),
          bit(bit_shift + l - 1).flip());
    }
  }

  CHKERR solve_init(bit_shift + max_nb_levels);
  CHKERR post_proc(bit_shift + max_nb_levels);

  // Clear pipelines
  pipeline_mng->getOpDomainRhsPipeline().clear();
  pipeline_mng->getOpDomainLhsPipeline().clear();

  // Enforce boundary conditions by removing DOFs on symmetry axis and fixed
  // positions
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "SYMETRY",
                                           "U", 0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "SYMETRY",
                                           "L", 0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX", "U",
                                           0, SPACE_DIM);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX", "L",
                                           0, 0);

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode FreeSurface::assembleSystem() {
  MoFEMFunctionBegin;

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
  auto set_domain_general = [&](auto &pipeline) {
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(new OpSetHOWeightsOnFace());
    pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
    pipeline.push_back(
        new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline.push_back(new OpSetInvJacH1ForFace(inv_jac_ptr));

    pipeline.push_back(
        new OpCalculateVectorFieldValuesDot<U_FIELD_DIM>("U", dot_u_ptr));
    pipeline.push_back(
        new OpCalculateVectorFieldValues<U_FIELD_DIM>("U", u_ptr));
    pipeline.push_back(
        new OpCalculateVectorFieldGradient<U_FIELD_DIM, SPACE_DIM>("U",
                                                                   grad_u_ptr));

    pipeline.push_back(new OpCalculateScalarFieldValuesDot("H", dot_h_ptr));
    pipeline.push_back(new OpCalculateScalarFieldValues("H", h_ptr));
    pipeline.push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_h_ptr));
    pipeline.push_back(new OpCalculateScalarFieldValues("G", g_ptr));
    pipeline.push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("G", grad_g_ptr));
    pipeline.push_back(new OpCalculateScalarFieldValues("P", p_ptr));
    pipeline.push_back(
        new OpCalculateDivergenceVectorFieldValues<SPACE_DIM, coord_type>(
            "U", div_u_ptr));
  };

  auto set_domain_rhs = [&](auto &pipeline) {
    pipeline.push_back(new OpRhsU("U", dot_u_ptr, u_ptr, grad_u_ptr, h_ptr,
                                  grad_h_ptr, g_ptr, p_ptr));
    pipeline.push_back(new OpRhsH<false>("H", u_ptr, dot_h_ptr, h_ptr,
                                         grad_h_ptr, grad_g_ptr));
    pipeline.push_back(new OpRhsG<false>("G", h_ptr, grad_h_ptr, g_ptr));
    pipeline.push_back(new OpBaseTimesScalarField(
        "P", div_u_ptr, [](const double r, const double, const double) {
          return cylindrical(r);
        }));
    pipeline.push_back(new OpBaseTimesScalarField(
        "P", p_ptr, [](const double r, const double, const double) {
          return eps * cylindrical(r);
        }));
  };

  auto set_domain_lhs = [&](auto &pipeline) {
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
  };

  auto set_boundary_rhs = [&](auto &pipeline) {
    pipeline.push_back(
        new OpCalculateVectorFieldValues<U_FIELD_DIM>("U", u_ptr));
    pipeline.push_back(new OpCalculateScalarFieldValues("L", lambda_ptr));
    pipeline.push_back(new OpNormalConstrainRhs("L", u_ptr));
    pipeline.push_back(new OpNormalForcebRhs("U", lambda_ptr));
  };

  auto set_boundary_lhs = [&](auto &pipeline) {
    pipeline.push_back(new OpNormalConstrainLhs("L", "U"));
  };

  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule);

  set_domain_general(pipeline_mng->getOpDomainRhsPipeline());
  set_domain_general(pipeline_mng->getOpDomainLhsPipeline());
  set_domain_rhs(pipeline_mng->getOpDomainRhsPipeline());
  set_domain_lhs(pipeline_mng->getOpDomainLhsPipeline());
  set_boundary_rhs(pipeline_mng->getOpBoundaryRhsPipeline());
  set_boundary_lhs(pipeline_mng->getOpBoundaryLhsPipeline());

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
      std::pair<boost::shared_ptr<BoundaryEle>, boost::shared_ptr<VectorDouble>>
          p)
      : dM(dm), postProc(post_proc), liftFE(p.first), liftVec(p.second) {}
  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;
    constexpr int save_every_nth_step = 1;
    if (ts_step % save_every_nth_step == 0) {
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc,
                                      this->getCacheWeakPtr());
      CHKERR postProc->writeFile(
          "out_step_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");

      // MOFEM_LOG("FS", Sev::verbose)
      //     << "writing vector in binary to vector.dat ...";
      // PetscViewer viewer;
      // PetscViewerBinaryOpen(PETSC_COMM_WORLD, "vector.dat", FILE_MODE_WRITE,
      //                       &viewer);
      // VecView(ts_u, viewer);
      // PetscViewerDestroy(&viewer);
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
  boost::shared_ptr<BoundaryEle> liftFE;
  boost::shared_ptr<VectorDouble> liftVec;
};

//! [Solve]
MoFEMErrorCode FreeSurface::solveSystem() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  auto dm = simple->getDM();

  auto get_fe_post_proc = [&]() {
    auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
    post_proc_fe->generateReferenceElementMesh();
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateHOJacForFace(jac_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(inv_jac_ptr));

    post_proc_fe->addFieldValuesPostProc("U");
    post_proc_fe->addFieldValuesPostProc("H");
    post_proc_fe->addFieldValuesPostProc("P");
    post_proc_fe->addFieldValuesPostProc("G");
    post_proc_fe->addFieldValuesGradientPostProc("U", 2);
    post_proc_fe->addFieldValuesGradientPostProc("H", 2);
    post_proc_fe->addFieldValuesGradientPostProc("G", 2);
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

  auto ts = pipeline_mng->createTSIM();
  CHKERR TSSetType(ts, TSALPHA);

  auto set_post_proc_monitor = [&](auto dm) {
    MoFEMFunctionBegin;
    boost::shared_ptr<FEMethod> null_fe;
    auto monitor_ptr =
        boost::make_shared<Monitor>(dm, get_fe_post_proc(), get_lift_fe());
    CHKERR DMMoFEMTSSetMonitor(dm, ts, simple->getDomainFEName(), null_fe,
                               null_fe, monitor_ptr);
    MoFEMFunctionReturn(0);
  };
  CHKERR set_post_proc_monitor(dm);

  auto set_section_monitor = [&](auto solver) {
    MoFEMFunctionBegin;
    SNES snes;
    CHKERR TSGetSNES(solver, &snes);
    PetscViewerAndFormat *vf;
    CHKERR PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,
                                      PETSC_VIEWER_DEFAULT, &vf);
    CHKERR SNESMonitorSet(
        snes,
        (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal, void *))SNESMonitorFields,
        vf, (MoFEMErrorCode(*)(void **))PetscViewerAndFormatDestroy);

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
  CHKERR TSSolve(ts, NULL);
  CHKERR TSGetTime(ts, &ftime);

  MoFEMFunctionReturn(0);
}

Range FreeSurface::findEntitiesCrossedByPhaseInterface() {

  auto &moab = mField.get_moab();
  auto bit_mng = mField.getInterface<BitRefManager>();

  Range vertices;
  CHK_THROW_MESSAGE(bit_mng->getEntitiesByTypeAndRefLevel(
                        bit(0), BitRefLevel().set(), MBVERTEX, vertices),
                    "can not get vertices on bit 0");

  auto &dofs_mi = mField.get_dofs()->get<Unique_mi_tag>();
  auto field_bit_number = mField.get_field_bit_number("H");

  Range plus_range, minus_range;
  std::vector<EntityHandle> plus, minus;

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

  MOFEM_LOG("SELF", Sev::noisy) << "Plus range " << plus_range << endl;
  MOFEM_LOG("SELF", Sev::noisy) << "Minus range " << minus_range << endl;

  auto get_elems = [&](auto &ents) {
    Range adj;
    CHK_MOAB_THROW(
        moab.get_adjacencies(ents, 2, false, adj, moab::Interface::UNION),
        "can not get adjacencies");
    CHK_THROW_MESSAGE(
        bit_mng->filterEntitiesByRefLevel(bit(0), BitRefLevel().set(), adj),
        "can not filter elements with bit 0");
    return adj;
  };

  auto ele_plus = get_elems(plus_range);
  auto ele_minus = get_elems(minus_range);
  auto common = intersect(ele_plus, ele_minus);
  ele_plus = subtract(ele_plus, common);
  ele_minus = subtract(ele_minus, common);

  Range all;
  CHK_THROW_MESSAGE(
      bit_mng->getEntitiesByDimAndRefLevel(bit(0), BitRefLevel().set(), 2, all),
      "can not get vertices on bit 0");
  all = subtract(all, ele_plus);
  all = subtract(all, ele_minus);

  Range conn;
  CHK_MOAB_THROW(moab.get_connectivity(all, conn, true), "");
  all = get_elems(conn);

  return all;
}

Range FreeSurface::getBitSkin(BitRefLevel bit, BitRefLevel mask) {
  auto &moab = mField.get_moab();
  moab::Skinner skin(&moab);
  ParallelComm *pcomm =
      ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
  Range bit_ents;
  CHK_THROW_MESSAGE(
      mField.getInterface<BitRefManager>()->getEntitiesByDimAndRefLevel(
          bit, mask, SPACE_DIM, bit_ents),
      "can't get bit level");
  Range bit_skin;
  CHK_MOAB_THROW(skin.find_skin(0, bit_ents, false, bit_skin),
                 "can't get skin");
  pcomm->filter_pstatus(bit_skin, PSTATUS_SHARED | PSTATUS_MULTISHARED,
                        PSTATUS_NOT, -1, nullptr);
  return bit_skin;
}

Range FreeSurface::findEntitiesOnNextLevel(const Range &ents) {
  auto &moab = mField.get_moab();
  moab::Skinner skin(&moab);
  ParallelComm *pcomm =
      ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
  Range level_skin;
  CHK_MOAB_THROW(skin.find_skin(0, ents, false, level_skin), "can't get skin");
  CHK_MOAB_THROW(pcomm->filter_pstatus(level_skin,
                                       PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                       PSTATUS_NOT, -1, nullptr),
                 "can't filter");
  level_skin = subtract(level_skin, bodySkinBitAll);
  CHK_MOAB_THROW(moab.get_connectivity(level_skin, level_skin),
                 "can't get connectivity of from skin edges");
  Range adj;
  CHK_MOAB_THROW(mField.get_moab().get_adjacencies(level_skin, 2, false, adj,
                                                   moab::Interface::UNION),
                 "can't get adjacencies");
  Range next_level_ents = subtract(ents, adj);
  return next_level_ents;
}

MoFEMErrorCode FreeSurface::setBitLevels() {
  auto &moab = mField.get_moab();
  MoFEMFunctionBegin;

  auto update_and_filter = [&](auto ents, auto bit, auto mask) {
    CHKERR mField.getInterface<BitRefManager>()->updateRangeByChildren(ents,
                                                                       ents);
    CHKERR mField.getInterface<BitRefManager>()->filterEntitiesByRefLevel(
        bit, mask, ents);
    return ents.subset_by_dimension(2);
  };

  std::vector<Range> levels;

  levels.push_back(findEntitiesCrossedByPhaseInterface());
  levels.back() = update_and_filter(levels.back(), bit(1), BitRefLevel().set());
  CHKERR save_range(moab, "1_level.vtk", levels.back());

  for (auto l = 2; l <= max_nb_levels; ++l) {
    levels.push_back(findEntitiesOnNextLevel(levels.back()));
    levels.back() =
        update_and_filter(levels.back(), bit(l), BitRefLevel().set());
    CHKERR save_range(moab, boost::lexical_cast<std::string>(l) + "_level.vtk",
                      levels.back());
  }

  auto reset_bits = [&]() {
    MoFEMFunctionBegin;
    auto ref_ents_ptr = mField.get_ref_ents();
    const auto hi_dit = ref_ents_ptr->end();
    for (auto dit = ref_ents_ptr->begin(); dit != hi_dit; ++dit) {
      auto &bit = (*const_cast<RefEntity *>(dit->get())->getBitRefLevelPtr());
      for (int l = bit_shift; l != BITREFLEVEL_SIZE; ++l) {
        bit[l] = 0;
      }
    }
    MoFEMFunctionReturn(0);
  };

  auto mark_skins = [&](auto &&ents, auto m) {
    auto &moab = mField.get_moab();
    moab::Skinner skin(&moab);
    ParallelComm *pcomm =
        ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
    MoFEMFunctionBegin;
    Range level_skin;
    CHKERR skin.find_skin(0, ents, false, level_skin);
    CHKERR pcomm->filter_pstatus(level_skin,
                                 PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                 PSTATUS_NOT, -1, nullptr);
    level_skin = subtract(level_skin, bodySkinBitAll);
    CHKERR mField.get_moab().get_adjacencies(level_skin, 0, false, level_skin,
                                             moab::Interface::UNION);
    CHKERR mField.getInterface<BitRefManager>()->addBitRefLevel(level_skin,
                                                                marker(m));
    MoFEMFunctionReturn(0);
  };

  auto set_bits = [&]() {
    MoFEMFunctionBegin;

    int l = 1;
    for (auto &r : levels) {
      Range l_mesh;
      CHKERR mField.getInterface<BitRefManager>()->getEntitiesByDimAndRefLevel(
          bit(bit_shift + l - 1), BitRefLevel().set(), SPACE_DIM, l_mesh);
      Range r_parents;
      CHKERR mField.getInterface<BitRefManager>()->updateRangeByParent(
          r, r_parents);
      r = r.subset_by_dimension(2);
      r_parents = r_parents.subset_by_dimension(2);
      CHKERR mark_skins(unite(r, r_parents), l);
      l_mesh = subtract(l_mesh, r_parents);
      l_mesh.merge(r);

      Range conn;
      CHKERR mField.get_moab().get_connectivity(l_mesh.subset_by_dimension(2),
                                                conn, true);
      Range edges;
      CHKERR mField.get_moab().get_adjacencies(l_mesh.subset_by_dimension(2), 1,
                                               false, edges,
                                               moab::Interface::UNION);
      l_mesh.merge(conn);
      l_mesh.merge(edges);

      CHKERR mField.getInterface<BitRefManager>()->addBitRefLevel(
          l_mesh, bit(bit_shift + l));

      ++l;
    }
    MoFEMFunctionReturn(0);
  };

  CHKERR reset_bits();

  {
    Range zero_mesh;
    CHKERR mField.getInterface<BitRefManager>()->getEntitiesByRefLevel(
        bit(0), BitRefLevel().set(), zero_mesh);
    Range conn;
    CHKERR mField.get_moab().get_connectivity(zero_mesh.subset_by_dimension(2),
                                              conn, true);
    Range edges;
    CHKERR mField.get_moab().get_adjacencies(
        zero_mesh.subset_by_dimension(2), 1, false, edges, moab::Interface::UNION);
    zero_mesh.merge(conn);
    zero_mesh.merge(edges);
    CHKERR mField.getInterface<BitRefManager>()->addBitRefLevel(zero_mesh,
                                                                bit(bit_shift));
  }

  CHKERR set_bits();

  for (auto l = 1; l <= max_nb_levels; ++l) {
    CHKERR mField.getInterface<BitRefManager>()->writeBitLevelByDim(
        bit(bit_shift + l), BitRefLevel().set(), SPACE_DIM,
        (boost::lexical_cast<std::string>(l) + "_level_mesh.vtk").c_str(),
        "VTK", "");
    CHKERR mField.getInterface<BitRefManager>()->writeBitLevelByDim(
        BitRefLevel().set(), bit(bit_shift + l).flip(), 0,
        (boost::lexical_cast<std::string>(l) + "_level_remove.vtk").c_str(),
        "VTK", "");
    CHKERR mField.getInterface<BitRefManager>()->writeBitLevelByDim(
        marker(l), bit(bit_shift + l - 1).flip(), 0,
        (boost::lexical_cast<std::string>(l) + "_level_marker.vtk").c_str(),
        "VTK", "");
  }

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
