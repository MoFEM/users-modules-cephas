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

extern "C" {
#include <petsc/private/tsimpl.h>
typedef struct {
  PetscReal stage_time;
  PetscReal shift_V;
  PetscReal scale_F;
  Vec X0, Xa, X1;
  Vec V0, Va, V1;
  PetscReal Alpha_m;
  PetscReal Alpha_f;
  PetscReal Gamma;
  PetscInt order;
  Vec vec_sol_prev;
  Vec vec_lte_work;
  TSStepStatus status;
} TS_Alpha;
}

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
  using BoundaryParentEle = EdgeElementForcesAndSourcesCoreOnChildParent;
  using PostProcEle = PostProcFaceOnRefinedMesh;
  using PostProcEdgeEle = PostProcEdgeOnRefinedMesh;
};

using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomianParentEle = ElementsAndOps<SPACE_DIM>::DomianParentEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using BoundaryParentEle = ElementsAndOps<SPACE_DIM>::BoundaryParentEle;

using EntData = EntitiesFieldData::EntData;

using PostProcEle = ElementsAndOps<SPACE_DIM>::PostProcEle;
using PostProcEdgeEle = ElementsAndOps<SPACE_DIM>::PostProcEdgeEle;

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

// mesh refinement
static constexpr int max_nb_levels = 2;
static constexpr int bit_shift = 10;

auto exe_test_hook = [](FEMethod *fe_ptr) {
  return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
      bit_shift + max_nb_levels);
};

constexpr int order = 4; ///< approximation order

// Physical parameters
constexpr double a0 = 0.98;
constexpr double rho_m = 0.998;
constexpr double mu_m = 0.0101;
constexpr double rho_p = 0.0012;
constexpr double mu_p = 0.000182;
constexpr double lambda = 7.4;
constexpr double W = 0.25;

template <int T> constexpr int powof2() {
  if constexpr (T == 0)
    return 1;
  else
    return powof2<T - 1>() * 2;
};

// Model parameters
constexpr double h = 0.025 / powof2<max_nb_levels>(); // mesh size
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

auto init_h = [](double r, double y, double theta) {
  return kernel_eye(r, y, theta);
};

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

/**
 * @brief set levels of projection operators, which project field data from
 * parent entities, to child, up to to level, i.e. last mesh refinement.
 *
 */
template <typename ELE, typename PARENT>
MoFEMErrorCode
set_parent_dofs(MoFEM::Interface &m_field, boost::shared_ptr<ELE> &fe_top,
                ForcesAndSourcesCore::UserDataOperator::OpType op,
                ExtractParentType<PARENT>, std::string field_name = string()) {
  MoFEMFunctionBegin;

  using OpType = ForcesAndSourcesCore::UserDataOperator::OpType;

  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
  auto det_ptr = boost::make_shared<VectorDouble>();

  BitRefLevel bit_marker(0);
  for (auto l = 0; l <= max_nb_levels; ++l)
    bit_marker |= bit(bit_shift + l);

  boost::function<void(boost::shared_ptr<ForcesAndSourcesCore>, int)>
      add_parent_level =
          [&](boost::shared_ptr<ForcesAndSourcesCore> parent_fe_pt, int level) {
            if (level > 0) {

              auto fe_ptr_current =
                  boost::shared_ptr<ForcesAndSourcesCore>(new PARENT(m_field));

              if (op == OpType::OPSPACE) {
                if (typeid(PARENT) == typeid(DomianParentEle)) {
                  fe_ptr_current->getOpPtrVector().push_back(
                      new OpCalculateHOJacForFace(jac_ptr));
                  fe_ptr_current->getOpPtrVector().push_back(
                      new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
                  fe_ptr_current->getOpPtrVector().push_back(
                      new OpSetInvJacH1ForFace(inv_jac_ptr));
                }
              }

              add_parent_level(
                  boost::dynamic_pointer_cast<ForcesAndSourcesCore>(
                      fe_ptr_current),
                  level - 1);

              if (op == OpType::OPSPACE) {

                parent_fe_pt->getOpPtrVector().push_back(

                    new OpAddParentEntData(

                        H1, OpType::OPSPACE, fe_ptr_current,

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

  MoFEMErrorCode makeRefProblem();

  MoFEM::Interface &mField;

  struct AlphaData : EntityStorage {
    int adjIdx;
  };

  boost::shared_ptr<std::vector<AlphaData>> dofsIndices;
  std::vector<DofIdx> dofsIndicesIdx;

private:
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();

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
  CHKERR assembleSystem();
  CHKERR solveSystem();
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

  auto add_boundary_entities = [&]() {
    MoFEMFunctionBegin;
    bodySkinBit0 = getBitSkin(bit(0), BitRefLevel().set());
    bodySkinBitAll = bodySkinBit0;
    for (auto l = 0; l != max_nb_levels; ++l)
      CHKERR mField.getInterface<BitRefManager>()->updateRangeByChildren(
          bodySkinBitAll, bodySkinBitAll);

    EntityHandle &boundary_meshset = simple->getBoundaryMeshSet();
    CHKERR mField.get_moab().clear_meshset(&boundary_meshset, 1);
    CHKERR mField.get_moab().add_entities(boundary_meshset, bodySkinBitAll);
    Range conn;
    CHKERR mField.get_moab().get_connectivity(
        bodySkinBitAll.subset_by_dimension(1), conn, true);
    CHKERR mField.get_moab().add_entities(boundary_meshset, conn);
    MoFEMFunctionReturn(0);
  };

  CHKERR add_boundary_entities();

#ifndef NDEBUG
  Range bdy_ents;
  CHKERR mField.get_moab().get_entities_by_handle(simple->getBoundaryMeshSet(),
                                                  bdy_ents);
  CHKERR mField.getInterface<BitRefManager>()->filterEntitiesByRefLevel(
      bit(max_nb_levels), BitRefLevel().set(), bdy_ents);
  CHKERR save_range(mField.get_moab(),
                    boost::lexical_cast<std::string>(mField.get_comm_rank()) +
                        "_bdy_ents.vtk",
                    bdy_ents);
#endif

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

  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setFieldOrder("P", order - 1);
  CHKERR simple->setFieldOrder("H", order);
  CHKERR simple->setFieldOrder("G", order);
  CHKERR simple->setFieldOrder("L", order);

  // Simple interface will resolve adjacency to DOFs of parent of the
  // element. Using that information MAtrixManager  allocate appropriately
  // size of matrix.
  simple->getParentAdjacencies() = true;
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

    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPSPACE,
                           ExtractParentType<DomianParentEle>(), std::string());
    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "H");

    pipeline.push_back(new OpCalculateScalarFieldValues("H", h_ptr));
    pipeline.push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_h_ptr));

    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "G");
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
      CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                             ExtractParentType<DomianParentEle>(), "H");
      pipeline.push_back(new OpRhsH<true>("H", nullptr, nullptr, h_ptr,
                                          grad_h_ptr, grad_g_ptr));
      CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                             ExtractParentType<DomianParentEle>(), "G");
      pipeline.push_back(new OpRhsG<true>("G", h_ptr, grad_h_ptr, g_ptr));
    };

    auto set_domain_lhs = [&](auto &pipeline, auto &fe) {
      set_generic(pipeline, fe);
      CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                             ExtractParentType<DomianParentEle>(), "H");
      CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPCOL,
                             ExtractParentType<DomianParentEle>(), "H");
      pipeline.push_back(new OpLhsH_dH<true>("H", nullptr, h_ptr, grad_g_ptr));
      CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPCOL,
                             ExtractParentType<DomianParentEle>(), "G");
      pipeline.push_back(new OpLhsH_dG<true>("H", "G", h_ptr));
      CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                             ExtractParentType<DomianParentEle>(), "G");
      pipeline.push_back(new OpLhsG_dG("G"));
      CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPCOL,
                             ExtractParentType<DomianParentEle>(), "H");
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
    auto snes_ctx_ptr = smartGetDMSnesCtx(subdm);

    auto set_section_monitor = [&](auto solver) {
      MoFEMFunctionBegin;
      CHKERR SNESMonitorSet(snes,
                            (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal,
                                               void *))MoFEMSNESMonitorFields,
                            (void *)(snes_ctx_ptr.get()), nullptr);
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

  CHKERR solve_init(0);
  CHKERR post_proc(0);

  CHKERR makeRefProblem();

  CHKERR solve_init(bit_shift + max_nb_levels);
  CHKERR post_proc(bit_shift + max_nb_levels);

  // Clear pipelines
  pipeline_mng->getOpDomainRhsPipeline().clear();
  pipeline_mng->getOpDomainLhsPipeline().clear();

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
  auto set_domain_general = [&](auto &pipeline, auto &fe) {
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(new OpSetHOWeightsOnFace());
    pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
    pipeline.push_back(
        new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline.push_back(new OpSetInvJacH1ForFace(inv_jac_ptr));

    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPSPACE,
                           ExtractParentType<DomianParentEle>(), std::string());

    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "U");
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

    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "H");
    pipeline.push_back(new OpCalculateScalarFieldValuesDot("H", dot_h_ptr));
    pipeline.push_back(new OpCalculateScalarFieldValues("H", h_ptr));
    pipeline.push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_h_ptr));

    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "G");
    pipeline.push_back(new OpCalculateScalarFieldValues("G", g_ptr));
    pipeline.push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("G", grad_g_ptr));

    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "P");
    pipeline.push_back(new OpCalculateScalarFieldValues("P", p_ptr));
  };

  auto set_domain_rhs = [&](auto &pipeline, auto &fe) {
    set_domain_general(pipeline, fe);
    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "U");
    pipeline.push_back(new OpRhsU("U", dot_u_ptr, u_ptr, grad_u_ptr, h_ptr,
                                  grad_h_ptr, g_ptr, p_ptr));
    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "H");
    pipeline.push_back(new OpRhsH<false>("H", u_ptr, dot_h_ptr, h_ptr,
                                         grad_h_ptr, grad_g_ptr));
    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "G");
    pipeline.push_back(new OpRhsG<false>("G", h_ptr, grad_h_ptr, g_ptr));
    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "P");
    pipeline.push_back(new OpBaseTimesScalarField(
        "P", div_u_ptr, [](const double r, const double, const double) {
          return cylindrical(r);
        }));
    pipeline.push_back(new OpBaseTimesScalarField(
        "P", p_ptr, [](const double r, const double, const double) {
          return eps * cylindrical(r);
        }));
  };

  auto set_domain_lhs = [&](auto &pipeline, auto &fe) {
    set_domain_general(pipeline, fe);
    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "U");
    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPCOL,
                           ExtractParentType<DomianParentEle>(), "U");
    pipeline.push_back(new OpLhsU_dU("U", u_ptr, grad_u_ptr, h_ptr));
    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPCOL,
                           ExtractParentType<DomianParentEle>(), "H");
    pipeline.push_back(
        new OpLhsU_dH("U", "H", dot_u_ptr, u_ptr, grad_u_ptr, h_ptr, g_ptr));
    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPCOL,
                           ExtractParentType<DomianParentEle>(), "G");
    pipeline.push_back(new OpLhsU_dG("U", "G", grad_h_ptr));

    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "H");
    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPCOL,
                           ExtractParentType<DomianParentEle>(), "U");
    pipeline.push_back(new OpLhsH_dU("H", "U", grad_h_ptr));
    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPCOL,
                           ExtractParentType<DomianParentEle>(), "H");
    pipeline.push_back(new OpLhsH_dH<false>("H", u_ptr, h_ptr, grad_g_ptr));
    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPCOL,
                           ExtractParentType<DomianParentEle>(), "G");
    pipeline.push_back(new OpLhsH_dG<false>("H", "G", h_ptr));

    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "G");
    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPCOL,
                           ExtractParentType<DomianParentEle>(), "H");
    pipeline.push_back(new OpLhsG_dH<false>("G", "H", h_ptr));
    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPCOL,
                           ExtractParentType<DomianParentEle>(), "G");
    pipeline.push_back(new OpLhsG_dG("G"));

    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "P");
    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPCOL,
                           ExtractParentType<DomianParentEle>(), "U");
    pipeline.push_back(new OpMixScalarTimesDiv(
        "P", "U",
        [](const double r, const double, const double) {
          return cylindrical(r);
        },
        true, false));
    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPCOL,
                           ExtractParentType<DomianParentEle>(), "P");
    pipeline.push_back(
        new OpDomainMassP("P", "P", [](double r, double, double) {
          return eps * cylindrical(r);
        }));
  };

  auto set_boundary_rhs = [&](auto &pipeline, auto &fe) {
    CHKERR set_parent_dofs(mField, fe, BoundaryEleOp::OPSPACE,
                           ExtractParentType<BoundaryParentEle>(),
                           std::string());

    CHKERR set_parent_dofs(mField, fe, BoundaryEleOp::OPROW,
                           ExtractParentType<BoundaryParentEle>(), "U");
    pipeline.push_back(
        new OpCalculateVectorFieldValues<U_FIELD_DIM>("U", u_ptr));
    CHKERR set_parent_dofs(mField, fe, BoundaryEleOp::OPROW,
                           ExtractParentType<BoundaryParentEle>(), "L");
    pipeline.push_back(new OpCalculateScalarFieldValues("L", lambda_ptr));
    pipeline.push_back(new OpNormalConstrainRhs("L", u_ptr));
    CHKERR set_parent_dofs(mField, fe, BoundaryEleOp::OPROW,
                           ExtractParentType<BoundaryParentEle>(), "U");
    pipeline.push_back(new OpNormalForceRhs("U", lambda_ptr));
  };

  auto set_boundary_lhs = [&](auto &pipeline, auto &fe) {
    CHKERR set_parent_dofs(mField, fe, BoundaryEleOp::OPSPACE,
                           ExtractParentType<BoundaryParentEle>(),
                           std::string());
    CHKERR set_parent_dofs(mField, fe, BoundaryEleOp::OPROW,
                           ExtractParentType<BoundaryParentEle>(), "L");
    CHKERR set_parent_dofs(mField, fe, BoundaryEleOp::OPCOL,
                           ExtractParentType<BoundaryParentEle>(), "U");
    pipeline.push_back(new OpNormalConstrainLhs("L", "U"));
  };

  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);

  pipeline_mng->getDomainRhsFE()->exeTestHook = exe_test_hook;
  pipeline_mng->getDomainLhsFE()->exeTestHook = exe_test_hook;

  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule);

  pipeline_mng->getBoundaryRhsFE()->exeTestHook = exe_test_hook;
  pipeline_mng->getBoundaryLhsFE()->exeTestHook = exe_test_hook;

  set_domain_rhs(pipeline_mng->getOpDomainRhsPipeline(),
                 pipeline_mng->getDomainRhsFE());
  set_domain_lhs(pipeline_mng->getOpDomainLhsPipeline(),
                 pipeline_mng->getDomainLhsFE());

  set_boundary_rhs(pipeline_mng->getOpBoundaryRhsPipeline(),
                   pipeline_mng->getBoundaryRhsFE());
  set_boundary_lhs(pipeline_mng->getOpBoundaryLhsPipeline(),
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
      postProc->elementsMap
          .clear(); // clear map of post-processed elements, new set is
                    // created each time mesh is refined.
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc,
                                      this->getCacheWeakPtr());
      CHKERR postProc->writeFile(
          "out_step_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");

      postProcEdge->elementsMap
          .clear(); // clear map and post proc mesh after each mesh refinment
      postProcEdge->postProcMesh.delete_mesh();
      CHKERR DMoFEMLoopFiniteElements(dM, "bFE", postProcEdge,
                                      this->getCacheWeakPtr());
      CHKERR postProcEdge->writeFile(
          "out_step_bdy_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");

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
  boost::shared_ptr<PostProcEdgeEle> postProcEdge;
  boost::shared_ptr<BoundaryEle> liftFE;
  boost::shared_ptr<VectorDouble> liftVec;
};

static FreeSurface *free_surface_ptr = nullptr;

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

    auto u_ptr = boost::make_shared<MatrixDouble>();
    auto grad_u_ptr = boost::make_shared<MatrixDouble>();
    auto h_ptr = boost::make_shared<VectorDouble>();
    auto grad_h_ptr = boost::make_shared<MatrixDouble>();
    auto p_ptr = boost::make_shared<VectorDouble>();
    auto g_ptr = boost::make_shared<VectorDouble>();

    post_proc_fe->exeTestHook = exe_test_hook;

    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateHOJacForFace(jac_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(inv_jac_ptr));

    CHKERR set_parent_dofs(mField, post_proc_fe, DomainEleOp::OPSPACE,
                           ExtractParentType<DomianParentEle>(), std::string());

    CHKERR set_parent_dofs(mField, post_proc_fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "U");
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<U_FIELD_DIM>("U", u_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<U_FIELD_DIM, SPACE_DIM>("U",
                                                                   grad_u_ptr));

    CHKERR set_parent_dofs(mField, post_proc_fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "H");
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("H", h_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_h_ptr));

    CHKERR set_parent_dofs(mField, post_proc_fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "P");
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("P", p_ptr));
    CHKERR set_parent_dofs(mField, post_proc_fe, DomainEleOp::OPROW,
                           ExtractParentType<DomianParentEle>(), "G");
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("G", g_ptr));

    post_proc_fe->getOpPtrVector().push_back(

        new OpPostProcMap<2, 2>(
            post_proc_fe->postProcMesh, post_proc_fe->mapGaussPts,

            OpPostProcMap<2, 2>::DataMapVec{
                {"H", h_ptr}, {"P", p_ptr}, {"G", g_ptr}},

            OpPostProcMap<2, 2>::DataMapMat{{"U", u_ptr},
                                            {"H_GRAD", grad_h_ptr}},

            OpPostProcMap<2, 2>::DataMapMat{{"GRAD_U", grad_u_ptr}}

            )

    );

    return post_proc_fe;
  };

  auto get_bdy_post_proc_fe = [&]() {
    auto post_proc_fe = boost::make_shared<PostProcEdgeEle>(mField);
    post_proc_fe->generateReferenceElementMesh();

    post_proc_fe->exeTestHook = exe_test_hook;

    auto u_ptr = boost::make_shared<MatrixDouble>();
    auto p_ptr = boost::make_shared<VectorDouble>();
    auto lambda_ptr = boost::make_shared<VectorDouble>();

    CHKERR set_parent_dofs(mField, post_proc_fe, BoundaryEleOp::OPSPACE,
                           ExtractParentType<BoundaryParentEle>(),
                           std::string());

    CHKERR set_parent_dofs(mField, post_proc_fe, BoundaryEleOp::OPROW,
                           ExtractParentType<BoundaryParentEle>(), "U");
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<U_FIELD_DIM>("U", u_ptr));
    CHKERR set_parent_dofs(mField, post_proc_fe, BoundaryEleOp::OPROW,
                           ExtractParentType<BoundaryParentEle>(), "L");
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("L", lambda_ptr));
    CHKERR set_parent_dofs(mField, post_proc_fe, BoundaryEleOp::OPROW,
                           ExtractParentType<BoundaryParentEle>(), "P");
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("P", p_ptr));

    post_proc_fe->getOpPtrVector().push_back(

        new OpPostProcMap<2, 2>(
            post_proc_fe->postProcMesh, post_proc_fe->mapGaussPts,

            OpPostProcMap<2, 2>::DataMapVec{{"L", lambda_ptr}, {"P", p_ptr}},

            OpPostProcMap<2, 2>::DataMapMat{{"U", u_ptr}},

            OpPostProcMap<2, 2>::DataMapMat()

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

    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPSPACE,
                           ExtractParentType<BoundaryParentEle>(),
                           std::string());

    CHKERR set_parent_dofs(mField, fe, DomainEleOp::OPROW,
                           ExtractParentType<BoundaryParentEle>(), "P");
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
  CHKERR TSSetUp(ts);

  free_surface_ptr = this;
  // We create vector with AlphaData. AlphaData is storage for previous indexing
  // of entities.
  dofsIndices = boost::make_shared<std::vector<AlphaData>>();
  auto prb_ptr = getProblemPtr(dm);
  dofsIndices->resize(mField.get_field_ents()->size());
  dofsIndicesIdx.clear();
  dofsIndicesIdx.reserve(prb_ptr->numeredRowDofsPtr->size());

  auto refine_mesh = [](TS ts) {
    MoFEMFunctionBeginHot;
    auto &m_field = free_surface_ptr->mField;
    auto &dofs_indices = free_surface_ptr->dofsIndices;
    auto &dofs_indices_idx = free_surface_ptr->dofsIndicesIdx;
    auto simple = m_field.getInterface<Simple>();
    auto dm = simple->getDM();
    auto prb_ptr = getProblemPtr(dm);

    auto field_ents_ptr = m_field.get_field_ents();
    auto dofs_ptr = prb_ptr->numeredRowDofsPtr;

    auto set_data_on_entity_with_current_indices = [&]() {
      MoFEMFunctionBegin;
      // Here we use functionality which allows to attach week pointer to field
      // entity. We use property, that nodes are number contiguously on given
      // entity, so we have to store only first index.
      int ent_idx = 0;
      for (auto &e_ptr : *field_ents_ptr) {
        const auto &uid = e_ptr->getLocalUniqueId();
        const auto lo_uid = FieldEntity::getLoFieldEntityUId(uid);
        const auto hi_uid = FieldEntity::getHiFieldEntityUId(uid);
        const auto hi = dofs_ptr->get<Unique_mi_tag>().upper_bound(hi_uid);
        (*dofs_indices)[ent_idx].adjIdx = dofs_indices_idx.size();
        for (auto lo = dofs_ptr->get<Unique_mi_tag>().lower_bound(lo_uid);
             lo != hi; ++lo) {
          dofs_indices_idx.push_back((*lo)->getPetscLocalDofIdx());
        }
        e_ptr->getWeakStoragePtr() = boost::shared_ptr<AlphaData>(
            dofs_indices, &((*dofs_indices)[ent_idx]));
        ++ent_idx;
      }
      MoFEMFunctionReturn(0);
    };

    CHKERR set_data_on_entity_with_current_indices();

    // Copy internal vector of Alpha method, to new vector. Using storage
    // AlphaData.
    auto copy_data = [&](Vec v_old, Vec v_new) {
      MoFEMFunctionBegin;
      if (v_old) {
        double *a_new;
        const double *a_old;
        CHKERR VecGetArray(v_new, &a_new);
        CHKERR VecGetArrayRead(v_old, &a_old);

        int ent_idx = 0;
        for (auto &e_ptr : *field_ents_ptr) {
          if (auto ptr = e_ptr->getWeakStoragePtr().lock()) {
            if (auto alpha_ptr = boost::dynamic_pointer_cast<AlphaData>(ptr)) {
              auto adj = alpha_ptr->adjIdx;
              const auto &uid = e_ptr->getLocalUniqueId();
              const auto lo_uid = FieldEntity::getLoFieldEntityUId(uid);
              const auto hi_uid = FieldEntity::getHiFieldEntityUId(uid);
              const auto hi =
                  dofs_ptr->get<Unique_mi_tag>().upper_bound(hi_uid);
              for (auto lo = dofs_ptr->get<Unique_mi_tag>().lower_bound(lo_uid);
                   lo != hi; ++lo) {
                a_new[(*lo)->getPetscLocalDofIdx()] =
                    a_old[dofs_indices_idx[adj + (*lo)->getEntDofIdx()]];
              }
            } else {
#ifndef NDEBUG
              SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                      "Expected alpha data on entity");
#endif
            }
          } else {
#ifndef NDEBUG
            SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                    "Expected data on entity");
#endif
          }
          ++ent_idx;
        }

        CHKERR VecRestoreArray(v_new, &a_new);
        CHKERR VecRestoreArrayRead(v_old, &a_old);
      }
      MoFEMFunctionReturn(0);
    };

    auto get_current_solution = [&]() {
      auto solution = smartCreateDMVector(dm);
      CHKERR DMoFEMMeshToLocalVector(simple->getDM(), solution, INSERT_VALUES,
                                     SCATTER_FORWARD);
      CHKERR VecGhostUpdateBegin(solution, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(solution, INSERT_VALUES, SCATTER_FORWARD);
      return solution;
    };

    auto solution_current = get_current_solution();

    CHKERR free_surface_ptr->makeRefProblem();

    auto reset_snes = [&]() {
      MoFEMFunctionBegin;
      SNES snes;
      CHKERR TSGetSNES(ts, &snes);
      CHKERR SNESReset(snes);
      CHKERR SNESSetDM(snes, dm);
      CHKERR SNESMonitorCancel(snes);

      auto set_section_monitor = [&](auto snes) {
        MoFEMFunctionBegin;
        auto snes_ctx_ptr = smartGetDMSnesCtx(simple->getDM());
        CHKERR SNESMonitorSet(snes,
                              (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal,
                                                 void *))MoFEMSNESMonitorFields,
                              (void *)(snes_ctx_ptr.get()), nullptr);
        MoFEMFunctionReturn(0);
      };

      CHKERR set_section_monitor(snes);

      MoFEMFunctionReturn(0);
    };

    auto set_dm_section = [&](auto dm) {
      MoFEMFunctionBegin;
      auto section =
          m_field.getInterface<ISManager>()->sectionCreate(prb_ptr->getName());
      CHKERR DMSetSection(dm, section);
      MoFEMFunctionReturn(0);
    };

    auto update_private_alpha_method_data = [&]() {
      MoFEMFunctionBegin;

      TS_Alpha *th = (TS_Alpha *)ts->data;

      Vec X0 = th->X0;
      Vec Xa = th->Xa;
      Vec X1 = th->X1;
      Vec V0 = th->V0;
      Vec Va = th->Va;
      Vec V1 = th->V1;
      Vec vec_sol_prev = th->vec_sol_prev;
      Vec vec_lte_work = th->vec_lte_work;

      Vec nX0;
      CHKERR DMCreateGlobalVector_MoFEM(dm, &nX0);
      Vec nXa;
      CHKERR VecDuplicate(nX0, &nXa);
      Vec nX1;
      CHKERR VecDuplicate(nX0, &nX1);
      Vec nV0;
      CHKERR VecDuplicate(nX0, &nV0);
      Vec nVa;
      CHKERR VecDuplicate(nX0, &nVa);
      Vec nV1;
      CHKERR VecDuplicate(nX0, &nV1);
      Vec n_vec_sol_prev;
      CHKERR VecDuplicate(nX0, &n_vec_sol_prev);
      Vec n_vec_lte_work;
      CHKERR VecDuplicate(nX0, &n_vec_lte_work);

      CHKERR copy_data(X0, nX0);
      CHKERR copy_data(Xa, nXa);
      CHKERR copy_data(X1, nX1);
      CHKERR copy_data(V0, nV0);
      CHKERR copy_data(Va, nVa);
      CHKERR copy_data(V1, nV1);
      CHKERR copy_data(vec_sol_prev, n_vec_sol_prev);
      CHKERR copy_data(vec_lte_work, n_vec_lte_work);

      CHKERR VecDestroy(&X0);
      CHKERR VecDestroy(&Xa);
      CHKERR VecDestroy(&X1);
      CHKERR VecDestroy(&V0);
      CHKERR VecDestroy(&Va);
      CHKERR VecDestroy(&V1);
      CHKERR VecDestroy(&vec_sol_prev);
      CHKERR VecDestroy(&vec_lte_work);

      th->X0 = nX0;
      th->Xa = nXa;
      th->X1 = nX1;
      th->V0 = nV0;
      th->Va = nV1;
      th->V1 = nVa;
      th->vec_sol_prev = n_vec_sol_prev;
      th->vec_lte_work = n_vec_lte_work;

      MoFEMFunctionReturn(0);
    };

    auto update_solution = [&]() {
      MoFEMFunctionBegin;
      for (auto f : {"U", "P", "H", "G", "L"}) {
        CHKERR m_field.getInterface<FieldBlas>()->setField(0, f);
      }
      auto solution_new = smartCreateDMVector(dm);
      CHKERR copy_data(solution_current, solution_new);
      CHKERR VecGhostUpdateBegin(solution_current, INSERT_VALUES,
                                 SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(solution_current, INSERT_VALUES,
                               SCATTER_FORWARD);

      CHKERR TSSetSolution(ts, solution_new);
      MoFEMFunctionReturn(0);
    };

    CHKERR set_dm_section(dm);
    CHKERR reset_snes();
    CHKERR update_private_alpha_method_data();
    CHKERR update_solution();

    MoFEMFunctionReturnHot(0);
  };

  CHKERR TSSetPreStep(ts, refine_mesh);

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
  std::vector<EntityHandle> plus, minus, out_of_scope;

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

  CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(plus_range);
  CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(minus_range);

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
  CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(conn);
  all = get_elems(conn);
  CHK_MOAB_THROW(moab.get_connectivity(all, conn, true), "");
  CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(conn);
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
  level_skin = subtract(level_skin, bodySkinBitAll);
  Range level_skin_owned;
  CHKERR pcomm->filter_pstatus(level_skin, PSTATUS_NOT_OWNED, PSTATUS_NOT, -1,
                               &level_skin_owned);
  Range level_skin_not_owned;
  CHKERR pcomm->filter_pstatus(level_skin, PSTATUS_NOT_OWNED, PSTATUS_AND, -1,
                               &level_skin_not_owned);
  CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(
      level_skin_owned);
  CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(
      level_skin_not_owned);
  auto level_skin_sync = intersect(level_skin_owned, level_skin_not_owned);
  CHK_MOAB_THROW(pcomm->filter_pstatus(level_skin_sync,
                                       PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                       PSTATUS_OR, -1, nullptr),
                 "can't filter");
  level_skin = subtract(level_skin, level_skin_sync);

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
  std::vector<Range> levels;
  MoFEMFunctionBegin;

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

  auto update_and_filter = [&](auto ents, auto bit, auto mask) {
    CHKERR mField.getInterface<BitRefManager>()->updateRangeByChildren(ents,
                                                                       ents);
    CHKERR mField.getInterface<BitRefManager>()->filterEntitiesByRefLevel(
        bit, mask, ents);
    return ents.subset_by_dimension(2);
  };

  auto mark_skins = [&](auto &&ents, auto m) {
    auto &moab = mField.get_moab();
    moab::Skinner skin(&moab);
    ParallelComm *pcomm =
        ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
    MoFEMFunctionBegin;
    Range level_skin;
    CHKERR skin.find_skin(0, ents, false, level_skin);
    Range level_skin_owned;
    CHKERR pcomm->filter_pstatus(level_skin, PSTATUS_NOT_OWNED, PSTATUS_NOT, -1,
                                 &level_skin_owned);
    Range level_skin_not_owned;
    CHKERR pcomm->filter_pstatus(level_skin, PSTATUS_NOT_OWNED, PSTATUS_AND, -1,
                                 &level_skin_not_owned);
    CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(
        level_skin_owned);
    CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(
        level_skin_not_owned);
    auto level_skin_sync = intersect(level_skin_owned, level_skin_not_owned);
    CHK_MOAB_THROW(pcomm->filter_pstatus(level_skin_sync,
                                         PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                         PSTATUS_OR, -1, nullptr),
                   "can't filter");
    level_skin = subtract(level_skin, level_skin_sync);

    level_skin = subtract(level_skin, bodySkinBitAll);
    CHKERR mField.get_moab().get_adjacencies(level_skin, 0, false, level_skin,
                                             moab::Interface::UNION);
    CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(
        level_skin);
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

      CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(l_mesh);
      CHKERR mField.getInterface<BitRefManager>()->addBitRefLevel(
          l_mesh, bit(bit_shift + l));

      ++l;
    }
    MoFEMFunctionReturn(0);
  };

  CHKERR reset_bits();

  levels.push_back(findEntitiesCrossedByPhaseInterface());
  levels.back() = update_and_filter(levels.back(), bit(1), BitRefLevel().set());
#ifndef NDEBUG
  CHKERR save_range(moab, "1_level.vtk", levels.back());
#endif

  for (auto l = 2; l <= max_nb_levels; ++l) {
    levels.push_back(findEntitiesOnNextLevel(levels.back()));
    levels.back() =
        update_and_filter(levels.back(), bit(l), BitRefLevel().set());
#ifndef NDEBUG
    CHKERR save_range(moab, boost::lexical_cast<std::string>(l) + "_level.vtk",
                      levels.back());
#endif
  }

  {
    Range zero_mesh;
    CHKERR mField.getInterface<BitRefManager>()->getEntitiesByRefLevel(
        bit(0), BitRefLevel().set(), zero_mesh);
    Range conn;
    CHKERR mField.get_moab().get_connectivity(zero_mesh.subset_by_dimension(2),
                                              conn, true);
    Range edges;
    CHKERR mField.get_moab().get_adjacencies(zero_mesh.subset_by_dimension(2),
                                             1, false, edges,
                                             moab::Interface::UNION);
    zero_mesh.merge(conn);
    zero_mesh.merge(edges);

    CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(zero_mesh);
    CHKERR mField.getInterface<BitRefManager>()->addBitRefLevel(zero_mesh,
                                                                bit(bit_shift));
  }

  CHKERR set_bits();

#ifndef NDEBUG
  for (auto l = 1; l <= max_nb_levels; ++l) {
    CHKERR mField.getInterface<BitRefManager>()->writeBitLevelByDim(
        bit(bit_shift + l), BitRefLevel().set(), SPACE_DIM,
        (boost::lexical_cast<std::string>(l) + "_level_mesh" +
         boost::lexical_cast<std::string>(mField.get_comm_rank()) + ".vtk")
            .c_str(),
        "VTK", "");
    CHKERR mField.getInterface<BitRefManager>()->writeBitLevelByDim(
        BitRefLevel().set(), bit(bit_shift + l).flip(), 0,
        (boost::lexical_cast<std::string>(l) + "_level_remove" +
         boost::lexical_cast<std::string>(mField.get_comm_rank()) + ".vtk")
            .c_str(),
        "VTK", "");
    CHKERR mField.getInterface<BitRefManager>()->writeBitLevelByDim(
        marker(l), bit(bit_shift + l - 1).flip(), 0,
        (boost::lexical_cast<std::string>(l) + "_level_marker" +
         boost::lexical_cast<std::string>(mField.get_comm_rank()) + ".vtk")
            .c_str(),
        "VTK", "");
  }
#endif

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode FreeSurface::makeRefProblem() {
  auto simple = mField.getInterface<Simple>();
  auto bc_mng = mField.getInterface<BcManager>();
  MoFEMFunctionBegin;

  CHKERR setBitLevels();

  BitRefLevel bit_marker;
  for (auto l = 1; l <= max_nb_levels; ++l)
    bit_marker |= marker(l);
  simple->getBitAdjEnt() = BitRefLevel().set(); // bit_marker;

  BitRefLevel bit_level(0);
  for (auto l = 0; l <= max_nb_levels; ++l)
    bit_level |= bit(bit_shift + l);
  simple->getBitRefLevel() = bit_level;
  simple->getBitRefLevelMask() = BitRefLevel().set();

  // auto ref_fe_ptr = mField.get_ref_finite_elements();
  // const_cast<RefElement_multiIndex *>(ref_fe_ptr)->clear();
  // auto fe_ptr = mField.get_finite_elements();
  // const_cast<FiniteElement_multiIndex *>(fe_ptr)->clear();
  // auto fe_adj_ptr = mField.get_ents_elements_adjacency();
  // const_cast<FieldEntityEntFiniteElementAdjacencyMap_multiIndex
  // *>(fe_adj_ptr)
  //     ->clear();

  // mField.getInterface<ProblemsManager>()->synchroniseProblemEntities =
  //     PETSC_TRUE;
  simple->buildFiniteElements();
  simple->reSetUp(true);

  auto get_ents_bit_ref = [&](auto bit, auto mask) {
    Range ents;
    CHK_THROW_MESSAGE(
        mField.getInterface<BitRefManager>()->getEntitiesByRefLevel(bit, mask,
                                                                    ents),
        "Can not get ents on bit ref level");
    return ents;
  };

  Range ents_to_remove;
  ents_to_remove.merge(get_ents_bit_ref(BitRefLevel().set(), bit_level.flip()));
  for (auto l = 1; l <= max_nb_levels; ++l) {
    ents_to_remove.merge(
        get_ents_bit_ref(marker(l), bit(bit_shift + l - 1).flip()));
  }

  for (auto field : {"U", "P", "H", "G", "L"}) {
    CHKERR
    mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
        simple->getProblemName(), field, ents_to_remove);
  }

  // Takes edges or faces, find their children which are also edges or faces,
  // and remove higher order approximation from those entities. Since
  // approximation is provided by parents (underlying entities).a
  auto get_children_edges = [&]() {
    Range children_edges;
    for (auto l = 0; l != max_nb_levels; ++l) {
      auto bit_mng = mField.getInterface<BitRefManager>();
      Range bit_ents =
          get_ents_bit_ref(bit(bit_shift + l), BitRefLevel().set());
      Range bit_children_edges;
      CHKERR bit_mng->updateRangeByChildren(bit_ents.subset_by_dimension(1),
                                            bit_children_edges);
      CHKERR bit_mng->filterEntitiesByRefLevel(bit(bit_shift + l + 1),
                                               bit(bit_shift + l).flip(),
                                               bit_children_edges);
      children_edges.merge(bit_children_edges.subset_by_dimension(1));
      Range bit_children_faces;
      CHKERR bit_mng->updateRangeByChildren(bit_ents.subset_by_dimension(2),
                                            bit_children_faces);
      CHKERR bit_mng->filterEntitiesByRefLevel(bit(bit_shift + l + 1),
                                               bit(bit_shift + l).flip(),
                                               bit_children_faces);
      children_edges.merge(bit_children_faces.subset_by_dimension(2));
    }
    return children_edges;
  };

  auto children_edges = get_children_edges();

#ifndef NDEBUG
  CHKERR save_range(mField.get_moab(), "children_edges.vtk", children_edges);
#endif

  for (auto field : {"U", "H", "G", "L"}) {
    CHKERR
    mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
        simple->getProblemName(), field, children_edges, 0, MAX_DOFS_ON_ENTITY,
        2, std::max(2, order));
  }

  for (auto field : {"P"}) {
    CHKERR
    mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
        simple->getProblemName(), field, children_edges, 0, MAX_DOFS_ON_ENTITY,
        1, std::max(1, order - 1));
  }

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
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "ZERO",
                                           "L", 0, 0);

  MoFEMFunctionReturn(0);
};

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
