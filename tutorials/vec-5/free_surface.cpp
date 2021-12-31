/**
 * \file approximation.cpp
 * \FreeSurface approximation.cpp
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
constexpr bool explict_convection = false;
constexpr bool diffusive_flux_term = false; ///< add diffusive flux_term

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = PipelineManager::FaceEle;
  using DomainEleOp = DomainEle::UserDataOperator;
  using BoundaryEle = PipelineManager::EdgeEle;
  using BoundaryEleOp = BoundaryEle::UserDataOperator;
  using PostProcEle = PostProcFaceOnRefinedMesh;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = VolumeElementForcesAndSourcesCore;
  using DomainEleOp = DomainEle::UserDataOperator;
  using BoundaryEle = PipelineManager::FaceEle;
  using BoundaryEleOp = BoundaryEle::UserDataOperator;
  using PostProcEle = PostProcVolumeOnRefinedMesh;
};

using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;

using EntData = DataForcesAndSourcesCore::EntData;

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
constexpr double a0 = 0;
constexpr double rho_p = 1;
constexpr double rho_m = 1;
constexpr double mu_p = 1;
constexpr double mu_m = 1;
constexpr double lambda = 1;
constexpr double W = 0.25;

// Model parameters
constexpr double h = 0.02; // mesh size
constexpr double eta = 2 * h;
constexpr double eta2 = 1;//eta * eta;

// Numerical parameteres
constexpr double md = 0;//1e-2;
constexpr double eps = 1e-10;
constexpr double tol = std::numeric_limits<float>::epsilon();

constexpr double rho_ave = (rho_p + rho_m) / 2;
constexpr double rho_diff = (rho_p - rho_m) / 2;
const double kappa = (3. / (4. * sqrt(2. * W))) * (lambda / eta);

auto integration_rule = [](int, int, int approx_order) {
  return 4 * approx_order;
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
auto cut_off = [](const double h) {
  return my_max(my_min(h));
};
auto d_cut_off = [](const double h) {
  if (h >= -1 && h < 1)
    return 1.;
  else
    return 0.;
};

auto phase_function = [](const double h, const double p, const double m) {
  return rho_diff * cut_off(h) + rho_ave;
};

auto d_phase_function_h = [](const double h, const double p, const double m) {
  return rho_diff * d_cut_off(h);
};

auto get_f = [](const double h) { return 4 * W * h * (h * h - 1); };
auto get_f_dh = [](const double h) { return 4 * W * (3 * h * h - 1); };

auto get_M0 = [](auto h) { return md; };
auto get_M0_dh = [](auto h) { return 0; };
auto get_M2 = [](auto h) { return md * (1 - h * h); };
auto get_M2_dh = [](auto h) { return -md * 2 * h *h; };
auto get_M3 = [](auto h) {
  const double h2 = h * h;
  const double h3 = h2 * h;
  if (h >= 0)
    return md * (2 * h3 - 3 * h2 + 1);
  else
    return md * (-2 * h3 - 3 * h2 + 1);
};

auto get_M3_dh = [](auto h) {
  if (h >= 0)
    return md * (6 * h * (h - 1));
  else
    return md * (-6 * h * (h + 1));
};

auto get_M = [](auto h) { return get_M2(h); };
auto get_M_dh = [](auto h) { return get_M2_dh(h); };

auto get_J = [](auto h, auto &t_g) {
  FTensor::Tensor1<double, U_FIELD_DIM> t_J;
  t_J(i) = -rho_diff * get_M(h) * t_g(i);
  return t_J;
};
auto get_J_dh = [](auto h, auto &t_g) {
  FTensor::Tensor1<double, U_FIELD_DIM> t_J_dh;
  t_J_dh(i) = -rho_diff * get_M_dh(h) * t_g(i);
  return t_J_dh;
};
auto get_J_dg = [](auto h) { return -rho_diff * get_M(h); };

auto get_D = [](const double A) {
  FTensor::Ddg<double, SPACE_DIM, SPACE_DIM> t_D;
  t_D(i, j, k, l) = A * ((t_kd(i, k) ^ t_kd(j, l)) / 4.);
  return t_D;
};

constexpr double R = 0.25;
auto kernel = [](double r, double y, double) {
  const double d = sqrt(r * r + y * y);
  return tanh((R - d) / (eta * sqrt(2)));
};

auto init_h = [](double r, double y, double theta) {
  return kernel(r, y, theta);
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
  CHKERR simple->getOptions();
  CHKERR simple->loadFile();

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode FreeSurface::setupProblem() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  CHKERR simple->addDomainField("U", H1, DEMKOWICZ_JACOBI_BASE, U_FIELD_DIM);
  CHKERR simple->addDomainField("P", H1, DEMKOWICZ_JACOBI_BASE, 1);
  CHKERR simple->addDomainField("H", H1, DEMKOWICZ_JACOBI_BASE, 1);
  CHKERR simple->addDomainField("G", H1, DEMKOWICZ_JACOBI_BASE, 1);
  CHKERR simple->addBoundaryField("U", H1, DEMKOWICZ_JACOBI_BASE, U_FIELD_DIM);
  CHKERR simple->addBoundaryField("H", H1, DEMKOWICZ_JACOBI_BASE, 1);
  CHKERR simple->addBoundaryField("L", H1, DEMKOWICZ_JACOBI_BASE, 1);

  constexpr int order = 2;
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

  auto post_proc = [&]() {
    MoFEMFunctionBegin;
    auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
    post_proc_fe->generateReferenceElementMesh();

    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

    post_proc_fe->addFieldValuesPostProc("H");
    post_proc_fe->addFieldValuesPostProc("G");
    post_proc_fe->addFieldValuesGradientPostProc("G", 2);
    post_proc_fe->addFieldValuesGradientPostProc("H", 2);

    CHKERR DMoFEMLoopFiniteElements(dm, "dFE", post_proc_fe);
    CHKERR post_proc_fe->writeFile("out_init.h5m");

    MoFEMFunctionReturn(0);
  };

  auto solve_init = [&]() {
    MoFEMFunctionBegin;

    auto set_generic = [&](auto &pipeline) {
      auto det_ptr = boost::make_shared<VectorDouble>();
      auto jac_ptr = boost::make_shared<MatrixDouble>();
      auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
      pipeline.push_back(new OpSetHOWeightsOnFace());
      pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
      pipeline.push_back(
          new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
      pipeline.push_back(new OpSetInvJacH1ForFace(inv_jac_ptr));

      pipeline.push_back(new OpCalculateScalarFieldValues("H", h_ptr));
      pipeline.push_back(
          new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_h_ptr));

      pipeline.push_back(new OpCalculateScalarFieldValues("G", g_ptr));
      pipeline.push_back(
          new OpCalculateScalarFieldGradient<SPACE_DIM>("G", grad_g_ptr));
    };

    auto set_domain_rhs = [&](auto &pipeline) {
      pipeline.push_back(new OpRhsH<true>("H", nullptr, nullptr, h_ptr,
                                          grad_h_ptr, grad_g_ptr));
      pipeline.push_back(new OpRhsG("G", h_ptr, grad_h_ptr, g_ptr));
    };

    auto set_domain_lhs = [&](auto &pipeline) {
      pipeline.push_back(new OpLhsH_dH<true>("H", nullptr, h_ptr, grad_g_ptr));
      pipeline.push_back(new OpLhsH_dG("H", "G", h_ptr));
      pipeline.push_back(new OpLhsG_dH("G", "H", h_ptr));
      pipeline.push_back(new OpLhsG_dG("G"));
    };

    auto create_subdm = [&]() {
      DM subdm;
      CHKERR DMCreate(mField.get_comm(), &subdm);
      CHKERR DMSetType(subdm, "DMMOFEM");
      CHKERR DMMoFEMCreateSubDM(subdm, dm, "SUB");
      CHKERR DMMoFEMAddElement(subdm, simple->getDomainFEName().c_str());
      CHKERR DMMoFEMSetSquareProblem(subdm, PETSC_TRUE);
      CHKERR DMMoFEMAddSubFieldRow(subdm, "H");
      CHKERR DMMoFEMAddSubFieldRow(subdm, "G");
      CHKERR DMMoFEMAddSubFieldCol(subdm, "H");
      CHKERR DMMoFEMAddSubFieldCol(subdm, "G");
      CHKERR DMSetUp(subdm);
      return SmartPetscObj<DM>(subdm);
    };

    auto subdm = create_subdm();
    CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
    CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);

    set_generic(pipeline_mng->getOpDomainRhsPipeline());
    set_domain_rhs(pipeline_mng->getOpDomainRhsPipeline());
    set_generic(pipeline_mng->getOpDomainLhsPipeline());
    set_domain_lhs(pipeline_mng->getOpDomainLhsPipeline());

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

  CHKERR solve_init();
  CHKERR post_proc();

  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                           "SYMETRY", "U", 0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(),
                                           "SYMETRY", "L", 0, 0);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX",
                                           "U", 0, SPACE_DIM);
  CHKERR bc_mng->removeBlockDOFsOnEntities(simple->getProblemName(), "FIX",
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
                                  grad_h_ptr, g_ptr, grad_g_ptr, p_ptr));
    pipeline.push_back(
        new OpRhsH("H", u_ptr, dot_h_ptr, h_ptr, grad_h_ptr, grad_g_ptr));
    pipeline.push_back(new OpRhsG("G", h_ptr, grad_h_ptr, g_ptr));
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
    pipeline.push_back(
        new OpLhsU_dU("U", u_ptr, grad_u_ptr, h_ptr, grad_g_ptr));
    pipeline.push_back(new OpLhsU_dH("U", "H", dot_u_ptr, u_ptr, grad_u_ptr,
                                     h_ptr, g_ptr, grad_g_ptr));
    pipeline.push_back(
        new OpLhsU_dG("U", "G", grad_u_ptr, h_ptr, grad_h_ptr, grad_g_ptr));

    pipeline.push_back(new OpLhsH_dU("H", "U", grad_h_ptr));
    pipeline.push_back(new OpLhsH_dH("H", u_ptr, h_ptr, grad_g_ptr));
    pipeline.push_back(new OpLhsH_dG("H", "G", h_ptr));

    pipeline.push_back(new OpLhsG_dH("G", "H", h_ptr));
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
  Monitor(SmartPetscObj<DM> dm, boost::shared_ptr<PostProcEle> post_proc)
      : dM(dm), postProc(post_proc){};
  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;
    constexpr int save_every_nth_step = 10;
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
    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcEle> postProc;
};

//! [Solve]
MoFEMErrorCode FreeSurface::solveSystem() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  auto dm = simple->getDM();

  auto assemble_mass_mat = [&](auto M) {
    MoFEMFunctionBegin;
    auto fe = boost::make_shared<DomainEle>(mField);
    CHKERR MatZeroEntries(M);
    fe->getOpPtrVector().push_back(new OpSetHOWeightsOnFace());
    fe->getOpPtrVector().push_back(new OpDomainMassU(
        "U", "U", [&](double r, double, double) { return 1; }));
    fe->getOpPtrVector().push_back(new OpDomainMassH(
        "H", "H", [&](double r, double, double) { return 1; }));
    fe->getOpPtrVector().push_back(new OpDomainMassH(
        "G", "G", [&](double r, double, double) { return 1; }));
    fe->getRuleHook = integration_rule;

    // Set values on diagonal
    auto D = smartCreateDMVector(dm);
    const MoFEM::Problem *problem_ptr;
    CHKERR DMMoFEMGetProblemPtr(dm, &problem_ptr);
    double *a;
    CHKERR VecGetArray(D, &a);
    auto dofs = problem_ptr->numeredRowDofsPtr;
    auto fields = {"L", "P"};
    for (auto f : fields) {
      const auto bit_number = mField.get_field_bit_number(f);
      auto it = dofs->get<Unique_mi_tag>().lower_bound(
          FieldEntity::getLoBitNumberUId(bit_number));
      auto hi_it = dofs->get<Unique_mi_tag>().upper_bound(
          FieldEntity::getHiBitNumberUId(bit_number));
      for (; it != hi_it; ++it) {
        const auto loc_idx = (*it)->getPetscLocalDofIdx();
        a[loc_idx] = 1;
      }
    }
    CHKERR VecRestoreArray(D, &a);

    CHKERR MatDiagonalSet(M, D, ADD_ALL_VALUES);
    fe->B = M;
    CHKERR DMoFEMLoopFiniteElements(dm, "dFE", fe);
    CHKERR MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    MoFEMFunctionReturn(0);
  };

  auto set_mass_ksp = [&](auto ksp, auto M) {
    MoFEMFunctionBegin;
    CHKERR KSPSetOperators(ksp, M, M);
    CHKERR KSPSetFromOptions(ksp);
    CHKERR KSPSetUp(ksp);
    MoFEMFunctionReturn(0);
  };

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

  auto set_ts = [&](auto solver) {
    MoFEMFunctionBegin;
    SNES snes;
    CHKERR TSGetSNES(solver, &snes);
    KSP ksp;
    CHKERR SNESGetKSP(snes, &ksp);
    MoFEMFunctionReturn(0);
  };

  CHKERR pipeline_mng->setDomainExplicitRhsIntegrationRule(integration_rule);
  auto fe_explicit_rhs = pipeline_mng->getDomainExplicitRhsFE();

  auto M = smartCreateDMMatrix(dm);
  auto ksi_ptr = boost::make_shared<double>();
  auto ksp = createKSP(mField.get_comm());
  auto h_ptr = boost::make_shared<VectorDouble>();

  auto fe_global_terms = boost::make_shared<DomainEle>(mField);
  fe_global_terms->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues("H", h_ptr));
  fe_global_terms->getRuleHook = integration_rule;

  CHKERR assemble_mass_mat(M);
  CHKERR set_mass_ksp(ksp, M);

  auto caluclate_global_terms = [&]() {
    MoFEMFunctionBegin;
    MoFEMFunctionReturn(0);
  };

  auto solve_explicit_rhs = [&]() {
    MoFEMFunctionBegin;
    MOFEM_LOG("FS", Sev::verbose) << "Solve explicit term -> Start";
    if (fe_explicit_rhs->vecAssembleSwitch) {
      CHKERR VecGhostUpdateBegin(fe_explicit_rhs->ts_F, ADD_VALUES,
                                 SCATTER_REVERSE);
      CHKERR VecGhostUpdateEnd(fe_explicit_rhs->ts_F, ADD_VALUES,
                               SCATTER_REVERSE);
      CHKERR VecAssemblyBegin(fe_explicit_rhs->ts_F);
      CHKERR VecAssemblyEnd(fe_explicit_rhs->ts_F);
      CHKERR KSPSolve(ksp, fe_explicit_rhs->ts_F, fe_explicit_rhs->ts_F);
      *fe_explicit_rhs->vecAssembleSwitch = false;
    }
    MOFEM_LOG("FS", Sev::verbose) << "Solve explicit term <- Done";
    MoFEMFunctionReturn(0);
  };

  auto set_domain_rhs_explicit = [&](auto &pipeline) {
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(new OpSetHOWeightsOnFace());
    pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
    pipeline.push_back(
        new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline.push_back(new OpSetInvJacH1ForFace(inv_jac_ptr));

    auto h_ptr = boost::make_shared<VectorDouble>();
    auto grad_h_ptr = boost::make_shared<MatrixDouble>();
    auto u_ptr = boost::make_shared<MatrixDouble>();
    auto grad_u_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(
        new OpCalculateVectorFieldValues<U_FIELD_DIM>("U", u_ptr));
    pipeline.push_back(
        new OpCalculateVectorFieldGradient<U_FIELD_DIM, SPACE_DIM>("U",
                                                                   grad_u_ptr));
    pipeline.push_back(new OpCalculateScalarFieldValues("H", h_ptr));
    pipeline.push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_h_ptr));

    pipeline.push_back(new OpRhsExplicitTermU("U", u_ptr, grad_u_ptr, h_ptr));
    pipeline.push_back(new OpRhsExplicitTermH("H", u_ptr, grad_h_ptr));
  };

  set_domain_rhs_explicit(pipeline_mng->getOpDomainExplicitRhsPipeline());

  fe_explicit_rhs->preProcessHook = caluclate_global_terms;
  fe_explicit_rhs->postProcessHook = solve_explicit_rhs;

  MoFEM::SmartPetscObj<TS> ts;
  ts = pipeline_mng->createTSIMEX();

  CHKERR TSSetType(ts, TSARKIMEX);
  CHKERR TSARKIMEXSetType(ts, TSARKIMEXA2);

  auto set_post_proc_monitor = [&](auto dm) {
    MoFEMFunctionBegin;
    boost::shared_ptr<FEMethod> null_fe;
    auto monitor_ptr = boost::make_shared<Monitor>(dm, get_fe_post_proc());
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
