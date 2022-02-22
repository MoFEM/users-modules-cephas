#include <stdlib.h>
#include <BasicFiniteElements.hpp>
using namespace MoFEM;

static char help[] = "...\n\n";

using DomainEle = PipelineManager::FaceEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = PipelineManager::EdgeEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcFaceOnRefinedMesh;

using OpBoundaryMass = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<1, 1>;
using OpBoundaryTimeScalarField = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesScalarField<1>;
using OpBoundarySource = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<1, 1>;

using AssemblyDomainEleOp =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::OpBase;
using AssemblyBoundaryEleOp =
    FormsIntegrators<BoundaryEleOp>::Assembly<PETSC>::OpBase;

FTensor::Index<'i', 2> i;

/** \brief Integrate the domain tangent matrix (LHS)

\f[
\sum\limits_j {\left[ {\int\limits_{{\Omega _e}} {\left( {{a_n}\nabla {\phi _i}
\cdot \nabla {\phi _j} - a_n^3\nabla {\phi _i}\left( {\nabla u \cdot \nabla
{\phi _j}} \right)\nabla u} \right)d{\Omega _e}} } \right]\delta {U_j}}  =
\int\limits_{{\Omega _e}} {{\phi _i}fd{\Omega _e}}  - \int\limits_{{\Omega _e}}
{\nabla {\phi _i}{a_n}\nabla ud{\Omega _e}} \\
{a_n} = \frac{1}{{{{\left( {1 +
{{\left| {\nabla u} \right|}^2}} \right)}^{\frac{1}{2}}}}}
\f]

*/
struct OpDomainTangentMatrix : public AssemblyDomainEleOp {
public:
  OpDomainTangentMatrix(std::string row_field_name, std::string col_field_name,
                        boost::shared_ptr<MatrixDouble> field_grad_mat)
      : AssemblyDomainEleOp(row_field_name, col_field_name,
                            DomainEleOp::OPROWCOL),
        fieldGradMat(field_grad_mat) {}

  MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data) {
    MoFEMFunctionBegin;

    auto &locLhs = AssemblyDomainEleOp::locMat;

    // get element area
    const double area = getMeasure();

    // get number of integration points
    const int nb_integration_points = getGaussPts().size2();
    // get integration weights
    auto t_w = getFTensor0IntegrationWeight();
    // get gradient of the field at integration points
    auto t_field_grad = getFTensor1FromMat<2>(*fieldGradMat);

    // get derivatives of base functions on row
    auto t_row_diff_base = row_data.getFTensor1DiffN<2>();

    // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL MATRIX
    for (int gg = 0; gg != nb_integration_points; gg++) {

      const double a = t_w * area;
      const double an = 1. / std::sqrt(1 + t_field_grad(i) * t_field_grad(i));

      for (int rr = 0; rr != AssemblyDomainEleOp::nbRows; ++rr) {
        // get derivatives of base functions on column
        auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);

        for (int cc = 0; cc != AssemblyDomainEleOp::nbRows; cc++) {

          // calculate components of the local matrix
          locLhs(rr, cc) += (t_row_diff_base(i) * t_col_diff_base(i)) * an * a -
                            t_row_diff_base(i) *
                                (t_field_grad(i) * t_col_diff_base(i)) *
                                t_field_grad(i) * an * an * an * a;

          // move to the derivatives of the next base functions on column
          ++t_col_diff_base;
        }

        // move to the derivatives of the next base functions on row
        ++t_row_diff_base;
      }

      // move to the weight of the next integration point
      ++t_w;
      // move to the gradient of the field at the next integration point
      ++t_field_grad;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> fieldGradMat;
};

/** \brief Integrate the domain residual vector (RHS)

\f[
\sum\limits_j {\left[ {\int\limits_{{\Omega _e}} {\left( {{a_n}\nabla {\phi _i}
\cdot \nabla {\phi _j} - a_n^3\nabla {\phi _i}\left( {\nabla u \cdot \nabla
{\phi _j}} \right)\nabla u} \right)d{\Omega _e}} } \right]\delta {U_j}}  =
\int\limits_{{\Omega _e}} {{\phi _i}fd{\Omega _e}}  - \int\limits_{{\Omega _e}}
{\nabla {\phi _i}{a_n}\nabla ud{\Omega _e}} \\
{a_n} = \frac{1}{{{{\left( {1 +
{{\left| {\nabla u} \right|}^2}} \right)}^{\frac{1}{2}}}}}
\f]

*/
struct OpDomainResidualVector : public AssemblyDomainEleOp {
public:
  OpDomainResidualVector(std::string field_name,
                         boost::shared_ptr<MatrixDouble> field_grad_mat)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        fieldGradMat(field_grad_mat) {}

  MoFEMErrorCode iNtegrate(EntData &data) {
    MoFEMFunctionBegin;

    auto &nf = AssemblyDomainEleOp::locF;

    // get element area
    const double area = getMeasure();

    // get number of integration points
    const int nb_integration_points = getGaussPts().size2();
    // get integration weights
    auto t_w = getFTensor0IntegrationWeight();
    // get gradient of the field at integration points
    auto t_field_grad = getFTensor1FromMat<2>(*fieldGradMat);

    // get base functions
    auto t_base = data.getFTensor0N();
    // get derivatives of base functions
    auto t_diff_base = data.getFTensor1DiffN<2>();

    // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL VECTOR
    for (int gg = 0; gg != nb_integration_points; gg++) {

      const double a = t_w * area;
      const double an = 1. / std::sqrt(1 + t_field_grad(i) * t_field_grad(i));

      for (int rr = 0; rr != AssemblyDomainEleOp::nbRows; rr++) {

        // calculate components of the local vector
        // remember to use -= here due to PETSc consideration of Residual Vec
        nf[rr] += (t_diff_base(i) * t_field_grad(i)) * an * a;

        // move to the next base function
        ++t_base;
        // move to the derivatives of the next base function
        ++t_diff_base;
      }

      // move to the weight of the next integration point
      ++t_w;
      // move to the gradient of the field at the next integration point
      ++t_field_grad;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> fieldGradMat;
};

struct MinimalSurfaceEqn {
public:
  MinimalSurfaceEqn(MoFEM::Interface &m_field);

  // Declaration of the main function to run analysis
  MoFEMErrorCode runProgram();

private:
  // Declaration of other main functions called in runProgram()
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();

  // Function to calculate the Boundary term
  static double boundaryFunction(const double x, const double y,
                                 const double z) {
    return sin(2 * M_PI * (x + y));
  }

  // Main interfaces
  MoFEM::Interface &mField;

  // Object to mark boundary entities for the assembling of domain elements
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
  Range boundaryEnts;
};

MinimalSurfaceEqn::MinimalSurfaceEqn(MoFEM::Interface &m_field)
    : mField(m_field) {}

MoFEMErrorCode MinimalSurfaceEqn::runProgram() {
  MoFEMFunctionBegin;

  readMesh();
  setupProblem();
  setIntegrationRules();
  boundaryCondition();
  assembleSystem();
  solveSystem();
  outputResults();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MinimalSurfaceEqn::readMesh() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();
  CHKERR mField.getInterface(simple);
  CHKERR simple->getOptions();
  CHKERR simple->loadFile();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MinimalSurfaceEqn::setupProblem() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();
  CHKERR simple->addDomainField("U", H1, AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);
  CHKERR simple->addBoundaryField("U", H1, AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);

  int order = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setUp();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MinimalSurfaceEqn::setIntegrationRules() {
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

MoFEMErrorCode MinimalSurfaceEqn::boundaryCondition() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();

  auto get_ents_on_mesh_skin = [&]() {
    Range body_ents;
    CHKERR mField.get_moab().get_entities_by_dimension(0, 2, body_ents);
    Skinner skin(&mField.get_moab());
    Range skin_ents;
    CHKERR skin.find_skin(0, body_ents, false, skin_ents);
    // filter not owned entities, those are not on boundary
    Range boundary_ents;
    ParallelComm *pcomm =
        ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
    pcomm->filter_pstatus(skin_ents, PSTATUS_SHARED | PSTATUS_MULTISHARED,
                          PSTATUS_NOT, -1, &boundary_ents);

    Range skin_verts;
    mField.get_moab().get_connectivity(boundary_ents, skin_verts, true);
    boundary_ents.merge(skin_verts);
    boundaryEnts = boundary_ents;
    return boundary_ents;
  };

  auto mark_boundary_dofs = [&](Range &&skin_edges) {
    auto problem_manager = mField.getInterface<ProblemsManager>();
    auto marker_ptr = boost::make_shared<std::vector<unsigned char>>();
    problem_manager->markDofs(simple->getProblemName(), ROW,
                              ProblemsManager::OR, skin_edges, *marker_ptr);
    return marker_ptr;
  };

  // Get global local vector of marked DOFs. Is global, since is set for all
  // DOFs on processor. Is local since only DOFs on processor are in the
  // vector. To access DOFs use local indices.
  boundaryMarker = mark_boundary_dofs(get_ents_on_mesh_skin());

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MinimalSurfaceEqn::assembleSystem() {
  MoFEMFunctionBegin;
  auto add_domain_base_ops = [&](auto &pipeline) {
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
    pipeline.push_back(new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline.push_back(new OpSetInvJacH1ForFace(inv_jac_ptr));
    pipeline.push_back(new OpSetHOWeightsOnFace());
  };

  auto add_domain_lhs_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));
    auto grad_u_at_gauss_pts = boost::make_shared<MatrixDouble>();
    pipeline.push_back(
        new OpCalculateScalarFieldGradient<2>("U", grad_u_at_gauss_pts));
    pipeline.push_back(
        new OpDomainTangentMatrix("U", "U", grad_u_at_gauss_pts));
    pipeline.push_back(new OpUnSetBc("U"));
  };

  auto add_domain_rhs_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpSetBc("U", true, boundaryMarker));
    auto grad_u_at_gauss_pts = boost::make_shared<MatrixDouble>();
    pipeline.push_back(
        new OpCalculateScalarFieldGradient<2>("U", grad_u_at_gauss_pts));
    pipeline.push_back(new OpDomainResidualVector("U", grad_u_at_gauss_pts));
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
    pipeline.push_back(new OpCalculateScalarFieldValues("U", u_at_gauss_pts));
    pipeline.push_back(new OpBoundaryTimeScalarField(
        "U", u_at_gauss_pts,
        [](const double, const double, const double) { return 1; }));
    pipeline.push_back(new OpBoundarySource("U", boundaryFunction));
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

MoFEMErrorCode MinimalSurfaceEqn::solveSystem() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();

  auto set_fieldsplit_preconditioner = [&](auto snes) {
    MoFEMFunctionBeginHot;
    KSP ksp;
    CHKERR SNESGetKSP(snes, &ksp);
    PC pc;
    CHKERR KSPGetPC(ksp, &pc);
    PetscBool is_pcfs = PETSC_FALSE;
    PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &is_pcfs);

    if (is_pcfs == PETSC_TRUE) {
      auto bc_mng = mField.getInterface<BcManager>();
      auto name_prb = simple->getProblemName();
      SmartPetscObj<IS> is_all_bc;
      CHKERR mField.getInterface<ISManager>()->isCreateProblemFieldAndRank(
          name_prb, ROW, "U", 0, 1, is_all_bc, &boundaryEnts);
      int is_all_bc_size;
      CHKERR ISGetSize(is_all_bc, &is_all_bc_size);
      MOFEM_LOG("EXAMPLE", Sev::inform)
          << "Field split block size " << is_all_bc_size;
      CHKERR PCFieldSplitSetIS(pc, PETSC_NULL,
                               is_all_bc); // boundary block
    }
    MoFEMFunctionReturnHot(0);
  };

  // Create global RHS and solution vectors
  auto dm = simple->getDM();
  SmartPetscObj<Vec> global_rhs, global_solution;
  CHKERR DMCreateGlobalVector_MoFEM(dm, global_rhs);
  global_solution = smartVectorDuplicate(global_rhs);

  // Create nonlinear solver (SNES)
  auto pipeline_mng = mField.getInterface<PipelineManager>();
  auto solver = pipeline_mng->createSNES();
  CHKERR SNESSetFromOptions(solver);
  CHKERR set_fieldsplit_preconditioner(solver);
  CHKERR SNESSetUp(solver);

  // Solve the system
  CHKERR SNESSolve(solver, global_rhs, global_solution);

  // Scatter result data on the mesh
  CHKERR DMoFEMMeshToGlobalVector(dm, global_solution, INSERT_VALUES,
                                  SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MinimalSurfaceEqn::outputResults() {
  MoFEMFunctionBegin;

  auto post_proc = boost::make_shared<PostProcEle>(mField);
  CHKERR post_proc->generateReferenceElementMesh();
  CHKERR post_proc->addFieldValuesPostProc("U");

  auto *simple = mField.getInterface<Simple>();
  auto dm = simple->getDM();
  CHKERR DMoFEMLoopFiniteElements(dm, simple->getDomainFEName(), post_proc);

  CHKERR post_proc->writeFile("out_result.h5m");

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
    MinimalSurfaceEqn minimal_surface_problem(m_field);
    CHKERR minimal_surface_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}