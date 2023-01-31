/**
 * @file poisson_3d_homogeneous.cpp
 * @example poisson_3d_homogeneous.cpp
 * @brief Poisson problem 3D 
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <BasicFiniteElements.hpp>
#include <poisson_3d_homogeneous.hpp>

using namespace MoFEM;
using namespace Poisson3DHomogeneousOperators;

using PostProcVolEle =
    PostProcBrokenMeshInMoab<VolumeElementForcesAndSourcesCore>;

static char help[] = "...\n\n";

struct Poisson3DHomogeneous {
public:
  Poisson3DHomogeneous(MoFEM::Interface &m_field);

  // Declaration of the main function to run analysis
  MoFEMErrorCode runProgram();

private:
  // Declaration of other main functions called in runProgram()
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();

  // MoFEM interfaces
  MoFEM::Interface &mField;
  Simple *simpleInterface;

  // Field name and approximation order
  std::string domainField;
  int oRder;

};

Poisson3DHomogeneous::Poisson3DHomogeneous(MoFEM::Interface &m_field)
    : domainField("U"), mField(m_field) {}

MoFEMErrorCode Poisson3DHomogeneous::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson3DHomogeneous::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR simpleInterface->addDomainField(domainField, H1,
                                         AINSWORTH_BERNSTEIN_BEZIER_BASE, 1);

  int oRder = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &oRder, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder(domainField, oRder);

  CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson3DHomogeneous::boundaryCondition() {
  MoFEMFunctionBegin;

  auto bc_mng = mField.getInterface<BcManager>();

  CHKERR bc_mng->removeBlockDOFsOnEntities<BcScalarMeshsetType<BLOCKSET>>(
      simpleInterface->getProblemName(), "BOUNDARY_CONDITION", domainField,
      true);
  CHKERR bc_mng->removeBlockDOFsOnEntities<TemperatureCubitBcData>(
      simpleInterface->getProblemName(), domainField, true, false, true);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson3DHomogeneous::assembleSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  { // Push operators to the Pipeline that is responsible for calculating LHS
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpDomainLhsMatrixK(domainField, domainField));
  }

  { // Push operators to the Pipeline that is responsible for calculating LHS
    auto get_bc_hook = [&]() {
      EssentialPreProc<TemperatureCubitBcData> hook(
          mField, pipeline_mng->getDomainRhsFE(), {});
      return hook;
    };
    pipeline_mng->getDomainRhsFE()->preProcessHook = get_bc_hook();

    using DomainEle = PipelineManager::VolEle;
    using DomainEleOp = DomainEle::UserDataOperator;
    using OpInternal = FormsIntegrators<DomainEleOp>::Assembly<
        PETSC>::LinearForm<GAUSS>::OpBaseTimesScalar<1>;

    auto u_vals_ptr = boost::make_shared<VectorDouble>();
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpCalculateScalarFieldValues(domainField, u_vals_ptr));
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpInternal(domainField, u_vals_ptr,
                       [](double, double, double) constexpr { return -1; }));

    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpDomainRhsVectorF(domainField));
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson3DHomogeneous::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto rule_lhs = [](int, int, int p) -> int { return 2 * (p - 1); };
  auto rule_rhs = [](int, int, int p) -> int { return p; };

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(rule_lhs);
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule_rhs);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson3DHomogeneous::solveSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto ksp_solver = pipeline_mng->createKSP();
  CHKERR KSPSetFromOptions(ksp_solver);
  CHKERR KSPSetUp(ksp_solver);

  // Create RHS and solution vectors
  auto dm = simpleInterface->getDM();
  auto F = smartCreateDMVector(dm);
  auto D = smartVectorDuplicate(F);

  // Solve the system
  CHKERR KSPSolve(ksp_solver, F, D);

  // Scatter result data on the mesh
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson3DHomogeneous::outputResults() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();

  auto post_proc_fe = boost::make_shared<PostProcVolEle>(mField);

  auto u_ptr = boost::make_shared<VectorDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(domainField, u_ptr));

  using OpPPMap = OpPostProcMapInMoab<3, 3>;

  post_proc_fe->getOpPtrVector().push_back(

      new OpPPMap(

          post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

          {{domainField, u_ptr}},

          {},

          {},

          {})

  );

  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_result.h5m");

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Poisson3DHomogeneous::runProgram() {
  MoFEMFunctionBegin;

  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();
  CHKERR setIntegrationRules();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR outputResults();

  MoFEMFunctionReturn(0);
}

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

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
    Poisson3DHomogeneous poisson_problem(m_field);
    CHKERR poisson_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}