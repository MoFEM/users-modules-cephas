#include <BasicFiniteElements.hpp>
#include <MoFEM.hpp>
using namespace MoFEM;

static char help[] = "...\n\n";

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = PipelineManager::FaceEle;
  using BoundaryEle = PipelineManager::EdgeEle;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = VolumeElementForcesAndSourcesCore;
  using BoundaryEle = PipelineManager::FaceEle;
};

constexpr int SPACE_DIM = 3;
constexpr int BASE_DIM = 1;
constexpr AssemblyType A = AssemblyType::PETSC; //< selected assembly type
constexpr IntegrationType I =
    IntegrationType::GAUSS;                     //< selected integration type
constexpr CoordinateTypes coord_type = CARTESIAN;

using EntData = EntitiesFieldData::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

using AssemblyDomainEleOp = FormsIntegrators<DomainEleOp>::Assembly<A>::OpBase;
using AssemblyBoundaryEleOp =
    FormsIntegrators<BoundaryEleOp>::Assembly<A>::OpBase;

// new operators
using OpBoundarySourceRhs = FormsIntegrators<BoundaryEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<3, 3>;

int order = 2; ///< Order
// material parameters
double mu;      ///< magnetic constant  N / A2
double epsilon; ///< regularization paramater

struct Magnetostatics {
public:
  Magnetostatics(MoFEM::Interface &m_field);

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

  // boundary condition ranges
  Range naturalBcRange;
};

Magnetostatics::Magnetostatics(MoFEM::Interface &m_field) : mField(m_field) {}

//! [Read mesh]
MoFEMErrorCode Magnetostatics::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Setup problem]
MoFEMErrorCode Magnetostatics::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR simpleInterface->addDomainField("MAGNETIC_POTENTIAL", HCURL,
                                         DEMKOWICZ_JACOBI_BASE, 1);
  // CHKERR simpleInterface->addDomainField("MESH_NODE_POSITIONS", H1,
  //                                        AINSWORTH_LEGENDRE_BASE, 3);

  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder("MAGNETIC_POTENTIAL", order);

  CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}
//! [Setup problem]

//! [Boundary condition]
MoFEMErrorCode Magnetostatics::boundaryCondition() {
  MoFEMFunctionBegin;

  auto natural_bc = [&]() {
    Range boundary_entities;
    auto bc_mng = mField.getInterface<BcManager>();
    CHKERR bc_mng->pushMarkDOFsOnEntities(simpleInterface->getProblemName(),
                                          "MAGNETIC_POTENTIAL", "MAGNETIC_POTENTIAL", 0, 0);

    auto &bc_map = bc_mng->getBcMapByBlockName();
    for (auto b : bc_map) {
      if (std::regex_match(b.first, std::regex("(.*)NATURAL(.*)"))) {
        boundary_entities.merge(*(b.second->getBcEntsPtr()));
      }
    }
    // Add vertices to boundary entities
    Range boundary_vertices;
    CHKERR mField.get_moab().get_connectivity(boundary_entities,
                                              boundary_vertices, true);
    boundary_entities.merge(boundary_vertices);

    return boundary_entities;
  };

  naturalBcRange = natural_bc();

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Assemble system]
MoFEMErrorCode Magnetostatics::assembleSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  // Push operators to the Pipeline that is responsible for calculating LHS
  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pipeline_mng->getOpDomainLhsPipeline(), {HCURL});
  // pipeline_mng->getOpDomainLhsPipeline().push_back(
  //     new OpDomainLhsMatrixK(field_name, field_name));

  // Push operators to the Pipeline that is responsible for calculating RHS

  // auto set_values_to_bc_dofs = [&](auto &fe) {
  //   auto get_bc_hook = [&]() {
  //     EssentialPreProc<TemperatureCubitBcData> hook(mField, fe, {});
  //     return hook;
  //   };
  //   fe->preProcessHook = get_bc_hook();
  // };

  // auto calculate_residual_from_set_values_on_bc = [&](auto &pipeline) {
  //   using DomainEle =
  //       PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::DomainEle;
  //   using DomainEleOp = DomainEle::UserDataOperator;
  //   using OpInternal = FormsIntegrators<DomainEleOp>::Assembly<
  //       PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, 1, SPACE_DIM>;

  //   auto grad_u_vals_ptr = boost::make_shared<MatrixDouble>();
  //   pipeline_mng->getOpDomainRhsPipeline().push_back(
  //       new OpCalculateScalarFieldGradient<SPACE_DIM>(field_name,
  //                                                     grad_u_vals_ptr));
  //   pipeline_mng->getOpDomainRhsPipeline().push_back(
  //       new OpInternal(field_name, grad_u_vals_ptr,
  //                      [](double, double, double) constexpr { return -1;
  //                      }));
  // };

  CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
      pipeline_mng->getOpDomainRhsPipeline(), {HCURL});
  // set_values_to_bc_dofs(pipeline_mng->getDomainRhsFE());
  // calculate_residual_from_set_values_on_bc(
  //     pipeline_mng->getOpDomainRhsPipeline());
  // pipeline_mng->getOpDomainRhsPipeline().push_back(
  //     new OpDomainRhsVectorF(field_name));

  boost::function<FTensor::Tensor1<double, 3>(const double, const double,
                                                const double)>
      natural_fun = [](double x, double y, double z) {
        const double r = sqrt(x * x + y * y);
        return FTensor::Tensor1<double, 3>{(-y / r), (x / r), 0.};
      };

  // boundary Rhs
  pipeline_mng->getOpBoundaryRhsPipeline().push_back(
      new OpBoundarySourceRhs("MAGNETIC_POTENTIAL", natural_fun,
                              boost::make_shared<Range>(naturalBcRange)));

  MoFEMFunctionReturn(0);
}
  //! [Assemble system]

  //! [Set integration rules]
  MoFEMErrorCode
  Magnetostatics::setIntegrationRules() {
    MoFEMFunctionBegin;

    auto rule_lhs = [](int, int, int p) -> int { return 2 * p + 1; };
    auto rule_rhs = [](int, int, int p) -> int { return 2 * p + 1; };

    auto pipeline_mng = mField.getInterface<PipelineManager>();
    CHKERR pipeline_mng->setDomainLhsIntegrationRule(rule_lhs);
    CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule_rhs);

    MoFEMFunctionReturn(0);
  }
//! [Set integration rules]

//! [Solve system]
MoFEMErrorCode Magnetostatics::solveSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto ksp_solver = pipeline_mng->createKSP();
  CHKERR KSPSetFromOptions(ksp_solver);
  CHKERR KSPSetUp(ksp_solver);

  // Create RHS and solution vectors
  auto dm = simpleInterface->getDM();
  auto F = createDMVector(dm);
  auto D = vectorDuplicate(F);

  // Solve the system
  CHKERR KSPSolve(ksp_solver, F, D);

  // Scatter result data on the mesh
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  MoFEMFunctionReturn(0);
}
//! [Solve system]

//! [Output results]
MoFEMErrorCode Magnetostatics::outputResults() {
  MoFEMFunctionBegin;

  // PostProcBrokenMeshInMoab<VolumeElementForcesAndSourcesCore>
  // post_proc(mField);

  // CHKERR addHOOpsVol("MESH_NODE_POSITIONS", post_proc, false, true, false,
  //                    true);

  // auto pos_ptr = boost::make_shared<MatrixDouble>();
  // auto field_val_ptr = boost::make_shared<MatrixDouble>();

  // post_proc.getOpPtrVector().push_back(
  //     new OpCalculateVectorFieldValues<SPACE_DIM>("MESH_NODE_POSITIONS",
  //                                                 pos_ptr));
  // post_proc.getOpPtrVector().push_back(
  //     new OpCalculateHVecVectorField<SPACE_DIM>("MAGNETIC_POTENTIAL",
  //                                               field_val_ptr));

  // using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

  // post_proc.getOpPtrVector().push_back(

  //     new OpPPMap(

  //         post_proc.getPostProcMesh(), post_proc.getMapGaussPts(),

  //         {},

  //         {{"MESH_NODE_POSITIONS", pos_ptr},
  //          {"MAGNETIC_POTENTIAL", field_val_ptr}},

  //         {},

  //         {}

  //         )

  // );

  // post_proc.getOpPtrVector().push_back(new OpPostProcessCurl(
  //     blockData, post_proc.getPostProcMesh(), post_proc.getMapGaussPts()));
  // CHKERR DMoFEMLoopFiniteElements(blockData.dM, blockData.feName.c_str(),
  //                                 &post_proc);
  // CHKERR post_proc.writeFile("out_values.h5m");

  MoFEMFunctionReturn(0);
}
//! [Output results]

//! [Run program]
MoFEMErrorCode Magnetostatics::runProgram() {
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
//! [Run program]

//! [Main]
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
    Magnetostatics ex(m_field);
    CHKERR ex.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}
//! [Main]