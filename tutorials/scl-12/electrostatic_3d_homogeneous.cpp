/**
 * @file electrostatic_3d_homogeneous.cpp
 * @example electrostatic_3d_homogeneous.cpp
 * @brief Poisson problem 3D
 *
 * @copyright Copyright (c) 2023
 *
 */

constexpr auto domainField = "U";
#include <BasicFiniteElements.hpp>
#include <PoissonOperators.hpp>
#include <electrostatic_3d_homogeneous.hpp>


using namespace MoFEM;
using namespace Electrostatic3DHomogeneousOperators;

using PostProcVolEle =
    PostProcBrokenMeshInMoab<VolumeElementForcesAndSourcesCore>;

static char help[] = "...\n\n";

struct Electrostatic3DHomogeneous {
public:
  Electrostatic3DHomogeneous(MoFEM::Interface &m_field);

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
  boost::shared_ptr<std::map<int, VolBlockData<SPACE_DIM>>> vol_block_sets_ptr; ///////
  boost::shared_ptr<std::map<int, SurfBlockData<SPACE_DIM>>> surf_block_sets_ptr;
  Simple *simpleInterface;
  boost::shared_ptr<ForcesAndSourcesCore> interface_rhs_fe;
  boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> common_data_ptr;

  // Field name and approximation order
  std::string domainField;
  int oRder;
};

Electrostatic3DHomogeneous::Electrostatic3DHomogeneous(
    MoFEM::Interface &m_field)
    : domainField("U"), mField(m_field) {}

//! [Read mesh]
MoFEMErrorCode Electrostatic3DHomogeneous::readMesh() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();

  MoFEMFunctionReturn(0);
}

//! [Setup problem]
MoFEMErrorCode Electrostatic3DHomogeneous::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR simpleInterface->addDomainField(domainField, H1,
                                         AINSWORTH_LEGENDRE_BASE, 1);

  int oRder = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &oRder, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder(domainField, oRder);

  CHKERR simpleInterface->setUp();
  // Range interface_ents;

  // for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {

  //   if (bit->getName().compare(0, 9, "INTERFACE") == 0) {
  //     const int id = bit->getMeshsetId();
  //     cout << bit->getMeshsetId() << endl;
  //     Range ents;
  //     CHKERR mField.get_moab().get_entities_by_dimension(bit->getMeshset(), 2,
  //                                                        ents, true);
  //     interface_ents.merge(ents);
  //   }
  // }


  /////
  vol_block_sets_ptr = boost::make_shared<std::map<int, VolBlockData<SPACE_DIM>>>();
  Range electric_tets;
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
    if (bit->getName().compare(0, 12, "MAT_ELECTRIC") == 0) {
      const int id = bit->getMeshsetId();
      auto &block_data = (*vol_block_sets_ptr)[id];

      CHKERR mField.get_moab().get_entities_by_dimension(bit->getMeshset(), SPACE_DIM,
                                                         block_data.tEts, true);
      electric_tets.merge(block_data.tEts);

      std::vector<double> attributes;
      bit->getAttributes(attributes);
      if (attributes.size() < 1) {
        SETERRQ1(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                 "should be at least 1 attributes but is %d",
                 attributes.size());
      }
      block_data.iD = id;
      block_data.epsPermit = attributes[0];
    }
  }

  surf_block_sets_ptr = boost::make_shared<std::map<int, SurfBlockData<SPACE_DIM>>>();
  Range interface_tris;
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
    if (bit->getName().compare(0, 12, "INT_ELECTRIC") == 0) {
      const int id = bit->getMeshsetId();
      auto &block_data = (*surf_block_sets_ptr)[id];

      CHKERR mField.get_moab().get_entities_by_dimension(bit->getMeshset(), SPACE_DIM -1,
                                                         block_data.tRis, true);
      interface_tris.merge(block_data.tRis);

      std::vector<double> attributes;
      bit->getAttributes(attributes);
      if (attributes.size() < 1) {
        SETERRQ1(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                 "should be at least 1 attributes but is %d",
                 attributes.size());
      }

      block_data.iD = id;
      block_data.sigma = attributes[0];
    }
  }

  CHKERR mField.add_finite_element("INTERFACE");
  CHKERR mField.modify_finite_element_add_field_row("INTERFACE", domainField);
  CHKERR mField.modify_finite_element_add_field_col("INTERFACE", domainField);
  CHKERR mField.modify_finite_element_add_field_data("INTERFACE", domainField);
  CHKERR mField.add_ents_to_finite_element_by_dim(interface_tris, MBTRI,
                                                  "INTERFACE");

  CHKERR simpleInterface->defineFiniteElements();
  CHKERR simpleInterface->defineProblem(PETSC_TRUE);
  CHKERR simpleInterface->buildFields();
  CHKERR simpleInterface->buildFiniteElements();
  CHKERR simpleInterface->buildProblem();


  CHKERR mField.build_finite_elements("INTERFACE");
  CHKERR DMMoFEMAddElement(simpleInterface->getDM(), "INTERFACE");
  CHKERR simpleInterface->buildProblem();
  ///
  // CHKERR
  // mField.getInterface<CommInterface>()->synchroniseEntities(interface_tris);
  CHKERR simpleInterface->buildProblem();

  // CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]
MoFEMErrorCode Electrostatic3DHomogeneous::boundaryCondition() {
  MoFEMFunctionBegin;

  auto bc_mng = mField.getInterface<BcManager>();

  // Remove BCs from blockset name "BOUNDARY_CONDITION" or SETU, note that you
  // can use regular expression to put list of blocksets;
  CHKERR bc_mng->removeBlockDOFsOnEntities<BcScalarMeshsetType<BLOCKSET>>(
      simpleInterface->getProblemName(), "(BOUNDARY_CONDITION|SETU)",
      domainField, true);

  // Remove BCs from cubit TEMPERATURESET, i.e. set by cubit, and meshsets named
  // FIX_SCALAR (default name to name boundary conditions for scalar fields)
  CHKERR bc_mng->removeBlockDOFsOnEntities<TemperatureCubitBcData>(
      simpleInterface->getProblemName(), domainField, true, false, true);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Electrostatic3DHomogeneous::assembleSystem() {
  MoFEMFunctionBegin;
  auto pipeline_mng = mField.getInterface<PipelineManager>();
  common_data_ptr = boost::make_shared<DataAtIntegrationPts<SPACE_DIM>>(mField);

  auto add_domain_lhs_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpBlockPermittivity(
        common_data_ptr, vol_block_sets_ptr, domainField));
  };

  add_domain_lhs_ops(pipeline_mng->getOpDomainLhsPipeline());


  { // Push operators to the Pipeline that is responsible for calculating LHS
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pipeline_mng->getOpDomainLhsPipeline(), {H1});
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpDomainLhsMatrixK(domainField, domainField, common_data_ptr));
  }

  { // Push operators to the Pipeline that is responsible for calculating LHS
    auto set_values_to_bc_dofs = [&](auto &fe) {
      auto get_bc_hook = [&]() {
        EssentialPreProc<TemperatureCubitBcData> hook(mField, fe, {});
        return hook;
      };
      fe->preProcessHook = get_bc_hook();
    };

    auto calculate_residual_from_set_values_on_bc = [&](auto &pipeline) {
      using DomainEle =PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::DomainEle;
      using DomainEleOp = DomainEle::UserDataOperator;
      using OpInternal = FormsIntegrators<DomainEleOp>::Assembly<
          PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<BASE_DIM, FIELD_DIM, SPACE_DIM>;

      auto grad_u_vals_ptr = boost::make_shared<MatrixDouble>();
      pipeline.push_back(
          new OpCalculateScalarFieldGradient<SPACE_DIM>(domainField,
                                                        grad_u_vals_ptr));
      pipeline.push_back(
          new OpInternal(domainField, grad_u_vals_ptr,
                         [](double, double, double) constexpr { return -1; }));
    };

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        pipeline_mng->getOpDomainRhsPipeline(), {H1});
    set_values_to_bc_dofs(pipeline_mng->getDomainRhsFE());
    calculate_residual_from_set_values_on_bc(
        pipeline_mng->getOpDomainRhsPipeline());

    interface_rhs_fe = boost::shared_ptr<ForcesAndSourcesCore>(
        new FaceElementForcesAndSourcesCore(mField));

    {
      interface_rhs_fe->getOpPtrVector().push_back(new OpBlockChargeDensity(
          common_data_ptr, surf_block_sets_ptr, domainField));

      interface_rhs_fe->getOpPtrVector().push_back(
          new OpInterfaceRhsVectorF(domainField, common_data_ptr));

    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Electrostatic3DHomogeneous::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto rule_lhs = [](int, int, int p) -> int { return 2 * (p - 1); };
  auto rule_rhs = [](int, int, int p) -> int { return p; };

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(rule_lhs);
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule_rhs);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Electrostatic3DHomogeneous::solveSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto ksp_solver = pipeline_mng->createKSP();

  boost::shared_ptr<ForcesAndSourcesCore> null; ///< Null element does nothing
  CHKERR DMMoFEMKSPSetComputeRHS(simpleInterface->getDM(), "INTERFACE",
                                 interface_rhs_fe, null, null);

  pipeline_mng->getDomainLhsFE().reset();

  auto post_proc_fe = boost::make_shared<PostProcVolEle>(mField); ///

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

MoFEMErrorCode Electrostatic3DHomogeneous::outputResults() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();

  auto post_proc_fe = boost::make_shared<PostProcVolEle>(mField);

  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

  // constexpr auto SPACE_DIM = 3; // dimension of problem

  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateHOJac<SPACE_DIM>(jac_ptr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpSetHOInvJacToScalarBases<SPACE_DIM>(H1, inv_jac_ptr));

  auto u_ptr = boost::make_shared<VectorDouble>();
  auto grad_u_ptr = boost::make_shared<MatrixDouble>();

  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(domainField, u_ptr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateScalarFieldGradient<SPACE_DIM>(domainField, grad_u_ptr));
  using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;


  auto neg_grad_u_ptr = boost::make_shared<MatrixDouble>();
  post_proc_fe->getOpPtrVector().push_back(
      new OpNegativeGradient<SPACE_DIM>(neg_grad_u_ptr, grad_u_ptr));


  post_proc_fe->getOpPtrVector().push_back(

      new OpPPMap(post_proc_fe->getPostProcMesh(),
                  post_proc_fe->getMapGaussPts(),

                  OpPPMap::DataMapVec{{"U", u_ptr}},

                  OpPPMap::DataMapMat{{"ELECTRIC FILED", neg_grad_u_ptr}},

                  OpPPMap::DataMapMat{},

                  OpPPMap::DataMapMat{}

                  )

  );

  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_result3D.h5m");

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Electrostatic3DHomogeneous::runProgram() {
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
    Electrostatic3DHomogeneous poisson_problem(m_field);
    CHKERR poisson_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}