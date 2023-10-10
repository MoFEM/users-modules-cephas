/**
 * @file Electrostatics.cpp
 * \example Electrostatics.cpp
 *  */
// #ifndef __ELECTROSTATICS_CPP__
// #define __ELECTROSTATICS_CPP__

#include <electrostatics.hpp>
// #define ELogTag                                                                \
//   MOFEM_LOG_CHANNEL("WORLD");                                                  \
//   MOFEM_LOG_FUNCTION();                                                        \
//   MOFEM_LOG_TAG("WORLD", "Electrostatics");

struct ElectrostaticHomogeneous {
public:
  ElectrostaticHomogeneous(MoFEM::Interface &m_field);

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
  MoFEMErrorCode getAlphaPart();
  MoFEMErrorCode outputResults();

  // MoFEM interfaces
  MoFEM::Interface &mField;

  boost::shared_ptr<std::map<int, BlockData>> perm_block_sets_ptr;
  boost::shared_ptr<std::map<int, BlockData>> int_block_sets_ptr; ///////
  Simple *simpleInterface;
  boost::shared_ptr<ForcesAndSourcesCore> interface_rhs_fe;
  boost::shared_ptr<DataAtIntegrationPts> common_data_ptr;

  std::string domainField;
  int oRder;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
  double alphaPart = 0.0;
  double ALPHA = 0.0;
  std::vector<moab::Range> FloatingblockconstBC;

  SmartPetscObj<Vec> petscVec;
  enum VecElements { ZERO = 0, LAST_ELEMENT };
};

ElectrostaticHomogeneous::ElectrostaticHomogeneous(MoFEM::Interface &m_field)
    : domainField("POTENTIAL"), mField(m_field) {}

//! [Read mesh]
MoFEMErrorCode ElectrostaticHomogeneous::readMesh() {
  MoFEMFunctionBegin;
  CHKERR mField.getInterface(simpleInterface);
  CHKERR simpleInterface->getOptions();
  CHKERR simpleInterface->loadFile();
  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Setup problem]
MoFEMErrorCode ElectrostaticHomogeneous::setupProblem() {
  MoFEMFunctionBegin;

  CHKERR simpleInterface->addDomainField(domainField, H1,
                                         AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simpleInterface->addBoundaryField(domainField, H1,
                                           AINSWORTH_LEGENDRE_BASE, 1);

  int oRder = 2;

  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &oRder, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder(domainField, oRder);

  common_data_ptr = boost::make_shared<DataAtIntegrationPts>(mField);

  perm_block_sets_ptr = boost::make_shared<std::map<int, BlockData>>();
  Range electrIcs;
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
    if (bit->getName().compare(0, 12, "MAT_ELECTRIC") == 0) {
      const int id = bit->getMeshsetId();
      auto &block_data = (*perm_block_sets_ptr)[id];

      CHKERR mField.get_moab().get_entities_by_dimension(
          bit->getMeshset(), SPACE_DIM, block_data.blockDomains, true);
      electrIcs.merge(block_data.blockDomains);

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
  int_block_sets_ptr = boost::make_shared<std::map<int, BlockData>>();
  Range interfIcs;
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
    if (bit->getName().compare(0, 12, "INT_ELECTRIC") == 0) {
      const int id = bit->getMeshsetId();
      auto &block_data = (*int_block_sets_ptr)[id];

      CHKERR mField.get_moab().get_entities_by_dimension(
          bit->getMeshset(), SPACE_DIM - 1, block_data.blockInterfaces, true);
      interfIcs.merge(block_data.blockInterfaces);

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
  CHKERR mField.add_ents_to_finite_element_by_dim(electrIcs, SPACE_DIM,
                                                  "INTERFACE");
  CHKERR mField.add_ents_to_finite_element_by_dim(interfIcs, SPACE_DIM - 1,
                                                  "INTERFACE");

  CHKERR simpleInterface->defineFiniteElements();
  CHKERR simpleInterface->defineProblem(PETSC_TRUE);
  CHKERR simpleInterface->buildFields();
  CHKERR simpleInterface->buildFiniteElements();
  CHKERR simpleInterface->buildProblem();

  CHKERR mField.build_finite_elements("INTERFACE");
  CHKERR DMMoFEMAddElement(simpleInterface->getDM(), "INTERFACE");

  CHKERR simpleInterface->buildProblem();

  // initialise petsc vector for required processor
  int local_size;
  if (mField.get_comm_rank() == 0) // get_comm_rank() gets processor number

    local_size = LAST_ELEMENT; // last element gives size of vector

  else
    // other processors (e.g. 1, 2, 3, etc.)
    local_size = 0; // local size of vector is zero on other processors

  petscVec = createSmartVectorMPI(mField.get_comm(), local_size, LAST_ELEMENT);

  MoFEMFunctionReturn(0);
}
//! [Setup problem]

//! [Boundary condition]
MoFEMErrorCode ElectrostaticHomogeneous::boundaryCondition() {
  MoFEMFunctionBegin;

  auto bc_mng = mField.getInterface<BcManager>();

  // Remove_BCs_from_blockset name "BOUNDARY_CONDITION";
  CHKERR bc_mng->removeBlockDOFsOnEntities<BcScalarMeshsetType<BLOCKSET>>(
      simpleInterface->getProblemName(), "BOUNDARY_CONDITION",
      std::string(domainField), true);

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Assemble system]
MoFEMErrorCode ElectrostaticHomogeneous::assembleSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getBoundaryLhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();
  // auto side_proc_fe = boost::make_shared<SideEle>(mField);

  common_data_ptr = boost::make_shared<DataAtIntegrationPts>(mField);
  // auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto add_domain_lhs_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpBlockPermittivity(
        common_data_ptr, perm_block_sets_ptr, domainField));
  };

  add_domain_lhs_ops(pipeline_mng->getOpDomainLhsPipeline());
  auto epsilon = [&](const double, const double, const double) {
    return common_data_ptr->blockPermittivity;
  };
  { // Push operators to the Pipeline that is responsible for calculating LHS
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        pipeline_mng->getOpDomainLhsPipeline(), {H1});
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpDomainLhsMatrixK(domainField, domainField, epsilon));
  }

  { // Push operators to the Pipeline that is responsible for
    // calculating
    // LHS
    auto set_values_to_bc_dofs = [&](auto &fe) {
      auto get_bc_hook = [&]() {
        EssentialPreProc<TemperatureCubitBcData> hook(mField, fe, {});
        return hook;
      };
      fe->preProcessHook = get_bc_hook();
    };
    // Set essential BC

    // CHKERR set_bc_essential(domainField, "FLOATING_ELECTRODE");

    auto calculate_residual_from_set_values_on_bc = [&](auto &pipeline) {
      using OpInternal =
          FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
              GAUSS>::OpGradTimesTensor<BASE_DIM, FIELD_DIM, SPACE_DIM>;

      auto grad_u_vals_ptr = boost::make_shared<MatrixDouble>();
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateScalarFieldGradient<SPACE_DIM>(domainField,
                                                        grad_u_vals_ptr));
      add_domain_lhs_ops(pipeline_mng->getOpDomainRhsPipeline());
      auto minus_epsilon = [&](double, double, double) constexpr {
        return -common_data_ptr->blockPermittivity;
      };
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpInternal(domainField, grad_u_vals_ptr, minus_epsilon));
    };

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        pipeline_mng->getOpDomainRhsPipeline(), {H1});
    set_values_to_bc_dofs(pipeline_mng->getDomainRhsFE());
    calculate_residual_from_set_values_on_bc(
        pipeline_mng->getOpDomainRhsPipeline());

    interface_rhs_fe = boost::shared_ptr<ForcesAndSourcesCore>(
        new intElementForcesAndSourcesCore(mField));

    {
      interface_rhs_fe->getOpPtrVector().push_back(new OpBlockChargeDensity(
          common_data_ptr, int_block_sets_ptr, domainField));

      auto sIgma = [&](const double, const double, const double) {
        return common_data_ptr->chrgDens;
      };

      interface_rhs_fe->getOpPtrVector().push_back(
          new OpInterfaceRhsVectorF(domainField, sIgma));
    }
  }

  MoFEMFunctionReturn(0);
}
//! [Assemble system]

//! [Set integration rules]
MoFEMErrorCode ElectrostaticHomogeneous::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto rule_lhs = [](int, int, int p) -> int { return 2 * p + 1; };
  auto rule_rhs = [](int, int, int p) -> int { return p + 1; };

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(rule_lhs);
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule_rhs);

  MoFEMFunctionReturn(0);
}
//! [Set integration rules]

//! [Solve system]
MoFEMErrorCode ElectrostaticHomogeneous::solveSystem() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto ksp_solver = pipeline_mng->createKSP();

  boost::shared_ptr<ForcesAndSourcesCore> null; ///< Null element does
  CHKERR DMMoFEMKSPSetComputeRHS(simpleInterface->getDM(), "INTERFACE",
                                 interface_rhs_fe, null, null);

  auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
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
  auto bc_mng = mField.getInterface<BcManager>();

  MoFEMFunctionReturn(0);
}
//! [Solve system]
MoFEMErrorCode ElectrostaticHomogeneous::getAlphaPart() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getBoundaryLhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();

  auto get_entities_on_floating = [&]() {
    Range constBCEdges;
    std::list<std::string> electrodeNames;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      std::string entity_name = it->getName();

      if (entity_name.compare(0, 9, "ELECTRODE") == 0) {
        electrodeNames.push_back(entity_name);
        std::cout << entity_name << std::endl;
        CHKERR it->getMeshsetIdEntitiesByDimension(
            mField.get_moab(), SPACE_DIM - 1, constBCEdges, true);
      }
    }
    for (const auto &name : electrodeNames) {
      std::cout << "Electrode Name: " << name << std::endl;
    }

    std::cout << "Total Electrodes: " << electrodeNames.size() << std::endl;
    std::cout << "Hll" << constBCEdges.size() << std::endl;
    return constBCEdges;
  };
  auto get_entities_on_floatingssssssss = [&]() {
    std::vector<moab::Range> electrodeRanges;
    std::list<std::string> electrodeNames;

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      std::string entity_name = it->getName();

      if (entity_name.compare(0, 9, "ELECTRODE") == 0) {
        moab::Range constBCEdges;
        CHKERR it->getMeshsetIdEntitiesByDimension(
            mField.get_moab(), SPACE_DIM - 1, constBCEdges, true);

        electrodeNames.push_back(entity_name);
        electrodeRanges.push_back(constBCEdges);

        std::cout << "Electrode Name: " << entity_name << std::endl;
        std::cout << "Electrode Size: " << constBCEdges.size() << std::endl;
        std::cout << "Electrode Range: " << constBCEdges << std::endl;
      }
    }

    std::cout << "Total Electrodes: " << electrodeNames.size() << std::endl;

    // Print the electrode names and ranges

    return electrodeRanges;
  };

  FloatingblockconstBC =
      get_entities_on_floatingssssssss(); 
      /////////
                                          // for (const auto &electrodeRange :
                                          // electrodeRanges) {\\noooooooooooooo
                                          // FloatingblockconstBC =
                                          // electrodeRange;

  auto op_loop_side = new OpLoopSide<SideEle>(
      mField, simpleInterface->getDomainFEName(), SPACE_DIM);

  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
  op_loop_side->getOpPtrVector().push_back(
      new OpCalculateHOJac<SPACE_DIM>(jac_ptr));
  op_loop_side->getOpPtrVector().push_back(
      new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
  op_loop_side->getOpPtrVector().push_back(
      new OpSetHOInvJacToScalarBases<SPACE_DIM>(H1, inv_jac_ptr));

  auto grad_grad_ptr = boost::make_shared<MatrixDouble>();
  auto alpha_ptr = boost::make_shared<VectorDouble>();
  auto grad_projection_ptr = boost::make_shared<VectorDouble>();

  op_loop_side->getOpPtrVector().push_back(
      new OpCalculateScalarFieldGradient<SPACE_DIM>(domainField,
                                                    grad_grad_ptr));

  pipeline_mng->getOpBoundaryRhsPipeline().push_back(op_loop_side);
  // pipeline_mng->getOpBoundaryRhsPipeline().push_back(
  //     new OpAlpha<SPACE_DIM>(domainField, grad_grad_ptr, petscVec,
  //                            boost::make_shared<Range>(FloatingblockconstBC)));
  pipeline_mng->getOpBoundaryRhsPipeline().push_back(new OpAlpha<SPACE_DIM>(
      domainField, grad_grad_ptr, petscVec,
      boost::make_shared<std::vector<moab::Range>>(FloatingblockconstBC)));
  CHKERR VecZeroEntries(petscVec);
  CHKERR pipeline_mng->loopFiniteElements();

  CHKERR VecAssemblyBegin(petscVec);
  CHKERR VecAssemblyEnd(petscVec);
  if (!mField.get_comm_rank()) { // if
    const double *array;
    CHKERR VecGetArrayRead(petscVec, &array);

    ALPHA = array[ZERO];
    std::cout << std::setprecision(8) << "ALFA: " << ALPHA << std::endl;
    // ELogTag;
    // MOFEM_LOG("WORLD", Sev::inform) << "ALFA: " << ALPHA;

    CHKERR VecRestoreArrayRead(petscVec, &array);
  }

  MoFEMFunctionReturn(0);
}
//! [Output results]
MoFEMErrorCode ElectrostaticHomogeneous::outputResults() {
  MoFEMFunctionBegin;
  auto pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainRhsFE().reset();
  pipeline_mng->getDomainLhsFE().reset();
  pipeline_mng->getBoundaryLhsFE().reset();
  pipeline_mng->getBoundaryRhsFE().reset();

  pipeline_mng->getDomainLhsFE().reset();
  auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
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
      new OpNegativeGradient(neg_grad_u_ptr, grad_u_ptr));

  post_proc_fe->getOpPtrVector().push_back(

      new OpPPMap(post_proc_fe->getPostProcMesh(),
                  post_proc_fe->getMapGaussPts(),

                  OpPPMap::DataMapVec{{"POTENTIAL", u_ptr}},

                  OpPPMap::DataMapMat{{"ELECTRIC FILED", neg_grad_u_ptr}},

                  OpPPMap::DataMapMat{},

                  OpPPMap::DataMapMat{}

                  )

  );

  pipeline_mng->getDomainRhsFE() = post_proc_fe;
  CHKERR pipeline_mng->loopFiniteElements();
  CHKERR post_proc_fe->writeFile("out_resultFD.h5m");

  MoFEMFunctionReturn(0);
}
//! [Output results]

//! [Run program]
MoFEMErrorCode ElectrostaticHomogeneous::runProgram() {
  MoFEMFunctionBegin;

  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();
  CHKERR setIntegrationRules();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR outputResults();
  CHKERR getAlphaPart();
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
    ElectrostaticHomogeneous Electrostatics_problem(m_field);
    CHKERR Electrostatics_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}
//! [Main]
