/**
 * @file Electrostatichomogeneous.cpp
 * \example Electrostatichomogeneous.cpp
 *  */
// #ifndef __ELECTROSTATICHOMOGENEOUS_CPP__
// #define __ELECTROSTATICHOMOGENEOUS_CPP__
#ifndef EXECUTABLE_DIMENSION
#define EXECUTABLE_DIMENSION 3
#endif
#include <MoFEM.hpp>

constexpr auto domainField = "POTENTIAL";
constexpr int BASE_DIM = 1;
constexpr int FIELD_DIM = 1;
constexpr int SPACE_DIM = EXECUTABLE_DIMENSION;
using namespace MoFEM;

using DomainEle = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::DomainEle;
using IntEle = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::BoundaryEle;
using DomainEleOp = DomainEle::UserDataOperator;
using IntEleOp = IntEle::UserDataOperator;
using EntData = EntitiesFieldData::EntData;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;
using SideEle = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::FaceSideEle;
using SideEleOp = SideEle::UserDataOperator;

template <int SPACE_DIM> struct intPostproc {};

template <> struct intPostproc<2> {
  using intEle = MoFEM::EdgeElementForcesAndSourcesCore;
};

template <> struct intPostproc<3> {
  using intEle = MoFEM::FaceElementForcesAndSourcesCore;
};

using intPostProcElementForcesAndSourcesCore = intPostproc<SPACE_DIM>::intEle;

using OpDomainLhsMatrixK = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpGradGrad<BASE_DIM, FIELD_DIM, SPACE_DIM>;

using OpInterfaceRhsVectorF = FormsIntegrators<IntEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<BASE_DIM, FIELD_DIM>;

using OpDirchBoundaryMassL = FormsIntegrators<IntEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<BASE_DIM, FIELD_DIM>; /// opMass

using OpDirchBoundarySourceR = FormsIntegrators<IntEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<BASE_DIM, FIELD_DIM>; /// OpSourceL

static char help[] = "...\n\n";

template <int SPACE_DIM> struct BlockData {
  int iD;
  double sigma;
  double epsPermit;
  Range blockInterfaces;
  Range blockDomains;
};
template <int SPACE_DIM> struct DataAtIntegrationPts {

  SmartPetscObj<Vec> petscVec;
  double blockPermittivity;
  double chrgDens;
  DataAtIntegrationPts(MoFEM::Interface &m_field) {
    blockPermittivity = 0;
    chrgDens = 0;
  }
};

template <int SPACE_DIM>
struct OpNegativeGradient : public ForcesAndSourcesCore::UserDataOperator {
  OpNegativeGradient(boost::shared_ptr<MatrixDouble> grad_u_negative,
                     boost::shared_ptr<MatrixDouble> grad_u)
      : ForcesAndSourcesCore::UserDataOperator(NOSPACE, OPLAST),
        gradUNegative(grad_u_negative), gradU(grad_u) {}

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    const size_t nb_gauss_pts = getGaussPts().size2();
    gradUNegative->resize(SPACE_DIM, nb_gauss_pts, false);
    gradUNegative->clear();

    auto t_grad_u = getFTensor1FromMat<SPACE_DIM>(*gradU);

    auto t_negative_grad_u = getFTensor1FromMat<SPACE_DIM>(*gradUNegative);

    FTensor::Index<'I', SPACE_DIM> I;

    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      t_negative_grad_u(I) = -t_grad_u(I);

      ++t_grad_u;
      ++t_negative_grad_u;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> gradUNegative;
  boost::shared_ptr<MatrixDouble> gradU;
};
template <int SPACE_DIM> struct OpBlockChargeDensity : public IntEleOp {
  OpBlockChargeDensity(
      boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> common_data_ptr,
      boost::shared_ptr<std::map<int, BlockData<SPACE_DIM>>> int_block_sets_ptr,
      const std::string &field_name)
      : IntEleOp(field_name, field_name, OPROWCOL, false),
        commonDataPtr(common_data_ptr), intBlockSetsPtr(int_block_sets_ptr) {
    std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
    doEntities[MBVERTEX] = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;
    for (const auto &m : *intBlockSetsPtr) {
      if (m.second.blockInterfaces.find(getFEEntityHandle()) !=
          m.second.blockInterfaces.end()) {
        commonDataPtr->chrgDens = m.second.sigma;
      }
    }
    MoFEMFunctionReturn(0);
  }

protected:
  boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> commonDataPtr;
  boost::shared_ptr<std::map<int, BlockData<SPACE_DIM>>> intBlockSetsPtr;
};

template <int SPACE_DIM> struct OpBlockPermittivity : public DomainEleOp {

  OpBlockPermittivity(
      boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> common_data_ptr,
      boost::shared_ptr<map<int, BlockData<SPACE_DIM>>> perm_block_sets_ptr,
      const std::string &field_name)
      : DomainEleOp(field_name, field_name, OPROWCOL, false),
        commonDataPtr(common_data_ptr), permBlockSetsPtr(perm_block_sets_ptr) {
    std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
    doEntities[MBVERTEX] = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        EntitiesFieldData::EntData &row_data,
                        EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;
    for (auto &m : (*permBlockSetsPtr)) {
      if (m.second.blockDomains.find(getFEEntityHandle()) !=
          m.second.blockDomains.end()) {
        commonDataPtr->blockPermittivity = m.second.epsPermit;
      }
    }
    MoFEMFunctionReturn(0);
  }

protected:
  boost::shared_ptr<map<int, BlockData<SPACE_DIM>>> permBlockSetsPtr;
  boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> commonDataPtr;
};

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
  MoFEMErrorCode outputResults();

  // MoFEM interfaces
  MoFEM::Interface &mField;

  boost::shared_ptr<std::map<int, BlockData<SPACE_DIM>>> perm_block_sets_ptr;
  boost::shared_ptr<std::map<int, BlockData<SPACE_DIM>>>
      int_block_sets_ptr; ///////
  Simple *simpleInterface;
  boost::shared_ptr<ForcesAndSourcesCore> interface_rhs_fe;
  boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> common_data_ptr;

  std::string domainField;
  int oRder;
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
  double alphaPart = 0.5;
  double alphaA = 0.;
  Range FloatingElectrodes;
  Range blockconstBC;
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

  int oRder = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &oRder, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder(domainField, oRder);

  common_data_ptr = boost::make_shared<DataAtIntegrationPts<SPACE_DIM>>(mField);

  perm_block_sets_ptr =
      boost::make_shared<std::map<int, BlockData<SPACE_DIM>>>();
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
  int_block_sets_ptr =
      boost::make_shared<std::map<int, BlockData<SPACE_DIM>>>();
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
  ////////////

  auto *simple = mField.getInterface<Simple>();

  // Mark BC for blocksets. Blockset name is
  // "FLOATING_ELECTRODE"
  CHKERR bc_mng->pushMarkDOFsOnEntities(simpleInterface->getProblemName(),
                                        "FLOATING_ELECTRODE", domainField, 0, 1,
                                        true);

  boundaryMarker =
      bc_mng->getMergedBlocksMarker(vector<string>{"FLOATING_ELECTRODE"});

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

  common_data_ptr = boost::make_shared<DataAtIntegrationPts<SPACE_DIM>>(mField);
  // auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto add_domain_lhs_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpBlockPermittivity<SPACE_DIM>(
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
        new OpSetBc(domainField, true, boundaryMarker));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpDomainLhsMatrixK(domainField, domainField, epsilon));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpUnSetBc(domainField));
  }
  { // Push operators to the Pipeline that is responsible for
    // calculating
    // LHS
    MoFEMFunctionBegin;
    auto set_values_to_bc_dofs = [&](auto &fe) {
      auto get_bc_hook = [&]() {
        EssentialPreProc<TemperatureCubitBcData> hook(mField, fe, {});
        return hook;
      };
      fe->preProcessHook = get_bc_hook();
    };
    // Set essential BC
    auto set_boundary = [&]() {
      MoFEMFunctionBegin;
      CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
          pipeline_mng->getOpBoundaryLhsPipeline(), {});

      auto set_bc_essential = [&](auto domainField, auto block_name) {
        MoFEMFunctionBegin;
        auto bc_mng = mField.getInterface<BcManager>();

        auto &bcs = bc_mng->getBcMapByBlockName();
        for (auto &bc : bcs) {
          if (bc_mng->checkBlock(bc, block_name)) {
            // auto val = bc.second->bcAttributes[0];
            pipeline_mng->getOpBoundaryLhsPipeline().push_back(
                new OpSetBc(domainField, false, bc.second->getBcMarkersPtr()));
            pipeline_mng->getOpBoundaryLhsPipeline().push_back(
                new OpDirchBoundaryMassL(
                    domainField, domainField,
                    [](double, double, double) { return 1; },
                    bc.second->getBcEntsPtr()));
            auto boundary_val_function = [&](const double, const double,
                                             const double) {
              return alphaPart;
            };
            pipeline_mng->getOpBoundaryRhsPipeline().push_back(
                new OpDirchBoundarySourceR(
                    domainField, boundary_val_function,
                    boost::make_shared<Range>(bc.second->bcEnts)));
            pipeline_mng->getOpBoundaryLhsPipeline().push_back(
                new OpUnSetBc(domainField));

            // Set essential BC
          }
        }

        MoFEMFunctionReturn(0);
      };

      CHKERR set_bc_essential(domainField, "FLOATING_ELECTRODE");

      MoFEMFunctionReturn(0);
    };
    // CHKERR set_bc_essential(domainField, "FLOATING_ELECTRODE");

    auto calculate_residual_from_set_values_on_bc = [&](auto &pipeline) {
      using OpInternal =
          FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
              GAUSS>::OpGradTimesTensor<BASE_DIM, FIELD_DIM, SPACE_DIM>;

      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpSetBc(domainField, true, boundaryMarker));

      auto grad_u_vals_ptr = boost::make_shared<MatrixDouble>();
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateScalarFieldGradient<SPACE_DIM>(domainField,
                                                        grad_u_vals_ptr));

      auto minus_epsilon = [&](double, double, double) constexpr {
        return -common_data_ptr->blockPermittivity;
      };

      add_domain_lhs_ops(pipeline_mng->getOpDomainRhsPipeline());

      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpInternal(domainField, grad_u_vals_ptr, minus_epsilon));
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpUnSetBc(domainField));
    };

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        pipeline_mng->getOpDomainRhsPipeline(), {H1});

    set_values_to_bc_dofs(pipeline_mng->getDomainRhsFE());
    calculate_residual_from_set_values_on_bc(
        pipeline_mng->getOpDomainRhsPipeline());

    interface_rhs_fe = boost::shared_ptr<ForcesAndSourcesCore>(
        new intPostProcElementForcesAndSourcesCore(mField));

    interface_rhs_fe->getOpPtrVector().push_back(
        new OpBlockChargeDensity<SPACE_DIM>(common_data_ptr, int_block_sets_ptr,
                                            domainField));

    auto sIgma = [&](const double, const double, const double) {
      return common_data_ptr->chrgDens;
    };

    interface_rhs_fe->getOpPtrVector().push_back(
        new OpInterfaceRhsVectorF(domainField, sIgma));
    CHKERR set_boundary();
    MoFEMFunctionReturn(0);
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

  // interface_rhs_fe->getRuleHook = PoissonExample::FaceRule();

  MoFEMFunctionReturn(0);
}
//! [Set integration rules]

//! [Solve system]
MoFEMErrorCode ElectrostaticHomogeneous::solveSystem() {
  MoFEMFunctionBegin;

  //   auto pipeline_mng = mField.getInterface<PipelineManager>();

  //   auto ksp_solver = pipeline_mng->createKSP();

  //   boost::shared_ptr<ForcesAndSourcesCore> null; ///< Null element does
  //   CHKERR DMMoFEMKSPSetComputeRHS(simpleInterface->getDM(),
  //   "INTERFACE",
  //                                  interface_rhs_fe, null, null);

  //   auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
  //   CHKERR KSPSetFromOptions(ksp_solver);
  //   CHKERR KSPSetUp(ksp_solver);

  //   // Create RHS and solution vectors
  //   auto dm = simpleInterface->getDM();
  //   auto F = smartCreateDMVector(dm);
  //   auto D = smartVectorDuplicate(F);

  //   // Solve the system
  //   CHKERR KSPSolve(ksp_solver, F, D);

  //   // Scatter result data on the mesh
  //   CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  //   CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
  //   CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

  //   MoFEMFunctionReturn(0);
  // }
  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto ksp_solver = pipeline_mng->createKSP();

  boost::shared_ptr<ForcesAndSourcesCore> null; ///< Null element does
  CHKERR DMMoFEMKSPSetComputeRHS(simpleInterface->getDM(), "INTERFACE",
                                 interface_rhs_fe, null, null);

  auto post_proc_fe = boost::make_shared<PostProcEle>(mField);

  auto set_fieldsplit_preconditioner_ksp = [&](auto ksp) {
    MoFEMFunctionBeginHot;
    PC pc;
    CHKERR KSPGetPC(ksp, &pc);
    PetscBool is_pcfs = PETSC_FALSE;
    PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &is_pcfs);
    if (is_pcfs == PETSC_TRUE) {
      auto bc_mng = mField.getInterface<BcManager>();
      auto name_prb = simpleInterface->getProblemName();
      auto is_all_bc =
          bc_mng->getBlockIS(name_prb, "FLOATING_ELECTRODE", domainField, 0, 0);
      // bc_mng->getBlockIS(name_prb, "BOUNDARY_CONDITION", domainField, 0, 0);
      int is_all_bc_size;
      CHKERR ISGetSize(is_all_bc, &is_all_bc_size);
      // MOFEM_LOG("EXAMPLE", Sev::inform)
      //     << "Field split block size " << is_all_bc_size;
      CHKERR PCFieldSplitSetIS(pc, PETSC_NULL,
                               is_all_bc); // boundary block
    }
    MoFEMFunctionReturnHot(0);
  };

  CHKERR KSPSetFromOptions(ksp_solver);
  CHKERR set_fieldsplit_preconditioner_ksp(ksp_solver);
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
//! [Solve system]

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
      new OpNegativeGradient<SPACE_DIM>(neg_grad_u_ptr, grad_u_ptr));

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
  alphaA = alphaPart;
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
    ElectrostaticHomogeneous poisson_problem(m_field);
    CHKERR poisson_problem.runProgram();
  }
  CATCH_ERRORS;

  // Finish work: cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}
//! [Main]
