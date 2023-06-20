/**
 * @file Electrostatichomogeneous.cpp
 * \example Electrostatichomogeneous.cpp
 *  */
// #ifndef __ELECTROSTATICHOMOGENEOUS_CPP__
// #define __ELECTROSTATICHOMOGENEOUS_CPP__

#include <stdlib.h>
#include <BasicFiniteElements.hpp>

// #include <electrostatic_3d_homogeneous.hpp>
// #include <electrostatic_2d_homogeneous.hpp>
constexpr auto domainField = "U";
constexpr int SPACE_DIM = EXECUTABLE_DIMENSION;
using namespace MoFEM;
// using namespace Electrostatic3DHomogeneousOperators;
// using namespace Electrostatic2DHomogeneousOperators;
using EntData = EntitiesFieldData::EntData;
using PostProcVolEle = PostProcBrokenMeshInMoab<VolumeElementForcesAndSourcesCore>;
using PostProcFaceEle =PostProcBrokenMeshInMoab<FaceElementForcesAndSourcesCore>;

template <int SPACE_DIM> struct BlockData {};

template <>
struct BlockData<2> {
  int iD;
  double sigma;
  Range eDge;
  double epsPermit;
  Range tRis;
};

template <>
struct BlockData<3> {
  int iD;
  double sigma;
  Range eDge;
  double epsPermit;
  Range tRis;
  Range tEts;
};

template <int SPACE_DIM>
struct DataAtIntegrationPts {
  SmartPetscObj<Vec> petscVec;
  double blockPermittivity;
  double chrgDens;

  DataAtIntegrationPts(MoFEM::Interface& m_field) {
    blockPermittivity = 0;
    chrgDens = 0;
  }
};

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
/////2222D
template <int SPACE_DIM> struct  OpBlockChargeDensity : public OpEdgeEle{};
template<>
struct OpBlockChargeDensity : public OpEdgeEle<2> {

  OpBlockChargeDensity(
      boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> common_data_ptr,
      boost::shared_ptr<map<int, BlockData<SPACE_DIM>>> edge_block_sets_ptr,
      const std::string &field_name)
      : OpEdgeEle(field_name, field_name, OPROWCOL, false),
        commonDataPtr(common_data_ptr), edgeBlockSetsPtr(edge_block_sets_ptr) {
    std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
    doEntities[MBVERTEX] = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        EntitiesFieldData::EntData &row_data,
                        EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;
    for (const auto &m : *edgeBlockSetsPtr) {
      if (m.second.eDge.find(getFEEntityHandle()) != m.second.eDge.end()) {
        commonDataPtr->chrgDens = m.second.sigma;
      }
    }
    MoFEMFunctionReturn(0);
  }

protected:
  boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> commonDataPtr;
  boost::shared_ptr<map<int, BlockData<SPACE_DIM>>> edgeBlockSetsPtr;

};

template<>
struct OpBlockPermittivity : public OpFaceEle<2> {

  OpBlockPermittivity(
      boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> common_data_ptr,
      boost::shared_ptr<map<int, BlockData<SPACE_DIM>>> surf_block_sets_ptr,
      const std::string &field_name)
      : OpFaceEle(field_name, field_name, OPROWCOL, false),
        commonDataPtr(common_data_ptr), surfBlockSetsPtr(surf_block_sets_ptr) {
    std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
    doEntities[MBVERTEX] = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        EntitiesFieldData::EntData &row_data,
                        EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;
    for (auto &m : (*surfBlockSetsPtr)) {
      if (m.second.tRis.find(getFEEntityHandle()) != m.second.tRis.end()) {
        commonDataPtr->blockPermittivity = m.second.epsPermit;
      }
    }
    MoFEMFunctionReturn(0);
  }

protected:
  boost::shared_ptr<map<int, BlockData<SPACE_DIM>>> surfBlockSetsPtr;
  boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> commonDataPtr;
};
//33333333D
template<>
struct OpBlockChargeDensity : public OpFaceEle<3> {

  OpBlockChargeDensity(
      boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> common_data_ptr,
      boost::shared_ptr<map<int, SurfBlockData<SPACE_DIM>>> surf_block_sets_ptr,
      const std::string &field_name)
      : OpFaceEle(field_name, field_name, OPROWCOL, false),
        commonDataPtr(common_data_ptr), surfBlockSetsPtr(surf_block_sets_ptr) {
    std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
    doEntities[MBVERTEX] = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        EntitiesFieldData::EntData &row_data,
                        EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;
    for (auto &m : (*surfBlockSetsPtr)) {
      if (m.second.tRis.find(getFEEntityHandle()) != m.second.tRis.end()) {
        commonDataPtr->chrgDens = m.second.sigma;
      }
    }
    MoFEMFunctionReturn(0);
  }

protected:
  boost::shared_ptr<map<int, SurfBlockData<SPACE_DIM>>> surfBlockSetsPtr;
  boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> commonDataPtr;
};
template<>
struct OpBlockPermittivity : public OpVolEle<3> {

  OpBlockPermittivity(
      boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> common_data_ptr,
      boost::shared_ptr<map<int, VolBlockData<SPACE_DIM>>> vol_block_sets_ptr,
      const std::string &field_name)
      : OpVolEle(field_name, field_name, OPROWCOL, false),
        commonDataPtr(common_data_ptr), volBlockSetsPtr(vol_block_sets_ptr) {
    std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
    doEntities[MBVERTEX] = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        EntitiesFieldData::EntData &row_data,
                        EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;
    for (auto &m : (*volBlockSetsPtr)) {
      if (m.second.tEts.find(getFEEntityHandle()) != m.second.tEts.end()) {
        commonDataPtr->blockPermittivity = m.second.epsPermit;
      }
    }
    MoFEMFunctionReturn(0);
  }

protected:
  boost::shared_ptr<map<int, VolBlockData<SPACE_DIM>>>volBlockSetsPtr;
  boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>>commonDataPtr;
};
//333333333333

template <int SPACE_DIM> struct OpDomainLhsMatrixK {};
template <> struct OpDomainLhsMatrixK : public OpFaceEle <2> {
using OpFaceEle = MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;
};
template <> struct OpDomainLhsMatrixK : public OpVolEle <3> {
using OpVolEle = MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator;
};


// template <int SPACE_DIM> struct OpInterfaceRhsVectorF {};
// template <> struct OpInterfaceRhsVectorF : public OpEdgeEle <2> {
// using OpEdgeEle = MoFEM::EdgeElementForcesAndSourcesCore::UserDataOperator;
// };
// template <> struct OpInterfaceRhsVectorF : public OpFaceEle <3> {
// using OpFaceEle = MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;
// };

// template <int SPACE_DIM> struct OpNegativeGradient : public ForcesAndSourcesCore::UserDataOperator  {};

// template <int SPACE_DIM> struct OpBlockChargeDensity {};
// template <> struct OpBlockChargeDensity : public OpEdgeEle <2> {
// using OpEdgeEle = MoFEM::EdgeElementForcesAndSourcesCore::UserDataOperator;
// }
// template <> struct OpBlockChargeDensity : public OpFaceEle <3> {
// using OpFaceEle = MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;
// }

// template <int SPACE_DIM> struct OpBlockPermittivity {};
// template <> struct OpBlockChargeDensity : public OpEdgeEle <2> {
// using OpEdgeEle = MoFEM::EdgeElementForcesAndSourcesCore::UserDataOperator;
// }
// template <> struct OpBlockChargeDensity : public OpFaceEle <3> {
// using OpFaceEle = MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;
// }


// static char help[] = "...\n\n";
// struct ElectrostaticHomogeneous {
// public:
//   Electrostatic3DHomogeneous(MoFEM::Interface &m_field);

//   // Declaration of the main function to run analysis
//   MoFEMErrorCode runProgram();

// private:
//   // Declaration of other main functions called in runProgram()
//   MoFEMErrorCode readMesh();
//   MoFEMErrorCode setupProblem();
//   MoFEMErrorCode boundaryCondition();
//   MoFEMErrorCode assembleSystem();
//   MoFEMErrorCode setIntegrationRules();
//   MoFEMErrorCode solveSystem();
//   MoFEMErrorCode outputResults();

//   // MoFEM interfaces
//   MoFEM::Interface &mField;
//     boost::shared_ptr<std::map<int, EdgeBlockData<SPACE_DIM>>> edge_block_sets_ptr;
//     boost::shared_ptr<std::map<int, SurfBlockData<SPACE_DIM>>> surf_block_sets_ptr;
//     boost::shared_ptr<std::map<int, VolBlockData<SPACE_DIM>>> vol_block_sets_ptr; ///////

//     Simple *simpleInterface;

//     boost::shared_ptr<ForcesAndSourcesCore> interface_rhs_fe;
//     boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> common_data_ptr;
//   // Field name and approximation order
//   std::string domainField;
//   int oRder;
//   };

// ElectrostaticHomogeneous::ElectrostaticHomogeneous(
//     MoFEM::Interface &m_field)
//     : domainField("U"), mField(m_field) {}

// //! [Read mesh]
// MoFEMErrorCode ElectrostaticHomogeneous::readMesh() {
//   MoFEMFunctionBegin;

//   CHKERR mField.getInterface(simpleInterface);
//   CHKERR simpleInterface->getOptions();
//   CHKERR simpleInterface->loadFile();

//   MoFEMFunctionReturn(0);
// }
// //! [Read mesh]
// //! [Setup problem]
// MoFEMErrorCode ElectrostaticHomogeneous::setupProblem() {
//   MoFEMFunctionBegin;

//   CHKERR simpleInterface->addDomainField(domainField, H1,
//                                          AINSWORTH_LEGENDRE_BASE, 1);

//   int oRder = 3;
//   CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &oRder, PETSC_NULL);
//   CHKERR simpleInterface->setFieldOrder(domainField, oRder);
// if (SPACE_DIM == 2){
//   edge_block_sets_ptr = boost::make_shared<std::map<int, EdgeBlockData<SPACE_DIM>>>();
//   Range interface_edges;
//   for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
//     if (bit->getName().compare(0, 12, "INT_ELECTRIC") == 0) {
//       const int id = bit->getMeshsetId();
//       auto &block_data = (*edge_block_sets_ptr)[id];

//       CHKERR mField.get_moab().get_entities_by_dimension(bit->getMeshset(), SPACE_DIM -1,
//                                                          block_data.eDge, true);
//       interface_edges.merge(block_data.eDge);

//       std::vector<double> attributes;
//       bit->getAttributes(attributes);
//       if (attributes.size() < 1) {
//         SETERRQ1(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
//                  "should be at least 1 attributes but is %d",
//                  attributes.size());
//       }

//       block_data.iD = id;
//       block_data.sigma = attributes[0];
//     }
//   }
// }
//   surf_block_sets_ptr = boost::make_shared<std::map<int, SurfBlockData<SPACE_DIM>>>();
//   Range electric_tris;
//   for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
//     if (bit->getName().compare(0, 12, "MAT_ELECTRIC") == 0) {
//       const int id = bit->getMeshsetId();
//       auto &block_data = (*surf_block_sets_ptr)[id];

//       CHKERR mField.get_moab().get_entities_by_dimension(bit->getMeshset(), SPACE_DIM,
//                                                          block_data.tRis, true);
//       electric_tris.merge(block_data.tRis);

//       std::vector<double> attributes;
//       bit->getAttributes(attributes);
//       if (attributes.size() < 1) {
//         SETERRQ1(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
//                  "should be at least 1 attributes but is %d",
//                  attributes.size());
//       }
//       block_data.iD = id;
//       block_data.epsPermit = attributes[0];
//     }
//   } else if (SPACE_DIM == 3){
//       surf_block_sets_ptr = boost::make_shared<std::map<int, SurfBlockData<SPACE_DIM>>>();
//   Range electric_tris;
//   for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
//     if (bit->getName().compare(0, 12, "MAT_ELECTRIC") == 0) {
//       const int id = bit->getMeshsetId();
//       auto &block_data = (*surf_block_sets_ptr)[id];

//       CHKERR mField.get_moab().get_entities_by_dimension(bit->getMeshset(), SPACE_DIM,
//                                                          block_data.tRis, true);
//       electric_tris.merge(block_data.tRis);

//       std::vector<double> attributes;
//       bit->getAttributes(attributes);
//       if (attributes.size() < 1) {
//         SETERRQ1(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
//                  "should be at least 1 attributes but is %d",
//                  attributes.size());
//       }
//       block_data.iD = id;
//       block_data.epsPermit = attributes[0];
//     }
//   }

// vol_block_sets_ptr = boost::make_shared<std::map<int, VolBlockData<SPACE_DIM>>>();
//   Range electric_tets;
//   for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
//     if (bit->getName().compare(0, 12, "MAT_ELECTRIC") == 0) {
//       const int id = bit->getMeshsetId();
//       auto &block_data = (*vol_block_sets_ptr)[id];

//       CHKERR mField.get_moab().get_entities_by_dimension(bit->getMeshset(), SPACE_DIM,
//                                                          block_data.tEts, true);
//       electric_tets.merge(block_data.tEts);

//       std::vector<double> attributes;
//       bit->getAttributes(attributes);
//       if (attributes.size() < 1) {
//         SETERRQ1(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
//                  "should be at least 1 attributes but is %d",
//                  attributes.size());
//       }
//       block_data.iD = id;
//       block_data.epsPermit = attributes[0];
//     }
//   }
//   }
//   CHKERR mField.add_finite_element("INTERFACE");
//   CHKERR mField.modify_finite_element_add_field_row("INTERFACE", domainField);
//   CHKERR mField.modify_finite_element_add_field_col("INTERFACE", domainField);
//   CHKERR mField.modify_finite_element_add_field_data("INTERFACE", domainField);
//   if (SPACE_DIM==2){
//      CHKERR mField.add_ents_to_finite_element_by_dim(interface_edges, MBEDGE,
//                                                          "INTERFACE");
//   }
//   else if (SPACE_DIM==3){
//   CHKERR mField.add_ents_to_finite_element_by_dim(interface_tris, MBTRI,
//                                                   "INTERFACE");
//   }

//   CHKERR simpleInterface->defineFiniteElements();
//   CHKERR simpleInterface->defineProblem(PETSC_TRUE);
//   CHKERR simpleInterface->buildFields();
//   CHKERR simpleInterface->buildFiniteElements();
//   CHKERR simpleInterface->buildProblem();


//   CHKERR mField.build_finite_elements("INTERFACE");
//   CHKERR DMMoFEMAddElement(simpleInterface->getDM(), "INTERFACE");
//   CHKERR simpleInterface->buildProblem();
//   MoFEMFunctionReturn(0);
// }
// //! [Setup problem]

// //! [Boundary condition]
// MoFEMErrorCode ElectrostaticHomogeneous::boundaryCondition() {
//   MoFEMFunctionBegin;

//   auto bc_mng = mField.getInterface<BcManager>();

//   // Remove BCs from blockset name "BOUNDARY_CONDITION";
//   CHKERR bc_mng->removeBlockDOFsOnEntities<BcScalarMeshsetType<BLOCKSET>>(
//       simpleInterface->getProblemName(), "BOUNDARY_CONDITION",
//       std::string(domainField), true);

//   MoFEMFunctionReturn(0);
// }


// //! [Boundary condition]
// //! [Assemble system]
// MoFEMErrorCode ElectrostaticHomogeneous::assembleSystem() {
//   MoFEMFunctionBegin;

//   auto pipeline_mng = mField.getInterface<PipelineManager>();
//   common_data_ptr = boost::make_shared<DataAtIntegrationPts<SPACE_DIM>>(mField);
//   auto add_domain_lhs_ops = [&](auto &pipeline) {
//     if (SPACE_DIM ==2){
//     pipeline.push_back(new OpBlockPermittivity(
//         common_data_ptr, surf_block_sets_ptr, domainField));
//     }
//     else if(SPACE_DIM ==3){
//     pipeline.push_back(new OpBlockPermittivity(
//     common_data_ptr, vol_block_sets_ptr, domainField));
//     }

//   };

//   add_domain_lhs_ops(pipeline_mng->getOpDomainLhsPipeline());


//   { // Push operators to the Pipeline that is responsible for calculating LHS
//     CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
//         pipeline_mng->getOpDomainLhsPipeline(), {H1});
//     pipeline_mng->getOpDomainLhsPipeline().push_back(
//         new OpDomainLhsMatrixK(domainField, domainField, common_data_ptr));
//   }
//  { // Push operators to the Pipeline that is responsible for calculating LHS

//     auto set_values_to_bc_dofs = [&](auto &fe) {
//       auto get_bc_hook = [&]() {
//         EssentialPreProc<TemperatureCubitBcData> hook(mField, fe, {});
//         return hook;
//       };
//       fe->preProcessHook = get_bc_hook();
//     };

//     auto calculate_residual_from_set_values_on_bc = [&](auto &pipeline) {
//       using DomainEle =
//           PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::DomainEle;
//       using DomainEleOp = DomainEle::UserDataOperator;
//       using OpInternal = FormsIntegrators<DomainEleOp>::Assembly<
//           PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<BASE_DIM, FIELD_DIM, SPACE_DIM>;

//       auto grad_u_vals_ptr = boost::make_shared<MatrixDouble>();
//       pipeline_mng->getOpDomainRhsPipeline().push_back(
//           new OpCalculateScalarFieldGradient<SPACE_DIM>(domainField,
//                                                         grad_u_vals_ptr));
//       pipeline_mng->getOpDomainRhsPipeline().push_back(
//           new OpInternal(domainField, grad_u_vals_ptr,
//                          [](double, double, double) constexpr { return -1; }));
//     };

//     CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
//         pipeline_mng->getOpDomainRhsPipeline(), {H1});
//     set_values_to_bc_dofs(pipeline_mng->getDomainRhsFE());
//     calculate_residual_from_set_values_on_bc(
//         pipeline_mng->getOpDomainRhsPipeline());


//     interface_rhs_fe = boost::shared_ptr<ForcesAndSourcesCore>(
//         new EdgeElementForcesAndSourcesCore(mField));
//         {
// interface_rhs_fe->getOpPtrVector().push_back(new OpBlockChargeDensity(
//           common_data_ptr, edge_block_sets_ptr, domainField));

//       interface_rhs_fe->getOpPtrVector().push_back(
//           new OpInterfaceRhsVectorF(domainField, common_data_ptr));
    
//   }
//  }
//   MoFEMFunctionReturn(0);
// }
// //! [Assemble system]
// MoFEMErrorCode ElectrostaticHomogeneous::setIntegrationRules() {
//   MoFEMFunctionBegin;

//   auto rule_lhs = [](int, int, int p) -> int { return 2 * (p - 1); };
//   auto rule_rhs = [](int, int, int p) -> int { return p; };

//   auto pipeline_mng = mField.getInterface<PipelineManager>();
//   CHKERR pipeline_mng->setDomainLhsIntegrationRule(rule_lhs);
//   CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule_rhs);

//   // interface_rhs_fe->getRuleHook = PoissonExample::FaceRule();

//   MoFEMFunctionReturn(0);
// }
// //! [Set integration rules]

// //! [Solve system]
// MoFEMErrorCode ElectrostaticHomogeneous::solveSystem() {
//   MoFEMFunctionBegin;

//   auto pipeline_mng = mField.getInterface<PipelineManager>();

//   auto ksp_solver = pipeline_mng->createKSP();

//   boost::shared_ptr<ForcesAndSourcesCore> null; ///< Null element does nothing
//   CHKERR DMMoFEMKSPSetComputeRHS(simpleInterface->getDM(), "INTERFACE",
//                                  interface_rhs_fe, null, null);

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
// //! [Solve system]

// //! [Output results]
// MoFEMErrorCode ElectrostaticHomogeneous::outputResults() {
//   MoFEMFunctionBegin;

//   auto pipeline_mng = mField.getInterface<PipelineManager>();
//   pipeline_mng->getDomainLhsFE().reset();

//   auto post_proc_fe = boost::make_shared<PostProcFaceEle>(mField);
  
//   auto post_proc_fe = boost::make_shared<PostProcVolEle>(mField);

//   auto det_ptr = boost::make_shared<VectorDouble>();
//   auto jac_ptr = boost::make_shared<MatrixDouble>();
//   auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

//   post_proc_fe->getOpPtrVector().push_back(
//       new OpCalculateHOJac<SPACE_DIM>(jac_ptr));
//   post_proc_fe->getOpPtrVector().push_back(
//       new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
//   post_proc_fe->getOpPtrVector().push_back(
//       new OpSetHOInvJacToScalarBases<SPACE_DIM>(H1, inv_jac_ptr));

//   auto u_ptr = boost::make_shared<VectorDouble>();
//   auto grad_u_ptr = boost::make_shared<MatrixDouble>();
//   post_proc_fe->getOpPtrVector().push_back(
//       new OpCalculateScalarFieldValues(domainField, u_ptr));

//   post_proc_fe->getOpPtrVector().push_back(
//       new OpCalculateScalarFieldGradient<SPACE_DIM>(domainField, grad_u_ptr));

//   using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

//   auto neg_grad_u_ptr = boost::make_shared<MatrixDouble>();
//   post_proc_fe->getOpPtrVector().push_back(
//       new OpNegativeGradient<SPACE_DIM>(neg_grad_u_ptr, grad_u_ptr));

//   post_proc_fe->getOpPtrVector().push_back(

//       new OpPPMap(post_proc_fe->getPostProcMesh(),
//                   post_proc_fe->getMapGaussPts(),

//                   OpPPMap::DataMapVec{{"U", u_ptr}},

//                   OpPPMap::DataMapMat{{"ELECTRIC FILED", neg_grad_u_ptr}},

//                   OpPPMap::DataMapMat{},

//                   OpPPMap::DataMapMat{}

//                   )

//   );

//   pipeline_mng->getDomainRhsFE() = post_proc_fe;
//   CHKERR pipeline_mng->loopFiniteElements();
//   CHKERR post_proc_fe->writeFile("out_result2D.h5m");

//   MoFEMFunctionReturn(0);
// }
// //! [Output results]

// //! [Run program]
// MoFEMErrorCode ElectrostaticHomogeneous::runProgram() {
//   MoFEMFunctionBegin;

//   CHKERR readMesh();
//   CHKERR setupProblem();
//   CHKERR boundaryCondition();
//   CHKERR setIntegrationRules();
//   CHKERR assembleSystem();
//   CHKERR solveSystem();
//   CHKERR outputResults();

//   MoFEMFunctionReturn(0);
// }
// //! [Run program]

// //! [Main]
// int main(int argc, char *argv[]) {

//   // Initialisation of MoFEM/PETSc and MOAB data structures
//   const char param_file[] = "param_file.petsc";
//   MoFEM::Core::Initialize(&argc, &argv, param_file, help);

//   // Error handling
//   try {
//     // Register MoFEM discrete manager in PETSc
//     DMType dm_name = "DMMOFEM";
//     CHKERR DMRegister_MoFEM(dm_name);

//     // Create MOAB instance
//     moab::Core mb_instance;              // mesh database
//     moab::Interface &moab = mb_instance; // mesh database interface

//     // Create MoFEM instance
//     MoFEM::Core core(moab);           // finite element database
//     MoFEM::Interface &m_field = core; // finite element interface

//     // Run the main analysis
//     Electrostatic2DHomogeneous poisson_problem(m_field);
//     CHKERR poisson_problem.runProgram();
//   }
//   CATCH_ERRORS;

//   // Finish work: cleaning memory, getting statistics, etc.
//   MoFEM::Core::Finalize();

//   return 0;
// }
// //! [Main]

// // #endif //__ELECTROSTATICHOMOGENEOUS_CPP__