/**
 * @file Electrostatichomogeneous.cpp
 * \example Electrostatichomogeneous.cpp
 *  */
// #ifndef __ELECTROSTATICHOMOGENEOUS_CPP__
// #define __ELECTROSTATICHOMOGENEOUS_CPP__

#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <PoissonOperators.hpp>

// #include <electrostatic_3d_homogeneous.hpp>
// #include <electrostatic_2d_homogeneous.hpp>
constexpr auto domainField = "POTENTIAL";
constexpr int BASE_DIM = 1;
constexpr int FIELD_DIM = 1;
// constexpr int SPACE_DIM = EXECUTABLE_DIMENSION;
constexpr int SPACE_DIM = 3;
// using namespace MoFEM;
// template <int SPACE_DIM> struct ElementsAndOps {};
// template <> struct ElementsAndOps<2> {
// using DomainEle = MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;
// using IntEle= MoFEM::EdgeElementForcesAndSourcesCore::UserDataOperator;
// };
// template <> struct ElementsAndOps<3> {
// using DomainEle =MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator;
// using IntEle= MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;
// };

// template <int SPACE_DIM>
// using DomainEleOp = typename ElementsAndOps<SPACE_DIM>::DomainEleOp;

// template <int SPACE_DIM>
// using IntEleOp = typename ElementsAndOps<SPACE_DIM>::IntEleOp;
 
// using OpFaceEle = MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;
// using OpEdgeEle = MoFEM::EdgeElementForcesAndSourcesCore::UserDataOperator;


using DomainEle = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::DomainEle;
using IntEle = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::BoundaryEle;
using DomainEleOp = DomainEle::UserDataOperator;
using IntEleOp = IntEle::UserDataOperator;

// using DomainEle = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::DomainEle;
// using IntEle = PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::BoundaryEle;
// using DomainEleOp = DomainEle::UserDataOperator;
// using IntEleOp = IntEle::UserDataOperator;

using EntData = EntitiesFieldData::EntData;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;
using PostProcVolEle =PostProcBrokenMeshInMoab<VolumeElementForcesAndSourcesCore>;
// template <int SPACE_DIM> struct PostProcEle {};
// // template <> struct PostProcEle<2> {
// using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;
// // };
// // template <> struct PostProcEle<3> {
// using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;
// // };



static char help[] = "...\n\n";
// template <int SPACE_DIM>
// struct BlockData {
//     int iD;
//     double sigma;
//     double epsPermit;
//     Range blockEnts;
// };
// // auto tets  = blockEnts.subset_by_dimension(SPACE_DIM-1);
template <int SPACE_DIM> struct BlockData {};

template <>
struct BlockData<2> {
  int iD;
  double sigma;
  Range eDges;
  double epsPermit;
  Range tRis;
  Range tEts;
};

template <>
struct BlockData<3> {
  int iD;
  double sigma;
  Range eDges;
  double epsPermit;
  Range tRis;
  Range tEts;
};
template <int SPACE_DIM> 
struct DataAtIntegrationPts {

  SmartPetscObj<Vec> petscVec;
  double blockPermittivity;
  double chrgDens;
  DataAtIntegrationPts(MoFEM::Interface &m_field) {
    blockPermittivity=0;
    chrgDens=0;
  }
};

template <int SPACE_DIM> 
struct OpDomainLhsMatrixK : public DomainEleOp {
public:
  bool sYmm;
  OpDomainLhsMatrixK(std::string row_field_name, std::string col_field_name,
                     boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> common_data_ptr)
      : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
        commonDataPtr(common_data_ptr) {
    sYmm = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;

    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();
// namespace ElectrostaticHomogeneousOperators {
FTensor::Index<'i', SPACE_DIM> i;

    if (nb_row_dofs && nb_col_dofs) {

      locLhs.resize(nb_row_dofs, nb_col_dofs, false);
      locLhs.clear();

      // get element area
      const double area = getMeasure();

      // get number of integration points
      const int nb_integration_points = getGaussPts().size2();
      // get integration weights
      auto t_w = getFTensor0IntegrationWeight();

      // get derivatives of base functions on row
      auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL MATRIX
      for (int gg = 0; gg != nb_integration_points; gg++) {
        const double a = t_w * area;

        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          // get derivatives of base functions on column
          auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

          for (int cc = 0; cc != nb_col_dofs; cc++) {
            locLhs(rr, cc) += t_row_diff_base(i) * t_col_diff_base(i) * a *
                              commonDataPtr->blockPermittivity;

            // move to the derivatives of the next base functions on column
            ++t_col_diff_base;
          }

          // move to the derivatives of the next base functions on row
          ++t_row_diff_base;
        }

        // move to the weight of the next integration point
        ++t_w;
      }

      // FILL VALUES OF LOCAL MATRIX ENTRIES TO THE GLOBAL MATRIX
      CHKERR MatSetValues(getKSPB(), row_data, col_data, &locLhs(0, 0),
                          ADD_VALUES);
      if (row_side != col_side || row_type != col_type) {
        transLocLhs.resize(nb_col_dofs, nb_row_dofs, false);
        noalias(transLocLhs) = trans(locLhs);
        CHKERR MatSetValues(getKSPB(), col_data, row_data, &transLocLhs(0, 0),
                            ADD_VALUES);
      }
    }

    MoFEMFunctionReturn(0);
  }

private:
  MatrixDouble locLhs, transLocLhs;
  boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> commonDataPtr;
};
template <int SPACE_DIM>
struct OpInterfaceRhsVectorF : public IntEleOp{
public:
 OpInterfaceRhsVectorF(std::string field_name,
                        boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> common_data_ptr)
      : IntEleOp(field_name, IntEleOp::OPROW),
        commonDataPtr(common_data_ptr) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    const int nb_dofs = data.getIndices().size();

    if (nb_dofs) {
      locRhs.resize(nb_dofs, false);
      locRhs.clear();

      // get element area
      const double area = getMeasure();

      // get number of integration points
      const int nb_integration_points = getGaussPts().size2();
      // get integration weights
      auto t_w = getFTensor0IntegrationWeight();

      // get base function
      auto t_base = data.getFTensor0N();


      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL VECTOR
      for (int gg = 0; gg != nb_integration_points; gg++) {
        const double a = t_w * area;
        // double y = getGaussPts()(1, gg);
        double y =  getCoordsAtGaussPts()(gg, 1);
        for (int rr = 0; rr != nb_dofs; rr++) {
          locRhs[rr] += t_base * a * commonDataPtr->chrgDens;

          ++t_base;
        }

        // move to the weight of the next integration point
        ++t_w;
      }

      // FILL VALUES OF LOCAL VECTOR ENTRIES TO THE GLOBAL VECTOR

      // Ignoring DOFs on boundary (index -1)
    CHKERR VecSetOption(getKSPf(), VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
      CHKERR VecSetValues(getKSPf(), data, &*locRhs.begin(), ADD_VALUES);
    }

    MoFEMFunctionReturn(0);
  }

private:
  VectorDouble locRhs;
  boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> commonDataPtr;
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
template <int SPACE_DIM>
struct OpBlockChargeDensity : public IntEleOp {};

template <>
struct OpBlockChargeDensity<2> : public IntEleOp {
  OpBlockChargeDensity(
      boost::shared_ptr<DataAtIntegrationPts<2>> common_data_ptr,
      boost::shared_ptr<std::map<int, BlockData<2>>> edge_block_sets_ptr,
      const std::string& field_name)
      : IntEleOp(field_name, field_name, OPROWCOL, false),
        commonDataPtr(common_data_ptr), edgeBlockSetsPtr(edge_block_sets_ptr) {
    std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
    doEntities[MBVERTEX] = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        EntitiesFieldData::EntData &row_data,
                        EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;
    for (const auto& m : *edgeBlockSetsPtr) {
      if (m.second.eDges.find(getFEEntityHandle()) != m.second.eDges.end()) {
        commonDataPtr->chrgDens = m.second.sigma;
      }
    }
    MoFEMFunctionReturn(0);
  }

protected:
  boost::shared_ptr<DataAtIntegrationPts<2>> commonDataPtr;
  boost::shared_ptr<std::map<int, BlockData<2>>> edgeBlockSetsPtr;
};

template <>
struct OpBlockChargeDensity<3> : public IntEleOp {
  OpBlockChargeDensity(
      boost::shared_ptr<DataAtIntegrationPts<3>> common_data_ptr,
      boost::shared_ptr<std::map<int, BlockData<3>>> surf_block_sets_ptr,
      const std::string& field_name)
      : IntEleOp(field_name, field_name, OPROWCOL, false),
        commonDataPtr(common_data_ptr), surfBlockSetsPtr(surf_block_sets_ptr) {
    std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
    doEntities[MBVERTEX] = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        EntitiesFieldData::EntData &row_data,
                        EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;
    for (const auto& m : *surfBlockSetsPtr) {
      if (m.second.tRis.find(getFEEntityHandle()) != m.second.tRis.end()) {
        commonDataPtr->chrgDens = m.second.sigma;
      }
    }
    MoFEMFunctionReturn(0);
  }

protected:
  boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> commonDataPtr;
  boost::shared_ptr<std::map<int, BlockData<3>>> surfBlockSetsPtr;
};
template <int SPACE_DIM>
struct OpBlockPermittivity : public DomainEleOp {};

template <>
struct OpBlockPermittivity<2> : public DomainEleOp {;
  OpBlockPermittivity(
      boost::shared_ptr<DataAtIntegrationPts<2>> common_data_ptr,
      boost::shared_ptr<map<int, BlockData<2>>> surf_block_sets_ptr,
      const std::string &field_name)
      : DomainEleOp(field_name, field_name, OPROWCOL, false),
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
  boost::shared_ptr<map<int,  BlockData<2>>> surfBlockSetsPtr;
  boost::shared_ptr<DataAtIntegrationPts<2>> commonDataPtr;
};

template <>
struct OpBlockPermittivity<3> : public DomainEleOp {;
  OpBlockPermittivity(
      boost::shared_ptr<DataAtIntegrationPts<3>> common_data_ptr,
      boost::shared_ptr<map<int, BlockData<3>>> vol_block_sets_ptr,
      const std::string &field_name)
      : DomainEleOp(field_name, field_name, OPROWCOL, false),
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
      if (m.second.tRis.find(getFEEntityHandle()) != m.second.tRis.end()) {
        commonDataPtr->blockPermittivity = m.second.epsPermit;
      }
    }
    MoFEMFunctionReturn(0);
  }

protected:
  boost::shared_ptr<map<int,  BlockData<3>>> volBlockSetsPtr;
  boost::shared_ptr<DataAtIntegrationPts<3>> commonDataPtr;
};

// }
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
  boost::shared_ptr<std::map<int, BlockData<SPACE_DIM>>> int_block_sets_ptr; ///////
  Simple *simpleInterface;
  boost::shared_ptr<ForcesAndSourcesCore> interface_rhs_fe;
  boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> common_data_ptr;

  std::string domainField;
  int oRder;
};  

ElectrostaticHomogeneous::ElectrostaticHomogeneous(
    MoFEM::Interface &m_field)
    : domainField("POTENTIAL"), mField(m_field) {}

//! [Read mesh]
MoFEMErrorCode ElectrostaticHomogeneous::readMesh() {
  MoFEMFunctionBegin;

  //  MOFEM_LOG("ELECTROSTATICS", Sev::inform)
  //     << "Read mesh for problem in " << 3;
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

  int oRder = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &oRder, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder(domainField, oRder);

// using BlockSetsPtrType = boost::shared_ptr<map<int, BlockData<SPACE_DIM>>>;

perm_block_sets_ptr = boost::make_shared<map<int, BlockData<SPACE_DIM>>>();
Range electric_tRis;
Range electric_tEts;
for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
  if (bit->getName().compare(0, 12, "MAT_ELECTRIC") == 0) {
    const int id = bit->getMeshsetId();
    auto& block_data = (*perm_block_sets_ptr)[id];{

    if (SPACE_DIM == 2) {
      CHKERR mField.get_moab().get_entities_by_dimension(bit->getMeshset(), SPACE_DIM,
                                                         block_data.tRis, true);
      electric_tRis.merge(block_data.tRis);

    } else if (SPACE_DIM == 3) {
      CHKERR mField.get_moab().get_entities_by_dimension(bit->getMeshset(), SPACE_DIM,
                                                         block_data.tEts, true);
      electric_tEts.merge(block_data.tEts);
    }
      }
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

int_block_sets_ptr = boost::make_shared<map<int, BlockData<SPACE_DIM>>>();
Range interface_eDges;
Range interface_tRis;

for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
  if (bit->getName().compare(0, 12, "INT_ELECTRIC") == 0) {
    const int id = bit->getMeshsetId();
    auto& block_data = (*int_block_sets_ptr)[id];{

    if (SPACE_DIM == 2) {
      CHKERR mField.get_moab().get_entities_by_dimension(bit->getMeshset(), SPACE_DIM - 1,
                                                         block_data.eDges, true);
      interface_eDges.merge(block_data.eDges);
    } else if (SPACE_DIM == 3) {
      CHKERR mField.get_moab().get_entities_by_dimension(bit->getMeshset(), SPACE_DIM-1,
                                                         block_data.tRis, true);
      interface_tRis.merge(block_data.tEts);
    }
    }
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
  

  if (SPACE_DIM == 2) {
  CHKERR mField.add_ents_to_finite_element_by_dim(interface_eDges, MBEDGE, "INTERFACE");
} else if (SPACE_DIM == 3) {
  CHKERR mField.add_ents_to_finite_element_by_dim(interface_tRis, MBTRI, "INTERFACE");
}

  CHKERR simpleInterface->defineFiniteElements();
  CHKERR simpleInterface->defineProblem(PETSC_TRUE);
  CHKERR simpleInterface->buildFields();
  CHKERR simpleInterface->buildFiniteElements();

  CHKERR mField.build_finite_elements("INTERFACE");
  CHKERR DMMoFEMAddElement(simpleInterface->getDM(), "INTERFACE");

  CHKERR simpleInterface->buildProblem();

  // CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}
//! [Setup problem]

//! [Boundary condition]
MoFEMErrorCode ElectrostaticHomogeneous::boundaryCondition() {
  MoFEMFunctionBegin;

  auto bc_mng = mField.getInterface<BcManager>();

  // Remove BCs from blockset name "BOUNDARY_CONDITION";
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
  common_data_ptr = boost::make_shared<DataAtIntegrationPts<SPACE_DIM>>(mField);
  auto add_domain_lhs_ops = [&](auto &pipeline) {
    pipeline.push_back(new OpBlockPermittivity<SPACE_DIM>(
        common_data_ptr, perm_block_sets_ptr, domainField));
  };

  add_domain_lhs_ops(pipeline_mng->getOpDomainLhsPipeline());


  { // Push operators to the Pipeline that is responsible for calculating LHS
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        pipeline_mng->getOpDomainLhsPipeline(), {H1});
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpDomainLhsMatrixK <SPACE_DIM>(domainField, domainField, common_data_ptr));
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
      using DomainEle =
          PipelineManager::ElementsAndOpsByDim<SPACE_DIM>::DomainEle;
      using DomainEleOp = DomainEle::UserDataOperator;
      using OpInternal = FormsIntegrators<DomainEleOp>::Assembly<
          PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<BASE_DIM, FIELD_DIM, SPACE_DIM>;

      auto grad_u_vals_ptr = boost::make_shared<MatrixDouble>();
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpCalculateScalarFieldGradient<SPACE_DIM>(domainField,
                                                        grad_u_vals_ptr));
      pipeline_mng->getOpDomainRhsPipeline().push_back(
          new OpInternal(domainField, grad_u_vals_ptr,
                         [](double, double, double) constexpr { return -1; }));
    };

    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        pipeline_mng->getOpDomainRhsPipeline(), {H1});
    set_values_to_bc_dofs(pipeline_mng->getDomainRhsFE());
    calculate_residual_from_set_values_on_bc(
        pipeline_mng->getOpDomainRhsPipeline());


    interface_rhs_fe = boost::shared_ptr<ForcesAndSourcesCore>(
        new EdgeElementForcesAndSourcesCore(mField));
        {
interface_rhs_fe->getOpPtrVector().push_back
(new OpBlockChargeDensity<SPACE_DIM>(common_data_ptr, int_block_sets_ptr, domainField));

      interface_rhs_fe->getOpPtrVector().push_back(
          new OpInterfaceRhsVectorF<SPACE_DIM>(domainField, common_data_ptr));
  
  }
 }
  MoFEMFunctionReturn(0);
}
//! [Assemble system]

//! [Set integration rules]
MoFEMErrorCode ElectrostaticHomogeneous::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto rule_lhs = [](int, int, int p) -> int { return 2 * (p - 1); };
  auto rule_rhs = [](int, int, int p) -> int { return p; };

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

  auto pipeline_mng = mField.getInterface<PipelineManager>();

  auto ksp_solver = pipeline_mng->createKSP();

  boost::shared_ptr<ForcesAndSourcesCore> null; ///< Null element does nothing
  CHKERR DMMoFEMKSPSetComputeRHS(simpleInterface->getDM(), "INTERFACE",
                                 interface_rhs_fe, null, null);

  auto post_proc_fe = boost::make_shared<PostProcVolEle>(mField);
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
//! [Solve system]

//! [Output results]
MoFEMErrorCode ElectrostaticHomogeneous::outputResults() {
  MoFEMFunctionBegin;

  auto pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();

  auto post_proc_fe = boost::make_shared<PostProcVolEle>(mField);

  auto det_ptr = boost::make_shared<VectorDouble>();
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

  // constexpr auto SPACE_DIM = 2; // dimension of problem

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
  CHKERR post_proc_fe->writeFile("out_resultGD.h5m");

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
