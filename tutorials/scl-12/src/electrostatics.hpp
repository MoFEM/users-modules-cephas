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

using intElementForcesAndSourcesCore = intPostproc<SPACE_DIM>::intEle;

using OpDomainLhsMatrixK = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpGradGrad<BASE_DIM, FIELD_DIM, SPACE_DIM>;

using OpInterfaceRhsVectorF = FormsIntegrators<IntEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<BASE_DIM, FIELD_DIM>;

static char help[] = "...\n\n";

struct BlockData {
  int iD;
  double sigma;
  double epsPermit;
  Range blockInterfaces;
  Range blockDomains;
};
struct DataAtIntegrationPts {

  SmartPetscObj<Vec> petscVec;
  double blockPermittivity;
  double chrgDens;
  DataAtIntegrationPts(MoFEM::Interface &m_field) {
    blockPermittivity = 0;
    chrgDens = 0;
  }
};

template <int SPACE_DIM> struct OpAlpha : public IntEleOp {
public:
  OpAlpha(std::string field_name, boost::shared_ptr<MatrixDouble> grad_ptr,
          SmartPetscObj<Vec> alpha_vec, boost::shared_ptr<Range> ents_ptr)
      : IntEleOp(field_name, SideEleOp::OPROW), gradPtr(grad_ptr),
        petscVec(alpha_vec), entsPtr(ents_ptr) {
    std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
    doEntities[MBVERTEX] = true;
  }

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    FTensor::Index<'i', SPACE_DIM> i;

    const auto fe_ent = getFEEntityHandle();
    const auto nb_gauss_pts = getGaussPts().size2();

    auto t_field_grad = getFTensor1FromMat<SPACE_DIM>(*(gradPtr));
    auto t_w = getFTensor0IntegrationWeight();
    auto t_normal = getFTensor1NormalsAtGaussPts();
    const double area = getMeasure();
    double alphaPart = 0;

    for (const auto &entity : *entsPtr) {
      if (entity == fe_ent) {

        for (int gg = 0; gg != nb_gauss_pts; gg++) {
          FTensor::Tensor1<double, SPACE_DIM> t_r;
          t_r(i) = t_normal(i);
          t_r.normalize();
          alphaPart += -(t_field_grad(i) * t_r(i)) * t_w * area;
          // std::cout << alphaPart << std::endl;
          ++t_field_grad;
          ++t_normal;
          ++t_w;
        }
      }
    }
    int index = 0;
    CHKERR ::VecSetValues(petscVec, 1, &index, &alphaPart, ADD_VALUES);

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> gradPtr;
  SmartPetscObj<Vec> petscVec;
  boost::shared_ptr<Range> entsPtr;
};

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
struct OpBlockChargeDensity : public IntEleOp {
  OpBlockChargeDensity(
      boost::shared_ptr<DataAtIntegrationPts> common_data_ptr,
      boost::shared_ptr<std::map<int, BlockData>> int_block_sets_ptr,
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
  boost::shared_ptr<DataAtIntegrationPts> commonDataPtr;
  boost::shared_ptr<std::map<int, BlockData>> intBlockSetsPtr;
};

struct OpBlockPermittivity : public DomainEleOp {

  OpBlockPermittivity(
      boost::shared_ptr<DataAtIntegrationPts> common_data_ptr,
      boost::shared_ptr<map<int, BlockData>> perm_block_sets_ptr,
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
  boost::shared_ptr<map<int, BlockData>> permBlockSetsPtr;
  boost::shared_ptr<DataAtIntegrationPts> commonDataPtr;
};
