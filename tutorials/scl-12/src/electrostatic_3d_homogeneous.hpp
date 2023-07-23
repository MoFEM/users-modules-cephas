/**
 * @file poisson_3d_homogeneous.hpp
 * @example poisson_3d_homogeneous.hpp
 * @brief Operator for 3D poisson problem example
 * @date 2023-01-31
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __ELECTROSTATIC_3D_HOMOGENEOUS_HPP__
#define __ELECTROSTATIC_3D_HOMOGENEOUS_HPP__

#include <stdlib.h>
#include <BasicFiniteElements.hpp>
constexpr int BASE_DIM = 1;
constexpr int FIELD_DIM = 1;
const int SPACE_DIM = 3; 

using OpVolEle = MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator;                
using OpFaceEle = MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator; 
using EntData = EntitiesFieldData::EntData;

template <int SPACE_DIM>
struct VolBlockData {
  int iD;
  double epsPermit;
  Range tEts; ///< constrains elements in block set
};
template <int SPACE_DIM>
struct SurfBlockData {
  int iD;
  double sigma;
  Range tRis; ///
};
namespace Electrostatic3DHomogeneousOperators {


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

template <int SPACE_DIM> 
struct OpDomainLhsMatrixK : public OpVolEle {
public:
  OpDomainLhsMatrixK(std::string row_field_name, std::string col_field_name,
                     boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> common_data_ptr)
      : OpVolEle(row_field_name, col_field_name, OpVolEle::OPROWCOL),
        commonDataPtr(common_data_ptr) {
    sYmm = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;

    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();
    FTensor::Index<'i', SPACE_DIM> i;
    if (nb_row_dofs && nb_col_dofs) {

      locLhs.resize(nb_row_dofs, nb_col_dofs, false);
      locLhs.clear();

      // get element volume
      const double volume = getMeasure();

      // get number of integration points
      const int nb_integration_points = getGaussPts().size2();
      // get integration weights
      auto t_w = getFTensor0IntegrationWeight();

      // get derivatives of base functions on row
      auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL MATRIX
      for (int gg = 0; gg != nb_integration_points; gg++) {
        const double a = t_w * volume;

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
struct OpInterfaceRhsVectorF : public OpFaceEle {
public:
  OpInterfaceRhsVectorF(std::string field_name,
                        boost::shared_ptr<DataAtIntegrationPts<SPACE_DIM>> common_data_ptr)
      : OpFaceEle(field_name, OpFaceEle::OPROW),
        commonDataPtr(common_data_ptr) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    const int nb_dofs = data.getIndices().size();

    if (nb_dofs) {
      locRhs.resize(nb_dofs, false);
      locRhs.clear();

      // get element volume
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

        for (int rr = 0; rr != nb_dofs; rr++) {
          locRhs[rr] += t_base * a * commonDataPtr->chrgDens;

          ++t_base;
        }
        // move to the weight of the next integration point
        ++t_w;
        // move to the coordinates of the next integration point
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
struct OpBlockChargeDensity : public OpFaceEle {

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
template <int SPACE_DIM>
struct OpBlockPermittivity : public OpVolEle {

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
};     // namespace Electrostatic3DHomogeneousOperators

#endif //__ELECTROSTATIC_3D_HOMOGENEOUS_HPP__