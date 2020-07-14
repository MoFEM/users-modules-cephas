#ifndef __POISSON3DHOMOGENEOUS_HPP__
#define __POISSON3DHOMOGENEOUS_HPP__

#include <stdlib.h>
#include <BasicFiniteElements.hpp>

using VolEle = MoFEM::VolumeElementForcesAndSourcesCore;

using OpVolEle = MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator;

using EntData = DataForcesAndSourcesCore::EntData;

namespace Poisson3DHomogeneousOperators {

FTensor::Index<'i', 3> i;

const double body_source = 1.;

struct OpDomainLhsMatrixK : public OpVolEle {
public:
  OpDomainLhsMatrixK(std::string row_field_name, std::string col_field_name)
      : OpVolEle(row_field_name, col_field_name, OpVolEle::OPROWCOL) {
    sYmm = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;

    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();

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
      auto t_row_diff_base = row_data.getFTensor1DiffN<3>();

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL MATRIX
      for (int gg = 0; gg != nb_integration_points; gg++) {
        const double a = t_w * volume;

        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          // get derivatives of base functions on column
          auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

          for (int cc = 0; cc != nb_col_dofs; cc++) {
            locLhs(rr, cc) += t_row_diff_base(i) * t_col_diff_base(i) * a;

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
};

struct OpDomainRhsVectorF : public OpVolEle {
public:
  OpDomainRhsVectorF(std::string field_name)
      : OpVolEle(field_name, OpVolEle::OPROW) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    const int nb_dofs = data.getIndices().size();

    if (nb_dofs) {
      locRhs.resize(nb_dofs, false);
      locRhs.clear();

      // get element volume
      const double volume = getMeasure();

      // get number of integration points
      const int nb_integration_points = getGaussPts().size2();
      // get integration weights
      auto t_w = getFTensor0IntegrationWeight();

      // get base function
      auto t_base = data.getFTensor0N();

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL VECTOR
      for (int gg = 0; gg != nb_integration_points; gg++) {
        const double a = t_w * volume;

        for (int rr = 0; rr != nb_dofs; rr++) {
          locRhs[rr] += t_base * body_source * a;

          // move to the next base function
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
};

}; // namespace Poisson3DHomogeneousOperators

#endif //__POISSON3DHOMOGENEOUS_HPP__