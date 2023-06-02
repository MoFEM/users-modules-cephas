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

using VolEle = MoFEM::VolumeElementForcesAndSourcesCore;

using OpVolEle = MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator;

using FaceEle = MoFEM::FaceElementForcesAndSourcesCore;                     //
using OpFaceEle = MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator; //

using EntData = EntitiesFieldData::EntData;

namespace Electrostatic3DHomogeneousOperators {

FTensor::Index<'i', 3> i;

const double body_source = 5.;

struct OpDomainLhsMatrixK : public OpVolEle {
public:
  OpDomainLhsMatrixK(std::string row_field_name, std::string col_field_name,
                     double rel_permit)
      : OpVolEle(row_field_name, col_field_name, OpVolEle::OPROWCOL),
        relPermit(rel_permit) {
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
            locLhs(rr, cc) +=
                t_row_diff_base(i) * t_col_diff_base(i) * a * relPermit;

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
  double relPermit;
};

struct OpInterfaceRhsVectorF : public OpFaceEle {
public:
  OpInterfaceRhsVectorF(std::string field_name, double SrfcChrgDens)
      : OpFaceEle(field_name, OpFaceEle::OPROW), chrgDens(SrfcChrgDens) {}

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
        // double y = getGaussPts()(1, gg);
        double y = getCoordsAtGaussPts()(gg, 1);

        for (int rr = 0; rr != nb_dofs; rr++) {
          double charge_density = chrgDens;
          if (y < 0) {
            charge_density *= -1;
          }

          locRhs[rr] += t_base * charge_density * a;
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
  double chrgDens;
};
// struct OpNegativeGradient: public OpVolEle {
//   OpNegativeGradient(boost::shared_ptr<MoFEM::Types::MatrixDouble> gradU,
//                     boost::shared_ptr<MoFEM::Types::MatrixDouble> gradUNegative)
//       : OpVolEle(gradU(gradU), gradUNegative(gradUNegative) {}

//   MoFEMErrorCode doWork(EntityType type, const EntData &data, double *row_data,
//                         const double *col_data) {
//     MoFEMFunctionBegin;
//     const int nb_integration_points = getGaussPts().size2();
//     gradUNegative->resize(nb_integration_points());
//     gradUNegative->clear();

//       for (int gg = 0; gg != nb_integration_points; gg++) {
//       (gradUNegative)[gg] = -(gradU)[gg];
//     }

//     MoFEMFunctionReturn(0);
//   }

// private:
//   boost::shared_ptr<MoFEM::Types::MatrixDouble> gradU;
//   boost::shared_ptr<MoFEM::Types::MatrixDouble> gradUNegative;
// };

struct OpNegativeGradient : public ForcesAndSourcesCore::UserDataOperator {
public:
  OpNegativeGradient(boost::shared_ptr<MatrixDouble> grad_u_neg,
                     boost::shared_ptr<MatrixDouble> grad_u)
      : ForcesAndSourcesCore::UserDataOperator(NOSPACE, OPLAST),
        gradUNeg(grad_u_neg), gradU(grad_u) {}

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    const size_t nb_gauss_pts = getGaussPts().size2();
    gradUNeg->resize(3, nb_gauss_pts, false);
    gradUNeg->clear();

    for (size_t gg = 0; gg < nb_gauss_pts; gg++) {
      for (size_t i = 0; i < 3; i++) {
        (*gradUNeg)(i, gg) = -(*gradU)(i, gg);//gg->OpRow;
      }
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> gradUNeg;
  boost::shared_ptr<MatrixDouble> gradU;
};
// template <int space_dim>
// struct OpNegativeGradient
// : public ForcesAndSourcesCore::UserDataOperator {
//   OpNegativeGradient(boost::shared_ptr<MatrixDouble> grad_u_negative,
//                               boost::shared_ptr<MatrixDouble> grad_u)
//       : ForcesAndSourcesCore::UserDataOperator(),
//         gradUNegative(grad_u_negative), gradU(grad_u) {}

//   MoFEMErrorCode doWork(int side, EntityType type,
//                         DataForcesAndSourcesCore::EntData &data) {
//     MoFEMFunctionBegin;

//     const size_t nb_gauss_pts = getGaussPts().size2();
//     gradUNegative->resize(space_dim, nb_gauss_pts, false);
//     gradUNegative->clear();

//     auto t_grad_u = getFTensor1FromMat<space_dim>(*gradU);

//     auto t_negative_grad_u = getFTensor1FromMat<space_dim>(*gradUNegative);

//     FTensor::Index<'I', space_dim> I;

//     for (int gg = 0; gg != nb_gauss_pts; gg++) {
//       t_negative_grad_u(I) = -t_grad_u(I);

//       ++t_grad_u;
//       ++t_negative_grad_u;
//     }

//     MoFEMFunctionReturn(0);
//   }

// private:
//   boost::shared_ptr<MatrixDouble> gradUNegative;
//   boost::shared_ptr<MatrixDouble> gradU;
// };

};     // namespace Electrostatic3DHomogeneousOperators

#endif //__ELECTROSTATIC_3D_HOMOGENEOUS_HPP__