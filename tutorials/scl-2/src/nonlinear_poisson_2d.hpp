#ifndef __NONLINEARPOISSON2D_HPP__
#define __NONLINEARPOISSON2D_HPP__

#include <stdlib.h>
#include <BasicFiniteElements.hpp>

using FaceEle = MoFEM::FaceElementForcesAndSourcesCore;
using EdgeEle = MoFEM::EdgeElementForcesAndSourcesCore;

using OpFaceEle = MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;
using OpEdgeEle = MoFEM::EdgeElementForcesAndSourcesCore::UserDataOperator;

using EntData = DataForcesAndSourcesCore::EntData;

namespace NonlinearPoissonOps {

FTensor::Index<'i', 2> i;

typedef boost::function<double(const double, const double, const double)>
    ScalarFunc;

struct DataAtGaussPoints {
  // This struct is for data required for the update of Newton's iteration

  VectorDouble fieldValue;
  MatrixDouble fieldGrad;
};

struct OpDomainTangentMatrix : public OpFaceEle {
public:
  OpDomainTangentMatrix(
      std::string row_field_name, std::string col_field_name,
      boost::shared_ptr<DataAtGaussPoints> &common_data,
      boost::shared_ptr<std::vector<bool>> boundary_marker = nullptr)
      : OpFaceEle(row_field_name, col_field_name, OpFaceEle::OPROWCOL),
        commonData(common_data), boundaryMarker(boundary_marker) {
    sYmm = false;
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

      // get element area
      const double area = getMeasure();

      // get number of integration points
      const int nb_integration_points = getGaussPts().size2();
      // get integration weights
      auto t_w = getFTensor0IntegrationWeight();
      // get solution (field value) at integration points
      auto t_field = getFTensor0FromVec(commonData->fieldValue);
      // get gradient of field at integration points
      auto t_field_grad = getFTensor1FromMat<2>(commonData->fieldGrad);

      // get base functions on row
      auto t_row_base = row_data.getFTensor0N();
      // get derivatives of base functions on row
      auto t_row_diff_base = row_data.getFTensor1DiffN<2>();

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL MATRIX
      for (int gg = 0; gg != nb_integration_points; gg++) {
        const double a = t_w * area;

        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          // get base functions on column
          auto t_col_base = col_data.getFTensor0N(gg, 0);
          // get derivatives of base functions on column
          auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);

          for (int cc = 0; cc != nb_col_dofs; cc++) {
            locLhs(rr, cc) += (((1 + t_field * t_field) * t_row_diff_base(i) *
                                t_col_diff_base(i)) +
                               (2.0 * t_field * t_field_grad(i) *
                                t_row_diff_base(i) * t_col_base)) *
                              a;

            // move to the next base functions on column
            ++t_col_base;
            // move to the derivatives of the next base function on column
            ++t_col_diff_base;
          }

          // move to the next base function on row
          ++t_row_base;
          // move to the derivatives of the next base function on row
          ++t_row_diff_base;
        }

        // move to the weight of the next integration point
        ++t_w;
        // move to the solution (field value) at the next integration point
        ++t_field;
        // move to the gradient of field value at the next integration point
        ++t_field_grad;
      }

      // FILL VALUES OF LOCAL MATRIX ENTRIES TO THE GLOBAL MATRIX

      // store original row indices
      auto row_indices = row_data.getIndices();
      // mark the boundary DOFs (as -1) and fill only domain DOFs
      if (boundaryMarker) {
        for (int r = 0; r != row_data.getIndices().size(); ++r) {
          if ((*boundaryMarker)[row_data.getLocalIndices()[r]]) {
            row_data.getIndices()[r] = -1;
          }
        }
      }
      // fill value to local stiffness matrix ignoring boundary DOFs
      CHKERR MatSetValues(getSNESB(), row_data, col_data, &locLhs(0, 0),
                          ADD_VALUES);
      // revert back row indices to the original
      row_data.getIndices().data().swap(row_indices.data());
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<DataAtGaussPoints> commonData;
  boost::shared_ptr<std::vector<bool>> boundaryMarker;
  MatrixDouble locLhs;
};

struct OpDomainResidualVector : public OpFaceEle {
public:
  OpDomainResidualVector(
      std::string field_name, ScalarFunc source_term_function,
      boost::shared_ptr<DataAtGaussPoints> &common_data,
      boost::shared_ptr<std::vector<bool>> boundary_marker = nullptr)
      : OpFaceEle(field_name, OpFaceEle::OPROW),
        sourceTermFunc(source_term_function), commonData(common_data),
        boundaryMarker(boundary_marker) {}

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
      // get coordinates of the integration point
      auto t_coords = getFTensor1CoordsAtGaussPts();
      // get solution (field value) at integration point
      auto t_field = getFTensor0FromVec(commonData->fieldValue);
      // get gradient of field value of integration point
      auto t_field_grad = getFTensor1FromMat<2>(commonData->fieldGrad);

      // get base function
      auto t_base = data.getFTensor0N();
      // get derivatives of base function
      auto t_grad_base = data.getFTensor1DiffN<2>();

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL VECTOR
      for (int gg = 0; gg != nb_integration_points; gg++) {
        const double a = t_w * area;
        double body_source =
            sourceTermFunc(t_coords(0), t_coords(1), t_coords(2));

        // calculate the local vector
        for (int rr = 0; rr != nb_dofs; rr++) {
          locRhs[rr] -=
              (t_base * body_source -
               t_grad_base(i) * t_field_grad(i) * (1 + t_field * t_field)) *
              a;

          // move to the next base function
          ++t_base;
          // move to the derivatives of the next base function
          ++t_grad_base;
        }

        // move to the weight of the next integration point
        ++t_w;
        // move to the coordinates of the next integration point
        ++t_coords;
        // move to the solution (field value) at the next integration point
        ++t_field;
        // move to the gradient of field value at the next integration point
        ++t_field_grad;
      }

      // FILL VALUES OF LOCAL VECTOR ENTRIES TO THE GLOBAL VECTOR

      // store original row indices
      auto row_indices = data.getIndices();
      // mark the boundary DOFs (as -1) and fill only domain DOFs
      if (boundaryMarker) {
        for (int r = 0; r != data.getIndices().size(); ++r) {
          if ((*boundaryMarker)[data.getLocalIndices()[r]]) {
            data.getIndices()[r] = -1;
          }
        }
      }
      // fill value to local vector ignoring boundary DOFs
      CHKERR VecSetOption(getSNESf(), VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
      CHKERR VecSetValues(getSNESf(), data, &*locRhs.begin(), ADD_VALUES);
      // revert back the indices
      data.getIndices().data().swap(row_indices.data());
    }

    MoFEMFunctionReturn(0);
  }

private:
  ScalarFunc sourceTermFunc;
  boost::shared_ptr<DataAtGaussPoints> commonData;
  boost::shared_ptr<std::vector<bool>> boundaryMarker;
  VectorDouble locRhs;
};

struct OpBoundaryTangentMatrix : public OpEdgeEle {
public:
  OpBoundaryTangentMatrix(std::string row_field_name,
                          std::string col_field_name)
      : OpEdgeEle(row_field_name, col_field_name, OpEdgeEle::OPROWCOL) {
    sYmm = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;

    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();
    // cerr << nb_row_dofs;

    if (nb_row_dofs && nb_col_dofs) {
      locBoundaryLhs.resize(nb_row_dofs, nb_col_dofs, false);
      locBoundaryLhs.clear();

      // get (boundary) element length
      const double edge = getMeasure();

      // get number of integration points
      const int nb_integration_points = getGaussPts().size2();
      // get integration weights
      auto t_w = getFTensor0IntegrationWeight();

      // get base function on row
      auto t_row_base = row_data.getFTensor0N();

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL MATRIX
      for (int gg = 0; gg != nb_integration_points; gg++) {
        const double a = t_w * edge;

        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          // get base function on column
          auto t_col_base = col_data.getFTensor0N(gg, 0);

          for (int cc = 0; cc != nb_col_dofs; cc++) {
            locBoundaryLhs(rr, cc) += t_row_base * t_col_base * a;

            // move to the next base function on column
            ++t_col_base;
          }

          // move to the next base function on row
          ++t_row_base;
        }

        // move to the weight of the next integration point
        ++t_w;
      }

      // FILL VALUES OF LOCAL MATRIX ENTRIES TO THE GLOBAL MATRIX
      CHKERR MatSetValues(getSNESB(), row_data, col_data, &locBoundaryLhs(0, 0),
                          ADD_VALUES);

      if (row_side != col_side || row_type != col_type) {
        transLocBoundaryLhs.resize(nb_col_dofs, nb_row_dofs, false);
        noalias(transLocBoundaryLhs) = trans(locBoundaryLhs);
        CHKERR MatSetValues(getSNESB(), col_data, row_data,
                            &transLocBoundaryLhs(0, 0), ADD_VALUES);
      }
      // cerr << locBoundaryLhs << endl;
      // cerr << transLocBoundaryLhs << endl;
    }

    MoFEMFunctionReturn(0);
  }

private:
  MatrixDouble locBoundaryLhs, transLocBoundaryLhs;
};

struct OpBoundaryResidualVector : public OpEdgeEle {
public:
  OpBoundaryResidualVector(std::string field_name, ScalarFunc boundary_function,
                           boost::shared_ptr<DataAtGaussPoints> &common_data)
      : OpEdgeEle(field_name, OpEdgeEle::OPROW),
        boundaryFunc(boundary_function), commonData(common_data) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    const int nb_dofs = data.getIndices().size();

    if (nb_dofs) {

      locBoundaryRhs.resize(nb_dofs, false);
      locBoundaryRhs.clear();

      // get (boundary) element length
      const double edge = getMeasure();

      // get number of integration points
      const int nb_integration_points = getGaussPts().size2();
      // get integration weights
      auto t_w = getFTensor0IntegrationWeight();
      // get coordinates at integration point
      auto t_coords = getFTensor1CoordsAtGaussPts();
      // get solution (field value) at integration point
      auto t_field = getFTensor0FromVec(commonData->fieldValue);

      // get base function
      auto t_base = data.getFTensor0N();

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL VECTOR
      for (int gg = 0; gg != nb_integration_points; gg++) {
        const double a = t_w * edge;
        double boundary_term =
            boundaryFunc(t_coords(0), t_coords(1), t_coords(2));

        // calculate the local vector
        for (int rr = 0; rr != nb_dofs; rr++) {
          locBoundaryRhs[rr] -= t_base * (boundary_term - t_field) * a;

          // move to the next base function
          ++t_base;
        }

        // move to the weight of the next integration point
        ++t_w;
        // move to the coordinates of the next integration point
        ++t_coords;
        // move to the solution (field value) at the next integration point
        ++t_field;
      }

      // FILL VALUES OF LOCAL VECTOR ENTRIES TO THE GLOBAL VECTOR
      CHKERR VecSetValues(getSNESf(), data, &*locBoundaryRhs.begin(),
                          ADD_VALUES);
    }

    MoFEMFunctionReturn(0);
  }

private:
  ScalarFunc boundaryFunc;
  boost::shared_ptr<DataAtGaussPoints> commonData;
  VectorDouble locBoundaryRhs;
};

}; // namespace NonlinearPoissonOps

#endif //__NONLINEARPOISSON2D_HPP__