#ifndef __MINIMALSURFACE_HPP__
#define __MINIMALSURFACE_HPP__

#include <stdlib.h>
#include <BasicFiniteElements.hpp>

using FaceEle = MoFEM::FaceElementForcesAndSourcesCore;
using EdgeEle = MoFEM::EdgeElementForcesAndSourcesCore;

using OpFaceEle = MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;
using OpEdgeEle = MoFEM::EdgeElementForcesAndSourcesCore::UserDataOperator;

using EntData = DataForcesAndSourcesCore::EntData;

namespace MinimalSurfaceOperators {

FTensor::Index<'i', 2> i;

typedef boost::function<double(const double, const double, const double)>
    ScalarFunc;

struct DataAtGaussPoints {
  // This struct is for data required for the update of Newton's iteration

  VectorDouble fieldValue;
  MatrixDouble fieldGrad;
};

/** \brief Integrate the domain tangent matrix (LHS)

\f[
\sum\limits_j {\left[ {\int\limits_{{\Omega _e}} {\left( {{a_n}\nabla {\phi _i}
\cdot \nabla {\phi _j} - a_n^3\nabla {\phi _i}\left( {\nabla u \cdot \nabla
{\phi _j}} \right)\nabla u} \right)d{\Omega _e}} } \right]\delta {U_j}}  =
\int\limits_{{\Omega _e}} {{\phi _i}fd{\Omega _e}}  - \int\limits_{{\Omega _e}}
{\nabla {\phi _i}{a_n}\nabla ud{\Omega _e}} \\
{a_n} = \frac{1}{{{{\left( {1 +
{{\left| {\nabla u} \right|}^2}} \right)}^{\frac{1}{2}}}}}
\f]

*/
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
      // get gradient of the field at integration points
      auto t_field_grad = getFTensor1FromMat<2>(commonData->fieldGrad);

      // get derivatives of base functions on row
      auto t_row_diff_base = row_data.getFTensor1DiffN<2>();

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL MATRIX
      for (int gg = 0; gg != nb_integration_points; gg++) {

        const double a = t_w * area;
        const double an = 1. / sqrt(1 + t_field_grad(i) * t_field_grad(i));

        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          // get derivatives of base functions on column
          auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);

          for (int cc = 0; cc != nb_col_dofs; cc++) {

            // calculate components of the local matrix
            locLhs(rr, cc) +=
                (t_row_diff_base(i) * t_col_diff_base(i)) * an * a -
                t_row_diff_base(i) * (t_field_grad(i) * t_col_diff_base(i)) *
                    t_field_grad(i) * an * an * an * a;

            // move to the derivatives of the next base functions on column
            ++t_col_diff_base;
          }

          // move to the derivatives of the next base functions on row
          ++t_row_diff_base;
        }

        // move to the weight of the next integration point
        ++t_w;
        // move to the gradient of the field at the next integration point
        ++t_field_grad;
      }

      // FILL VALUES OF THE GLOBAL MATRIX ENTRIES FROM THE LOCAL ONES

      // store original row indices
      auto row_indices = row_data.getIndices();
      // mark the boundary DOFs as -1 to avoid filling values of those DOFs
      if (boundaryMarker) {
        for (int r = 0; r != row_data.getIndices().size(); ++r) {
          if ((*boundaryMarker)[row_data.getLocalIndices()[r]]) {
            row_data.getIndices()[r] = -1;
          }
        }
      }
      // fill values to global stiffness matrix ignoring boundary DOFs
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

/** \brief Integrate the domain residual vector (RHS)

\f[
\sum\limits_j {\left[ {\int\limits_{{\Omega _e}} {\left( {{a_n}\nabla {\phi _i}
\cdot \nabla {\phi _j} - a_n^3\nabla {\phi _i}\left( {\nabla u \cdot \nabla
{\phi _j}} \right)\nabla u} \right)d{\Omega _e}} } \right]\delta {U_j}}  =
\int\limits_{{\Omega _e}} {{\phi _i}fd{\Omega _e}}  - \int\limits_{{\Omega _e}}
{\nabla {\phi _i}{a_n}\nabla ud{\Omega _e}} \\
{a_n} = \frac{1}{{{{\left( {1 +
{{\left| {\nabla u} \right|}^2}} \right)}^{\frac{1}{2}}}}}
\f]

*/
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
      // get gradient of the field at integration points
      auto t_field_grad = getFTensor1FromMat<2>(commonData->fieldGrad);

      // get base functions
      auto t_base = data.getFTensor0N();
      // get derivatives of base functions
      auto t_diff_base = data.getFTensor1DiffN<2>();

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL VECTOR
      for (int gg = 0; gg != nb_integration_points; gg++) {

        const double a = t_w * area;
        double source_term =
            sourceTermFunc(t_coords(0), t_coords(1), t_coords(2));
        const double an = 1. / sqrt(1 + t_field_grad(i) * t_field_grad(i));

        for (int rr = 0; rr != nb_dofs; rr++) {

          // calculate components of the local vector
          // remember to use -= here due to PETSc consideration of Residual Vec
          locRhs[rr] -=
              (t_base * source_term - t_diff_base(i) * t_field_grad(i)) * an *
              a;

          // move to the next base function
          ++t_base;
          // move to the derivatives of the next base function
          ++t_diff_base;
        }

        // move to the weight of the next integration point
        ++t_w;
        // move to the coordinates of the next integration point
        ++t_coords;
        // move to the gradient of the field at the next integration point
        ++t_field_grad;
      }

      // FILL VALUES OF THE GLOBAL VECTOR ENTRIES FROM THE LOCAL ONES

      auto row_indices = data.getIndices();
      // Mark the boundary DOFs and fill only domain DOFs
      if (boundaryMarker) {
        for (int r = 0; r != data.getIndices().size(); ++r) {
          if ((*boundaryMarker)[data.getLocalIndices()[r]]) {
            data.getIndices()[r] = -1;
          }
        }
      }
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

/** \brief Integrate the boundary tangent matrix (LHS)

\f[
\sum\limits_j {\left( {\int\limits_{\partial {\Omega _e}} {{\phi _i} \cdot {\phi
_j}d\partial {\Omega _e}} } \right)\delta {U_j}}  = \int\limits_{\partial
{\Omega _e}} {{\phi _i}\left( {\overline u  - u} \right)d\partial {\Omega _e}}
\f]

*/
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

    if (nb_row_dofs && nb_col_dofs) {

      locBoundaryLhs.resize(nb_row_dofs, nb_col_dofs, false);
      locBoundaryLhs.clear();

      // get edge length
      const double edge = getMeasure();

      // get number of integration points
      const int nb_integration_points = getGaussPts().size2();
      // get integration weights
      auto t_w = getFTensor0IntegrationWeight();

      // get base functions on row
      auto t_row_base = row_data.getFTensor0N();

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL MATRIX
      for (int gg = 0; gg != nb_integration_points; gg++) {

        const double a = t_w * edge;

        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          // get base functions on column
          auto t_col_base = col_data.getFTensor0N(gg, 0);

          for (int cc = 0; cc != nb_col_dofs; cc++) {

            // calculate components of the local matrix
            locBoundaryLhs(rr, cc) += t_row_base * t_col_base * a;

            // move to the next base functions on column
            ++t_col_base;
          }

          // base functions on row
          ++t_row_base;
        }
        ++t_w; // move to the next integration weight
      }

      // FILL VALUES OF THE GLOBAL MATRIX ENTRIES FROM THE LOCAL ONES

      CHKERR MatSetValues(getSNESB(), row_data, col_data, &locBoundaryLhs(0, 0),
                          ADD_VALUES);
      if (row_side != col_side || row_type != col_type) {
        transLocBoundaryLhs.resize(nb_col_dofs, nb_row_dofs, false);
        noalias(transLocBoundaryLhs) = trans(locBoundaryLhs);
        CHKERR MatSetValues(getSNESB(), col_data, row_data,
                            &transLocBoundaryLhs(0, 0), ADD_VALUES);
      }
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<DataAtGaussPoints> commonData;
  MatrixDouble locBoundaryLhs, transLocBoundaryLhs;
};

/** \brief Integrate the boundary residual vector (RHS)

\f[
\sum\limits_j {\left( {\int\limits_{\partial {\Omega _e}} {{\phi _i} \cdot {\phi
_j}d\partial {\Omega _e}} } \right)\delta {U_j}}  = \int\limits_{\partial
{\Omega _e}} {{\phi _i}\left( {\overline u  - u} \right)d\partial {\Omega _e}}
\f]

*/
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

      // get element area
      const double edge = getMeasure();

      // get number of integration points
      const int nb_integration_points = getGaussPts().size2();
      // get integration weights
      auto t_w = getFTensor0IntegrationWeight();
      // get coordinates of integration points
      auto t_coords = getFTensor1CoordsAtGaussPts();
      // get field values at integration points
      auto t_field = getFTensor0FromVec(commonData->fieldValue);

      // get base functions
      auto t_base = data.getFTensor0N();

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL VECTOR
      for (int gg = 0; gg != nb_integration_points; gg++) {

        const double a = t_w * edge;
        double boundary_term =
            boundaryFunc(t_coords(0), t_coords(1), t_coords(2));

        for (int rr = 0; rr != nb_dofs; rr++) {

          // calculate components of the local vector
          // remember to use -= here due to PETSc consideration of Residual Vec
          locBoundaryRhs[rr] -= t_base * (boundary_term - t_field) * a;

          // move to the next base function
          ++t_base;
        }

        // move to the weight of the next integration point
        ++t_w;
        // move to the coordinates of the next integration point
        ++t_coords;
        // move to the field values at the next integration point
        ++t_field;
      }

      // FILL VALUES OF THE GLOBAL VECTOR ENTRIES FROM THE LOCAL ONES

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

}; // namespace MinimalSurfaceOperators

#endif //__MINIMALSURFACE_HPP__