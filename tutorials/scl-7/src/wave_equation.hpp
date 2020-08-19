/**
 * \file wave_equation.hpp
 * \example wave_equation.hpp
 *
 * \brief Operators for time-dependent Wave Equation.
 *
 */
/* This file is part of MoFEM.
 * MoFEM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * MoFEM is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with MoFEM. If not, see <http://www.gnu.org/licenses/>. */

#ifndef __WAVEEQUATION_HPP__
#define __WAVEEQUATION_HPP__

#include <stdlib.h>
#include <BasicFiniteElements.hpp>

using FaceEle = MoFEM::FaceElementForcesAndSourcesCore;
using EdgeEle = MoFEM::EdgeElementForcesAndSourcesCore;

using OpFaceEle = MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;
using OpEdgeEle = MoFEM::EdgeElementForcesAndSourcesCore::UserDataOperator;

using EntData = DataForcesAndSourcesCore::EntData;

namespace WaveEquationOperators {

FTensor::Index<'i', 2> i;

typedef boost::function<double(const double, const double, const double,
                               const double)>
    ScalarFunc;

const double wave_speed = 1.;

struct DataAtGaussPoints {
  // This struct is for data required for the update of Newton's iteration

  VectorDouble fieldValue; // field value at integration point
  MatrixDouble fieldGrad;  // field gradient at integration point
  VectorDouble fieldDot;   // time derivative of field at integration point
};

/**
 * @brief Monitor solution
 *
 * This functions is called by TS solver at the end of each step. It is used
 * to output results to the hard drive.
 */
struct Monitor : public FEMethod {

  Monitor(SmartPetscObj<DM> &dm,
          boost::shared_ptr<PostProcFaceOnRefinedMesh> &post_proc)
      : dM(dm), postProc(post_proc){};

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  const int save_every_nth_step = 1;

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;
    if (ts_step % save_every_nth_step == 0) {
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc);
      CHKERR postProc->writeFile(
          "out_level_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
      // cerr << "\n \t Monitor ... \n";
    }
    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcFaceOnRefinedMesh> postProc;
};

// ****************************************************************
// OPERATORS FOR LEFT HAND SIDE
// ****************************************************************

/** \brief Integrate the domain tangent matrix (LHS), part UU

\f[
\left[\begin{array}{cc}
\sigma M_{U U} & -M_{U V} \\
K_{V U} & \sigma M_{V V}
\end{array}\right]\left\{\begin{array}{c}
\delta U \\
\delta V
\end{array}\right\}=-\left[\begin{array}{c}
R^{(1)} \\
R^{(2)}
\end{array}\right]
\f]

*/
struct OpTangentLhsUU : public OpFaceEle {
public:
  OpTangentLhsUU(std::string row_field_name, std::string col_field_name,
                 boost::shared_ptr<std::vector<bool>> boundary_marker = nullptr)
      : OpFaceEle(row_field_name, col_field_name, OpFaceEle::OPROWCOL),
        boundaryMarker(boundary_marker) {
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

      // get element area
      const double area = getMeasure();

      // get number of integration points
      const int nb_integration_points = getGaussPts().size2();
      // get integration weights
      auto t_w = getFTensor0IntegrationWeight();
      // get gradient of the field at integration points
      // auto t_field_grad = getFTensor1FromMat<2>(commonData->fieldGrad);

      // get base functions on row
      auto t_row_base = row_data.getFTensor0N();
      // get derivatives of base functions on row
      // auto t_row_diff_base = row_data.getFTensor1DiffN<2>();

      // The shift \f[ \sigma  = {\left. {\frac{{\partial u}}{{\partial \dot
      // u}}} \right|_{{u^n}}} \f]
      const double sigma = getFEMethod()->ts_a;

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL MATRIX
      for (int gg = 0; gg != nb_integration_points; gg++) {

        const double a = t_w * area;

        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          // get base functions on column
          auto t_col_base = col_data.getFTensor0N(gg, 0);
          // get derivatives of base functions on column
          // auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);

          for (int cc = 0; cc != nb_col_dofs; cc++) {

            // calculate components of the local matrix
            locLhs(rr, cc) += (sigma * t_row_base * t_col_base + 0.0 * 0.0) * a;

            // move to the next base functions on column
            ++t_col_base;
            // move to the derivatives of the next base functions on column
            // ++t_col_diff_base;
          }

          // move to the next base functions on row
          ++t_row_base;
          // move to the derivatives of the next base functions on row
          // ++t_row_diff_base;
        }

        // move to the weight of the next integration point
        ++t_w;
        // // move to the gradient of the field at the next integration point
        // ++t_field_grad;
      }

      // FILL VALUES OF THE GLOBAL MATRIX ENTRIES FROM THE LOCAL ONES

      // store original row indices
      auto row_indices = row_data.getIndices();
      // mark indices of boundary DOFs as -1
      if (boundaryMarker) {
        for (int r = 0; r != row_data.getIndices().size(); ++r) {
          if ((*boundaryMarker)[row_data.getLocalIndices()[r]]) {
            row_data.getIndices()[r] = -1;
          }
        }
      }
      // Fill value to local stiffness matrix ignoring boundary DOFs
      CHKERR MatSetValues(getTSB(), row_data, col_data, &locLhs(0, 0),
                          ADD_VALUES);
      // Revert back row indices to the original
      row_data.getIndices().data().swap(row_indices.data());

      // store original column row indices
      auto col_indices = col_data.getIndices();
      // mark indices of boundary DOFs as -1
      if (boundaryMarker) {
        for (int c = 0; c != col_data.getIndices().size(); ++c) {
          if ((*boundaryMarker)[col_data.getLocalIndices()[c]]) {
            col_data.getIndices()[c] = -1;
          }
        }
      }
      // Fill values of symmetric local stiffness matrix
      if (row_side != col_side || row_type != col_type) {
        transLocLhs.resize(nb_col_dofs, nb_row_dofs, false);
        noalias(transLocLhs) = trans(locLhs);
        CHKERR MatSetValues(getTSB(), col_data, row_data, &transLocLhs(0, 0),
                            ADD_VALUES);
      }
      // Revert back row indices to the original
      col_data.getIndices().data().swap(col_indices.data());
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<std::vector<bool>> boundaryMarker;
  boost::shared_ptr<DataAtGaussPoints> commonData;
  MatrixDouble locLhs, transLocLhs;
};

/** \brief Integrate the domain tangent matrix (LHS), part UV

\f[
\left[\begin{array}{cc}
\sigma M_{U U} & -M_{U V} \\
K_{V U} & \sigma M_{V V}
\end{array}\right]\left\{\begin{array}{c}
\delta U \\
\delta V
\end{array}\right\}=-\left[\begin{array}{c}
R^{(1)} \\
R^{(2)}
\end{array}\right]
\f]

*/
struct OpTangentLhsUV : public OpFaceEle {
public:
  OpTangentLhsUV(std::string row_field_name, std::string col_field_name,
                 boost::shared_ptr<std::vector<bool>> boundary_marker = nullptr)
      : OpFaceEle(row_field_name, col_field_name, OpFaceEle::OPROWCOL),
        boundaryMarker(boundary_marker) {
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

      // get element area
      const double area = getMeasure();

      // get number of integration points
      const int nb_integration_points = getGaussPts().size2();
      // get integration weights
      auto t_w = getFTensor0IntegrationWeight();
      // get gradient of the field at integration points
      // auto t_field_grad = getFTensor1FromMat<2>(commonData->fieldGrad);

      // get base functions on row
      auto t_row_base = row_data.getFTensor0N();
      // get derivatives of base functions on row
      // auto t_row_diff_base = row_data.getFTensor1DiffN<2>();

      // The shift \f[ \sigma  = {\left. {\frac{{\partial u}}{{\partial \dot
      // u}}} \right|_{{u^n}}} \f]
      const double sigma = getFEMethod()->ts_a;

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL MATRIX
      for (int gg = 0; gg != nb_integration_points; gg++) {

        const double a = t_w * area;

        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          // get base functions on column
          auto t_col_base = col_data.getFTensor0N(gg, 0);
          // get derivatives of base functions on column
          // auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);

          for (int cc = 0; cc != nb_col_dofs; cc++) {

            // calculate components of the local matrix
            locLhs(rr, cc) += -(t_row_base * t_col_base + 0.0 * 0.0) * a;

            // move to the next base functions on column
            ++t_col_base;
            // move to the derivatives of the next base functions on column
            // ++t_col_diff_base;
          }

          // move to the next base functions on row
          ++t_row_base;
          // move to the derivatives of the next base functions on row
          // ++t_row_diff_base;
        }

        // move to the weight of the next integration point
        ++t_w;
        // // move to the gradient of the field at the next integration point
        // ++t_field_grad;
      }

      // FILL VALUES OF THE GLOBAL MATRIX ENTRIES FROM THE LOCAL ONES

      // store original row indices
      auto row_indices = row_data.getIndices();
      // mark indices of boundary DOFs as -1
      if (boundaryMarker) {
        for (int r = 0; r != row_data.getIndices().size(); ++r) {
          if ((*boundaryMarker)[row_data.getLocalIndices()[r]]) {
            row_data.getIndices()[r] = -1;
          }
        }
      }
      // Fill value to local stiffness matrix ignoring boundary DOFs
      CHKERR MatSetValues(getTSB(), row_data, col_data, &locLhs(0, 0),
                          ADD_VALUES);
      // Revert back row indices to the original
      row_data.getIndices().data().swap(row_indices.data());

      // store original column row indices
      auto col_indices = col_data.getIndices();
      // mark indices of boundary DOFs as -1
      if (boundaryMarker) {
        for (int c = 0; c != col_data.getIndices().size(); ++c) {
          if ((*boundaryMarker)[col_data.getLocalIndices()[c]]) {
            col_data.getIndices()[c] = -1;
          }
        }
      }
      // Fill values of symmetric local stiffness matrix
      if (row_side != col_side || row_type != col_type) {
        transLocLhs.resize(nb_col_dofs, nb_row_dofs, false);
        noalias(transLocLhs) = trans(locLhs);
        CHKERR MatSetValues(getTSB(), col_data, row_data, &transLocLhs(0, 0),
                            ADD_VALUES);
      }
      // Revert back row indices to the original
      col_data.getIndices().data().swap(col_indices.data());
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<std::vector<bool>> boundaryMarker;
  boost::shared_ptr<DataAtGaussPoints> commonData;
  MatrixDouble locLhs, transLocLhs;
};

/** \brief Integrate the domain tangent matrix (LHS), part VU

\f[
\left[\begin{array}{cc}
\sigma M_{U U} & -M_{U V} \\
K_{V U} & \sigma M_{V V}
\end{array}\right]\left\{\begin{array}{c}
\delta U \\
\delta V
\end{array}\right\}=-\left[\begin{array}{c}
R^{(1)} \\
R^{(2)}
\end{array}\right]
\f]

*/
struct OpTangentLhsVU : public OpFaceEle {
public:
  OpTangentLhsVU(std::string row_field_name, std::string col_field_name,
                 boost::shared_ptr<std::vector<bool>> boundary_marker = nullptr)
      : OpFaceEle(row_field_name, col_field_name, OpFaceEle::OPROWCOL),
        boundaryMarker(boundary_marker) {
    sYmm = true;
    // cerr << "True ... \n";
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;
    // cerr << "Tangent ... \n";
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
      // auto t_field_grad = getFTensor1FromMat<2>(commonData->fieldGrad);

      // get base functions on row
      // auto t_row_base = row_data.getFTensor0N();
      // get derivatives of base functions on row
      auto t_row_diff_base = row_data.getFTensor1DiffN<2>();

      // The shift \f[ \sigma  = {\left. {\frac{{\partial u}}{{\partial \dot
      // u}}} \right|_{{u^n}}} \f]
      // const double sigma = getFEMethod()->ts_a;

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL MATRIX
      for (int gg = 0; gg != nb_integration_points; gg++) {

        const double a = t_w * area;

        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          // get base functions on column
          auto t_col_base = col_data.getFTensor0N(gg, 0);
          // get derivatives of base functions on column
          auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);

          for (int cc = 0; cc != nb_col_dofs; cc++) {

            // calculate components of the local matrix
            locLhs(rr, cc) +=
                (wave_speed * t_row_diff_base(i) * t_col_diff_base(i)) * a;

            // move to the next base functions on column
            // ++t_col_base;
            // move to the derivatives of the next base functions on column
            ++t_col_diff_base;
          }

          // move to the next base functions on row
          // ++t_row_base;
          // move to the derivatives of the next base functions on row
          ++t_row_diff_base;
        }

        // move to the weight of the next integration point
        ++t_w;
        // // move to the gradient of the field at the next integration point
        // ++t_field_grad;
      }

      // FILL VALUES OF THE GLOBAL MATRIX ENTRIES FROM THE LOCAL ONES

      // store original row indices
      auto row_indices = row_data.getIndices();
      // mark indices of boundary DOFs as -1
      if (boundaryMarker) {
        for (int r = 0; r != row_data.getIndices().size(); ++r) {
          if ((*boundaryMarker)[row_data.getLocalIndices()[r]]) {
            row_data.getIndices()[r] = -1;
          }
        }
      }
      // Fill value to local stiffness matrix ignoring boundary DOFs
      CHKERR MatSetValues(getTSB(), row_data, col_data, &locLhs(0, 0),
                          ADD_VALUES);
      // Revert back row indices to the original
      row_data.getIndices().data().swap(row_indices.data());

      // store original column row indices
      auto col_indices = col_data.getIndices();
      // mark indices of boundary DOFs as -1
      if (boundaryMarker) {
        for (int c = 0; c != col_data.getIndices().size(); ++c) {
          if ((*boundaryMarker)[col_data.getLocalIndices()[c]]) {
            col_data.getIndices()[c] = -1;
          }
        }
      }
      // Fill values of symmetric local stiffness matrix
      if (row_side != col_side || row_type != col_type) {
        transLocLhs.resize(nb_col_dofs, nb_row_dofs, false);
        noalias(transLocLhs) = trans(locLhs);
        CHKERR MatSetValues(getTSB(), col_data, row_data, &transLocLhs(0, 0),
                            ADD_VALUES);
      }
      // Revert back row indices to the original
      col_data.getIndices().data().swap(col_indices.data());
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<std::vector<bool>> boundaryMarker;
  MatrixDouble locLhs, transLocLhs;
};

/** \brief Integrate the domain tangent matrix (LHS), part VV

\f[
\left[\begin{array}{cc}
\sigma M_{U U} & -M_{U V} \\
K_{V U} & \sigma M_{V V}
\end{array}\right]\left\{\begin{array}{c}
\delta U \\
\delta V
\end{array}\right\}=-\left[\begin{array}{c}
R^{(1)} \\
R^{(2)}
\end{array}\right]
\f]

*/
struct OpTangentLhsVV : public OpFaceEle {
public:
  OpTangentLhsVV(std::string row_field_name, std::string col_field_name,
                 boost::shared_ptr<std::vector<bool>> boundary_marker = nullptr)
      : OpFaceEle(row_field_name, col_field_name, OpFaceEle::OPROWCOL),
        boundaryMarker(boundary_marker) {
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

      // get element area
      const double area = getMeasure();

      // get number of integration points
      const int nb_integration_points = getGaussPts().size2();
      // get integration weights
      auto t_w = getFTensor0IntegrationWeight();
      // get gradient of the field at integration points
      // auto t_field_grad = getFTensor1FromMat<2>(commonData->fieldGrad);

      // get base functions on row
      auto t_row_base = row_data.getFTensor0N();
      // get derivatives of base functions on row
      // auto t_row_diff_base = row_data.getFTensor1DiffN<2>();

      // The shift \f[ \sigma  = {\left. {\frac{{\partial u}}{{\partial \dot
      // u}}} \right|_{{u^n}}} \f]
      const double sigma = getFEMethod()->ts_a;

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL MATRIX
      for (int gg = 0; gg != nb_integration_points; gg++) {

        const double a = t_w * area;

        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          // get base functions on column
          auto t_col_base = col_data.getFTensor0N(gg, 0);
          // get derivatives of base functions on column
          // auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);

          for (int cc = 0; cc != nb_col_dofs; cc++) {

            // calculate components of the local matrix
            locLhs(rr, cc) += (sigma * t_row_base * t_col_base + 0.0 * 0.0) * a;

            // move to the next base functions on column
            ++t_col_base;
            // move to the derivatives of the next base functions on column
            // ++t_col_diff_base;
          }

          // move to the next base functions on row
          ++t_row_base;
          // move to the derivatives of the next base functions on row
          // ++t_row_diff_base;
        }

        // move to the weight of the next integration point
        ++t_w;
        // // move to the gradient of the field at the next integration point
        // ++t_field_grad;
      }

      // FILL VALUES OF THE GLOBAL MATRIX ENTRIES FROM THE LOCAL ONES

      // store original row indices
      auto row_indices = row_data.getIndices();
      // mark indices of boundary DOFs as -1
      if (boundaryMarker) {
        for (int r = 0; r != row_data.getIndices().size(); ++r) {
          if ((*boundaryMarker)[row_data.getLocalIndices()[r]]) {
            row_data.getIndices()[r] = -1;
          }
        }
      }
      // Fill value to local stiffness matrix ignoring boundary DOFs
      CHKERR MatSetValues(getTSB(), row_data, col_data, &locLhs(0, 0),
                          ADD_VALUES);
      // Revert back row indices to the original
      row_data.getIndices().data().swap(row_indices.data());

      // store original column row indices
      auto col_indices = col_data.getIndices();
      // mark indices of boundary DOFs as -1
      if (boundaryMarker) {
        for (int c = 0; c != col_data.getIndices().size(); ++c) {
          if ((*boundaryMarker)[col_data.getLocalIndices()[c]]) {
            col_data.getIndices()[c] = -1;
          }
        }
      }
      // Fill values of symmetric local stiffness matrix
      if (row_side != col_side || row_type != col_type) {
        transLocLhs.resize(nb_col_dofs, nb_row_dofs, false);
        noalias(transLocLhs) = trans(locLhs);
        CHKERR MatSetValues(getTSB(), col_data, row_data, &transLocLhs(0, 0),
                            ADD_VALUES);
      }
      // Revert back row indices to the original
      col_data.getIndices().data().swap(col_indices.data());
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<std::vector<bool>> boundaryMarker;
  boost::shared_ptr<DataAtGaussPoints> commonData;
  MatrixDouble locLhs, transLocLhs;
};

/** \brief Integrate the residual (vector RHS), part 1 (u)

\f[
\left[\begin{array}{cc}
\sigma M_{U U} & -M_{U V} \\
K_{V U} & \sigma M_{V V}
\end{array}\right]\left\{\begin{array}{c}
\delta U \\
\delta V
\end{array}\right\}=-\left[\begin{array}{c}
R^{(1)} \\
R^{(2)}
\end{array}\right]
\f]

*/
struct OpResidualRhsU : public OpFaceEle {
public:
  OpResidualRhsU(std::string field_name, ScalarFunc source_term_function,
                 boost::shared_ptr<DataAtGaussPoints> &common_data_u,
                 boost::shared_ptr<DataAtGaussPoints> &common_data_v,
                 boost::shared_ptr<std::vector<bool>> boundary_marker = nullptr)
      : OpFaceEle(field_name, OpFaceEle::OPROW),
        sourceTermFunc(source_term_function), commonDataU(common_data_u),
        commonDataV(common_data_v), boundaryMarker(boundary_marker) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    // cerr << "Function ... \n";
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
      auto t_v = getFTensor0FromVec(commonDataV->fieldValue);
      // get gradient of the field at integration points
      // auto t_field_grad = getFTensor1FromMat<2>(commonData->fieldGrad);
      // get time derivative of field at integration points
      auto t_field_u_dot = getFTensor0FromVec(commonDataU->fieldDot);

      // get base functions
      auto t_base = data.getFTensor0N();
      // get derivatives of base functions
      // auto t_diff_base = data.getFTensor1DiffN<2>();

      // get time
      const double time = getFEMethod()->ts_t;

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL VECTOR
      for (int gg = 0; gg != nb_integration_points; gg++) {

        const double a = t_w * area;
        double source_term =
            sourceTermFunc(t_coords(0), t_coords(1), t_coords(2), time);

        for (int rr = 0; rr != nb_dofs; rr++) {

          // calculate components of the local vector
          locRhs[rr] += (t_base * t_field_u_dot - t_base * t_v) * a;

          // move to the next base function
          ++t_base;
          // move to the derivatives of the next base function
          // ++t_diff_base;
        }

        // move to the weight of the next integration point
        ++t_w;
        // move to the coordinates of the next integration point
        ++t_coords;
        // move to field v at the next integration point
        ++t_v;
        // move to the gradient of the field at the next integration point
        // ++t_field_grad;
        // move to time derivative of the field at the next integration point
        ++t_field_u_dot;
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
      CHKERR VecSetOption(getTSf(), VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
      CHKERR VecSetValues(getTSf(), data, &*locRhs.begin(), ADD_VALUES);

      // revert back the indices
      data.getIndices().data().swap(row_indices.data());
    }

    MoFEMFunctionReturn(0);
  }

private:
  ScalarFunc sourceTermFunc;
  boost::shared_ptr<std::vector<bool>> boundaryMarker;
  boost::shared_ptr<DataAtGaussPoints> commonDataU;
  boost::shared_ptr<DataAtGaussPoints> commonDataV;
  VectorDouble locRhs;
};

/** \brief Integrate the residual (vector RHS), part 2 (v)

\f[
\left[\begin{array}{cc}
\sigma M_{U U} & -M_{U V} \\
K_{V U} & \sigma M_{V V}
\end{array}\right]\left\{\begin{array}{c}
\delta U \\
\delta V
\end{array}\right\}=-\left[\begin{array}{c}
R^{(1)} \\
R^{(2)}
\end{array}\right]
\f]

*/
struct OpResidualRhsV : public OpFaceEle {
public:
  OpResidualRhsV(std::string field_name, ScalarFunc source_term_function,
                 boost::shared_ptr<DataAtGaussPoints> &common_data_u,
                 boost::shared_ptr<DataAtGaussPoints> &common_data_v,
                 boost::shared_ptr<std::vector<bool>> boundary_marker = nullptr)
      : OpFaceEle(field_name, OpFaceEle::OPROW),
        sourceTermFunc(source_term_function), commonDataU(common_data_u),
        commonDataV(common_data_v), boundaryMarker(boundary_marker) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    // cerr << "Function ... \n";
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
      auto t_field_u_grad = getFTensor1FromMat<2>(commonDataU->fieldGrad);
      // get time derivative of field at integration points
      auto t_field_v_dot = getFTensor0FromVec(commonDataV->fieldDot);

      // get base functions
      auto t_base = data.getFTensor0N();
      // get derivatives of base functions
      auto t_diff_base = data.getFTensor1DiffN<2>();

      // get time
      const double time = getFEMethod()->ts_t;

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL VECTOR
      for (int gg = 0; gg != nb_integration_points; gg++) {

        const double a = t_w * area;
        double source_term =
            sourceTermFunc(t_coords(0), t_coords(1), t_coords(2), time);

        for (int rr = 0; rr != nb_dofs; rr++) {

          // calculate components of the local vector
          locRhs[rr] +=
              (t_base * t_field_v_dot +
               t_diff_base(i) * t_field_u_grad(i) * wave_speed - source_term) *
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
        // move to field v at the next integration point
        // ++t_v;
        // move to the gradient of the field at the next integration point
        ++t_field_u_grad;
        // move to time derivative of the field at the next integration point
        ++t_field_v_dot;
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
      CHKERR VecSetOption(getTSf(), VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
      CHKERR VecSetValues(getTSf(), data, &*locRhs.begin(), ADD_VALUES);

      // revert back the indices
      data.getIndices().data().swap(row_indices.data());
    }

    MoFEMFunctionReturn(0);
  }

private:
  ScalarFunc sourceTermFunc;
  boost::shared_ptr<std::vector<bool>> boundaryMarker;
  boost::shared_ptr<DataAtGaussPoints> commonDataU;
  boost::shared_ptr<DataAtGaussPoints> commonDataV;
  VectorDouble locRhs;
};

/** \brief Integrate LHS of the boundary elements using Least Squares approx.,
 * This operator will be for both u and v fields

\f[
\begin{aligned}
&\delta {\bf U}=g({\bf x}, t)-u({\bf x}, t) \text { on } \partial \Omega\\
&\delta {\bf V}=\frac{\partial g({\bf x}, t)}{\partial t}-v({\bf x}, t) \text {
on } \partial \Omega\\
&\left[\begin{array}{cc}
{\bf M}_{U U} & {\bf 0} \\
{\bf 0} & {\bf M}_{V V}
\end{array}\right]\left\{\begin{array}{l}
\delta {\bf U} \\
\delta {\bf V}
\end{array}\right\}=-\left[\begin{array}{c}
{\bf R}_{U} \\
{\bf R}_{V}
\end{array}\right]
\end{aligned}
\f]

*/
struct OpBoundaryLhs : public OpEdgeEle {
public:
  OpBoundaryLhs(std::string row_field_name, std::string col_field_name)
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
      CHKERR MatSetValues(getTSB(), row_data, col_data, &locBoundaryLhs(0, 0),
                          ADD_VALUES);

      if (row_side != col_side || row_type != col_type) {
        transLocBoundaryLhs.resize(nb_col_dofs, nb_row_dofs, false);
        noalias(transLocBoundaryLhs) = trans(locBoundaryLhs);
        CHKERR MatSetValues(getTSB(), col_data, row_data,
                            &transLocBoundaryLhs(0, 0), ADD_VALUES);
      }
    }

    MoFEMFunctionReturn(0);
  }

private:
  MatrixDouble locBoundaryLhs, transLocBoundaryLhs;
};

/** \brief Integrate RHS of the boundary elements using Least Squares approx.,
 * This operator will be for both u and v fields

\f[
\begin{aligned}
&\delta {\bf U}=g({\bf x}, t)-u({\bf x}, t) \text { on } \partial \Omega\\
&\delta {\bf V}=\frac{\partial g({\bf x}, t)}{\partial t}-v({\bf x}, t) \text {
on } \partial \Omega\\
&\left[\begin{array}{cc}
{\bf M}_{U U} & {\bf 0} \\
{\bf 0} & {\bf M}_{V V}
\end{array}\right]\left\{\begin{array}{l}
\delta {\bf U} \\
\delta {\bf V}
\end{array}\right\}=-\left[\begin{array}{c}
{\bf R}_{U} \\
{\bf R}_{V}
\end{array}\right]
\end{aligned}
\f]

*/
struct OpBoundaryRhs : public OpEdgeEle {
public:
  OpBoundaryRhs(std::string field_name, ScalarFunc boundary_function,
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
      auto t_u = getFTensor0FromVec(commonData->fieldValue);

      // get base function
      auto t_base = data.getFTensor0N();

      // get time
      const double time = getFEMethod()->ts_t;

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL VECTOR
      for (int gg = 0; gg != nb_integration_points; gg++) {
        const double a = t_w * edge;
        double boundary_term =
            boundaryFunc(t_coords(0), t_coords(1), t_coords(2), time);

        // calculate the local vector
        for (int rr = 0; rr != nb_dofs; rr++) {
          locBoundaryRhs[rr] -= t_base * (boundary_term - t_u) * a;

          // move to the next base function
          ++t_base;
        }

        // move to the weight of the next integration point
        ++t_w;
        // move to the coordinates of the next integration point
        ++t_coords;
        // move to the solution (field value) at the next integration point
        ++t_u;
      }

      // FILL VALUES OF LOCAL VECTOR ENTRIES TO THE GLOBAL VECTOR
      CHKERR VecSetValues(getTSf(), data, &*locBoundaryRhs.begin(), ADD_VALUES);
    }

    MoFEMFunctionReturn(0);
  }

private:
  ScalarFunc boundaryFunc;
  boost::shared_ptr<DataAtGaussPoints> commonData;
  VectorDouble locBoundaryRhs;
};

}; // namespace WaveEquationOperators

#endif //__WAVEEQUATION_HPP__