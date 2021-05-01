#ifndef __POISSON2DNONHOMOGENEOUS_HPP__
#define __POISSON2DNONHOMOGENEOUS_HPP__

#include <stdlib.h>
#include <BasicFiniteElements.hpp>

using FaceEle = MoFEM::FaceElementForcesAndSourcesCore;
using EdgeEle = MoFEM::EdgeElementForcesAndSourcesCore;

using OpFaceEle = MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;
using OpEdgeEle = MoFEM::EdgeElementForcesAndSourcesCore::UserDataOperator;

using EntData = DataForcesAndSourcesCore::EntData;

namespace Thermal2DOperators {

constexpr auto VecSetValues = MoFEM::VecSetValues<MoFEM::EssentialBcStorage>;
constexpr auto MatSetValues = MoFEM::MatSetValues<MoFEM::EssentialBcStorage>;

FTensor::Index<'i', 2> i;

// const double body_source = 10;
typedef boost::function<double(const double, const double, const double)>
    ScalarFunc;

struct OpDomainLhs : public OpFaceEle {
public:
  OpDomainLhs(std::string row_field_name, std::string col_field_name)
      : OpFaceEle(row_field_name, col_field_name, OpFaceEle::OPROWCOL) {
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

      // get derivatives of base functions on row
      auto t_row_diff_base = row_data.getFTensor1DiffN<2>();

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL MATRIX
      for (int gg = 0; gg != nb_integration_points; gg++) {

        const double a = t_w * area;

        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          // get derivatives of base functions on column
          auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);

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

      // Fill value to local stiffness matrix ignoring boundary DOFs
      CHKERR MatSetValues(getKSPB(), row_data, col_data, &locLhs(0, 0),
                          ADD_VALUES);

      // Fill values of symmetric local stiffness matrix
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
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
  MatrixDouble locLhs, transLocLhs;
};

struct OpDomainRhs : public OpFaceEle {
public:
  OpDomainRhs(std::string field_name, ScalarFunc source_term_function)
      : OpFaceEle(field_name, OpFaceEle::OPROW),
        sourceTermFunc(source_term_function) {}

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

      // get base functions
      auto t_base = data.getFTensor0N();

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL VECTOR
      for (int gg = 0; gg != nb_integration_points; gg++) {

        const double a = t_w * area;
        double body_source =
            sourceTermFunc(t_coords(0), t_coords(1), t_coords(2));

        for (int rr = 0; rr != nb_dofs; rr++) {

          locRhs[rr] += t_base * body_source * a;

          // move to the next base function
          ++t_base;
        }

        // move to the weight of the next integration point
        ++t_w;
        // move to the coordinates of the next integration point
        ++t_coords;
      }

      // FILL VALUES OF THE GLOBAL VECTOR ENTRIES FROM THE LOCAL ONES
      CHKERR VecSetValues(getKSPf(), data, &*locRhs.begin(), ADD_VALUES);
    }

    MoFEMFunctionReturn(0);
  }

private:
  ScalarFunc sourceTermFunc;
  VectorDouble locRhs;
};

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

    if (nb_row_dofs && nb_col_dofs) {

      locBoundaryLhs.resize(nb_row_dofs, nb_col_dofs, false);
      locBoundaryLhs.clear();

      // get (boundary) element length
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

            locBoundaryLhs(rr, cc) += t_row_base * t_col_base * a;

            // move to the next base functions on column
            ++t_col_base;
          }
          // move to the next base functions on row
          ++t_row_base;
        }

        // move to the weight of the next integration point
        ++t_w;
      }

      // FILL VALUES OF LOCAL MATRIX ENTRIES TO THE GLOBAL MATRIX
      CHKERR MatSetValues(getKSPB(), row_data, col_data, &locBoundaryLhs(0, 0),
                          ADD_VALUES);
      if (row_side != col_side || row_type != col_type) {
        transLocBoundaryLhs.resize(nb_col_dofs, nb_row_dofs, false);
        noalias(transLocBoundaryLhs) = trans(locBoundaryLhs);
        CHKERR MatSetValues(getKSPB(), col_data, row_data,
                            &transLocBoundaryLhs(0, 0), ADD_VALUES);
      }
    }

    MoFEMFunctionReturn(0);
  }

private:
  MatrixDouble locBoundaryLhs, transLocBoundaryLhs;
};

struct OpBoundaryRhs : public OpEdgeEle {
public:
  OpBoundaryRhs(std::string field_name, ScalarFunc boundary_function)
      : OpEdgeEle(field_name, OpEdgeEle::OPROW),
        boundaryFunc(boundary_function) {}

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

      // get base function
      auto t_base = data.getFTensor0N();

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL VECTOR
      for (int gg = 0; gg != nb_integration_points; gg++) {

        const double a = t_w * edge;
        double boundary_term =
            boundaryFunc(t_coords(0), t_coords(1), t_coords(2));

        for (int rr = 0; rr != nb_dofs; rr++) {

          locBoundaryRhs[rr] += t_base * boundary_term * a;

          // move to the next base function
          ++t_base;
        }

        // move to the weight of the next integration point
        ++t_w;
        // move to the coordinates of the next integration point
        ++t_coords;
      }

      // FILL VALUES OF LOCAL VECTOR ENTRIES TO THE GLOBAL VECTOR
      CHKERR VecSetValues(getKSPf(), data, &*locBoundaryRhs.begin(),
                          ADD_VALUES);
    }

    MoFEMFunctionReturn(0);
  }

private:
  ScalarFunc boundaryFunc;
  VectorDouble locBoundaryRhs;
};

// New LHS and RHS for domain bc Neumann

struct OpBoundaryFluxLhs : public OpEdgeEle {
public:
  OpBoundaryFluxLhs(std::string row_field_name, std::string col_field_name)
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

      // get (boundary) element length
      const double edge = getMeasure();

      // get number of integration points
      const int nb_integration_points = getGaussPts().size2();
      // get integration weights
      auto t_w = getFTensor0IntegrationWeight();

      // get base functions on row
      auto t_row_base = row_data.getFTensor0N();
      auto t_normal = getFTensor1Normal();
      // get time
      const double time = getFEMethod()->ts_t;   
      
      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL MATRIX
      for (int gg = 0; gg != nb_integration_points; gg++) {
        const double a = t_w * edge;

        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          // get base functions on column
          auto t_col_base = col_data.getFTensor0N(gg, 0);

          for (int cc = 0; cc != nb_col_dofs; cc++) {

            locBoundaryLhs(rr, cc) += t_row_base * t_col_base * a;

            // move to the next base functions on column
            ++t_col_base;
          }
          // move to the next base functions on row
          ++t_row_base;
        }

        // move to the weight of the next integration point
        ++t_w;
      }

      // FILL VALUES OF LOCAL MATRIX ENTRIES TO THE GLOBAL MATRIX
      CHKERR MatSetValues(getKSPB(), row_data, col_data, &locBoundaryLhs(0, 0),
                          ADD_VALUES);
      if (row_side != col_side || row_type != col_type) {
        transLocBoundaryLhs.resize(nb_col_dofs, nb_row_dofs, false);
        noalias(transLocBoundaryLhs) = trans(locBoundaryLhs);
        CHKERR MatSetValues(getKSPB(), col_data, row_data,
                            &transLocBoundaryLhs(0, 0), ADD_VALUES);
      }
    }

    MoFEMFunctionReturn(0);
  }

private:
  MatrixDouble locBoundaryLhs, transLocBoundaryLhs;
};

struct OpBoundaryFluxRhs : public OpEdgeEle {
public:
  OpBoundaryFluxRhs(std::string field_name, ScalarFunc boundary_function)
      : OpEdgeEle(field_name, OpEdgeEle::OPROW),
        boundaryFunc(boundary_function) {}

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

      // get base function
      auto t_base = data.getFTensor0N();
      auto t_normal = getFTensor1Normal();
      // get time
      const double time = getFEMethod()->ts_t;      

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL VECTOR
      for (int gg = 0; gg != nb_integration_points; gg++) {

        const double a = t_w * edge;
        double boundary_term =
            boundaryFunc(t_coords(0), t_coords(1), t_coords(2));

        for (int rr = 0; rr != nb_dofs; rr++) {

          locBoundaryRhs[rr] -= t_base * boundary_term * a *t_normal(rr);

          // move to the next base function
          ++t_base;
          
        }

        // move to the weight of the next integration point
        ++t_w;
        // move to the coordinates of the next integration point
        ++t_coords;
      }

      // FILL VALUES OF LOCAL VECTOR ENTRIES TO THE GLOBAL VECTOR
      CHKERR VecSetValues(getKSPf(), data, &*locBoundaryRhs.begin(),
                          ADD_VALUES);
    }

    MoFEMFunctionReturn(0);
  }

private:
  ScalarFunc boundaryFunc;
  VectorDouble locBoundaryRhs;
};

// end new code for Neumann


}; // namespace Thermal2DOperators

#endif //__POISSON2DNONHOMOGENEOUS_HPP__