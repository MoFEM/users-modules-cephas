#ifndef __THERMAL2D_HPP__
#define __THERMAL2D_HPP__

#include <stdlib.h>
#include <BasicFiniteElements.hpp>

using FaceEle = MoFEM::FaceElementForcesAndSourcesCore;
using EdgeEle = MoFEM::EdgeElementForcesAndSourcesCore;

using OpFaceEle = MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;
using OpEdgeEle = MoFEM::EdgeElementForcesAndSourcesCore::UserDataOperator;

using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = FaceElementForcesAndSourcesCoreBase;
using DomainEleOp = DomainEle::UserDataOperator;
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

struct DataAtGaussPoints {
  // This struct is for data required for the update of Newton's iteration

  VectorDouble fieldValue; // field value at integration point
  MatrixDouble fieldGrad;  // field gradient at integration point
  VectorDouble fieldDot;   // time derivative of field at integration point
};

struct OpKut : public OpFaceEle {
public:
  OpKut(std::string row_field_name, std::string col_field_name, 
        boost::shared_ptr<MatrixDouble> mat_D,boost::shared_ptr<DataAtGaussPoints> &common_data)
      : OpFaceEle(row_field_name, col_field_name, OpFaceEle::OPROWCOL), 
        matD(mat_D), commonData(common_data) {
    sYmm = false;
  }
protected:
  boost::shared_ptr<MatrixDouble> matD;

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
      
 
      auto t_D = getFTensor4DdgFromMat<2, 2, 0>(*matD);
      constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();

      auto get_tensor1 = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor1<double *, 2>(
                                             &m(r + 0, c), 
                                             &m(r + 1, c));
      };

      FTensor::Index<'i', 2> i;
      FTensor::Index<'j', 2> j;
      FTensor::Index<'k', 2> k;
      FTensor::Index<'l', 2> l;

      // // get derivatives of base functions on column
      // auto t_row_diff_base = row_data.getFTensor1DiffN<2>(); 

      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL MATRIX
      for (int gg = 0; gg != nb_integration_points; gg++) {
      // get solution (field value) at integration point
      auto t_temp = getFTensor0FromVec(commonData->fieldValue);

        const double a = t_w * area;
        // get derivatives of base functions on column
        auto t_row_diff_base = row_data.getFTensor1DiffN<2>(gg,0); 
        for (int rr = 0; rr != nb_row_dofs/2; ++rr) {

          // get the base functions on column (Temperature)
          auto t_col_base = col_data.getFTensor0N(gg, 0);

          for (int cc = 0; cc != nb_col_dofs; cc++) {
            auto t_subLocMat = get_tensor1(locLhs, 2*rr, cc);  
            t_subLocMat(i) -= t_row_diff_base(j) * t_col_base * a * t_D(i,j,k,l) * t_kd(k, l);

            // move to the derivatives of the next base functions on column
            ++t_col_base;
          }

          // move to the derivatives of the next base functions on row
          ++t_row_diff_base;


        }
          // move to the weight of the next integration point
          ++t_w;
          // // move to the field at the next integration point
          // ++t_temp;
      }

      // FILL VALUES OF LOCAL MATRIX ENTRIES TO THE GLOBAL MATRIX

      // Fill value to local stiffness matrix ignoring boundary DOFs
      CHKERR MatSetValues(getKSPB(), row_data, col_data, &locLhs(0, 0),
                          ADD_VALUES);

      // Fill values of symmetric local stiffness matrix
      // if (row_side != col_side || row_type != col_type) {
      //   transLocLhs.resize(nb_col_dofs, nb_row_dofs, false);
      //   noalias(transLocLhs) = trans(locLhs);
      //   CHKERR MatSetValues(getKSPB(), col_data, row_data, &transLocLhs(0, 0),
      //                       ADD_VALUES);
      // }
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<std::vector<unsigned char>> boundaryMarker;
  MatrixDouble locLhs, transLocLhs;
  boost::shared_ptr<DataAtGaussPoints> commonData;
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

struct OpBoundaryLhs_tm : public OpEdgeEle {
public:
  OpBoundaryLhs_tm(std::string row_field_name, std::string col_field_name,
  boost::shared_ptr<MatrixDouble> mat_D,boost::shared_ptr<DataAtGaussPoints> &common_data)
      : OpEdgeEle(row_field_name, col_field_name, OpEdgeEle::OPROWCOL),matD(mat_D), commonData(common_data) {
    sYmm = false;
  }
  
protected:
  boost::shared_ptr<MatrixDouble> matD;

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
      // auto t_row_base = row_data.getFTensor0N();

      // // get solution (field value) at integration point
      auto t_temp = getFTensor0FromVec(commonData->fieldValue);

      auto t_D = getFTensor4DdgFromMat<2, 2, 0>(*matD);
      constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();

      
      auto get_tensor1 = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor1<double *, 2>(
                                             &m(r + 0, c), 
                                             &m(r + 1, c));
      };

      FTensor::Index<'i', 2> i;
      FTensor::Index<'j', 2> j;
      FTensor::Index<'k', 2> k;
      FTensor::Index<'l', 2> l;

      // get derivatives of base functions on column
      auto t_row_diff_base = row_data.getFTensor1DiffN<2>();

      // to the face area
      auto t_normal = getFTensor1Normal();
      t_normal(i) /= sqrt(t_normal(j) * t_normal(j));

      cerr << "t_normal " << t_normal << endl; 
      // if( t_normal(0) - 1. < 1.e-7){
      // cerr << "col_data.getIndices() "<< col_data.getIndices() <<endl;
      // }
      // START THE LOOP OVER INTEGRATION POINTS TO CALCULATE LOCAL MATRIX
      for (int gg = 0; gg != nb_integration_points; gg++) {
        const double a = t_w * edge;
        // get base functions on row
        auto t_row_base = row_data.getFTensor0N(gg,0);
        for (int rr = 0; rr != nb_row_dofs/2; ++rr) {
          // get base functions on column
          auto t_col_base = col_data.getFTensor0N(gg, 0);

          for (int cc = 0; cc != nb_col_dofs; cc++) {
            auto t_subLocMat = get_tensor1(locBoundaryLhs, 2*rr, cc);  
            t_subLocMat(i) -= (t_row_base * t_col_base * a * t_D(i,j,k,l) * t_kd(k, l))*t_normal(j);


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
      // if (row_side != col_side || row_type != col_type) {
      //   transLocBoundaryLhs.resize(nb_col_dofs, nb_row_dofs, false);
      //   noalias(transLocBoundaryLhs) = trans(locBoundaryLhs);
      //   CHKERR MatSetValues(getKSPB(), col_data, row_data,
      //                       &transLocBoundaryLhs(0, 0), ADD_VALUES);
      // }
    }

    MoFEMFunctionReturn(0);
  }

private:
  MatrixDouble locBoundaryLhs, transLocBoundaryLhs;
  boost::shared_ptr<DataAtGaussPoints> commonData;
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

// Start Elastic Part

template <int DIM_01, int DIM_23, int S = 0>
struct OpTensorTimesSymmetricTensorNew
    : public ForcesAndSourcesCore::UserDataOperator {

  using EntData = DataForcesAndSourcesCore::EntData;
  using UserOp = ForcesAndSourcesCore::UserDataOperator;

  OpTensorTimesSymmetricTensorNew(const std::string field_name,
                               boost::shared_ptr<MatrixDouble> in_mat,
                               boost::shared_ptr<MatrixDouble> out_mat,
                               boost::shared_ptr<MatrixDouble> d_mat,
                               boost::shared_ptr<MatrixDouble> t_mat,
                               boost::shared_ptr<DataAtGaussPoints> &common_data)
      : UserOp(field_name, OPROW), inMat(in_mat), outMat(out_mat), dMat(d_mat), tMat(t_mat), commonData(common_data) {
    // Only is run for vertices
    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
    if (!inMat)
      THROW_MESSAGE("Pointer for in mat is null");
    if (!outMat)
      THROW_MESSAGE("Pointer for out mat is null");
    if (!dMat)
      THROW_MESSAGE("Pointer for tensor mat is null");
     if (!tMat)
      THROW_MESSAGE("Pointer for thermal tensor mat is null");     
  }

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    const size_t nb_gauss_pts = getGaussPts().size2();
    auto t_D = getFTensor4DdgFromMat<DIM_01, DIM_23, S>(*(dMat));
    auto t_thD = getFTensor4DdgFromMat<DIM_01, DIM_23, S>(*(tMat));
    auto t_in = getFTensor2SymmetricFromMat<DIM_01>(*(inMat));
    outMat->resize((DIM_23 * (DIM_23 + 1)) / 2, nb_gauss_pts, false);
    auto t_out = getFTensor2SymmetricFromMat<DIM_23>(*(outMat));
    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
    // get solution (field value) at integration point
    auto t_temp = getFTensor0FromVec(commonData->fieldValue);
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      t_out(i, j) = t_D(i, j, k, l) * t_in(k, l) - t_thD(i, j, k, l) * t_kd(k, l) *t_temp;
      ++t_in;
      ++t_out;
      ++t_temp;
    }
    MoFEMFunctionReturn(0);
  }

private:
  FTensor::Index<'i', DIM_01> i;
  FTensor::Index<'j', DIM_01> j;
  FTensor::Index<'k', DIM_23> k;
  FTensor::Index<'l', DIM_23> l;

  boost::shared_ptr<MatrixDouble> inMat;
  boost::shared_ptr<MatrixDouble> outMat;
  boost::shared_ptr<MatrixDouble> dMat;
  boost::shared_ptr<MatrixDouble> tMat;
  boost::shared_ptr<DataAtGaussPoints> commonData;
};

template <int DIM> struct OpPostProcElastic : public DomainEleOp {
  OpPostProcElastic(const std::string field_name,
                    moab::Interface &post_proc_mesh,
                    std::vector<EntityHandle> &map_gauss_pts,
                    boost::shared_ptr<MatrixDouble> m_strain_ptr,
                    boost::shared_ptr<MatrixDouble> m_stress_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;
  boost::shared_ptr<MatrixDouble> mStrainPtr;
  boost::shared_ptr<MatrixDouble> mStressPtr;
};
//! [Class definition]

//! [Postprocessing constructor]
template <int DIM>
OpPostProcElastic<DIM>::OpPostProcElastic(
    const std::string field_name, moab::Interface &post_proc_mesh,
    std::vector<EntityHandle> &map_gauss_pts,
    boost::shared_ptr<MatrixDouble> m_strain_ptr,
    boost::shared_ptr<MatrixDouble> m_stress_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW), postProcMesh(post_proc_mesh),
      mapGaussPts(map_gauss_pts), mStrainPtr(m_strain_ptr),
      mStressPtr(m_stress_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}
//! [Postprocessing constructor]

//! [Postprocessing]
template <int DIM>
MoFEMErrorCode OpPostProcElastic<DIM>::doWork(int side, EntityType type,
                                              EntData &data) {
  MoFEMFunctionBegin;

  auto get_tag = [&](const std::string name) {
    std::array<double, 9> def;
    std::fill(def.begin(), def.end(), 0);
    Tag th;
    CHKERR postProcMesh.tag_get_handle(name.c_str(), 9, MB_TYPE_DOUBLE, th,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       def.data());
    return th;
  };

  MatrixDouble3by3 mat(3, 3);

  auto set_matrix_symm = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != DIM; ++r)
      for (size_t c = 0; c != DIM; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_plain_stress_strain = [&](auto &mat, auto &t) -> MatrixDouble3by3 & {
    //poisson_ratio is not passed in!
    mat(2, 2) = -0. * (t(0, 0) + t(1, 1));
    return mat;
  };

  auto set_tag = [&](auto th, auto gg, MatrixDouble3by3 &mat) {
    return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
                                     &*mat.data().begin());
  };

  auto th_strain = get_tag("STRAIN");
  auto th_stress = get_tag("STRESS");

  size_t nb_gauss_pts = data.getN().size1();
  auto t_strain = getFTensor2SymmetricFromMat<DIM>(*(mStrainPtr));
  auto t_stress = getFTensor2SymmetricFromMat<DIM>(*(mStressPtr));

  switch (DIM) {
  case 2:
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      CHKERR set_tag(
          th_strain, gg,
          set_plain_stress_strain(set_matrix_symm(t_strain), t_stress));
      CHKERR set_tag(th_stress, gg, set_matrix_symm(t_stress));
      ++t_strain;
      ++t_stress;
    }
    break;
  case 3:
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      CHKERR set_tag(th_strain, gg, set_matrix_symm(t_strain));
      CHKERR set_tag(th_stress, gg, set_matrix_symm(t_stress));
      ++t_strain;
      ++t_stress;
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
            "Not implemeneted dimension");
  }

  MoFEMFunctionReturn(0);
}
//! [Postprocessing]
// End Elastic Part



}; // namespace Thermal2DOperators

#endif //__THERMAL2D_HPP__