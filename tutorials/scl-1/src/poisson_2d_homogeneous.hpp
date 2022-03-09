// Define name if it has not been defined yet
#ifndef __POISSON2DHOMOGENEOUS_HPP__
#define __POISSON2DHOMOGENEOUS_HPP__

// Include standard library and Header file for basic finite elements
// implementation
#include <stdlib.h>
#include <BasicFiniteElements.hpp>

// Use of alias for some specific functions
// We are solving Poisson's equation in 2D so Face element is used
using FaceEle = MoFEM::FaceElementForcesAndSourcesCore;
using OpFaceEle = MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;
using OpEdgeEle = PipelineManager::EdgeEle::UserDataOperator;
using OpSkeletonEle = OpEdgeEle;
using FaceSide = MoFEM::FaceElementForcesAndSourcesCoreOnSideSwitch<0>;
using OpFaceSide = FaceSide::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;

// Namespace that contains necessary UDOs, will be included in the main program
namespace Poisson2DHomogeneousOperators {

// Declare FTensor index for 2D problem
FTensor::Index<'i', 2> i;

// For simplicity, source term f will be constant throughout the domain
const double body_source = 5.;
enum ElementSide { LEFT_SIDE = 0, RIGHT_SIDE };
// data for skeleton computation
map<int, VectorInt> indicesRowSideMap;
map<int, VectorInt> indicesColSideMap;
map<int, MatrixDouble> rowBaseSideMap;
map<int, MatrixDouble> colBaseSideMap;
map<int, VectorDouble> uSideMap;

struct OpDomainLhsMatrixK : public OpFaceEle {
public:
  OpDomainLhsMatrixK(std::string row_field_name, std::string col_field_name)
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

struct OpDomainRhsVectorF : public OpFaceEle {
public:
  OpDomainRhsVectorF(std::string field_name)
      : OpFaceEle(field_name, OpFaceEle::OPROW) {}

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
      CHKERR VecSetValues(getKSPf(), data, &locRhs(0), ADD_VALUES);
    }

    MoFEMFunctionReturn(0);
  }

private:
  VectorDouble locRhs;
};

struct OpCalculateSideData : public OpFaceSide {
  OpCalculateSideData(std::string field_name)
      : OpFaceSide(field_name, field_name, OpFaceSide::OPROW) {
    // FIXME: check if necessary
    std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
    doEntities[MBTRI] = doEntities[MBQUAD] = true;
  }
  OpCalculateSideData(std::string field_name, std::string col_field_name)
      : OpFaceSide(field_name, col_field_name, OpFaceSide::OPROWCOL) {
    // FIXME: check if necessary
    std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
    doEntities[MBTRI] = doEntities[MBQUAD] = true;
  }

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    int nb_gauss_pts = getGaussPts().size2();
    int nb_in_loop = getFEMethod()->nInTheLoop;

    auto &vec_u = uSideMap[nb_in_loop];
    indicesRowSideMap[nb_in_loop] = data.getIndices();
    rowBaseSideMap[nb_in_loop] = data.getN();

    vec_u.resize(nb_gauss_pts, false);
    vec_u.clear();

    for (int gg = 0; gg != nb_gauss_pts; gg++)
      vec_u(gg) = inner_prod(trans(data.getN(gg)), data.getFieldData());

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBeginHot;
    if (row_type == MBTRI || row_type == MBQUAD)
      if (col_type == MBTRI || col_type == MBQUAD) {
        auto nb_in_loop = getFEMethod()->nInTheLoop;
        // auto nb_in_loop = getFEMethod()->getFaceSense();
        auto nb_in_loop2 = getEdgeSideNumber();
        auto nb_in_loop3 = getDirection();
        auto &vec_u = uSideMap[nb_in_loop];
        indicesColSideMap[nb_in_loop] = col_data.getIndices();
        colBaseSideMap[nb_in_loop] = col_data.getN();
        indicesRowSideMap[nb_in_loop] = row_data.getIndices();
        rowBaseSideMap[nb_in_loop] = row_data.getN();
      }

    MoFEMFunctionReturnHot(0);
  }
};

struct OpComputeJumpOnSkeleton : public OpSkeletonEle {
  OpComputeJumpOnSkeleton(std::string field_name,
                          boost::shared_ptr<FaceSide> side_fe)
      : OpSkeletonEle(NOSPACE, OpSkeletonEle::OPLAST), sideFe(side_fe) {

  }

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    indicesRowSideMap.clear();
    rowBaseSideMap.clear();
    uSideMap.clear();

    CHKERR loopSideFaces("dFE", *sideFe);
    // CHKERR loopSide("dFE",
    //                 dynamic_cast<MoFEM::ForcesAndSourcesCore *>(&sideFe), 2);

#ifndef NDEBUG
    if (uSideMap.size() != 2 || rowBaseSideMap.size() != 2)
      MOFEM_LOG("WORLD", Sev::error) << " Problem with the side Map";
#endif // NDEBUG

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<FaceSide> sideFe;
};

template <typename T> inline auto get_ntensor(T &base_mat) {
  return FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(
      &*base_mat.data().begin());
};

template <typename T> inline auto get_ntensor(T &base_mat, int gg, int bb) {
  double *ptr = &base_mat(gg, bb);
  return FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(ptr);
};

struct OpDomainRhsPenalty : public OpSkeletonEle {
public:
  OpDomainRhsPenalty(std::string field_name, const double penalty = 1e-8)
      : OpSkeletonEle(field_name, OpSkeletonEle::OPROW), mPenalty(penalty) {
    std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
    doEntities[MBVERTEX] = true;
  }

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    auto &idx_rmap = indicesRowSideMap;
    auto t_base_l = get_ntensor(rowBaseSideMap.at(LEFT_SIDE));
    auto t_base_r = get_ntensor(rowBaseSideMap.at(RIGHT_SIDE));
    const size_t nb_dofs = idx_rmap.at(LEFT_SIDE).size();

    VectorDouble nf_l(nb_dofs, false);
    nf_l.clear();
    VectorDouble nf_r(nb_dofs, false);
    nf_r.clear();
    auto nb_in_loop = getFEMethod()->nInTheLoop;

    auto &u_l = uSideMap.at(LEFT_SIDE);
    auto &u_r = uSideMap.at(RIGHT_SIDE);

    const double area = getMeasure();
    const int nb_integration_points = getGaussPts().size2();
    auto t_w = getFTensor0IntegrationWeight();

    for (int gg = 0; gg != nb_integration_points; gg++) {
      const double a = t_w * area * mPenalty;
      auto u_jump = u_l(gg) - u_r(gg);
      for (int rr = 0; rr != nb_dofs; rr++) {

        nf_r[rr] += t_base_l * u_jump * a;
        nf_l[rr] -= t_base_r * u_jump * a;
        ++t_base_l;
        ++t_base_r;
      }

      ++t_w;
    }

    CHKERR VecSetValues(getKSPf(), idx_rmap.at(LEFT_SIDE).size(),
                        &*idx_rmap.at(LEFT_SIDE).begin(), &*nf_l.data().begin(),
                        ADD_VALUES);
    CHKERR VecSetValues(getKSPf(), idx_rmap.at(RIGHT_SIDE).size(),
                        &*idx_rmap.at(RIGHT_SIDE).begin(),
                        &*nf_r.data().begin(), ADD_VALUES);

    MoFEMFunctionReturn(0);
  }

private:
  const double mPenalty;
};

struct OpDomainLhsPenalty : public OpSkeletonEle {
public:
  OpDomainLhsPenalty(std::string row_field_name, std::string col_field_name,
                     const double penalty = 1e-8)
      : OpSkeletonEle(row_field_name, col_field_name, OpSkeletonEle::OPROWCOL),
        mPenalty(penalty) {
    std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
    doEntities[MBVERTEX] = true;
    sYmm = false;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    if (row_type != MBVERTEX || col_type != MBVERTEX)
      MoFEMFunctionReturnHot(0);

    EntityHandle ent = getFEEntityHandle();
    auto t_w = getFTensor0IntegrationWeight();

    array<map<int, VectorInt>, 2> idx_map;
    idx_map[ROW] = indicesRowSideMap;
    idx_map[COL] = indicesColSideMap;

    // shape funcs
    auto t_row_base_l = get_ntensor(rowBaseSideMap.at(LEFT_SIDE));
    auto t_row_base_r = get_ntensor(rowBaseSideMap.at(RIGHT_SIDE));

    const size_t nb_rows = idx_map[ROW].at(LEFT_SIDE).size();
    const size_t nb_cols = idx_map[COL].at(LEFT_SIDE).size();

    if (!nb_cols)
      MoFEMFunctionReturnHot(0);

    std::array<std::array<MatrixDouble, 2>, 2> locMat;
    for (auto side0 : {LEFT_SIDE, RIGHT_SIDE})
      for (auto side1 : {LEFT_SIDE, RIGHT_SIDE}) {
        locMat[side0][side1].resize(nb_rows, nb_cols, false);
        locMat[side0][side1].clear();
      }

    const size_t nb_integration_pts = getGaussPts().size2();
    const size_t nb_row_base_functions = rowBaseSideMap.at(LEFT_SIDE).size2();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      const double alpha = getMeasure() * t_w * mPenalty;

      auto t_mat_rr = locMat[RIGHT_SIDE][RIGHT_SIDE].data().begin();
      auto t_mat_rl = locMat[RIGHT_SIDE][LEFT_SIDE].data().begin();
      auto t_mat_lr = locMat[LEFT_SIDE][RIGHT_SIDE].data().begin();
      auto t_mat_ll = locMat[LEFT_SIDE][LEFT_SIDE].data().begin();

      size_t rr = 0;
      for (; rr != nb_rows; ++rr) {

        auto t_col_base_l = get_ntensor(colBaseSideMap.at(LEFT_SIDE), gg, 0);
        auto t_col_base_r = get_ntensor(colBaseSideMap.at(RIGHT_SIDE), gg, 0);

        for (size_t cc = 0; cc != nb_cols; ++cc) {

          *t_mat_rr += alpha * t_row_base_r * t_col_base_r;
          *t_mat_rl -= alpha * t_row_base_r * t_col_base_l;
          *t_mat_lr -= alpha * t_row_base_l * t_col_base_r;
          *t_mat_ll += alpha * t_row_base_l * t_col_base_l;

          ++t_mat_rr;
          ++t_mat_rl;
          ++t_mat_lr;
          ++t_mat_ll;
          ++t_col_base_r;
          ++t_col_base_l;
        }

        ++t_row_base_r;
        ++t_row_base_l;
      }
      for (; rr < nb_row_base_functions; ++rr) {
        ++t_row_base_r;
        ++t_row_base_l;
      }

      ++t_w;
    }

    for (auto s0 : {LEFT_SIDE, RIGHT_SIDE})
      for (auto s1 : {LEFT_SIDE, RIGHT_SIDE})
        CHKERR ::MatSetValues(getKSPB(), idx_map[ROW].at(s0).size(),
                              &*idx_map[ROW].at(s0).begin(),
                              idx_map[COL].at(s1).size(),
                              &*idx_map[COL].at(s1).begin(),
                              &*locMat[s0][s1].data().begin(), ADD_VALUES);

    MoFEMFunctionReturn(0);
  }

private:
  const double mPenalty;
};

struct OpBoundaryLhs : public OpEdgeEle {
public:
  OpBoundaryLhs(std::string row_field_name, std::string col_field_name,
                boost::shared_ptr<FaceSide> side_fe, double penalty)
      : OpEdgeEle(row_field_name, col_field_name, OpEdgeEle::OPROWCOL),
        sideFE(side_fe), pEnalty(penalty) {
    sYmm = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;

    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();

    if (row_type != MBVERTEX || col_type != MBVERTEX)
      MoFEMFunctionReturnHot(0);

    indicesRowSideMap.clear();
    indicesColSideMap.clear();
    rowBaseSideMap.clear();
    uSideMap.clear();

    CHKERR loopSideFaces("dFE", *sideFE);

    EntityHandle ent = getFEEntityHandle();
    auto t_w = getFTensor0IntegrationWeight();

    array<map<int, VectorInt>, 2> idx_map;
    idx_map[ROW] = indicesRowSideMap;
    idx_map[COL] = indicesColSideMap;

    // shape funcs
    auto t_row_base = get_ntensor(rowBaseSideMap.at(0));

    const size_t nb_rows = idx_map[ROW].at(0).size();
    const size_t nb_cols = idx_map[COL].at(0).size();

    if (!nb_cols)
      MoFEMFunctionReturnHot(0);

    MatrixDouble locMat(nb_rows, nb_cols, false);
    locMat.clear();

    const size_t nb_integration_pts = getGaussPts().size2();
    const size_t nb_row_base_functions = rowBaseSideMap.at(0).size2();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      const double alpha = getMeasure() * t_w * pEnalty;

      auto t_mat = locMat.data().begin();

      size_t rr = 0;
      for (; rr != nb_rows; ++rr) {

        auto t_col_base = get_ntensor(colBaseSideMap.at(0), gg, 0);
    
        for (size_t cc = 0; cc != nb_cols; ++cc) {

          *t_mat += alpha * t_row_base * t_col_base;

          ++t_mat;
          ++t_col_base;
  
        }

        ++t_row_base;
      }
      for (; rr < nb_row_base_functions; ++rr) {
        ++t_row_base;
      }

      ++t_w;
    }

    CHKERR ::MatSetValues(
        getKSPB(), idx_map[ROW].at(0).size(), &*idx_map[ROW].at(0).begin(),
        idx_map[COL].at(0).size(), &*idx_map[COL].at(0).begin(),
        &*locMat.data().begin(), ADD_VALUES);

    MoFEMFunctionReturn(0);
  }

private:
  MatrixDouble locBoundaryLhs, transLocBoundaryLhs;
  boost::shared_ptr<FaceSide> sideFE;
  double pEnalty;
};

struct OpBoundaryRhs : public OpEdgeEle {
public:
  OpBoundaryRhs(std::string field_name, ScalarFun boundary_function)
      : OpEdgeEle(field_name, OpEdgeEle::OPROW),
        boundaryFunc(boundary_function) {
    std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], true);
  }

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
  ScalarFun boundaryFunc;
  VectorDouble locBoundaryRhs;
};

}; // namespace Poisson2DHomogeneousOperators

#endif //__POISSON2DHOMOGENEOUS_HPP__