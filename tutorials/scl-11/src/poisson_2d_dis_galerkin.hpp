/**
 * \file poisson_2d_dis_galerkin.hpp
 * \example poisson_2d_dis_galerkin.hpp
 *
 * Example of implementation for discontinuous Galerkin.
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


// Define name if it has not been defined yet
#ifndef __POISSON2DISGALERKIN_HPP__
#define __POISSON2DISGALERKIN_HPP__

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
namespace Poisson2DiscontGalerkinOperators {

// Declare FTensor index for 2D problem
FTensor::Index<'i', 2> i;

enum ElementSide { LEFT_SIDE = 0, RIGHT_SIDE };
// data for skeleton computation
std::array<VectorInt, 2> indicesRowSideMap;
std::array<VectorInt, 2> indicesColSideMap;
std::array<MatrixDouble, 2> rowBaseSideMap;
std::array<MatrixDouble, 2> colBaseSideMap;

struct OpCalculateSideData : public OpFaceSide {

  OpCalculateSideData(std::string field_name, std::string col_field_name)
      : OpFaceSide(field_name, col_field_name, OpFaceSide::OPROWCOL) {

    std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
    doEntities[MBTRI] = doEntities[MBQUAD] = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBeginHot;
    if (row_type == MBTRI || row_type == MBQUAD)
      if (col_type == MBTRI || col_type == MBQUAD) {
        auto nb_in_loop = getFEMethod()->nInTheLoop;

        indicesColSideMap[nb_in_loop] = col_data.getIndices();
        colBaseSideMap[nb_in_loop] = col_data.getN();
        indicesRowSideMap[nb_in_loop] = row_data.getIndices();
        rowBaseSideMap[nb_in_loop] = row_data.getN();
      }

    MoFEMFunctionReturnHot(0);
  }
};

template <typename T> inline auto get_ntensor(T &base_mat) {
  return FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(
      &*base_mat.data().begin());
};

template <typename T> inline auto get_ntensor(T &base_mat, int gg, int bb) {
  double *ptr = &base_mat(gg, bb);
  return FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(ptr);
};

struct OpDomainLhsPenalty : public OpSkeletonEle {
public:
  OpDomainLhsPenalty(boost::shared_ptr<FaceSide> side_fe,
                     const double penalty = 1e-8)
      : OpSkeletonEle(NOSPACE, OpEdgeEle::OPLAST), sideFe(side_fe),
        mPenalty(penalty) {}

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    CHKERR loopSideFaces("dFE", *sideFe);

    EntityHandle ent = getFEEntityHandle();
    auto t_w = getFTensor0IntegrationWeight();

    array<std::array<VectorInt, 2> *, 2> idx_map{&indicesRowSideMap,
                                                 &indicesColSideMap};
    const size_t nb_rows = (*idx_map[ROW])[LEFT_SIDE].size();
    const size_t nb_cols = (*idx_map[COL])[LEFT_SIDE].size();

#ifndef NDEBUG
    if (*(idx_map[ROW])[LEFT_SIDE].size() != *(idx_map[ROW])[RIGHT_SIDE].size())
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
              "Code is implemented that number of base functions on both sides "
              "is the same");
    if (*(idx_map[COL])[LEFT_SIDE].size() != *(idx_map[COL])[RIGHT_SIDE].size())
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
              "Code is implemented that number of base functions on both sides "
              "is the same");
#endif 

    if (!nb_rows)
      MoFEMFunctionReturnHot(0);

    std::array<std::array<MatrixDouble, 2>, 2> locMat;
    for (auto side0 : {LEFT_SIDE, RIGHT_SIDE})
      for (auto side1 : {LEFT_SIDE, RIGHT_SIDE}) {
        locMat[side0][side1].resize(nb_rows, nb_cols, false);
        locMat[side0][side1].clear();
      }

    const size_t nb_integration_pts = getGaussPts().size2();
    const size_t nb_row_base_functions = rowBaseSideMap.at(LEFT_SIDE).size2();

    // shape funcs
    auto t_row_base_l = get_ntensor(rowBaseSideMap.at(LEFT_SIDE));
    auto t_row_base_r = get_ntensor(rowBaseSideMap.at(RIGHT_SIDE));


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
        CHKERR ::MatSetValues(getKSPB(), (*idx_map[ROW])[s0].size(),
                              &*(*idx_map[ROW])[s0].begin(),
                              (*idx_map[COL])[s1].size(),
                              &*(*idx_map[COL])[s1].begin(),
                              &*locMat[s0][s1].data().begin(), ADD_VALUES);

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<FaceSide> sideFe;
  const double mPenalty;

};

struct OpL2BoundaryLhs : public OpEdgeEle {
public:
  OpL2BoundaryLhs(boost::shared_ptr<FaceSide> side_fe, double penalty)
      : OpEdgeEle(NOSPACE, OpEdgeEle::OPLAST), sideFE(side_fe),
        pEnalty(penalty) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    CHKERR loopSideFaces("dFE", *sideFE);

    EntityHandle ent = getFEEntityHandle();
    auto t_w = getFTensor0IntegrationWeight();

    // shape funcs
    auto t_row_base = get_ntensor(rowBaseSideMap.at(0));

    const size_t nb_rows = indicesRowSideMap[0].size();
    const size_t nb_cols = indicesColSideMap[0].size();

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
        getKSPB(), indicesRowSideMap[0].size(), &*indicesRowSideMap[0].begin(),
        indicesColSideMap[0].size(), &*indicesColSideMap[0].begin(),
        &*locMat.data().begin(), ADD_VALUES);

    MoFEMFunctionReturn(0);
  }

private:
  MatrixDouble locBoundaryLhs;
  boost::shared_ptr<FaceSide> sideFE;
  double pEnalty;
};

}; // namespace Poisson2DiscontGalerkinOperators

#endif //__POISSON2DISGALERKIN_HPP__