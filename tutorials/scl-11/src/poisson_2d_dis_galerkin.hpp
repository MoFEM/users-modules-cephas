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
std::array<MatrixDouble, 2> rowDiffBaseSideMap;
std::array<MatrixDouble, 2> colDiffBaseSideMap;

constexpr double phi =
    -1; // 1 - symmetric Nitsche, 0 - nonsymmetric, -1 antisymmetric

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

    if ((CN::Dimension(row_type) == 2) && (CN::Dimension(col_type) == 2)) {
      auto nb_in_loop = getFEMethod()->nInTheLoop;
      indicesColSideMap[nb_in_loop] = col_data.getIndices();
      colBaseSideMap[nb_in_loop] = col_data.getN();
      colDiffBaseSideMap[nb_in_loop] = col_data.getDiffN();
      indicesRowSideMap[nb_in_loop] = row_data.getIndices();
      rowBaseSideMap[nb_in_loop] = row_data.getN();
      rowDiffBaseSideMap[nb_in_loop] = row_data.getN();
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

template <typename T> inline auto get_diff_ntensor(T &base_mat) {
  double *ptr = &*base_mat.data().begin();
  return FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2>(ptr, &ptr[1]);
};

template <typename T>
inline auto get_diff_ntensor(T &base_mat, int gg, int bb) {
  double *ptr = &base_mat(gg, 2 * bb);
  return FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2>(ptr, &ptr[1]);
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

    auto t_normal = getFTensor1Normal();
    t_normal.normalize();

    const size_t nb_integration_pts = getGaussPts().size2();
    constexpr std::array<int, 4> sign_array{1, -1, -1, 1};

    for (auto s0 : {LEFT_SIDE, RIGHT_SIDE}) {

      const auto nb_rows = indicesRowSideMap[s0].size();

      if (nb_rows) {

        const auto nb_row_base_functions = rowBaseSideMap[s0].size2();
        for (auto s1 : {LEFT_SIDE, RIGHT_SIDE}) {

          const auto sign = sign_array[s0 * 2 + s1];
          const auto sign_row = sign_array[s0];
          const auto sign_col = sign_array[s1];

          const auto nb_cols = indicesColSideMap[s1].size();
          locMat.resize(nb_rows, nb_cols, false);
          locMat.clear();

          auto t_row_base = get_ntensor(rowBaseSideMap[s0]);
          auto t_diff_row_base = get_diff_ntensor(rowDiffBaseSideMap[0]);
          auto t_w = getFTensor0IntegrationWeight();

          for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

            const double alpha = getMeasure() * t_w;

            auto t_mat = locMat.data().begin();

            size_t rr = 0;
            for (; rr != nb_rows; ++rr) {
              auto t_col_base = get_ntensor(colBaseSideMap[s1], gg, 0);
              auto t_diff_col_base =
                  get_diff_ntensor(colDiffBaseSideMap[s1], gg, 0);
              for (size_t cc = 0; cc != nb_cols; ++cc) {

                // TODO: This is not effient constants should be precalculated
                // outside loops

                *t_mat += (alpha * mPenalty * sign) * t_row_base * t_col_base;
                *t_mat -= (alpha * sign) * t_row_base *
                          (t_diff_col_base(i) * t_normal(i));
                *t_mat -= (alpha * sign * phi) *
                          (t_diff_row_base(i) * t_normal(i)) * t_col_base;
                *t_mat +=
                    (alpha * sign * phi) * (t_diff_row_base(i) * t_normal(i)) *
                    (t_diff_col_base(i) * t_normal(i)) / (mPenalty * mPenalty);

                ++t_col_base;
                ++t_diff_col_base;
                ++t_mat;
              }

              ++t_row_base;
            }

            for (; rr < nb_row_base_functions; ++rr) {
              ++t_row_base;
              ++t_diff_row_base;
            }

            ++t_w;
          }

          CHKERR ::MatSetValues(getKSPB(), indicesRowSideMap[s0].size(),
                                &*indicesRowSideMap[s0].begin(),
                                indicesColSideMap[s1].size(),
                                &*indicesColSideMap[s1].begin(),
                                &*locMat.data().begin(), ADD_VALUES);
        }
      }
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<FaceSide> sideFe;
  const double mPenalty;
  MatrixDouble locMat;
};

struct OpL2BoundaryLhs : public OpEdgeEle {
public:
  OpL2BoundaryLhs(boost::shared_ptr<FaceSide> side_fe, double penalty)
      : OpEdgeEle(NOSPACE, OpEdgeEle::OPLAST), sideFE(side_fe),
        pEnalty(penalty) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    CHKERR loopSideFaces("dFE", *sideFE);

    auto t_normal = getFTensor1Normal();
    t_normal.normalize();

    auto t_w = getFTensor0IntegrationWeight();

    // shape funcs
    auto t_row_base = get_ntensor(rowBaseSideMap[0]);
    auto t_diff_row_base = get_diff_ntensor(rowDiffBaseSideMap[0]);

    const size_t nb_rows = indicesRowSideMap[0].size();
    const size_t nb_cols = indicesColSideMap[0].size();

    if (!nb_cols)
      MoFEMFunctionReturnHot(0);

    locMat.resize(nb_rows, nb_cols, false);
    locMat.clear();

    const size_t nb_integration_pts = getGaussPts().size2();
    const size_t nb_row_base_functions = rowBaseSideMap[0].size2();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      const double alpha = getMeasure() * t_w;

      auto t_mat = locMat.data().begin();

      size_t rr = 0;
      for (; rr != nb_rows; ++rr) {

        auto t_col_base = get_ntensor(colBaseSideMap[0], gg, 0);
        auto t_diff_col_base = get_diff_ntensor(colDiffBaseSideMap[0], gg, 0);

        for (size_t cc = 0; cc != nb_cols; ++cc) {

          // TODO: This is not effient constants should be precalculated
          // outside loops

          *t_mat += (alpha * pEnalty) * t_row_base * t_col_base;
          *t_mat -= alpha * t_row_base * (t_diff_col_base(i) * t_normal(i));
          *t_mat -=
              alpha * phi * (t_diff_row_base(i) * t_normal(i)) * t_col_base;
          *t_mat += alpha * phi * (t_diff_row_base(i) * t_normal(i)) *
                    (t_diff_col_base(i) * t_normal(i)) / (pEnalty * pEnalty);

          ++t_mat;
          ++t_col_base;
          ++t_diff_col_base;
        }

        ++t_row_base;
      }
      for (; rr < nb_row_base_functions; ++rr) {
        ++t_row_base;
        ++t_diff_row_base;
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
  MatrixDouble locMat;
};

}; // namespace Poisson2DiscontGalerkinOperators

#endif //__POISSON2DISGALERKIN_HPP__