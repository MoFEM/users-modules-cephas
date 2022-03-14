/**
 * \file PoissonDiscontinousGalerkin.hpp
 * \example PoissonDiscontinousGalerkin.hpp
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

// Namespace that contains necessary UDOs, will be included in the main program
namespace Poisson2DiscontGalerkinOperators {

// Declare FTensor index for 2D problem
FTensor::Index<'i', SPACE_DIM> i;

enum ElementSide { LEFT_SIDE = 0, RIGHT_SIDE };
// data for skeleton computation
std::array<VectorInt, 2> indicesRowSideMap;
std::array<VectorInt, 2> indicesColSideMap;
std::array<MatrixDouble, 2> rowBaseSideMap;
std::array<MatrixDouble, 2> colBaseSideMap;
std::array<MatrixDouble, 2> rowDiffBaseSideMap;
std::array<MatrixDouble, 2> colDiffBaseSideMap;
std::array<double, 2> areaMap;

auto testing_fun = [](auto base, auto diff_base, auto p, auto phi,
                      auto nitche) {
  return (base - nitche * phi * diff_base / p);
};

auto testing_fun_plus = [](auto base, auto diff_base, auto p, auto phi,
                           auto nitche) {
  return nitche * phi * diff_base / p;
};

auto tested_fun = [](auto base, auto diff_base, auto p, auto nitche) {
  return -p * (base - nitche * diff_base / p);
};

struct OpCalculateSideData : public FaceSideOp {

  OpCalculateSideData(std::string field_name, std::string col_field_name)
      : FaceSideOp(field_name, col_field_name, FaceSideOp::OPROWCOL) {

    std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);

    for (auto t = moab::CN::TypeDimensionMap[SPACE_DIM].first;
         t <= moab::CN::TypeDimensionMap[SPACE_DIM].second; ++t)
      doEntities[t] = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBeginHot;

    if ((CN::Dimension(row_type) == SPACE_DIM) &&
        (CN::Dimension(col_type) == SPACE_DIM)) {
      const auto nb_in_loop = getFEMethod()->nInTheLoop;
      indicesRowSideMap[nb_in_loop] = row_data.getIndices();
      indicesColSideMap[nb_in_loop] = col_data.getIndices();
      rowBaseSideMap[nb_in_loop] = row_data.getN();
      colBaseSideMap[nb_in_loop] = col_data.getN();
      rowDiffBaseSideMap[nb_in_loop] = row_data.getDiffN();
      colDiffBaseSideMap[nb_in_loop] = col_data.getDiffN();
      areaMap[nb_in_loop] = getMeasure();
    } else {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Should not happen");
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

struct OpL2LhsPenalty : public BoundaryEleOp {
public:
  OpL2LhsPenalty(boost::shared_ptr<FaceSideEle> side_fe_ptr,
                     bool is_boundary)
      : BoundaryEleOp(NOSPACE, BoundaryEleOp::OPLAST), sideFEPtr(side_fe_ptr),
        isBoundary(is_boundary) {}

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    CHKERR loopSideFaces("dFE", *sideFEPtr);

    const double s = getMeasure() / (areaMap[0] + areaMap[1]);
    const double p = penalty * s;

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

          const auto nb_cols = indicesColSideMap[s1].size();
          locMat.resize(nb_rows, nb_cols, false);
          locMat.clear();

          auto t_row_base = get_ntensor(rowBaseSideMap[s0]);
          auto t_diff_row_base = get_diff_ntensor(rowDiffBaseSideMap[s0]);
          auto t_w = getFTensor0IntegrationWeight();

          for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

            const double alpha = getMeasure() * t_w * sign;

            auto t_mat = locMat.data().begin();

            size_t rr = 0;
            for (; rr != nb_rows; ++rr) {
              auto t_col_base = get_ntensor(colBaseSideMap[s1], gg, 0);
              auto t_diff_col_base =
                  get_diff_ntensor(colDiffBaseSideMap[s1], gg, 0);

              const double row_q = t_diff_row_base(i) * t_normal(i);
              const double row_val =
                  alpha * testing_fun(t_row_base, row_q, p, phi, nitsche);
              const double row_val_plus =
                  alpha * testing_fun_plus(t_row_base, row_q, p, phi, nitsche);

              for (size_t cc = 0; cc != nb_cols; ++cc) {

                const double col_q = t_diff_col_base(i) * t_normal(i);
                *t_mat -= row_val * tested_fun(t_col_base, col_q, p, nitsche);
                *t_mat -= row_val_plus * col_q;

                ++t_col_base;
                ++t_diff_col_base;
                ++t_mat;
              }

              ++t_row_base;
              ++t_diff_row_base;
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

          if (isBoundary)
            MoFEMFunctionReturnHot(0);
        }
      }
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<FaceSideEle> sideFEPtr;
  MatrixDouble locMat;
  bool isBoundary;

};

struct OpL2BoundaryRhs : public BoundaryEleOp {
public:
  OpL2BoundaryRhs(boost::shared_ptr<FaceSideEle> side_fe_ptr, ScalarFun fun)
      : BoundaryEleOp(NOSPACE, BoundaryEleOp::OPLAST), sideFEPtr(side_fe_ptr),
        sourceFun(fun) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    CHKERR loopSideFaces("dFE", *sideFEPtr);

    const double s = getMeasure() / (areaMap[0]);
    const double p = penalty * s;

    auto t_normal = getFTensor1Normal();
    t_normal.normalize();

    auto t_w = getFTensor0IntegrationWeight();

    const size_t nb_rows = indicesRowSideMap[0].size();

    if (!nb_rows)
      MoFEMFunctionReturnHot(0);

    rhsF.resize(nb_rows, false);
    rhsF.clear();

    const size_t nb_integration_pts = getGaussPts().size2();
    const size_t nb_row_base_functions = rowBaseSideMap[0].size2();

    // shape funcs
    auto t_row_base = get_ntensor(rowBaseSideMap[0]);
    auto t_diff_row_base = get_diff_ntensor(rowDiffBaseSideMap[0]);
    auto t_coords = getFTensor1CoordsAtGaussPts();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      const double alpha = getMeasure() * t_w;

      const double source_val =
          -p * sourceFun(t_coords(0), t_coords(1), t_coords(2));

      auto t_f = rhsF.data().begin();

      size_t rr = 0;
      for (; rr != nb_rows; ++rr) {

        const double row_q = t_diff_row_base(i) * t_normal(i);
        const double row_val =
            alpha * testing_fun(t_row_base, row_q, p, phi, nitsche);
        *t_f -= row_val * source_val;

        ++t_row_base;
        ++t_diff_row_base;
        ++t_f;
      }

      for (; rr < nb_row_base_functions; ++rr) {
        ++t_row_base;
        ++t_diff_row_base;
      }

      ++t_w;
      ++t_coords;
    }

    CHKERR ::VecSetValues(getKSPf(), indicesRowSideMap[0].size(),
                          &*indicesRowSideMap[0].begin(), &*rhsF.data().begin(),
                          ADD_VALUES);

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<FaceSideEle> sideFEPtr;
  ScalarFun sourceFun;
  VectorDouble rhsF;
};

}; // namespace Poisson2DiscontGalerkinOperators

#endif //__POISSON2DISGALERKIN_HPP__