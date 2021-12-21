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

namespace FreeSurfaceOps {

struct OpNormalConstrainRhs : public AssemblyBoundaryEleOp {
  OpNormalConstrainRhs(const std::string field_name,
                       boost::shared_ptr<MatrixDouble> u_ptr)
      : AssemblyBoundaryEleOp(field_name, field_name,
                              AssemblyBoundaryEleOp::OPROW),
        uPtr(u_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data) {
    MoFEMFunctionBegin;

    auto t_w = getFTensor0IntegrationWeight();
    auto t_normal = getFTensor1Normal();
    auto t_u = getFTensor1FromMat<SPACE_DIM>(*uPtr);
    auto t_row_base = row_data.getFTensor0N();
    auto t_coords = getFTensor1CoordsAtGaussPts();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * cylindrical(r);

      int bb = 0;
      for (; bb != nbRows; ++bb) {
        locF[bb] += alpha * t_row_base * (t_normal(i) * t_u(i));
        ++t_row_base;
      }

      for (; bb < nbRowBaseFunctions; ++bb)
        ++t_row_base;

      ++t_w;
      ++t_u;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> uPtr;
};

struct OpNormalForcebRhs : public AssemblyBoundaryEleOp {
  OpNormalForcebRhs(const std::string field_name,
                    boost::shared_ptr<VectorDouble> lambda_ptr)
      : AssemblyBoundaryEleOp(field_name, field_name,
                              AssemblyDomainEleOp::OPROW),
        lambdaPtr(lambda_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data) {
    MoFEMFunctionBegin;

    auto t_w = getFTensor0IntegrationWeight();
    auto t_normal = getFTensor1Normal();
    auto t_lambda = getFTensor0FromVec(*lambdaPtr);
    auto t_row_base = row_data.getFTensor0N();
    auto t_coords = getFTensor1CoordsAtGaussPts();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      auto t_nf = getFTensor1FromArray<U_FIELD_DIM, U_FIELD_DIM>(locF);

      const double r = t_coords(0);
      const double alpha = t_w * cylindrical(r);

      int bb = 0;
      for (; bb != nbRows / U_FIELD_DIM; ++bb) {

        t_nf(i) += alpha * t_row_base * t_normal(i) * t_lambda;
        ++t_row_base;
        ++t_nf;
      }

      for (; bb < nbRowBaseFunctions; ++bb)
        ++t_row_base;

      ++t_w;
      ++t_lambda;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<VectorDouble> lambdaPtr;
};

struct OpNormalConstrainLhs : public AssemblyBoundaryEleOp {

  OpNormalConstrainLhs(const std::string field_name_row,
                       const std::string field_name_col)
      : AssemblyBoundaryEleOp(field_name_row, field_name_col,
                              AssemblyBoundaryEleOp::OPROWCOL) {
    assembleTranspose = true;
    sYmm = false;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    auto t_w = getFTensor0IntegrationWeight();
    auto t_normal = getFTensor1Normal();
    auto t_row_base = row_data.getFTensor0N();
    auto t_coords = getFTensor1CoordsAtGaussPts();

    for (int gg = 0; gg != nbIntegrationPts; ++gg) {

      auto t_mat = getFTensor1FromPtr<U_FIELD_DIM>(&locMat(0, 0));

      const double r = t_coords(0);
      const double alpha = t_w * cylindrical(r);

      int rr = 0;
      for (; rr != nbRows; ++rr) {

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        const double a = alpha * t_row_base;

        for (int cc = 0; cc != nbCols / U_FIELD_DIM; ++cc) {
          t_mat(i) += (a * t_col_base) * t_normal(i);
          ++t_col_base;
          ++t_mat;
        }
        ++t_row_base;
      }

      for (; rr < nbRowBaseFunctions; ++rr)
        ++t_row_base;

      ++t_w;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  };
};

/**
 * @brief Rhs for U
 *
 */
struct OpRhsU : public AssemblyDomainEleOp {

  OpRhsU(const std::string field_name,
         boost::shared_ptr<MatrixDouble> dot_u_ptr,
         boost::shared_ptr<MatrixDouble> u_ptr,
         boost::shared_ptr<MatrixDouble> grad_u_ptr,
         boost::shared_ptr<VectorDouble> h_ptr,
         boost::shared_ptr<MatrixDouble> grad_h_ptr,
         boost::shared_ptr<VectorDouble> g_ptr,
         boost::shared_ptr<MatrixDouble> grad_g_ptr,
         boost::shared_ptr<VectorDouble> p_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        dotUPtr(dot_u_ptr), uPtr(u_ptr), gradUPtr(grad_u_ptr), hPtr(h_ptr),
        gradHPtr(grad_h_ptr), gPtr(g_ptr), gradGPtr(grad_g_ptr), pPtr(p_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_dot_u = getFTensor1FromMat<U_FIELD_DIM>(*dotUPtr);
    auto t_u = getFTensor1FromMat<U_FIELD_DIM>(*uPtr);
    auto t_p = getFTensor0FromVec(*pPtr);
    auto t_grad_u = getFTensor2FromMat<U_FIELD_DIM, SPACE_DIM>(*gradUPtr);
    auto t_h = getFTensor0FromVec(*hPtr);
    auto t_grad_h = getFTensor1FromMat<SPACE_DIM>(*gradHPtr);
    auto t_g = getFTensor0FromVec(*gPtr);
    auto t_grad_g = getFTensor1FromMat<SPACE_DIM>(*gradGPtr);
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto t_base = data.getFTensor0N();
    auto t_diff_base = data.getFTensor1DiffN<SPACE_DIM>();

    auto t_w = getFTensor0IntegrationWeight();

    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<double>();
    FTensor::Tensor2_symmetric<double, SPACE_DIM> t_stress;
    FTensor::Tensor1<double, U_FIELD_DIM> t_phase_force;
    FTensor::Tensor1<double, U_FIELD_DIM> t_inertia_force;
    FTensor::Tensor1<double, U_FIELD_DIM> t_a0;
    t_a0(i) = 0;
    t_a0(SPACE_DIM - 1) = a0;

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      const double rho = phase_function(t_h, rho_p, rho_m);
      const double mu = phase_function(t_h, mu_p, mu_m);

      auto t_D = get_D(2 * mu);
      auto t_J = get_J(t_h, t_grad_g);

      t_inertia_force(i) = rho * (t_dot_u(i) + t_a0(i));
      t_stress(i, j) = (t_D(i, j, k, l) * t_grad_u(k, l) + t_kd(i, j) * t_p);
      t_phase_force(i) = (t_J(i) - kappa * t_g * t_grad_h(i));

      auto t_nf = getFTensor1FromArray<U_FIELD_DIM, U_FIELD_DIM>(locF);

      int bb = 0;
      for (; bb != nbRows / U_FIELD_DIM; ++bb) {

        t_nf(i) += (t_base * alpha) * (t_inertia_force(i) + t_phase_force(i));
        t_nf(i) += (t_diff_base(j) * alpha) * t_stress(i, j);

        // When we move to C++17 add if constexpr()
        if (coord_type == CYLINDRICAL) {
          t_nf(0) +=
              (t_base * (alpha / t_coords(0))) * ((2 * mu) * t_u(0) + t_p);
        }

        ++t_base;
        ++t_diff_base;
        ++t_nf;
      }

      for (; bb < nbRowBaseFunctions; ++bb) {
        ++t_diff_base;
        ++t_base;
      }

      ++t_dot_u;
      ++t_u;
      ++t_grad_u;
      ++t_h;
      ++t_grad_h;
      ++t_g;
      ++t_p;

      ++t_w;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> dotUPtr;
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<MatrixDouble> gradUPtr;
  boost::shared_ptr<VectorDouble> hPtr;
  boost::shared_ptr<MatrixDouble> gradHPtr;
  boost::shared_ptr<VectorDouble> gPtr;
  boost::shared_ptr<MatrixDouble> gradGPtr;
  boost::shared_ptr<VectorDouble> pPtr;
};

/**
 * @brief Lhs for U dU
 *
 */
struct OpLhsU_dU : public AssemblyDomainEleOp {

  OpLhsU_dU(const std::string field_name, boost::shared_ptr<MatrixDouble> u_ptr,
            boost::shared_ptr<MatrixDouble> grad_u_ptr,
            boost::shared_ptr<VectorDouble> h_ptr)
      : AssemblyDomainEleOp(field_name, field_name,
                            AssemblyDomainEleOp::OPROWCOL),
        uPtr(u_ptr), gradUPtr(grad_u_ptr), hPtr(h_ptr) {
    sYmm = false;
    assembleTranspose = false;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_u = getFTensor1FromMat<U_FIELD_DIM>(*uPtr);
    auto t_grad_u = getFTensor2FromMat<U_FIELD_DIM, SPACE_DIM>(*gradUPtr);
    auto t_h = getFTensor0FromVec(*hPtr);
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto t_row_base = row_data.getFTensor0N();
    auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();

    auto t_w = getFTensor0IntegrationWeight();

    auto get_mat = [&](const int rr) {
      return getFTensor2FromArray<SPACE_DIM, SPACE_DIM, SPACE_DIM>(locMat, rr);
    };

    auto ts_a = getTSa();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);
      const double rho = phase_function(t_h, rho_p, rho_m);
      const double mu = phase_function(t_h, mu_p, mu_m);

      const double beta = alpha * rho * ts_a;
      auto t_D = get_D(2 * mu);

      int rr = 0;
      for (; rr != nbRows / U_FIELD_DIM; ++rr) {

        auto t_mat = get_mat(rr * U_FIELD_DIM);
        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

        FTensor::Christof<double, SPACE_DIM, SPACE_DIM> t_d_stress;
        // I mix up the indices here so that it behaves like a
        // Dg.  That way I don't have to have a separate wrapper
        // class Christof_Expr, which simplifies things.
        t_d_stress(l, j, k) = t_D(i, j, k, l) * (alpha * t_row_diff_base(i));

        for (int cc = 0; cc != nbCols / U_FIELD_DIM; ++cc) {

          const double bb = t_row_base * t_col_base;

          t_mat(i, j) += beta * bb;
          t_mat(i, j) += t_d_stress(i, j, k) * t_col_diff_base(k);

          // When we move to C++17 add if constexpr()
          if (coord_type == CYLINDRICAL) {
            t_mat(0, 0) += (bb * (alpha / t_coords(0))) * (2 * mu);
          }

          ++t_mat;
          ++t_col_base;
          ++t_col_diff_base;
        }

        ++t_row_base;
        ++t_row_diff_base;
      }

      for (; rr < nbRowBaseFunctions; ++rr) {
        ++t_row_diff_base;
        ++t_row_base;
      }

      ++t_u;
      ++t_grad_u;
      ++t_h;

      ++t_coords;
      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<MatrixDouble> gradUPtr;
  boost::shared_ptr<VectorDouble> hPtr;
};

/**
 * @brief Lhs for U dH
 *
 */
struct OpLhsU_dH : public AssemblyDomainEleOp {

  OpLhsU_dH(const std::string field_name_u, const std::string field_name_h,
            boost::shared_ptr<MatrixDouble> dot_u_ptr,
            boost::shared_ptr<MatrixDouble> u_ptr,
            boost::shared_ptr<MatrixDouble> grad_u_ptr,
            boost::shared_ptr<VectorDouble> h_ptr,
            boost::shared_ptr<VectorDouble> g_ptr,
            boost::shared_ptr<MatrixDouble> grad_g_ptr)
      : AssemblyDomainEleOp(field_name_u, field_name_h,
                            AssemblyDomainEleOp::OPROWCOL),
        dotUPtr(dot_u_ptr), uPtr(u_ptr), gradUPtr(grad_u_ptr), hPtr(h_ptr),
        gPtr(g_ptr), gradGPtr(grad_g_ptr) {
    sYmm = false;
    assembleTranspose = false;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_dot_u = getFTensor1FromMat<U_FIELD_DIM>(*dotUPtr);
    auto t_u = getFTensor1FromMat<U_FIELD_DIM>(*uPtr);
    auto t_grad_u = getFTensor2FromMat<U_FIELD_DIM, SPACE_DIM>(*gradUPtr);
    auto t_h = getFTensor0FromVec(*hPtr);
    auto t_g = getFTensor0FromVec(*gPtr);
    auto t_grad_g = getFTensor1FromMat<SPACE_DIM>(*gradGPtr);
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto t_row_base = row_data.getFTensor0N();
    auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();

    auto t_w = getFTensor0IntegrationWeight();

    FTensor::Tensor2_symmetric<double, SPACE_DIM> t_stress_dh;
    FTensor::Tensor1<double, U_FIELD_DIM> t_phase_force_dh;
    FTensor::Tensor1<double, U_FIELD_DIM> t_inertia_force_dh;
    FTensor::Tensor1<double, U_FIELD_DIM> t_a0;
    t_a0(i) = 0;
    t_a0(SPACE_DIM - 1) = a0;

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      auto rho_dh = d_phase_function_h(t_h, rho_p, rho_m);
      auto mu_dh = d_phase_function_h(t_h, mu_p, mu_m);

      auto t_D_dh = get_D(2 * mu_dh);
      auto t_J_dh = get_J_dh(t_h, t_grad_g);

      t_inertia_force_dh(i) = alpha * (rho_dh * (t_dot_u(i) + t_a0(i)));
      t_stress_dh(i, j) = alpha * (t_D_dh(i, j, k, l) * t_grad_u(k, l));
      t_phase_force_dh(i) = alpha * t_J_dh(i);
      const double t_phase_force_d_diff_h = alpha * kappa * t_g;

      int rr = 0;
      for (; rr != nbRows / U_FIELD_DIM; ++rr) {

        auto t_mat =
            getFTensor1FromMat<U_FIELD_DIM, 1>(locMat, rr * U_FIELD_DIM);
        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

        for (int cc = 0; cc != nbCols; ++cc) {

          const double bb = t_row_base * t_col_base;
          t_mat(i) += t_inertia_force_dh(i) * bb;
          t_mat(i) += (t_row_diff_base(j) * t_col_base) * t_stress_dh(i, j);
          t_mat(i) += t_phase_force_dh(i) * t_col_base;
          t_mat(i) += t_phase_force_d_diff_h * t_col_diff_base(i);

          // When we move to C++17 add if constexpr()
          if (coord_type == CYLINDRICAL) {
            t_mat(0) += (bb * (alpha / t_coords(0))) * (2 * mu_dh * t_u(0));
          }

          ++t_mat;
          ++t_col_base;
          ++t_col_diff_base;
        }

        ++t_row_base;
        ++t_row_diff_base;
      }

      for (; rr < nbRowBaseFunctions; ++rr) {
        ++t_row_diff_base;
        ++t_row_base;
      }

      ++t_dot_u;
      ++t_u;
      ++t_grad_u;
      ++t_h;
      ++t_g;
      ++t_grad_g;
      ++t_coords;
      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> dotUPtr;
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<MatrixDouble> gradUPtr;
  boost::shared_ptr<VectorDouble> hPtr;
  boost::shared_ptr<VectorDouble> gPtr;
  boost::shared_ptr<MatrixDouble> gradGPtr;
};

/**
 * @brief Lhs for G dH
 *
 */
struct OpLhsU_dG : public AssemblyDomainEleOp {

  OpLhsU_dG(const std::string field_name_u, const std::string field_name_h,
            boost::shared_ptr<MatrixDouble> dot_u_ptr,
            boost::shared_ptr<MatrixDouble> u_ptr,
            boost::shared_ptr<MatrixDouble> grad_u_ptr,
            boost::shared_ptr<VectorDouble> h_ptr,
            boost::shared_ptr<MatrixDouble> grad_h_ptr,
            boost::shared_ptr<VectorDouble> g_ptr,
            boost::shared_ptr<MatrixDouble> grad_g_ptr)
      : AssemblyDomainEleOp(field_name_u, field_name_h,
                            AssemblyDomainEleOp::OPROWCOL),
        dotUPtr(dot_u_ptr), uPtr(u_ptr), gradUPtr(grad_u_ptr), hPtr(h_ptr),
        gradHPtr(grad_h_ptr), gPtr(g_ptr), gradGPtr(grad_g_ptr) {
    sYmm = false;
    assembleTranspose = false;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_dot_u = getFTensor1FromMat<U_FIELD_DIM>(*dotUPtr);
    auto t_u = getFTensor1FromMat<U_FIELD_DIM>(*uPtr);
    auto t_grad_u = getFTensor2FromMat<U_FIELD_DIM, SPACE_DIM>(*gradUPtr);
    auto t_h = getFTensor0FromVec(*hPtr);
    auto t_grad_h = getFTensor1FromMat<SPACE_DIM>(*gradHPtr);
    auto t_g = getFTensor0FromVec(*gPtr);
    auto t_grad_g = getFTensor1FromMat<SPACE_DIM>(*gradGPtr);
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto t_row_base = row_data.getFTensor0N();
    auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();

    auto t_w = getFTensor0IntegrationWeight();

    FTensor::Tensor1<double, U_FIELD_DIM> t_phase_force_d_g;

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      auto rho = phase_function(t_h, rho_p, rho_m);
      auto mu = phase_function(t_h, mu_p, mu_m);

      const double t_phase_force_dg = alpha * get_J_dg(t_h, t_grad_g);
      t_phase_force_d_g(i) = alpha * kappa * t_grad_h(i);

      int rr = 0;
      for (; rr != nbRows / U_FIELD_DIM; ++rr) {

        auto t_mat =
            getFTensor1FromMat<U_FIELD_DIM, 1>(locMat, rr * U_FIELD_DIM);
        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

        for (int cc = 0; cc != nbCols; ++cc) {

          const double bb = t_row_base * t_col_base;
          t_mat(i) += t_phase_force_d_g(i) * bb;
          t_mat(i) += t_phase_force_dg * t_col_diff_base(i);

          ++t_mat;
          ++t_col_base;
          ++t_col_diff_base;
        }

        ++t_row_base;
        ++t_row_diff_base;
      }

      for (; rr < nbRowBaseFunctions; ++rr) {
        ++t_row_diff_base;
        ++t_row_base;
      }

      ++t_dot_u;
      ++t_u;
      ++t_grad_u;
      ++t_h;
      ++t_g;
      ++t_grad_g;
      ++t_coords;
      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> dotUPtr;
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<MatrixDouble> gradUPtr;
  boost::shared_ptr<VectorDouble> hPtr;
  boost::shared_ptr<MatrixDouble> gradHPtr;
  boost::shared_ptr<VectorDouble> gPtr;
  boost::shared_ptr<MatrixDouble> gradGPtr;
};

struct OpRhsH : public AssemblyDomainEleOp {

  OpRhsH(const std::string field_name,
         boost::shared_ptr<VectorDouble> dot_h_ptr,
         boost::shared_ptr<VectorDouble> h_ptr,
         boost::shared_ptr<MatrixDouble> grad_g_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        dotHPtr(dot_h_ptr), hPtr(h_ptr), gradGPtr(grad_g_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_dot_h = getFTensor0FromVec(*dotHPtr);
    auto t_h = getFTensor0FromVec(*hPtr);
    auto t_grad_g = getFTensor1FromMat<SPACE_DIM>(*gradGPtr);
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto t_base = data.getFTensor0N();
    auto t_diff_base = data.getFTensor1DiffN<SPACE_DIM>();
    auto t_w = getFTensor0IntegrationWeight();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      const double m = get_M(t_h) * alpha;

      int bb = 0;
      for (; bb != nbRows; ++bb) {
        locF[bb] += (t_base * alpha) * t_dot_h;
        locF[bb] += (t_diff_base(i) * m) * t_grad_g(i);
        ++t_base;
        ++t_diff_base;
      }

      for (; bb < nbRowBaseFunctions; ++bb) {
        ++t_base;
        ++t_diff_base;
      }

      ++t_dot_h;
      ++t_h;
      ++t_grad_g;

      ++t_coords;
      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<VectorDouble> dotHPtr;
  boost::shared_ptr<VectorDouble> hPtr;
  boost::shared_ptr<MatrixDouble> gradGPtr;
};

/**
 * @brief Lhs for H dU
 *
 */
struct OpLhsH_dU : public AssemblyDomainEleOp {
  OpLhsH_dU(const std::string h_field_name, const std::string u_field_name)
      : AssemblyDomainEleOp(h_field_name, u_field_name,
                            AssemblyDomainEleOp::OPROWCOL) {
    sYmm = false;
    assembleTranspose = false;
  }
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;
    MoFEMFunctionReturn(0);
  }

private:
};

/**
 * @brief Lhs for H dH
 *
 */
struct OpLhsH_dH : public AssemblyDomainEleOp {

  OpLhsH_dH(const std::string field_name, boost::shared_ptr<VectorDouble> h_ptr,
            boost::shared_ptr<MatrixDouble> grad_g_ptr)
      : AssemblyDomainEleOp(field_name, field_name,
                            AssemblyDomainEleOp::OPROWCOL),
        hPtr(h_ptr), gradGPtr(grad_g_ptr) {
    sYmm = false;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_h = getFTensor0FromVec(*hPtr);
    auto t_grad_g = getFTensor1FromMat<SPACE_DIM>(*gradGPtr);

    auto t_row_base = row_data.getFTensor0N();
    auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_coords = getFTensor1CoordsAtGaussPts();
    auto ts_a = getTSa();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      auto m_dh = get_M_dh(t_h) * alpha;

      int rr = 0;
      for (; rr != nbRows; ++rr) {

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

        for (int cc = 0; cc != nbCols; ++cc) {

          locMat(rr, cc) += (t_row_base * t_col_base * alpha) * ts_a;
          locMat(rr, cc) +=
              (t_row_diff_base(i) * t_grad_g(i)) * (t_col_base * m_dh);

          ++t_col_base;
          ++t_col_diff_base;
        }

        ++t_row_base;
        ++t_row_diff_base;
      }

      for (; rr < nbRowBaseFunctions; ++rr) {
        ++t_row_base;
        ++t_row_diff_base;
      }

      ++t_h;
      ++t_grad_g;
      ++t_w;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<VectorDouble> hPtr;
  boost::shared_ptr<MatrixDouble> gradGPtr;
};

struct OpRhsG : public AssemblyDomainEleOp {

  OpRhsG(const std::string field_name, boost::shared_ptr<VectorDouble> h_ptr,
         boost::shared_ptr<MatrixDouble> grad_h_ptr,
         boost::shared_ptr<VectorDouble> g_ptr, )
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        dotHPtr(dot_h_ptr), hPtr(h_ptr), gradGPtr(grad_g_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_h = getFTensor0FromVec(*hPtr);
    auto t_grad_h = getFTensor1FromMat<SPACE_DIM>(*gradGPtr);
    auto t_g = getFTensor0FromVec(*gPtr);
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto t_base = data.getFTensor0N();
    auto t_diff_base = data.getFTensor1DiffN<SPACE_DIM>();
    auto t_w = getFTensor0IntegrationWeight();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      const double f = get_F(t_h);

      int bb = 0;
      for (; bb != nbRows; ++bb) {
        locF[bb] += (t_base * alpha) * (t_g + h);
        locF[bb] -= (t_diff_base(i) * eta * eta) * t_grad_h(i);
        ++t_base;
        ++t_diff_base;
      }

      for (; bb < nbRowBaseFunctions; ++bb) {
        ++t_base;
        ++t_diff_base;
      }

      ++t_dot_h;
      ++t_h;
      ++t_grad_g;

      ++t_coords;
      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<VectorDouble> dotHPtr;
  boost::shared_ptr<VectorDouble> hPtr;
  boost::shared_ptr<MatrixDouble> gradGPtr;
};

/**
 * @brief Lhs for H dH
 *
 */
struct OpLhsG_dH : public AssemblyDomainEleOp {

  OpLhsH_dH(const std::string field_name,
            boost::shared_ptr<VectorDouble> h_ptrr)
      : AssemblyDomainEleOp(field_name, field_name,
                            AssemblyDomainEleOp::OPROWCOL),
        hPtr(h_ptr) {
    sYmm = false;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_h = getFTensor0FromVec(*hPtr);

    auto t_row_base = row_data.getFTensor0N();
    auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_coords = getFTensor1CoordsAtGaussPts();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      const double f_dh = get_F_dh(t_h) * alpha;
      const double beta = eta * eta * alpha;

      int rr = 0;
      for (; rr != nbRows; ++rr) {

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

        for (int cc = 0; cc != nbCols; ++cc) {

          locMat(rr, cc) += (t_row_base * t_col_base * alpha) * f_dh;
          locMat(rr, cc) -= (t_row_diff_base(i) * beta) * t_col_diff_base(i);

          ++t_col_base;
          ++t_col_diff_base;
        }

        ++t_row_base;
        ++t_row_diff_base;
      }

      for (; rr < nbRowBaseFunctions; ++rr) {
        ++t_row_base;
        ++t_row_diff_base;
      }

      ++t_h;
      ++t_grad_g;
      ++t_w;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<VectorDouble> hPtr;
};

/**
 * @brief Lhs for H dH
 *
 */
struct OpLhsG_dG : public AssemblyDomainEleOp {

  OpLhsH_dH(const std::string field_name,
            boost::shared_ptr<VectorDouble> h_ptrr)
      : AssemblyDomainEleOp(field_name, field_name,
                            AssemblyDomainEleOp::OPROWCOL),
        hPtr(h_ptr) {
    sYmm = false;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_h = getFTensor0FromVec(*hPtr);

    auto t_row_base = row_data.getFTensor0N();
    auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_coords = getFTensor1CoordsAtGaussPts();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      const double f_dh = get_F_dh(t_h) * alpha;
      const double beta = eta * eta * alpha;

      int rr = 0;
      for (; rr != nbRows; ++rr) {

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

        for (int cc = 0; cc != nbCols; ++cc) {

          locMat(rr, cc) += (t_row_base * t_col_base * alpha);

          ++t_col_base;
          ++t_col_diff_base;
        }

        ++t_row_base;
        ++t_row_diff_base;
      }

      for (; rr < nbRowBaseFunctions; ++rr) {
        ++t_row_base;
        ++t_row_diff_base;
      }

      ++t_h;
      ++t_grad_g;
      ++t_w;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<VectorDouble> hPtr;
};

/**
 * @brief Explict term for IMEX method
 *
 */
struct OpRhsExplicitTermU : public AssemblyDomainEleOp {

  OpRhsExplicitTermU(const std::string field_name,
                     boost::shared_ptr<MatrixDouble> u_ptr,
                     boost::shared_ptr<MatrixDouble> grad_u_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        uPtr(u_ptr), gradUPtr(grad_u_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_coords = getFTensor1CoordsAtGaussPts();
    auto t_u = getFTensor1FromMat<U_FIELD_DIM>(*uPtr);
    auto t_grad_u = getFTensor2FromMat<U_FIELD_DIM, SPACE_DIM>(*gradUPtr);

    auto t_base = data.getFTensor0N();
    auto t_diff_base = data.getFTensor1DiffN<SPACE_DIM>();

    auto t_w = getFTensor0IntegrationWeight();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      FTensor::Tensor1<double, U_FIELD_DIM> t_convection;
      t_convection(i) = t_u(j) * t_grad_u(i, j);

      auto t_nf = getFTensor1FromArray<U_FIELD_DIM, U_FIELD_DIM>(locF);

      int bb = 0;
      for (; bb != nbRows / U_FIELD_DIM; ++bb) {

        t_nf(i) += (t_base * alpha) * t_convection(i);

        ++t_base;
        ++t_nf;
      }

      for (; bb < nbRowBaseFunctions; ++bb) {
        ++t_base;
      }

      ++t_u;
      ++t_grad_u;

      ++t_w;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<MatrixDouble> gradUPtr;
};

/**
 * @brief Explict term for IMEX method
 *
 */
struct OpRhsExplicitTermH : public AssemblyDomainEleOp {

  OpRhsExplicitTermH(const std::string field_name,
                     boost::shared_ptr<VectorDouble> phi_ptr,
                     boost::shared_ptr<double> ksi_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        phiPtr(phi_ptr), ksiPtr(ksi_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_coords = getFTensor1CoordsAtGaussPts();
    auto t_base = data.getFTensor0N();
    auto t_phi = getFTensor0FromVec(*phiPtr);
    const double ksi = *ksiPtr;

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);
      auto nf_ptr = &locF[0];

      int rr = 0;
      for (; rr != nbRows; ++rr) {

        (*nf_ptr) += (alpha * lambda) * (ksi * t_base);

        ++nf_ptr;
        ++t_base;
      }

      ++t_phi;
      ++t_w;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<VectorDouble> phiPtr;
  boost::shared_ptr<double> ksiPtr;
};



} // namespace FreeSurfaceOps