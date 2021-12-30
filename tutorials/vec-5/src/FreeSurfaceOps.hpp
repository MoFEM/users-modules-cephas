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
    FTensor::Tensor1<double, U_FIELD_DIM> t_convection;
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

      t_inertia_force(i) = (rho * alpha) * (t_dot_u(i) - t_a0(i));
      if constexpr (!explict_convection)
        t_convection(i) = (rho * alpha) * (t_u(j) * t_grad_u(i, j));
      else
        t_convection(i) = 0;
      t_stress(i, j) =
          alpha * (t_D(i, j, k, l) * t_grad_u(k, l) + t_kd(i, j) * t_p);
      if constexpr (diffusive_flux_term)
        t_phase_force(i) =
            -alpha * (t_J(j) * t_grad_u(i, j) + kappa * t_g * t_grad_h(i));
      else
        t_phase_force(i) = -alpha * kappa * t_g * t_grad_h(i);
      auto t_nf = getFTensor1FromArray<U_FIELD_DIM, U_FIELD_DIM>(locF);

      int bb = 0;
      for (; bb != nbRows / U_FIELD_DIM; ++bb) {

        t_nf(i) +=
            t_base * (t_inertia_force(i) + t_convection(i) + t_phase_force(i));
        t_nf(i) += t_diff_base(j) * t_stress(i, j);

        // When we move to C++17 add if constexpr()
        if constexpr (coord_type == CYLINDRICAL) {
          t_nf(0) += (t_base * (alpha / t_coords(0))) * (2 * mu * t_u(0) + t_p);
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
      ++t_grad_g;
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
            boost::shared_ptr<VectorDouble> h_ptr,
            boost::shared_ptr<MatrixDouble> grad_g_ptr)
      : AssemblyDomainEleOp(field_name, field_name,
                            AssemblyDomainEleOp::OPROWCOL),
        uPtr(u_ptr), gradUPtr(grad_u_ptr), hPtr(h_ptr), gradGPtr(grad_g_ptr) {
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
    auto t_grad_g = getFTensor1FromMat<SPACE_DIM>(*gradGPtr);
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

      const double beta0 = alpha * rho;
      const double beta1 = beta0 * ts_a;
      auto t_D = get_D(alpha * 2 * mu);
      auto t_J = get_J(t_h, t_grad_g);

      FTensor::Tensor1<double, SPACE_DIM> t_phase_force_du;
      t_phase_force_du(i) = -alpha * t_J(i);

      int rr = 0;
      for (; rr != nbRows / U_FIELD_DIM; ++rr) {

        auto t_mat = get_mat(rr * U_FIELD_DIM);
        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

        FTensor::Christof<double, SPACE_DIM, SPACE_DIM> t_d_stress;
        // I mix up the indices here so that it behaves like a
        // Dg.  That way I don't have to have a separate wrapper
        // class Christof_Expr, which simplifies things.
        t_d_stress(l, j, k) = t_D(i, j, k, l) * t_row_diff_base(i);

        for (int cc = 0; cc != nbCols / U_FIELD_DIM; ++cc) {

          const double bb = t_row_base * t_col_base;

          t_mat(i, j) += (beta1 * bb) * t_kd(i, j);
          if constexpr (!explict_convection) {
            t_mat(i, j) += (beta0 * bb) * t_grad_u(i, j);
            t_mat(i, j) += (beta0 * t_row_base) * (t_col_diff_base(k) * t_u(k));
          }
          if constexpr (diffusive_flux_term)
            t_mat(i, j) += (t_row_base * t_phase_force_du(k)) *
                           t_col_diff_base(k) * t_kd(i, j);
          t_mat(i, j) += t_d_stress(i, j, k) * t_col_diff_base(k);

          // When we move to C++17 add if constexpr()
          if constexpr (coord_type == CYLINDRICAL) {
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
      ++t_grad_g;

      ++t_coords;
      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<MatrixDouble> gradUPtr;
  boost::shared_ptr<VectorDouble> hPtr;
  boost::shared_ptr<MatrixDouble> gradGPtr;
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
    FTensor::Tensor1<double, SPACE_DIM> t_convection;
    FTensor::Tensor1<double, U_FIELD_DIM> t_a0;
    t_a0(i) = 0;
    t_a0(SPACE_DIM - 1) = a0;

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      const double rho_dh = d_phase_function_h(t_h, rho_p, rho_m);
      const double mu_dh = d_phase_function_h(t_h, mu_p, mu_m);

      auto t_D_dh = get_D(alpha * mu_dh);
      auto t_J_dh = get_J_dh(t_h, t_grad_g);
      if constexpr (!explict_convection)
        t_convection(i) = (rho_dh * alpha) * (t_u(j) * t_grad_u(i, j));
      else
        t_convection(i) = 0;
      t_inertia_force_dh(i) = (alpha * rho_dh) * (t_dot_u(i) - t_a0(i));
      t_stress_dh(i, j) = t_D_dh(i, j, k, l) * t_grad_u(k, l);
      t_phase_force_dh(i) = -alpha * t_J_dh(j) * t_grad_u(i, j);
      const double t_phase_force_g_dh = -alpha * kappa * t_g;

      int rr = 0;
      for (; rr != nbRows / U_FIELD_DIM; ++rr) {

        auto t_mat =
            getFTensor1FromMat<U_FIELD_DIM, 1>(locMat, rr * U_FIELD_DIM);
        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

        for (int cc = 0; cc != nbCols; ++cc) {

          const double bb = t_row_base * t_col_base;
          t_mat(i) += (t_inertia_force_dh(i) + t_convection(i)) * bb;
          t_mat(i) += (t_row_diff_base(j) * t_col_base) * t_stress_dh(i, j);
          if constexpr (diffusive_flux_term)
            t_mat(i) += t_phase_force_dh(i) * t_col_base;
          t_mat(i) += t_phase_force_g_dh * t_col_diff_base(i);

          // When we move to C++17 add if constexpr()
          if constexpr (coord_type == CYLINDRICAL) {
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
            boost::shared_ptr<MatrixDouble> grad_u_ptr,
            boost::shared_ptr<VectorDouble> h_ptr,
            boost::shared_ptr<MatrixDouble> grad_h_ptr,
            boost::shared_ptr<MatrixDouble> grad_g_ptr)
      : AssemblyDomainEleOp(field_name_u, field_name_h,
                            AssemblyDomainEleOp::OPROWCOL),
        gradUPtr(grad_u_ptr), hPtr(h_ptr), gradHPtr(grad_h_ptr),
        gradGPtr(grad_g_ptr) {
    sYmm = false;
    assembleTranspose = false;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_grad_u = getFTensor2FromMat<U_FIELD_DIM, SPACE_DIM>(*gradUPtr);
    auto t_h = getFTensor0FromVec(*hPtr);
    auto t_grad_h = getFTensor1FromMat<SPACE_DIM>(*gradHPtr);
    auto t_grad_g = getFTensor1FromMat<SPACE_DIM>(*gradGPtr);
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto t_row_base = row_data.getFTensor0N();
    auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();

    auto t_w = getFTensor0IntegrationWeight();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      const double rho = phase_function(t_h, rho_p, rho_m);

      const double J_dg = -alpha * get_J_dg(t_h);
      FTensor::Tensor1<double, SPACE_DIM> t_phase_force_dg;
      t_phase_force_dg(i) = -alpha * kappa * t_grad_h(i);

      int rr = 0;
      for (; rr != nbRows / U_FIELD_DIM; ++rr) {

        auto t_mat =
            getFTensor1FromMat<U_FIELD_DIM, 1>(locMat, rr * U_FIELD_DIM);
        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

        for (int cc = 0; cc != nbCols; ++cc) {

          const double bb = t_row_base * t_col_base;
          t_mat(i) += t_phase_force_dg(i) * bb;
          if constexpr (diffusive_flux_term)
            t_mat(i) +=
                (t_row_base * J_dg) * (t_col_diff_base(j) * t_grad_u(i, j));

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

      ++t_grad_u;
      ++t_h;
      ++t_grad_h;
      ++t_grad_g;
      ++t_coords;
      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> gradUPtr;
  boost::shared_ptr<VectorDouble> hPtr;
  boost::shared_ptr<MatrixDouble> gradHPtr;
  boost::shared_ptr<MatrixDouble> gradGPtr;
};

struct OpRhsH : public AssemblyDomainEleOp {

  OpRhsH(const std::string field_name, boost::shared_ptr<MatrixDouble> u_ptr,
         boost::shared_ptr<VectorDouble> dot_h_ptr,
         boost::shared_ptr<VectorDouble> h_ptr,
         boost::shared_ptr<MatrixDouble> grad_h_ptr,
         boost::shared_ptr<MatrixDouble> grad_g_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        uPtr(u_ptr), dotHPtr(dot_h_ptr), hPtr(h_ptr), gradHPtr(grad_h_ptr),
        gradGPtr(grad_g_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_dot_h = getFTensor0FromVec(*dotHPtr);
    auto t_h = getFTensor0FromVec(*hPtr);
    auto t_u = getFTensor1FromMat<U_FIELD_DIM>(*uPtr);
    auto t_grad_h = getFTensor1FromMat<SPACE_DIM>(*gradHPtr);
    auto t_grad_g = getFTensor1FromMat<SPACE_DIM>(*gradGPtr);
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto t_base = data.getFTensor0N();
    auto t_diff_base = data.getFTensor1DiffN<SPACE_DIM>();
    auto t_w = getFTensor0IntegrationWeight();

    for (int gg = 0; gg != nbIntegrationPts; ++gg) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      const double m = get_M(t_h) * alpha;

      int bb = 0;
      for (; bb != nbRows; ++bb) {
        locF[bb] += (t_base * alpha) * (t_dot_h);
        if constexpr (!explict_convection)
          locF[bb] += (t_base * alpha) * (t_grad_h(i) * t_u(i));
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
      ++t_u;
      ++t_grad_h;

      ++t_coords;
      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<VectorDouble> dotHPtr;
  boost::shared_ptr<VectorDouble> hPtr;
  boost::shared_ptr<MatrixDouble> gradHPtr;
  boost::shared_ptr<MatrixDouble> gradGPtr;
};

/**
 * @brief Lhs for H dU
 *
 */
struct OpLhsH_dU : public AssemblyDomainEleOp {
  OpLhsH_dU(const std::string h_field_name, const std::string u_field_name,
            boost::shared_ptr<MatrixDouble> grad_h_ptr)
      : AssemblyDomainEleOp(h_field_name, u_field_name,
                            AssemblyDomainEleOp::OPROWCOL),
        gradHPtr(grad_h_ptr) {
    sYmm = false;
    assembleTranspose = false;
  }
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    if constexpr (!explict_convection) {

      const double vol = getMeasure();
      auto t_grad_h = getFTensor1FromMat<SPACE_DIM>(*gradHPtr);
      auto t_coords = getFTensor1CoordsAtGaussPts();

      auto t_row_base = row_data.getFTensor0N();
      auto t_w = getFTensor0IntegrationWeight();

      for (int gg = 0; gg != nbIntegrationPts; gg++) {

        const double alpha = t_w * vol;

        int rr = 0;
        for (; rr != nbRows; ++rr) {
          auto t_mat = getFTensor1FromPtr<U_FIELD_DIM>(&locMat(rr, 0));
          auto t_col_base = col_data.getFTensor0N(gg, 0);
          for (int cc = 0; cc != nbCols / U_FIELD_DIM; ++cc) {
            t_mat(i) += (t_row_base * t_col_base * alpha) * t_grad_h(i);
            ++t_mat;
            ++t_col_base;
          }
          ++t_row_base;
        }

        for (; rr < nbRowBaseFunctions; ++rr)
          ++t_row_base;

        ++t_grad_h;
        ++t_w;
      }
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> gradHPtr;
};

/**
 * @brief Lhs for H dH
 *
 */
struct OpLhsH_dH : public AssemblyDomainEleOp {

  OpLhsH_dH(const std::string field_name, boost::shared_ptr<MatrixDouble> u_ptr,
            boost::shared_ptr<VectorDouble> h_ptr,
            boost::shared_ptr<MatrixDouble> grad_g_ptr)
      : AssemblyDomainEleOp(field_name, field_name,
                            AssemblyDomainEleOp::OPROWCOL),
        uPtr(u_ptr), hPtr(h_ptr), gradGPtr(grad_g_ptr) {
    sYmm = false;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_h = getFTensor0FromVec(*hPtr);
    auto t_grad_g = getFTensor1FromMat<SPACE_DIM>(*gradGPtr);
    auto t_u = getFTensor1FromMat<U_FIELD_DIM>(*uPtr);

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
          if constexpr (!explict_convection) {
            locMat(rr, cc) +=
                (t_row_base * alpha) * (t_col_diff_base(i) * t_u(i));
          }
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

      ++t_u;
      ++t_h;
      ++t_grad_g;
      ++t_w;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<VectorDouble> hPtr;
  boost::shared_ptr<MatrixDouble> gradGPtr;
};

/**
 * @brief Lhs for H dH
 *
 */
struct OpLhsH_dG : public AssemblyDomainEleOp {

  OpLhsH_dG(const std::string field_name_h, const std::string field_name_g,
            boost::shared_ptr<VectorDouble> h_ptr)
      : AssemblyDomainEleOp(field_name_h, field_name_g,
                            AssemblyDomainEleOp::OPROWCOL),
        hPtr(h_ptr) {
    sYmm = false;
    assembleTranspose = false;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_h = getFTensor0FromVec(*hPtr);

    auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_coords = getFTensor1CoordsAtGaussPts();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      auto m = get_M(t_h) * alpha;

      int rr = 0;
      for (; rr != nbRows; ++rr) {
        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

        for (int cc = 0; cc != nbCols; ++cc) {
          locMat(rr, cc) += (t_row_diff_base(i) * t_col_diff_base(i)) * m;

          ++t_col_diff_base;
        }

        ++t_row_diff_base;
      }

      for (; rr < nbRowBaseFunctions; ++rr) {
        ++t_row_diff_base;
      }

      ++t_h;
      ++t_w;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<VectorDouble> hPtr;
};

struct OpRhsG : public AssemblyDomainEleOp {

  OpRhsG(const std::string field_name, boost::shared_ptr<VectorDouble> h_ptr,
         boost::shared_ptr<MatrixDouble> grad_h_ptr,
         boost::shared_ptr<VectorDouble> g_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        hPtr(h_ptr), gradHPtr(grad_h_ptr), gPtr(g_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_h = getFTensor0FromVec(*hPtr);
    auto t_grad_h = getFTensor1FromMat<SPACE_DIM>(*gradHPtr);
    auto t_g = getFTensor0FromVec(*gPtr);
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto t_base = data.getFTensor0N();
    auto t_diff_base = data.getFTensor1DiffN<SPACE_DIM>();
    auto t_w = getFTensor0IntegrationWeight();

    for (int gg = 0; gg != nbIntegrationPts; ++gg) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);
      const double f = get_f(t_h);

      int bb = 0;
      for (; bb != nbRows; ++bb) {
        locF[bb] += (t_base * alpha) * (t_g - f);
        locF[bb] -= (t_diff_base(i) * (eta2 * alpha)) * t_grad_h(i);
        ++t_base;
        ++t_diff_base;
      }

      for (; bb < nbRowBaseFunctions; ++bb) {
        ++t_base;
        ++t_diff_base;
      }

      ++t_h;
      ++t_grad_h;
      ++t_g;

      ++t_coords;
      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<VectorDouble> hPtr;
  boost::shared_ptr<MatrixDouble> gradHPtr;
  boost::shared_ptr<VectorDouble> gPtr;
};

/**
 * @brief Lhs for H dH
 *
 */
struct OpLhsG_dH : public AssemblyDomainEleOp {

  OpLhsG_dH(const std::string field_name_g, const std::string field_name_h,
            boost::shared_ptr<VectorDouble> h_ptr)
      : AssemblyDomainEleOp(field_name_g, field_name_h,
                            AssemblyDomainEleOp::OPROWCOL),
        hPtr(h_ptr) {
    sYmm = false;
    assembleTranspose = false;
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

    for (int gg = 0; gg != nbIntegrationPts; ++gg) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      const double f_dh = get_f_dh(t_h) * alpha;
      const double beta = eta2 * alpha;

      int rr = 0;
      for (; rr != nbRows; ++rr) {

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

        for (int cc = 0; cc != nbCols; ++cc) {

          locMat(rr, cc) -= (t_row_base * t_col_base) * f_dh;
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

  OpLhsG_dG(const std::string field_name)
      : AssemblyDomainEleOp(field_name, field_name,
                            AssemblyDomainEleOp::OPROWCOL) {
    sYmm = false;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();

    auto t_row_base = row_data.getFTensor0N();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_coords = getFTensor1CoordsAtGaussPts();

    for (int gg = 0; gg != nbIntegrationPts; ++gg) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      int rr = 0;
      for (; rr != nbRows; ++rr) {
        auto t_col_base = col_data.getFTensor0N(gg, 0);
        const double beta = alpha * t_row_base;
        for (int cc = 0; cc != nbCols; ++cc) {
          locMat(rr, cc) += (t_col_base * beta);
          ++t_col_base;
        }

        ++t_row_base;
      }

      for (; rr < nbRowBaseFunctions; ++rr) {
        ++t_row_base;
      }

      ++t_w;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
};

/**
 * @brief Explict term for IMEX method
 *
 */
struct OpRhsExplicitTermU : public AssemblyDomainEleOp {

  OpRhsExplicitTermU(const std::string field_name,
                     boost::shared_ptr<MatrixDouble> u_ptr,
                     boost::shared_ptr<MatrixDouble> grad_u_ptr,
                     boost::shared_ptr<VectorDouble> h_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        uPtr(u_ptr), gradUPtr(grad_u_ptr), hPtr(h_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    if constexpr (explict_convection) {

      const double vol = getMeasure();
      auto t_coords = getFTensor1CoordsAtGaussPts();
      auto t_u = getFTensor1FromMat<U_FIELD_DIM>(*uPtr);
      auto t_grad_u = getFTensor2FromMat<U_FIELD_DIM, SPACE_DIM>(*gradUPtr);
      auto t_h = getFTensor0FromVec(*hPtr);

      auto t_base = data.getFTensor0N();
      auto t_diff_base = data.getFTensor1DiffN<SPACE_DIM>();

      auto t_w = getFTensor0IntegrationWeight();

      for (int gg = 0; gg != nbIntegrationPts; gg++) {

        const double r = t_coords(0);
        const double alpha = t_w * vol * cylindrical(r);

        const double rho = phase_function(t_h, rho_p, rho_m);

        FTensor::Tensor1<double, U_FIELD_DIM> t_convection;
        t_convection(i) = t_u(j) * t_grad_u(i, j) / rho;

        auto t_nf = getFTensor1FromArray<U_FIELD_DIM, U_FIELD_DIM>(locF);

        int bb = 0;
        for (; bb != nbRows / U_FIELD_DIM; ++bb) {

          t_nf(i) -= (t_base * alpha) * t_convection(i);

          ++t_base;
          ++t_nf;
        }

        for (; bb < nbRowBaseFunctions; ++bb) {
          ++t_base;
        }

        ++t_u;
        ++t_grad_u;
        ++t_h;

        ++t_w;
        ++t_coords;
      }
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<MatrixDouble> gradUPtr;
  boost::shared_ptr<VectorDouble> hPtr;
};

/**
 * @brief Explict term for IMEX method
 *
 */
struct OpRhsExplicitTermH : public AssemblyDomainEleOp {

  OpRhsExplicitTermH(const std::string field_name,
                     boost::shared_ptr<MatrixDouble> u_ptr,
                     boost::shared_ptr<MatrixDouble> grad_h_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        uPtr(u_ptr), gradHPtr(grad_h_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    if constexpr (explict_convection) {

      const double vol = getMeasure();
      auto t_w = getFTensor0IntegrationWeight();
      auto t_coords = getFTensor1CoordsAtGaussPts();
      auto t_base = data.getFTensor0N();

      auto t_u = getFTensor1FromMat<U_FIELD_DIM>(*uPtr);
      auto t_grad_h = getFTensor1FromMat<SPACE_DIM>(*gradHPtr);

      for (int gg = 0; gg != nbIntegrationPts; gg++) {

        const double r = t_coords(0);
        const double alpha = t_w * vol * cylindrical(r);
        auto nf_ptr = &locF[0];

        int rr = 0;
        for (; rr != nbRows; ++rr) {

          (*nf_ptr) -= (alpha * t_base) * t_u(i) * t_grad_h(i);

          ++nf_ptr;
          ++t_base;
        }

        ++t_u;
        ++t_grad_h;
        ++t_w;
        ++t_coords;
      }
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<MatrixDouble> gradHPtr;
};

} // namespace FreeSurfaceOps