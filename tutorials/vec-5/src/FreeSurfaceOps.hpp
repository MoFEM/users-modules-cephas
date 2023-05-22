namespace FreeSurfaceOps {

struct OpCalculateLift : public BoundaryEleOp {
  OpCalculateLift(const std::string field_name,
                  boost::shared_ptr<VectorDouble> p_ptr,
                  boost::shared_ptr<VectorDouble> lift_ptr,
                  boost::shared_ptr<Range> ents_ptr)
      : BoundaryEleOp(field_name, field_name, BoundaryEleOp::OPROW),
        pPtr(p_ptr), liftPtr(lift_ptr), entsPtr(ents_ptr) {
    std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
    doEntities[MBEDGE] = true;
  }

  MoFEMErrorCode doWork(int row_side, EntityType row_type,
                        HookeElement::EntData &data) {
    MoFEMFunctionBegin;

    const auto fe_ent = getFEEntityHandle();
    if (entsPtr->find(fe_ent) != entsPtr->end()) {

      auto t_w = getFTensor0IntegrationWeight();
      auto t_p = getFTensor0FromVec(*pPtr);
      auto t_normal = getFTensor1Normal();
      auto t_coords = getFTensor1CoordsAtGaussPts();
      auto t_lift = getFTensor1FromArray<SPACE_DIM, SPACE_DIM>(*liftPtr);

      const auto nb_int_points = getGaussPts().size2();

      for (int gg = 0; gg != nb_int_points; gg++) {

        const double r = t_coords(0);
        const double alpha = cylindrical(r) * t_w;
        t_lift(i) -= t_normal(i) * (t_p * alpha);

        ++t_w;
        ++t_p;
        ++t_coords;
      }
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<VectorDouble> pPtr;
  boost::shared_ptr<VectorDouble> liftPtr;
  boost::shared_ptr<Range> entsPtr;
};

struct OpNormalConstrainRhs : public AssemblyBoundaryEleOp {
  OpNormalConstrainRhs(const std::string field_name,
                       boost::shared_ptr<MatrixDouble> u_ptr)
      : AssemblyBoundaryEleOp(field_name, field_name,
                              AssemblyBoundaryEleOp::OPROW),
        uPtr(u_ptr) {}

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data) {
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

struct OpNormalForceRhs : public AssemblyBoundaryEleOp {
  OpNormalForceRhs(const std::string field_name,
                   boost::shared_ptr<VectorDouble> lambda_ptr)
      : AssemblyBoundaryEleOp(field_name, field_name,
                              AssemblyDomainEleOp::OPROW),
        lambdaPtr(lambda_ptr) {}

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data) {
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

struct OpWettingAngleRhs : public AssemblyBoundaryEleOp {
  OpWettingAngleRhs(const std::string field_name,
                    boost::shared_ptr<MatrixDouble> grad_h_ptr,
                    boost::shared_ptr<Range> ents_ptr = nullptr,
                    double wetting_angle = 0)
      : AssemblyBoundaryEleOp(field_name, field_name,
                              AssemblyBoundaryEleOp::OPROW),
        gradHPtr(grad_h_ptr), entsPtr(ents_ptr), wettingAngle(wetting_angle) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data) {
    MoFEMFunctionBegin;
    if (entsPtr) {
      if (entsPtr->find(AssemblyBoundaryEleOp::getFEEntityHandle()) ==
          entsPtr->end())
        MoFEMFunctionReturnHot(0);
    }
    const double area = getMeasure();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    auto t_grad_h = getFTensor1FromMat<SPACE_DIM>(*gradHPtr);
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto s = wetting_angle_sub_stepping(getFEMethod()->ts_step);

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * cylindrical(r) * area;
      const double h_grad_norm = sqrt(t_grad_h(i) * t_grad_h(i) +
                                      std::numeric_limits<double>::epsilon());
      const double cos_angle = std::cos(M_PI * wettingAngle / 180);
      const double rhs_wetting = s * eta2 * h_grad_norm * cos_angle;

      // cerr << "pass "
      //      << h_grad_norm <<"\n";
      int bb = 0;
      for (; bb != nbRows; ++bb) {
        locF[bb] += alpha * t_row_base * rhs_wetting;
        ++t_row_base;
      }

      for (; bb < nbRowBaseFunctions; ++bb)
        ++t_row_base;

      ++t_w;
      ++t_grad_h;
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> gradHPtr;
  boost::shared_ptr<Range> entsPtr;
  double wettingAngle;
};

struct OpNormalConstrainLhs : public AssemblyBoundaryEleOp {

  OpNormalConstrainLhs(const std::string field_name_row,
                       const std::string field_name_col)
      : AssemblyBoundaryEleOp(field_name_row, field_name_col,
                              AssemblyBoundaryEleOp::OPROWCOL) {
    assembleTranspose = true;
    sYmm = false;
  }

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data) {
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

struct OpWettingAngleLhs : public BoundaryEleOp {

  OpWettingAngleLhs(
      const std::string row_field_name,
      boost::shared_ptr<MatrixDouble> grad_h_ptr,
      boost::shared_ptr<std::vector<VectorInt>> col_ind_ptr,
      boost::shared_ptr<std::vector<MatrixDouble>> col_diff_base_ptr,
      boost::shared_ptr<Range> ents_ptr = nullptr, double wetting_angle = 0)
      : BoundaryEleOp(row_field_name, BoundaryEleOp::OPROW),
        gradHPtr(grad_h_ptr), colIndicesPtr(col_ind_ptr),
        colDiffBaseFunctionsPtr(col_diff_base_ptr), entsPtr(ents_ptr),
        wettingAngle(wetting_angle) {}

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;
    if (entsPtr) {
      if (entsPtr->find(BoundaryEleOp::getFEEntityHandle()) == entsPtr->end())
        MoFEMFunctionReturnHot(0);
    }
    const double area = getMeasure();

    const auto row_size = data.getIndices().size();
    if (row_size == 0)
      MoFEMFunctionReturnHot(0);

    auto integrate = [&](auto col_indicies, auto &col_diff_base_functions) {
      MoFEMFunctionBegin;

      const auto col_size = col_indicies.size();

      locMat.resize(row_size, col_size, false);
      locMat.clear();
      int nb_gp = getGaussPts().size2();
      int nb_rows = data.getIndices().size();

      auto t_w = getFTensor0IntegrationWeight();
      auto t_coords = getFTensor1CoordsAtGaussPts();
      auto t_grad_h = getFTensor1FromMat<SPACE_DIM>(*gradHPtr);
      auto t_row_base = data.getFTensor0N();
      int nb_row_base_functions = data.getN().size2();

      auto s = wetting_angle_sub_stepping(getFEMethod()->ts_step);

      for (int gg = 0; gg != nb_gp; ++gg) {

        const double r = t_coords(0);
        const double alpha = t_w * area * cylindrical(r);
        const double h_grad_norm = sqrt(t_grad_h(i) * t_grad_h(i) +
                                        std::numeric_limits<double>::epsilon());
        const double one_over_h_grad_norm = 1. / h_grad_norm;
        const double beta = s * alpha * eta2 * one_over_h_grad_norm *
                            std::cos(M_PI * wettingAngle / 180);

        int rr = 0;
        for (; rr != nb_rows; ++rr) {
          const double delta = beta * t_row_base;

          auto ptr = &col_diff_base_functions(gg, 0);
          auto t_col_diff_base = getFTensor1FromPtr<SPACE_DIM>(ptr);

          for (int cc = 0; cc != col_size; ++cc) {
            locMat(rr, cc) += delta * t_col_diff_base(i) * t_grad_h(i);
            ++t_col_diff_base;
          }
          ++t_row_base;
        }

        for (; rr < nb_row_base_functions; ++rr) {
          ++t_row_base;
        }

        ++t_grad_h;
        ++t_w;
        ++t_coords;
      }

      MoFEMFunctionReturn(0);
    };

    for (auto c = 0; c != colIndicesPtr->size(); ++c) {

      auto &col_ind = (*colIndicesPtr)[c];
      if (col_ind.size()) {
        auto &diff_base = (*colDiffBaseFunctionsPtr)[c];

        CHKERR integrate(col_ind, diff_base);

        CHKERR MatSetValues(getKSPB(), data.getIndices().size(),
                            &*data.getIndices().begin(), col_ind.size(),
                            &*col_ind.begin(), &locMat(0, 0), ADD_VALUES);
      }
    }

    MoFEMFunctionReturn(0);
  }

private:
  MatrixDouble locMat;

  boost::shared_ptr<MatrixDouble> gradHPtr;
  boost::shared_ptr<Range> entsPtr;
  double wettingAngle;
  boost::shared_ptr<std::vector<VectorInt>> colIndicesPtr;
  boost::shared_ptr<std::vector<MatrixDouble>> colDiffBaseFunctionsPtr;
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
         boost::shared_ptr<VectorDouble> p_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        dotUPtr(dot_u_ptr), uPtr(u_ptr), gradUPtr(grad_u_ptr), hPtr(h_ptr),
        gradHPtr(grad_h_ptr), gPtr(g_ptr), pPtr(p_ptr) {}

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_dot_u = getFTensor1FromMat<U_FIELD_DIM>(*dotUPtr);
    auto t_u = getFTensor1FromMat<U_FIELD_DIM>(*uPtr);
    auto t_p = getFTensor0FromVec(*pPtr);
    auto t_grad_u = getFTensor2FromMat<U_FIELD_DIM, SPACE_DIM>(*gradUPtr);
    auto t_h = getFTensor0FromVec(*hPtr);
    auto t_grad_h = getFTensor1FromMat<SPACE_DIM>(*gradHPtr);
    auto t_g = getFTensor0FromVec(*gPtr);
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto t_base = data.getFTensor0N();
    auto t_diff_base = data.getFTensor1DiffN<SPACE_DIM>();

    auto t_w = getFTensor0IntegrationWeight();

    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<double>();
    FTensor::Tensor2_symmetric<double, SPACE_DIM> t_stress;
    FTensor::Tensor1<double, U_FIELD_DIM> t_phase_force;
    FTensor::Tensor1<double, U_FIELD_DIM> t_inertia_force;
    FTensor::Tensor1<double, U_FIELD_DIM> t_convection;
    FTensor::Tensor1<double, U_FIELD_DIM> t_buoyancy;
    FTensor::Tensor1<double, U_FIELD_DIM> t_gravity;
    FTensor::Tensor1<double, U_FIELD_DIM> t_forces;

    t_buoyancy(i) = 0;
    t_gravity(i) = 0;

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      const double rho = phase_function(t_h, rho_diff, rho_ave);
      const double mu = phase_function(t_h, mu_diff, mu_ave);

      auto t_D = get_D(2 * mu);

      t_inertia_force(i) = (rho * alpha) * (t_dot_u(i));
      // t_buoyancy(SPACE_DIM - 1) = -(alpha * rho * a0) * t_h;
      t_gravity(SPACE_DIM - 1) = (alpha * rho * a0);
      t_phase_force(i) = -alpha * kappa * t_g * t_grad_h(i);
      t_convection(i) = (rho * alpha) * (t_u(j) * t_grad_u(i, j));

      t_stress(i, j) =
          alpha * (t_D(i, j, k, l) * t_grad_u(k, l) + t_kd(i, j) * t_p);

      auto t_nf = getFTensor1FromArray<U_FIELD_DIM, U_FIELD_DIM>(locF);

      t_forces(i) = t_inertia_force(i) + t_buoyancy(i) + t_gravity(i) +
                    t_convection(i) + t_phase_force(i);

      int bb = 0;
      for (; bb != nbRows / U_FIELD_DIM; ++bb) {

        t_nf(i) += t_base * t_forces(i);
        t_nf(i) += t_diff_base(j) * t_stress(i, j);

        if (coord_type == CYLINDRICAL) {
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

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data) {
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
    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<double>();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);
      const double rho = phase_function(t_h, rho_diff, rho_ave);
      const double mu = phase_function(t_h, mu_diff, mu_ave);

      const double beta0 = alpha * rho;
      const double beta1 = beta0 * ts_a;
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

          t_mat(i, j) += (beta1 * bb) * t_kd(i, j);
          t_mat(i, j) += (beta0 * bb) * t_grad_u(i, j);
          t_mat(i, j) +=
              (beta0 * t_row_base) * t_kd(i, j) * (t_col_diff_base(k) * t_u(k));
          t_mat(i, j) += t_d_stress(i, j, k) * t_col_diff_base(k);

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

struct OpLoopSideGetDataForSideEle : ForcesAndSourcesCore::UserDataOperator {

  using UDO = ForcesAndSourcesCore::UserDataOperator;

  OpLoopSideGetDataForSideEle(
      const std::string field_name,
      boost::shared_ptr<std::vector<VectorInt>> col_indices_ptr,
      boost::shared_ptr<std::vector<MatrixDouble>> col_diff_basefunctions_ptr)
      : UDO(field_name, UDO::OPCOL), colIndicesPtr(col_indices_ptr),
        colDiffBaseFunctionsPtr(col_diff_basefunctions_ptr) {}

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    if (type == MBVERTEX) {
      colIndicesPtr->clear();
      colDiffBaseFunctionsPtr->clear();
    }

    colIndicesPtr->push_back(data.getIndices());
    colDiffBaseFunctionsPtr->push_back(data.getDiffN());

    MoFEMFunctionReturn(0);
  }

protected:
  boost::shared_ptr<std::vector<VectorInt>> colIndicesPtr;
  boost::shared_ptr<std::vector<MatrixDouble>> colDiffBaseFunctionsPtr;
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
            boost::shared_ptr<VectorDouble> g_ptr)
      : AssemblyDomainEleOp(field_name_u, field_name_h,
                            AssemblyDomainEleOp::OPROWCOL),
        dotUPtr(dot_u_ptr), uPtr(u_ptr), gradUPtr(grad_u_ptr), hPtr(h_ptr),
        gPtr(g_ptr) {
    sYmm = false;
    assembleTranspose = false;
  }

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_dot_u = getFTensor1FromMat<U_FIELD_DIM>(*dotUPtr);
    auto t_u = getFTensor1FromMat<U_FIELD_DIM>(*uPtr);
    auto t_grad_u = getFTensor2FromMat<U_FIELD_DIM, SPACE_DIM>(*gradUPtr);
    auto t_h = getFTensor0FromVec(*hPtr);
    auto t_g = getFTensor0FromVec(*gPtr);
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto t_row_base = row_data.getFTensor0N();
    auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();

    auto t_w = getFTensor0IntegrationWeight();

    FTensor::Tensor2_symmetric<double, SPACE_DIM> t_stress_dh;
    FTensor::Tensor1<double, U_FIELD_DIM> t_phase_force_dh;
    FTensor::Tensor1<double, U_FIELD_DIM> t_inertia_force_dh;
    FTensor::Tensor1<double, U_FIELD_DIM> t_convection_dh;
    FTensor::Tensor1<double, U_FIELD_DIM> t_buoyancy_dh;
    FTensor::Tensor1<double, U_FIELD_DIM> t_forces_dh;

    t_buoyancy_dh(i) = 0;

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      const double rho = phase_function(t_h, rho_diff, rho_ave);
      const double rho_dh = d_phase_function_h(t_h, rho_diff);
      const double mu_dh = d_phase_function_h(t_h, mu_diff);

      auto t_D_dh = get_D(2 * mu_dh);

      t_inertia_force_dh(i) = (alpha * rho_dh) * t_dot_u(i);
      // t_buoyancy_dh(SPACE_DIM - 1) = -(alpha * a0) * (rho + rho_dh * t_h);
      t_convection_dh(i) = (rho_dh * alpha) * (t_u(j) * t_grad_u(i, j));
      const double t_phase_force_g_dh = -alpha * kappa * t_g;
      t_forces_dh(i) =
          t_inertia_force_dh(i) + t_buoyancy_dh(i) + t_convection_dh(i);

      t_stress_dh(i, j) = alpha * (t_D_dh(i, j, k, l) * t_grad_u(k, l));

      int rr = 0;
      for (; rr != nbRows / U_FIELD_DIM; ++rr) {

        auto t_mat =
            getFTensor1FromMat<U_FIELD_DIM, 1>(locMat, rr * U_FIELD_DIM);
        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

        for (int cc = 0; cc != nbCols; ++cc) {

          const double bb = t_row_base * t_col_base;
          t_mat(i) += t_forces_dh(i) * bb;
          t_mat(i) += (t_phase_force_g_dh * t_row_base) * t_col_diff_base(i);
          t_mat(i) += (t_row_diff_base(j) * t_col_base) * t_stress_dh(i, j);

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
};

/**
 * @brief Lhs for G dH
 *
 */
struct OpLhsU_dG : public AssemblyDomainEleOp {

  OpLhsU_dG(const std::string field_name_u, const std::string field_name_h,
            boost::shared_ptr<MatrixDouble> grad_h_ptr)
      : AssemblyDomainEleOp(field_name_u, field_name_h,
                            AssemblyDomainEleOp::OPROWCOL),
        gradHPtr(grad_h_ptr) {
    sYmm = false;
    assembleTranspose = false;
  }

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_grad_h = getFTensor1FromMat<SPACE_DIM>(*gradHPtr);
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto t_row_base = row_data.getFTensor0N();
    auto t_w = getFTensor0IntegrationWeight();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      FTensor::Tensor1<double, SPACE_DIM> t_phase_force_dg;
      t_phase_force_dg(i) = -alpha * kappa * t_grad_h(i);

      int rr = 0;
      for (; rr != nbRows / U_FIELD_DIM; ++rr) {
        auto t_mat =
            getFTensor1FromMat<U_FIELD_DIM, 1>(locMat, rr * U_FIELD_DIM);
        auto t_col_base = col_data.getFTensor0N(gg, 0);

        for (int cc = 0; cc != nbCols; ++cc) {
          const double bb = t_row_base * t_col_base;
          t_mat(i) += t_phase_force_dg(i) * bb;

          ++t_mat;
          ++t_col_base;
        }

        ++t_row_base;
      }

      for (; rr < nbRowBaseFunctions; ++rr) {
        ++t_row_base;
      }

      ++t_grad_h;
      ++t_coords;
      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> gradHPtr;
};

template <bool I> struct OpRhsH : public AssemblyDomainEleOp {

  OpRhsH(const std::string field_name, boost::shared_ptr<MatrixDouble> u_ptr,
         boost::shared_ptr<VectorDouble> dot_h_ptr,
         boost::shared_ptr<VectorDouble> h_ptr,
         boost::shared_ptr<MatrixDouble> grad_h_ptr,
         boost::shared_ptr<MatrixDouble> grad_g_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        uPtr(u_ptr), dotHPtr(dot_h_ptr), hPtr(h_ptr), gradHPtr(grad_h_ptr),
        gradGPtr(grad_g_ptr) {}

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_coords = getFTensor1CoordsAtGaussPts();
    auto t_base = data.getFTensor0N();
    auto t_diff_base = data.getFTensor1DiffN<SPACE_DIM>();

#ifndef NDEBUG
    if (data.getDiffN().size1() != data.getN().size1())
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "wrong size 1");
    if (data.getDiffN().size2() != data.getN().size2() * SPACE_DIM) {
      MOFEM_LOG("SELF", Sev::error)
          << "Side " << rowSide << " " << CN::EntityTypeName(rowType);
      MOFEM_LOG("SELF", Sev::error) << data.getN();
      MOFEM_LOG("SELF", Sev::error) << data.getDiffN();
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "wrong size 2");
    }
#endif

    if constexpr (I) {

      auto t_h = getFTensor0FromVec(*hPtr);
      auto t_grad_g = getFTensor1FromMat<SPACE_DIM>(*gradGPtr);

      for (int gg = 0; gg != nbIntegrationPts; ++gg) {

        const double r = t_coords(0);
        const double alpha = t_w * vol * cylindrical(r);

        const double set_h = init_h(t_coords(0), t_coords(1), t_coords(2));
        const double m = get_M(set_h) * alpha;

        int bb = 0;
        for (; bb != nbRows; ++bb) {
          locF[bb] += (t_base * alpha) * (t_h - set_h);
          locF[bb] += (t_diff_base(i) * m) * t_grad_g(i);
          ++t_base;
          ++t_diff_base;
        }

        for (; bb < nbRowBaseFunctions; ++bb) {
          ++t_base;
          ++t_diff_base;
        }

        ++t_h;
        ++t_grad_g;

        ++t_coords;
        ++t_w;
      }

    } else {

      auto t_dot_h = getFTensor0FromVec(*dotHPtr);
      auto t_h = getFTensor0FromVec(*hPtr);
      auto t_u = getFTensor1FromMat<U_FIELD_DIM>(*uPtr);
      auto t_grad_h = getFTensor1FromMat<SPACE_DIM>(*gradHPtr);
      auto t_grad_g = getFTensor1FromMat<SPACE_DIM>(*gradGPtr);

      for (int gg = 0; gg != nbIntegrationPts; ++gg) {

        const double r = t_coords(0);
        const double alpha = t_w * vol * cylindrical(r);

        const double m = get_M(t_h) * alpha;

        int bb = 0;
        for (; bb != nbRows; ++bb) {
          locF[bb] += (t_base * alpha) * (t_dot_h);
          locF[bb] += (t_base * alpha) * (t_grad_h(i) * t_u(i));
          locF[bb] += (t_diff_base(i) * t_grad_g(i)) * m;
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
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_grad_h = getFTensor1FromMat<SPACE_DIM>(*gradHPtr);
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto t_row_base = row_data.getFTensor0N();
    auto t_w = getFTensor0IntegrationWeight();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const auto r = t_coords(0);
      const auto alpha = t_w * vol * cylindrical(r);
      auto t_mat = getFTensor1FromPtr<U_FIELD_DIM>(&locMat(0, 0));
      FTensor::Tensor1<double, SPACE_DIM> t_row;

      int rr = 0;
      for (; rr != nbRows; ++rr) {
        t_row(i) = (alpha * t_row_base) * t_grad_h(i);
        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (int cc = 0; cc != nbCols / U_FIELD_DIM; ++cc) {
          t_mat(i) += t_row(i) * t_col_base;
          ++t_mat;
          ++t_col_base;
        }
        ++t_row_base;
      }

      for (; rr < nbRowBaseFunctions; ++rr)
        ++t_row_base;

      ++t_grad_h;
      ++t_w;
      ++t_coords;
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
template <bool I> struct OpLhsH_dH : public AssemblyDomainEleOp {

  OpLhsH_dH(const std::string field_name, boost::shared_ptr<MatrixDouble> u_ptr,
            boost::shared_ptr<VectorDouble> h_ptr,
            boost::shared_ptr<MatrixDouble> grad_g_ptr)
      : AssemblyDomainEleOp(field_name, field_name,
                            AssemblyDomainEleOp::OPROWCOL),
        uPtr(u_ptr), hPtr(h_ptr), gradGPtr(grad_g_ptr) {
    sYmm = false;
  }

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_coords = getFTensor1CoordsAtGaussPts();
    auto t_row_base = row_data.getFTensor0N();
    auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();

#ifndef NDEBUG
    if (row_data.getDiffN().size1() != row_data.getN().size1())
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "wrong size 1");
    if (row_data.getDiffN().size2() != row_data.getN().size2() * SPACE_DIM) {
      MOFEM_LOG("SELF", Sev::error)
          << "Side " << rowSide << " " << CN::EntityTypeName(rowType);
      MOFEM_LOG("SELF", Sev::error) << row_data.getN();
      MOFEM_LOG("SELF", Sev::error) << row_data.getDiffN();
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "wrong size 2");
    }

    if (col_data.getDiffN().size1() != col_data.getN().size1())
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "wrong size 1");
    if (col_data.getDiffN().size2() != col_data.getN().size2() * SPACE_DIM) {
      MOFEM_LOG("SELF", Sev::error)
          << "Side " << rowSide << " " << CN::EntityTypeName(rowType);
      MOFEM_LOG("SELF", Sev::error) << col_data.getN();
      MOFEM_LOG("SELF", Sev::error) << col_data.getDiffN();
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "wrong size 2");
    }
#endif

    if constexpr (I) {

      auto t_h = getFTensor0FromVec(*hPtr);
      auto t_grad_g = getFTensor1FromMat<SPACE_DIM>(*gradGPtr);

      for (int gg = 0; gg != nbIntegrationPts; gg++) {

        const double r = t_coords(0);
        const double alpha = t_w * vol * cylindrical(r);

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

    } else {

      auto t_h = getFTensor0FromVec(*hPtr);
      auto t_grad_g = getFTensor1FromMat<SPACE_DIM>(*gradGPtr);
      auto t_u = getFTensor1FromMat<U_FIELD_DIM>(*uPtr);

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
                (t_row_base * alpha) * (t_col_diff_base(i) * t_u(i));
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
template <bool I> struct OpLhsH_dG : public AssemblyDomainEleOp {

  OpLhsH_dG(const std::string field_name_h, const std::string field_name_g,
            boost::shared_ptr<VectorDouble> h_ptr)
      : AssemblyDomainEleOp(field_name_h, field_name_g,
                            AssemblyDomainEleOp::OPROWCOL),
        hPtr(h_ptr) {
    sYmm = false;
    assembleTranspose = false;
  }

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_h = getFTensor0FromVec(*hPtr);

    auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_coords = getFTensor1CoordsAtGaussPts();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double r = t_coords(0);
      const double alpha = t_w * vol * cylindrical(r);

      double set_h;
      if constexpr (I)
        set_h = init_h(t_coords(0), t_coords(1), t_coords(2));
      else
        set_h = t_h;

      auto m = get_M(set_h) * alpha;

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

template <bool I> struct OpRhsG : public AssemblyDomainEleOp {

  OpRhsG(const std::string field_name, boost::shared_ptr<VectorDouble> h_ptr,
         boost::shared_ptr<MatrixDouble> grad_h_ptr,
         boost::shared_ptr<VectorDouble> g_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        hPtr(h_ptr), gradHPtr(grad_h_ptr), gPtr(g_ptr) {}

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data) {
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

      double set_h;
      if constexpr (I)
        set_h = init_h(t_coords(0), t_coords(1), t_coords(2));
      else
        set_h = t_h;

      const double f = get_f(set_h);

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
template <bool I> struct OpLhsG_dH : public AssemblyDomainEleOp {

  OpLhsG_dH(const std::string field_name_g, const std::string field_name_h,
            boost::shared_ptr<VectorDouble> h_ptr)
      : AssemblyDomainEleOp(field_name_g, field_name_h,
                            AssemblyDomainEleOp::OPROWCOL),
        hPtr(h_ptr) {
    sYmm = false;
    assembleTranspose = false;
  }

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data) {
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

          if constexpr (I == false)
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
    sYmm = true;
  }

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data) {
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

} // namespace FreeSurfaceOps