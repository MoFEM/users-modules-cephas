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

namespace OpPlasticTools {

//! [Common data]
struct CommonData : public OpElasticTools::CommonData {
  boost::shared_ptr<VectorDouble> plasticSurfacePtr;
  boost::shared_ptr<MatrixDouble> plasticFlowPtr;
  boost::shared_ptr<VectorDouble> plasticTauPtr;
  boost::shared_ptr<MatrixDouble> plasticStrainPtr;
  boost::shared_ptr<MatrixDouble> plasticStrainDotPtr;
};
//! [Common data]

//! [Lambda functions]
auto diff_stress = []() {
  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  FTensor::Ddg<double, 2, 2> t_diff_stress;
  t_diff_stress(i, j, k, l) = 0;
  for (size_t ii = 0; ii != 2; ++ii) {
    t_diff_stress(ii, ii, ii, ii) += 0.5;
    for (size_t jj = ii; jj != 2; ++jj) {
      t_diff_stress(ii, jj, ii, jj) += 0.5;
    }
  }
  return t_diff_stress;
};

auto trace = [](auto &t_stress) {
  constexpr double third = boost::math::constants::third<double>();
  return (t_stress(0, 0) + t_stress(1, 1)) * third;
};

auto deviator = [](auto &t_stress, auto &&trace) {
  FTensor::Index<'I', 3> I;
  FTensor::Index<'J', 3> J;
  FTensor::Tensor2_symmetric<double, 3> t_dev;
  t_dev(I, J) = 0;
  for (int ii = 0; ii != 2; ++ii)
    for (int jj = ii; jj != 2; ++jj)
      t_dev(ii, jj) = t_stress(ii, jj);
  t_dev(0, 0) -= trace;
  t_dev(1, 1) -= trace;
  t_dev(2, 2) -= trace;
  return t_dev;
};

auto diff_dev_stress = [](auto &&t_diff_stress) {
  FTensor::Index<'I', 3> I;
  FTensor::Index<'J', 3> J;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  FTensor::Ddg<double, 3, 2> t_diff_dev_stress;

  t_diff_dev_stress(I, J, k, l) = 0;
  for (int ii = 0; ii != 2; ++ii)
    for (int jj = ii; jj != 2; ++jj)
      for (int kk = 0; kk != 2; ++kk)
        for (int ll = kk; ll != 2; ++ll)
          t_diff_dev_stress(ii, jj, kk, ll) = t_diff_stress(ii, jj, kk, ll);

  constexpr double third = boost::math::constants::third<double>();

  t_diff_dev_stress(0, 0, 0, 0) -= third;
  t_diff_dev_stress(0, 0, 1, 1) -= third;

  t_diff_dev_stress(1, 1, 0, 0) -= third;
  t_diff_dev_stress(1, 1, 1, 1) -= third;

  t_diff_dev_stress(2, 2, 0, 0) -= third;
  t_diff_dev_stress(2, 2, 1, 1) -= third;

  return t_diff_dev_stress;
};

auto platsic_surface = [](auto &&t_dev) {
  FTensor::Index<'I', 3> I;
  FTensor::Index<'I', 3> J;
  return t_dev(I, J) * t_dev(I, J);
};

auto plastic_flow = [](auto &f, auto &&t_dev_stress, auto &&t_diff_dev_stress) {
  FTensor::Index<'I', 3> I;
  FTensor::Index<'J', 3> J;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  FTensor::Tensor2_symmetric<double, 2> t_diff_f;
  t_diff_f(k, l) = 2. * (t_dev_stress(I, J) * t_diff_dev_stress(I, J, k, l));
  return t_diff_f;
};

auto diff_plastic_flow_dstress = [](auto &f, auto &t_flow,
                                    auto &&t_diff_dev_stress) {
  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;
  FTensor::Index<'M', 3> M;
  FTensor::Index<'N', 3> N;

  FTensor::Ddg<double, 2, 2> t_diff_flow;
  t_diff_flow(i, j, k, l) = 2. * (t_diff_dev_stress(M, N, i, j) * t_diff_dev_stress(M, N, k, l));

  return t_diff_flow;
};

auto diff_plastic_flow_dstrain = [](auto &t_D,
                                    auto &&t_diff_plastic_flow_dstress) {
  FTensor::Index<'i', 2> j;
  FTensor::Index<'j', 2> i;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;
  FTensor::Index<'m', 2> m;
  FTensor::Index<'m', 2> n;

  FTensor::Ddg<double, 2, 2> t_diff_flow;
  t_diff_flow(i, j, k, l) =
      t_diff_plastic_flow_dstress(i, j, m, n) * t_D(m, n, k, l);

  return t_diff_flow;
};

auto contrains = [](double tau, double f) {
  return tau;
  // if ((tau + f - sigmaY2) >= 0)
  //   return -f;
  // else
  //   return tau;
};

auto sign = [](auto x) {
  if (x >= 0)
    return 1;
  else
    return -1;
};

auto constrain_dtau = [](auto tau, auto f) {
  return 1;
  // return (1 - sign(tau + f - sigmaY2)) / 2.;
};

auto diff_constrain_df = [](auto tau, auto f) {
  return 0;
  // return (-1 - sign(tau + f - sigmaY2)) / 2.;
};

auto diff_constrain_dstress = [](auto &&diff_constrain_df,
                                 auto &&t_plastic_flow) {
  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Tensor2_symmetric<double, 2> t_diff_constrain_dstress;
  t_diff_constrain_dstress(i, j) = diff_constrain_df * t_plastic_flow(i, j);
  return t_diff_constrain_dstress;
};

auto diff_constrain_du = [](auto &t_D, auto &&t_diff_constrain_dstress) {
  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;
  FTensor::Tensor2_symmetric<double, 2> t_diff_constrain_du;
  t_diff_constrain_du(k, l) = t_diff_constrain_dstress(i, j) * t_D(i, j, k, l);
  return t_diff_constrain_du;
};
//! [Lambda functions]

//! [Operators definitions]
struct OpCalculatePlasticSurface : public DomianEleOp {
  OpCalculatePlasticSurface(const std::string field_name,
                            boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpPlasticStress : public DomianEleOp {
  OpPlasticStress(const std::string field_name,
                  boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculatePlasticFlowRhs : public DomianEleOp {
  OpCalculatePlasticFlowRhs(const std::string field_name,
                            boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculateContrainsRhs : public DomianEleOp {
  OpCalculateContrainsRhs(const std::string field_name,
                          boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculatePlasticInternalForceLhs_dEP : public DomianEleOp {
  OpCalculatePlasticInternalForceLhs_dEP(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpCalculatePlasticFlowLhs_dU : public DomianEleOp {
  OpCalculatePlasticFlowLhs_dU(const std::string row_field_name,
                               const std::string col_field_name,
                               boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpCalculatePlasticFlowLhs_dEP : public DomianEleOp {
  OpCalculatePlasticFlowLhs_dEP(const std::string row_field_name,
                                const std::string col_field_name,
                                boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpCalculatePlasticFlowLhs_dTAU : public DomianEleOp {
  OpCalculatePlasticFlowLhs_dTAU(const std::string row_field_name,
                                 const std::string col_field_name,
                                 boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpCalculateContrainsLhs_dU : public DomianEleOp {
  OpCalculateContrainsLhs_dU(const std::string row_field_name,
                             const std::string col_field_name,
                             boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpCalculateContrainsLhs_dEP : public DomianEleOp {
  OpCalculateContrainsLhs_dEP(const std::string row_field_name,
                              const std::string col_field_name,
                              boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpCalculateContrainsLhs_dTAU : public DomianEleOp {
  OpCalculateContrainsLhs_dTAU(const std::string row_field_name,
                               const std::string col_field_name,
                               boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpPostProcPlastic : public DomianEleOp {
  OpPostProcPlastic(const std::string field_name,
                    moab::Interface &post_proc_mesh,
                    std::vector<EntityHandle> &map_gauss_pts,
                    boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;
  boost::shared_ptr<CommonData> commonDataPtr;
};
//! [Operators definitions]

OpCalculatePlasticSurface::OpCalculatePlasticSurface(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

MoFEMErrorCode
OpCalculatePlasticSurface::doWork(int side, EntityType type,
                                  DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  const size_t nb_gauss_pts = commonDataPtr->mStressPtr->size2();
  auto t_stress = getFTensor2SymmetricFromMat<2>(*(commonDataPtr->mStressPtr));
  auto t_strain = getFTensor2SymmetricFromMat<2>(*(commonDataPtr->mStrainPtr)); 

  commonDataPtr->plasticSurfacePtr->resize(nb_gauss_pts, false);
  commonDataPtr->plasticFlowPtr->resize(3, nb_gauss_pts, false);
  auto t_flow =
      getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticFlowPtr));

  for (auto &f : *(commonDataPtr->plasticSurfacePtr)) {
    f = platsic_surface(deviator(t_stress, trace(t_stress)));
    auto t_flow_tmp = plastic_flow(f,

                                   deviator(t_stress, trace(t_stress)),

                                   diff_dev_stress(diff_stress()));
    t_flow(i, j) = t_flow_tmp(i, j);
    ++t_flow;
    ++t_stress;
    ++t_strain;
  }

  MoFEMFunctionReturn(0);
}

OpPlasticStress::OpPlasticStress(const std::string field_name,
                                 boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Calculate stress]
MoFEMErrorCode
OpPlasticStress::doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = commonDataPtr->mStrainPtr->size2();
  commonDataPtr->mStressPtr->resize(3, nb_gauss_pts);
  auto &t_D = commonDataPtr->tD;
  auto t_strain = getFTensor2SymmetricFromMat<2>(*(commonDataPtr->mStrainPtr));
  auto t_platic_strain =
      getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticStrainPtr));
  auto t_stress = getFTensor2SymmetricFromMat<2>(*(commonDataPtr->mStressPtr));

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
    t_stress(i, j) = t_D(i, j, k, l) * (t_strain(k, l) - t_platic_strain(k, l));
    ++t_strain;
    ++t_platic_strain;
    ++t_stress;
  }

  MoFEMFunctionReturn(0);
}
//! [Calculate stress]

OpCalculatePlasticFlowRhs::OpCalculatePlasticFlowRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode
OpCalculatePlasticFlowRhs::doWork(int side, EntityType type,
                                  DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    auto t_flow =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticFlowPtr));
    auto t_plastic_strain_dot =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticStrainDotPtr));
    auto t_tau = getFTensor0FromVec(*(commonDataPtr->plasticTauPtr));
    auto &t_D = commonDataPtr->tD;

    const size_t nb_integration_pts = data.getN().size1();
    const size_t nb_base_functions = data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_base = data.getFTensor0N();

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      FTensor::Tensor2_symmetric<FTensor::PackPtr<double *, 3>, 2> t_nf{
          &nf[0], &nf[1], &nf[2]};
      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        t_nf(i, j) += alpha * t_base *
                      (t_D(i, j, k, l) * t_plastic_strain_dot(k, l) -
                       t_tau * t_flow(i, j));

        ++t_base;
        ++t_nf;
      }
      for (; bb < nb_base_functions; ++bb)
        ++t_base;

      ++t_flow;
      ++t_plastic_strain_dot;
      ++t_tau;
      ++t_w;
    }

    CHKERR VecSetValues(getTSf(), data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculateContrainsRhs::OpCalculateContrainsRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode
OpCalculateContrainsRhs::doWork(int side, EntityType type,
                                DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    auto t_tau = getFTensor0FromVec(*(commonDataPtr->plasticTauPtr));
    auto t_f = getFTensor0FromVec(*(commonDataPtr->plasticSurfacePtr));
    auto t_w = getFTensor0IntegrationWeight();

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    auto t_base = data.getFTensor0N();
    const size_t nb_integration_pts = data.getN().size1();
    const size_t nb_base_functions = data.getN().size2();
    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      size_t bb = 0;
      for (; bb != nb_dofs; ++bb) {
        nf[bb] += alpha * t_base * contrains(t_tau, t_f);
        ++t_base;
      }
      for (; bb != nb_base_functions; ++bb)
        ++t_base;

      ++t_tau;
      ++t_f;
      ++t_w;
    }

    CHKERR VecSetValues(getTSf(), data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticInternalForceLhs_dEP::OpCalculatePlasticInternalForceLhs_dEP(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpCalculatePlasticInternalForceLhs_dEP::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_functions = row_data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_diff_base = row_data.getFTensor1DiffN<2>();
    auto &t_D = commonDataPtr->tD;

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      size_t rr = 0;
      for (; rr != nb_row_dofs / 2; ++rr) {

        FTensor::Christof<FTensor::PackPtr<double *, 3>, 2, 2> t_mat{

            &locMat(2 * rr + 0, 0), &locMat(2 * rr + 0, 1),
            &locMat(2 * rr + 0, 2),

            &locMat(2 * rr + 1, 0), &locMat(2 * rr + 1, 1),
            &locMat(2 * rr + 1, 2)

        };

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs / 3; ++cc) {

          // I mix up the indices here so that it behaves like a
          // Dg.  That way I don't have to have a separate wrapper
          // class Christof_Expr, which simplifies things.
          // You cyclicly has to shift index, i, j, k -> l, i, j
          FTensor::Christof<double, 2, 2> t_tmp;
          t_tmp(l, i, k) =
              (t_D(i, j, k, l) * ((alpha * t_col_base) * t_row_diff_base(j)));
          for (int ii = 0; ii != 2; ++ii)
            for (int kk = 0; kk != 2; ++kk)
              for (int ll = 0; ll != 2; ++ll)
                t_mat(ii, kk, ll) -= t_tmp(ii, kk, ll);

          ++t_mat;
          ++t_col_base;
        }

        ++t_row_diff_base;
      }

      for (; rr < nb_row_base_functions; ++rr)
        ++t_row_diff_base;

      ++t_w;
    }

    MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                 ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticFlowLhs_dU::OpCalculatePlasticFlowLhs_dU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpCalculatePlasticFlowLhs_dU::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_functions = row_data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    auto t_f = getFTensor0FromVec(*(commonDataPtr->plasticSurfacePtr));
    auto t_tau = getFTensor0FromVec(*(commonDataPtr->plasticTauPtr));
    auto t_flow =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticFlowPtr));
    auto &t_D = commonDataPtr->tD;


    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;
      auto t_diff_plastic_flow = diff_plastic_flow_dstrain(
          t_D, diff_plastic_flow_dstress(t_f, t_flow,
                                         diff_dev_stress(diff_stress())));

      size_t rr = 0;
      for (; rr != nb_row_dofs / 3; ++rr) {

        FTensor::Dg<FTensor::PackPtr<double *, 2>, 2, 2> t_mat{
            &locMat(3 * rr + 0, 0),
            &locMat(3 * rr + 0, 1),

            &locMat(3 * rr + 1, 0),
            &locMat(3 * rr + 1, 1),

            &locMat(3 * rr + 2, 0),
            &locMat(3 * rr + 2, 1)

        };

        const double c0 = alpha * t_row_base * t_tau;

        auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs / 2; ++cc) {

          t_mat(i, j, l) -=
              c0 * t_diff_plastic_flow(i, j, l, k) * t_col_diff_base(k);

          ++t_mat;
          ++t_col_diff_base;
        }

        ++t_row_base;
      }
      for (; rr < nb_row_base_functions; ++rr)
        ++t_row_base;

      ++t_w;
      ++t_flow;
      ++t_f;
      ++t_tau;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticFlowLhs_dEP::OpCalculatePlasticFlowLhs_dEP(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
        sYmm = false;
      }

MoFEMErrorCode OpCalculatePlasticFlowLhs_dEP::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_functions = row_data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    auto t_f = getFTensor0FromVec(*(commonDataPtr->plasticSurfacePtr));
    auto t_tau = getFTensor0FromVec(*(commonDataPtr->plasticTauPtr));
    auto t_flow =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticFlowPtr));

    auto &t_D = commonDataPtr->tD;

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;
      double beta = alpha * getTSa();

      size_t rr = 0;
      for (; rr != nb_row_dofs / 3; ++rr) {

        const double c0 = alpha * t_row_base * t_tau;
        const double c1 = beta * t_row_base;

        auto t_diff_plastic_flow = diff_plastic_flow_dstrain(
            t_D, diff_plastic_flow_dstress(t_f, t_flow,
                                           diff_dev_stress(diff_stress())));

        FTensor::Ddg<FTensor::PackPtr<double *, 3>, 2, 2> t_mat{

            &locMat(3 * rr + 0, 0), &locMat(3 * rr + 0, 1),
            &locMat(3 * rr + 0, 2),

            &locMat(3 * rr + 1, 0), &locMat(3 * rr + 1, 1),
            &locMat(3 * rr + 1, 2),

            &locMat(3 * rr + 2, 0), &locMat(3 * rr + 2, 1),
            &locMat(3 * rr + 2, 2)

        };

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs / 3; ++cc) {

          FTensor::Ddg<double, 2, 2> t_tmp;
          t_tmp(i, j, k, l) =
              t_col_base *
              (c1 * t_D(i, j, k, l) + c0 * t_diff_plastic_flow(i, j, k, l));

          for (int ii = 0; ii != 2; ++ii)
            for (int jj = ii; jj != 2; ++jj)
              for (int kk = 0; kk != 2; ++kk)
                for (int ll = 0; ll != 2; ++ll)
                  t_mat(ii, jj, kk, ll) += t_tmp(ii, jj, kk, ll);

          ++t_mat;
          ++t_col_base;
        }

        ++t_row_base;
      }
      for (; rr < nb_row_base_functions; ++rr)
        ++t_row_base;

      ++t_w;
      ++t_f;
      ++t_tau;
      ++t_flow;
    }

    CHKERR MatSetValues(getTSB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticFlowLhs_dTAU::OpCalculatePlasticFlowLhs_dTAU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpCalculatePlasticFlowLhs_dTAU::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_flow =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticFlowPtr));

    auto t_row_base = row_data.getFTensor0N();

    auto &t_D = commonDataPtr->tD;

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      for (size_t rr = 0; rr != nb_row_dofs / 3; ++rr) {

        FTensor::Tensor2_symmetric<FTensor::PackPtr<double *, 1>, 2> t_mat{
            &locMat(3 * rr + 0, 0), &locMat(3 * rr + 1, 0),
            &locMat(3 * rr + 2, 0)};

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs; cc++) {

          t_mat(i, j) -= alpha * t_row_base * t_col_base * t_flow(i, j);
          ++t_mat;
          ++t_col_base;
        }

        ++t_row_base;
      }

      ++t_w;
      ++t_flow;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculateContrainsLhs_dU::OpCalculateContrainsLhs_dU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpCalculateContrainsLhs_dU::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_functions = row_data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    auto t_f = getFTensor0FromVec(*(commonDataPtr->plasticSurfacePtr));
    auto t_tau = getFTensor0FromVec(*(commonDataPtr->plasticTauPtr));
    auto t_flow =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticFlowPtr));
    auto &t_D = commonDataPtr->tD;

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      auto mat_ptr = locMat.data().begin();
      auto t_diff_constrains_du = diff_constrain_du(
          t_D, diff_constrain_dstress(diff_constrain_df(t_tau, t_f), t_flow));

      FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_mat{&locMat(0, 0),
                                                               &locMat(0, 1)};

      size_t rr = 0;
      for (; rr != nb_row_dofs / 3; ++rr) {

        auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs / 2; cc++) {

          t_mat(i) += alpha * t_row_base * t_diff_constrains_du(i, j) *
                      t_col_diff_base(j);

          ++t_col_diff_base;
        }

        ++t_row_base;
        ++t_mat;
      }
      for (; rr != nb_row_base_functions; ++rr)
        ++t_row_base;

      ++t_f;
      ++t_tau;
      ++t_row_base;
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculateContrainsLhs_dEP::OpCalculateContrainsLhs_dEP(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpCalculateContrainsLhs_dEP::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_functions = row_data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    auto t_f = getFTensor0FromVec(*(commonDataPtr->plasticSurfacePtr));
    auto t_tau = getFTensor0FromVec(*(commonDataPtr->plasticTauPtr));
    auto t_flow =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticFlowPtr));
    auto &t_D = commonDataPtr->tD;

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      auto mat_ptr = locMat.data().begin();
      auto t_diff_constrains_du = diff_constrain_du(
          t_D, diff_constrain_dstress(diff_constrain_df(t_tau, t_f), t_flow));

      FTensor::Tensor2_symmetric<FTensor::PackPtr<double *, 3>, 2> t_mat{
          &locMat(0, 0), &locMat(0, 1), &locMat(0, 2)};

      size_t rr = 0;
      for (; rr != nb_row_dofs / 3; ++rr) {

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs / 3; cc++) {

          t_mat(i, j) -=
              alpha * t_row_base * t_col_base * t_diff_constrains_du(i, j);

          ++t_col_base;
        }

        ++t_row_base;
        ++t_mat;
      }
      for (; rr != nb_row_base_functions; ++rr)
        ++t_row_base;

      ++t_f;
      ++t_tau;
      ++t_row_base;
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculateContrainsLhs_dTAU::OpCalculateContrainsLhs_dTAU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpCalculateContrainsLhs_dTAU::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_functions = row_data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    auto t_f = getFTensor0FromVec(*(commonDataPtr->plasticSurfacePtr));
    auto t_tau = getFTensor0FromVec(*(commonDataPtr->plasticTauPtr));

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      auto mat_ptr = locMat.data().begin();

      size_t rr = 0;
      for (; rr != nb_row_dofs; ++rr) {

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs; cc++) {

          *mat_ptr +=
              alpha * t_row_base * t_col_base * constrain_dtau(t_tau, t_f);

          ++mat_ptr;
          ++t_col_base;
        }
        ++t_row_base;
      }
      for (; rr != nb_row_base_functions; ++rr)
        ++t_row_base;

      ++t_f;
      ++t_tau;
      ++t_row_base;
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpPostProcPlastic::OpPostProcPlastic(
    const std::string field_name, moab::Interface &post_proc_mesh,
    std::vector<EntityHandle> &map_gauss_pts,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(field_name, DomianEleOp::OPROW), postProcMesh(post_proc_mesh),
      mapGaussPts(map_gauss_pts), commonDataPtr(common_data_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Postprocessing]
MoFEMErrorCode
OpPostProcPlastic::doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  std::array<double, 9> def;
  std::fill(def.begin(), def.end(), 0);

  auto get_tag = [&](const std::string name, size_t size) {
    Tag th;
    CHKERR postProcMesh.tag_get_handle(name.c_str(), size, MB_TYPE_DOUBLE, th,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       def.data());
    return th;
  };

  MatrixDouble3by3 mat(3, 3);

  auto set_matrix_2d = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != 2; ++r)
      for (size_t c = 0; c != 2; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_matrix_3d = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != 3; ++r)
      for (size_t c = 0; c != 3; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_matrix_2d_symm = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != 2; ++r)
      for (size_t c = 0; c != 2; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_scalar = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    mat(0, 0) = t;
    return mat;
  };

  auto set_tag = [&](auto th, auto gg, MatrixDouble3by3 &mat) {
    return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
                                     &*mat.data().begin());
  };

  auto th_plastic_surface = get_tag("PLASTIC_SURFACE", 1);
  auto th_plastic_flow = get_tag("PLASTIC_FLOW", 9);

  auto t_flow = getFTensor2FromMat<2, 2>(*(commonDataPtr->plasticFlowPtr));
  size_t gg = 0;
  for (auto &f : *(commonDataPtr->plasticSurfacePtr)) {
    CHKERR set_tag(th_plastic_surface, gg, set_scalar(f));
    CHKERR set_tag(th_plastic_flow, gg, set_matrix_3d(t_flow));
    ++gg;
    ++t_flow;
  }

  MoFEMFunctionReturn(0);
}
//! [Postprocessing]
}; // namespace OpPlasticTools