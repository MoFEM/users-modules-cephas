/* \file SimpleContact.cpp
  \brief Implementation of simple contact element
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

#include <MoFEM.hpp>
using namespace MoFEM;
#include <SimpleContact.hpp>
using namespace boost::numeric;

extern "C" {
#include <triangle_ncc_rule.h>
}

constexpr double SimpleContactProblem::TOL;

MoFEMErrorCode
SimpleContactProblem::SimpleContactElement::setGaussPts(int order) {
  MoFEMFunctionBegin;
  if (newtonCotes) {
    int rule = order + 2;
    int nb_gauss_pts = triangle_ncc_order_num(rule);
    gaussPtsMaster.resize(3, nb_gauss_pts, false);
    gaussPtsSlave.resize(3, nb_gauss_pts, false);
    double xy_coords[2 * nb_gauss_pts];
    double w_array[nb_gauss_pts];
    triangle_ncc_rule(rule, nb_gauss_pts, xy_coords, w_array);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      gaussPtsMaster(0, gg) = xy_coords[gg * 2];
      gaussPtsMaster(1, gg) = xy_coords[gg * 2 + 1];
      gaussPtsMaster(2, gg) = w_array[gg];
      gaussPtsSlave(0, gg) = xy_coords[gg * 2];
      gaussPtsSlave(1, gg) = xy_coords[gg * 2 + 1];
      gaussPtsSlave(2, gg) = w_array[gg];
    }
  } else {
    CHKERR ContactEle::setDefaultGaussPts(2);
  }
  MoFEMFunctionReturn(0);
}

template <bool CONVECT_MASTER>
MoFEMErrorCode
SimpleContactProblem::ConvectSlaveIntegrationPts::convectSlaveIntegrationPts() {
  MoFEMFunctionBegin;

  auto get_material_dofs_from_coords = [&]() {
    MoFEMFunctionBegin;
    materialCoords.resize(18, false);
    int num_nodes;
    const EntityHandle *conn;
    CHKERR fePtr->mField.get_moab().get_connectivity(fePtr->getFEEntityHandle(),
                                                     conn, num_nodes, true);
    CHKERR fePtr->mField.get_moab().get_coords(conn, 6,
                                               &*materialCoords.data().begin());
    CHKERR fePtr->getNodeData(materialPositionsField, materialCoords, false);
    MoFEMFunctionReturn(0);
  };

  auto get_dofs_data_for_slave_and_master = [&] {
    slaveSpatialCoords.resize(3, 3, false);
    slaveMaterialCoords.resize(3, 3, false);
    masterSpatialCoords.resize(3, 3, false);
    masterMaterialCoords.resize(3, 3, false);
    for (size_t n = 0; n != 3; ++n) {
      for (size_t d = 0; d != 3; ++d) {
        masterSpatialCoords(n, d) = spatialCoords(3 * n + d);
        slaveSpatialCoords(n, d) = spatialCoords(3 * (n + 3) + d);
        masterMaterialCoords(n, d) = materialCoords(3 * n + d);
        slaveMaterialCoords(n, d) = materialCoords(3 * (n + 3) + d);
      }
    }
  };

  auto calculate_shape_base_functions = [&](const int nb_gauss_pts) {
    MoFEMFunctionBegin;
    if (nb_gauss_pts != fePtr->gaussPtsMaster.size2())
      SETERRQ2(
          PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
          "Inconsistent size of slave and master integration points (%d != %d)",
          nb_gauss_pts, fePtr->gaussPtsMaster.size2());
    slaveN.resize(nb_gauss_pts, 3, false);
    masterN.resize(nb_gauss_pts, 3, false);
    CHKERR Tools::shapeFunMBTRI(&slaveN(0, 0), &fePtr->gaussPtsSlave(0, 0),
                                &fePtr->gaussPtsSlave(1, 0), nb_gauss_pts);
    CHKERR Tools::shapeFunMBTRI(&masterN(0, 0), &fePtr->gaussPtsMaster(0, 0),
                                &fePtr->gaussPtsMaster(1, 0), nb_gauss_pts);
    MoFEMFunctionReturn(0);
  };

  auto get_diff_ksi_master = [&]() -> MatrixDouble & {
    if (CONVECT_MASTER)
      return diffKsiMaster;
    else
      return diffKsiSlave;
  };

  auto get_diff_ksi_slave = [&]() -> MatrixDouble & {
    if (CONVECT_MASTER)
      return diffKsiSlave;
    else
      return diffKsiMaster;
  };

  auto get_slave_material_coords = [&]() -> MatrixDouble & {
    if (CONVECT_MASTER)
      return slaveMaterialCoords;
    else
      return masterMaterialCoords;
  };

  auto get_master_gauss_pts = [&]() -> MatrixDouble & {
    if (CONVECT_MASTER)
      return fePtr->gaussPtsMaster;
    else
      return fePtr->gaussPtsSlave;
  };

  auto get_slave_spatial_coords = [&]() -> MatrixDouble & {
    if (CONVECT_MASTER)
      return slaveSpatialCoords;
    else
      return masterSpatialCoords;
  };

  auto get_master_spatial_coords = [&]() -> MatrixDouble & {
    if (CONVECT_MASTER)
      return masterSpatialCoords;
    else
      return slaveSpatialCoords;
  };

  auto get_slave_n = [&]() -> MatrixDouble & {
    if (CONVECT_MASTER)
      return slaveN;
    else
      return masterN;
  };

  auto get_master_n = [&]() -> MatrixDouble & {
    if (CONVECT_MASTER)
      return masterN;
    else
      return slaveN;
  };

  auto convect_points = [get_diff_ksi_master, get_diff_ksi_slave,
                         get_slave_material_coords, get_master_gauss_pts,
                         get_slave_spatial_coords, get_master_spatial_coords,
                         get_slave_n, get_master_n](const int nb_gauss_pts) {
    MatrixDouble3by3 A(2, 2);
    MatrixDouble3by3 invA(2, 2);
    VectorDouble3 F(2);

    auto get_t_coords = [](auto &m) {
      return FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3>{
          &m(0, 0), &m(0, 1), &m(0, 2)};
    };

    auto get_t_xi = [](auto &m) {
      return FTensor::Tensor1<FTensor::PackPtr<double *, 1>, 2>{&m(0, 0),
                                                                &m(1, 0)};
    };

    auto get_t_diff = []() {
      return FTensor::Tensor1<FTensor::PackPtr<double const *, 2>, 2>{
          &Tools::diffShapeFunMBTRI[0], &Tools::diffShapeFunMBTRI[1]};
    };

    auto get_t_tau = []() {
      FTensor::Tensor2<double, 3, 2> t_tau;
      return t_tau;
    };

    auto get_t_x = []() {
      FTensor::Tensor1<double, 3> t_x;
      return t_x;
    };

    auto get_t_F = [&]() {
      return FTensor::Tensor1<FTensor::PackPtr<double *, 0>, 2>{&F[0], &F[1]};
    };

    auto get_t_A = [&](auto &m) {
      return FTensor::Tensor2<FTensor::PackPtr<double *, 0>, 2, 2>{
          &m(0, 0), &m(0, 1), &m(1, 0), &m(1, 1)};
    };

    auto get_diff_ksi = [](auto &m, const int gg) {
      return FTensor::Tensor2<FTensor::PackPtr<double *, 1>, 2, 3>(
          &m(0, gg), &m(1, gg), &m(2, gg), &m(3, gg), &m(4, gg), &m(5, gg));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'I', 2> I;
    FTensor::Index<'J', 2> J;
    FTensor::Index<'K', 2> K;
    FTensor::Index<'L', 2> L;

    get_diff_ksi_master().resize(6, 3 * nb_gauss_pts, false);
    get_diff_ksi_slave().resize(6, 3 * nb_gauss_pts, false);

    auto t_xi_master = get_t_xi(get_master_gauss_pts());
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      auto t_tau = get_t_tau();
      auto t_x_slave = get_t_x();
      auto t_x_master = get_t_x();
      auto t_mat = get_t_A(A);
      auto t_f = get_t_F();

      auto newton_solver = [&]() {
        auto get_values = [&]() {
          t_tau(i, I) = 0;
          t_x_slave(i) = 0;
          t_x_master(i) = 0;

          auto t_slave_material_coords =
              get_t_coords(get_slave_material_coords());
          auto t_slave_spatial_coords =
              get_t_coords(get_slave_spatial_coords());
          auto t_master_spatial_coords =
              get_t_coords(get_master_spatial_coords());
          double *slave_base = &get_slave_n()(gg, 0);
          double *master_base = &get_master_n()(gg, 0);
          auto t_diff = get_t_diff();
          for (size_t n = 0; n != 3; ++n) {

            t_tau(i, J) += t_diff(J) * t_slave_material_coords(i);
            t_x_slave(i) += (*slave_base) * t_slave_spatial_coords(i);
            t_x_master(i) += (*master_base) * t_master_spatial_coords(i);

            ++t_diff;
            ++t_slave_material_coords;
            ++t_slave_spatial_coords;
            ++t_master_spatial_coords;
            ++slave_base;
            ++master_base;
          }
        };

        auto assemble = [&]() {
          t_mat(I, J) = 0;
          auto t_master_spatial_coords =
              get_t_coords(get_master_spatial_coords());
          auto t_diff = get_t_diff();
          for (size_t n = 0; n != 3; ++n) {
            t_mat(I, J) += t_diff(J) * t_tau(i, I) * t_master_spatial_coords(i);
            ++t_diff;
            ++t_master_spatial_coords;
          };
          t_f(I) = t_tau(i, I) * (t_x_slave(i) - t_x_master(i));
        };

        auto update = [&]() {
          t_xi_master(I) += t_f(I);
          get_master_n()(gg, 0) =
              Tools::shapeFunMBTRI0(t_xi_master(0), t_xi_master(1));
          get_master_n()(gg, 1) =
              Tools::shapeFunMBTRI1(t_xi_master(0), t_xi_master(1));
          get_master_n()(gg, 2) =
              Tools::shapeFunMBTRI2(t_xi_master(0), t_xi_master(1));
        };

        auto linear_solver = [&]() {
          ublas::lu_factorize(A);
          ublas::inplace_solve(A, F, ublas::unit_lower_tag());
          ublas::inplace_solve(A, F, ublas::upper_tag());
        };

        auto invert_A = [&]() {
          ublas::lu_factorize(A);
          invA.resize(2, 2, false);
          noalias(invA) = ublas::identity_matrix<double>(2);
          ublas::inplace_solve(A, invA, ublas::unit_lower_tag());
          ublas::inplace_solve(A, invA, ublas::upper_tag());
        };

        auto nonlinear_solve = [&]() {
          constexpr double tol = 1e-12;
          constexpr int max_it = 10;
          int it = 0;
          double eps;

          do {

            get_values();
            assemble();
            linear_solver();
            update();

            eps = norm_2(F);

          } while (eps > tol && (it++) < max_it);
        };

        nonlinear_solve();
        get_values();
        assemble();
        invert_A();

        auto get_diff_slave = [&]() {
          auto t_inv_A = get_t_A(invA);
          auto t_diff_xi_slave = get_diff_ksi(get_diff_ksi_slave(), 3 * gg);
          double *slave_base = &get_slave_n()(gg, 0);
          for (size_t n = 0; n != 3; ++n) {
            t_diff_xi_slave(I, i) = t_inv_A(I, J) * t_tau(i, J) * (*slave_base);
            ++t_diff_xi_slave;
            ++slave_base;
          }
        };

        auto get_diff_master = [&]() {
          auto t_inv_A = get_t_A(invA);
          auto t_diff_xi_master = get_diff_ksi(get_diff_ksi_master(), 3 * gg);
          auto t_diff = get_t_diff();
          double *master_base = &get_master_n()(gg, 0);
          FTensor::Tensor4<double, 2, 2, 2, 2> t_diff_A;
          t_diff_A(I, J, K, L) = -t_inv_A(I, K) * t_inv_A(L, J);
          for (size_t n = 0; n != 3; ++n) {
            t_diff_xi_master(I, i) =
                (t_diff_A(I, J, K, L) * (t_f(J) * t_diff(L))) * t_tau(i, K) -
                t_inv_A(I, J) * t_tau(i, J) * (*master_base);
            ++t_diff_xi_master;
            ++master_base;
            ++t_diff;
          }
        };

        get_diff_master();
        get_diff_slave();
      };

      newton_solver();

      ++t_xi_master;
    }
  };

  const int nb_gauss_pts = fePtr->gaussPtsSlave.size2();
  CHKERR fePtr->getNodeData(sparialPositionsField, spatialCoords);
  CHKERR get_material_dofs_from_coords();
  get_dofs_data_for_slave_and_master();
  CHKERR calculate_shape_base_functions(nb_gauss_pts);
  convect_points(nb_gauss_pts);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::ConvectMasterContactElement::setGaussPts(int order) {
  MoFEMFunctionBegin;
  CHKERR SimpleContactElement::setGaussPts(order);
  CHKERR convectPtr->convectSlaveIntegrationPts<true>();
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::ConvectSlaveContactElement::setGaussPts(int order) {
  MoFEMFunctionBegin;
  CHKERR SimpleContactElement::setGaussPts(order);
  CHKERR convectPtr->convectSlaveIntegrationPts<false>();
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetNormalSlave::doWork(int side,
                                                              EntityType type,
                                                              EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  FTensor::Index<'i', 3> i;

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  const double *normal_slave_ptr = &getNormalSlave()[0];

  commonDataSimpleContact->normalVectorSlavePtr.get()->resize(3);
  commonDataSimpleContact->normalVectorSlavePtr.get()->clear();

  auto t_normal =
      get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

  for (int ii = 0; ii != 3; ++ii)
    t_normal(ii) = normal_slave_ptr[ii];

  const double normal_length = sqrt(t_normal(i) * t_normal(i));
  t_normal(i) = t_normal(i) / normal_length;

  commonDataSimpleContact->areaSlave = 0.5 * normal_length;

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetNormalAndTangentsFace::doWork(
    int side, EntityType type, EntData &data) {

  MoFEMFunctionBegin;
  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);
  
  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);
  
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  commonDataSimpleContact->normalVectorFacePtr.get()->resize(3);
  commonDataSimpleContact->normalVectorFacePtr.get()->clear();

  commonDataSimpleContact->tangentOneVectorFacePtr->resize(3, false);
  commonDataSimpleContact->tangentOneVectorFacePtr->clear();

  commonDataSimpleContact->tangentTwoVectorFacePtr->resize(3, false);
  commonDataSimpleContact->tangentTwoVectorFacePtr->clear();

  const double *normal_slave_ptr = &getNormal()[0];

  auto t_normal =
      get_tensor_vec(*(commonDataSimpleContact->normalVectorFacePtr));

  for (int ii = 0; ii != 3; ++ii)
    t_normal(ii) = normal_slave_ptr[ii];

  const double normal_length = sqrt(t_normal(i) * t_normal(i));
  t_normal(i) = t_normal(i) / normal_length;

  const double *tangent_one_slave_ptr = &getTangent1()[0];
  // const double *tangent_two_slave_ptr = &getTangent2()[0];

  auto t_tangent_1 =
      get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorFacePtr));

  auto t_tangent_2 =
      get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorFacePtr));

  for (int ii = 0; ii != 3; ++ii) {
    t_tangent_1(ii) = tangent_one_slave_ptr[ii];
    // t_tangent_2(ii) = tangent_two_slave_ptr[ii];
  }

  const double l_tan_1 = sqrt(t_tangent_1(i) * t_tangent_1(i));

  t_tangent_2(j) = FTensor::levi_civita(i, j, k) * t_tangent_1(i) * t_normal(k);

  const double l_tan_2 = sqrt(t_tangent_2(i) * t_tangent_2(i));

  t_tangent_1(i) = t_tangent_1(i) / l_tan_1;
  t_tangent_2(i) = t_tangent_2(i) / l_tan_2;

  MoFEMFunctionReturn(0);
    }

MoFEMErrorCode SimpleContactProblem::OpGetNormalForTri::doWork(int side,
                                                              EntityType type,
                                                              EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  FTensor::Index<'i', 3> i;

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  const double *normal_slave_ptr = &getNormal()[0];

  commonDataSimpleContact->normalVectorSlavePtr.get()->resize(3);
  commonDataSimpleContact->normalVectorSlavePtr.get()->clear();

  auto t_normal =
      get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

  for (int ii = 0; ii != 3; ++ii)
    t_normal(ii) = normal_slave_ptr[ii];

  const double normal_length = sqrt(t_normal(i) * t_normal(i));
  t_normal(i) = t_normal(i) / normal_length;

  commonDataSimpleContact->areaSlave = 0.5 * normal_length;

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetOrthonormalTangents::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  commonDataExtendedContact->tangentOneVectorSlavePtr->resize(3, false);
  commonDataExtendedContact->tangentOneVectorSlavePtr->clear();

  commonDataExtendedContact->tangentTwoVectorSlavePtr->resize(3, false);
  commonDataExtendedContact->tangentTwoVectorSlavePtr->clear();

  const double *tangent_one_slave_ptr = &getTangentSlaveOne()[0];
  const double *tangent_two_slave_ptr = &getTangentSlaveTwo()[0];

  auto t_tangent_1 =
      get_tensor_vec(*(commonDataExtendedContact->tangentOneVectorSlavePtr));

  auto t_tangent_2 =
      get_tensor_vec(*(commonDataExtendedContact->tangentTwoVectorSlavePtr));

for (int ii = 0; ii != 3; ++ii) {
  t_tangent_1(ii) = tangent_one_slave_ptr[ii];
  t_tangent_2(ii) = tangent_two_slave_ptr[ii];
}

const double l_tan_1 = sqrt(t_tangent_1(i) * t_tangent_1(i));

const double l_tan_2 = sqrt(t_tangent_2(i) * t_tangent_2(i));

t_tangent_1(i) = t_tangent_1(i) / l_tan_1;
t_tangent_2(i) = t_tangent_2(i) / l_tan_2;

MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetNormalMaster::doWork(int side,
                                                               EntityType type,
                                                               EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  FTensor::Index<'i', 3> i;

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  const double *normal_master_ptr = &getNormalMaster()[0];
  commonDataSimpleContact->normalVectorMasterPtr.get()->resize(3);
  commonDataSimpleContact->normalVectorMasterPtr.get()->clear();

  auto t_normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorMasterPtr.get()[0]);
  for (int ii = 0; ii != 3; ++ii)
    t_normal(ii) = normal_master_ptr[ii];

  const double normal_length = sqrt(t_normal(i) * t_normal(i));
  t_normal(i) = t_normal(i) / normal_length;
  commonDataSimpleContact->areaMaster = 0.5 * normal_length;

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetPositionAtGaussPtsMaster::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;
  const int nb_dofs = data.getFieldData().size();

  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  if (type == MBVERTEX) {
    commonDataSimpleContact->positionAtGaussPtsMasterPtr.get()->resize(
        3, nb_gauss_pts, false);

    commonDataSimpleContact->positionAtGaussPtsMasterPtr.get()->clear();
  }

  auto t_position_master = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsMasterPtr);

  int nb_base_fun_col = data.getFieldData().size() / 3;

  FTensor::Index<'i', 3> i;

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

    FTensor::Tensor1<double *, 3> t_field_data_master(
        &data.getFieldData()[0], &data.getFieldData()[1],
        &data.getFieldData()[2], 3);

    for (int bb = 0; bb != nb_base_fun_col; bb++) {
      t_position_master(i) += t_base_master * t_field_data_master(i);

      ++t_base_master;
      ++t_field_data_master;
    }
    ++t_position_master;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetPositionAtGaussPtsSlave::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;
  const int nb_dofs = data.getFieldData().size();

  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();
  if (type == MBVERTEX) {
    commonDataSimpleContact->positionAtGaussPtsSlavePtr.get()->resize(
        3, nb_gauss_pts, false);
    commonDataSimpleContact->positionAtGaussPtsSlavePtr.get()->clear();
  }

  auto t_position_slave = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsSlavePtr);

  int nb_base_fun_col = data.getFieldData().size() / 3;

  FTensor::Index<'i', 3> i;

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));

    FTensor::Tensor1<double *, 3> t_field_data_slave(
        &data.getFieldData()[0], &data.getFieldData()[1],
        &data.getFieldData()[2], 3); // in-between

    for (int bb = 0; bb != nb_base_fun_col; bb++) {
      t_position_slave(i) += t_base_slave * t_field_data_slave(i);

      ++t_base_slave;
      ++t_field_data_slave;
    }
    ++t_position_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetGapSlave::doWork(int side,
                                                           EntityType type,
                                                           EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  const int nb_gauss_pts = data.getN().size1();

  commonDataSimpleContact->gapPtr.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->gapPtr.get()->clear();

  FTensor::Index<'i', 3> i;

  auto t_position_master_gp = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsMasterPtr);
  auto t_position_slave_gp = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsSlavePtr);

  auto t_gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto t_normal_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    t_gap_ptr -=
        t_normal_at_gp(i) * (t_position_slave_gp(i) - t_position_master_gp(i));
    ++t_position_slave_gp;
    ++t_position_master_gp;
    ++t_gap_ptr;
  } // for gauss points

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetLagMulAtGaussPtsSlave::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  if (type == MBVERTEX) {
    commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->resize(nb_gauss_pts);
    commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->clear();
  }

  int nb_base_fun_row = data.getFieldData().size();

  auto t_lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    FTensor::Tensor0<double *> t_base_lambda(&data.getN()(gg, 0));

    FTensor::Tensor0<double *> t_field_data_slave(&data.getFieldData()[0]);
    for (int bb = 0; bb != nb_base_fun_row; bb++) {
      t_lagrange_slave += t_base_lambda * t_field_data_slave;
      ++t_base_lambda;
      ++t_field_data_slave;
    }
    ++t_lagrange_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetFromHdivLagMulAtGaussPtsSlave::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBTRI)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->clear();

  auto get_tensor_vec = [](VectorDouble &n, const int r) {
    return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
  };

  FTensor::Index<'i', 3> i;

  auto t_normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

  auto t_traction =
      getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

  auto t_lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    t_lagrange_slave += t_traction(i) * t_normal(i);
    ++t_traction;
    ++t_lagrange_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetFromHdivLagMulAtGaussPtsSlavePostProc::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);
  
  if (type != MBTRI)
    MoFEMFunctionReturnHot(0);

    const int nb_gauss_pts = data.getN().size1();

      commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->resize(nb_gauss_pts);
      commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->clear();
    

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    FTensor::Index<'i', 3> i;

    auto t_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

    auto t_lagrange_slave =
        getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      t_lagrange_slave += t_traction(i) * t_normal(i);
      ++t_traction;
      ++t_lagrange_slave;
    }

    MoFEMFunctionReturn(0);
  }

MoFEMErrorCode SimpleContactProblem::OpCalLagrangeMultPostProc::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  if (type == MBVERTEX) {
    commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->resize(nb_gauss_pts);
    commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->clear();
  }

  int nb_base_fun_row = data.getFieldData().size();

  auto t_lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    FTensor::Tensor0<double *> t_base_lambda(&data.getN()(gg, 0));

    FTensor::Tensor0<double *> t_field_data_slave(&data.getFieldData()[0]);
    for (int bb = 0; bb != nb_base_fun_row; bb++) {
      t_lagrange_slave += t_base_lambda * t_field_data_slave;
      ++t_base_lambda;
      ++t_field_data_slave;
    }
    ++t_lagrange_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpPrintLagMulAtGaussPtsSlave::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  auto t_lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
  auto t_gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);
  cout << "-----------------------------" << endl;

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    cout << "gp: " << gg << " | gap: " << t_gap_ptr
         << " | lm: " << t_lagrange_slave
         << " | gap * lm = " << t_gap_ptr * t_lagrange_slave << endl;
    ++t_lagrange_slave;
    ++t_gap_ptr;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpLagGapProdGaussPtsSlave::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  commonDataSimpleContact->lagGapProdPtr.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->lagGapProdPtr.get()->clear();

  int nb_base_fun_row = data.getFieldData().size();

  auto t_lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto t_lag_gap_prod_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagGapProdPtr);

  auto t_gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    t_lag_gap_prod_slave += t_gap_ptr * t_lagrange_slave;
    ++t_gap_ptr;
    ++t_lag_gap_prod_slave;
    ++t_lagrange_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalContactTractionOnMaster::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  const int nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    const int nb_gauss_pts = data.getN().size1();
    int nb_base_fun_col = nb_dofs / 3;

    vecF.resize(nb_dofs, false);
    vecF.clear();

    const double area_m =
        commonDataSimpleContact->areaMaster; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    FTensor::Index<'i', 3> i;

    auto t_lagrange_slave =
        getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

    auto t_const_unit_n = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto t_w = getFTensor0IntegrationWeightMaster();

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      double val_m = t_w * area_m;

      auto t_base_master = data.getFTensor0N(gg, 0);
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_assemble_m{
          &vecF[0], &vecF[1], &vecF[2]};

      for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
        const double m = val_m * t_base_master * t_lagrange_slave;
        t_assemble_m(i) -= m * t_const_unit_n(i);
        ++t_base_master;
        ++t_assemble_m;
      }

      ++t_lagrange_slave;
      ++t_w;
    } // for gauss points

    CHKERR VecSetValues(getSNESf(), data, &vecF[0], ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalContactTractionOnSlave::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  const int nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    const int nb_gauss_pts = data.getN().size1();
    int nb_base_fun_col = nb_dofs / 3;

    vecF.resize(nb_dofs, false);
    vecF.clear();

    const double area_m =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    FTensor::Index<'i', 3> i;

    auto t_lagrange_slave =
        getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

    auto t_const_unit_n = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto t_w = getFTensor0IntegrationWeightSlave();

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      double val_m = t_w * area_m;

      auto t_base_slave = data.getFTensor0N(gg, 0);
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_assemble_m{
          &vecF[0], &vecF[1], &vecF[2]};

      for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
        const double m = val_m * t_base_slave * t_lagrange_slave;
        t_assemble_m(i) += m * t_const_unit_n(i);
        ++t_base_slave;
        ++t_assemble_m;
      }

      ++t_lagrange_slave;
      ++t_w;
    } // for gauss points

    CHKERR VecSetValues(getSNESf(), data, &vecF[0], ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalIntCompFunSlave::doWork(int side, EntityType type,
                                                   EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  int nb_base_fun_col = data.getFieldData().size();
  const double area_s =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  vecR.resize(nb_base_fun_col, false);
  vecR.clear();

  auto t_lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
  auto t_gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);
  auto t_w = getFTensor0IntegrationWeightSlave();

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    const double val_s =
        t_w * area_s *
        SimpleContactProblem::ConstrainFunction(cN, t_gap_gp, t_lagrange_slave);

    auto t_base_lambda = data.getFTensor0N(gg, 0);
    for (int bbr = 0; bbr != nb_base_fun_col; bbr++) {
      vecR[bbr] += val_s * t_base_lambda;

      ++t_base_lambda; // update rows
    }

    ++t_lagrange_slave;
    ++t_gap_gp;
    ++t_w;
  } // for gauss points

  CHKERR VecSetValues(getSNESf(), data, &vecR[0], ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalContactTractionOverLambdaMasterSlave::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {
    const int nb_gauss_pts = row_data.getN().size1();

    int nb_base_fun_row = row_data.getFieldData().size() / 3;
    int nb_base_fun_col = col_data.getFieldData().size();

    const double area_master =
        commonDataSimpleContact->areaMaster; // same area in master and slave

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor1<double *, 3>(&m(r + 0, c), &m(r + 1, c),
                                           &m(r + 2, c));
    };

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    FTensor::Index<'i', 3> i;

    NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
    NN.clear();

    auto const_unit_n =
        get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr.get()));

    auto t_w = getFTensor0IntegrationWeightMaster();

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      double val_m = t_w * area_master;
      auto t_base_master = row_data.getFTensor0N(gg, 0);

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 0);
        auto t_base_lambda = col_data.getFTensor0N(gg, 0);
        const double m = val_m * t_base_master;
        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
          const double n = m * t_base_lambda;
          t_assemble_m(i) -= n * const_unit_n(i);
          ++t_assemble_m;
          ++t_base_lambda; // update cols slave
        }
        ++t_base_master; // update rows master
      }
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalContactTractionOverLambdaSlaveSlave::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {
    const int nb_gauss_pts = row_data.getN().size1();

    int nb_base_fun_row = row_data.getFieldData().size() / 3;
    int nb_base_fun_col = col_data.getFieldData().size();

    const double area_slave =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor1<double *, 3>(&m(r + 0, c), &m(r + 1, c),
                                           &m(r + 2, c));
    };

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    FTensor::Index<'i', 3> i;

    NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
    NN.clear();

    auto const_unit_n =
        get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr.get()));

    auto t_w = getFTensor0IntegrationWeightSlave();

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      double val_m = t_w * area_slave;
      auto t_base_master = row_data.getFTensor0N(gg, 0);

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 0);
        auto t_base_lambda = col_data.getFTensor0N(gg, 0);
        const double m = val_m * t_base_master;
        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
          const double n = m * t_base_lambda;
          t_assemble_m(i) += n * const_unit_n(i);
          ++t_assemble_m;
          ++t_base_lambda; // update cols slave
        }
        ++t_base_master; // update rows master
      }
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalDerIntCompFunOverLambdaSlaveSlave::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {
    const int nb_gauss_pts = row_data.getN().size1();
    const double area_slave =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    NN.resize(nb_row, nb_col, false);
    NN.clear();

    auto t_lagrange_slave =
        getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
    auto t_gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);
    auto t_w = getFTensor0IntegrationWeightSlave();

    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      const double val_s = SimpleContactProblem::ConstrainFunction_dl(
                               cN, t_gap_gp, t_lagrange_slave) *
                           t_w * area_slave;

      FTensor::Tensor0<FTensor::PackPtr<double *, 1>> t_mat(
          &*NN.data().begin());

      auto t_base_lambda_row = row_data.getFTensor0N(gg, 0);
      for (int bbr = 0; bbr != nb_row; ++bbr) {
        auto t_base_lambda_col = col_data.getFTensor0N(gg, 0);
        const double s = val_s * t_base_lambda_row;
        for (int bbc = 0; bbc != nb_col; ++bbc) {

          t_mat += s * t_base_lambda_col;

          ++t_mat;
          ++t_base_lambda_col; // update cols
        }
        ++t_base_lambda_row; // update rows
      }

      ++t_lagrange_slave;
      ++t_gap_gp;
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalDerIntCompFunOverSpatPosSlaveMaster::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {

    const int nb_gauss_pts = row_data.getN().size1();
    int nb_base_fun_row = row_data.getFieldData().size();
    int nb_base_fun_col = col_data.getFieldData().size() / 3;

    const double area_master =
        commonDataSimpleContact->areaMaster; // same area in master and slave

    NN.resize(nb_base_fun_row, 3 * nb_base_fun_col, false);
    NN.clear();

    auto get_tensor_from_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    FTensor::Index<'i', 3> i;

    auto t_const_unit_n =
        get_tensor_from_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

    auto t_const_unit_master =
        get_tensor_from_vec(*(commonDataSimpleContact->normalVectorMasterPtr));

    auto t_lagrange_slave =
        getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
    auto t_gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    auto t_w = getFTensor0IntegrationWeightMaster();
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      const double val_m = SimpleContactProblem::ConstrainFunction_dg(
                               cN, t_gap_gp, t_lagrange_slave) *
                           t_w * area_master;

      auto t_base_lambda = row_data.getFTensor0N(gg, 0);
      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

        auto t_base_master = col_data.getFTensor0N(gg, 0);
        FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_mat{
            &NN(bbr, 0), &NN(bbr, 1), &NN(bbr, 2)};
        const double m = val_m * t_base_lambda;

        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

          t_mat(i) += t_const_unit_n(i) * m * t_base_master;

          ++t_base_master; // update rows
          ++t_mat;
        }
        ++t_base_lambda; // update cols master
      }

      ++t_gap_gp;
      ++t_lagrange_slave;
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalDerIntCompFunOverSpatPosSlaveSlave::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {

    const int nb_gauss_pts = row_data.getN().size1();
    int nb_base_fun_row = row_data.getFieldData().size();
    int nb_base_fun_col = col_data.getFieldData().size() / 3;

    const double area_slave =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    NN.resize(nb_base_fun_row, 3 * nb_base_fun_col, false);
    NN.clear();

    auto get_tensor_from_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    FTensor::Index<'i', 3> i;

    auto t_lagrange_slave =
        getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
    auto t_gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    auto t_const_unit_n =
        get_tensor_from_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

    auto t_w = getFTensor0IntegrationWeightSlave();
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      const double val_m = SimpleContactProblem::ConstrainFunction_dg(
                               cN, t_gap_gp, t_lagrange_slave) *
                           t_w * area_slave;

      auto t_base_lambda = row_data.getFTensor0N(gg, 0);
      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

        auto t_base_slave = col_data.getFTensor0N(gg, 0);
        FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_mat{
            &NN(bbr, 0), &NN(bbr, 1), &NN(bbr, 2)};
        const double m = val_m * t_base_lambda;

        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

          t_mat(i) -= t_const_unit_n(i) * m * t_base_slave;

          ++t_base_slave; // update rows
          ++t_mat;
        }
        ++t_base_lambda; // update cols master
      }

      ++t_gap_gp;
      ++t_lagrange_slave;
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpMakeVtkSlave::doWork(int side,
                                                            EntityType type,
                                                            EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);
  
  int nb_dofs = data.getFieldData().size();
  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);
  int nb_gauss_pts = data.getN().size1();

  double def_vals;
  def_vals = 0;

  double def_vals_3[3];
  def_vals_3[0] = def_vals_3[1] = def_vals_3[2] = 0;

  Tag th_gap;
  CHKERR moabOut.tag_get_handle("GAP", 1, MB_TYPE_DOUBLE, th_gap,
                                MB_TAG_CREAT | MB_TAG_SPARSE, &def_vals);

  Tag th_lag_mult;
  if (lagFieldSet)
    CHKERR moabOut.tag_get_handle("LAGRANGE_MULTIPLIER", 1, MB_TYPE_DOUBLE,
                                  th_lag_mult, MB_TAG_CREAT | MB_TAG_SPARSE,
                                  &def_vals);

  Tag th_lag_gap_prod;
  if (lagFieldSet)
    CHKERR moabOut.tag_get_handle("LAG_GAP_PROD", 1, MB_TYPE_DOUBLE,
                                  th_lag_gap_prod, MB_TAG_CREAT | MB_TAG_SPARSE,
                                  &def_vals);

  Tag th_lag_3_lagrange;
  CHKERR moabOut.tag_get_handle("3_LAGRANGE_MULTIPLIER", 3, MB_TYPE_DOUBLE,
                                th_lag_3_lagrange, MB_TAG_CREAT | MB_TAG_SPARSE,
                                &def_vals);

  auto get_tag_pos = [&](const std::string name) {
    Tag th;
    constexpr std::array<double, 3> def_vals = {0, 0, 0};
    CHKERR moabOut.tag_get_handle(name.c_str(), 3, MB_TYPE_DOUBLE, th,
                                  MB_TAG_CREAT | MB_TAG_SPARSE,
                                  def_vals.data());
    return th;
  };
  auto th_pos_master = get_tag_pos("MASTER_SPATIAL_POSITION");
  auto th_pos_slave = get_tag_pos("SLAVE_SPATIAL_POSITION");
  auto th_master_coords = get_tag_pos("MASTER_GAUSS_PTS_COORDS");

  EntityHandle new_vertex = getFEEntityHandle();

  auto t_gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto t_lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
  auto t_lag_gap_prod_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagGapProdPtr);
  auto t_position_master = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsMasterPtr);
  auto t_position_slave = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsSlavePtr);
  auto t_traction =
      getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

  std::array<double, 3> pos_vec;

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    const double *slave_coords_ptr = &(getCoordsAtGaussPtsSlave()(gg, 0));
    CHKERR moabOut.create_vertex(slave_coords_ptr, new_vertex);
    CHKERR moabOut.tag_set_data(th_gap, &new_vertex, 1, &t_gap_ptr);

    CHKERR moabOut.tag_set_data(th_lag_gap_prod, &new_vertex, 1,
                                &t_lag_gap_prod_slave);
    CHKERR moabOut.tag_set_data(th_lag_mult, &new_vertex, 1, &t_lagrange_slave);

    

    auto get_vec_ptr = [&](auto t) {
      for (int dd = 0; dd != 3; ++dd)
        pos_vec[dd] = t(dd);
      return pos_vec.data();
    };

    CHKERR moabOut.tag_set_data(th_lag_3_lagrange, &new_vertex, 1,
                                get_vec_ptr(t_traction));

    CHKERR moabOut.tag_set_data(th_pos_master, &new_vertex, 1,
                                get_vec_ptr(t_position_master));
    CHKERR moabOut.tag_set_data(th_pos_slave, &new_vertex, 1,
                                get_vec_ptr(t_position_slave));
    const double *master_coords_ptr = &(getCoordsAtGaussPtsMaster()(gg, 0));
    CHKERR moabOut.tag_set_data(th_master_coords, &new_vertex, 1,
                                master_coords_ptr);

    ++t_gap_ptr;
    ++t_lagrange_slave;
    ++t_lag_gap_prod_slave;
    ++t_position_master;
    ++t_position_slave;
    ++t_traction;
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpMakeTestTextFile::doWork(int side,
                                                                EntityType type,
                                                                EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  int nb_dofs = data.getFieldData().size();
  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);
  int nb_gauss_pts = data.getN().size1();

  auto t_gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto t_lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
  double d_gap;
  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    const double d_lambda =
        std::abs(t_lagrange_slave) < TOL ? 0.0 : t_lagrange_slave;
    d_gap = std::abs(t_gap_ptr) < TOL ? 0.0 : t_gap_ptr;
    mySplit << d_lambda << " " << d_gap << " " << std::endl;

    ++t_gap_ptr;
    ++t_lagrange_slave;
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpPostProcContactContinuous::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBVERTEX)
    PetscFunctionReturn(0);

  double def_VAL[9];
  bzero(def_VAL, 9 * sizeof(double));

  Tag th_lag_mult;

  CHKERR postProcMesh.tag_get_handle("LAGRANGE_MULTIPLIER", 1, MB_TYPE_DOUBLE,
                                     th_lag_mult, MB_TAG_CREAT | MB_TAG_SPARSE,
                                     def_VAL);

  auto t_lagrange =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  const int nb_gauss_pts =
      commonDataSimpleContact->lagMultAtGaussPtsPtr->size();

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    CHKERR postProcMesh.tag_set_data(th_lag_mult, &mapGaussPts[gg], 1,
                                     &t_lagrange);
    ++t_lagrange;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpConstrainDomainRhs::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);
    const size_t nb_base_functions = data.getN().size2() / 3;
    auto t_w = getFTensor0IntegrationWeight();
    // auto t_base = data.getFTensor1N<3>();
    // auto t_diff_base = data.getFTensor2DiffN<3, 3>();
    auto t_lambdas = getFTensor2FromMat<3, 3>(
        *(commonDataSimpleContact->lagMatMultAtGaussPtsPtr));
    auto t_x = getFTensor1FromMat<3>(*(commonDataSimpleContact->xAtPts));
    auto t_X = getFTensor1FromMat<3>(*(commonDataSimpleContact->xInitAtPts));
    auto t_grad =
        getFTensor2FromMat<3, 3>(*(commonDataSimpleContact->mGradPtr));
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      const double alpha = getVolume() * t_w;
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf{&nf[0], &nf[1],
                                                              &nf[2]};
      // t_grad(0, 0) -= 1;
      // t_grad(1, 1) -= 1;
      // t_grad(2, 2) -= 1;
      auto t_diff_base = data.getFTensor2DiffN<3,3>(gg, 0);
      auto t_base = data.getFTensor1N<3>(gg, 0);

      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        const double t_div_base =
            t_diff_base(0, 0) + t_diff_base(1, 1) + t_diff_base(2, 2);
        t_nf(i) +=  alpha * (t_base(j) * t_grad(i, j));
        // t_nf(i) += alpha * (t_base(i) * t_grad(i, j));
        t_nf(i) +=  alpha * t_div_base * (t_x(i) /*- t_X(i)*/);
        ++t_nf;
        ++t_base;
        ++t_diff_base;
      }
      // for (; bb < nb_base_functions; ++bb) {
      //   ++t_base;
      //   ++t_diff_base;
      // }
      ++t_lambdas;
      ++t_x;
      ++t_X;
      ++t_grad;
      ++t_w;
    }
    CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpInternalDomainContactRhs::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);
    const size_t nb_base_functions = data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    // auto t_base = data.getFTensor0N();
    // auto t_diff_base = data.getFTensor1DiffN<3>();
    auto t_lambdas = getFTensor2FromMat<3, 3>(
        *(commonDataSimpleContact->lagMatMultAtGaussPtsPtr));
    auto t_div_stress = getFTensor1FromMat<3>(
        *(commonDataSimpleContact->contactLagrangeHdivDivergencePtr));

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      const double alpha = getVolume() * t_w;
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf{&nf[0], &nf[1],
                                                              &nf[2]};

      auto t_diff_base = data.getFTensor1DiffN<3>(gg, 0);
      auto t_base = data.getFTensor0N(gg, 0);
      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        t_nf(i) += alpha * t_base * t_div_stress(i);
        t_nf(i) += alpha * t_diff_base(j) * t_lambdas(i, j);
        ++t_nf;
        ++t_base;
        ++t_diff_base;
      }

      // for (; bb < nb_base_functions; ++bb) {
      //   ++t_base;
      //   ++t_diff_base;
      // }

      ++t_div_stress;
      ++t_lambdas;
      ++t_w;
      ++t_grad;
    }
    CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpConstrainBoundaryTraction::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPtsSlave().size2();
  
  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    if (/*side == 0 &&*/ type == MBTRI) {
      commonDataSimpleContact->contactTractionPtr->resize(3, nb_gauss_pts);
      commonDataSimpleContact->contactTractionPtr->clear();
    }

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto t_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);
    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));
    size_t nb_base_functions = data.getN().size2() / 3;
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      auto t_base = data.getFTensor1N<3>(gg, 0);
      auto t_field_data = data.getFTensor1FieldData<3>();
      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        // cerr << " WTF "
        //      << "   bb    " << bb << "   gg   " << gg << "   t_base       "
        //      << t_base << "    t_field_data      " << t_field_data << "\n";
        t_traction(j) += (t_base(i) * t_normal(i)) * t_field_data(j);

        ++t_field_data;
        ++t_base;
      }
      // for (; bb < nb_base_functions; ++bb)
      //   ++t_base;

      ++t_traction;
    }
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpConstrainBoundaryTractionForFace::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPts().size2();

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    if (/*side == 0 &&*/ type == MBTRI) {
      commonDataSimpleContact->contactTractionPtr->resize(3, nb_gauss_pts);
      commonDataSimpleContact->contactTractionPtr->clear();
    }

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto t_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorFacePtr.get()[0], 0);
    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));
    size_t nb_base_functions = data.getN().size2() / 3;
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      auto t_base = data.getFTensor1N<3>(gg, 0);
      auto t_field_data = data.getFTensor1FieldData<3>();
      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        // cerr << " WTF "
        //      << "   bb    " << bb << "   gg   " << gg << "   t_base       "
        //      << t_base << "    t_field_data      " << t_field_data << "\n";
        t_traction(j) += (t_base(i) * t_normal(i)) * t_field_data(j);

        ++t_field_data;
        ++t_base;
      }
      // for (; bb < nb_base_functions; ++bb)
      //   ++t_base;

      ++t_traction;
    }
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpConstrainBoundaryTractionPostProc::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  
  const size_t nb_gauss_pts = getGaussPts().size2();
  if (side == 0 && type == MBTRI) {
    commonDataSimpleContact->contactTractionPtr->resize(3, nb_gauss_pts);
    commonDataSimpleContact->contactTractionPtr->clear();
  }

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };


    auto t_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

    size_t nb_base_functions = data.getN().size2() / 3;
    auto t_base = data.getFTensor1N<3>();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      auto t_field_data = data.getFTensor1FieldData<3>();
      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        // cerr << " WTF "
        //      << "   bb    " << bb << "   gg   " << gg << "   t_base       "
        //      << t_base << "    t_field_data      " << t_field_data << "\n";
        //this 2 is comming from the triangle transformation (is it eventually done time 0.5?)
        t_traction(j) += (t_base(i) * t_normal(i)) * t_field_data(j);

        ++t_field_data;
        ++t_base;
      }
      for (; bb < nb_base_functions; ++bb)
        ++t_base;

      ++t_traction;
    }
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpConstrainBoundaryRhs::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPtsSlave().size2();
  const size_t nb_dofs = data.getIndices().size();

  if (nb_dofs) {
    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    // cerr << " GAUSS !! " << nb_gauss_pts << "\n";
    // // FTensor::Tensor1<double, 2> t_direction{getDirection()[0],
    //                                                 getDirection()[1]};
    // FTensor::Tensor1<double, 2> t_normal{-t_direction(1),
    // t_direction(0)}; const double l = sqrt(t_normal(i) *
    // t_normal(i)); t_normal(i) /= l; t_direction(i) /= l;

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;

    const double area_m =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto t_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto t_w = getFTensor0IntegrationWeightSlave();
    // auto t_disp = getFTensor1FromMat<2>(*(commonDataPtr->contactDispPtr));

    auto position_master = getFTensor1FromMat<3>(
        *commonDataSimpleContact->positionAtGaussPtsMasterPtr);
    auto t_x_slave = getFTensor1FromMat<3>(
        *commonDataSimpleContact->positionAtGaussPtsSlavePtr);
    auto t_x_master = getFTensor1FromMat<3>(
        *commonDataSimpleContact->positionAtGaussPtsMasterPtr);

    auto t_X_slave = getFTensor1FromMat<3>(
        *commonDataSimpleContact->meshPositionAtGaussPtsSlavePtr);
    auto t_X_master = getFTensor1FromMat<3>(
        *commonDataSimpleContact->meshPositionAtGaussPtsMasterPtr);

    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

    auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    // auto t_coords = getFTensor1CoordsAtGaussPtsSlave();
    size_t nb_base_functions = data.getN().size2() / 3;
    // auto t_base = data.getFTensor1N<3>();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf{&nf[0], &nf[1],
                                                              &nf[2]};

      auto t_base = data.getFTensor1N<3>(gg, 0);
      const double alpha = t_w * area_m;

      //
      auto t_contact_normal = get_tensor_vec(
          commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

      FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
      t_contact_normal_tensor(i, j) = t_contact_normal(i) * t_contact_normal(j);
      FTensor::Tensor2<double, 3, 3> t_contact_tangent_tensor;

      auto t_tangent_1_at_gp = get_tensor_vec(
          *(commonDataSimpleContact->tangentOneVectorSlavePtr), 0);

      auto t_tangent_2_at_gp = get_tensor_vec(
          *(commonDataSimpleContact->tangentTwoVectorSlavePtr), 0);

      FTensor::Tensor1<double, 3> t_rhs_constrains;
      const double normal_traction = t_traction(i) * t_contact_normal(i);
      t_rhs_constrains(i) =
          (gap_ptr + cN * normal_traction +
                      std::abs(gap_ptr + cN * normal_traction)) *
          t_contact_normal(i);

      // cerr << "t_rhs_constrains " << t_rhs_constrains << "\n";

      // t_rhs_constrains(i) =
      //     (-t_contact_normal(j) * t_x_master(j) - cN * normal_traction +
      //      std::abs(gap_ptr - cN * normal_traction)) *
      //     t_contact_normal(i);

      //  t_rhs_constrains(i) =
      //      (t_contact_normal(j) * t_x_master(j) + cN * normal_traction -
      //      std::abs(gap_ptr - cN * normal_traction)) * t_contact_normal(i);

      //  t_rhs_constrains(i) =
      //      (gap_ptr + cN * normal_traction -
      //                  std::abs(gap_ptr - cN * normal_traction)) *
      //      t_contact_normal(i);

      //  cerr << "t_rhs_constrains     " << t_rhs_constrains << "  gap_ptr  "
      //       << gap_ptr << "   normal_traction  " << normal_traction << "\n";
      //  double check = gap_ptr - cN * normal_traction;

      //  cerr << " ASDASDASD    " << check << " gap " << gap_ptr << "\n";

      //  t_rhs_constrains(i) =
      //      (-normal_traction - (-normal_traction - cN * gap_ptr +
      //                           std::abs(-normal_traction - cN * gap_ptr)) /
      //                              2.) *
      //      t_contact_normal(i);

      //  cerr << "C fun RHS vector " << t_rhs_constrains << "\n";

      // t_rhs_constrains(i) =
      //     (normal_traction - (normal_traction - cN * gap_ptr +
      //                          std::abs(normal_traction - cN * gap_ptr)) /
      //                             2.) *
      //     t_contact_normal(i);

      //  t_rhs_constrains(i) =
      //      t_contact_normal(i) *
      //     constrian(gap0(t_coords, t_contact_normal),
      //                    gap(t_disp, t_contact_normal),
      //                    normal_traction(t_traction, t_contact_normal));
      FTensor::Tensor1<double, 3> t_rhs_tangent_disp, t_rhs_tangent_traction,
          t_rhs_normal_traction;

      FTensor::Tensor1<double, 3> tangent_1_disp, tangent_2_disp,
          tangent_1_disp_master, tangent_2_disp_master, tangent_1_traction,
          tangent_2_traction;

      t_rhs_normal_traction(i) =
          cN * t_contact_normal_tensor(i, j) * t_traction(j);

      tangent_1_disp(i) = t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
                          (t_x_slave(j) /*- t_X_slave(j)*/);

      tangent_2_disp(i) = t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) *
                          (t_x_slave(j) /*- t_X_slave(j)*/);

      tangent_1_disp_master(i) = t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
                          (t_x_master(j) /*- t_X_master(j)*/);

      tangent_2_disp_master(i) = t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) *
                          (t_x_master(j) /*- t_X_master(j)*/);

      tangent_1_traction(i) =
          t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) * t_traction(j);

      tangent_2_traction(i) =
          t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) * t_traction(j);
      // cerr << "t_tangent_1_at_gp " <<t_tangent_1_at_gp << "\n ";
      // cerr << "t_tangent_2_at_gp " <<t_tangent_2_at_gp << "\n ";

      auto t_field_data = data.getFTensor1FieldData<3>();
      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        const double beta = alpha * (t_base(i) * t_normal(i));
        // const double beta = alpha * (t_base(i) * t_normal(i));
        // const double beta = alpha * (t_base(i) * t_normal(i));

        //////////
        // t_nf(i) += beta * t_rhs_constrains(i);

        //11/08/2020
        t_nf(i) -= beta * tangent_1_disp(i);
        t_nf(i) -= beta * tangent_2_disp(i);
        // t_nf(i) += beta * t_contact_normal_tensor(i, j) * t_x_slave(j);

        // t_nf(i) += beta * t_contact_normal_tensor(i, j);
        
        //07/08/2020
        t_nf(i) -= beta * t_contact_normal_tensor(i, j) * (t_x_master(j) /*- t_X_master(j)*/);
        // t_nf(i) -= beta * tangent_1_disp_master(i);
        // t_nf(i) -= beta * tangent_2_disp_master(i);

        // t_nf(i) -= beta * t_x_master(i);

        t_nf(i) += cN * beta * tangent_1_traction(i);
        t_nf(i) += cN * beta * tangent_2_traction(i);
        //////////

        // t_nf(i) += beta * (t_x_slave(i) - t_X_slave(i));

        // This is for tangent 28/05
        // t_nf(i) += alpha * cN *
        //            (t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
        //             t_traction(j) * t_base(k) * t_normal(k));

        // t_nf(i) += alpha * cN *
        //            (t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) *
        //             t_traction(j) * t_base(k) * t_normal(k));

        // t_nf(i) += beta * t_rhs_normal_traction(i);

        //  cerr << "beta " << beta << "\n";
        ++t_nf;
        ++t_base;
        ++t_field_data;
      }
      // for (; bb < nb_base_functions; ++bb)
      //  ++t_base;

      ++t_x_slave;
      ++t_x_master;
      ++t_traction;
      ++t_X_slave;
      ++t_X_master;
      ++t_w;
      ++gap_ptr;
    }

    CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpConstrainBoundaryRhsForFace::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {
    
    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;

    const double area_m = getMeasure(); // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto t_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorFacePtr.get()[0], 0);

    auto t_w = getFTensor0IntegrationWeight();

    auto t_x =
        getFTensor1FromMat<3>(*commonDataSimpleContact->xFaceAtPts);

    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

    // auto t_coords = getFTensor1CoordsAtGaussPtsSlave();
    size_t nb_base_functions = data.getN().size2() / 3;
    // auto t_base = data.getFTensor1N<3>();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf{&nf[0], &nf[1],
                                                              &nf[2]};
      auto t_base = data.getFTensor1N<3>(gg, 0);
      const double alpha = t_w * area_m;
      
      //
      auto t_contact_normal = get_tensor_vec(
          commonDataSimpleContact->normalVectorFacePtr.get()[0], 0);
      
      FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
      t_contact_normal_tensor(i, j) = t_contact_normal(i) * t_contact_normal(j);
      FTensor::Tensor2<double, 3, 3> t_contact_tangent_tensor;
      
      auto t_tangent_1_at_gp = get_tensor_vec(
          *(commonDataSimpleContact->tangentOneVectorFacePtr), 0);

      auto t_tangent_2_at_gp = get_tensor_vec(
          *(commonDataSimpleContact->tangentTwoVectorFacePtr), 0);
      
      FTensor::Tensor1<double, 3> t_rhs_constrains;
      const double normal_traction = t_traction(i) * t_contact_normal(i);
      
      FTensor::Tensor1<double, 3> t_rhs_tangent_disp, t_rhs_tangent_traction,
          t_rhs_normal_traction;

      FTensor::Tensor1<double, 3> tangent_1_disp, tangent_2_disp,
          tangent_1_disp_master, tangent_2_disp_master, tangent_1_traction,
          tangent_2_traction;

      t_rhs_normal_traction(i) =
          cN * t_contact_normal_tensor(i, j) * t_traction(j);

      tangent_1_disp(i) = t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
                          t_x(j);

      tangent_2_disp(i) = t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) *
                          t_x(j);
      
      tangent_1_traction(i) =
          t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) * t_traction(j);

      tangent_2_traction(i) =
          t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) * t_traction(j);
      
      auto t_field_data = data.getFTensor1FieldData<3>();
      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        const double beta = alpha * (t_base(i) * t_normal(i));
        // t_nf(i) -= beta * tangent_1_disp(i);
        // t_nf(i) -= beta * tangent_2_disp(i);

        // t_nf(i) += cN * beta * tangent_1_traction(i);
        // t_nf(i) += cN * beta * tangent_2_traction(i);
        //////////

        t_nf(i) -= beta * t_x(i);
        t_nf(i) += cN * beta * t_traction(i);

        ++t_nf;
        ++t_base;
        ++t_field_data;
      }


      ++t_x;
      ++t_traction;
      ++t_w;
    }

    CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpPassHdivToMasterVariationX::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPtsSlave().size2();
  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {
    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;

    const double area_m =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto t_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto position_master = getFTensor1FromMat<3>(
        *commonDataSimpleContact->positionAtGaussPtsMasterPtr);
    auto t_x_slave = getFTensor1FromMat<3>(
        *commonDataSimpleContact->positionAtGaussPtsSlavePtr);
    auto t_x_master = getFTensor1FromMat<3>(
        *commonDataSimpleContact->positionAtGaussPtsMasterPtr);

    auto t_X_slave = getFTensor1FromMat<3>(
        *commonDataSimpleContact->meshPositionAtGaussPtsSlavePtr);
    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

    auto t_X_master = getFTensor1FromMat<3>(
        *commonDataSimpleContact->meshPositionAtGaussPtsMasterPtr);

    auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    // auto t_coords = getFTensor1CoordsAtGaussPtsSlave();
    size_t nb_base_functions = data.getN().size2() / 3;
    // auto t_base = data.getFTensor1N<3>();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf{&nf[0], &nf[1],
                                                              &nf[2]};

      auto t_base = data.getFTensor1N<3>(gg, 0);
      const double alpha = t_w * area_m;

      //
      auto t_contact_normal = get_tensor_vec(
          commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

      FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
      t_contact_normal_tensor(i, j) = t_contact_normal(i) * t_contact_normal(j);
      

      auto t_tangent_1_at_gp = get_tensor_vec(
          *(commonDataSimpleContact->tangentOneVectorSlavePtr), 0);

      auto t_tangent_2_at_gp = get_tensor_vec(
          *(commonDataSimpleContact->tangentTwoVectorSlavePtr), 0);

      FTensor::Tensor1<double, 3> t_rhs_constrains;
      const double normal_traction = t_traction(i) * t_contact_normal(i);

      FTensor::Tensor1<double, 3>  t_rhs_tangent_traction,
          t_rhs_normal_traction, t_rhs_displ;

      FTensor::Tensor1<double, 3> tangent_1_disp, tangent_2_disp,
          tangent_1_traction, tangent_2_traction;

      t_rhs_displ(j) = t_x_master(j) - t_X_master(j);
      // t_rhs_displ(2) += 0.1;
      // t_rhs_tangent_traction(i) =
      //     cN * t_contact_tangent_tensor(i, j) * t_traction(j);

      // t_rhs_normal_traction(i) =
      //     cN * t_contact_normal_tensor(i, j) * t_traction(j);

      tangent_1_disp(i) = t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
                          (t_x_slave(j) - t_x_master(j));

      tangent_2_disp(i) = t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) *
                          (t_x_slave(j) - t_x_master(j));

      tangent_1_traction(i) =
          t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) * t_traction(j);

      tangent_2_traction(i) =
          t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) * t_traction(j);

      auto t_field_data = data.getFTensor1FieldData<3>();
      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        const double beta = alpha * (t_base(i) * t_normal(i));

        t_nf(j) -= beta * t_rhs_displ(j);

        ++t_base;

        ++t_nf;
        ++t_field_data;
      }
      for (; bb < nb_base_functions; ++bb)
        ++t_base;

      ++t_x_slave;
      ++t_x_master;
      ++t_traction;
      ++t_X_slave;
      ++t_X_master;
      ++t_w;
      ++gap_ptr;
    }
    CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpPassHdivToMasterNormal::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPtsMaster().size2();
  const size_t nb_dofs = data.getIndices().size();

  if (nb_dofs) {
    // cerr << "rhs " << nb_dofs << "\n";

    const double area_m =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto t_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto t_tangent_1_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr), 0);

    auto t_tangent_2_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr), 0);

    auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    FTensor::Tensor1<double, 3>  tangent_1_traction,
        tangent_2_traction;

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);
    const size_t nb_base_functions = data.getN().size2();
    auto t_w = getFTensor0IntegrationWeightMaster();
    auto t_base = data.getFTensor0N();
    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      // cerr << " traction " << t_traction(j) * t_normal(j) << "\n";
      tangent_1_traction(i) =
          t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) * t_traction(j);

      tangent_2_traction(i) =
          t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) * t_traction(j);

      const double alpha = area_m * t_w;
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf{&nf[0], &nf[1],
                                                              &nf[2]};
      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        ///////
        //Master 11/08/2020
        // t_nf(i) -= alpha * t_base * t_normal(i) * t_traction(j) * t_normal(j);
        // t_nf(i) -= alpha * t_base * tangent_1_traction(i);
        // t_nf(i) -= alpha * t_base * tangent_2_traction(i);

        t_nf(i) -= alpha * t_base * t_traction(i);

        //for augmented
        t_nf(i) -= alpha * t_base * gap_ptr * t_normal(i);

        ///////
        // t_nf(i) -= alpha * t_base  * t_traction(i) ;
        // t_normal(j); t_nf(i) -= alpha * t_base * t_tangent_1_at_gp(i) *
        // t_traction(j) * t_tangent_1_at_gp(j); t_nf(i) -= alpha * t_base *
        // t_tangent_2_at_gp(i) * t_traction(j) * t_tangent_2_at_gp(j);
        // t_nf(i) -= alpha * t_base * t_traction(i);
        ++t_nf;
        ++t_base;
      }
      for (; bb < nb_base_functions; ++bb) {
        ++t_base;
      }
      ++t_traction;
      ++t_w;
      ++gap_ptr;
    }
    CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetMeshPositionAtGaussPtsMaster::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  const int nb_dofs = data.getFieldData().size();

  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  if (type == MBVERTEX) {
    commonDataSimpleContact->meshPositionAtGaussPtsMasterPtr.get()->resize(
        3, nb_gauss_pts, false);

    commonDataSimpleContact->meshPositionAtGaussPtsMasterPtr.get()->clear();
  }

  auto position_master = getFTensor1FromMat<3>(
      *commonDataSimpleContact->meshPositionAtGaussPtsMasterPtr);

  int nb_base_fun_col = data.getFieldData().size() / 3;

  FTensor::Index<'i', 3> i;

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

    FTensor::Tensor1<double *, 3> t_field_data_master(
        &data.getFieldData()[0], &data.getFieldData()[1],
        &data.getFieldData()[2], 3);

    for (int bb = 0; bb != nb_base_fun_col; bb++) {
      position_master(i) += t_base_master * t_field_data_master(i);

      ++t_base_master;
      ++t_field_data_master;
    }
    ++position_master;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetMeshPositionAtGaussPtsSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  const int nb_dofs = data.getFieldData().size();

  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  if (type == MBVERTEX) {
    commonDataSimpleContact->meshPositionAtGaussPtsSlavePtr.get()->resize(
        3, nb_gauss_pts, false);

    commonDataSimpleContact->meshPositionAtGaussPtsSlavePtr.get()->clear();
  }

  auto position_slave = getFTensor1FromMat<3>(
      *commonDataSimpleContact->meshPositionAtGaussPtsSlavePtr);

  int nb_base_fun_col = data.getFieldData().size() / 3;

  FTensor::Index<'i', 3> i;

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));

    FTensor::Tensor1<double *, 3> t_field_data_slave(
        &data.getFieldData()[0], &data.getFieldData()[1],
        &data.getFieldData()[2], 3);

    for (int bb = 0; bb != nb_base_fun_col; bb++) {
      position_slave(i) += t_base_slave * t_field_data_slave(i);

      ++t_base_slave;
      ++t_field_data_slave;
    }
    ++position_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpConstrainDomainLhs_dU::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();
  if (row_nb_dofs && col_nb_dofs) {
    auto t_w = getFTensor0IntegrationWeight();
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();
    transLocMat.resize(col_nb_dofs, row_nb_dofs, false);
    
    size_t nb_base_functions = row_data.getN().size2() / 3;
    // auto t_row_base = row_data.getFTensor1N<3>();
    // auto t_row_diff_base = row_data.getFTensor2DiffN<3, 3>();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      const double alpha = getVolume() * t_w;

      auto t_row_base = row_data.getFTensor1N<3>(gg, 0);
      auto t_row_diff_base = row_data.getFTensor2DiffN<3, 3>(gg, 0);

      size_t rr = 0;
      for (; rr != row_nb_dofs / 3; ++rr) {
        FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_mat_diag{
            &locMat(3 * rr + 0, 0), &locMat(3 * rr + 1, 1),
            &locMat(3 * rr + 2, 2)};
        // FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3> t_mat{
        //     &locMat(3 * rr + 0, 0), &locMat(3 * rr + 0, 1),
        //     &locMat(3 * rr + 0, 2), &locMat(3 * rr + 1, 0),
        //     &locMat(3 * rr + 1, 1), &locMat(3 * rr + 1, 2),
        //     &locMat(3 * rr + 2, 0), &locMat(3 * rr + 2, 1),
        //     &locMat(3 * rr + 2, 2)};

        const double t_row_div_base = t_row_diff_base(0, 0) +
                                      t_row_diff_base(1, 1) +
                                      t_row_diff_base(2, 2);
        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);
        for (size_t cc = 0; cc != col_nb_dofs / 3; ++cc) {
          t_mat_diag(i) +=  alpha * t_row_base(j) * t_col_diff_base(j);
          t_mat_diag(i) +=  alpha * t_row_div_base * t_col_base;
          ++t_col_base;
          ++t_col_diff_base;
          ++t_mat_diag;
          // ++t_mat;
        }
        ++t_row_diff_base;
        ++t_row_base;
      }
      // for (; rr < nb_base_functions; ++rr) {
      //   ++t_row_diff_base;
      //   ++t_row_base;
      // }
      ++t_w;
    }
    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
    noalias(transLocMat) = trans(locMat);
    CHKERR MatSetValues(getSNESB(), col_data, row_data,
                        &*transLocMat.data().begin(), ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpConstrainBoundaryLhs_dU_dlambda_Master::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPtsMaster().size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();
  if (row_nb_dofs && col_nb_dofs) {

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    const double area_m = commonDataSimpleContact->areaSlave;
    //  FTensor::Tensor1<double, 3> t_direction{getDirection()[0],
    //                                                      getDirection()[1]};
    //        FTensor::Tensor1<double, 2> t_normal{-t_direction(1),
    //        t_direction(0)};
    //  const double l = sqrt(t_normal(i) * t_normal(i));
    //  t_normal(i) /= l;
    //      t_direction(i) /= l;
    // auto t_traction =
    //     getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto sign = [](double x) {
      if (x == 0)
        return 0;
      else if (x > 0)
        return 1;
      else
        return -1;
    };

    auto t_contact_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto t_tangent_1_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr), 0);

    auto t_tangent_2_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr), 0);

    // auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    auto t_w = getFTensor0IntegrationWeightMaster();
    // here

    size_t nb_face_functions = col_data.getN().size2() / 3;
    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      const double alpha = t_w * area_m;
      auto t_row_base = row_data.getFTensor0N(gg, 0);
      //  auto t_contact_normal = normal(t_coords, t_disp);
      FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
      t_contact_normal_tensor(i, j) = t_contact_normal(i) * t_contact_normal(j);
      FTensor::Tensor2<double, 3, 3> t_contact_tangent_tensor;
      t_contact_tangent_tensor(i, j) = t_contact_normal_tensor(i, j);
      t_contact_tangent_tensor(0, 0) -= 1;
      t_contact_tangent_tensor(1, 1) -= 1;
      t_contact_tangent_tensor(2, 2) -= 1;
      // auto diff_constrain = diff_constrains_dgap(
      //     gap0(t_coords, t_contact_normal), gap(t_disp, t_contact_normal),
      //     normal_traction(t_traction, t_contact_normal));
      // const double normal_traction = t_traction(i) * t_contact_normal(i);
      // plus/minus depends on master slave

      //  const double diff_constrain_d_slave =
      //      -(1 - sign(gap_ptr - cN * normal_traction)) / 2.;

      // const double diff_constrain_d_slave =
      //     (-1 - sign(-normal_traction - cN * gap_ptr)) * cN / 2.;

      // t_nf(i) += alpha * t_base * t_normal(i) * t_traction(j) *
      // t_normal(j);

      size_t rr = 0;
      for (; rr != row_nb_dofs / 3; ++rr) {
        FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3> t_mat(
            &locMat(3 * rr + 0, 0), &locMat(3 * rr + 0, 1),
            &locMat(3 * rr + 0, 2), &locMat(3 * rr + 1, 0),
            &locMat(3 * rr + 1, 1), &locMat(3 * rr + 1, 2),
            &locMat(3 * rr + 2, 0), &locMat(3 * rr + 2, 1),
            &locMat(3 * rr + 2, 2));
        auto t_col_base = col_data.getFTensor1N<3>(gg, 0);
        size_t cc = 0;
        for (; cc != col_nb_dofs / 3; ++cc) {
          const double col_base = t_col_base(i) * t_contact_normal(i);
          const double col_base_t_1 = t_col_base(i) * t_tangent_1_at_gp(i);
          const double col_base_t_2 = t_col_base(i) * t_tangent_2_at_gp(i);
          const double beta = alpha * col_base * t_row_base;
          const double beta_t_1 = alpha * col_base_t_1 * t_row_base;
          const double beta_t_2 = alpha * col_base_t_2 * t_row_base;

          //Master 11/08/2020
          // t_mat(i, j) -= beta * t_contact_normal_tensor(i, j);
          t_mat(0, 0) -= beta;
          t_mat(1, 1) -= beta;
          t_mat(2, 2) -= beta;

          // t_mat(i, j) -= beta * t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j);
          // t_mat(i, j) -= beta * t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j);

          // cerr << "WTF  ~~~~~   " << t_mat << "\n";

          ++t_col_base;
          ++t_mat;
        }
        for (; cc < nb_face_functions; ++cc)
          ++t_col_base;

        ++t_row_base;
      }

      // ++t_traction;
      ++t_w;
      // ++gap_ptr;
    }
    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}


MoFEMErrorCode
SimpleContactProblem::OpConstrainBoundaryLhs_dU_Slave_tied::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPtsSlave().size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();
  if (row_nb_dofs && col_nb_dofs) {

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    const double area_m = commonDataSimpleContact->areaSlave;
    //  FTensor::Tensor1<double, 3> t_direction{getDirection()[0],
    //                                                      getDirection()[1]};
    //        FTensor::Tensor1<double, 2> t_normal{-t_direction(1),
    //        t_direction(0)};
    //  const double l = sqrt(t_normal(i) * t_normal(i));
    //  t_normal(i) /= l;
    //      t_direction(i) /= l;
    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto sign = [](double x) {
      if (x == 0)
        return 0;
      else if (x > 0)
        return 1;
      else
        return -1;
    };

    auto t_contact_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    auto t_w = getFTensor0IntegrationWeightSlave();
    auto t_row_base = row_data.getFTensor1N<3>();
    size_t nb_face_functions = row_data.getN().size2() / 3;
    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    auto t_tangent_1_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr), 0);

    auto t_tangent_2_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr), 0);

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      const double alpha = t_w * area_m;
      //  auto t_contact_normal = normal(t_coords, t_disp);
      FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
      t_contact_normal_tensor(i, j) = t_contact_normal(i) * t_contact_normal(j);
      FTensor::Tensor2<double, 3, 3> t_contact_tangent_tensor;
      t_contact_tangent_tensor(i, j) = t_contact_normal_tensor(i, j);
      t_contact_tangent_tensor(0, 0) -= 1;
      t_contact_tangent_tensor(1, 1) -= 1;
      t_contact_tangent_tensor(2, 2) -= 1;
      // auto diff_constrain = diff_constrains_dgap(
      //     gap0(t_coords, t_contact_normal), gap(t_disp, t_contact_normal),
      //     normal_traction(t_traction, t_contact_normal));
      const double normal_traction = t_traction(i) * t_contact_normal(i);
      // plus/minus depends on master slave

      const double diff_constrain_d_slave =
          (-1 + sign(gap_ptr - cN * normal_traction)) / 2.;

      //  const double diff_constrain_d_slave =
      //      (1 - sign(-normal_traction - cN * gap_ptr)) * cN / 2.;

      //  const double diff_constrain_d_slave =
      //      (-1 - sign(normal_traction - cN * gap_ptr)) * cN / 2.;

      // t_nf(i) += alpha * t_base * t_normal(i) * t_traction(j) *
      // t_normal(j);

      size_t rr = 0;
      for (; rr != row_nb_dofs / 3; ++rr) {

        FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_mat_diag{
            &locMat(3 * rr + 0, 0), &locMat(3 * rr + 1, 1),
            &locMat(3 * rr + 2, 2)};
        FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3> t_mat(
            &locMat(3 * rr + 0, 0), &locMat(3 * rr + 0, 1),
            &locMat(3 * rr + 0, 2), &locMat(3 * rr + 1, 0),
            &locMat(3 * rr + 1, 1), &locMat(3 * rr + 1, 2),
            &locMat(3 * rr + 2, 0), &locMat(3 * rr + 2, 1),
            &locMat(3 * rr + 2, 2));
        const double row_base_n = t_row_base(i) * t_contact_normal(i);
        const double row_base_t_1 = t_row_base(i) * t_tangent_1_at_gp(i);
        const double row_base_t_2 = t_row_base(i) * t_tangent_2_at_gp(i);

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != col_nb_dofs / 3; ++cc) {
          const double beta_n = alpha * row_base_n * t_col_base;
          const double beta_t_1 = alpha * row_base_n * t_col_base;
          const double beta_t_2 = alpha * row_base_n * t_col_base;

          // t_mat(0, 0) += beta_n;

          // t_mat(1, 1) += beta_n;

          // t_mat(2, 2) += beta_n;
          t_mat_diag(i) += beta_n;
          t_mat(i, j) -=
              beta_t_1 * t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j);
          t_mat(i, j) -=
              beta_t_2 * t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j);

          //  t_mat(i, j) += beta_n * t_contact_tangent_tensor(i, j);

          // t_mat(i, j) += beta_n * t_tangent_1_at_gp(i) *
          // t_tangent_1_at_gp(j); t_mat(i, j) += beta_n *
          // t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j);

          ++t_col_base;
          ++t_mat;
          ++t_mat_diag;
        }
        ++t_row_base;
      }
      for (; rr < nb_face_functions; ++rr)
        ++t_row_base;

      ++t_traction;
      ++t_w;
      ++gap_ptr;
    }
    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpConstrainBoundaryLhs_dU_Master_tied::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPtsSlave().size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();
  if (row_nb_dofs && col_nb_dofs) {

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    const double area_m = commonDataSimpleContact->areaSlave;
    //  FTensor::Tensor1<double, 3> t_direction{getDirection()[0],
    //                                                      getDirection()[1]};
    //        FTensor::Tensor1<double, 2> t_normal{-t_direction(1),
    //        t_direction(0)};
    //  const double l = sqrt(t_normal(i) * t_normal(i));
    //  t_normal(i) /= l;
    //      t_direction(i) /= l;
    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto sign = [](double x) {
      if (x == 0)
        return 0;
      else if (x > 0)
        return 1;
      else
        return -1;
    };

    auto t_contact_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    auto t_w = getFTensor0IntegrationWeightSlave();
    auto t_row_base = row_data.getFTensor1N<3>();
    size_t nb_face_functions = row_data.getN().size2() / 3;
    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    auto t_tangent_1_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr), 0);

    auto t_tangent_2_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr), 0);

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      const double alpha = t_w * area_m;
      //  auto t_contact_normal = normal(t_coords, t_disp);
      FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
      t_contact_normal_tensor(i, j) = t_contact_normal(i) * t_contact_normal(j);
      FTensor::Tensor2<double, 3, 3> t_contact_tangent_tensor;
      t_contact_tangent_tensor(i, j) = t_contact_normal_tensor(i, j);
      t_contact_tangent_tensor(0, 0) -= 1;
      t_contact_tangent_tensor(1, 1) -= 1;
      t_contact_tangent_tensor(2, 2) -= 1;
      // auto diff_constrain = diff_constrains_dgap(
      //     gap0(t_coords, t_contact_normal), gap(t_disp, t_contact_normal),
      //     normal_traction(t_traction, t_contact_normal));
      const double normal_traction = t_traction(i) * t_contact_normal(i);
      // plus/minus depends on master slave

      const double diff_constrain_d_slave =
          (-1 + sign(gap_ptr - cN * normal_traction)) / 2.;

      //  const double diff_constrain_d_slave =
      //      (1 - sign(-normal_traction - cN * gap_ptr)) * cN / 2.;

      //  const double diff_constrain_d_slave =
      //      (-1 - sign(normal_traction - cN * gap_ptr)) * cN / 2.;

      // t_nf(i) += alpha * t_base * t_normal(i) * t_traction(j) *
      // t_normal(j);

      size_t rr = 0;
      for (; rr != row_nb_dofs / 3; ++rr) {

        FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_mat_diag{
            &locMat(3 * rr + 0, 0), &locMat(3 * rr + 1, 1),
            &locMat(3 * rr + 2, 2)};
        FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3> t_mat(
            &locMat(3 * rr + 0, 0), &locMat(3 * rr + 0, 1),
            &locMat(3 * rr + 0, 2), &locMat(3 * rr + 1, 0),
            &locMat(3 * rr + 1, 1), &locMat(3 * rr + 1, 2),
            &locMat(3 * rr + 2, 0), &locMat(3 * rr + 2, 1),
            &locMat(3 * rr + 2, 2));
        const double row_base_n = t_row_base(i) * t_contact_normal(i);
        const double row_base_t_1 = t_row_base(i) * t_tangent_1_at_gp(i);
        const double row_base_t_2 = t_row_base(i) * t_tangent_2_at_gp(i);

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != col_nb_dofs / 3; ++cc) {
          const double beta_n = alpha * row_base_n * t_col_base;
          const double beta_t_1 = alpha * row_base_t_1 * t_col_base;
          const double beta_t_2 = alpha * row_base_t_2 * t_col_base;

          // t_mat(0, 0) += beta_n;

          // t_mat(1, 1) += beta_n;

          // t_mat(2, 2) += beta_n;
          // t_mat_diag(i) -= beta_n;
          // t_mat(i, j) -=
          //     beta_t_1 * t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j);
          // t_mat(i, j) -=
          //     beta_t_2 * t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j);

          //07/08/2020
          t_mat(i, j) -= beta_n * t_contact_normal_tensor(i, j);
          // t_mat(i, j) -= beta_n * t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j);
          // t_mat(i, j) -= beta_n * t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j);

          // t_mat(i, j) += beta_n * t_tangent_1_at_gp(i) *
          // t_tangent_1_at_gp(j); t_mat(i, j) += beta_n *
          // t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j);

          ++t_col_base;
          ++t_mat;
          ++t_mat_diag;
        }
        ++t_row_base;
      }
      for (; rr < nb_face_functions; ++rr)
        ++t_row_base;

      ++t_traction;
      ++t_w;
      ++gap_ptr;
    }
    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpConstrainBoundaryLhs_dxdTraction::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPtsSlave().size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();
  //  cerr << "row_side " << row_side << " col_nb_dofs " << col_nb_dofs << "\n";

  //  cerr << "row_type " << row_type << " col_type " << col_type << "\n";

  if (row_nb_dofs && col_nb_dofs) {

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    const double area_m =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto sign = [](double x) {
      if (x == 0)
        return 0;
      else if (x > 0)
        return 1;
      else
        return -1;
    };

    auto t_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();
    // FTensor::Tensor1<double, 2> t_direction{getDirection()[0],
    //                                                 getDirection()[1]};
    // FTensor::Tensor1<double, 2> t_normal{-t_direction(1), t_direction(0)};
    // const double l = sqrt(t_normal(i) * t_normal(i));
    // t_normal(i) /= l;
    // t_direction(i) /= l;

    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

    auto t_tangent_1_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr), 0);

    auto t_tangent_2_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr), 0);

    auto t_w = getFTensor0IntegrationWeightSlave();
    auto t_row_base = row_data.getFTensor1N<3>();
    size_t nb_face_functions = row_data.getN().size2() / 3;
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      //  cerr << "Traction " << t_traction << "\n";

      const double alpha = t_w * area_m;
      //  auto t_contact_normal = normal(t_coords, t_disp);
      FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
      t_contact_normal_tensor(i, j) = t_normal(i) * t_normal(j);

      // const double diff_traction = diff_constrains_dtraction(
      // gap0(t_coords, t_contact_normal), gap(t_disp, t_contact_normal),
      // normal_traction(t_traction, t_contact_normal));
      const double normal_traction = t_traction(i) * t_normal(i);

      //  t_rhs_constrains(i) =
      //      (-normal_traction - (-normal_traction - cN * gap_ptr +
      //                           std::abs(-normal_traction - cN * gap_ptr)) /
      //                              2.) *
      //      t_contact_normal(i);

      // const double diff_constrain_d_lambda =
      //     cN * (1 + sign(gap_ptr - cN * normal_traction)) / 2.;

      const double diff_constrain_d_lambda =
          cN * (1 + sign(gap_ptr + cN * normal_traction));

      //  const double diff_constrain_d_lambda =
      //      (1 + sign(-cN * normal_traction + gap_ptr)) / 2.;

      //  const double diff_constrain_d_lambda =
      //      (1 - sign(normal_traction - cN * gap_ptr)) / 2.;
      size_t rr = 0;
      for (; rr != row_nb_dofs / 3; ++rr) {
        FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3> t_mat(
            &locMat(3 * rr + 0, 0), &locMat(3 * rr + 0, 1),
            &locMat(3 * rr + 0, 2), &locMat(3 * rr + 1, 0),
            &locMat(3 * rr + 1, 1), &locMat(3 * rr + 1, 2),
            &locMat(3 * rr + 2, 0), &locMat(3 * rr + 2, 1),
            &locMat(3 * rr + 2, 2));

        const double row_base = t_row_base(i) * t_normal(i);
        auto t_col_base = col_data.getFTensor1N<3>(gg, 0);
        for (size_t cc = 0; cc != col_nb_dofs / 3; ++cc) {
          const double col_base = t_col_base(i) * t_normal(i);
          const double beta = alpha * row_base * col_base;
        //do not forget
          // t_mat(i, j) +=
          //     (beta * diff_constrain_d_lambda) * t_contact_normal_tensor(i, j);




          //  t_mat(i, j) += 2 * beta * cN * t_contact_tangent_tensor(i, j);

          // Here 28/05
          // t_mat(i, j) +=
          //     beta * cN * t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j);

          // t_mat(i, j) +=
          //     beta * cN * t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j);
          // Until here 28/05
          t_mat(i, j) +=
              cN * beta * t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j);

          t_mat(i, j) += cN * beta * t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j);

          //  t_mat(i, j) += beta * cN * t_contact_normal_tensor(i, j);

          //  t_mat(i, j) += beta * t_contact_normal_tensor(i, j);

          ++t_col_base;
          ++t_mat;
        }
        ++t_row_base;
      }
      for (; rr < nb_face_functions; ++rr)
        ++t_row_base;

      ++t_traction;
      ++t_w;
      ++gap_ptr;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpConstrainBoundaryLhs_dxdTractionForFace::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();

  if (row_nb_dofs && col_nb_dofs) {

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    const double area_m = getMeasure(); // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto t_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorFacePtr.get()[0], 0);

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

    auto t_tangent_1_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorFacePtr), 0);

    auto t_tangent_2_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorFacePtr), 0);

    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor1N<3>();
    size_t nb_face_functions = row_data.getN().size2() / 3;
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      //  cerr << "Traction " << t_traction << "\n";

      const double alpha = t_w * area_m;
      //  auto t_contact_normal = normal(t_coords, t_disp);
      FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
      t_contact_normal_tensor(i, j) = t_normal(i) * t_normal(j);

      const double normal_traction = t_traction(i) * t_normal(i);

      size_t rr = 0;
      for (; rr != row_nb_dofs / 3; ++rr) {
        FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3> t_mat(
            &locMat(3 * rr + 0, 0), &locMat(3 * rr + 0, 1),
            &locMat(3 * rr + 0, 2), &locMat(3 * rr + 1, 0),
            &locMat(3 * rr + 1, 1), &locMat(3 * rr + 1, 2),
            &locMat(3 * rr + 2, 0), &locMat(3 * rr + 2, 1),
            &locMat(3 * rr + 2, 2));

        const double row_base = t_row_base(i) * t_normal(i);
        auto t_col_base = col_data.getFTensor1N<3>(gg, 0);
        for (size_t cc = 0; cc != col_nb_dofs / 3; ++cc) {
          const double col_base = t_col_base(i) * t_normal(i);
          const double beta = alpha * row_base * col_base;
          // Until here 28/05

          t_mat(i, j) +=
              cN * beta * t_normal(i) * t_normal(j);

          t_mat(i, j) +=
              cN * beta * t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j);

          t_mat(i, j) +=
              cN * beta * t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j);


          ++t_col_base;
          ++t_mat;
        }
        ++t_row_base;
      }
      for (; rr < nb_face_functions; ++rr)
        ++t_row_base;

      ++t_traction;
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpConstrainBoundaryLhs_dU_Slave::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPtsSlave().size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();
  if (row_nb_dofs && col_nb_dofs) {

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    const double area_m = commonDataSimpleContact->areaSlave;
    //  FTensor::Tensor1<double, 3> t_direction{getDirection()[0],
    //                                                      getDirection()[1]};
    //        FTensor::Tensor1<double, 2> t_normal{-t_direction(1),
    //        t_direction(0)};
    //  const double l = sqrt(t_normal(i) * t_normal(i));
    //  t_normal(i) /= l;
    //      t_direction(i) /= l;
    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto sign = [](double x) {
      if (x == 0)
        return 0;
      else if (x > 0)
        return 1;
      else
        return -1;
    };

    auto t_contact_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    auto t_w = getFTensor0IntegrationWeightSlave();
    auto t_row_base = row_data.getFTensor1N<3>();
    size_t nb_face_functions = row_data.getN().size2() / 3;
    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    auto t_tangent_1_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr), 0);

    auto t_tangent_2_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr), 0);

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      const double alpha = t_w * area_m;
      //  auto t_contact_normal = normal(t_coords, t_disp);
      FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
      t_contact_normal_tensor(i, j) = t_contact_normal(i) * t_contact_normal(j);
      FTensor::Tensor2<double, 3, 3> t_contact_tangent_tensor;
      t_contact_tangent_tensor(i, j) = t_contact_normal_tensor(i, j);
      t_contact_tangent_tensor(0, 0) -= 1;
      t_contact_tangent_tensor(1, 1) -= 1;
      t_contact_tangent_tensor(2, 2) -= 1;
      // auto diff_constrain = diff_constrains_dgap(
      //     gap0(t_coords, t_contact_normal), gap(t_disp, t_contact_normal),
      //     normal_traction(t_traction, t_contact_normal));
      const double normal_traction = t_traction(i) * t_contact_normal(i);
      // plus/minus depends on master slave

      const double diff_constrain_d_slave =
          -(1 + sign(gap_ptr + cN * normal_traction));

      // const double diff_constrain_d_slave =
      //     -(/*-1*/ +sign(gap_ptr - cN * normal_traction)) /*/ 2.*/;

      //  const double diff_constrain_d_slave =
      //      (1 - sign(-normal_traction - cN * gap_ptr)) * cN / 2.;

      //  const double diff_constrain_d_slave =
      //      (-1 - sign(normal_traction - cN * gap_ptr)) * cN / 2.;

      // t_nf(i) += alpha * t_base * t_normal(i) * t_traction(j) * t_normal(j);

      size_t rr = 0;
      for (; rr != row_nb_dofs / 3; ++rr) {
        FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3> t_mat(
            &locMat(3 * rr + 0, 0), &locMat(3 * rr + 0, 1),
            &locMat(3 * rr + 0, 2), &locMat(3 * rr + 1, 0),
            &locMat(3 * rr + 1, 1), &locMat(3 * rr + 1, 2),
            &locMat(3 * rr + 2, 0), &locMat(3 * rr + 2, 1),
            &locMat(3 * rr + 2, 2));
        const double row_base = t_row_base(i) * t_contact_normal(i);
        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != col_nb_dofs / 3; ++cc) {
          const double beta = alpha * row_base * t_col_base;
          //here
          // t_mat(i, j) +=
          //     (beta * diff_constrain_d_slave) * t_contact_normal_tensor(i, j);

          //  t_mat(i, j) += beta * t_contact_tangent_tensor(i, j);
          //11/08/2020
          t_mat(i, j) -= beta * t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j);
          t_mat(i, j) -= beta * t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j);
          // t_mat(i, j) += beta * t_contact_normal(i) * t_contact_normal(j);

          // t_mat(0, 0) += beta;
          // t_mat(1, 1) += beta;
          // t_mat(2, 2) += beta;

          ++t_col_base;
          ++t_mat;
        }
        ++t_row_base;
      }
      for (; rr < nb_face_functions; ++rr)
        ++t_row_base;

      ++t_traction;
      ++t_w;
      ++gap_ptr;
    }
    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpConstrainBoundaryLhs_dU_ForFace::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();
  if (row_nb_dofs && col_nb_dofs) {

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    const double area_m = getMeasure();
     auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto t_contact_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorFacePtr.get()[0], 0);

    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor1N<3>();
    size_t nb_face_functions = row_data.getN().size2() / 3;
    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    auto t_tangent_1_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorFacePtr), 0);

    auto t_tangent_2_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorFacePtr), 0);

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      const double alpha = t_w * area_m;
      //  auto t_contact_normal = normal(t_coords, t_disp);
      FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
      t_contact_normal_tensor(i, j) = t_contact_normal(i) * t_contact_normal(j);
      FTensor::Tensor2<double, 3, 3> t_contact_tangent_tensor;
      
      size_t rr = 0;
      for (; rr != row_nb_dofs / 3; ++rr) {
        FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3> t_mat(
            &locMat(3 * rr + 0, 0), &locMat(3 * rr + 0, 1),
            &locMat(3 * rr + 0, 2), &locMat(3 * rr + 1, 0),
            &locMat(3 * rr + 1, 1), &locMat(3 * rr + 1, 2),
            &locMat(3 * rr + 2, 0), &locMat(3 * rr + 2, 1),
            &locMat(3 * rr + 2, 2));
        const double row_base = t_row_base(i) * t_contact_normal(i);
        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != col_nb_dofs / 3; ++cc) {
          const double beta = alpha * row_base * t_col_base;

          // 11/08/2020
          t_mat(i, j) -= beta * t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j);
          t_mat(i, j) -= beta * t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j);
          t_mat(i, j) -= beta * t_contact_normal(i) * t_contact_normal(j);

          ++t_col_base;
          ++t_mat;
        }
        ++t_row_base;
      }
      for (; rr < nb_face_functions; ++rr)
        ++t_row_base;

      ++t_traction;
      ++t_w;
    }
    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpConstrainBoundaryLhs_dU_Master::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPtsMaster().size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();
  if (row_nb_dofs && col_nb_dofs) {

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    transLocMat.resize(row_nb_dofs, col_nb_dofs, false);
    transLocMat.clear();

    const double area_m = commonDataSimpleContact->areaSlave;
    //  FTensor::Tensor1<double, 3> t_direction{getDirection()[0],
    //                                                      getDirection()[1]};
    //        FTensor::Tensor1<double, 2> t_normal{-t_direction(1),
    //        t_direction(0)};
    //  const double l = sqrt(t_normal(i) * t_normal(i));
    //  t_normal(i) /= l;
    //      t_direction(i) /= l;
    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto sign = [](double x) {
      if (x == 0)
        return 0;
      else if (x > 0)
        return 1;
      else
        return -1;
    };

    auto t_contact_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    auto t_w = getFTensor0IntegrationWeightMaster();
    auto t_row_base = row_data.getFTensor1N<3>();
    size_t nb_face_functions = row_data.getN().size2() / 3;

    auto t_tangent_1_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr), 0);

    auto t_tangent_2_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr), 0);

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      const double alpha = t_w * area_m;
      //  auto t_contact_normal = normal(t_coords, t_disp);
      FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
      t_contact_normal_tensor(i, j) = t_contact_normal(i) * t_contact_normal(j);
      FTensor::Tensor2<double, 3, 3> t_contact_tangent_tensor;
      t_contact_tangent_tensor(i, j) = t_contact_normal_tensor(i, j);
      t_contact_tangent_tensor(0, 0) -= 1;
      t_contact_tangent_tensor(1, 1) -= 1;
      t_contact_tangent_tensor(2, 2) -= 1;
      // auto diff_constrain = diff_constrains_dgap(
      //     gap0(t_coords, t_contact_normal), gap(t_disp, t_contact_normal),
      //     normal_traction(t_traction, t_contact_normal));
      const double normal_traction = t_traction(i) * t_contact_normal(i);
      // plus/minus depends on master slave

      const double diff_constrain_d_master =
          (1 + sign(gap_ptr + cN * normal_traction));

      // const double diff_constrain_d_master =
      //     -(1 - sign(gap_ptr - cN * normal_traction)) /*/ 2.*/;

      // const double diff_constrain_d_master =
      //     (-1 + sign(-normal_traction - cN * gap_ptr)) * cN / 2.;

      // const double diff_constrain_d_master =
      //     (1 + sign(normal_traction - cN * gap_ptr))  / 2.;

      size_t rr = 0;
      for (; rr != row_nb_dofs / 3; ++rr) {
        FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3> t_mat(
            &locMat(3 * rr + 0, 0), &locMat(3 * rr + 0, 1),
            &locMat(3 * rr + 0, 2), &locMat(3 * rr + 1, 0),
            &locMat(3 * rr + 1, 1), &locMat(3 * rr + 1, 2),
            &locMat(3 * rr + 2, 0), &locMat(3 * rr + 2, 1),
            &locMat(3 * rr + 2, 2));

        FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3> t_mat_trans(
            &transLocMat(3 * rr + 0, 0), &transLocMat(3 * rr + 0, 1),
            &transLocMat(3 * rr + 0, 2), &transLocMat(3 * rr + 1, 0),
            &transLocMat(3 * rr + 1, 1), &transLocMat(3 * rr + 1, 2),
            &transLocMat(3 * rr + 2, 0), &transLocMat(3 * rr + 2, 1),
            &transLocMat(3 * rr + 2, 2));
        const double row_base = t_row_base(i) * t_contact_normal(i);
        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != col_nb_dofs / 3; ++cc) {
          const double beta = alpha * row_base * t_col_base;
          ///
          // t_mat(i, j) +=
          //     (beta * diff_constrain_d_master) * t_contact_normal_tensor(i, j);
          ///
          t_mat(i, j) -= beta * t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j);
          t_mat(i, j) -= beta * t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j);

          // t_mat(i, j) += beta  *
          //                t_contact_normal_tensor(i, j);

          // t_mat(0, 0) -= beta;
          // t_mat(1, 1) -= beta;
          // t_mat(2, 2) -= beta;

          // t_mat_trans(i, j) -= beta * t_contact_normal_tensor(i, j);

          // cerr << "beta" << beta <<"\n";
          ++t_col_base;
          ++t_mat;
          ++t_mat_trans;
        }
        ++t_row_base;
      }
      // for (; rr < nb_face_functions; ++rr)
      //   ++t_row_base;

      ++t_traction;
      ++t_w;
      ++gap_ptr;
    }
    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);

    MatrixDouble help_matrix;
    help_matrix.resize(col_nb_dofs, row_nb_dofs, false);
    help_matrix.clear();
    noalias(help_matrix) = trans(transLocMat);
    CHKERR MatSetValues(getSNESB(), col_data, row_data,
                        &*help_matrix.data().begin(), ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactOperatorsRhs(
    boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrang_field_name) {
  MoFEMFunctionBegin;

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalSlave(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsMaster(field_name,
                                        common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsSlave(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetGapSlave(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                     common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpCalContactTractionOnSlave(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(new OpCalIntCompFunSlave(
      lagrang_field_name, common_data_simple_contact, cnValue));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setMasterForceOperatorsRhs(
    boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrang_field_name) {
  MoFEMFunctionBegin;

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalSlave(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalMaster(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                     common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpCalContactTractionOnMaster(field_name, common_data_simple_contact));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactOperatorsLhs(
    boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrang_field_name) {
  MoFEMFunctionBegin;

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalSlave(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalMaster(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsMaster(field_name,
                                        common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsSlave(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetGapSlave(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                     common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalContactTractionOverLambdaSlaveSlave(
          field_name, lagrang_field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalDerIntCompFunOverLambdaSlaveSlave(
          lagrang_field_name, common_data_simple_contact, cnValue));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalDerIntCompFunOverSpatPosSlaveMaster(
          field_name, lagrang_field_name, common_data_simple_contact, cnValue));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalDerIntCompFunOverSpatPosSlaveSlave(
          field_name, lagrang_field_name, common_data_simple_contact, cnValue));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setMasterForceOperatorsLhs(
    boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrang_field_name) {
  MoFEMFunctionBegin;

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalSlave(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalMaster(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                     common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalContactTractionOverLambdaMasterSlave(
          field_name, lagrang_field_name, common_data_simple_contact));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactOperatorsLhs(
    boost::shared_ptr<ConvectMasterContactElement> fe_lhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrang_field_name) {
  MoFEMFunctionBegin;
  CHKERR setContactOperatorsLhs(
      boost::dynamic_pointer_cast<SimpleContactElement>(fe_lhs_simple_contact),
      common_data_simple_contact, field_name, lagrang_field_name);

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalculateGradPositionXi(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpLhsConvectIntegrationPtsConstrainMasterGap(
          lagrang_field_name, field_name, common_data_simple_contact, cnValue,
          ContactOp::FACESLAVESLAVE,
          fe_lhs_simple_contact->getConvectPtr()->getDiffKsiSpatialSlave()));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpLhsConvectIntegrationPtsConstrainMasterGap(
          lagrang_field_name, field_name, common_data_simple_contact, cnValue,
          ContactOp::FACESLAVEMASTER,
          fe_lhs_simple_contact->getConvectPtr()->getDiffKsiSpatialMaster()));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setMasterForceOperatorsLhs(
    boost::shared_ptr<ConvectSlaveContactElement> fe_lhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrang_field_name) {
  MoFEMFunctionBegin;

  CHKERR setMasterForceOperatorsLhs(
      boost::dynamic_pointer_cast<SimpleContactElement>(fe_lhs_simple_contact),
      common_data_simple_contact, field_name, lagrang_field_name);

  fe_lhs_simple_contact->getOpPtrVector().push_back(new OpCalculateGradLambdaXi(
      lagrang_field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpLhsConvectIntegrationPtsContactTraction(
          field_name, field_name, common_data_simple_contact,
          ContactOp::FACEMASTERSLAVE,
          fe_lhs_simple_contact->getConvectPtr()->getDiffKsiSpatialSlave()));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpLhsConvectIntegrationPtsContactTraction(
          field_name, field_name, common_data_simple_contact,
          ContactOp::FACEMASTERMASTER,
          fe_lhs_simple_contact->getConvectPtr()->getDiffKsiSpatialMaster()));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactOperatorsForPostProc(
    boost::shared_ptr<SimpleContactElement> fe_post_proc_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    MoFEM::Interface &m_field, string field_name, string lagrang_field_name,
    moab::Interface &moab_out, bool lagrange_field) {
  MoFEMFunctionBegin;

  map<int, SimpleContactPrismsData>::iterator sit =
      setOfSimpleContactPrism.begin();
  for (; sit != setOfSimpleContactPrism.end(); sit++) {

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpGetNormalMaster(field_name, common_data_simple_contact));

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpGetNormalSlave(field_name, common_data_simple_contact));

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpGetPositionAtGaussPtsMaster(field_name,
                                          common_data_simple_contact));

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpGetPositionAtGaussPtsSlave(field_name,
                                         common_data_simple_contact));

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpGetGapSlave(field_name, common_data_simple_contact));

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                       common_data_simple_contact));

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpLagGapProdGaussPtsSlave(lagrang_field_name,
                                      common_data_simple_contact));

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpMakeVtkSlave(m_field, field_name, common_data_simple_contact,
                           moab_out, lagrange_field));
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactOperatorsForPostProcHdiv(
    boost::shared_ptr<SimpleContactElement> fe_post_proc_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    MoFEM::Interface &m_field, string field_name, string lagrang_field_name,
    moab::Interface &moab_out, bool lagrange_field) {
  MoFEMFunctionBegin;

  map<int, SimpleContactPrismsData>::iterator sit =
      setOfSimpleContactPrism.begin();
  for (; sit != setOfSimpleContactPrism.end(); sit++) {

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpGetNormalMaster(field_name, common_data_simple_contact));

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpGetNormalSlave(field_name, common_data_simple_contact));

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpGetPositionAtGaussPtsMaster(field_name,
                                          common_data_simple_contact));

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpGetPositionAtGaussPtsSlave(field_name,
                                         common_data_simple_contact));

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpGetGapSlave(field_name, common_data_simple_contact));

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpConstrainBoundaryTraction(lagrang_field_name,
                                        common_data_simple_contact));

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpGetFromHdivLagMulAtGaussPtsSlave(lagrang_field_name,
                                       common_data_simple_contact));

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpLagGapProdGaussPtsSlave(field_name, common_data_simple_contact));

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpMakeVtkSlave(m_field, field_name, common_data_simple_contact,
                           moab_out, lagrange_field));
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setPostProcContactOperatorsHdiv(
    boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc_contact_ptr,
    const std::string field_name, const std::string lagrang_field_name,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact) {
  MoFEMFunctionBegin;

  post_proc_contact_ptr->getOpPtrVector().push_back(
      new OpGetNormalForTri(field_name, common_data_simple_contact));

  post_proc_contact_ptr->getOpPtrVector().push_back(
      new OpConstrainBoundaryTractionPostProc(lagrang_field_name,
                                              common_data_simple_contact));

  post_proc_contact_ptr->getOpPtrVector().push_back(
      new OpGetFromHdivLagMulAtGaussPtsSlavePostProc(
          lagrang_field_name, common_data_simple_contact));

  post_proc_contact_ptr->getOpPtrVector().push_back(
      new OpPostProcContactContinuous(
          field_name, field_name, post_proc_contact_ptr->postProcMesh,
          post_proc_contact_ptr->mapGaussPts, common_data_simple_contact));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setPostProcContactOperators(
    boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc_contact_ptr,
    const std::string field_name, const std::string lagrang_field_name,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact) {
  MoFEMFunctionBegin;

  post_proc_contact_ptr->getOpPtrVector().push_back(
      new OpCalLagrangeMultPostProc(lagrang_field_name,
                                    common_data_simple_contact));

  post_proc_contact_ptr->getOpPtrVector().push_back(
      new OpPostProcContactContinuous(
          lagrang_field_name, field_name, post_proc_contact_ptr->postProcMesh,
          post_proc_contact_ptr->mapGaussPts, common_data_simple_contact));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactOperatorsRhsOperatorsHdiv3DSurface(
    boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrang_field_name) {
  MoFEMFunctionBegin;

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalSlave(field_name, common_data_simple_contact));
  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalMaster(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetOrthonormalTangents(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsMaster(field_name,
                                        common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetMeshPositionAtGaussPtsMaster("MESH_NODE_POSITIONS",
                                            common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetMeshPositionAtGaussPtsSlave("MESH_NODE_POSITIONS",
                                           common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsSlave(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetGapSlave(field_name, common_data_simple_contact));

  // tets until here

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpConstrainBoundaryTraction(lagrang_field_name,
                                      common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(new OpConstrainBoundaryRhs(
      lagrang_field_name, common_data_simple_contact, cnValue));

  //=====
  //   fe_rhs_simple_contact->getOpPtrVector().push_back(
  //       new OpConstrainBoundaryRhsMaster(lagrang_field_name,
  //                                  common_data_simple_contact, cnValue));
  //====

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpPassHdivToMasterNormal(field_name, common_data_simple_contact));

  // fe_rhs_simple_contact->getOpPtrVector().push_back(
  //     new OpPassHdivToMasterVariationX(lagrang_field_name,
  //                                      common_data_simple_contact));


  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpPassHdivToSlaveNormal(field_name,
      common_data_simple_contact, cnValue));

  //   fe_rhs_simple_contact->getOpPtrVector().push_back(
  //       new OpPassHdivToSlaveVariationX(lagrang_field_name,
  //                                        common_data_simple_contact));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::setContactOperatorsRhsOperatorsHdiv3DForFace(
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_rhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrange_field_name) {
  MoFEMFunctionBegin;
  cerr << "0000000000000000\n";

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalAndTangentsFace(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<3>(
          field_name, common_data_simple_contact->xFaceAtPts));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpConstrainBoundaryTractionForFace(lagrange_field_name,
                                      common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpConstrainBoundaryRhsForFace(lagrange_field_name,
                                        common_data_simple_contact, cnValue));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::setContactOperatorsLhsOperatorsHdiv3DForFace(
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_lhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrange_field_name) {
  MoFEMFunctionBegin;

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalAndTangentsFace(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpConstrainBoundaryTractionForFace(lagrange_field_name,
                                             common_data_simple_contact));

  ////
  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpConstrainBoundaryLhs_dxdTractionForFace(
          lagrange_field_name, lagrange_field_name, common_data_simple_contact,
          cnValue));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpConstrainBoundaryLhs_dU_ForFace(lagrange_field_name, field_name,
                                          common_data_simple_contact, cnValue));


  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpPassHdivToSlaveNormal::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPtsSlave().size2();
  const size_t nb_dofs = data.getIndices().size();

  if (nb_dofs) {
    // cerr << "rhs " << nb_dofs << "\n";

    const double area_m =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto t_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);
    const size_t nb_base_functions = data.getN().size2();
    auto t_w = getFTensor0IntegrationWeightSlave();
    auto t_base = data.getFTensor0N();
    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));
    FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
    t_contact_normal_tensor(i, j) = t_normal(i) * t_normal(j);
    FTensor::Tensor2<double, 3, 3> t_contact_tangent_tensor;
    t_contact_tangent_tensor(i, j) = t_contact_normal_tensor(i, j);
    t_contact_tangent_tensor(0, 0) -= 1;
    t_contact_tangent_tensor(1, 1) -= 1;
    t_contact_tangent_tensor(2, 2) -= 1;

    FTensor::Tensor1<double, 3> tangent_1_traction, tangent_2_traction;

    auto t_tangent_1_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr), 0);

    auto t_tangent_2_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr), 0);

    auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      
      tangent_1_traction(i) =
          t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) * t_traction(j);

      tangent_2_traction(i) =
          t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) * t_traction(j);
      // cerr << " tangent_1_traction " << tangent_1_traction << "\n";
      // cerr << " tangent_2_traction " << tangent_2_traction << "\n";

      const double alpha = area_m * t_w;
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf{&nf[0], &nf[1],
                                                              &nf[2]};
      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {

        // t_nf(i) += alpha * t_base * t_normal(i) * t_traction(j) *
        // t_normal(j);
        t_nf(i) += alpha * t_base * tangent_1_traction(i);
        t_nf(i) += alpha * t_base * tangent_2_traction(i);

        // t_nf(i) -= alpha * t_base * t_tangent_1_at_gp(i) *
        //            t_tangent_1_at_gp(j) * t_traction(j);

        // t_nf(i) -= alpha * t_base * t_tangent_2_at_gp(i) *
        //            t_tangent_2_at_gp(j) * t_traction(j);

        // t_nf(i) += alpha * t_base * cN * t_contact_tangent_tensor(i,j) *
        // t_traction(j);
        ++t_nf;
        ++t_base;
      }
      for (; bb < nb_base_functions; ++bb) {
        ++t_base;
      }
      ++t_traction;
      ++t_w;
      ++gap_ptr;
    }
    // cerr << "t_nf    ";

    // for (int ii = 0; ii != nb_dofs; ++ii) {
    //   cerr << "  " << nf[ii];
    // }
    // cerr << "\n";
    CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpPassHdivToSlaveVariationX::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPtsSlave().size2();
  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {
    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    // cerr << " GAUSS !! " << nb_gauss_pts << "\n";
    // // FTensor::Tensor1<double, 2> t_direction{getDirection()[0],
    //                                                 getDirection()[1]};
    // FTensor::Tensor1<double, 2> t_normal{-t_direction(1),
    // t_direction(0)}; const double l = sqrt(t_normal(i) *
    // t_normal(i)); t_normal(i) /= l; t_direction(i) /= l;

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;

    const double area_m =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto t_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto t_w = getFTensor0IntegrationWeightSlave();
    // auto t_disp = getFTensor1FromMat<2>(*(commonDataPtr->contactDispPtr));

    auto position_master = getFTensor1FromMat<3>(
        *commonDataSimpleContact->positionAtGaussPtsMasterPtr);
    auto t_x_slave = getFTensor1FromMat<3>(
        *commonDataSimpleContact->positionAtGaussPtsSlavePtr);
    auto t_x_master = getFTensor1FromMat<3>(
        *commonDataSimpleContact->positionAtGaussPtsMasterPtr);

    auto t_X_slave = getFTensor1FromMat<3>(
        *commonDataSimpleContact->meshPositionAtGaussPtsSlavePtr);
    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

    auto t_X_master = getFTensor1FromMat<3>(
        *commonDataSimpleContact->meshPositionAtGaussPtsMasterPtr);

    auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    // auto t_coords = getFTensor1CoordsAtGaussPtsSlave();
    size_t nb_base_functions = data.getN().size2() / 3;
    // auto t_base = data.getFTensor1N<3>();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf{&nf[0], &nf[1],
                                                              &nf[2]};

      auto t_base = data.getFTensor1N<3>(gg, 0);
      const double alpha = t_w * area_m;

      //
      auto t_contact_normal = get_tensor_vec(
          commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

      FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
      t_contact_normal_tensor(i, j) = t_contact_normal(i) * t_contact_normal(j);
      FTensor::Tensor2<double, 3, 3> t_contact_tangent_tensor;
      t_contact_tangent_tensor(i, j) = t_contact_normal_tensor(i, j);
      t_contact_tangent_tensor(0, 0) -= 1;
      t_contact_tangent_tensor(1, 1) -= 1;
      t_contact_tangent_tensor(2, 2) -= 1;

      auto t_tangent_1_at_gp = get_tensor_vec(
          *(commonDataSimpleContact->tangentOneVectorSlavePtr), 0);

      auto t_tangent_2_at_gp = get_tensor_vec(
          *(commonDataSimpleContact->tangentTwoVectorSlavePtr), 0);

      FTensor::Tensor1<double, 3> t_rhs_constrains;
      const double normal_traction = t_traction(i) * t_contact_normal(i);
      // t_rhs_constrains(i) =
      //     (gap_ptr - (gap_ptr - cN * normal_traction +
      //                 std::abs(gap_ptr - cN * normal_traction)) /
      //                    2.) *
      //     t_contact_normal(i);

      //  t_rhs_constrains(i) =
      //      (t_contact_normal(j) * t_x_master(j) + cN * normal_traction -
      //      std::abs(gap_ptr - cN * normal_traction)) * t_contact_normal(i);

      //  t_rhs_constrains(i) =
      //      (gap_ptr + cN * normal_traction -
      //                  std::abs(gap_ptr - cN * normal_traction)) *
      //      t_contact_normal(i);

      //  cerr << "t_rhs_constrains     " << t_rhs_constrains << "  gap_ptr  "
      //       << gap_ptr << "   normal_traction  " << normal_traction << "\n";
      //  double check = gap_ptr - cN * normal_traction;

      //  cerr << " ASDASDASD    " << check << " gap " << gap_ptr << "\n";

      //  t_rhs_constrains(i) =
      //      (-normal_traction - (-normal_traction - cN * gap_ptr +
      //                           std::abs(-normal_traction - cN * gap_ptr)) /
      //                              2.) *
      //      t_contact_normal(i);

      //  cerr << "C fun RHS vector " << t_rhs_constrains << "\n";

      // t_rhs_constrains(i) =
      //     (normal_traction - (normal_traction - cN * gap_ptr +
      //                          std::abs(normal_traction - cN * gap_ptr)) /
      //                             2.) *
      //     t_contact_normal(i);

      //  t_rhs_constrains(i) =
      //      t_contact_normal(i) *
      //     constrian(gap0(t_coords, t_contact_normal),
      //                    gap(t_disp, t_contact_normal),
      //                    normal_traction(t_traction, t_contact_normal));
      FTensor::Tensor1<double, 3> t_rhs_tangent_disp, t_rhs_tangent_traction,
          t_rhs_normal_traction, t_rhs_displ;

      FTensor::Tensor1<double, 3> tangent_1_disp, tangent_2_disp,
          tangent_1_traction, tangent_2_traction;

      t_rhs_tangent_disp(i) =
          t_contact_tangent_tensor(i, j) * (t_x_slave(j) - t_x_master(j));
      t_rhs_displ(j) = t_x_slave(j) - t_X_slave(j);
      // t_rhs_tangent_traction(i) =
      //     cN * t_contact_tangent_tensor(i, j) * t_traction(j);

      // t_rhs_normal_traction(i) =
      //     cN * t_contact_normal_tensor(i, j) * t_traction(j);

      tangent_1_disp(i) = t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
                          (t_x_slave(j) - t_x_master(j));

      tangent_2_disp(i) = t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) *
                          (t_x_slave(j) - t_x_master(j));

      tangent_1_traction(i) =
          t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) * t_traction(j);

      tangent_2_traction(i) =
          t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) * t_traction(j);

      //  cerr << " t_traction " << t_traction << "\n";
      //  cerr << " t_rhs_tangent_traction " << t_rhs_tangent_traction <<
      //  "\n";
      // cerr << " t_rhs_tangent_disp " << t_rhs_tangent_disp << "\n";
      // cerr << " normal_traction " << normal_traction << "\n";
      //  cerr << " t_x_slave " << t_x_slave << " t_X_slave " << t_X_slave
      //  <<
      //  "\n"  ;

      auto t_field_data = data.getFTensor1FieldData<3>();
      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        const double beta = alpha * (t_base(i) * t_normal(i));

        t_nf(j) += beta * t_rhs_displ(j);
        t_nf(j) -=
            beta * t_contact_normal_tensor(j, i) * (t_x_master(i) - t_X_master(i));
        // t_nf(i) += beta * t_rhs_constrains(i);
        // t_nf(i) -= beta * t_rhs_tangent_disp(i);
        // // Ignatios previous
        // //  t_nf(i) += beta * t_rhs_tangent_disp(i);
        // //  t_nf(i) += beta * t_rhs_tangent_traction(i);
        // //  t_nf(i) += alpha * cN *
        // //             (t_contact_tangent_tensor(i, j) * t_base(k) *
        // //             t_normal(k)) * t_traction(j);

        // t_nf(i) += beta * tangent_1_disp(i);
        // t_nf(i) += beta * tangent_2_disp(i);

        // t_nf(i) += beta * tangent_1_traction(i);
        // t_nf(i) += beta * tangent_2_traction(i);

        // // This is for tangent 28/05
        // t_nf(i) += alpha * cN *
        //            (t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
        //             t_traction(j) * t_base(k) * t_normal(k));

        // t_nf(i) += alpha * cN *
        //            (t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) *
        //             t_traction(j) * t_base(k) * t_normal(k));

        // t_nf(i) += beta * t_rhs_normal_traction(i);

        //  cerr << "beta " << beta << "\n";

        ++t_base;

        ++t_nf;
        ++t_field_data;
      }
      // for (; bb < nb_base_functions; ++bb)
      //   ++t_base;

      ++t_x_slave;
      ++t_x_master;
      ++t_traction;
      ++t_X_slave;
      ++t_X_master;
      // ++t_coords;
      ++t_w;
      ++gap_ptr;
    }
    // cerr << "nf     ";
    // for (int kk = 0; kk != nb_dofs; ++kk) {
    //   cerr << " " <<nf[kk];
    // }
    // cerr << "\n";
    CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpConstrainBoundaryLhs_dU_dlambda_Slave::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPtsMaster().size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();
  if (row_nb_dofs && col_nb_dofs) {

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    const double area_m = commonDataSimpleContact->areaSlave;
    //  FTensor::Tensor1<double, 3> t_direction{getDirection()[0],
    //                                                      getDirection()[1]};
    //        FTensor::Tensor1<double, 2> t_normal{-t_direction(1),
    //        t_direction(0)};
    //  const double l = sqrt(t_normal(i) * t_normal(i));
    //  t_normal(i) /= l;
    //      t_direction(i) /= l;
    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto sign = [](double x) {
      if (x == 0)
        return 0;
      else if (x > 0)
        return 1;
      else
        return -1;
    };

    auto t_contact_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    auto t_w = getFTensor0IntegrationWeightMaster();
    // here

    size_t nb_face_functions = col_data.getN().size2() / 3;
    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    auto t_tangent_1_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr), 0);

    auto t_tangent_2_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr), 0);

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      const double alpha = t_w * area_m;
      auto t_row_base = row_data.getFTensor0N(gg, 0);
      //  auto t_contact_normal = normal(t_coords, t_disp);
      FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
      t_contact_normal_tensor(i, j) = t_contact_normal(i) * t_contact_normal(j);
      FTensor::Tensor2<double, 3, 3> t_contact_tangent_tensor;
      t_contact_tangent_tensor(i, j) = t_contact_normal_tensor(i, j);
      t_contact_tangent_tensor(0, 0) -= 1;
      t_contact_tangent_tensor(1, 1) -= 1;
      t_contact_tangent_tensor(2, 2) -= 1;
      // auto diff_constrain = diff_constrains_dgap(
      //     gap0(t_coords, t_contact_normal), gap(t_disp, t_contact_normal),
      //     normal_traction(t_traction, t_contact_normal));
      const double normal_traction = t_traction(i) * t_contact_normal(i);
      // plus/minus depends on master slave

      //  const double diff_constrain_d_slave =
      //      -(1 - sign(gap_ptr - cN * normal_traction)) / 2.;

      const double diff_constrain_d_slave =
          (-1 - sign(-normal_traction - cN * gap_ptr)) * cN / 2.;

      // t_nf(i) += alpha * t_base * t_normal(i) * t_traction(j) *
      // t_normal(j);

      size_t rr = 0;
      for (; rr != row_nb_dofs / 3; ++rr) {

        FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_mat_diag{
            &locMat(3 * rr + 0, 0), &locMat(3 * rr + 1, 1),
            &locMat(3 * rr + 2, 2)};

        FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3> t_mat(
            &locMat(3 * rr + 0, 0), &locMat(3 * rr + 0, 1),
            &locMat(3 * rr + 0, 2), &locMat(3 * rr + 1, 0),
            &locMat(3 * rr + 1, 1), &locMat(3 * rr + 1, 2),
            &locMat(3 * rr + 2, 0), &locMat(3 * rr + 2, 1),
            &locMat(3 * rr + 2, 2));
        auto t_col_base = col_data.getFTensor1N<3>(gg, 0);
        size_t cc = 0;
        for (; cc != col_nb_dofs / 3; ++cc) {
          const double col_base = t_col_base(i) * t_contact_normal(i);
          const double beta = alpha * col_base * t_row_base;
          // t_mat(i, j) += beta * t_contact_normal_tensor(i, j);

          t_mat(i, j) += beta * t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j);
          t_mat(i, j) += beta * t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j);

          // t_mat(0, 0) += beta;

          // t_mat(1, 1) += beta;
          // t_mat(2, 2) += beta;
          // t_mat_diag(i) += beta;
          // t_mat(i, j) += beta * cN * t_contact_tangent_tensor(i, j);
          // cerr << "WTF  ~~~~~   " << t_mat << "\n";

          ++t_col_base;
          ++t_mat_diag;
          ++t_mat;
        }
        // for (; cc < nb_face_functions; ++cc)
        //   ++t_col_base;

        ++t_row_base;
      }

      ++t_traction;
      ++t_w;
      ++gap_ptr;
    }
    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}


MoFEMErrorCode SimpleContactProblem::setContactOperatorsRhsOperatorsHdiv3DVolume(
    boost::shared_ptr<VolumeElementForcesAndSourcesCore> fe_hdiv_rhs_slave_tet,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrang_field_name) {
  MoFEMFunctionBegin;
  cerr << "XXXXXXXXXXXXXXXXXXXXXXXX\n";
  fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
      new OpCalculateHVecTensorField<3, 3>(
          lagrang_field_name,
          common_data_simple_contact->lagMatMultAtGaussPtsPtr));

  fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
      new OpCalculateHVecTensorDivergence<3, 3>(
          lagrang_field_name,
          common_data_simple_contact->contactLagrangeHdivDivergencePtr));

  fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<3, 3>(
          field_name, common_data_simple_contact->mGradPtr));

  fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<3>(field_name,
                                          common_data_simple_contact->xAtPts));

  fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<3>(
          "MESH_NODE_POSITIONS", common_data_simple_contact->xInitAtPts));

  fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
      new OpConstrainDomainRhs(lagrang_field_name, common_data_simple_contact));

  fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
      new OpInternalDomainContactRhs(field_name, common_data_simple_contact));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactOperatorsLhsOperatorsHdiv3DSurface(
    boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrange_field_name) {
  MoFEMFunctionBegin;

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalSlave(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalMaster(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetOrthonormalTangents(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsMaster(field_name,
                                        common_data_simple_contact));

  //   fe_lhs_simple_contact->getOpPtrVector().push_back(
  //       new OpGetMeshPositionAtGaussPtsMaster(field_name,
  //                                         common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsSlave(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetGapSlave(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpConstrainBoundaryTraction(lagrange_field_name,
                                      common_data_simple_contact));

    // fe_lhs_simple_contact->getOpPtrVector().push_back(
    //     new OpConstrainBoundaryLhs_dTractionMaster(
    //         lagrange_field_name, lagrange_field_name,
    //         common_data_simple_contact, cnValue));

  ////
  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpConstrainBoundaryLhs_dxdTraction(
          lagrange_field_name, lagrange_field_name, common_data_simple_contact,
          cnValue));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpConstrainBoundaryLhs_dU_Slave(lagrange_field_name, field_name,
                                          common_data_simple_contact, cnValue));
  ////

  // fe_lhs_simple_contact->getOpPtrVector().push_back(
  //     new OpConstrainBoundaryLhs_dU_Master(lagrange_field_name, field_name,
  //                                          common_data_simple_contact,
  //                                          cnValue));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpConstrainBoundaryLhs_dU_dlambda_Master(
          field_name, lagrange_field_name, common_data_simple_contact,
          cnValue));

  /// until here

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpConstrainBoundaryLhs_dU_dlambda_Slave(
          field_name, lagrange_field_name, common_data_simple_contact,
          cnValue));

  // fe_lhs_simple_contact->getOpPtrVector().push_back(
  //     new OpConstrainBoundaryLhs_dU_Slave_tied(lagrange_field_name, field_name,
  //                                              common_data_simple_contact,
  //                                              cnValue));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpConstrainBoundaryLhs_dU_Master_tied(lagrange_field_name, field_name,
                                               common_data_simple_contact,
                                               cnValue));

  //   fe_lhs_simple_contact->getOpPtrVector().push_back(
  //       new OpConstrainBoundaryLhs_dU_SlaveSlave_tied(
  //           lagrange_field_name, field_name,
  //           common_data_simple_contact, cnValue));*/

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactOperatorsLhsOperatorsHdiv3DVolume(
    boost::shared_ptr<VolumeElementForcesAndSourcesCore> fe_hdiv_lhs_slave_tet,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrange_field_name) {
  MoFEMFunctionBegin;

  fe_hdiv_lhs_slave_tet->getOpPtrVector().push_back(new OpConstrainDomainLhs_dU(
      lagrange_field_name, field_name, common_data_simple_contact));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalculateGradLambdaXi::doWork(int side, EntityType type,
                                                      EntData &data) {
  MoFEMFunctionBegin;
  const int nb_dofs = data.getFieldData().size();
  const int nb_integration_pts = getGaussPtsSlave().size2();
  auto &xi_grad_mat = *(commonDataSimpleContact->gradKsiLambdaAtGaussPtsPtr);
  xi_grad_mat.resize(2, nb_integration_pts, false);
  if (type == MBVERTEX)
    xi_grad_mat.clear();

  FTensor::Index<'I', 2> I;

  if (nb_dofs) {

    auto t_diff_lambda_xi = getFTensor1FromMat<2>(xi_grad_mat);

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      auto t_data = data.getFTensor0FieldData();
      auto t_diff_base = data.getFTensor1DiffN<2>(gg, 0);
      for (size_t bb = 0; bb != nb_dofs; ++bb) {
        t_diff_lambda_xi(I) += t_diff_base(I) * t_data;
        ++t_data;
        ++t_diff_base;
      }
      ++t_diff_lambda_xi;
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpLhsConvectIntegrationPtsContactTraction::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const int nb_row_dofs = row_data.getIndices().size();
  const int nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && (nb_col_dofs && col_type == MBVERTEX)) {

    const int nb_gauss_pts = getGaussPtsSlave().size2();
    int nb_base_fun_row = nb_row_dofs / 3;
    int nb_base_fun_col = nb_col_dofs / 3;

    matLhs.resize(nb_row_dofs, nb_col_dofs, false);
    matLhs.clear();

    const double area_s =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'I', 2> I;

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n[0], &n[1], &n[2]);
    };

    auto t_const_unit_n =
        get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr));
    auto t_diff_lambda_xi = getFTensor1FromMat<2>(
        *(commonDataSimpleContact->gradKsiLambdaAtGaussPtsPtr));
    auto t_w = getFTensor0IntegrationWeightSlave();

    auto get_diff_ksi = [](auto &m, auto gg) {
      return FTensor::Tensor2<FTensor::PackPtr<double *, 1>, 2, 3>(
          &m(0, gg), &m(1, gg), &m(2, gg), &m(3, gg), &m(4, gg), &m(5, gg));
    };

    auto t_base_row = row_data.getFTensor0N();

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      double val_s = t_w * area_s;
      auto t_base_row = row_data.getFTensor0N(gg, 0);

      for (int rr = 0; rr != nb_base_fun_row; ++rr) {

        FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3> t_mat{
            &matLhs(3 * rr + 0, 0), &matLhs(3 * rr + 0, 1),
            &matLhs(3 * rr + 0, 2),

            &matLhs(3 * rr + 1, 0), &matLhs(3 * rr + 1, 1),
            &matLhs(3 * rr + 1, 2),

            &matLhs(3 * rr + 2, 0), &matLhs(3 * rr + 2, 1),
            &matLhs(3 * rr + 2, 2)};

        auto t_diff_convect = get_diff_ksi(*diffConvect, 3 * gg);

        for (int cc = 0; cc != nb_base_fun_col; ++cc) {
          t_mat(i, j) -= val_s * t_base_row * t_const_unit_n(i) *
                         (t_diff_lambda_xi(I) * t_diff_convect(I, j));

          ++t_diff_convect;
          ++t_mat;
        }

        ++t_base_row;
      }

      ++t_diff_lambda_xi;
      ++t_w;
    } // for gauss points

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*matLhs.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalculateGradPositionXi::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;
  const int nb_dofs = data.getFieldData().size();
  const int nb_integration_pts = getGaussPtsSlave().size2();
  auto &xi_grad_mat = *(commonDataSimpleContact->gradKsiPositionAtGaussPtsPtr);
  xi_grad_mat.resize(6, nb_integration_pts, false);
  if (type == MBVERTEX)
    xi_grad_mat.clear();

  FTensor::Index<'i', 3> i;
  FTensor::Index<'I', 2> I;

  if (nb_dofs) {

    auto t_grad_pos_xi = getFTensor2FromMat<3, 2>(xi_grad_mat);

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      auto t_data = data.getFTensor1FieldData<3>();
      auto t_diff_base = data.getFTensor1DiffN<2>(gg, 0);
      for (size_t bb = 0; bb != nb_dofs / 3; ++bb) {
        t_grad_pos_xi(i, I) += t_diff_base(I) * t_data(i);
        ++t_data;
        ++t_diff_base;
      }
      ++t_grad_pos_xi;
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpLhsConvectIntegrationPtsConstrainMasterGap::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col && col_type == MBVERTEX) {

    const int nb_gauss_pts = row_data.getN().size1();
    int nb_base_fun_row = nb_row;
    int nb_base_fun_col = nb_col / 3;

    const double area_slave =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    matLhs.resize(nb_row, nb_col, false);
    matLhs.clear();

    auto get_diff_ksi = [](auto &m, auto gg) {
      return FTensor::Tensor2<FTensor::PackPtr<double *, 1>, 2, 3>(
          &m(0, gg), &m(1, gg), &m(2, gg), &m(3, gg), &m(4, gg), &m(5, gg));
    };

    auto get_tensor_from_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'I', 2> I;

    auto t_const_unit_n =
        get_tensor_from_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

    auto t_lagrange_slave =
        getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
    auto t_gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);
    auto &xi_grad_mat =
        *(commonDataSimpleContact->gradKsiPositionAtGaussPtsPtr);
    auto t_grad = getFTensor2FromMat<3, 2>(xi_grad_mat);

    auto t_w = getFTensor0IntegrationWeightSlave();
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      const double val_m = SimpleContactProblem::ConstrainFunction(
                               cN, t_gap_gp, t_lagrange_slave) *
                           t_w * area_slave;
      const double val_diff_m_l = SimpleContactProblem::ConstrainFunction_dl(
                                      cN, t_gap_gp, t_lagrange_slave) *
                                  t_w * area_slave;
      const double val_diff_m_g = SimpleContactProblem::ConstrainFunction_dg(
                                      cN, t_gap_gp, t_lagrange_slave) *
                                  t_w * area_slave;

      auto t_base_lambda = row_data.getFTensor0N(gg, 0);
      auto t_base_diff_lambda = row_data.getFTensor1DiffN<2>(gg, 0);

      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_mat{
          &matLhs(0, 0), &matLhs(0, 1), &matLhs(0, 2)};

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

        auto t_base_diff_disp = col_data.getFTensor1DiffN<2>(gg, 0);
        auto t_diff_convect = get_diff_ksi(*diffConvect, 3 * gg);

        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

          t_mat(i) += t_base_lambda * val_diff_m_g * t_const_unit_n(j) *
                      t_grad(j, I) * t_diff_convect(I, i);

          ++t_base_diff_disp;
          ++t_diff_convect;
          ++t_mat;
        }

        ++t_base_lambda;
        ++t_base_diff_lambda;
      }

      ++t_gap_gp;
      ++t_lagrange_slave;
      ++t_grad;
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*matLhs.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}
