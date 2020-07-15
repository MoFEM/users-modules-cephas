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
    CHKERR ContactEle::setDefaultGaussPts(2 * order);
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

MoFEMErrorCode SimpleContactProblem::OpGetOrthonormalTangents::doWork(
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

  commonDataSimpleContact->tangentOneVectorSlavePtr->resize(3, false);
  commonDataSimpleContact->tangentOneVectorSlavePtr->clear();

  commonDataSimpleContact->tangentTwoVectorSlavePtr->resize(3, false);
  commonDataSimpleContact->tangentTwoVectorSlavePtr->clear();

  auto t_normal =
      get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

  const double *tangent_one_slave_ptr = &getTangentSlaveOne()[0];
  const double *tangent_two_slave_ptr = &getTangentSlaveTwo()[0];

  auto t_tangent_1 =
      get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr));

  auto t_tangent_2 =
      get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr));

  for (int ii = 0; ii != 3; ++ii) {
    t_tangent_1(ii) = tangent_one_slave_ptr[ii];
    // t_tangent_2(ii) = tangent_two_slave_ptr[ii];
  }

  const double l_tan_1 = sqrt(t_tangent_1(i) * t_tangent_1(i));

  // const double l_tan_2 = sqrt(t_tangent_2(i) * t_tangent_2(i));

  t_tangent_1(i) = t_tangent_1(i) / l_tan_1;

  t_tangent_2(k) = FTensor::levi_civita(i, j, k) * t_normal(i) * t_tangent_1(j);
  // cerr << "t_normal   " << t_normal << "\n";
  // cerr << "t_tangent_1   " << t_tangent_1 << "\n";
  // cerr << "t_tangent_2   " << t_tangent_2 << "\n";

  // t_tangent_2(i) = t_tangent_2(i) / l_tan_2;

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

MoFEMErrorCode
SimpleContactProblem::OpGetPreviousPositionAtGaussPtsSlave::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;
  const int nb_dofs = data.getFieldData().size();

  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();
  if (type == MBVERTEX) {
    commonDataSimpleContact->prevPositionAtGaussPtsSlavePtr.get()->resize(
        3, nb_gauss_pts, false);
    commonDataSimpleContact->prevPositionAtGaussPtsSlavePtr.get()->clear();
  }

  auto t_prev_position_slave = getFTensor1FromMat<3>(
      *commonDataSimpleContact->prevPositionAtGaussPtsSlavePtr);

  int nb_base_fun_col = data.getFieldData().size() / 3;

  FTensor::Index<'i', 3> i;

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));

    FTensor::Tensor1<double *, 3> t_field_data_slave(
        &data.getFieldData()[0], &data.getFieldData()[1],
        &data.getFieldData()[2], 3); // in-between

    for (int bb = 0; bb != nb_base_fun_col; ++bb) {
      t_prev_position_slave(i) += t_base_slave * t_field_data_slave(i);

      ++t_base_slave;
      ++t_field_data_slave;
    }
    ++t_prev_position_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpGetPreviousPositionAtGaussPtsMaster::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;
  const int nb_dofs = data.getFieldData().size();

  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  if (type == MBVERTEX) {
    commonDataSimpleContact->prevPositionAtGaussPtsMasterPtr.get()->resize(
        3, nb_gauss_pts, false);

    commonDataSimpleContact->prevPositionAtGaussPtsMasterPtr.get()->clear();
  }

  auto t_prev_position_master = getFTensor1FromMat<3>(
      *commonDataSimpleContact->prevPositionAtGaussPtsMasterPtr);

  int nb_base_fun_col = data.getFieldData().size() / 3;

  FTensor::Index<'i', 3> i;

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

    FTensor::Tensor1<double *, 3> t_field_data_master(
        &data.getFieldData()[0], &data.getFieldData()[1],
        &data.getFieldData()[2], 3);

    for (int bb = 0; bb != nb_base_fun_col; ++bb) {
      t_prev_position_master(i) += t_base_master * t_field_data_master(i);

      ++t_base_master;
      ++t_field_data_master;
    }
    ++t_prev_position_master;
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

MoFEMErrorCode SimpleContactProblem::OpGetTangentGapVelocitySlave::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  const int nb_gauss_pts = data.getN().size1();

  commonDataSimpleContact->tangentGapPtr.get()->resize(2, nb_gauss_pts,
                                                         false);
  commonDataSimpleContact->tangentGapPtr.get()->clear();

  FTensor::Index<'i', 3> i;

  auto t_position_master_gp = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsMasterPtr);

  auto t_position_slave_gp = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsSlavePtr);

  auto t_prev_position_master_gp = getFTensor1FromMat<3>(
      *commonDataSimpleContact->prevPositionAtGaussPtsMasterPtr);

  auto t_prev_position_slave_gp = getFTensor1FromMat<3>(
      *commonDataSimpleContact->prevPositionAtGaussPtsSlavePtr);

  auto t_tangent_gap_ptr =
      getFTensor1FromMat<2>(*commonDataSimpleContact->tangentGapPtr);

  auto t_tangent_1_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr));

  auto t_tangent_2_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr));

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    t_tangent_gap_ptr(0) +=
        t_tangent_1_at_gp(i) *
        (t_position_slave_gp(i) - t_position_master_gp(i) -
         (t_prev_position_slave_gp(i) - t_prev_position_master_gp(i)));

    t_tangent_gap_ptr(1) +=
        t_tangent_2_at_gp(i) *
        (t_position_slave_gp(i) - t_position_master_gp(i) -
         -(t_prev_position_slave_gp(i) - t_prev_position_master_gp(i)));

    ++t_position_slave_gp;
    ++t_position_master_gp;
    ++t_prev_position_slave_gp;
    ++t_prev_position_master_gp;
    ++t_tangent_gap_ptr;
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
    commonDataSimpleContact->allLambdasPtr.get()->resize(3, nb_gauss_pts, false);
    commonDataSimpleContact->allLambdasPtr.get()->clear();

    commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->resize(nb_gauss_pts);
    commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->clear();
  }

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;

  int nb_base_fun_row = data.getFieldData().size() / 3;

  auto t_lagrange_slave_3 =
      getFTensor1FromMat<3>(*commonDataSimpleContact->allLambdasPtr);

  auto t_lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto t_normal =
      get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    FTensor::Tensor0<double *> t_base_lambda(&data.getN()(gg, 0));

    FTensor::Tensor1<double *, 3> t_field_data_slave(
        &data.getFieldData()[0], &data.getFieldData()[1],
        &data.getFieldData()[2], 3);

    for (int bb = 0; bb != nb_base_fun_row; ++bb) {
      t_lagrange_slave_3(i) += t_base_lambda * t_field_data_slave(i);
      t_lagrange_slave += t_normal(i) * t_base_lambda * t_field_data_slave(i);
      ++t_base_lambda;
      ++t_field_data_slave;
    }
    // cerr << "Normal"
    //      << "     " << t_lagrange_slave << "\n";

    ++t_lagrange_slave_3;
    ++t_lagrange_slave;
  }

    MoFEMFunctionReturn(0);
  }

MoFEMErrorCode
SimpleContactProblem::OpGetTangentLagrange::doWork(int side, EntityType type,
                                                   EntData &data) {
  MoFEMFunctionBegin;
  const int nb_dofs = data.getFieldData().size();

  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  if (type == MBVERTEX) {
    commonDataSimpleContact->tangentLambdasPtr.get()->resize(2, nb_gauss_pts,
                                                               false);

    commonDataSimpleContact->tangentLambdasPtr.get()->clear();
  }

  auto t_tangent_lagrange =
      getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

  int nb_base_fun_col = data.getFieldData().size() / 3;

  FTensor::Index<'i', 3> i;

  auto t_lagrange_slave_3 =
      getFTensor1FromMat<3>(*commonDataSimpleContact->allLambdasPtr);

  auto t_tangent_1_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr));

  auto t_tangent_2_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr));

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    // FTensor::Tensor0<double *> t_base_lag_slave(&data.getN()(gg, 0));

    // FTensor::Tensor1<double *, 3> t_field_data_slave(
    //     &data.getFieldData()[0], &data.getFieldData()[1],
    //     &data.getFieldData()[2], 3);

    // for (int bb = 0; bb != nb_base_fun_col; ++bb) {
    t_tangent_lagrange(0) += t_tangent_1_at_gp(i) * t_lagrange_slave_3(i);
    t_tangent_lagrange(1) += t_tangent_2_at_gp(i) * t_lagrange_slave_3(i);
    
    // cerr << t_tangent_lagrange(0) << "     " << t_tangent_lagrange(1) << "\n";
    
    // ++t_base_lag_slave;
    // ++t_field_data_slave;
    // }
    ++t_tangent_lagrange;
    ++t_lagrange_slave_3;
  } 

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetAugmentedLambdaSlave::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  const int nb_gauss_pts = data.getN().size1();

  commonDataSimpleContact->augmentedLambdasPtr.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->augmentedLambdasPtr.get()->clear();

  FTensor::Index<'i', 3> i;

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  auto t_gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto t_lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    t_aug_lambda_ptr += t_lagrange_slave + cN * t_gap_ptr;
    ++t_aug_lambda_ptr;
    ++t_lagrange_slave;
    ++t_gap_ptr;
  } // for gauss points

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetTangentAugmentedLambdaSlave::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;
  const int nb_dofs = data.getFieldData().size();

  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  commonDataSimpleContact->tangentAugmentedLambdasPtr.get()->resize(
      2, nb_gauss_pts, false);

  commonDataSimpleContact->tangentAugmentedLambdasPtr.get()->clear();
  commonDataSimpleContact->normAugTangentLambdasPtr.get()->resize(
      nb_gauss_pts);
  commonDataSimpleContact->normAugTangentLambdasPtr.get()->clear();

  auto t_tangent_augmented_lagrange = getFTensor1FromMat<2>(
      *commonDataSimpleContact->tangentAugmentedLambdasPtr);

  auto t_tangent_lagrange =
      getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

  auto t_tangent_gap_ptr =
      getFTensor1FromMat<2>(*commonDataSimpleContact->tangentGapPtr);

  auto t_norm_tan_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->normAugTangentLambdasPtr);

  FTensor::Index<'i', 2> i;

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    t_tangent_augmented_lagrange(i) +=
        t_tangent_lagrange(i) + cNtAngentPtr * t_tangent_gap_ptr(i);

    t_norm_tan_aug_lambda_ptr =
        sqrt(t_tangent_augmented_lagrange(i) * t_tangent_augmented_lagrange(i));
    ++t_tangent_lagrange;
    ++t_tangent_augmented_lagrange;
    ++t_tangent_gap_ptr;
    ++t_norm_tan_aug_lambda_ptr;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalContactAugmentedTractionOverLambdaMasterSlave::
    doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
           EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {
    const int nb_gauss_pts = row_data.getN().size1();

    int nb_base_fun_row = row_data.getFieldData().size() / 3;
    int nb_base_fun_col = col_data.getFieldData().size() / 3;

    const double area_master =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 3, 3>(
          &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2),
          &m(r + 1, c + 0), &m(r + 1, c + 1), &m(r + 1, c + 2),
          &m(r + 2, c + 0), &m(r + 2, c + 1), &m(r + 2, c + 2));
    };

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
    NN.clear();

    auto const_unit_n = get_tensor_vec(
        *(commonDataSimpleContact->normalVectorSlavePtr.get()));

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    VectorDouble unit_vector;
    unit_vector.resize(3, false);
    unit_vector(0) = unit_vector(1) = unit_vector(2) = 1.;

    auto t_unit = get_tensor_vec(unit_vector);
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
        ++t_w;
        ++t_aug_lambda_ptr;
        continue;
      }
      double val_m = t_w * area_master;
      auto t_base_master = row_data.getFTensor0N(gg, 0);

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

        auto t_base_lambda = col_data.getFTensor0N(gg, 0);
        const double m = val_m * t_base_master;
        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

          auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);
          const double n = m * t_base_lambda;
          t_assemble_m(i, j) += n * const_unit_n(i) * const_unit_n(j);
          // ++t_assemble_m;
          ++t_base_lambda; // update cols slave
        }
        ++t_base_master; // update rows master
      }
      ++t_w;
      ++t_aug_lambda_ptr;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalContactAugmentedTractionOverLambdaSlaveSlave::doWork(
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
    int nb_base_fun_col = col_data.getFieldData().size() / 3;

    const double area_slave =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 3, 3>(
          &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2),
          &m(r + 1, c + 0), &m(r + 1, c + 1), &m(r + 1, c + 2),
          &m(r + 2, c + 0), &m(r + 2, c + 1), &m(r + 2, c + 2));
    };

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
    NN.clear();

    auto const_unit_n = get_tensor_vec(
        *(commonDataSimpleContact->normalVectorSlavePtr.get()));

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
        ++t_w;
        ++t_aug_lambda_ptr;
        continue;
      }

      double val_m = t_w * area_slave;
      auto t_base_master = row_data.getFTensor0N(gg, 0);

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        auto t_base_lambda = col_data.getFTensor0N(gg, 0);
        const double m = val_m * t_base_master;
        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
          auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);
          const double n = m * t_base_lambda;
          
          t_assemble_m(i, j) -= n * const_unit_n(i) * const_unit_n(j);

          // ++t_assemble_m;
          ++t_base_lambda; // update cols slave
        }
        ++t_base_master; // update rows master
      }
      ++t_w;
      ++t_aug_lambda_ptr;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGapConstraintOverLambda::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const int nb_row = row_data.getIndices().size() / 3;
  const int nb_col = col_data.getIndices().size() / 3;

  if (nb_row && nb_col) {
    const int nb_gauss_pts = row_data.getN().size1();
    const double area_slave =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    NN.resize(3 * nb_row, 3 * nb_col, false);
    NN.clear();

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 3, 3>(
          &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2),
          &m(r + 1, c + 0), &m(r + 1, c + 1), &m(r + 1, c + 2),
          &m(r + 2, c + 0), &m(r + 2, c + 1), &m(r + 2, c + 2));
    };

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    auto t_lagrange_slave =
        getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_const_unit_n =
        get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      if (t_aug_lambda_ptr <= 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
        ++t_w;
        ++t_aug_lambda_ptr;
        ++t_lagrange_slave;
        // cerr << "OK " << t_aug_lambda_ptr <<"\n";
        continue;
      }

      const double val_s = -t_w * area_slave / cN;
      // cerr << "t_w  " << t_w << "\n";

      // cerr << "area_slave  " << area_slave << "\n";

      // cerr << "cN  " << cN << "\n";
      // FTensor::Tensor0<FTensor::PackPtr<double *, 1>> t_mat(
      //     &*NN.data().begin());

      auto t_base_lambda_row = row_data.getFTensor0N(gg, 0);
      for (int bbr = 0; bbr != nb_row; ++bbr) {
        auto t_base_lambda_col = col_data.getFTensor0N(gg, 0);
        const double s = val_s * t_base_lambda_row;
        for (int bbc = 0; bbc != nb_col; ++bbc) {
          auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

          t_assemble_m(i, j) +=
              s * t_base_lambda_col * t_const_unit_n(i) * t_const_unit_n(j);

          // ++t_mat;
          ++t_base_lambda_col; // update cols
        }
        ++t_base_lambda_row; // update rows
      }

      ++t_lagrange_slave;
      ++t_w;
      ++t_aug_lambda_ptr;
    }
    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGapConstraintOverSpatialMaster::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {

    const int nb_gauss_pts = row_data.getN().size1();
    int nb_base_fun_row = row_data.getFieldData().size() / 3;
    int nb_base_fun_col = col_data.getFieldData().size() / 3;

    const double area_master =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    const double area_slave =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
    NN.clear();

    auto get_tensor_from_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 3, 3>(
          &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2),
          &m(r + 1, c + 0), &m(r + 1, c + 1), &m(r + 1, c + 2),
          &m(r + 2, c + 0), &m(r + 2, c + 1), &m(r + 2, c + 2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    auto t_const_unit_n =
        get_tensor_from_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

    auto t_const_unit_master = get_tensor_from_vec(
        *(commonDataSimpleContact->normalVectorMasterPtr));

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_w = getFTensor0IntegrationWeightSlave();
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
        ++t_w;
        ++t_aug_lambda_ptr;
        continue;
      }

      const double val_m = t_w * area_slave;

      auto t_base_lambda = row_data.getFTensor0N(gg, 0);
      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

        auto t_base_master = col_data.getFTensor0N(gg, 0);
        FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_mat{
            &NN(bbr, 0), &NN(bbr, 1), &NN(bbr, 2)};
        const double m = val_m * t_base_lambda;

        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
          auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

          t_assemble_m(i, j) +=
              t_const_unit_n(i) * t_const_unit_n(j) * m * t_base_master;

          ++t_base_master; // update rows
          // ++t_mat;
        }
        ++t_base_lambda; // update cols master
      }

      ++t_aug_lambda_ptr;
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGapConstraintOverSpatialSlave::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {

    const int nb_gauss_pts = row_data.getN().size1();
    int nb_base_fun_row = row_data.getFieldData().size() / 3;
    int nb_base_fun_col = col_data.getFieldData().size() / 3;

    const double area_slave =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    NN.resize( 3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
    NN.clear();

    auto get_tensor_from_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 3, 3>(
          &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2),
          &m(r + 1, c + 0), &m(r + 1, c + 1), &m(r + 1, c + 2),
          &m(r + 2, c + 0), &m(r + 2, c + 1), &m(r + 2, c + 2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    auto t_const_unit_n =
        get_tensor_from_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_w = getFTensor0IntegrationWeightSlave();
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
        ++t_w;
        ++t_aug_lambda_ptr;
        continue;
      }

      const double val_m = t_w * area_slave;

      auto t_base_lambda = row_data.getFTensor0N(gg, 0);
      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

        auto t_base_slave = col_data.getFTensor0N(gg, 0);
        // FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_mat{
        //     &NN(bbr, 0), &NN(bbr, 1), &NN(bbr, 2)};
        const double m = val_m * t_base_lambda;

        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
        auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);
        t_assemble_m(i, j) -=
            t_const_unit_n(i) * t_const_unit_n(j) * m * t_base_slave;

        ++t_base_slave; // update rows
        // ++t_mat;
        }
        ++t_base_lambda; // update cols master
      }

      ++t_aug_lambda_ptr;
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactAugmentedFrictionMasterMaster::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto t_w = getFTensor0IntegrationWeightSlave();

  const double area_master = commonDataSimpleContact->areaSlave;

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<2>(
      *commonDataSimpleContact->tangentAugmentedLambdasPtr);

  auto t_norm_tang_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->normAugTangentLambdasPtr);

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  auto t_tangent_lambda =
      getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

  auto t_tangent_1_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr));

  auto t_tangent_2_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr));

  // const double cNTangentPtr = *cNTangentPtr.get();
  // const double cNNormalPtr = *cNNormalPtr.get();
  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    // cerr << " t_aug_lambda_ptr " << t_aug_lambda_ptr << "\n";
    if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
      ++t_norm_tang_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_aug_lambda_ptr;
      ++t_w;
      // cerr << "no contact\n";
      continue;
    }

    double stick_slip_check =
        t_norm_tang_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

    // cerr << " stick_slip_check " << stick_slip_check << "\n";

    bool stick = true;
    if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
      stick = false;

    const double val_m = t_w * area_master;

    FTensor::Tensor0<double *> t_base_master_col(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base_master_row(&row_data.getN()(gg, 0));
      const double m = val_m * t_base_master_col;

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

        if (stick) {
          // cerr << "stick \n";

          // cerr << "t_tangent_1_at_gp " << t_tangent_1_at_gp << "\n";
          // cerr << "t_tangent_2_at_gp " << t_tangent_2_at_gp << "\n";
          // cerr << "normal " << normal <<"\n";
          // cerr << "sad   " << m * cNTangentPtr * t_base_master_row << "\n";
          t_assemble_s(i, j) += m * cNTangentPtr * t_tangent_1_at_gp(i) *
                                t_tangent_1_at_gp(j) * t_base_master_row;
          t_assemble_s(i, j) += m * cNTangentPtr * t_tangent_2_at_gp(i) *
                                t_tangent_2_at_gp(j) * t_base_master_row;

          // t_assemble_s(i, j) += m * cNTangentPtr * t_tangent_2_at_gp(i) *
          //                       t_tangent_1_at_gp(j) * t_base_master_row;
          // t_assemble_s(i, j) += m * cNTangentPtr * t_tangent_1_at_gp(i) *
          //                       t_tangent_2_at_gp(j) * t_base_master_row;

        } else {
          // cerr << "stick \n";
          // normal gap
          // cerr << "slip \n";

          t_assemble_s(i, j) += muTan * m * cNNormalPtr *
                                t_tangent_aug_lambda_ptr(0) *
                                t_tangent_1_at_gp(i) * normal(j) *
                                t_base_master_row / t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) += muTan * m * cNNormalPtr *
                                t_tangent_aug_lambda_ptr(1) *
                                t_tangent_2_at_gp(i) * normal(j) *
                                t_base_master_row / t_norm_tang_aug_lambda_ptr;

          // t_assemble_s(i, j) += muTan * m * cNNormalPtr *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_2_at_gp(i) * normal(j) *
          //                       t_base_master_row / t_norm_tang_aug_lambda_ptr;

          // t_assemble_s(i, j) += muTan * m * cNNormalPtr *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_1_at_gp(i) * normal(j) *
          //                       t_base_master_row / t_norm_tang_aug_lambda_ptr;

          // tangent gap aug lambda

          t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_1_at_gp(i) *
                                t_tangent_1_at_gp(j) * t_base_master_row /
                                t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_2_at_gp(i) *
                                t_tangent_2_at_gp(j) * t_base_master_row /
                                t_norm_tang_aug_lambda_ptr;

          // t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) *
          //                       t_base_master_row / t_norm_tang_aug_lambda_ptr;

          // t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j) *
          //                       t_base_master_row / t_norm_tang_aug_lambda_ptr;

          const double pow_3 = t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_aug_lambda_ptr(0) *
                                t_tangent_aug_lambda_ptr(0) *
                                t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
                                t_base_master_row / pow_3;

          t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_aug_lambda_ptr(1) *
                                t_tangent_aug_lambda_ptr(1) *
                                t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) *
                                t_base_master_row / pow_3;

          t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_aug_lambda_ptr(0) *
                                t_tangent_aug_lambda_ptr(1) *
                                t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j) *
                                t_base_master_row / pow_3;

          t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_aug_lambda_ptr(1) *
                                t_tangent_aug_lambda_ptr(0) *
                                t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) *
                                t_base_master_row / pow_3;
          //
          // t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) *
          //                       t_base_master_row / pow_3;

          // t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j) *
          //                       t_base_master_row / pow_3;

          // t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) *
          //                       t_base_master_row / pow_3;

          // t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
          //                       t_base_master_row / pow_3;
        }

        ++t_base_master_row; // update rows
      }
      ++t_base_master_col; // update cols slave
    }
    ++t_w;
    ++t_norm_tang_aug_lambda_ptr;
    ++t_tangent_aug_lambda_ptr;
    ++t_aug_lambda_ptr;
  }

  CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactAugmentedFrictionMasterSlave::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto t_w = getFTensor0IntegrationWeightSlave();

  const double area_master = commonDataSimpleContact->areaSlave;

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<2>(
      *commonDataSimpleContact->tangentAugmentedLambdasPtr);

  auto t_norm_tang_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->normAugTangentLambdasPtr);

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  auto t_tangent_lambda =
      getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

  auto t_tangent_1_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr));

  auto t_tangent_2_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr));

  // const double cNTangentPtr = *cNTangentPtr.get();
  // const double cNNormalPtr = *cNNormalPtr.get();
  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
      ++t_norm_tang_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_aug_lambda_ptr;
      ++t_w;
      continue;
    }

    double stick_slip_check =
        t_norm_tang_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

    bool stick = true;
    if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
      stick = false;

    const double val_m = t_w * area_master;

    FTensor::Tensor0<double *> t_base_slave_col(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base_master_row(&row_data.getN()(gg, 0));
      const double m = val_m * t_base_slave_col;

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {

        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

        if (stick) {
          t_assemble_s(i, j) -= m * cNTangentPtr * t_tangent_1_at_gp(i) *
                                t_tangent_1_at_gp(j) * t_base_master_row;
          t_assemble_s(i, j) -= m * cNTangentPtr * t_tangent_2_at_gp(i) *
                                t_tangent_2_at_gp(j) * t_base_master_row;

          // t_assemble_s(i, j) -= m * cNTangentPtr * t_tangent_2_at_gp(i) *
          //                       t_tangent_1_at_gp(j) * t_base_master_row;
          // t_assemble_s(i, j) -= m * cNTangentPtr * t_tangent_1_at_gp(i) *
          //                       t_tangent_2_at_gp(j) * t_base_master_row;
        } else {

          // normal gap

          t_assemble_s(i, j) -= muTan * m * cNNormalPtr *
                                t_tangent_aug_lambda_ptr(0) *
                                t_tangent_1_at_gp(i) * normal(j) *
                                t_base_master_row / t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) -= muTan * m * cNNormalPtr *
                                t_tangent_aug_lambda_ptr(1) *
                                t_tangent_2_at_gp(i) * normal(j) *
                                t_base_master_row / t_norm_tang_aug_lambda_ptr;

          // t_assemble_s(i, j) -= muTan * m * cNNormalPtr *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_2_at_gp(i) * normal(j) *
          //                       t_base_master_row / t_norm_tang_aug_lambda_ptr;

          // t_assemble_s(i, j) -= muTan * m * cNNormalPtr *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_1_at_gp(i) * normal(j) *
          //                       t_base_master_row / t_norm_tang_aug_lambda_ptr;

          // tangent gap aug lambda

          t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_1_at_gp(i) *
                                t_tangent_1_at_gp(j) * t_base_master_row /
                                t_norm_tang_aug_lambda_ptr;
          t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_2_at_gp(i) *
                                t_tangent_2_at_gp(j) * t_base_master_row /
                                t_norm_tang_aug_lambda_ptr;

          // t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) *
          //                       t_base_master_row / t_norm_tang_aug_lambda_ptr;
          // t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j) *
          //                       t_base_master_row / t_norm_tang_aug_lambda_ptr;

          const double pow_3 = t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_aug_lambda_ptr(0) *
                                t_tangent_aug_lambda_ptr(0) *
                                t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
                                t_base_master_row / pow_3;

          t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_aug_lambda_ptr(1) *
                                t_tangent_aug_lambda_ptr(1) *
                                t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) *
                                t_base_master_row / pow_3;

          t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_aug_lambda_ptr(0) *
                                t_tangent_aug_lambda_ptr(1) *
                                t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j) *
                                t_base_master_row / pow_3;

          t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_aug_lambda_ptr(1) *
                                t_tangent_aug_lambda_ptr(0) *
                                t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) *
                                t_base_master_row / pow_3;

                                //

          // t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr *
          //                       m * t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) *
          //                       t_base_master_row / pow_3;

          // t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr *
          //                       m * t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j) *
          //                       t_base_master_row / pow_3;

          // t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr *
          //                       m * t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) *
          //                       t_base_master_row / pow_3;

          // t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr *
          //                       m * t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
          //                       t_base_master_row / pow_3;


        }

        ++t_base_master_row; // update rows
      }
      ++t_base_slave_col; // update cols slave
    }
    ++t_w;
    ++t_norm_tang_aug_lambda_ptr;
    ++t_tangent_aug_lambda_ptr;
    ++t_aug_lambda_ptr;
  }

  CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalContactAugmentedTangentLambdaOverLambdaMasterSlave::
    doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
           EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {
    const int nb_gauss_pts = row_data.getN().size1();

    int nb_base_fun_row = row_data.getFieldData().size() / 3;
    int nb_base_fun_col = col_data.getFieldData().size() / 3;

    const double area_master =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    // auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    //   return FTensor::Tensor1<double *, 3>(&m(r + 0, c), &m(r + 1, c),
    //                                        &m(r + 2, c));
    // };

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 3, 3>(
          &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2),
          &m(r + 1, c + 0), &m(r + 1, c + 1), &m(r + 1, c + 2),
          &m(r + 2, c + 0), &m(r + 2, c + 1), &m(r + 2, c + 2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
    NN.clear();

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<2>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_norm_tang_aug_lambda_ptr = getFTensor0FromVec(
        *commonDataSimpleContact->normAugTangentLambdasPtr);

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_lambda =
        getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

    auto t_tangent_1_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr));

    auto t_tangent_2_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr));

    auto normal =
        get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
        ++t_norm_tang_aug_lambda_ptr;
        ++t_tangent_aug_lambda_ptr;
        ++t_aug_lambda_ptr;
        ++t_w;
        continue;
      }

      double val_m = t_w * area_master;
      auto t_base_master = row_data.getFTensor0N(gg, 0);

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 0);
        auto t_base_lambda = col_data.getFTensor0N(gg, 0);
        const double m = val_m * t_base_master;
        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

          auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);
          const double n = m * t_base_lambda;

          double stick_slip_check =
              t_norm_tang_aug_lambda_ptr + muTan * t_aug_lambda_ptr;
          bool stick = true;
          if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
            stick = false;
          if (stick) {
          } else {
            t_assemble_s(i, j) += muTan * n * t_tangent_aug_lambda_ptr(0) *
                                  t_tangent_1_at_gp(i) * normal(j) /
                                  t_norm_tang_aug_lambda_ptr;

            t_assemble_s(i, j) += muTan * n * t_tangent_aug_lambda_ptr(1) *
                                  t_tangent_2_at_gp(i) * normal(j) /
                                  t_norm_tang_aug_lambda_ptr;
          }
            // t_assemble_s(i, j) += muTan * n * t_tangent_aug_lambda_ptr(0) *
            //                       t_tangent_2_at_gp(i) * normal(j) /
            //                       t_norm_tang_aug_lambda_ptr;

            // t_assemble_s(i, j) += muTan * n * t_tangent_aug_lambda_ptr(1) *
            //                       t_tangent_1_at_gp(i) * normal(j) /
            //                       t_norm_tang_aug_lambda_ptr;

            ++t_base_lambda; // update cols slave
          }
        ++t_base_master; // update rows master
      }
      ++t_w;
      ++t_norm_tang_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_aug_lambda_ptr;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::
    OpCalContactAugmentedTangentLambdaOverLambdaTanMasterSlave::doWork(
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
    int nb_base_fun_col = col_data.getFieldData().size() / 3;

    const double area_master =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    // auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    //   return FTensor::Tensor1<double *, 3>(&m(r + 0, c), &m(r + 1, c),
    //                                        &m(r + 2, c));
    // };


    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 3, 3>(
          &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2),
          &m(r + 1, c + 0), &m(r + 1, c + 1), &m(r + 1, c + 2),
          &m(r + 2, c + 0), &m(r + 2, c + 1), &m(r + 2, c + 2));
    };

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
    NN.clear();

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<2>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_norm_tang_aug_lambda_ptr = getFTensor0FromVec(
        *commonDataSimpleContact->normAugTangentLambdasPtr);

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_lambda =
        getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

    auto t_tangent_1_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr));

    auto t_tangent_2_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr));

    VectorDouble unit_vec;
    unit_vec.resize(3,false);
    unit_vec(0) = unit_vec(1) = unit_vec(2) = 1.;
    auto t_unit = get_tensor_vec(unit_vec);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      if (t_aug_lambda_ptr > 0 /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
        ++t_norm_tang_aug_lambda_ptr;
        ++t_tangent_aug_lambda_ptr;
        ++t_aug_lambda_ptr;
        ++t_w;
        continue;
      }

      double stick_slip_check =
          t_norm_tang_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

      bool stick = true;
      if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
        stick = false;

      double val_m = t_w * area_master;
      auto t_base_master = row_data.getFTensor0N(gg, 0);

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        
        auto t_base_lambda = col_data.getFTensor0N(gg, 0);
        const double m = val_m * t_base_master;
        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
          const double n = m * t_base_lambda;
          auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);
          if (stick) {
            // cerr << "stick\n";
            // cerr << "n   " << n << "\n";

            t_assemble_m(i, j) -=
                n * t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j);

            t_assemble_m(i, j) -=
                n * t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j);

            // t_assemble_m(i, j) -=
            //     n * t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j);

            // t_assemble_m(i, j) -=
            //     n * t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j);

          } else {

            // cerr << "Slip\n";
            const double pow_3 = t_norm_tang_aug_lambda_ptr *
                                 t_norm_tang_aug_lambda_ptr *
                                 t_norm_tang_aug_lambda_ptr;

            t_assemble_m(i, j) += n * muTan * t_aug_lambda_ptr *
                                  t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) 
                                  / t_norm_tang_aug_lambda_ptr;


            // t_assemble_m(i, j) += n * muTan * t_aug_lambda_ptr *
            //                       t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) 
            //                       / t_norm_tang_aug_lambda_ptr;

            t_assemble_m(i, j) -= n * muTan * t_aug_lambda_ptr *
                                  t_tangent_aug_lambda_ptr(0) *
                                  t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
                                  t_tangent_aug_lambda_ptr(0) / pow_3;

            // t_assemble_m(i, j) -= n * muTan * t_aug_lambda_ptr *
            //                       t_tangent_aug_lambda_ptr(0) *
            //                       t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) *
            //                       t_tangent_aug_lambda_ptr(0) / pow_3;

            // t_assemble_m(i) -=
            //     n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(0) *
            //     t_tangent_1_at_gp(i) * t_tangent_aug_lambda_ptr(0) / pow_3;

            t_assemble_m(i, j) -= n * muTan * t_aug_lambda_ptr *
                                  t_tangent_aug_lambda_ptr(0) *
                                  t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) *
                                  t_tangent_aug_lambda_ptr(1) / pow_3;

            // t_assemble_m(i, j) -= n * muTan * t_aug_lambda_ptr *
            //                       t_tangent_aug_lambda_ptr(0) *
            //                       t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
            //                       t_tangent_aug_lambda_ptr(1) / pow_3;

            // t_assemble_m(i) -=
            //     n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(0) *
            //     t_tangent_2_at_gp(i) * t_tangent_aug_lambda_ptr(1) / pow_3;

            // ++t_assemble_m;

            t_assemble_m(i, j) += n * muTan * t_aug_lambda_ptr *
                                  t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) /
                                  t_norm_tang_aug_lambda_ptr;

                                  // t_assemble_m(i, j) += n * muTan * t_aug_lambda_ptr *
                                  // t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j) /
                                  // t_norm_tang_aug_lambda_ptr;

            // t_assemble_m(i) += n * muTan * t_aug_lambda_ptr *
            //                    t_tangent_2_at_gp(i) /
            //                    t_norm_tang_aug_lambda_ptr;

            t_assemble_m(i, j) -=
                n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(1) *
                t_tangent_aug_lambda_ptr(1) * t_tangent_2_at_gp(i) *
                t_tangent_2_at_gp(j) / pow_3;

                // t_assemble_m(i, j) -=
                // n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(1) *
                // t_tangent_aug_lambda_ptr(1) * t_tangent_1_at_gp(i) *
                // t_tangent_2_at_gp(j) / pow_3;

            // t_assemble_m(i) -=
            //     n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(1) *
            //     t_tangent_aug_lambda_ptr(1) * t_tangent_2_at_gp(i) / pow_3;

            t_assemble_m(i, j) -=
                n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(1) *
                t_tangent_aug_lambda_ptr(0) * t_tangent_1_at_gp(i) *
                t_tangent_2_at_gp(j) / pow_3;

                // t_assemble_m(i, j) -=
                // n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(1) *
                // t_tangent_aug_lambda_ptr(0) * t_tangent_2_at_gp(i) *
                // t_tangent_2_at_gp(j) / pow_3;

            // t_assemble_m(i) -=
            //     n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(1) *
            //     t_tangent_aug_lambda_ptr(0) * t_tangent_1_at_gp(i) / pow_3;
          }

          // ++t_assemble_m;
          ++t_base_lambda; // update cols slave
        }
        ++t_base_master; // update rows master
      }
      ++t_w;
      ++t_norm_tang_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_aug_lambda_ptr;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactAugmentedFrictionSlaveSlave::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto t_w = getFTensor0IntegrationWeightSlave();

  const double area_master = commonDataSimpleContact->areaSlave;

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<2>(
      *commonDataSimpleContact->tangentAugmentedLambdasPtr);

  auto t_norm_tang_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->normAugTangentLambdasPtr);

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  auto t_tangent_lambda =
      getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

  auto t_tangent_1_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr));

  auto t_tangent_2_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr));

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
      ++t_norm_tang_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_aug_lambda_ptr;
      ++t_w;
      // cerr << "no friction\n";
      continue;
    }

    double stick_slip_check =
        t_norm_tang_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

    bool stick = true;
    if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
      stick = false;

    const double val_m = t_w * area_master;

    FTensor::Tensor0<double *> t_base_slave_col(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base_slave_row(&row_data.getN()(gg, 0));
      const double m = val_m * t_base_slave_col;

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

        if (stick) {
          t_assemble_s(i, j) += m * cNTangentPtr * t_tangent_1_at_gp(i) *
                                t_tangent_1_at_gp(j) * t_base_slave_row;
          t_assemble_s(i, j) += m * cNTangentPtr * t_tangent_2_at_gp(i) *
                                t_tangent_2_at_gp(j) * t_base_slave_row;

          // t_assemble_s(i, j) += m * cNTangentPtr * t_tangent_2_at_gp(i) *
          //                       t_tangent_1_at_gp(j) * t_base_slave_row;
          // t_assemble_s(i, j) += m * cNTangentPtr * t_tangent_1_at_gp(i) *
          //                       t_tangent_2_at_gp(j) * t_base_slave_row;


        } else {

          // normal gap

          t_assemble_s(i, j) += muTan * m * cNNormalPtr *
                                t_tangent_aug_lambda_ptr(0) *
                                t_tangent_1_at_gp(i) * normal(j) *
                                t_base_slave_row / t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) += muTan * m * cNNormalPtr *
                                t_tangent_aug_lambda_ptr(1) *
                                t_tangent_2_at_gp(i) * normal(j) *
                                t_base_slave_row / t_norm_tang_aug_lambda_ptr;

          // t_assemble_s(i, j) += muTan * m * cNNormalPtr *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_2_at_gp(i) * normal(j) *
          //                       t_base_slave_row / t_norm_tang_aug_lambda_ptr;

          // t_assemble_s(i, j) += muTan * m * cNNormalPtr *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_1_at_gp(i) * normal(j) *
          //                       t_base_slave_row / t_norm_tang_aug_lambda_ptr;

          // tangent gap aug lambda

          t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_1_at_gp(i) *
                                t_tangent_1_at_gp(j) * t_base_slave_row /
                                t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_2_at_gp(i) *
                                t_tangent_2_at_gp(j) * t_base_slave_row /
                                t_norm_tang_aug_lambda_ptr;

          // t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) *
          //                       t_base_slave_row / t_norm_tang_aug_lambda_ptr;

          // t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j) *
          //                       t_base_slave_row / t_norm_tang_aug_lambda_ptr;

          const double pow_3 = t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_aug_lambda_ptr(0) *
                                t_tangent_aug_lambda_ptr(0) *
                                t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
                                t_base_slave_row / pow_3;

          t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_aug_lambda_ptr(1) *
                                t_tangent_aug_lambda_ptr(1) *
                                t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) *
                                t_base_slave_row / pow_3;

          t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_aug_lambda_ptr(0) *
                                t_tangent_aug_lambda_ptr(1) *
                                t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j) *
                                t_base_slave_row / pow_3;

          t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_aug_lambda_ptr(1) *
                                t_tangent_aug_lambda_ptr(0) *
                                t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) *
                                t_base_slave_row / pow_3;
                                //
          // t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) *
          //                       t_base_slave_row / pow_3;

          // t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j) *
          //                       t_base_slave_row / pow_3;

          // t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) *
          //                       t_base_slave_row / pow_3;

          // t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
          //                       t_base_slave_row / pow_3;
        }

        ++t_base_slave_row; // update rows
      }
      ++t_base_slave_col; // update cols slave
    }
    ++t_w;
    ++t_norm_tang_aug_lambda_ptr;
    ++t_tangent_aug_lambda_ptr;
    ++t_aug_lambda_ptr;
  }
  CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactAugmentedFrictionSlaveMaster::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto t_w = getFTensor0IntegrationWeightSlave();

  const double area_master = commonDataSimpleContact->areaSlave;

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<2>(
      *commonDataSimpleContact->tangentAugmentedLambdasPtr);

  auto t_norm_tang_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->normAugTangentLambdasPtr);

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  auto t_tangent_lambda =
      getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

  auto t_tangent_1_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr));

  auto t_tangent_2_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr));

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
      ++t_norm_tang_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_aug_lambda_ptr;
      ++t_w;
      continue;
    }

    double stick_slip_check =
        t_norm_tang_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

    bool stick = true;
    if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
      stick = false;

    const double val_m = t_w * area_master;

    FTensor::Tensor0<double *> t_base_master_col(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base_slave_row(&row_data.getN()(gg, 0));
      const double m = val_m * t_base_master_col;

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {

        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

        if (stick) {
          t_assemble_s(i, j) -= m * cNTangentPtr * t_tangent_1_at_gp(i) *
                                t_tangent_1_at_gp(j) * t_base_slave_row;
          t_assemble_s(i, j) -= m * cNTangentPtr * t_tangent_2_at_gp(i) *
                                t_tangent_2_at_gp(j) * t_base_slave_row;

          // t_assemble_s(i, j) -= m * cNTangentPtr * t_tangent_2_at_gp(i) *
          //                       t_tangent_1_at_gp(j) * t_base_slave_row;
          // t_assemble_s(i, j) -= m * cNTangentPtr * t_tangent_1_at_gp(i) *
          //                       t_tangent_2_at_gp(j) * t_base_slave_row;

        } else {

          // normal gap

          t_assemble_s(i, j) -= muTan * m * cNNormalPtr *
                                t_tangent_aug_lambda_ptr(0) *
                                t_tangent_1_at_gp(i) * normal(j) *
                                t_base_slave_row / t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) -= muTan * m * cNNormalPtr *
                                t_tangent_aug_lambda_ptr(1) *
                                t_tangent_2_at_gp(i) * normal(j) *
                                t_base_slave_row / t_norm_tang_aug_lambda_ptr;

          // t_assemble_s(i, j) -= muTan * m * cNNormalPtr *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_2_at_gp(i) * normal(j) *
          //                       t_base_slave_row / t_norm_tang_aug_lambda_ptr;

          // t_assemble_s(i, j) -= muTan * m * cNNormalPtr *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_1_at_gp(i) * normal(j) *
          //                       t_base_slave_row / t_norm_tang_aug_lambda_ptr;

          // tangent gap aug lambda


          t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_1_at_gp(i) *
                                t_tangent_1_at_gp(j) * t_base_slave_row /
                                t_norm_tang_aug_lambda_ptr;
          t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_2_at_gp(i) *
                                t_tangent_2_at_gp(j) * t_base_slave_row /
                                t_norm_tang_aug_lambda_ptr;

          // t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) *
          //                       t_base_slave_row / t_norm_tang_aug_lambda_ptr;
          // t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j) *
          //                       t_base_slave_row / t_norm_tang_aug_lambda_ptr;

          const double pow_3 = t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_aug_lambda_ptr(0) *
                                t_tangent_aug_lambda_ptr(0) *
                                t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
                                t_base_slave_row / pow_3;

          t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_aug_lambda_ptr(1) *
                                t_tangent_aug_lambda_ptr(1) *
                                t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) *
                                t_base_slave_row / pow_3;

          t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_aug_lambda_ptr(0) *
                                t_tangent_aug_lambda_ptr(1) *
                                t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j) *
                                t_base_slave_row / pow_3;

          t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_aug_lambda_ptr(1) *
                                t_tangent_aug_lambda_ptr(0) *
                                t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) *
                                t_base_slave_row / pow_3;

                                //

          // t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) *
          //                       t_base_slave_row / pow_3;

          // t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j) *
          //                       t_base_slave_row / pow_3;

          // t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) *
          //                       t_base_slave_row / pow_3;

          // t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr * m *
          //                       t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
          //                       t_base_slave_row / pow_3;
        }

        ++t_base_slave_row; // update rows
      }
      ++t_base_master_col; // update cols slave
    }
    ++t_w;
    ++t_norm_tang_aug_lambda_ptr;
    ++t_tangent_aug_lambda_ptr;
    ++t_aug_lambda_ptr;
  }

  CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalContactAugmentedTangentLambdaOverLambdaSlaveSlave::
    doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
           EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {
    const int nb_gauss_pts = row_data.getN().size1();

    int nb_base_fun_row = row_data.getFieldData().size() / 3;
    int nb_base_fun_col = col_data.getFieldData().size() / 3;

    const double area_master =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    // auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    //   return FTensor::Tensor1<double *, 3>(&m(r + 0, c), &m(r + 1, c),
    //                                        &m(r + 2, c));
    // };

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 3, 3>(
          &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2),
          &m(r + 1, c + 0), &m(r + 1, c + 1), &m(r + 1, c + 2),
          &m(r + 2, c + 0), &m(r + 2, c + 1), &m(r + 2, c + 2));
    };

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
    NN.clear();

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<2>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_norm_tang_aug_lambda_ptr = getFTensor0FromVec(
        *commonDataSimpleContact->normAugTangentLambdasPtr);

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_lambda =
        getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

    auto t_tangent_1_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr));

    auto t_tangent_2_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr));

    auto normal =
        get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
        ++t_norm_tang_aug_lambda_ptr;
        ++t_tangent_aug_lambda_ptr;
        ++t_aug_lambda_ptr;
        ++t_w;
        continue;
      }

      double val_m = t_w * area_master;
      auto t_base_master = row_data.getFTensor0N(gg, 0);

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        // auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 0);
        auto t_base_lambda = col_data.getFTensor0N(gg, 0);
        const double m = val_m * t_base_master;
        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
          auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);
          const double n = m * t_base_lambda;

          double stick_slip_check =
              t_norm_tang_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

          bool stick = true;
          if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
            stick = false;
          if (stick) {
          } else {
            t_assemble_m(i, j) -= muTan * n * t_tangent_aug_lambda_ptr(0) *
                                  t_tangent_1_at_gp(i) * normal(j) /
                                  t_norm_tang_aug_lambda_ptr;

            t_assemble_m(i, j) -= muTan * n * t_tangent_aug_lambda_ptr(1) *
                                  t_tangent_2_at_gp(i) * normal(j) /
                                  t_norm_tang_aug_lambda_ptr;
          }
          // t_assemble_m(i, j) -= muTan * n * t_tangent_aug_lambda_ptr(0) *
          //                       t_tangent_2_at_gp(i) * normal(j) /
          //                       t_norm_tang_aug_lambda_ptr;

          // t_assemble_m(i, j) -= muTan * n * t_tangent_aug_lambda_ptr(1) *
          //                       t_tangent_1_at_gp(i) * normal(j) /
          //                       t_norm_tang_aug_lambda_ptr;
          // ++t_assemble_m;
          ++t_base_lambda; // update cols slave
        }
        ++t_base_master; // update rows master
      }
      ++t_w;
      ++t_norm_tang_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_aug_lambda_ptr;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::
    OpCalContactAugmentedTangentLambdaOverLambdaTanSlaveSlave::doWork(
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
    int nb_base_fun_col = col_data.getFieldData().size() / 3;

    const double area_master =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    // auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    //   return FTensor::Tensor1<double *, 3>(&m(r + 0, c), &m(r + 1, c),
    //                                        &m(r + 2, c));
    // };

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 3, 3>(
          &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2),
          &m(r + 1, c + 0), &m(r + 1, c + 1), &m(r + 1, c + 2),
          &m(r + 2, c + 0), &m(r + 2, c + 1), &m(r + 2, c + 2));
    };

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
    NN.clear();

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<2>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_norm_tang_aug_lambda_ptr = getFTensor0FromVec(
        *commonDataSimpleContact->normAugTangentLambdasPtr);

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_lambda =
        getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

    auto t_tangent_1_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr));

    auto t_tangent_2_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr));

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
        ++t_norm_tang_aug_lambda_ptr;
        ++t_tangent_aug_lambda_ptr;
        ++t_aug_lambda_ptr;
        ++t_w;
        continue;
      }

      double stick_slip_check =
          t_norm_tang_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

      bool stick = true;
      if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
        stick = false;

      double val_m = t_w * area_master;
      auto t_base_master = row_data.getFTensor0N(gg, 0);

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        
        auto t_base_lambda = col_data.getFTensor0N(gg, 0);
        const double m = val_m * t_base_master;
        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
          const double n = m * t_base_lambda;
          auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

          if (stick) {
            // cerr << "stick\n";
            // cerr << "muTan " << muTan << "\n";
            // cerr << "t_norm_tang_aug_lambda_ptr " << t_norm_tang_aug_lambda_ptr
            //      << "\n";
            // cerr << "t_aug_lambda_ptr " << t_aug_lambda_ptr << "\n";
            
            t_assemble_m(i, j) +=
                n * t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j);
            t_assemble_m(i, j) +=
                n * t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j);

            // t_assemble_m(i, j) +=
            //     n * t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j);
            // t_assemble_m(i, j) +=
            //     n * t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j);

          } else {
            // cerr << "1slip\n";

            const double pow_3 = t_norm_tang_aug_lambda_ptr *
                                 t_norm_tang_aug_lambda_ptr *
                                 t_norm_tang_aug_lambda_ptr;

            t_assemble_m(i, j) -= n * muTan * t_aug_lambda_ptr *
                                  t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) /
                                  t_norm_tang_aug_lambda_ptr;

        
            t_assemble_m(i, j) += n * muTan * t_aug_lambda_ptr *
                                  t_tangent_aug_lambda_ptr(0) *
                                  t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
                                  t_tangent_aug_lambda_ptr(0) / pow_3;

            t_assemble_m(i, j) += n * muTan * t_aug_lambda_ptr *
                                  t_tangent_aug_lambda_ptr(0) *
                                  t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) *
                                  t_tangent_aug_lambda_ptr(1) / pow_3;
                                  //
            // t_assemble_m(i, j) -= n * muTan * t_aug_lambda_ptr *
            //                       t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) /
            //                       t_norm_tang_aug_lambda_ptr;

            // t_assemble_m(i, j) += n * muTan * t_aug_lambda_ptr *
            //                       t_tangent_aug_lambda_ptr(0) *
            //                       t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) *
            //                       t_tangent_aug_lambda_ptr(0) / pow_3;

            // t_assemble_m(i, j) += n * muTan * t_aug_lambda_ptr *
            //                       t_tangent_aug_lambda_ptr(0) *
            //                       t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) *
            //                       t_tangent_aug_lambda_ptr(1) / pow_3;


            t_assemble_m(i, j) -= n * muTan * t_aug_lambda_ptr *
                                  t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) /
                                  t_norm_tang_aug_lambda_ptr;

           
            t_assemble_m(i, j) +=
                n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(1) *
                t_tangent_aug_lambda_ptr(1) * t_tangent_2_at_gp(i) *
                t_tangent_2_at_gp(j) / pow_3;

           
            t_assemble_m(i, j) +=
                n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(1) *
                t_tangent_aug_lambda_ptr(0) * t_tangent_1_at_gp(i) *
                t_tangent_2_at_gp(j) / pow_3;
                //
            // t_assemble_m(i, j) -= n * muTan * t_aug_lambda_ptr *
            //                       t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j) /
            //                       t_norm_tang_aug_lambda_ptr;

            // t_assemble_m(i, j) +=
            //     n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(1) *
            //     t_tangent_aug_lambda_ptr(1) * t_tangent_1_at_gp(i) *
            //     t_tangent_2_at_gp(j) / pow_3;

            // t_assemble_m(i, j) +=
            //     n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(1) *
            //     t_tangent_aug_lambda_ptr(0) * t_tangent_2_at_gp(i) *
            //     t_tangent_2_at_gp(j) / pow_3;
          }

          // ++t_assemble_m;
          ++t_base_lambda; // update cols slave
        }
        ++t_base_master; // update rows master
      }
      ++t_w;
      ++t_norm_tang_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_aug_lambda_ptr;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalAugmentedTangentialContConditionDispSlaveSlave::
    doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
           DataForcesAndSourcesCore::EntData &row_data,
           DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

  // auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
  //   return FTensor::Tensor2<double *, 2, 3>(&m(r + 0, c + 0), &m(r + 0, c + 1),
  //                                           &m(r + 0, c + 2), &m(r + 1, c + 0),
  //                                           &m(r + 1, c + 1), &m(r + 1, c + 2));
  // };

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor_from_mat_1 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 0, c + 1),
                                         &m(r + 0, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto t_w = getFTensor0IntegrationWeightSlave();

  const double area_master = commonDataSimpleContact->areaSlave;

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<2>(
      *commonDataSimpleContact->tangentAugmentedLambdasPtr);

  auto t_norm_tang_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->normAugTangentLambdasPtr);

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  auto t_tangent_lambda =
      getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

  auto t_tangent_1_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr));

  auto t_tangent_2_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr));

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
      ++t_norm_tang_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_aug_lambda_ptr;
      ++t_w;
      continue;
    }

    double stick_slip_check =
        t_norm_tang_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

    bool stick = true;
    if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
      stick = false;

    const double val_m = t_w * area_master;

    FTensor::Tensor0<double *> t_base_master_col(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base_slave_row(&row_data.getN()(gg, 0));
      const double m = val_m * t_base_master_col;

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {

        // auto t_assemble_s = get_tensor_from_mat(NN, 2 * bbr, 3 * bbc);

        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);
        // auto t_assemble_s_2 = get_tensor_from_mat_1(NN, 2 * bbr + 1, 3 * bbc);

        if (stick) {
          t_assemble_s(i, j) += m * t_base_slave_row * t_tangent_1_at_gp(j) *
                                t_tangent_1_at_gp(i);
          t_assemble_s(i, j) += m * t_base_slave_row * t_tangent_2_at_gp(j) *
                             t_tangent_2_at_gp(i);

          // t_assemble_s(i, j) += m * t_base_slave_row * t_tangent_1_at_gp(j) *
          //                       t_tangent_2_at_gp(i);
          // t_assemble_s(i, j) += m * t_base_slave_row * t_tangent_2_at_gp(j) *
          //                       t_tangent_1_at_gp(i);

        } else {

          // normal gap

 
          t_assemble_s(i, j) +=
              muTan * m * cNNormalPtr * t_tangent_aug_lambda_ptr(0) *
              t_tangent_1_at_gp(i) * normal(j) * t_base_slave_row /
              (t_norm_tang_aug_lambda_ptr * cNTangentPtr);

          t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * m *
                                t_tangent_1_at_gp(j) * t_tangent_1_at_gp(i) *
                                t_base_slave_row / (t_norm_tang_aug_lambda_ptr);

          // t_assemble_s(i, j) +=
          //     muTan * m * cNNormalPtr * t_tangent_aug_lambda_ptr(0) *
          //     t_tangent_2_at_gp(i) * normal(j) * t_base_slave_row /
          //     (t_norm_tang_aug_lambda_ptr * cNTangentPtr);

          // t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * m *
          //                       t_tangent_1_at_gp(j) * t_tangent_2_at_gp(i) *
          //                       t_base_slave_row / (t_norm_tang_aug_lambda_ptr);

          const double pow_3 = t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) +=
              muTan * t_aug_lambda_ptr * m * t_tangent_aug_lambda_ptr(0) *
              t_tangent_aug_lambda_ptr(0) * t_tangent_1_at_gp(i) *
              t_tangent_1_at_gp(j) * t_base_slave_row / pow_3;

          t_assemble_s(i, j) +=
              muTan * t_aug_lambda_ptr * m * t_tangent_aug_lambda_ptr(0) *
              t_tangent_aug_lambda_ptr(1) * t_tangent_1_at_gp(i) *
              t_tangent_2_at_gp(j) * t_base_slave_row / pow_3;

          // t_assemble_s(i, j) +=
          //     muTan * t_aug_lambda_ptr * m * t_tangent_aug_lambda_ptr(0) *
          //     t_tangent_aug_lambda_ptr(0) * t_tangent_2_at_gp(i) *
          //     t_tangent_1_at_gp(j) * t_base_slave_row / pow_3;

          // t_assemble_s(i, j) +=
          //     muTan * t_aug_lambda_ptr * m * t_tangent_aug_lambda_ptr(0) *
          //     t_tangent_aug_lambda_ptr(1) * t_tangent_2_at_gp(i) *
          //     t_tangent_2_at_gp(j) * t_base_slave_row / pow_3;

         

          t_assemble_s(i, j) +=
              muTan * m * cNNormalPtr * t_tangent_aug_lambda_ptr(1) *
              t_tangent_2_at_gp(i) * normal(j) * t_base_slave_row /
              (t_norm_tang_aug_lambda_ptr * cNTangentPtr);

          t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * m *
                                t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) *
                                t_base_slave_row / t_norm_tang_aug_lambda_ptr;

          // t_assemble_s(i, j) +=
          //     muTan * m * cNNormalPtr * t_tangent_aug_lambda_ptr(1) *
          //     t_tangent_1_at_gp(i) * normal(j) * t_base_slave_row /
          //     (t_norm_tang_aug_lambda_ptr * cNTangentPtr);

          // t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * m *
          //                       t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j) *
          //                       t_base_slave_row / t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) +=
              muTan * t_aug_lambda_ptr * m * t_tangent_aug_lambda_ptr(1) *
              t_tangent_aug_lambda_ptr(1) * t_tangent_2_at_gp(i) *
              t_tangent_2_at_gp(j) * t_base_slave_row / pow_3;

          t_assemble_s(i, j) +=
              muTan * t_aug_lambda_ptr * m * t_tangent_aug_lambda_ptr(0) *
              t_tangent_aug_lambda_ptr(1) * t_tangent_2_at_gp(i) *
              t_tangent_1_at_gp(j) * t_base_slave_row / pow_3;

          // t_assemble_s(i, j) +=
          //     muTan * t_aug_lambda_ptr * m * t_tangent_aug_lambda_ptr(1) *
          //     t_tangent_aug_lambda_ptr(1) * t_tangent_1_at_gp(i) *
          //     t_tangent_2_at_gp(j) * t_base_slave_row / pow_3;

          // t_assemble_s(i, j) +=
          //     muTan * t_aug_lambda_ptr * m * t_tangent_aug_lambda_ptr(0) *
          //     t_tangent_aug_lambda_ptr(1) * t_tangent_1_at_gp(i) *
          //     t_tangent_1_at_gp(j) * t_base_slave_row / pow_3;
        }

        ++t_base_slave_row; // update rows
      }
      ++t_base_master_col; // update cols slave
    }
    ++t_w;
    ++t_norm_tang_aug_lambda_ptr;
    ++t_tangent_aug_lambda_ptr;
    ++t_aug_lambda_ptr;
  }

  CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalAugmentedTangentialContConditionDispSlaveMaster::
    doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
           DataForcesAndSourcesCore::EntData &row_data,
           DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

  // auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
  //   return FTensor::Tensor2<double *, 2, 3>(&m(r + 0, c + 0), &m(r + 0, c + 1),
  //                                           &m(r + 0, c + 2), &m(r + 1, c + 0),
  //                                           &m(r + 1, c + 1), &m(r + 1, c + 2));
  // };

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };
  auto get_tensor_from_mat_1 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 0, c + 1),
                                         &m(r + 0, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto t_w = getFTensor0IntegrationWeightSlave();

  const double area_master = commonDataSimpleContact->areaSlave;

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<2>(
      *commonDataSimpleContact->tangentAugmentedLambdasPtr);

  auto t_norm_tang_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->normAugTangentLambdasPtr);

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  auto t_tangent_lambda =
      getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

  auto t_tangent_1_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr));

  auto t_tangent_2_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr));

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
      ++t_norm_tang_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_aug_lambda_ptr;
      ++t_w;
      continue;
    }

    double stick_slip_check =
        t_norm_tang_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

    bool stick = true;
    if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
      stick = false;

    const double val_m = t_w * area_master;

    FTensor::Tensor0<double *> t_base_master_col(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base_slave_row(&row_data.getN()(gg, 0));
      const double m = val_m * t_base_master_col;

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {

        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

        // auto t_assemble_s_1 = get_tensor_from_mat_1(NN, 2 * bbr, 3 * bbc);
        // auto t_assemble_s_2 = get_tensor_from_mat_1(NN, 2 * bbr + 1, 3 * bbc);

        if (stick) {

          t_assemble_s(i, j) -= m * t_base_slave_row * t_tangent_1_at_gp(i) *
                                t_tangent_1_at_gp(j);
          t_assemble_s(i, j) -= m * t_base_slave_row * t_tangent_2_at_gp(i) *
                                t_tangent_2_at_gp(j);

          // t_assemble_s(i, j) -= m * t_base_slave_row * t_tangent_2_at_gp(i) *
          //                       t_tangent_1_at_gp(j);
          // t_assemble_s(i, j) -= m * t_base_slave_row * t_tangent_1_at_gp(i) *
          //                       t_tangent_2_at_gp(j);

        } else {

          t_assemble_s(i, j) -=
              muTan * m * cNNormalPtr * t_tangent_aug_lambda_ptr(0) *
              t_tangent_1_at_gp(i) * normal(j) * t_base_slave_row /
              (t_norm_tang_aug_lambda_ptr * cNTangentPtr);

          t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * m *
                                  t_tangent_1_at_gp(j) * t_tangent_1_at_gp(i) *
                                  t_base_slave_row /
                                  (t_norm_tang_aug_lambda_ptr);

          // t_assemble_s(i, j) -=
          //     muTan * m * cNNormalPtr * t_tangent_aug_lambda_ptr(0) *
          //     t_tangent_2_at_gp(i) * normal(j) * t_base_slave_row /
          //     (t_norm_tang_aug_lambda_ptr * cNTangentPtr);

          // t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * m *
          //                       t_tangent_1_at_gp(j) * t_tangent_2_at_gp(i) *
          //                       t_base_slave_row / (t_norm_tang_aug_lambda_ptr);



          const double pow_3 = t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) -=
              muTan * t_aug_lambda_ptr * m * t_tangent_aug_lambda_ptr(0) *
              t_tangent_aug_lambda_ptr(0) * t_tangent_1_at_gp(i) *
              t_tangent_1_at_gp(j) * t_base_slave_row / pow_3;

          t_assemble_s(i, j) -=
              muTan * t_aug_lambda_ptr * m * t_tangent_aug_lambda_ptr(0) *
              t_tangent_aug_lambda_ptr(1) * t_tangent_1_at_gp(i) *
              t_tangent_2_at_gp(j) * t_base_slave_row / pow_3;

          // t_assemble_s(i, j) -=
          //     muTan * t_aug_lambda_ptr * m * t_tangent_aug_lambda_ptr(0) *
          //     t_tangent_aug_lambda_ptr(0) * t_tangent_2_at_gp(i) *
          //     t_tangent_1_at_gp(j) * t_base_slave_row / pow_3;

          // t_assemble_s(i, j) -=
          //     muTan * t_aug_lambda_ptr * m * t_tangent_aug_lambda_ptr(0) *
          //     t_tangent_aug_lambda_ptr(1) * t_tangent_2_at_gp(i) *
          //     t_tangent_2_at_gp(j) * t_base_slave_row / pow_3;



          t_assemble_s(i, j) -=
              muTan * m * cNNormalPtr * t_tangent_aug_lambda_ptr(1) *
              t_tangent_2_at_gp(i) * normal(j) * t_base_slave_row /
              (t_norm_tang_aug_lambda_ptr * cNTangentPtr);

          t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * m *
                                t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) *
                                t_base_slave_row / t_norm_tang_aug_lambda_ptr;

// t_assemble_s(i, j) -=
//               muTan * m * cNNormalPtr * t_tangent_aug_lambda_ptr(1) *
//               t_tangent_1_at_gp(i) * normal(j) * t_base_slave_row /
//               (t_norm_tang_aug_lambda_ptr * cNTangentPtr);

//           t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * m *
//                                 t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j) *
//                                 t_base_slave_row / t_norm_tang_aug_lambda_ptr;


          t_assemble_s(i, j) -=
              muTan * t_aug_lambda_ptr * m * t_tangent_aug_lambda_ptr(1) *
              t_tangent_aug_lambda_ptr(1) * t_tangent_2_at_gp(i) *
              t_tangent_2_at_gp(j) * t_base_slave_row / pow_3;

          t_assemble_s(i, j) -=
              muTan * t_aug_lambda_ptr * m * t_tangent_aug_lambda_ptr(0) *
              t_tangent_aug_lambda_ptr(1) * t_tangent_2_at_gp(i) *
              t_tangent_1_at_gp(j) * t_base_slave_row / pow_3;

          // t_assemble_s(i, j) -=
          //     muTan * t_aug_lambda_ptr * m * t_tangent_aug_lambda_ptr(1) *
          //     t_tangent_aug_lambda_ptr(1) * t_tangent_1_at_gp(i) *
          //     t_tangent_2_at_gp(j) * t_base_slave_row / pow_3;

          // t_assemble_s(i, j) -=
          //     muTan * t_aug_lambda_ptr * m * t_tangent_aug_lambda_ptr(0) *
          //     t_tangent_aug_lambda_ptr(1) * t_tangent_1_at_gp(i) *
          //     t_tangent_1_at_gp(j) * t_base_slave_row / pow_3;

          // ===========++++++++========


        }

        ++t_base_slave_row; // update rows
      }
      ++t_base_master_col; // update cols slave
    }
    ++t_w;
    ++t_norm_tang_aug_lambda_ptr;
    ++t_tangent_aug_lambda_ptr;
    ++t_aug_lambda_ptr;
  }

  CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::
    OpCalAugmentedTangentialContConditionLambdaNormSlaveSlave::doWork(
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
    int nb_base_fun_col = col_data.getFieldData().size() / 3;

    const double area_master =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    // auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    //   return FTensor::Tensor1<double *, 2>(&m(r + 0, c + 0), &m(r + 1, c + 0));
    // };

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 3, 3>(
          &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2),
          &m(r + 1, c + 0), &m(r + 1, c + 1), &m(r + 1, c + 2),
          &m(r + 2, c + 0), &m(r + 2, c + 1), &m(r + 2, c + 2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
    NN.clear();

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<2>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_norm_tang_aug_lambda_ptr = getFTensor0FromVec(
        *commonDataSimpleContact->normAugTangentLambdasPtr);

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_lambda =
        getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

    auto t_tangent_1_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr));

    auto t_tangent_2_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr));
    auto normal =
        get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      double val_m = t_w * area_master;
      auto t_base_row_lambda = row_data.getFTensor0N(gg, 0);

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        auto t_base_lambda = col_data.getFTensor0N(gg, 0);
        const double m = val_m * t_base_row_lambda;
        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
          const double n = m * t_base_lambda;
          auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

          if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
            // t_assemble_m(0, 0) -= n / cNTangentPtr;
            // //++t_assemble_m;
            // t_assemble_m(1, 1) -= n / cNTangentPtr;
          } else {
            double stick_slip_check =
                t_norm_tang_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

            bool stick = true;
            if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
              stick = false;

            if (stick) {
              // t_assemble_m(0) += n ;
              // ++t_assemble_m;
              // t_assemble_m(1) += n ;

            } else {
              t_assemble_m(i, j) -= n * muTan * t_tangent_aug_lambda_ptr(0) *
                                    t_tangent_1_at_gp(i) * normal(j) /
                                    (cNTangentPtr * t_norm_tang_aug_lambda_ptr);

              t_assemble_m(i, j) -= n * muTan * t_tangent_aug_lambda_ptr(1) *
                                    t_tangent_2_at_gp(i) * normal(j) /
                                    (cNTangentPtr * t_norm_tang_aug_lambda_ptr);

              // t_assemble_m(i, j) -= n * muTan * t_tangent_aug_lambda_ptr(0) *
              //                       t_tangent_2_at_gp(i) * normal(j) /
              //                       (cNTangentPtr * t_norm_tang_aug_lambda_ptr);

              // t_assemble_m(i, j) -= n * muTan * t_tangent_aug_lambda_ptr(1) *
              //                       t_tangent_1_at_gp(i) * normal(j) /
              //                       (cNTangentPtr * t_norm_tang_aug_lambda_ptr);
            }
          }
          ++t_base_lambda; // update cols slave
        }
        ++t_base_row_lambda; // update rows master
      }
      ++t_w;
      ++t_norm_tang_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_aug_lambda_ptr;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalAugmentedTangentialContConditionLambdaTanSlaveSlave::
    doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
           EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {
    const int nb_gauss_pts = row_data.getN().size1();

    int nb_base_fun_row = row_data.getFieldData().size() / 3;
    int nb_base_fun_col = col_data.getFieldData().size() / 3;

    const double area_master =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    // auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    //   return FTensor::Tensor2<double *, 2, 2>(
    //       &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 1, c + 0),
    //       &m(r + 1, c + 1));
    // };

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 3, 3>(
          &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2),
          &m(r + 1, c + 0), &m(r + 1, c + 1), &m(r + 1, c + 2),
          &m(r + 2, c + 0), &m(r + 2, c + 1), &m(r + 2, c + 2));
    };

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
    NN.clear();

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<2>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_norm_tang_aug_lambda_ptr = getFTensor0FromVec(
        *commonDataSimpleContact->normAugTangentLambdasPtr);

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_lambda =
        getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

    auto t_tangent_1_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorSlavePtr));

    auto t_tangent_2_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorSlavePtr));

    auto normal =
        get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      double val_m = t_w * area_master;
      auto t_base_row_lambda = row_data.getFTensor0N(gg, 0);

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        auto t_base_lambda = col_data.getFTensor0N(gg, 0);
        const double m = val_m * t_base_row_lambda;
        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
          const double n = m * t_base_lambda;
          auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

          if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
            t_assemble_m(i, j) -=
                n * t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) / cNTangentPtr;

            t_assemble_m(i, j) -=
                n * t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) / cNTangentPtr;

            // t_assemble_m(i, j) -=
            //     n * t_tangent_2_at_gp(i) * t_tangent_1_at_gp(j) / cNTangentPtr;

            // t_assemble_m(i, j) -=
            //     n * t_tangent_1_at_gp(i) * t_tangent_2_at_gp(j) / cNTangentPtr;
          } else {
            double stick_slip_check =
                t_norm_tang_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

            bool stick = true;
            if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
              stick = false;

            if (stick) {
              // cerr << "stick\n";
              // t_assemble_m(0) += n ;
              // ++t_assemble_m;
              // t_assemble_m(1) += n ;

            } else {
              // cerr << "slip\n";

              const double pow_3 = t_norm_tang_aug_lambda_ptr *
                                   t_norm_tang_aug_lambda_ptr *
                                   t_norm_tang_aug_lambda_ptr;

              t_assemble_m(i, j) -=
                  n * t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) / cNTangentPtr;

              t_assemble_m(i, j) -=
                  n * t_tangent_2_at_gp(i) * t_tangent_2_at_gp(j) / cNTangentPtr;

              t_assemble_m(i, j) -= n * muTan * t_aug_lambda_ptr *
                                    t_tangent_1_at_gp(i) *
                                    t_tangent_1_at_gp(j) /
                                    (t_norm_tang_aug_lambda_ptr * cNTangentPtr);

              t_assemble_m(i, j) -= n * muTan * t_aug_lambda_ptr *
                                    t_tangent_2_at_gp(i) *
                                    t_tangent_2_at_gp(j) /
                                    (t_norm_tang_aug_lambda_ptr * cNTangentPtr);

              t_assemble_m(i, j) += n * muTan * t_aug_lambda_ptr *
                                    t_tangent_aug_lambda_ptr(0) *
                                    t_tangent_aug_lambda_ptr(0) *
                                    t_tangent_1_at_gp(i) * t_tangent_1_at_gp(j) /
                                    (pow_3 * cNTangentPtr);

              t_assemble_m(i, j) +=
                  n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(0) *
                  t_tangent_aug_lambda_ptr(1) * t_tangent_1_at_gp(i) *
                  t_tangent_2_at_gp(j) / (pow_3 * cNTangentPtr);

              t_assemble_m(i, j) +=
                  n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(1) *
                  t_tangent_aug_lambda_ptr(1) * t_tangent_2_at_gp(i) *
                  t_tangent_2_at_gp(j) / (pow_3 * cNTangentPtr);

              t_assemble_m(i, j) +=
                  n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(0) *
                  t_tangent_aug_lambda_ptr(1) * t_tangent_2_at_gp(i) *
                  t_tangent_1_at_gp(j) / (pow_3 * cNTangentPtr);

              //
              // t_assemble_m(i, j) -= n * t_tangent_2_at_gp(i) *
              //                       t_tangent_1_at_gp(j) / cNTangentPtr;

              // t_assemble_m(i, j) -= n * t_tangent_1_at_gp(i) *
              //                       t_tangent_2_at_gp(j) / cNTangentPtr;

              // t_assemble_m(i, j) -= n * muTan * t_aug_lambda_ptr *
              //                       t_tangent_2_at_gp(i) *
              //                       t_tangent_1_at_gp(j) /
              //                       (t_norm_tang_aug_lambda_ptr * cNTangentPtr);

              // t_assemble_m(i, j) -= n * muTan * t_aug_lambda_ptr *
              //                       t_tangent_1_at_gp(i) *
              //                       t_tangent_2_at_gp(j) /
              //                       (t_norm_tang_aug_lambda_ptr * cNTangentPtr);

              // t_assemble_m(i, j) +=
              //     n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(0) *
              //     t_tangent_aug_lambda_ptr(0) * t_tangent_2_at_gp(i) *
              //     t_tangent_1_at_gp(j) / (pow_3 * cNTangentPtr);

              // t_assemble_m(i, j) +=
              //     n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(0) *
              //     t_tangent_aug_lambda_ptr(1) * t_tangent_2_at_gp(i) *
              //     t_tangent_2_at_gp(j) / (pow_3 * cNTangentPtr);

              // t_assemble_m(i, j) +=
              //     n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(1) *
              //     t_tangent_aug_lambda_ptr(1) * t_tangent_1_at_gp(i) *
              //     t_tangent_2_at_gp(j) / (pow_3 * cNTangentPtr);

              // t_assemble_m(i, j) +=
              //     n * muTan * t_aug_lambda_ptr * t_tangent_aug_lambda_ptr(0) *
              //     t_tangent_aug_lambda_ptr(1) * t_tangent_1_at_gp(i) *
              //     t_tangent_1_at_gp(j) / (pow_3 * cNTangentPtr);

            }
          }
          ++t_base_lambda; // update cols slave
        }
        ++t_base_row_lambda; // update rows master
      }
      ++t_w;
      ++t_norm_tang_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_aug_lambda_ptr;
    }
    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalContactAugmentedTractionOverSpatialMasterMaster::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto t_w = getFTensor0IntegrationWeightSlave();

  const double area_master = commonDataSimpleContact->areaSlave;

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
      // cerr << "Pass Right\n";

      ++t_w;
      ++t_aug_lambda_ptr;
      continue;
    }

    const double val_m = t_w * area_master * cN;

    FTensor::Tensor0<double *> t_base_master_col(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base_master_row(&row_data.getN()(gg, 0));
      const double m = val_m * t_base_master_col;

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {

        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

        t_assemble_s(i, j) += m * normal(i) * normal(j) * t_base_master_row;

        ++t_base_master_row; // update rows
      }
      ++t_base_master_col; // update cols slave
    }
    ++t_w;
    ++t_aug_lambda_ptr;
  }

  CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalContactAugmentedTractionOverSpatialMasterSlave::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto t_w = getFTensor0IntegrationWeightSlave();

  const double area_master = commonDataSimpleContact->areaSlave;

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);
  
  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
      ++t_w;
      ++t_aug_lambda_ptr;
      // cerr << "Pass Wrong\n";
      continue;
    }

    const double val_m = t_w * area_master * cN;

    FTensor::Tensor0<double *> t_base_slave_col(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base_master_row(&row_data.getN()(gg, 0));
      const double m = val_m * t_base_slave_col;

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {

        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

        t_assemble_s(i, j) -= m * normal(i) * normal(j) * t_base_master_row;

        ++t_base_master_row; // update rows
      }
      ++t_base_slave_col; // update cols slave
    }
    ++t_w;
    ++t_aug_lambda_ptr;
  }

  CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalContactAugmentedTractionOverSpatialSlaveSlave::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto t_w = getFTensor0IntegrationWeightSlave();

  const double area_slave = commonDataSimpleContact->areaSlave;

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
      ++t_w;
      ++t_aug_lambda_ptr;
      continue;
    }

    const double val_s = t_w * area_slave * cN;

    FTensor::Tensor0<double *> t_base_slave_col(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base_slave_row(&row_data.getN()(gg, 0));
      const double s = val_s * t_base_slave_col;

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {

        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

        t_assemble_s(i, j) += s * normal(i) * normal(j) * t_base_slave_row;

        ++t_base_slave_row; // update rows
      }
      ++t_base_slave_col; // update cols slave
    }
    ++t_w;
    ++t_aug_lambda_ptr;
  }

  CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalContactAugmentedTractionOverSpatialSlaveMaster::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto t_w = getFTensor0IntegrationWeightSlave();

  const double area_slave = commonDataSimpleContact->areaSlave;

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
      ++t_w;
      ++t_aug_lambda_ptr;
      continue;
    }

    const double val_s = t_w * area_slave * cN;

    FTensor::Tensor0<double *> t_base_master_col(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base_slave_row(&row_data.getN()(gg, 0));
      const double s = val_s * t_base_master_col;

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {

        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

        t_assemble_s(i, j) -= s * normal(i) * normal(j) * t_base_slave_row;

        ++t_base_slave_row; // update rows
      }
      ++t_base_master_col; // update cols slave
    }
    ++t_w;
    ++t_aug_lambda_ptr;
  }

  CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                      ADD_VALUES);

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

    CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
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

    CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
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

MoFEMErrorCode SimpleContactProblem::OpGapConstraintOnLambda::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  int nb_base_fun_col = data.getFieldData().size() / 3;
  const double area_s =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  vecR.resize(3 * nb_base_fun_col, false);
  vecR.clear();

  auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

  auto t_lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
  auto t_gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);
  auto t_w = getFTensor0IntegrationWeightSlave();

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  FTensor::Index<'i', 3> i;

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    double branch_gg;
    if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
      branch_gg = -t_lagrange_slave / cN;
    } else {
      branch_gg = t_gap_gp;
    }

    const double val_s = t_w * area_s * branch_gg;
    auto t_base_lambda = data.getFTensor0N(gg, 0);
    for (int bbr = 0; bbr != nb_base_fun_col; ++bbr) {
      auto t_assemble_m = get_tensor_vec(vecR, 3 * bbr);
      t_assemble_m(i) += val_s * t_base_lambda * const_unit_n(i);

      ++t_base_lambda; // update rows
    }

    ++t_lagrange_slave;
    ++t_gap_gp;
    ++t_aug_lambda_ptr;
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

MoFEMErrorCode SimpleContactProblem::OpCalAugmentedTractionRhsSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_dofs = data.getIndices().size();
  if (nb_dofs) {
    const int nb_gauss_pts = data.getN().size1();
    int nb_base_fun_col = data.getFieldData().size() / 3;

    vecF.resize(nb_dofs, false);
    vecF.clear();

    const double area_s =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    FTensor::Index<'i', 3> i;

    auto const_unit_n = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto t_w = getFTensor0IntegrationWeightSlave();
    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
        ++t_aug_lambda_ptr;
        ++t_w;
        continue;
      }
      const double val_s = t_w * area_s;

      FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));

      for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

        auto t_assemble_s = get_tensor_vec(vecF, 3 * bbc);

        t_assemble_s(i) -=
            val_s * const_unit_n(i) * t_aug_lambda_ptr * t_base_slave;

        ++t_base_slave;
      }
      ++t_aug_lambda_ptr;
      ++t_w;
    } // for gauss points

    CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalAugmentedTractionRhsMaster::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    const int nb_gauss_pts = data.getN().size1();
    const int nb_base_fun_col = data.getFieldData().size() / 3;

    vecF.resize(3 * nb_base_fun_col,
                false); // the last false in ublas
                        // resize will destroy (not
                        // preserved) the old
                        // values
    vecF.clear();

    const double area_m =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    FTensor::Index<'i', 3> i;

    auto const_unit_n = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto t_w = getFTensor0IntegrationWeightSlave();
    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
        ++t_aug_lambda_ptr;
        ++t_w;
        continue;
      }
      const double val_m = t_w * area_m;

      FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

      for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

        auto t_assemble_m = get_tensor_vec(vecF, 3 * bbc);

        t_assemble_m(i) +=
            val_m * const_unit_n(i) * t_aug_lambda_ptr * t_base_master;

        ++t_base_master;
      }
      ++t_aug_lambda_ptr;
      ++t_w;
    } // for gauss points

    CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalAugmentedTangentTractionsRhsMaster::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    const int nb_gauss_pts = data.getN().size1();
    const int nb_base_fun_col = data.getFieldData().size() / 3;

    vecF.resize(nb_dofs,
                false); // the last false in ublas
                        // resize will destroy (not
                        // preserved) the old
                        // values
    vecF.clear();

    const double area_m =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    FTensor::Index<'i', 3> i;

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<2>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_tangent_gap =
        getFTensor1FromMat<2>(*commonDataSimpleContact->tangentGapPtr);

    auto t_norm_aug_lambda_ptr = getFTensor0FromVec(
        *commonDataSimpleContact->normAugTangentLambdasPtr);

    auto t_tangent_lambda =
        getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

    auto t_tangent_1_at_gp = get_tensor_vec(
        *(commonDataSimpleContact->tangentOneVectorSlavePtr), 0);

    auto t_tangent_2_at_gp = get_tensor_vec(
        *(commonDataSimpleContact->tangentTwoVectorSlavePtr), 0);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
        // cerr << " RHS NO CONTACT!\n";
        ++t_aug_lambda_ptr;
        ++t_tangent_aug_lambda_ptr;
        ++t_tangent_gap;
        ++t_norm_aug_lambda_ptr;
        ++t_tangent_lambda;
        ++t_w;
        continue;
      }

      double stick_slip_check = t_norm_aug_lambda_ptr + muTan * t_aug_lambda_ptr;
      bool stick = true;
      if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
        stick = false;

      // cerr << "t_norm_aug_lambda_ptr > -muTan * t_aug_lambda_ptr"
      //      << t_norm_aug_lambda_ptr
      //     + muTan * t_aug_lambda_ptr <<"\n";

      const double val_m = t_w * area_m;

      FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

      for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

        auto t_assemble_m = get_tensor_vec(vecF, 3 * bbc);

        if (stick) { // stick

          // cerr << "t_norm_aug_lambda_ptr "
          //      << t_norm_aug_lambda_ptr
          //     <<"\n";

          // cerr << "muTan * t_aug_lambda_ptr  " << muTan * t_aug_lambda_ptr
          //      << "\n";

          // cerr << "RHS   stick "<< stick_slip_check <<"\n ";
              t_assemble_m(i) -=
              val_m * t_tangent_1_at_gp(i) *
              (t_tangent_aug_lambda_ptr(0) /*+ t_tangent_aug_lambda_ptr(1)*/) *
              t_base_master;

          t_assemble_m(i) -=
              val_m * t_tangent_2_at_gp(i) *
              (/*t_tangent_aug_lambda_ptr(0)*/ + t_tangent_aug_lambda_ptr(1)) *
              t_base_master;

        } else { // slip
          // cerr << "RHS   slip\n";

          t_assemble_m(i) +=
              val_m * t_tangent_1_at_gp(i) * t_tangent_aug_lambda_ptr(0) *
              muTan * t_aug_lambda_ptr * t_base_master / t_norm_aug_lambda_ptr;

          t_assemble_m(i) +=
              val_m * t_tangent_2_at_gp(i) * t_tangent_aug_lambda_ptr(1) *
              muTan * t_aug_lambda_ptr * t_base_master / t_norm_aug_lambda_ptr;

          // t_assemble_m(i) +=
          //     val_m * t_tangent_2_at_gp(i) * t_tangent_aug_lambda_ptr(0) *
          //     muTan * t_aug_lambda_ptr * t_base_master / t_norm_aug_lambda_ptr;

          // t_assemble_m(i) +=
          //     val_m * t_tangent_1_at_gp(i) * t_tangent_aug_lambda_ptr(1) *
          //     muTan * t_aug_lambda_ptr * t_base_master / t_norm_aug_lambda_ptr;
        }

        ++t_base_master;
      }

      ++t_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_tangent_gap;
      ++t_norm_aug_lambda_ptr;
      ++t_tangent_lambda;
      ++t_w;
    } // for gauss points

    CHKERR VecSetValues(getSNESf(), data, &vecF[0], ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalAugmentedTangentTractionsRhsSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    const int nb_gauss_pts = data.getN().size1();
    const int nb_base_fun_col = data.getFieldData().size() / 3;

    vecF.resize(nb_dofs,
                false); // the last false in ublas
                        // resize will destroy (not
                        // preserved) the old
                        // values
    vecF.clear();

    const double area_m =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    FTensor::Index<'i', 3> i;

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<2>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_tangent_gap =
        getFTensor1FromMat<2>(*commonDataSimpleContact->tangentGapPtr);

    auto t_norm_aug_lambda_ptr = getFTensor0FromVec(
        *commonDataSimpleContact->normAugTangentLambdasPtr);

    auto t_tangent_lambda =
        getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

    auto t_tangent_1_at_gp = get_tensor_vec(
        *(commonDataSimpleContact->tangentOneVectorSlavePtr), 0);

    auto t_tangent_2_at_gp = get_tensor_vec(
        *(commonDataSimpleContact->tangentTwoVectorSlavePtr), 0);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
        ++t_aug_lambda_ptr;
        ++t_tangent_aug_lambda_ptr;
        ++t_tangent_gap;
        ++t_norm_aug_lambda_ptr;
        ++t_tangent_lambda;
        ++t_w;
        continue;
      }

      double stick_slip_check = t_norm_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

      bool stick = true;
      if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
        stick = false;

      const double val_m = t_w * area_m;

      FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));

      for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

        auto t_assemble_m = get_tensor_vec(vecF, 3 * bbc);

        if (stick) { // stick
          // cerr << "stick\n";
          t_assemble_m(i) += val_m * t_tangent_1_at_gp(i) *
                             t_tangent_aug_lambda_ptr(0) * t_base_slave;

          t_assemble_m(i) += val_m * t_tangent_2_at_gp(i) *
                             t_tangent_aug_lambda_ptr(1) * t_base_slave;

          // t_assemble_m(i) += val_m * t_tangent_2_at_gp(i) *
          //                    t_tangent_aug_lambda_ptr(0) * t_base_slave;

          // t_assemble_m(i) += val_m * t_tangent_1_at_gp(i) *
          //                    t_tangent_aug_lambda_ptr(1) * t_base_slave;

        } else { // slip
          // cerr << "slip\n";

          t_assemble_m(i) -=
              val_m * t_tangent_1_at_gp(i) * t_tangent_aug_lambda_ptr(0) *
              muTan * t_aug_lambda_ptr * t_base_slave / t_norm_aug_lambda_ptr;

          t_assemble_m(i) -=
              val_m * t_tangent_2_at_gp(i) * t_tangent_aug_lambda_ptr(1) *
              muTan * t_aug_lambda_ptr * t_base_slave / t_norm_aug_lambda_ptr;

          // t_assemble_m(i) -=
          //     val_m * t_tangent_2_at_gp(i) * t_tangent_aug_lambda_ptr(0) *
          //     muTan * t_aug_lambda_ptr * t_base_slave / t_norm_aug_lambda_ptr;

          // t_assemble_m(i) -=
          //     val_m * t_tangent_1_at_gp(i) * t_tangent_aug_lambda_ptr(1) *
          //     muTan * t_aug_lambda_ptr * t_base_slave / t_norm_aug_lambda_ptr;
        }

        ++t_base_slave;
      }

      ++t_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_tangent_gap;
      ++t_norm_aug_lambda_ptr;
      ++t_tangent_lambda;
      ++t_w;
    } // for gauss points

    CHKERR VecSetValues(getSNESf(), data, &vecF[0], ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalAugmentedTangentialContCondition::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    const int nb_gauss_pts = data.getN().size1();
    const int nb_base_fun_col = data.getFieldData().size() / 3;

    vecF.resize(nb_dofs,
                false); // the last false in ublas
                        // resize will destroy (not
                        // preserved) the old
                        // values
    vecF.clear();

    const double area_m =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto get_tensor_vec_2 = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 2>(&n(r + 0), &n(r + 1));
    };

    // FTensor::Index<'i', 2> i;

    FTensor::Index<'j', 3> j;

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<2>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_tangent_gap =
        getFTensor1FromMat<2>(*commonDataSimpleContact->tangentGapPtr);

    auto t_norm_aug_lambda_ptr = getFTensor0FromVec(
        *commonDataSimpleContact->normAugTangentLambdasPtr);

    auto t_tangent_lambda =
        getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

    auto t_tangent_1_at_gp = get_tensor_vec(
        *(commonDataSimpleContact->tangentOneVectorSlavePtr), 0);

    auto t_tangent_2_at_gp = get_tensor_vec(
        *(commonDataSimpleContact->tangentTwoVectorSlavePtr), 0);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      double stick_slip_check = t_norm_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

      bool stick = true;
      if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
        stick = false;

      // cerr << "t_norm_aug_lambda_ptr + muTan * t_aug_lambda_ptr "
      //      << t_norm_aug_lambda_ptr + muTan * t_aug_lambda_ptr
      //      << "   t_aug_lambda_ptr " << t_aug_lambda_ptr << "\n";
      const double val_m = t_w * area_m;

      FTensor::Tensor0<double *> t_base_slave_lambda(&data.getN()(gg, 0));

      for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

        auto t_assemble_m = get_tensor_vec(vecF, 3 * bbc);
        // cerr << "t_aug_lambda_ptr   " << t_aug_lambda_ptr << "\n";
        if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
          // cerr << "NO CONTACT " <<"\n";
          t_assemble_m(j) -= val_m * t_tangent_lambda(0) * t_tangent_1_at_gp(j) *
                             t_base_slave_lambda / cNTangentPtr;

          t_assemble_m(j) -= val_m * t_tangent_lambda(1) *
                             t_tangent_2_at_gp(j) * t_base_slave_lambda /
                             cNTangentPtr;

          // t_assemble_m(j) -= val_m * t_tangent_lambda(0) *
          //                    t_tangent_2_at_gp(j) * t_base_slave_lambda /
          //                    cNTangentPtr;

          // t_assemble_m(j) -= val_m * t_tangent_lambda(1) *
          //                    t_tangent_1_at_gp(j) * t_base_slave_lambda /
          //                    cNTangentPtr;

        } else {
          if (stick) { // stick
            // cerr << "t_tangent_gap " << t_tangent_gap
           
            t_assemble_m(j) += val_m *
                               t_tangent_gap(0) * t_tangent_1_at_gp(j) *
                               t_base_slave_lambda;

            t_assemble_m(j) += val_m * t_tangent_gap(1) * t_tangent_2_at_gp(j) *
                               t_base_slave_lambda;

            // t_assemble_m(j) += val_m * t_tangent_gap(1) * t_tangent_1_at_gp(j) *
            //                    t_base_slave_lambda;

            // t_assemble_m(j) += val_m * t_tangent_gap(0) * t_tangent_2_at_gp(j) *
            //                    t_base_slave_lambda;

          } else { // slip

            
            ///======

            // cerr << "t_tangent_lambda " << t_tangent_lambda << "\n";
            t_assemble_m(j) -=
                val_m *
                (t_tangent_lambda(0) + t_tangent_aug_lambda_ptr(0) * muTan *
                                           t_aug_lambda_ptr /
                                           t_norm_aug_lambda_ptr) *
                t_tangent_1_at_gp(j) * t_base_slave_lambda / cNTangentPtr;

            t_assemble_m(j) -=
                val_m *
                (t_tangent_lambda(1) + t_tangent_aug_lambda_ptr(1) * muTan *
                                           t_aug_lambda_ptr /
                                           t_norm_aug_lambda_ptr) *
                t_tangent_2_at_gp(j) * t_base_slave_lambda / cNTangentPtr;

            // t_assemble_m(j) -=
            //     val_m *
            //     (t_tangent_lambda(0) + t_tangent_aug_lambda_ptr(0) * muTan *
            //                                t_aug_lambda_ptr /
            //                                t_norm_aug_lambda_ptr) *
            //     t_tangent_2_at_gp(j) * t_base_slave_lambda / cNTangentPtr;

            // t_assemble_m(j) -=
            //     val_m *
            //     (t_tangent_lambda(1) + t_tangent_aug_lambda_ptr(1) * muTan *
            //                                t_aug_lambda_ptr /
            //                                t_norm_aug_lambda_ptr) *
            //     t_tangent_1_at_gp(j) * t_base_slave_lambda / cNTangentPtr;

          }
        }

        ++t_base_slave_lambda;
      }

      ++t_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_tangent_gap;
      ++t_norm_aug_lambda_ptr;
      ++t_tangent_lambda;
      ++t_w;
    } // for gauss points

    CHKERR VecSetValues(getSNESf(), data, &vecF[0], ADD_VALUES);
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

  FTensor::Index<'i', 2> i;

  Tag th_tan_lag;
  if (isTangLagrange)
    CHKERR moabOut.tag_get_handle("TANGENT_LAGRANGE_MAG", 1, MB_TYPE_DOUBLE,
                                  th_tan_lag, MB_TAG_CREAT | MB_TAG_SPARSE,
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

  auto t_tangent_lagrange =
      getFTensor1FromMat<2>(*commonDataSimpleContact->tangentLambdasPtr);

  std::array<double, 3> pos_vec;

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    const double *slave_coords_ptr = &(getCoordsAtGaussPtsSlave()(gg, 0));
    CHKERR moabOut.create_vertex(slave_coords_ptr, new_vertex);
    CHKERR moabOut.tag_set_data(th_gap, &new_vertex, 1, &t_gap_ptr);

    CHKERR moabOut.tag_set_data(th_lag_gap_prod, &new_vertex, 1,
                                &t_lag_gap_prod_slave);
    CHKERR moabOut.tag_set_data(th_lag_mult, &new_vertex, 1, &t_lagrange_slave);
  
    if (isTangLagrange){
    double mag_tan_lag = sqrt(t_tangent_lagrange(i) * t_tangent_lagrange(i));
    CHKERR moabOut.tag_set_data(th_tan_lag, &new_vertex, 1, &mag_tan_lag);
    }
   
    auto get_vec_ptr = [&](auto t) {
      for (int dd = 0; dd != 3; ++dd)
        pos_vec[dd] = t(dd);
      return pos_vec.data();
    };

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
    

    if (isTangLagrange)
      ++t_tangent_lagrange;
      
    
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

MoFEMErrorCode SimpleContactProblem::setContactOperatorsRhs(
    boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrange_field_name) {
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
      new OpGetLagMulAtGaussPtsSlave(lagrange_field_name,
                                     common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGetGaussPtsState(
      lagrange_field_name, common_data_simple_contact, cnValue));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpCalContactTractionOnSlave(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(new OpCalIntCompFunSlave(
      lagrange_field_name, common_data_simple_contact, cnValue));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactAugmentedOperatorsRhs(
    boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrang_field_name) {
  MoFEMFunctionBegin;

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalSlave(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalMaster(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsMaster(field_name,
                                        common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsSlave(field_name,
                                       common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetGapSlave(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                     common_data_simple_contact));

  // fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGetGaussPtsState(
  //     lagrang_field_name, common_data_simple_contact, cnValue, true));

  // augmented

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetAugmentedLambdaSlave(field_name, common_data_simple_contact,
                                    cnValue));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpCalAugmentedTractionRhsSlave(field_name,
                                         common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpCalAugmentedTractionRhsMaster(field_name,
                                          common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGapConstraintOnLambda(
      lagrang_field_name, common_data_simple_contact, cnValue));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactAugmentedOperatorsLhs(
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
      new OpGetPositionAtGaussPtsSlave(field_name,
                                       common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetGapSlave(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                     common_data_simple_contact));

  // augmented stuff

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetAugmentedLambdaSlave(field_name, common_data_simple_contact,
                                    cnValue));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalContactAugmentedTractionOverLambdaMasterSlave(
          field_name, lagrang_field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalContactAugmentedTractionOverLambdaSlaveSlave(
          field_name, lagrang_field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalContactAugmentedTractionOverSpatialMasterMaster(field_name, field_name, cnValue,
                                             common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalContactAugmentedTractionOverSpatialMasterSlave(field_name, field_name, cnValue,
                                            common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalContactAugmentedTractionOverSpatialSlaveSlave(field_name, field_name, cnValue,
                                           common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalContactAugmentedTractionOverSpatialSlaveMaster(field_name, field_name, cnValue,
                                            common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGapConstraintOverLambda(
          lagrang_field_name, common_data_simple_contact, cnValue));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGapConstraintOverSpatialMaster(field_name, lagrang_field_name,
                                             common_data_simple_contact,
                                             cnValue));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGapConstraintOverSpatialSlave(field_name, lagrang_field_name,
                                            common_data_simple_contact,
                                            cnValue));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactFrictionAugmentedOperatorsRhs(
    boost::shared_ptr<SimpleContactElement> fe_rhs_extended_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrang_field_name,
    string tangent_lagrang_field_name, string previously_converged_spat) {
  MoFEMFunctionBegin;

  fe_rhs_extended_contact->getOpPtrVector().push_back(
      new OpGetNormalSlave(field_name, common_data_simple_contact));

  fe_rhs_extended_contact->getOpPtrVector().push_back(
      new OpGetNormalMaster(field_name, common_data_simple_contact));

  fe_rhs_extended_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsMaster(field_name,
                                        common_data_simple_contact));

  fe_rhs_extended_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsSlave(field_name,
                                       common_data_simple_contact));

  fe_rhs_extended_contact->getOpPtrVector().push_back(
      new OpGetGapSlave(field_name, common_data_simple_contact));

  fe_rhs_extended_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                     common_data_simple_contact));

  fe_rhs_extended_contact->getOpPtrVector().push_back(
      new OpGetAugmentedLambdaSlave(field_name, common_data_simple_contact,
                                    cnValue));

  fe_rhs_extended_contact->getOpPtrVector().push_back(
      new OpGetOrthonormalTangents(field_name, common_data_simple_contact));

  fe_rhs_extended_contact->getOpPtrVector().push_back(
      new OpGetPreviousPositionAtGaussPtsSlave(previously_converged_spat,
                                               common_data_simple_contact));

  fe_rhs_extended_contact->getOpPtrVector().push_back(
      new OpGetPreviousPositionAtGaussPtsMaster(previously_converged_spat,
                                                common_data_simple_contact));

  fe_rhs_extended_contact->getOpPtrVector().push_back(
      new OpGetTangentGapVelocitySlave(field_name,
                                       common_data_simple_contact));

  fe_rhs_extended_contact->getOpPtrVector().push_back(new OpGetTangentLagrange(
      lagrang_field_name, common_data_simple_contact));

  fe_rhs_extended_contact->getOpPtrVector().push_back(
      new OpGetTangentAugmentedLambdaSlave(lagrang_field_name,
                                           common_data_simple_contact,
                                           cTangentValue));

  fe_rhs_extended_contact->getOpPtrVector().push_back(
      new OpCalAugmentedTangentTractionsRhsMaster(
          field_name, common_data_simple_contact, muTangent));

  fe_rhs_extended_contact->getOpPtrVector().push_back(
      new OpCalAugmentedTangentTractionsRhsSlave(field_name,
                                                 common_data_simple_contact,
                                                 muTangent));

  fe_rhs_extended_contact->getOpPtrVector().push_back(
      new OpCalAugmentedTangentialContCondition(lagrang_field_name,
                                                common_data_simple_contact,
                                                muTangent, cTangentValue));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactFrictionAugmentedOperatorsLhs(
    boost::shared_ptr<SimpleContactElement> fe_lhs_extended_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrang_field_name,
    string tangent_lagrang_field_name, string previously_converged_spat) {
  MoFEMFunctionBegin;

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpGetNormalSlave(field_name, common_data_simple_contact));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpGetNormalMaster(field_name, common_data_simple_contact));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsMaster(field_name,
                                        common_data_simple_contact));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsSlave(field_name,
                                       common_data_simple_contact));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpGetGapSlave(field_name, common_data_simple_contact));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                     common_data_simple_contact));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpGetAugmentedLambdaSlave(field_name, common_data_simple_contact,
                                    cnValue));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpGetOrthonormalTangents(field_name, common_data_simple_contact));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpGetPreviousPositionAtGaussPtsSlave(previously_converged_spat,
                                               common_data_simple_contact));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpGetPreviousPositionAtGaussPtsMaster(previously_converged_spat,
                                                common_data_simple_contact));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpGetTangentGapVelocitySlave(field_name,
                                       common_data_simple_contact));

  fe_lhs_extended_contact->getOpPtrVector().push_back(new OpGetTangentLagrange(
      lagrang_field_name, common_data_simple_contact));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpGetTangentAugmentedLambdaSlave(lagrang_field_name,
                                           common_data_simple_contact,
                                           cTangentValue));

  //
  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpContactAugmentedFrictionMasterMaster(
          field_name, field_name, common_data_simple_contact, muTangent,
          cTangentValue, cnValue));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpContactAugmentedFrictionMasterSlave(
          field_name, field_name, common_data_simple_contact, muTangent,
          cTangentValue, cnValue));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpCalContactAugmentedTangentLambdaOverLambdaMasterSlave(
          field_name, lagrang_field_name, common_data_simple_contact,
          muTangent));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpCalContactAugmentedTangentLambdaOverLambdaTanMasterSlave(
          field_name, lagrang_field_name, common_data_simple_contact,
          muTangent));

  //

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpContactAugmentedFrictionSlaveSlave(
          field_name, field_name, common_data_simple_contact, muTangent,
          cTangentValue, cnValue));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpContactAugmentedFrictionSlaveMaster(
          field_name, field_name, common_data_simple_contact, muTangent,
          cTangentValue, cnValue));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpCalContactAugmentedTangentLambdaOverLambdaSlaveSlave(
          field_name, lagrang_field_name, common_data_simple_contact,
          muTangent));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpCalContactAugmentedTangentLambdaOverLambdaTanSlaveSlave(
          field_name, lagrang_field_name, common_data_simple_contact,
          muTangent));

  //

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpCalAugmentedTangentialContConditionDispSlaveSlave(
          lagrang_field_name, field_name, common_data_simple_contact,
          muTangent, cTangentValue, cnValue));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpCalAugmentedTangentialContConditionDispSlaveMaster(
          lagrang_field_name, field_name, common_data_simple_contact,
          muTangent, cTangentValue, cnValue));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpCalAugmentedTangentialContConditionLambdaNormSlaveSlave(
          lagrang_field_name, lagrang_field_name,
          common_data_simple_contact, muTangent, cTangentValue));

  fe_lhs_extended_contact->getOpPtrVector().push_back(
      new OpCalAugmentedTangentialContConditionLambdaTanSlaveSlave(
          lagrang_field_name, lagrang_field_name,
          common_data_simple_contact, muTangent, cTangentValue));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setMasterForceOperatorsRhs(
    boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrange_field_name) {
  MoFEMFunctionBegin;

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalSlave(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalMaster(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrange_field_name,
                                     common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpCalContactTractionOnMaster(field_name, common_data_simple_contact));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactOperatorsLhs(
    boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrange_field_name) {
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
      new OpGetLagMulAtGaussPtsSlave(lagrange_field_name,
                                     common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalContactTractionOverLambdaSlaveSlave(
          field_name, lagrange_field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalDerIntCompFunOverLambdaSlaveSlave(
          lagrange_field_name, common_data_simple_contact, cnValue));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalDerIntCompFunOverSpatPosSlaveMaster(
          field_name, lagrange_field_name, common_data_simple_contact,
          cnValue));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalDerIntCompFunOverSpatPosSlaveSlave(
          field_name, lagrange_field_name, common_data_simple_contact,
          cnValue));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setMasterForceOperatorsLhs(
    boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrange_field_name) {
  MoFEMFunctionBegin;

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalSlave(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalMaster(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrange_field_name,
                                     common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalContactTractionOverLambdaMasterSlave(
          field_name, lagrange_field_name, common_data_simple_contact));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactOperatorsLhs(
    boost::shared_ptr<ConvectMasterContactElement> fe_lhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrange_field_name) {
  MoFEMFunctionBegin;
  CHKERR setContactOperatorsLhs(
      boost::dynamic_pointer_cast<SimpleContactElement>(fe_lhs_simple_contact),
      common_data_simple_contact, field_name, lagrange_field_name);

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalculateGradPositionXi(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpLhsConvectIntegrationPtsConstrainMasterGap(
          lagrange_field_name, field_name, common_data_simple_contact, cnValue,
          ContactOp::FACESLAVESLAVE,
          fe_lhs_simple_contact->getConvectPtr()->getDiffKsiSpatialSlave()));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpLhsConvectIntegrationPtsConstrainMasterGap(
          lagrange_field_name, field_name, common_data_simple_contact, cnValue,
          ContactOp::FACESLAVEMASTER,
          fe_lhs_simple_contact->getConvectPtr()->getDiffKsiSpatialMaster()));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setMasterForceOperatorsLhs(
    boost::shared_ptr<ConvectSlaveContactElement> fe_lhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrange_field_name) {
  MoFEMFunctionBegin;

  CHKERR setMasterForceOperatorsLhs(
      boost::dynamic_pointer_cast<SimpleContactElement>(fe_lhs_simple_contact),
      common_data_simple_contact, field_name, lagrange_field_name);

  fe_lhs_simple_contact->getOpPtrVector().push_back(new OpCalculateGradLambdaXi(
      lagrange_field_name, common_data_simple_contact));

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
    MoFEM::Interface &m_field, string field_name, string lagrange_field_name,
    moab::Interface &moab_out, bool alm_flag,
    bool lagrange_field, bool is_friction) {
  MoFEMFunctionBegin;

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalMaster(field_name, common_data_simple_contact));

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalSlave(field_name, common_data_simple_contact));

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsMaster(field_name,
                                        common_data_simple_contact));

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsSlave(field_name, common_data_simple_contact));

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpGetGapSlave(field_name, common_data_simple_contact));

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrange_field_name,
                                     common_data_simple_contact));

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpLagGapProdGaussPtsSlave(lagrange_field_name,
                                    common_data_simple_contact));

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpGetOrthonormalTangents(field_name, common_data_simple_contact));

  if(is_friction)
    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpGetTangentLagrange("LAGMULT",
                                 common_data_simple_contact));

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpMakeVtkSlave(m_field, field_name, common_data_simple_contact,
                         moab_out, lagrange_field, is_friction));

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpGetGaussPtsState(lagrange_field_name, common_data_simple_contact,
                             cnValue, alm_flag));

  fe_post_proc_simple_contact->getOpPtrVector().push_back(new OpGetContactArea(
      lagrange_field_name, common_data_simple_contact, cnValue, alm_flag));

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

MoFEMErrorCode SimpleContactProblem::OpGetGaussPtsState::doWork(int side,
                                                                EntityType type,
                                                                EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  vecR.resize(CommonDataSimpleContact::LAST_ELEMENT, false);
  vecR.clear();

  auto t_lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
  auto t_gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    vecR[CommonDataSimpleContact::TOTAL] += 1;
    if (!almFlag &&
        SimpleContactProblem::State(cN, t_gap_gp, t_lagrange_slave)) {
      vecR[CommonDataSimpleContact::ACTIVE] += 1;
    }

    if (almFlag &&
        SimpleContactProblem::StateALM(cN, t_gap_gp, t_lagrange_slave)) {
      vecR[CommonDataSimpleContact::ACTIVE] += 1;
    }

    ++t_lagrange_slave;
    ++t_gap_gp;
  } // for gauss points

  constexpr std::array<int, 2> indices = {CommonDataSimpleContact::ACTIVE,
                                          CommonDataSimpleContact::TOTAL};
  CHKERR VecSetValues(commonDataSimpleContact->gaussPtsStateVec, 2,
                      indices.data(), &vecR[0], ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetContactArea::doWork(int side,
                                                              EntityType type,
                                                              EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  int nb_base_fun_col = data.getFieldData().size();
  const double area_s =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  vecR.resize(CommonDataSimpleContact::LAST_ELEMENT, false);
  vecR.clear();

  auto t_lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
  auto t_gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);
  auto t_w = getFTensor0IntegrationWeightSlave();

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    const double val_s = t_w * area_s;
    vecR[CommonDataSimpleContact::TOTAL] += val_s;
    if (SimpleContactProblem::State(cN, t_gap_gp, t_lagrange_slave) &&
        !almFlag) {
      vecR[CommonDataSimpleContact::ACTIVE] += val_s;
    }
    if (SimpleContactProblem::StateALM(cN, t_gap_gp, t_lagrange_slave) &&
        almFlag) {
      vecR[CommonDataSimpleContact::ACTIVE] += val_s;
    }

    ++t_lagrange_slave;
    ++t_gap_gp;
    ++t_w;
  } // for gauss points

  constexpr std::array<int, 2> indices = {
      CommonDataSimpleContact::ACTIVE,
      CommonDataSimpleContact::TOTAL,
  };
  CHKERR VecSetValues(commonDataSimpleContact->contactAreaVec, 2,
                      indices.data(), &vecR[0], ADD_VALUES);

  MoFEMFunctionReturn(0);
}