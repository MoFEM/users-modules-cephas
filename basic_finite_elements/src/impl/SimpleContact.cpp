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

using namespace MoFEM;
using namespace boost::numeric;

#include <IntegrationRules.hpp>

constexpr double SimpleContactProblem::TOL;
constexpr int SimpleContactProblem::LAGRANGE_RANK;
constexpr int SimpleContactProblem::POSITION_RANK;

MoFEMErrorCode
SimpleContactProblem::SimpleContactElement::setGaussPts(int order) {
  MoFEMFunctionBegin;
  if (newtonCotes) {
    int rule = order + 2;
    int nb_gauss_pts = IntRules::NCC::triangle_ncc_order_num(rule);
    gaussPtsMaster.resize(3, nb_gauss_pts, false);
    gaussPtsSlave.resize(3, nb_gauss_pts, false);
    double xy_coords[2 * nb_gauss_pts];
    double w_array[nb_gauss_pts];
    IntRules::NCC::triangle_ncc_rule(rule, nb_gauss_pts, xy_coords, w_array);

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
    MatrixDouble3by3 inv_matA(2, 2);
    VectorDouble3 copy_F(2);
    FTensor::Tensor1<FTensor::PackPtr<double *, 0>, 2> t_copy_F(&copy_F[0],
                                                                &copy_F[1]);
    FTensor::Tensor2<double *, 2, 2> t_inv_matA(
        &inv_matA(0, 0), &inv_matA(0, 1), &inv_matA(1, 0), &inv_matA(1, 1));

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

        auto invert_2_by_2 = [&](MatrixDouble3by3 &inv_mat_A,
                                 MatrixDouble3by3 &mat_A) {
          double det_A;
          CHKERR determinantTensor2by2(mat_A, det_A);
          CHKERR invertTensor2by2(mat_A, det_A, inv_mat_A);
        };

        auto linear_solver = [&]() {
          invert_2_by_2(inv_matA, A);
          t_copy_F(J) = t_f(J);
          t_f(I) = t_inv_matA(I, J) * t_copy_F(J);
        };

        auto invert_A = [&]() { invert_2_by_2(invA, A); };

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

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

    FTensor::Tensor1<double *, 3> t_field_data_master(
        &data.getFieldData()[0], &data.getFieldData()[1],
        &data.getFieldData()[2], 3);

    for (int bb = 0; bb != nb_base_fun_col; ++bb) {
      t_position_master(i) += t_base_master * t_field_data_master(i);

      ++t_base_master;
      ++t_field_data_master;
    }
    ++t_position_master;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpGetMatPosForDisplAtGaussPtsMaster::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;
  const int nb_dofs = data.getFieldData().size();

  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  auto t_new_spat_pos_master = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsMasterPtr);

  int nb_base_fun_col = data.getFieldData().size() / 3;

  FTensor::Index<'i', 3> i;

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

    FTensor::Tensor1<double *, 3> t_field_data_master(
        &data.getFieldData()[0], &data.getFieldData()[1],
        &data.getFieldData()[2], 3);

    for (int bb = 0; bb != nb_base_fun_col; ++bb) {
      t_new_spat_pos_master(i) -= t_base_master * t_field_data_master(i);

      ++t_base_master;
      ++t_field_data_master;
    }
    ++t_new_spat_pos_master;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpGetDeformationFieldForDisplAtGaussPtsMaster::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;
  const int nb_dofs = data.getFieldData().size();

  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  auto t_new_spat_pos_master = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsMasterPtr);

  int nb_base_fun_col = data.getFieldData().size() / 3;

  FTensor::Index<'i', 3> i;

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

    FTensor::Tensor1<double *, 3> t_field_data_master(
        &data.getFieldData()[0], &data.getFieldData()[1],
        &data.getFieldData()[2], 3);

    for (int bb = 0; bb != nb_base_fun_col; ++bb) {
      t_new_spat_pos_master(i) += t_base_master * t_field_data_master(i);

      ++t_base_master;
      ++t_field_data_master;
    }
    ++t_new_spat_pos_master;
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

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));

    FTensor::Tensor1<double *, 3> t_field_data_slave(
        &data.getFieldData()[0], &data.getFieldData()[1],
        &data.getFieldData()[2], 3);

    for (int bb = 0; bb != nb_base_fun_col; ++bb) {
      t_position_slave(i) += t_base_slave * t_field_data_slave(i);

      ++t_base_slave;
      ++t_field_data_slave;
    }
    ++t_position_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetMatPosForDisplAtGaussPtsSlave::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;
  const int nb_dofs = data.getFieldData().size();

  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  auto t_new_spat_pos_slave = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsSlavePtr);

  int nb_base_fun_col = data.getFieldData().size() / 3;

  FTensor::Index<'i', 3> i;

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));

    FTensor::Tensor1<double *, 3> t_field_data_slave(
        &data.getFieldData()[0], &data.getFieldData()[1],
        &data.getFieldData()[2], 3);

    for (int bb = 0; bb != nb_base_fun_col; ++bb) {
      t_new_spat_pos_slave(i) -= t_base_slave * t_field_data_slave(i);

      ++t_base_slave;
      ++t_field_data_slave;
    }
    ++t_new_spat_pos_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpGetDeformationFieldForDisplAtGaussPtsSlave::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;
  const int nb_dofs = data.getFieldData().size();

  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  auto t_new_spat_pos_slave = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsSlavePtr);

  int nb_base_fun_col = data.getFieldData().size() / 3;

  FTensor::Index<'i', 3> i;

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));

    FTensor::Tensor1<double *, 3> t_field_data_slave(
        &data.getFieldData()[0], &data.getFieldData()[1],
        &data.getFieldData()[2], 3);

    for (int bb = 0; bb != nb_base_fun_col; ++bb) {
      t_new_spat_pos_slave(i) += t_base_slave * t_field_data_slave(i);

      ++t_base_slave;
      ++t_field_data_slave;
    }
    ++t_new_spat_pos_slave;
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

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
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

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    FTensor::Tensor0<double *> t_base_lambda(&data.getN()(gg, 0));

    FTensor::Tensor0<double *> t_field_data_slave(&data.getFieldData()[0]);
    for (int bb = 0; bb != nb_base_fun_row; ++bb) {
      t_lagrange_slave += t_base_lambda * t_field_data_slave;
      ++t_base_lambda;
      ++t_field_data_slave;
    }
    ++t_lagrange_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetNormLagMulAtTriGaussPtsSlave::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  if (type == MBVERTEX) {
    commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->resize(nb_gauss_pts);
    commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->clear();
  }

  int nb_base_fun_row = data.getFieldData().size();

  auto t_lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    FTensor::Tensor0<double *> t_base_lambda(&data.getN()(gg, 0));

    FTensor::Tensor0<double *> t_field_data_slave(&data.getFieldData()[0]);
    for (int bb = 0; bb != nb_base_fun_row; ++bb) {
      t_lagrange_slave += t_base_lambda * t_field_data_slave;
      ++t_base_lambda;
      ++t_field_data_slave;
    }
    ++t_lagrange_slave;
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
    int nb_base_fun_col = col_data.getFieldData().size();

    const double area_master =
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

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      if (t_aug_lambda_ptr > 0 && std::abs(t_aug_lambda_ptr) > ALM_TOL) {
        ++t_w;
        ++t_aug_lambda_ptr;
        continue;
      }
      double val_m = t_w * area_master;
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

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      if (t_aug_lambda_ptr > 0 && std::abs(t_aug_lambda_ptr) > ALM_TOL) {
        ++t_w;
        ++t_aug_lambda_ptr;
        continue;
      }

      double val_m = t_w * area_slave;
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
      ++t_aug_lambda_ptr;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGapConstraintAugmentedOverLambda::doWork(
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
    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      if (t_aug_lambda_ptr <= 0) {
        ++t_w;
        ++t_aug_lambda_ptr;
        ++t_lagrange_slave;
        continue;
      }

      const double val_s = -t_w * area_slave / cN;

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
      ++t_w;
      ++t_aug_lambda_ptr;
    }
    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpGapConstraintAugmentedOverSpatialMaster::doWork(
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
        commonDataSimpleContact->areaSlave; // same area in master and slave

    const double area_slave =
        commonDataSimpleContact->areaSlave; // same area in master and slave

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

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_w = getFTensor0IntegrationWeightSlave();
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      if (t_aug_lambda_ptr > 0 && std::abs(t_aug_lambda_ptr) > ALM_TOL) {
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

          t_mat(i) += t_const_unit_n(i) * m * t_base_master;

          ++t_base_master; // update rows
          ++t_mat;
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
SimpleContactProblem::OpGapConstraintAugmentedOverSpatialSlave::doWork(
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

    auto t_const_unit_n =
        get_tensor_from_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_w = getFTensor0IntegrationWeightSlave();
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      if (t_aug_lambda_ptr > 0 && std::abs(t_aug_lambda_ptr) > ALM_TOL) {
        ++t_w;
        ++t_aug_lambda_ptr;
        continue;
      }

      const double val_m = t_w * area_slave;

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

      ++t_aug_lambda_ptr;
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalContactAugmentedTractionOverSpatialMasterMaster::
    doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
           EntData &row_data,
           EntData &col_data) {
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

    if (t_aug_lambda_ptr > 0 /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
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
SimpleContactProblem::OpCalContactAugmentedTractionOverSpatialMasterSlave::
    doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
           EntData &row_data,
           EntData &col_data) {
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

    if (t_aug_lambda_ptr > 0 /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
      ++t_w;
      ++t_aug_lambda_ptr;
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

MoFEMErrorCode
SimpleContactProblem::OpCalContactAugmentedTractionOverSpatialSlaveSlave::
    doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
           EntData &row_data,
           EntData &col_data) {
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

    if (t_aug_lambda_ptr > 0 /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
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
SimpleContactProblem::OpCalContactAugmentedTractionOverSpatialSlaveMaster::
    doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
           EntData &row_data,
           EntData &col_data) {
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

    if (t_aug_lambda_ptr > 0 /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
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

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
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

    const double area_m = commonDataSimpleContact->areaSlave;

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
  const double cn_value = *cNPtr.get();
  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    const double val_s = t_w * area_s *
                         SimpleContactProblem::ConstrainFunction(
                             cn_value, t_gap_gp, t_lagrange_slave);
    auto t_base_lambda = data.getFTensor0N(gg, 0);
    for (int bbr = 0; bbr != nb_base_fun_col; ++bbr) {
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

MoFEMErrorCode SimpleContactProblem::OpGapConstraintAugmentedRhs::doWork(
    int side, EntityType type, EntData &data) {
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

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    double branch_gg;
    if (t_aug_lambda_ptr > 0 && std::abs(t_aug_lambda_ptr) > ALM_TOL) {
      branch_gg = -t_lagrange_slave / cN;
    } else {
      branch_gg = t_gap_gp;
    }

    const double val_s = t_w * area_s * branch_gg;
    auto t_base_lambda = data.getFTensor0N(gg, 0);
    for (int bbr = 0; bbr != nb_base_fun_col; ++bbr) {
      vecR[bbr] += val_s * t_base_lambda;

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
    int side, EntityType type, EntData &data) {
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
      if (t_aug_lambda_ptr > 0 /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
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
    int side, EntityType type, EntData &data) {
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

    auto const_unit_n = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto t_w = getFTensor0IntegrationWeightSlave();
    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      if (t_aug_lambda_ptr > 0 /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
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
    const double cn_value = *cNPtr.get();

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      const double val_s = SimpleContactProblem::ConstrainFunction_dl(
                               cn_value, t_gap_gp, t_lagrange_slave) *
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
        commonDataSimpleContact->areaSlave; // same area in master and slave

    const double area_slave =
        commonDataSimpleContact->areaSlave; // same area in master and slave

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

    auto t_w = getFTensor0IntegrationWeightSlave();
    const double cn_value = *cNPtr.get();

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      const double val_m = SimpleContactProblem::ConstrainFunction_dg(
                               cn_value, t_gap_gp, t_lagrange_slave) *
                           t_w * area_slave;

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
    const double cn_value = *cNPtr.get();
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      const double val_m = SimpleContactProblem::ConstrainFunction_dg(
                               cn_value, t_gap_gp, t_lagrange_slave) *
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

  const EntityHandle prism_ent = getNumeredEntFiniteElementPtr()->getEnt();
  EntityHandle tri_ent;
  if (stateTagSide == MASTER_SIDE) {
    CHKERR mField.get_moab().side_element(prism_ent, 2, 3, tri_ent);
  }
  if (stateTagSide == SLAVE_SIDE) {
    CHKERR mField.get_moab().side_element(prism_ent, 2, 4, tri_ent);
  }

  int nb_dofs = data.getFieldData().size();
  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);
  int nb_gauss_pts = data.getN().size1();

  double def_val = 0.;

  Tag th_gap;
  CHKERR moabOut.tag_get_handle("GAP", 1, MB_TYPE_DOUBLE, th_gap,
                                MB_TAG_CREAT | MB_TAG_SPARSE, &def_val);

  Tag th_lag_mult;
  CHKERR moabOut.tag_get_handle("LAGMULT", 1, MB_TYPE_DOUBLE, th_lag_mult,
                                MB_TAG_CREAT | MB_TAG_SPARSE, &def_val);

  Tag th_lag_gap_prod;
  CHKERR moabOut.tag_get_handle("LAG_GAP_PROD", 1, MB_TYPE_DOUBLE,
                                th_lag_gap_prod, MB_TAG_CREAT | MB_TAG_SPARSE,
                                &def_val);

  int def_val_int = 0;

  Tag th_state;
  CHKERR moabOut.tag_get_handle("STATE", 1, MB_TYPE_INTEGER, th_state,
                                MB_TAG_CREAT | MB_TAG_SPARSE, &def_val_int);

  Tag th_state_side;
  if (stateTagSide > 0) {
    CHKERR mField.get_moab().tag_get_handle(
        "STATE", 1, MB_TYPE_INTEGER, th_state_side,
        MB_TAG_CREAT | MB_TAG_SPARSE, &def_val_int);
  }

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

  auto th_tangent_vector_1 = get_tag_pos("TANGENT_VECTOR_1");
  auto th_tangent_vector_2 = get_tag_pos("TANGENT_VECTOR_2");

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

  auto t_state_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->gaussPtsStatePtr);

  std::array<double, 3> pos_vec;

  int count_active_pts = 0;

  auto set_float_precision = [](const double x) {
    if (std::abs(x) < std::numeric_limits<float>::epsilon())
      return 0.;
    else
      return x;
  };

auto t_shift_tensor = getFTensor1FromMat<3>(
      *commonDataSimpleContact->shiftTensor);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    const double *slave_coords_ptr = &(getCoordsAtGaussPtsSlave()(gg, 0));
    CHKERR moabOut.create_vertex(slave_coords_ptr, new_vertex);

    if (gg == 0) {
            FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;

      auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n[0], &n[1], &n[2]);
    };

      auto normal_original_slave =
      get_tensor_vec(*commonDataSimpleContact->normalVectorSlavePtr);
// cerr << "normal_original_slave  " <<  normal_original_slave << "\n";
constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
FTensor::Tensor2<double, 3, 3> t_help;
t_help(i, j) = t_kd(i, j) - normal_original_slave(i) * normal_original_slave(j);
// cerr << "t_help " << t_help <<"\n";
      FTensor::Tensor1<double, 3> t_one;
      
      t_one(i) =  t_shift_tensor(i);
      // cerr <<"One " << t_one << "\n";
      CHKERR moabOut.tag_set_data(th_tangent_vector_1, &new_vertex, 1,
                                       &t_one(0));
      ++t_shift_tensor;
      
      FTensor::Tensor1<double, 3> t_two;
      t_two(i) =  t_shift_tensor(i);
      // cerr <<"Two " << t_two << "\n";
      CHKERR moabOut.tag_set_data(th_tangent_vector_2, &new_vertex, 1,
                                       &t_two(0));
    }

    double gap_vtk = set_float_precision(t_gap_ptr);
    CHKERR moabOut.tag_set_data(th_gap, &new_vertex, 1, &gap_vtk);

    double lag_gap_prod_vtk = set_float_precision(t_lag_gap_prod_slave);
    CHKERR moabOut.tag_set_data(th_lag_gap_prod, &new_vertex, 1,
                                &lag_gap_prod_vtk);

    double lagrange_slave_vtk = set_float_precision(t_lagrange_slave);
    CHKERR moabOut.tag_set_data(th_lag_mult, &new_vertex, 1,
                                &lagrange_slave_vtk);

    int state = 0;
    if (t_state_ptr > 0.5) {
      state = 1;
      ++count_active_pts;
    }
    CHKERR moabOut.tag_set_data(th_state, &new_vertex, 1, &state);

    auto get_vec_ptr = [&](auto t) {
      for (int dd = 0; dd != 3; ++dd)
        pos_vec[dd] = set_float_precision(t(dd));
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
    ++t_state_ptr;
  }

  if (stateTagSide > 0) {
    int state_side = 0;
    if (count_active_pts >= nb_gauss_pts / 2) {
      state_side = 1;
    }
    CHKERR mField.get_moab().tag_set_data(th_state_side, &tri_ent, 1,
                                          &state_side);
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
    string field_name, string lagrange_field_name, bool is_alm,
    bool is_eigen_pos_field, string eigen_pos_field_name,
    bool use_reference_coordinates) {
  MoFEMFunctionBegin;

  fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGetNormalSlaveALE(
      "MESH_NODE_POSITIONS", common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGetNormalMasterALE(
      "MESH_NODE_POSITIONS", common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsMaster(field_name,
                                        common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsSlave(field_name, common_data_simple_contact));

  if (is_eigen_pos_field) {
    fe_rhs_simple_contact->getOpPtrVector().push_back(
        new OpGetDeformationFieldForDisplAtGaussPtsMaster(
            eigen_pos_field_name, common_data_simple_contact));

    fe_rhs_simple_contact->getOpPtrVector().push_back(
        new OpGetDeformationFieldForDisplAtGaussPtsSlave(
            eigen_pos_field_name, common_data_simple_contact));
    if (use_reference_coordinates) {
      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetMatPosForDisplAtGaussPtsMaster("MESH_NODE_POSITIONS",
                                                  common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetMatPosForDisplAtGaussPtsSlave("MESH_NODE_POSITIONS",
                                                 common_data_simple_contact));
    }
  }

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetGapSlave(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrange_field_name,
                                     common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGetGaussPtsState(
      lagrange_field_name, common_data_simple_contact, cnValue, is_alm));

  if (!is_alm) {
    fe_rhs_simple_contact->getOpPtrVector().push_back(
        new OpCalContactTractionOnSlave(field_name,
                                        common_data_simple_contact));

    fe_rhs_simple_contact->getOpPtrVector().push_back(new OpCalIntCompFunSlave(
        lagrange_field_name, common_data_simple_contact, cnValuePtr));
  } else {
    fe_rhs_simple_contact->getOpPtrVector().push_back(
        new OpGetAugmentedLambdaSlave(field_name, common_data_simple_contact,
                                      cnValue));

    fe_rhs_simple_contact->getOpPtrVector().push_back(
        new OpCalAugmentedTractionRhsSlave(field_name,
                                           common_data_simple_contact));

    fe_rhs_simple_contact->getOpPtrVector().push_back(
        new OpGapConstraintAugmentedRhs(lagrange_field_name,
                                        common_data_simple_contact, cnValue));
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setMasterForceOperatorsRhs(
    boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrange_field_name, bool is_alm,
    bool is_eigen_pos_field, string eigen_pos_field_name,
    bool use_reference_coordinates) {
  MoFEMFunctionBegin;

  fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGetNormalSlaveALE(
      "MESH_NODE_POSITIONS", common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGetNormalMasterALE(
      "MESH_NODE_POSITIONS", common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrange_field_name,
                                     common_data_simple_contact));

  if (!is_alm) {
    fe_rhs_simple_contact->getOpPtrVector().push_back(
        new OpCalContactTractionOnMaster(field_name,
                                         common_data_simple_contact));
  } else {

    fe_rhs_simple_contact->getOpPtrVector().push_back(
        new OpGetPositionAtGaussPtsMaster(field_name,
                                          common_data_simple_contact));

    fe_rhs_simple_contact->getOpPtrVector().push_back(
        new OpGetPositionAtGaussPtsSlave(field_name,
                                         common_data_simple_contact));

    if (is_eigen_pos_field) {
      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetDeformationFieldForDisplAtGaussPtsMaster(
              eigen_pos_field_name, common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetDeformationFieldForDisplAtGaussPtsSlave(
              eigen_pos_field_name, common_data_simple_contact));

      if (use_reference_coordinates) {
        fe_rhs_simple_contact->getOpPtrVector().push_back(
            new OpGetMatPosForDisplAtGaussPtsMaster(
                "MESH_NODE_POSITIONS", common_data_simple_contact));

        fe_rhs_simple_contact->getOpPtrVector().push_back(
            new OpGetMatPosForDisplAtGaussPtsSlave("MESH_NODE_POSITIONS",
                                                   common_data_simple_contact));
      }
    }

    fe_rhs_simple_contact->getOpPtrVector().push_back(
        new OpGetGapSlave(field_name, common_data_simple_contact));

    fe_rhs_simple_contact->getOpPtrVector().push_back(
        new OpGetAugmentedLambdaSlave(field_name, common_data_simple_contact,
                                      cnValue));

    fe_rhs_simple_contact->getOpPtrVector().push_back(
        new OpCalAugmentedTractionRhsMaster(field_name,
                                            common_data_simple_contact));
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactOperatorsLhs(
    boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrange_field_name, bool is_alm,
    bool is_eigen_pos_field, string eigen_pos_field_name,
    bool use_reference_coordinates) {
  MoFEMFunctionBegin;

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsMaster(field_name,
                                        common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(new OpGetNormalSlaveALE(
      "MESH_NODE_POSITIONS", common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(new OpGetNormalMasterALE(
      "MESH_NODE_POSITIONS", common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsSlave(field_name, common_data_simple_contact));

  if (is_eigen_pos_field) {
    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpGetDeformationFieldForDisplAtGaussPtsMaster(
            eigen_pos_field_name, common_data_simple_contact));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpGetDeformationFieldForDisplAtGaussPtsSlave(
            eigen_pos_field_name, common_data_simple_contact));

    if (use_reference_coordinates) {
      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetMatPosForDisplAtGaussPtsMaster("MESH_NODE_POSITIONS",
                                                  common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetMatPosForDisplAtGaussPtsSlave("MESH_NODE_POSITIONS",
                                                 common_data_simple_contact));
    }
  }

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetGapSlave(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrange_field_name,
                                     common_data_simple_contact));
  if (!is_alm) {
    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalContactTractionOverLambdaSlaveSlave(
            field_name, lagrange_field_name, common_data_simple_contact));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalDerIntCompFunOverLambdaSlaveSlave(
            lagrange_field_name, common_data_simple_contact, cnValuePtr));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalDerIntCompFunOverSpatPosSlaveMaster(
            lagrange_field_name, field_name, common_data_simple_contact,
            cnValuePtr));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalDerIntCompFunOverSpatPosSlaveSlave(
            lagrange_field_name, field_name, common_data_simple_contact,
            cnValuePtr));
  } else {
    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpGetAugmentedLambdaSlave(field_name, common_data_simple_contact,
                                      cnValue));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalContactAugmentedTractionOverLambdaSlaveSlave(
            field_name, lagrange_field_name, common_data_simple_contact));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalContactAugmentedTractionOverSpatialSlaveSlave(
            field_name, field_name, cnValue, common_data_simple_contact));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalContactAugmentedTractionOverSpatialSlaveMaster(
            field_name, field_name, cnValue, common_data_simple_contact));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpGapConstraintAugmentedOverLambda(
            lagrange_field_name, common_data_simple_contact, cnValue));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpGapConstraintAugmentedOverSpatialMaster(
            field_name, lagrange_field_name, common_data_simple_contact,
            cnValue));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpGapConstraintAugmentedOverSpatialSlave(
            field_name, lagrange_field_name, common_data_simple_contact,
            cnValue));
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setMasterForceOperatorsLhs(
    boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrange_field_name, bool is_alm,
    bool is_eigen_pos_field, string eigen_pos_field_name,
    bool use_reference_coordinates) {
  MoFEMFunctionBegin;

  fe_lhs_simple_contact->getOpPtrVector().push_back(new OpGetNormalSlaveALE(
      "MESH_NODE_POSITIONS", common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(new OpGetNormalMasterALE(
      "MESH_NODE_POSITIONS", common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrange_field_name,
                                     common_data_simple_contact));
  if (!is_alm) {
    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalContactTractionOverLambdaMasterSlave(
            field_name, lagrange_field_name, common_data_simple_contact));
  } else {

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpGetPositionAtGaussPtsMaster(field_name,
                                          common_data_simple_contact));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpGetPositionAtGaussPtsSlave(field_name,
                                         common_data_simple_contact));

    if (is_eigen_pos_field) {
      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetDeformationFieldForDisplAtGaussPtsMaster(
              eigen_pos_field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetDeformationFieldForDisplAtGaussPtsSlave(
              eigen_pos_field_name, common_data_simple_contact));
      if (use_reference_coordinates) {
        fe_lhs_simple_contact->getOpPtrVector().push_back(
            new OpGetMatPosForDisplAtGaussPtsMaster(
                "MESH_NODE_POSITIONS", common_data_simple_contact));

        fe_lhs_simple_contact->getOpPtrVector().push_back(
            new OpGetMatPosForDisplAtGaussPtsSlave("MESH_NODE_POSITIONS",
                                                   common_data_simple_contact));
      }
    }

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpGetGapSlave(field_name, common_data_simple_contact));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpGetAugmentedLambdaSlave(field_name, common_data_simple_contact,
                                      cnValue));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalContactAugmentedTractionOverLambdaMasterSlave(
            field_name, lagrange_field_name, common_data_simple_contact));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalContactAugmentedTractionOverSpatialMasterMaster(
            field_name, field_name, cnValue, common_data_simple_contact));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalContactAugmentedTractionOverSpatialMasterSlave(
            field_name, field_name, cnValue, common_data_simple_contact));
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactOperatorsLhs(
    boost::shared_ptr<ConvectMasterContactElement> fe_lhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrange_field_name, bool is_alm,
    bool is_eigen_pos_field, string eigen_pos_field_name,
    bool use_reference_coordinates) {
  MoFEMFunctionBegin;
  CHKERR setContactOperatorsLhs(
      boost::dynamic_pointer_cast<SimpleContactElement>(fe_lhs_simple_contact),
      common_data_simple_contact, field_name, lagrange_field_name, is_alm,
      is_eigen_pos_field, eigen_pos_field_name);

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpCalculateGradPositionXi(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpLhsConvectIntegrationPtsConstrainMasterGap(
          lagrange_field_name, field_name, common_data_simple_contact,
          cnValuePtr, ContactOp::FACESLAVESLAVE,
          fe_lhs_simple_contact->getConvectPtr()->getDiffKsiSpatialSlave()));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpLhsConvectIntegrationPtsConstrainMasterGap(
          lagrange_field_name, field_name, common_data_simple_contact,
          cnValuePtr, ContactOp::FACESLAVEMASTER,
          fe_lhs_simple_contact->getConvectPtr()->getDiffKsiSpatialMaster()));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setMasterForceOperatorsLhs(
    boost::shared_ptr<ConvectSlaveContactElement> fe_lhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrange_field_name, bool is_alm,
    bool is_eigen_pos_field, string eigen_pos_field_name,
    bool use_reference_coordinates) {
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
    moab::Interface &moab_out, bool alm_flag, bool is_eigen_pos_field,
    string eigen_pos_field_name, bool use_reference_coordinates,
    StateTagSide state_tag_side) {
  MoFEMFunctionBegin;

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalMasterALE("MESH_NODE_POSITIONS",
                               common_data_simple_contact));

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpGetNormalSlaveALE("MESH_NODE_POSITIONS",
                              common_data_simple_contact));

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsMaster(field_name,
                                        common_data_simple_contact));

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsSlave(field_name, common_data_simple_contact));

  if (is_eigen_pos_field) {
    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpGetDeformationFieldForDisplAtGaussPtsMaster(
            eigen_pos_field_name, common_data_simple_contact));

    fe_post_proc_simple_contact->getOpPtrVector().push_back(
        new OpGetDeformationFieldForDisplAtGaussPtsSlave(
            eigen_pos_field_name, common_data_simple_contact));
    if (use_reference_coordinates) {
      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetMatPosForDisplAtGaussPtsMaster("MESH_NODE_POSITIONS",
                                                  common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetMatPosForDisplAtGaussPtsSlave("MESH_NODE_POSITIONS",
                                                 common_data_simple_contact));
    }
  }

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpGetGapSlave(field_name, common_data_simple_contact));

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrange_field_name,
                                     common_data_simple_contact));

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpLagGapProdGaussPtsSlave(lagrange_field_name,
                                    common_data_simple_contact));

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpGetGaussPtsState(lagrange_field_name, common_data_simple_contact,
                             cnValue, alm_flag));

  fe_post_proc_simple_contact->getOpPtrVector().push_back(
      new OpMakeVtkSlave(m_field, field_name, common_data_simple_contact,
                         moab_out, state_tag_side));

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
    const double cn_value = *cNPtr.get();
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      const double val_m = SimpleContactProblem::ConstrainFunction(
                               cn_value, t_gap_gp, t_lagrange_slave) *
                           t_w * area_slave;
      const double val_diff_m_l = SimpleContactProblem::ConstrainFunction_dl(
                                      cn_value, t_gap_gp, t_lagrange_slave) *
                                  t_w * area_slave;
      const double val_diff_m_g = SimpleContactProblem::ConstrainFunction_dg(
                                      cn_value, t_gap_gp, t_lagrange_slave) *
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

MoFEMErrorCode
SimpleContactProblem::OpCalculateDeformation::doWork(int side, EntityType type,
                                                     EntData &row_data) {

  MoFEMFunctionBegin;
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  // get number of integration points
  const int nb_integration_pts = getGaussPts().size2();

  auto t_h = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->hMat);
  auto t_H = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->HMat);

  commonDataSimpleContact->detHVec->resize(nb_integration_pts, false);
  commonDataSimpleContact->invHMat->resize(9, nb_integration_pts, false);
  commonDataSimpleContact->FMat->resize(9, nb_integration_pts, false);

  commonDataSimpleContact->detHVec->clear();
  commonDataSimpleContact->invHMat->clear();
  commonDataSimpleContact->FMat->clear();

  auto t_detH = getFTensor0FromVec(*commonDataSimpleContact->detHVec);
  auto t_invH = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->invHMat);
  auto t_F = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->FMat);

  for (int gg = 0; gg != nb_integration_pts; ++gg) {
    CHKERR determinantTensor3by3(t_H, t_detH);
    CHKERR invertTensor3by3(t_H, t_detH, t_invH);
    t_F(i, j) = t_h(i, k) * t_invH(k, j);
    ++t_h;
    ++t_H;
    ++t_detH;
    ++t_invH;
    ++t_F;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpLoopForSideOfContactPrism::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBTRI)
    MoFEMFunctionReturnHot(0);

  int side_number;
  if (faceType == ContactOp::FACEMASTER)
    side_number = 3;
  else
    side_number = 4;

  const EntityHandle tri = getSideEntity(side_number, type);

  CHKERR loopSideVolumes(sideFeName, *sideFe, 3, tri);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalMatForcesALEMaster::doWork(int side, EntityType type,
                                                      EntData &row_data) {

  MoFEMFunctionBegin;

  // get number of dofs on row
  nbRows = row_data.getIndices().size();
  // if no dofs on row, exit that work, nothing to do here
  if (!nbRows)
    MoFEMFunctionReturnHot(0);

  vecF.resize(nbRows, false);
  vecF.clear();

  // get number of integration points
  nbIntegrationPts = getGaussPtsMaster().size2();

  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data);

  // assemble local matrix
  CHKERR aSsemble(row_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalMatForcesALEMaster::iNtegrate(EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  // in case the tet is not in database
  if (commonDataSimpleContact->FMat->size1() != 9)
    MoFEMFunctionReturnHot(0);

  const int nb_base_fun_col = nbRows / 3;

  auto get_tensor_vec = [](VectorDouble &n, const int r) {
    return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto t_F = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->FMat);
  auto normal_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->normalVectorMasterPtr), 0);

  auto t_w = getFTensor0IntegrationWeightMaster();
  double &area_master = commonDataSimpleContact->areaMaster;
  for (int gg = 0; gg != nbIntegrationPts; ++gg) {
    const double val_m = area_master * t_w;
    FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

      const double s = val_m * t_base_master * lagrange_slave;

      auto t_assemble_s = get_tensor_vec(vecF, 3 * bbc);

      t_assemble_s(i) -= s * t_F(j, i) * normal_at_gp(j);

      ++t_base_master;
    }
    ++t_F;
    ++lagrange_slave;
    ++t_w;
  } // for gauss points

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalMatForcesALEMaster::aSsemble(EntData &row_data) {
  MoFEMFunctionBegin;

  // get pointer to first global index on row
  const int *row_indices = &*row_data.getIndices().data().begin();
  auto &data = *commonDataSimpleContact;
  if (data.forcesOnlyOnEntitiesRow.empty())
    MoFEMFunctionReturnHot(0);

  rowIndices.resize(nbRows, false);
  noalias(rowIndices) = row_data.getIndices();
  row_indices = &rowIndices[0];
  VectorDofs &dofs = row_data.getFieldDofs();
  VectorDofs::iterator dit = dofs.begin();
  for (int ii = 0; dit != dofs.end(); ++dit, ++ii) {
    if (data.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
        data.forcesOnlyOnEntitiesRow.end()) {
      rowIndices[ii] = -1;
    }
  }

  CHKERR VecSetValues(getSNESf(), nbRows, row_indices, &*vecF.data().begin(),
                      ADD_VALUES);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalMatForcesALESlave::doWork(int side, EntityType type,
                                                     EntData &row_data) {

  MoFEMFunctionBegin;

  // get number of dofs on row
  nbRows = row_data.getIndices().size();
  // if no dofs on row, exit that work, nothing to do here
  if (!nbRows)
    MoFEMFunctionReturnHot(0);

  vecF.resize(nbRows, false);
  vecF.clear();

  // get number of integration points
  nbIntegrationPts = getGaussPtsSlave().size2();

  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data);

  // assemble local matrix
  CHKERR aSsemble(row_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalMatForcesALESlave::iNtegrate(EntData &data) {
  MoFEMFunctionBegin;

  int nb_base_fun_col = nbRows / 3;

  auto get_tensor_vec = [](VectorDouble &n, const int r) {
    return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto t_F = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->FMat);
  auto normal_at_gp =
      get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr), 0);
  auto t_w = getFTensor0IntegrationWeightSlave();
  double &area_slave = commonDataSimpleContact->areaSlave;
  for (int gg = 0; gg != nbIntegrationPts; ++gg) {
    double val_s = t_w * area_slave;
    FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));
    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
      const double s = val_s * t_base_master * lagrange_slave;
      auto t_assemble_s = get_tensor_vec(vecF, 3 * bbc);
      t_assemble_s(i) -= s * t_F(j, i) * normal_at_gp(j);
      ++t_base_master;
    }
    ++t_F;
    ++lagrange_slave;
    ++t_w;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalMatForcesALESlave::aSsemble(EntData &row_data) {
  MoFEMFunctionBegin;

  // get pointer to first global index on row
  const int *row_indices = &*row_data.getIndices().data().begin();
  auto &data = *commonDataSimpleContact;
  if (data.forcesOnlyOnEntitiesRow.empty())
    MoFEMFunctionReturnHot(0);

  rowIndices.resize(nbRows, false);
  noalias(rowIndices) = row_data.getIndices();
  row_indices = &rowIndices[0];
  VectorDofs &dofs = row_data.getFieldDofs();
  VectorDofs::iterator dit = dofs.begin();
  for (int ii = 0; dit != dofs.end(); ++dit, ++ii) {
    if (data.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
        data.forcesOnlyOnEntitiesRow.end()) {
      rowIndices[ii] = -1;
    }
  }

  CHKERR VecSetValues(getSNESf(), nbRows, row_indices, &*vecF.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpGetNormalSlaveALE::doWork(int side, EntityType type,
                                                  EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  commonDataSimpleContact->normalVectorSlavePtr->resize(3, false);

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto normal_original_slave =
      get_tensor_vec(*commonDataSimpleContact->normalVectorSlavePtr);

  commonDataSimpleContact->tangentOneVectorSlavePtr->resize(3, false);
  commonDataSimpleContact->tangentOneVectorSlavePtr->clear();

  commonDataSimpleContact->tangentTwoVectorSlavePtr->resize(3, false);
  commonDataSimpleContact->tangentTwoVectorSlavePtr->clear();

  auto tangent_0_slave =
      get_tensor_vec(*commonDataSimpleContact->tangentOneVectorSlavePtr);
  auto tangent_1_slave =
      get_tensor_vec(*commonDataSimpleContact->tangentTwoVectorSlavePtr);

  auto t_N = data.getFTensor1DiffN<2>(0, 0);
  auto t_dof = data.getFTensor1FieldData<3>();

  for (unsigned int dd = 0; dd != 3; ++dd) {
    tangent_0_slave(i) += t_dof(i) * t_N(0);
    tangent_1_slave(i) += t_dof(i) * t_N(1);
    ++t_dof;
    ++t_N;
  }

  commonDataSimpleContact->shiftTensor->resize(3, 2, false);
  commonDataSimpleContact->shiftTensor->clear();

  normal_original_slave(i) =
      FTensor::levi_civita(i, j, k) * tangent_0_slave(j) * tangent_1_slave(k);

  const double normal_length =
      sqrt(normal_original_slave(i) * normal_original_slave(i));
  normal_original_slave(i) = normal_original_slave(i) / normal_length;

  commonDataSimpleContact->areaSlave = 0.5 * normal_length;


    VectorDouble eig_values(3, false);
  eig_values.clear();
  MatrixDouble eigen_vectors(3, 3, false);
  eigen_vectors.clear();
  MatrixDouble symm_matrix(3, 3, false);
  symm_matrix.clear();


constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
  auto get_tensor_from_mat = [](MatrixDouble &m) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(0, 0), &m(0, 1), &m(0, 2), &m(1, 0),
        &m(1, 1), &m(1, 2), &m(2, 0), &m(2, 1),
        &m(2, 2));
  };

  auto get_tensor_from_mat_no_p = [](MatrixDouble &m) {
    return FTensor::Tensor2<double, 3, 3>(
        m(0, 0), m(0, 1), m(0, 2), m(1, 0),
        m(1, 1), m(1, 2), m(2, 0), m(2, 1),
        m(2, 2));
  };

  auto get_tensor_from_vec_no_p = [](VectorDouble &m) {
    return FTensor::Tensor1<double, 3>(
        m(0), m(1), m(2));
  };

  auto get_vec_from_mat_by_cols = [](MatrixDouble &m, const int c) {
    return FTensor::Tensor1<double, 3>(m(0, c), m(1, c), m(2, c));
  };

    auto get_vec_from_mat_by_cols_by_ptr = [](MatrixDouble &m, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(0, c), &m(1, c), &m(2, c));
  };

  auto get_vec_from_mat_by_rows = [](MatrixDouble &m, const int r) {
    return FTensor::Tensor1<double *, 3>(&m(r, 0), &m(r, 1), &m(r, 2));
  };


  auto make_givens_rotation = [&](int N, int ii, int jj, double theta) {
    FTensor::Tensor2<double, 3, 3> R;
    R(i, j) = 0.;
    R(0, 0) = R(1, 1) = R(2, 2) = 1.;
    double c = cos(theta);
    double s = sin(theta);
    R(ii, ii) = c;
    R(jj, jj) = c;
    R(ii, jj) = s;
    R(jj, ii) = s;
    return R;
  };

  auto construct_subspace_rotation_mtx = [&](int N, int ii,
                                             FTensor::Tensor1<double, 3> &AnglesCol, FTensor::Tensor2<double, 3, 3> &R) {
  MoFEMFunctionBegin;

    for (int jj = N - 1; jj != ii; --jj) {
    auto r_recusive = make_givens_rotation(N, ii, jj, AnglesCol(jj));
    auto r_copy = R;
    R(i, k) = r_recusive(i, j) * r_copy(j, k);
    }
MoFEMFunctionReturn(0);
    
  };

  auto solve_rotation_angles_in_sub_dim = [&](int N, int ii, FTensor::Tensor1<double, 3> &Vcol) {
    FTensor::Tensor1<double, 3> angles_work;
    angles_work(i) = 0;
    double r = 1.;
    for (int jj = N - 1; jj != ii; --jj) {
      double y = Vcol(jj);
      r = r * cos(angles_work(jj));
      angles_work(jj) = (abs(r) > eps) ? asin(y / r) : 0.;
    }

    return angles_work;
  };

  auto reduce_dimension_by_one = [&](int N, int ii, MatrixDouble &Vwork, MatrixDouble &AnglesMtx) {
    MoFEMFunctionBegin;
    //passing by pointers
    auto angles_mat = get_vec_from_mat_by_rows(AnglesMtx, ii);
    auto t_v_work = get_tensor_from_mat(Vwork);
    
    auto v_col = get_vec_from_mat_by_cols(Vwork, ii);
    
    FTensor::Tensor2<double, 3, 3> t_v_to_mult;
    t_v_to_mult(i, j) = t_v_work(i, j);
    // Vcol ← Vwork[:, i ]
    auto angles_col = solve_rotation_angles_in_sub_dim(N, ii, v_col);

    FTensor::Tensor2<double, 3, 3> R;
    R(i, j) = 0.;
    R(0, 0) = R(1, 1) = R(2, 2) = 1.;
    CHKERR construct_subspace_rotation_mtx(N, ii, angles_col, R);
    //for return
    t_v_work(i, k) = R(j, i) * t_v_to_mult(j, k);
    angles_mat(i) = angles_col(i);
    MoFEMFunctionReturn(0);
  };

  auto orient_eigenvectors = [&](MatrixDouble &eigenVectors) {
    MoFEMFunctionBegin;
    int N = 3;
    FTensor::Tensor1<double, 3> sign_flip_vec;
    MatrixDouble v_work;
    v_work.resize(3, 3, false);
    v_work.clear();

    MatrixDouble angle_mat;
    angle_mat.resize(3, 3, false);
    angle_mat.clear();

    auto t_v_work = get_tensor_from_mat(v_work);
    auto t_eigen_vectors = get_tensor_from_mat(eigenVectors);

    t_v_work(i, j) = t_eigen_vectors(i, j);
    for (int ii = 0; ii != N; ++ii){
      sign_flip_vec(ii) = (abs(t_v_work(ii, ii))  < eps || t_v_work(ii, ii)  > 0 ) ? 1 : -1;
      auto t_v_work_col = get_vec_from_mat_by_cols(v_work, ii);
      auto t_v_work_col_ptr = get_vec_from_mat_by_cols_by_ptr(v_work, ii);
      t_v_work_col_ptr(i) = t_v_work_col(i) * sign_flip_vec(ii);
      CHKERR reduce_dimension_by_one(N, ii, v_work, angle_mat);
    }

    FTensor::Tensor2<double, 3, 3> t_s;
    t_s(i, j) = 0.;
    for (int ii = 0; ii != N; ++ii){
    t_s(ii, ii) = sign_flip_vec(ii);   
    }
    // cerr << " sign_flip_vec " << sign_flip_vec <<"\n";
    
    t_v_work(i, j) = t_eigen_vectors(i, j);
    // cerr << " t_eigen_vectors " << t_eigen_vectors <<"\n";
    // cerr << " t_v_work " << t_v_work <<"\n";
    t_eigen_vectors(i, k) = t_v_work(i, j) * t_s(j, k);
    // cerr << " t_eigen_vectors " << t_eigen_vectors <<"\n";
        // Eor ← Esort
            // return Vor, Eor, AnglesMtx, SignFlipVec, SortIndices
            MoFEMFunctionReturn(0);
  };

  auto t_symm_matrix = get_tensor_from_mat(symm_matrix);

  t_symm_matrix(i, j) = t_kd(i, j) - normal_original_slave(i) * normal_original_slave(j);
  t_symm_matrix(1, 1) += 1.e-6;
//  cerr << "t_symm_matrix  " << t_symm_matrix << "\n";
//   cerr << "normal_original_slave " << normal_original_slave << "\n";

  CHKERR computeEigenValuesSymmetric(symm_matrix, eig_values, eigen_vectors);

MatrixDouble trans_symm_matrix(3, 3, false);
  trans_symm_matrix.clear();
  // noalias(trans_symm_matrix) = trans(eigen_vectors);
  
  // cerr << "eigen_vectors  " << eigen_vectors << "\n";
  //   cerr << "eig_values " << eig_values << "\n";
  auto t_shift_tensor = getFTensor1FromMat<3>(
      *commonDataSimpleContact->shiftTensor);

  auto t_eigen_vals =
      getFTensor0FromVec(eig_values);

      auto nb_uniq = get_uniq_nb<3>(&*eig_values.data().begin());
      
      auto t_eig_vec_to_sort = get_tensor_from_mat_no_p(eigen_vectors);
      auto t_eig_values_to_sort = get_tensor_from_vec_no_p(eig_values);
      auto t_mat_eigen_vectors = get_tensor_from_mat(trans_symm_matrix);
      // if (nb_uniq == 2) {
        // cerr << "before; values:  " << t_eig_values_to_sort
        FTensor::Tensor2<double, 3, 3> t_check_3by3;
        t_check_3by3(i, j) = t_eig_vec_to_sort(j, i);
        // cerr << "  vec: " << t_check_3by3 << "\n";
        sortEigenVals<3>(t_eig_values_to_sort, t_eig_vec_to_sort, nb_uniq);
        // cerr << "after; values:  " << t_eig_values_to_sort
        //      << "  vec: " << t_eig_vec_to_sort << "\n";
        //      cerr << "1 t_eigen_vectors " << t_eigen_vectors <<"\n";
            //  t_mat_eigen_vectors(i, j) = t_eig_vec_to_sort(j, i);
            //  cerr << "trans_symm_matrix 1 " <<  trans_symm_matrix << "\n"; 
            
            // cerr << "eigen_vectors 1 " << eigen_vectors << "\n";
             auto t_eigen_vectors_2 = get_tensor_from_mat(eigen_vectors); 
            t_eigen_vectors_2(i,j) =t_eig_vec_to_sort(i,j);
            noalias(trans_symm_matrix) = trans(eigen_vectors);
             CHKERR orient_eigenvectors(trans_symm_matrix);
            //  cerr << "eigen_vectors 2 " << eigen_vectors << "\n";
            //  t_mat_eigen_vectors(i, j) = t_eig_vec_to_sort(j, i);

             
        //      cerr << "2 t_eigen_vectors " << t_mat_eigen_vectors <<"\n";
            //  cerr << " trans_symm_matrix " <<  trans_symm_matrix << "\n";
      // }
auto t_eigen_vectors = getFTensor1FromMat<3>(trans_symm_matrix);
  int count = 0;
  // cerr << "normal_original_slave  " << normal_original_slave << "\n";
  // cerr << "trans_symm_matrix 2 " <<  trans_symm_matrix << "\n"; 
  for (int ii = 0; ii != 3; ++ii) {
    // 
    // cerr << "t_eigen_vectors  " <<  t_eigen_vectors << "\n"; 
    double vec_length = sqrt(t_eigen_vectors(i) * t_eigen_vectors(i));
    double projection = sqrt(abs(t_eigen_vectors(i) * normal_original_slave(i)));
    // cerr << "projection  " << projection <<"\n";
    if (projection < 1.e-3) {
      ++count;
      // if(t_eigen_vectors(2) > 0)
      t_shift_tensor(i) = t_eigen_vectors(i);
      // else 
      // t_shift_tensor(i) = -t_eigen_vectors(i);
      // cerr <<"eigen  " <<  t_eigen_vectors << "\n";
      // cerr << "normal_original_slave  " << normal_original_slave << "\n";
      // cerr <<"eigen val   " << t_eigen_vals   <<"  shift tensor " << count << "  " << t_shift_tensor << "\n";
      // cerr << "projection " << projection << "\n";
      ++t_shift_tensor;
    }
    // cerr << "vec_length " << vec_length << "  " << ii << " " <<
    // t_eigen_vectors << "\n"; cerr << "projection " << projection << "\n";
    ++t_eigen_vectors;
    ++t_eigen_vals;
}
if(count != 2){
  cerr << "Mistake!! " << count <<  "\n";
}
// cerr <<"shift  " <<  t_shift_tensor << "\n"; 


  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpGetNormalMasterALE::doWork(int side, EntityType type,
                                                   EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  commonDataSimpleContact->normalVectorMasterPtr->resize(3, false);

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto normal_original_master =
      get_tensor_vec(*commonDataSimpleContact->normalVectorMasterPtr);

  commonDataSimpleContact->tangentOneVectorMasterPtr->resize(3, false);
  commonDataSimpleContact->tangentOneVectorMasterPtr->clear();

  commonDataSimpleContact->tangentTwoVectorMasterPtr->resize(3, false);
  commonDataSimpleContact->tangentTwoVectorMasterPtr->clear();

  auto tangent_0_master =
      get_tensor_vec(*commonDataSimpleContact->tangentOneVectorMasterPtr);
  auto tangent_1_master =
      get_tensor_vec(*commonDataSimpleContact->tangentTwoVectorMasterPtr);

  auto t_N = data.getFTensor1DiffN<2>(0, 0);
  auto t_dof = data.getFTensor1FieldData<3>();

  for (unsigned int dd = 0; dd != 3; ++dd) {
    tangent_0_master(i) += t_dof(i) * t_N(0);
    tangent_1_master(i) += t_dof(i) * t_N(1);
    ++t_dof;
    ++t_N;
  }

  normal_original_master(i) =
      FTensor::levi_civita(i, j, k) * tangent_0_master(j) * tangent_1_master(k);

  const double normal_length =
      sqrt(normal_original_master(i) * normal_original_master(i));
  normal_original_master(i) = normal_original_master(i) / normal_length;

  commonDataSimpleContact->areaMaster = 0.5 * normal_length;

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpContactTractionSlaveSlave_dX::iNtegrate(
    EntData &row_data, EntData &col_data) {
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

  auto get_tensor_vec_3 = [&](VectorDouble3 &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  auto make_vec_der = [](FTensor::Tensor1<double *, 2> t_N,
                         FTensor::Tensor1<double *, 3> t_1,
                         FTensor::Tensor1<double *, 3> t_2) {
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;
    FTensor::Tensor2<double, 3, 3> t_n;
    t_n(i, j) = 0;
    t_n(i, j) += FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
    t_n(i, j) -= FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
    return t_n;
  };

  matLhs.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  matLhs.clear();

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto t_1 = get_tensor_vec(*commonDataSimpleContact->tangentOneVectorSlavePtr);
  auto t_2 = get_tensor_vec(*commonDataSimpleContact->tangentTwoVectorSlavePtr);
  auto t_w = getFTensor0IntegrationWeightSlave();

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    auto t_N = col_data.getFTensor1DiffN<2>(gg, 0);

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

      FTensor::Tensor0<double *> t_base_slave(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        const double s = t_w * lagrange_slave * t_base_slave;

        auto t_d_n = make_vec_der(t_N, t_1, t_2);

        auto t_assemble_s = get_tensor_from_mat(matLhs, 3 * bbr, 3 * bbc);

        t_assemble_s(i, j) += 0.5 * s * t_d_n(i, j);

        ++t_base_slave; // update rows
      }
      ++t_N;
    }
    ++lagrange_slave;
    ++t_w;
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpContactTractionMasterSlave_dX::iNtegrate(
    EntData &row_data, EntData &col_data) {
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

  auto get_tensor_vec_3 = [&](VectorDouble3 &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  auto make_vec_der = [](FTensor::Tensor1<double *, 2> t_N,
                         FTensor::Tensor1<double *, 3> t_1,
                         FTensor::Tensor1<double *, 3> t_2) {
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;
    FTensor::Tensor2<double, 3, 3> t_n;
    t_n(i, j) = 0;
    t_n(i, j) += FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
    t_n(i, j) -= FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
    return t_n;
  };

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  const double area_s = commonDataSimpleContact->areaSlave;

  const double area_m = commonDataSimpleContact->areaMaster;

  auto t_1 = get_tensor_vec(*commonDataSimpleContact->tangentOneVectorSlavePtr);
  auto t_2 = get_tensor_vec(*commonDataSimpleContact->tangentTwoVectorSlavePtr);

  auto t_const_unit_slave =
      get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

  auto t_w = getFTensor0IntegrationWeightMaster();
  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    auto t_N = col_data.getFTensor1DiffN<2>(gg, 0);

    const double mult_s = 0.5 * t_w * lagrange_slave * area_m / area_s;

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

      FTensor::Tensor0<double *> t_base_master(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        const double s = mult_s * t_base_master;

        auto t_d_n = make_vec_der(t_N, t_1, t_2);

        auto t_assemble_s = get_tensor_from_mat(matLhs, 3 * bbr, 3 * bbc);

        t_assemble_s(i, j) -=
            s * (-t_const_unit_slave(i) * t_d_n(k, j) * t_const_unit_slave(k) +
                 t_d_n(i, j));

        ++t_base_master; // update rows
      }
      ++t_N;
    }
    ++lagrange_slave;
    ++t_w;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactTractionMasterMaster_dX::iNtegrate(
    EntData &row_data, EntData &col_data) {
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

  auto get_tensor_vec_3 = [&](VectorDouble3 &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  auto make_vec_der = [](FTensor::Tensor1<double *, 2> t_N,
                         FTensor::Tensor1<double *, 3> t_1,
                         FTensor::Tensor1<double *, 3> t_2) {
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;
    FTensor::Tensor2<double, 3, 3> t_n;
    t_n(i, j) = 0;
    t_n(i, j) += FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
    t_n(i, j) -= FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
    return t_n;
  };

  matLhs.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  matLhs.clear();

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto t_1 =
      get_tensor_vec(*commonDataSimpleContact->tangentOneVectorMasterPtr);
  auto t_2 =
      get_tensor_vec(*commonDataSimpleContact->tangentTwoVectorMasterPtr);

  auto t_w = getFTensor0IntegrationWeightMaster();

  auto t_const_unit_master =
      get_tensor_vec(*(commonDataSimpleContact->normalVectorMasterPtr));

  auto t_const_unit_slave =
      get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

  const double area_m = commonDataSimpleContact->areaMaster;
  const double area_s = commonDataSimpleContact->areaSlave;

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    auto t_N = col_data.getFTensor1DiffN<2>(gg, 0);

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
      const double mult_m = 0.5 * t_w * lagrange_slave;
      FTensor::Tensor0<double *> t_base_master(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        const double s = mult_m * t_base_master;

        auto t_d_n = make_vec_der(t_N, t_1, t_2);

        auto t_assemble_s = get_tensor_from_mat(matLhs, 3 * bbr, 3 * bbc);

        t_assemble_s(i, j) -=
            s * t_d_n(k, j) * t_const_unit_master(k) * t_const_unit_slave(i);

        ++t_base_master; // update rows
      }
      ++t_N;
    }
    ++lagrange_slave;
    ++t_w;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalDerIntCompFunSlaveSlave_dX::iNtegrate(
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();
  int nb_base_fun_row = row_data.getFieldData().size();
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

  matLhs.resize(nb_base_fun_row, 3 * nb_base_fun_col, false);
  matLhs.clear();

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto get_vec_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 0, c + 1),
                                         &m(r + 0, c + 2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  auto make_vec_der = [](FTensor::Tensor1<double *, 2> t_N,
                         FTensor::Tensor1<double *, 3> t_1,
                         FTensor::Tensor1<double *, 3> t_2) {
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;
    FTensor::Tensor2<double, 3, 3> t_n;
    t_n(i, j) = 0;
    t_n(i, j) += FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
    t_n(i, j) -= FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
    return t_n;
  };

  auto x_m = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsMasterPtr);
  auto x_s = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsSlavePtr);
  auto t_lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  const double length_normal = commonDataSimpleContact->areaSlave;

  auto normal_at_gp =
      get_tensor_vec(*commonDataSimpleContact->normalVectorSlavePtr);

  auto t_w = getFTensor0IntegrationWeightSlave();
  auto t_1 = get_tensor_vec(*commonDataSimpleContact->tangentOneVectorSlavePtr);
  auto t_2 = get_tensor_vec(*commonDataSimpleContact->tangentTwoVectorSlavePtr);
  auto t_gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);
  const double cn_value = *cNPtr.get();
  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    double val_s = t_w * 0.5;
    auto t_N = col_data.getFTensor1DiffN<2>(gg, 0);

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
      FTensor::Tensor0<double *> t_base_lambda(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        const double s = val_s * t_base_lambda;

        auto t_d_n = make_vec_der(t_N, t_1, t_2);

        auto assemble_mat = get_vec_from_mat(matLhs, bbr, 3 * bbc);

        assemble_mat(j) -=
            0.5 *
            (t_d_n(i, j) - normal_at_gp(i) * t_d_n(k, j) * normal_at_gp(k)) *
            (x_s(i) - x_m(i)) * s * cn_value *
            (1 + SimpleContactProblem::Sign(t_lagrange_slave -
                                            cn_value * t_gap_gp));

        assemble_mat(j) += t_d_n(i, j) * normal_at_gp(i) * s *
                           SimpleContactProblem::ConstrainFunction(
                               cn_value, t_gap_gp, t_lagrange_slave);

        ++t_base_lambda; // update rows
      }
      ++t_N;
    }

    ++x_m;
    ++x_s;
    ++t_lagrange_slave;
    ++t_gap_gp;
    ++t_w;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpContactMaterialLhs::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;
  if (row_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);
  row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);
  nb_gauss_pts = row_data.getN().size1();

  nb_base_fun_row = row_data.getFieldData().size() / rankRow;
  nb_base_fun_col = col_data.getFieldData().size() / rankCol;

  matLhs.resize(rankRow * nb_base_fun_row, rankCol * nb_base_fun_col, false);
  matLhs.clear();

  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data, col_data);

  // assemble local matrix
  CHKERR aSsemble(row_data, col_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpContactALELhs::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;
  row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);
  nb_gauss_pts = row_data.getN().size1();

  nb_base_fun_row = row_data.getFieldData().size() / rankRow;
  nb_base_fun_col = col_data.getFieldData().size() / rankCol;

  matLhs.resize(rankRow * nb_base_fun_row, rankCol * nb_base_fun_col, false);
  matLhs.clear();

  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data, col_data);

  // assemble local matrix
  CHKERR aSsemble(row_data, col_data);

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

  commonDataSimpleContact->gaussPtsStatePtr->resize(nb_gauss_pts, false);
  commonDataSimpleContact->gaussPtsStatePtr->clear();

  auto t_state_gp =
      getFTensor0FromVec(*commonDataSimpleContact->gaussPtsStatePtr);

  auto t_lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
  auto t_gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    vecR[CommonDataSimpleContact::TOTAL] += 1;

    if (!almFlag &&
        SimpleContactProblem::State(cN, t_gap_gp, t_lagrange_slave)) {
      vecR[CommonDataSimpleContact::ACTIVE] += 1;
      t_state_gp = 1;
    }

    if (almFlag &&
        SimpleContactProblem::StateALM(cN, t_gap_gp, t_lagrange_slave)) {
      vecR[CommonDataSimpleContact::ACTIVE] += 1;
      t_state_gp = 1;
    }

    ++t_lagrange_slave;
    ++t_gap_gp;
    ++t_state_gp;
  } // for gauss points

  constexpr std::array<int, 2> indices = {CommonDataSimpleContact::ACTIVE,
                                          CommonDataSimpleContact::TOTAL};
  CHKERR VecSetValues(commonDataSimpleContact->gaussPtsStateVec, 2,
                      indices.data(), &vecR[0], ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactMaterialLhs::aSsemble(EntData &row_data,
                                                     EntData &col_data) {

  MoFEMFunctionBegin;

  // get pointer to first global index on row
  const int *row_indices = &*row_data.getIndices().data().begin();
  // get pointer to first global index on column
  const int *col_indices = &*col_data.getIndices().data().begin();

  auto &data = *commonDataSimpleContact;
  if (data.forcesOnlyOnEntitiesRow.empty())
    MoFEMFunctionReturnHot(0);

  if (!data.forcesOnlyOnEntitiesRow.empty()) {
    rowIndices.resize(row_nb_dofs, false);
    noalias(rowIndices) = row_data.getIndices();
    row_indices = &rowIndices[0];
    VectorDofs &dofs = row_data.getFieldDofs();
    VectorDofs::iterator dit = dofs.begin();
    for (int ii = 0; dit != dofs.end(); ++dit, ++ii) {
      if (data.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
          data.forcesOnlyOnEntitiesRow.end()) {
        rowIndices[ii] = -1;
      }
    }
  }

  // assemble local matrix
  CHKERR MatSetValues(getSNESB(), row_nb_dofs, row_indices, col_nb_dofs,
                      col_indices, &*matLhs.data().begin(), ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactALELhs::aSsemble(EntData &row_data,
                                                EntData &col_data) {

  MoFEMFunctionBegin;

  // get pointer to first global index on row
  const int *row_indices = &*row_data.getIndices().data().begin();
  // get pointer to first global index on column
  const int *col_indices = &*col_data.getIndices().data().begin();

  // assemble local matrix
  CHKERR MatSetValues(getSNESB(), row_nb_dofs, row_indices, col_nb_dofs,
                      col_indices, &*matLhs.data().begin(), ADD_VALUES);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactMaterialVolOnSideLhs_dX_dx::iNtegrate(
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto t_w = getFTensor0IntegrationWeight();

  auto t_inv_H = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->invHMat);

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto normal_at_gp = get_tensor_vec(*normalVector);

  const double area_m = aRea;

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    double a = -t_w * lagrange_slave * area_m;

    auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

    int bbc = 0;
    for (; bbc != nb_base_fun_col; ++bbc) {

      FTensor::Tensor0<double *> t_row_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; ++bbr) {

        auto t_assemble = get_tensor2(matLhs, 3 * bbr, 3 * bbc);
        // TODO: handle hoGeometry

        t_assemble(i, j) += a * t_row_base * t_inv_H(k, i) *
                            t_col_diff_base(k) * normal_at_gp(j);

        ++t_row_base;
      }
      ++t_col_diff_base;
    }
    ++t_w;
    ++t_inv_H;
    ++lagrange_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactMaterialVolOnSideLhs_dX_dX::iNtegrate(
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;
  FTensor::Index<'m', 3> m;

  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto t_w = getFTensor0IntegrationWeight();

  auto t_h = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->hMat);
  auto t_inv_H = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->invHMat);

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto normal_at_gp = get_tensor_vec(*normalVector);

  const double area_m = aRea;

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    const double a = -t_w * lagrange_slave * area_m;
    auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

    int bbc = 0;
    for (; bbc != nb_base_fun_col; ++bbc) {

      FTensor::Tensor0<double *> t_row_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; ++bbr) {

        auto t_assemble = get_tensor2(matLhs, 3 * bbr, 3 * bbc);

        t_assemble(i, j) -= a * t_row_base * t_inv_H(l, j) *
                            t_col_diff_base(m) * t_inv_H(m, i) * t_h(k, l) *
                            normal_at_gp(k);

        ++t_row_base;
      }
      ++t_col_diff_base;
    }
    ++t_w;
    ++t_h;
    ++t_inv_H;
    ++lagrange_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactMaterialMasterOnFaceLhs_dX_dX::iNtegrate(
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto make_vec_der = [](FTensor::Tensor1<double *, 2> t_N,
                         FTensor::Tensor1<double *, 3> t_1,
                         FTensor::Tensor1<double *, 3> t_2) {
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;
    FTensor::Tensor2<double, 3, 3> t_n;
    t_n(i, j) = 0;
    t_n(i, j) += FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
    t_n(i, j) -= FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
    return t_n;
  };

  auto t_F = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->FMat);

  auto t_w = getFTensor0IntegrationWeightMaster();
  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
  auto t_1 =
      get_tensor_vec(*commonDataSimpleContact->tangentOneVectorMasterPtr);
  auto t_2 =
      get_tensor_vec(*commonDataSimpleContact->tangentTwoVectorMasterPtr);
  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    auto t_N = col_data.getFTensor1DiffN<2>(gg, 0);
    const double val = 0.5 * t_w * lagrange_slave;

    int bbc = 0;
    for (; bbc != nb_base_fun_col; ++bbc) {

      FTensor::Tensor0<double *> t_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; ++bbr) {

        auto t_assemble = get_tensor2(matLhs, 3 * bbr, 3 * bbc);
        // TODO: handle hoGeometry
        auto t_d_n = make_vec_der(t_N, t_1, t_2);
        t_assemble(i, k) -= val * t_base * t_F(j, i) * t_d_n(j, k);

        ++t_base;
      }
      ++t_N;
    }
    ++t_F;
    ++t_w;
    ++lagrange_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactMaterialSlaveOnFaceLhs_dX_dX::iNtegrate(
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto make_vec_der = [](FTensor::Tensor1<double *, 2> t_N,
                         FTensor::Tensor1<double *, 3> t_1,
                         FTensor::Tensor1<double *, 3> t_2) {
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;
    FTensor::Tensor2<double, 3, 3> t_n;
    t_n(i, j) = 0;
    t_n(i, j) += FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
    t_n(i, j) -= FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
    return t_n;
  };

  // commonDataSimpleContact->faceRowData = nullptr;

  auto t_F = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->FMat);

  auto t_w = getFTensor0IntegrationWeightMaster();
  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
  auto t_1 = get_tensor_vec(*commonDataSimpleContact->tangentOneVectorSlavePtr);
  auto t_2 = get_tensor_vec(*commonDataSimpleContact->tangentTwoVectorSlavePtr);
  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    auto t_N = col_data.getFTensor1DiffN<2>(gg, 0);
    const double val = 0.5 * t_w * lagrange_slave;

    int bbc = 0;
    for (; bbc != nb_base_fun_col; ++bbc) {

      FTensor::Tensor0<double *> t_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; ++bbr) {

        auto t_assemble = get_tensor2(matLhs, 3 * bbr, 3 * bbc);
        // TODO: handle hoGeometry
        auto t_d_n = make_vec_der(t_N, t_1, t_2);
        t_assemble(i, k) -= val * t_base * t_F(j, i) * t_d_n(j, k);

        ++t_base;
      }
      ++t_N;
    }
    ++t_F;
    ++t_w;
    ++lagrange_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactMaterialMasterSlaveLhs_dX_dLagmult::iNtegrate(
    EntData &row_data,
    EntData &col_data) {

  MoFEMFunctionBegin;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 1, c + 0),
                                         &m(r + 2, c + 0));
  };

  auto t_F = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->FMat);

  auto t_w = getFTensor0IntegrationWeightMaster();

  auto normal_master_at_gp =
      get_tensor_vec(*commonDataSimpleContact->normalVectorMasterPtr);

  const double area_m = commonDataSimpleContact->areaMaster;

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    FTensor::Tensor0<double *> t_col_base(&col_data.getN()(gg, 0));

    const double val = t_w * area_m;

    int bbc = 0;
    for (; bbc != nb_base_fun_col; ++bbc) {

      FTensor::Tensor0<double *> t_row_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; ++bbr) {

        auto t_assemble = get_tensor_from_mat(matLhs, 3 * bbr, bbc);

        t_assemble(i) -=
            val * t_row_base * t_F(j, i) * normal_master_at_gp(j) * t_col_base;

        ++t_row_base;
      }
      ++t_col_base;
    }
    ++t_F;
    ++t_w;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactMaterialSlaveSlaveLhs_dX_dLagmult::iNtegrate(
    EntData &row_data,
    EntData &col_data) {

  MoFEMFunctionBegin;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 1, c + 0),
                                         &m(r + 2, c + 0));
  };

  auto t_F = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->FMat);

  auto t_w = getFTensor0IntegrationWeightSlave();

  auto normal_master_at_gp =
      get_tensor_vec(*commonDataSimpleContact->normalVectorSlavePtr);

  const double area_m = commonDataSimpleContact->areaSlave;

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    FTensor::Tensor0<double *> t_col_base(&col_data.getN()(gg, 0));

    const double val = t_w * area_m;

    int bbc = 0;
    for (; bbc != nb_base_fun_col; ++bbc) {

      FTensor::Tensor0<double *> t_row_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; ++bbr) {

        auto t_assemble = get_tensor_from_mat(matLhs, 3 * bbr, bbc);

        t_assemble(i) -=
            val * t_row_base * t_F(j, i) * normal_master_at_gp(j) * t_col_base;

        ++t_row_base;
      }
      ++t_col_base;
    }
    ++t_F;
    ++t_w;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpContactMaterialVolOnSideLhs::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;
  if (row_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);
  row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);
  nb_gauss_pts = row_data.getN().size1();

  nb_base_fun_row = row_data.getFieldData().size() / 3;
  nb_base_fun_col = col_data.getFieldData().size() / 3;

  matLhs.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  matLhs.clear();

  normalVector->resize(3, false);
  tangentOne->resize(3, false);
  tangentTwo->resize(3, false);

  if (isMaster) {
    normalVector = commonDataSimpleContact->normalVectorMasterPtr;
    tangentOne = commonDataSimpleContact->tangentOneVectorMasterPtr;
    tangentTwo = commonDataSimpleContact->tangentOneVectorMasterPtr;
    aRea = commonDataSimpleContact->areaMaster;
  } else {
    normalVector = commonDataSimpleContact->normalVectorSlavePtr;
    tangentOne = commonDataSimpleContact->tangentOneVectorSlavePtr;
    tangentTwo = commonDataSimpleContact->tangentOneVectorSlavePtr;
    aRea = commonDataSimpleContact->areaSlave;
  }

  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data, col_data);

  // assemble local matrix
  CHKERR aSsemble(row_data, col_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpContactMaterialVolOnSideLhs::aSsemble(
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;

  // get pointer to first global index on row
  const int *row_indices = &*row_data.getIndices().data().begin();
  // get pointer to first global index on column
  const int *col_indices = &*col_data.getIndices().data().begin();

  auto &data = *commonDataSimpleContact;
  if (!data.forcesOnlyOnEntitiesRow.empty()) {
    rowIndices.resize(row_nb_dofs, false);
    noalias(rowIndices) = row_data.getIndices();
    row_indices = &rowIndices[0];
    VectorDofs &dofs = row_data.getFieldDofs();
    VectorDofs::iterator dit = dofs.begin();
    for (int ii = 0; dit != dofs.end(); ++dit, ++ii) {
      if (data.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
          data.forcesOnlyOnEntitiesRow.end()) {
        rowIndices[ii] = -1;
      }
    }
  }

  // assemble local matrix
  CHKERR MatSetValues(getSNESB(), row_nb_dofs, row_indices, col_nb_dofs,
                      col_indices, &*matLhs.data().begin(), ADD_VALUES);

  MoFEMFunctionReturn(0);
}

// setup operators for calculation of active set
MoFEMErrorCode SimpleContactProblem::setContactOperatorsRhsALEMaterial(
    boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact_ale,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    const string field_name, const string mesh_node_field_name,
    const string lagrange_field_name, const string side_fe_name) {
  MoFEMFunctionBegin;

  boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnContactPrismSide>
      fe_mat_side_rhs_master = boost::make_shared<
          VolumeElementForcesAndSourcesCoreOnContactPrismSide>(mField);

  fe_mat_side_rhs_master->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<3, 3>(
          mesh_node_field_name, common_data_simple_contact->HMat));
  fe_mat_side_rhs_master->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<3, 3>(
          field_name, common_data_simple_contact->hMat));

  boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnContactPrismSide>
      fe_mat_side_rhs_slave = boost::make_shared<
          VolumeElementForcesAndSourcesCoreOnContactPrismSide>(mField);

  fe_mat_side_rhs_slave->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<3, 3>(
          mesh_node_field_name, common_data_simple_contact->HMat));
  fe_mat_side_rhs_slave->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<3, 3>(
          field_name, common_data_simple_contact->hMat));

  fe_rhs_simple_contact_ale->getOpPtrVector().push_back(new OpGetNormalSlaveALE(
      "MESH_NODE_POSITIONS", common_data_simple_contact));

  fe_rhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpGetNormalMasterALE("MESH_NODE_POSITIONS",
                               common_data_simple_contact));

  // fe_rhs_simple_contact_ale->getOpPtrVector().push_back(
  //     new OpGetPositionAtGaussPtsMaster(field_name,
  //                                       common_data_simple_contact));

  // fe_rhs_simple_contact_ale->getOpPtrVector().push_back(
  //     new OpGetPositionAtGaussPtsSlave(field_name,
  //     common_data_simple_contact));

  // fe_rhs_simple_contact_ale->getOpPtrVector().push_back(
  //     new OpGetGapSlave(field_name, common_data_simple_contact));

  fe_rhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrange_field_name,
                                     common_data_simple_contact));

  fe_mat_side_rhs_master->getOpPtrVector().push_back(new OpCalculateDeformation(
      mesh_node_field_name, common_data_simple_contact, false));

  fe_rhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpLoopForSideOfContactPrism(mesh_node_field_name,
                                      fe_mat_side_rhs_master, side_fe_name,
                                      ContactOp::FACEMASTER));

  fe_rhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpCalMatForcesALEMaster(mesh_node_field_name,
                                  common_data_simple_contact));

  fe_mat_side_rhs_slave->getOpPtrVector().push_back(new OpCalculateDeformation(
      mesh_node_field_name, common_data_simple_contact, false));

  fe_rhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpLoopForSideOfContactPrism(mesh_node_field_name,
                                      fe_mat_side_rhs_slave, side_fe_name,
                                      ContactOp::FACESLAVE));

  fe_rhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpCalMatForcesALESlave(mesh_node_field_name,
                                 common_data_simple_contact));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactOperatorsLhsALEMaterial(
    boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact_ale,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    const string field_name, const string mesh_node_field_name,
    const string lagrange_field_name, const string side_fe_name) {
  MoFEMFunctionBegin;

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(new OpGetNormalSlaveALE(
      mesh_node_field_name, common_data_simple_contact));

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpGetNormalMasterALE(mesh_node_field_name,
                               common_data_simple_contact));

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrange_field_name,
                                     common_data_simple_contact));

  boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnContactPrismSide>
      feMatSideLhs_dx = boost::make_shared<
          VolumeElementForcesAndSourcesCoreOnContactPrismSide>(mField);

  feMatSideLhs_dx->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<3, 3>(
          mesh_node_field_name, common_data_simple_contact->HMat));

  feMatSideLhs_dx->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<3, 3>(
          field_name, common_data_simple_contact->hMat));

  //   // Master derivative over spatial
  feMatSideLhs_dx->getOpPtrVector().push_back(new OpCalculateDeformation(
      mesh_node_field_name, common_data_simple_contact, false));

  feMatSideLhs_dx->getOpPtrVector().push_back(
      new OpContactMaterialVolOnSideLhs_dX_dX(
          mesh_node_field_name, mesh_node_field_name,
          common_data_simple_contact, true));

  feMatSideLhs_dx->getOpPtrVector().push_back(
      new OpContactMaterialVolOnSideLhs_dX_dx(
          mesh_node_field_name, field_name, common_data_simple_contact, true));

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpLoopForSideOfContactPrism(mesh_node_field_name, feMatSideLhs_dx,
                                      side_fe_name, ContactOp::FACEMASTER));

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpContactMaterialMasterOnFaceLhs_dX_dX(
          mesh_node_field_name, mesh_node_field_name,
          common_data_simple_contact, POSITION_RANK, POSITION_RANK));

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpContactMaterialMasterSlaveLhs_dX_dLagmult(
          mesh_node_field_name, lagrange_field_name, common_data_simple_contact,
          POSITION_RANK, LAGRANGE_RANK));

  boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnContactPrismSide>
      feMatSideLhsSlave_dx = boost::make_shared<
          VolumeElementForcesAndSourcesCoreOnContactPrismSide>(mField);

  feMatSideLhsSlave_dx->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<3, 3>(
          mesh_node_field_name, common_data_simple_contact->HMat));

  feMatSideLhsSlave_dx->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<3, 3>(
          field_name, common_data_simple_contact->hMat));

  feMatSideLhsSlave_dx->getOpPtrVector().push_back(new OpCalculateDeformation(
      mesh_node_field_name, common_data_simple_contact, false));

  feMatSideLhsSlave_dx->getOpPtrVector().push_back(
      new OpContactMaterialVolOnSideLhs_dX_dX(
          mesh_node_field_name, mesh_node_field_name,
          common_data_simple_contact, false));

  feMatSideLhsSlave_dx->getOpPtrVector().push_back(
      new OpContactMaterialVolOnSideLhs_dX_dx(
          mesh_node_field_name, field_name, common_data_simple_contact, false));

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpLoopForSideOfContactPrism(mesh_node_field_name,
                                      feMatSideLhsSlave_dx, side_fe_name,
                                      ContactOp::FACESLAVE));

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpContactMaterialSlaveOnFaceLhs_dX_dX(
          mesh_node_field_name, mesh_node_field_name,
          common_data_simple_contact, POSITION_RANK, POSITION_RANK));

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpContactMaterialSlaveSlaveLhs_dX_dLagmult(
          mesh_node_field_name, lagrange_field_name, common_data_simple_contact,
          POSITION_RANK, LAGRANGE_RANK));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactOperatorsLhsALE(
    boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact_ale,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    const string field_name, const string mesh_node_field_name,
    const string lagrange_field_name, bool is_eigen_pos_field,
    string eigen_pos_field_name) {
  MoFEMFunctionBegin;

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(new OpGetNormalSlaveALE(
      mesh_node_field_name, common_data_simple_contact));

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpGetNormalMasterALE(mesh_node_field_name,
                               common_data_simple_contact));

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsMaster(field_name,
                                        common_data_simple_contact));

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsSlave(field_name, common_data_simple_contact));

  if (is_eigen_pos_field) {
    fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
        new OpGetDeformationFieldForDisplAtGaussPtsMaster(
            eigen_pos_field_name, common_data_simple_contact));

    fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
        new OpGetDeformationFieldForDisplAtGaussPtsSlave(
            eigen_pos_field_name, common_data_simple_contact));
  }

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpGetGapSlave(field_name, common_data_simple_contact));

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrange_field_name,
                                     common_data_simple_contact));

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpContactTractionSlaveSlave_dX(field_name, mesh_node_field_name,
                                         common_data_simple_contact,
                                         POSITION_RANK, POSITION_RANK));

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpContactTractionMasterSlave_dX(field_name, mesh_node_field_name,
                                          common_data_simple_contact,
                                          POSITION_RANK, POSITION_RANK));

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpContactTractionMasterMaster_dX(field_name, mesh_node_field_name,
                                           common_data_simple_contact,
                                           POSITION_RANK, POSITION_RANK));

  fe_lhs_simple_contact_ale->getOpPtrVector().push_back(
      new OpCalDerIntCompFunSlaveSlave_dX(
          lagrange_field_name, mesh_node_field_name, cnValuePtr,
          common_data_simple_contact, LAGRANGE_RANK, POSITION_RANK));
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

MoFEMErrorCode SimpleContactProblem::OpGetLagMulAtGaussPtsSlave3D::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  if (type == MBVERTEX) {
    commonDataSimpleContact->allLambdasPtr.get()->resize(3, nb_gauss_pts,
                                                         false);
    commonDataSimpleContact->allLambdasPtr.get()->clear();

  }

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'I', 2> I;
  int nb_base_fun_row = data.getFieldData().size() / 2;

  auto t_lagrange_slave_3 =
      getFTensor1FromMat<3>(*commonDataSimpleContact->allLambdasPtr);

  auto get_shift_tensor = [](auto &m) {
    return FTensor::Tensor2<FTensor::PackPtr<double *, 1>, 2, 3>(
        &m(0, 0), &m(1, 0), &m(2, 0), &m(0, 1), &m(1, 1), &m(2, 1));
  };

  auto t_shift_tensor =
      get_shift_tensor(*(commonDataSimpleContact->shiftTensor));

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    FTensor::Tensor0<double *> t_base_lambda(&data.getN()(gg, 0));

    FTensor::Tensor1<double *, 2> t_field_data_slave(
        &data.getFieldData()[0], &data.getFieldData()[1], 2);

    for (int bb = 0; bb != nb_base_fun_row; ++bb) {
      t_lagrange_slave_3(i) += t_base_lambda * t_shift_tensor(I,i) * t_field_data_slave(I);
      ++t_base_lambda;
      ++t_field_data_slave;
    }

    ++t_lagrange_slave_3;
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

  commonDataSimpleContact->tangentGapPtr.get()->resize(3, nb_gauss_pts, false);
  commonDataSimpleContact->tangentGapPtr.get()->clear();

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  auto t_position_master_gp = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsMasterPtr);

  auto t_position_slave_gp = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsSlavePtr);

  auto t_prev_position_master_gp = getFTensor1FromMat<3>(
      *commonDataSimpleContact->prevPositionAtGaussPtsMasterPtr);

  auto t_prev_position_slave_gp = getFTensor1FromMat<3>(
      *commonDataSimpleContact->prevPositionAtGaussPtsSlavePtr);

  auto t_tangent_gap_ptr =
      getFTensor1FromMat<3>(*commonDataSimpleContact->tangentGapPtr);
  
  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();

    auto t_normal =
      get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr));
  
  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    t_tangent_gap_ptr(i) = 
       (t_kd(i, j) - t_normal(i) * t_normal(j)) *
        (t_position_slave_gp(j) - t_position_master_gp(j) -
         (t_prev_position_slave_gp(j) - t_prev_position_master_gp(j)));
    ++t_position_slave_gp;
    ++t_position_master_gp;
    ++t_prev_position_slave_gp;
    ++t_prev_position_master_gp;
    ++t_tangent_gap_ptr;
  } // for gauss points

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
    commonDataSimpleContact->tangentLambdasPtr.get()->resize(3, nb_gauss_pts,
                                                             false);

    commonDataSimpleContact->tangentLambdasPtr.get()->clear();
  }

  auto t_tangent_lagrange =
      getFTensor1FromMat<3>(*commonDataSimpleContact->tangentLambdasPtr);

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  auto t_lagrange_slave_3 =
      getFTensor1FromMat<3>(*commonDataSimpleContact->allLambdasPtr);
  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();

  auto t_normal =
      get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr));
  
  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    t_tangent_lagrange(i) = ( t_kd(i, j) - t_normal(i)* t_normal(j)) * t_lagrange_slave_3(j);

    ++t_tangent_lagrange;
    ++t_lagrange_slave_3;
  }

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
      3, nb_gauss_pts, false);

  commonDataSimpleContact->tangentAugmentedLambdasPtr.get()->clear();
  commonDataSimpleContact->normAugTangentLambdasPtr.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->normAugTangentLambdasPtr.get()->clear();

  auto t_tangent_augmented_lagrange = getFTensor1FromMat<3>(
      *commonDataSimpleContact->tangentAugmentedLambdasPtr);

  auto t_tangent_lagrange =
      getFTensor1FromMat<3>(*commonDataSimpleContact->tangentLambdasPtr);

  auto t_tangent_gap_ptr =
      getFTensor1FromMat<3>(*commonDataSimpleContact->tangentGapPtr);

  auto t_norm_tan_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->normAugTangentLambdasPtr);

  FTensor::Index<'i', 3> i;

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    t_tangent_augmented_lagrange(i) =
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
SimpleContactProblem::OpCalAugmentedTangentTractionsRhsMaster::doWork(
    int side, EntityType type, EntData &data) {
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
        commonDataSimpleContact->areaMaster; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    auto t_w = getFTensor0IntegrationWeightMaster();

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<3>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_norm_of_tan_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->normAugTangentLambdasPtr);

    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
    auto normal =
        get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      // cerr << " OpCalAugmentedTangentTractionsRhsMaster\n";
      if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
        // cerr << " RHS NO CONTACT!\n";
        ++t_aug_lambda_ptr;
        ++t_tangent_aug_lambda_ptr;
        ++t_norm_of_tan_aug_lambda_ptr;
        ++t_w;
        // cerr << "OPEN 3 !\n";
        continue;
      }
    
      double stick_slip_check =
          t_norm_of_tan_aug_lambda_ptr + muTan * t_aug_lambda_ptr;
      bool stick = true;
      if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
        stick = false;

      const double val_m = t_w * area_m;

      FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

      for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

        auto t_assemble_m = get_tensor_vec(vecF, 3 * bbc);

        if (stick) { // stick
              t_assemble_m(i) -=
              val_m * t_tangent_aug_lambda_ptr(i) *
              t_base_master;
              // cerr << "stick\n";
              // cerr << "STICK 3 !\n";
        } else { // slip
// cerr << "slip\n";
            //Right
            t_assemble_m(i) +=
              val_m * t_tangent_aug_lambda_ptr(i) *
              muTan * t_aug_lambda_ptr * t_base_master / t_norm_of_tan_aug_lambda_ptr;
              

        // t_assemble_m(i) +=  normal(i) / t_norm_of_tan_aug_lambda_ptr;

// cerr << "SLIP 3 !\n";
        }
        ++t_base_master;
      }

      ++t_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_norm_of_tan_aug_lambda_ptr;
      ++t_w;
    } // for gauss points

    CHKERR VecSetValues(getSNESf(), data, &vecF[0], ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalAugmentedTangentTractionsRhsSlave::doWork(
    int side, EntityType type, EntData &data) {
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
    FTensor::Index<'j', 3> j;

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<3>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_norm_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->normAugTangentLambdasPtr);

    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
    auto normal =
        get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
        ++t_aug_lambda_ptr;
        ++t_tangent_aug_lambda_ptr;
        ++t_norm_aug_lambda_ptr;
        ++t_w;
        // cerr << "OPEN 2!\n";
        continue;
      }

      double stick_slip_check =
          t_norm_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

      bool stick = true;
      if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
        stick = false;

      const double val_m = t_w * area_m;

      FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));

      for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

        auto t_assemble_m = get_tensor_vec(vecF, 3 * bbc);

        if (stick) { // stick

          t_assemble_m(i) += val_m *
                             t_tangent_aug_lambda_ptr(i) * t_base_slave;
          // cerr << "STICK 2!\n";

        } else { // slip
          t_assemble_m(i) -= val_m *
                             t_tangent_aug_lambda_ptr(i) * muTan *
                             t_aug_lambda_ptr * t_base_slave /
                             t_norm_aug_lambda_ptr;
          // cerr << "SLIP 2 !\n";
        }

        ++t_base_slave;
      }

      ++t_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_norm_aug_lambda_ptr;
      ++t_w;
    } // for gauss points

    CHKERR VecSetValues(getSNESf(), data, &vecF[0], ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalAugmentedTangentialContCondition::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    const int nb_gauss_pts = data.getN().size1();
    const int nb_base_fun_col = data.getFieldData().size() / 2;

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
    FTensor::Index<'J', 2> J;

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<3>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_tangent_gap =
        getFTensor1FromMat<3>(*commonDataSimpleContact->tangentGapPtr);

    auto t_norm_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->normAugTangentLambdasPtr);

    auto t_tangent_lambda =
        getFTensor1FromMat<3>(*commonDataSimpleContact->tangentLambdasPtr);

    auto get_shift_tensor = [](auto &m) {
      return FTensor::Tensor2<FTensor::PackPtr<double *, 1>, 2, 3>(
          &m(0, 0), &m(1, 0), &m(2, 0), &m(0, 1), &m(1, 1), &m(2, 1));
    };

    auto t_shift_tensor =
        get_shift_tensor(*(commonDataSimpleContact->shiftTensor));

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      double stick_slip_check =
          t_norm_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

      bool stick = true;
      if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
        stick = false;

      const double val_m = t_w * area_m;

      FTensor::Tensor0<double *> t_base_slave_lambda(&data.getN()(gg, 0));

      for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

        auto t_assemble_m = get_tensor_vec_2(vecF, 2 * bbc);
        // cerr << "t_aug_lambda_ptr   " << t_aug_lambda_ptr << "\n";
        if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
// cerr << "OPEN!\n";
          t_assemble_m(J) -=
              val_m * t_shift_tensor(J,j) * t_tangent_lambda(j) * t_base_slave_lambda / cNTangentPtr;

        } else {
          if (stick) { // stick
// cerr << "STICK!\n";
            t_assemble_m(J) += val_m * t_shift_tensor(J,j) * t_tangent_gap(j) * t_base_slave_lambda;

          } else { // slip
// cerr << "SLIP!\n";
            ///======
            t_assemble_m(J) -= val_m * t_shift_tensor(J,j) *
                               (t_tangent_lambda(j) +
                                t_tangent_aug_lambda_ptr(j) * muTan *
                                    t_aug_lambda_ptr / t_norm_aug_lambda_ptr) *
                               t_base_slave_lambda / cNTangentPtr;
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
SimpleContactProblem::OpContactAugmentedFrictionMasterMaster::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data,
    EntData &col_data) {
  MoFEMFunctionBegin;
  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

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
  FTensor::Index<'k', 3> k;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto t_w = getFTensor0IntegrationWeightMaster();

  const double area_master = commonDataSimpleContact->areaMaster;

  auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<3>(
      *commonDataSimpleContact->tangentAugmentedLambdasPtr);

  auto t_norm_tang_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->normAugTangentLambdasPtr);

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  // auto t_tangent_lambda =
  //     getFTensor1FromMat<3>(*commonDataSimpleContact->tangentLambdasPtr);

  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
  auto t_normal =
      get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

  // const double cNTangentPtr = *cNTangentPtr.get();
  // const double cNNormalPtr = *cNNormalPtr.get();
  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    
    if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
      ++t_norm_tang_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_aug_lambda_ptr;
      ++t_w;
      // cerr << "no contact LHS\n";
      // PetscPrintf(PETSC_COMM_WORLD, "Pets no contact\n");
      continue;
    }

    double stick_slip_check =
        t_norm_tang_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

    // cerr << " stick_slip_check " << stick_slip_check << "\n";

    bool stick = true;
    if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
      stick = false;
// cerr << "stick LHS " << stick << "\n";
    const double val_m = t_w * area_master;

    FTensor::Tensor0<double *> t_base_master_col(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

      FTensor::Tensor0<double *> t_base_master_row(&row_data.getN()(gg, 0));
      const double m = val_m * t_base_master_col;

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);
        
        if (stick) {  
          // cerr << "stick\n";
          t_assemble_s(i, j) += m * t_base_master_row * cNTangentPtr * (t_kd(i, j) - t_normal(i) * t_normal(j));
        } else {
          // cerr << "slip\n";
          const double prod_val = muTan * m * t_base_master_row ;
          long double inv_leng = 1. / t_norm_tang_aug_lambda_ptr;
          // cerr << " cNNormalPtr " << cNNormalPtr << "\n";
          // Right 1
          t_assemble_s(i, j) += prod_val * cNNormalPtr *
                                t_tangent_aug_lambda_ptr(i) * t_normal(j) / t_norm_tang_aug_lambda_ptr;          
          //Right 2)
          t_assemble_s(i, j) -= t_aug_lambda_ptr * cNTangentPtr * prod_val *
                                (t_kd(i, j) - t_normal(i) * t_normal(j)) / t_norm_tang_aug_lambda_ptr;

          long double pow_3 = 1. / (t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr);
          
          // long double pow_3 = (t_norm_tang_aug_lambda_ptr *
          //                      t_norm_tang_aug_lambda_ptr *
          //                      t_norm_tang_aug_lambda_ptr);
          
          //Right 3)
          t_assemble_s(i, j) += t_aug_lambda_ptr * cNTangentPtr * prod_val *
                                t_tangent_aug_lambda_ptr(i) *
                                t_tangent_aug_lambda_ptr(k) * (t_kd(k, j) - t_normal(k) * t_normal(j)) * pow_3;
                                
// cerr <<" norm " << t_norm_tang_aug_lambda_ptr << "\n";
// cerr << " vector " << t_tangent_aug_lambda_ptr << "\n";
//           t_assemble_s(i, j) += cNTangentPtr * t_base_master_col *
//                                 t_normal(i) * (t_kd(k, j) - t_normal(k) * t_normal(j))* 
//                                 t_tangent_aug_lambda_ptr(k) * pow_3;



          // FTensor::Tensor1<double, 3> t1;
          // t1(i) = 1.;
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
    }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactAugmentedFrictionMasterSlave::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data,
    EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {

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
  FTensor::Index<'k', 3> k;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto t_w = getFTensor0IntegrationWeightMaster();

  const double area_master = commonDataSimpleContact->areaMaster;

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<3>(
      *commonDataSimpleContact->tangentAugmentedLambdasPtr);

  auto t_norm_tang_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->normAugTangentLambdasPtr);

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  auto t_tangent_lambda =
      getFTensor1FromMat<3>(*commonDataSimpleContact->tangentLambdasPtr);

  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
  auto t_normal =
      get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

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
          t_assemble_s(i, j) -= m * cNTangentPtr * (t_kd(i, j) - t_normal(i)* t_normal(j)) * t_base_master_row;
        } else {


          t_assemble_s(i, j) -= muTan * m * cNNormalPtr *
                                t_tangent_aug_lambda_ptr(i) * normal(j) *
                                t_base_master_row / t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * (t_kd(i, j) - t_normal(i)* t_normal(j)) * t_base_master_row /
                                t_norm_tang_aug_lambda_ptr;

          const double pow_3 = t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr * m *
                                t_tangent_aug_lambda_ptr(i) *
                                t_tangent_aug_lambda_ptr(k) * (t_kd(k, j) - t_normal(k)* t_normal(j)) *
                                t_base_master_row / pow_3;
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
    }

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
    int nb_base_fun_col = col_data.getFieldData().size();

    const double area_master =
        commonDataSimpleContact->areaMaster; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor1<double *, 3>(&m(r + 0, c), &m(r + 1, c),
                                           &m(r + 2, c));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
    NN.clear();

    auto t_w = getFTensor0IntegrationWeightMaster();

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<3>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_norm_tang_aug_lambda_ptr = getFTensor0FromVec(
        *commonDataSimpleContact->normAugTangentLambdasPtr);

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_lambda =
        getFTensor1FromMat<3>(*commonDataSimpleContact->tangentLambdasPtr);

    auto normal =
        get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);
    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();

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
        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, 0);
        auto t_base_lambda = col_data.getFTensor0N(gg, 0);
        const double m = val_m * t_base_master;
        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
        
          const double n = m * t_base_lambda;

          double stick_slip_check =
              t_norm_tang_aug_lambda_ptr + muTan * t_aug_lambda_ptr;
          bool stick = true;
          if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
            stick = false;
          if (stick) {
          } else {
            t_assemble_s(i) += muTan * n * t_tangent_aug_lambda_ptr(i) / t_norm_tang_aug_lambda_ptr;
          }
            ++t_base_lambda; // update cols slave
            ++t_assemble_s;
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
    int nb_base_fun_col = col_data.getFieldData().size() / 2;

    const double area_master =
        commonDataSimpleContact->areaMaster; // same area in master and slave

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 3, 2>(
          &m(r + 0, c + 0), &m(r + 0, c + 1),
          &m(r + 1, c + 0), &m(r + 1, c + 1),
          &m(r + 2, c + 0), &m(r + 2, c + 1));
    };

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;

    FTensor::Index<'I', 2> I;
    FTensor::Index<'J', 2> J;
    FTensor::Index<'K', 2> K;


    NN.resize(3 * nb_base_fun_row, 2 * nb_base_fun_col, false);
    NN.clear();

    auto t_w = getFTensor0IntegrationWeightMaster();

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<3>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_norm_tang_aug_lambda_ptr = getFTensor0FromVec(
        *commonDataSimpleContact->normAugTangentLambdasPtr);

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_lambda =
        getFTensor1FromMat<3>(*commonDataSimpleContact->tangentLambdasPtr);

    auto t_normal =
        get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

    VectorDouble unit_vec;
    unit_vec.resize(3, false);
    unit_vec(0) = unit_vec(1) = unit_vec(2) = 1.;
    auto t_unit = get_tensor_vec(unit_vec);
    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();

  auto get_shift_tensor = [](auto &m) {
    return FTensor::Tensor2<FTensor::PackPtr<double *, 1>, 2, 3>(
        &m(0, 0), &m(1, 0), &m(2, 0), &m(0, 1), &m(1, 1), &m(2, 1));
  };

    auto t_shift_tensor =
        get_shift_tensor(*(commonDataSimpleContact->shiftTensor));

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
          auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 2 * bbc);
          if (stick) {

            t_assemble_m(i, J) -=
                n * t_shift_tensor(J, j) * (t_kd(i, j) -  t_normal(i) * t_normal(j)) ;

          } else {

            // cerr << "Slip\n";
            const double pow_3 = t_norm_tang_aug_lambda_ptr *
                                 t_norm_tang_aug_lambda_ptr *
                                 t_norm_tang_aug_lambda_ptr;

        t_assemble_m(i, J) += n * muTan * t_aug_lambda_ptr *
                                  t_shift_tensor(J, j) * (t_kd(i, j) -  t_normal(i) * t_normal(j)) 
                                  / t_norm_tang_aug_lambda_ptr;

        t_assemble_m(i, J) -=
            n * muTan * t_aug_lambda_ptr * t_shift_tensor(J, j) * t_tangent_aug_lambda_ptr(i) *
            t_tangent_aug_lambda_ptr(j) / pow_3;
          }
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
    EntData &row_data,
    EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {

  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = col_data.getFieldData().size() / 3;
// cerr << "col_data.getIndices() new  " << col_data.getIndices() << "\n";
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
  FTensor::Index<'k', 3> k;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto t_w = getFTensor0IntegrationWeightSlave();

  const double area_master = commonDataSimpleContact->areaSlave;

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<3>(
      *commonDataSimpleContact->tangentAugmentedLambdasPtr);

  auto t_norm_tang_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->normAugTangentLambdasPtr);

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  auto t_tangent_lambda =
      getFTensor1FromMat<3>(*commonDataSimpleContact->tangentLambdasPtr);

  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
// cerr << "col_data.getIndices()  " << col_data.getIndices() <<"\n";
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
          // cerr << "stick \n";
          t_assemble_s(i, j) += m * cNTangentPtr * (t_kd(i, j) - normal(i) * normal(j))
                                 * t_base_slave_row;
          // t_assemble_s(i, j) += m * cNTangentPtr * t_base_slave_row;
        } else {
          // cerr << "slip \n";
          t_assemble_s(i, j) += muTan * m * cNNormalPtr *
                                t_tangent_aug_lambda_ptr(i) * normal(j) *
                                t_base_slave_row / t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * (t_kd(i, j) - normal(i) * normal(j)) * t_base_slave_row /
                                t_norm_tang_aug_lambda_ptr;

          const double pow_3 = t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr *
                                m * t_tangent_aug_lambda_ptr(i) *
                                t_tangent_aug_lambda_ptr(j) *
                                t_base_slave_row / pow_3;
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
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactAugmentedFrictionSlaveMaster::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data,
    EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {

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
  FTensor::Index<'k', 3> k;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto t_w = getFTensor0IntegrationWeightSlave();

  const double area_master = commonDataSimpleContact->areaSlave;

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<3>(
      *commonDataSimpleContact->tangentAugmentedLambdasPtr);

  auto t_norm_tang_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->normAugTangentLambdasPtr);

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  auto t_tangent_lambda =
      getFTensor1FromMat<3>(*commonDataSimpleContact->tangentLambdasPtr);

  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();

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
          t_assemble_s(i, j) -= m * cNTangentPtr * (t_kd(i, j) - normal(i) * normal(j) ) * t_base_slave_row;
        } else {

          t_assemble_s(i, j) -= muTan * m * cNNormalPtr *
                                t_tangent_aug_lambda_ptr(i) *
                                normal(j) *
                                t_base_slave_row / t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) += muTan * t_aug_lambda_ptr * cNTangentPtr * m *
                                (t_kd(i, j) - normal(i) * normal(j) ) *
                                t_base_slave_row / t_norm_tang_aug_lambda_ptr;

          const double pow_3 = t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr;

          t_assemble_s(i, j) -= muTan * t_aug_lambda_ptr * cNTangentPtr * m *
                                t_tangent_aug_lambda_ptr(i) *
                                t_tangent_aug_lambda_ptr(k) * (t_kd(k, j) - normal(k) * normal(j) ) *
                                t_base_slave_row / pow_3;
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
  }
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
    int nb_base_fun_col = col_data.getFieldData().size();

    const double area_master =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor1<double *, 3>(&m(r + 0, c), &m(r + 1, c),
                                           &m(r + 2, c));
    };

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
    NN.clear();

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<3>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_norm_tang_aug_lambda_ptr = getFTensor0FromVec(
        *commonDataSimpleContact->normAugTangentLambdasPtr);

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_lambda =
        getFTensor1FromMat<3>(*commonDataSimpleContact->tangentLambdasPtr);

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
          const double n = m * t_base_lambda;

          double stick_slip_check =
              t_norm_tang_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

          bool stick = true;
          if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
            stick = false;

          if (stick) {
          } else {
            t_assemble_m(i) -= muTan * n * t_tangent_aug_lambda_ptr(i) / t_norm_tang_aug_lambda_ptr;
          }
          ++t_base_lambda; // update cols slave
          ++t_assemble_m;
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
    int nb_base_fun_col = col_data.getFieldData().size() / 2;

    const double area_master =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 3, 2>(
          &m(r + 0, c + 0), &m(r + 0, c + 1),
          &m(r + 1, c + 0), &m(r + 1, c + 1),
          &m(r + 2, c + 0), &m(r + 2, c + 1));
    };

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;

    FTensor::Index<'J', 2> J;

    NN.resize(3 * nb_base_fun_row, 2 * nb_base_fun_col, false);
    NN.clear();

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<3>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_norm_tang_aug_lambda_ptr = getFTensor0FromVec(
        *commonDataSimpleContact->normAugTangentLambdasPtr);

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_lambda =
        getFTensor1FromMat<3>(*commonDataSimpleContact->tangentLambdasPtr);

    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
    auto t_normal =
        get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

    auto get_shift_tensor = [](auto &m) {
      return FTensor::Tensor2<FTensor::PackPtr<double *, 1>, 2, 3>(
          &m(0, 0), &m(1, 0), &m(2, 0), &m(0, 1), &m(1, 1), &m(2, 1));
    };

    auto t_shift_tensor =
        get_shift_tensor(*(commonDataSimpleContact->shiftTensor));

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
          auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 2 * bbc);

          if (stick) {
            
            t_assemble_m(i, J) +=
                n * t_shift_tensor(J, j) * ( t_kd(i, j) - t_normal(i) * t_normal(j));
            
          } else {
            
            const double pow_3 = t_norm_tang_aug_lambda_ptr *
                                 t_norm_tang_aug_lambda_ptr *
                                 t_norm_tang_aug_lambda_ptr;

            t_assemble_m(i, J) -= n * muTan * t_aug_lambda_ptr *
                                 t_shift_tensor(J, j) * ( t_kd(i, j) - t_normal(i) * t_normal(j)) /
                                  t_norm_tang_aug_lambda_ptr;

            // t_assemble_m(i, j) += n * muTan * 
            //                       t_tangent_aug_lambda_ptr(i) *
            //                        t_normal(j) / pow_3;

            t_assemble_m(i, J) += n * muTan * t_aug_lambda_ptr *
                                  t_shift_tensor(J, j) *
                                  t_tangent_aug_lambda_ptr(i) *
                                  t_tangent_aug_lambda_ptr(j) / pow_3;
          }
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
           EntData &row_data,
           EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {

  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 2;
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 2, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2));
  };

  auto get_tensor_from_mat_1 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 0, c + 1),
                                         &m(r + 0, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'I', 2> I;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  NN.resize(2 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto t_w = getFTensor0IntegrationWeightSlave();

  const double area_master = commonDataSimpleContact->areaSlave;

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<3>(
      *commonDataSimpleContact->tangentAugmentedLambdasPtr);

  auto t_norm_tang_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->normAugTangentLambdasPtr);

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  auto t_tangent_lambda =
      getFTensor1FromMat<3>(*commonDataSimpleContact->tangentLambdasPtr);

  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
  
  auto get_shift_tensor = [](auto &m) {
    return FTensor::Tensor2<FTensor::PackPtr<double *, 1>, 2, 3>(
        &m(0, 0), &m(1, 0), &m(2, 0), &m(0, 1), &m(1, 1), &m(2, 1));
  };

  auto t_shift_tensor =
      get_shift_tensor(*(commonDataSimpleContact->shiftTensor));

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

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        auto t_assemble_s = get_tensor_from_mat(NN, 2 * bbr, 3 * bbc);
        if (stick) {
          t_assemble_s(I, j) += m * t_base_slave_row * t_shift_tensor(I,i) * (t_kd(i, j) - normal(i) * normal(j));
        } else {
          t_assemble_s(I, j) +=
              muTan * m * cNNormalPtr * t_shift_tensor(I,i) * t_tangent_aug_lambda_ptr(i) *
              normal(j) * t_base_slave_row /
              (t_norm_tang_aug_lambda_ptr * cNTangentPtr);
              
          t_assemble_s(I, j) -= muTan * t_aug_lambda_ptr * m * t_shift_tensor(I,i) *
                                (t_kd(i, j) - normal(i) * normal(j)) *
                                t_base_slave_row / (t_norm_tang_aug_lambda_ptr);

          const double pow_3 = t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr;

          t_assemble_s(I, j) +=
              muTan * t_aug_lambda_ptr * m * t_shift_tensor(I,i) * t_tangent_aug_lambda_ptr(i) *
              t_tangent_aug_lambda_ptr(j)  * t_base_slave_row / pow_3;
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
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalAugmentedTangentialContConditionDispSlaveMaster::
    doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
           EntData &row_data,
           EntData &col_data) {
  MoFEMFunctionBegin;

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {

  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 2;
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 2, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2));
  };
  auto get_tensor_from_mat_1 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 0, c + 1),
                                         &m(r + 0, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'I', 2> I;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  NN.resize(2 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto t_w = getFTensor0IntegrationWeightSlave();

  const double area_master = commonDataSimpleContact->areaSlave;

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<3>(
      *commonDataSimpleContact->tangentAugmentedLambdasPtr);

  auto t_norm_tang_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->normAugTangentLambdasPtr);

  auto t_aug_lambda_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

  auto t_tangent_lambda =
      getFTensor1FromMat<3>(*commonDataSimpleContact->tangentLambdasPtr);

  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();

  auto get_shift_tensor = [](auto &m) {
    return FTensor::Tensor2<FTensor::PackPtr<double *, 1>, 2, 3>(
        &m(0, 0), &m(1, 0), &m(2, 0), &m(0, 1), &m(1, 1), &m(2, 1));
  };

  auto t_shift_tensor =
      get_shift_tensor(*(commonDataSimpleContact->shiftTensor));

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

        auto t_assemble_s = get_tensor_from_mat(NN, 2 * bbr, 3 * bbc);

        if (stick) {

          t_assemble_s(I, j) -= m * t_base_slave_row * t_shift_tensor(I, i) * (t_kd(i, j) - normal(i) * normal(j));
        } else {
          t_assemble_s(I, j) -=
              muTan * m * cNNormalPtr * t_shift_tensor(I, i) * t_tangent_aug_lambda_ptr(i) *
              normal(j) * t_base_slave_row /
              (t_norm_tang_aug_lambda_ptr * cNTangentPtr);

          t_assemble_s(I, j) += muTan * t_aug_lambda_ptr * m * t_shift_tensor(I, i) *
                                (t_kd(i, j) - normal(i) * normal(j)) *
                                t_base_slave_row / (t_norm_tang_aug_lambda_ptr);

          const double pow_3 = t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr *
                               t_norm_tang_aug_lambda_ptr;

          t_assemble_s(I, j) -=
              muTan * t_aug_lambda_ptr * m * t_shift_tensor(I, i) * t_tangent_aug_lambda_ptr(i) *
               t_tangent_aug_lambda_ptr(j) * t_base_slave_row / pow_3;
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
  }
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

    int nb_base_fun_row = row_data.getFieldData().size() / 2;
    int nb_base_fun_col = col_data.getFieldData().size();

    const double area_master =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor1<double *, 2>(&m(r + 0, c), &m(r + 1, c));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    FTensor::Index<'I', 2> I;
    FTensor::Index<'J', 2> J;

    NN.resize(2 * nb_base_fun_row, nb_base_fun_col, false);
    NN.clear();

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<3>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_norm_tang_aug_lambda_ptr = getFTensor0FromVec(
        *commonDataSimpleContact->normAugTangentLambdasPtr);

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_lambda =
        getFTensor1FromMat<3>(*commonDataSimpleContact->tangentLambdasPtr);

    auto normal =
        get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

    auto get_shift_tensor = [](auto &m) {
      return FTensor::Tensor2<FTensor::PackPtr<double *, 1>, 2, 3>(
          &m(0, 0), &m(1, 0), &m(2, 0), &m(0, 1), &m(1, 1), &m(2, 1));
    };

    auto t_shift_tensor =
        get_shift_tensor(*(commonDataSimpleContact->shiftTensor));

    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    if (t_aug_lambda_ptr > 0.) {
      ++t_w;
      ++t_norm_tang_aug_lambda_ptr;
      ++t_tangent_aug_lambda_ptr;
      ++t_aug_lambda_ptr;
      continue;
    }

      double val_m = t_w * area_master;
      auto t_base_row_lambda = row_data.getFTensor0N(gg, 0);

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        // auto t_assemble_m = get_tensor_from_mat(NN, 2 * bbr, 0);
        auto t_base_lambda = col_data.getFTensor0N(gg, 0);
        const double m = val_m * t_base_row_lambda;
        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
          const double n = m * t_base_lambda;


          auto t_assemble_m = get_tensor_from_mat(NN, 2 * bbr, bbc);
          // if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {
          // } else {
            double stick_slip_check =
                t_norm_tang_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

            bool stick = true;
            if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
              stick = false;

            if (stick) {

            } else {
              t_assemble_m(I) -= n * muTan * t_shift_tensor(I, i) * t_tangent_aug_lambda_ptr(i) /
                                    (cNTangentPtr * t_norm_tang_aug_lambda_ptr);
            }
          // }
          ++t_base_lambda; // update cols slave
        }
        ++t_base_row_lambda; // update rows master
        // ++t_assemble_m;
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

    int nb_base_fun_row = row_data.getFieldData().size() / 2;
    int nb_base_fun_col = col_data.getFieldData().size() / 2;

    const double area_master =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 2, 2>(
          &m(r + 0, c + 0), &m(r + 0, c + 1),
          &m(r + 1, c + 0), &m(r + 1, c + 1));
    };

    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;

    FTensor::Index<'I', 2> I;
    FTensor::Index<'J', 2> J;

    NN.resize(2 * nb_base_fun_row, 2 * nb_base_fun_col, false);
    NN.clear();

    auto t_w = getFTensor0IntegrationWeightSlave();

    auto t_tangent_aug_lambda_ptr = getFTensor1FromMat<3>(
        *commonDataSimpleContact->tangentAugmentedLambdasPtr);

    auto t_norm_tang_aug_lambda_ptr = getFTensor0FromVec(
        *commonDataSimpleContact->normAugTangentLambdasPtr);

    auto t_aug_lambda_ptr =
        getFTensor0FromVec(*commonDataSimpleContact->augmentedLambdasPtr);

    auto t_tangent_lambda =
        getFTensor1FromMat<3>(*commonDataSimpleContact->tangentLambdasPtr);

    auto normal =
        get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

    auto get_shift_tensor = [](auto &m) {
      return FTensor::Tensor2<FTensor::PackPtr<double *, 1>, 2, 3>(
          &m(0, 0), &m(1, 0), &m(2, 0), &m(0, 1), &m(1, 1), &m(2, 1));
    };

    auto t_shift_tensor =
        get_shift_tensor(*(commonDataSimpleContact->shiftTensor));

    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      double val_m = t_w * area_master;
      auto t_base_row_lambda = row_data.getFTensor0N(gg, 0);

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        auto t_base_lambda = col_data.getFTensor0N(gg, 0);
        const double m = val_m * t_base_row_lambda;
        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
          const double n = m * t_base_lambda;
          auto t_assemble_m = get_tensor_from_mat(NN, 2 * bbr, 2 * bbc);

          if (t_aug_lambda_ptr > 0. /*&& std::abs(t_aug_lambda_ptr) > ALM_TOL*/) {

            t_assemble_m(I, J) -=
                n * t_shift_tensor(I, i) * t_shift_tensor(J, i) / cNTangentPtr;
          } else {
            double stick_slip_check =
                t_norm_tang_aug_lambda_ptr + muTan * t_aug_lambda_ptr;

            bool stick = true;
            if (stick_slip_check > 0. /*&& std::fabs(stick_slip_check) > ALM_TOL*/)
              stick = false;

            if (stick) {

            } else {
              // cerr << "slip\n";

              const double pow_3 = t_norm_tang_aug_lambda_ptr *
                                   t_norm_tang_aug_lambda_ptr *
                                   t_norm_tang_aug_lambda_ptr;

              t_assemble_m(I, J) -=
                  n * t_shift_tensor(I, i) * t_shift_tensor(J, i) / cNTangentPtr;

              t_assemble_m(I, J) -= n * muTan * t_aug_lambda_ptr *
                                    t_shift_tensor(I, i) * t_shift_tensor(J, i) /
                                    (t_norm_tang_aug_lambda_ptr * cNTangentPtr);

              t_assemble_m(I, J) +=
                  n * muTan * t_aug_lambda_ptr *  t_shift_tensor(I, i) * t_tangent_aug_lambda_ptr(i) *
                  t_shift_tensor(J, j)  * t_tangent_aug_lambda_ptr(j) / (pow_3 * cNTangentPtr);

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
SimpleContactProblem::OpCalContactAugmentedTractionOverLambdaSlaveSlave3D::doWork(
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
SimpleContactProblem::OpCalContactAugmentedTractionOverLambdaMasterSlave3D::
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

 MoFEMErrorCode SimpleContactProblem::OpPostProcContactOnTri::doWork(int side, EntityType type,
                                           EntData &data) {
    MoFEMFunctionBegin;

    
    if (type != MBVERTEX)
      MoFEMFunctionReturnHot(0);

    // auto get_tag_handle = [&](auto name, auto size) {
    //   Tag th;
    //   std::vector<double> def_vals(size, 0.0);
    //   CHKERR outputMesh.tag_get_handle(name, size, MB_TYPE_DOUBLE, th,
    //                                    MB_TAG_CREAT | MB_TAG_SPARSE,
    //                                    def_vals.data());
    //   return th;
    // };

    auto get_tag_double = [&](const std::string name, auto size) {
      // double def_vals;
      // def_vals = 0.;
      std::vector<double> def_vals(size, 0.0);
      Tag th;

      CHKERR postProcMesh.tag_get_handle(name.c_str(), size, MB_TYPE_DOUBLE, th,
                                         MB_TAG_CREAT | MB_TAG_SPARSE,
                                         def_vals.data());

      return th;
    };

    auto set_tag = [&](auto th, auto gg, double &data) {
      return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1, &data);
    };

    auto set_tag_vec = [&](auto th, auto gg, MatrixDouble3by3 &vec) {
      // for (auto &v : vec.data())
      //   v = set_float_precision(v);
      return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
                                       &*vec.data().begin());
    };

    auto th_normal = get_tag_double("NORMAL_LAMBDA", 1);
    auto th_tangent = get_tag_double("TANGENT_LAMBDA", 1);
    auto th_ratio = get_tag_double("RATIO_LAMBDA", 1);
    auto th_tangent_diff = get_tag_double("TANGENT_DIFF", 1);
    auto th_tangent_vector_1 = get_tag_double("TANGENT_VECTOR_1", 3);
    auto th_tangent_vector_2 = get_tag_double("TANGENT_VECTOR_2", 3);

    size_t nb_gauss_pts = getGaussPts().size2();
    auto get_tensor_vec = [](VectorDouble &n) {
      return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
    };

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'I', 2> I;

    FTensor::Tensor1<double, 3> t_vec_1;

    FTensor::Tensor1<double, 3> t_vec_2;


    auto t_lagrange_slave_3 =
        getFTensor1FromMat<3>(*commonDataSimpleContact->allLambdasPtr);

    auto t_lagrange_slave_2 =
        getFTensor1FromMat<2>(*commonDataSimpleContact->tanCheckLambdasPtr);

    auto t_lagrange_slave =
        getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
    auto t_normal_at_gp =
        get_tensor_vec(*(commonDataSimpleContact->normalVectorTriPtr));

    FTensor::Tensor1<double, 3> t_tangent;

    auto t_shift_tensor = getFTensor1FromMat<3>(
      *commonDataSimpleContact->shiftTensor);

EntityHandle new_vertex = getFEEntityHandle();
const double *slave_coords_ptr = &(getCoordsAtGaussPts()(0, 0));
CHKERR postProcMesh.create_vertex(slave_coords_ptr, new_vertex);

    CHKERR postProcMesh.tag_set_data(th_tangent_vector_1, &new_vertex, 1,
                                     &t_shift_tensor(0));
    double length = sqrt(t_shift_tensor(i) * t_shift_tensor(i));
    // cerr << length << "\n";
    t_vec_1(i) = t_shift_tensor(i);
    ++t_shift_tensor;
    length = sqrt(t_shift_tensor(i) * t_shift_tensor(i));
    t_vec_2(i) = t_shift_tensor(i);

    // CHKERR set_tag_vec(th_tangent_vector_2, gg, *t_shift_tensor);
    CHKERR postProcMesh.tag_set_data(th_tangent_vector_2, &new_vertex, 1,
                                     &t_shift_tensor(0));
    // cerr << length << "\n";
    double product = sqrt(t_vec_1(i) * t_vec_2(i));
    
    if(isnan(product)){
      // cerr << "t_vec_1  " << t_vec_1 <<"\n";
      // cerr << "t_vec_2  " << t_vec_2 <<"\n";
    }

    // cerr << "product  " << product <<"\n";
    // ++t_shift_tensor;

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      // t_tangent(i) = t_lagrange_slave_3(j) * ( t_kd(i, j) - t_normal_at_gp(i)
      // * t_normal_at_gp(j) );

      double normal_lambda = sqrt(t_lagrange_slave * t_lagrange_slave);
      double tangent_lambda =
          sqrt(t_lagrange_slave_3(i) * t_lagrange_slave_3(i));
      double tangent_check =
          sqrt(t_lagrange_slave_2(I) * t_lagrange_slave_2(I)) - tangent_lambda;
      double ratio = tangent_lambda / normal_lambda;
      CHKERR set_tag(th_normal, gg, normal_lambda);

      CHKERR set_tag(th_tangent, gg, tangent_lambda);
      CHKERR set_tag(th_tangent_diff, gg, tangent_check);
      CHKERR set_tag(th_ratio, gg, ratio);


    // CHKERR set_tag_vec(th_tangent_vector_1, gg, *t_shift_tensor);
      ++t_lagrange_slave_3;
      ++t_lagrange_slave;
      ++t_lagrange_slave_2;
    }

    ++(commonDataSimpleContact->countTri);
    // cerr << "TRI Print !!! " << commonDataSimpleContact->countTri << "\n";

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode SimpleContactProblem::OpGetOrthonormalTangentsTri::doWork(
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

  commonDataSimpleContact->tangentOneVectorTriPtr->resize(3, false);
  commonDataSimpleContact->tangentOneVectorTriPtr->clear();

  commonDataSimpleContact->tangentTwoVectorTriPtr->resize(3, false);
  commonDataSimpleContact->tangentTwoVectorTriPtr->clear();

  auto t_normal =
      get_tensor_vec(*(commonDataSimpleContact->normalVectorTriPtr));

  const double *tangent_one_slave_ptr = &getTangent1()[0];
  const double *tangent_two_slave_ptr = &getTangent2()[0];

  auto t_tangent_1 =
      get_tensor_vec(*(commonDataSimpleContact->tangentOneVectorTriPtr));

  auto t_tangent_2 =
      get_tensor_vec(*(commonDataSimpleContact->tangentTwoVectorTriPtr));

  for (int ii = 0; ii != 3; ++ii) {
    t_tangent_1(ii) = tangent_one_slave_ptr[ii];
    // t_tangent_2(ii) = tangent_two_slave_ptr[ii];
  }

  const double l_tan_1 = sqrt(t_tangent_1(i) * t_tangent_1(i));

  // const double l_tan_2 = sqrt(t_tangent_2(i) * t_tangent_2(i));

  t_tangent_1(i) = t_tangent_1(i) / l_tan_1;

  t_tangent_2(k) = FTensor::levi_civita(i, j, k) * t_normal(i) * t_tangent_1(j);

  MoFEMFunctionReturn(0);
}


MoFEMErrorCode SimpleContactProblem::OpGetLagMulAtGaussPtsTri::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  if (type == MBVERTEX) {
    commonDataSimpleContact->allLambdasPtr.get()->resize(3, nb_gauss_pts,
                                                         false);
    commonDataSimpleContact->allLambdasPtr.get()->clear();
    commonDataSimpleContact->tanCheckLambdasPtr.get()->resize(2, nb_gauss_pts,
                                                         false);
    commonDataSimpleContact->tanCheckLambdasPtr.get()->clear();
  }

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'I', 2> I;

  int nb_base_fun_row = data.getFieldData().size() / 2;

  auto t_lagrange_slave_3 =
      getFTensor1FromMat<3>(*commonDataSimpleContact->allLambdasPtr);

  auto t_lagrange_slave_2 =
      getFTensor1FromMat<2>(*commonDataSimpleContact->tanCheckLambdasPtr);


  auto t_normal =
      get_tensor_vec(*(commonDataSimpleContact->normalVectorSlavePtr));

  auto get_shift_tensor = [](auto &m) {
    return FTensor::Tensor2<FTensor::PackPtr<double *, 1>, 2, 3>(
        &m(0, 0), &m(1, 0), &m(2, 0), &m(0, 1), &m(1, 1), &m(2, 1));
  };

  auto t_shift_tensor =
      get_shift_tensor(*(commonDataSimpleContact->shiftTensor));


  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    FTensor::Tensor0<double *> t_base_lambda(&data.getN()(gg, 0));

    FTensor::Tensor1<double *, 2> t_field_data_slave(
        &data.getFieldData()[0], &data.getFieldData()[1], 2);

    for (int bb = 0; bb != nb_base_fun_row; ++bb) {
      t_lagrange_slave_3(i) += t_base_lambda * t_shift_tensor(I, i) * t_field_data_slave(I);
      t_lagrange_slave_2(I) += t_base_lambda * t_field_data_slave(I);
      ++t_base_lambda;
      ++t_field_data_slave;
    }
    // cerr << "Normal"
    //      << "     " << t_lagrange_slave << "\n";

    ++t_lagrange_slave_3;
    ++t_lagrange_slave_2;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetNormalTri::doWork(int side,
                                                            EntityType type,
                                                            EntData &data) {
  MoFEMFunctionBegin;

if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

++(commonDataSimpleContact->countTri);
// cerr << "TRI!!! " << commonDataSimpleContact->countTri <<"\n";

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  commonDataSimpleContact->normalVectorSlavePtr->resize(3, false);

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto normal_original_slave =
      get_tensor_vec(*commonDataSimpleContact->normalVectorSlavePtr);

  commonDataSimpleContact->tangentOneVectorSlavePtr->resize(3, false);
  commonDataSimpleContact->tangentOneVectorSlavePtr->clear();

  commonDataSimpleContact->tangentTwoVectorSlavePtr->resize(3, false);
  commonDataSimpleContact->tangentTwoVectorSlavePtr->clear();

  auto tangent_0_slave =
      get_tensor_vec(*commonDataSimpleContact->tangentOneVectorSlavePtr);
  auto tangent_1_slave =
      get_tensor_vec(*commonDataSimpleContact->tangentTwoVectorSlavePtr);

  auto t_N = data.getFTensor1DiffN<2>(0, 0);
  auto t_dof = data.getFTensor1FieldData<3>();

  for (unsigned int dd = 0; dd != 3; ++dd) {
    tangent_0_slave(i) += t_dof(i) * t_N(0);
    tangent_1_slave(i) += t_dof(i) * t_N(1);
    ++t_dof;
    ++t_N;
  }

  // commonDataSimpleContact->shiftTensor->resize(3, 2, false);
  // commonDataSimpleContact->shiftTensor->clear();

  normal_original_slave(i) =
      FTensor::levi_civita(i, j, k) * tangent_0_slave(j) * tangent_1_slave(k);

  const double normal_length =
      sqrt(normal_original_slave(i) * normal_original_slave(i));
  normal_original_slave(i) = normal_original_slave(i) / normal_length;

  commonDataSimpleContact->areaSlave = 0.5 * normal_length;


  commonDataSimpleContact->shiftTensor->resize(3, 2, false);
  commonDataSimpleContact->shiftTensor->clear();

  
    VectorDouble eig_values(3, false);
  eig_values.clear();
  MatrixDouble eigen_vectors(3, 3, false);
  eigen_vectors.clear();
  MatrixDouble symm_matrix(3, 3, false);
  symm_matrix.clear();


constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
  auto get_tensor_from_mat = [](MatrixDouble &m) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(0, 0), &m(0, 1), &m(0, 2), &m(1, 0),
        &m(1, 1), &m(1, 2), &m(2, 0), &m(2, 1),
        &m(2, 2));
  };

  auto get_tensor_from_mat_no_p = [](MatrixDouble &m) {
    return FTensor::Tensor2<double, 3, 3>(
        m(0, 0), m(0, 1), m(0, 2), m(1, 0),
        m(1, 1), m(1, 2), m(2, 0), m(2, 1),
        m(2, 2));
  };

  auto get_tensor_from_vec_no_p = [](VectorDouble &m) {
    return FTensor::Tensor1<double, 3>(
        m(0), m(1), m(2));
  };

  auto get_vec_from_mat_by_cols = [](MatrixDouble &m, const int c) {
    return FTensor::Tensor1<double, 3>(m(0, c), m(1, c), m(2, c));
  };

    auto get_vec_from_mat_by_cols_by_ptr = [](MatrixDouble &m, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(0, c), &m(1, c), &m(2, c));
  };

  auto get_vec_from_mat_by_rows = [](MatrixDouble &m, const int r) {
    return FTensor::Tensor1<double *, 3>(&m(r, 0), &m(r, 1), &m(r, 2));
  };


  auto make_givens_rotation = [&](int N, int ii, int jj, double theta) {
    FTensor::Tensor2<double, 3, 3> R;
    R(i, j) = 0.;
    R(0, 0) = R(1, 1) = R(2, 2) = 1.;
    double c = cos(theta);
    double s = sin(theta);
    R(ii, ii) = c;
    R(jj, jj) = c;
    R(ii, jj) = s;
    R(jj, ii) = s;
    return R;
  };

  auto construct_subspace_rotation_mtx = [&](int N, int ii,
                                             FTensor::Tensor1<double, 3> &AnglesCol, FTensor::Tensor2<double, 3, 3> &R) {
  MoFEMFunctionBegin;

    for (int jj = N - 1; jj != ii; --jj) {
    auto r_recusive = make_givens_rotation(N, ii, jj, AnglesCol(jj));
    auto r_copy = R;
    R(i, k) = r_recusive(i, j) * r_copy(j, k);
    }
MoFEMFunctionReturn(0);
    
  };

  auto solve_rotation_angles_in_sub_dim = [&](int N, int ii, FTensor::Tensor1<double, 3> &Vcol) {
    FTensor::Tensor1<double, 3> angles_work;
    angles_work(i) = 0;
    double r = 1.;
    for (int jj = N - 1; jj != ii; --jj) {
      double y = Vcol(jj);
      r = r * cos(angles_work(jj));
      angles_work(jj) = (abs(r) > eps) ? asin(y / r) : 0.;
    }

    return angles_work;
  };

  auto reduce_dimension_by_one = [&](int N, int ii, MatrixDouble &Vwork, MatrixDouble &AnglesMtx) {
    MoFEMFunctionBegin;
    //passing by pointers
    auto angles_mat = get_vec_from_mat_by_rows(AnglesMtx, ii);
    auto t_v_work = get_tensor_from_mat(Vwork);
    
    auto v_col = get_vec_from_mat_by_cols(Vwork, ii);
    
    FTensor::Tensor2<double, 3, 3> t_v_to_mult;
    t_v_to_mult(i, j) = t_v_work(i, j);
    // Vcol ← Vwork[:, i ]
    auto angles_col = solve_rotation_angles_in_sub_dim(N, ii, v_col);

    FTensor::Tensor2<double, 3, 3> R;
    R(i, j) = 0.;
    R(0, 0) = R(1, 1) = R(2, 2) = 1.;
    CHKERR construct_subspace_rotation_mtx(N, ii, angles_col, R);
    //for return
    t_v_work(i, k) = R(j, i) * t_v_to_mult(j, k);
    angles_mat(i) = angles_col(i);
    MoFEMFunctionReturn(0);
  };

  auto orient_eigenvectors = [&](MatrixDouble &eigenVectors) {
    MoFEMFunctionBegin;
    int N = 3;
    FTensor::Tensor1<double, 3> sign_flip_vec;
    MatrixDouble v_work;
    v_work.resize(3, 3, false);
    v_work.clear();

    MatrixDouble angle_mat;
    angle_mat.resize(3, 3, false);
    angle_mat.clear();

    auto t_v_work = get_tensor_from_mat(v_work);
    auto t_eigen_vectors = get_tensor_from_mat(eigenVectors);

    t_v_work(i, j) = t_eigen_vectors(i, j);
    for (int ii = 0; ii != N; ++ii){
      sign_flip_vec(ii) = (abs(t_v_work(ii, ii))  < eps || t_v_work(ii, ii)  > 0 ) ? 1 : -1;
      auto t_v_work_col = get_vec_from_mat_by_cols(v_work, ii);
      auto t_v_work_col_ptr = get_vec_from_mat_by_cols_by_ptr(v_work, ii);
      t_v_work_col_ptr(i) = t_v_work_col(i) * sign_flip_vec(ii);
      CHKERR reduce_dimension_by_one(N, ii, v_work, angle_mat);
    }

    FTensor::Tensor2<double, 3, 3> t_s;
    t_s(i, j) = 0.;
    for (int ii = 0; ii != N; ++ii){
    t_s(ii, ii) = sign_flip_vec(ii);   
    }
    // cerr << " sign_flip_vec " << sign_flip_vec <<"\n";
    
    t_v_work(i, j) = t_eigen_vectors(i, j);
    // cerr << " t_eigen_vectors " << t_eigen_vectors <<"\n";
    // cerr << " t_v_work " << t_v_work <<"\n";
    t_eigen_vectors(i, k) = t_v_work(i, j) * t_s(j, k);
    // cerr << " t_eigen_vectors " << t_eigen_vectors <<"\n";
        // Eor ← Esort
            // return Vor, Eor, AnglesMtx, SignFlipVec, SortIndices
            MoFEMFunctionReturn(0);
  };

  auto t_symm_matrix = get_tensor_from_mat(symm_matrix);

  t_symm_matrix(i, j) = t_kd(i, j) - normal_original_slave(i) * normal_original_slave(j);
// cerr << "t_symm_matrix  " << t_symm_matrix << "\n";
//   cerr << "normal_original_slave " << normal_original_slave << "\n";

  CHKERR computeEigenValuesSymmetric(symm_matrix, eig_values, eigen_vectors);
  
MatrixDouble trans_symm_matrix(3, 3, false);
  trans_symm_matrix.clear();
  
  
  // cerr << "eigen_vectors  " << eigen_vectors << " trans_symm_matrix " <<  trans_symm_matrix << "\n";
    // cerr << "eig_values " << eig_values << "\n";
  auto t_shift_tensor = getFTensor1FromMat<3>(
      *commonDataSimpleContact->shiftTensor);

  auto t_eigen_vals =
      getFTensor0FromVec(eig_values);

      auto nb_uniq = get_uniq_nb<3>(&*eig_values.data().begin());
      
      auto t_eig_vec_to_sort = get_tensor_from_mat_no_p(eigen_vectors);
      auto t_eig_values_to_sort = get_tensor_from_vec_no_p(eig_values);
      auto t_mat_eigen_vectors = get_tensor_from_mat(trans_symm_matrix);
      // if (nb_uniq == 2) {
        // cerr << "before; values:  " << t_eig_values_to_sort
        //      << "  vec: " << t_eig_vec_to_sort << "\n";
        sortEigenVals<3>(t_eig_values_to_sort, t_eig_vec_to_sort, nb_uniq);
        // cerr << "after; values:  " << t_eig_values_to_sort
        //      << "  vec: " << t_eig_vec_to_sort << "\n";
        //      cerr << "1 t_eigen_vectors " << t_eigen_vectors <<"\n";
            //  t_mat_eigen_vectors(i, j) = t_eig_vec_to_sort(j, i);
            //  cerr << "trans_symm_matrix 1 " <<  trans_symm_matrix << "\n"; 
            
            // cerr << "eigen_vectors 1 " << eigen_vectors << "\n";
            auto t_eigen_vectors_2 = get_tensor_from_mat(eigen_vectors); 
            t_eigen_vectors_2(i,j) =t_eig_vec_to_sort(i,j);
            noalias(trans_symm_matrix) = trans(eigen_vectors);
             CHKERR orient_eigenvectors(trans_symm_matrix);
            //  cerr << "eigen_vectors 2 " << eigen_vectors << "\n";
            //  t_mat_eigen_vectors(i, j) = t_eig_vec_to_sort(j, i);

             
        //      cerr << "2 t_eigen_vectors " << t_mat_eigen_vectors <<"\n";
            //  cerr << " trans_symm_matrix " <<  trans_symm_matrix << "\n";
      // }
auto t_eigen_vectors = getFTensor1FromMat<3>(trans_symm_matrix);
  int count = 0;
  // cerr << "normal_original_slave  " << normal_original_slave << "\n";
  // cerr << "trans_symm_matrix 2 " <<  trans_symm_matrix << "\n"; 
  for (int ii = 0; ii != 3; ++ii) {
    
    // cerr << "t_eigen_vectors  " <<  t_eigen_vectors << "\n"; 
    double vec_length = sqrt(t_eigen_vectors(i) * t_eigen_vectors(i));
    double projection = sqrt(abs(t_eigen_vectors(i) * normal_original_slave(i)));
    // cerr <<"eigen  " <<  t_eig_vec_to_sort << "\n";
    // cerr <<"eigen val check   " << t_eigen_vectors <<"\n";
    // cerr << "projection " << projection << "\n";
    if (projection < 1.e-3) {
      ++count;
      // if(t_eigen_vectors(2) > 0)
      t_shift_tensor(i) = t_eigen_vectors(i);
      // else 
      // t_shift_tensor(i) = -t_eigen_vectors(i);
      // cerr <<"eigen  " <<  t_eigen_vectors << "\n";
      // cerr << "normal_original_slave  " << normal_original_slave << "\n";
      // cerr <<"eigen val   " << t_eigen_vals   <<"  shift tensor " << count << "  " << t_shift_tensor << "\n";
      // cerr << "projection " << projection << "\n";
      // cerr <<"eigen val   " << t_eigen_vals <<"\n";
      ++t_shift_tensor;
    }
    // cerr << "vec_length " << vec_length << "  " << ii << " " <<
    // t_eigen_vectors << "\n"; cerr << "projection " << projection << "\n";
     
    ++t_eigen_vectors;
    ++t_eigen_vals;
}
if(count != 2){
  cerr << "Mistake!! " << count <<  "\n";
}


  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactFrictionAugmentedOperatorsRhs(
    boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrange_field_name,
    string friction_lagrange_field_name, string previously_converged_spat) {
  MoFEMFunctionBegin

      fe_rhs_simple_contact->getOpPtrVector()
          .push_back(new OpGetNormalSlaveALE("MESH_NODE_POSITIONS",
                                             common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGetNormalMasterALE(
      "MESH_NODE_POSITIONS", common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsMaster(field_name,
                                        common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsSlave(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetGapSlave(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave3D(friction_lagrange_field_name,
                                       common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrange_field_name,
                                     common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetAugmentedLambdaSlave(field_name, common_data_simple_contact,
                                    cnValue));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPreviousPositionAtGaussPtsSlave(previously_converged_spat,
                                               common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPreviousPositionAtGaussPtsMaster(previously_converged_spat,
                                                common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetTangentGapVelocitySlave(field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetTangentLagrange(friction_lagrange_field_name, common_data_simple_contact));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpGetTangentAugmentedLambdaSlave(
          friction_lagrange_field_name, common_data_simple_contact, cTangentValue));

  //Normal
  // fe_rhs_simple_contact->getOpPtrVector().push_back(
  //     new OpCalAugmentedTractionRhsMaster(field_name,
  //                                         common_data_simple_contact));
  // fe_rhs_simple_contact->getOpPtrVector().push_back(
  //     new OpCalAugmentedTractionRhsSlave(field_name,
  //                                        common_data_simple_contact));

  // fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGapConstraintOnLambda(
  //     lagrange_field_name, common_data_simple_contact, cnValue));

  //Friction
  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpCalAugmentedTangentTractionsRhsMaster(
          field_name, common_data_simple_contact, muTangent));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpCalAugmentedTangentTractionsRhsSlave(
          field_name, common_data_simple_contact, muTangent));

  fe_rhs_simple_contact->getOpPtrVector().push_back(
      new OpCalAugmentedTangentialContCondition(friction_lagrange_field_name,
                                                common_data_simple_contact,
                                                muTangent, cTangentValue));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::setContactFrictionAugmentedOperatorsLhs(
    boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
    string field_name, string lagrange_field_name, 
    string friction_lagrange_field_name,  string previously_converged_spat) {
  MoFEMFunctionBegin;

  fe_lhs_simple_contact->getOpPtrVector().push_back(new
  OpGetNormalSlaveALE(
      "MESH_NODE_POSITIONS", common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(new
  OpGetNormalMasterALE(
      "MESH_NODE_POSITIONS", common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsMaster(field_name,
                                        common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsSlave(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetGapSlave(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave3D(friction_lagrange_field_name,
                                     common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetLagMulAtGaussPtsSlave(lagrange_field_name,
                                     common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetAugmentedLambdaSlave(field_name, common_data_simple_contact,
                                    cnValue));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPreviousPositionAtGaussPtsSlave(previously_converged_spat,
                                               common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetPreviousPositionAtGaussPtsMaster(previously_converged_spat,
                                                common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetTangentGapVelocitySlave(field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetTangentLagrange(friction_lagrange_field_name, common_data_simple_contact));

  fe_lhs_simple_contact->getOpPtrVector().push_back(
      new OpGetTangentAugmentedLambdaSlave(
          friction_lagrange_field_name, common_data_simple_contact, cTangentValue));

    // //Normal x-master
    //   fe_lhs_simple_contact->getOpPtrVector().push_back(
    //       new OpCalContactAugmentedTractionOverSpatialMasterMaster(
    //           field_name, field_name, cnValue, common_data_simple_contact));

    //   fe_lhs_simple_contact->getOpPtrVector().push_back(
    //       new OpCalContactAugmentedTractionOverSpatialMasterSlave(
    //           field_name, field_name, cnValue, common_data_simple_contact));

    //     fe_lhs_simple_contact->getOpPtrVector().push_back(
    //         new OpCalContactAugmentedTractionOverLambdaMasterSlave3D(
    //             field_name, lagrange_field_name, common_data_simple_contact));
    // //Normal x-slave
    //   fe_lhs_simple_contact->getOpPtrVector().push_back(
    //       new OpCalContactAugmentedTractionOverSpatialSlaveMaster(
    //           field_name, field_name, cnValue, common_data_simple_contact));

    //   fe_lhs_simple_contact->getOpPtrVector().push_back(
    //       new OpCalContactAugmentedTractionOverSpatialSlaveSlave(
    //           field_name, field_name, cnValue, common_data_simple_contact));

    //     fe_lhs_simple_contact->getOpPtrVector().push_back(
    //         new OpCalContactAugmentedTractionOverLambdaSlaveSlave3D(
    //             field_name, lagrange_field_name, common_data_simple_contact));

    // //Normal lambda-slave
    //     fe_lhs_simple_contact->getOpPtrVector().push_back(
    //         new OpGapConstraintOverLambda(lagrange_field_name,
    //                                       common_data_simple_contact, cnValue));

    //     fe_lhs_simple_contact->getOpPtrVector().push_back(
    //         new OpGapConstraintOverSpatialMaster(field_name, lagrange_field_name,
    //                                              common_data_simple_contact,
    //                                              cnValue));

    //     fe_lhs_simple_contact->getOpPtrVector().push_back(
    //         new OpGapConstraintOverSpatialSlave(field_name, lagrange_field_name,
    //                                             common_data_simple_contact,
    //                                             cnValue));

//Friction
//
    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpContactAugmentedFrictionMasterMaster(
            field_name, field_name, common_data_simple_contact, muTangent,
            cTangentValue, cnValue));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpContactAugmentedFrictionMasterSlave(
            field_name, field_name, common_data_simple_contact, muTangent,
            cTangentValue, cnValue));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalContactAugmentedTangentLambdaOverLambdaMasterSlave(
            field_name, lagrange_field_name, common_data_simple_contact,
            muTangent));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalContactAugmentedTangentLambdaOverLambdaTanMasterSlave(
            field_name, friction_lagrange_field_name, common_data_simple_contact,
            muTangent));
//
    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpContactAugmentedFrictionSlaveSlave(
            field_name, field_name, common_data_simple_contact, muTangent,
            cTangentValue, cnValue));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpContactAugmentedFrictionSlaveMaster(
            field_name, field_name, common_data_simple_contact, muTangent,
            cTangentValue, cnValue));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalContactAugmentedTangentLambdaOverLambdaSlaveSlave(
            field_name, lagrange_field_name, common_data_simple_contact,
            muTangent));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalContactAugmentedTangentLambdaOverLambdaTanSlaveSlave(
            field_name, friction_lagrange_field_name, common_data_simple_contact,
            muTangent));
//
    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalAugmentedTangentialContConditionDispSlaveSlave(
            friction_lagrange_field_name, field_name, common_data_simple_contact,
            muTangent, cTangentValue, cnValue));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalAugmentedTangentialContConditionDispSlaveMaster(
            friction_lagrange_field_name, field_name, common_data_simple_contact,
            muTangent, cTangentValue, cnValue));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalAugmentedTangentialContConditionLambdaNormSlaveSlave(
            friction_lagrange_field_name, lagrange_field_name, common_data_simple_contact,
            muTangent, cTangentValue));

    fe_lhs_simple_contact->getOpPtrVector().push_back(
        new OpCalAugmentedTangentialContConditionLambdaTanSlaveSlave(
            friction_lagrange_field_name, friction_lagrange_field_name, common_data_simple_contact,
            muTangent, cTangentValue));

  MoFEMFunctionReturn(0);
}
