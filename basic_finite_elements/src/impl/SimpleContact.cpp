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

MoFEMErrorCode
SimpleContactProblem::SimpleContactElement::setGaussPts(int order) {
  MoFEMFunctionBegin;
  int rule = order + 2;
  int nb_gauss_pts = triangle_ncc_order_num(rule);
  SimpleContactProblem::SimpleContactElement::gaussPtsMaster.resize(
      3, nb_gauss_pts, false);
  SimpleContactProblem::SimpleContactElement::gaussPtsSlave.resize(
      3, nb_gauss_pts, false);
  double xy_coords[2 * nb_gauss_pts];
  double w_array[nb_gauss_pts];
  triangle_ncc_rule(rule, nb_gauss_pts, xy_coords, w_array);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    SimpleContactProblem::SimpleContactElement::gaussPtsMaster(0, gg) =
        xy_coords[gg * 2];
    SimpleContactProblem::SimpleContactElement::gaussPtsMaster(1, gg) =
        xy_coords[gg * 2 + 1];
    SimpleContactProblem::SimpleContactElement::gaussPtsMaster(2, gg) =
        w_array[gg];
    SimpleContactProblem::SimpleContactElement::gaussPtsSlave(0, gg) =
        xy_coords[gg * 2];
    SimpleContactProblem::SimpleContactElement::gaussPtsSlave(1, gg) =
        xy_coords[gg * 2 + 1];
    SimpleContactProblem::SimpleContactElement::gaussPtsSlave(2, gg) =
        w_array[gg];
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetNormalSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
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
  const double *tangent_one_slave_ptr = &getTangentSlaveOne()[0];
  const double *tangent_two_slave_ptr = &getTangentSlaveTwo()[0];

  commonDataSimpleContact->normalVectorSlavePtr.get()->resize(3);
  commonDataSimpleContact->normalVectorSlavePtr.get()->clear();

  commonDataSimpleContact->tangentOneVectorSlavePtr.get()->resize(3);
  commonDataSimpleContact->tangentOneVectorSlavePtr.get()->clear();

  commonDataSimpleContact->tangentTwoVectorSlavePtr.get()->resize(3);
  commonDataSimpleContact->tangentTwoVectorSlavePtr.get()->clear();

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);
  for (int ii = 0; ii != 3; ++ii)
    normal(ii) = normal_slave_ptr[ii];

  const double normal_length = sqrt(normal(i) * normal(i));
  normal(i) = normal(i) / normal_length;

  commonDataSimpleContact->areaSlave = 0.5 * normal_length;

  auto t_1 = get_tensor_vec(
      commonDataSimpleContact->tangentOneVectorSlavePtr.get()[0]);
  for (int ii = 0; ii != 3; ++ii)
    t_1(ii) = tangent_one_slave_ptr[ii];

  const double t_1_length = sqrt(t_1(i) * t_1(i));
  t_1(i) = t_1(i) / t_1_length;

  auto t_2 = get_tensor_vec(
      commonDataSimpleContact->tangentTwoVectorSlavePtr.get()[0]);
  for (int ii = 0; ii != 3; ++ii)
    t_2(ii) = tangent_two_slave_ptr[ii];

  const double t_2_length = sqrt(t_2(i) * t_2(i));
  t_2(i) = t_2(i) / t_2_length;

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetNormalMaster::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
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

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorMasterPtr.get()[0]);
  for (int ii = 0; ii != 3; ++ii)
    normal(ii) = normal_master_ptr[ii];

  const double normal_length = sqrt(normal(i) * normal(i));
  normal(i) = normal(i) / normal_length;
  commonDataSimpleContact->areaSlave = 0.5 * normal_length;

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetNormalSlaveForSide::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  FTensor::Index<'i', 3> i;

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  // CHKERR loopSideVolumes(sideFeName, *sideFe, 3);

  const double *normal_slave_ptr = &getNormalSlave()[0];
  commonDataSimpleContact->normalVectorSlavePtr.get()->resize(3);
  commonDataSimpleContact->normalVectorSlavePtr.get()->clear();

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);
  for (int ii = 0; ii != 3; ++ii)
    normal(ii) = normal_slave_ptr[ii];

  const double normal_length = sqrt(normal(i) * normal(i));
  normal(i) = normal(i) / normal_length;
  commonDataSimpleContact->areaSlave = 0.5 * normal_length;

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactConstraintMatrixMasterSlaveForSide::doWork(
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

  // CHKERR loopSideVolumes(sideFeName, *sideFe);

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = col_data.getFieldData().size();

  const double area_slave =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 1, c + 0),
                                         &m(r + 2, c + 0));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;

  NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
  NN.clear();

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    double val_m = getGaussPtsMaster()(2, gg) * area_slave;

    FTensor::Tensor0<double *> t_base_lambda(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
      FTensor::Tensor0<double *> t_base_master(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
        const double m = val_m * t_base_lambda * t_base_master;

        auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, bbc);
        t_assemble_m(i) -= m * const_unit_n(i);

        ++t_base_master; // update rows master
      }
      ++t_base_lambda; // update cols slave
    }
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetPositionAtGaussPtsMaster::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
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

  auto position_master = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsMasterPtr);

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

MoFEMErrorCode SimpleContactProblem::OpCalProjStressesAtGaussPtsMaster::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  
  if (type != MBVERTEX )
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  commonDataSimpleContact->projNormalStressAtMaster.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->projNormalStressAtMaster.get()->clear();

  auto get_tensor2 = [](MatrixDouble3by3 &m) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(0, 0), &m(0, 1), &m(0, 2), &m(1, 0),
        &m(1, 1), &m(1, 2), &m(2, 0), &m(2, 1),
        &m(2, 2));
  };
  
  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };
  
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  auto normal_at_gp_master =
      get_tensor_vec(commonDataSimpleContact->normalVectorMasterPtr.get()[0]);

  auto normal_at_gp_slave =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto proj_normal_stress_master =
      getFTensor0FromVec(*commonDataSimpleContact->projNormalStressAtMaster);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    // cerr << "After1 "
    //      << commonDataSimpleContact->elasticityCommonData.sTress[gg] << "\n";
    // auto t_stress =
    //     get_tensor2(commonDataSimpleContact->elasticityCommonData.sTress[gg]);

    
    MatrixDouble3by3 &stress =
        commonDataSimpleContact->elasticityCommonDataMaster.sTress[gg];
    
    FTensor::Tensor2<double *, 3, 3> t_stress(
        &stress(0, 0), &stress(0, 1), &stress(0, 2), &stress(1, 0),
        &stress(1, 1), &stress(1, 2), &stress(2, 0), &stress(2, 1),
        &stress(2, 2));

    // proj_normal_stress_master +=
    //     t_stress(i, j) * normal_at_gp_master(i) * normal_at_gp_master(j);

    proj_normal_stress_master +=
        t_stress(i, j) * normal_at_gp_master(i) * normal_at_gp_slave(j);

    ++proj_normal_stress_master;
    // ++normal_at_gp_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalRelativeErrorNormalLagrangeMasterAndSlaveDifference::
    doWork(int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  commonDataSimpleContact->relErrorLagNormalStressAtSlave.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->relErrorLagNormalStressAtSlave.get()->clear();

  commonDataSimpleContact->relErrorLagNormalStressAtMaster.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->relErrorLagNormalStressAtMaster.get()->clear();

  auto proj_normal_stress_master =
      getFTensor0FromVec(*commonDataSimpleContact->projNormalStressAtMaster);

  auto proj_normal_stress_slave =
      getFTensor0FromVec(*commonDataSimpleContact->projNormalStressAtSlave);

  auto diff_master =
      getFTensor0FromVec(*commonDataSimpleContact->diffNormalLagMaster);

  auto diff_slave =
      getFTensor0FromVec(*commonDataSimpleContact->diffNormalLagSlave);

auto lag_mult =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

auto error_master = getFTensor0FromVec(
    *commonDataSimpleContact->relErrorLagNormalStressAtMaster);

auto error_slave = getFTensor0FromVec(
    *commonDataSimpleContact->relErrorLagNormalStressAtSlave);

for (int gg = 0; gg != nb_gauss_pts; gg++) {

  error_slave = fabs(diff_slave / proj_normal_stress_slave);
  error_master = fabs(diff_master / proj_normal_stress_master);

  ++proj_normal_stress_master;
  ++proj_normal_stress_slave;
  ++diff_slave;
  ++diff_master;
  ++lag_mult;
  ++error_slave;
  ++error_master;
}

MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpCalNormalLagrangeMasterAndSlaveDifference::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  commonDataSimpleContact->diffNormalLagSlave.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->diffNormalLagSlave.get()->clear();

  commonDataSimpleContact->diffNormalLagMaster.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->diffNormalLagMaster.get()->clear();

  auto proj_normal_stress_master =
      getFTensor0FromVec(*commonDataSimpleContact->projNormalStressAtMaster);

  auto proj_normal_stress_slave =
      getFTensor0FromVec(*commonDataSimpleContact->projNormalStressAtSlave);

  auto diff_master =
      getFTensor0FromVec(*commonDataSimpleContact->diffNormalLagMaster);

  auto diff_slave =
      getFTensor0FromVec(*commonDataSimpleContact->diffNormalLagSlave);

auto lag_mult =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    diff_slave += lag_mult + proj_normal_stress_slave;
    diff_master += lag_mult + proj_normal_stress_master;
    ++proj_normal_stress_master;
    ++proj_normal_stress_slave;
    ++diff_slave;
    ++diff_master;
    ++lag_mult;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalProjStressesAtGaussPtsSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  
  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  commonDataSimpleContact->projNormalStressAtSlave.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->projNormalStressAtSlave.get()->clear();

  auto get_tensor2 = [](MatrixDouble3by3 &m) {
    return FTensor::Tensor2<double *, 3, 3>(&m(0, 0), &m(0, 1), &m(0, 2),
                                            &m(1, 0), &m(1, 1), &m(1, 2),
                                            &m(2, 0), &m(2, 1), &m(2, 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  auto normal_at_gp_slave =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto proj_normal_stress_slave =
      getFTensor0FromVec(*commonDataSimpleContact->projNormalStressAtSlave);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    // auto t_stress =
    //     get_tensor2(commonDataSimpleContact->elasticityCommonData.sTress[gg]);

    MatrixDouble3by3 &stress =
        commonDataSimpleContact->elasticityCommonData.sTress[gg];
    
    FTensor::Tensor2<double *, 3, 3> t_stress(
        &stress(0, 0), &stress(0, 1), &stress(0, 2), &stress(1, 0),
        &stress(1, 1), &stress(1, 2), &stress(2, 0), &stress(2, 1),
        &stress(2, 2));

    proj_normal_stress_slave +=
        t_stress(i, j) * normal_at_gp_slave(i) * normal_at_gp_slave(j);

    // cerr << "gg " << gg << " proj_normal_stress_slave "
    //      << proj_normal_stress_slave << " \n";

    ++proj_normal_stress_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetPositionAtGaussPtsSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
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

  auto position_slave = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsSlavePtr);

  int nb_base_fun_col = data.getFieldData().size() / 3;

  FTensor::Index<'i', 3> i;

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));

    FTensor::Tensor1<double *, 3> t_field_data_slave(
        &data.getFieldData()[0], &data.getFieldData()[1],
        &data.getFieldData()[2], 3); // in-between

    for (int bb = 0; bb != nb_base_fun_col; bb++) {
      position_slave(i) += t_base_slave * t_field_data_slave(i);

      ++t_base_slave;
      ++t_field_data_slave;
    }
    ++position_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetGapSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
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

  auto position_master_gp = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsMasterPtr);
  auto position_slave_gp = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsSlavePtr);

  auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto normal_at_gp =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    gap_ptr -= normal_at_gp(i) * (position_slave_gp(i) - position_master_gp(i));
    ++position_slave_gp;
    ++position_master_gp;
    ++gap_ptr;
  } // for gauss points

  auto gap_ptr_2 = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetGapSlaveALE::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
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

  auto position_master_gp = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsMasterPtr);
  auto position_slave_gp = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsSlavePtr);

  auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);
  auto length_normal =
      getFTensor0FromVec(*commonDataSimpleContact->normalSlaveLengthALEPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    auto normal_at_gp =
        get_tensor_vec(commonDataSimpleContact->normalSlaveALE[gg]);

    gap_ptr -= normal_at_gp(i) *
               (position_slave_gp(i) - position_master_gp(i)) / length_normal;

    ++position_slave_gp;
    ++position_master_gp;
    ++gap_ptr;
    ++length_normal;
  } // for gauss points

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetLagMulAtGaussPtsSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  if (type == MBVERTEX) {
    commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->resize(nb_gauss_pts);
    commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->clear();
  }

  int nb_base_fun_row = data.getFieldData().size();

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    FTensor::Tensor0<double *> t_base_lambda(&data.getN()(gg, 0));

    FTensor::Tensor0<double *> t_field_data_slave(&data.getFieldData()[0]);
    for (int bb = 0; bb != nb_base_fun_row; bb++) {
      lagrange_slave += t_base_lambda * t_field_data_slave;
      ++t_base_lambda;
      ++t_field_data_slave;
    }
    ++lagrange_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetLagMulAtGaussPtsSlaveL2::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;


  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  if (type == MBTRI) {
    commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->resize(nb_gauss_pts);
    commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->clear();
  }

  int nb_base_fun_row = data.getFieldData().size();

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    FTensor::Tensor0<double *> t_base_lambda(&data.getN()(gg, 0));

    FTensor::Tensor0<double *> t_field_data_slave(&data.getFieldData()[0]);
    for (int bb = 0; bb != nb_base_fun_row; bb++) {
      lagrange_slave += t_base_lambda * t_field_data_slave;
      cerr << "CAL lagrange!!!!   base lambda " << t_base_lambda << "  gg " << gg << "\n";
      ++t_base_lambda;
      ++t_field_data_slave;
    }

    ++lagrange_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpPrintLagMulAtGaussPtsSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
  auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);
  cout << "-----------------------------" << endl;

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    cout << "gp: " << gg << " | gap: " << gap_ptr << " | lm: " << lagrange_slave
         << " | gap * lm = " << gap_ptr * lagrange_slave << endl;
    ++lagrange_slave;
    ++gap_ptr;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetLagMulAtGaussPtsSlaveHdiv::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBTRI)
    MoFEMFunctionReturnHot(0);

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();
  commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->clear();

  int nb_base_fun_row = data.getFieldData().size();

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  //auto t_lag_base = data.getFTensor1N<3>();

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  auto normal_at_gp =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    auto t_lag_base = data.getFTensor1N<3>(gg, 0);
    FTensor::Tensor0<double *> t_field_data_slave(&data.getFieldData()[0]);
    for (int bb = 0; bb != nb_base_fun_row; bb++) {
      // cerr << "Lagbase "  << t_lag_base << "    GG " << gg <<" \n";
      lagrange_slave +=
          t_field_data_slave /* normal_at_gp(i) */ * t_lag_base(0);
      ++t_field_data_slave;
      ++t_lag_base;
    }
    // cerr << " Lagrange gp slave tri " << lagrange_slave << "\n ";
    ++lagrange_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetLagMulAtGaussPtsSlaveHdiv3D::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  // if (type != MBTRI)
  //   MoFEMFunctionReturnHot(0);

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();
  // cerr << "GAUSS " << nb_gauss_pts << "\n";

  commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->clear();

commonDataSimpleContact->lagMultAtGaussPtsT1Ptr.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->lagMultAtGaussPtsT1Ptr.get()->clear();

commonDataSimpleContact->lagMultAtGaussPtsT2Ptr.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->lagMultAtGaussPtsT2Ptr.get()->clear();

auto lagrange_slave =
    getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

    auto lagrange_slave_t1 =
    getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsT1Ptr);

    auto lagrange_slave_t2 =
    getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsT2Ptr);

  
// commonDataSimpleContact->lagMultAtGaussPtsPt3D.get()->resize(3, nb_gauss_pts,
//                                                              false);
// commonDataSimpleContact->lagMultAtGaussPtsPt3D.get()->clear();

int nb_base_fun_row = data.getFieldData().size() / 3;

// auto lagrange_slave_3_d =
//     getFTensor1FromMat<3>(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

// auto t_lag_base = data.getFTensor1N<3>();

auto get_tensor_vec = [](VectorDouble &n) {
  return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
};

FTensor::Index<'i', 3> i;
FTensor::Index<'j', 3> j;
auto normal_at_gp =
    get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

    auto tangent_t1_at_gp_slave = get_tensor_vec(
      commonDataSimpleContact->tangentOneVectorSlavePtr.get()[0]);

  auto tangent_t2_at_gp_slave = get_tensor_vec(
      commonDataSimpleContact->tangentTwoVectorSlavePtr.get()[0]);

// auto t_n_diff_hvec = data.getFTensor2DiffN<Tensor_Dim1, Tensor_Dim1>();

const int nb_dofs = data.getFieldData().size();
if (!nb_dofs)
  MoFEMFunctionReturnHot(0);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    auto t_lag_base = data.getFTensor1N<3>(gg, 0);
    // auto t_dof = data.getFTensor1FieldData<1>();//we loop over each dov
    //FTensor::Tensor0<double *> t_field_data_slave(&data.getFieldData()[0]);
    FTensor::Tensor1<double *, 3> t_field_data_slave(
        &data.getFieldData()[0], &data.getFieldData()[1],
        &data.getFieldData()[2], 3);

    for (int bb = 0; bb != nb_dofs/3; ++bb) {
      lagrange_slave -= (t_field_data_slave(0) + t_field_data_slave(1) +
                         t_field_data_slave(2)) *
                        t_lag_base(i) * normal_at_gp(i);

      lagrange_slave_t1 -= (t_field_data_slave(0) + t_field_data_slave(1) +
                         t_field_data_slave(2)) *
                        t_lag_base(i) * tangent_t1_at_gp_slave(i);
      lagrange_slave_t2 -= (t_field_data_slave(0) + t_field_data_slave(1) +
                         t_field_data_slave(2)) *
                        t_lag_base(i) * tangent_t2_at_gp_slave(i);
      // lagrange_slave +=
      //     t_field_data_slave(1) * t_lag_base(0) /* normal_at_gp(i)*/;
      // lagrange_slave +=
      //     t_field_data_slave(2) * t_lag_base(0) /* normal_at_gp(i)*/;
      // cerr << " Data Lag mult " << t_field_data_slave << " type " << type
      //      << " nb_dofs " << nb_dofs << "\n ";
      // cerr << " t_lag_base " << t_lag_base << " normal " << normal_at_gp
      //       << " gg "<< gg << "\n ";
      ++t_field_data_slave;
        ++t_lag_base;
      // ++t_dof;
    }
    // cerr << "  gp slave tri 3D " << lagrange_slave << " TYPE " << type << "\n ";
    ++lagrange_slave;
    ++lagrange_slave_t1;
    ++lagrange_slave_t2;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpConstrainBoundaryRhs::doWork(
    int side, EntityType type,
    DataForcesAndSourcesCore::EntData &data){
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
      t_contact_normal_tensor(i, j) =
          t_contact_normal(i) * t_contact_normal(j);
       FTensor::Tensor2<double, 3, 3> t_contact_tangent_tensor;
       t_contact_tangent_tensor(i, j) = t_contact_normal_tensor(i, j);
       t_contact_tangent_tensor(0, 0) -= 1;
       t_contact_tangent_tensor(1, 1) -= 1;
       t_contact_tangent_tensor(2, 2) -= 1;
       FTensor::Tensor1<double, 3> t_rhs_constrains;
       const double normal_traction = t_traction(i) * t_contact_normal(i);
       t_rhs_constrains(i) =
           (gap_ptr - (gap_ptr - cN * normal_traction +
                       std::abs(gap_ptr - cN * normal_traction)) /
                          2.) * t_contact_normal(i);

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
       t_rhs_tangent_disp(i) =
           t_contact_tangent_tensor(i, j) * (t_x_slave(j) - t_X_slave(j));
       t_rhs_tangent_traction(i) =
            cN * t_contact_tangent_tensor(i, j) * t_traction(j);
       t_rhs_normal_traction(i) =
           cN * t_contact_normal_tensor(i, j) * t_traction(j);
       //  cerr << " t_traction " << t_traction << "\n";
       //  cerr << " t_rhs_tangent_traction " << t_rhs_tangent_traction << "\n";
        // cerr << " t_rhs_tangent_disp " << t_rhs_tangent_disp << "\n";
        // cerr << " normal_traction " << normal_traction << "\n";
       //  cerr << " t_x_slave " << t_x_slave << " t_X_slave " << t_X_slave <<
       //  "\n"  ;

       auto t_field_data = data.getFTensor1FieldData<3>();
       size_t bb = 0;
       for (; bb != nb_dofs / 3; ++bb) {
         const double beta = alpha * (t_base(i) * t_normal(i));
         //  t_nf(i) += beta * t_rhs_constrains(i);
         t_nf(i) += beta * t_rhs_tangent_disp(i);
         t_nf(i) += beta * t_rhs_tangent_traction(i);
         t_nf(i) += alpha * cN *
                    (t_contact_tangent_tensor(i, j) * t_base(k) * t_normal(k)) *
                    t_traction(j);
         //  t_nf(i) += beta * t_rhs_normal_traction(i);

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
       // ++t_coords;
       ++t_w;
       ++gap_ptr;
    }
    CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }
 MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpConstrainBoundaryRhsMaster::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPtsMaster().size2();
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

    const double area_m =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto t_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto t_w = getFTensor0IntegrationWeightMaster();
    // auto t_disp = getFTensor1FromMat<2>(*(commonDataPtr->contactDispPtr));

    auto position_master = getFTensor1FromMat<3>(
        *commonDataSimpleContact->positionAtGaussPtsMasterPtr);
    auto t_x_slave = getFTensor1FromMat<3>(
        *commonDataSimpleContact->positionAtGaussPtsSlavePtr);
    auto t_x_master = getFTensor1FromMat<3>(
        *commonDataSimpleContact->positionAtGaussPtsMasterPtr);

    auto t_x_master_mesh = getFTensor1FromMat<3>(
        *commonDataSimpleContact->meshPositionAtGaussPtsMasterPtr);
    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

    auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    // auto t_coords = getFTensor1CoordsAtGaussPtsSlave();
    size_t nb_base_functions = data.getN().size2() / 3;
    auto t_base = data.getFTensor1N<3>();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf{&nf[0], &nf[1],
                                                              &nf[2]};
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
      FTensor::Tensor1<double, 3> t_rhs_constrains;
      const double normal_traction = t_traction(i) * t_contact_normal(i);
      // t_rhs_constrains(i) =
      //     (gap_ptr - (gap_ptr - cN * normal_traction +
      //                 std::abs(gap_ptr - cN * normal_traction)) /
      //                    2.) *
      //     t_contact_normal(i);

      t_rhs_constrains(i) =
          (-normal_traction - (-normal_traction - cN * gap_ptr +
                               std::abs(-normal_traction - cN * gap_ptr)) /
                                  2.) *
          t_contact_normal(i);

      //  t_rhs_constrains(i) =
      //      t_contact_normal(i) *
      //     constrian(gap0(t_coords, t_contact_normal),
      //                    gap(t_disp, t_contact_normal),
      //                    normal_traction(t_traction, t_contact_normal));
      FTensor::Tensor1<double, 3> t_rhs_tangent_disp, t_rhs_tangent_traction;
      t_rhs_tangent_disp(i) =
          t_contact_tangent_tensor(i, j) * (t_x_slave(j) - t_x_master(j));
      t_rhs_tangent_traction(i) =
          cN * t_contact_tangent_tensor(i, j) * t_traction(j);
      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        const double beta = alpha * (t_base(i) * t_normal(i));
        // t_nf(i) += beta * (t_x_master(i) - t_x_master_mesh(i));
        t_nf(i) += beta * t_rhs_tangent_disp(i);
        t_nf(i) += beta * t_rhs_tangent_traction(i);
        ++t_nf;
        ++t_base;
      }
      for (; bb < nb_base_functions; ++bb)
        ++t_base;

      ++t_x_slave;
      ++t_x_master;
      ++t_x_master_mesh;
      ++t_traction;
      // ++t_coords;
      ++t_w;
      ++gap_ptr;
    }
    CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpConstrainBoundaryLhs_dTraction::doWork(
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

    const double area_m =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto get_tensor_vec = [](VectorDouble &n, const int r) {
      return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
    };

    auto t_normal = get_tensor_vec(
        commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

    auto t_base_normal =
        getFTensor0FromVec(*commonDataSimpleContact->contactBaseNormal);

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();
    // FTensor::Tensor1<double, 2> t_direction{getDirection()[0],
    //                                                 getDirection()[1]};
    // FTensor::Tensor1<double, 2> t_normal{-t_direction(1), t_direction(0)};
    // const double l = sqrt(t_normal(i) * t_normal(i));
    // t_normal(i) /= l;
    // t_direction(i) /= l;
    auto t_x_slave = getFTensor1FromMat<3>(
        *commonDataSimpleContact->positionAtGaussPtsSlavePtr);
    auto t_x_master = getFTensor1FromMat<3>(
        *commonDataSimpleContact->positionAtGaussPtsMasterPtr);
    auto t_traction =
        getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));

    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor1N<3>();
    size_t nb_face_functions = row_data.getN().size2() / 3;
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      const double alpha = t_w * area_m;
      //  auto t_contact_normal = normal(t_coords, t_disp);
       FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
       t_contact_normal_tensor(i, j) =
          t_normal(i) * t_normal(j);
       FTensor::Tensor2<double, 3, 3> t_contact_tangent_tensor;
       t_contact_tangent_tensor(i, j) = t_contact_normal_tensor(i, j);
       t_contact_tangent_tensor(0, 0) -= 1;
       t_contact_tangent_tensor(1, 1) -= 1;
       t_contact_tangent_tensor(2, 2) -= 1;
        // const double diff_traction = diff_constrains_dtraction(
        // gap0(t_coords, t_contact_normal), gap(t_disp, t_contact_normal),
        // normal_traction(t_traction, t_contact_normal));
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
          //  t_mat(i, j) +=
          //      (beta * diff_traction) * t_contact_normal_tensor(i, j);
           t_mat(i, j) += beta * cN * t_contact_tangent_tensor(i, j);
           ++t_col_base;
           ++t_mat;
        }
       ++t_row_base;
      }
      for (; rr < nb_face_functions; ++rr) 
          ++t_row_base;

      ++t_x_master;
      ++t_x_slave;
      ++t_traction;
      ++t_w;
     }
  CHKERR MatSetValues(getSNESB(), row_data, col_data,
                                &*locMat.data().begin(), ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
 }

 MoFEMErrorCode SimpleContactProblem::OpConstrainBoundaryLhs_dxdTraction::doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
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

     auto t_base_normal =
         getFTensor0FromVec(*commonDataSimpleContact->contactBaseNormal);

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

     auto t_w = getFTensor0IntegrationWeightSlave();
     auto t_row_base = row_data.getFTensor1N<3>();
     size_t nb_face_functions = row_data.getN().size2() / 3;
     for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      //  cerr << "Traction " << t_traction << "\n";

       const double alpha = t_w * area_m;
       //  auto t_contact_normal = normal(t_coords, t_disp);
       FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
       t_contact_normal_tensor(i, j) = t_normal(i) * t_normal(j);
       FTensor::Tensor2<double, 3, 3> t_contact_tangent_tensor;
       t_contact_tangent_tensor(i, j) = t_contact_normal_tensor(i, j);
       t_contact_tangent_tensor(0, 0) -= 1;
       t_contact_tangent_tensor(1, 1) -= 1;
       t_contact_tangent_tensor(2, 2) -= 1;
       // const double diff_traction = diff_constrains_dtraction(
       // gap0(t_coords, t_contact_normal), gap(t_disp, t_contact_normal),
       // normal_traction(t_traction, t_contact_normal));
       const double normal_traction = t_traction(i) * t_normal(i);

      //  t_rhs_constrains(i) =
      //      (-normal_traction - (-normal_traction - cN * gap_ptr +
      //                           std::abs(-normal_traction - cN * gap_ptr)) /
      //                              2.) *
      //      t_contact_normal(i);

      //  const double diff_constrain_d_lambda =
      //     cN * (1 + sign(gap_ptr - cN * normal_traction)) / 2.;

       const double diff_constrain_d_lambda =
           (1 + sign(-normal_traction - cN * gap_ptr)) / 2.;
      
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

        //  auto row_base = row_data.getFTensor1N<3>();
        //  auto row_base = col_data.getFTensor0N(gg, 0);
        //  cerr << "diff_constrain_d_lambda "<< diff_constrain_d_lambda << "\n ";
         FTensor::Tensor1<double, 3> t_tan_row_base;
         t_tan_row_base(i) = alpha * t_row_base(j) * t_contact_tangent_tensor(i, j);
         const double row_base = t_row_base(i) * t_normal(i);
         auto t_col_base = col_data.getFTensor1N<3>(gg, 0);
         for (size_t cc = 0; cc != col_nb_dofs / 3; ++cc) {
           const double col_base = t_col_base(i) * t_normal(i);
           const double beta = alpha * row_base * col_base;
           
           //  t_mat(i, j) +=
           //      (beta * diff_constrain_d_lambda) *
           //      t_contact_normal_tensor(i, j);
           t_mat(i, j) += 2.* beta * cN * t_contact_tangent_tensor(i, j);

          //  t_mat(i, j) += beta * cN * t_contact_normal_tensor(i, j);

           //  t_mat(i, j) += beta * t_contact_normal_tensor(i, j);

           //  cerr << "t_mat " << t_mat << "\n";
           //  cerr << "t_contact_normal_tensor " << t_contact_normal_tensor
           //       << "\n";
           //  cerr << "beta " <<  beta <<"\n";
           //  cerr << "t_row_base " << t_row_base << "\n";
           //  cerr << "t_col_base " << t_col_base << "\n";

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
     //  cerr << "locMat  " <<  locMat << "\n";
     CHKERR MatSetValues(getSNESB(), row_data, col_data,
                         &*locMat.data().begin(), ADD_VALUES);
   }
   MoFEMFunctionReturn(0);
 }

 MoFEMErrorCode SimpleContactProblem::OpConstrainBoundaryLhs_dU_Slave::doWork(
     int row_side, int col_side, EntityType row_type, EntityType col_type,
     DataForcesAndSourcesCore::EntData &row_data, DataForcesAndSourcesCore::EntData &col_data) {
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
     for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
       const double alpha = t_w * area_m;
       //  auto t_contact_normal = normal(t_coords, t_disp);
       FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
       t_contact_normal_tensor(i, j) =
           t_contact_normal(i) * t_contact_normal(j);
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
           (1 - sign(-normal_traction - cN * gap_ptr)) * cN / 2.;

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
          //  t_mat(i, j) +=
          //      (beta * diff_constrain_d_slave) * t_contact_normal_tensor(i, j);
          
           t_mat(i, j) += beta * t_contact_tangent_tensor(i, j);
          
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
     CHKERR MatSetValues(getSNESB(), row_data, col_data,
                         &*locMat.data().begin(), ADD_VALUES);
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
      //here
      

      size_t nb_face_functions = col_data.getN().size2() / 3;
      locMat.resize(row_nb_dofs, col_nb_dofs, false);
      locMat.clear();
      
      for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
        const double alpha = t_w * area_m;
        auto t_row_base = row_data.getFTensor0N(gg, 0);
        //  auto t_contact_normal = normal(t_coords, t_disp);
        FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
        t_contact_normal_tensor(i, j) =
            t_contact_normal(i) * t_contact_normal(j);
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
            t_mat(i, j) -=
                beta  * t_contact_normal_tensor(i, j);

            // cerr << "WTF  ~~~~~   " << t_mat << "\n";

            ++t_col_base;
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
      CHKERR MatSetValues(getSNESB(), row_data, col_data,
                          &*locMat.data().begin(), ADD_VALUES);
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

      for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
        const double alpha = t_w * area_m;
        auto t_row_base = row_data.getFTensor0N(gg, 0);
        //  auto t_contact_normal = normal(t_coords, t_disp);
        FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
        t_contact_normal_tensor(i, j) =
            t_contact_normal(i) * t_contact_normal(j);
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
            t_mat(i, j) += beta * t_contact_normal_tensor(i, j);

            // t_mat(i, j) += beta * cN * t_contact_tangent_tensor(i, j);
            // cerr << "WTF  ~~~~~   " << t_mat << "\n";

            ++t_col_base;
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
      CHKERR MatSetValues(getSNESB(), row_data, col_data,
                          &*locMat.data().begin(), ADD_VALUES);
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
  
      for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
        const double alpha = t_w * area_m;
        //  auto t_contact_normal = normal(t_coords, t_disp);
        FTensor::Tensor2<double, 3, 3> t_contact_normal_tensor;
        t_contact_normal_tensor(i, j) =
            t_contact_normal(i) * t_contact_normal(j);
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
        
        // const double diff_constrain_d_master =
        //     (1 - sign(gap_ptr - cN * normal_traction)) / 2.;

        const double diff_constrain_d_master =
            (-1 + sign(-normal_traction - cN * gap_ptr)) * cN / 2.;

        // const double diff_constrain_d_master =
        //     (1 + sign(normal_traction - cN * gap_ptr)) * cN / 2.;

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
            t_mat(i, j) +=
                (beta * diff_constrain_d_master) * t_contact_normal_tensor(i, j);
            
            // t_mat(i, j) -= beta * t_contact_tangent_tensor(i, j);
            

            // t_mat(i, j) += beta  *
            //                t_contact_normal_tensor(i, j);

            // t_mat(0, 0) += beta;
            // t_mat(1, 1) += beta;
            // t_mat(2, 2) += beta;

            // t_mat_trans(i, j) += beta * t_contact_normal_tensor(i, j);
            
            // cerr << "beta" << beta <<"\n";
            ++t_col_base;
            ++t_mat;
            ++t_mat_trans;
          }
          ++t_row_base;
        }
        for (; rr < nb_face_functions; ++rr)
          ++t_row_base;

        ++t_traction;
        ++t_w;
        ++gap_ptr;
      }
      CHKERR MatSetValues(getSNESB(), row_data, col_data,
                          &*locMat.data().begin(), ADD_VALUES);

      MatrixDouble help_matrix;
      help_matrix.resize(col_nb_dofs, row_nb_dofs, false);
      help_matrix.clear();
      noalias(help_matrix) = trans(transLocMat);
      CHKERR MatSetValues(getSNESB(), col_data, row_data,
                          &*help_matrix.data().begin(), ADD_VALUES);
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode SimpleContactProblem::OpConstrainBoundaryTraction::doWork(
      int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;
    const size_t nb_gauss_pts = getGaussPtsSlave().size2();
    // if (side == 0 && type == MBTRI) {
      commonDataSimpleContact->contactTractionPtr->resize(3, nb_gauss_pts);
      commonDataSimpleContact->contactTractionPtr->clear();
      commonDataSimpleContact->contactBaseNormal->resize(nb_gauss_pts);
      commonDataSimpleContact->contactBaseNormal->clear();
    // }
    const size_t nb_dofs = data.getIndices().size();
    if (nb_dofs) {
      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;

      auto get_tensor_vec = [](VectorDouble &n, const int r) {
        return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
      };

      auto t_normal = get_tensor_vec(
          commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

      // auto t_base_normal =
      //     getFTensor0FromVec(*commonDataSimpleContact->contactBaseNormal);

      auto t_traction =
          getFTensor1FromMat<3>(*(commonDataSimpleContact->contactTractionPtr));
      size_t nb_base_functions = data.getN().size2() / 3;
      auto t_base = data.getFTensor1N<3>();
      for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
        auto t_field_data = data.getFTensor1FieldData<3>();
        size_t bb = 0;
        for (; bb != nb_dofs / 3; ++bb) {
          t_traction(j) += (t_base(i) * t_normal(i)) * t_field_data(j);
          
          // t_base_normal += t_base(i) * t_normal(i);
          ++t_field_data;
          ++t_base;
        }
        for (; bb < nb_base_functions; ++bb)
          ++t_base;

        // cerr << "t_traction " << t_traction << "\n";
        ++t_traction;
        // ++t_base_normal;
      }
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
    auto t_base = data.getFTensor0N();
    auto t_diff_base = data.getFTensor1DiffN<3>();
    auto t_lambdas =
        getFTensor2FromMat<3, 3>(*(commonDataSimpleContact->lagMatMultAtGaussPtsPtr));
    auto t_div_stress = getFTensor1FromMat<3>(
        *(commonDataSimpleContact->contactLagrangeHdivDivergencePtr));
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      const double alpha = getMeasure() * t_w;
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf{&nf[0], &nf[1],
                                                              &nf[2]};
      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        t_nf(i) += alpha * t_base * t_div_stress(i);
        t_nf(i) += alpha * t_diff_base(j) * t_lambdas(i, j);
        ++t_nf;
        ++t_base;
        ++t_diff_base;
        // cerr << "t_div_stress" << t_div_stress << "\n";
        // cerr << "alpha  " << alpha << "\n";
      }
      for (; bb < nb_base_functions; ++bb) {
        ++t_base;
        ++t_diff_base;
      }
      ++t_div_stress;
      ++t_lambdas;
      ++t_w;
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

      const double alpha = area_m * t_w;
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf{&nf[0], &nf[1],
                                                              &nf[2]};
      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        t_nf(i) -= alpha * t_base * t_normal(i) * t_traction(j) * t_normal(j);
        ++t_nf;
        ++t_base;

      }
      for (; bb < nb_base_functions; ++bb) {
        ++t_base;
      }
      ++t_traction;
      ++t_w;
    }
    CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }
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

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      // cerr << " traction " << t_traction(j) * t_normal(j) << "\n";

      const double alpha = area_m * t_w;
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf{&nf[0], &nf[1],
                                                              &nf[2]};
      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        t_nf(i) += alpha * t_base * t_normal(i) * t_traction(j) * t_normal(j);
        // t_nf(i) += alpha * t_base * cN * t_contact_tangent_tensor(i,j) * t_traction(j);
        ++t_nf;
        ++t_base;
      }
      for (; bb < nb_base_functions; ++bb) {
        ++t_base;
      }
      ++t_traction;
      ++t_w;
    }
    CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
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
    auto t_base = data.getFTensor1N<3>();
    auto t_diff_base = data.getFTensor2DiffN<3, 3>();
    auto t_lambdas = getFTensor2FromMat<3, 3>(
        *(commonDataSimpleContact->lagMatMultAtGaussPtsPtr));
    auto t_x = getFTensor1FromMat<3>(*(commonDataSimpleContact->xAtPts));
    auto t_X = getFTensor1FromMat<3>(*(commonDataSimpleContact->xInitAtPts));
    auto t_grad =
        getFTensor2FromMat<3, 3>(*(commonDataSimpleContact->mGradPtr));
      for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
        const double alpha = getMeasure() * t_w;
        FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf{&nf[0], &nf[1],
                                                                &nf[2]};
        t_grad(0,0) -= 1;
        t_grad(1,1) -= 1;
        t_grad(2,2) -= 1;

        size_t bb = 0;
        for (; bb != nb_dofs / 3; ++bb) {
          const double t_div_base =
              t_diff_base(0, 0) + t_diff_base(1, 1) + t_diff_base(2, 2);
          t_nf(i) += alpha * (t_base(j) * t_grad(i, j));
          t_nf(i) += alpha * t_div_base * (t_x(i) - t_X(i));
          ++t_nf;
          ++t_base;
          ++t_diff_base;
        }
         for (; bb < nb_base_functions; ++bb) {
         ++t_base;
         ++t_diff_base;
        }
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

MoFEMErrorCode SimpleContactProblem::OpLagGapProdGaussPtsSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  commonDataSimpleContact->lagGapProdPtr.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->lagGapProdPtr.get()->clear();

  int nb_base_fun_row = data.getFieldData().size();

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto lag_gap_prod_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagGapProdPtr);

  auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    lag_gap_prod_slave += gap_ptr * lagrange_slave;
    ++gap_ptr;
    ++lag_gap_prod_slave;
    ++lagrange_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalFReConMaster::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();
  int nb_base_fun_col = data.getFieldData().size() / 3;

  vec_f.resize(3 * nb_base_fun_col,
               false); // the last false in ublas
                       // resize will destroy (not
                       // preserved) the old
                       // values
  vec_f.clear();

  const double area_m =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto get_tensor_vec = [](VectorDouble &n, const int r) {
    return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
  };

  FTensor::Index<'i', 3> i;

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    double val_m = getGaussPtsMaster()(2, gg) * area_m;

    FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
      const double m = val_m * t_base_master * lagrange_slave;
      auto t_assemble_m = get_tensor_vec(vec_f, 3 * bbc);
      t_assemble_m(i) -= m * const_unit_n(i);

      ++t_base_master;
    }

    ++lagrange_slave;
  } // for gauss points

  Vec f;
  if (F == PETSC_NULL) {
    f = getFEMethod()->snes_f;
  } else {
    f = F;
  }

  const int nb_col = data.getIndices().size();

  CHKERR VecSetValues(f, nb_col, &data.getIndices()[0], &vec_f[0], ADD_VALUES);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalPenaltyRhsMaster::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();
  int nb_base_fun_col = data.getFieldData().size() / 3;

  vec_f.resize(3 * nb_base_fun_col,
               false); // the last false in ublas
                       // resize will destroy (not
                       // preserved) the old
                       // values
  vec_f.clear();

  const double area_m =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto get_tensor_vec = [](VectorDouble &n, const int r) {
    return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
  };

  FTensor::Index<'i', 3> i;

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

  auto t_w = getFTensor0IntegrationWeightMaster();
  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    if (gap_gp > 0) {
      ++gap_gp;
      ++t_w;
      continue;
    }

    const double val_m = t_w * area_m * cN;

    FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
      
      auto t_assemble_m = get_tensor_vec(vec_f, 3 * bbc);
      
      t_assemble_m(i) += val_m * const_unit_n(i) * gap_gp * t_base_master;

      ++t_base_master;
    }
    ++gap_gp;
    ++t_w;
  } // for gauss points

  Vec f;
  if (F == PETSC_NULL) {
    f = getFEMethod()->snes_f;
  } else {
    f = F;
  }

  const int nb_col = data.getIndices().size();

  CHKERR VecSetValues(f, nb_col, &data.getIndices()[0], &vec_f[0], ADD_VALUES);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalNitscheCStressRhsMaster::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();
  int nb_base_fun_col = data.getFieldData().size() / 3;

  vec_f.resize(3 * nb_base_fun_col,
               false); // the last false in ublas
                       // resize will destroy (not
                       // preserved) the old
                       // values
  vec_f.clear();

  const double area_m =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto get_tensor_vec = [](VectorDouble &n, const int r) {
    return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorMasterPtr.get()[0], 0);

  auto const_unit_n_slave = get_tensor_vec(
      commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

  auto t_w = getFTensor0IntegrationWeightMaster();
  auto proj_normal_stress_master =
      getFTensor0FromVec(*commonDataSimpleContact->projNormalStressAtMaster);
  auto proj_normal_stress_slave =
      getFTensor0FromVec(*commonDataSimpleContact->projNormalStressAtSlave);

  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto tilde_c_nitsche_fun =
      getFTensor0FromVec(*commonDataSimpleContact->tildeCFunNitschePtr);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    // MatrixDouble3by3 &stress =
    //     commonDataSimpleContact->elasticityCommonDataMaster.sTress[gg];

    // FTensor::Tensor2<double *, 3, 3> t_stress(
    //     &stress(0, 0), &stress(0, 1), &stress(0, 2), &stress(1, 0),
    //     &stress(1, 1), &stress(1, 2), &stress(2, 0), &stress(2, 1),
    //     &stress(2, 2));

    // if (gap_gp > 0) {
    //   ++gap_gp;
    // ++t_w;
    // ++proj_normal_stress_master;
    // ++proj_normal_stress_slave;
    //   continue;
    // }
    // ++gap_gp;

    // cerr << "Stresses "
    //      << 0.5 * (proj_normal_stress_master + proj_normal_stress_slave)
    //      << "\n";
    // Here for average stress
    const double val_m =
        t_w * area_m ;

    FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

      auto t_assemble_m = get_tensor_vec(vec_f, 3 * bbc);

      // t_assemble_m(i) -=
      //     val_m *
      //     (omegaVal * proj_normal_stress_slave * const_unit_n_slave(i) -
      //      (1. - omegaVal) * t_stress(i,j) * const_unit_n(j)) *
      //     t_base_master;

      t_assemble_m(i) +=
          val_m *
          (omegaVal * proj_normal_stress_slave - (1. - omegaVal) *
           proj_normal_stress_master ) *
          const_unit_n_slave(i) * t_base_master;

      t_assemble_m(i) -=
          val_m * tilde_c_nitsche_fun * const_unit_n_slave(i) * t_base_master;

      ++t_base_master;
    }
    ++t_w;
    ++proj_normal_stress_master;
    ++proj_normal_stress_slave;
    ++tilde_c_nitsche_fun;
  } // for gauss points

  Vec f;
  if (F == PETSC_NULL) {
    f = getFEMethod()->snes_f;
    } else {
      f = F;
    }

    const int nb_col = data.getIndices().size();

    CHKERR VecSetValues(f, nb_col, &data.getIndices()[0], &vec_f[0],
                        ADD_VALUES);
    MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalPenaltyRhsSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();
  int nb_base_fun_col = data.getFieldData().size() / 3;

  vec_f.resize(3 * nb_base_fun_col,
               false); // the last false in ublas
                       // resize will destroy (not
                       // preserved) the old
                       // values
  vec_f.clear();

  const double area_s =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto get_tensor_vec = [](VectorDouble &n, const int r) {
    return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
  };

  FTensor::Index<'i', 3> i;

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

  auto t_w = getFTensor0IntegrationWeightSlave();
  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
  if (gap_gp > 0) {
      ++gap_gp;
      ++t_w;
      continue;
    }
    const double val_s = t_w * area_s * cN;

    FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

      auto t_assemble_s = get_tensor_vec(vec_f, 3 * bbc);

      t_assemble_s(i) -= val_s * const_unit_n(i) * gap_gp * t_base_slave;

      ++t_base_slave;
    }
    ++gap_gp;
    ++t_w;
  } // for gauss points

  Vec f;
  if (F == PETSC_NULL) {
    f = getFEMethod()->snes_f;
  } else {
    f = F;
  }

  const int nb_col = data.getIndices().size();

  CHKERR VecSetValues(f, nb_col, &data.getIndices()[0], &vec_f[0], ADD_VALUES);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalNitscheCStressRhsSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();
  int nb_base_fun_col = data.getFieldData().size() / 3;

  vec_f.resize(3 * nb_base_fun_col,
               false); // the last false in ublas
                       // resize will destroy (not
                       // preserved) the old
                       // values
  vec_f.clear();

  const double area_s =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto get_tensor_vec = [](VectorDouble &n, const int r) {
    return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

  auto const_unit_master =
      get_tensor_vec(commonDataSimpleContact->normalVectorMasterPtr.get()[0], 0);

  auto t_w = getFTensor0IntegrationWeightSlave();

  auto proj_normal_stress_master =
      getFTensor0FromVec(*commonDataSimpleContact->projNormalStressAtMaster);
  auto proj_normal_stress_slave =
      getFTensor0FromVec(*commonDataSimpleContact->projNormalStressAtSlave);

  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto tilde_c_nitsche_fun =
      getFTensor0FromVec(*commonDataSimpleContact->tildeCFunNitschePtr);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    // if (gap_gp > 0) {
    //   ++gap_gp;
    //   ++t_w;
    //   ++proj_normal_stress_master;
    //   ++proj_normal_stress_slave;
    //   continue;
    // }
    // ++gap_gp;

    // cerr << "Stresses Master "
    //      << proj_normal_stress_master << " Stresses Slave "
    //      <<proj_normal_stress_slave <<"\n";
    // 0.5 * (sigma_master + sigma_slave)
    const double val_s = t_w * area_s ;

    MatrixDouble3by3 &stress =
        commonDataSimpleContact->elasticityCommonDataMaster.sTress[gg];

    FTensor::Tensor2<double *, 3, 3> t_stress(
        &stress(0, 0), &stress(0, 1), &stress(0, 2), &stress(1, 0),
        &stress(1, 1), &stress(1, 2), &stress(2, 0), &stress(2, 1),
        &stress(2, 2));

    // MatrixDouble3by3 &stress =
    //     commonDataSimpleContact->elasticityCommonData.sTress[gg];

    // FTensor::Tensor2<double *, 3, 3> t_stress(
    //     &stress(0, 0), &stress(0, 1), &stress(0, 2), &stress(1, 0),
    //     &stress(1, 1), &stress(1, 2), &stress(2, 0), &stress(2, 1),
    //     &stress(2, 2));

    FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

      auto t_assemble_s = get_tensor_vec(vec_f, 3 * bbc);

      // t_assemble_s(i) -= t_w * area_s * omegaVal  * t_stress(i, j) *
      //                    const_unit_n(j) * t_base_slave;

      // t_assemble_s(i) +=
      //     val_s *
      //     (omegaVal * proj_normal_stress_slave * const_unit_n(i) -
      //      (1 - omegaVal) * t_stress(k, j) * const_unit_master(j) *
      //          const_unit_master(k) *const_unit_master(i)) *
      //     t_base_slave;


  //15/03/2020
      t_assemble_s(i) -=
          val_s *
          (omegaVal * proj_normal_stress_slave -
           (1. - omegaVal) * proj_normal_stress_master ) * const_unit_n(i) *
          t_base_slave;

      t_assemble_s(i) +=
          val_s * tilde_c_nitsche_fun * const_unit_n(i) * t_base_slave;

      ++t_base_slave;
    }

    ++t_w;
    ++proj_normal_stress_master;
    ++proj_normal_stress_slave;
    ++tilde_c_nitsche_fun;
  } // for gauss points

  Vec f;
  if (F == PETSC_NULL) {
    f = getFEMethod()->snes_f;
    } else {
      f = F;
    }

    const int nb_col = data.getIndices().size();

    CHKERR VecSetValues(f, nb_col, &data.getIndices()[0], &vec_f[0],
                        ADD_VALUES);
    MoFEMFunctionReturn(0);
}

template <int S>
static MoFEMErrorCode get_jac_contact(DataForcesAndSourcesCore::EntData &col_data,
                              int gg, MatrixDouble &jac_stress,
                              MatrixDouble &jac) {
  jac.clear();
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  int nb_col = col_data.getFieldData().size();
  double *diff_ptr =
      const_cast<double *>(&(col_data.getDiffN(gg, nb_col / 3)(0, 0)));
      //cerr << "Lets Check\n";
  // First two indices 'i','j' derivatives of 1st Piola-stress, third index 'k'
  // is displacement component
  FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1_0(
      &jac_stress(3 * 0 + 0, S + 0), &jac_stress(3 * 0 + 0, S + 1),
      &jac_stress(3 * 0 + 0, S + 2), &jac_stress(3 * 0 + 1, S + 0),
      &jac_stress(3 * 0 + 1, S + 1), &jac_stress(3 * 0 + 1, S + 2),
      &jac_stress(3 * 0 + 2, S + 0), &jac_stress(3 * 0 + 2, S + 1),
      &jac_stress(3 * 0 + 2, S + 2), &jac_stress(3 * 1 + 0, S + 0),
      &jac_stress(3 * 1 + 0, S + 1), &jac_stress(3 * 1 + 0, S + 2),
      &jac_stress(3 * 1 + 1, S + 0), &jac_stress(3 * 1 + 1, S + 1),
      &jac_stress(3 * 1 + 1, S + 2), &jac_stress(3 * 1 + 2, S + 0),
      &jac_stress(3 * 1 + 2, S + 1), &jac_stress(3 * 1 + 2, S + 2),
      &jac_stress(3 * 2 + 0, S + 0), &jac_stress(3 * 2 + 0, S + 1),
      &jac_stress(3 * 2 + 0, S + 2), &jac_stress(3 * 2 + 1, S + 0),
      &jac_stress(3 * 2 + 1, S + 1), &jac_stress(3 * 2 + 1, S + 2),
      &jac_stress(3 * 2 + 2, S + 0), &jac_stress(3 * 2 + 2, S + 1),
      &jac_stress(3 * 2 + 2, S + 2));
  FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1_1(
      &jac_stress(3 * 0 + 0, S + 3), &jac_stress(3 * 0 + 0, S + 4),
      &jac_stress(3 * 0 + 0, S + 5), &jac_stress(3 * 0 + 1, S + 3),
      &jac_stress(3 * 0 + 1, S + 4), &jac_stress(3 * 0 + 1, S + 5),
      &jac_stress(3 * 0 + 2, S + 3), &jac_stress(3 * 0 + 2, S + 4),
      &jac_stress(3 * 0 + 2, S + 5), &jac_stress(3 * 1 + 0, S + 3),
      &jac_stress(3 * 1 + 0, S + 4), &jac_stress(3 * 1 + 0, S + 5),
      &jac_stress(3 * 1 + 1, S + 3), &jac_stress(3 * 1 + 1, S + 4),
      &jac_stress(3 * 1 + 1, S + 5), &jac_stress(3 * 1 + 2, S + 3),
      &jac_stress(3 * 1 + 2, S + 4), &jac_stress(3 * 1 + 2, S + 5),
      &jac_stress(3 * 2 + 0, S + 3), &jac_stress(3 * 2 + 0, S + 4),
      &jac_stress(3 * 2 + 0, S + 5), &jac_stress(3 * 2 + 1, S + 3),
      &jac_stress(3 * 2 + 1, S + 4), &jac_stress(3 * 2 + 1, S + 5),
      &jac_stress(3 * 2 + 2, S + 3), &jac_stress(3 * 2 + 2, S + 4),
      &jac_stress(3 * 2 + 2, S + 5));
  FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1_2(
      &jac_stress(3 * 0 + 0, S + 6), &jac_stress(3 * 0 + 0, S + 7),
      &jac_stress(3 * 0 + 0, S + 8), &jac_stress(3 * 0 + 1, S + 6),
      &jac_stress(3 * 0 + 1, S + 7), &jac_stress(3 * 0 + 1, S + 8),
      &jac_stress(3 * 0 + 2, S + 6), &jac_stress(3 * 0 + 2, S + 7),
      &jac_stress(3 * 0 + 2, S + 8), &jac_stress(3 * 1 + 0, S + 6),
      &jac_stress(3 * 1 + 0, S + 7), &jac_stress(3 * 1 + 0, S + 8),
      &jac_stress(3 * 1 + 1, S + 6), &jac_stress(3 * 1 + 1, S + 7),
      &jac_stress(3 * 1 + 1, S + 8), &jac_stress(3 * 1 + 2, S + 6),
      &jac_stress(3 * 1 + 2, S + 7), &jac_stress(3 * 1 + 2, S + 8),
      &jac_stress(3 * 2 + 0, S + 6), &jac_stress(3 * 2 + 0, S + 7),
      &jac_stress(3 * 2 + 0, S + 8), &jac_stress(3 * 2 + 1, S + 6),
      &jac_stress(3 * 2 + 1, S + 7), &jac_stress(3 * 2 + 1, S + 8),
      &jac_stress(3 * 2 + 2, S + 6), &jac_stress(3 * 2 + 2, S + 7),
      &jac_stress(3 * 2 + 2, S + 8));
  // Derivate of 1st Piola-stress multiplied by gradient of defamation for
  // base function (dd) and displacement component (rr)
  FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3> t2_1_0(
      &jac(0, 0), &jac(1, 0), &jac(2, 0), &jac(3, 0), &jac(4, 0), &jac(5, 0),
      &jac(6, 0), &jac(7, 0), &jac(8, 0));
  FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3> t2_1_1(
      &jac(0, 1), &jac(1, 1), &jac(2, 1), &jac(3, 1), &jac(4, 1), &jac(5, 1),
      &jac(6, 1), &jac(7, 1), &jac(8, 1));
  FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3> t2_1_2(
      &jac(0, 2), &jac(1, 2), &jac(2, 2), &jac(3, 2), &jac(4, 2), &jac(5, 2),
      &jac(6, 2), &jac(7, 2), &jac(8, 2));
  FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> diff(
      diff_ptr, &diff_ptr[1], &diff_ptr[2]);
  for (int dd = 0; dd != nb_col / 3; ++dd) {
    t2_1_0(i, j) += t3_1_0(i, j, k) * diff(k);
    t2_1_1(i, j) += t3_1_1(i, j, k) * diff(k);
    t2_1_2(i, j) += t3_1_2(i, j, k) * diff(k);
    ++t2_1_0;
    ++t2_1_1;
    ++t2_1_2;
    ++diff;
  }
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode SimpleContactProblem::OpStressDerivativeGapMaster_dx::getJac(
    DataForcesAndSourcesCore::EntData &col_data, int gg) {
  return get_jac_contact<0>(col_data, gg,
                                           commonData.jacStress[gg], jac);
}

MoFEMErrorCode SimpleContactProblem::OpStressDerivativeGapMaster_dx::doWork(
    int side, EntityType type,
    DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  int nb_row = data.getIndices().size();
//cerr << "RHS  " << nb_row << "\n";
  if (nb_row == 0)
    MoFEMFunctionReturnHot(0);

  // if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
  //     dAta.tEts.end()) {
  //   MoFEMFunctionReturnHot(0);
  // }

// cerr << "Test Passed  " << nb_row << "\n";
  
  // const int nb_base_functions = row_data.getN().size2();
  const int nb_gauss_pts = data.getN().size1();
  int nb_base_fun_row = data.getFieldData().size() / 3;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'m', 3> m;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;

  vec_f.resize(3 * nb_base_fun_row,
               false); // the last false in ublas
                       // resize will destroy (not
                       // preserved) the old
                       // values
  vec_f.clear();

  const double area_m =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto get_tensor_vec = [](VectorDouble &n, const int r) {
    return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
  };

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorMasterPtr.get()[0], 0);

  auto const_unit_slave = get_tensor_vec(
      commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

  auto t_w = getFTensor0IntegrationWeight();

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  // k.resize(nb_row, nb_row, false);
  // k.clear();
  jac.resize(9, nb_row, false);
  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto tilde_c_nitsche_fun =
      getFTensor0FromVec(*commonDataSimpleContact->tildeCFunNitschePtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    MatrixDouble3by3 &stress =
        commonDataSimpleContact->elasticityCommonData.sTress[gg];

    // cerr << "RHS  " << commonData.jacStress[gg] << "\n";
    // if (gap_gp > 0) {
    //   ++gap_gp;
    //   ++t_w;
    //   continue;
    // }
  
    CHKERR getJac(data, gg);
    // double val = getVolume() * getGaussPts()(3, gg);
    // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
    //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
    // }
    // commonData.jacStress[gg]

    FTensor::Tensor2<double *, 3, 3> t_stress(
        &stress(0, 0), &stress(0, 1), &stress(0, 2), &stress(1, 0),
        &stress(1, 1), &stress(1, 2), &stress(2, 0), &stress(2, 1),
        &stress(2, 2));

    FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1(
        &jac(3 * 0 + 0, 0), &jac(3 * 0 + 0, 1), &jac(3 * 0 + 0, 2),
        &jac(3 * 0 + 1, 0), &jac(3 * 0 + 1, 1), &jac(3 * 0 + 1, 2),
        &jac(3 * 0 + 2, 0), &jac(3 * 0 + 2, 1), &jac(3 * 0 + 2, 2),
        &jac(3 * 1 + 0, 0), &jac(3 * 1 + 0, 1), &jac(3 * 1 + 0, 2),
        &jac(3 * 1 + 1, 0), &jac(3 * 1 + 1, 1), &jac(3 * 1 + 1, 2),
        &jac(3 * 1 + 2, 0), &jac(3 * 1 + 2, 1), &jac(3 * 1 + 2, 2),
        &jac(3 * 2 + 0, 0), &jac(3 * 2 + 0, 1), &jac(3 * 2 + 0, 2),
        &jac(3 * 2 + 1, 0), &jac(3 * 2 + 1, 1), &jac(3 * 2 + 1, 2),
        &jac(3 * 2 + 2, 0), &jac(3 * 2 + 2, 1), &jac(3 * 2 + 2, 2));

    // FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1(
    //     &commonData.jacStress[gg](3 * 0 + 0, 0), &commonData.jacStress[gg](3 * 0 + 0, 1), &commonData.jacStress[gg](3 * 0 + 0, 2),
    //     &commonData.jacStress[gg](3 * 0 + 1, 0), &commonData.jacStress[gg](3 * 0 + 1, 1), &commonData.jacStress[gg](3 * 0 + 1, 2),
    //     &commonData.jacStress[gg](3 * 0 + 2, 0), &commonData.jacStress[gg](3 * 0 + 2, 1), &commonData.jacStress[gg](3 * 0 + 2, 2),
    //     &commonData.jacStress[gg](3 * 1 + 0, 0), &commonData.jacStress[gg](3 * 1 + 0, 1), &commonData.jacStress[gg](3 * 1 + 0, 2),
    //     &commonData.jacStress[gg](3 * 1 + 1, 0), &commonData.jacStress[gg](3 * 1 + 1, 1), &commonData.jacStress[gg](3 * 1 + 1, 2),
    //     &commonData.jacStress[gg](3 * 1 + 2, 0), &commonData.jacStress[gg](3 * 1 + 2, 1), &commonData.jacStress[gg](3 * 1 + 2, 2),
    //     &commonData.jacStress[gg](3 * 2 + 0, 0), &commonData.jacStress[gg](3 * 2 + 0, 1), &commonData.jacStress[gg](3 * 2 + 0, 2),
    //     &commonData.jacStress[gg](3 * 2 + 1, 0), &commonData.jacStress[gg](3 * 2 + 1, 1), &commonData.jacStress[gg](3 * 2 + 1, 2),
    //     &commonData.jacStress[gg](3 * 2 + 2, 0), &commonData.jacStress[gg](3 * 2 + 2, 1), &commonData.jacStress[gg](3 * 2 + 2, 2));

    // for (int cc = 0; cc != nb_col / 3; cc++) {
    //   FTensor::Tensor1<double *, 3> diff_base_functions =
    //       row_data.getFTensor1DiffN<3>(gg, 0);
    //   FTensor::Tensor2<double *, 3, 3> lhs(
    //       &k(0, 3 * cc + 0), &k(0, 3 * cc + 1), &k(0, 3 * cc + 2),
    //       &k(1, 3 * cc + 0), &k(1, 3 * cc + 1), &k(1, 3 * cc + 2),
    //       &k(2, 3 * cc + 0), &k(2, 3 * cc + 1), &k(2, 3 * cc + 2), 3 * nb_col);
    //   for (int rr = 0; rr != nb_row / 3; rr++) {
    //     lhs(i, j) += val * t3_1(i, m, j) * diff_base_functions(m);
    //     ++diff_base_functions;
    //     ++lhs;
    //   }
    //   ++t3_1;
    // }
    
    //half for the average stress story
    const double val_m = thetaSVal * tilde_c_nitsche_fun * (1 - omegaVal) *  t_w * area_m / cN;
    const double val_m_2 = thetaSVal * omegaVal * t_w * area_m / cN;
    // cerr << "\n \n"
    //      << "start " << jac << "\n";
    FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_row; ++bbc) {

      auto t_assemble_m = get_tensor_vec(vec_f, 3 * bbc);

      // t_assemble_m(j) -=
      //     val_m_2 * t_stress(j, m) * const_unit_n(m) * t_base_master;


      t_assemble_m(j) += val_m * t3_1(i, m, j) * const_unit_n(i) *
                         const_unit_slave(m); // base not needed

      MatrixDouble t_tan;
      t_tan.resize(3, 3, false);

      auto t_assemble_tan = get_tensor_from_mat(t_tan, 0, 0);
      t_assemble_tan(i, j) = const_unit_n(i) * const_unit_n(j);
      t_assemble_tan(0, 0) -= 1.;
      t_assemble_tan(1, 1) -= 1.;
      t_assemble_tan(2, 2) -= 1.;

      t_assemble_m(i) -=
          val_m_2 * (t_assemble_tan(j, l) * t_stress(l, k) * const_unit_n(k)) *
          (t3_1(i, m, l) * const_unit_n(m) * t_assemble_tan(l, j));

      // if (type == MBEDGE)
        // cerr <<" " <<  t_assemble_m << "\n";

      ++t3_1;
      ++t_base_master;
    }
    ++t_w;
    ++gap_gp;
    ++tilde_c_nitsche_fun;
    }
  // cerr << "\n \n"
  //      << "start f_n " << vec_f << "\n";
  //CHKERR aSemble(side, type, data);
  Vec f;
  if (F == PETSC_NULL) {
    f = getFEMethod()->snes_f;
  } else {
    f = F;
  }

  const int nb_col = data.getIndices().size();

  CHKERR VecSetValues(f, nb_col, &data.getIndices()[0], &vec_f[0], ADD_VALUES);

  MoFEMFunctionReturn(0);
}


MoFEMErrorCode SimpleContactProblem::OpLoopForRowDataOnMaster::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  int nb_row = data.getIndices().size();
  if (nb_row == 0)
    MoFEMFunctionReturnHot(0);
  
  if(type == MBVERTEX)
    commonDataSimpleContact->masterRowData->clear();

        // masterRowData =
        //     boost::make_shared<std::vector<DataForcesAndSourcesCore::EntData
        //     *>>();

        commonDataSimpleContact->masterRowData->push_back(&data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpLoopForRowDataOnSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;


  int nb_row = data.getIndices().size();
  if (nb_row == 0)
    MoFEMFunctionReturnHot(0);

  if (type == MBVERTEX)
    commonDataSimpleContact->slaveRowData->clear();

  // masterRowData =
  //     boost::make_shared<std::vector<DataForcesAndSourcesCore::EntData *>>();

  commonDataSimpleContact->slaveRowData->push_back(&data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpStressDerivativeGapSlave_dx::getJac(
    DataForcesAndSourcesCore::EntData &col_data, int gg) {
  return get_jac_contact<0>(col_data, gg, commonData.jacStress[gg], jac);
}

MoFEMErrorCode SimpleContactProblem::OpStressDerivativeGapSlave_dx::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  int nb_row = data.getIndices().size();
  if (nb_row == 0)
    MoFEMFunctionReturnHot(0);

  // if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
  //     dAta.tEts.end()) {
  //   MoFEMFunctionReturnHot(0);
  // }
  // cerr << "rows " << nb_row << " types " << type
  //      << "\n";
  // const int nb_base_functions = row_data.getN().size2();
  const int nb_gauss_pts = data.getN().size1();
  int nb_base_fun_row = data.getFieldData().size() / 3;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'m', 3> m;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;

  vec_f.resize(3 * nb_base_fun_row,
               false); // the last false in ublas
                       // resize will destroy (not
                       // preserved) the old
                       // values
  vec_f.clear();

  const double area_s =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto get_tensor_vec = [](VectorDouble &n, const int r) {
    return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
  };

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

  auto t_w = getFTensor0IntegrationWeight();

  // k.resize(nb_row, nb_row, false);
  // k.clear();
  jac.resize(9, nb_row, false);
  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

auto tilde_c_nitsche_fun =
      getFTensor0FromVec(*commonDataSimpleContact->tildeCFunNitschePtr);

auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
  return FTensor::Tensor2<double *, 3, 3>(
      &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
      &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
      &m(r + 2, c + 2));
};

for (int gg = 0; gg != nb_gauss_pts; gg++) {

  MatrixDouble3by3 &stress =
      commonDataSimpleContact->elasticityCommonData.sTress[gg];

  // if (gap_gp > 0) {
  //   ++gap_gp;
  //   ++t_w;
  //   continue;
  // }
  CHKERR getJac(data, gg);
  // double val = getVolume() * getGaussPts()(3, gg);
  // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
  //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
  // }

  FTensor::Tensor2<double *, 3, 3> t_stress(
      &stress(0, 0), &stress(0, 1), &stress(0, 2), &stress(1, 0), &stress(1, 1),
      &stress(1, 2), &stress(2, 0), &stress(2, 1), &stress(2, 2));

  FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1(
      &jac(3 * 0 + 0, 0), &jac(3 * 0 + 0, 1), &jac(3 * 0 + 0, 2),
      &jac(3 * 0 + 1, 0), &jac(3 * 0 + 1, 1), &jac(3 * 0 + 1, 2),
      &jac(3 * 0 + 2, 0), &jac(3 * 0 + 2, 1), &jac(3 * 0 + 2, 2),
      &jac(3 * 1 + 0, 0), &jac(3 * 1 + 0, 1), &jac(3 * 1 + 0, 2),
      &jac(3 * 1 + 1, 0), &jac(3 * 1 + 1, 1), &jac(3 * 1 + 1, 2),
      &jac(3 * 1 + 2, 0), &jac(3 * 1 + 2, 1), &jac(3 * 1 + 2, 2),
      &jac(3 * 2 + 0, 0), &jac(3 * 2 + 0, 1), &jac(3 * 2 + 0, 2),
      &jac(3 * 2 + 1, 0), &jac(3 * 2 + 1, 1), &jac(3 * 2 + 1, 2),
      &jac(3 * 2 + 2, 0), &jac(3 * 2 + 2, 1), &jac(3 * 2 + 2, 2));

  // FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1(
  //     &commonData.jacStress[gg](3 * 0 + 0, 0),
  //     &commonData.jacStress[gg](3 * 0 + 0, 1),
  //     &commonData.jacStress[gg](3 * 0 + 0, 2),
  //     &commonData.jacStress[gg](3 * 0 + 1, 0),
  //     &commonData.jacStress[gg](3 * 0 + 1, 1),
  //     &commonData.jacStress[gg](3 * 0 + 1, 2),
  //     &commonData.jacStress[gg](3 * 0 + 2, 0),
  //     &commonData.jacStress[gg](3 * 0 + 2, 1),
  //     &commonData.jacStress[gg](3 * 0 + 2, 2),
  //     &commonData.jacStress[gg](3 * 1 + 0, 0),
  //     &commonData.jacStress[gg](3 * 1 + 0, 1),
  //     &commonData.jacStress[gg](3 * 1 + 0, 2),
  //     &commonData.jacStress[gg](3 * 1 + 1, 0),
  //     &commonData.jacStress[gg](3 * 1 + 1, 1),
  //     &commonData.jacStress[gg](3 * 1 + 1, 2),
  //     &commonData.jacStress[gg](3 * 1 + 2, 0),
  //     &commonData.jacStress[gg](3 * 1 + 2, 1),
  //     &commonData.jacStress[gg](3 * 1 + 2, 2),
  //     &commonData.jacStress[gg](3 * 2 + 0, 0),
  //     &commonData.jacStress[gg](3 * 2 + 0, 1),
  //     &commonData.jacStress[gg](3 * 2 + 0, 2),
  //     &commonData.jacStress[gg](3 * 2 + 1, 0),
  //     &commonData.jacStress[gg](3 * 2 + 1, 1),
  //     &commonData.jacStress[gg](3 * 2 + 1, 2),
  //     &commonData.jacStress[gg](3 * 2 + 2, 0),
  //     &commonData.jacStress[gg](3 * 2 + 2, 1),
  //     &commonData.jacStress[gg](3 * 2 + 2, 2));

  // for (int cc = 0; cc != nb_col / 3; cc++) {
  //   FTensor::Tensor1<double *, 3> diff_base_functions =
  //       row_data.getFTensor1DiffN<3>(gg, 0);
  //   FTensor::Tensor2<double *, 3, 3> lhs(
  //       &k(0, 3 * cc + 0), &k(0, 3 * cc + 1), &k(0, 3 * cc + 2),
  //       &k(1, 3 * cc + 0), &k(1, 3 * cc + 1), &k(1, 3 * cc + 2),
  //       &k(2, 3 * cc + 0), &k(2, 3 * cc + 1), &k(2, 3 * cc + 2), 3 *
  //       nb_col);
  //   for (int rr = 0; rr != nb_row / 3; rr++) {
  //     lhs(i, j) += val * t3_1(i, m, j) * diff_base_functions(m);
  //     ++diff_base_functions;
  //     ++lhs;
  //   }
  //   ++t3_1;
  // }

  // half for the average stress story
  // cerr << " Theta S Val " << thetaSVal << "\n";
  // cerr << " Omega Val " << omegaVal << "\n";
  // cerr << " cN " << cN << "\n";

  const double val_s =
      thetaSVal * tilde_c_nitsche_fun * omegaVal * t_w * area_s / cN;

  const double val_s_2 = omegaVal * thetaSVal * t_w * area_s / cN;
  // cerr << "cN " << cN << "\n";
  // cerr << "omegaVal " << omegaVal << "\n";
  // cerr << "thetaSVal " << thetaSVal << "\n";

  FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));

  for (int bbc = 0; bbc != nb_base_fun_row; ++bbc) {

    auto t_assemble_s = get_tensor_vec(vec_f, 3 * bbc);

    // t_assemble_s(j) +=
    //     val_s_2 * t_stress(j, m) * const_unit_n(m) * t_base_slave;



//15/03/2020
    t_assemble_s(j) -= val_s * t3_1(i, m, j) * const_unit_n(i) *
                       const_unit_n(m); // base not needed

    MatrixDouble t_tan;
    t_tan.resize(3, 3, false);

    auto t_assemble_tan = get_tensor_from_mat(t_tan, 0, 0);
    t_assemble_tan(i, j) = const_unit_n(i) * const_unit_n(j);
    t_assemble_tan(0, 0) -= 1.;
    t_assemble_tan(1, 1) -= 1.;
    t_assemble_tan(2, 2) -= 1.;


    t_assemble_s(i) -=
        val_s_2 * (t_assemble_tan(j, l) * t_stress(l, k) * const_unit_n(k)) *
        (t3_1(i, m, l) * const_unit_n(m) * t_assemble_tan(l, j));

    // t_assemble_s(i) -=
    //     val_s_2 * ((const_unit_n(l) * t_stress(l, k) * const_unit_n(k) -
    //                 tilde_c_nitsche_fun) * const_unit_n(j) +
    //                t_assemble_tan(j, l) * t_stress(l, k) * const_unit_n(k)) * t3_1(j, m, i) * const_unit_n(m) ;

    ++t3_1;
    ++t_base_slave;
  }
  ++t_w;
  ++gap_gp;
  ++tilde_c_nitsche_fun;
  }

  // CHKERR aSemble(side, type, data);
  Vec f;
  if (F == PETSC_NULL) {
    f = getFEMethod()->snes_f;
  } else {
    f = F;
  }

  const int nb_col = data.getIndices().size();

  CHKERR VecSetValues(f, nb_col, &data.getIndices()[0], &vec_f[0], ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpStressDerivativeGapMasterMaster_dx::getJac(
    DataForcesAndSourcesCore::EntData &col_data, int gg) {
  return get_jac_contact<0>(col_data, gg, commonData.jacStress[gg], jac);
}

MoFEMErrorCode
SimpleContactProblem::OpStressDerivativeGapMasterMaster_dx::getJacRow(
    DataForcesAndSourcesCore::EntData &row_data, int gg) {
  return get_jac_contact<0>(row_data, gg, commonData.jacStress[gg], jac_row);
}

MoFEMErrorCode SimpleContactProblem::OpStressDerivativeGapMasterMaster_dx::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  
  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

  

  if (col_side == MBVERTEX && row_side == MBVERTEX)
    // cerr << "Passed\n";

  const double area_slave =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'m', 3> m;
  FTensor::Index<'k', 3> k;

  const double area_s =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto const_unit_n_master =
      get_tensor_vec(commonDataSimpleContact->normalVectorMasterPtr.get()[0]);

  auto const_unit_n_slave = get_tensor_vec(
      commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto t_w = getFTensor0IntegrationWeight();

  // k.resize(nb_row, nb_row, false);
  // k.clear();
  jac.resize(9, nb_col, false);

  jac_row.resize(9, nb_row, false);

  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto nitsche_gap_diff_prod =
      getFTensor0FromVec(*commonDataSimpleContact->nitscheGapDiffProductPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    // if (gap_gp > 0) {
    //   ++gap_gp;
    //   continue;
    // }
    // ++gap_gp;

    // cerr << "Jack! " << commonData.jacStress[gg] << " \n";
    CHKERR getJac(col_data, gg);
    // cerr << "2qwe qwe\n";
    // double val = getVolume() * getGaussPts()(3, gg);
    // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
    //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
    // }
    FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1(
        &jac(3 * 0 + 0, 0), &jac(3 * 0 + 0, 1), &jac(3 * 0 + 0, 2),
        &jac(3 * 0 + 1, 0), &jac(3 * 0 + 1, 1), &jac(3 * 0 + 1, 2),
        &jac(3 * 0 + 2, 0), &jac(3 * 0 + 2, 1), &jac(3 * 0 + 2, 2),
        &jac(3 * 1 + 0, 0), &jac(3 * 1 + 0, 1), &jac(3 * 1 + 0, 2),
        &jac(3 * 1 + 1, 0), &jac(3 * 1 + 1, 1), &jac(3 * 1 + 1, 2),
        &jac(3 * 1 + 2, 0), &jac(3 * 1 + 2, 1), &jac(3 * 1 + 2, 2),
        &jac(3 * 2 + 0, 0), &jac(3 * 2 + 0, 1), &jac(3 * 2 + 0, 2),
        &jac(3 * 2 + 1, 0), &jac(3 * 2 + 1, 1), &jac(3 * 2 + 1, 2),
        &jac(3 * 2 + 2, 0), &jac(3 * 2 + 2, 1), &jac(3 * 2 + 2, 2));
    // for (int cc = 0; cc != nb_col / 3; cc++) {
    //   FTensor::Tensor1<double *, 3> diff_base_functions =
    //       row_data.getFTensor1DiffN<3>(gg, 0);
    //   FTensor::Tensor2<double *, 3, 3> lhs(
    //       &k(0, 3 * cc + 0), &k(0, 3 * cc + 1), &k(0, 3 * cc + 2),
    //       &k(1, 3 * cc + 0), &k(1, 3 * cc + 1), &k(1, 3 * cc + 2),
    //       &k(2, 3 * cc + 0), &k(2, 3 * cc + 1), &k(2, 3 * cc + 2), 3 *
    //       nb_col);
    //   for (int rr = 0; rr != nb_row / 3; rr++) {
    //     lhs(i, j) += val * t3_1(i, m, j) * diff_base_functions(m);
    //     ++diff_base_functions;
    //     ++lhs;
    //   }
    //   ++t3_1;
    // }

    // half for the average stress story
    const double val_m = 0.5 * t_w * area_s;
    FTensor::Tensor0<double *> t_base_master_col(&col_data.getN()(gg, 0));
    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
      // cerr << "2\n";

      CHKERR getJacRow(row_data, gg);
      // cerr << "2qwe qwe\n";
      // double val = getVolume() * getGaussPts()(3, gg);
      // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
      //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
      // }
      FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1_row(
          &jac_row(3 * 0 + 0, 0), &jac_row(3 * 0 + 0, 1),
          &jac_row(3 * 0 + 0, 2), &jac_row(3 * 0 + 1, 0),
          &jac_row(3 * 0 + 1, 1), &jac_row(3 * 0 + 1, 2),
          &jac_row(3 * 0 + 2, 0), &jac_row(3 * 0 + 2, 1),
          &jac_row(3 * 0 + 2, 2), &jac_row(3 * 1 + 0, 0),
          &jac_row(3 * 1 + 0, 1), &jac_row(3 * 1 + 0, 2),
          &jac_row(3 * 1 + 1, 0), &jac_row(3 * 1 + 1, 1),
          &jac_row(3 * 1 + 1, 2), &jac_row(3 * 1 + 2, 0),
          &jac_row(3 * 1 + 2, 1), &jac_row(3 * 1 + 2, 2),
          &jac_row(3 * 2 + 0, 0), &jac_row(3 * 2 + 0, 1),
          &jac_row(3 * 2 + 0, 2), &jac_row(3 * 2 + 1, 0),
          &jac_row(3 * 2 + 1, 1), &jac_row(3 * 2 + 1, 2),
          &jac_row(3 * 2 + 2, 0), &jac_row(3 * 2 + 2, 1),
          &jac_row(3 * 2 + 2, 2));

      FTensor::Tensor0<double *> t_base_master_row(&row_data.getN()(gg, 0));

      double sum_prod = 1. + nitsche_gap_diff_prod;
      double diff_prod = 1. - nitsche_gap_diff_prod;

      if (nitsche_gap_diff_prod == 0) {
        sum_prod *= 2.;
        diff_prod *= 2.;
      }

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        // cerr << "213123\n";

        auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

          
        t_assemble_m(m, k) -= 2. * val_m * (1. - omegaVal) * t3_1(i, j, k) *
                              const_unit_n_master(j) * const_unit_n_slave(i) *
                              const_unit_n_slave(m) *
                              t_base_master_row; // base not needed

        //         cerr << sum_prod << "sum_prod\n";
        //         cerr << diff_prod << "diff_prod\n";
        // cerr << omegaVal << "omegaVal\n";

        t_assemble_m(m, k) +=
            val_m *
            ( -sum_prod * (1. - omegaVal) * t3_1(i, j, k) *
                 const_unit_n_master(i) * const_unit_n_slave(j) -
             diff_prod * cN * t_base_master_col * const_unit_n_slave(k)) *
            (-const_unit_n_slave(m) * t_base_master_row +
             (1. - omegaVal) * thetaSVal * t3_1_row(i, j, m) *
                 const_unit_n_master(j) * const_unit_n_slave(i) /
                 cN); // base not needed

        // cerr << " diff_prod " << diff_prod << "\n";
        // cerr << " sum_prod " << sum_prod << "\n";

        // t_assemble_m(k, m) -= val_m * diff_prod * cN * t_base_master_col *
        //                       const_unit_n_slave(k) * const_unit_n_slave(m) *
        //                       t_base_master_row;

        // t_assemble_m(m, k) += val_m * (1. - omegaVal) * t3_1_row(i, j, k)
        // *
        //                       const_unit_n_master(i) *
        //                       const_unit_n_master(j) *
        //                       const_unit_n_slave(m) *
        //                       t_base_master_col; // base not needed

        // t_assemble_m(m, k) += t_assemble_m(k, m);

        ++t_base_master_row;
        ++t3_1_row;
      }
      ++t3_1;
      ++t_base_master_col;
    }
    ++t_w;
    ++nitsche_gap_diff_prod;
  }
  // CHKERR aSemble(side, type, data);
  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
    } else {
      aij = Aij;
    }

    CHKERR MatSetValues(
        aij, nb_row, &row_data.getIndices()[0],
        nb_col, &col_data.getIndices()[0], &*NN.data().begin(), ADD_VALUES);

    MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpStressDerivativeGapSlaveMaster_dx::getJac(
    DataForcesAndSourcesCore::EntData &col_data, int gg) {
  return get_jac_contact<0>(col_data, gg, commonDataMaster.jacStress[gg], jac);
}

MoFEMErrorCode
SimpleContactProblem::OpStressDerivativeGapSlaveMaster_dx::getJacRow(
    DataForcesAndSourcesCore::EntData &col_data, int gg) {
  return get_jac_contact<0>(col_data, gg, commonData.jacStress[gg], jac_row);
}

MoFEMErrorCode
SimpleContactProblem::OpStressDerivativeGapSlaveMaster_dx::doWork(
    int col_side, EntityType col_type,
    DataForcesAndSourcesCore::EntData &col_data) {

  MoFEMFunctionBegin;

  // const int nb_row = row_data.getIndices().size();
  // if (!nb_row)
  //   MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);

  // boost::shared_ptr<std::vector<DataForcesAndSourcesCore::EntData *>>
  //     masterRowData;
  for (std::vector<DataForcesAndSourcesCore::EntData *>::iterator row_data =
           commonDataSimpleContact->slaveRowData->begin();
       row_data != commonDataSimpleContact->slaveRowData->end(); ++row_data) {

    const int nb_row = (*row_data)->getIndices().size();
    if (!nb_row)
      continue;

    // cerr << " row SlaveMaster asd" << (*row_data)->getIndices() << "\n";
    // cerr << " col " << col_data.getIndices() << "\n";

    const int nb_gauss_pts = (*row_data)->getN().size1();

    int nb_base_fun_row = (*row_data)->getFieldData().size() / 3;
    int nb_base_fun_col = col_data.getFieldData().size() / 3;

    // cerr << "OpStressDerivativeGapMasterSlave_dx "
    //      << " Row " << nb_row << " Col " << nb_col << " indices "
    //      << col_data.getIndices() << "\n";

    // cerr << " base funs row " << nb_base_fun_row << " base funs  cols "
    //      << nb_base_fun_col << " indices rows " << (*row_data)->getIndices()
    //      << "\n";

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

    NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
    NN.clear();

    NNT.resize(3 * nb_base_fun_col, 3 * nb_base_fun_row, false);
    NNT.clear();

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'m', 3> m;
    FTensor::Index<'k', 3> k;

    const double area_s =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto const_unit_n =
        get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

    auto const_unit_n_master =
        get_tensor_vec(commonDataSimpleContact->normalVectorMasterPtr.get()[0]);

    auto t_w = getFTensor0IntegrationWeight();

    // k.resize(nb_row, nb_row, false);
    // k.clear();
    jac.resize(9, nb_col, false);
    jac_row.resize(9, nb_row, false);

    auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);
    auto nitsche_gap_diff_prod =
        getFTensor0FromVec(*commonDataSimpleContact->nitscheGapDiffProductPtr);
    for (int gg = 0; gg != nb_gauss_pts; gg++) {

        double sum_prod = 1. + nitsche_gap_diff_prod;
        double diff_prod = 1. - nitsche_gap_diff_prod;

        if (nitsche_gap_diff_prod == 0) {
          sum_prod *= 2;
          diff_prod *= 2;
        }

        // if (gap_gp > 0) {
        //   ++gap_gp;
        //   continue;
        // }
        // ++gap_gp;

        CHKERR getJac(col_data, gg);
        // double val = getVolume() * getGaussPts()(3, gg);
        // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
        //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
        // }
        FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1(
            &jac(3 * 0 + 0, 0), &jac(3 * 0 + 0, 1), &jac(3 * 0 + 0, 2),
            &jac(3 * 0 + 1, 0), &jac(3 * 0 + 1, 1), &jac(3 * 0 + 1, 2),
            &jac(3 * 0 + 2, 0), &jac(3 * 0 + 2, 1), &jac(3 * 0 + 2, 2),
            &jac(3 * 1 + 0, 0), &jac(3 * 1 + 0, 1), &jac(3 * 1 + 0, 2),
            &jac(3 * 1 + 1, 0), &jac(3 * 1 + 1, 1), &jac(3 * 1 + 1, 2),
            &jac(3 * 1 + 2, 0), &jac(3 * 1 + 2, 1), &jac(3 * 1 + 2, 2),
            &jac(3 * 2 + 0, 0), &jac(3 * 2 + 0, 1), &jac(3 * 2 + 0, 2),
            &jac(3 * 2 + 1, 0), &jac(3 * 2 + 1, 1), &jac(3 * 2 + 1, 2),
            &jac(3 * 2 + 2, 0), &jac(3 * 2 + 2, 1), &jac(3 * 2 + 2, 2));
        // for (int cc = 0; cc != nb_col / 3; cc++) {
        //   FTensor::Tensor1<double *, 3> diff_base_functions =
        //       (*row_data)->getFTensor1DiffN<3>(gg, 0);
        //   FTensor::Tensor2<double *, 3, 3> lhs(
        //       &k(0, 3 * cc + 0), &k(0, 3 * cc + 1), &k(0, 3 * cc + 2),
        //       &k(1, 3 * cc + 0), &k(1, 3 * cc + 1), &k(1, 3 * cc + 2),
        //       &k(2, 3 * cc + 0), &k(2, 3 * cc + 1), &k(2, 3 * cc + 2), 3 *
        //       nb_col);
        //   for (int rr = 0; rr != nb_row / 3; rr++) {
        //     lhs(i, j) += val * t3_1(i, m, j) * diff_base_functions(m);
        //     ++diff_base_functions;
        //     ++lhs;
        //   }
        //   ++t3_1;
        // }

        FTensor::Tensor0<double *> t_base_master_col(&col_data.getN()(gg, 0));
        // half for the average stress story
        const double val_m = 0.5 * t_w * area_s;
        for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
          FTensor::Tensor0<double *> t_base_slave_row(
              &(*row_data)->getN()(gg, 0));

          // CHKERR getJacRow(row_data, gg);
          // // cerr << "2qwe qwe\n";
          // // double val = getVolume() * getGaussPts()(3, gg);
          // // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
          // //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
          // // }
          // FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1_row(
          //     &jac_row(3 * 0 + 0, 0), &jac_row(3 * 0 + 0, 1),
          //     &jac_row(3 * 0 + 0, 2), &jac_row(3 * 0 + 1, 0),
          //     &jac_row(3 * 0 + 1, 1), &jac_row(3 * 0 + 1, 2),
          //     &jac_row(3 * 0 + 2, 0), &jac_row(3 * 0 + 2, 1),
          //     &jac_row(3 * 0 + 2, 2), &jac_row(3 * 1 + 0, 0),
          //     &jac_row(3 * 1 + 0, 1), &jac_row(3 * 1 + 0, 2),
          //     &jac_row(3 * 1 + 1, 0), &jac_row(3 * 1 + 1, 1),
          //     &jac_row(3 * 1 + 1, 2), &jac_row(3 * 1 + 2, 0),
          //     &jac_row(3 * 1 + 2, 1), &jac_row(3 * 1 + 2, 2),
          //     &jac_row(3 * 2 + 0, 0), &jac_row(3 * 2 + 0, 1),
          //     &jac_row(3 * 2 + 0, 2), &jac_row(3 * 2 + 1, 0),
          //     &jac_row(3 * 2 + 1, 1), &jac_row(3 * 2 + 1, 2),
          //     &jac_row(3 * 2 + 2, 0), &jac_row(3 * 2 + 2, 1),
          //     &jac_row(3 * 2 + 2, 2));

          CHKERR getJacRow(**row_data, gg);

          // double val = getVolume() * getGaussPts()(3, gg);
          // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
          //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
          // }
          // cerr << " Row: gg " << gg << " rows " << nb_base_fun_row << "\n";
          FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1_row(
              &jac_row(3 * 0 + 0, 0), &jac_row(3 * 0 + 0, 1),
              &jac_row(3 * 0 + 0, 2), &jac_row(3 * 0 + 1, 0),
              &jac_row(3 * 0 + 1, 1), &jac_row(3 * 0 + 1, 2),
              &jac_row(3 * 0 + 2, 0), &jac_row(3 * 0 + 2, 1),
              &jac_row(3 * 0 + 2, 2), &jac_row(3 * 1 + 0, 0),
              &jac_row(3 * 1 + 0, 1), &jac_row(3 * 1 + 0, 2),
              &jac_row(3 * 1 + 1, 0), &jac_row(3 * 1 + 1, 1),
              &jac_row(3 * 1 + 1, 2), &jac_row(3 * 1 + 2, 0),
              &jac_row(3 * 1 + 2, 1), &jac_row(3 * 1 + 2, 2),
              &jac_row(3 * 2 + 0, 0), &jac_row(3 * 2 + 0, 1),
              &jac_row(3 * 2 + 0, 2), &jac_row(3 * 2 + 1, 0),
              &jac_row(3 * 2 + 1, 1), &jac_row(3 * 2 + 1, 2),
              &jac_row(3 * 2 + 2, 0), &jac_row(3 * 2 + 2, 1),
              &jac_row(3 * 2 + 2, 2));
          for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

            auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);
            auto t_assemble_s = get_tensor_from_mat(NNT, 3 * bbc, 3 * bbr);

            // t_assemble_m(m, k) += val_m * t3_1(i, j, k) *
            // const_unit_n_master(i) *
            //                       const_unit_n_master(j) * const_unit_n(m) *
            //                       t_base_slave_row; // base not needed

            t_assemble_m(m, k) += 2. * val_m * (1 - omegaVal) * t3_1(i, j, k) *
                                  const_unit_n(i) * const_unit_n_master(j) *
                                  const_unit_n(m) *
                                  t_base_slave_row; // base not needed

            t_assemble_m(m, k) -=
                val_m *
                ( -sum_prod * (1. - omegaVal) * t3_1(i, j, k) * const_unit_n(i) *
                     const_unit_n_master(j) +
                 diff_prod * cN * t_base_master_col * const_unit_n(k)) *
                (const_unit_n(m) * t_base_slave_row -
                 omegaVal * thetaSVal * t3_1_row(i, j, m) * const_unit_n(i) *
                     const_unit_n(j) / cN);

            //(sum_prod * (1. - omegaVal) * t3_1(i, j, k) *
            //      const_unit_n_master(i) * const_unit_n_master(j) -
            // diff_prod * cN * t_base_master_col * const_unit_n_slave(k))

            // t_assemble_s(m, k) += val_m * t3_1_row(i, j, k) *
            //                       const_unit_n_master(i) *
            //                       const_unit_n_master(j) *
            //                       const_unit_n(m) * t_base_master_col; //
            //                       base not needed
            // // if (gap_gp <= 0)
            //   t_assemble_m(k, m) += val_m * t3_1_row(i, j, k) *
            //   const_unit_n(i) *
            //                         const_unit_n(j) * const_unit_n(m) *
            //                         t_base_master_col; // base not needed

            // cerr << "Master Stress : " << t_assemble_m << "\n";
            ++t3_1_row;
            ++t_base_slave_row;
          }
          ++t3_1;
          ++t_base_master_col;
        }
        ++t_w;
        ++nitsche_gap_diff_prod;
    }
    // CHKERR aSemble(side, type, data);
    Mat aij;
    if (Aij == PETSC_NULL) {
      aij = getFEMethod()->snes_B;
    } else {
      aij = Aij;
    }

//  cerr << "Cols New " << (*row_data)->getIndices() << " \n";

//     cerr << "Rows New " << col_data.getIndices() << " \n";

//      cerr << " NN " << NN << "\n";
    CHKERR MatSetValues(aij, nb_row, &(*row_data)->getIndices()[0], nb_col,
                        &col_data.getIndices()[0], &*NN.data().begin(),
                        ADD_VALUES);
// noalias(NNT) = trans(NN);
    // CHKERR MatSetValues(aij, nb_col, &col_data.getIndices()[0], nb_row,
    //                     &(*row_data)->getIndices()[0], &*NNT.data().begin(),
    //                     ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpStressDerivativeGapSlaveMaster_dx2::getJac(
    DataForcesAndSourcesCore::EntData &col_data, int gg) {
  return get_jac_contact<0>(col_data, gg, commonData.jacStress[gg], jac);
}

MoFEMErrorCode
SimpleContactProblem::OpStressDerivativeGapSlaveMaster_dx2::getJacRow(
    DataForcesAndSourcesCore::EntData &col_data, int gg) {
  return get_jac_contact<0>(col_data, gg, commonDataSlave.jacStress[gg],
                            jac_row);
}

MoFEMErrorCode
SimpleContactProblem::OpStressDerivativeGapSlaveMaster_dx2::doWork(
    int col_side, EntityType col_type,
    DataForcesAndSourcesCore::EntData &col_data) {

  MoFEMFunctionBegin;

  // const int nb_row = row_data.getIndices().size();
  // if (!nb_row)
  //   MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);

  // boost::shared_ptr<std::vector<DataForcesAndSourcesCore::EntData *>>
  //     masterRowData;
  for (std::vector<DataForcesAndSourcesCore::EntData *>::iterator row_data =
           commonDataSimpleContact->slaveRowData->begin();
       row_data != commonDataSimpleContact->slaveRowData->end(); ++row_data) {

    const int nb_row = (*row_data)->getIndices().size();
    if (!nb_row)
      continue;

    // cerr << " row SlaveMaster asd" << (*row_data)->getIndices() << "\n";
    // cerr << " col " << col_data.getIndices() << "\n";

    const int nb_gauss_pts = (*row_data)->getN().size1();

    int nb_base_fun_row = (*row_data)->getFieldData().size() / 3;
    int nb_base_fun_col = col_data.getFieldData().size() / 3;

    // cerr << "OpStressDerivativeGapMasterSlave_dx "
    //      << " Row " << nb_row << " Col " << nb_col << " indices "
    //      << col_data.getIndices() << "\n";

    // cerr << " base funs row " << nb_base_fun_row << " base funs  cols "
    //      << nb_base_fun_col << " indices rows " << (*row_data)->getIndices()
    //      << "\n";

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

    NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
    NN.clear();

    NNT.resize(3 * nb_base_fun_col, 3 * nb_base_fun_row, false);
    NNT.clear();

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'m', 3> m;
    FTensor::Index<'k', 3> k;

    const double area_s =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto const_unit_n =
        get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

    auto const_unit_n_master =
        get_tensor_vec(commonDataSimpleContact->normalVectorMasterPtr.get()[0]);

    auto t_w = getFTensor0IntegrationWeightSlave();

    // k.resize(nb_row, nb_row, false);
    // k.clear();
    jac.resize(9, nb_col, false);
    jac_row.resize(9, nb_row, false);

    auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    for (int gg = 0; gg != nb_gauss_pts; gg++) {

      // if (gap_gp > 0) {
      //   ++gap_gp;
      //   continue;
      // }
      // ++gap_gp;

      CHKERR getJac(col_data, gg);
      // double val = getVolume() * getGaussPts()(3, gg);
      // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
      //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
      // }
      FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1(
          &jac(3 * 0 + 0, 0), &jac(3 * 0 + 0, 1), &jac(3 * 0 + 0, 2),
          &jac(3 * 0 + 1, 0), &jac(3 * 0 + 1, 1), &jac(3 * 0 + 1, 2),
          &jac(3 * 0 + 2, 0), &jac(3 * 0 + 2, 1), &jac(3 * 0 + 2, 2),
          &jac(3 * 1 + 0, 0), &jac(3 * 1 + 0, 1), &jac(3 * 1 + 0, 2),
          &jac(3 * 1 + 1, 0), &jac(3 * 1 + 1, 1), &jac(3 * 1 + 1, 2),
          &jac(3 * 1 + 2, 0), &jac(3 * 1 + 2, 1), &jac(3 * 1 + 2, 2),
          &jac(3 * 2 + 0, 0), &jac(3 * 2 + 0, 1), &jac(3 * 2 + 0, 2),
          &jac(3 * 2 + 1, 0), &jac(3 * 2 + 1, 1), &jac(3 * 2 + 1, 2),
          &jac(3 * 2 + 2, 0), &jac(3 * 2 + 2, 1), &jac(3 * 2 + 2, 2));
      // for (int cc = 0; cc != nb_col / 3; cc++) {
      //   FTensor::Tensor1<double *, 3> diff_base_functions =
      //       (*row_data)->getFTensor1DiffN<3>(gg, 0);
      //   FTensor::Tensor2<double *, 3, 3> lhs(
      //       &k(0, 3 * cc + 0), &k(0, 3 * cc + 1), &k(0, 3 * cc + 2),
      //       &k(1, 3 * cc + 0), &k(1, 3 * cc + 1), &k(1, 3 * cc + 2),
      //       &k(2, 3 * cc + 0), &k(2, 3 * cc + 1), &k(2, 3 * cc + 2), 3 *
      //       nb_col);
      //   for (int rr = 0; rr != nb_row / 3; rr++) {
      //     lhs(i, j) += val * t3_1(i, m, j) * diff_base_functions(m);
      //     ++diff_base_functions;
      //     ++lhs;
      //   }
      //   ++t3_1;
      // }

      FTensor::Tensor0<double *> t_base_slave_col(&col_data.getN()(gg, 0));
      // half for the average stress story
      const double val_m = 0.5 * t_w * area_s;
      for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
        FTensor::Tensor0<double *> t_base_master_row(
            &(*row_data)->getN()(gg, 0));

        // CHKERR getJacRow(row_data, gg);
        // // cerr << "2qwe qwe\n";
        // // double val = getVolume() * getGaussPts()(3, gg);
        // // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
        // //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
        // // }
        // FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1_row(
        //     &jac_row(3 * 0 + 0, 0), &jac_row(3 * 0 + 0, 1),
        //     &jac_row(3 * 0 + 0, 2), &jac_row(3 * 0 + 1, 0),
        //     &jac_row(3 * 0 + 1, 1), &jac_row(3 * 0 + 1, 2),
        //     &jac_row(3 * 0 + 2, 0), &jac_row(3 * 0 + 2, 1),
        //     &jac_row(3 * 0 + 2, 2), &jac_row(3 * 1 + 0, 0),
        //     &jac_row(3 * 1 + 0, 1), &jac_row(3 * 1 + 0, 2),
        //     &jac_row(3 * 1 + 1, 0), &jac_row(3 * 1 + 1, 1),
        //     &jac_row(3 * 1 + 1, 2), &jac_row(3 * 1 + 2, 0),
        //     &jac_row(3 * 1 + 2, 1), &jac_row(3 * 1 + 2, 2),
        //     &jac_row(3 * 2 + 0, 0), &jac_row(3 * 2 + 0, 1),
        //     &jac_row(3 * 2 + 0, 2), &jac_row(3 * 2 + 1, 0),
        //     &jac_row(3 * 2 + 1, 1), &jac_row(3 * 2 + 1, 2),
        //     &jac_row(3 * 2 + 2, 0), &jac_row(3 * 2 + 2, 1),
        //     &jac_row(3 * 2 + 2, 2));

        CHKERR getJacRow(**row_data, gg);

        // double val = getVolume() * getGaussPts()(3, gg);
        // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
        //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
        // }
        // cerr << " Row: gg " << gg << " rows " << nb_base_fun_row << "\n";
        FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1_row(
            &jac_row(3 * 0 + 0, 0), &jac_row(3 * 0 + 0, 1),
            &jac_row(3 * 0 + 0, 2), &jac_row(3 * 0 + 1, 0),
            &jac_row(3 * 0 + 1, 1), &jac_row(3 * 0 + 1, 2),
            &jac_row(3 * 0 + 2, 0), &jac_row(3 * 0 + 2, 1),
            &jac_row(3 * 0 + 2, 2), &jac_row(3 * 1 + 0, 0),
            &jac_row(3 * 1 + 0, 1), &jac_row(3 * 1 + 0, 2),
            &jac_row(3 * 1 + 1, 0), &jac_row(3 * 1 + 1, 1),
            &jac_row(3 * 1 + 1, 2), &jac_row(3 * 1 + 2, 0),
            &jac_row(3 * 1 + 2, 1), &jac_row(3 * 1 + 2, 2),
            &jac_row(3 * 2 + 0, 0), &jac_row(3 * 2 + 0, 1),
            &jac_row(3 * 2 + 0, 2), &jac_row(3 * 2 + 1, 0),
            &jac_row(3 * 2 + 1, 1), &jac_row(3 * 2 + 1, 2),
            &jac_row(3 * 2 + 2, 0), &jac_row(3 * 2 + 2, 1),
            &jac_row(3 * 2 + 2, 2));

        for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

          auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);
          auto t_assemble_s = get_tensor_from_mat(NNT, 3 * bbc, 3 * bbr);

          // t_assemble_m(m, k) += val_m * t3_1(i, j, k) *
          // const_unit_n_master(i) *
          //                       const_unit_n_master(j) * const_unit_n(m) *
          //                       t_base_master_row; // base not needed

          t_assemble_s(m, k) += val_m * t3_1_row(i, j, k) *
                                const_unit_n_master(i) *
                                const_unit_n_master(j) * const_unit_n(m) *
                                t_base_slave_col; // base not needed

          t_assemble_m(k, m) += val_m * t3_1_row(i, j, k) * const_unit_n(i) *
                                const_unit_n(j) * const_unit_n(m) *
                                t_base_slave_col; // base not needed
          // cerr << "Master Stress : " << t_assemble_m << "\n";
          ++t3_1_row;
          ++t_base_master_row;
        }
        ++t3_1;
        ++t_base_slave_col;
      }
      ++t_w;
    }
    // CHKERR aSemble(side, type, data);
    Mat aij;
    if (Aij == PETSC_NULL) {
      aij = getFEMethod()->snes_B;
    } else {
      aij = Aij;
    }
    // cerr << " NN " << NN << "\n";
    // CHKERR MatSetValues(aij, nb_row, &(*row_data)->getIndices()[0], nb_col,
    //                     &col_data.getIndices()[0], &*NN.data().begin(),
    //                     ADD_VALUES);

    // CHKERR MatSetValues(aij, nb_col, &col_data.getIndices()[0], nb_row,
    //                     &(*row_data)->getIndices()[0], &*NNT.data().begin(),
    //                     ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

// MoFEMErrorCode
// SimpleContactProblem::OpStressDerivativeGapSlaveMaster_dx::doWork(
//     int row_side, int col_side, EntityType row_type, EntityType col_type,
//     DataForcesAndSourcesCore::EntData &row_data,
//     DataForcesAndSourcesCore::EntData &col_data) {
//   MoFEMFunctionBegin;

//   const int nb_row = row_data.getIndices().size();
//   if (!nb_row)
//     MoFEMFunctionReturnHot(0);
//   const int nb_col = col_data.getIndices().size();
//   if (!nb_col)
//     MoFEMFunctionReturnHot(0);
//   const int nb_gauss_pts = row_data.getN().size1();
//   cerr << "Row " << nb_row << " Col " << nb_col << "\n";
//   int nb_base_fun_row =
//       row_data.getFieldData().size() / 3;
//   int nb_base_fun_col = col_data.getFieldData().size() / 3;

//   const double area_slave =
//       commonDataSimpleContact->areaSlave; // same area in master and slave

//   auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
//     return FTensor::Tensor2<double *, 3, 3>(
//         &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
//         &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
//         &m(r + 2, c + 2));
//   };

//   auto get_tensor_vec = [](VectorDouble &n) {
//     return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
//   };

//   NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
//   NN.clear();

//   FTensor::Index<'i', 3> i;
//   FTensor::Index<'j', 3> j;
//   FTensor::Index<'m', 3> m;
//   FTensor::Index<'k', 3> k;

//   const double area_s =
//       commonDataSimpleContact->areaSlave; // same area in master and slave

//   auto const_unit_n_master =
//       get_tensor_vec(commonDataSimpleContact->normalVectorMasterPtr.get()[0]);

//   auto const_unit_n_slave =
//       get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

//   auto t_w = getFTensor0IntegrationWeight();

//   // k.resize(nb_row, nb_row, false);
//   // k.clear();
//   jac.resize(9, nb_col, false);

//   auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

//   for (int gg = 0; gg != nb_gauss_pts; gg++) {

//     //   if (gap_gp > 0) {
//     //   ++gap_gp;
//     //   continue;
//     // }
//     // ++gap_gp;
  
//       // cerr << "Jack! " << commonData.jacStress[gg] << " \n";
//       CHKERR getJac(col_data, gg);
//       // cerr << "2qwe qwe\n";
//       // double val = getVolume() * getGaussPts()(3, gg);
//       // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
//       //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
//       // }
//       FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1(
//           &jac(3 * 0 + 0, 0), &jac(3 * 0 + 0, 1), &jac(3 * 0 + 0, 2),
//           &jac(3 * 0 + 1, 0), &jac(3 * 0 + 1, 1), &jac(3 * 0 + 1, 2),
//           &jac(3 * 0 + 2, 0), &jac(3 * 0 + 2, 1), &jac(3 * 0 + 2, 2),
//           &jac(3 * 1 + 0, 0), &jac(3 * 1 + 0, 1), &jac(3 * 1 + 0, 2),
//           &jac(3 * 1 + 1, 0), &jac(3 * 1 + 1, 1), &jac(3 * 1 + 1, 2),
//           &jac(3 * 1 + 2, 0), &jac(3 * 1 + 2, 1), &jac(3 * 1 + 2, 2),
//           &jac(3 * 2 + 0, 0), &jac(3 * 2 + 0, 1), &jac(3 * 2 + 0, 2),
//           &jac(3 * 2 + 1, 0), &jac(3 * 2 + 1, 1), &jac(3 * 2 + 1, 2),
//           &jac(3 * 2 + 2, 0), &jac(3 * 2 + 2, 1), &jac(3 * 2 + 2, 2));
//       // for (int cc = 0; cc != nb_col / 3; cc++) {
//       //   FTensor::Tensor1<double *, 3> diff_base_functions =
//       //       row_data.getFTensor1DiffN<3>(gg, 0);
//       //   FTensor::Tensor2<double *, 3, 3> lhs(
//       //       &k(0, 3 * cc + 0), &k(0, 3 * cc + 1), &k(0, 3 * cc + 2),
//       //       &k(1, 3 * cc + 0), &k(1, 3 * cc + 1), &k(1, 3 * cc + 2),
//       //       &k(2, 3 * cc + 0), &k(2, 3 * cc + 1), &k(2, 3 * cc + 2), 3 *
//       //       nb_col);
//       //   for (int rr = 0; rr != nb_row / 3; rr++) {
//       //     lhs(i, j) += val * t3_1(i, m, j) * diff_base_functions(m);
//       //     ++diff_base_functions;
//       //     ++lhs;
//       //   }
//       //   ++t3_1;
//       // }

//       // half for the average stress story
//       const double val_m = 0.5 * t_w * area_s;
//       for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
//         // cerr << "2\n";

//         FTensor::Tensor0<double *> t_base_slave_row(&row_data.getN()(gg, 0));

//         for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
//           // cerr << "213123\n";

//           auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

//           t_assemble_m(m, k) -= val_m *
//                                 (t3_1(i, j, k) * const_unit_n_master(i) *
//                                  const_unit_n_master(j)) *
//                                 const_unit_n_slave(m) *
//                                 t_base_slave_row; // base not needed

//           ++t_base_slave_row;
//         }
//         ++t3_1;
//       }
//       ++t_w;
//     }
//     // CHKERR aSemble(side, type, data);
//     Mat aij;
//     if (Aij == PETSC_NULL) {
//       aij = getFEMethod()->snes_B;
//     } else {
//       aij = Aij;
//     }

//     CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
//                         &col_data.getIndices()[0], &*NN.data().begin(),
//                         ADD_VALUES);

//     MoFEMFunctionReturn(0);
// }



MoFEMErrorCode SimpleContactProblem::OpLagMultOnVertex::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  int nb_dofs = data.getFieldData().size();
  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);
  int nb_gauss_pts = data.getN().size1();

  double def_vals[1];
  bzero(def_vals, 1 * sizeof(double));
  Tag th_lagmult;
  CHKERR moabOut.tag_get_handle("VERTEX_LAGMULT", 1, MB_TYPE_DOUBLE, th_lagmult,
                                MB_TAG_CREAT | MB_TAG_SPARSE, def_vals);

  double lagrange_multipliers[1] = {0};
  double coords[3];
  EntityHandle new_vertex = getFEEntityHandle();

  VectorDouble all_coords = getCoordsSlave();

  for (int ii = 0; ii != 3; ++ii) {

    for (int dd = 0; dd != 3; dd++) {
      coords[dd] = all_coords[ ii * 3 + dd];
    }
    lagrange_multipliers[0] = data.getFieldData()[ii];

    CHKERR moabOut.create_vertex(&coords[0], new_vertex);
    CHKERR moabOut.tag_set_data(th_lagmult, &new_vertex, 1,
                                lagrange_multipliers);
  }
  MoFEMFunctionReturn(0);
  }


MoFEMErrorCode SimpleContactProblem::OpStressDerivativeGapSlaveSlave_dx::getJac(
    DataForcesAndSourcesCore::EntData &col_data, int gg) {
  return get_jac_contact<0>(col_data, gg, commonData.jacStress[gg], jac);
}

MoFEMErrorCode SimpleContactProblem::OpStressDerivativeGapSlaveSlave_dx::getJacRow(
    DataForcesAndSourcesCore::EntData &row_data, int gg) {
  return get_jac_contact<0>(row_data, gg, commonData.jacStress[gg], jac_row);
}

MoFEMErrorCode SimpleContactProblem::OpStressDerivativeGapSlaveSlave_dx::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

  const double area_slave =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'m', 3> m;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'f', 3> f;
  FTensor::Index<'l', 3> l;

  const double area_s =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto t_w = getFTensor0IntegrationWeight();

  // k.resize(nb_row, nb_row, false);
  // k.clear();
  jac.resize(9, nb_col, false);
  jac_row.resize(9, nb_row, false);

  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto nitsche_gap_diff_prod =
      getFTensor0FromVec(*commonDataSimpleContact->nitscheGapDiffProductPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    double sum_prod = 1. + nitsche_gap_diff_prod;
    double diff_prod = 1. - nitsche_gap_diff_prod;
    double coeff = 1.;
    double coeff_2 = 1.;
    double coeff_3 = 1.;
    if (nitsche_gap_diff_prod == 0) {
      // sum_prod *= 2;
      // diff_prod *= 2;
      coeff = 0.;
      coeff_2 = 2.;
      coeff_3 = -1999.;
    }

    // if (gap_gp > 0) {
    //   ++gap_gp;
    //   continue;
    // }
    // ++gap_gp;

    CHKERR getJac(col_data, gg);
    // double val = getVolume() * getGaussPts()(3, gg);
    // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
    //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
    // }
    FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1(
        &jac(3 * 0 + 0, 0), &jac(3 * 0 + 0, 1), &jac(3 * 0 + 0, 2),
        &jac(3 * 0 + 1, 0), &jac(3 * 0 + 1, 1), &jac(3 * 0 + 1, 2),
        &jac(3 * 0 + 2, 0), &jac(3 * 0 + 2, 1), &jac(3 * 0 + 2, 2),
        &jac(3 * 1 + 0, 0), &jac(3 * 1 + 0, 1), &jac(3 * 1 + 0, 2),
        &jac(3 * 1 + 1, 0), &jac(3 * 1 + 1, 1), &jac(3 * 1 + 1, 2),
        &jac(3 * 1 + 2, 0), &jac(3 * 1 + 2, 1), &jac(3 * 1 + 2, 2),
        &jac(3 * 2 + 0, 0), &jac(3 * 2 + 0, 1), &jac(3 * 2 + 0, 2),
        &jac(3 * 2 + 1, 0), &jac(3 * 2 + 1, 1), &jac(3 * 2 + 1, 2),
        &jac(3 * 2 + 2, 0), &jac(3 * 2 + 2, 1), &jac(3 * 2 + 2, 2));
    // for (int cc = 0; cc != nb_col / 3; cc++) {
    //   FTensor::Tensor1<double *, 3> diff_base_functions =
    //       row_data.getFTensor1DiffN<3>(gg, 0);
    //   FTensor::Tensor2<double *, 3, 3> lhs(
    //       &k(0, 3 * cc + 0), &k(0, 3 * cc + 1), &k(0, 3 * cc + 2),
    //       &k(1, 3 * cc + 0), &k(1, 3 * cc + 1), &k(1, 3 * cc + 2),
    //       &k(2, 3 * cc + 0), &k(2, 3 * cc + 1), &k(2, 3 * cc + 2), 3 *
    //       nb_col);
    //   for (int rr = 0; rr != nb_row / 3; rr++) {
    //     lhs(i, j) += val * t3_1(i, m, j) * diff_base_functions(m);
    //     ++diff_base_functions;
    //     ++lhs;
    //   }
    //   ++t3_1;
    // }

    FTensor::Tensor0<double *> t_base_slave_col(&col_data.getN()(gg, 0));
    // half for the average stress story
    const double val_m = 0.5 * t_w * area_s;
    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
      FTensor::Tensor0<double *> t_base_slave_row(&row_data.getN()(gg, 0));

      CHKERR getJacRow(row_data, gg);
      // cerr << "2qwe qwe\n";
      // double val = getVolume() * getGaussPts()(3, gg);
      // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
      //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
      // }
      FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1_row(
          &jac_row(3 * 0 + 0, 0), &jac_row(3 * 0 + 0, 1),
          &jac_row(3 * 0 + 0, 2), &jac_row(3 * 0 + 1, 0),
          &jac_row(3 * 0 + 1, 1), &jac_row(3 * 0 + 1, 2),
          &jac_row(3 * 0 + 2, 0), &jac_row(3 * 0 + 2, 1),
          &jac_row(3 * 0 + 2, 2), &jac_row(3 * 1 + 0, 0),
          &jac_row(3 * 1 + 0, 1), &jac_row(3 * 1 + 0, 2),
          &jac_row(3 * 1 + 1, 0), &jac_row(3 * 1 + 1, 1),
          &jac_row(3 * 1 + 1, 2), &jac_row(3 * 1 + 2, 0),
          &jac_row(3 * 1 + 2, 1), &jac_row(3 * 1 + 2, 2),
          &jac_row(3 * 2 + 0, 0), &jac_row(3 * 2 + 0, 1),
          &jac_row(3 * 2 + 0, 2), &jac_row(3 * 2 + 1, 0),
          &jac_row(3 * 2 + 1, 1), &jac_row(3 * 2 + 1, 2),
          &jac_row(3 * 2 + 2, 0), &jac_row(3 * 2 + 2, 1),
          &jac_row(3 * 2 + 2, 2));

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

        auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

        // t_assemble_m(m, k) -= val_m * t3_1(i, j, k) * const_unit_n(i) *
        //                       const_unit_n(j) * const_unit_n(m) *
        //                       t_base_slave_row; // base not needed

        // t_assemble_m(m, k) -= val_m * t3_1(i, j, k) * const_unit_n(i) *
        //                       const_unit_n(j) * const_unit_n(m) *
        //                       t_base_slave_row; // base not needed

        t_assemble_m(m, k) -= 2. * coeff_2 * val_m * omegaVal * t3_1(i, j, k) *
                              const_unit_n(i) * const_unit_n(j) *
                              const_unit_n(m) *
                              t_base_slave_row; // base not needed

        // cerr << " sum_prod  " << sum_prod << " diff_prod  " << diff_prod
        //      << "\n";

        t_assemble_m(m, k) +=
            val_m *
            (coeff_2 * sum_prod * omegaVal * t3_1(i, j, k) * const_unit_n(i) *
                 const_unit_n(j) +
             coeff_2 * diff_prod * cN * t_base_slave_col * const_unit_n(k)) *
            (const_unit_n(m) * t_base_slave_row -
             coeff * omegaVal * thetaSVal * t3_1_row(i, j, m) *
                 const_unit_n(i) * const_unit_n(j) / cN); // base not needed

        MatrixDouble t_tan;
        t_tan.resize(3, 3, false);

        auto t_assemble_tan = get_tensor_from_mat(t_tan, 0, 0);
        t_assemble_tan(i, j) = const_unit_n(i) * const_unit_n(j);
        t_assemble_tan(0, 0) -= 1.;
        t_assemble_tan(1, 1) -= 1.;
        t_assemble_tan(2, 2) -= 1.;

        t_assemble_m(i, f) -=
            2. * val_m * omegaVal * thetaSVal *
            (t_assemble_tan(j, l) * t3_1(l, k, f) * const_unit_n(k)) *
            (t3_1_row(i, m, l) * const_unit_n(m) * t_assemble_tan(l, j)) / cN;

        // if(gap_gp > 0)
        // t_assemble_m(k, m) -= val_m * t3_1_row(i, j, k) *
        //                       const_unit_n(i) *
        //                       const_unit_n(j) * const_unit_n(m) *
        //                       t_base_slave_col; // base not needed

        ++t3_1_row;
        ++t_base_slave_row;
      }
      ++t_base_slave_col;
      ++t3_1;
    }
    ++t_w;
    ++gap_gp;
    ++nitsche_gap_diff_prod;
    }
    // CHKERR aSemble(side, type, data);
    Mat aij;
    if (Aij == PETSC_NULL) {
      aij = getFEMethod()->snes_B;
    } else {
      aij = Aij;
    }

    CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
                        &col_data.getIndices()[0], &*NN.data().begin(),
                        ADD_VALUES);

    MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpStressDerivativeGapMasterSlave_dx::getJac(
    DataForcesAndSourcesCore::EntData &col_data, int gg) {
  return get_jac_contact<0>(col_data, gg, commonData.jacStress[gg], jac);
}

MoFEMErrorCode
SimpleContactProblem::OpStressDerivativeGapMasterSlave_dx::getJacRow(
    DataForcesAndSourcesCore::EntData &row_data, int gg) {
  return get_jac_contact<0>(row_data, gg, commonDataMaster.jacStress[gg],
                            jac_row);
}

// MoFEMErrorCode SimpleContactProblem::OpStressDerivativeGapMasterSlave_dx::doWork(
//     int row_side, int col_side, EntityType row_type, EntityType col_type,
//     DataForcesAndSourcesCore::EntData &row_data,
//     DataForcesAndSourcesCore::EntData &col_data) {


MoFEMErrorCode SimpleContactProblem::OpStressDerivativeGapMasterSlave_dx::doWork(
    int col_side, EntityType col_type,
    DataForcesAndSourcesCore::EntData &col_data) {

  MoFEMFunctionBegin;

  // const int nb_row = row_data.getIndices().size();
  // if (!nb_row)
  //   MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);

  // boost::shared_ptr<std::vector<DataForcesAndSourcesCore::EntData *>>
  //     masterRowData;
  for (std::vector<DataForcesAndSourcesCore::EntData *>::iterator row_data =
           commonDataSimpleContact->masterRowData->begin();
       row_data != commonDataSimpleContact->masterRowData->end(); ++row_data) {

    const int nb_row = (*row_data)->getIndices().size();
    if (!nb_row)
      continue;

    const int nb_gauss_pts = (*row_data)->getN().size1();

    int nb_base_fun_row = (*row_data)->getFieldData().size() / 3;
    int nb_base_fun_col = col_data.getFieldData().size() / 3;

    // cerr << "OpStressDerivativeGapMasterSlave_dx "
    //      << " Row " << nb_row << " Col " << nb_col << " indices "
    //      << col_data.getIndices() << "\n";

    // cerr << " base funs row " << nb_base_fun_row << " base funs  cols "
    //      << nb_base_fun_col << " indices rows " << (*row_data)->getIndices()
    //      << "\n";

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

    NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
    NN.clear();

    NNT.resize(3 * nb_base_fun_col, 3 * nb_base_fun_row, false);
    NNT.clear();

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'m', 3> m;
    FTensor::Index<'k', 3> k;
    FTensor::Index<'f', 3> f;
    FTensor::Index<'l', 3> l;

    const double area_s =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto const_unit_n =
        get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

    auto const_unit_n_master =
        get_tensor_vec(commonDataSimpleContact->normalVectorMasterPtr.get()[0]);

    auto t_w = getFTensor0IntegrationWeight();

    // k.resize(nb_row, nb_row, false);
    // k.clear();
    jac.resize(9, nb_col, false);
    jac_row.resize(9, nb_row, false);

    auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    auto nitsche_gap_diff_prod =
        getFTensor0FromVec(*commonDataSimpleContact->nitscheGapDiffProductPtr);

    auto tilde_c_fun =
        getFTensor0FromVec(*commonDataSimpleContact->tildeCFunNitschePtr);

    auto proj_normal_stress_master =
        getFTensor0FromVec(*commonDataSimpleContact->projNormalStressAtMaster);
    auto proj_normal_stress_slave =
        getFTensor0FromVec(*commonDataSimpleContact->projNormalStressAtSlave);
    for (int gg = 0; gg != nb_gauss_pts; gg++) {

      double sum_prod = 1. + nitsche_gap_diff_prod;
      double diff_prod = 1. - nitsche_gap_diff_prod;
      double coef = 1. ;
      double coef_2 = 1.;
      if (nitsche_gap_diff_prod == 0) {
        // sum_prod *= 2;
        // diff_prod *= 2;
        coef = 2.;
        coef_2 = 0.;
      }

      // if (gap_gp > 0) {
      //   ++gap_gp;
      //   continue;
      // }
      // ++gap_gp;

      CHKERR getJac(col_data, gg);
      // double val = getVolume() * getGaussPts()(3, gg);
      // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
      //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
      // }
      // FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1(
      //     &jac(3 * 0 + 0, 0), &jac(3 * 0 + 0, 1), &jac(3 * 0 + 0, 2),
      //     &jac(3 * 0 + 1, 0), &jac(3 * 0 + 1, 1), &jac(3 * 0 + 1, 2),
      //     &jac(3 * 0 + 2, 0), &jac(3 * 0 + 2, 1), &jac(3 * 0 + 2, 2),
      //     &jac(3 * 1 + 0, 0), &jac(3 * 1 + 0, 1), &jac(3 * 1 + 0, 2),
      //     &jac(3 * 1 + 1, 0), &jac(3 * 1 + 1, 1), &jac(3 * 1 + 1, 2),
      //     &jac(3 * 1 + 2, 0), &jac(3 * 1 + 2, 1), &jac(3 * 1 + 2, 2),
      //     &jac(3 * 2 + 0, 0), &jac(3 * 2 + 0, 1), &jac(3 * 2 + 0, 2),
      //     &jac(3 * 2 + 1, 0), &jac(3 * 2 + 1, 1), &jac(3 * 2 + 1, 2),
      //     &jac(3 * 2 + 2, 0), &jac(3 * 2 + 2, 1), &jac(3 * 2 + 2, 2));

      FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1(
          &jac(3 * 0 + 0, 0), &jac(3 * 0 + 0, 1), &jac(3 * 0 + 0, 2),
          &jac(3 * 0 + 1, 0), &jac(3 * 0 + 1, 1), &jac(3 * 0 + 1, 2),
          &jac(3 * 0 + 2, 0), &jac(3 * 0 + 2, 1), &jac(3 * 0 + 2, 2),
          &jac(3 * 1 + 0, 0), &jac(3 * 1 + 0, 1), &jac(3 * 1 + 0, 2),
          &jac(3 * 1 + 1, 0), &jac(3 * 1 + 1, 1), &jac(3 * 1 + 1, 2),
          &jac(3 * 1 + 2, 0), &jac(3 * 1 + 2, 1), &jac(3 * 1 + 2, 2),
          &jac(3 * 2 + 0, 0), &jac(3 * 2 + 0, 1), &jac(3 * 2 + 0, 2),
          &jac(3 * 2 + 1, 0), &jac(3 * 2 + 1, 1), &jac(3 * 2 + 1, 2),
          &jac(3 * 2 + 2, 0), &jac(3 * 2 + 2, 1), &jac(3 * 2 + 2, 2));

      // for (int cc = 0; cc != nb_col / 3; cc++) {
      //   FTensor::Tensor1<double *, 3> diff_base_functions =
      //       (*row_data)->getFTensor1DiffN<3>(gg, 0);
      //   FTensor::Tensor2<double *, 3, 3> lhs(
      //       &k(0, 3 * cc + 0), &k(0, 3 * cc + 1), &k(0, 3 * cc + 2),
      //       &k(1, 3 * cc + 0), &k(1, 3 * cc + 1), &k(1, 3 * cc + 2),
      //       &k(2, 3 * cc + 0), &k(2, 3 * cc + 1), &k(2, 3 * cc + 2), 3 *
      //       nb_col);
      //   for (int rr = 0; rr != nb_row / 3; rr++) {
      //     lhs(i, j) += val * t3_1(i, m, j) * diff_base_functions(m);
      //     ++diff_base_functions;
      //     ++lhs;
      //   }
      //   ++t3_1;
      // }

      FTensor::Tensor0<double *> t_base_slave_col(&col_data.getN()(gg, 0));
      // half for the average stress story
      const double val_m = 0.5 * t_w   * area_s;
      for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
        FTensor::Tensor0<double *> t_base_master_row(&(*row_data)->getN()(gg, 0));

        // CHKERR getJacRow(row_data, gg);
        // // cerr << "2qwe qwe\n";
        // // double val = getVolume() * getGaussPts()(3, gg);
        // // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
        // //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
        // // }
        // FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1_row(
        //     &jac_row(3 * 0 + 0, 0), &jac_row(3 * 0 + 0, 1),
        //     &jac_row(3 * 0 + 0, 2), &jac_row(3 * 0 + 1, 0),
        //     &jac_row(3 * 0 + 1, 1), &jac_row(3 * 0 + 1, 2),
        //     &jac_row(3 * 0 + 2, 0), &jac_row(3 * 0 + 2, 1),
        //     &jac_row(3 * 0 + 2, 2), &jac_row(3 * 1 + 0, 0),
        //     &jac_row(3 * 1 + 0, 1), &jac_row(3 * 1 + 0, 2),
        //     &jac_row(3 * 1 + 1, 0), &jac_row(3 * 1 + 1, 1),
        //     &jac_row(3 * 1 + 1, 2), &jac_row(3 * 1 + 2, 0),
        //     &jac_row(3 * 1 + 2, 1), &jac_row(3 * 1 + 2, 2),
        //     &jac_row(3 * 2 + 0, 0), &jac_row(3 * 2 + 0, 1),
        //     &jac_row(3 * 2 + 0, 2), &jac_row(3 * 2 + 1, 0),
        //     &jac_row(3 * 2 + 1, 1), &jac_row(3 * 2 + 1, 2),
        //     &jac_row(3 * 2 + 2, 0), &jac_row(3 * 2 + 2, 1),
        //     &jac_row(3 * 2 + 2, 2));

        CHKERR getJacRow(**row_data, gg);

        // double val = getVolume() * getGaussPts()(3, gg);
        // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
        //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
        // }
        // cerr << " Row: gg " << gg << " rows " << nb_base_fun_row << "\n";
        FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1_row(
            &jac_row(3 * 0 + 0, 0), &jac_row(3 * 0 + 0, 1),
            &jac_row(3 * 0 + 0, 2), &jac_row(3 * 0 + 1, 0),
            &jac_row(3 * 0 + 1, 1), &jac_row(3 * 0 + 1, 2),
            &jac_row(3 * 0 + 2, 0), &jac_row(3 * 0 + 2, 1),
            &jac_row(3 * 0 + 2, 2), &jac_row(3 * 1 + 0, 0),
            &jac_row(3 * 1 + 0, 1), &jac_row(3 * 1 + 0, 2),
            &jac_row(3 * 1 + 1, 0), &jac_row(3 * 1 + 1, 1),
            &jac_row(3 * 1 + 1, 2), &jac_row(3 * 1 + 2, 0),
            &jac_row(3 * 1 + 2, 1), &jac_row(3 * 1 + 2, 2),
            &jac_row(3 * 2 + 0, 0), &jac_row(3 * 2 + 0, 1),
            &jac_row(3 * 2 + 0, 2), &jac_row(3 * 2 + 1, 0),
            &jac_row(3 * 2 + 1, 1), &jac_row(3 * 2 + 1, 2),
            &jac_row(3 * 2 + 2, 0), &jac_row(3 * 2 + 2, 1),
            &jac_row(3 * 2 + 2, 2));

        for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

          auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

          auto t_assemble_s = get_tensor_from_mat(NNT, 3 * bbc, 3 * bbr);

          // t_assemble_m(m, k) -= val_m * t3_1(i, j, k) * const_unit_n(i) *
          //                       const_unit_n(j) * const_unit_n(m) *
          //                       t_base_master_row; // base not needed

          // cerr << " C FUN " << tilde_c_fun << " GAP " << gap_gp << " MASTER "
          //      << proj_normal_stress_master << " SLAVE "
          //      << proj_normal_stress_slave << " sum_prod " << sum_prod
          //      << " diff_prod " << diff_prod << "\n";

          // cerr << "tilde_c_fun" << tilde_c_fun << " gap_gp " << gap_gp
          //      << " nitsche_gap_diff_prod " << nitsche_gap_diff_prod
          //      << " proj_normal_stress_master " << proj_normal_stress_master
          //      << " proj_normal_stress_slave " << proj_normal_stress_slave
          //      << "\n";

          // if (bbr == 0)
          //   continue;

          t_assemble_m(m, k) += 2. * coef * val_m * omegaVal * t3_1(i, j, k) *
                                const_unit_n(i) * const_unit_n(j) *
                                const_unit_n(m) *
                                t_base_master_row; // base not needed

          t_assemble_m(m, k) +=
              val_m *
              (sum_prod * coef * omegaVal * t3_1(i, j, k) * const_unit_n(i) *
                   const_unit_n(j) +
                coef * diff_prod * cN * t_base_slave_col * const_unit_n(k)) *
              (-const_unit_n(m) * t_base_master_row +
               (1. - omegaVal) * thetaSVal * t3_1_row(i, j, m) *
                   const_unit_n_master(i) * const_unit_n(j) /
                   cN); // base not needed

          MatrixDouble t_tan;
          t_tan.resize(3, 3, false);

          auto t_assemble_tan = get_tensor_from_mat(t_tan, 0, 0);
          t_assemble_tan(i, j) = const_unit_n(i) * const_unit_n(j);
          t_assemble_tan(0, 0) -= 1.;
          t_assemble_tan(1, 1) -= 1.;
          t_assemble_tan(2, 2) -= 1.;

          t_assemble_m(i, f) -=
              2. * val_m * omegaVal * thetaSVal *
              (t_assemble_tan(j, l) * t3_1(l, k, f) * const_unit_n(k)) *
              (t3_1_row(i, m, l) * const_unit_n(m) * t_assemble_tan(l, j)) / cN;

          // if (nitsche_gap_diff_prod == 0)
          // t_assemble_m(m, k) = 2. * t_assemble_m(m, k);

          // if (gap_gp > 0){
          //   t_assemble_m(k, m) -= val_m * t3_1_row(i, j, k) *
          //                         const_unit_n_master(i) *
          //                         const_unit_n_master(j) * const_unit_n(m) *
          //                         t_base_slave_col; // base not needed
          // }
          ++t3_1_row;
          ++t_base_master_row;
        }
        ++t3_1;
        ++t_base_slave_col;
      }
      ++t_w;
      ++gap_gp;
      ++nitsche_gap_diff_prod;
      ++tilde_c_fun;
      ++proj_normal_stress_master;
      ++proj_normal_stress_slave;
    }
    // CHKERR aSemble(side, type, data);
    Mat aij;
    if (Aij == PETSC_NULL) {
      aij = getFEMethod()->snes_B;
    } else {
      aij = Aij;
    }
   
    
    CHKERR MatSetValues(aij, nb_row, &(*row_data)->getIndices()[0], nb_col,
                        &col_data.getIndices()[0], &*NN.data().begin(),
                        ADD_VALUES);

    // noalias(NNT) = trans(NN);
    // CHKERR MatSetValues(aij, nb_col, &col_data.getIndices()[0], nb_row,
    //                     &(*row_data)->getIndices()[0], &*NNT.data().begin(),
    //                     ADD_VALUES);
}
    MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpStressDerivativeGapMasterSlave_dxCorrect::getJac(
    DataForcesAndSourcesCore::EntData &col_data, int gg) {
  return get_jac_contact<0>(col_data, gg, commonData.jacStress[gg], jac);
}

MoFEMErrorCode
SimpleContactProblem::OpStressDerivativeGapMasterSlave_dxCorrect::getJacRow(
    DataForcesAndSourcesCore::EntData &row_data, int gg) {
  return get_jac_contact<0>(row_data, gg, commonData.jacStress[gg], jac_row);
}

// MoFEMErrorCode
// SimpleContactProblem::OpStressDerivativeGapMasterSlave_dx::doWork(
//     int row_side, int col_side, EntityType row_type, EntityType col_type,
//     DataForcesAndSourcesCore::EntData &row_data,
//     DataForcesAndSourcesCore::EntData &col_data) {

MoFEMErrorCode SimpleContactProblem::OpStressDerivativeGapMasterSlave_dxCorrect::doWork
    (int row_side, int col_side, EntityType row_type, EntityType col_type,
           DataForcesAndSourcesCore::EntData &row_data,
           DataForcesAndSourcesCore::EntData &col_data)
{

  MoFEMFunctionBegin;

  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);

    cerr << " row " << row_data.getIndices() << "\n";
    cerr << " col " << col_data.getIndices() << "\n";

    const int nb_gauss_pts = row_data.getN().size1();
    // commonDataMaster
    int nb_base_fun_row = row_data.getFieldData().size() / 3;
    int nb_base_fun_col = col_data.getFieldData().size() / 3;

    // cerr << "OpStressDerivativeGapMasterSlave_dx "
    //      << " Row " << nb_row << " Col " << nb_col << " indices "
    //      << col_data.getIndices() << "\n";

    // cerr << " base funs row " << nb_base_fun_row << " base funs  cols "
    //      << nb_base_fun_col << " indices rows " << nb_row.getIndices()
    //      << "\n";

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

    NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
    NN.clear();

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'m', 3> m;
    FTensor::Index<'k', 3> k;

    const double area_s =
        commonDataSimpleContact->areaSlave; // same area in master and slave

    auto const_unit_n =
        get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

    auto const_unit_n_master =
        get_tensor_vec(commonDataSimpleContact->normalVectorMasterPtr.get()[0]);

    auto t_w = getFTensor0IntegrationWeightMaster();

    // k.resize(nb_row, nb_row, false);
    // k.clear();
    jac.resize(9, nb_col, false);
    jac_row.resize(9, nb_row, false);

    auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    for (int gg = 0; gg != nb_gauss_pts; gg++) {

      // if (gap_gp > 0) {
      //   ++gap_gp;
      //   continue;
      // }
      // ++gap_gp;

      CHKERR getJac(col_data, gg);
      // double val = getVolume() * getGaussPts()(3, gg);
      // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
      //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
      // }
      FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1(
          &jac(3 * 0 + 0, 0), &jac(3 * 0 + 0, 1), &jac(3 * 0 + 0, 2),
          &jac(3 * 0 + 1, 0), &jac(3 * 0 + 1, 1), &jac(3 * 0 + 1, 2),
          &jac(3 * 0 + 2, 0), &jac(3 * 0 + 2, 1), &jac(3 * 0 + 2, 2),
          &jac(3 * 1 + 0, 0), &jac(3 * 1 + 0, 1), &jac(3 * 1 + 0, 2),
          &jac(3 * 1 + 1, 0), &jac(3 * 1 + 1, 1), &jac(3 * 1 + 1, 2),
          &jac(3 * 1 + 2, 0), &jac(3 * 1 + 2, 1), &jac(3 * 1 + 2, 2),
          &jac(3 * 2 + 0, 0), &jac(3 * 2 + 0, 1), &jac(3 * 2 + 0, 2),
          &jac(3 * 2 + 1, 0), &jac(3 * 2 + 1, 1), &jac(3 * 2 + 1, 2),
          &jac(3 * 2 + 2, 0), &jac(3 * 2 + 2, 1), &jac(3 * 2 + 2, 2));
      // for (int cc = 0; cc != nb_col / 3; cc++) {
      //   FTensor::Tensor1<double *, 3> diff_base_functions =
      //       nb_row.getFTensor1DiffN<3>(gg, 0);
      //   FTensor::Tensor2<double *, 3, 3> lhs(
      //       &k(0, 3 * cc + 0), &k(0, 3 * cc + 1), &k(0, 3 * cc + 2),
      //       &k(1, 3 * cc + 0), &k(1, 3 * cc + 1), &k(1, 3 * cc + 2),
      //       &k(2, 3 * cc + 0), &k(2, 3 * cc + 1), &k(2, 3 * cc + 2), 3 *
      //       nb_col);
      //   for (int rr = 0; rr != nb_row / 3; rr++) {
      //     lhs(i, j) += val * t3_1(i, m, j) * diff_base_functions(m);
      //     ++diff_base_functions;
      //     ++lhs;
      //   }
      //   ++t3_1;
      // }

      FTensor::Tensor0<double *> t_base_slave_col(&col_data.getN()(gg, 0));
      // half for the average stress story
      const double val_m = 0.5 * t_w * area_s;
      for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
        FTensor::Tensor0<double *> t_base_master_row(
            &row_data.getN()(gg, 0));

        CHKERR getJacRow(row_data, gg);

        // double val = getVolume() * getGaussPts()(3, gg);
        // if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
        //   val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
        // }
        FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3> t3_1_row(
            &jac_row(3 * 0 + 0, 0), &jac_row(3 * 0 + 0, 1),
            &jac_row(3 * 0 + 0, 2), &jac_row(3 * 0 + 1, 0),
            &jac_row(3 * 0 + 1, 1), &jac_row(3 * 0 + 1, 2),
            &jac_row(3 * 0 + 2, 0), &jac_row(3 * 0 + 2, 1),
            &jac_row(3 * 0 + 2, 2), &jac_row(3 * 1 + 0, 0),
            &jac_row(3 * 1 + 0, 1), &jac_row(3 * 1 + 0, 2),
            &jac_row(3 * 1 + 1, 0), &jac_row(3 * 1 + 1, 1),
            &jac_row(3 * 1 + 1, 2), &jac_row(3 * 1 + 2, 0),
            &jac_row(3 * 1 + 2, 1), &jac_row(3 * 1 + 2, 2),
            &jac_row(3 * 2 + 0, 0), &jac_row(3 * 2 + 0, 1),
            &jac_row(3 * 2 + 0, 2), &jac_row(3 * 2 + 1, 0),
            &jac_row(3 * 2 + 1, 1), &jac_row(3 * 2 + 1, 2),
            &jac_row(3 * 2 + 2, 0), &jac_row(3 * 2 + 2, 1),
            &jac_row(3 * 2 + 2, 2));

        for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

          auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

          // t_assemble_m(m, k) += val_m * t3_1(i, j, k) * const_unit_n_master(i) *
          //                       const_unit_n_master(j) * const_unit_n(m) *
          //                       t_base_master_row; // base not needed

          t_assemble_m(k, m) += val_m * t3_1_row(i, j, k) *
                                const_unit_n_master(i) *
                                const_unit_n_master(j) * const_unit_n(m) *
                                t_base_slave_col; // base not needed
          ++t3_1_row;
          ++t_base_master_row;
        }
        ++t3_1;
        ++t_base_slave_col;
      }
      ++t_w;
    }
    // CHKERR aSemble(side, type, data);
    Mat aij;
    if (Aij == PETSC_NULL) {
      aij = getFEMethod()->snes_B;
    } else {
      aij = Aij;
    }

    CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
                        &col_data.getIndices()[0], &*NN.data().begin(),
                        ADD_VALUES);
  
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalFReConSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();
  int nb_base_fun_col = data.getFieldData().size() / 3;

  vec_f.resize(3 * nb_base_fun_col,
               false); // the last false in ublas
                       // resize will destroy (not
                       // preserved) the old
                       // values
  vec_f.clear();

  const double area_s =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto get_tensor_vec = [](VectorDouble &n, const int r) {
    return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
  };

  FTensor::Index<'i', 3> i;
  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0], 0);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    double val_s = getGaussPtsSlave()(2, gg) * area_s;

    FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

      const double s = val_s * t_base_slave * lagrange_slave;

      auto t_assemble_s = get_tensor_vec(vec_f, 3 * bbc);

      t_assemble_s(i) += s * const_unit_n(i);
      ++t_base_slave;
    }

    ++lagrange_slave;
  } // for gauss points

  const int nb_col = data.getIndices().size();

  Vec f;
  if (F == PETSC_NULL) {
    f = getFEMethod()->snes_f;
  } else {
    f = F;
  }

  CHKERR VecSetValues(f, nb_col, &data.getIndices()[0], &vec_f[0], ADD_VALUES);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalFReConSlaveALE::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();
  int nb_base_fun_col = data.getFieldData().size() / 3;

  vec_f.resize(3 * nb_base_fun_col,
               false); // the last false in ublas
                       // resize will destroy (not
                       // preserved) the old
                       // values
  vec_f.clear();

  auto get_tensor_vec = [](VectorDouble &n, const int r) {
    return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
  };

  FTensor::Index<'i', 3> i;
  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto length_normal =
      getFTensor0FromVec(*commonDataSimpleContact->normalSlaveLengthALEPtr);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    auto normal_at_gp =
        get_tensor_vec(commonDataSimpleContact->normalSlaveALE[gg], 0);

    double val_s = getGaussPtsSlave()(2, gg) * length_normal * 0.5;

    FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

      const double s = val_s * t_base_slave * lagrange_slave;

      auto t_assemble_s = get_tensor_vec(vec_f, 3 * bbc);

      t_assemble_s(i) += s * normal_at_gp(i) / length_normal;
      ++t_base_slave;
    }

    ++lagrange_slave;
    ++length_normal;
  } // for gauss points

  const int nb_col = data.getIndices().size();

  Vec f;
  if (F == PETSC_NULL) {
    f = getFEMethod()->snes_f;
  } else {
    f = F;
  }

  CHKERR VecSetValues(f, nb_col, &data.getIndices()[0], &vec_f[0], ADD_VALUES);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalFReConMasterALE::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();
  int nb_base_fun_col = data.getFieldData().size() / 3;

  vec_f.resize(3 * nb_base_fun_col,
               false); // the last false in ublas
                       // resize will destroy (not
                       // preserved) the old
                       // values
  vec_f.clear();

  auto get_tensor_vec = [](VectorDouble &n, const int r) {
    return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
  };

  FTensor::Index<'i', 3> i;
  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto length_normal =
      getFTensor0FromVec(*commonDataSimpleContact->normalSlaveLengthALEPtr);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    auto normal_at_gp =
        get_tensor_vec(commonDataSimpleContact->normalSlaveALE[gg], 0);

    double val_s = getGaussPtsSlave()(2, gg) * length_normal * 0.5;

    FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

      const double s = val_s * t_base_master * lagrange_slave;

      auto t_assemble_s = get_tensor_vec(vec_f, 3 * bbc);

      t_assemble_s(i) -= s * normal_at_gp(i) / length_normal;
      ++t_base_master;
    }

    ++lagrange_slave;
    ++length_normal;
  } // for gauss points

  const int nb_col = data.getIndices().size();

  Vec f;
  if (F == PETSC_NULL) {
    f = getFEMethod()->snes_f;
  } else {
    f = F;
  }

  CHKERR VecSetValues(f, nb_col, &data.getIndices()[0], &vec_f[0], ADD_VALUES);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpLoopMasterForSide::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBTRI)
    MoFEMFunctionReturnHot(0);

  const EntityHandle tri_master = getSideEntity(3, MBTRI);
  CHKERR loopSideVolumes(sideFeName, *sideFe, 3, tri_master);

  MoFEMFunctionReturn(0);
}



MoFEMErrorCode SimpleContactProblem::OpLoopSlaveForSide::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBTRI)
    MoFEMFunctionReturnHot(0);

  const EntityHandle tri_slave = getSideEntity(4, type);
  CHKERR loopSideVolumes(sideFeName, *sideFe, 3, tri_slave);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpLoopSlaveForSideFace::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBTRI)
    MoFEMFunctionReturnHot(0);

  const EntityHandle tri_slave = getSideEntity(4, type);
  CHKERR loopSideFaces(sideFeName, *faceSideFe, 2, tri_slave);

  MoFEMFunctionReturn(0);
}

// MoFEMErrorCode SimpleContactProblem::OpCalMatForcesALEMaster::doWork(
//     int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
//   MoFEMFunctionBegin;

//   if (data.getIndices().size() == 0)
//     MoFEMFunctionReturnHot(0);

//   const int nb_gauss_pts = data.getN().size1();
//   int nb_base_fun_col = data.getFieldData().size() / 3;

//   vec_f.resize(3 * nb_base_fun_col,
//                false); // the last false in ublas
//                        // resize will destroy (not
//                        // preserved) the old
//                        // values
//   vec_f.clear();

//   auto get_tensor_vec = [](VectorDouble &n, const int r) {
//     return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
//   };

//   FTensor::Index<'i', 3> i;
//   FTensor::Index<'j', 3> j;
//   auto lagrange_slave =
//       getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

//   auto t_F = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->FMat);

//   for (int gg = 0; gg != nb_gauss_pts; ++gg) {
//     auto normal_at_gp =
//         get_tensor_vec(commonDataSimpleContact->normalMasterALE[gg], 0);

//     double val_s = getGaussPtsMaster()(2, gg) * 0.5;

//     FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

//     for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

//       const double s = val_s * t_base_master * lagrange_slave;

//       auto t_assemble_s = get_tensor_vec(vec_f, 3 * bbc);

//       t_assemble_s(i) -= s * t_F(j, i) * normal_at_gp(j);
//       ++t_base_master;
//     }
//     ++t_F;
//     ++lagrange_slave;
//   } // for gauss points

//   const int nb_col = data.getIndices().size();

//   Vec f;
//   if (F == PETSC_NULL) {
//     f = getFEMethod()->snes_f;
//   } else {
//     f = F;
//   }

//   CHKERR VecSetValues(f, nb_col, &data.getIndices()[0], &vec_f[0], ADD_VALUES);
//   MoFEMFunctionReturn(0);
// }

MoFEMErrorCode SimpleContactProblem::OpCalMatForcesALEMaster::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &row_data) {

  MoFEMFunctionBegin;

  // get number of dofs on row
  nbRows = row_data.getIndices().size();
  // if no dofs on row, exit that work, nothing to do here
  if (!nbRows)
    MoFEMFunctionReturnHot(0);

  vec_f.resize(nbRows, false);
  vec_f.clear();

  // get number of integration points
  nbIntegrationPts = getGaussPtsMaster().size2();

  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data);

  // assemble local matrix
  CHKERR aSsemble(row_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalMatForcesALEMaster::iNtegrate(
    DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
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

  for (int gg = 0; gg != nbIntegrationPts; ++gg) {
    auto normal_at_gp =
        get_tensor_vec(commonDataSimpleContact->normalMasterALE[gg], 0);

    double val_s = getGaussPtsMaster()(2, gg) * 0.5;

    FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

      const double s = val_s * t_base_master * lagrange_slave;

      auto t_assemble_s = get_tensor_vec(vec_f, 3 * bbc);

      t_assemble_s(i) -= s * t_F(j, i) * normal_at_gp(j);

      ++t_base_master;
    }
    ++t_F;
    ++lagrange_slave;
  } // for gauss points

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalMatForcesALEMaster::aSsemble(
    DataForcesAndSourcesCore::EntData &row_data) {
  MoFEMFunctionBegin;

  // get pointer to first global index on row
  const int *row_indices = &*row_data.getIndices().data().begin();

  auto &data = *commonDataSimpleContact;
  if (!data.forcesOnlyOnEntitiesRow.empty()) {
    rowIndices.resize(nbRows, false);
    noalias(rowIndices) = row_data.getIndices();
    row_indices = &rowIndices[0];
    VectorDofs &dofs = row_data.getFieldDofs();
    VectorDofs::iterator dit = dofs.begin();
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (data.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
          data.forcesOnlyOnEntitiesRow.end()) {
        rowIndices[ii] = -1;
      }
    }
  }

  auto get_f = [&]() {
    Vec my_f;
    if (F == PETSC_NULL) {
      switch (getFEMethod()->ts_ctx) {
      case FEMethod::CTX_TSSETIFUNCTION: {
        const_cast<FEMethod *>(getFEMethod())->snes_ctx =
            FEMethod::CTX_SNESSETFUNCTION;
        const_cast<FEMethod *>(getFEMethod())->snes_x = getFEMethod()->ts_u;
        const_cast<FEMethod *>(getFEMethod())->snes_f = getFEMethod()->ts_F;
        break;
      }
      default:
        break;
      }
      my_f = getFEMethod()->snes_f;
    } else {
      my_f = F;
    }
    return my_f;
  };

  auto vec_assemble = [&](Vec my_f) {
    MoFEMFunctionBegin;
    CHKERR VecSetOption(my_f, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
    CHKERR VecSetValues(my_f, nbRows, row_indices, &*vec_f.data().begin(),
                        ADD_VALUES);
    MoFEMFunctionReturn(0);
  };

  // assemble local matrix
  CHKERR vec_assemble(get_f());

  MoFEMFunctionReturn(0);
}

// MoFEMErrorCode SimpleContactProblem::OpCalMatForcesALESlave::doWork(
//     int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
//   MoFEMFunctionBegin;

//   if (data.getIndices().size() == 0)
//     MoFEMFunctionReturnHot(0);

//   const int nb_gauss_pts = data.getN().size1();
//   int nb_base_fun_col = data.getFieldData().size() / 3;

//   vec_f.resize(3 * nb_base_fun_col,
//                false); // the last false in ublas
//                        // resize will destroy (not
//                        // preserved) the old
//                        // values
//   vec_f.clear();

//   auto get_tensor_vec = [](VectorDouble &n, const int r) {
//     return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
//   };

//   FTensor::Index<'i', 3> i;
//   FTensor::Index<'j', 3> j;
//   auto lagrange_slave =
//       getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

//   auto t_F = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->FMat);

//   for (int gg = 0; gg != nb_gauss_pts; ++gg) {
//     auto normal_at_gp =
//         get_tensor_vec(commonDataSimpleContact->normalSlaveALE[gg], 0);

//     double val_s = getGaussPtsSlave()(2, gg) * 0.5;

//     FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

//     for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

//       const double s = val_s * t_base_master * lagrange_slave;

//       auto t_assemble_s = get_tensor_vec(vec_f, 3 * bbc);

//       t_assemble_s(i) -= s * t_F(j, i) * normal_at_gp(j);
//       ++t_base_master;
//     }
//     ++t_F;
//     ++lagrange_slave;
//   } // for gauss points

//   const int nb_col = data.getIndices().size();

//   Vec f;
//   if (F == PETSC_NULL) {
//     f = getFEMethod()->snes_f;
//   } else {
//     f = F;
//   }

//   CHKERR VecSetValues(f, nb_col, &data.getIndices()[0], &vec_f[0], ADD_VALUES);
//   MoFEMFunctionReturn(0);
// }

MoFEMErrorCode SimpleContactProblem::OpCalMatForcesALESlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &row_data) {

  MoFEMFunctionBegin;

  // get number of dofs on row
  nbRows = row_data.getIndices().size();
  // if no dofs on row, exit that work, nothing to do here
  if (!nbRows)
    MoFEMFunctionReturnHot(0);

  vec_f.resize(nbRows, false);
  vec_f.clear();

  // get number of integration points
  nbIntegrationPts = getGaussPtsSlave().size2();

  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data);

  // assemble local matrix
  CHKERR aSsemble(row_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalMatForcesALESlave::iNtegrate(
    DataForcesAndSourcesCore::EntData &data) {
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
  for (int gg = 0; gg != nbIntegrationPts; ++gg) {
    auto normal_at_gp =
        get_tensor_vec(commonDataSimpleContact->normalSlaveALE[gg], 0);

    double val_s = getGaussPtsSlave()(2, gg) * 0.5;
    FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));
  

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
      const double s = val_s * t_base_master * lagrange_slave;
      auto t_assemble_s = get_tensor_vec(vec_f, 3 * bbc);

      t_assemble_s(i) -= s * t_F(j, i) * normal_at_gp(j);
      ++t_base_master;
    }
    ++t_F;
    ++lagrange_slave;
  } // for gauss points

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalMatForcesALESlave::aSsemble(
    DataForcesAndSourcesCore::EntData &row_data) {
  MoFEMFunctionBegin;

  // get pointer to first global index on row
  const int *row_indices = &*row_data.getIndices().data().begin();

  auto &data = *commonDataSimpleContact;
  if (!data.forcesOnlyOnEntitiesRow.empty()) {
    rowIndices.resize(nbRows, false);
    noalias(rowIndices) = row_data.getIndices();
    row_indices = &rowIndices[0];
    VectorDofs &dofs = row_data.getFieldDofs();
    VectorDofs::iterator dit = dofs.begin();
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (data.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
          data.forcesOnlyOnEntitiesRow.end()) {
        rowIndices[ii] = -1;
      }
    }
  }

  auto get_f = [&]() {
    Vec my_f;
    if (F == PETSC_NULL) {
      switch (getFEMethod()->ts_ctx) {
      case FEMethod::CTX_TSSETIFUNCTION: {
        const_cast<FEMethod *>(getFEMethod())->snes_ctx =
            FEMethod::CTX_SNESSETFUNCTION;
        const_cast<FEMethod *>(getFEMethod())->snes_x = getFEMethod()->ts_u;
        const_cast<FEMethod *>(getFEMethod())->snes_f = getFEMethod()->ts_F;
        break;
      }
      default:
        break;
      }
      my_f = getFEMethod()->snes_f;
    } else {
      my_f = F;
    }
    return my_f;
  };

  auto vec_assemble = [&](Vec my_f) {
    MoFEMFunctionBegin;
    CHKERR VecSetOption(my_f, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
    CHKERR VecSetValues(my_f, nbRows, row_indices, &*vec_f.data().begin(),
                        ADD_VALUES);
    MoFEMFunctionReturn(0);
  };

  // assemble local matrix
  CHKERR vec_assemble(get_f());

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalTildeCFunSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  commonDataSimpleContact->tildeCFunPtr.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->tildeCFunPtr.get()->clear();

  commonDataSimpleContact->lambdaGapDiffProductPtr.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->lambdaGapDiffProductPtr.get()->clear();

  auto lambda_gap_diff_prod =
      getFTensor0FromVec(*commonDataSimpleContact->lambdaGapDiffProductPtr);

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto tilde_c_fun = getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    const double cg = cN * gap_gp;

    const double lambda_gap_diff = lagrange_slave - cg;
    const double regular_abs = fabs(lambda_gap_diff);

    tilde_c_fun =
        (lagrange_slave + cg - pow(regular_abs, r) / r); // is lagMult Correct?

    const double exponent = r - 1.;

    double sign = 0.;
    sign = (lambda_gap_diff == 0) ? 0 : (lambda_gap_diff < 0) ? -1 : 1;
    // cerr << "tilde_c_fun in fun " << tilde_c_fun<<"\n";
    // if (lagrange_slave == 0. && cg == 0.)
    // sign = 1.;
    lambda_gap_diff_prod = sign * pow(regular_abs, exponent);
    ++lagrange_slave;
    ++gap_gp;
    ++lambda_gap_diff_prod;
    ++tilde_c_fun;
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalTildeCFunNitscheSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  commonDataSimpleContact->tildeCFunNitschePtr.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->tildeCFunNitschePtr.get()->clear();

  commonDataSimpleContact->nitscheGapDiffProductPtr.get()->resize(nb_gauss_pts);
  commonDataSimpleContact->nitscheGapDiffProductPtr.get()->clear();

  auto nitsche_gap_diff_prod =
      getFTensor0FromVec(*commonDataSimpleContact->nitscheGapDiffProductPtr);

  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto proj_normal_stress_master =
      getFTensor0FromVec(*commonDataSimpleContact->projNormalStressAtMaster);
  auto proj_normal_stress_slave =
      getFTensor0FromVec(*commonDataSimpleContact->projNormalStressAtSlave);

  auto tilde_c_fun = getFTensor0FromVec(*commonDataSimpleContact->tildeCFunNitschePtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    const double cg = cN * gap_gp;

    // const double regular_abs = fabs(lambda_gap_diff);
    const double p_n_omega = omegaVal * proj_normal_stress_slave -
                             (1 - omegaVal) * proj_normal_stress_master;
    const double lambda_gap_diff = p_n_omega - cg;
    const double lambda_gap_sum = p_n_omega + cg;

    const double regular_abs = fabs(lambda_gap_sum);
    tilde_c_fun =
        0.5 * (lambda_gap_diff + pow(regular_abs, r) / r); // is lagMult Correct?
    // cerr << "tilde_c_fun " << tilde_c_fun << "\n";
    const double exponent = r - 1.;

    double sign = 0.;
    sign = (std::abs(lambda_gap_sum) < 1.e-8) ? 0 : (lambda_gap_sum < 0) ? -1 : 1;
    // cerr << "tilde_c_fun in fun " << tilde_c_fun<<"\n";
    // if (lagrange_slave == 0. && cg == 0.)
    // sign = 1.;
    nitsche_gap_diff_prod = sign * pow(regular_abs, exponent);

    ++gap_gp;
    ++nitsche_gap_diff_prod;
     ++proj_normal_stress_slave;
     ++proj_normal_stress_master;
    ++tilde_c_fun;
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalIntTildeCFunSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  int nb_base_fun_col = data.getFieldData().size();
  const double area_s =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  vecR.resize(nb_base_fun_col, false); // the last false in ublas
                                       // resize will destroy (not
                                       // preserved) the old values
  vecR.clear();

  auto tilde_c_fun = getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    double val_s = getGaussPtsSlave()(2, gg) * area_s;

    FTensor::Tensor0<double *> t_base_lambda(&data.getN()(gg, 0));

    for (int bbr = 0; bbr != nb_base_fun_col; bbr++) {
      const double s = val_s * t_base_lambda * tilde_c_fun;
      vecR[bbr] += s;  // it is a plus since F always scales with -1
      ++t_base_lambda; // update rows
    }
    ++tilde_c_fun;
  } // for gauss points

  Vec f;
  if (F == PETSC_NULL) {
    f = getFEMethod()->snes_f;
  } else {
    f = F;
  }

  CHKERR VecSetValues(f, nb_base_fun_col, &data.getIndices()[0], &vecR[0],
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalIntTildeCFunSlaveL2::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  int nb_base_fun_col = data.getFieldData().size();
  const double area_s =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  vecR.resize(nb_base_fun_col, false); // the last false in ublas
                                       // resize will destroy (not
                                       // preserved) the old values
  vecR.clear();

  auto tilde_c_fun = getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    double val_s = getGaussPts()(2, gg) * area_s;
    FTensor::Tensor0<double *> t_base_lambda(&data.getN()(gg, 0));

    for (int bbr = 0; bbr != nb_base_fun_col; bbr++) {
      const double s = val_s * t_base_lambda * tilde_c_fun;
      vecR[bbr] += s;  // it is a plus since F always scales with -1
      ++t_base_lambda; // update rows
    }
    ++tilde_c_fun;
  } // for gauss points

  Vec f;
  if (F == PETSC_NULL) {
    f = getFEMethod()->snes_f;
  } else {
    f = F;
  }
  CHKERR VecSetValues(f, nb_base_fun_col, &data.getIndices()[0], &vecR[0],
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalIntTildeCFunSlaveALE::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  int nb_base_fun_col = data.getFieldData().size();
  const double area_s =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  vecR.resize(nb_base_fun_col, false); // the last false in ublas
                                       // resize will destroy (not
                                       // preserved) the old values
  vecR.clear();

  auto tilde_c_fun = getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

  auto length_normal =
      getFTensor0FromVec(*commonDataSimpleContact->normalSlaveLengthALEPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    double val_s = getGaussPtsSlave()(2, gg) * 0.5 * length_normal;

    FTensor::Tensor0<double *> t_base_lambda(&data.getN()(gg, 0));

    for (int bbr = 0; bbr != nb_base_fun_col; bbr++) {
      const double s = val_s * t_base_lambda * tilde_c_fun;
      vecR[bbr] += s;  // it is a plus since F always scales with -1
      ++t_base_lambda; // update rows
    }
    ++tilde_c_fun;
    ++length_normal;
  } // for gauss points

  Vec f;
  if (F == PETSC_NULL) {
    f = getFEMethod()->snes_f;
  } else {
    f = F;
  }

  CHKERR VecSetValues(f, nb_base_fun_col, &data.getIndices()[0], &vecR[0],
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalIntTildeCFunSlaveHdiv::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  int nb_base_fun_col = data.getFieldData().size();
  const double area_s =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  vecR.resize(nb_base_fun_col, false); // the last false in ublas
                                       // resize will destroy (not
                                       // preserved) the old values
  vecR.clear();

  auto tilde_c_fun = getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);
 //auto t_lag_base = data.getFTensor1N<3>();
  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    double val_s = getGaussPtsSlave()(2, gg) * area_s * tilde_c_fun;
    auto t_lag_base = data.getFTensor1N<3>(gg, 0);

    for (int bbr = 0; bbr != nb_base_fun_col; bbr++) {
      const double s = val_s * t_lag_base(0) ;
      vecR[bbr] += s; // it is a plus since F always scales with -1
      ++t_lag_base;   // update rows
    }
    ++tilde_c_fun;
  } // for gauss points

  Vec f;
  if (F == PETSC_NULL) {
    f = getFEMethod()->snes_f;
  } else {
    f = F;
  }

  CHKERR VecSetValues(f, nb_base_fun_col, &data.getIndices()[0], &vecR[0],
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalIntTildeCFunSlaveHdiv3D::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  int nb_base_fun_col = data.getFieldData().size();
  const double area_s =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  vecR.resize(nb_base_fun_col, false); // the last false in ublas
                                       // resize will destroy (not
                                       // preserved) the old values
  vecR.clear();

  auto tilde_c_fun = getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

 auto lagrange_slave_t1 =
    getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsT1Ptr);

    auto lagrange_slave_t2 =
    getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsT2Ptr);

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;

  auto normal_at_gp_slave =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto tangent_t1_at_gp_slave = get_tensor_vec(
      commonDataSimpleContact->tangentOneVectorSlavePtr.get()[0]);

  auto tangent_t2_at_gp_slave = get_tensor_vec(
      commonDataSimpleContact->tangentTwoVectorSlavePtr.get()[0]);

  const int nb_dofs = data.getFieldData().size();
  if (!nb_dofs)
    MoFEMFunctionReturnHot(0);

  auto t_w = getFTensor0IntegrationWeight();

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    // cerr << " Tilde c fun " << tilde_c_fun << "\n";
    const double val_s = t_w * area_s * tilde_c_fun;
    const double val_t_1 = t_w * area_s * lagrange_slave_t1;
    const double val_t_2 = t_w * area_s * lagrange_slave_t2;
    auto t_lag_base = data.getFTensor1N<3>(gg, 0);

    FTensor::Tensor1<double *, 3> t_field_data_slave(
        &data.getFieldData()[0], &data.getFieldData()[1],
        &data.getFieldData()[2], 3);

    for (int bbr = 0; bbr != nb_dofs / 3; bbr++) {
      const double s = val_s * t_lag_base(i) * normal_at_gp_slave(i);

      const double t_1 = val_t_1 * t_lag_base(i) * tangent_t1_at_gp_slave(i);
      const double t_2 = val_t_2 * t_lag_base(i) * tangent_t2_at_gp_slave(i);
      // cerr << "Val s " << val_s << " nb_dofs " << nb_dofs << "\n";
      // cerr << "base normal " << t_lag_base(i) * normal_at_gp_slave(i) <<
      // "\n";
      const double check = t_lag_base(i) * normal_at_gp_slave(i) *
                           t_lag_base(i) * normal_at_gp_slave(i);
      vecR[3 * bbr] -= s;
      vecR[3 * bbr + 1] -= t_1;
      vecR[3 * bbr + 2] -= t_2;
      // if (gg == 0 && check > 1.e-8 ) {

      //   // cerr << "t_lag_base " << t_lag_base << " bbr " << bbr << " side "
      //   //      << side << "\n";

      //   const double lag_sum = t_field_data_slave(0) + t_field_data_slave(1) +
      //       t_field_data_slave(2);
      //   vecR[3 * bbr + 1] -= t_field_data_slave(i) * tangent_t1_at_gp_slave(i);
      //   vecR[3 * bbr + 2] -= t_field_data_slave(i) * tangent_t2_at_gp_slave(i);
      // }
      // vecR[bbr + 1] += s;
      // vecR[bbr + 2] += s; // it is a plus since F always scales with -1
      ++t_lag_base;       // update rows
      ++t_field_data_slave;
    }
    ++tilde_c_fun;
    ++lagrange_slave_t1;
    ++lagrange_slave_t2;
    ++t_w;
  } // for gauss points

  Vec f;
  if (F == PETSC_NULL) {
    f = getFEMethod()->snes_f;
    } else {
      f = F;
    }
    // cerr << " Rhs " << nb_dofs/3 << "\n";

    CHKERR VecSetValues(f, nb_dofs, &data.getIndices()[0], &vecR[0],
                        ADD_VALUES);

    MoFEMFunctionReturn(0);
}

// MoFEMErrorCode SimpleContactProblem::OpCalIntTildeCFunSlaveHdiv::doWork(
//     int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
//   MoFEMFunctionBegin;

//   if (type != MBTRI)
//     MoFEMFunctionReturnHot(0);

//   if (data.getIndices().size() == 0)
//     MoFEMFunctionReturnHot(0);

//   const int nb_gauss_pts = data.getN().size1();

//   int nb_base_fun_col = data.getFieldData().size();
//   const double area_s =
//       commonDataSimpleContact->areaSlave; // same area in master and slave

//   vecR.resize(nb_base_fun_col, false); // the last false in ublas
//                                        // resize will destroy (not
//                                        // preserved) the old values
//   vecR.clear();

//   auto tilde_c_fun = getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

//   auto t_lag_base = data.getFTensor1N<3>();

//   auto get_tensor_vec = [](VectorDouble &n) {
//     return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
//   };

//   FTensor::Index<'i', 3> i;
//   auto normal_at_gp =
//       get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

//   for (int gg = 0; gg != nb_gauss_pts; gg++) {
//     double val_s = getGaussPtsSlave()(2, gg) * area_s;

//     //FTensor::Tensor0<double *> t_base_lambda(&data.getN()(gg, 0));

//     for (int bbr = 0; bbr != nb_base_fun_col; bbr++) {
//       const double s = val_s * t_lag_base(0) *  /* normal_at_gp(i) */tilde_c_fun;
//       vecR[bbr] += s; // it is a plus since F always scales with -1
//       ++t_lag_base;
//     }
//     ++tilde_c_fun;
//   } // for gauss points

//   // auto tilde_c_fun_2 =
//   //     getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

//   Vec f;
//   if (F == PETSC_NULL) {
//     f = getFEMethod()->snes_f;
//   } else {
//     f = F;
//   }

//   CHKERR VecSetValues(f, nb_base_fun_col, &data.getIndices()[0], &vecR[0],
//                       ADD_VALUES);

//   MoFEMFunctionReturn(0);
// }

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
    auto t_row_base = row_data.getFTensor1N<3>();
    auto t_row_diff_base = row_data.getFTensor2DiffN<3, 3>();
     for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
       const double alpha = getMeasure() * t_w;
       size_t rr = 0;
      for (; rr != row_nb_dofs / 3; ++rr) {
       FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_mat_diag{
       & locMat(3 * rr + 0, 0), &locMat(3 * rr + 1, 1), &locMat(3 * rr + 2, 2) };
       FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3> t_mat{
           &locMat(3 * rr + 0, 0), &locMat(3 * rr + 0, 1),
           &locMat(3 * rr + 0, 2), &locMat(3 * rr + 1, 0),
           &locMat(3 * rr + 1, 1), &locMat(3 * rr + 1, 2),
           &locMat(3 * rr + 2, 0), &locMat(3 * rr + 2, 1),
           &locMat(3 * rr + 2, 2)};
       const double t_row_div_base = t_row_diff_base(0, 0) +
                                     t_row_diff_base(1, 1) +
                                     t_row_diff_base(2, 2);
       auto t_col_base = col_data.getFTensor0N(gg, 0);
       auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);
       for (size_t cc = 0; cc != col_nb_dofs / 3; ++cc) {
       t_mat_diag(i) += alpha * t_row_base(j) * t_col_diff_base(j);
       t_mat_diag(i) += alpha * t_row_div_base * t_col_base;
       ++t_col_base;
       ++t_col_diff_base;
       ++t_mat_diag;
       ++t_mat;
        }
      ++t_row_diff_base;
     ++t_row_base;
      }
     for (; rr < nb_base_functions; ++rr) {
     ++t_row_diff_base;
     ++t_row_base;
      }
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
SimpleContactProblem::OpContactConstraintMatrixMasterSlave::doWork(
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
  int nb_base_fun_col = col_data.getFieldData().size();

  const double area_slave =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 1, c + 0),
                                         &m(r + 2, c + 0));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;

  NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
  NN.clear();

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    double val_m = getGaussPtsMaster()(2, gg) * area_slave;

    FTensor::Tensor0<double *> t_base_lambda(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
      FTensor::Tensor0<double *> t_base_master(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
        const double m = val_m * t_base_lambda * t_base_master;

        auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, bbc);
        t_assemble_m(i) -= m * const_unit_n(i);

        ++t_base_master; // update rows master
      }
      ++t_base_lambda; // update cols slave
    }
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactConstraintMatrixMasterSlaveH1L2::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  if (commonDataSimpleContact->l2ColData == nullptr)
    MoFEMFunctionReturnHot(0);

  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = commonDataSimpleContact->l2ColData->getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = commonDataSimpleContact->l2ColData->getFieldData().size();
  cerr << " col base !! " << nb_base_fun_col << "\n";
  const double area_slave =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 1, c + 0),
                                         &m(r + 2, c + 0));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;

  NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
  NN.clear();

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    double val_m = getGaussPtsMaster()(2, gg) * area_slave;

    FTensor::Tensor0<double *> t_base_lambda(&commonDataSimpleContact->l2ColData->getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
      FTensor::Tensor0<double *> t_base_master(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
        const double m = val_m * t_base_lambda * t_base_master;

        auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, bbc);
        t_assemble_m(i) -= m * const_unit_n(i);

        ++t_base_master; // update rows master
      }
      cerr << "base lambda " << t_base_lambda << "  gg " << gg << "\n";
      ++t_base_lambda; // update cols slave
    }
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  cerr << "col data number " << nb_col << "\n";

  cerr << "col data indices "
       << commonDataSimpleContact->l2ColData->getIndices() << "\n";
  CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
                      &commonDataSimpleContact->l2ColData->getIndices()[0],
                      &*NN.data().begin(), ADD_VALUES);

  // cerr << "Passed \n";

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactConstraintMatrixMasterSlaveHdiv::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  if (col_type != MBTRI)
    MoFEMFunctionReturnHot(0);

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
  int nb_base_fun_col = col_data.getFieldData().size();

  const double area_slave =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 1, c + 0),
                                         &m(r + 2, c + 0));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;

  NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
  NN.clear();

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    double val_m = getGaussPtsMaster()(2, gg) * area_slave;
    // auto t_lag_base = col_data.getFTensor1N<3>();
    auto t_lag_base = col_data.getFTensor1N<3>(gg, 0);
    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
      FTensor::Tensor0<double *> t_base_master(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
        const double m =
            val_m * /*const_unit_n(i) */ t_lag_base(0) * t_base_master;

        auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, bbc);
        t_assemble_m(i) -= m * const_unit_n(i);

        ++t_base_master; // update rows master
      }
      ++t_lag_base; // update cols slave
    }
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactConstraintMatrixSlaveSlave::doWork(
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
  int nb_base_fun_col = col_data.getFieldData().size();

  const double area_slave =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 1, c + 0),
                                         &m(r + 2, c + 0));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;

  NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
  NN.clear();

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    double val_s = getGaussPtsSlave()(2, gg) * area_slave;

    FTensor::Tensor0<double *> t_base_lambda(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base_slave(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
        const double s = val_s * t_base_lambda * t_base_slave;

        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, bbc);

        t_assemble_s(i) += s * const_unit_n(i);

        ++t_base_slave; // update rows
      }
      ++t_base_lambda; // update cols slave
    }
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactConstraintMatrixSlaveSlaveH1L2::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  if (commonDataSimpleContact->l2ColData == nullptr)
    MoFEMFunctionReturnHot(0);
  // Both sides are needed since both sides contribute their shape
  // function to the stiffness matrix
  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = commonDataSimpleContact->l2ColData->getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = commonDataSimpleContact->l2ColData->getFieldData().size();

  const double area_slave =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 1, c + 0),
                                         &m(r + 2, c + 0));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;

  NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
  NN.clear();

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    double val_s = getGaussPtsSlave()(2, gg) * area_slave;

    FTensor::Tensor0<double *> t_base_lambda(&commonDataSimpleContact->l2ColData->getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base_slave(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
        const double s = val_s * t_base_lambda * t_base_slave;

        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, bbc);

        t_assemble_s(i) += s * const_unit_n(i);

        ++t_base_slave; // update rows
      }
      ++t_base_lambda; // update cols slave
    }
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
                      &commonDataSimpleContact->l2ColData->getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactConstraintMatrixSlaveSlaveALE::doWork(
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
  int nb_base_fun_col = col_data.getFieldData().size();

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 1, c + 0),
                                         &m(r + 2, c + 0));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;

  NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
  NN.clear();

  auto length_normal =
      getFTensor0FromVec(*commonDataSimpleContact->normalSlaveLengthALEPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    double val_s = getGaussPtsSlave()(2, gg);
    auto normal_at_gp =
        get_tensor_vec(commonDataSimpleContact->normalSlaveALE[gg]);

    FTensor::Tensor0<double *> t_base_lambda(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base_slave(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
        const double s = val_s * t_base_lambda * t_base_slave;

        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, bbc);

        t_assemble_s(i) += s * normal_at_gp(i) * 0.5;

        ++t_base_slave; // update rows
      }
      ++t_base_lambda; // update cols slave
    }
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactConstraintMatrixMasterSlaveALE::doWork(
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
  int nb_base_fun_col = col_data.getFieldData().size();

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 1, c + 0),
                                         &m(r + 2, c + 0));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;

  NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
  NN.clear();

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    double val_m = getGaussPtsMaster()(2, gg);
    auto normal_at_gp =
        get_tensor_vec(commonDataSimpleContact->normalSlaveALE[gg]);

    FTensor::Tensor0<double *> t_base_lambda(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base_master(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
        const double m = val_m * t_base_lambda * t_base_master;

        auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, bbc);

        t_assemble_m(i) -= m * normal_at_gp(i) * 0.5;

        ++t_base_master; // update rows
      }
      ++t_base_lambda; // update cols slave
    }
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetTangentMaster::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int ngp = data.getN().size1(); // this is number of base functions for
                                       // vertices, 6 bases functions for prism

  if (type == MBVERTEX)
    commonDataSimpleContact->tangentMasterALE.resize(ngp);
  unsigned int nb_dofs = data.getFieldData().size();
  // tangent vectors to face F3
  for (unsigned int gg = 0; gg != ngp; ++gg) {
    if (type == MBVERTEX) {
      commonDataSimpleContact->tangentMasterALE[gg].resize(2);
      commonDataSimpleContact->tangentMasterALE[gg][0].resize(3);
      commonDataSimpleContact->tangentMasterALE[gg][1].resize(3);
      commonDataSimpleContact->tangentMasterALE[gg][0].clear();
      commonDataSimpleContact->tangentMasterALE[gg][1].clear();
    }

    for (unsigned int dd = 0; dd != 3; ++dd) {
      commonDataSimpleContact->tangentMasterALE[gg][0][dd] +=
          cblas_ddot(nb_dofs / 3, &data.getDiffN()(gg, 0), 2,
                     &data.getFieldData()[dd], 3); // tangent-1
      commonDataSimpleContact->tangentMasterALE[gg][1][dd] +=
          cblas_ddot(nb_dofs / 3, &data.getDiffN()(gg, 1), 2,
                     &data.getFieldData()[dd], 3); // tangent-2
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetTangentSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int ngp = data.getN().size1(); // this is number of base functions for
                                       // vertices, 6 bases functions for prism

  if (type == MBVERTEX)
    commonDataSimpleContact->tangentSlaveALE.resize(ngp);

  unsigned int nb_dofs = data.getFieldData().size();
  // tangent vectors to face F4
  for (unsigned int gg = 0; gg != ngp; ++gg) {

    if (type == MBVERTEX) {
      commonDataSimpleContact->tangentSlaveALE[gg].resize(2);
      commonDataSimpleContact->tangentSlaveALE[gg][0].resize(3);
      commonDataSimpleContact->tangentSlaveALE[gg][1].resize(3);
      commonDataSimpleContact->tangentSlaveALE[gg][0].clear();
      commonDataSimpleContact->tangentSlaveALE[gg][1].clear();
    }

    for (unsigned int dd = 0; dd != 3; ++dd) {
      commonDataSimpleContact->tangentSlaveALE[gg][0][dd] +=
          cblas_ddot(nb_dofs / 3, &data.getDiffN()(gg, 0), 2,
                     &data.getFieldData()[dd], 3); // tangent-1
      commonDataSimpleContact->tangentSlaveALE[gg][1][dd] +=
          cblas_ddot(nb_dofs / 3, &data.getDiffN()(gg, 1), 2,
                     &data.getFieldData()[dd], 3); // tangent-2
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetNormalMasterALE::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  int gp_gp = data.getN().size1();

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  commonDataSimpleContact->normalMasterALE.resize(gp_gp);

  commonDataSimpleContact->normalMasterLengthALEPtr.get()->resize(gp_gp);
  commonDataSimpleContact->normalMasterLengthALEPtr.get()->clear();

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto f3_length_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->normalMasterLengthALEPtr);

  for (int gg = 0; gg != gp_gp; ++gg) {

    commonDataSimpleContact->normalMasterALE[gg].resize(3);
    commonDataSimpleContact->normalMasterALE[gg].clear();

    auto normal_original_master =
        get_tensor_vec(commonDataSimpleContact->normalMasterALE[gg]);

    auto tangent_0_master =
        get_tensor_vec(commonDataSimpleContact->tangentMasterALE[gg][0]);
    auto tangent_1_master =
        get_tensor_vec(commonDataSimpleContact->tangentMasterALE[gg][1]);

    normal_original_master(i) =
        FTensor::levi_civita(i, j, k) * tangent_0_master(j) * tangent_1_master(k);

    f3_length_ptr = sqrt(normal_original_master(i) * normal_original_master(i));

    ++f3_length_ptr;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetNormalSlaveALE::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  int gp_gp = data.getN().size1();

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  commonDataSimpleContact->normalSlaveALE.resize(gp_gp);

  commonDataSimpleContact->normalSlaveLengthALEPtr.get()->resize(gp_gp);
  commonDataSimpleContact->normalSlaveLengthALEPtr.get()->clear();

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto f4_length_ptr =
      getFTensor0FromVec(*commonDataSimpleContact->normalSlaveLengthALEPtr);

  for (int gg = 0; gg != gp_gp; ++gg) {

    commonDataSimpleContact->normalSlaveALE[gg].resize(3);
    commonDataSimpleContact->normalSlaveALE[gg].clear();

    auto normal_original_slave =
        get_tensor_vec(commonDataSimpleContact->normalSlaveALE[gg]);

    auto tangent_0_slave =
        get_tensor_vec(commonDataSimpleContact->tangentSlaveALE[gg][0]);
    auto tangent_1_slave =
        get_tensor_vec(commonDataSimpleContact->tangentSlaveALE[gg][1]);

    normal_original_slave(i) = FTensor::levi_civita(i, j, k) *
                                tangent_0_slave(j) * tangent_1_slave(k);

    f4_length_ptr = sqrt(normal_original_slave(i) * normal_original_slave(i));

    ++f4_length_ptr;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactConstraintMatrixSlaveSlave_dX::doWork(
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

  auto get_tensor_vec_3 = [&](VectorDouble3 &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  auto make_normal_vec_der =
      [&](VectorDouble3 &der_ksi, VectorDouble3 &der_eta,
          VectorDouble3 &normal_der, MatrixDouble &der_normal_mat,
          FTensor::Tensor0<double *> &t_N_over_ksi,
          FTensor::Tensor0<double *> &t_N_over_eta,
          boost::shared_ptr<CommonDataSimpleContact> &commonDataSimpleContact,
          const int &gg) {
        der_normal_mat.clear();

        auto t_tan_1 =
            get_tensor_vec(commonDataSimpleContact->tangentSlaveALE[gg][0]);
        auto t_tan_2 =
            get_tensor_vec(commonDataSimpleContact->tangentSlaveALE[gg][1]);
        for (int dd = 0; dd != 3; ++dd) {

          der_ksi.clear();
          der_eta.clear();
          normal_der.clear();

          der_ksi[dd] = t_N_over_ksi;
          der_eta[dd] = t_N_over_eta;

          auto t_normal_der = get_tensor_vec_3(normal_der);
          auto t_dn_xi = get_tensor_vec_3(der_ksi);
          auto t_dn_eta = get_tensor_vec_3(der_eta);

          t_normal_der(i) +=
              FTensor::levi_civita(i, j, k) * t_dn_xi(j) * t_tan_2(k);

          t_normal_der(i) +=
              FTensor::levi_civita(i, j, k) * t_tan_1(j) * t_dn_eta(k);

          for (int kk = 0; kk != 3; ++kk) {
            der_normal_mat(kk, dd) += t_normal_der(kk);
          }
        }
      };

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  VectorDouble3 normal_der(3);
  VectorDouble3 der_ksi(3), der_eta(3);

  MatrixDouble der_normal_mat;
  der_normal_mat.resize(3, 3, false);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    FTensor::Tensor0<double *> t_N_over_ksi(&col_data.getDiffN()(gg, 0));

    FTensor::Tensor0<double *> t_N_over_eta(&col_data.getDiffN()(gg, 1));

    double val_s = getGaussPtsSlave()(2, gg);

    FTensor::Tensor0<double *> t_base_slave_row_X(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base_slave(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
        const double s = val_s * lagrange_slave * t_base_slave;

        make_normal_vec_der(der_ksi, der_eta, normal_der, der_normal_mat,
                            t_N_over_ksi, t_N_over_eta, commonDataSimpleContact,
                            gg);

        auto t_d_n = get_tensor_from_mat(der_normal_mat, 0, 0);

        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

        t_assemble_s(i, j) += 0.5 * s * t_d_n(i, j);

        ++t_base_slave; // update rows
      }
      ++t_base_slave_row_X; // update cols slave
      ++t_N_over_ksi;       // move pointers by two
      ++t_N_over_ksi;
      ++t_N_over_eta;
      ++t_N_over_eta;
    }
    ++lagrange_slave;
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpContactSimplePenaltyMasterMaster::doWork(
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

  auto t_w = getFTensor0IntegrationWeightMaster();

  const double area_slave = commonDataSimpleContact->areaSlave;

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    if (gap_gp > 0) {
      ++gap_gp;
      ++t_w;
      continue;
    }

const double val_m = t_w * area_slave * cN;

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
    ++gap_gp;
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpContactSimplePenaltyMasterSlave::doWork(
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

  auto t_w = getFTensor0IntegrationWeightMaster();

  const double area_slave = commonDataSimpleContact->areaSlave;

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);
  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    if (gap_gp > 0) {
      ++gap_gp;
      ++t_w;
      continue;
    }

    const double val_m = t_w * area_slave * cN;

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
    ++gap_gp;
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpContactSimplePenaltySlaveSlave::doWork(
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

  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);
  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    if (gap_gp > 0) {
      ++gap_gp;
      ++t_w;
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
    ++gap_gp;
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpContactSimplePenaltySlaveMaster::doWork(
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

  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);
  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    if (gap_gp > 0) {
      ++t_w;
      ++gap_gp;
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
    ++gap_gp;
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactConstraintMatrixMasterSlave_dX::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  // derrivatives only for vertices in cols
  if (col_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

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

  auto make_normal_vec_der =
      [&](VectorDouble3 &der_ksi, VectorDouble3 &der_eta,
          VectorDouble3 &normal_der, MatrixDouble &der_normal_mat,
          FTensor::Tensor0<double *> &t_N_over_ksi,
          FTensor::Tensor0<double *> &t_N_over_eta,
          boost::shared_ptr<CommonDataSimpleContact> &commonDataSimpleContact,
          const int &gg) {
        der_normal_mat.clear();

        auto t_tan_1 =
            get_tensor_vec(commonDataSimpleContact->tangentSlaveALE[gg][0]);
        auto t_tan_2 =
            get_tensor_vec(commonDataSimpleContact->tangentSlaveALE[gg][1]);
        for (int dd = 0; dd != 3; ++dd) {

          der_ksi.clear();
          der_eta.clear();
          normal_der.clear();

          der_ksi[dd] = t_N_over_ksi;
          der_eta[dd] = t_N_over_eta;

          auto t_normal_der = get_tensor_vec_3(normal_der);
          auto t_dn_xi = get_tensor_vec_3(der_ksi);
          auto t_dn_eta = get_tensor_vec_3(der_eta);

          t_normal_der(i) +=
              FTensor::levi_civita(i, j, k) * t_dn_xi(j) * t_tan_2(k);

          t_normal_der(i) +=
              FTensor::levi_civita(i, j, k) * t_tan_1(j) * t_dn_eta(k);

          for (int kk = 0; kk != 3; ++kk) {
            der_normal_mat(kk, dd) += t_normal_der(kk);
          }
        }
      };

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  VectorDouble3 normal_der(3);
  VectorDouble3 der_ksi(3), der_eta(3);

  MatrixDouble der_normal_mat;
  der_normal_mat.resize(3, 3, false);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    FTensor::Tensor0<double *> t_N_over_ksi(&col_data.getDiffN()(gg, 0));

    FTensor::Tensor0<double *> t_N_over_eta(&col_data.getDiffN()(gg, 1));

    double val_s = getGaussPtsSlave()(2, gg);

    FTensor::Tensor0<double *> t_base_slave_row_X(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base_master(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
        const double s = val_s * lagrange_slave * t_base_master;

        make_normal_vec_der(der_ksi, der_eta, normal_der, der_normal_mat,
                            t_N_over_ksi, t_N_over_eta, commonDataSimpleContact,
                            gg);

        auto t_d_n = get_tensor_from_mat(der_normal_mat, 0, 0);

        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, 3 * bbc);

        t_assemble_s(i, j) -= 0.5 * s * t_d_n(i, j);

        ++t_base_master; // update rows
      }
      ++t_base_slave_row_X; // update cols slave
      ++t_N_over_ksi;       // move pointers by two
      ++t_N_over_ksi;
      ++t_N_over_eta;
      ++t_N_over_eta;
    }
    ++lagrange_slave;
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactConstraintMatrixSlaveSlaveHdiv::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  if (col_type != MBTRI)
    MoFEMFunctionReturnHot(0);

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
  int nb_base_fun_col = col_data.getFieldData().size();

  const double area_slave =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 1, c + 0),
                                         &m(r + 2, c + 0));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;

  NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
  NN.clear();

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  //auto t_lag_base = col_data.getFTensor1N<3>();

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    double val_s = getGaussPtsSlave()(2, gg) * area_slave;
    auto t_lag_base = col_data.getFTensor1N<3>(gg, 0);

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base_slave(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
        const double s = val_s * t_lag_base(0)/* const_unit_n(i)*/ * t_base_slave;

        auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, bbc);

        t_assemble_s(i) += s * const_unit_n(i);

        ++t_base_slave; // update rows
      }
      ++t_lag_base; // update cols slave
    }
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpDerivativeBarTildeCFunOLambdaSlaveSlave::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size();

  int nb_base_fun_col = col_data.getFieldData().size();

  const double area_slave =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  NN.resize(nb_base_fun_row, nb_base_fun_col,
            false); // the last false in ublas resize will destroy
                    // (not preserved) the old values
  NN.clear();

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto lambda_gap_diff_prod =
      getFTensor0FromVec(*commonDataSimpleContact->lambdaGapDiffProductPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    const double val_s = getGaussPtsSlave()(2, gg) * area_slave;

    FTensor::Tensor0<double *> t_base_lambda_col(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
      FTensor::Tensor0<double *> t_base_lambda_row(&row_data.getN()(gg, 0));
      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
        if (bbr == bbc) {
          if (fabs(gap_gp) < 1.e-8 && fabs(lagrange_slave) < 1.e-8) {
            NN(bbr, bbc) = 0;
          } else {
            NN(bbr, bbc) += (1. - lambda_gap_diff_prod) * val_s *
                            t_base_lambda_row * t_base_lambda_col;
          }

        } else {
          if (fabs(gap_gp) < 1.e-8 && fabs(lagrange_slave) < 1.e-8) {
          } else {
            NN(bbr, bbc) += (1. - lambda_gap_diff_prod) * val_s *
                            t_base_lambda_row * t_base_lambda_col;
          }
        }
        ++t_base_lambda_row; // update rows
      }
      ++t_base_lambda_col; // update cols
    }
    ++lagrange_slave;
    ++gap_gp;
    ++lambda_gap_diff_prod;
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  CHKERR MatSetValues(aij, nb_base_fun_row, &row_data.getIndices()[0],
                      nb_base_fun_col, // ign: is shift row right here?
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpDerivativeBarTildeCFunOLambdaSlaveSlaveL2L2::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size();

  int nb_base_fun_col = col_data.getFieldData().size();

  const double area_slave =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  NN.resize(nb_base_fun_row, nb_base_fun_col,
            false); // the last false in ublas resize will destroy
                    // (not preserved) the old values
  NN.clear();

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto lambda_gap_diff_prod =
      getFTensor0FromVec(*commonDataSimpleContact->lambdaGapDiffProductPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    const double val_s = getGaussPts()(2, gg) * area_slave;

    FTensor::Tensor0<double *> t_base_lambda_col(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
      FTensor::Tensor0<double *> t_base_lambda_row(&row_data.getN()(gg, 0));
      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
        if (bbr == bbc) {
          if (fabs(gap_gp) < 1.e-8 && fabs(lagrange_slave) < 1.e-8) {
            NN(bbr, bbc) = 0;
          } else {
            NN(bbr, bbc) += (1. - lambda_gap_diff_prod) * val_s *
                            t_base_lambda_row * t_base_lambda_col;
          }

        } else {
          if (fabs(gap_gp) < 1.e-8 && fabs(lagrange_slave) < 1.e-8) {
          } else {
            NN(bbr, bbc) += (1. - lambda_gap_diff_prod) * val_s *
                            t_base_lambda_row * t_base_lambda_col;
          }
        }
        ++t_base_lambda_row; // update rows
      }
      ++t_base_lambda_col; // update cols
    }
    ++lagrange_slave;
    ++gap_gp;
    ++lambda_gap_diff_prod;
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  CHKERR MatSetValues(aij, nb_base_fun_row, &row_data.getIndices()[0],
                      nb_base_fun_col, // ign: is shift row right here?
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpDerivativeBarTildeCFunOLambdaSlaveSlaveALE::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size();

  int nb_base_fun_col = col_data.getFieldData().size();

  NN.resize(nb_base_fun_row, nb_base_fun_col,
            false); // the last false in ublas resize will destroy
                    // (not preserved) the old values
  NN.clear();

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto lambda_gap_diff_prod =
      getFTensor0FromVec(*commonDataSimpleContact->lambdaGapDiffProductPtr);

  auto length_normal =
      getFTensor0FromVec(*commonDataSimpleContact->normalSlaveLengthALEPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    const double val_s = getGaussPtsSlave()(2, gg) * 0.5 * length_normal;

    FTensor::Tensor0<double *> t_base_lambda_col(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
      FTensor::Tensor0<double *> t_base_lambda_row(&row_data.getN()(gg, 0));
      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
        if (bbr == bbc) {
          if (fabs(gap_gp) < 1.e-8 && fabs(lagrange_slave) < 1.e-8) {
            NN(bbr, bbc) = 0;
          } else {
            NN(bbr, bbc) += (1. - lambda_gap_diff_prod) * val_s *
                            t_base_lambda_row * t_base_lambda_col;
          }

        } else {
          if (fabs(gap_gp) < 1.e-8 && fabs(lagrange_slave) < 1.e-8) {
          } else {
            NN(bbr, bbc) += (1. - lambda_gap_diff_prod) * val_s *
                            t_base_lambda_row * t_base_lambda_col;
          }
        }
        ++t_base_lambda_row; // update rows
      }
      ++t_base_lambda_col; // update cols
    }
    ++lagrange_slave;
    ++gap_gp;
    ++lambda_gap_diff_prod;
    ++length_normal;
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  CHKERR MatSetValues(aij, nb_base_fun_row, &row_data.getIndices()[0],
                      nb_base_fun_col, // ign: is shift row right here?
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpDerivativeBarTildeCFunOLambdaSlaveSlaveHdiv::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  if (col_type != MBTRI || row_type != MBTRI)
    MoFEMFunctionReturnHot(0);

  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size();

  int nb_base_fun_col = col_data.getFieldData().size();

  const double area_slave =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  NN.resize(nb_base_fun_row, nb_base_fun_col,
            false); // the last false in ublas resize will destroy
                    // (not preserved) the old values
  NN.clear();

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto lambda_gap_diff_prod =
      getFTensor0FromVec(*commonDataSimpleContact->lambdaGapDiffProductPtr);

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    const double val_s = getGaussPtsSlave()(2, gg) * area_slave;

    auto t_lag_base_col = col_data.getFTensor1N<3>(gg, 0);
    for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
      auto t_lag_base_row = row_data.getFTensor1N<3>(gg, 0);
      for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
        if (bbr == bbc) {
          if (fabs(gap_gp) < 1.e-8 && fabs(lagrange_slave) < 1.e-8) {
            NN(bbr, bbc) = 0;
          } else {
            NN(bbr, bbc) += (1. - lambda_gap_diff_prod) * val_s *
                            (/*const_unit_n(i) */ t_lag_base_row(0)) *
                            (t_lag_base_col(0) /* const_unit_n(i)*/);
          }

        } else {
          if (fabs(gap_gp) < 1.e-8 && fabs(lagrange_slave) < 1.e-8) {
          } else {
            NN(bbr, bbc) += (1. - lambda_gap_diff_prod) * val_s *
                            (/*const_unit_n(i) */ t_lag_base_row(0)) *
                            (t_lag_base_col(0) /* const_unit_n(i)*/);
          }
        }
        ++t_lag_base_row; // update rows
      }
      ++t_lag_base_col; // update cols
    }
    ++lagrange_slave;
    ++gap_gp;
    ++lambda_gap_diff_prod;
  }
  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }
  CHKERR MatSetValues(aij, nb_base_fun_row, &row_data.getIndices()[0],
                      nb_base_fun_col, // ign: is shift row right here?
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpDerivativeBarTildeCFunODisplacementsSlaveMaster::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
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

  const double area_slave =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  NN.resize(nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto get_tensor_from_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto get_vec_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 0, c + 1),
                                         &m(r + 0, c + 2));
  };

  FTensor::Index<'i', 3> i;

  auto lambda_gap_diff_prod =
      getFTensor0FromVec(*commonDataSimpleContact->lambdaGapDiffProductPtr);

  auto const_unit_n = get_tensor_from_vec(
      commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
   const double val_m = (1. + lambda_gap_diff_prod) * cN * getGaussPtsMaster()(2, gg) * area_slave;

    FTensor::Tensor0<double *> t_base_master(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
      FTensor::Tensor0<double *> t_base_lambda(&row_data.getN()(gg, 0));
      const double m = val_m * t_base_master;

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

        auto assemble_mat = get_vec_from_mat(NN, bbr, 3 * bbc);

        assemble_mat(i) += const_unit_n(i) *  
                           m * t_base_lambda;

        ++t_base_lambda; // update rows
      }
      ++t_base_master; // update cols master
    }
    ++lambda_gap_diff_prod;
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  // Assemble NN to final Aij vector based on its global indices
  CHKERR MatSetValues(aij, nb_base_fun_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpDerivativeBarTildeCFunODisplacementsSlaveMasterALE::
    doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
           DataForcesAndSourcesCore::EntData &row_data,
           DataForcesAndSourcesCore::EntData &col_data) {
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

  const double area_slave =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  NN.resize(nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto get_tensor_from_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto get_vec_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 0, c + 1),
                                         &m(r + 0, c + 2));
  };

  FTensor::Index<'i', 3> i;

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto lambda_gap_diff_prod =
      getFTensor0FromVec(*commonDataSimpleContact->lambdaGapDiffProductPtr);

  auto tilde_c_fun = getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

  auto length_normal =
      getFTensor0FromVec(*commonDataSimpleContact->normalSlaveLengthALEPtr);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    double val_m = getGaussPtsMaster()(2, gg) * 0.5;

    auto normal_at_gp =
        get_tensor_from_vec(commonDataSimpleContact->normalSlaveALE[gg]);

    FTensor::Tensor0<double *> t_base_master(&col_data.getN()(gg, 0));

    FTensor::Tensor1<double *, 3> t_field_data_master(
        &col_data.getFieldData()[0], &col_data.getFieldData()[1],
        &col_data.getFieldData()[2], 3);

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
      FTensor::Tensor0<double *> t_base_lambda(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        const double m = val_m * t_base_lambda * t_base_master;

        auto assemble_mat = get_vec_from_mat(NN, bbr, 3 * bbc);

        assemble_mat(i) +=
            normal_at_gp(i) * cN * (1. + lambda_gap_diff_prod) * m;

        ++t_base_lambda; // update rows
      }

      ++t_base_master; // update cols master

      ++t_field_data_master;
    }

    ++tilde_c_fun;
    ++lagrange_slave;
    ++gap_gp;
    ++lambda_gap_diff_prod;
    ++length_normal;
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  // Assemble NN to final Aij vector based on its global indices
  CHKERR MatSetValues(aij, nb_base_fun_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpDerivativeBarTildeCFunODisplacementsSlaveMasterALE_dX::
    doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
           DataForcesAndSourcesCore::EntData &row_data,
           DataForcesAndSourcesCore::EntData &col_data) {
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

  const double area_slave =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  NN.resize(nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto get_tensor_from_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto get_vec_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 0, c + 1),
                                         &m(r + 0, c + 2));
  };

  FTensor::Index<'i', 3> i;

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto lambda_gap_diff_prod =
      getFTensor0FromVec(*commonDataSimpleContact->lambdaGapDiffProductPtr);

  auto tilde_c_fun = getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

  auto length_normal =
      getFTensor0FromVec(*commonDataSimpleContact->normalSlaveLengthALEPtr);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    double val_m = getGaussPtsMaster()(2, gg) * length_normal * 0.5;

    auto normal_at_gp =
        get_tensor_from_vec(commonDataSimpleContact->normalSlaveALE[gg]);

    FTensor::Tensor0<double *> t_base_master(&col_data.getN()(gg, 0));

    FTensor::Tensor1<double *, 3> t_field_data_master(
        &col_data.getFieldData()[0], &col_data.getFieldData()[1],
        &col_data.getFieldData()[2], 3);

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
      FTensor::Tensor0<double *> t_base_lambda(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        const double m = val_m * t_base_lambda * t_base_master;

        auto assemble_mat = get_vec_from_mat(NN, bbr, 3 * bbc);

        assemble_mat(i) +=
            normal_at_gp(i) * cN * (1. + lambda_gap_diff_prod) * m;

        ++t_base_lambda; // update rows
      }

      ++t_base_master; // update cols master

      ++t_field_data_master;
    }

    ++tilde_c_fun;
    ++lagrange_slave;
    ++gap_gp;
    ++lambda_gap_diff_prod;
    ++length_normal;
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  // Assemble NN to final Aij vector based on its global indices
  CHKERR MatSetValues(aij, nb_base_fun_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpCalculateDeformation::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &row_data) {

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
  auto t_invH =
      getFTensor2FromMat<3, 3>(*commonDataSimpleContact->invHMat);
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

MoFEMErrorCode 
SimpleContactProblem::OpCalculateHdivDivergenceResidual::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {

  MoFEMFunctionBegin;

  const int nb_row = data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);

  // cerr << " Vol Lambda " << "\n";

  const int nb_gauss_pts = data.getN().size1();
  int nb_base_fun_row = data.getFieldData().size() / 3;

  vec_f.resize(3 * nb_base_fun_row,
               false); // the last false in ublas
                       // resize will destroy (not
                       // preserved) the old
                       // values
  vec_f.clear();

  const double vol = getVolume(); // same area in master and slave
  auto t_w = getFTensor0IntegrationWeight();

  FTensor::Index<'i', 3> i;

  auto get_tensor_vec = [](VectorDouble &n, const int r) {
    return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
  };

  auto t_div_P = getFTensor1FromMat<3>(*(commonDataSimpleContact->divPAtPts));
  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    double val_s_tet = vol * t_w;

    FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));
    for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

      const double s = val_s_tet * t_base_slave;
      
      auto t_assemble_s = get_tensor_vec(vec_f, 3 * bbr);
      
      t_assemble_s(i) -= s * t_div_P(i);
      ++t_base_slave;
    }

    ++t_w;
    ++t_div_P;
  } // for gauss points

  Vec f;
  if (F == PETSC_NULL) {
    f = getFEMethod()->snes_f;
  } else {
    f = F;
  }

  // cerr << " Vol Lambda " << vec_f << " type " << type << "\n";
  CHKERR VecSetValues(f, nb_row, &data.getIndices()[0], &vec_f[0], ADD_VALUES);

  MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode 
SimpleContactProblem::OpCalculateHdivDivergenceResidualOnDispl::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {

  MoFEMFunctionBegin;

  const int nb_row = data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);

  // cerr << " Vol Lambda " << "\n";

  const int nb_gauss_pts = data.getN().size1();
  int nb_base_fun_row = data.getFieldData().size() / 3;

  vec_f.resize(3 * nb_base_fun_row,
               false); // the last false in ublas
                       // resize will destroy (not
                       // preserved) the old
                       // values
  vec_f.clear();

  const double vol = getVolume(); // same area in master and slave
  auto t_w = getFTensor0IntegrationWeight();

  FTensor::Index<'i', 3> i;

  auto get_tensor_vec = [](VectorDouble &n, const int r) {
    return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
  };

  auto t_x = getFTensor1FromMat<3>(*(commonDataSimpleContact->xAtPts));
  auto t_X = getFTensor1FromMat<3>(*(commonDataSimpleContact->xInitAtPts));
  auto t_n_diff_hvec = data.getFTensor2DiffN<3, 3>();
  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    double val_s_tet = vol * t_w;

    //FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));
    for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

      const double s = val_s_tet * t_n_diff_hvec(i,i) ;
      
      auto t_assemble_s = get_tensor_vec(vec_f, 3 * bbr);
      
      t_assemble_s(i) -= s * (t_x(i) - t_X(i));
          ++t_n_diff_hvec;
    }

    ++t_w;
    ++t_x;
    ++t_X;
  } // for gauss points

  Vec f;
  if (F == PETSC_NULL) {
    f = getFEMethod()->snes_f;
  } else {
    f = F;
  }

  // cerr << " Vol Lambda " << vec_f << " type " << type << "\n";
  CHKERR VecSetValues(f, nb_row, &data.getIndices()[0], &vec_f[0], ADD_VALUES);

  MoFEMFunctionReturn(0);
    }

MoFEMErrorCode
SimpleContactProblem::OpDerivativeBarTildeCFunODisplacementsSlaveMasterHdiv::
    doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
           DataForcesAndSourcesCore::EntData &row_data,
           DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  if (row_type != MBTRI)
    MoFEMFunctionReturnHot(0);

  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
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

  auto get_vec_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 0, c + 1),
                                         &m(r + 0, c + 2));
  };

  FTensor::Index<'i', 3> i;

  auto lambda_gap_diff_prod =
      getFTensor0FromVec(*commonDataSimpleContact->lambdaGapDiffProductPtr);

  auto const_unit_n = get_tensor_from_vec(
      commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    double val_m = cN * (1. + lambda_gap_diff_prod) * getGaussPtsSlave()(2, gg) *
                   area_slave;

    FTensor::Tensor0<double *> t_base_master(&col_data.getN()(gg, 0));
   

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
      auto t_lag_base_row = row_data.getFTensor1N<3>(gg, 0);
      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        // const double m =
        //     val_m * t_lag_base_row(0) /* const_unit_n(i)*/ * t_base_master;

        auto assemble_mat = get_vec_from_mat(NN, bbr, 3 * bbc);

        assemble_mat(i) += const_unit_n(i) * val_m *
                           t_lag_base_row(0) /* const_unit_n(i)*/ *
                           t_base_master;

        ++t_lag_base_row; // update rows
      }

      ++t_base_master; // update cols master
    }

    ++lambda_gap_diff_prod;
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  // Assemble NN to final Aij vector based on its global indices
  CHKERR MatSetValues(aij, nb_base_fun_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpDerivativeBarTildeCFunODisplacementsSlaveSlave::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
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

  const double area_slave =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  NN.resize(nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto get_vec_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 0, c + 1),
                                         &m(r + 0, c + 2));
  };

  FTensor::Index<'i', 3> i;

  auto lambda_gap_diff_prod =
      getFTensor0FromVec(*commonDataSimpleContact->lambdaGapDiffProductPtr);

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    
    const double val_s = getGaussPtsSlave()(2, gg) * area_slave * cN *
                         (1. + lambda_gap_diff_prod);

    FTensor::Tensor0<double *> t_base_slave(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
      FTensor::Tensor0<double *> t_base_lambda(&row_data.getN()(gg, 0));
      const double s = val_s * t_base_slave ;
      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
      
        auto assemble_mat = get_vec_from_mat(NN, bbr, 3 * bbc);

        assemble_mat(i) -= const_unit_n(i) * s * t_base_lambda;

        ++t_base_lambda; // update rows
      }
      ++t_base_slave; // update cols slave
    }
    ++lambda_gap_diff_prod;
    ++gap_gp;
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  // Assemble NN to final Aij vector based on its global indices
  CHKERR MatSetValues(aij, nb_base_fun_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpDerivativeBarTildeCFunODisplacementsSlaveSlaveALE::
    doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
           DataForcesAndSourcesCore::EntData &row_data,
           DataForcesAndSourcesCore::EntData &col_data) {
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

  NN.resize(nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto get_vec_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 0, c + 1),
                                         &m(r + 0, c + 2));
  };

  FTensor::Index<'i', 3> i;

  auto x_m = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsMasterPtr);
  auto x_s = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsSlavePtr);
  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto lambda_gap_diff_prod =
      getFTensor0FromVec(*commonDataSimpleContact->lambdaGapDiffProductPtr);

  auto tilde_c_fun = getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

  auto length_normal =
      getFTensor0FromVec(*commonDataSimpleContact->normalSlaveLengthALEPtr);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    double val_s = getGaussPtsSlave()(2, gg) * 0.5;

    auto normal_at_gp =
        get_tensor_vec(commonDataSimpleContact->normalSlaveALE[gg]);

    FTensor::Tensor0<double *> t_base_slave(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
      FTensor::Tensor0<double *> t_base_lambda(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        const double s = val_s * t_base_lambda * t_base_slave;

        auto assemble_mat = get_vec_from_mat(NN, bbr, 3 * bbc);

        assemble_mat(i) -=
            normal_at_gp(i) * cN * (1. + lambda_gap_diff_prod) * s;

        ++t_base_lambda; // update rows
      }

      ++t_base_slave; // update cols slave
    }

    ++x_m;
    ++x_s;
    ++lagrange_slave;
    ++gap_gp;
    ++lambda_gap_diff_prod;
    ++tilde_c_fun;
    ++length_normal;
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  // Assemble NN to final Aij vector based on its global indices
  CHKERR MatSetValues(aij, nb_base_fun_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpDerivativeBarTildeCFunODisplacementsSlaveSlaveALE_dX::
    doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
           DataForcesAndSourcesCore::EntData &row_data,
           DataForcesAndSourcesCore::EntData &col_data) {
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

  NN.resize(nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto get_vec_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 0, c + 1),
                                         &m(r + 0, c + 2));
  };

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor_vec_3 = [&](VectorDouble3 &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  auto make_normal_vec_der =
      [&](VectorDouble3 &der_ksi, VectorDouble3 &der_eta,
          VectorDouble3 &normal_der, MatrixDouble &der_normal_mat,
          FTensor::Tensor0<double *> &t_N_over_ksi,
          FTensor::Tensor0<double *> &t_N_over_eta,
          boost::shared_ptr<CommonDataSimpleContact> &commonDataSimpleContact,
          const int &gg) {
        der_normal_mat.clear();

        auto t_tan_1 =
            get_tensor_vec(commonDataSimpleContact->tangentSlaveALE[gg][0]);
        auto t_tan_2 =
            get_tensor_vec(commonDataSimpleContact->tangentSlaveALE[gg][1]);
        for (int dd = 0; dd != 3; ++dd) {

          der_ksi.clear();
          der_eta.clear();
          normal_der.clear();

          der_ksi[dd] = t_N_over_ksi;
          der_eta[dd] = t_N_over_eta;

          auto t_normal_der = get_tensor_vec_3(normal_der);
          auto t_dn_xi = get_tensor_vec_3(der_ksi);
          auto t_dn_eta = get_tensor_vec_3(der_eta);

          t_normal_der(i) +=
              FTensor::levi_civita(i, j, k) * t_dn_xi(j) * t_tan_2(k);

          t_normal_der(i) +=
              FTensor::levi_civita(i, j, k) * t_tan_1(j) * t_dn_eta(k);

          for (int kk = 0; kk != 3; ++kk) {
            der_normal_mat(kk, dd) += t_normal_der(kk);
          }
        }
      };

  auto x_m = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsMasterPtr);
  auto x_s = getFTensor1FromMat<3>(
      *commonDataSimpleContact->positionAtGaussPtsSlavePtr);
  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto lambda_gap_diff_prod =
      getFTensor0FromVec(*commonDataSimpleContact->lambdaGapDiffProductPtr);

  auto length_normal =
      getFTensor0FromVec(*commonDataSimpleContact->normalSlaveLengthALEPtr);

  VectorDouble3 normal_der(3);
  VectorDouble3 der_ksi(3), der_eta(3);

  MatrixDouble der_normal_mat;
  der_normal_mat.resize(3, 3, false);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    double val_s = getGaussPtsSlave()(2, gg) * 0.5;

    auto normal_at_gp =
        get_tensor_vec(commonDataSimpleContact->normalSlaveALE[gg]);

    FTensor::Tensor0<double *> t_N_over_ksi(&col_data.getDiffN()(gg, 0));

    FTensor::Tensor0<double *> t_N_over_eta(&col_data.getDiffN()(gg, 1));

    FTensor::Tensor0<double *> t_base_slave_row_X(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
      FTensor::Tensor0<double *> t_base_lambda(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        const double s = val_s * t_base_lambda /* t_base_slave_row_X*/;

        make_normal_vec_der(der_ksi, der_eta, normal_der, der_normal_mat,
                            t_N_over_ksi, t_N_over_eta, commonDataSimpleContact,
                            gg);

        auto t_d_n = get_tensor_from_mat(der_normal_mat, 0, 0);

        auto assemble_mat = get_vec_from_mat(NN, bbr, 3 * bbc);

        assemble_mat(j) -= t_d_n(i, j) * (x_s(i) - x_m(i)) * cN *
                           (1. + lambda_gap_diff_prod) * s;

        assemble_mat(j) += lagrange_slave * t_d_n(i, j) * normal_at_gp(i) *
                           (1. - lambda_gap_diff_prod) * s / (length_normal);

        ++t_base_lambda; // update rows
      }
      ++t_base_slave_row_X; // update cols slave
      ++t_N_over_ksi;       // move pointers by two
      ++t_N_over_ksi;
      ++t_N_over_eta;
      ++t_N_over_eta;
    }

    ++x_m;
    ++x_s;
    ++lagrange_slave;
    ++lambda_gap_diff_prod;
    ++length_normal;
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  // Assemble NN to final Aij vector based on its global indices
  CHKERR MatSetValues(aij, nb_base_fun_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpDerivativeBarTildeCFunODisplacementsSlaveSlaveHdiv::
    doWork(int row_side, int col_side, EntityType row_type, EntityType col_type,
           DataForcesAndSourcesCore::EntData &row_data,
           DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  if (row_type != MBTRI)
    MoFEMFunctionReturnHot(0);
  const int nb_row = row_data.getIndices().size();
  if (!nb_row)
    MoFEMFunctionReturnHot(0);
  const int nb_col = col_data.getIndices().size();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();
  int nb_base_fun_row = row_data.getFieldData().size();
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

  const double area_slave =
      commonDataSimpleContact->areaSlave; // same area in master and slave

  NN.resize(nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto get_vec_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 0, c + 1),
                                         &m(r + 0, c + 2));
  };

  FTensor::Index<'i', 3> i;

  auto lambda_gap_diff_prod =
      getFTensor0FromVec(*commonDataSimpleContact->lambdaGapDiffProductPtr);
      
  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    double val_s = cN * (1. + lambda_gap_diff_prod) *
                   getGaussPtsSlave()(2, gg) * area_slave;
    
    FTensor::Tensor0<double *> t_base_slave(&col_data.getN()(gg, 0));
    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
      auto t_lag_base_row = row_data.getFTensor1N<3>(gg, 0);
      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        const double s =
            val_s * t_lag_base_row(0) /* const_unit_n(i)*/ * t_base_slave;

        auto assemble_mat = get_vec_from_mat(NN, bbr, 3 * bbc);

        assemble_mat(i) -=
            const_unit_n(i) * s;

        ++t_lag_base_row; // update rows
      }

      ++t_base_slave; // update cols slave
    }

    ++lambda_gap_diff_prod;
  }

  Mat aij;
  if (Aij == PETSC_NULL) {
    aij = getFEMethod()->snes_B;
  } else {
    aij = Aij;
  }

  // Assemble NN to final Aij vector based on its global indices
  CHKERR MatSetValues(aij, nb_base_fun_row, &row_data.getIndices()[0], nb_col,
                      &col_data.getIndices()[0], &*NN.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpMakeVtkSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);
 
  int nb_dofs = data.getFieldData().size();
  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);
  int nb_gauss_pts = data.getN().size1();
  cerr << "CHEK\n";
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

  Tag th_proj_normal_master;
  CHKERR moabOut.tag_get_handle("PROJECTED_NORMAL_MASTER", 1, MB_TYPE_DOUBLE,
                                th_proj_normal_master,
                                MB_TAG_CREAT | MB_TAG_SPARSE, &def_vals);

  Tag th_proj_normal_slave;
  CHKERR moabOut.tag_get_handle("PROJECTED_NORMAL_SLAVE", 1, MB_TYPE_DOUBLE,
                                th_proj_normal_slave,
                                MB_TAG_CREAT | MB_TAG_SPARSE, &def_vals);
  Tag th_lag_gap_prod;
  if (lagFieldSet)
    CHKERR moabOut.tag_get_handle("LAG_GAP_PROD", 1, MB_TYPE_DOUBLE,
                                  th_lag_gap_prod, MB_TAG_CREAT | MB_TAG_SPARSE,
                                  &def_vals);

  Tag th_diff_lag_with_normal_master_traction;
  if (lagFieldSet)
    CHKERR moabOut.tag_get_handle("DIFF_NORMAL_LAG_WITH_MASTER", 1, MB_TYPE_DOUBLE,
                                  th_diff_lag_with_normal_master_traction,
                                  MB_TAG_CREAT | MB_TAG_SPARSE, &def_vals);

  Tag th_diff_lag_with_normal_slave_traction;
  if (lagFieldSet)
    CHKERR moabOut.tag_get_handle("DIFF_NORMAL_LAG_WITH_SLAVE", 1,
                                  MB_TYPE_DOUBLE, th_diff_lag_with_normal_slave_traction,
                                  MB_TAG_CREAT | MB_TAG_SPARSE, &def_vals);

  Tag th_error_lag_master;
  if (lagFieldSet)
    CHKERR moabOut.tag_get_handle("ERROR_LAG_MASTER", 1, MB_TYPE_DOUBLE,
                                  th_error_lag_master,
                                  MB_TAG_CREAT | MB_TAG_SPARSE, &def_vals);

  Tag th_error_lag_slave;
  if (lagFieldSet)
    CHKERR moabOut.tag_get_handle("ERROR_LAG_SLAVE", 1, MB_TYPE_DOUBLE,
                                  th_error_lag_slave,
                                  MB_TAG_CREAT | MB_TAG_SPARSE, &def_vals);

  double coords[3];
  EntityHandle new_vertex = getFEEntityHandle();

  auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  
    auto lagrange_slave =
        getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
    auto lag_gap_prod_slave =
        getFTensor0FromVec(*commonDataSimpleContact->lagGapProdPtr);

  auto projected_stress_master =
      getFTensor0FromVec(*commonDataSimpleContact->projNormalStressAtMaster);

  auto projected_stress_slave =
      getFTensor0FromVec(*commonDataSimpleContact->projNormalStressAtSlave);

  auto diff_normal_lag_master =
      getFTensor0FromVec(*commonDataSimpleContact->diffNormalLagMaster);

  auto diff_normal_lag_slave =
      getFTensor0FromVec(*commonDataSimpleContact->diffNormalLagSlave);

  auto rel_error_master = getFTensor0FromVec(
      *commonDataSimpleContact->relErrorLagNormalStressAtMaster);

  auto rel_error_slave = getFTensor0FromVec(
      *commonDataSimpleContact->relErrorLagNormalStressAtSlave);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    for (int dd = 0; dd != 3; ++dd) {
      coords[dd] = getCoordsAtGaussPtsSlave()(gg, dd);
    }

    VectorDouble &data_disp = data.getFieldData();
    CHKERR moabOut.create_vertex(&coords[0], new_vertex);

    CHKERR moabOut.tag_set_data(th_gap, &new_vertex, 1, &gap_ptr);

    CHKERR moabOut.tag_set_data(th_proj_normal_master, &new_vertex, 1,
                                &projected_stress_master);

    CHKERR moabOut.tag_set_data(th_proj_normal_slave, &new_vertex, 1,
                                &projected_stress_slave);
    if (lagFieldSet){
      CHKERR moabOut.tag_set_data(th_lag_gap_prod, &new_vertex, 1,
                                  &lag_gap_prod_slave);
      CHKERR moabOut.tag_set_data(th_lag_mult, &new_vertex, 1, &lagrange_slave);

      CHKERR moabOut.tag_set_data(th_diff_lag_with_normal_master_traction,
                                  &new_vertex, 1, &diff_normal_lag_master);
      CHKERR moabOut.tag_set_data(th_diff_lag_with_normal_slave_traction,
                                  &new_vertex, 1, &diff_normal_lag_slave);
      CHKERR moabOut.tag_set_data(th_error_lag_slave, &new_vertex, 1,
                                  &rel_error_slave);
      CHKERR moabOut.tag_set_data(th_error_lag_master, &new_vertex, 1,
                                  &rel_error_master);
        }

    // cerr << "Post proc gap " << gap_ptr<<"\n";
        ++projected_stress_slave;
        ++projected_stress_master;
        ++gap_ptr;
        if (lagFieldSet) {
          ++diff_normal_lag_master;
          ++diff_normal_lag_slave;
          ++lagrange_slave;
          ++lag_gap_prod_slave;
          ++rel_error_master;
          ++rel_error_slave;
    }
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactMaterialVolOnMasterSideLhs::aSsemble(
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

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
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (data.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
          data.forcesOnlyOnEntitiesRow.end()) {
        rowIndices[ii] = -1;
      }
    }
  }

  if (!data.forcesOnlyOnEntitiesCol.empty()) {
    colIndices.resize(col_nb_dofs, false);
    noalias(colIndices) = col_data.getIndices();
    col_indices = &colIndices[0];
    VectorDofs &dofs = col_data.getFieldDofs();
    VectorDofs::iterator dit = dofs.begin();
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (data.forcesOnlyOnEntitiesCol.find((*dit)->getEnt()) ==
          data.forcesOnlyOnEntitiesCol.end()) {
        colIndices[ii] = -1;
      }
    }
  }

  Mat B = getFEMethod()->ksp_B != PETSC_NULL ? getFEMethod()->ksp_B
                                             : getFEMethod()->snes_B;
  // assemble local matrix
  CHKERR MatSetValues(B, row_nb_dofs, row_indices, col_nb_dofs, col_indices,
                      &*NN.data().begin(), ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactMaterialVolOnSlaveSideLhs::aSsemble(
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

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
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (data.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
          data.forcesOnlyOnEntitiesRow.end()) {
        rowIndices[ii] = -1;
      }
    }
  }

  if (!data.forcesOnlyOnEntitiesCol.empty()) {
    colIndices.resize(col_nb_dofs, false);
    noalias(colIndices) = col_data.getIndices();
    col_indices = &colIndices[0];
    VectorDofs &dofs = col_data.getFieldDofs();
    VectorDofs::iterator dit = dofs.begin();
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (data.forcesOnlyOnEntitiesCol.find((*dit)->getEnt()) ==
          data.forcesOnlyOnEntitiesCol.end()) {
        colIndices[ii] = -1;
      }
    }
  }

  Mat B = getFEMethod()->ksp_B != PETSC_NULL ? getFEMethod()->ksp_B
                                             : getFEMethod()->snes_B;
  // assemble local matrix
  CHKERR MatSetValues(B, row_nb_dofs, row_indices, col_nb_dofs, col_indices,
                      &*NN.data().begin(), ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
    SimpleContactProblem::OpContactMaterialVolOnMasterSideLhs::doWork(
        int row_side, int col_side, EntityType row_type, EntityType col_type,
        DataForcesAndSourcesCore::EntData &row_data,
        DataForcesAndSourcesCore::EntData &col_data) {

  MoFEMFunctionBegin;

  if (commonDataSimpleContact->faceRowData == nullptr)
    MoFEMFunctionReturnHot(0);

  if (row_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  row_nb_dofs = commonDataSimpleContact->faceRowData->getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);

  nb_gauss_pts = commonDataSimpleContact->faceRowData->getN().size1();

  nb_base_fun_row = commonDataSimpleContact->faceRowData->getFieldData().size() / 3;
  nb_base_fun_col = col_data.getFieldData().size() / 3;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  // integrate local matrix for entity block
  CHKERR iNtegrate(*(commonDataSimpleContact->faceRowData), col_data);

  // assemble local matrix
  CHKERR aSsemble(*(commonDataSimpleContact->faceRowData), col_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactMaterialVolOnSlaveSideLhs::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

  MoFEMFunctionBegin;

  if (commonDataSimpleContact->faceRowData == nullptr)
    MoFEMFunctionReturnHot(0);

  if (row_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  row_nb_dofs = commonDataSimpleContact->faceRowData->getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);

  nb_gauss_pts = commonDataSimpleContact->faceRowData->getN().size1();

  nb_base_fun_row =
      commonDataSimpleContact->faceRowData->getFieldData().size() / 3;
  nb_base_fun_col = col_data.getFieldData().size() / 3;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  // integrate local matrix for entity block
  CHKERR iNtegrate(*(commonDataSimpleContact->faceRowData), col_data);

  // assemble local matrix
  CHKERR aSsemble(*(commonDataSimpleContact->faceRowData), col_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpLoopMasterForSideLhs::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  if (row_type != MBTRI || col_type != MBTRI)
    MoFEMFunctionReturnHot(0);

  // const int row_nb_dofs = row_data.getIndices().size();
  // if (!row_nb_dofs)
  //   MoFEMFunctionReturnHot(0);
  // const int col_nb_dofs = col_data.getIndices().size();
  // if (!col_nb_dofs)
  //   MoFEMFunctionReturnHot(0);

  commonDataSimpleContact->faceRowData = &row_data;
  const EntityHandle tri_master = getSideEntity(3, MBTRI);
  CHKERR loopSideVolumes(sideFeName, *sideFe, 3, tri_master);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpGetRowColDataForSideFace::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  if (row_type != MBTRI || col_type != MBTRI)
    MoFEMFunctionReturnHot(0);
  cerr << " DATA !!! \n";
  commonDataSimpleContact->l2RowData = &row_data;

  commonDataSimpleContact->l2ColData = &col_data;

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpLoopMasterForSideLhsTest::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  if (row_type != MBVERTEX || col_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  const int row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  const int col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);

  //commonDataSimpleContact->faceRowData = &row_data;
  const EntityHandle tri_master = getSideEntity(3, MBTRI);
  CHKERR loopSideVolumes(sideFeName, *sideFe, 3, tri_master);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpLoopSlaveForSideLhsTest::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  if (row_type != MBVERTEX || col_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  const int row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  const int col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);

  // commonDataSimpleContact->faceRowData = &row_data;
  const EntityHandle tri_master = getSideEntity(4, MBTRI);
  CHKERR loopSideVolumes(sideFeName, *sideFe, 3, tri_master);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpLoopMasterSlaveForSideLhsTest::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  // if (row_type != MBTRI || col_type != MBTRI)
  //   MoFEMFunctionReturnHot(0);

  const int row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  const int col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);

  commonDataSimpleContact->faceRowData = &row_data;
  const EntityHandle tri_slave = getSideEntity(4, MBTRI);
  CHKERR loopSideVolumes(sideFeName, *sideFe, 3, tri_slave);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpLoopSlaveMasterForSideLhsTest::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  // if (row_type != MBTRI || col_type != MBTRI)
  //   MoFEMFunctionReturnHot(0);

  const int row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  const int col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);

  commonDataSimpleContact->faceRowData = &row_data;
  const EntityHandle tri_master = getSideEntity(4, MBTRI);
  CHKERR loopSideVolumes(sideFeName, *sideFe, 3, tri_master);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpLoopSlaveForSideLhs::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  // row_nb_dofs = row_data.getIndices().size();
  // if (!row_nb_dofs)
  //   MoFEMFunctionReturnHot(0);
  // col_nb_dofs = col_data.getIndices().size();
  // if (!col_nb_dofs)
  //   MoFEMFunctionReturnHot(0);
  if (row_type != MBTRI || col_type != MBTRI)
    MoFEMFunctionReturnHot(0);

  const EntityHandle tri_slave = getSideEntity(4, MBTRI);
  CHKERR loopSideVolumes(sideFeName, *sideFe, 3, tri_slave);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpContactMaterialMasterLhs_dX_dX::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

  MoFEMFunctionBegin;

  row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);
  nb_gauss_pts = row_data.getN().size1();

  nb_base_fun_row = row_data.getFieldData().size() / 3;
  nb_base_fun_col = col_data.getFieldData().size() / 3;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  diagonal_block = (row_type == col_type) && (row_side == col_side);

  if (col_type == MBVERTEX) {
    commonDataSimpleContact->faceRowData = &row_data;
    const EntityHandle tri_master = getSideEntity(3, MBTRI);
    CHKERR loopSideVolumes(sideFeName, *sideFe, 3, tri_master);
  }

  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data, col_data);

  // assemble local matrix
  CHKERR aSsemble(row_data, col_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpContactMaterialMasterSlaveLhs_dX_dLagmult::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

  MoFEMFunctionBegin;

  row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);
  nb_gauss_pts = row_data.getN().size1();

  nb_base_fun_row = row_data.getFieldData().size() / 3;
  nb_base_fun_col = col_data.getFieldData().size();

  NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
  NN.clear();

  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data, col_data);

  // assemble local matrix
  CHKERR aSsemble(row_data, col_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactMaterialMasterSlaveLhs_dX_dX::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

  MoFEMFunctionBegin;

  row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);
  nb_gauss_pts = row_data.getN().size1();

  nb_base_fun_row = row_data.getFieldData().size() / 3;
  nb_base_fun_col = col_data.getFieldData().size() / 3;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data, col_data);

  // assemble local matrix
  CHKERR aSsemble(row_data, col_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactMaterialSlaveSlaveLhs_dX_dLagmult::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

  MoFEMFunctionBegin;

  row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);
  nb_gauss_pts = row_data.getN().size1();

  nb_base_fun_row = row_data.getFieldData().size() / 3;
  nb_base_fun_col = col_data.getFieldData().size();

  NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
  NN.clear();

  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data, col_data);

  // assemble local matrix
  CHKERR aSsemble(row_data, col_data);

  MoFEMFunctionReturn(0);
}


MoFEMErrorCode
SimpleContactProblem::OpContactMaterialMasterLhs_dX_dX::iNtegrate(
    DataForcesAndSourcesCore::EntData &row_data, DataForcesAndSourcesCore::EntData &col_data) {

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

  auto get_tensor_vec_3 = [&](VectorDouble3 &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto make_normal_vec_der =
      [&](VectorDouble3 &der_ksi, VectorDouble3 &der_eta,
          VectorDouble3 &normal_der, MatrixDouble &der_normal_mat,
          FTensor::Tensor0<double *> &t_N_over_ksi,
          FTensor::Tensor0<double *> &t_N_over_eta,
          boost::shared_ptr<CommonDataSimpleContact> &commonDataSimpleContact,
          const int &gg) {
        der_normal_mat.clear();

        auto t_tan_1 =
            get_tensor_vec(commonDataSimpleContact->tangentMasterALE[gg][0]);
        auto t_tan_2 =
            get_tensor_vec(commonDataSimpleContact->tangentMasterALE[gg][1]);
        for (int dd = 0; dd != 3; ++dd) {

          der_ksi.clear();
          der_eta.clear();
          normal_der.clear();

          der_ksi[dd] = t_N_over_ksi;
          der_eta[dd] = t_N_over_eta;

          auto t_normal_der = get_tensor_vec_3(normal_der);
          auto t_dn_xi = get_tensor_vec_3(der_ksi);
          auto t_dn_eta = get_tensor_vec_3(der_eta);

          t_normal_der(i) +=
              FTensor::levi_civita(i, j, k) * t_dn_xi(j) * t_tan_2(k);

          t_normal_der(i) +=
              FTensor::levi_civita(i, j, k) * t_tan_1(j) * t_dn_eta(k);

          for (int kk = 0; kk != 3; ++kk) {
            der_normal_mat(kk, dd) += t_normal_der(kk);
          }
        }
      };

  MatrixDouble der_normal_mat;
  der_normal_mat.resize(3, 3, false);

  VectorDouble3 normal_der(3);
  VectorDouble3 der_ksi(3);
  VectorDouble3 der_eta(3);

  commonDataSimpleContact->faceRowData = nullptr;
  
  auto t_F = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->FMat);

  auto t_w = getFTensor0IntegrationWeightMaster();
  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    const double val = 0.5 * t_w * lagrange_slave;
    FTensor::Tensor0<double *> t_N_over_ksi(&col_data.getDiffN()(gg, 0));
    FTensor::Tensor0<double *> t_N_over_eta(&col_data.getDiffN()(gg, 1));

    int bbc = 0;
    for (; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; bbr++) {

        make_normal_vec_der(der_ksi, der_eta, normal_der, der_normal_mat,
                            t_N_over_ksi, t_N_over_eta, commonDataSimpleContact, gg);

        auto d_n = get_tensor2(der_normal_mat, 0, 0);

        auto t_assemble = get_tensor2(NN, 3 * bbr, 3 * bbc);
        // TODO: handle hoGeometry

        t_assemble(i, k) -=
             val * t_base * t_F(j, i) * d_n(j, k);

        ++t_base;
      }
      ++t_N_over_ksi;
      ++t_N_over_ksi;
      ++t_N_over_eta;
      ++t_N_over_eta;
    }
    ++t_F;
    ++t_w;
    ++lagrange_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactMaterialMasterSlaveLhs_dX_dLagmult::iNtegrate(
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

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

  commonDataSimpleContact->faceRowData = nullptr;

  auto t_F = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->FMat);

  auto t_w = getFTensor0IntegrationWeightMaster();

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    auto normal_master_at_gp =
        get_tensor_vec(commonDataSimpleContact->normalMasterALE[gg]);

    FTensor::Tensor0<double *> t_col_base(&col_data.getN()(gg, 0));

    const double val = 0.5 * t_w;
    
    int bbc = 0;
    for (; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_row_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; bbr++) {

        auto t_assemble = get_tensor_from_mat(NN, 3 * bbr, bbc);
        // TODO: handle hoGeometry

        t_assemble(i) -= val * t_row_base * t_F(j, i) *
                            normal_master_at_gp(j) * t_col_base;
                            
        // cerr << "~~~~~~~~~~" << "\n";

        // cerr << NN <<"\n";

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
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

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

  commonDataSimpleContact->faceRowData = nullptr;

  auto t_F = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->FMat);

  auto t_w = getFTensor0IntegrationWeightSlave();

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    auto normal_slave_at_gp =
        get_tensor_vec(commonDataSimpleContact->normalSlaveALE[gg]);

    FTensor::Tensor0<double *> t_col_base(&col_data.getN()(gg, 0));

    const double val = 0.5 * t_w;

    int bbc = 0;
    for (; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_row_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; bbr++) {

        auto t_assemble = get_tensor_from_mat(NN, 3 * bbr, bbc);
        // TODO: handle hoGeometry

        t_assemble(i) -=
            val * t_row_base * t_F(j, i) * normal_slave_at_gp(j) * t_col_base;

        // cerr << "~~~~~~~~~~" << "\n";

        // cerr << NN <<"\n";

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
SimpleContactProblem::OpContactMaterialMasterSlaveLhs_dX_dX::iNtegrate(
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

  MoFEMFunctionBegin;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'m', 3> m;
  FTensor::Index<'k', 3> k;

  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor1<double *, 3>(&m(r + 0, c + 0), &m(r + 1, c + 0),
                                         &m(r + 2, c + 0));
  };

  auto get_tensor_vec_3 = [&](VectorDouble3 &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto make_normal_vec_der =
      [&](VectorDouble3 &der_ksi, VectorDouble3 &der_eta,
          VectorDouble3 &normal_der, MatrixDouble &der_normal_mat,
          FTensor::Tensor0<double *> &t_N_over_ksi,
          FTensor::Tensor0<double *> &t_N_over_eta,
          boost::shared_ptr<CommonDataSimpleContact> &commonDataSimpleContact,
          const int &gg) {
        der_normal_mat.clear();

        auto t_tan_1 =
            get_tensor_vec(commonDataSimpleContact->tangentSlaveALE[gg][0]);
        auto t_tan_2 =
            get_tensor_vec(commonDataSimpleContact->tangentSlaveALE[gg][1]);
        for (int dd = 0; dd != 3; ++dd) {

          der_ksi.clear();
          der_eta.clear();
          normal_der.clear();

          der_ksi[dd] = t_N_over_ksi;
          der_eta[dd] = t_N_over_eta;

          auto t_normal_der = get_tensor_vec_3(normal_der);
          auto t_dn_xi = get_tensor_vec_3(der_ksi);
          auto t_dn_eta = get_tensor_vec_3(der_eta);

          t_normal_der(i) +=
              FTensor::levi_civita(i, j, k) * t_dn_xi(j) * t_tan_2(k);

          t_normal_der(i) +=
              FTensor::levi_civita(i, j, k) * t_tan_1(j) * t_dn_eta(k);

          for (int kk = 0; kk != 3; ++kk) {
            der_normal_mat(kk, dd) += t_normal_der(kk);
          }
        }
      };

  VectorDouble3 normal_der(3);
  VectorDouble3 der_ksi(3), der_eta(3);

  MatrixDouble der_normal_mat;
  der_normal_mat.resize(3, 3, false);

  commonDataSimpleContact->faceRowData = nullptr;

  auto t_F = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->FMat);

  auto t_w = getFTensor0IntegrationWeightMaster();

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto length_normal =
      getFTensor0FromVec(*commonDataSimpleContact->normalSlaveLengthALEPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    auto normal_master_at_gp =
        get_tensor_vec(commonDataSimpleContact->normalMasterALE[gg]);

    auto normal_slave_at_gp =
        get_tensor_vec(commonDataSimpleContact->normalSlaveALE[gg]);

    const double val = 0.5 * t_w * lagrange_slave;

    FTensor::Tensor0<double *> t_N_over_ksi(&col_data.getDiffN()(gg, 0));

    FTensor::Tensor0<double *> t_N_over_eta(&col_data.getDiffN()(gg, 1));

    int bbc = 0;
    for (; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_row_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; bbr++) {

        make_normal_vec_der(der_ksi, der_eta, normal_der, der_normal_mat,
                            t_N_over_ksi, t_N_over_eta, commonDataSimpleContact,
                            gg);

        auto t_d_n = get_tensor2(der_normal_mat, 0, 0);

        auto t_assemble = get_tensor2(NN, 3 * bbr, 3 * bbc);
        // TODO: handle hoGeometry

        t_assemble(i, k) -= val * t_row_base * t_F(j, i) *
                            normal_master_at_gp(j) * normal_slave_at_gp(m) *
                            t_d_n(m, k) ;// / length_normal;

        ++t_row_base;
      }
      ++t_N_over_ksi;
      ++t_N_over_ksi;
      ++t_N_over_eta;
      ++t_N_over_eta;
    }
    ++t_F;
    ++t_w;
    ++lagrange_slave;
    ++length_normal;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpContactMaterialMasterLhs::aSsemble(
    DataForcesAndSourcesCore::EntData &row_data, DataForcesAndSourcesCore::EntData &col_data) {

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
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (data.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
          data.forcesOnlyOnEntitiesRow.end()) {
        rowIndices[ii] = -1;
      }
    }
  }

  if (!data.forcesOnlyOnEntitiesCol.empty()) {
    colIndices.resize(col_nb_dofs, false);
    noalias(colIndices) = col_data.getIndices();
    col_indices = &colIndices[0];
    VectorDofs &dofs = col_data.getFieldDofs();
    VectorDofs::iterator dit = dofs.begin();
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (data.forcesOnlyOnEntitiesCol.find((*dit)->getEnt()) ==
          data.forcesOnlyOnEntitiesCol.end()) {
        colIndices[ii] = -1;
      }
    }
  }

  Mat B = getFEMethod()->ksp_B != PETSC_NULL ? getFEMethod()->ksp_B
                                             : getFEMethod()->snes_B;
  // assemble local matrix
  CHKERR MatSetValues(B, row_nb_dofs, row_indices, col_nb_dofs, col_indices,
                      &*NN.data().begin(), ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpContactMaterialMasterSlaveLhs::aSsemble(
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

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
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (data.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
          data.forcesOnlyOnEntitiesRow.end()) {
        rowIndices[ii] = -1;
      }
    }
  }

  if (!data.forcesOnlyOnEntitiesCol.empty()) {
    colIndices.resize(col_nb_dofs, false);
    noalias(colIndices) = col_data.getIndices();
    col_indices = &colIndices[0];
    VectorDofs &dofs = col_data.getFieldDofs();
    VectorDofs::iterator dit = dofs.begin();
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (data.forcesOnlyOnEntitiesCol.find((*dit)->getEnt()) ==
          data.forcesOnlyOnEntitiesCol.end()) {
        colIndices[ii] = -1;
      }
    }
  }

  Mat B = getFEMethod()->ksp_B != PETSC_NULL ? getFEMethod()->ksp_B
                                             : getFEMethod()->snes_B;
  // assemble local matrix
  CHKERR MatSetValues(B, row_nb_dofs, row_indices, col_nb_dofs, col_indices,
                      &*NN.data().begin(), ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactMaterialVolOnMasterSideLhs_dX_dx::iNtegrate(
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

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

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    auto normal_at_gp =
        get_tensor_vec(commonDataSimpleContact->normalMasterALE[gg]);

    double a = -0.5 * t_w * lagrange_slave;
    
    auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

    int bbc = 0;
    for (; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_row_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; bbr++) {

        auto t_assemble = get_tensor2(NN, 3 * bbr, 3 * bbc);
        // TODO: handle hoGeometry

        t_assemble(i, j) +=
            a * t_row_base * t_inv_H(k, i) * t_col_diff_base(k) *
            normal_at_gp(j);

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
SimpleContactProblem::OpContactMaterialVolOnSlaveSideLhs_dX_dx::iNtegrate(
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

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

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    auto normal_at_gp =
        get_tensor_vec(commonDataSimpleContact->normalSlaveALE[gg]);

    double a = -0.5 * t_w * lagrange_slave;

    auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

    int bbc = 0;
    for (; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_row_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; bbr++) {

        auto t_assemble = get_tensor2(NN, 3 * bbr, 3 * bbc);
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

MoFEMErrorCode SimpleContactProblem::OpContactMaterialSlaveLhs_dX_dx::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

  MoFEMFunctionBegin;

  if (col_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  commonDataSimpleContact->faceRowData = &row_data;
  const EntityHandle tri_slave = getSideEntity(4, MBTRI);
  CHKERR loopSideVolumes(sideFeName, *sideFe, 3, tri_slave);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpContactMaterialSlaveLhs::aSsemble(
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

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
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (data.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
          data.forcesOnlyOnEntitiesRow.end()) {
        rowIndices[ii] = -1;
      }
    }
  }

  if (!data.forcesOnlyOnEntitiesCol.empty()) {
    colIndices.resize(col_nb_dofs, false);
    noalias(colIndices) = col_data.getIndices();
    col_indices = &colIndices[0];
    VectorDofs &dofs = col_data.getFieldDofs();
    VectorDofs::iterator dit = dofs.begin();
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (data.forcesOnlyOnEntitiesCol.find((*dit)->getEnt()) ==
          data.forcesOnlyOnEntitiesCol.end()) {
        colIndices[ii] = -1;
      }
    }
  }

  Mat B = getFEMethod()->ksp_B != PETSC_NULL ? getFEMethod()->ksp_B
                                             : getFEMethod()->snes_B;
  // assemble local matrix
  CHKERR MatSetValues(B, row_nb_dofs, row_indices, col_nb_dofs, col_indices,
                      &*NN.data().begin(), ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpContactMaterialSlaveLhs_dX_dX::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

  MoFEMFunctionBegin;

  row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);
  nb_gauss_pts = row_data.getN().size1();

  nb_base_fun_row = row_data.getFieldData().size() / 3;
  nb_base_fun_col = col_data.getFieldData().size() / 3;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  diagonal_block = (row_type == col_type) && (row_side == col_side);

  if (col_type == MBVERTEX) {
    commonDataSimpleContact->faceRowData = &row_data;
    const EntityHandle tri_slave = getSideEntity(4, MBTRI);
    CHKERR loopSideVolumes(sideFeName, *sideFe, 3, tri_slave);
  }

  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data, col_data);

  // assemble local matrix
  CHKERR aSsemble(row_data, col_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactMaterialSlaveLhs_dX_dX::iNtegrate(
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

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

  auto get_tensor_vec_3 = [&](VectorDouble3 &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto make_normal_vec_der =
      [&](VectorDouble3 &der_ksi, VectorDouble3 &der_eta,
          VectorDouble3 &normal_der, MatrixDouble &der_normal_mat,
          FTensor::Tensor0<double *> &t_N_over_ksi,
          FTensor::Tensor0<double *> &t_N_over_eta,
          boost::shared_ptr<CommonDataSimpleContact> &commonDataSimpleContact,
          const int &gg) {
        der_normal_mat.clear();

        auto t_tan_1 =
            get_tensor_vec(commonDataSimpleContact->tangentSlaveALE[gg][0]);
        auto t_tan_2 =
            get_tensor_vec(commonDataSimpleContact->tangentSlaveALE[gg][1]);
        for (int dd = 0; dd != 3; ++dd) {

          der_ksi.clear();
          der_eta.clear();
          normal_der.clear();

          der_ksi[dd] = t_N_over_ksi;
          der_eta[dd] = t_N_over_eta;

          auto t_normal_der = get_tensor_vec_3(normal_der);
          auto t_dn_xi = get_tensor_vec_3(der_ksi);
          auto t_dn_eta = get_tensor_vec_3(der_eta);

          t_normal_der(i) +=
              FTensor::levi_civita(i, j, k) * t_dn_xi(j) * t_tan_2(k);

          t_normal_der(i) +=
              FTensor::levi_civita(i, j, k) * t_tan_1(j) * t_dn_eta(k);

          for (int kk = 0; kk != 3; ++kk) {
            der_normal_mat(kk, dd) += t_normal_der(kk);
          }
        }
      };

  MatrixDouble der_normal_mat;
  der_normal_mat.resize(3, 3, false);

  VectorDouble3 normal_der(3);
  VectorDouble3 der_ksi(3);
  VectorDouble3 der_eta(3);

  commonDataSimpleContact->faceRowData = nullptr;

  auto t_F = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->FMat);

  auto t_w = getFTensor0IntegrationWeightSlave();
  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    const double val = 0.5 * t_w * lagrange_slave;
    FTensor::Tensor0<double *> t_N_over_ksi(&col_data.getDiffN()(gg, 0));
    FTensor::Tensor0<double *> t_N_over_eta(&col_data.getDiffN()(gg, 1));

    int bbc = 0;
    for (; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; bbr++) {

        make_normal_vec_der(der_ksi, der_eta, normal_der, der_normal_mat,
                            t_N_over_ksi, t_N_over_eta, commonDataSimpleContact,
                            gg);

        auto d_n = get_tensor2(der_normal_mat, 0, 0);

        auto t_assemble = get_tensor2(NN, 3 * bbr, 3 * bbc);
        // TODO: handle hoGeometry

        t_assemble(i, k) -= val * t_base * t_F(j, i) * d_n(j, k);

        ++t_base;
      }
      ++t_N_over_ksi;
      ++t_N_over_ksi;
      ++t_N_over_eta;
      ++t_N_over_eta;
    }
    ++t_F;
    ++t_w;
    ++lagrange_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpContactMaterialMasterLhs_dX_dx::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

  MoFEMFunctionBegin;

  
  if (col_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

   commonDataSimpleContact->faceRowData = &row_data;
   const EntityHandle tri_master = getSideEntity(3, MBTRI);
   CHKERR loopSideVolumes(sideFeName, *sideFe, 3, tri_master);

   MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactMaterialVolOnMasterSideLhs_dX_dX::iNtegrate(
    DataForcesAndSourcesCore::EntData &row_data, DataForcesAndSourcesCore::EntData &col_data) {

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

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    auto normal_at_gp =
        get_tensor_vec(commonDataSimpleContact->normalMasterALE[gg]);

    double a = -0.5 * t_w * lagrange_slave;

    auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

    int bbc = 0;
    for (; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_row_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; bbr++) {

        auto t_assemble = get_tensor2(NN, 3 * bbr, 3 * bbc);

        // TODO: handle hoGeometry

        t_assemble(i, j) += -1.0 * a * t_row_base * t_inv_H(l, j) *
                            t_col_diff_base(m) * t_inv_H(m, i) * t_h(k, l) *
                            normal_at_gp(k);

        ++t_row_base;
      }
      ++t_col_diff_base;
    }
    ++t_w;
    ++t_h;
    ++t_inv_H;
    // ++t_normal;
    ++lagrange_slave;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SimpleContactProblem::OpContactMaterialVolOnSlaveSideLhs_dX_dX::iNtegrate(
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

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

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    auto normal_at_gp =
        get_tensor_vec(commonDataSimpleContact->normalSlaveALE[gg]);

    double a = -0.5 * t_w * lagrange_slave;

    auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

    int bbc = 0;
    for (; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_row_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; bbr++) {

        auto t_assemble = get_tensor2(NN, 3 * bbr, 3 * bbc);

        // TODO: handle hoGeometry

        t_assemble(i, j) += -1.0 * a * t_row_base * t_inv_H(l, j) *
                            t_col_diff_base(m) * t_inv_H(m, i) * t_h(k, l) *
                            normal_at_gp(k);

        ++t_row_base;
      }
      ++t_col_diff_base;
    }
    ++t_w;
    ++t_h;
    ++t_inv_H;
    // ++t_normal;
    ++lagrange_slave;
  }

  MoFEMFunctionReturn(0);
}

// MoFEMErrorCode
// SimpleContactProblem::OpContactMaterialVolOnMasterSideLhs_dX_dLagmult::iNtegrate(
//     DataForcesAndSourcesCore::EntData &row_data,
//     DataForcesAndSourcesCore::EntData &col_data) {

//   MoFEMFunctionBegin;

//   FTensor::Index<'i', 3> i;
//   FTensor::Index<'j', 3> j;
//   FTensor::Index<'k', 3> k;

//   auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
//     return FTensor::Tensor2<double *, 3, 3>(
//         &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
//         &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
//         &m(r + 2, c + 2));
//   };

//   auto get_tensor_vec = [](VectorDouble &n) {
//     return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
//   };

//   auto t_w = getFTensor0IntegrationWeight();

//   FTensor::Tensor2<double, 3, 3> t_d;

//   auto t_F = getFTensor2FromMat<3, 3>(*commonDataSimpleContact->FMat);

//   for (int gg = 0; gg != nb_gauss_pts; gg++) {

//     auto normal_at_gp =
//         get_tensor_vec(commonDataSimpleContact->normalMasterALE[gg]);

//     double a = -0.5 * t_w;

//     FTensor::Tensor0<double *> t_col_base(&col_data.getN()(gg, 0));

//     int bbc = 0;
//     for (; bbc != nb_base_fun_col; bbc++) {

//       FTensor::Tensor0<double *> t_row_base(&row_data.getN()(gg, 0));

//       int bbr = 0;
//       for (; bbr != nb_base_fun_row; bbr++) {

//         auto t_assemble = get_tensor2(NN, 3 * bbr, 3 * bbc);

//         // TODO: handle hoGeometry

//         t_assemble(i, k) -= a * t_row_base * t_col_base * t_F(j, i) *
//                             normal_at_gp(j) * normal_at_gp(k);

//         ++t_row_base;
//       }
//       ++t_col_base;
//     }
//     ++t_w;
//   }

//   MoFEMFunctionReturn(0);
// }
