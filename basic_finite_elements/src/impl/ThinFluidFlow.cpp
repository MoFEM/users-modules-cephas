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
#include <ThinFluidFlow.hpp>

using namespace MoFEM;
using namespace boost::numeric;

MoFEMErrorCode ThinFluidFlowProblem::setThinFluidFlowOperatorsRhs(
    boost::shared_ptr<ThinFluidFlowElement> fe_rhs,
    boost::shared_ptr<CommonData> common_data,
    string spatial_field_name, string pressure_field_name) {
  MoFEMFunctionBegin;

  fe_rhs->getOpPtrVector().push_back(
      new OpGetNormalSlave(spatial_field_name, common_data));

  fe_rhs->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsMaster(spatial_field_name, common_data));

  fe_rhs->getOpPtrVector().push_back(
      new OpGetPositionAtGaussPtsSlave(spatial_field_name, common_data));

  fe_rhs->getOpPtrVector().push_back(
      new OpGetGapSlave(spatial_field_name, common_data));

  fe_rhs->getOpPtrVector().push_back(
      new OpGetPressureAtGaussPtsSlave(pressure_field_name, common_data));

  MoFEMFunctionReturn(0);
}


MoFEMErrorCode ThinFluidFlowProblem::OpGetNormalSlave::doWork(int side,
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

  commonData->normalVectorSlavePtr.get()->resize(3);
  commonData->normalVectorSlavePtr.get()->clear();

  auto t_normal =
      get_tensor_vec(*(commonData->normalVectorSlavePtr));

  for (int ii = 0; ii != 3; ++ii)
    t_normal(ii) = normal_slave_ptr[ii];

  const double normal_length = sqrt(t_normal(i) * t_normal(i));
  t_normal(i) = t_normal(i) / normal_length;

  commonData->areaSlave = 0.5 * normal_length;

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThinFluidFlowProblem::OpGetPositionAtGaussPtsSlave::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;
  const int nb_dofs = data.getFieldData().size();

  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();
  if (type == MBVERTEX) {
    commonData->positionAtGaussPtsSlavePtr.get()->resize(
        3, nb_gauss_pts, false);
    commonData->positionAtGaussPtsSlavePtr.get()->clear();
  }

  auto t_position_slave = getFTensor1FromMat<3>(
      *commonData->positionAtGaussPtsSlavePtr);

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

MoFEMErrorCode ThinFluidFlowProblem::OpGetPositionAtGaussPtsMaster::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;
  const int nb_dofs = data.getFieldData().size();

  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  if (type == MBVERTEX) {
    commonData->positionAtGaussPtsMasterPtr.get()->resize(
        3, nb_gauss_pts, false);

    commonData->positionAtGaussPtsMasterPtr.get()->clear();
  }

  auto t_position_master = getFTensor1FromMat<3>(
      *commonData->positionAtGaussPtsMasterPtr);

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

MoFEMErrorCode ThinFluidFlowProblem::OpGetGapSlave::doWork(int side,
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

  commonData->gapPtr.get()->resize(nb_gauss_pts);
  commonData->gapPtr.get()->clear();

  FTensor::Index<'i', 3> i;

  auto t_position_master_gp = getFTensor1FromMat<3>(
      *commonData->positionAtGaussPtsMasterPtr);
  auto t_position_slave_gp = getFTensor1FromMat<3>(
      *commonData->positionAtGaussPtsSlavePtr);

  auto t_gap_ptr = getFTensor0FromVec(*commonData->gapPtr);

  auto t_normal_at_gp =
      get_tensor_vec(*(commonData->normalVectorSlavePtr));

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    t_gap_ptr -=
        t_normal_at_gp(i) * (t_position_slave_gp(i) - t_position_master_gp(i));
    ++t_position_slave_gp;
    ++t_position_master_gp;
    ++t_gap_ptr;
  } // for gauss points

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThinFluidFlowProblem::OpGetPressureAtGaussPtsSlave::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  if (type == MBVERTEX) {
    commonData->pressureAtGaussPtsPtr.get()->resize(nb_gauss_pts);
    commonData->pressureAtGaussPtsPtr.get()->clear();
  }

  int nb_base_fun_row = data.getFieldData().size();

  auto t_lagrange_slave =
      getFTensor0FromVec(*commonData->pressureAtGaussPtsPtr);

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


