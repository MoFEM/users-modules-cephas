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

  fe_rhs->getOpPtrVector().push_back(
      new OpGetPressureGradAtGaussPtsSlave(pressure_field_name, common_data));

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
  
  cout << "Normal: " << t_normal(0) << " " << t_normal(1) << " " << t_normal(2) << endl;

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
    
    cout << "Slave: " << t_position_slave_gp << " | Master: " << t_position_master_gp << " | Gap: " << t_gap_ptr << endl;

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

  auto t_pressure =
      getFTensor0FromVec(*commonData->pressureAtGaussPtsPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    FTensor::Tensor0<double *> t_base(&data.getN()(gg, 0));

    FTensor::Tensor0<double *> t_field_data(&data.getFieldData()[0]);
    for (int bb = 0; bb != nb_base_fun_row; bb++) {
      t_pressure += t_base * t_field_data;
      ++t_base;
      ++t_field_data;
    }
    cout << "Pressure: " << t_pressure <<  endl;
    ++t_pressure;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThinFluidFlowProblem::OpGetPressureGradAtGaussPtsSlave::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  const int nb_gauss_pts = data.getN().size1();

  if (type == MBVERTEX) {
    commonData->pressureGradAtGaussPtsPtr.get()->resize(2, nb_gauss_pts);
    commonData->pressureGradAtGaussPtsPtr.get()->clear();
  }

  int nb_base_fun_row = data.getFieldData().size();

  auto t_pressure_grad =
      getFTensor1FromMat<2>(*commonData->pressureGradAtGaussPtsPtr);

  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;

  FTensor::Tensor2<double, 2, 2> t_jac;
  FTensor::Tensor1<double, 2> t_base_diff_jac;

  for (int ii = 0; ii < 2; ii++) {
    for (int jj = 0; jj < 2; jj++) {
      t_jac(ii, jj) = getInvJacMaster()(ii, jj);
    }
  }

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    auto t_base_diff = data.getFTensor1DiffN<2>(gg, 0);

    FTensor::Tensor0<double *> t_field_data(&data.getFieldData()[0]);

    for (int bb = 0; bb != nb_base_fun_row; bb++) {
      t_base_diff_jac(i) = t_jac(i, j) * t_base_diff(j);
      cout << t_field_data << " | " << t_base_diff(0) << " " << t_base_diff(1) << endl;
      t_pressure_grad(i) += t_base_diff_jac(i) * t_field_data;
      ++t_base_diff;
      ++t_field_data;
    }
    cout << "Press Grad:" << t_pressure_grad(0) << " " << t_pressure_grad(1) << endl;
    ++t_pressure_grad;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThinFluidFlowProblem::setPostProcOperators(
    boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc_ptr,
    const std::string spatial_field_name, const std::string pressure_field_name,
    boost::shared_ptr<CommonData> common_data) {
  MoFEMFunctionBegin;

  post_proc_ptr->getOpPtrVector().push_back(
      new OpCalPressurePostProc(pressure_field_name,
                                    common_data));

  post_proc_ptr->getOpPtrVector().push_back(
      new OpPostProcContinuous(
          pressure_field_name, spatial_field_name, post_proc_ptr->postProcMesh,
          post_proc_ptr->mapGaussPts, common_data));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThinFluidFlowProblem::OpCalPressurePostProc::doWork(
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

MoFEMErrorCode ThinFluidFlowProblem::OpPostProcContinuous::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBVERTEX)
    PetscFunctionReturn(0);

  double def_VAL[9];
  bzero(def_VAL, 9 * sizeof(double));

  Tag th_lag_mult;

  CHKERR postProcMesh.tag_get_handle("PRESSURE", 1, MB_TYPE_DOUBLE,
                                     th_lag_mult, MB_TAG_CREAT | MB_TAG_SPARSE,
                                     def_VAL);

  auto t_lagrange =
      getFTensor0FromVec(*commonData->pressureAtGaussPtsPtr);

  const int nb_gauss_pts =
      commonData->pressureAtGaussPtsPtr->size();

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    CHKERR postProcMesh.tag_set_data(th_lag_mult, &mapGaussPts[gg], 1,
                                     &t_lagrange);
    ++t_lagrange;
  }

  MoFEMFunctionReturn(0);
}


// MoFEMErrorCode SimpleContactProblem::OpCalContactTractionOnSlave::doWork(
//     int side, EntityType type, EntData &data) {
//   MoFEMFunctionBegin;

//   const int nb_dofs = data.getIndices().size();
//   if (nb_dofs) {

//     const int nb_gauss_pts = data.getN().size1();
//     int nb_base_fun_col = nb_dofs / 3;

//     vecF.resize(nb_dofs, false);
//     vecF.clear();

//     const double area_m =
//         commonData->areaSlave; // same area in master and slave

//     auto get_tensor_vec = [](VectorDouble &n, const int r) {
//       return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
//     };

//     FTensor::Index<'i', 3> i;

//     auto t_lagrange_slave =
//         getFTensor0FromVec(*commonData->lagMultAtGaussPtsPtr);

//     auto t_const_unit_n = get_tensor_vec(
//         commonData->normalVectorSlavePtr.get()[0], 0);

//     auto t_w = getFTensor0IntegrationWeightSlave();

//     for (int gg = 0; gg != nb_gauss_pts; ++gg) {

//       double val_m = t_w * area_m;

//       auto t_base_slave = data.getFTensor0N(gg, 0);
//       FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_assemble_m{
//           &vecF[0], &vecF[1], &vecF[2]};

//       for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
//         const double m = val_m * t_base_slave * t_lagrange_slave;
//         t_assemble_m(i) += m * t_const_unit_n(i);
//         ++t_base_slave;
//         ++t_assemble_m;
//       }

//       ++t_lagrange_slave;
//       ++t_w;
//     } // for gauss points

//     CHKERR VecSetValues(getSNESf(), data, &vecF[0], ADD_VALUES);
//   }
//   MoFEMFunctionReturn(0);
// }


