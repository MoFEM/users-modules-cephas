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

MoFEMErrorCode SimpleContactProblem::OpGetLagMulAtGaussPtsSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

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
      coords[dd] = all_coords[ii * 3 + dd];
    }
    lagrange_multipliers[0] = data.getFieldData()[ii];

    CHKERR moabOut.create_vertex(&coords[0], new_vertex);
    CHKERR moabOut.tag_set_data(th_lagmult, &new_vertex, 1,
                                lagrange_multipliers);
  }
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
    const double regular_abs = abs(lambda_gap_diff);

    tilde_c_fun =
        (lagrange_slave + cg - pow(regular_abs, r) / r); // is lagMult Correct?

    const double exponent = r - 1.;

    double sign = 0.;
    sign = (lambda_gap_diff == 0) ? 0 : (lambda_gap_diff < 0) ? -1 : 1;

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

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto lambda_gap_diff_prod =
      getFTensor0FromVec(*commonDataSimpleContact->lambdaGapDiffProductPtr);

  auto tilde_c_fun = getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

  auto const_unit_n = get_tensor_from_vec(
      commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    double val_m = getGaussPtsMaster()(2, gg) * area_slave;

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
            const_unit_n(i) * cN * (1. + lambda_gap_diff_prod) * m;

        ++t_base_lambda; // update rows
      }

      ++t_base_master; // update cols master

      ++t_field_data_master;
    }

    ++tilde_c_fun;
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

  auto const_unit_n =
      get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    double val_s = getGaussPtsSlave()(2, gg) * area_slave;

    FTensor::Tensor0<double *> t_base_slave(&col_data.getN()(gg, 0));

    for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
      FTensor::Tensor0<double *> t_base_lambda(&row_data.getN()(gg, 0));

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
        const double s = val_s * t_base_lambda * t_base_slave;

        auto assemble_mat = get_vec_from_mat(NN, bbr, 3 * bbc);

        assemble_mat(i) -=
            const_unit_n(i) * cN * (1. + lambda_gap_diff_prod) * s;

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

  double coords[3];
  EntityHandle new_vertex = getFEEntityHandle();

  auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
  auto lag_gap_prod_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagGapProdPtr);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    for (int dd = 0; dd != 3; ++dd) {
      coords[dd] = getCoordsAtGaussPtsSlave()(gg, dd);
    }

    VectorDouble &data_disp = data.getFieldData();
    CHKERR moabOut.create_vertex(&coords[0], new_vertex);

    CHKERR moabOut.tag_set_data(th_gap, &new_vertex, 1, &gap_ptr);

    if (lagFieldSet) {
      CHKERR moabOut.tag_set_data(th_lag_gap_prod, &new_vertex, 1,
                                  &lag_gap_prod_slave);
      CHKERR moabOut.tag_set_data(th_lag_mult, &new_vertex, 1, &lagrange_slave);
    }

    // cerr << "Post proc gap " << gap_ptr<<"\n";

    ++gap_ptr;

    if (lagFieldSet) {
      ++lagrange_slave;
      ++lag_gap_prod_slave;
    }
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::OpMakeTestTextFile::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  if (type != MBVERTEX)
    MoFEMFunctionReturnHot(0);
  int nb_dofs = data.getFieldData().size();
  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);
  int nb_gauss_pts = data.getN().size1();
  auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);
  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
  double d_lambda, d_gap;
  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    d_lambda = fabs(lagrange_slave) < 1e-16 ? 0.0 : lagrange_slave;
    d_gap = fabs(gap_ptr) < 1e-16 ? 0.0 : gap_ptr;
    mySplit << d_lambda << " " << d_gap << " " << std::endl;
    ++gap_ptr;
    ++lagrange_slave;
  }
  MoFEMFunctionReturn(0);
}