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
  SimpleContactProblem::SimpleContactElement::gaussPtsMaster.resize(3, nb_gauss_pts, false);
  SimpleContactProblem::SimpleContactElement::gaussPtsSlave.resize(3, nb_gauss_pts, false);
  double xy_coords[2 * nb_gauss_pts];
  double w_array[nb_gauss_pts];
  triangle_ncc_rule(rule, nb_gauss_pts, xy_coords, w_array);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    SimpleContactProblem::SimpleContactElement::gaussPtsMaster(0, gg) = xy_coords[gg * 2];
    SimpleContactProblem::SimpleContactElement::gaussPtsMaster(1, gg) = xy_coords[gg * 2 + 1];
    SimpleContactProblem::SimpleContactElement::gaussPtsMaster(2, gg) = w_array[gg];
    SimpleContactProblem::SimpleContactElement::gaussPtsSlave(0, gg) = xy_coords[gg * 2];
    SimpleContactProblem::SimpleContactElement::gaussPtsSlave(1, gg) = xy_coords[gg * 2 + 1];
    SimpleContactProblem::SimpleContactElement::gaussPtsSlave(2, gg) = w_array[gg];
  }

  MoFEMFunctionReturn(0);
}


PetscErrorCode SimpleContactProblem::OpGetNormalSlave::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    PetscFunctionReturn(0);

  if (type != MBVERTEX)
    PetscFunctionReturn(0);

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

PetscErrorCode SimpleContactProblem::OpGetNormalMaster::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    PetscFunctionReturn(0);

  if (type != MBVERTEX)
    PetscFunctionReturn(0);

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

PetscErrorCode SimpleContactProblem::OpGetNormalSlaveForSide::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    PetscFunctionReturn(0);

  if (type != MBVERTEX)
    PetscFunctionReturn(0);

  FTensor::Index<'i', 3> i;

  auto get_tensor_vec = [](VectorDouble &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };
  
  CHKERR loopSideVolumes(sideFeName, *sideFe, 0);

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

  //CHKERR loopSideVolumes(sideFeName, *sideFe);

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

  CHKERR MatSetValues(aij, nb_row, &row_data.getIndices()[0],
                      nb_col, &col_data.getIndices()[0], &*NN.data().begin(),
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
    PetscFunctionReturn(0);

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
    PetscFunctionReturn(0);

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

  const int nb_gauss_pts = data.getN().size1();

  if (type == MBVERTEX) {
    commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->resize(nb_gauss_pts);
    commonDataSimpleContact->lagMultAtGaussPtsPtr
        .get()
        ->clear();
  }

  int nb_base_fun_row = data.getFieldData().size();

  auto lagrange_slave =
      getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    FTensor::Tensor0<double *> t_base_lambda(
        &data.getN()(gg, 0));

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

    MoFEMErrorCode SimpleContactProblem::OpGetLagMulAtGaussPtsSlaveHdiv::doWork(
        int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;

      if (type != MBTRI)
        MoFEMFunctionReturnHot(0);

      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      const int nb_gauss_pts = data.getN().size1();
      commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->resize(nb_gauss_pts);
      commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->clear();

      int nb_base_fun_row = data.getFieldData().size();

      auto lagrange_slave =
          getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

      auto t_lag_base = data.getFTensor1N<3>();

      auto get_tensor_vec = [](VectorDouble &n) {
        return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
      };

      FTensor::Index<'i', 3> i;
      auto normal_at_gp =
          get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

      for (int gg = 0; gg != nb_gauss_pts; gg++) {
        FTensor::Tensor0<double *> t_field_data_slave(&data.getFieldData()[0]);
        for (int bb = 0; bb != nb_base_fun_row; bb++) {
          auto t_lag_base_help = data.getFTensor1N<3>(gg, bb);
          lagrange_slave +=
              t_field_data_slave * normal_at_gp(i) * t_lag_base(i);
          ++t_field_data_slave;
          ++t_lag_base;
        }
        ++lagrange_slave;
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

    PetscErrorCode SimpleContactProblem::OpCalFReConMaster::doWork(
        int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
      PetscFunctionBegin;

      if (data.getIndices().size() == 0)
        PetscFunctionReturn(0);

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

      CHKERR VecSetValues(f, nb_col, &data.getIndices()[0],
                          &vec_f[0], ADD_VALUES);
      PetscFunctionReturn(0);
    }

    PetscErrorCode SimpleContactProblem::OpCalFReConSlave::doWork(
        int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
      PetscFunctionBegin;

      if (data.getIndices().size() == 0)
        PetscFunctionReturn(0);

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

      CHKERR VecSetValues(f, nb_col, &data.getIndices()[0],
                          &vec_f[0], ADD_VALUES);
      PetscFunctionReturn(0);
    }

    PetscErrorCode SimpleContactProblem::OpCalFReConSlaveALE::doWork(
        int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
      PetscFunctionBegin;

      if (data.getIndices().size() == 0)
        PetscFunctionReturn(0);

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

        CHKERR VecSetValues(f, nb_col, &data.getIndices()[0], &vec_f[0],
                            ADD_VALUES);
        PetscFunctionReturn(0);
    }

    PetscErrorCode SimpleContactProblem::OpCalFReConMasterALE::doWork(
        int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
      PetscFunctionBegin;

      if (data.getIndices().size() == 0)
        PetscFunctionReturn(0);

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

      CHKERR VecSetValues(f, nb_col, &data.getIndices()[0], &vec_f[0],
                          ADD_VALUES);
      PetscFunctionReturn(0);
    }

    MoFEMErrorCode SimpleContactProblem::OpCalTildeCFunSlave::doWork(
        int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;

      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      if (type != MBVERTEX)
        MoFEMFunctionReturnHot(0);

      const int nb_gauss_pts = data.getN().size1();

      commonDataSimpleContact->tildeCFunPtr.get()->resize(nb_gauss_pts);
      commonDataSimpleContact->tildeCFunPtr.get()->clear();

      commonDataSimpleContact->lambdaGapDiffProductPtr.get()->resize(
          nb_gauss_pts);
      commonDataSimpleContact->lambdaGapDiffProductPtr.get()->clear();

      auto lambda_gap_diff_prod =
          getFTensor0FromVec(*commonDataSimpleContact->lambdaGapDiffProductPtr);

      auto lagrange_slave =
          getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

      auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

      auto tilde_c_fun =
          getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

      for (int gg = 0; gg != nb_gauss_pts; gg++) {
        const double cg = cN * gap_gp;

        const double lambda_gap_diff = lagrange_slave - cg;
        const double regular_abs = abs(lambda_gap_diff);

        tilde_c_fun = (lagrange_slave + cg -
                       pow(regular_abs, r) / r); // is lagMult Correct?

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

      auto tilde_c_fun =
          getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

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

      CHKERR VecSetValues(f, nb_base_fun_col,
                          &data.getIndices()[0], &vecR[0], ADD_VALUES);

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

      auto tilde_c_fun =
          getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

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

      if (type != MBTRI)
        MoFEMFunctionReturnHot(0);

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

      auto tilde_c_fun =
          getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

      auto t_lag_base = data.getFTensor1N<3>();

      auto get_tensor_vec = [](VectorDouble &n) {
        return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
      };

      FTensor::Index<'i', 3> i;
      auto normal_at_gp =
          get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

      for (int gg = 0; gg != nb_gauss_pts; gg++) {
        double val_s = getGaussPtsSlave()(2, gg) * area_s;

        FTensor::Tensor0<double *> t_base_lambda(&data.getN()(gg, 0));

        for (int bbr = 0; bbr != nb_base_fun_col; bbr++) {
          const double s =
              val_s * t_lag_base(i) * normal_at_gp(i) * tilde_c_fun;
          vecR[bbr] += s; // it is a plus since F always scales with -1
          ++t_lag_base;
        }
        ++tilde_c_fun;
      } // for gauss points

      auto tilde_c_fun_2 =
          getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

      Vec f;
      if (F == PETSC_NULL) {
        f = getFEMethod()->snes_f;
      } else {
        f = F;
      }

      CHKERR VecSetValues(f, nb_base_fun_col,
                          &data.getIndices()[0], &vecR[0], ADD_VALUES);

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

      CHKERR MatSetValues(
          aij, nb_row, &row_data.getIndices()[0], nb_col,
          &col_data.getIndices()[0], &*NN.data().begin(), ADD_VALUES);

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

    auto t_lag_base = col_data.getFTensor1N<3>();

    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      double val_m = getGaussPtsMaster()(2, gg) * area_slave;

      for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
        FTensor::Tensor0<double *> t_base_master(&row_data.getN()(gg, 0));

        for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
          const double m =
              val_m * const_unit_n(i) * t_lag_base(i) * t_base_master;

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

    CHKERR MatSetValues(
        aij, nb_row, &row_data.getIndices()[0], nb_col,
        &col_data.getIndices()[0], &*NN.data().begin(), ADD_VALUES);

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

  PetscErrorCode SimpleContactProblem::OpGetTangentMaster::doWork(
      int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    if (data.getFieldData().size() == 0)
      PetscFunctionReturn(0);

    const int ngp =
        data.getN().size1(); // this is number of base functions for
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


    PetscErrorCode 
    SimpleContactProblem::OpGetTangentSlave::doWork(
        int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;

      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);
      
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

    PetscErrorCode SimpleContactProblem::OpGetNormalSlaveALE::doWork(
        int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;

      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      if (type != MBVERTEX)
        PetscFunctionReturn(0);

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

          f4_length_ptr =
              sqrt(normal_original_slave(i) * normal_original_slave(i));

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
            &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2),
            &m(r + 1, c + 0), &m(r + 1, c + 1), &m(r + 1, c + 2),
            &m(r + 2, c + 0), &m(r + 2, c + 1), &m(r + 2, c + 2));
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

      auto make_normal_vec_der = [&](VectorDouble3 &der_ksi,
                                     VectorDouble3 &der_eta,
                                     VectorDouble3 &normal_der,
                                     MatrixDouble &der_normal_mat,
                                     FTensor::Tensor0<double *> &t_N_over_ksi,
                                     FTensor::Tensor0<double *> &t_N_over_eta,
                                     boost::shared_ptr<CommonDataSimpleContact>
                                         &commonDataSimpleContact,
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
                                t_N_over_ksi, t_N_over_eta,
                                commonDataSimpleContact, gg);

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
          &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2),
          &m(r + 1, c + 0), &m(r + 1, c + 1), &m(r + 1, c + 2),
          &m(r + 2, c + 0), &m(r + 2, c + 1), &m(r + 2, c + 2));
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
                              t_N_over_ksi, t_N_over_eta,
                              commonDataSimpleContact, gg);

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

    auto t_lag_base = col_data.getFTensor1N<3>();

    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      double val_s = getGaussPtsSlave()(2, gg) * area_slave;

      for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

        FTensor::Tensor0<double *> t_base_slave(&row_data.getN()(gg, 0));

        for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
          const double s =
              val_s * t_lag_base(i) * const_unit_n(i) * t_base_slave;

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

    CHKERR MatSetValues(
        aij, nb_row, &row_data.getIndices()[0], nb_col,
        &col_data.getIndices()[0], &*NN.data().begin(), ADD_VALUES);

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

    CHKERR MatSetValues(
        aij, nb_base_fun_row, &row_data.getIndices()[0],
        nb_base_fun_col, // ign: is shift row right here?
        &col_data.getIndices()[0], &*NN.data().begin(), ADD_VALUES);

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
                              (const_unit_n(i) * t_lag_base_row(i)) *
                              (t_lag_base_col(i) * const_unit_n(i));
            }

          } else {
            if (fabs(gap_gp) < 1.e-8 && fabs(lagrange_slave) < 1.e-8) {
            } else {
              NN(bbr, bbc) += (1. - lambda_gap_diff_prod) * val_s *
                              (const_unit_n(i) * t_lag_base_row(i)) *
                              (t_lag_base_col(i) * const_unit_n(i));
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
    CHKERR MatSetValues(
        aij, nb_base_fun_row, &row_data.getIndices()[0],
        nb_base_fun_col, // ign: is shift row right here?
        &col_data.getIndices()[0], &*NN.data().begin(), ADD_VALUES);

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

    auto tilde_c_fun =
        getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

    auto const_unit_n =
        get_tensor_from_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

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
    CHKERR MatSetValues(
        aij, nb_base_fun_row, &row_data.getIndices()[0],
        nb_col, &col_data.getIndices()[0], &*NN.data().begin(), ADD_VALUES);

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode
  SimpleContactProblem::OpDerivativeBarTildeCFunODisplacementsSlaveMasterALE::
      doWork(int row_side, int col_side, EntityType row_type,
             EntityType col_type, DataForcesAndSourcesCore::EntData &row_data,
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

    auto tilde_c_fun =
        getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

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
      CHKERR MatSetValues(aij, nb_base_fun_row, &row_data.getIndices()[0],
                          nb_col, &col_data.getIndices()[0],
                          &*NN.data().begin(), ADD_VALUES);

      MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode
  SimpleContactProblem::OpDerivativeBarTildeCFunODisplacementsSlaveMasterALE_dX::
      doWork(int row_side, int col_side, EntityType row_type,
             EntityType col_type, DataForcesAndSourcesCore::EntData &row_data,
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

    auto tilde_c_fun =
        getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

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

  MoFEMErrorCode SimpleContactProblem::
      OpDerivativeBarTildeCFunODisplacementsSlaveMasterHdiv::doWork(
          int row_side, int col_side, EntityType row_type, EntityType col_type,
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

    auto lagrange_slave =
        getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

    auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    auto lambda_gap_diff_prod =
        getFTensor0FromVec(*commonDataSimpleContact->lambdaGapDiffProductPtr);

    auto tilde_c_fun =
        getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

    auto const_unit_n =
        get_tensor_from_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      double val_m = getGaussPtsMaster()(2, gg) * area_slave;

      FTensor::Tensor0<double *> t_base_master(&col_data.getN()(gg, 0));

      FTensor::Tensor1<double *, 3> t_field_data_master(
          &col_data.getFieldData()[0], &col_data.getFieldData()[1],
          &col_data.getFieldData()[2], 3);

      for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
        auto t_lag_base_row = row_data.getFTensor1N<3>(gg, 0);
        for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
          const double m =
              val_m * t_lag_base_row(i) * const_unit_n(i) * t_base_master;

          auto assemble_mat = get_vec_from_mat(NN, bbr, 3 * bbc);

          assemble_mat(i) +=
              const_unit_n(i) * cN * (1. + lambda_gap_diff_prod) * m;

          ++t_lag_base_row; // update rows
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
    CHKERR MatSetValues(
        aij, nb_base_fun_row, &row_data.getIndices()[0],
        nb_col, &col_data.getIndices()[0], &*NN.data().begin(), ADD_VALUES);

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

    auto tilde_c_fun =
        getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

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
    CHKERR MatSetValues(
        aij, nb_base_fun_row, &row_data.getIndices()[0],
        nb_col, &col_data.getIndices()[0], &*NN.data().begin(), ADD_VALUES);

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode
  SimpleContactProblem::OpDerivativeBarTildeCFunODisplacementsSlaveSlaveALE::
      doWork(int row_side, int col_side, EntityType row_type,
             EntityType col_type, DataForcesAndSourcesCore::EntData &row_data,
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

    auto tilde_c_fun =
        getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

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
      CHKERR MatSetValues(aij, nb_base_fun_row, &row_data.getIndices()[0],
                          nb_col, &col_data.getIndices()[0],
                          &*NN.data().begin(), ADD_VALUES);

      MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode
  SimpleContactProblem::OpDerivativeBarTildeCFunODisplacementsSlaveSlaveALE_dX::
      doWork(int row_side, int col_side, EntityType row_type,
             EntityType col_type, DataForcesAndSourcesCore::EntData &row_data,
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
          &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2),
          &m(r + 1, c + 0), &m(r + 1, c + 1), &m(r + 1, c + 2),
          &m(r + 2, c + 0), &m(r + 2, c + 1), &m(r + 2, c + 2));
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
                              t_N_over_ksi, t_N_over_eta,
                              commonDataSimpleContact, gg);

          auto t_d_n = get_tensor_from_mat(der_normal_mat, 0, 0);

          auto assemble_mat = get_vec_from_mat(NN, bbr, 3 * bbc);

          assemble_mat(j) -= t_d_n(i, j) * (x_s(i) - x_m(i)) * cN *
                             (1. + lambda_gap_diff_prod) * s;

          assemble_mat(j) += lagrange_slave * t_d_n(i, j) * normal_at_gp(i) *
                             (1. - lambda_gap_diff_prod) * s  /
                             (length_normal );

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
      doWork(int row_side, int col_side, EntityType row_type,
             EntityType col_type, DataForcesAndSourcesCore::EntData &row_data,
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

    auto x_m = getFTensor1FromMat<3>(
        *commonDataSimpleContact->positionAtGaussPtsMasterPtr);
    auto x_s = getFTensor1FromMat<3>(
        *commonDataSimpleContact->positionAtGaussPtsSlavePtr);
    auto lagrange_slave =
        getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

    auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

    auto lambda_gap_diff_prod =
        getFTensor0FromVec(*commonDataSimpleContact->lambdaGapDiffProductPtr);

    auto tilde_c_fun =
        getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

    auto const_unit_n =
        get_tensor_vec(commonDataSimpleContact->normalVectorSlavePtr.get()[0]);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      double val_s = getGaussPtsSlave()(2, gg) * area_slave;

      FTensor::Tensor0<double *> t_base_slave(&col_data.getN()(gg, 0));

      for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
        auto t_lag_base_row = row_data.getFTensor1N<3>(gg, 0);

        for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
          const double s =
              val_s * t_lag_base_row(i) * const_unit_n(i) * t_base_slave;

          auto assemble_mat = get_vec_from_mat(NN, bbr, 3 * bbc);

          assemble_mat(i) -=
              const_unit_n(i) * cN * (1. + lambda_gap_diff_prod) * s;

          ++t_lag_base_row; // update rows
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
    CHKERR MatSetValues(
        aij, nb_base_fun_row, &row_data.getIndices()[0],
        nb_col, &col_data.getIndices()[0], &*NN.data().begin(), ADD_VALUES);

    MoFEMFunctionReturn(0);
  }

  PetscErrorCode SimpleContactProblem::OpMakeVtkSlave::doWork(
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
    CHKERR moabOut.tag_get_handle("LAGRANGE_MULTIPLIER", 1, MB_TYPE_DOUBLE,
                                  th_lag_mult, MB_TAG_CREAT | MB_TAG_SPARSE,
                                  &def_vals);
    Tag th_lag_gap_prod;
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

      CHKERR moabOut.tag_set_data(th_lag_mult, &new_vertex, 1, &lagrange_slave);

      CHKERR moabOut.tag_set_data(th_lag_gap_prod, &new_vertex, 1,
                                  &lag_gap_prod_slave);

      // cerr << "Post proc gap " << gap_ptr<<"\n";

      ++gap_ptr;
      ++lagrange_slave;
      ++lag_gap_prod_slave;
    }
    MoFEMFunctionReturn(0);
  }