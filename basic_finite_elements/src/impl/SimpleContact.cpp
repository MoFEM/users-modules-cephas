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

MoFEMErrorCode SimpleContactProblem::CommonDataSimpleContact::getContactTags(
    const EntityHandle fe_ent, const int nb_gauss_pts,
    boost::shared_ptr<MatrixDouble> tag_ptr, Tag tag_ref) {
  MoFEMFunctionBegin;
  double *tag_data;
  int tag_size;
  CHKERR mField.get_moab().tag_get_by_ptr(tag_ref, &fe_ent, 1,
                                          (const void **)&tag_data, &tag_size);

  if (tag_size == 1) {
    tag_ptr->resize(9, nb_gauss_pts, false);
    tag_ptr->clear();
    void const *tag_data[] = {&*tag_ptr->data().begin()};
    const int tag_size = tag_ptr->data().size();
    CHKERR mField.get_moab().tag_set_by_ptr(tag_ref, &fe_ent, 1, tag_data,
                                            &tag_size);
  } else if (tag_size != nb_gauss_pts * 9) {
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "Wrong size of the tag, data inconsistency");
  } else {
    MatrixAdaptor tag_vec =
        MatrixAdaptor(9, nb_gauss_pts,
                      ublas::shallow_array_adaptor<double>(tag_size, tag_data));

    *tag_ptr = tag_vec;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleContactProblem::CommonDataSimpleContact::getContactTags(
    const EntityHandle fe_ent, const int nb_gauss_pts,
    boost::shared_ptr<VectorDouble> &tag_ptr, Tag tag_ref) {
  MoFEMFunctionBegin;
  double *tag_data;
  int tag_size;
  CHKERR mField.get_moab().tag_get_by_ptr(tag_ref, &fe_ent, 1,
                                          (const void **)&tag_data, &tag_size);

  if (tag_size == 1) {
    tag_ptr->resize(nb_gauss_pts);
    tag_ptr->clear();
    void const *tag_data[] = {&*tag_ptr->begin()};
    const int tag_size = tag_ptr->size();
    CHKERR mField.get_moab().tag_set_by_ptr(tag_ref, &fe_ent, 1, tag_data,
                                            &tag_size);
  } else if (tag_size != nb_gauss_pts) {
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "Wrong size of the tag, data inconsistency");
  } else {
    VectorAdaptor tag_vec = VectorAdaptor(
        tag_size, ublas::shallow_array_adaptor<double>(tag_size, tag_data));
    *tag_ptr = tag_vec;
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
  commonDataSimpleContact->normalVectorPtr.get()->resize(3);
  commonDataSimpleContact->normalVectorPtr.get()->clear();

  auto normal =
      get_tensor_vec(commonDataSimpleContact->normalVectorPtr.get()[0]);
  for (int ii = 0; ii != 3; ++ii)
    normal(ii) = normal_slave_ptr[ii];

  const double normal_length = sqrt(normal(i) * normal(i));
  normal(i) = normal(i) / normal_length;
  commonDataSimpleContact->areaCommon = 0.5 * normal_length;

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
      get_tensor_vec(commonDataSimpleContact->normalVectorPtr.get()[0]);

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
          get_tensor_vec(commonDataSimpleContact->normalVectorPtr.get()[0]);

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
          commonDataSimpleContact->areaCommon; // same area in master and slave

      auto get_tensor_vec = [](VectorDouble &n, const int r) {
        return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
      };

      FTensor::Index<'i', 3> i;

      auto lagrange_slave =
          getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

      auto const_unit_n =
          get_tensor_vec(commonDataSimpleContact->normalVectorPtr.get()[0], 0);

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

      const int nb_col = data.getIndices().size();

      CHKERR VecSetValues(getFEMethod()->snes_f, nb_col, &data.getIndices()[0],
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

      const double *normal_f4_ptr = &getNormalSlave()[0];

      const double area_s =
          commonDataSimpleContact->areaCommon; // same area in master and slave

      auto get_tensor_vec = [](VectorDouble &n, const int r) {
        return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1), &n(r + 2));
      };

      FTensor::Index<'i', 3> i;
      auto lagrange_slave =
          getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

      auto const_unit_n =
          get_tensor_vec(commonDataSimpleContact->normalVectorPtr.get()[0], 0);

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

      CHKERR VecSetValues(getFEMethod()->snes_f, nb_col, &data.getIndices()[0],
                          &vec_f[0], ADD_VALUES);
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
          commonDataSimpleContact->areaCommon; // same area in master and slave

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

      auto tilde_c_fun_2 =
          getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

      CHKERR VecSetValues(getFEMethod()->snes_f, nb_base_fun_col,
                          &data.getIndices()[0], &vecR[0], ADD_VALUES);

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
          commonDataSimpleContact->areaCommon; // same area in master and slave

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
          get_tensor_vec(commonDataSimpleContact->normalVectorPtr.get()[0]);

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

      CHKERR VecSetValues(getFEMethod()->snes_f, nb_base_fun_col,
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

      const double area_common =
          commonDataSimpleContact->areaCommon; // same area in master and slave

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
          get_tensor_vec(commonDataSimpleContact->normalVectorPtr.get()[0]);

      for (int gg = 0; gg != nb_gauss_pts; gg++) {
        double val_m = getGaussPtsMaster()(2, gg) * area_common;

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

      if (Aij == PETSC_NULL) {
        Aij = getFEMethod()->snes_B;
      }

      CHKERR MatSetValues(
          getFEMethod()->snes_B, nb_row, &row_data.getIndices()[0], nb_col,
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

    const double area_common =
        commonDataSimpleContact->areaCommon; // same area in master and slave

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
        get_tensor_vec(commonDataSimpleContact->normalVectorPtr.get()[0]);

    auto t_lag_base = col_data.getFTensor1N<3>();

    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      double val_m = getGaussPtsMaster()(2, gg) * area_common;

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

    if (Aij == PETSC_NULL) {
      Aij = getFEMethod()->snes_B;
    }

    CHKERR MatSetValues(
        getFEMethod()->snes_B, nb_row, &row_data.getIndices()[0], nb_col,
        &col_data.getIndices()[0], &*NN.data().begin(), ADD_VALUES);

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

    const double area_common =
        commonDataSimpleContact->areaCommon; // same area in master and slave

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
        get_tensor_vec(commonDataSimpleContact->normalVectorPtr.get()[0]);

    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      double val_s = getGaussPtsSlave()(2, gg) * area_common;

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

    if (Aij == PETSC_NULL) {
      Aij = getFEMethod()->snes_B;
    }

    CHKERR MatSetValues(
        getFEMethod()->snes_B, nb_row, &row_data.getIndices()[0], nb_col,
        &col_data.getIndices()[0], &*NN.data().begin(), ADD_VALUES);

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

    const double area_common =
        commonDataSimpleContact->areaCommon; // same area in master and slave

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
        get_tensor_vec(commonDataSimpleContact->normalVectorPtr.get()[0]);

    auto t_lag_base = col_data.getFTensor1N<3>();

    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      double val_s = getGaussPtsSlave()(2, gg) * area_common;

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

    if (Aij == PETSC_NULL) {
      Aij = getFEMethod()->snes_B;
    }

    CHKERR MatSetValues(
        getFEMethod()->snes_B, nb_row, &row_data.getIndices()[0], nb_col,
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

    const double area_common =
        commonDataSimpleContact->areaCommon; // same area in master and slave

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
      const double val_s = getGaussPtsSlave()(2, gg) * area_common;

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

    CHKERR MatSetValues(
        getFEMethod()->snes_B, nb_base_fun_row, &row_data.getIndices()[0],
        nb_base_fun_col, // ign: is shift row right here?
        &col_data.getIndices()[0], &*NN.data().begin(), ADD_VALUES);

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

    const double area_common =
        commonDataSimpleContact->areaCommon; // same area in master and slave

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
        get_tensor_vec(commonDataSimpleContact->normalVectorPtr.get()[0]);

    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      const double val_s = getGaussPtsSlave()(2, gg) * area_common;

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

    CHKERR MatSetValues(
        getFEMethod()->snes_B, nb_base_fun_row, &row_data.getIndices()[0],
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

    const double area_common =
        commonDataSimpleContact->areaCommon; // same area in master and slave

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
        get_tensor_from_vec(commonDataSimpleContact->normalVectorPtr.get()[0]);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      double val_m = getGaussPtsMaster()(2, gg) * area_common;

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

    // Assemble NN to final Aij vector based on its global indices
    CHKERR MatSetValues(
        getFEMethod()->snes_B, nb_base_fun_row, &row_data.getIndices()[0],
        nb_col, &col_data.getIndices()[0], &*NN.data().begin(), ADD_VALUES);

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

    const double area_common =
        commonDataSimpleContact->areaCommon; // same area in master and slave

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
        get_tensor_from_vec(commonDataSimpleContact->normalVectorPtr.get()[0]);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      double val_m = getGaussPtsMaster()(2, gg) * area_common;

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

    // Assemble NN to final Aij vector based on its global indices
    CHKERR MatSetValues(
        getFEMethod()->snes_B, nb_base_fun_row, &row_data.getIndices()[0],
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

    const double area_common =
        commonDataSimpleContact->areaCommon; // same area in master and slave

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
        get_tensor_vec(commonDataSimpleContact->normalVectorPtr.get()[0]);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      double val_s = getGaussPtsSlave()(2, gg) * area_common;

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

    // Assemble NN to final Aij vector based on its global indices
    CHKERR MatSetValues(
        getFEMethod()->snes_B, nb_base_fun_row, &row_data.getIndices()[0],
        nb_col, &col_data.getIndices()[0], &*NN.data().begin(), ADD_VALUES);

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

    const double area_common =
        commonDataSimpleContact->areaCommon; // same area in master and slave

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
        get_tensor_vec(commonDataSimpleContact->normalVectorPtr.get()[0]);

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      double val_s = getGaussPtsSlave()(2, gg) * area_common;

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

    // Assemble NN to final Aij vector based on its global indices
    CHKERR MatSetValues(
        getFEMethod()->snes_B, nb_base_fun_row, &row_data.getIndices()[0],
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

    // double def_vals[9];
    // bzero(def_vals, 9 * sizeof(double));

    // Tag th_strain;
    // CHKERR moabOut.tag_get_handle("PLASTIC_STRAIN0", 9, MB_TYPE_DOUBLE,
    //                               th_strain, MB_TAG_CREAT | MB_TAG_SPARSE,
    //                               def_vals);

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
    // CHKERR commonDataSimpleContact->getContactTags(
    //     new_vertex, nb_gauss_pts, commonDataSimpleContact->gapPtr,
    //     commonDataSimpleContact->thGap);

    // CHKERR commonDataSimpleContact->getContactTags(
    //     new_vertex, nb_gauss_pts,
    //     commonDataSimpleContact->lagMultAtGaussPtsPtr,
    //     commonDataSimpleContact->thLagrangeMultiplier);

    // CHKERR commonDataSimpleContact->getContactTags(
    //     new_vertex, nb_gauss_pts,
    //     commonDataSimpleContact->lagGapProdPtr,
    //     commonDataSimpleContact->thLagGapProd);

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