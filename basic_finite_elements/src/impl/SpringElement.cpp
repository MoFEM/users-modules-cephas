/* \file SpringElement.cpp
  \brief Implementation of spring boundary condition on triangular surfaces
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

#include <SpringElement.hpp>
using namespace boost::numeric;

MoFEMErrorCode MetaSpringBC::OpSpringFs::doWork(int side, EntityType type,
                                                EntData &data) {

  MoFEMFunctionBegin;
  // check that the faces have associated degrees of freedom
  const int nb_dofs = data.getIndices().size();
  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);
  if (dAta.tRis.find(getFEEntityHandle()) == dAta.tRis.end()) {
    MoFEMFunctionReturnHot(0);
  }
  CHKERR commonDataPtr->getBlockData(dAta);

  // size of force vector associated to the entity
  // set equal to the number of degrees of freedom of associated with the
  // entity
  nF.resize(nb_dofs, false);
  nF.clear();

  // get number of Gauss points
  const int nb_gauss_pts = data.getN().size1();

  // get integration weights
  auto t_w = getFTensor0IntegrationWeight();

  // FTensor indices
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  const double normal_stiffness = commonDataPtr->springStiffnessNormal;
  const double tangent_stiffness = commonDataPtr->springStiffnessTangent;

  // create a 3d vector to be used as the normal to the face with length equal
  // to the face area
  FTensor::Tensor1<double, 3> t_normal;
  FTensor::Tensor2<double, 3, 3> t_normal_projection;
  FTensor::Tensor2<double, 3, 3> t_tangent_projection;

  // Extract solution at Gauss points
  auto t_solution_at_gauss_point =
      getFTensor1FromMat<3>(*commonDataPtr->xAtPts);
  auto t_init_solution_at_gauss_point =
      getFTensor1FromMat<3>(*commonDataPtr->xInitAtPts);
  FTensor::Tensor1<double, 3> t_displacement_at_gauss_point;

  auto t_normal_1 = getFTensor1FromMat<3>(commonDataPtr->normalVector);
  // loop over all Gauss points of the face
  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    t_normal(i) = t_normal_1(i);

    const double normal_length = sqrt(t_normal(k) * t_normal(k));
    t_normal(i) = t_normal(i) / normal_length;
    t_normal_projection(i, j) = t_normal(i) * t_normal(j);
    t_tangent_projection(i, j) = t_normal_projection(i, j);
    t_tangent_projection(0, 0) -= 1;
    t_tangent_projection(1, 1) -= 1;
    t_tangent_projection(2, 2) -= 1;
    // Calculate the displacement at the Gauss point
    if (is_spatial_position) { // "SPATIAL_POSITION"
      t_displacement_at_gauss_point(i) =
          t_solution_at_gauss_point(i) - t_init_solution_at_gauss_point(i);
    } else { // e.g. "DISPLACEMENT" or "U"
      t_displacement_at_gauss_point(i) = t_solution_at_gauss_point(i);
    }

    double w = t_w * 0.5 * normal_length; // area was constant

    auto t_base_func = data.getFTensor0N(gg, 0);

    // create a vector t_nf whose pointer points an array of 3 pointers
    // pointing to nF  memory location of components
    FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf(&nF[0], &nF[1],
                                                            &nF[2]);

    for (int rr = 0; rr != nb_dofs / 3; ++rr) { // loop over the nodes

      t_nf(i) += w * t_base_func * normal_stiffness *
                 t_normal_projection(i, j) * t_displacement_at_gauss_point(j);
      t_nf(i) -= w * t_base_func * tangent_stiffness *
                 t_tangent_projection(i, j) * t_displacement_at_gauss_point(j);

      // move to next base function
      ++t_base_func;
      // move the pointer to next element of t_nf
      ++t_nf;
    }
    // move to next integration weight
    ++t_w;
    // move to the solutions at the next Gauss point
    ++t_solution_at_gauss_point;
    ++t_init_solution_at_gauss_point;
    ++t_normal_1;
  }
  // add computed values of spring in the global right hand side vector
  Vec f = getFEMethod()->ksp_f != PETSC_NULL ? getFEMethod()->ksp_f
                                             : getFEMethod()->snes_f;
  CHKERR VecSetValues(f, nb_dofs, &data.getIndices()[0], &nF[0], ADD_VALUES);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaSpringBC::OpSpringKs::doWork(int row_side, int col_side,
                                                EntityType row_type,
                                                EntityType col_type,
                                                EntData &row_data,
                                                EntData &col_data) {
  MoFEMFunctionBegin;
  // check if the volumes have associated degrees of freedom
  const int row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);

  const int col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);

  if (dAta.tRis.find(getFEEntityHandle()) == dAta.tRis.end()) {
    MoFEMFunctionReturnHot(0);
  }

  CHKERR commonDataPtr->getBlockData(dAta);
  // size associated to the entity
  locKs.resize(row_nb_dofs, col_nb_dofs, false);
  locKs.clear();

  // get number of Gauss points
  const int row_nb_gauss_pts = row_data.getN().size1();
  if (!row_nb_gauss_pts) // check if number of Gauss point <> 0
    MoFEMFunctionReturnHot(0);

  // get intergration weights
  auto t_w = getFTensor0IntegrationWeight();

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  // create a 3d vector to be used as the normal to the face with length equal
  // to the face area
  FTensor::Tensor1<double, 3> t_normal;

  // First tangent vector
  auto t_tangent1_ptr = getFTensor1Tangent1AtGaussPts();
  FTensor::Tensor1<double, 3> t_tangent1;
  t_tangent1(i) = t_tangent1_ptr(i);

  // Second tangent vector, such that t_n = t_t1 x t_t2 | t_t2 = t_n x t_t1
  FTensor::Tensor1<double, 3> t_tangent2;
  t_tangent2(i) = FTensor::levi_civita(i, j, k) * t_normal(j) * t_tangent1(k);

  auto t_tangent2_ptr = getFTensor1Tangent2AtGaussPts();
  t_normal(i) =
      FTensor::levi_civita(i, j, k) * t_tangent1(j) * t_tangent2_ptr(k);

  auto normal_original_slave =
      getFTensor1FromMat<3>(commonDataPtr->normalVector);

  FTensor::Tensor2<double, 3, 3> t_normal_projection;
  FTensor::Tensor2<double, 3, 3> t_tangent_projection;
  // loop over the Gauss points
  for (int gg = 0; gg != row_nb_gauss_pts; gg++) {
    t_normal(i) = normal_original_slave(i);
    const double normal_length = sqrt(t_normal(k) * t_normal(k));
    t_normal(i) = t_normal(i) / normal_length;
    t_normal_projection(i, j) = t_normal(i) * t_normal(j);
    t_tangent_projection(i, j) = t_normal_projection(i, j);
    t_tangent_projection(0, 0) -= 1;
    t_tangent_projection(1, 1) -= 1;
    t_tangent_projection(2, 2) -= 1;

    // get area and integration weight
    double w = 0.5 * t_w * normal_length;

    auto t_row_base_func = row_data.getFTensor0N(gg, 0);

    for (int rr = 0; rr != row_nb_dofs / 3; rr++) {
      auto assemble_m = getFTensor2FromArray<3, 3, 3>(locKs, 3 * rr);
      auto t_col_base_func = col_data.getFTensor0N(gg, 0);
      for (int cc = 0; cc != col_nb_dofs / 3; cc++) {
        assemble_m(i, j) += w * t_row_base_func * t_col_base_func *
                            commonDataPtr->springStiffnessNormal *
                            t_normal_projection(i, j);
        assemble_m(i, j) -= w * t_row_base_func * t_col_base_func *
                            commonDataPtr->springStiffnessTangent *
                            t_tangent_projection(i, j);
        ++t_col_base_func;
        ++assemble_m;
      }
      ++t_row_base_func;
    }
    // move to next integration weight
    ++t_w;
    ++normal_original_slave;
  }

  // Add computed values of spring stiffness to the global LHS matrix
  CHKERR MatSetValues(getKSPB(), row_data, col_data, &locKs(0, 0), ADD_VALUES);

  // is symmetric
  if (row_side != col_side || row_type != col_type) {
    transLocKs.resize(col_nb_dofs, row_nb_dofs, false);
    noalias(transLocKs) = trans(locKs);

    CHKERR MatSetValues(getKSPB(), col_data, row_data, &transLocKs(0, 0),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaSpringBC::OpSpringKs_dX::doWork(int row_side, int col_side,
                                                   EntityType row_type,
                                                   EntityType col_type,
                                                   EntData &row_data,
                                                   EntData &col_data) {
  MoFEMFunctionBegin;

  // check if the volumes have associated degrees of freedom
  const int row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);

  const int col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);

  if (dAta.tRis.find(getFEEntityHandle()) == dAta.tRis.end()) {
    MoFEMFunctionReturnHot(0);
  }

  CHKERR commonDataPtr->getBlockData(dAta);
  // size associated to the entity
  locKs.resize(row_nb_dofs, col_nb_dofs, false);
  locKs.clear();

  // get number of Gauss points
  const int row_nb_gauss_pts = row_data.getN().size1();
  if (!row_nb_gauss_pts) // check if number of Gauss point <> 0
    MoFEMFunctionReturnHot(0);

  // get intergration weights
  auto t_w = getFTensor0IntegrationWeight();

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;

  auto make_vec_der = [&](auto t_N, auto t_1, auto t_2) {
    FTensor::Tensor2<double, 3, 3> t_n;
    t_n(i, j) = 0;
    t_n(i, j) += FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
    t_n(i, j) -= FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
    return t_n;
  };

  auto make_vec_der_2 = [&](auto t_N, auto t_1, auto t_2) {
    FTensor::Tensor1<double, 3> t_normal;
    t_normal(i) = FTensor::levi_civita(i, j, k) * t_1(j) * t_2(k);
    const double normal_norm = sqrt(t_normal(i) * t_normal(i));
    FTensor::Tensor2<double, 3, 3> t_n;
    t_n(i, j) = 0;
    t_n(i, j) += FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
    t_n(i, j) -= FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
    FTensor::Tensor2<double, 3, 3> t_result;
    t_result(i, j) = 0;
    t_result(i, j) =
        t_n(i, j) / normal_norm - t_normal(i) * t_n(k, j) * t_normal(k) /
                                      (normal_norm * normal_norm * normal_norm);
    return t_result;
  };

  const double normal_stiffness = commonDataPtr->springStiffnessNormal;
  const double tangent_stiffness = commonDataPtr->springStiffnessTangent;

  // create a 3d vector to be used as the normal to the face with length equal
  // to the face area
  FTensor::Tensor1<double, 3> t_normal;
  FTensor::Tensor2<double, 3, 3> t_normal_projection;
  FTensor::Tensor2<double, 3, 3> t_tangent_projection;
  FTensor::Tensor1<double, 3> t_displacement_at_gauss_point;

  // Extract solution at Gauss points
  auto t_solution_at_gauss_point =
      getFTensor1FromMat<3>(*commonDataPtr->xAtPts);
  auto t_init_solution_at_gauss_point =
      getFTensor1FromMat<3>(*commonDataPtr->xInitAtPts);

  auto t_1 = getFTensor1FromMat<3>(commonDataPtr->tangent1);
  auto t_2 = getFTensor1FromMat<3>(commonDataPtr->tangent2);

  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();

  auto normal_at_gp = getFTensor1FromMat<3>(commonDataPtr->normalVector);

  // loop over the Gauss points
  for (int gg = 0; gg != row_nb_gauss_pts; gg++) {
    // get area and integration weight
    t_normal(i) = normal_at_gp(i);
    const double normal_length = sqrt(t_normal(k) * t_normal(k));
    t_normal(i) = t_normal(i) / normal_length;

    t_normal_projection(i, j) = t_normal(i) * t_normal(j);
    t_tangent_projection(i, j) = t_normal_projection(i, j);
    t_tangent_projection(0, 0) -= 1;
    t_tangent_projection(1, 1) -= 1;
    t_tangent_projection(2, 2) -= 1;

    double w = 0.5 * t_w;

    t_displacement_at_gauss_point(i) =
        t_solution_at_gauss_point(i) - t_init_solution_at_gauss_point(i);

    auto t_row_base_func = row_data.getFTensor0N(gg, 0);

    for (int rr = 0; rr != row_nb_dofs / 3; rr++) {
      auto t_col_base_func = col_data.getFTensor0N(gg, 0);
      auto t_N = col_data.getFTensor1DiffN<2>(gg, 0);
      auto assemble_m = getFTensor2FromArray<3, 3, 3>(locKs, 3 * rr);
      for (int cc = 0; cc != col_nb_dofs / 3; cc++) {
        auto t_d_n = make_vec_der(t_N, t_1, t_2);
        auto t_d_n_2 = make_vec_der_2(t_N, t_1, t_2);

        assemble_m(i, j) -= w * normal_length * t_col_base_func *
                            normal_stiffness * t_row_base_func *
                            t_normal_projection(i, j);

        assemble_m(i, j) += w * normal_length * t_col_base_func *
                            tangent_stiffness * t_row_base_func *
                            t_tangent_projection(i, j);

        assemble_m(i, j) +=
            w * (normal_stiffness - tangent_stiffness) * t_row_base_func *
            (t_d_n(i, j) * (t_normal(k) * t_displacement_at_gauss_point(k)) +
             normal_length * t_normal(i) *
                 (t_d_n_2(k, j) * t_displacement_at_gauss_point(k)));
        // constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
        assemble_m(i, j) += w * (t_d_n(l, j) * t_normal(l)) *
                            tangent_stiffness * t_row_base_func * t_kd(i, k) *
                            t_displacement_at_gauss_point(k);

        ++t_col_base_func;
        ++t_N;
        ++assemble_m;
      }
      ++t_row_base_func;
    }
    // move to next integration weight
    ++t_w;
    ++t_solution_at_gauss_point;
    ++t_init_solution_at_gauss_point;
    ++t_1;
    ++t_2;
    ++normal_at_gp;
  }

  // Add computed values of spring stiffness to the global LHS matrix
  CHKERR MatSetValues(getKSPB(), row_data, col_data, &locKs(0, 0), ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaSpringBC::SpringALEMaterialVolOnSideLhs::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;

  if (dataAtSpringPts->faceRowData == nullptr)
    MoFEMFunctionReturnHot(0);

  if (row_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  row_nb_dofs = dataAtSpringPts->faceRowData->getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);

  nb_gauss_pts = dataAtSpringPts->faceRowData->getN().size1();

  nb_base_fun_row = dataAtSpringPts->faceRowData->getFieldData().size() / 3;
  nb_base_fun_col = col_data.getFieldData().size() / 3;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  // integrate local matrix for entity block
  CHKERR iNtegrate(*(dataAtSpringPts->faceRowData), col_data);

  // assemble local matrix
  CHKERR aSsemble(*(dataAtSpringPts->faceRowData), col_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
MetaSpringBC::SpringALEMaterialVolOnSideLhs::aSsemble(EntData &row_data,
                                                      EntData &col_data) {

  MoFEMFunctionBegin;

  // get pointer to first global index on row
  const int *row_indices = &*row_data.getIndices().data().begin();
  // get pointer to first global index on column
  const int *col_indices = &*col_data.getIndices().data().begin();

  auto &data = *dataAtSpringPts;
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

  if (!data.forcesOnlyOnEntitiesCol.empty()) {
    colIndices.resize(col_nb_dofs, false);
    noalias(colIndices) = col_data.getIndices();
    col_indices = &colIndices[0];
    VectorDofs &dofs = col_data.getFieldDofs();
    VectorDofs::iterator dit = dofs.begin();
    for (int ii = 0; dit != dofs.end(); ++dit, ++ii) {
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

MoFEMErrorCode MetaSpringBC::SpringALEMaterialVolOnSideLhs_dX_dx::iNtegrate(
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;

  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };
  FTensor::Tensor1<double, 3> t_normal;
  FTensor::Tensor2<double, 3, 3> t_normal_projection;
  FTensor::Tensor2<double, 3, 3> t_tangent_projection;

  const double normal_stiffness = dataAtSpringPts->springStiffnessNormal;
  const double tangent_stiffness = dataAtSpringPts->springStiffnessTangent;

  // Extract solution at Gauss points
  auto t_solution_at_gauss_point =
      getFTensor1FromMat<3>(*dataAtSpringPts->xAtPts);
  auto t_init_solution_at_gauss_point =
      getFTensor1FromMat<3>(*dataAtSpringPts->xInitAtPts);
  FTensor::Tensor1<double, 3> t_displacement_at_gauss_point;

  auto t_normal_1 = getFTensor1FromMat<3>(dataAtSpringPts->normalVector);
  auto t_w = getFTensor0IntegrationWeight();

  auto t_inv_H = getFTensor2FromMat<3, 3>(*dataAtSpringPts->invHMat);
  auto t_F = getFTensor2FromMat<3, 3>(*dataAtSpringPts->FMat);
  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    t_normal(i) = t_normal_1(i);

    const double normal_length = sqrt(t_normal(k) * t_normal(k));
    t_normal(i) = t_normal(i) / normal_length;
    t_normal_projection(i, j) = t_normal(i) * t_normal(j);
    t_tangent_projection(i, j) = t_normal_projection(i, j);
    t_tangent_projection(0, 0) -= 1;
    t_tangent_projection(1, 1) -= 1;
    t_tangent_projection(2, 2) -= 1;

    t_displacement_at_gauss_point(i) =
        t_solution_at_gauss_point(i) - t_init_solution_at_gauss_point(i);

    double w = 0.5 * t_w * normal_length;

    auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);
    FTensor::Tensor0<double *> t_col_base(&col_data.getN()(gg, 0));
    int bbc = 0;
    for (; bbc != nb_base_fun_col; ++bbc) {

      FTensor::Tensor0<double *> t_row_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; ++bbr) {

        auto t_assemble = get_tensor2(NN, 3 * bbr, 3 * bbc);

        t_assemble(i, j) -= w * t_row_base * t_col_base * normal_stiffness *
                            t_F(k, i) * t_normal_projection(k, j);
        t_assemble(i, j) += w * t_row_base * t_col_base * tangent_stiffness *
                            t_F(k, i) * t_tangent_projection(k, j);

        t_assemble(i, j) -= w * normal_stiffness * t_row_base * t_inv_H(k, i) *
                            t_col_diff_base(k) * t_normal_projection(j, l) *
                            t_displacement_at_gauss_point(l);
        t_assemble(i, j) += w * tangent_stiffness * t_row_base * t_inv_H(k, i) *
                            t_col_diff_base(k) * t_tangent_projection(j, l) *
                            t_displacement_at_gauss_point(l);

        ++t_row_base;
      }
      ++t_col_diff_base;
      ++t_col_base;
    }
    ++t_w;
    ++t_solution_at_gauss_point;
    ++t_init_solution_at_gauss_point;
    ++t_normal_1;
    ++t_inv_H;
    ++t_F;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
MetaSpringBC::OpSpringALEMaterialLhs::aSsemble(EntData &row_data,
                                               EntData &col_data) {

  MoFEMFunctionBegin;

  // get pointer to first global index on row
  const int *row_indices = &*row_data.getIndices().data().begin();
  // get pointer to first global index on column
  const int *col_indices = &*col_data.getIndices().data().begin();

  auto &data = *dataAtSpringPts;
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

  if (!data.forcesOnlyOnEntitiesCol.empty()) {
    colIndices.resize(col_nb_dofs, false);
    noalias(colIndices) = col_data.getIndices();
    col_indices = &colIndices[0];
    VectorDofs &dofs = col_data.getFieldDofs();
    VectorDofs::iterator dit = dofs.begin();
    for (int ii = 0; dit != dofs.end(); ++dit, ++ii) {
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

MoFEMErrorCode MetaSpringBC::OpSpringALEMaterialLhs_dX_dX::doWork(
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

  nb_base_fun_row = row_data.getFieldData().size() / 3;
  nb_base_fun_col = col_data.getFieldData().size() / 3;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  diagonal_block = (row_type == col_type) && (row_side == col_side);

  if (col_type == MBVERTEX) {
    dataAtSpringPts->faceRowData = &row_data;
    CHKERR loopSideVolumes(sideFeName, *sideFe);
  }

  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data, col_data);

  // assemble local matrix
  CHKERR aSsemble(row_data, col_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
MetaSpringBC::OpSpringALEMaterialLhs_dX_dX::iNtegrate(EntData &row_data,
                                                      EntData &col_data) {

  MoFEMFunctionBegin;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;

  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto make_vec_der = [&](auto t_N, auto t_1, auto t_2) {
    FTensor::Tensor2<double, 3, 3> t_n;
    t_n(i, j) = 0;
    t_n(i, j) += FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
    t_n(i, j) -= FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
    return t_n;
  };

  auto make_vec_der_2 = [&](auto t_N, auto t_1, auto t_2) {
    FTensor::Tensor1<double, 3> t_normal;
    t_normal(i) = FTensor::levi_civita(i, j, k) * t_1(j) * t_2(k);
    const double normal_norm = sqrt(t_normal(i) * t_normal(i));
    FTensor::Tensor2<double, 3, 3> t_n;
    t_n(i, j) = 0;
    t_n(i, j) += FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
    t_n(i, j) -= FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
    FTensor::Tensor2<double, 3, 3> t_result;
    t_result(i, j) = 0;
    t_result(i, j) =
        t_n(i, j) / normal_norm - t_normal(i) * t_n(k, j) * t_normal(k) /
                                      (normal_norm * normal_norm * normal_norm);
    return t_result;
  };

  dataAtSpringPts->faceRowData = nullptr;
  CHKERR loopSideVolumes(sideFeName, *sideFe);

  auto t_F = getFTensor2FromMat<3, 3>(*dataAtSpringPts->FMat);

  auto t_w = getFTensor0IntegrationWeight();
  auto t_1 = getFTensor1FromMat<3>(dataAtSpringPts->tangent1);
  auto t_2 = getFTensor1FromMat<3>(dataAtSpringPts->tangent2);

  const double normal_stiffness = dataAtSpringPts->springStiffnessNormal;
  const double tangent_stiffness = dataAtSpringPts->springStiffnessTangent;

  // create a 3d vector to be used as the normal to the face with length equal
  // to the face area
  FTensor::Tensor1<double, 3> t_normal;
  FTensor::Tensor2<double, 3, 3> t_normal_projection;
  FTensor::Tensor2<double, 3, 3> t_tangent_projection;

  // Extract solution at Gauss points
  auto t_solution_at_gauss_point =
      getFTensor1FromMat<3>(*dataAtSpringPts->xAtPts);
  auto t_init_solution_at_gauss_point =
      getFTensor1FromMat<3>(*dataAtSpringPts->xInitAtPts);
  FTensor::Tensor1<double, 3> t_displacement_at_gauss_point;

  auto t_normal_1 = getFTensor1FromMat<3>(dataAtSpringPts->normalVector);
  constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    t_normal(i) = t_normal_1(i);

    const double normal_length = sqrt(t_normal(k) * t_normal(k));
    t_normal(i) = t_normal(i) / normal_length;
    t_normal_projection(i, j) = t_normal(i) * t_normal(j);
    t_tangent_projection(i, j) = t_normal_projection(i, j);
    t_tangent_projection(0, 0) -= 1;
    t_tangent_projection(1, 1) -= 1;
    t_tangent_projection(2, 2) -= 1;
    // Calculate the displacement at the Gauss point
    t_displacement_at_gauss_point(i) =
        t_solution_at_gauss_point(i) - t_init_solution_at_gauss_point(i);
    double val = 0.5 * t_w;

    auto t_N = col_data.getFTensor1DiffN<2>(gg, 0);
    auto t_col_base_func = col_data.getFTensor0N(gg, 0);
    int bbc = 0;
    for (; bbc != nb_base_fun_col; ++bbc) {

      auto t_row_base_func = row_data.getFTensor0N(gg, 0);

      int bbr = 0;
      for (; bbr != nb_base_fun_row; ++bbr) {

        auto t_d_n = make_vec_der(t_N, t_1, t_2);

        auto t_assemble = get_tensor2(NN, 3 * bbr, 3 * bbc);

        auto t_d_n_2 = make_vec_der_2(t_N, t_1, t_2);

        t_assemble(i, j) += val * normal_length * t_col_base_func *
                            normal_stiffness * t_row_base_func * t_F(k, i) *
                            t_normal_projection(k, j);

        t_assemble(i, j) -= val * normal_length * t_col_base_func *
                            tangent_stiffness * t_row_base_func * t_F(k, i) *
                            t_tangent_projection(k, j);

        t_assemble(i, j) -=
            val * (normal_stiffness - tangent_stiffness) * t_row_base_func *
            t_F(l, i) *
            (t_d_n(l, j) * (t_normal(k) * t_displacement_at_gauss_point(k)) +
             normal_length * t_normal(l) *
                 (t_d_n_2(k, j) * t_displacement_at_gauss_point(k)));
        // constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
        t_assemble(i, j) -= val * (t_d_n(l, j) * t_normal(l)) *
                            tangent_stiffness * t_row_base_func * t_F(k, i) *
                            t_displacement_at_gauss_point(k);

        ++t_row_base_func;
      }
      ++t_N;
      ++t_col_base_func;
    }
    ++t_F;
    ++t_w;
    ++t_1;
    ++t_2;
    ++t_normal_1;
    ++t_solution_at_gauss_point;
    ++t_init_solution_at_gauss_point;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaSpringBC::SpringALEMaterialVolOnSideLhs_dX_dX::iNtegrate(
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;
  FTensor::Index<'m', 3> m;
  FTensor::Index<'q', 3> q;

  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  const double normal_stiffness = dataAtSpringPts->springStiffnessNormal;
  const double tangent_stiffness = dataAtSpringPts->springStiffnessTangent;

  // create a 3d vector to be used as the normal to the face with length equal
  // to the face area
  FTensor::Tensor1<double, 3> t_normal;
  FTensor::Tensor2<double, 3, 3> t_normal_projection;
  FTensor::Tensor2<double, 3, 3> t_tangent_projection;

  // Extract solution at Gauss points
  auto t_solution_at_gauss_point =
      getFTensor1FromMat<3>(*dataAtSpringPts->xAtPts);
  auto t_init_solution_at_gauss_point =
      getFTensor1FromMat<3>(*dataAtSpringPts->xInitAtPts);
  FTensor::Tensor1<double, 3> t_displacement_at_gauss_point;

  auto t_normal_1 = getFTensor1FromMat<3>(dataAtSpringPts->normalVector);

  auto t_w = getFTensor0IntegrationWeight();

  auto t_h = getFTensor2FromMat<3, 3>(*dataAtSpringPts->hMat);
  auto t_inv_H = getFTensor2FromMat<3, 3>(*dataAtSpringPts->invHMat);

  FTensor::Tensor2<double, 3, 3> t_d;

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    t_normal(i) = t_normal_1(i);

    const double normal_length = sqrt(t_normal(k) * t_normal(k));
    t_normal(i) = t_normal(i) / normal_length;
    t_normal_projection(i, j) = t_normal(i) * t_normal(j);
    t_tangent_projection(i, j) = t_normal_projection(i, j);
    t_tangent_projection(0, 0) -= 1;
    t_tangent_projection(1, 1) -= 1;
    t_tangent_projection(2, 2) -= 1;
    // Calculate the displacement at the Gauss point
    t_displacement_at_gauss_point(i) =
        t_solution_at_gauss_point(i) - t_init_solution_at_gauss_point(i);

    double val = 0.5 * t_w * normal_length;

    auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

    int bbc = 0;
    for (; bbc != nb_base_fun_col; ++bbc) {

      FTensor::Tensor0<double *> t_row_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; ++bbr) {

        auto t_assemble = get_tensor2(NN, 3 * bbr, 3 * bbc);

        t_assemble(i, j) +=
            val * t_row_base * normal_stiffness * t_inv_H(l, j) *
            t_col_diff_base(m) * t_inv_H(m, i) * t_h(k, l) *
            (t_normal_projection(k, q) * t_displacement_at_gauss_point(q));

        t_assemble(i, j) -=
            val * t_row_base * tangent_stiffness * t_inv_H(l, j) *
            t_col_diff_base(m) * t_inv_H(m, i) * t_h(k, l) *
            (t_tangent_projection(k, q) * t_displacement_at_gauss_point(q));

        ++t_row_base;
      }
      ++t_col_diff_base;
    }
    ++t_w;
    ++t_h;
    ++t_inv_H;
    ++t_solution_at_gauss_point;
    ++t_init_solution_at_gauss_point;
    ++t_normal_1;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaSpringBC::OpCalculateDeformation::doWork(int side,
                                                            EntityType type,
                                                            EntData &row_data) {

  MoFEMFunctionBegin;
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  // get number of integration points
  const int nb_integration_pts = getGaussPts().size2();

  auto t_h = getFTensor2FromMat<3, 3>(*commonDataPtr->hMat);
  auto t_H = getFTensor2FromMat<3, 3>(*commonDataPtr->HMat);

  commonDataPtr->detHVec->resize(nb_integration_pts, false);
  commonDataPtr->invHMat->resize(9, nb_integration_pts, false);
  commonDataPtr->FMat->resize(9, nb_integration_pts, false);

  commonDataPtr->detHVec->clear();
  commonDataPtr->invHMat->clear();
  commonDataPtr->FMat->clear();

  auto t_detH = getFTensor0FromVec(*commonDataPtr->detHVec);
  auto t_invH = getFTensor2FromMat<3, 3>(*commonDataPtr->invHMat);
  auto t_F = getFTensor2FromMat<3, 3>(*commonDataPtr->FMat);

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

MoFEMErrorCode MetaSpringBC::OpSpringFsMaterial::doWork(int side,
                                                        EntityType type,
                                                        EntData &row_data) {

  MoFEMFunctionBegin;

  if (dAta.tRis.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tRis.end()) {
    MoFEMFunctionReturnHot(0);
  }

  // get number of dofs on row
  nbRows = row_data.getIndices().size();
  // if no dofs on row, exit that work, nothing to do here
  if (!nbRows)
    MoFEMFunctionReturnHot(0);

  nF.resize(nbRows, false);
  nF.clear();

  // get number of integration points
  nbIntegrationPts = getGaussPts().size2();

  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data);

  // assemble local matrix
  CHKERR aSsemble(row_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaSpringBC::OpSpringFsMaterial::iNtegrate(EntData &data) {
  MoFEMFunctionBegin;

  CHKERR loopSideVolumes(sideFeName, *sideFe);

  // check that the faces have associated degrees of freedom
  const int nb_dofs = data.getIndices().size();
  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  CHKERR dataAtPts->getBlockData(dAta);

  // get integration weights
  auto t_w = getFTensor0IntegrationWeight();

  // FTensor indices
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  const double normal_stiffness = dataAtPts->springStiffnessNormal;
  const double tangent_stiffness = dataAtPts->springStiffnessTangent;

  // create a 3d vector to be used as the normal to the face with length equal
  // to the face area
  FTensor::Tensor1<double, 3> t_normal;
  FTensor::Tensor2<double, 3, 3> t_normal_projection;
  FTensor::Tensor2<double, 3, 3> t_tangent_projection;

  // Extract solution at Gauss points
  auto t_solution_at_gauss_point = getFTensor1FromMat<3>(*dataAtPts->xAtPts);
  auto t_init_solution_at_gauss_point =
      getFTensor1FromMat<3>(*dataAtPts->xInitAtPts);
  FTensor::Tensor1<double, 3> t_displacement_at_gauss_point;

  auto t_normal_1 = getFTensor1FromMat<3>(dataAtPts->normalVector);

  auto t_F = getFTensor2FromMat<3, 3>(*dataAtPts->FMat);

  // loop over all Gauss points of the face
  for (int gg = 0; gg != nbIntegrationPts; ++gg) {
    t_normal(i) = t_normal_1(i);

    const double normal_length = sqrt(t_normal(k) * t_normal(k));
    t_normal(i) = t_normal(i) / normal_length;
    t_normal_projection(i, j) = t_normal(i) * t_normal(j);
    t_tangent_projection(i, j) = t_normal_projection(i, j);
    t_tangent_projection(0, 0) -= 1;
    t_tangent_projection(1, 1) -= 1;
    t_tangent_projection(2, 2) -= 1;
    // Calculate the displacement at the Gauss point
    t_displacement_at_gauss_point(i) =
        t_solution_at_gauss_point(i) - t_init_solution_at_gauss_point(i);

    double w = t_w * 0.5 * normal_length; // area was constant

    auto t_base_func = data.getFTensor0N(gg, 0);

    // create a vector t_nf whose pointer points an array of 3 pointers
    // pointing to nF  memory location of components
    FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf(&nF[0], &nF[1],
                                                            &nF[2]);

    for (int rr = 0; rr != nb_dofs / 3; ++rr) { // loop over the nodes

      t_nf(i) -= w * t_base_func * normal_stiffness * t_F(k, i) *
                 t_normal_projection(k, j) * t_displacement_at_gauss_point(j);
      t_nf(i) += w * t_base_func * tangent_stiffness * t_F(k, i) *
                 t_tangent_projection(k, j) * t_displacement_at_gauss_point(j);

      // move to next base function
      ++t_base_func;
      // move the pointer to next element of t_nf
      ++t_nf;
    }
    // move to next integration weight
    ++t_w;
    // move to the solutions at the next Gauss point
    ++t_solution_at_gauss_point;
    ++t_init_solution_at_gauss_point;
    ++t_normal_1;
    ++t_F;
  }
  // add computed values of spring in the global right hand side vector
  // Vec f = getFEMethod()->ksp_f != PETSC_NULL ? getFEMethod()->ksp_f
  //                                            : getFEMethod()->snes_f;
  // CHKERR VecSetValues(f, nb_dofs, &data.getIndices()[0], &nF[0], ADD_VALUES);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaSpringBC::OpSpringFsMaterial::aSsemble(EntData &row_data) {
  MoFEMFunctionBegin;

  // get pointer to first global index on row
  const int *row_indices = &*row_data.getIndices().data().begin();

  auto &data = *dataAtPts;
  if (!data.forcesOnlyOnEntitiesRow.empty()) {
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
  }

  CHKERR VecSetOption(getSNESf(), VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  CHKERR VecSetValues(getSNESf(), nbRows, row_indices, &*nF.data().begin(),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaSpringBC::OpGetTangentSpEle::doWork(int side,
                                                       EntityType type,
                                                       EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    PetscFunctionReturn(0);

  ngp = data.getN().size1();

  unsigned int nb_dofs = data.getFieldData().size() / 3;
  FTensor::Index<'i', 3> i;
  if (type == MBVERTEX) {
    dataAtIntegrationPts->tangent1.resize(3, ngp, false);
    dataAtIntegrationPts->tangent1.clear();

    dataAtIntegrationPts->tangent2.resize(3, ngp, false);
    dataAtIntegrationPts->tangent2.clear();
  }

  auto t_1 = getFTensor1FromMat<3>(dataAtIntegrationPts->tangent1);
  auto t_2 = getFTensor1FromMat<3>(dataAtIntegrationPts->tangent2);

  for (unsigned int gg = 0; gg != ngp; ++gg) {
    auto t_N = data.getFTensor1DiffN<2>(gg, 0);
    auto t_dof = data.getFTensor1FieldData<3>();

    for (unsigned int dd = 0; dd != nb_dofs; ++dd) {
      t_1(i) += t_dof(i) * t_N(0);
      t_2(i) += t_dof(i) * t_N(1);
      ++t_dof;
      ++t_N;
    }
    ++t_1;
    ++t_2;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaSpringBC::OpGetNormalSpEle::doWork(int side, EntityType type,
                                                      EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    MoFEMFunctionReturnHot(0);

  ngp = data.getN().size1();

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  if (type == MBVERTEX) {
    dataAtIntegrationPts->normalVector.resize(3, ngp, false);
    dataAtIntegrationPts->normalVector.clear();
  }

  auto normal_original_slave =
      getFTensor1FromMat<3>(dataAtIntegrationPts->normalVector);

  auto tangent_1 = getFTensor1FromMat<3>(dataAtIntegrationPts->tangent1);
  auto tangent_2 = getFTensor1FromMat<3>(dataAtIntegrationPts->tangent2);

  for (unsigned int gg = 0; gg != ngp; ++gg) {
    normal_original_slave(i) =
        FTensor::levi_civita(i, j, k) * tangent_1(j) * tangent_2(k);
    ++normal_original_slave;
    ++tangent_1;
    ++tangent_2;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaSpringBC::OpSpringALEMaterialLhs_dX_dx::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;

  if (col_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  dataAtSpringPts->faceRowData = &row_data;
  CHKERR loopSideVolumes(sideFeName, *sideFe);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
MetaSpringBC::addSpringElements(MoFEM::Interface &m_field,
                                const std::string field_name,
                                const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;

  // Define boundary element that operates on rows, columns and data of a
  // given field
  CHKERR m_field.add_finite_element("SPRING", MF_ZERO);
  CHKERR m_field.modify_finite_element_add_field_row("SPRING", field_name);
  CHKERR m_field.modify_finite_element_add_field_col("SPRING", field_name);
  CHKERR m_field.modify_finite_element_add_field_data("SPRING", field_name);
  if (m_field.check_field(mesh_nodals_positions)) {
    CHKERR m_field.modify_finite_element_add_field_data("SPRING",
                                                        mesh_nodals_positions);
  }
  // Add entities to that element, here we add all triangles with SPRING_BC
  // from cubit
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
    if (bit->getName().compare(0, 9, "SPRING_BC") == 0) {
      CHKERR m_field.add_ents_to_finite_element_by_type(bit->getMeshset(),
                                                        MBTRI, "SPRING");
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaSpringBC::addSpringElementsALE(
    MoFEM::Interface &m_field, const std::string field_name,
    const std::string mesh_nodals_positions, Range &spring_triangles) {
  MoFEMFunctionBegin;

  // Define boundary element that operates on rows, columns and data of a
  // given field
  CHKERR m_field.add_finite_element("SPRING_ALE", MF_ZERO);
  CHKERR m_field.modify_finite_element_add_field_row("SPRING_ALE", field_name);
  CHKERR m_field.modify_finite_element_add_field_col("SPRING_ALE", field_name);
  CHKERR m_field.modify_finite_element_add_field_data("SPRING_ALE", field_name);
  CHKERR m_field.modify_finite_element_add_field_row("SPRING_ALE",
                                                     mesh_nodals_positions);
  CHKERR m_field.modify_finite_element_add_field_col("SPRING_ALE",
                                                     mesh_nodals_positions);
  CHKERR m_field.modify_finite_element_add_field_data("SPRING_ALE",
                                                      mesh_nodals_positions);

  CHKERR m_field.add_ents_to_finite_element_by_type(spring_triangles, MBTRI,
                                                    "SPRING_ALE");

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaSpringBC::setSpringOperators(
    MoFEM::Interface &m_field,
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ptr,
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_rhs_ptr,
    const std::string field_name, const std::string mesh_nodals_positions, double stiffness_scale) {
  MoFEMFunctionBegin;

  // Push operators to instances for springs
  // loop over blocks
  boost::shared_ptr<MetaSpringBC::DataAtIntegrationPtsSprings> commonDataPtr =
      boost::make_shared<MetaSpringBC::DataAtIntegrationPtsSprings>(m_field, stiffness_scale);
  CHKERR commonDataPtr->getParameters();

  if (m_field.check_field(mesh_nodals_positions)) {
    CHKERR addHOOpsFace3D(mesh_nodals_positions, *fe_spring_lhs_ptr, false,
                          false);
    CHKERR addHOOpsFace3D(mesh_nodals_positions, *fe_spring_rhs_ptr, false,
                          false);
  }

  for (auto &sitSpring : commonDataPtr->mapSpring) {

    fe_spring_lhs_ptr->getOpPtrVector().push_back(
        new OpGetTangentSpEle(mesh_nodals_positions, commonDataPtr));
    fe_spring_lhs_ptr->getOpPtrVector().push_back(
        new OpGetNormalSpEle(mesh_nodals_positions, commonDataPtr));
    fe_spring_lhs_ptr->getOpPtrVector().push_back(
        new OpSpringKs(commonDataPtr, sitSpring.second, field_name));

    fe_spring_rhs_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<3>(field_name, commonDataPtr->xAtPts));
    fe_spring_rhs_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<3>(mesh_nodals_positions,
                                            commonDataPtr->xInitAtPts));
    fe_spring_rhs_ptr->getOpPtrVector().push_back(
        new OpGetTangentSpEle(mesh_nodals_positions, commonDataPtr));
    fe_spring_rhs_ptr->getOpPtrVector().push_back(
        new OpGetNormalSpEle(mesh_nodals_positions, commonDataPtr));
    fe_spring_rhs_ptr->getOpPtrVector().push_back(
        new OpSpringFs(commonDataPtr, sitSpring.second, field_name));
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaSpringBC::setSpringOperatorsMaterial(
    MoFEM::Interface &m_field,
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ptr_dx,
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ptr_dX,
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_rhs_ptr,
    boost::shared_ptr<MetaSpringBC::DataAtIntegrationPtsSprings>
        data_at_integration_pts,
    const std::string field_name, const std::string mesh_nodals_positions,
    std::string side_fe_name) {
  MoFEMFunctionBegin;

  // Push operators to instances for springs
  // loop over blocks
  CHKERR data_at_integration_pts->getParameters();

  if (m_field.check_field(mesh_nodals_positions)) {
    CHKERR addHOOpsFace3D(mesh_nodals_positions, *fe_spring_lhs_ptr_dx, false,
                          false);
    CHKERR addHOOpsFace3D(mesh_nodals_positions, *fe_spring_lhs_ptr_dX, false,
                          false);
    CHKERR addHOOpsFace3D(mesh_nodals_positions, *fe_spring_rhs_ptr, false,
                          false);
  }

  for (auto &sitSpring : data_at_integration_pts->mapSpring) {

    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> feMatSideRhs =
        boost::make_shared<VolumeElementForcesAndSourcesCoreOnSide>(m_field);

    feMatSideRhs->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(
            mesh_nodals_positions, data_at_integration_pts->HMat));
    feMatSideRhs->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(
            field_name, data_at_integration_pts->hMat));

    feMatSideRhs->getOpPtrVector().push_back(new OpCalculateDeformation(
        mesh_nodals_positions, data_at_integration_pts));

    fe_spring_rhs_ptr->getOpPtrVector().push_back(
        new OpGetTangentSpEle(mesh_nodals_positions, data_at_integration_pts));
    fe_spring_rhs_ptr->getOpPtrVector().push_back(
        new OpGetNormalSpEle(mesh_nodals_positions, data_at_integration_pts));

    fe_spring_rhs_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<3>(field_name,
                                            data_at_integration_pts->xAtPts));
    fe_spring_rhs_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<3>(
            mesh_nodals_positions, data_at_integration_pts->xInitAtPts));

    fe_spring_rhs_ptr->getOpPtrVector().push_back(
        new OpSpringFsMaterial(mesh_nodals_positions, data_at_integration_pts,
                               feMatSideRhs, side_fe_name, sitSpring.second));

    fe_spring_lhs_ptr_dx->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<3>(field_name,
                                            data_at_integration_pts->xAtPts));
    fe_spring_lhs_ptr_dx->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<3>(
            mesh_nodals_positions, data_at_integration_pts->xInitAtPts));
    fe_spring_lhs_ptr_dx->getOpPtrVector().push_back(
        new OpGetTangentSpEle(mesh_nodals_positions, data_at_integration_pts));
    fe_spring_lhs_ptr_dx->getOpPtrVector().push_back(
        new OpGetNormalSpEle(mesh_nodals_positions, data_at_integration_pts));

    fe_spring_lhs_ptr_dx->getOpPtrVector().push_back(
        new OpSpringKs_dX(data_at_integration_pts, sitSpring.second, field_name,
                          mesh_nodals_positions));

    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> feMatSideLhs_dx =
        boost::make_shared<VolumeElementForcesAndSourcesCoreOnSide>(m_field);

    feMatSideLhs_dx->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(
            mesh_nodals_positions, data_at_integration_pts->HMat));
    feMatSideLhs_dx->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(
            field_name, data_at_integration_pts->hMat));

    feMatSideLhs_dx->getOpPtrVector().push_back(new OpCalculateDeformation(
        mesh_nodals_positions, data_at_integration_pts));

    feMatSideLhs_dx->getOpPtrVector().push_back(
        new SpringALEMaterialVolOnSideLhs_dX_dx(
            mesh_nodals_positions, field_name, data_at_integration_pts));

    fe_spring_lhs_ptr_dX->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<3>(field_name,
                                            data_at_integration_pts->xAtPts));
    fe_spring_lhs_ptr_dX->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<3>(
            mesh_nodals_positions, data_at_integration_pts->xInitAtPts));
    fe_spring_lhs_ptr_dX->getOpPtrVector().push_back(
        new OpGetTangentSpEle(mesh_nodals_positions, data_at_integration_pts));
    fe_spring_lhs_ptr_dX->getOpPtrVector().push_back(
        new OpGetNormalSpEle(mesh_nodals_positions, data_at_integration_pts));

    fe_spring_lhs_ptr_dX->getOpPtrVector().push_back(
        new OpSpringALEMaterialLhs_dX_dx(mesh_nodals_positions, field_name,
                                         data_at_integration_pts,
                                         feMatSideLhs_dx, side_fe_name));

    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> feMatSideLhs_dX =
        boost::make_shared<VolumeElementForcesAndSourcesCoreOnSide>(m_field);

    feMatSideLhs_dX->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(
            mesh_nodals_positions, data_at_integration_pts->HMat));
    feMatSideLhs_dX->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(
            field_name, data_at_integration_pts->hMat));

    feMatSideLhs_dX->getOpPtrVector().push_back(new OpCalculateDeformation(
        mesh_nodals_positions, data_at_integration_pts));

    feMatSideLhs_dX->getOpPtrVector().push_back(
        new SpringALEMaterialVolOnSideLhs_dX_dX(mesh_nodals_positions,
                                                mesh_nodals_positions,
                                                data_at_integration_pts));

    fe_spring_lhs_ptr_dX->getOpPtrVector().push_back(
        new OpSpringALEMaterialLhs_dX_dX(
            mesh_nodals_positions, mesh_nodals_positions,
            data_at_integration_pts, feMatSideLhs_dX, side_fe_name));
  }

  MoFEMFunctionReturn(0);
}