/** \file HookeElement.cpp
 * \example HookeElement.cpp
 * \brief Operators and data structures for linear elastic analysis
 *
 * See as well header file HookeElement.hpp
 * 
 * Implemention of operators for Hooke material. Implementation is extended to
 * the case when the mesh is moving as results of topological changes, also the
 * calculation of material forces and associated tangent matrices are added to
 * implementation.
 * 
 * In other words spatial deformation is small but topological changes large. 
 
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
#include <HookeElement.hpp>

HookeElement::OpCalculateStrainAle::OpCalculateStrainAle(
    const std::string row_field, const std::string col_field,
    boost::shared_ptr<DataAtIntegrationPts> &data_at_pts)
    : VolUserDataOperator(row_field, col_field, OPROW, false),
      dataAtPts(data_at_pts) {
  doEdges = false;
  doQuads = false;
  doTris = false;
  doTets = false;
  doPrisms = false;
}

MoFEMErrorCode HookeElement::OpCalculateStrainAle::doWork(int row_side,
                                                          EntityType row_type,
                                                          EntData &row_data) {
  MoFEMFunctionBegin;
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  // get number of integration points
  const int nb_integration_pts = getGaussPts().size2();
  auto t_h = getFTensor2FromMat<3, 3>(*dataAtPts->hMat);
  auto t_H = getFTensor2FromMat<3, 3>(*dataAtPts->HMat);

  dataAtPts->detHVec->resize(nb_integration_pts, false);
  dataAtPts->invHMat->resize(9, nb_integration_pts, false);
  dataAtPts->FMat->resize(9, nb_integration_pts, false);
  dataAtPts->smallStrainMat->resize(6, nb_integration_pts, false);

  auto t_detH = getFTensor0FromVec(*dataAtPts->detHVec);
  auto t_invH = getFTensor2FromMat<3, 3>(*dataAtPts->invHMat);
  auto t_F = getFTensor2FromMat<3, 3>(*dataAtPts->FMat);
  auto t_strain = getFTensor2SymmetricFromMat<3>(*dataAtPts->smallStrainMat);

  for (int gg = 0; gg != nb_integration_pts; ++gg) {
    CHKERR determinantTensor3by3(t_H, t_detH);
    CHKERR invertTensor3by3(t_H, t_detH, t_invH);
    t_F(i, j) = t_h(i, k) * t_invH(k, j);
    t_strain(i, j) = (t_F(i, j) || t_F(j, i)) / 2.;

    t_strain(0, 0) -= 1;
    t_strain(1, 1) -= 1;
    t_strain(2, 2) -= 1;

    ++t_strain;
    ++t_h;
    ++t_H;
    ++t_detH;
    ++t_invH;
    ++t_F;
  }
  MoFEMFunctionReturn(0);
}

HookeElement::OpCalculateEnergy::OpCalculateEnergy(
    const std::string row_field, const std::string col_field,
    boost::shared_ptr<DataAtIntegrationPts> data_at_pts, Vec ghost_vec)
    : VolUserDataOperator(row_field, col_field, OPROW, true),
      dataAtPts(data_at_pts), ghostVec(ghost_vec) {
  doEdges = false;
  doQuads = false;
  doTris = false;
  doTets = false;
  doPrisms = false;
  if (ghostVec != PETSC_NULL) {
    ierr = PetscObjectReference((PetscObject)ghostVec);
    CHKERRABORT(PETSC_COMM_SELF, ierr);
  }
}

HookeElement::OpCalculateEnergy::~OpCalculateEnergy() {
  ierr = VecDestroy(&ghostVec);
  CHKERRABORT(PETSC_COMM_SELF, ierr);
}

MoFEMErrorCode HookeElement::OpCalculateEnergy::doWork(int row_side,
                                                       EntityType row_type,
                                                       EntData &row_data) {
  MoFEMFunctionBegin;

  // get number of integration points
  const int nb_integration_pts = getGaussPts().size2();
  auto t_strain = getFTensor2SymmetricFromMat<3>(*(dataAtPts->smallStrainMat));
  auto t_cauchy_stress =
      getFTensor2SymmetricFromMat<3>(*(dataAtPts->cauchyStressMat));
  dataAtPts->energyVec->resize(nb_integration_pts, false);
  auto t_energy = FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(
      &*(dataAtPts->energyVec->data().begin()));

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  for (int gg = 0; gg != nb_integration_pts; ++gg) {
    t_energy = (t_strain(i, j) * t_cauchy_stress(i, j)) / 2.;
    ++t_strain;
    ++t_cauchy_stress;
    ++t_energy;
  }

  if (ghostVec != PETSC_NULL) {
    // get element volume
    double vol = getVolume();
    // get intergrayion weights
    auto t_w = getFTensor0IntegrationWeight();
    auto &det_H = *dataAtPts->detHVec;
    auto t_energy = FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(
        &*(dataAtPts->energyVec->data().begin()));
    double *energy_ptr;
    CHKERR VecGetArray(ghostVec, &energy_ptr);
    for (int gg = 0; gg != nb_integration_pts; ++gg) {
      // calculate scalar weight times element volume
      double a = t_w * vol;
      if (getHoGaussPtsDetJac().size() && det_H.empty()) {
        a *= getHoGaussPtsDetJac()[gg];
      } else if (det_H.size()) {
        a *= det_H[gg];
      }
      energy_ptr[0] += a * t_energy;
      ++t_energy;
      ++t_w;
    }
    CHKERR VecRestoreArray(ghostVec, &energy_ptr);
  }

  MoFEMFunctionReturn(0);
}

HookeElement::OpCalculateEshelbyStress::OpCalculateEshelbyStress(
    const std::string row_field, const std::string col_field,
    boost::shared_ptr<DataAtIntegrationPts> data_at_pts)
    : VolUserDataOperator(row_field, col_field, OPROW, true),
      dataAtPts(data_at_pts) {
  doEdges = false;
  doQuads = false;
  doTris = false;
  doTets = false;
  doPrisms = false;
}

MoFEMErrorCode HookeElement::OpCalculateEshelbyStress::doWork(
    int row_side, EntityType row_type, EntData &row_data) {
  MoFEMFunctionBegin;
  // get number of integration points
  const int nb_integration_pts = getGaussPts().size2();
  auto t_energy = FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(
      &*(dataAtPts->energyVec->data().begin()));
  auto t_cauchy_stress =
      getFTensor2SymmetricFromMat<3>(*(dataAtPts->cauchyStressMat));
  auto t_F = getFTensor2FromMat<3, 3>(*(dataAtPts->FMat));
  dataAtPts->eshelbyStressMat->resize(9, nb_integration_pts, false);
  auto t_eshelby_stress =
      getFTensor2FromMat<3, 3>(*(dataAtPts->eshelbyStressMat));

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  for (int gg = 0; gg != nb_integration_pts; ++gg) {
    t_eshelby_stress(i, j) = -t_F(k, i) * t_cauchy_stress(k, j);
    t_eshelby_stress(0, 0) += t_energy;
    t_eshelby_stress(1, 1) += t_energy;
    t_eshelby_stress(2, 2) += t_energy;
    ++t_cauchy_stress;
    ++t_energy;
    ++t_eshelby_stress;
    ++t_F;
  }
  MoFEMFunctionReturn(0);
}

HookeElement::OpAssemble::OpAssemble(
    const std::string row_field, const std::string col_field,
    boost::shared_ptr<DataAtIntegrationPts> &data_at_pts, const char type,
    bool symm)
    : VolUserDataOperator(row_field, col_field, type, symm),
      dataAtPts(data_at_pts) {}

MoFEMErrorCode HookeElement::OpAssemble::doWork(int row_side, int col_side,
                                                EntityType row_type,
                                                EntityType col_type,
                                                EntData &row_data,
                                                EntData &col_data) {

  MoFEMFunctionBegin;

  // get number of dofs on row
  nbRows = row_data.getIndices().size();
  // if no dofs on row, exit that work, nothing to do here
  if (!nbRows)
    MoFEMFunctionReturnHot(0);

  // get number of dofs on column
  nbCols = col_data.getIndices().size();
  // if no dofs on Columbia, exit nothing to do here
  if (!nbCols)
    MoFEMFunctionReturnHot(0);

  // K_ij matrix will have 3 times the number of degrees of freedom of the
  // i-th entity set (nbRows)
  // and 3 times the number of degrees of freedom of the j-th entity set
  // (nbCols)
  K.resize(nbRows, nbCols, false);
  K.clear();

  // get number of integration points
  nbIntegrationPts = getGaussPts().size2();
  // check if entity block is on matrix diagonal
  if (row_side == col_side && row_type == col_type) {
    isDiag = true;
  } else {
    isDiag = false;
  }

  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data, col_data);

  // assemble local matrix
  CHKERR aSsemble(row_data, col_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HookeElement::OpAssemble::doWork(int row_side,
                                                EntityType row_type,
                                                EntData &row_data) {
  MoFEMFunctionBegin;

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

MoFEMErrorCode HookeElement::OpAssemble::iNtegrate(EntData &row_data,
                                                   EntData &col_data) {
  MoFEMFunctionBegin;
  SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "Not implemented");
  MoFEMFunctionReturn(0);
};

MoFEMErrorCode HookeElement::OpAssemble::iNtegrate(EntData &row_data) {
  MoFEMFunctionBegin;
  SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "Not implemented");
  MoFEMFunctionReturn(0);
};

MoFEMErrorCode HookeElement::OpAssemble::aSsemble(EntData &row_data,
                                                  EntData &col_data) {
  MoFEMFunctionBegin;

  // get pointer to first global index on row
  const int *row_indices = &*row_data.getIndices().data().begin();
  // get pointer to first global index on column
  const int *col_indices = &*col_data.getIndices().data().begin();

  auto &data = *dataAtPts;
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

  if (!data.forcesOnlyOnEntitiesCol.empty()) {
    colIndices.resize(nbCols, false);
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
  CHKERR MatSetValues(B, nbRows, row_indices, nbCols, col_indices,
                      &*K.data().begin(), ADD_VALUES);

  if (!isDiag && sYmm) {
    // if not diagonal term and since global matrix is symmetric assemble
    // transpose term.
    transK.resize(K.size2(), K.size1(), false);
    noalias(transK) = trans(K);
    CHKERR MatSetValues(B, nbCols, col_indices, nbRows, row_indices,
                        &*transK.data().begin(), ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HookeElement::OpAssemble::aSsemble(EntData &row_data) {
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
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (data.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
          data.forcesOnlyOnEntitiesRow.end()) {
        rowIndices[ii] = -1;
      }
    }
  }

  Vec F = getFEMethod()->snes_f;
  // assemble local matrix
  CHKERR VecSetValues(F, nbRows, row_indices, &*nF.data().begin(), ADD_VALUES);
  MoFEMFunctionReturn(0);
}

HookeElement::OpRhs_dx::OpRhs_dx(
    const std::string row_field, const std::string col_field,
    boost::shared_ptr<DataAtIntegrationPts> &data_at_pts)
    : OpAssemble(row_field, col_field, data_at_pts, OPROW) {}

MoFEMErrorCode HookeElement::OpRhs_dx::iNtegrate(EntData &row_data) {
  MoFEMFunctionBegin;

  auto get_tensor1 = [](VectorDouble &v, const int r) {
    return FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3>(
        &v(r + 0), &v(r + 1), &v(r + 2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;

  // get element volume
  double vol = getVolume();
  // get intergrayion weights
  auto t_w = getFTensor0IntegrationWeight();

  // get derivatives of base functions on rows
  auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
  const int row_nb_base_fun = row_data.getN().size2();
  auto t_cauchy_stress =
      getFTensor2SymmetricFromMat<3>(*dataAtPts->cauchyStressMat);

  // iterate over integration points
  for (int gg = 0; gg != nbIntegrationPts; ++gg) {

    // calculate scalar weight times element volume
    double a = t_w * vol;

    if (getHoGaussPtsDetJac().size()) {
      // If HO geometry
      a *= getHoGaussPtsDetJac()[gg];
    }

    auto t_nf = get_tensor1(nF, 0);

    int rr = 0;
    for (; rr != nbRows / 3; ++rr) {
      t_nf(i) += a * t_row_diff_base(j) * t_cauchy_stress(i, j);
      ++t_row_diff_base;
      ++t_nf;
    }

    for (; rr != row_nb_base_fun; ++rr)
      ++t_row_diff_base;

    ++t_w;
    ++t_cauchy_stress;
  }

  MoFEMFunctionReturn(0);
}

HookeElement::OpAleRhs_dx::OpAleRhs_dx(
    const std::string row_field, const std::string col_field,
    boost::shared_ptr<DataAtIntegrationPts> &data_at_pts)
    : OpAssemble(row_field, col_field, data_at_pts, OPROW) {}

MoFEMErrorCode HookeElement::OpAleRhs_dx::iNtegrate(EntData &row_data) {
  MoFEMFunctionBegin;

  auto get_tensor1 = [](VectorDouble &v, const int r) {
    return FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3>(
        &v(r + 0), &v(r + 1), &v(r + 2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  // get element volume
  double vol = getVolume();
  // get intergrayion weights
  auto t_w = getFTensor0IntegrationWeight();

  // get derivatives of base functions on rows
  auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
  const int row_nb_base_fun = row_data.getN().size2();
  auto t_cauchy_stress =
      getFTensor2SymmetricFromMat<3>(*dataAtPts->cauchyStressMat);
  auto &det_H = *dataAtPts->detHVec;
  auto t_invH = getFTensor2FromMat<3, 3>(*dataAtPts->invHMat);

  // iterate over integration points
  for (int gg = 0; gg != nbIntegrationPts; ++gg) {

    // calculate scalar weight times element volume
    double a = t_w * vol * det_H[gg];
    auto t_nf = get_tensor1(nF, 0);

    int rr = 0;
    for (; rr != nbRows / 3; ++rr) {
      FTensor::Tensor1<double, 3> t_row_diff_base_pulled;
      t_row_diff_base_pulled(i) = t_row_diff_base(j) * t_invH(j, i);
      t_nf(i) += a * t_row_diff_base_pulled(j) * t_cauchy_stress(i, j);
      ++t_row_diff_base;
      ++t_nf;
    }

    for (; rr != row_nb_base_fun; ++rr)
      ++t_row_diff_base;

    ++t_w;
    ++t_cauchy_stress;
    ++t_invH;
  }

  MoFEMFunctionReturn(0);
}

HookeElement::OpAleRhs_dX::OpAleRhs_dX(
    const std::string row_field, const std::string col_field,
    boost::shared_ptr<DataAtIntegrationPts> &data_at_pts)
    : OpAssemble(row_field, col_field, data_at_pts, OPROW) {}

MoFEMErrorCode HookeElement::OpAleRhs_dX::iNtegrate(EntData &row_data) {
  MoFEMFunctionBegin;

  auto get_tensor1 = [](VectorDouble &v, const int r) {
    return FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3>(
        &v(r + 0), &v(r + 1), &v(r + 2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  // get element volume
  double vol = getVolume();
  // get intergrayion weights
  auto t_w = getFTensor0IntegrationWeight();

  // get derivatives of base functions on rows
  auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
  const int row_nb_base_fun = row_data.getN().size2();
  auto t_eshelby_stress =
      getFTensor2FromMat<3, 3>(*dataAtPts->eshelbyStressMat);
  auto &det_H = *dataAtPts->detHVec;
  auto t_invH = getFTensor2FromMat<3, 3>(*dataAtPts->invHMat);

  // iterate over integration points
  for (int gg = 0; gg != nbIntegrationPts; ++gg) {

    // calculate scalar weight times element volume
    double a = t_w * vol * det_H[gg];
    auto t_nf = get_tensor1(nF, 0);

    int rr = 0;
    for (; rr != nbRows / 3; ++rr) {
      FTensor::Tensor1<double, 3> t_row_diff_base_pulled;
      t_row_diff_base_pulled(i) = t_row_diff_base(j) * t_invH(j, i);
      t_nf(i) += a * t_row_diff_base_pulled(j) * t_eshelby_stress(i, j);
      ++t_row_diff_base;
      ++t_nf;
    }

    for (; rr != row_nb_base_fun; ++rr)
      ++t_row_diff_base;

    ++t_w;
    ++t_eshelby_stress;
    ++t_invH;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HookeElement::setBlocks(
    MoFEM::Interface &m_field,
    boost::shared_ptr<map<int, BlockData>> &block_sets_ptr) {
  MoFEMFunctionBegin;

  if (!block_sets_ptr)
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "Pointer to block of sets is null");

  for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
           m_field, BLOCKSET | MAT_ELASTICSET, it)) {
    Mat_Elastic mydata;
    CHKERR it->getAttributeDataStructure(mydata);
    int id = it->getMeshsetId();
    auto &block_data = (*block_sets_ptr)[id];
    EntityHandle meshset = it->getMeshset();
    CHKERR m_field.get_moab().get_entities_by_dimension(meshset, 3,
                                                        block_data.tEts, true);
    block_data.iD = id;
    block_data.E = mydata.data.Young;
    block_data.PoissonRatio = mydata.data.Poisson;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HookeElement::addElasticElement(
    MoFEM::Interface &m_field,
    boost::shared_ptr<map<int, BlockData>> &block_sets_ptr,
    const std::string element_name, const std::string x_field,
    const std::string X_field, const bool ale) {
  MoFEMFunctionBegin;

  if (!block_sets_ptr)
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "Pointer to block of sets is null");

  CHKERR m_field.add_finite_element(element_name, MF_ZERO);
  CHKERR m_field.modify_finite_element_add_field_row(element_name, x_field);
  CHKERR m_field.modify_finite_element_add_field_col(element_name, x_field);
  CHKERR m_field.modify_finite_element_add_field_data(element_name, x_field);
  if (m_field.check_field(X_field)) {
    if (ale) {
      CHKERR m_field.modify_finite_element_add_field_row(element_name, X_field);
      CHKERR m_field.modify_finite_element_add_field_col(element_name, X_field);
    }
    CHKERR m_field.modify_finite_element_add_field_data(element_name, X_field);
  }

  for (auto &m : (*block_sets_ptr)) {
    CHKERR m_field.add_ents_to_finite_element_by_dim(m.second.tEts, 3,
                                                     element_name);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HookeElement::setOperators(
    boost::shared_ptr<ForcesAndSourcesCore> fe_lhs_ptr,
    boost::shared_ptr<ForcesAndSourcesCore> fe_rhs_ptr,
    boost::shared_ptr<map<int, BlockData>> block_sets_ptr,
    const std::string x_field, const std::string X_field, const bool ale,
    const bool field_disp, const EntityType type) {
  MoFEMFunctionBegin;

  if (!block_sets_ptr)
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "Pointer to block of sets is null");

  boost::shared_ptr<DataAtIntegrationPts> data_at_pts(
      new DataAtIntegrationPts());

  if (fe_lhs_ptr) {
    if (ale == PETSC_FALSE) {
      if (type == MBPRISM) {
        boost::shared_ptr<MatrixDouble> inv_jac_ptr(new MatrixDouble);
        fe_lhs_ptr->getOpPtrVector().push_back(
            new OpCalculateInvJacForFatPrism(inv_jac_ptr));
        fe_lhs_ptr->getOpPtrVector().push_back(
            new OpSetInvJacH1ForFatPrism(inv_jac_ptr));
      }
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpCalculateHomogeneousStiffness<true>(
              x_field, x_field, block_sets_ptr, data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpLhs_dx_dx<0>(x_field, x_field, data_at_pts));
    } else {
      if (type == MBPRISM) {
        boost::shared_ptr<MatrixDouble> inv_jac_ptr(new MatrixDouble);
        fe_lhs_ptr->getOpPtrVector().push_back(
            new OpCalculateInvJacForFatPrism(inv_jac_ptr));
        fe_lhs_ptr->getOpPtrVector().push_back(
            new OpSetInvJacH1ForFatPrism(inv_jac_ptr));
      }
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(X_field, data_at_pts->HMat));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpCalculateHomogeneousStiffness<true>(
              x_field, x_field, block_sets_ptr, data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(x_field, data_at_pts->hMat));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpCalculateStrainAle(x_field, x_field, data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpCalculateStress<0>(x_field, x_field, data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpAleLhs_dx_dx<0>(x_field, x_field, data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpAleLhs_dx_dX<0>(x_field, X_field, data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpCalculateEnergy(X_field, X_field, data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpCalculateEshelbyStress(X_field, X_field, data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpAleLhs_dX_dX<0>(X_field, X_field, data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpAleLhsPre_dX_dx<0>(X_field, x_field, data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpAleLhs_dX_dx(X_field, x_field, data_at_pts));
    }
  }

  if (fe_rhs_ptr) {

    if (ale == PETSC_FALSE) {
      if (type == MBPRISM) {
        boost::shared_ptr<MatrixDouble> inv_jac_ptr(new MatrixDouble);
        fe_rhs_ptr->getOpPtrVector().push_back(
            new OpCalculateInvJacForFatPrism(inv_jac_ptr));
        fe_rhs_ptr->getOpPtrVector().push_back(
            new OpSetInvJacH1ForFatPrism(inv_jac_ptr));
      }
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(x_field, data_at_pts->hMat));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpCalculateHomogeneousStiffness<0>(x_field, x_field,
                                                 block_sets_ptr, data_at_pts));
      if (field_disp) {
        fe_rhs_ptr->getOpPtrVector().push_back(
            new OpCalculateStrain<true>(x_field, x_field, data_at_pts));
      } else {
        fe_rhs_ptr->getOpPtrVector().push_back(
            new OpCalculateStrain<false>(x_field, x_field, data_at_pts));
      }
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpCalculateStress<0>(x_field, x_field, data_at_pts));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpRhs_dx(x_field, x_field, data_at_pts));
    } else {
      if (type == MBPRISM) {
        boost::shared_ptr<MatrixDouble> inv_jac_ptr(new MatrixDouble);
        fe_rhs_ptr->getOpPtrVector().push_back(
            new OpCalculateInvJacForFatPrism(inv_jac_ptr));
        fe_rhs_ptr->getOpPtrVector().push_back(
            new OpSetInvJacH1ForFatPrism(inv_jac_ptr));
      } 
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(X_field, data_at_pts->HMat));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpCalculateHomogeneousStiffness<0>(x_field, x_field,
                                                 block_sets_ptr, data_at_pts));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(x_field, data_at_pts->hMat));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpCalculateStrainAle(x_field, x_field, data_at_pts));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpCalculateStress<0>(x_field, x_field, data_at_pts));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpAleRhs_dx(x_field, x_field, data_at_pts));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpCalculateEnergy(X_field, X_field, data_at_pts));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpCalculateEshelbyStress(X_field, X_field, data_at_pts));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpAleRhs_dX(X_field, X_field, data_at_pts));
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HookeElement::calculateEnergy(
    DM dm, boost::shared_ptr<map<int, BlockData>> block_sets_ptr,
    const std::string x_field, const std::string X_field, const bool ale,
    const bool field_disp, Vec *v_energy_ptr) {
  MoFEMFunctionBegin;

  MoFEM::Interface *m_field_ptr;
  CHKERR DMoFEMGetInterfacePtr(dm, &m_field_ptr);

  int ghosts[] = {0};
  if (m_field_ptr->get_comm_rank() == 0) {
    CHKERR VecCreateGhost(m_field_ptr->get_comm(), 1, 1, 0, ghosts,
                          v_energy_ptr);
  } else {
    CHKERR VecCreateGhost(m_field_ptr->get_comm(), 0, 1, 1, ghosts,
                          v_energy_ptr);
  }

  boost::shared_ptr<DataAtIntegrationPts> data_at_pts(
      new DataAtIntegrationPts());

  boost::shared_ptr<ForcesAndSourcesCore> fe_ptr(
      new VolumeElementForcesAndSourcesCore(*m_field_ptr));

  if (ale == PETSC_FALSE) {
    fe_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(x_field, data_at_pts->hMat));
    fe_ptr->getOpPtrVector().push_back(new OpCalculateHomogeneousStiffness<0>(
        x_field, x_field, block_sets_ptr, data_at_pts));
    if (field_disp) {
      fe_ptr->getOpPtrVector().push_back(
          new OpCalculateStrain<true>(x_field, x_field, data_at_pts));
    } else {
      fe_ptr->getOpPtrVector().push_back(
          new OpCalculateStrain<false>(x_field, x_field, data_at_pts));
    }
    fe_ptr->getOpPtrVector().push_back(
        new OpCalculateStress<0>(x_field, x_field, data_at_pts));
    fe_ptr->getOpPtrVector().push_back(
        new OpCalculateEnergy(X_field, X_field, data_at_pts, *v_energy_ptr));
  } else {
    fe_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(X_field, data_at_pts->HMat));
    fe_ptr->getOpPtrVector().push_back(new OpCalculateHomogeneousStiffness<0>(
        x_field, x_field, block_sets_ptr, data_at_pts));
    fe_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(x_field, data_at_pts->hMat));
    fe_ptr->getOpPtrVector().push_back(
        new OpCalculateStrainAle(x_field, x_field, data_at_pts));
    fe_ptr->getOpPtrVector().push_back(
        new OpCalculateStress<0>(x_field, x_field, data_at_pts));
    fe_ptr->getOpPtrVector().push_back(
        new OpCalculateEnergy(X_field, X_field, data_at_pts, *v_energy_ptr));
  }

  CHKERR VecZeroEntries(*v_energy_ptr);
  CHKERR VecGhostUpdateBegin(*v_energy_ptr, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(*v_energy_ptr, INSERT_VALUES, SCATTER_FORWARD);

  fe_ptr->snes_ctx = SnesMethod::CTX_SNESNONE;
  PetscPrintf(PETSC_COMM_WORLD, "Calculate elastic energy  ...");
  CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", fe_ptr);
  PetscPrintf(PETSC_COMM_WORLD, " done\n");

  CHKERR VecAssemblyBegin(*v_energy_ptr);
  CHKERR VecAssemblyEnd(*v_energy_ptr);
  CHKERR VecGhostUpdateBegin(*v_energy_ptr, ADD_VALUES, SCATTER_REVERSE);
  CHKERR VecGhostUpdateEnd(*v_energy_ptr, ADD_VALUES, SCATTER_REVERSE);
  CHKERR VecGhostUpdateBegin(*v_energy_ptr, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(*v_energy_ptr, INSERT_VALUES, SCATTER_FORWARD);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode HookeElement::OpAleLhs_dX_dx::iNtegrate(EntData &row_data,
                                                       EntData &col_data) {
  MoFEMFunctionBegin;

  // get sub-block (3x3) of local stiffens matrix, here represented by
  // second order tensor
  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;
  FTensor::Index<'m', 3> m;
  FTensor::Index<'n', 3> n;

  // get element volume
  double vol = getVolume();

  // get intergrayion weights
  auto t_w = getFTensor0IntegrationWeight();

  // get derivatives of base functions on rows
  auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
  const int row_nb_base_fun = row_data.getN().size2();

  auto t_invH = getFTensor2FromMat<3, 3>(*dataAtPts->invHMat);
  auto &det_H = *dataAtPts->detHVec;

  auto get_eshelby_stress_dx = [this]() {
    FTensor::Tensor4<FTensor::PackPtr<double *, 1>, 3, 3, 3, 3>
        t_eshelby_stress_dx;
    int mm = 0;
    for (int ii = 0; ii != 3; ++ii)
      for (int jj = 0; jj != 3; ++jj)
        for (int kk = 0; kk != 3; ++kk)
          for (int ll = 0; ll != 3; ++ll)
            t_eshelby_stress_dx.ptr(ii, jj, kk, ll) =
                &(*dataAtPts->eshelbyStress_dx)(mm++, 0);
    return t_eshelby_stress_dx;
  };

  auto t_eshelby_stress_dx = get_eshelby_stress_dx();

  // iterate over integration points
  for (int gg = 0; gg != nbIntegrationPts; ++gg) {

    // calculate scalar weight times element volume
    double a = t_w * vol * det_H[gg];

    // iterate over row base functions
    int rr = 0;
    for (; rr != nbRows / 3; ++rr) {

      // get sub matrix for the row
      auto t_m = get_tensor2(K, 3 * rr, 0);

      FTensor::Tensor1<double, 3> t_row_diff_base_pulled;
      t_row_diff_base_pulled(i) = t_row_diff_base(j) * t_invH(j, i);

      FTensor::Tensor3<double, 3, 3, 3> t_row_stress_dx;
      t_row_stress_dx(i, k, l) =
          a * t_row_diff_base_pulled(j) * t_eshelby_stress_dx(i, j, k, l);

      // get derivatives of base functions for columns
      auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

      // iterate column base functions
      for (int cc = 0; cc != nbCols / 3; ++cc) {

        t_m(i, k) += t_row_stress_dx(i, k, l) * t_col_diff_base(l);

        // move to next column base function
        ++t_col_diff_base;

        // move to next block of local stiffens matrix
        ++t_m;
      }

      // move to next row base function
      ++t_row_diff_base;
    }

    for (; rr != row_nb_base_fun; ++rr)
      ++t_row_diff_base;

    ++t_w;
    ++t_invH;
    ++t_eshelby_stress_dx;
  }

  MoFEMFunctionReturn(0);
}