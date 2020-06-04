/**
 * \brief Operators and data structures for nonlinear elastic material
 *
 * Implementation of nonlinear elastic element.
 *
 * This file is part of MoFEM.
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
#include <Projection10NodeCoordsOnField.hpp>

#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <adolc/adolc.h>
#include <NonLinearElasticElement.hpp>

NonlinearElasticElement::MyVolumeFE::MyVolumeFE(MoFEM::Interface &m_field)
    : VolumeElementForcesAndSourcesCore(m_field), A(PETSC_NULL), F(PETSC_NULL),
      addToRule(1) {
  int ghosts[] = {0};
  if (mField.get_comm_rank() == 0) {
    ierr = VecCreateGhost(mField.get_comm(), 1, 1, 0, ghosts, &V);
  } else {
    ierr = VecCreateGhost(mField.get_comm(), 0, 1, 1, ghosts, &V);
  }
  CHKERRABORT(PETSC_COMM_SELF, ierr);
}

NonlinearElasticElement::MyVolumeFE::~MyVolumeFE() { 
  ierr = VecDestroy(&V);
  CHKERRABORT(PETSC_COMM_SELF, ierr);
}

int NonlinearElasticElement::MyVolumeFE::getRule(int order) {
  return 2 * (order - 1) + addToRule;
};

MoFEMErrorCode NonlinearElasticElement::MyVolumeFE::preProcess() {
  MoFEMFunctionBegin;

  CHKERR VolumeElementForcesAndSourcesCore::preProcess();

  if (A != PETSC_NULL) {
    snes_B = A;
  }

  if (F != PETSC_NULL) {
    snes_f = F;
  }

  switch (snes_ctx) {
  case CTX_SNESNONE:
    CHKERR VecZeroEntries(V);
    CHKERR VecGhostUpdateBegin(V, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(V, INSERT_VALUES, SCATTER_FORWARD);
    break;
  default:
    break;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NonlinearElasticElement::MyVolumeFE::postProcess() {
  MoFEMFunctionBegin;

  double *array;

  switch (snes_ctx) {
  case CTX_SNESNONE:
    CHKERR VecAssemblyBegin(V);
    CHKERR VecAssemblyEnd(V);
    CHKERR VecGhostUpdateBegin(V, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(V, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateBegin(V, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(V, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGetArray(V, &array);
    eNergy = array[0];
    CHKERR VecRestoreArray(V, &array);
    break;
  default:
    break;
  }

  CHKERR VolumeElementForcesAndSourcesCore::postProcess();

  MoFEMFunctionReturn(0);
}

NonlinearElasticElement::NonlinearElasticElement(MoFEM::Interface &m_field,
                                                 short int tag)
    : feRhs(m_field), feLhs(m_field), feEnergy(m_field), mField(m_field),
      tAg(tag) {}

NonlinearElasticElement::OpGetDataAtGaussPts::OpGetDataAtGaussPts(
    const std::string field_name,
    std::vector<VectorDouble> &values_at_gauss_pts,
    std::vector<MatrixDouble> &gardient_at_gauss_pts)
    : VolumeElementForcesAndSourcesCore::UserDataOperator(
          field_name, UserDataOperator::OPROW),
      valuesAtGaussPts(values_at_gauss_pts),
      gradientAtGaussPts(gardient_at_gauss_pts), zeroAtType(MBVERTEX) {}

MoFEMErrorCode NonlinearElasticElement::OpGetDataAtGaussPts::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  const int nb_dofs = data.getFieldData().size();
  const int nb_base_functions = data.getN().size2();
  if (nb_dofs == 0) {
    MoFEMFunctionReturnHot(0);
  }
  const int nb_gauss_pts = data.getN().size1();
  const int rank = data.getFieldDofs()[0]->getNbOfCoeffs();

  // initialize
  if (type == zeroAtType) {
    valuesAtGaussPts.resize(nb_gauss_pts);
    gradientAtGaussPts.resize(nb_gauss_pts);
    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      valuesAtGaussPts[gg].resize(rank, false);
      gradientAtGaussPts[gg].resize(rank, 3, false);
    }
    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      valuesAtGaussPts[gg].clear();
      gradientAtGaussPts[gg].clear();
    }
  }

  auto base_function = data.getFTensor0N();
  auto diff_base_functions = data.getFTensor1DiffN<3>();
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  if (rank == 1) {

    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      auto field_data = data.getFTensor0FieldData();
      double &val = valuesAtGaussPts[gg][0];
      FTensor::Tensor1<double *, 3> grad(&gradientAtGaussPts[gg](0, 0),
                                         &gradientAtGaussPts[gg](0, 1),
                                         &gradientAtGaussPts[gg](0, 2));
      int bb = 0;
      for (; bb != nb_dofs; bb++) {
        val += base_function * field_data;
        grad(i) += diff_base_functions(i) * field_data;
        ++diff_base_functions;
        ++base_function;
        ++field_data;
      }
      for (; bb != nb_base_functions; bb++) {
        ++diff_base_functions;
        ++base_function;
      }
    }

  } else if (rank == 3) {

    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      auto field_data = data.getFTensor1FieldData<3>();
      FTensor::Tensor1<double *, 3> values(&valuesAtGaussPts[gg][0],
                                           &valuesAtGaussPts[gg][1],
                                           &valuesAtGaussPts[gg][2]);
      FTensor::Tensor2<double *, 3, 3> gradient(
          &gradientAtGaussPts[gg](0, 0), &gradientAtGaussPts[gg](0, 1),
          &gradientAtGaussPts[gg](0, 2), &gradientAtGaussPts[gg](1, 0),
          &gradientAtGaussPts[gg](1, 1), &gradientAtGaussPts[gg](1, 2),
          &gradientAtGaussPts[gg](2, 0), &gradientAtGaussPts[gg](2, 1),
          &gradientAtGaussPts[gg](2, 2));
      int bb = 0;
      for (; bb != nb_dofs / 3; bb++) {
        values(i) += base_function * field_data(i);
        gradient(i, j) += field_data(i) * diff_base_functions(j);
        ++diff_base_functions;
        ++base_function;
        ++field_data;
      }
      for (; bb != nb_base_functions; bb++) {
        ++diff_base_functions;
        ++base_function;
      }
    }

  } else {
    // FIXME: THat part is inefficient
    VectorDouble &values = data.getFieldData();
    for (int gg = 0; gg < nb_gauss_pts; gg++) {
      VectorAdaptor N = data.getN(gg, nb_dofs / rank);
      MatrixAdaptor diffN = data.getDiffN(gg, nb_dofs / rank);
      for (int dd = 0; dd < nb_dofs / rank; dd++) {
        for (int rr1 = 0; rr1 < rank; rr1++) {
          valuesAtGaussPts[gg][rr1] += N[dd] * values[rank * dd + rr1];
          for (int rr2 = 0; rr2 < 3; rr2++) {
            gradientAtGaussPts[gg](rr1, rr2) +=
                diffN(dd, rr2) * values[rank * dd + rr1];
          }
        }
      }
    }
  }

  MoFEMFunctionReturn(0);
}

NonlinearElasticElement::OpGetCommonDataAtGaussPts::OpGetCommonDataAtGaussPts(
    const std::string field_name, CommonData &common_data)
    : OpGetDataAtGaussPts(field_name, common_data.dataAtGaussPts[field_name],
                          common_data.gradAtGaussPts[field_name]) {}

NonlinearElasticElement::OpJacobianPiolaKirchhoffStress::
    OpJacobianPiolaKirchhoffStress(const std::string field_name,
                                   BlockData &data, CommonData &common_data,
                                   int tag, bool jacobian, bool ale,
                                   bool field_disp)
    : VolumeElementForcesAndSourcesCore::UserDataOperator(
          field_name, UserDataOperator::OPROW),
      dAta(data), commonData(common_data), tAg(tag), adlocReturnValue(0),
      jAcobian(jacobian), fUnction(!jacobian), aLe(ale), fieldDisp(field_disp) {

}

MoFEMErrorCode
NonlinearElasticElement::OpJacobianPiolaKirchhoffStress::calculateStress(
    const int gg) {
  MoFEMFunctionBegin;

  ierr = dAta.materialAdoublePtr->calculateP_PiolaKirchhoffI(
      dAta, getNumeredEntFiniteElementPtr());
  CHKERRG(ierr);
  if (aLe) {
    dAta.materialAdoublePtr->P =
        dAta.materialAdoublePtr->detH *
        prod(dAta.materialAdoublePtr->P, trans(dAta.materialAdoublePtr->invH));
  }
  commonData.sTress[gg].resize(3, 3, false);
  for (int dd1 = 0; dd1 < 3; dd1++) {
    for (int dd2 = 0; dd2 < 3; dd2++) {
      dAta.materialAdoublePtr->P(dd1, dd2) >>=
          (commonData.sTress[gg])(dd1, dd2);
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
NonlinearElasticElement::OpJacobianPiolaKirchhoffStress::recordTag(
    const int gg) {
  MoFEMFunctionBegin;

  trace_on(tAg, 0);

  dAta.materialAdoublePtr->F.resize(3, 3, false);

  if (!aLe) {

    nbActiveVariables = 0;
    for (int dd1 = 0; dd1 < 3; dd1++) {
      for (int dd2 = 0; dd2 < 3; dd2++) {
        dAta.materialAdoublePtr->F(dd1, dd2) <<= (*ptrh)[gg](dd1, dd2);
        if (fieldDisp) {
          if (dd1 == dd2) {
            dAta.materialAdoublePtr->F(dd1, dd2) += 1;
          }
        }
        nbActiveVariables++;
      }
    }

  } else {

    nbActiveVariables = 0;

    dAta.materialAdoublePtr->h.resize(3, 3, false);
    for (int dd1 = 0; dd1 < 3; dd1++) {
      for (int dd2 = 0; dd2 < 3; dd2++) {
        dAta.materialAdoublePtr->h(dd1, dd2) <<= (*ptrh)[gg](dd1, dd2);
        nbActiveVariables++;
      }
    }

    dAta.materialAdoublePtr->H.resize(3, 3, false);
    for (int dd1 = 0; dd1 < 3; dd1++) {
      for (int dd2 = 0; dd2 < 3; dd2++) {
        dAta.materialAdoublePtr->H(dd1, dd2) <<= (*ptrH)[gg](dd1, dd2);
        nbActiveVariables++;
      }
    }

    CHKERR dAta.materialAdoublePtr->dEterminant(dAta.materialAdoublePtr->H,
                                                dAta.materialAdoublePtr->detH);
    dAta.materialAdoublePtr->invH.resize(3, 3, false);
    CHKERR dAta.materialAdoublePtr->iNvert(dAta.materialAdoublePtr->detH,
                                           dAta.materialAdoublePtr->H,
                                           dAta.materialAdoublePtr->invH);
    noalias(dAta.materialAdoublePtr->F) =
        prod(dAta.materialAdoublePtr->h, dAta.materialAdoublePtr->invH);
  }

  CHKERR dAta.materialAdoublePtr->setUserActiveVariables(nbActiveVariables);
  CHKERR calculateStress(gg);

  trace_off();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
NonlinearElasticElement::OpJacobianPiolaKirchhoffStress::playTag(const int gg) {
  MoFEMFunctionBeginHot;

  int r;

  if (fUnction) {
    commonData.sTress[gg].resize(3, 3, false);
    // play recorder for values
    r = ::function(tAg, 9, nbActiveVariables, &activeVariables[0],
                   &commonData.sTress[gg](0, 0));
    if (r < adlocReturnValue) { // function is locally analytic
      SETERRQ1(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
               "ADOL-C function evaluation with error r = %d", r);
    }
  }

  if (jAcobian) {
    commonData.jacStress[gg].resize(9, nbActiveVariables, false);
    double *jac_ptr[] = {
        &(commonData.jacStress[gg](0, 0)), &(commonData.jacStress[gg](1, 0)),
        &(commonData.jacStress[gg](2, 0)), &(commonData.jacStress[gg](3, 0)),
        &(commonData.jacStress[gg](4, 0)), &(commonData.jacStress[gg](5, 0)),
        &(commonData.jacStress[gg](6, 0)), &(commonData.jacStress[gg](7, 0)),
        &(commonData.jacStress[gg](8, 0))};
    // play recorder for jacobians
    r = jacobian(tAg, 9, nbActiveVariables, &activeVariables[0], jac_ptr);
    if (r < adlocReturnValue) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
              "ADOL-C function evaluation with error");
    }
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode NonlinearElasticElement::OpJacobianPiolaKirchhoffStress::doWork(
    int row_side, EntityType row_type,
    DataForcesAndSourcesCore::EntData &row_data) {
  MoFEMFunctionBegin;

  // do it only once, no need to repeat this for edges,faces or tets
  if (row_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tEts.end()) {
    MoFEMFunctionReturnHot(0);
  }

  int nb_dofs = row_data.getFieldData().size();
  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);
  dAta.materialAdoublePtr->commonDataPtr = &commonData;
  dAta.materialAdoublePtr->opPtr = this;

  int nb_gauss_pts = row_data.getN().size1();
  commonData.sTress.resize(nb_gauss_pts);
  commonData.jacStress.resize(nb_gauss_pts);

  ptrh = &(commonData.gradAtGaussPts[commonData.spatialPositions]);
  if (aLe) {
    ptrH = &(commonData.gradAtGaussPts[commonData.meshPositions]);
  }

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    dAta.materialAdoublePtr->gG = gg;

    // Record tag and calculate stress
    if (recordTagForIntegrationPoint(gg)) {
      CHKERR recordTag(gg);
    }

    // Set active variables vector
    if (jAcobian || (!recordTagForIntegrationPoint(gg))) {
      activeVariables.resize(nbActiveVariables, false);
      if (!aLe) {
        for (int dd1 = 0; dd1 < 3; dd1++) {
          for (int dd2 = 0; dd2 < 3; dd2++) {
            activeVariables(dd1 * 3 + dd2) = (*ptrh)[gg](dd1, dd2);
          }
        }
      } else {
        for (int dd1 = 0; dd1 < 3; dd1++) {
          for (int dd2 = 0; dd2 < 3; dd2++) {
            activeVariables(dd1 * 3 + dd2) = (*ptrh)[gg](dd1, dd2);
          }
        }
        for (int dd1 = 0; dd1 < 3; dd1++) {
          for (int dd2 = 0; dd2 < 3; dd2++) {
            activeVariables(9 + dd1 * 3 + dd2) = (*ptrH)[gg](dd1, dd2);
          }
        }
      }
      CHKERR dAta.materialAdoublePtr->setUserActiveVariables(activeVariables);

      // Play tag and calculate stress or tangent
      if (jAcobian || (!recordTagForIntegrationPoint(gg))) {
        CHKERR playTag(gg);
      }
    }
  }

  MoFEMFunctionReturn(0);
}

NonlinearElasticElement::OpJacobianEnergy::OpJacobianEnergy(
    const std::string
        field_name, ///< field name for spatial positions or displacements
    BlockData &data, CommonData &common_data, int tag, bool gradient,
    bool hessian, bool ale, bool field_disp)
    : VolumeElementForcesAndSourcesCore::UserDataOperator(
          field_name, UserDataOperator::OPROW),
      dAta(data), commonData(common_data), tAg(tag), gRadient(gradient),
      hEssian(hessian), aLe(ale), fieldDisp(field_disp) {}

MoFEMErrorCode
NonlinearElasticElement::OpJacobianEnergy::calculateEnergy(const int gg) {
  MoFEMFunctionBegin;
  CHKERR dAta.materialAdoublePtr->calculateElasticEnergy(
      dAta, getNumeredEntFiniteElementPtr());
  dAta.materialAdoublePtr->eNergy >>= commonData.eNergy[gg];
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
NonlinearElasticElement::OpJacobianEnergy::recordTag(const int gg) {
  MoFEMFunctionBegin;

  trace_on(tAg, 0);

  dAta.materialAdoublePtr->F.resize(3, 3, false);

  if (!aLe) {

    nbActiveVariables = 0;
    for (int dd1 = 0; dd1 < 3; dd1++) {
      for (int dd2 = 0; dd2 < 3; dd2++) {
        dAta.materialAdoublePtr->F(dd1, dd2) <<= (*ptrh)[gg](dd1, dd2);
        if (fieldDisp) {
          if (dd1 == dd2) {
            dAta.materialAdoublePtr->F(dd1, dd2) += 1;
          }
        }
        nbActiveVariables++;
      }
    }

  } else {

    nbActiveVariables = 0;

    dAta.materialAdoublePtr->h.resize(3, 3, false);
    for (int dd1 = 0; dd1 < 3; dd1++) {
      for (int dd2 = 0; dd2 < 3; dd2++) {
        dAta.materialAdoublePtr->h(dd1, dd2) <<= (*ptrh)[gg](dd1, dd2);
        nbActiveVariables++;
      }
    }

    dAta.materialAdoublePtr->H.resize(3, 3, false);
    for (int dd1 = 0; dd1 < 3; dd1++) {
      for (int dd2 = 0; dd2 < 3; dd2++) {
        dAta.materialAdoublePtr->H(dd1, dd2) <<= (*ptrH)[gg](dd1, dd2);
        nbActiveVariables++;
      }
    }

    CHKERR dAta.materialAdoublePtr->dEterminant(dAta.materialAdoublePtr->H,
                                                dAta.materialAdoublePtr->detH);
    dAta.materialAdoublePtr->invH.resize(3, 3, false);
    CHKERR dAta.materialAdoublePtr->iNvert(dAta.materialAdoublePtr->detH,
                                           dAta.materialAdoublePtr->H,
                                           dAta.materialAdoublePtr->invH);
    noalias(dAta.materialAdoublePtr->F) =
        prod(dAta.materialAdoublePtr->h, dAta.materialAdoublePtr->invH);
  }

  CHKERR dAta.materialAdoublePtr->setUserActiveVariables(nbActiveVariables);
  CHKERR calculateEnergy(gg);

  trace_off();

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
NonlinearElasticElement::OpJacobianEnergy::playTag(const int gg) {
  MoFEMFunctionBegin;

  if (gRadient) {
    commonData.jacEnergy[gg].resize(nbActiveVariables, false);
    int r = ::gradient(tAg, nbActiveVariables, &activeVariables[0],
                       &commonData.jacEnergy[gg][0]);
    if (r < 0) {
      // That means that energy function is not smooth and derivative
      // can not be calculated,
      SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
              "ADOL-C function evaluation with error");
    }
  }

  if (hEssian) {
    commonData.hessianEnergy[gg].resize(nbActiveVariables * nbActiveVariables,
                                        false);
    double *H[nbActiveVariables];
    for (int n = 0; n != nbActiveVariables; n++) {
      H[n] = &(commonData.hessianEnergy[gg][n * nbActiveVariables]);
    }
    int r = ::hessian(tAg, nbActiveVariables, &*activeVariables.begin(), H);
    if (r < 0) {
      // That means that energy function is not smooth and derivative
      // can not be calculated,
      SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
              "ADOL-C function evaluation with error");
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NonlinearElasticElement::OpJacobianEnergy::doWork(
    int row_side, EntityType row_type,
    DataForcesAndSourcesCore::EntData &row_data) {
  MoFEMFunctionBegin;

  // do it only once, no need to repeat this for edges,faces or tets
  if (row_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tEts.end()) {
    MoFEMFunctionReturnHot(0);
  }

  int nb_dofs = row_data.getFieldData().size();
  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);
  dAta.materialAdoublePtr->commonDataPtr = &commonData;
  dAta.materialAdoublePtr->opPtr = this;

  int nb_gauss_pts = row_data.getN().size1();
  commonData.eNergy.resize(nb_gauss_pts);
  commonData.jacEnergy.resize(nb_gauss_pts);

  ptrh = &(commonData.gradAtGaussPts[commonData.spatialPositions]);
  if (aLe) {
    ptrH = &(commonData.gradAtGaussPts[commonData.meshPositions]);
  }

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    dAta.materialAdoublePtr->gG = gg;

    // Record tag and calualte stress
    if (recordTagForIntegrationPoint(gg)) {
      CHKERR recordTag(gg);
    }

    activeVariables.resize(nbActiveVariables, false);
    if (!aLe) {
      for (int dd1 = 0; dd1 < 3; dd1++) {
        for (int dd2 = 0; dd2 < 3; dd2++) {
          activeVariables(dd1 * 3 + dd2) = (*ptrh)[gg](dd1, dd2);
        }
      }
    } else {
      for (int dd1 = 0; dd1 < 3; dd1++) {
        for (int dd2 = 0; dd2 < 3; dd2++) {
          activeVariables(dd1 * 3 + dd2) = (*ptrh)[gg](dd1, dd2);
        }
      }
      for (int dd1 = 0; dd1 < 3; dd1++) {
        for (int dd2 = 0; dd2 < 3; dd2++) {
          activeVariables(9 + dd1 * 3 + dd2) = (*ptrH)[gg](dd1, dd2);
        }
      }
    }
    CHKERR dAta.materialAdoublePtr->setUserActiveVariables(activeVariables);

    // Play tag and calculate stress or tangent
    CHKERR playTag(gg);
  }

  MoFEMFunctionReturn(0);
}

NonlinearElasticElement::OpRhsPiolaKirchhoff::OpRhsPiolaKirchhoff(
    const std::string field_name, BlockData &data, CommonData &common_data)
    : VolumeElementForcesAndSourcesCore::UserDataOperator(
          field_name, UserDataOperator::OPROW),
      dAta(data), commonData(common_data), aLe(false) {}

MoFEMErrorCode NonlinearElasticElement::OpRhsPiolaKirchhoff::aSemble(
    int row_side, EntityType row_type,
    DataForcesAndSourcesCore::EntData &row_data) {
  MoFEMFunctionBegin;

  int nb_dofs = row_data.getIndices().size();
  int *indices_ptr = &row_data.getIndices()[0];
  if (!dAta.forcesOnlyOnEntitiesRow.empty()) {
    iNdices.resize(nb_dofs, false);
    noalias(iNdices) = row_data.getIndices();
    indices_ptr = &iNdices[0];
    VectorDofs &dofs = row_data.getFieldDofs();
    VectorDofs::iterator dit = dofs.begin();
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (dAta.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
          dAta.forcesOnlyOnEntitiesRow.end()) {
        iNdices[ii] = -1;
      }
    }
  }
  CHKERR VecSetValues(getFEMethod()->snes_f, nb_dofs, indices_ptr, &nf[0],
                      ADD_VALUES);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NonlinearElasticElement::OpRhsPiolaKirchhoff::doWork(
    int row_side, EntityType row_type,
    DataForcesAndSourcesCore::EntData &row_data) {
  MoFEMFunctionBegin;

  if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tEts.end()) {
    MoFEMFunctionReturnHot(0);
  }

  const int nb_dofs = row_data.getIndices().size();
  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);
  if ((unsigned int)nb_dofs > 3 * row_data.getN().size2()) {
    SETERRQ(PETSC_COMM_SELF, 1, "data inconsistency");
  }
  const int nb_base_functions = row_data.getN().size2();
  const int nb_gauss_pts = row_data.getN().size1();

  nf.resize(nb_dofs, false);
  nf.clear();

  FTensor::Tensor1<double *, 3> diff_base_functions =
      row_data.getFTensor1DiffN<3>();
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    double val = getVolume() * getGaussPts()(3, gg);
    if ((!aLe) && getHoGaussPtsDetJac().size() > 0) {
      val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
    }
    MatrixDouble3by3 &stress = commonData.sTress[gg];
    FTensor::Tensor2<double *, 3, 3> t3(
        &stress(0, 0), &stress(0, 1), &stress(0, 2), &stress(1, 0),
        &stress(1, 1), &stress(1, 2), &stress(2, 0), &stress(2, 1),
        &stress(2, 2));
    FTensor::Tensor1<double *, 3> rhs(&nf[0], &nf[1], &nf[2], 3);
    int bb = 0;
    for (; bb != nb_dofs / 3; bb++) {
      rhs(i) += val * t3(i, j) * diff_base_functions(j);
      ++rhs;
      ++diff_base_functions;
    }
    for (; bb != nb_base_functions; bb++) {
      ++diff_base_functions;
    }
  }

  CHKERR aSemble(row_side, row_type, row_data);

  MoFEMFunctionReturn(0);
}

NonlinearElasticElement::OpEnergy::OpEnergy(const std::string field_name,
                                            BlockData &data,
                                            CommonData &common_data, Vec ghost_vec,
                                            bool field_disp)
    : VolumeElementForcesAndSourcesCore::UserDataOperator(
          field_name, UserDataOperator::OPROW),
      dAta(data), commonData(common_data), ghostVec(ghost_vec), fieldDisp(field_disp) {
  ierr = PetscObjectReference((PetscObject)ghostVec);
  CHKERRABORT(PETSC_COMM_SELF, ierr);
}

NonlinearElasticElement::OpEnergy::~OpEnergy() { 
  ierr = VecDestroy(&ghostVec);
  CHKERRABORT(PETSC_COMM_SELF, ierr);
}

MoFEMErrorCode NonlinearElasticElement::OpEnergy::doWork(
    int row_side, EntityType row_type,
    DataForcesAndSourcesCore::EntData &row_data) {
  MoFEMFunctionBegin;

  if (row_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);
  if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tEts.end()) {
    MoFEMFunctionReturnHot(0);
  }

  std::vector<MatrixDouble> &F =
      (commonData.gradAtGaussPts[commonData.spatialPositions]);
  dAta.materialDoublePtr->F.resize(3, 3, false);

  double *energy_ptr;
  CHKERR VecGetArray(ghostVec, &energy_ptr);

  for (unsigned int gg = 0; gg != row_data.getN().size1(); ++gg) {
    double val = getVolume() * getGaussPts()(3, gg);
    if (getHoGaussPtsDetJac().size() > 0) {
      val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
    }

    noalias(dAta.materialDoublePtr->F) = F[gg];
    if (fieldDisp) {
      for (int dd = 0; dd < 3; dd++) {
        dAta.materialDoublePtr->F(dd, dd) += 1;
      }
    }

    int nb_active_variables = 0;
    CHKERR dAta.materialDoublePtr->setUserActiveVariables(nb_active_variables);
    CHKERR dAta.materialDoublePtr->calculateElasticEnergy(
        dAta, getNumeredEntFiniteElementPtr());
    energy_ptr[0] += val * dAta.materialDoublePtr->eNergy;
  }
  CHKERR VecRestoreArray(ghostVec, &energy_ptr);

  MoFEMFunctionReturn(0);
}

NonlinearElasticElement::OpLhsPiolaKirchhoff_dx::OpLhsPiolaKirchhoff_dx(
    const std::string vel_field, const std::string field_name, BlockData &data,
    CommonData &common_data)
    : VolumeElementForcesAndSourcesCore::UserDataOperator(
          vel_field, field_name, UserDataOperator::OPROWCOL),
      dAta(data), commonData(common_data), aLe(false) {}

template <int S>
static MoFEMErrorCode get_jac(DataForcesAndSourcesCore::EntData &col_data,
                              int gg, MatrixDouble &jac_stress,
                              MatrixDouble &jac) {
  jac.clear();
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  int nb_col = col_data.getFieldData().size();
  double *diff_ptr =
      const_cast<double *>(&(col_data.getDiffN(gg, nb_col / 3)(0, 0)));
  // First two indices 'i','j' derivatives of 1st Piola-stress, third index 'k'
  // is displacement component
  FTensor::Tensor3<FTensor::PackPtr<double *,3>, 3, 3, 3> t3_1_0(
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
  FTensor::Tensor3<FTensor::PackPtr<double *,3>, 3, 3, 3> t3_1_1(
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
  FTensor::Tensor3<FTensor::PackPtr<double *,3>, 3, 3, 3> t3_1_2(
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

MoFEMErrorCode NonlinearElasticElement::OpLhsPiolaKirchhoff_dx::getJac(
    DataForcesAndSourcesCore::EntData &col_data, int gg) {
  return get_jac<0>(col_data, gg, commonData.jacStress[gg], jac);
}

MoFEMErrorCode NonlinearElasticElement::OpLhsPiolaKirchhoff_dx::aSemble(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  int nb_row = row_data.getIndices().size();
  int nb_col = col_data.getIndices().size();

  int *row_indices_ptr = &row_data.getIndices()[0];
  int *col_indices_ptr = &col_data.getIndices()[0];

  /*for(int dd1 = 0;dd1<k.size1();dd1++) {
    for(int dd2 = 0;dd2<k.size2();dd2++) {
      if(k(dd1,dd2)!=k(dd1,dd2)) {
        SETERRQ(PETSC_COMM_SELF,1,"Wrong result");
      }
    }
  }*/

  if (!dAta.forcesOnlyOnEntitiesRow.empty()) {
    rowIndices.resize(nb_row, false);
    noalias(rowIndices) = row_data.getIndices();
    row_indices_ptr = &rowIndices[0];
    VectorDofs &dofs = row_data.getFieldDofs();
    VectorDofs::iterator dit = dofs.begin();
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (dAta.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
          dAta.forcesOnlyOnEntitiesRow.end()) {
        rowIndices[ii] = -1;
      }
    }
  }

  if (!dAta.forcesOnlyOnEntitiesCol.empty()) {
    colIndices.resize(nb_col, false);
    noalias(colIndices) = col_data.getIndices();
    col_indices_ptr = &colIndices[0];
    VectorDofs &dofs = col_data.getFieldDofs();
    VectorDofs::iterator dit = dofs.begin();
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (dAta.forcesOnlyOnEntitiesCol.find((*dit)->getEnt()) ==
          dAta.forcesOnlyOnEntitiesCol.end()) {
        colIndices[ii] = -1;
      }
    }
  }

  CHKERR MatSetValues(getFEMethod()->snes_B, nb_row, row_indices_ptr, nb_col,
                      col_indices_ptr, &k(0, 0), ADD_VALUES);

  // is symmetric
  if (row_side != col_side || row_type != col_type) {

    row_indices_ptr = &row_data.getIndices()[0];
    col_indices_ptr = &col_data.getIndices()[0];

    if (!dAta.forcesOnlyOnEntitiesCol.empty()) {
      rowIndices.resize(nb_row, false);
      noalias(rowIndices) = row_data.getIndices();
      row_indices_ptr = &rowIndices[0];
      VectorDofs &dofs = row_data.getFieldDofs();
      VectorDofs::iterator dit = dofs.begin();
      for (int ii = 0; dit != dofs.end(); dit++, ii++) {
        if (dAta.forcesOnlyOnEntitiesCol.find((*dit)->getEnt()) ==
            dAta.forcesOnlyOnEntitiesCol.end()) {
          rowIndices[ii] = -1;
        }
      }
    }

    if (!dAta.forcesOnlyOnEntitiesRow.empty()) {
      colIndices.resize(nb_col, false);
      noalias(colIndices) = col_data.getIndices();
      col_indices_ptr = &colIndices[0];
      VectorDofs &dofs = col_data.getFieldDofs();
      VectorDofs::iterator dit = dofs.begin();
      for (int ii = 0; dit != dofs.end(); dit++, ii++) {
        if (dAta.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
            dAta.forcesOnlyOnEntitiesRow.end()) {
          colIndices[ii] = -1;
        }
      }
    }

    trans_k.resize(nb_col, nb_row, false);
    noalias(trans_k) = trans(k);
    CHKERR MatSetValues(getFEMethod()->snes_B, nb_col, col_indices_ptr, nb_row,
                        row_indices_ptr, &trans_k(0, 0), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NonlinearElasticElement::OpLhsPiolaKirchhoff_dx::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  int nb_row = row_data.getIndices().size();
  int nb_col = col_data.getIndices().size();
  if (nb_row == 0)
    MoFEMFunctionReturnHot(0);
  if (nb_col == 0)
    MoFEMFunctionReturnHot(0);

  if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tEts.end()) {
    MoFEMFunctionReturnHot(0);
  }

  // const int nb_base_functions = row_data.getN().size2();
  const int nb_gauss_pts = row_data.getN().size1();

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'m', 3> m;

  k.resize(nb_row, nb_col, false);
  k.clear();
  jac.resize(9, nb_col, false);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    CHKERR getJac(col_data, gg);
    double val = getVolume() * getGaussPts()(3, gg);
    if ((!aLe) && (getHoGaussPtsDetJac().size() > 0)) {
      val *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
    }
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
    for (int cc = 0; cc != nb_col / 3; cc++) {
      FTensor::Tensor1<double *, 3> diff_base_functions =
          row_data.getFTensor1DiffN<3>(gg, 0);
      FTensor::Tensor2<double *, 3, 3> lhs(
          &k(0, 3 * cc + 0), &k(0, 3 * cc + 1), &k(0, 3 * cc + 2),
          &k(1, 3 * cc + 0), &k(1, 3 * cc + 1), &k(1, 3 * cc + 2),
          &k(2, 3 * cc + 0), &k(2, 3 * cc + 1), &k(2, 3 * cc + 2), 3 * nb_col);
      for (int rr = 0; rr != nb_row / 3; rr++) {
        lhs(i, j) += val * t3_1(i, m, j) * diff_base_functions(m);
        ++diff_base_functions;
        ++lhs;
      }
      ++t3_1;
    }
  }

  CHKERR aSemble(row_side, col_side, row_type, col_type, row_data, col_data);

  MoFEMFunctionReturn(0);
}

NonlinearElasticElement::OpLhsPiolaKirchhoff_dX::OpLhsPiolaKirchhoff_dX(
    const std::string vel_field, const std::string field_name, BlockData &data,
    CommonData &common_data)
    : OpLhsPiolaKirchhoff_dx(vel_field, field_name, data, common_data) {
  sYmm = false;
}

MoFEMErrorCode NonlinearElasticElement::OpLhsPiolaKirchhoff_dX::getJac(
    DataForcesAndSourcesCore::EntData &col_data, int gg) {
  return get_jac<9>(col_data, gg, commonData.jacStress[gg], jac);
}

MoFEMErrorCode NonlinearElasticElement::OpLhsPiolaKirchhoff_dX::aSemble(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  int nb_row = row_data.getIndices().size();
  int nb_col = col_data.getIndices().size();

  int *row_indices_ptr = &row_data.getIndices()[0];
  if (!dAta.forcesOnlyOnEntitiesRow.empty()) {
    rowIndices.resize(nb_row, false);
    noalias(rowIndices) = row_data.getIndices();
    row_indices_ptr = &rowIndices[0];
    VectorDofs &dofs = row_data.getFieldDofs();
    VectorDofs::iterator dit = dofs.begin();
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (dAta.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
          dAta.forcesOnlyOnEntitiesRow.end()) {
        rowIndices[ii] = -1;
      }
    }
  }

  int *col_indices_ptr = &col_data.getIndices()[0];
  if (!dAta.forcesOnlyOnEntitiesCol.empty()) {
    colIndices.resize(nb_col, false);
    noalias(colIndices) = col_data.getIndices();
    col_indices_ptr = &colIndices[0];
    VectorDofs &dofs = col_data.getFieldDofs();
    VectorDofs::iterator dit = dofs.begin();
    for (int ii = 0; dit != dofs.end(); dit++, ii++) {
      if (dAta.forcesOnlyOnEntitiesCol.find((*dit)->getEnt()) ==
          dAta.forcesOnlyOnEntitiesCol.end()) {
        colIndices[ii] = -1;
      }
    }
  }

  /*for(int dd1 = 0;dd1<k.size1();dd1++) {
    for(int dd2 = 0;dd2<k.size2();dd2++) {
      if(k(dd1,dd2)!=k(dd1,dd2)) {
        SETERRQ(PETSC_COMM_SELF,1,"Wrong result");
      }
    }
  }*/

  CHKERR MatSetValues(getFEMethod()->snes_B, nb_row, row_indices_ptr, nb_col,
                      col_indices_ptr, &k(0, 0), ADD_VALUES);

  MoFEMFunctionReturn(0);
}

NonlinearElasticElement::OpJacobianEshelbyStress::OpJacobianEshelbyStress(
    const std::string field_name, BlockData &data, CommonData &common_data,
    int tag, bool jacobian, bool ale)
    : OpJacobianPiolaKirchhoffStress(field_name, data, common_data, tag,
                                     jacobian, ale, false) {}

MoFEMErrorCode
NonlinearElasticElement::OpJacobianEshelbyStress::calculateStress(
    const int gg) {
  MoFEMFunctionBeginHot;

  CHKERR dAta.materialAdoublePtr->calculateSiGma_EshelbyStress(
      dAta, getNumeredEntFiniteElementPtr());
  if (aLe) {
    dAta.materialAdoublePtr->SiGma = dAta.materialAdoublePtr->detH *
                                     prod(dAta.materialAdoublePtr->SiGma,
                                          trans(dAta.materialAdoublePtr->invH));
  }
  commonData.sTress[gg].resize(3, 3, false);
  for (int dd1 = 0; dd1 < 3; dd1++) {
    for (int dd2 = 0; dd2 < 3; dd2++) {
      dAta.materialAdoublePtr->SiGma(dd1, dd2) >>=
          (commonData.sTress[gg])(dd1, dd2);
    }
  }

  MoFEMFunctionReturnHot(0);
}

NonlinearElasticElement::OpRhsEshelbyStress::OpRhsEshelbyStress(
    const std::string field_name, BlockData &data, CommonData &common_data)
    : OpRhsPiolaKirchhoff(field_name, data, common_data) {}

NonlinearElasticElement::OpLhsEshelby_dx::OpLhsEshelby_dx(
    const std::string vel_field, const std::string field_name, BlockData &data,
    CommonData &common_data)
    : OpLhsPiolaKirchhoff_dX(vel_field, field_name, data, common_data) {}

MoFEMErrorCode NonlinearElasticElement::OpLhsEshelby_dx::getJac(
    DataForcesAndSourcesCore::EntData &col_data, int gg) {
  return get_jac<0>(col_data, gg, commonData.jacStress[gg], jac);
}

NonlinearElasticElement::OpLhsEshelby_dX::OpLhsEshelby_dX(
    const std::string vel_field, const std::string field_name, BlockData &data,
    CommonData &common_data)
    : OpLhsPiolaKirchhoff_dx(vel_field, field_name, data, common_data) {}

MoFEMErrorCode NonlinearElasticElement::OpLhsEshelby_dX::getJac(
    DataForcesAndSourcesCore::EntData &col_data, int gg) {
  return get_jac<9>(col_data, gg, commonData.jacStress[gg], jac);
}

MoFEMErrorCode NonlinearElasticElement::setBlocks(
    boost::shared_ptr<FunctionsToCalculatePiolaKirchhoffI<double>>
        materialDoublePtr,
    boost::shared_ptr<FunctionsToCalculatePiolaKirchhoffI<adouble>>
        materialAdoublePtr) {
  MoFEMFunctionBegin;

  if (!materialDoublePtr) {
    SETERRQ(mField.get_comm(), MOFEM_DATA_INCONSISTENCY,
            "Pointer for materialDoublePtr not allocated");
  }
  if (!materialAdoublePtr) {
    SETERRQ(mField.get_comm(), MOFEM_DATA_INCONSISTENCY,
            "Pointer for materialAdoublePtr not allocated");
  }

  for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
           mField, BLOCKSET | MAT_ELASTICSET, it)) {
    Mat_Elastic mydata;
    CHKERR it->getAttributeDataStructure(mydata);
    int id = it->getMeshsetId();
    EntityHandle meshset = it->getMeshset();
    CHKERR mField.get_moab().get_entities_by_type(meshset, MBTET,
                                                  setOfBlocks[id].tEts, true);
    setOfBlocks[id].iD = id;
    setOfBlocks[id].E = mydata.data.Young;
    setOfBlocks[id].PoissonRatio = mydata.data.Poisson;
    setOfBlocks[id].materialDoublePtr = materialDoublePtr;
    setOfBlocks[id].materialAdoublePtr = materialAdoublePtr;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NonlinearElasticElement::addElement(
    const std::string element_name,
    const std::string spatial_position_field_name,
    const std::string material_position_field_name, const bool ale) {
  MoFEMFunctionBegin;

  CHKERR mField.add_finite_element(element_name, MF_ZERO);
  CHKERR mField.modify_finite_element_add_field_row(
      element_name, spatial_position_field_name);
  CHKERR mField.modify_finite_element_add_field_col(
      element_name, spatial_position_field_name);
  CHKERR mField.modify_finite_element_add_field_data(
      element_name, spatial_position_field_name);
  if (mField.check_field(material_position_field_name)) {
    if (ale) {
      CHKERR mField.modify_finite_element_add_field_row(
          element_name, material_position_field_name);
      CHKERR mField.modify_finite_element_add_field_col(
          element_name, material_position_field_name);
    }
    CHKERR mField.modify_finite_element_add_field_data(
        element_name, material_position_field_name);
  }

  std::map<int, BlockData>::iterator sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    CHKERR mField.add_ents_to_finite_element_by_type(sit->second.tEts, MBTET,
                                                     element_name);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NonlinearElasticElement::setOperators(
    const std::string spatial_position_field_name,
    const std::string material_position_field_name, const bool ale,
    const bool field_disp) {
  MoFEMFunctionBegin;

  commonData.spatialPositions = spatial_position_field_name;
  commonData.meshPositions = material_position_field_name;

  // Rhs
  feRhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(spatial_position_field_name, commonData));
  if (mField.check_field(material_position_field_name)) {
    feRhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
        material_position_field_name, commonData));
  }
  std::map<int, BlockData>::iterator sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    feRhs.getOpPtrVector().push_back(new OpJacobianPiolaKirchhoffStress(
        spatial_position_field_name, sit->second, commonData, tAg, false, ale,
        field_disp));
    feRhs.getOpPtrVector().push_back(new OpRhsPiolaKirchhoff(
        spatial_position_field_name, sit->second, commonData));
  }

  // Energy
  feEnergy.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(spatial_position_field_name, commonData));
  if (mField.check_field(material_position_field_name)) {
    feEnergy.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
        material_position_field_name, commonData));
  }
  sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    feEnergy.getOpPtrVector().push_back(
        new OpEnergy(spatial_position_field_name, sit->second, commonData,
                     feEnergy.V, field_disp));
  }

  // Lhs
  feLhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(spatial_position_field_name, commonData));
  if (mField.check_field(material_position_field_name)) {
    feLhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
        material_position_field_name, commonData));
  }
  sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    feLhs.getOpPtrVector().push_back(new OpJacobianPiolaKirchhoffStress(
        spatial_position_field_name, sit->second, commonData, tAg, true, ale,
        field_disp));
    feLhs.getOpPtrVector().push_back(new OpLhsPiolaKirchhoff_dx(
        spatial_position_field_name, spatial_position_field_name, sit->second,
        commonData));
  }

  MoFEMFunctionReturn(0);
}
