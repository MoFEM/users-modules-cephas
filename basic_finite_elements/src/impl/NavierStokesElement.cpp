/**
 * \file NavierStokesElement.cpp
 * \example NavierStokesElement.cpp
 *
 * \brief Implementation of operators for fluid flow
 *
 * Implementation of operators for simulation of the fluid flow governed by
 * Stokes and Navier-Stokes equations, and computation of the drag
 * force on a fluid-solid inteface
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
using namespace boost::numeric;
#include <MethodForForceScaling.hpp>
#include <NavierStokesElement.hpp>

MoFEMErrorCode NavierStokesElement::setNavierStokesOperators(
    boost::shared_ptr<VolumeElementForcesAndSourcesCore> fe_rhs_ptr,
    boost::shared_ptr<VolumeElementForcesAndSourcesCore> fe_lhs_ptr,
    const std::string velocity_field, const std::string pressure_field,
    boost::shared_ptr<CommonData> common_data, const EntityType type) {
  MoFEMFunctionBegin;

  for (auto &sit : common_data->setOfBlocksData) {

    if (type == MBPRISM) {
      boost::shared_ptr<MatrixDouble> inv_jac_ptr(new MatrixDouble);
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpMultiplyDeterminantOfJacobianAndWeightsForFatPrisms());
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpCalculateInvJacForFatPrism(inv_jac_ptr));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpSetInvJacH1ForFatPrism(inv_jac_ptr));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpMultiplyDeterminantOfJacobianAndWeightsForFatPrisms());
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpCalculateInvJacForFatPrism(inv_jac_ptr));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpSetInvJacH1ForFatPrism(inv_jac_ptr));
    }

    fe_lhs_ptr->getOpPtrVector().push_back(new OpCalculateVectorFieldValues<3>(
        velocity_field, common_data->velPtr));
    fe_lhs_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(velocity_field,
                                                 common_data->gradVelPtr));

    fe_lhs_ptr->getOpPtrVector().push_back(new OpAssembleLhsDiagLin(
        velocity_field, velocity_field, common_data, sit.second));
    fe_lhs_ptr->getOpPtrVector().push_back(new OpAssembleLhsDiagNonLin(
        velocity_field, velocity_field, common_data, sit.second));
    fe_lhs_ptr->getOpPtrVector().push_back(new OpAssembleLhsOffDiag(
        velocity_field, pressure_field, common_data, sit.second));

    fe_rhs_ptr->getOpPtrVector().push_back(new OpCalculateVectorFieldValues<3>(
        velocity_field, common_data->velPtr));
    fe_rhs_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(velocity_field,
                                                 common_data->gradVelPtr));
    fe_rhs_ptr->getOpPtrVector().push_back(new OpCalculateScalarFieldValues(
        pressure_field, common_data->pressPtr));

    fe_rhs_ptr->getOpPtrVector().push_back(
        new OpAssembleRhsVelocityLin(velocity_field, common_data, sit.second));
    fe_rhs_ptr->getOpPtrVector().push_back(new OpAssembleRhsVelocityNonLin(
        velocity_field, common_data, sit.second));
    fe_rhs_ptr->getOpPtrVector().push_back(
        new OpAssembleRhsPressure(pressure_field, common_data, sit.second));
  }

  MoFEMFunctionReturn(0);
};

MoFEMErrorCode NavierStokesElement::setStokesOperators(
    boost::shared_ptr<VolumeElementForcesAndSourcesCore> fe_rhs_ptr,
    boost::shared_ptr<VolumeElementForcesAndSourcesCore> fe_lhs_ptr,
    const std::string velocity_field, const std::string pressure_field,
    boost::shared_ptr<CommonData> common_data, const EntityType type) {
  MoFEMFunctionBegin;

  for (auto &sit : common_data->setOfBlocksData) {

    if (type == MBPRISM) {
      boost::shared_ptr<MatrixDouble> inv_jac_ptr(new MatrixDouble);
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpMultiplyDeterminantOfJacobianAndWeightsForFatPrisms());
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpCalculateInvJacForFatPrism(inv_jac_ptr));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpSetInvJacH1ForFatPrism(inv_jac_ptr));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpMultiplyDeterminantOfJacobianAndWeightsForFatPrisms());
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpCalculateInvJacForFatPrism(inv_jac_ptr));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpSetInvJacH1ForFatPrism(inv_jac_ptr));
    }

    fe_lhs_ptr->getOpPtrVector().push_back(new OpAssembleLhsDiagLin(
        velocity_field, velocity_field, common_data, sit.second));
    fe_lhs_ptr->getOpPtrVector().push_back(new OpAssembleLhsOffDiag(
        velocity_field, pressure_field, common_data, sit.second));

    fe_rhs_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(velocity_field,
                                                 common_data->gradVelPtr));

    fe_rhs_ptr->getOpPtrVector().push_back(new OpCalculateScalarFieldValues(
        pressure_field, common_data->pressPtr));
    fe_rhs_ptr->getOpPtrVector().push_back(
        new OpAssembleRhsVelocityLin(velocity_field, common_data, sit.second));
    fe_rhs_ptr->getOpPtrVector().push_back(
        new OpAssembleRhsPressure(pressure_field, common_data, sit.second));
  }

  MoFEMFunctionReturn(0);
};

MoFEMErrorCode NavierStokesElement::setCalcVolumeFluxOperators(
    boost::shared_ptr<VolumeElementForcesAndSourcesCore> fe_flux_ptr,
    const std::string velocity_field, boost::shared_ptr<CommonData> common_data,
    const EntityType type) {
  MoFEMFunctionBegin;

  for (auto &sit : common_data->setOfBlocksData) {

    if (type == MBPRISM) {
      boost::shared_ptr<MatrixDouble> inv_jac_ptr(new MatrixDouble);
      fe_flux_ptr->getOpPtrVector().push_back(
          new OpMultiplyDeterminantOfJacobianAndWeightsForFatPrisms());
      fe_flux_ptr->getOpPtrVector().push_back(
          new OpCalculateInvJacForFatPrism(inv_jac_ptr));
      fe_flux_ptr->getOpPtrVector().push_back(
          new OpSetInvJacH1ForFatPrism(inv_jac_ptr));
    }
    fe_flux_ptr->getOpPtrVector().push_back(new OpCalculateVectorFieldValues<3>(
        velocity_field, common_data->velPtr));
    fe_flux_ptr->getOpPtrVector().push_back(
        new OpCalcVolumeFlux(velocity_field, common_data, sit.second));
  }

  MoFEMFunctionReturn(0);
};

MoFEMErrorCode NavierStokesElement::setCalcDragOperators(
    boost::shared_ptr<FaceElementForcesAndSourcesCore> dragFe,
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideDragFe,
    std::string side_fe_name, const std::string velocity_field,
    const std::string pressure_field,
    boost::shared_ptr<CommonData> common_data) {
  MoFEMFunctionBegin;

  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

  for (auto &sit : common_data->setOfFacesData) {
    sideDragFe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(velocity_field,
                                                 common_data->gradVelPtr));
    dragFe->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(inv_jac_ptr));
    dragFe->getOpPtrVector().push_back(new OpSetInvJacH1ForFace(inv_jac_ptr));
    dragFe->getOpPtrVector().push_back(new OpCalculateScalarFieldValues(
        pressure_field, common_data->pressPtr));

    dragFe->getOpPtrVector().push_back(
        new NavierStokesElement::OpCalcDragTraction(
            velocity_field, sideDragFe, side_fe_name, common_data, sit.second));
    dragFe->getOpPtrVector().push_back(new NavierStokesElement::OpCalcDragForce(
        velocity_field, common_data, sit.second));
  }
  MoFEMFunctionReturn(0);
};

MoFEMErrorCode NavierStokesElement::setPostProcDragOperators(
    boost::shared_ptr<PostProcFaceOnRefinedMesh> postProcDragPtr,
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideDragFe,
    std::string side_fe_name, const std::string velocity_field,
    const std::string pressure_field,
    boost::shared_ptr<CommonData> common_data) {
  MoFEMFunctionBegin;

  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

  for (auto &sit : common_data->setOfFacesData) {
    sideDragFe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(velocity_field,
                                                 common_data->gradVelPtr));

    postProcDragPtr->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(inv_jac_ptr));
    postProcDragPtr->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(inv_jac_ptr));
    postProcDragPtr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<3>(velocity_field,
                                            common_data->velPtr));
    postProcDragPtr->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues(pressure_field,
                                         common_data->pressPtr));

    postProcDragPtr->getOpPtrVector().push_back(
        new NavierStokesElement::OpCalcDragTraction(
            velocity_field, sideDragFe, side_fe_name, common_data, sit.second));
    postProcDragPtr->getOpPtrVector().push_back(
        new NavierStokesElement::OpPostProcDrag(
            velocity_field, postProcDragPtr->postProcMesh,
            postProcDragPtr->mapGaussPts, common_data, sit.second));
  }
  MoFEMFunctionReturn(0);
};

MoFEMErrorCode NavierStokesElement::OpAssembleLhs::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;

  row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);

  if (blockData.eNts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      blockData.eNts.end()) {
    MoFEMFunctionReturnHot(0);
  }

  row_nb_gauss_pts = row_data.getN().size1();
  if (!row_nb_gauss_pts) {
    MoFEMFunctionReturnHot(0);
  }

  isOnDiagonal = (row_type == col_type) && (row_side == col_side);

  // Set size can clear local tangent matrix
  locMat.resize(row_nb_dofs, col_nb_dofs, false);
  locMat.clear();

  CHKERR iNtegrate(row_data, col_data);

  CHKERR aSsemble(row_data, col_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NavierStokesElement::OpAssembleLhs::aSsemble(EntData &row_data,
                                                            EntData &col_data) {
  MoFEMFunctionBegin;

  const int *row_ind = &*row_data.getIndices().begin();
  const int *col_ind = &*col_data.getIndices().begin();

  Mat B = getFEMethod()->ksp_B != PETSC_NULL ? getFEMethod()->ksp_B
                                             : getFEMethod()->snes_B;

  CHKERR MatSetValues(B, row_nb_dofs, row_ind, col_nb_dofs, col_ind,
                      &*locMat.data().begin(), ADD_VALUES);

  if (!diagonalBlock || (sYmm && !isOnDiagonal)) {
    locMat = trans(locMat);
    CHKERR MatSetValues(B, col_nb_dofs, col_ind, row_nb_dofs, row_ind,
                        &*locMat.data().begin(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
NavierStokesElement::OpAssembleLhsOffDiag::iNtegrate(EntData &row_data,
                                                     EntData &col_data) {
  MoFEMFunctionBegin;

  const int row_nb_base_functions = row_data.getN().size2();
  auto row_diff_base_functions = row_data.getFTensor1DiffN<3>();

  FTensor::Tensor1<double, 3> t1;
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  // INTEGRATION
  for (int gg = 0; gg != row_nb_gauss_pts; gg++) {

    // Get volume and integration weight
    double w = getVolume() * getGaussPts()(3, gg);

    int row_bb = 0;
    for (; row_bb != row_nb_dofs / 3; row_bb++) {

      t1(i) = w * row_diff_base_functions(i);

      auto base_functions = col_data.getFTensor0N(gg, 0);
      for (int col_bb = 0; col_bb != col_nb_dofs; col_bb++) {

        FTensor::Tensor1<double *, 3> k(&locMat(3 * row_bb + 0, col_bb),
                                        &locMat(3 * row_bb + 1, col_bb),
                                        &locMat(3 * row_bb + 2, col_bb));

        k(i) -= t1(i) * base_functions;
        ++base_functions;
      }
      ++row_diff_base_functions;
    }
    for (; row_bb != row_nb_base_functions; row_bb++) {
      ++row_diff_base_functions;
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
NavierStokesElement::OpAssembleLhsDiagLin::iNtegrate(EntData &row_data,
                                                     EntData &col_data) {
  MoFEMFunctionBegin;

  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  const int row_nb_base_functions = row_data.getN().size2();

  auto row_diff_base_functions = row_data.getFTensor1DiffN<3>();

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  // integrate local matrix for entity block
  for (int gg = 0; gg != row_nb_gauss_pts; gg++) {

    // Get volume and integration weight
    double w = getVolume() * getGaussPts()(3, gg);
    double const alpha = w * blockData.viscousCoef;

    int row_bb = 0;
    for (; row_bb != row_nb_dofs / 3; row_bb++) {

      auto col_diff_base_functions = col_data.getFTensor1DiffN<3>(gg, 0);

      const int final_bb = isOnDiagonal ? row_bb + 1 : col_nb_dofs / 3;

      int col_bb = 0;

      for (; col_bb != final_bb; col_bb++) {

        auto t_assemble = get_tensor2(locMat, 3 * row_bb, 3 * col_bb);

        for (int i : {0, 1, 2}) {
          for (int j : {0, 1, 2}) {
            t_assemble(i, i) +=
                alpha * row_diff_base_functions(j) * col_diff_base_functions(j);
            t_assemble(i, j) +=
                alpha * row_diff_base_functions(j) * col_diff_base_functions(i);
          }
        }

        // Next base function for column
        ++col_diff_base_functions;
      }

      // Next base function for row
      ++row_diff_base_functions;
    }

    for (; row_bb != row_nb_base_functions; row_bb++) {
      ++row_diff_base_functions;
    }
  }

  if (isOnDiagonal) {
    for (int row_bb = 0; row_bb != row_nb_dofs / 3; row_bb++) {
      int col_bb = 0;
      for (; col_bb != row_bb + 1; col_bb++) {
        auto t_assemble = get_tensor2(locMat, 3 * row_bb, 3 * col_bb);
        auto t_off_side = get_tensor2(locMat, 3 * col_bb, 3 * row_bb);
        t_off_side(i, j) = t_assemble(j, i);
      }
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
NavierStokesElement::OpAssembleLhsDiagNonLin::iNtegrate(EntData &row_data,
                                                        EntData &col_data) {
  MoFEMFunctionBegin;

  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  const int row_nb_base_functions = row_data.getN().size2();

  auto row_base_functions = row_data.getFTensor0N();

  auto t_u = getFTensor1FromMat<3>(*commonData->velPtr);
  auto t_u_grad = getFTensor2FromMat<3, 3>(*commonData->gradVelPtr);

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  // integrate local matrix for entity block
  for (int gg = 0; gg != row_nb_gauss_pts; gg++) {

    // Get volume and integration weight
    double w = getVolume() * getGaussPts()(3, gg);
    double const beta = w * blockData.inertiaCoef;

    int row_bb = 0;
    for (; row_bb != row_nb_dofs / 3; row_bb++) {

      auto col_diff_base_functions = col_data.getFTensor1DiffN<3>(gg, 0);
      auto col_base_functions = col_data.getFTensor0N(gg, 0);

      const int final_bb = col_nb_dofs / 3;
      int col_bb = 0;

      for (; col_bb != final_bb; col_bb++) {

        auto t_assemble = get_tensor2(locMat, 3 * row_bb, 3 * col_bb);

        for (int i : {0, 1, 2}) {
          for (int j : {0, 1, 2}) {
            t_assemble(i, j) +=
                beta * col_base_functions * t_u_grad(i, j) * row_base_functions;
            t_assemble(i, i) +=
                beta * t_u(j) * col_diff_base_functions(j) * row_base_functions;
          }
        }

        // Next base function for column
        ++col_diff_base_functions;
        ++col_base_functions;
      }

      // Next base function for row
      ++row_base_functions;
    }

    for (; row_bb != row_nb_base_functions; row_bb++) {
      ++row_base_functions;
    }

    ++t_u;
    ++t_u_grad;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NavierStokesElement::OpAssembleRhs::doWork(int row_side,
                                                          EntityType row_type,
                                                          EntData &row_data) {
  MoFEMFunctionBegin;
  // get number of dofs on row
  nbRows = row_data.getIndices().size();
  if (!nbRows)
    MoFEMFunctionReturnHot(0);
  if (blockData.eNts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      blockData.eNts.end()) {
    MoFEMFunctionReturnHot(0);
  }
  // get number of integration points
  nbIntegrationPts = getGaussPts().size2();

  // integrate local vector
  CHKERR iNtegrate(row_data);
  // assemble local vector
  CHKERR aSsemble(row_data);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NavierStokesElement::OpAssembleRhs::aSsemble(EntData &data) {
  MoFEMFunctionBegin;
  // get global indices of local vector
  const int *indices = &*data.getIndices().data().begin();
  // get values from local vector
  const double *vals = &*locVec.data().begin();
  Vec f = getFEMethod()->ksp_f != PETSC_NULL ? getFEMethod()->ksp_f
                                             : getFEMethod()->snes_f;
  // assemble vector
  CHKERR VecSetValues(f, nbRows, indices, vals, ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
NavierStokesElement::OpAssembleRhsVelocityLin::iNtegrate(EntData &data) {
  MoFEMFunctionBegin;

  auto get_tensor1 = [](VectorDouble &v, const int r) {
    return FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3>(
        &v(r + 0), &v(r + 1), &v(r + 2));
  };

  // set size of local vector
  locVec.resize(nbRows, false);
  // clear local entity vector
  locVec.clear();

  int nb_base_functions = data.getN().size2();

  // get base function gradient on rows
  auto t_v_grad = data.getFTensor1DiffN<3>();

  auto t_u_grad = getFTensor2FromMat<3, 3>(*commonData->gradVelPtr);
  auto t_p = getFTensor0FromVec(*commonData->pressPtr);

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  // loop over all integration points
  for (int gg = 0; gg != nbIntegrationPts; gg++) {

    double w = getVolume() * getGaussPts()(3, gg);

    // evaluate constant term
    const double alpha = w * blockData.viscousCoef;

    auto t_a = get_tensor1(locVec, 0);
    int rr = 0;

    // loop over base functions
    for (; rr != nbRows / 3; rr++) {
      // add to local vector source term

      t_a(i) += alpha * t_u_grad(i, j) * t_v_grad(j);
      t_a(j) += alpha * t_u_grad(i, j) * t_v_grad(i);

      t_a(i) -= w * t_p * t_v_grad(i);

      ++t_a;      // move to next element of local vector
      ++t_v_grad; // move to next gradient of base function
    }

    for (; rr != nb_base_functions; rr++) {
      ++t_v_grad;
    }

    ++t_u_grad;
    ++t_p;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
NavierStokesElement::OpAssembleRhsVelocityNonLin::iNtegrate(EntData &data) {
  MoFEMFunctionBegin;

  auto get_tensor1 = [](VectorDouble &v, const int r) {
    return FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3>(
        &v(r + 0), &v(r + 1), &v(r + 2));
  };

  // set size of local vector
  locVec.resize(nbRows, false);
  // clear local entity vector
  locVec.clear();

  // get base functions on entity
  auto t_v = data.getFTensor0N();

  int nb_base_functions = data.getN().size2();

  auto t_u = getFTensor1FromMat<3>(*commonData->velPtr);
  auto t_u_grad = getFTensor2FromMat<3, 3>(*commonData->gradVelPtr);

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  // loop over all integration points
  for (int gg = 0; gg != nbIntegrationPts; gg++) {

    double w = getVolume() * getGaussPts()(3, gg);
    // evaluate constant term
    const double beta = w * blockData.inertiaCoef;

    auto t_a = get_tensor1(locVec, 0);
    int rr = 0;

    // loop over base functions
    for (; rr != nbRows / 3; rr++) {
      // add to local vector source term

      t_a(i) += beta * t_v * t_u_grad(i, j) * t_u(j);

      ++t_a; // move to next element of local vector
      ++t_v; // move to next base function
    }

    for (; rr != nb_base_functions; rr++) {
      ++t_v;
    }

    ++t_u;
    ++t_u_grad;
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
NavierStokesElement::OpAssembleRhsPressure::iNtegrate(EntData &data) {
  MoFEMFunctionBegin;

  // commonData->getBlockData(blockData);

  // set size to local vector
  locVec.resize(nbRows, false);
  // clear local vector
  locVec.clear();
  // get base function
  auto t_q = data.getFTensor0N();

  int nb_base_functions = data.getN().size2();
  // get solution at integration point

  auto t_u_grad = getFTensor2FromMat<3, 3>(*commonData->gradVelPtr);

  // FTensor::Index<'i', 3> i;

  // make loop over integration points
  for (int gg = 0; gg != nbIntegrationPts; gg++) {
    // evaluate function on boundary and scale it by area and integration
    // weight

    double w = getVolume() * getGaussPts()(3, gg);

    // get element of vector
    FTensor::Tensor0<FTensor::PackPtr<double *, 1>> t_a(&*locVec.begin());
    int rr = 0;
    for (; rr != nbRows; rr++) {

      for (int ii : {0, 1, 2}) {
        t_a -= w * t_q * t_u_grad(ii, ii);
      }

      ++t_a;
      ++t_q;
    }

    for (; rr != nb_base_functions; rr++) {
      ++t_q;
    }

    ++t_u_grad;
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NavierStokesElement::OpCalcVolumeFlux::doWork(int side,
                                                             EntityType type,
                                                             EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBVERTEX)
    PetscFunctionReturn(0);

  if (blockData.eNts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      blockData.eNts.end()) {
    MoFEMFunctionReturnHot(0);
  }

  const int nb_gauss_pts = getGaussPts().size2();

  auto t_u = getFTensor1FromMat<3>(*commonData->velPtr);
  auto t_w = getFTensor0IntegrationWeight(); ///< Integration weight

  FTensor::Index<'i', 3> i;

  FTensor::Tensor1<double, 3> t_flux;
  t_flux(i) = 0.0; // Zero entries

  // loop over all integration points
  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    double vol = getVolume();
    t_flux(i) += t_w * vol * t_u(i);

    ++t_w;
    ++t_u;
  }

  // Set array of indices
  constexpr std::array<int, 3> indices = {0, 1, 2};

  // Assemble volumetric flux
  CHKERR VecSetValues(commonData->volumeFluxVec, 3, indices.data(), &t_flux(0),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NavierStokesElement::OpCalcDragForce::doWork(int side,
                                                            EntityType type,
                                                            EntData &data) {
  MoFEMFunctionBegin;
  if (type != MBVERTEX)
    PetscFunctionReturn(0);

  if (blockData.eNts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      blockData.eNts.end()) {
    MoFEMFunctionReturnHot(0);
  }

  const int nb_gauss_pts = getGaussPts().size2();

  auto pressure_drag_at_gauss_pts =
      getFTensor1FromMat<3>(*commonData->pressureDragTract);
  auto shear_drag_at_gauss_pts =
      getFTensor1FromMat<3>(*commonData->shearDragTract);
  auto total_drag_at_gauss_pts =
      getFTensor1FromMat<3>(*commonData->totalDragTract);

  FTensor::Tensor1<double, 3> t_pressure_drag_force;
  FTensor::Tensor1<double, 3> t_shear_drag_force;
  FTensor::Tensor1<double, 3> t_total_drag_force;

  FTensor::Index<'i', 3> i;

  t_pressure_drag_force(i) = 0.0; // Zero entries
  t_shear_drag_force(i) = 0.0;    // Zero entries
  t_total_drag_force(i) = 0.0;    // Zero entries

  auto t_w = getFTensor0IntegrationWeight();
  auto t_normal = getFTensor1NormalsAtGaussPts();

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    double nrm2 = sqrt(t_normal(i) * t_normal(i));
    double w = t_w * nrm2 * 0.5;

    t_pressure_drag_force(i) += w * pressure_drag_at_gauss_pts(i);
    t_shear_drag_force(i) += w * shear_drag_at_gauss_pts(i);
    t_total_drag_force(i) += w * total_drag_at_gauss_pts(i);

    ++t_w;
    ++t_normal;
    ++pressure_drag_at_gauss_pts;
    ++shear_drag_at_gauss_pts;
    ++total_drag_at_gauss_pts;
  }

  // Set array of indices
  constexpr std::array<int, 3> indices = {0, 1, 2};

  CHKERR VecSetValues(commonData->pressureDragForceVec, 3, indices.data(),
                      &t_pressure_drag_force(0), ADD_VALUES);
  CHKERR VecSetValues(commonData->shearDragForceVec, 3, indices.data(),
                      &t_shear_drag_force(0), ADD_VALUES);
  CHKERR VecSetValues(commonData->totalDragForceVec, 3, indices.data(),
                      &t_total_drag_force(0), ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NavierStokesElement::OpCalcDragTraction::doWork(int side,
                                                               EntityType type,
                                                               EntData &data) {
  MoFEMFunctionBegin;

  if (type != MBVERTEX)
    PetscFunctionReturn(0);

  if (blockData.eNts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      blockData.eNts.end()) {
    MoFEMFunctionReturnHot(0);
  }

  CHKERR loopSideVolumes(sideFeName, *sideFe);

  const int nb_gauss_pts = getGaussPts().size2();

  auto t_p = getFTensor0FromVec(*commonData->pressPtr);
  auto t_u_grad = getFTensor2FromMat<3, 3>(*commonData->gradVelPtr);

  auto t_normal = getFTensor1NormalsAtGaussPts();

  FTensor::Tensor1<double, 3> t_unit_normal;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  commonData->pressureDragTract->resize(3, nb_gauss_pts, false);
  commonData->pressureDragTract->clear();

  commonData->shearDragTract->resize(3, nb_gauss_pts, false);
  commonData->shearDragTract->clear();

  commonData->totalDragTract->resize(3, nb_gauss_pts, false);
  commonData->totalDragTract->clear();

  auto pressure_drag_at_gauss_pts =
      getFTensor1FromMat<3>(*commonData->pressureDragTract);
  auto shear_drag_at_gauss_pts =
      getFTensor1FromMat<3>(*commonData->shearDragTract);
  auto total_drag_at_gauss_pts =
      getFTensor1FromMat<3>(*commonData->totalDragTract);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    double nrm2 = sqrt(t_normal(i) * t_normal(i));
    t_unit_normal(i) = t_normal(i) / nrm2;

    double mu = blockData.fluidViscosity;

    pressure_drag_at_gauss_pts(i) = t_p * t_unit_normal(i);
    shear_drag_at_gauss_pts(i) =
        -mu * (t_u_grad(i, j) + t_u_grad(j, i)) * t_unit_normal(j);
    total_drag_at_gauss_pts(i) =
        pressure_drag_at_gauss_pts(i) + shear_drag_at_gauss_pts(i);

    ++pressure_drag_at_gauss_pts;
    ++shear_drag_at_gauss_pts;
    ++total_drag_at_gauss_pts;
    ++t_p;
    ++t_u_grad;
    ++t_normal;
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NavierStokesElement::OpPostProcDrag::doWork(int side,
                                                           EntityType type,
                                                           EntData &data) {
  MoFEMFunctionBegin;
  if (type != MBVERTEX)
    PetscFunctionReturn(0);

  double def_VAL[9];
  bzero(def_VAL, 9 * sizeof(double));

  if (blockData.eNts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      blockData.eNts.end()) {
    MoFEMFunctionReturnHot(0);
  }

  Tag th_velocity;
  Tag th_pressure;
  Tag th_velocity_grad;
  Tag th_shear_drag;
  Tag th_pressure_drag;
  Tag th_total_drag;

  CHKERR postProcMesh.tag_get_handle("VELOCITY", 3, MB_TYPE_DOUBLE, th_velocity,
                                     MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
  CHKERR postProcMesh.tag_get_handle("PRESSURE", 1, MB_TYPE_DOUBLE, th_pressure,
                                     MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
  CHKERR postProcMesh.tag_get_handle("VELOCITY_GRAD", 9, MB_TYPE_DOUBLE,
                                     th_velocity_grad,
                                     MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);

  CHKERR postProcMesh.tag_get_handle("PRESSURE_DRAG", 3, MB_TYPE_DOUBLE,
                                     th_pressure_drag,
                                     MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
  CHKERR postProcMesh.tag_get_handle("SHEAR_DRAG", 3, MB_TYPE_DOUBLE,
                                     th_shear_drag,
                                     MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
  CHKERR postProcMesh.tag_get_handle("TOTAL_DRAG", 3, MB_TYPE_DOUBLE,
                                     th_total_drag,
                                     MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);

  auto t_p = getFTensor0FromVec(*commonData->pressPtr);

  const int nb_gauss_pts = commonData->pressureDragTract->size2();

  MatrixDouble velGradMat;
  velGradMat.resize(3, 3);
  VectorDouble velVec;
  velVec.resize(3);
  VectorDouble pressDragVec;
  pressDragVec.resize(3);
  VectorDouble viscDragVec;
  viscDragVec.resize(3);
  VectorDouble totDragVec;
  totDragVec.resize(3);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    for (int i : {0, 1, 2}) {
      for (int j : {0, 1, 2}) {
        velGradMat(i, j) = (*commonData->gradVelPtr)(i * 3 + j, gg);
      }
      velVec(i) = (*commonData->velPtr)(i, gg);
      pressDragVec(i) = (*commonData->pressureDragTract)(i, gg);
      viscDragVec(i) = (*commonData->shearDragTract)(i, gg);
      totDragVec(i) = (*commonData->totalDragTract)(i, gg);
    }
    CHKERR postProcMesh.tag_set_data(th_velocity, &mapGaussPts[gg], 1,
                                     &velVec(0));
    CHKERR postProcMesh.tag_set_data(th_pressure, &mapGaussPts[gg], 1, &t_p);
    CHKERR postProcMesh.tag_set_data(th_velocity_grad, &mapGaussPts[gg], 1,
                                     &velGradMat(0, 0));

    CHKERR postProcMesh.tag_set_data(th_pressure_drag, &mapGaussPts[gg], 1,
                                     &pressDragVec(0));

    CHKERR postProcMesh.tag_set_data(th_shear_drag, &mapGaussPts[gg], 1,
                                     &viscDragVec(0));

    CHKERR postProcMesh.tag_set_data(th_total_drag, &mapGaussPts[gg], 1,
                                     &totDragVec(0));

    ++t_p;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NavierStokesElement::OpPostProcVorticity::doWork(int side,
                                                                EntityType type,
                                                                EntData &data) {
  MoFEMFunctionBegin;
  if (type != MBVERTEX)
    PetscFunctionReturn(0);
  double def_VAL[9];
  bzero(def_VAL, 9 * sizeof(double));

  if (blockData.eNts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      blockData.eNts.end()) {
    MoFEMFunctionReturnHot(0);
  }
  // commonData->getBlockData(blockData);

  Tag th_vorticity;
  Tag th_q;
  Tag th_l2;
  CHKERR postProcMesh.tag_get_handle("VORTICITY", 3, MB_TYPE_DOUBLE,
                                     th_vorticity, MB_TAG_CREAT | MB_TAG_SPARSE,
                                     def_VAL);
  CHKERR postProcMesh.tag_get_handle("Q", 1, MB_TYPE_DOUBLE, th_q,
                                     MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
  CHKERR postProcMesh.tag_get_handle("L2", 1, MB_TYPE_DOUBLE, th_l2,
                                     MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);

  auto t_u_grad = getFTensor2FromMat<3, 3>(*commonData->gradVelPtr);
  // auto p = getFTensor0FromVec(*commonData->pressPtr);

  const int nb_gauss_pts = commonData->gradVelPtr->size2();
  // const int nb_gauss_pts2 = commonData->pressPtr->size();

  // const double lambda = commonData->lAmbda;
  // const double mu = commonData->mU;

  // FTensor::Index<'i', 3> i;
  // FTensor::Index<'j', 3> j;
  // FTensor::Index<'j', 3> k;
  FTensor::Tensor1<double, 3> vorticity;
  // FTensor::Tensor2<double,3,3> t_s;
  double q;
  double l2;
  // FTensor::Tensor2<double, 3, 3> stress;
  MatrixDouble S;
  MatrixDouble Omega;
  MatrixDouble M;

  S.resize(3, 3);
  Omega.resize(3, 3);
  M.resize(3, 3);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    vorticity(0) = t_u_grad(2, 1) - t_u_grad(1, 2);
    vorticity(1) = t_u_grad(0, 2) - t_u_grad(2, 0);
    vorticity(2) = t_u_grad(1, 0) - t_u_grad(0, 1);

    q = 0;
    for (int i = 0; i != 3; i++) {
      for (int j = 0; j != 3; j++) {
        q += -0.5 * t_u_grad(i, j) * t_u_grad(j, i);
      }
    }
    for (int i = 0; i != 3; i++) {
      for (int j = 0; j != 3; j++) {
        S(i, j) = 0.5 * (t_u_grad(i, j) + t_u_grad(j, i));
        Omega(i, j) = 0.5 * (t_u_grad(i, j) - t_u_grad(j, i));
        M(i, j) = 0.0;
      }
    }

    for (int i = 0; i != 3; i++) {
      for (int j = 0; j != 3; j++) {
        for (int k = 0; k != 3; k++) {
          M(i, j) += S(i, k) * S(k, j) + Omega(i, k) * Omega(k, j);
        }
      }
    }

    MatrixDouble eigen_vectors = M;
    VectorDouble eigen_values(3);

    // LAPACK - eigenvalues and vectors. Applied twice for initial creates
    // memory space
    int n = 3, lda = 3, info, lwork = -1;
    double wkopt;
    info = lapack_dsyev('V', 'U', n, &(eigen_vectors.data()[0]), lda,
                        &(eigen_values.data()[0]), &wkopt, lwork);
    if (info != 0)
      SETERRQ1(PETSC_COMM_SELF, 1,
               "is something wrong with lapack_dsyev info = %d", info);
    lwork = (int)wkopt;
    double work[lwork];
    info = lapack_dsyev('V', 'U', n, &(eigen_vectors.data()[0]), lda,
                        &(eigen_values.data()[0]), work, lwork);
    if (info != 0)
      SETERRQ1(PETSC_COMM_SELF, 1,
               "is something wrong with lapack_dsyev info = %d", info);

    map<double, int> eigen_sort;
    eigen_sort[eigen_values[0]] = 0;
    eigen_sort[eigen_values[1]] = 1;
    eigen_sort[eigen_values[2]] = 2;

    // prin_stress_vect.clear();
    VectorDouble prin_vals_vect(3);
    prin_vals_vect.clear();

    int ii = 0;
    for (map<double, int>::reverse_iterator mit = eigen_sort.rbegin();
         mit != eigen_sort.rend(); mit++) {
      prin_vals_vect[ii] = eigen_values[mit->second];
      // for (int dd = 0; dd != 3; dd++) {
      //   prin_stress_vect(ii, dd) = eigen_vectors.data()[3 * mit->second +
      //   dd];
      // }
      ii++;
    }

    l2 = prin_vals_vect[1];
    // cout << prin_vals_vect << endl;
    // cout << "-0.5 sum: " << -0.5 * (prin_vals_vect[0] + prin_vals_vect[1]
    // + prin_vals_vect[2]) << endl; cout << "q: " << q << endl;

    // t_s(i,j) = 0.5*(t)

    // vorticity(0) = t_u_grad(1, 2) - t_u_grad(2, 1);
    // vorticity(1) = t_u_grad(2, 0) - t_u_grad(0, 2);
    // vorticity(2) = t_u_grad(0, 1) - t_u_grad(1, 0);

    CHKERR postProcMesh.tag_set_data(th_vorticity, &mapGaussPts[gg], 1,
                                     &vorticity(0));
    CHKERR postProcMesh.tag_set_data(th_q, &mapGaussPts[gg], 1, &q);
    CHKERR postProcMesh.tag_set_data(th_l2, &mapGaussPts[gg], 1, &l2);
    ++t_u_grad;
  }

  MoFEMFunctionReturn(0);
}

VectorDouble3 stokes_flow_velocity(double x, double y, double z) {
  double r = sqrt(x * x + y * y + z * z);
  double theta = acos(x / r);
  double phi = atan2(y, z);
  double ur = cos(theta) * (1.0 + 0.5 / (r * r * r) - 1.5 / r);
  double ut = -sin(theta) * (1.0 - 0.25 / (r * r * r) - 0.75 / r);
  VectorDouble3 res(3);
  res[0] = ur * cos(theta) - ut * sin(theta);
  res[1] = ur * sin(theta) * sin(phi) + ut * cos(theta) * sin(phi);
  res[2] = ur * sin(theta) * cos(phi) + ut * cos(theta) * cos(phi);
  return res;
}
