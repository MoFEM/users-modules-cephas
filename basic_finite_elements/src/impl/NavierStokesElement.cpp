/**
 * \file NavierStokesOperators.cpp
 * \example NavierStokesOperators.cpp
 *
 * \brief Implementation of operators for fluid flow
 *
 * Implementation of operators for computations of the fluid flow, governed by
 * for Stokes and Navier-Stokes equations, and computation of the viscous drag
 * force
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


    MoFEMErrorCode
    NavierStokesElement::OpAssembleG::doWork(int row_side, int col_side,
                                             EntityType row_type,
                                             EntityType col_type,
                                             EntData &row_data,
                                             EntData &col_data)
{

  MoFEMFunctionBegin;

  const int row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  const int col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);

  if (blockData.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      blockData.tEts.end()) {
    MoFEMFunctionReturnHot(0);
  }
  // commonData->getBlockData(blockData);

  // Set size can clear local tangent matrix
  locG.resize(row_nb_dofs, col_nb_dofs, false);
  locG.clear();
  const int row_nb_gauss_pts = row_data.getN().size1();
  if (!row_nb_gauss_pts)
    MoFEMFunctionReturnHot(0);
  const int row_nb_base_functions = row_data.getN().size2();
  auto row_diff_base_functions = row_data.getFTensor1DiffN<3>();

  // const double lambda = commonData->lAmbda;
  // const double mu = commonData->mU;

  FTensor::Tensor1<double, 3> t1;
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  // INTEGRATION
  for (int gg = 0; gg != row_nb_gauss_pts; gg++) {

    // Get volume and integration weight
    double w = getVolume() * getGaussPts()(3, gg);
    if (getHoGaussPtsDetJac().size() > 0) {
      w *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
    }

    int row_bb = 0;
    for (; row_bb != row_nb_dofs / 3; row_bb++) {

      t1(i) = w * row_diff_base_functions(i);

      auto base_functions = col_data.getFTensor0N(gg, 0);
      for (int col_bb = 0; col_bb != col_nb_dofs; col_bb++) {

        FTensor::Tensor1<double *, 3> k(&locG(3 * row_bb + 0, col_bb),
                                        &locG(3 * row_bb + 1, col_bb),
                                        &locG(3 * row_bb + 2, col_bb));

        k(i) -= t1(i) * base_functions;
        ++base_functions;
      }
      ++row_diff_base_functions;
    }
    for (; row_bb != row_nb_base_functions; row_bb++) {
      ++row_diff_base_functions;
    }
  }

  CHKERR MatSetValues(getFEMethod()->snes_B, row_nb_dofs,
                      &*row_data.getIndices().begin(), col_nb_dofs,
                      &*col_data.getIndices().begin(), &*locG.data().begin(),
                      ADD_VALUES);

  // ASSEMBLE THE TRANSPOSE
  locG = trans(locG);
  CHKERR MatSetValues(getFEMethod()->snes_B, col_nb_dofs,
                      &*col_data.getIndices().begin(), row_nb_dofs,
                      &*row_data.getIndices().begin(), &*locG.data().begin(),
                      ADD_VALUES);
  MoFEMFunctionReturn(0);
}

// PetscErrorCode
// NavierStokesElement::OpAssembleK::doWork(int row_side, int col_side,
//                                        EntityType row_type, EntityType col_type,
//                                        EntData &row_data, EntData &col_data) {
//   MoFEMFunctionBegin;

//   auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
//     return FTensor::Tensor2<double *, 3, 3>(
//         &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
//         &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
//         &m(r + 2, c + 2));
//   };

//   const int row_nb_dofs = row_data.getIndices().size();
//   if (!row_nb_dofs)
//     MoFEMFunctionReturnHot(0);
//   const int col_nb_dofs = col_data.getIndices().size();
//   if (!col_nb_dofs)
//     MoFEMFunctionReturnHot(0);
//   if (blockData.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
//       blockData.tEts.end()) {
//     PetscPrintf(PETSC_COMM_WORLD, "PROBLEM");
//     MoFEMFunctionReturnHot(0);
//   }
//   // commonData->getBlockData(blockData);

//   // const bool diagonal_block =
//   //     (row_type == col_type) && (row_side == col_side);
//   // get number of integration points
//   // Set size can clear local tangent matrix

//   locK.resize(row_nb_dofs, col_nb_dofs, false);
//   locK.clear();

//   const int row_nb_gauss_pts = row_data.getN().size1();
//   const int row_nb_base_functions = row_data.getN().size2();

//   auto row_diff_base_functions = row_data.getFTensor1DiffN<3>();
//   auto row_base_functions = row_data.getFTensor0N();

//   auto t_u = getFTensor1FromMat<3>(*commonData->dispPtr);
//   auto t_u_grad = getFTensor2FromMat<3, 3>(*commonData->gradDispPtr);

//   // const double mu = commonData->mU;
//   // const double lambda = commonData->lAmbda;

//   // integrate local matrix for entity block
//   for (int gg = 0; gg != row_nb_gauss_pts; gg++) {

//     // const MatrixAdaptor &diffN = row_data.getDiffN(gg,nb_dofs/3);

//     // Get volume and integration weight
//     double w = getVolume() * getGaussPts()(3, gg);
//     if (getHoGaussPtsDetJac().size() > 0) {
//       w *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
//     }

//     double const alpha = w * blockData.fluidViscosity;
//     double const beta = w * blockData.fluidDensity;

//     int row_bb = 0;
//     for (; row_bb != row_nb_dofs / 3; row_bb++) {

//       auto col_diff_base_functions = col_data.getFTensor1DiffN<3>(gg, 0);
//       auto col_base_functions = col_data.getFTensor0N(gg, 0);

//       // const int final_bb = diagonal_block ? row_bb + 1 : col_nb_dofs / 3;
//       const int final_bb = col_nb_dofs / 3;
//       int col_bb = 0;

//       // MatrixDouble mat = col_data.getDiffN();

//       for (; col_bb != final_bb; col_bb++) {

//         auto t_assemble = get_tensor2(locK, 3 * row_bb, 3 * col_bb);

//         for (int i : {0, 1, 2}) {
//           for (int j : {0, 1, 2}) {
//             t_assemble(i, i) +=
//                 alpha * row_diff_base_functions(j) * col_diff_base_functions(j);
//             t_assemble(i, j) +=
//                 alpha * row_diff_base_functions(j) * col_diff_base_functions(i);

//             t_assemble(i, j) +=
//                 beta * col_base_functions * t_u_grad(i, j) * row_base_functions;
//             t_assemble(i, i) +=
//                 beta * t_u(j) * col_diff_base_functions(j) * row_base_functions;
//             // t_assemble(j, i) += beta * col_base_functions * t_u_grad(i, j)
//             // * row_base_functions; t_assemble(i, i) += beta * t_u(j) *
//             // col_diff_base_functions(j) * row_base_functions;
//           }
//         }

//         // Next base function for column
//         ++col_diff_base_functions;
//         ++col_base_functions;
//       }

//       // Next base function for row
//       ++row_diff_base_functions;
//       ++row_base_functions;
//     }

//     for (; row_bb != row_nb_base_functions; row_bb++) {
//       ++row_diff_base_functions;
//       ++row_base_functions;
//     }

//     ++t_u;
//     ++t_u_grad;
//   }

//   // if (diagonal_block) {
//   //   FTensor::Index<'i', 3> i;
//   //   //FTensor::Index<'j', 3> j;
//   //   FTensor::Index<'k', 3> k;
//   //   //FTensor::Index<'l', 3> l;
//   //   for (int row_bb = 0; row_bb != row_nb_dofs / 3; row_bb++) {
//   //     int col_bb = 0;
//   //     for (; col_bb != row_bb + 1; col_bb++) {
//   //       auto t_assemble = get_tensor2(locK, 3 * row_bb, 3 * col_bb);
//   //       auto t_off_side = get_tensor2(locK, 3 * col_bb, 3 * row_bb);
//   //       t_off_side(i, k) = t_assemble(k, i);
//   //     }
//   //   }
//   // }

//   const int *row_ind = &*row_data.getIndices().begin();
//   const int *col_ind = &*col_data.getIndices().begin();
//   Mat B = getFEMethod()->ksp_B != PETSC_NULL ? getFEMethod()->ksp_B
//                                              : getFEMethod()->snes_B;
//   CHKERR MatSetValues(B, row_nb_dofs, row_ind, col_nb_dofs, col_ind,
//                       &locK(0, 0), ADD_VALUES);

//   // if (row_type != col_type || row_side != col_side) {
//   //   translocK.resize(col_nb_dofs, row_nb_dofs, false);
//   //   noalias(translocK) = trans(locK);
//   //   CHKERR MatSetValues(B, col_nb_dofs, col_ind, row_nb_dofs, row_ind,
//   //                       &translocK(0, 0), ADD_VALUES);
//   // }

//   MoFEMFunctionReturn(0);
// }

// MoFEMErrorCode NavierStokesElement::OpAssembleRhsVelocityLin::doWork(int row_side, EntityType row_type, EntData &row_data) {
//   MoFEMFunctionBegin;
//   // get number of dofs on row
//   nbRows = row_data.getIndices().size();
//   if (!nbRows)
//     MoFEMFunctionReturnHot(0);
//   if (blockData.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
//       blockData.tEts.end()) {
//     PetscPrintf(PETSC_COMM_WORLD, "PROBLEM");
//     MoFEMFunctionReturnHot(0);
//   }
//   // get number of integration points
//   nbIntegrationPts = getGaussPts().size2();
//   // integrate local vector
//   CHKERR iNtegrate(row_data);
//   // assemble local vector
//   CHKERR aSsemble(row_data);
//   MoFEMFunctionReturn(0);
// }

/**
 * \brief Integrate local entity vector
 * @param  data entity data on element row
 * @return      error code
 */
// MoFEMErrorCode
// NavierStokesElement::OpAssembleRhsVelocityLin::iNtegrate(EntData &data) {
//   MoFEMFunctionBegin;

//   auto get_tensor1 = [](VectorDouble &v, const int r) {
//     return FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3>(
//         &v(r + 0), &v(r + 1), &v(r + 2));
//   };

//   // commonData->getBlockData(blockData);

//   // set size of local vector
//   locVec.resize(nbRows, false);
//   // clear local entity vector
//   locVec.clear();
//   // get finite element volume
//   double vol = getVolume();

//   // get base functions on entity
//   auto t_v = data.getFTensor0N();

//   int nb_base_functions = data.getN().size2();

//   // get base function gradient on rows
//   auto t_v_grad = data.getFTensor1DiffN<3>();

//   auto t_u = getFTensor1FromMat<3>(*commonData->dispPtr);
//   auto t_u_grad = getFTensor2FromMat<3, 3>(*commonData->gradDispPtr);
//   auto t_p = getFTensor0FromVec(*commonData->pPtr);

//   FTensor::Index<'i', 3> i;
//   FTensor::Index<'j', 3> j;

//   // loop over all integration points
//   for (int gg = 0; gg != nbIntegrationPts; gg++) {

//     double w = getVolume() * getGaussPts()(3, gg);
//     // cout << "volume: " << getVolume() << endl;
//     // cout << "GP: " << getGaussPts()(3, gg) << endl;
//     if (getHoGaussPtsDetJac().size() > 0) {
//       w *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
//     }

//     // evaluate constant term
//     const double alpha = w * blockData.fluidViscosity;
//     const double beta = w * blockData.fluidDensity;

//     auto t_a = get_tensor1(locVec, 0);
//     int rr = 0;

//     // loop over base functions
//     for (; rr != nbRows / 3; rr++) {
//       // add to local vector source term

//       for (int i : {0, 1, 2}) {
//         for (int j : {0, 1, 2}) {
//           t_a(i) += alpha * t_u_grad(i, j) * t_v_grad(j);
//           t_a(j) += alpha * t_u_grad(i, j) * t_v_grad(i);
//           // cout << t_u_grad(i, j) << endl;
//         }
//         t_a(i) -= w * t_p * t_v_grad(i);
//         // cout << t_u(i) << endl;
//       }

//       t_a(i) += beta * t_v * t_u_grad(i, j) * t_u(j);

//       // t_a(i) += alpha * t_u_grad(i, j) * t_v_grad(j);
//       // t_a(j) += alpha * t_u_grad(i, j) * t_v_grad(i);

//       // t_a(i) -= w * t_p * t_v_grad(i);

//       ++t_a;      // move to next element of local vector
//       ++t_v;      // move to next base function
//       ++t_v_grad; // move to next gradient of base function
//     }

//     for (; rr != nb_base_functions; rr++) {
//       ++t_v;
//       ++t_v_grad;
//     }

//     ++t_u;
//     ++t_u_grad;
//     ++t_p;
//   }
//   MoFEMFunctionReturn(0);
// }

/**
 * \brief assemble local entity vector to the global right hand side
 * @param  data entity data, i.e. global indices of local vector
 * @return      error code
 */

// MoFEMErrorCode
// NavierStokesElement::OpAssembleRhsVelocityLin::aSsemble(EntData &data) {
//   MoFEMFunctionBegin;
//   // get global indices of local vector
//   const int *indices = &*data.getIndices().data().begin();
//   // get values from local vector
//   const double *vals = &*locVec.data().begin();
//   Vec f = getFEMethod()->ksp_f != PETSC_NULL ? getFEMethod()->ksp_f
//                                              : getFEMethod()->snes_f;
//   // assemble vector
//   CHKERR VecSetValues(f, nbRows, indices, vals, ADD_VALUES);
//   MoFEMFunctionReturn(0);
// }

// MoFEMErrorCode NavierStokesElement::OpAssembleRes_g::doWork(int row_side,
//                                                          EntityType row_type,
//                                                           EntData &row_data) {
//   MoFEMFunctionBegin;
//   // get number of dofs on row
//   nbRows = row_data.getIndices().size();
//   if (!nbRows)
//     MoFEMFunctionReturnHot(0);
//   if (blockData.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
//       blockData.tEts.end()) {
//     PetscPrintf(PETSC_COMM_WORLD, "PROBLEM");
//     MoFEMFunctionReturnHot(0);
//   }
//   // get number of integration points
//   nbIntegrationPts = getGaussPts().size2();
//   // integrate local vector
//   CHKERR iNtegrate(row_data);
//   // assemble local vector
//   CHKERR aSsemble(row_data);
//   MoFEMFunctionReturn(0);
// }

/**
 * \brief Integrate local constrains vector
 */
// MoFEMErrorCode NavierStokesElement::OpAssembleRes_g::iNtegrate(EntData &data) {
//   MoFEMFunctionBegin;

//   // commonData->getBlockData(blockData);

//   // set size to local vector
//   locVec.resize(nbRows, false);
//   // clear local vector
//   locVec.clear();
//   // get base function
//   auto t_q = data.getFTensor0N();

//   int nb_base_functions = data.getN().size2();
//   // get solution at integration point

//   auto t_u_grad = getFTensor2FromMat<3, 3>(*commonData->gradDispPtr);

//   // FTensor::Index<'i', 3> i;

//   // make loop over integration points
//   for (int gg = 0; gg != nbIntegrationPts; gg++) {
//     // evaluate function on boundary and scale it by area and integration
//     // weight

//     double w = getVolume() * getGaussPts()(3, gg);
//     if (getHoGaussPtsDetJac().size() > 0) {
//       w *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
//     }

//     // get element of vector
//     FTensor::Tensor0<FTensor::PackPtr<double *, 1>> t_a(&*locVec.begin());
//     int rr = 0;
//     for (; rr != nbRows; rr++) {

//       for (int ii : {0, 1, 2}) {
//         t_a -= w * t_q * t_u_grad(ii, ii);
//       }
//       // t_a -= w * t_l * t_u_grad(i,i);
//       ++t_a;
//       ++t_q;
//     }

//     for (; rr != nb_base_functions; rr++) {
//       ++t_q;
//     }

//     ++t_u_grad;
//   }
//   MoFEMFunctionReturn(0);
// }

/**
 * \brief assemble constrains vectors
 */
// MoFEMErrorCode NavierStokesElement::OpAssembleRes_g::aSsemble(EntData &data) {
//   MoFEMFunctionBegin;
//   const int *indices = &*data.getIndices().data().begin();
//   const double *vals = &*locVec.data().begin();
//   Vec f = getFEMethod()->ksp_f != PETSC_NULL ? getFEMethod()->ksp_f
//                                              : getFEMethod()->snes_f;
//   CHKERR VecSetValues(f, nbRows, indices, &*vals, ADD_VALUES);
//   MoFEMFunctionReturn(0);
// }

// PetscErrorCode NavierStokesElement::OpCalcPressureDrag::doWork(int side,
//                                                              EntityType type,
// //                                                              EntData &data) {
//   MoFEMFunctionBegin;
//   if (type != MBVERTEX)
//     PetscFunctionReturn(0);
//   double def_VAL[9];
//   bzero(def_VAL, 9 * sizeof(double));

//   // commonData->getBlockData(blockData);

//   auto t_p = getFTensor0FromVec(*commonData->pPtr);
//   const int nb_gauss_pts = commonData->pPtr->size();
//   auto t_normal = getFTensor1NormalsAtGaussPts();
//   // auto t_normal = getFTensor1Normal();

//   FTensor::Index<'i', 3> i;

//   for (int gg = 0; gg != nb_gauss_pts; gg++) {

//     double nrm2 = sqrt(t_normal(i) * t_normal(i));
//     t_normal(i) = t_normal(i) / nrm2;

//     double w = getArea() * getGaussPts()(2, gg);

//     // if (getHoGaussPtsDetJac().size() > 0) {
//     //   w *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
//     // }

//     for (int dd = 0; dd != 3; dd++) {
//       commonData->pressureDrag[dd] += w * t_p * t_normal(dd);
//     }

//     ++t_p;
//     ++t_normal;
//   }

//   MoFEMFunctionReturn(0);
// }

// PetscErrorCode NavierStokesElement::OpCalcViscousDrag::doWork(int side,
//                                                             EntityType type,
//                                                             EntData &data) {
//   MoFEMFunctionBegin;

//   if (type != MBVERTEX)
//     PetscFunctionReturn(0);

//   double def_VAL[9];
//   bzero(def_VAL, 9 * sizeof(double));

//   CHKERR loopSideVolumes("NAVIER_STOKES", *sideFe);

//   // commonData->getBlockData(blockData);

//   auto t_u_grad = getFTensor2FromMat<3, 3>(*commonData->gradDispPtr);
//   const int nb_gauss_pts = commonData->gradDispPtr->size2();
//   auto t_normal = getFTensor1NormalsAtGaussPts();
//   // auto t_normal = getFTensor1Normal();

//   FTensor::Index<'i', 3> i;

//   for (int gg = 0; gg != nb_gauss_pts; gg++) {

//     double nrm2 = sqrt(t_normal(i) * t_normal(i));
//     t_normal(i) = t_normal(i) / nrm2;

//     double w = getArea() * getGaussPts()(2, gg) * blockData.fluidViscosity;
//     // if (getHoGaussPtsDetJac().size() > 0) {
//     //   w *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
//     // }

//     for (int ii = 0; ii != 3; ii++) {
//       for (int jj = 0; jj != 3; jj++) {
//         commonData->viscousDrag[ii] +=
//             -w * (t_u_grad(ii, jj) + t_u_grad(jj, ii)) * t_normal(jj);
//       }
//     }

//     ++t_u_grad;
//     ++t_normal;
//   }
//   MoFEMFunctionReturn(0);
// }

// PetscErrorCode NavierStokesElement::OpPostProcVorticity::doWork(int side,
//                                                               EntityType type,
//                                                               EntData &data) {
//   MoFEMFunctionBegin;
//   if (type != MBVERTEX)
//     PetscFunctionReturn(0);
//   double def_VAL[9];
//   bzero(def_VAL, 9 * sizeof(double));

//   if (blockData.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
//       blockData.tEts.end()) {
//     MoFEMFunctionReturnHot(0);
//   }
//   // commonData->getBlockData(blockData);

//   Tag th_vorticity;
//   Tag th_q;
//   Tag th_l2;
//   CHKERR postProcMesh.tag_get_handle("VORTICITY", 3, MB_TYPE_DOUBLE,
//                                      th_vorticity, MB_TAG_CREAT | MB_TAG_SPARSE,
//                                      def_VAL);
//   CHKERR postProcMesh.tag_get_handle("Q", 1, MB_TYPE_DOUBLE, th_q,
//                                      MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
//   CHKERR postProcMesh.tag_get_handle("L2", 1, MB_TYPE_DOUBLE, th_l2,
//                                      MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);

//   auto t_u_grad = getFTensor2FromMat<3, 3>(*commonData->gradDispPtr);
//   // auto p = getFTensor0FromVec(*commonData->pPtr);

//   const int nb_gauss_pts = commonData->gradDispPtr->size2();
//   // const int nb_gauss_pts2 = commonData->pPtr->size();

//   // const double lambda = commonData->lAmbda;
//   // const double mu = commonData->mU;

//   // FTensor::Index<'i', 3> i;
//   // FTensor::Index<'j', 3> j;
//   // FTensor::Index<'j', 3> k;
//   FTensor::Tensor1<double, 3> vorticity;
//   // FTensor::Tensor2<double,3,3> t_s;
//   double q;
//   double l2;
//   // FTensor::Tensor2<double, 3, 3> stress;
//   MatrixDouble S;
//   MatrixDouble Omega;
//   MatrixDouble M;

//   S.resize(3, 3);
//   Omega.resize(3, 3);
//   M.resize(3, 3);

//   for (int gg = 0; gg != nb_gauss_pts; gg++) {

//     vorticity(0) = t_u_grad(2, 1) - t_u_grad(1, 2);
//     vorticity(1) = t_u_grad(0, 2) - t_u_grad(2, 0);
//     vorticity(2) = t_u_grad(1, 0) - t_u_grad(0, 1);

//     q = 0;
//     for (int i = 0; i != 3; i++) {
//       for (int j = 0; j != 3; j++) {
//         q += -0.5 * t_u_grad(i, j) * t_u_grad(j, i);
//       }
//     }
//     for (int i = 0; i != 3; i++) {
//       for (int j = 0; j != 3; j++) {
//         S(i, j) = 0.5 * (t_u_grad(i, j) + t_u_grad(j, i));
//         Omega(i, j) = 0.5 * (t_u_grad(i, j) - t_u_grad(j, i));
//         M(i, j) = 0.0;
//       }
//     }

//     for (int i = 0; i != 3; i++) {
//       for (int j = 0; j != 3; j++) {
//         for (int k = 0; k != 3; k++) {
//           M(i, j) += S(i, k) * S(k, j) + Omega(i, k) * Omega(k, j);
//         }
//       }
//     }

//     MatrixDouble eigen_vectors = M;
//     VectorDouble eigen_values(3);

//     // LAPACK - eigenvalues and vectors. Applied twice for initial creates
//     // memory space
//     int n = 3, lda = 3, info, lwork = -1;
//     double wkopt;
//     info = lapack_dsyev('V', 'U', n, &(eigen_vectors.data()[0]), lda,
//                         &(eigen_values.data()[0]), &wkopt, lwork);
//     if (info != 0)
//       SETERRQ1(PETSC_COMM_SELF, 1,
//                "is something wrong with lapack_dsyev info = %d", info);
//     lwork = (int)wkopt;
//     double work[lwork];
//     info = lapack_dsyev('V', 'U', n, &(eigen_vectors.data()[0]), lda,
//                         &(eigen_values.data()[0]), work, lwork);
//     if (info != 0)
//       SETERRQ1(PETSC_COMM_SELF, 1,
//                "is something wrong with lapack_dsyev info = %d", info);

//     map<double, int> eigen_sort;
//     eigen_sort[eigen_values[0]] = 0;
//     eigen_sort[eigen_values[1]] = 1;
//     eigen_sort[eigen_values[2]] = 2;

//     // prin_stress_vect.clear();
//     VectorDouble prin_vals_vect(3);
//     prin_vals_vect.clear();

//     int ii = 0;
//     for (map<double, int>::reverse_iterator mit = eigen_sort.rbegin();
//          mit != eigen_sort.rend(); mit++) {
//       prin_vals_vect[ii] = eigen_values[mit->second];
//       // for (int dd = 0; dd != 3; dd++) {
//       //   prin_stress_vect(ii, dd) = eigen_vectors.data()[3 * mit->second +
//       //   dd];
//       // }
//       ii++;
//     }

//     l2 = prin_vals_vect[1];
//     // cout << prin_vals_vect << endl;
//     // cout << "-0.5 sum: " << -0.5 * (prin_vals_vect[0] + prin_vals_vect[1] +
//     // prin_vals_vect[2]) << endl; cout << "q: " << q << endl;

//     // t_s(i,j) = 0.5*(t)

//     // vorticity(0) = t_u_grad(1, 2) - t_u_grad(2, 1);
//     // vorticity(1) = t_u_grad(2, 0) - t_u_grad(0, 2);
//     // vorticity(2) = t_u_grad(0, 1) - t_u_grad(1, 0);

//     CHKERR postProcMesh.tag_set_data(th_vorticity, &mapGaussPts[gg], 1,
//                                      &vorticity(0));
//     CHKERR postProcMesh.tag_set_data(th_q, &mapGaussPts[gg], 1, &q);
//     CHKERR postProcMesh.tag_set_data(th_l2, &mapGaussPts[gg], 1, &l2);
//     ++t_u_grad;
//   }

//   MoFEMFunctionReturn(0);
// }

// MoFEMErrorCode DirichletVelocityBc::iNitalize() {
//   MoFEMFunctionBegin;
//   if (mapZeroRows.empty() || !methodsOp.empty()) {
//     ParallelComm *pcomm =
//         ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);

//     for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
//       string name = it->getName();
//       if (name.compare(0, 3, "EXT") == 0) {
//         // if (name.compare(0, 7, "fdsfdsfsd") == 0) {

//         VectorDouble3 scaled_values(3);
//         // scaled_values[0] = 1.0;
//         // scaled_values[1] = 0.0;
//         // scaled_values[2] = 0.0;

//         for (int dim = 0; dim < 3; dim++) {
//           Range ents;
//           CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), dim,
//                                                      ents, true);
//           if (dim > 1) {
//             Range _edges;
//             CHKERR mField.get_moab().get_adjacencies(ents, 1, false, _edges,
//                                                      moab::Interface::UNION);
//             ents.insert(_edges.begin(), _edges.end());
//           }
//           if (dim > 0) {
//             Range _nodes;
//             CHKERR mField.get_moab().get_connectivity(ents, _nodes, true);
//             ents.insert(_nodes.begin(), _nodes.end());
//           }
//           for (Range::iterator eit = ents.begin(); eit != ents.end(); eit++) {
//             for (_IT_NUMEREDDOF_ROW_BY_NAME_ENT_PART_FOR_LOOP_(
//                      problemPtr, fieldName, *eit, pcomm->rank(), dof_ptr)) {
//               const boost::shared_ptr<NumeredDofEntity> &dof = *dof_ptr;
//               std::bitset<8> pstatus(dof->getPStatus());
//               if (pstatus.test(0))
//                 continue; // only local
//               if (dof->getEntType() == MBVERTEX) {

//                 EntityHandle ent = dof->getEnt();
//                 double coords[3];
//                 mField.get_moab().get_coords(&ent, 1,
//                                              coords); // get coordinates

//                 scaled_values =
//                     stokes_flow_velocity(coords[0], coords[1], coords[2]);

//                 if (dof->getDofCoeffIdx() == 0) {
//                   mapZeroRows[dof->getPetscGlobalDofIdx()] = scaled_values[0];
//                 }
//                 if (dof->getDofCoeffIdx() == 1) {
//                   mapZeroRows[dof->getPetscGlobalDofIdx()] = scaled_values[1];
//                 }
//                 if (dof->getDofCoeffIdx() == 2) {
//                   mapZeroRows[dof->getPetscGlobalDofIdx()] = scaled_values[2];
//                 }
//               } else {
//                 if (dof->getDofCoeffIdx() == 0) {
//                   mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
//                 }
//                 if (dof->getDofCoeffIdx() == 1) {
//                   mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
//                 }
//                 if (dof->getDofCoeffIdx() == 2) {
//                   mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
//                 }
//               }
//             }
//           }
//           // for (auto eit = ents.pair_begin(); eit != ents.pair_end(); ++eit) {
//           //   auto lo_dit =
//           //       problemPtr->getNumeredDofsRows()
//           //           ->get<Composite_Name_And_Ent_And_EntDofIdx_mi_tag>()
//           //           .lower_bound(boost::make_tuple(fieldName, eit->first,
//           //           0));
//           //   auto hi_dit =
//           //       problemPtr->getNumeredDofsRows()
//           //           ->get<Composite_Name_And_Ent_And_EntDofIdx_mi_tag>()
//           //           .upper_bound(boost::make_tuple(fieldName, eit->second,
//           //                                          MAX_DOFS_ON_ENTITY));
//           //   for (; lo_dit != hi_dit; ++lo_dit) {
//           //     auto &dof = *lo_dit;
//           //     if (dof->getEntType() == MBVERTEX) {
//           //       mapZeroRows[dof->getPetscGlobalDofIdx()] = scaled_values[0];
//           //     } else {
//           //       mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
//           //     }
//           //   }
//           // }
//           // for (Range::iterator eit = ents.begin(); eit != ents.end(); eit++)
//           // {
//           //   for (_IT_NUMEREDDOF_ROW_BY_NAME_ENT_PART_FOR_LOOP_(
//           //            problemPtr, fieldName, *eit, pcomm->rank(), dof_ptr)) {
//           //     NumeredDofEntity *dof = dof_ptr->get();
//           //     if (dof->getEntType() == MBVERTEX) {
//           //       mapZeroRows[dof->getPetscGlobalDofIdx()] = scaled_values[0];
//           //       // dof->getFieldData() = scaled_values[0];
//           //     } else {
//           //       mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
//           //       // dof->getFieldData() = 0;
//           //     }
//           //   }
//           //  }
//           // }
//         }
//       }
//     }
//     dofsIndices.resize(mapZeroRows.size());
//     dofsValues.resize(mapZeroRows.size());
//     int ii = 0;
//     std::map<DofIdx, FieldData>::iterator mit = mapZeroRows.begin();
//     for (; mit != mapZeroRows.end(); mit++, ii++) {
//       dofsIndices[ii] = mit->first;
//       dofsValues[ii] = mit->second;
//     }
//   }
//   MoFEMFunctionReturn(0);
// }

// MoFEMErrorCode DirichletVelocityBc::postProcess() {
//   MoFEMFunctionBegin;

//   switch (ts_ctx) {
//   case CTX_TSSETIFUNCTION: {
//     snes_ctx = CTX_SNESSETFUNCTION;
//     snes_x = ts_u;
//     snes_f = ts_F;
//     break;
//   }
//   case CTX_TSSETIJACOBIAN: {
//     snes_ctx = CTX_SNESSETJACOBIAN;
//     snes_B = ts_B;
//     break;
//   }
//   default:
//     break;
//   }

//   if (snes_B) {
//     CHKERR MatAssemblyBegin(snes_B, MAT_FINAL_ASSEMBLY);
//     CHKERR MatAssemblyEnd(snes_B, MAT_FINAL_ASSEMBLY);
//     CHKERR MatZeroRowsColumns(snes_B, dofsIndices.size(),
//                               dofsIndices.empty() ? PETSC_NULL
//                                                   : &dofsIndices[0],
//                               dIag, PETSC_NULL, PETSC_NULL);
//   }
//   if (snes_f) {
//     CHKERR VecAssemblyBegin(snes_f);
//     CHKERR VecAssemblyEnd(snes_f);
//     for (std::vector<int>::iterator vit = dofsIndices.begin();
//          vit != dofsIndices.end(); vit++) {
//       CHKERR VecSetValue(snes_f, *vit, 0., INSERT_VALUES);
//     }
//     CHKERR VecAssemblyBegin(snes_f);
//     CHKERR VecAssemblyEnd(snes_f);
//   }

//   MoFEMFunctionReturn(0);
// }

// MoFEMErrorCode DirichletPressureBc::iNitalize() {
//   MoFEMFunctionBegin;
//   if (mapZeroRows.empty() || !methodsOp.empty()) {
//     ParallelComm *pcomm =
//         ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);

//     for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
//       string name = it->getName();
//       if (name.compare(0, 5, "PRESS") == 0) {
//         // if (name.compare(0, 7, "fdsfdsfsd") == 0) {

//         VectorDouble scaled_values(1);
//         scaled_values[0] = 0.0;

//         for (int dim = 0; dim < 3; dim++) {
//           Range ents;
//           CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), dim,
//                                                      ents, true);
//           if (dim > 1) {
//             Range _edges;
//             CHKERR mField.get_moab().get_adjacencies(ents, 1, false, _edges,
//                                                      moab::Interface::UNION);
//             ents.insert(_edges.begin(), _edges.end());
//           }
//           if (dim > 0) {
//             Range _nodes;
//             CHKERR mField.get_moab().get_connectivity(ents, _nodes, true);
//             ents.insert(_nodes.begin(), _nodes.end());
//           }
//           for (auto eit = ents.pair_begin(); eit != ents.pair_end(); ++eit) {
//             auto lo_dit =
//                 problemPtr->getNumeredDofsRows()
//                     ->get<Composite_Name_And_Ent_And_EntDofIdx_mi_tag>()
//                     .lower_bound(boost::make_tuple(fieldName, eit->first, 0));
//             auto hi_dit =
//                 problemPtr->getNumeredDofsRows()
//                     ->get<Composite_Name_And_Ent_And_EntDofIdx_mi_tag>()
//                     .upper_bound(boost::make_tuple(fieldName, eit->second,
//                                                    MAX_DOFS_ON_ENTITY));
//             for (; lo_dit != hi_dit; ++lo_dit) {
//               auto &dof = *lo_dit;
//               if (dof->getEntType() == MBVERTEX) {
//                 mapZeroRows[dof->getPetscGlobalDofIdx()] = scaled_values[0];
//               } else {
//                 mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
//               }
//             }
//           }
//           // for (Range::iterator eit = ents.begin(); eit != ents.end(); eit++)
//           // {
//           //   for (_IT_NUMEREDDOF_ROW_BY_NAME_ENT_PART_FOR_LOOP_(
//           //            problemPtr, fieldName, *eit, pcomm->rank(), dof_ptr)) {
//           //     NumeredDofEntity *dof = dof_ptr->get();
//           //     if (dof->getEntType() == MBVERTEX) {
//           //       mapZeroRows[dof->getPetscGlobalDofIdx()] = scaled_values[0];
//           //       // dof->getFieldData() = scaled_values[0];
//           //     } else {
//           //       mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
//           //       // dof->getFieldData() = 0;
//           //     }
//           //   }
//           //  }
//           // }
//         }
//       }
//     }
//     dofsIndices.resize(mapZeroRows.size());
//     dofsValues.resize(mapZeroRows.size());
//     int ii = 0;
//     std::map<DofIdx, FieldData>::iterator mit = mapZeroRows.begin();
//     for (; mit != mapZeroRows.end(); mit++, ii++) {
//       dofsIndices[ii] = mit->first;
//       dofsValues[ii] = mit->second;
//     }
//   }
//   MoFEMFunctionReturn(0);
// }

// MoFEMErrorCode DirichletPressureBc::postProcess() {
//   MoFEMFunctionBegin;

//   switch (ts_ctx) {
//   case CTX_TSSETIFUNCTION: {
//     snes_ctx = CTX_SNESSETFUNCTION;
//     snes_x = ts_u;
//     snes_f = ts_F;
//     break;
//   }
//   case CTX_TSSETIJACOBIAN: {
//     snes_ctx = CTX_SNESSETJACOBIAN;
//     snes_B = ts_B;
//     break;
//   }
//   default:
//     break;
//   }

//   if (snes_B) {
//     CHKERR MatAssemblyBegin(snes_B, MAT_FINAL_ASSEMBLY);
//     CHKERR MatAssemblyEnd(snes_B, MAT_FINAL_ASSEMBLY);
//     CHKERR MatZeroRowsColumns(snes_B, dofsIndices.size(),
//                               dofsIndices.empty() ? PETSC_NULL
//                                                   : &dofsIndices[0],
//                               dIag, PETSC_NULL, PETSC_NULL);
//   }
//   if (snes_f) {
//     CHKERR VecAssemblyBegin(snes_f);
//     CHKERR VecAssemblyEnd(snes_f);
//     for (std::vector<int>::iterator vit = dofsIndices.begin();
//          vit != dofsIndices.end(); vit++) {
//       CHKERR VecSetValue(snes_f, *vit, 0., INSERT_VALUES);
//     }
//     CHKERR VecAssemblyBegin(snes_f);
//     CHKERR VecAssemblyEnd(snes_f);
//   }

//   MoFEMFunctionReturn(0);
// }

// VectorDouble3 stokes_flow_velocity(double x, double y, double z) {
//   double r = sqrt(x * x + y * y + z * z);
//   double theta = acos(x / r);
//   double phi = atan2(y, z);
//   double ur = cos(theta) * (1.0 + 0.5 / (r * r * r) - 1.5 / r);
//   double ut = -sin(theta) * (1.0 - 0.25 / (r * r * r) - 0.75 / r);
//   VectorDouble3 res(3);
//   res[0] = ur * cos(theta) - ut * sin(theta);
//   res[1] = ur * sin(theta) * sin(phi) + ut * cos(theta) * sin(phi);
//   res[2] = ur * sin(theta) * cos(phi) + ut * cos(theta) * cos(phi);
//   return res;
// }
