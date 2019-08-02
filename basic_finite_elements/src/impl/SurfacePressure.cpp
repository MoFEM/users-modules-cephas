/* \file SurfacePressure.cpp
  \brief Implementation of pressure and forces on triangles surface
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
#include <MethodForForceScaling.hpp>
#include <SurfacePressure.hpp>
#include <NodalForce.hpp>

using namespace boost::numeric;

MoFEMErrorCode NeummanForcesSurface::LinearVaringPresssure::getForce(
    const EntityHandle ent, const VectorDouble3 &coords,
    const VectorDouble3 &normal, VectorDouble3 &force) {
  MoFEMFunctionBegin;
  const double p = inner_prod(coords, linearConstants) + pressureShift;
  force = normal * p / norm_2(normal);
  MoFEMFunctionReturn(0);
}

NeummanForcesSurface::MyTriangleFE::MyTriangleFE(MoFEM::Interface &m_field)
    : FaceElementForcesAndSourcesCore(m_field), addToRule(1) {}

NeummanForcesSurface::OpNeumannForce::OpNeumannForce(
    const std::string field_name, Vec _F, bCForce &data,
    boost::ptr_vector<MethodForForceScaling> &methods_op, bool ho_geometry)
    : FaceElementForcesAndSourcesCore::UserDataOperator(
          field_name, UserDataOperator::OPROW),
      F(_F), dAta(data), methodsOp(methods_op), hoGeometry(ho_geometry) {}

MoFEMErrorCode NeummanForcesSurface::OpNeumannForce::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {

  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  EntityHandle ent = getNumeredEntFiniteElementPtr()->getEnt();
  if (dAta.tRis.find(ent) == dAta.tRis.end())
    MoFEMFunctionReturnHot(0);

  int rank = data.getFieldDofs()[0]->getNbOfCoeffs();
  int nb_row_dofs = data.getIndices().size() / rank;

  Nf.resize(data.getIndices().size(), false);
  Nf.clear();

  for (unsigned int gg = 0; gg < data.getN().size1(); gg++) {

    // get integration weight and Jacobian of integration point (area of face)
    double val = getGaussPts()(2, gg);
    if (hoGeometry) {
      val *= 0.5 * cblas_dnrm2(3, &getNormalsAtGaussPts()(gg, 0), 1);
    } else {
      val *= getArea();
    }

    // use data from module
    for (int rr = 0; rr < rank; rr++) {
      double force;
      if (rr == 0) {
        force = dAta.data.data.value3;
      } else if (rr == 1) {
        force = dAta.data.data.value4;
      } else if (rr == 2) {
        force = dAta.data.data.value5;
      } else {
        SETERRQ(PETSC_COMM_SELF, 1, "data inconsistency");
      }
      force *= dAta.data.data.value1;
      cblas_daxpy(nb_row_dofs, val * force, &data.getN()(gg, 0), 1, &Nf[rr],
                  rank);
    }
  }

  // Scale force using user defined scaling operator
  CHKERR MethodForForceScaling::applyScale(getFEMethod(), methodsOp, Nf);

  {
    Vec my_f;
    // If user vector is not set, use vector from snes or ts solvers
    if (F == PETSC_NULL) {
      switch (getFEMethod()->ts_ctx) {
      case FEMethod::CTX_TSSETIFUNCTION: {
        const_cast<FEMethod *>(getFEMethod())->snes_ctx =
            FEMethod::CTX_SNESSETFUNCTION;
        const_cast<FEMethod *>(getFEMethod())->snes_x = getFEMethod()->ts_u;
        const_cast<FEMethod *>(getFEMethod())->snes_f = getFEMethod()->ts_F;
        break;
      }
      default:
        break;
      }
      my_f = getFEMethod()->snes_f;
    } else {
      my_f = F;
    }

    // Assemble force into vector
    CHKERR VecSetValues(my_f, data.getIndices().size(), &data.getIndices()[0],
                        &Nf[0], ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

NeummanForcesSurface::OpNeumannForceAnalytical::OpNeumannForceAnalytical(
    const std::string field_name, Vec f, const Range tris,
    boost::ptr_vector<MethodForForceScaling> &methods_op,
    boost::shared_ptr<MethodForAnalyticalForce> &analytical_force_op,
    const bool ho_geometry)
    : FaceElementForcesAndSourcesCore::UserDataOperator(
          field_name, UserDataOperator::OPROW),
      F(f), tRis(tris), methodsOp(methods_op),
      analyticalForceOp(analytical_force_op), hoGeometry(ho_geometry) {}

MoFEMErrorCode NeummanForcesSurface::OpNeumannForceAnalytical::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  const EntityHandle ent = getNumeredEntFiniteElementPtr()->getEnt();
  if (tRis.find(ent) == tRis.end())
    MoFEMFunctionReturnHot(0);

  const int rank = data.getFieldDofs()[0]->getNbOfCoeffs();
  const int nb_row_dofs = data.getIndices().size() / rank;

  nF.resize(data.getIndices().size(), false);
  nF.clear();

  VectorDouble3 coords(3);
  VectorDouble3 normal(3);
  VectorDouble3 force(3);

  for (unsigned int gg = 0; gg != data.getN().size1(); ++gg) {

    // get integration weight and Jacobian of integration point (area of face)
    double val = getGaussPts()(2, gg);
    if (hoGeometry) {
      val *= 0.5 * cblas_dnrm2(3, &getNormalsAtGaussPts()(gg, 0), 1);
      for (int dd = 0; dd != 3; dd++) {
        coords[dd] = getHoCoordsAtGaussPts()(gg, dd);
        normal[dd] = getNormalsAtGaussPts()(gg, dd);
      }
    } else {
      val *= getArea();
      for (int dd = 0; dd != 3; dd++) {
        coords[dd] = getCoordsAtGaussPts()(gg, dd);
        normal = getNormal();
      }
    }

    if (analyticalForceOp) {
      CHKERR analyticalForceOp->getForce(ent, coords, normal, force);
      for (int rr = 0; rr != 3; ++rr) {
        cblas_daxpy(nb_row_dofs, val * force[rr], &data.getN()(gg, 0), 1,
                    &nF[rr], rank);
      }
    } else {
      SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA, "No force to apply");
    }
  }

  // Scale force using user defined scaling operator
  CHKERR MethodForForceScaling::applyScale(getFEMethod(), methodsOp, nF);

  {
    Vec my_f;
    // If user vector is not set, use vector from snes or ts solvers
    if (F == PETSC_NULL) {
      switch (getFEMethod()->ts_ctx) {
      case FEMethod::CTX_TSSETIFUNCTION: {
        const_cast<FEMethod *>(getFEMethod())->snes_ctx =
            FEMethod::CTX_SNESSETFUNCTION;
        const_cast<FEMethod *>(getFEMethod())->snes_x = getFEMethod()->ts_u;
        const_cast<FEMethod *>(getFEMethod())->snes_f = getFEMethod()->ts_F;
        break;
      }
      default:
        break;
      }
      my_f = getFEMethod()->snes_f;
    } else {
      my_f = F;
    }

    // Assemble force into vector
    CHKERR VecSetValues(my_f, data.getIndices().size(),
                        &*data.getIndices().data().begin(), &*nF.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

NeummanForcesSurface::OpNeumannPressure::OpNeumannPressure(
    const std::string field_name, Vec _F, bCPressure &data,
    boost::ptr_vector<MethodForForceScaling> &methods_op, bool ho_geometry)
    : FaceElementForcesAndSourcesCore::UserDataOperator(
          field_name, UserDataOperator::OPROW),
      F(_F), dAta(data), methodsOp(methods_op), hoGeometry(ho_geometry) {
        count = 0;
      }

MoFEMErrorCode NeummanForcesSurface::OpNeumannPressure::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {

  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  if (dAta.tRis.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tRis.end()) {
    MoFEMFunctionReturnHot(0);
  }

  int rank = data.getFieldDofs()[0]->getNbOfCoeffs();
  int nb_row_dofs = data.getIndices().size() / rank;

  Nf.resize(data.getIndices().size(), false);
  Nf.clear();

  for (unsigned int gg = 0; gg != data.getN().size1(); ++gg) {

    double val = getGaussPts()(2, gg);
    for (int rr = 0; rr != rank; ++rr) {

      double force;
      if (hoGeometry) {
        force = dAta.data.data.value1 * getNormalsAtGaussPts()(gg, rr);
      } else {
        force = dAta.data.data.value1 * getNormal()[rr];
      }
      cblas_daxpy(nb_row_dofs, 0.5 * val * force, &data.getN()(gg, 0), 1,
                  &Nf[rr], rank);
    }
  }

  CHKERR MethodForForceScaling::applyScale(getFEMethod(), methodsOp, Nf);
  {
    Vec my_f;
    if (F == PETSC_NULL) {
      switch (getFEMethod()->ts_ctx) {
      case FEMethod::CTX_TSSETIFUNCTION: {
        const_cast<FEMethod *>(getFEMethod())->snes_ctx =
            FEMethod::CTX_SNESSETFUNCTION;
        const_cast<FEMethod *>(getFEMethod())->snes_x = getFEMethod()->ts_u;
        const_cast<FEMethod *>(getFEMethod())->snes_f = getFEMethod()->ts_F;
        break;
      }
      default:
        break;
      }
      my_f = getFEMethod()->snes_f;
    } else {
      my_f = F;
    }
    CHKERR VecSetValues(my_f, data.getIndices().size(), &data.getIndices()[0],
                        &Nf[0], ADD_VALUES);

  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NeummanForcesSurface::OpGetTangent::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    PetscFunctionReturn(0);

  ngp = data.getN().size1();

  unsigned int nb_dofs = data.getFieldData().size() / 3;

  if (type == MBVERTEX) {
    dataAtIntegrationPts->tangent.resize(ngp);
    // tangent vectors to face F3
    for (unsigned int gg = 0; gg != ngp; ++gg) {
      dataAtIntegrationPts->tangent[gg].resize(2);
      dataAtIntegrationPts->tangent[gg][0].resize(3);
      dataAtIntegrationPts->tangent[gg][1].resize(3);
      dataAtIntegrationPts->tangent[gg][0].clear();
      dataAtIntegrationPts->tangent[gg][1].clear();
    }
  }

  for (unsigned int gg = 0; gg != ngp; ++gg) {
    for (unsigned int dd = 0; dd != 3; ++dd) {
      dataAtIntegrationPts->tangent[gg][0][dd] +=
          cblas_ddot(nb_dofs, &data.getDiffN()(gg, 0), 2, &data.getFieldData()[dd],
                     3); // tangent-1
      dataAtIntegrationPts->tangent[gg][1][dd] +=
          cblas_ddot(nb_dofs, &data.getDiffN()(gg, 1), 2, &data.getFieldData()[dd],
                     3); // tangent-2
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NeummanForcesSurface::OpNeumannPressureLhs_dx_dX::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  const int row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  const int col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  int nb_base_fun_row = row_data.getFieldData().size() / 3;
  int nb_base_fun_col = col_data.getFieldData().size() / 3;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor1 = [](VectorDouble3 &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto make_vec_der =
      [](VectorDouble3 &der_ksi, VectorDouble3 &der_eta,
         VectorDouble3 &normal_der, MatrixDouble &der_normal_mat,
         FTensor::Tensor0<double *> &t_N_over_ksi,
         FTensor::Tensor0<double *> &t_N_over_eta, MatrixDouble &spin,
         boost::shared_ptr<DataAtIntegrationPts> &dataAtIntegrationPts,
         const int &gg) {
        der_normal_mat.clear();

        for (int dd = 0; dd != 3; ++dd) {

          der_ksi.clear();
          der_eta.clear();
          normal_der.clear();

          der_ksi[dd] = t_N_over_ksi;
          der_eta[dd] = t_N_over_eta;

          spin.clear();
          CHKERR Spin(&*spin.data().begin(), &*der_ksi.data().begin());

          // n= t1 x t2 =  spin(t1)t2
          normal_der = prod(spin, dataAtIntegrationPts->tangent[gg][1]);

          spin.clear();
          CHKERR Spin(&*spin.data().begin(),
                      &*dataAtIntegrationPts->tangent[gg][0].data().begin());

          // n= t1 x t2 =  spin(t1)t2
          normal_der += prod(spin, der_eta);

          for (int kk = 0; kk != 3; ++kk) {
            der_normal_mat(kk, dd) += normal_der[kk];
          }
        }
      };

  MatrixDouble der_normal_mat;
  der_normal_mat.resize(3, 3, false);

  const bool diagonal_block = (row_type == col_type) && (row_side == col_side);

  MatrixDouble spin;
  spin.resize(3, 3, false);

  VectorDouble3 normal_der(3);
  VectorDouble3 der_ksi(3);
  VectorDouble3 der_eta(3);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    double val = getGaussPts()(2, gg); // * area;

    FTensor::Tensor0<double *> t_N_over_ksi(&col_data.getDiffN()(gg, 0));
    FTensor::Tensor0<double *> t_N_over_eta(&col_data.getDiffN()(gg, 1));

    int bbc = 0;
    for (; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; bbr++) {

        make_vec_der(der_ksi, der_eta, normal_der, der_normal_mat, t_N_over_ksi,
                     t_N_over_eta, spin, dataAtIntegrationPts, gg);

        auto d_n = get_tensor2(der_normal_mat, 0, 0);

        auto t_assemble = get_tensor2(NN, 3 * bbr, 3 * bbc);

        // TODO: handle hoGeometry 
        t_assemble(i, k) +=
            0.5 * val * dAta.data.data.value1 * t_base * d_n(i, k);

        ++t_base;
      }
      ++t_N_over_ksi;
      ++t_N_over_eta;
      ++t_N_over_ksi;
      ++t_N_over_eta;
    }
  }

  //scale matrix NN
  if (lambdaPtr) {
    NN *= *lambdaPtr;
  }

  CHKERR MatSetValues(
      getFEMethod()->snes_B, row_nb_dofs, &row_data.getIndices()[0],
      col_nb_dofs, &col_data.getIndices()[0], &*NN.data().begin(), ADD_VALUES);
      
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NeummanForcesSurface::OpCalculateDeformation::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &row_data) {

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

  auto t_detH = getFTensor0FromVec(*dataAtPts->detHVec);
  auto t_invH = getFTensor2FromMat<3, 3>(*dataAtPts->invHMat);
  auto t_F = getFTensor2FromMat<3, 3>(*dataAtPts->FMat);
  
  for (int gg = 0; gg != nb_integration_pts; ++gg) {
    CHKERR determinantTensor3by3(t_H, t_detH);
    CHKERR invertTensor3by3(t_H, t_detH, t_invH);
    t_F(i, j) = t_h(i, k) * t_invH(k, j);

    //t_h(i, j) = t_h(i, j) * 2.0;

    ++t_h;
    ++t_H;
    ++t_detH;
    ++t_invH;
    ++t_F;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NeummanForcesSurface::OpNeumannPressureMaterialRhs_dX::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &row_data) {

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

MoFEMErrorCode NeummanForcesSurface::OpNeumannPressureMaterialRhs_dX::iNtegrate(
    EntData &data) {
  MoFEMFunctionBegin;

  auto get_tensor1 = [](VectorDouble &v, const int r) {
    return FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3>(
        &v(r + 0), &v(r + 1), &v(r + 2));
  };

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  CHKERR loopSideVolumes(sideFeName, *sideFe);

  // get integration weights
  auto t_w = getFTensor0IntegrationWeight();

  auto t_normal = getFTensor1NormalsAtGaussPts();

  auto t_F = getFTensor2FromMat<3, 3>(*dataAtPts->FMat);

  //iterate over integration points
  for (int gg = 0; gg != nbIntegrationPts; ++gg) {

    FTensor::Tensor0<double *> t_base(&data.getN()(gg, 0));

    double a = -0.5 * t_w * dAta.data.data.value1; 
    auto t_nf = get_tensor1(nF, 0);

    int rr = 0;
    for (; rr != nbRows / 3; ++rr) {

      t_nf(i) += a * t_base * t_F(j, i) * t_normal(j);
      
      ++t_base;
      ++t_nf;
    }

    ++t_w;
    ++t_F;
    ++t_normal;
  }

  // if (lambdaPtr) {
  //   nF *= *lambdaPtr;
  // }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NeummanForcesSurface::OpNeumannPressureMaterialRhs_dX::aSsemble(
    EntData &row_data) {
  MoFEMFunctionBegin;

  // get pointer to first global index on row
  const int *row_indices = &*row_data.getIndices().data().begin();

  // auto &data = *dataAtPts;
  // if (!data.forcesOnlyOnEntitiesRow.empty()) {
  //   rowIndices.resize(nbRows, false);
  //   noalias(rowIndices) = row_data.getIndices();
  //   row_indices = &rowIndices[0];
  //   VectorDofs &dofs = row_data.getFieldDofs();
  //   VectorDofs::iterator dit = dofs.begin();
  //   for (int ii = 0; dit != dofs.end(); dit++, ii++) {
  //     if (data.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
  //         data.forcesOnlyOnEntitiesRow.end()) {
  //       rowIndices[ii] = -1;
  //     }
  //   }
  // }

  Vec F = getFEMethod()->snes_f;
  // assemble local matrix
  CHKERR VecSetValues(F, nbRows, row_indices, &*nF.data().begin(), ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NeummanForcesSurface::OpNeumannPressureMaterialLhs_dX_dX::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

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
    dataAtPts->faceRowData = &row_data;
    CHKERR loopSideVolumes(sideFeName, *sideFe);
  }

  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data, col_data);

  // assemble local matrix
  CHKERR aSsemble(row_data, col_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
NeummanForcesSurface::OpNeumannPressureMaterialLhs_dX_dX::iNtegrate(
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor1 = [](VectorDouble3 &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto make_vec_der =
      [](VectorDouble3 &der_ksi, VectorDouble3 &der_eta,
         VectorDouble3 &normal_der, MatrixDouble &der_normal_mat,
         FTensor::Tensor0<double *> &t_N_over_ksi,
         FTensor::Tensor0<double *> &t_N_over_eta, MatrixDouble &spin,
         boost::shared_ptr<DataAtIntegrationPtsMat> dataAtPts,
         const int &gg) {
        der_normal_mat.clear();

        for (int dd = 0; dd != 3; ++dd) {

          der_ksi.clear();
          der_eta.clear();
          normal_der.clear();

          der_ksi[dd] = t_N_over_ksi;
          der_eta[dd] = t_N_over_eta;

          spin.clear();
          CHKERR Spin(&*spin.data().begin(), &*der_ksi.data().begin());

          // n= t1 x t2 =  spin(t1)t2
          normal_der = prod(spin, dataAtPts->tangent[gg][1]);

          spin.clear();
          CHKERR Spin(&*spin.data().begin(),
                      &*dataAtPts->tangent[gg][0].data().begin());

          // n= t1 x t2 =  spin(t1)t2
          normal_der += prod(spin, der_eta);

          for (int kk = 0; kk != 3; ++kk) {
            der_normal_mat(kk, dd) += normal_der[kk];
          }
        }
      };

  MatrixDouble der_normal_mat;
  der_normal_mat.resize(3, 3, false);

  MatrixDouble spin;
  spin.resize(3, 3, false);

  VectorDouble3 normal_der(3);
  VectorDouble3 der_ksi(3);
  VectorDouble3 der_eta(3);

  dataAtPts->faceRowData = nullptr;
  CHKERR loopSideVolumes(sideFeName, *sideFe);

  auto t_F = getFTensor2FromMat<3, 3>(*dataAtPts->FMat);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    double val = getGaussPts()(2, gg); // * area;

    FTensor::Tensor0<double *> t_N_over_ksi(&col_data.getDiffN()(gg, 0));
    FTensor::Tensor0<double *> t_N_over_eta(&col_data.getDiffN()(gg, 1));

    int bbc = 0;
    for (; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; bbr++) {

        make_vec_der(der_ksi, der_eta, normal_der, der_normal_mat, t_N_over_ksi,
                     t_N_over_eta, spin, dataAtPts, gg);

        auto d_n = get_tensor2(der_normal_mat, 0, 0);  

        auto t_assemble = get_tensor2(NN, 3 * bbr, 3 * bbc);
        // TODO: handle hoGeometry 

        t_assemble(i, k) +=
            -0.5 * val * dAta.data.data.value1 * t_base * t_F(j, i) * d_n(j, k);

        ++t_base;
      }
      ++t_N_over_ksi;
      ++t_N_over_eta;
      ++t_N_over_ksi;
      ++t_N_over_eta;
    }
    ++t_F;
  }

  if (lambdaPtr) {
    NN *= *lambdaPtr;
  } 

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
NeummanForcesSurface::OpNeumannPressureMaterialLhs::aSsemble(
    EntData & row_data, EntData & col_data) {

  MoFEMFunctionBegin;

  // get pointer to first global index on row
  const int *row_indices = &*row_data.getIndices().data().begin();
  // get pointer to first global index on column
  const int *col_indices = &*col_data.getIndices().data().begin();

  auto &data = *dataAtPts;
  /*   if (!data.forcesOnlyOnEntitiesRow.empty()) {
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
    } */

  Mat B = getFEMethod()->ksp_B != PETSC_NULL ? getFEMethod()->ksp_B
                                             : getFEMethod()->snes_B;
  // assemble local matrix
  CHKERR MatSetValues(B, row_nb_dofs, row_indices, col_nb_dofs, col_indices,
                      &*NN.data().begin(), ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NeummanForcesSurface::OpNeumannPressureMaterialLhs_dX_dx::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

  MoFEMFunctionBegin;

  if (col_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  dataAtPts->faceRowData = &row_data;
  CHKERR loopSideVolumes(sideFeName, *sideFe);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
NeummanForcesSurface::OpNeumannPressureMaterialVolOnSideLhs::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {

  MoFEMFunctionBegin;

  if (dataAtPts->faceRowData == nullptr)
    MoFEMFunctionReturnHot(0);

  if (row_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  row_nb_dofs = dataAtPts->faceRowData->getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);

  nb_gauss_pts = dataAtPts->faceRowData->getN().size1();

  nb_base_fun_row = dataAtPts->faceRowData->getFieldData().size() / 3;
  nb_base_fun_col = col_data.getFieldData().size() / 3;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  //diagonal_block = (row_type == col_type) && (row_side == col_side);

  // integrate local matrix for entity block
  CHKERR iNtegrate(*(dataAtPts->faceRowData), col_data);

  // assemble local matrix
  CHKERR aSsemble(*(dataAtPts->faceRowData), col_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
NeummanForcesSurface::OpNeumannPressureMaterialVolOnSideLhs_dX_dx::iNtegrate(
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor1 = [](VectorDouble3 &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto t_w = getFTensor0IntegrationWeight();

  auto t_normal = getFTensor1NormalsAtGaussPts();

  auto t_inv_H = getFTensor2FromMat<3, 3>(*dataAtPts->invHMat);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

    int bbc = 0;
    for (; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_row_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; bbr++) {

        double a = -0.5 * t_w * dAta.data.data.value1;

        auto t_assemble = get_tensor2(NN, 3 * bbr, 3 * bbc);
        // TODO: handle hoGeometry 

        t_assemble(i, j) += a * t_row_base * t_inv_H(k, i) * t_col_diff_base(k) * t_normal(j);

        ++t_row_base;
      }
      ++t_col_diff_base;
    }
    ++t_w;
    ++t_normal;
    ++t_inv_H;
  }

  if (lambdaPtr) {
    NN *= *lambdaPtr;
  } 

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
NeummanForcesSurface::OpNeumannPressureMaterialVolOnSideLhs::aSsemble(
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;

  // get pointer to first global index on row
  const int *row_indices = &*row_data.getIndices().data().begin();
  // get pointer to first global index on column
  const int *col_indices = &*col_data.getIndices().data().begin();

  auto &data = *dataAtPts;
  /*   if (!data.forcesOnlyOnEntitiesRow.empty()) {
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
    } */

  Mat B = getFEMethod()->ksp_B != PETSC_NULL ? getFEMethod()->ksp_B
                                             : getFEMethod()->snes_B;
  // assemble local matrix
  CHKERR MatSetValues(B, row_nb_dofs, row_indices, col_nb_dofs, col_indices,
                      &*NN.data().begin(), ADD_VALUES);

/*   if (!isDiag && sYmm) {
    // if not diagonal term and since global matrix is symmetric assemble
    // transpose term.
    transK.resize(K.size2(), K.size1(), false);
    noalias(transK) = trans(K);
    CHKERR MatSetValues(B, nbCols, col_indices, nbRows, row_indices,
                        &*transK.data().begin(), ADD_VALUES);
  } */
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
NeummanForcesSurface::OpNeumannPressureMaterialVolOnSideLhs_dX_dX::iNtegrate(
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;
  FTensor::Index<'m', 3> m;

  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto get_tensor1 = [](VectorDouble3 &n) {
    return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
  };

  auto t_w = getFTensor0IntegrationWeight();

  auto t_normal = getFTensor1NormalsAtGaussPts();

  auto t_h = getFTensor2FromMat<3, 3>(*dataAtPts->hMat);
  auto t_inv_H = getFTensor2FromMat<3, 3>(*dataAtPts->invHMat);

  for (int gg = 0; gg != nb_gauss_pts; gg++) {

    auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);

    int bbc = 0;
    for (; bbc != nb_base_fun_col; bbc++) {

      FTensor::Tensor0<double *> t_row_base(&row_data.getN()(gg, 0));

      int bbr = 0;
      for (; bbr != nb_base_fun_row; bbr++) {

        double a = -0.5 * t_w * dAta.data.data.value1;

        auto t_assemble = get_tensor2(NN, 3 * bbr, 3 * bbc);
        // TODO: handle hoGeometry
        
        t_assemble(i, j) += -1.0 * a * t_row_base * t_inv_H(l, j) *
                            t_col_diff_base(m) * t_inv_H(m, i) * t_h(k, l) *
                            t_normal(k);

        ++t_row_base;
      }
      ++t_col_diff_base;
    }
    ++t_w;
    ++t_h;
    ++t_inv_H;
    ++t_normal;
  }

  if (lambdaPtr) {
    NN *= *lambdaPtr;
  } 

  MoFEMFunctionReturn(0);
}

NeummanForcesSurface::OpNeumannFlux::OpNeumannFlux(
    const std::string field_name, Vec _F, bCPressure &data,
    boost::ptr_vector<MethodForForceScaling> &methods_op, bool ho_geometry)
    : FaceElementForcesAndSourcesCore::UserDataOperator(
          field_name, UserDataOperator::OPROW),
      F(_F), dAta(data), methodsOp(methods_op), hoGeometry(ho_geometry) {}

MoFEMErrorCode NeummanForcesSurface::OpNeumannFlux::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  if (dAta.tRis.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tRis.end()) {
    MoFEMFunctionReturnHot(0);
  }

  int rank = data.getFieldDofs()[0]->getNbOfCoeffs();
  int nb_row_dofs = data.getIndices().size() / rank;

  Nf.resize(data.getIndices().size(), false);
  Nf.clear();

  for (unsigned int gg = 0; gg < data.getN().size1(); gg++) {

    double val = getGaussPts()(2, gg);
    double flux;
    if (hoGeometry) {
      double area = 0.5 * cblas_dnrm2(3, &getNormalsAtGaussPts()(gg, 0), 1);
      flux = dAta.data.data.value1 * area;
    } else {
      flux = dAta.data.data.value1 * getArea();
    }
    cblas_daxpy(nb_row_dofs, val * flux, &data.getN()(gg, 0), 1,
                &*Nf.data().begin(), 1);
  }

  CHKERR MethodForForceScaling::applyScale(getFEMethod(), methodsOp, Nf);
  {
    Vec my_f;
    if (F == PETSC_NULL) {
      switch (getFEMethod()->ts_ctx) {
      case FEMethod::CTX_TSSETIFUNCTION: {
        const_cast<FEMethod *>(getFEMethod())->snes_ctx =
            FEMethod::CTX_SNESSETFUNCTION;
        const_cast<FEMethod *>(getFEMethod())->snes_x = getFEMethod()->ts_u;
        const_cast<FEMethod *>(getFEMethod())->snes_f = getFEMethod()->ts_F;
        break;
      }
      default:
        break;
      }
      my_f = getFEMethod()->snes_f;
    } else {
      my_f = F;
    }
    CHKERR VecSetValues(my_f, data.getIndices().size(), &data.getIndices()[0],
                        &Nf[0], ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NeummanForcesSurface::addForce(const std::string field_name,
                                              Vec F, int ms_id,
                                              bool ho_geometry,
                                              bool block_set) {
  const CubitMeshSets *cubit_meshset_ptr;
  MeshsetsManager *mmanager_ptr;
  MoFEMFunctionBegin;
  CHKERR mField.getInterface(mmanager_ptr);
  if (block_set) {
    // Add data from block set.
    CHKERR mmanager_ptr->getCubitMeshsetPtr(ms_id, BLOCKSET,
                                            &cubit_meshset_ptr);
    std::vector<double> mydata;
    CHKERR cubit_meshset_ptr->getAttributes(mydata);
    VectorDouble force(mydata.size());
    for (unsigned int ii = 0; ii < mydata.size(); ii++) {
      force[ii] = mydata[ii];
    }
    // Read forces from BLOCKSET Force (if exists)
    if (force.empty()) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Force not given");
    }
    // Assign values from BLOCKSET FORCE to RHS vector. Info about native Cubit
    // BC data structure can be found in BCData.hpp
    const string name = "Force";
    strncpy(mapForce[ms_id].data.data.name, name.c_str(),
            name.size() > 5 ? 5 : name.size());
    double magnitude =
        sqrt(force[0] * force[0] + force[1] * force[1] + force[2] * force[2]);
    mapForce[ms_id].data.data.value1 = -magnitude; //< Force magnitude
    mapForce[ms_id].data.data.value2 = 0;
    mapForce[ms_id].data.data.value3 =
        force[0] / magnitude; //< X-component of force vector
    mapForce[ms_id].data.data.value4 =
        force[1] / magnitude; //< Y-component of force vector
    mapForce[ms_id].data.data.value5 =
        force[2] / magnitude; //< Z-component of force vector
    mapForce[ms_id].data.data.value6 = 0;
    mapForce[ms_id].data.data.value7 = 0;
    mapForce[ms_id].data.data.value8 = 0;
    mapForce[ms_id].data.data.zero[0] = 0;
    mapForce[ms_id].data.data.zero[1] = 0;
    mapForce[ms_id].data.data.zero[2] = 0;
    mapForce[ms_id].data.data.zero2 = 0;

    CHKERR mField.get_moab().get_entities_by_type(
        cubit_meshset_ptr->meshset, MBTRI, mapForce[ms_id].tRis, true);
    fe.getOpPtrVector().push_back(new OpNeumannForce(
        field_name, F, mapForce[ms_id], methodsOp, ho_geometry));

    // SETERRQ(PETSC_COMM_SELF,MOFEM_NOT_IMPLEMENTED,"Not implemented");
  } else {
    CHKERR mmanager_ptr->getCubitMeshsetPtr(ms_id, NODESET, &cubit_meshset_ptr);
    CHKERR cubit_meshset_ptr->getBcDataStructure(mapForce[ms_id].data);
    CHKERR mField.get_moab().get_entities_by_type(
        cubit_meshset_ptr->meshset, MBTRI, mapForce[ms_id].tRis, true);
    fe.getOpPtrVector().push_back(new OpNeumannForce(
        field_name, F, mapForce[ms_id], methodsOp, ho_geometry));
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NeummanForcesSurface::addPressure(const std::string field_name,
                                                 Vec F, int ms_id,
                                                 bool ho_geometry,
                                                 bool block_set) {

  const CubitMeshSets *cubit_meshset_ptr;
  MeshsetsManager *mmanager_ptr;
  MoFEMFunctionBegin;
  CHKERR mField.getInterface(mmanager_ptr);
  if (block_set) {
    CHKERR mmanager_ptr->getCubitMeshsetPtr(ms_id, BLOCKSET,
                                            &cubit_meshset_ptr);
    std::vector<double> mydata;
    CHKERR cubit_meshset_ptr->getAttributes(mydata);
    VectorDouble pressure(mydata.size());
    for (unsigned int ii = 0; ii < mydata.size(); ii++) {
      pressure[ii] = mydata[ii];
    }
    if (pressure.empty()) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Pressure not given");
    }
    const string name = "Pressure";
    strncpy(mapPressure[ms_id].data.data.name, name.c_str(),
            name.size() > 8 ? 8 : name.size());
    mapPressure[ms_id].data.data.flag1 = 0;
    mapPressure[ms_id].data.data.flag2 = 1;
    mapPressure[ms_id].data.data.value1 = pressure[0];
    mapPressure[ms_id].data.data.zero = 0;
    CHKERR mField.get_moab().get_entities_by_type(
        cubit_meshset_ptr->meshset, MBTRI, mapPressure[ms_id].tRis, true);
    fe.getOpPtrVector().push_back(new OpNeumannPressure(
        field_name, F, mapPressure[ms_id], methodsOp, ho_geometry));
  } else {
    CHKERR mmanager_ptr->getCubitMeshsetPtr(ms_id, SIDESET, &cubit_meshset_ptr);
    CHKERR cubit_meshset_ptr->getBcDataStructure(mapPressure[ms_id].data);
    CHKERR mField.get_moab().get_entities_by_type(
        cubit_meshset_ptr->meshset, MBTRI, mapPressure[ms_id].tRis, true);
    fe.getOpPtrVector().push_back(new OpNeumannPressure(
        field_name, F, mapPressure[ms_id], methodsOp, ho_geometry));
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NeummanForcesSurface::addPressure(
    const std::string x_field, const std::string X_field, Vec F,
    Mat aij, int ms_id, boost::shared_ptr<double> lambda_ptr, bool ho_geometry,
    bool block_set) {

  const CubitMeshSets *cubit_meshset_ptr;
  MeshsetsManager *mmanager_ptr;
  MoFEMFunctionBegin;
  CHKERR mField.getInterface(mmanager_ptr);
  if (block_set) {
    CHKERR mmanager_ptr->getCubitMeshsetPtr(ms_id, BLOCKSET,
                                            &cubit_meshset_ptr);
    std::vector<double> mydata;
    CHKERR cubit_meshset_ptr->getAttributes(mydata);
    VectorDouble pressure(mydata.size());
    for (unsigned int ii = 0; ii < mydata.size(); ii++) {
      pressure[ii] = mydata[ii];
    }
    if (pressure.empty()) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Pressure not given");
    }
    const string name = "Pressure";
    strncpy(mapPressure[ms_id].data.data.name, name.c_str(),
            name.size() > 8 ? 8 : name.size());
    mapPressure[ms_id].data.data.flag1 = 0;
    mapPressure[ms_id].data.data.flag2 = 1;
    mapPressure[ms_id].data.data.value1 = pressure[0];
    mapPressure[ms_id].data.data.zero = 0;
    CHKERR mField.get_moab().get_entities_by_type(
        cubit_meshset_ptr->meshset, MBTRI, mapPressure[ms_id].tRis, true);

  } else {
    CHKERR mmanager_ptr->getCubitMeshsetPtr(ms_id, SIDESET, &cubit_meshset_ptr);
    CHKERR cubit_meshset_ptr->getBcDataStructure(mapPressure[ms_id].data);
    CHKERR mField.get_moab().get_entities_by_type(
        cubit_meshset_ptr->meshset, MBTRI, mapPressure[ms_id].tRis, true);
  }

  // RIGHT-HAND SIDE

  fe.getOpPtrVector().push_back(new OpNeumannPressure(
      x_field, F, mapPressure[ms_id], methodsOp, ho_geometry));

  // LEFT-HAND SIDE

  boost::shared_ptr<NeummanForcesSurface::DataAtIntegrationPts>
      dataAtIntegrationPts =
          boost::make_shared<NeummanForcesSurface::DataAtIntegrationPts>();

  feLhs.getOpPtrVector().push_back(
      new OpGetTangent(X_field, dataAtIntegrationPts));

  feLhs.getOpPtrVector().push_back(new OpNeumannPressureLhs_dx_dX(
      x_field, X_field, dataAtIntegrationPts, aij, mapPressure[ms_id],
      lambda_ptr, ho_geometry));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NeummanForcesSurface::addPressureMaterial(
    const std::string x_field, const std::string X_field,
    std::string &side_fe_name, Vec F, Mat aij, int ms_id,
    boost::shared_ptr<double> lambda_ptr, bool ho_geometry, bool block_set) {

  const CubitMeshSets *cubit_meshset_ptr;
  MeshsetsManager *mmanager_ptr;
  MoFEMFunctionBegin;
  CHKERR mField.getInterface(mmanager_ptr);
  if (block_set) {
    CHKERR mmanager_ptr->getCubitMeshsetPtr(ms_id, BLOCKSET,
                                            &cubit_meshset_ptr);
    std::vector<double> mydata;
    CHKERR cubit_meshset_ptr->getAttributes(mydata);
    VectorDouble pressure(mydata.size());
    for (unsigned int ii = 0; ii < mydata.size(); ii++) {
      pressure[ii] = mydata[ii];
    }
    if (pressure.empty()) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Pressure not given");
    }
    const string name = "Pressure";
    strncpy(mapPressure[ms_id].data.data.name, name.c_str(),
            name.size() > 8 ? 8 : name.size());
    mapPressure[ms_id].data.data.flag1 = 0;
    mapPressure[ms_id].data.data.flag2 = 1;
    mapPressure[ms_id].data.data.value1 = pressure[0];
    mapPressure[ms_id].data.data.zero = 0;
    CHKERR mField.get_moab().get_entities_by_type(
        cubit_meshset_ptr->meshset, MBTRI, mapPressure[ms_id].tRis, true);

  } else {
    CHKERR mmanager_ptr->getCubitMeshsetPtr(ms_id, SIDESET, &cubit_meshset_ptr);
    CHKERR cubit_meshset_ptr->getBcDataStructure(mapPressure[ms_id].data);
    CHKERR mField.get_moab().get_entities_by_type(
        cubit_meshset_ptr->meshset, MBTRI, mapPressure[ms_id].tRis, true);
  }

  boost::shared_ptr<NeummanForcesSurface::DataAtIntegrationPtsMat> dataAtPts =
      boost::make_shared<NeummanForcesSurface::DataAtIntegrationPtsMat>();

  // RIGHT-HAND SIDE

  boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> feMatSideRhs =
      boost::make_shared<VolumeElementForcesAndSourcesCoreOnSide>(mField);

  feMatSideRhs->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<3, 3>(X_field, dataAtPts->HMat));
  feMatSideRhs->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<3, 3>(x_field, dataAtPts->hMat));
  feMatSideRhs->getOpPtrVector().push_back(
      new OpCalculateDeformation(X_field, dataAtPts, ho_geometry));

  feMatRhs.getOpPtrVector().push_back(new OpNeumannPressureMaterialRhs_dX(
      X_field, dataAtPts, feMatSideRhs, side_fe_name, F, mapPressure[ms_id],
      lambda_ptr, ho_geometry));

  // LEFT-HAND SIDE

  boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> feMatSideLhs_dx =
      boost::make_shared<VolumeElementForcesAndSourcesCoreOnSide>(mField);
  boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> feMatSideLhs_dX =
      boost::make_shared<VolumeElementForcesAndSourcesCoreOnSide>(mField);

  feMatSideLhs_dx->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<3, 3>(X_field, dataAtPts->HMat));
  feMatSideLhs_dx->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<3, 3>(x_field, dataAtPts->hMat));
  feMatSideLhs_dx->getOpPtrVector().push_back(
      new OpCalculateDeformation(X_field, dataAtPts, ho_geometry));
  feMatSideLhs_dx->getOpPtrVector().push_back(
      new OpNeumannPressureMaterialVolOnSideLhs_dX_dx(
          X_field, x_field, dataAtPts, aij, mapPressure[ms_id], lambda_ptr,
          ho_geometry));

  feMatLhs.getOpPtrVector().push_back(new OpNeumannPressureMaterialLhs_dX_dx(
      X_field, x_field, dataAtPts, feMatSideLhs_dx, side_fe_name, aij,
      mapPressure[ms_id], lambda_ptr, ho_geometry));

  feMatSideLhs_dX->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<3, 3>(X_field, dataAtPts->HMat));
  feMatSideLhs_dX->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<3, 3>(x_field, dataAtPts->hMat));
  feMatSideLhs_dX->getOpPtrVector().push_back(
      new OpCalculateDeformation(X_field, dataAtPts, ho_geometry));
  feMatSideLhs_dX->getOpPtrVector().push_back(
      new OpNeumannPressureMaterialVolOnSideLhs_dX_dX(
          X_field, X_field, dataAtPts, aij, mapPressure[ms_id], lambda_ptr,
          ho_geometry));

  feMatLhs.getOpPtrVector().push_back(new OpGetTangent(X_field, dataAtPts));
  feMatLhs.getOpPtrVector().push_back(new OpNeumannPressureMaterialLhs_dX_dX(
      X_field, X_field, dataAtPts, feMatSideLhs_dX, side_fe_name, aij,
      mapPressure[ms_id], lambda_ptr, ho_geometry));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
NeummanForcesSurface::addLinearPressure(const std::string field_name, Vec F,
                                        int ms_id, bool ho_geometry) {

  const CubitMeshSets *cubit_meshset_ptr;
  MeshsetsManager *mmanager_ptr;
  MoFEMFunctionBegin;
  CHKERR mField.getInterface(mmanager_ptr);
  CHKERR mmanager_ptr->getCubitMeshsetPtr(ms_id, BLOCKSET, &cubit_meshset_ptr);
  std::vector<double> mydata;
  CHKERR cubit_meshset_ptr->getAttributes(mydata);
  if (mydata.size() < 4)
    SETERRQ1(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
             "Should be four block attributes but is %d", mydata.size());
  VectorDouble3 pressure_coeffs(3);
  for (unsigned int ii = 0; ii != 3; ++ii) {
    pressure_coeffs[ii] = mydata[ii];
  }
  const double pressure_shift = mydata[3];

  Range tris;
  CHKERR mField.get_moab().get_entities_by_type(cubit_meshset_ptr->meshset,
                                                MBTRI, tris, true);
  boost::shared_ptr<MethodForAnalyticalForce> analytical_force_op(
      new LinearVaringPresssure(pressure_coeffs, pressure_shift));
  fe.getOpPtrVector().push_back(new OpNeumannForceAnalytical(
      field_name, F, tris, methodsOp, analytical_force_op, ho_geometry));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NeummanForcesSurface::addPreassure(const std::string field_name,
                                                  Vec F, int ms_id,
                                                  bool ho_geometry,
                                                  bool block_set) {
  return NeummanForcesSurface::addPressure(field_name, F, ms_id, ho_geometry,
                                           block_set);
}

MoFEMErrorCode NeummanForcesSurface::addFlux(const std::string field_name,
                                             Vec F, int ms_id,
                                             bool ho_geometry) {
  const CubitMeshSets *cubit_meshset_ptr;
  MeshsetsManager *mmanager_ptr;
  MoFEMFunctionBegin;
  CHKERR mField.getInterface(mmanager_ptr);
  CHKERR mmanager_ptr->getCubitMeshsetPtr(ms_id, SIDESET, &cubit_meshset_ptr);
  CHKERR cubit_meshset_ptr->getBcDataStructure(mapPressure[ms_id].data);
  CHKERR mField.get_moab().get_entities_by_type(
      cubit_meshset_ptr->meshset, MBTRI, mapPressure[ms_id].tRis, true);
  fe.getOpPtrVector().push_back(new OpNeumannFlux(
      field_name, F, mapPressure[ms_id], methodsOp, ho_geometry));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaNeummanForces::addNeumannBCElements(
    MoFEM::Interface &m_field, const std::string field_name,
    const std::string mesh_nodals_positions, Range *intersect_ptr) {
  MoFEMFunctionBegin;

  // Define boundary element that operates on rows, columns and data of a
  // given field
  CHKERR m_field.add_finite_element("FORCE_FE", MF_ZERO);
  CHKERR m_field.modify_finite_element_add_field_row("FORCE_FE", field_name);
  CHKERR m_field.modify_finite_element_add_field_col("FORCE_FE", field_name);
  CHKERR m_field.modify_finite_element_add_field_data("FORCE_FE", field_name);
  if (m_field.check_field(mesh_nodals_positions)) {
    CHKERR m_field.modify_finite_element_add_field_data("FORCE_FE",
                                                        mesh_nodals_positions);
  }
  // Add entities to that element, here we add all triangles with FORCESET
  // from cubit
  for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                  it)) {
    Range tris;
    CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI, tris,
                                                   true);
    if (intersect_ptr)
      tris = intersect(tris, *intersect_ptr);
    CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI, "FORCE_FE");
  }

  CHKERR m_field.add_finite_element("PRESSURE_FE", MF_ZERO);
  CHKERR m_field.modify_finite_element_add_field_row("PRESSURE_FE", field_name);
  CHKERR m_field.modify_finite_element_add_field_col("PRESSURE_FE", field_name);
  CHKERR m_field.modify_finite_element_add_field_data("PRESSURE_FE",
                                                      field_name);
  if (m_field.check_field(mesh_nodals_positions)) {
    CHKERR m_field.modify_finite_element_add_field_data("PRESSURE_FE",
                                                        mesh_nodals_positions);
  }

  for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,
                                                  SIDESET | PRESSURESET, it)) {
    Range tris;
    CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI, tris,
                                                   true);
    if (intersect_ptr)
      tris = intersect(tris, *intersect_ptr);
    CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI,
                                                      "PRESSURE_FE");
  }

  // Reading forces from BLOCKSET

  const string block_set_force_name("FORCE");
  // search for block named FORCE and add its attributes to FORCE_FE element
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
    if (it->getName().compare(0, block_set_force_name.length(),
                              block_set_force_name) == 0) {
      Range tris;
      CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI, tris,
                                                     true);
      if (intersect_ptr)
        tris = intersect(tris, *intersect_ptr);
      CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI,
                                                        "FORCE_FE");
    }
  }
  // search for block named PRESSURE and add its attributes to PRESSURE_FE
  // element
  const string block_set_pressure_name("PRESSURE");
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
    if (it->getName().compare(0, block_set_pressure_name.length(),
                              block_set_pressure_name) == 0) {
      Range tris;
      CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI, tris,
                                                     true);
      if (intersect_ptr)
        tris = intersect(tris, *intersect_ptr);
      CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI,
                                                        "PRESSURE_FE");
    }
  }

  // search for block named LINEAR_PRESSURE and add its attributes to
  // PRESSURE_FE element
  const string block_set_linear_pressure_name("LINEAR_PRESSURE");
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
    if (it->getName().compare(0, block_set_linear_pressure_name.length(),
                              block_set_linear_pressure_name) == 0) {
      Range tris;
      CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI, tris,
                                                     true);
      if (intersect_ptr)
        tris = intersect(tris, *intersect_ptr);
      CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI,
                                                        "PRESSURE_FE");
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaNeummanForces::setMomentumFluxOperators(
    MoFEM::Interface &m_field,
    boost::ptr_map<std::string, NeummanForcesSurface> &neumann_forces, Vec F,
    const std::string field_name, const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;

  string fe_name;
  fe_name = "FORCE_FE";
  neumann_forces.insert(fe_name, new NeummanForcesSurface(m_field));
  bool ho_geometry = m_field.check_field(mesh_nodals_positions);
  for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                  it)) {
    CHKERR neumann_forces.at(fe_name).addForce(
        field_name, F, it->getMeshsetId(), ho_geometry, false);
  }
  // Reading forces from BLOCKSET
  const string block_set_force_name("FORCE");
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
    if (it->getName().compare(0, block_set_force_name.length(),
                              block_set_force_name) == 0) {
      CHKERR neumann_forces.at(fe_name).addForce(
          field_name, F, it->getMeshsetId(), ho_geometry, true);
    }
  }

  fe_name = "PRESSURE_FE";
  neumann_forces.insert(fe_name, new NeummanForcesSurface(m_field));
  for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,
                                                  SIDESET | PRESSURESET, it)) {
    CHKERR neumann_forces.at(fe_name).addPressure(
        field_name, F, it->getMeshsetId(), ho_geometry, false);
  }

  // Reading pressures from BLOCKSET
  const string block_set_pressure_name("PRESSURE");
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
    if (it->getName().compare(0, block_set_pressure_name.length(),
                              block_set_pressure_name) == 0) {
      CHKERR neumann_forces.at(fe_name).addPressure(
          field_name, F, it->getMeshsetId(), ho_geometry, true);
    }
  }

  // Reading pressures from BLOCKSET
  const string block_set_linear_pressure_name("LINEAR_PRESSURE");
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
    if (it->getName().compare(0, block_set_linear_pressure_name.length(),
                              block_set_linear_pressure_name) == 0) {
      CHKERR neumann_forces.at(fe_name).addLinearPressure(
          field_name, F, it->getMeshsetId(), ho_geometry);
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaNeummanForces::addNeumannFluxBCElements(
    MoFEM::Interface &m_field, const std::string field_name,
    const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;

  CHKERR m_field.add_finite_element("FLUX_FE", MF_ZERO);
  CHKERR m_field.modify_finite_element_add_field_row("FLUX_FE", field_name);
  CHKERR m_field.modify_finite_element_add_field_col("FLUX_FE", field_name);
  CHKERR m_field.modify_finite_element_add_field_data("FLUX_FE", field_name);
  if (m_field.check_field(mesh_nodals_positions)) {
    CHKERR m_field.modify_finite_element_add_field_data("FLUX_FE",
                                                        mesh_nodals_positions);
  }

  for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,
                                                  SIDESET | PRESSURESET, it)) {
    Range tris;
    CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI, tris,
                                                   true);
    CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI, "FLUX_FE");
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaNeummanForces::setMassFluxOperators(
    MoFEM::Interface &m_field,
    boost::ptr_map<std::string, NeummanForcesSurface> &neumann_forces, Vec F,
    const std::string field_name, const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;

  string fe_name;
  fe_name = "FLUX_FE";
  neumann_forces.insert(fe_name, new NeummanForcesSurface(m_field));
  for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,
                                                  SIDESET | PRESSURESET, it)) {
    bool ho_geometry = m_field.check_field(mesh_nodals_positions);
    CHKERR neumann_forces.at(fe_name).addFlux(field_name, F, it->getMeshsetId(),
                                              ho_geometry);
  }
  MoFEMFunctionReturn(0);
}