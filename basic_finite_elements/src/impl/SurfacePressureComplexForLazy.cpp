/* \file SurfacePressureComplexForLazy.cpp
 * --------------------------------------------------------------
 *
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
#include <MethodForForceScaling.hpp>
#include <SurfacePressure.hpp>
#include <SurfacePressureComplexForLazy.hpp>

extern "C" {

MoFEMErrorCode Traction_hierarchical(int order, int *order_edge, double *N,
                                     double *N_face, double *N_edge[],
                                     double *t, double *t_edge[],
                                     double *t_face, double *traction, int gg);
MoFEMErrorCode Fext_h_hierarchical(
    int order, int *order_edge, double *N, double *N_face, double *N_edge[],
    double *diffN, double *diffN_face, double *diffN_edge[], double *t,
    double *t_edge[], double *t_face, double *dofs_x, double *dofs_x_edge[],
    double *dofs_x_face, double *idofs_x, double *idofs_x_edge[],
    double *idofs_x_face, double *Fext, double *Fext_egde[], double *Fext_face,
    double *iFext, double *iFext_egde[], double *iFext_face, int g_dim,
    const double *g_w);
MoFEMErrorCode KExt_hh_hierarchical(
    double eps, int order, int *order_edge, double *N, double *N_face,
    double *N_edge[], double *diffN, double *diffN_face, double *diffN_edge[],
    double *t, double *t_edge[], double *t_face, double *dofs_x,
    double *dofs_x_edge[], double *dofs_x_face, double *KExt_hh,
    double *KExt_egdeh[3], double *KExt_faceh, int g_dim, const double *g_w);
MoFEMErrorCode KExt_hh_hierarchical_edge(
    double eps, int order, int *order_edge, double *N, double *N_face,
    double *N_edge[], double *diffN, double *diffN_face, double *diffN_edge[],
    double *t, double *t_edge[], double *t_face, double *dofs_x,
    double *dofs_x_edge[], double *dofs_x_face, double *Khext_edge[3],
    double *KExt_edgeegde[3][3], double *KExt_faceedge[3], int g_dim,
    const double *g_w);
MoFEMErrorCode
KExt_hh_hierarchical_face(double eps, int order, int *order_edge, double *N,
                          double *N_face, double *N_edge[], double *diffN,
                          double *diffN_face, double *diffN_edge[], double *t,
                          double *t_edge[], double *t_face, double *dofs_x,
                          double *dofs_x_edge[], double *dofs_x_face,
                          double *KExt_hface, double *KExt_egdeface[3],
                          double *KExt_faceface, int g_dim, const double *g_w);

void tetcircumcenter_tp(double a[3], double b[3], double c[3], double d[3],
                        double circumcenter[3], double *xi, double *eta,
                        double *zeta);
void tricircumcenter3d_tp(double a[3], double b[3], double c[3],
                          double circumcenter[3], double *xi, double *eta);
}

NeumannForcesSurfaceComplexForLazy::AuxMethodSpatial::AuxMethodSpatial(
    const string &field_name, MyTriangleSpatialFE *my_ptr, const char type)
    : FaceElementForcesAndSourcesCore::UserDataOperator(field_name, type),
      myPtr(my_ptr) {}

NeumannForcesSurfaceComplexForLazy::AuxMethodMaterial::AuxMethodMaterial(
    const string &field_name, MyTriangleSpatialFE *my_ptr, const char type)
    : FaceElementForcesAndSourcesCore::UserDataOperator(field_name, type),
      myPtr(my_ptr){};

MoFEMErrorCode NeumannForcesSurfaceComplexForLazy::AuxMethodSpatial::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  switch (type) {
  case MBVERTEX: {
    if (data.getFieldData().size() != 9) {
      SETERRQ2(PETSC_COMM_SELF, 1,
               "it should be 9 dofs on vertices but is %d of field < %s >",
               data.getFieldData().size(), rowFieldName.c_str());
    }
    myPtr->N = &*data.getN().data().begin();
    myPtr->diffN = &*data.getDiffN().data().begin();
    myPtr->dOfs_x.resize(data.getFieldData().size());
    ublas::noalias(myPtr->dOfs_x) = data.getFieldData();
    myPtr->dofs_x = &*myPtr->dOfs_x.data().begin();
    myPtr->dOfs_x_indices.resize(data.getIndices().size());
    ublas::noalias(myPtr->dOfs_x_indices) = data.getIndices();
    myPtr->dofs_x_indices = &*myPtr->dOfs_x_indices.data().begin();
  } break;
  case MBEDGE: {
    myPtr->order_edge[side] = data.getDataOrder();
    myPtr->N_edge[side] = &*data.getN().data().begin();
    myPtr->diffN_edge[side] = &*data.getDiffN().data().begin();
    myPtr->dOfs_x_edge.resize(3);
    myPtr->dOfs_x_edge[side].resize(data.getFieldData().size());
    myPtr->dofs_x_edge[side] = &*myPtr->dOfs_x_edge[side].data().begin();
    myPtr->dOfs_x_edge_indices.resize(3);
    myPtr->dOfs_x_edge_indices[side].resize(data.getIndices().size());
    ublas::noalias(myPtr->dOfs_x_edge_indices[side]) = data.getIndices();
    myPtr->dofs_x_edge_indices[side] =
        &*myPtr->dOfs_x_edge_indices[side].data().begin();
  } break;
  case MBTRI: {
    myPtr->order_face = data.getDataOrder();
    myPtr->N_face = &*data.getN().data().begin();
    myPtr->diffN_face = &*data.getDiffN().data().begin();
    myPtr->dOfs_x_face.resize(data.getFieldData().size());
    ublas::noalias(myPtr->dOfs_x_face) = data.getFieldData();
    myPtr->dofs_x_face = &*myPtr->dOfs_x_face.data().begin();
    myPtr->dOfs_x_face_indices.resize(data.getIndices().size());
    ublas::noalias(myPtr->dOfs_x_face_indices) = data.getIndices();
    myPtr->dofs_x_face_indices = &*myPtr->dOfs_x_face_indices.data().begin();
  } break;
  default:
    SETERRQ(PETSC_COMM_SELF, 1, "unknown entity type");
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NeumannForcesSurfaceComplexForLazy::AuxMethodMaterial::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  switch (type) {
  case MBVERTEX: {
    if (data.getFieldData().size() != 9) {
      SETERRQ1(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
               "it should be 9 dofs on vertices but is %d",
               data.getFieldData().size());
    }
    if (data.getN().size2() != 3) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "it should 3 shape functions for 3 nodes");
    }
    myPtr->N = &*data.getN().data().begin();
    myPtr->diffN = &*data.getDiffN().data().begin();
    myPtr->dOfs_X_indices.resize(data.getIndices().size());
    ublas::noalias(myPtr->dOfs_X_indices) = data.getIndices();
    myPtr->dofs_X_indices = &*myPtr->dOfs_X_indices.data().begin();
    myPtr->dOfs_X.resize(data.getFieldData().size());
    ublas::noalias(myPtr->dOfs_X) = data.getFieldData();
    myPtr->dofs_X = &*myPtr->dOfs_X.data().begin();
  } break;
  case MBEDGE: {
    myPtr->order_edge_material[side] = data.getDataOrder();
    myPtr->dOfs_X_edge.resize(3);
    myPtr->dOfs_X_edge[side].resize(data.getFieldData().size());
    ublas::noalias(myPtr->dOfs_X_edge[side]) = data.getFieldData();
    myPtr->dofs_X_edge[side] = &*myPtr->dOfs_X_edge[side].data().begin();
  } break;
  case MBTRI: {
    myPtr->order_face_material = data.getDataOrder();
    myPtr->dOfs_X_face.resize(data.getFieldData().size());
    ublas::noalias(myPtr->dOfs_X_face) = data.getFieldData();
    myPtr->dofs_X_face = &*myPtr->dOfs_X_face.data().begin();
  } break;
  default:
    SETERRQ(PETSC_COMM_SELF, 1, "unknown entity type");
  }

  MoFEMFunctionReturn(0);
}

NeumannForcesSurfaceComplexForLazy::MyTriangleSpatialFE::MyTriangleSpatialFE(
    MoFEM::Interface &_mField, Mat _Aij, Vec &_F, double *scale_lhs,
    double *scale_rhs, std::string spatial_field_name,
    std::string mat_field_name)
    : FaceElementForcesAndSourcesCore(_mField), sCaleLhs(scale_lhs),
      sCaleRhs(scale_rhs), typeOfForces(CONSERVATIVE), eps(1e-8), uSeF(false) {

  meshPositionsFieldName = "NoNE";
  methodsOp.clear();

  Aij = _Aij;
  F = _F;

  snes_B = _Aij;
  snes_f = _F;

  if (mField.check_field(mat_field_name)) {
    getOpPtrVector().push_back(new AuxMethodMaterial(
        mat_field_name, this, ForcesAndSourcesCore::UserDataOperator::OPROW));
  }
  getOpPtrVector().push_back(new AuxMethodSpatial(
      spatial_field_name, this, ForcesAndSourcesCore::UserDataOperator::OPROW));
}

MoFEMErrorCode NeumannForcesSurfaceComplexForLazy::MyTriangleSpatialFE::rHs() {
  MoFEMFunctionBegin;

  auto &dataH1 = *dataOnElement[H1];

  fExtNode.resize(9);
  fExtFace.resize(dataH1.dataOnEntities[MBTRI][0].getFieldData().size());
  fExtEdge.resize(3);
  for (int ee = 0; ee < 3; ee++) {
    int nb_edge_dofs = dOfs_x_edge_indices[ee].size();
    if (nb_edge_dofs > 0) {
      fExtEdge[ee].resize(nb_edge_dofs);
      Fext_edge[ee] = &*fExtEdge[ee].data().begin();
    } else {
      Fext_edge[ee] = NULL;
    }
  }

  switch (typeOfForces) {
  case CONSERVATIVE:
    CHKERR Fext_h_hierarchical(
        order_face, order_edge,                                          // 2
        N, N_face, N_edge, diffN, diffN_face, diffN_edge,                // 8
        t_loc, NULL, NULL,                                               // 11
        dofs_x, dofs_x_edge, dofs_x_face,                                // 14
        NULL, NULL, NULL,                                                // 17
        &*fExtNode.data().begin(), Fext_edge, &*fExtFace.data().begin(), // 20
        NULL, NULL, NULL,                                                // 23
        gaussPts.size2(), &gaussPts(2, 0));
    break;
  case NONCONSERVATIVE:
    for (int ee = 0; ee < 3; ee++) {
      dOfs_X_edge.resize(3);
      unsigned int s = dOfs_X_edge[ee].size();
      dOfs_X_edge[ee].resize(dOfs_x_edge[ee].size(), true);
      for (; s < dOfs_X_edge[ee].size(); s++) {
        dOfs_X_edge[ee][s] = 0;
      }
      dofs_X_edge[ee] = &*dOfs_X_edge[ee].data().begin();
    }
    unsigned int s = dOfs_X_face.size();
    dOfs_X_face.resize(dOfs_x_face.size(), true);
    for (; s < dOfs_X_face.size(); s++) {
      dOfs_X_face[s] = 0;
    }
    dofs_X_face = &*dOfs_X_face.data().begin();

    CHKERR Fext_h_hierarchical(
        order_face, order_edge,                                          // 2
        N, N_face, N_edge, diffN, diffN_face, diffN_edge,                // 8
        t_loc, NULL, NULL,                                               // 11
        dofs_X, dofs_X_edge, dofs_X_face,                                // 14
        NULL, NULL, NULL,                                                // 17
        &*fExtNode.data().begin(), Fext_edge, &*fExtFace.data().begin(), // 20
        NULL, NULL, NULL,                                                // 23
        gaussPts.size2(), &gaussPts(2, 0));
    break;
  }

  Vec f = snes_f;
  if (uSeF)
    f = F;

  CHKERR VecSetValues(f, 9, dofs_x_indices, &*fExtNode.data().begin(),
                      ADD_VALUES);
  if (dOfs_x_face_indices.size() > 0) {
    CHKERR VecSetValues(f, dOfs_x_face_indices.size(), dofs_x_face_indices,
                        &*fExtFace.data().begin(), ADD_VALUES);
  }
  for (int ee = 0; ee < 3; ee++) {
    if (dOfs_x_edge_indices[ee].size() > 0) {
      CHKERR VecSetValues(f, dOfs_x_edge_indices[ee].size(),
                          dofs_x_edge_indices[ee], Fext_edge[ee], ADD_VALUES);
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NeumannForcesSurfaceComplexForLazy::MyTriangleSpatialFE::lHs() {
  MoFEMFunctionBegin;

  if (typeOfForces == NONCONSERVATIVE) {
    MoFEMFunctionReturnHot(0);
  }

  auto &dataH1 = *dataOnElement[H1];

  double center[3];
  tricircumcenter3d_tp(&coords.data()[0], &coords.data()[3], &coords.data()[6],
                       center, NULL, NULL);
  cblas_daxpy(3, -1, &coords.data()[0], 1, center, 1);
  double r = cblas_dnrm2(3, center, 1);

  kExtNodeNode.resize(9, 9);
  kExtEdgeNode.resize(3);
  for (int ee = 0; ee < 3; ee++) {
    kExtEdgeNode[ee].resize(dOfs_x_edge_indices[ee].size(), 9);
    Kext_edge_node[ee] = &*kExtEdgeNode[ee].data().begin();
  }
  kExtFaceNode.resize(dOfs_x_face_indices.size(), 9);
  CHKERR KExt_hh_hierarchical(
      r * eps, order_face, order_edge, N, N_face, N_edge, diffN, diffN_face,
      diffN_edge, t_loc, NULL, NULL, dofs_x, dofs_x_edge, dofs_x_face,
      &*kExtNodeNode.data().begin(), Kext_edge_node,
      &*kExtFaceNode.data().begin(), gaussPts.size2(), &gaussPts(2, 0));
  // cerr << kExtNodeNode << endl;
  CHKERR MatSetValues(snes_B, 9, dofs_x_indices, 9, dofs_x_indices,
                      &*kExtNodeNode.data().begin(), ADD_VALUES);
  CHKERR MatSetValues(snes_B, kExtFaceNode.size1(), dofs_x_face_indices, 9,
                      dofs_x_indices, &*kExtFaceNode.data().begin(),
                      ADD_VALUES);
  // cerr << kExtFaceNode << endl;
  for (int ee = 0; ee < 3; ee++) {
    // cerr << kExtEdgeNode[ee] << endl;
    CHKERR MatSetValues(snes_B, kExtEdgeNode[ee].size1(),
                        dofs_x_edge_indices[ee], 9, dofs_x_indices,
                        Kext_edge_node[ee], ADD_VALUES);
  }

  kExtNodeFace.resize(9, dOfs_x_face_indices.size());
  kExtEdgeFace.resize(3);
  for (int ee = 0; ee < 3; ee++) {
    kExtEdgeFace[ee].resize(
        dOfs_x_edge_indices[ee].size(),
        dataH1.dataOnEntities[MBTRI][0].getIndices().size());
    Kext_edge_face[ee] = &*kExtEdgeFace[ee].data().begin();
  }
  kExtFaceFace.resize(dOfs_x_face_indices.size(), dOfs_x_face_indices.size());
  CHKERR KExt_hh_hierarchical_face(
      r * eps, order_face, order_edge, N, N_face, N_edge, diffN, diffN_face,
      diffN_edge, t_loc, NULL, NULL, dofs_x, dofs_x_edge, dofs_x_face,
      &*kExtNodeFace.data().begin(), Kext_edge_face,
      &*kExtFaceFace.data().begin(), gaussPts.size2(), &gaussPts(2, 0));
  // cerr << "kExtNodeFace " << kExtNodeFace << endl;
  // cerr << "kExtFaceFace " << kExtFaceFace << endl;
  CHKERR MatSetValues(snes_B, 9, dofs_x_indices, kExtNodeFace.size2(),
                      dofs_x_face_indices, &*kExtNodeFace.data().begin(),
                      ADD_VALUES);
  CHKERR MatSetValues(snes_B, kExtFaceFace.size1(), dofs_x_face_indices,
                      kExtFaceFace.size2(), dofs_x_face_indices,
                      &*kExtFaceFace.data().begin(), ADD_VALUES);
  for (int ee = 0; ee < 3; ee++) {
    // cerr << "kExtEdgeFace " << kExtEdgeFace[ee] << endl;
    CHKERR MatSetValues(snes_B, kExtEdgeFace[ee].size1(),
                        dofs_x_edge_indices[ee], kExtFaceFace.size2(),
                        dofs_x_face_indices, Kext_edge_face[ee], ADD_VALUES);
  }

  kExtFaceEdge.resize(3);
  kExtNodeEdge.resize(3);
  kExtEdgeEdge.resize(3, 3);
  for (int ee = 0; ee < 3; ee++) {
    if (dOfs_x_edge_indices[ee].size() !=
        (unsigned int)(3 * NBEDGE_H1(order_edge[ee]))) {
      SETERRQ(PETSC_COMM_SELF, 1, "data inconsistency");
    }
    kExtFaceEdge[ee].resize(dOfs_x_face_indices.size(),
                            dOfs_x_edge_indices[ee].size());
    kExtNodeEdge[ee].resize(9, dOfs_x_edge_indices[ee].size());
    Kext_node_edge[ee] = &*kExtNodeEdge[ee].data().begin();
    Kext_face_edge[ee] = &*kExtFaceEdge[ee].data().begin();
    for (int EE = 0; EE < 3; EE++) {
      kExtEdgeEdge(EE, ee).resize(dOfs_x_edge_indices[EE].size(),
                                  dOfs_x_edge_indices[ee].size());
      Kext_edge_edge[EE][ee] = &*kExtEdgeEdge(EE, ee).data().begin();
    }
  }
  CHKERR KExt_hh_hierarchical_edge(
      r * eps, order_face, order_edge, N, N_face, N_edge, diffN, diffN_face,
      diffN_edge, t_loc, NULL, NULL, dofs_x, dofs_x_edge, dofs_x_face,
      Kext_node_edge, Kext_edge_edge, Kext_face_edge, gaussPts.size2(),
      &gaussPts(2, 0));
  for (int ee = 0; ee < 3; ee++) {
    CHKERR MatSetValues(snes_B, kExtFaceEdge[ee].size1(), dofs_x_face_indices,
                        kExtFaceEdge[ee].size2(), dofs_x_edge_indices[ee],
                        &*kExtFaceEdge[ee].data().begin(), ADD_VALUES);
    CHKERR MatSetValues(snes_B, 9, dofs_x_indices, kExtNodeEdge[ee].size2(),
                        dofs_x_edge_indices[ee],
                        &*kExtNodeEdge[ee].data().begin(), ADD_VALUES);
    for (int EE = 0; EE < 3; EE++) {
      CHKERR MatSetValues(snes_B, kExtEdgeEdge(EE, ee).size1(),
                          dofs_x_edge_indices[EE], kExtEdgeEdge(EE, ee).size2(),
                          dofs_x_edge_indices[ee], Kext_edge_edge[EE][ee],
                          ADD_VALUES);
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NeumannForcesSurfaceComplexForLazy::MyTriangleSpatialFE::
    reBaseToFaceLoocalCoordSystem(MatrixDouble &t_glob_nodal) {
  MoFEMFunctionBeginHot;
  double s1[3], s2[3], normal[3], q[9];
  CHKERR ShapeFaceBaseMBTRI(diffN, &*coords.data().begin(), normal, s1, s2);
  double nrm2_normal = cblas_dnrm2(3, normal, 1);
  cblas_dscal(3, 1. / nrm2_normal, normal, 1);
  cblas_dcopy(3, s1, 1, &q[0], 1);
  cblas_dcopy(3, s2, 1, &q[3], 1);
  cblas_dcopy(3, normal, 1, &q[6], 1);
  __CLPK_integer info;
  __CLPK_integer ipiv[3];
  info = lapack_dgesv(3, 3, q, 3, ipiv, &*t_glob_nodal.data().begin(), 3);
  if (info != 0) {
    SETERRQ1(PETSC_COMM_SELF, 1, "error solve dgesv info = %d", info);
  }
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode
NeumannForcesSurfaceComplexForLazy::MyTriangleSpatialFE::calcTraction() {
  MoFEMFunctionBeginHot;
  EntityHandle ent = numeredEntFiniteElementPtr->getEnt();
  map<int, bCPressure>::iterator mip = mapPressure.begin();
  tLoc.resize(3);
  tLoc[0] = tLoc[1] = tLoc[2] = 0;
  for (; mip != mapPressure.end(); mip++) {
    if (mip->second.tRis.find(ent) != mip->second.tRis.end()) {
      tLoc[2] -= mip->second.data.data.value1;
    }
  }
  tLocNodal.resize(3, 3);
  for (int nn = 0; nn < 3; nn++) {
    for (int dd = 0; dd < 3; dd++) {
      tLocNodal(nn, dd) = tLoc[dd];
    }
  }

  map<int, bCForce>::iterator mif = mapForce.begin();
  for (; mif != mapForce.end(); mif++) {
    if (mif->second.tRis.find(ent) != mif->second.tRis.end()) {
      tGlob.resize(3);
      tGlob[0] = mif->second.data.data.value3;
      tGlob[1] = mif->second.data.data.value4;
      tGlob[2] = mif->second.data.data.value5;
      tGlob *= mif->second.data.data.value1;
      tGlobNodal.resize(3, 3);
      for (int nn = 0; nn < 3; nn++) {
        for (int dd = 0; dd < 3; dd++) {
          tGlobNodal(nn, dd) = tGlob[dd];
        }
      }
      CHKERR reBaseToFaceLoocalCoordSystem(tGlobNodal);
      tLocNodal += tGlobNodal;
    }
  }

  VectorDouble scale(1, 1);
  CHKERR MethodForForceScaling::applyScale(this, methodsOp, scale);
  tLocNodal *= scale[0];

  t_loc = &*tLocNodal.data().begin();

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode
NeumannForcesSurfaceComplexForLazy::MyTriangleSpatialFE::preProcess() {
  MoFEMFunctionBeginHot;

  CHKERR PetscOptionsBegin(mField.get_comm(), "",
                           "Surface Pressure (complex for lazy)", "none");
  PetscBool is_conservative = PETSC_TRUE;
  CHKERR PetscOptionsBool("-is_conservative_force", "is conservative force", "",
                          PETSC_TRUE, &is_conservative, PETSC_NULL);
  if (is_conservative == PETSC_FALSE) {
    typeOfForces = NONCONSERVATIVE;
  }
  ierr = PetscOptionsEnd();
  CHKERRG(ierr);

  switch (ts_ctx) {
  case CTX_TSSETIFUNCTION: {
    snes_ctx = CTX_SNESSETFUNCTION;
    snes_f = ts_F;
    break;
  }
  case CTX_TSSETIJACOBIAN: {
    snes_ctx = CTX_SNESSETJACOBIAN;
    snes_B = ts_B;
    break;
  }
  default:
    break;
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode NeumannForcesSurfaceComplexForLazy::MyTriangleSpatialFE::
operator()() {
  MoFEMFunctionBeginHot;

  {

    {

      dofs_X = &*coords.data().begin();
      for (int ee = 0; ee < 3; ee++) {
        dofs_X_edge[ee] = NULL;
        idofs_X_edge[ee] = NULL;
        order_edge_material[ee] = 0;
      }
      dofs_X_face = NULL;
      idofs_X_face = NULL;
      order_face_material = 0;

      dofs_x = &*coords.data().begin();
      idofs_x = NULL;
      for (int ee = 0; ee < 3; ee++) {
        order_edge[ee] = 0;
        N_edge[ee] = NULL;
        diffN_edge[ee] = NULL;
        dofs_x_edge[ee] = NULL;
        idofs_x_edge[ee] = NULL;
      }
      order_face = 0;
      N_face = NULL;
      diffN_face = NULL;
      dofs_x_face = NULL;
      idofs_x_face = NULL;
    }

    CHKERR FaceElementForcesAndSourcesCore::operator()();
    CHKERR calcTraction();

    switch (snes_ctx) {
    case CTX_SNESNONE:
    case CTX_SNESSETFUNCTION: {
      tLocNodal *= *sCaleRhs;
      // cerr << "sCaleRhs " << *sCaleRhs << endl;
      // cerr << tLocNodal << endl;
      CHKERR rHs();
    } break;
    case CTX_SNESSETJACOBIAN: {
      tLocNodal *= *sCaleLhs;
      CHKERR lHs();
    } break;
    }
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode
NeumannForcesSurfaceComplexForLazy::MyTriangleSpatialFE::addForce(int ms_id) {
  MeshsetsManager *mesh_manager_ptr;
  const CubitMeshSets *cubit_meshset_ptr;

  MoFEMFunctionBeginHot;
  CHKERR mField.getInterface(mesh_manager_ptr);
  CHKERR mesh_manager_ptr->getCubitMeshsetPtr(ms_id, NODESET,
                                              &cubit_meshset_ptr);
  CHKERR cubit_meshset_ptr->getBcDataStructure(mapForce[ms_id].data);
  CHKERR mField.get_moab().get_entities_by_type(
      cubit_meshset_ptr->meshset, MBTRI, mapForce[ms_id].tRis, true);
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode
NeumannForcesSurfaceComplexForLazy::MyTriangleSpatialFE::addPressure(
    int ms_id) {
  MeshsetsManager *mesh_manager_ptr;
  const CubitMeshSets *cubit_meshset_ptr;

  MoFEMFunctionBeginHot;
  CHKERR mField.getInterface(mesh_manager_ptr);
  CHKERR mesh_manager_ptr->getCubitMeshsetPtr(ms_id, SIDESET,
                                              &cubit_meshset_ptr);
  CHKERR cubit_meshset_ptr->getBcDataStructure(mapPressure[ms_id].data);
  CHKERR mField.get_moab().get_entities_by_type(
      cubit_meshset_ptr->meshset, MBTRI, mapPressure[ms_id].tRis, true);
  MoFEMFunctionReturnHot(0);
}

NeumannForcesSurfaceComplexForLazy::NeumannForcesSurfaceComplexForLazy(
    MoFEM::Interface &m_field, Mat _Aij, Vec _F, std::string spatial_field_name,
    std::string material_field_name)
    : mField(m_field), feSpatial(m_field, _Aij, _F, NULL, NULL,
                                 spatial_field_name, material_field_name),
      spatialField(spatial_field_name), materialField(material_field_name) {

  double def_scale = 1.;
  const EntityHandle root_meshset = mField.get_moab().get_root_set();
  CHKERR mField.get_moab().tag_get_handle(
      "_LoadFactor_Scale_", 1, MB_TYPE_DOUBLE, thScale,
      MB_TAG_CREAT | MB_TAG_EXCL | MB_TAG_MESH, &def_scale);
  if (rval == MB_ALREADY_ALLOCATED) {
    rval = mField.get_moab().tag_get_by_ptr(thScale, &root_meshset, 1,
                                            (const void **)&sCale);
    MOAB_THROW(rval);
  } else {
    MOAB_THROW(rval);
    rval =
        mField.get_moab().tag_set_data(thScale, &root_meshset, 1, &def_scale);
    MOAB_THROW(rval);
    rval = mField.get_moab().tag_get_by_ptr(thScale, &root_meshset, 1,
                                            (const void **)&sCale);
    MOAB_THROW(rval);
  }

  feSpatial.sCaleLhs = sCale;
  feSpatial.sCaleRhs = sCale;
}

NeumannForcesSurfaceComplexForLazy::NeumannForcesSurfaceComplexForLazy(
    MoFEM::Interface &m_field, Mat _Aij, Vec _F, double *scale_lhs,
    double *scale_rhs, std::string spatial_field_name,
    std::string material_field_name)
    : mField(m_field), feSpatial(m_field, _Aij, _F, scale_lhs, scale_rhs,
                                 spatial_field_name, material_field_name),
      spatialField(spatial_field_name), materialField(material_field_name) {}

extern "C" {

// External forces
MoFEMErrorCode Traction_hierarchical(int order, int *order_edge, double *N,
                                     double *N_face, double *N_edge[],
                                     double *t, double *t_edge[],
                                     double *t_face, double *traction, int gg) {
  MoFEMFunctionBeginHot;
  int dd, ee;
  for (dd = 0; dd < 3; dd++)
    traction[dd] = cblas_ddot(3, &N[gg * 3], 1, &t[dd], 3);
  if (t_face != NULL) {
    int nb_dofs_face = NBFACETRI_H1(order);
    if (nb_dofs_face > 0) {
      for (dd = 0; dd < 3; dd++)
        traction[dd] += cblas_ddot(nb_dofs_face, &N_face[gg * nb_dofs_face], 1,
                                   &t_face[dd], 3);
    }
  }
  if (t_edge != NULL) {
    ee = 0;
    for (; ee < 3; ee++) {
      if (t_edge[ee] == NULL)
        continue;
      int nb_dofs_edge = NBEDGE_H1(order_edge[ee]);
      if (nb_dofs_edge > 0) {
        for (dd = 0; dd < 3; dd++) {
          traction[dd] +=
              cblas_ddot(nb_dofs_edge, &(N_edge[ee][gg * nb_dofs_edge]), 1,
                         &(t_edge[ee][dd]), 3);
        }
      }
    }
  }
  MoFEMFunctionReturnHot(0);
}
MoFEMErrorCode Fext_h_hierarchical(
    int order, int *order_edge, double *N, double *N_face, double *N_edge[],
    double *diffN, double *diffN_face, double *diffN_edge[], double *t,
    double *t_edge[], double *t_face, double *dofs_x, double *dofs_x_edge[],
    double *dofs_x_face, double *idofs_x, double *idofs_x_edge[],
    double *idofs_x_face, double *Fext, double *Fext_edge[], double *Fext_face,
    double *iFext, double *iFext_edge[], double *iFext_face, int g_dim,
    const double *g_w) {
  MoFEMFunctionBeginHot;
  int dd, nn, ee, gg;
  if (Fext != NULL)
    bzero(Fext, 9 * sizeof(double));
  if (iFext != NULL)
    bzero(iFext, 9 * sizeof(double));
  ee = 0;
  for (; ee < 3; ee++) {
    int nb_dofs_edge = NBEDGE_H1(order_edge[ee]);
    if (nb_dofs_edge == 0)
      continue;
    if (Fext_edge != NULL)
      bzero(Fext_edge[ee], 3 * nb_dofs_edge * sizeof(double));
    if (iFext_edge != NULL)
      bzero(iFext_edge[ee], 3 * nb_dofs_edge * sizeof(double));
  }
  int nb_dofs_face = NBFACETRI_H1(order);
  if (nb_dofs_face != 0) {
    if (Fext_face != NULL)
      bzero(Fext_face, 3 * nb_dofs_face * sizeof(double));
    if (iFext_face != NULL)
      bzero(iFext_face, 3 * nb_dofs_face * sizeof(double));
  }
  gg = 0;
  for (; gg < g_dim; gg++) {
    double traction[3] = {0, 0, 0};
    CHKERR Traction_hierarchical(order, order_edge, N, N_face, N_edge, t,
                                 t_edge, t_face, traction, gg);
    __CLPK_doublecomplex xnormal[3], xs1[3], xs2[3];
    CHKERR Normal_hierarchical(
        // FIXME: order of tractions approximation could be different for field
        // approx.
        order, order_edge, order, order_edge, diffN, diffN_face, diffN_edge,
        dofs_x, dofs_x_edge, dofs_x_face, idofs_x, idofs_x_edge, idofs_x_face,
        xnormal, xs1, xs2, gg);
    CHKERR Base_scale(xnormal, xs1, xs2);
    double normal_real[3], s1_real[3], s2_real[3];
    double normal_imag[3], s1_imag[3], s2_imag[3];
    for (dd = 0; dd < 3; dd++) {
      normal_real[dd] = 0.5 * xnormal[dd].r;
      normal_imag[dd] = 0.5 * xnormal[dd].i;
      s1_real[dd] = 0.5 * xs1[dd].r;
      s1_imag[dd] = 0.5 * xs1[dd].i;
      s2_real[dd] = 0.5 * xs2[dd].r;
      s2_imag[dd] = 0.5 * xs2[dd].i;
    }
    nn = 0;
    for (; nn < 3; nn++) {
      if (Fext != NULL)
        for (dd = 0; dd < 3; dd++) {
          // fprintf(stderr,"%d %f %f %f %f %f %f\n",
          // gg,g_w[gg],N[3*gg+nn],normal_real[dd],
          // traction[0],traction[1],traction[2]);
          Fext[3 * nn + dd] +=
              g_w[gg] * N[3 * gg + nn] * normal_real[dd] * traction[2];
          Fext[3 * nn + dd] +=
              g_w[gg] * N[3 * gg + nn] * s1_real[dd] * traction[0];
          Fext[3 * nn + dd] +=
              g_w[gg] * N[3 * gg + nn] * s2_real[dd] * traction[1];
        }
      if (iFext != NULL)
        for (dd = 0; dd < 3; dd++) {
          iFext[3 * nn + dd] +=
              g_w[gg] * N[3 * gg + nn] * normal_imag[dd] * traction[2];
          iFext[3 * nn + dd] +=
              g_w[gg] * N[3 * gg + nn] * s1_imag[dd] * traction[0];
          iFext[3 * nn + dd] +=
              g_w[gg] * N[3 * gg + nn] * s2_imag[dd] * traction[1];
        }
    }
    ee = 0;
    for (; ee < 3; ee++) {
      int nb_dofs_edge = NBEDGE_H1(order_edge[ee]);
      if (nb_dofs_edge == 0)
        continue;
      int nn = 0;
      for (; nn < nb_dofs_edge; nn++) {
        if (Fext_edge != NULL)
          for (dd = 0; dd < 3; dd++) {
            Fext_edge[ee][3 * nn + dd] += g_w[gg] *
                                          N_edge[ee][gg * nb_dofs_edge + nn] *
                                          normal_real[dd] * traction[2];
            Fext_edge[ee][3 * nn + dd] += g_w[gg] *
                                          N_edge[ee][gg * nb_dofs_edge + nn] *
                                          s1_real[dd] * traction[0];
            Fext_edge[ee][3 * nn + dd] += g_w[gg] *
                                          N_edge[ee][gg * nb_dofs_edge + nn] *
                                          s2_real[dd] * traction[1];
          }
        if (iFext_edge != NULL) {
          for (dd = 0; dd < 3; dd++) {
            iFext_edge[ee][3 * nn + dd] += g_w[gg] *
                                           N_edge[ee][gg * nb_dofs_edge + nn] *
                                           normal_imag[dd] * traction[2];
            iFext_edge[ee][3 * nn + dd] += g_w[gg] *
                                           N_edge[ee][gg * nb_dofs_edge + nn] *
                                           s1_imag[dd] * traction[0];
            iFext_edge[ee][3 * nn + dd] += g_w[gg] *
                                           N_edge[ee][gg * nb_dofs_edge + nn] *
                                           s2_imag[dd] * traction[1];
          }
        }
      }
    }
    if (nb_dofs_face != 0) {
      nn = 0;
      for (; nn < nb_dofs_face; nn++) {
        if (Fext_face != NULL)
          for (dd = 0; dd < 3; dd++) {
            Fext_face[3 * nn + dd] += g_w[gg] * N_face[gg * nb_dofs_face + nn] *
                                      normal_real[dd] * traction[2];
            Fext_face[3 * nn + dd] += g_w[gg] * N_face[gg * nb_dofs_face + nn] *
                                      s1_real[dd] * traction[0];
            Fext_face[3 * nn + dd] += g_w[gg] * N_face[gg * nb_dofs_face + nn] *
                                      s2_real[dd] * traction[1];
          }
        if (iFext_face != NULL)
          for (dd = 0; dd < 3; dd++) {
            iFext_face[3 * nn + dd] += g_w[gg] *
                                       N_face[gg * nb_dofs_face + nn] *
                                       normal_imag[dd] * traction[2];
            iFext_face[3 * nn + dd] += g_w[gg] *
                                       N_face[gg * nb_dofs_face + nn] *
                                       s1_imag[dd] * traction[0];
            iFext_face[3 * nn + dd] += g_w[gg] *
                                       N_face[gg * nb_dofs_face + nn] *
                                       s2_imag[dd] * traction[1];
          }
      }
    }
  }
  MoFEMFunctionReturnHot(0);
}
MoFEMErrorCode KExt_hh_hierarchical(
    double eps, int order, int *order_edge, double *N, double *N_face,
    double *N_edge[], double *diffN, double *diffN_face, double *diffN_edge[],
    double *t, double *t_edge[], double *t_face, double *dofs_x,
    double *dofs_x_edge[], double *dofs_x_face, double *KExt_hh,
    double *KExt_edgeh[], double *KExt_faceh, int g_dim, const double *g_w) {
  MoFEMFunctionBeginHot;
  int gg, dd, ii, nn, ee;
  bzero(KExt_hh, 9 * 9 * sizeof(double));
  if (KExt_edgeh != NULL) {
    for (ee = 0; ee < 3; ee++) {
      int nb_dofs_edge = NBEDGE_H1(order_edge[ee]);
      bzero(KExt_edgeh[ee], 9 * 3 * nb_dofs_edge * sizeof(double));
    }
  }
  int nb_dofs_face = NBFACETRI_H1(order);
  if (KExt_faceh != NULL) {
    bzero(KExt_faceh, 9 * 3 * nb_dofs_face * sizeof(double));
  }
  for (gg = 0; gg < g_dim; gg++) {
    double traction[3] = {0, 0, 0};
    CHKERR Traction_hierarchical(order, order_edge, N, N_face, N_edge, t,
                                 t_edge, t_face, traction, gg);
    //
    __CLPK_doublecomplex xnormal[3], xs1[3], xs2[3];
    double idofs_x[9];
    for (ii = 0; ii < 9; ii++) {
      bzero(idofs_x, 9 * sizeof(double));
      idofs_x[ii] = eps;
      CHKERR Normal_hierarchical(order, order_edge, // FIXME
                                 order, order_edge, diffN, diffN_face,
                                 diffN_edge, dofs_x, dofs_x_edge, dofs_x_face,
                                 idofs_x, NULL, NULL, xnormal, xs1, xs2, gg);
      CHKERR Base_scale(xnormal, xs1, xs2);
      double normal_imag[3], s1_imag[3], s2_imag[3];
      for (dd = 0; dd < 3; dd++) {
        normal_imag[dd] = 0.5 * xnormal[dd].i / eps;
        s1_imag[dd] = 0.5 * xs1[dd].i / eps;
        s2_imag[dd] = 0.5 * xs2[dd].i / eps;
      }
      nn = 0;
      for (; nn < 3; nn++) {
        for (dd = 0; dd < 3; dd++) {
          KExt_hh[ii + 9 * 3 * nn + 9 * dd] +=
              g_w[gg] * N[3 * gg + nn] * normal_imag[dd] * traction[2];
          KExt_hh[ii + 9 * 3 * nn + 9 * dd] +=
              g_w[gg] * N[3 * gg + nn] * s1_imag[dd] * traction[0];
          KExt_hh[ii + 9 * 3 * nn + 9 * dd] +=
              g_w[gg] * N[3 * gg + nn] * s2_imag[dd] * traction[1];
        }
      }
      if (KExt_edgeh != NULL) {
        for (ee = 0; ee < 3; ee++) {
          int nb_dofs_edge = NBEDGE_H1(order_edge[ee]);
          for (nn = 0; nn < nb_dofs_edge; nn++) {
            for (dd = 0; dd < 3; dd++) {
              KExt_edgeh[ee][ii + 9 * 3 * nn + 9 * dd] +=
                  g_w[gg] * N_edge[ee][nb_dofs_edge * gg + nn] *
                  normal_imag[dd] * traction[2];
              KExt_edgeh[ee][ii + 9 * 3 * nn + 9 * dd] +=
                  g_w[gg] * N_edge[ee][nb_dofs_edge * gg + nn] * s1_imag[dd] *
                  traction[0];
              KExt_edgeh[ee][ii + 9 * 3 * nn + 9 * dd] +=
                  g_w[gg] * N_edge[ee][nb_dofs_edge * gg + nn] * s2_imag[dd] *
                  traction[1];
            }
          }
        }
      }
      if (KExt_faceh != NULL) {
        for (nn = 0; nn < nb_dofs_face; nn++) {
          for (dd = 0; dd < 3; dd++) {
            KExt_faceh[ii + 3 * 9 * nn + 9 * dd] +=
                g_w[gg] * N_face[nb_dofs_face * gg + nn] * normal_imag[dd] *
                traction[2];
            KExt_faceh[ii + 3 * 9 * nn + 9 * dd] +=
                g_w[gg] * N_face[nb_dofs_face * gg + nn] * s1_imag[dd] *
                traction[0];
            KExt_faceh[ii + 3 * 9 * nn + 9 * dd] +=
                g_w[gg] * N_face[nb_dofs_face * gg + nn] * s2_imag[dd] *
                traction[1];
          }
        }
      }
    }
  }
  MoFEMFunctionReturnHot(0);
}
MoFEMErrorCode KExt_hh_hierarchical_edge(
    double eps, int order, int *order_edge, double *N, double *N_face,
    double *N_edge[], double *diffN, double *diffN_face, double *diffN_edge[],
    double *t, double *t_edge[], double *t_face, double *dofs_x,
    double *dofs_x_edge[], double *dofs_x_face, double *KExt_hedge[3],
    double *KExt_edgeedge[3][3], double *KExt_faceedge[3], int g_dim,
    const double *g_w) {
  MoFEMFunctionBeginHot;
  int gg, dd, ii, nn, ee, EE;
  int nb_dofs_face = NBFACETRI_H1(order);
  for (EE = 0; EE < 3; EE++) {
    int nb_dofs_edge_EE = NBEDGE_H1(order_edge[EE]);
    bzero(KExt_hedge[EE], 9 * 3 * nb_dofs_edge_EE * sizeof(double));
    if (KExt_edgeedge != NULL) {
      for (ee = 0; ee < 3; ee++) {
        int nb_dofs_edge = NBEDGE_H1(order_edge[ee]);
        bzero(KExt_edgeedge[EE][ee],
              3 * nb_dofs_edge_EE * 3 * nb_dofs_edge * sizeof(double));
      }
    }
    if (KExt_faceedge != NULL) {
      bzero(KExt_faceedge[EE],
            3 * nb_dofs_edge_EE * 3 * nb_dofs_face * sizeof(double));
    }
  }
  for (gg = 0; gg < g_dim; gg++) {
    double traction[3] = {0, 0, 0};
    CHKERR Traction_hierarchical(order, order_edge, N, N_face, N_edge, t,
                                 t_edge, t_face, traction, gg);
    for (EE = 0; EE < 3; EE++) {
      int nb_dofs_edge_EE = NBEDGE_H1(order_edge[EE]);
      double *idofs_x_edge[3] = {NULL, NULL, NULL};
      double idofs_x_edge_EE[3 * nb_dofs_edge_EE];
      idofs_x_edge[EE] = idofs_x_edge_EE;
      for (ii = 0; ii < 3 * nb_dofs_edge_EE; ii++) {
        bzero(idofs_x_edge_EE, 3 * nb_dofs_edge_EE * sizeof(double));
        idofs_x_edge_EE[ii] = eps;
        __CLPK_doublecomplex xnormal[3], xs1[3], xs2[3];
        CHKERR Normal_hierarchical(order, order_edge, // FIXME
                                   order, order_edge, diffN, diffN_face,
                                   diffN_edge, dofs_x, dofs_x_edge, dofs_x_face,
                                   NULL, idofs_x_edge, NULL, xnormal, xs1, xs2,
                                   gg);
        CHKERR Base_scale(xnormal, xs1, xs2);
        double normal_imag[3], s1_imag[3], s2_imag[3];
        for (dd = 0; dd < 3; dd++) {
          normal_imag[dd] = 0.5 * xnormal[dd].i / eps;
          s1_imag[dd] = 0.5 * xs1[dd].i / eps;
          s2_imag[dd] = 0.5 * xs2[dd].i / eps;
        }
        for (nn = 0; nn < 3; nn++) {
          for (dd = 0; dd < 3; dd++) {
            KExt_hedge[EE][ii + 3 * nb_dofs_edge_EE * 3 * nn +
                           3 * nb_dofs_edge_EE * dd] +=
                g_w[gg] * N[3 * gg + nn] * normal_imag[dd] * traction[2];
            KExt_hedge[EE][ii + 3 * nb_dofs_edge_EE * 3 * nn +
                           3 * nb_dofs_edge_EE * dd] +=
                g_w[gg] * N[3 * gg + nn] * s1_imag[dd] * traction[0];
            KExt_hedge[EE][ii + 3 * nb_dofs_edge_EE * 3 * nn +
                           3 * nb_dofs_edge_EE * dd] +=
                g_w[gg] * N[3 * gg + nn] * s2_imag[dd] * traction[1];
          }
        }
        if (KExt_edgeedge != NULL) {
          for (ee = 0; ee < 3; ee++) {
            int nb_dofs_edge = NBEDGE_H1(order_edge[ee]);
            for (nn = 0; nn < nb_dofs_edge; nn++) {
              for (dd = 0; dd < 3; dd++) {
                KExt_edgeedge[EE][ee][ii + 3 * nb_dofs_edge_EE * 3 * nn +
                                      3 * nb_dofs_edge_EE * dd] +=
                    g_w[gg] * N_edge[ee][nb_dofs_edge * gg + nn] *
                    normal_imag[dd] * traction[2];
                KExt_edgeedge[EE][ee][ii + 3 * nb_dofs_edge_EE * 3 * nn +
                                      3 * nb_dofs_edge_EE * dd] +=
                    g_w[gg] * N_edge[ee][nb_dofs_edge * gg + nn] * s1_imag[dd] *
                    traction[0];
                KExt_edgeedge[EE][ee][ii + 3 * nb_dofs_edge_EE * 3 * nn +
                                      3 * nb_dofs_edge_EE * dd] +=
                    g_w[gg] * N_edge[ee][nb_dofs_edge * gg + nn] * s2_imag[dd] *
                    traction[1];
              }
            }
          }
        }
        if (KExt_faceedge != NULL) {
          for (nn = 0; nn < nb_dofs_face; nn++) {
            for (dd = 0; dd < 3; dd++) {
              KExt_faceedge[EE][ii + 3 * nb_dofs_edge_EE * 3 * nn +
                                3 * nb_dofs_edge_EE * dd] +=
                  g_w[gg] * N_face[nb_dofs_face * gg + nn] * normal_imag[dd] *
                  traction[2];
              KExt_faceedge[EE][ii + 3 * nb_dofs_edge_EE * 3 * nn +
                                3 * nb_dofs_edge_EE * dd] +=
                  g_w[gg] * N_face[nb_dofs_face * gg + nn] * s1_imag[dd] *
                  traction[0];
              KExt_faceedge[EE][ii + 3 * nb_dofs_edge_EE * 3 * nn +
                                3 * nb_dofs_edge_EE * dd] +=
                  g_w[gg] * N_face[nb_dofs_face * gg + nn] * s2_imag[dd] *
                  traction[1];
            }
          }
        }
      }
    }
  }
  MoFEMFunctionReturnHot(0);
}
MoFEMErrorCode
KExt_hh_hierarchical_face(double eps, int order, int *order_edge, double *N,
                          double *N_face, double *N_edge[], double *diffN,
                          double *diffN_face, double *diffN_edge[], double *t,
                          double *t_edge[], double *t_face, double *dofs_x,
                          double *dofs_x_edge[], double *dofs_x_face,
                          double *KExt_hface, double *KExt_edgeface[3],
                          double *KExt_faceface, int g_dim, const double *g_w) {
  MoFEMFunctionBeginHot;
  int gg, dd, ii, nn, ee;
  int nb_dofs_face = NBFACETRI_H1(order);
  bzero(KExt_hface, 9 * 3 * nb_dofs_face * sizeof(double));
  if (KExt_edgeface != NULL) {
    for (ee = 0; ee < 3; ee++) {
      int nb_dofs_edge = NBEDGE_H1(order_edge[ee]);
      bzero(KExt_edgeface[ee],
            3 * nb_dofs_face * 3 * nb_dofs_edge * sizeof(double));
    }
  }
  if (KExt_faceface != NULL) {
    bzero(KExt_faceface, 3 * nb_dofs_face * 3 * nb_dofs_face * sizeof(double));
  }
  for (gg = 0; gg < g_dim; gg++) {
    double traction[3] = {0, 0, 0};
    CHKERR Traction_hierarchical(order, order_edge, N, N_face, N_edge, t,
                                 t_edge, t_face, traction, gg);
    double idofs_x_face[3 * nb_dofs_face];
    for (ii = 0; ii < 3 * nb_dofs_face; ii++) {
      bzero(idofs_x_face, 3 * nb_dofs_face * sizeof(double));
      idofs_x_face[ii] = eps;
      __CLPK_doublecomplex xnormal[3], xs1[3], xs2[3];
      CHKERR Normal_hierarchical(
          order, order_edge, // FIXME
          order, order_edge, diffN, diffN_face, diffN_edge, dofs_x, dofs_x_edge,
          dofs_x_face, NULL, NULL, idofs_x_face, xnormal, xs1, xs2, gg);
      CHKERR Base_scale(xnormal, xs1, xs2);
      double normal_imag[3], s1_imag[3], s2_imag[3];
      for (dd = 0; dd < 3; dd++) {
        normal_imag[dd] = 0.5 * xnormal[dd].i / eps;
        s1_imag[dd] = 0.5 * xs1[dd].i / eps;
        s2_imag[dd] = 0.5 * xs2[dd].i / eps;
      }
      for (nn = 0; nn < 3; nn++) {
        for (dd = 0; dd < 3; dd++) {
          KExt_hface[ii + 3 * nb_dofs_face * 3 * nn + 3 * nb_dofs_face * dd] +=
              g_w[gg] * N[3 * gg + nn] * normal_imag[dd] * traction[2];
          KExt_hface[ii + 3 * nb_dofs_face * 3 * nn + 3 * nb_dofs_face * dd] +=
              g_w[gg] * N[3 * gg + nn] * s1_imag[dd] * traction[0];
          KExt_hface[ii + 3 * nb_dofs_face * 3 * nn + 3 * nb_dofs_face * dd] +=
              g_w[gg] * N[3 * gg + nn] * s2_imag[dd] * traction[1];
        }
      }
      if (KExt_edgeface != NULL) {
        for (ee = 0; ee < 3; ee++) {
          int nb_dofs_edge = NBEDGE_H1(order_edge[ee]);
          for (nn = 0; nn < nb_dofs_edge; nn++) {
            for (dd = 0; dd < 3; dd++) {
              KExt_edgeface[ee][ii + 3 * nb_dofs_face * 3 * nn +
                                3 * nb_dofs_face * dd] +=
                  g_w[gg] * N_edge[ee][nb_dofs_edge * gg + nn] *
                  normal_imag[dd] * traction[2];
              KExt_edgeface[ee][ii + 3 * nb_dofs_face * 3 * nn +
                                3 * nb_dofs_face * dd] +=
                  g_w[gg] * N_edge[ee][nb_dofs_edge * gg + nn] * s1_imag[dd] *
                  traction[0];
              KExt_edgeface[ee][ii + 3 * nb_dofs_face * 3 * nn +
                                3 * nb_dofs_face * dd] +=
                  g_w[gg] * N_edge[ee][nb_dofs_edge * gg + nn] * s2_imag[dd] *
                  traction[1];
            }
          }
        }
      }
      if (KExt_faceface != NULL) {
        for (nn = 0; nn < nb_dofs_face; nn++) {
          for (dd = 0; dd < 3; dd++) {
            KExt_faceface[ii + 3 * nb_dofs_face * 3 * nn +
                          3 * nb_dofs_face * dd] +=
                g_w[gg] * N_face[nb_dofs_face * gg + nn] * normal_imag[dd] *
                traction[2];
            KExt_faceface[ii + 3 * nb_dofs_face * 3 * nn +
                          3 * nb_dofs_face * dd] +=
                g_w[gg] * N_face[nb_dofs_face * gg + nn] * s1_imag[dd] *
                traction[0];
            KExt_faceface[ii + 3 * nb_dofs_face * 3 * nn +
                          3 * nb_dofs_face * dd] +=
                g_w[gg] * N_face[nb_dofs_face * gg + nn] * s2_imag[dd] *
                traction[1];
          }
        }
      }
    }
  }
  MoFEMFunctionReturnHot(0);
}
}
