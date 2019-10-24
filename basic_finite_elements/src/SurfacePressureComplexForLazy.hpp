/* \file SurfacePressureComplexForLazy.hpp
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

#ifndef __COMPLEX_FOR_LAZY_NEUMANN_FORCES_HPP__
#define __COMPLEX_FOR_LAZY_NEUMANN_FORCES_HPP__

/** \brief NonLinear surface pressure element (obsolete implementation)
  * \ingroup nonlinear_elastic_elem

  \todo This is old implementation, need to be reimplemented, using
  auto-differentiation. It is well tested and works. well.

  */
struct NeumannForcesSurfaceComplexForLazy {

  struct MyTriangleSpatialFE;
  struct AuxMethodSpatial
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    MyTriangleSpatialFE *myPtr;
    AuxMethodSpatial(const string &field_name, MyTriangleSpatialFE *my_ptr,
                     const char type);
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct AuxMethodMaterial
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    MyTriangleSpatialFE *myPtr;
    AuxMethodMaterial(const string &field_name, MyTriangleSpatialFE *my_ptr,
                      const char type);
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct MyTriangleSpatialFE : public MoFEM::FaceElementForcesAndSourcesCore {

    double *sCaleLhs;
    double *sCaleRhs;
    enum FORCES { CONSERVATIVE = 1, NONCONSERVATIVE = 2 };

    FORCES typeOfForces;
    const double eps;
    bool uSeF;
    bool spatialDisp;

    Mat Aij;
    Vec F;

    MyTriangleSpatialFE(MoFEM::Interface &_mField, Mat _Aij, Vec &_F,
                        double *scale_lhs, double *scale_rhs, 
                        std::string spatial_field_name = "SPATIAL_POSITION",
                        std::string mat_field_name = "MESH_NODE_POSITIONS");

    int getRule(int order) { return max(1, order); };

    double *N;
    double *N_face;
    double *N_edge[3];
    double *diffN;
    double *diffN_face;
    double *diffN_edge[3];

    int order_face;
    int order_edge[3];
    double *dofs_x;
    double *dofs_x_edge[3];
    double *dofs_x_face;
    double *idofs_x;
    double *idofs_x_edge[3];
    double *idofs_x_face;
    int *dofs_x_indices;
    int *dofs_x_edge_indices[3];
    int *dofs_x_face_indices;

    int order_face_material;
    int order_edge_material[3];
    double *dofs_X;
    double *dofs_X_edge[3];
    double *dofs_X_face;
    double *idofs_X;
    double *idofs_X_edge[3];
    double *idofs_X_face;

    int *dofs_X_indices;

    VectorDouble tLoc, tGlob;
    MatrixDouble tLocNodal, tGlobNodal;
    double *t_loc;

    ublas::vector<int> dOfs_x_indices, dOfs_x_face_indices;
    ublas::vector<ublas::vector<int>> dOfs_x_edge_indices;
    ublas::vector<int> dOfs_X_indices, dOfs_X_face_indices;
    ublas::vector<ublas::vector<int>> dOfs_X_edge_indices;

    VectorDouble dOfs_x, dOfs_x_face;
    ublas::vector<VectorDouble> dOfs_x_edge;
    VectorDouble dOfs_X, dOfs_X_face;
    ublas::vector<VectorDouble> dOfs_X_edge;

    VectorDouble fExtNode, fExtFace;
    ublas::vector<VectorDouble> fExtEdge;
    double *Fext_edge[3];

    MatrixDouble kExtNodeNode, kExtFaceNode;
    ublas::vector<MatrixDouble> kExtEdgeNode;
    double *Kext_edge_node[3];

    MatrixDouble kExtNodeFace, kExtFaceFace;
    ublas::vector<MatrixDouble> kExtEdgeFace;
    double *Kext_edge_face[3];

    ublas::vector<MatrixDouble> kExtFaceEdge, kExtNodeEdge;
    ublas::matrix<MatrixDouble> kExtEdgeEdge;
    double *Kext_node_edge[3];
    double *Kext_face_edge[3];
    double *Kext_edge_edge[3][3];

    virtual MoFEMErrorCode calcTraction();
    virtual MoFEMErrorCode rHs();
    virtual MoFEMErrorCode lHs();

    MoFEMErrorCode preProcess();
    MoFEMErrorCode operator()();

    MoFEMErrorCode addForce(int ms_id);
    MoFEMErrorCode addPressure(int ms_id);

    struct bCForce {
      ForceCubitBcData data;
      Range tRis;
    };
    map<int, bCForce> mapForce;
    struct bCPressure {
      PressureCubitBcData data;
      Range tRis;
    };
    map<int, bCPressure> mapPressure;
    MoFEMErrorCode reBaseToFaceLoocalCoordSystem(MatrixDouble &t_glob_nodal);

    boost::ptr_vector<MethodForForceScaling> methodsOp;
  };

  MoFEM::Interface &mField;
  MyTriangleSpatialFE feSpatial;

  Tag thScale;

  double *sCale;
  MoFEMErrorCode setForceScale(double scale) {
    MoFEMFunctionBeginHot;
    *sCale = scale;
    MoFEMFunctionReturnHot(0);
  }

  MyTriangleSpatialFE &getLoopSpatialFe() { return feSpatial; }

  NeumannForcesSurfaceComplexForLazy(
      MoFEM::Interface &m_field, Mat _Aij, Vec _F, double *scale_lhs,
      double *scale_rhs, std::string spatial_field_name = "SPATIAL_POSITION",
      std::string material_field_name = "MESH_NODE_POSITIONS");
  NeumannForcesSurfaceComplexForLazy(
      MoFEM::Interface &m_field, Mat _Aij, Vec _F,
      std::string spatial_field_name = "SPATIAL_POSITION",
      std::string material_field_name = "MESH_NODE_POSITIONS");

private:
  const std::string spatialField;
  const std::string materialField;
};

/**
 * \depracted do not use name with spelling mistake.
 */
DEPRECATED typedef NeumannForcesSurfaceComplexForLazy
    NeummanForcesSurfaceComplexForLazy;

#endif //__COMPLEX_FOR_LAZY_NEUMANN_FORCES_HPP__
