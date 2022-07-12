/* \file SurfacePressureComplexForLazy.hpp
 */

/* MIT License
 *
 * Copyright (c) 2022
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

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
                          EntitiesFieldData::EntData &data);
  };

  struct AuxMethodMaterial
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    MyTriangleSpatialFE *myPtr;
    AuxMethodMaterial(const string &field_name, MyTriangleSpatialFE *my_ptr,
                      const char type);
    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data);
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

    DEPRECATED MoFEMErrorCode addPreassure(int ms_id) {
      return addPressure(ms_id);
    }

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
