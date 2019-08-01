/* \file SurfacePressure.hpp
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

#ifndef __SURFACE_PERSSURE_HPP__
#define __SURFACE_PERSSURE_HPP__

/** \brief Finite element and operators to apply force/pressures applied to
 * surfaces \ingroup mofem_static_boundary_conditions
 */
struct NeummanForcesSurface {

  MoFEM::Interface &mField;

  /**
   * \brief Analytical force method
   */
  struct MethodForAnalyticalForce {

    virtual ~MethodForAnalyticalForce() {}

    /**
     * User implemented analytical force
     * @param  coords coordinates of integration point
     * @param  normal normal at integration point
     * @param  force  returned force
     * @return        error code
     */
    virtual MoFEMErrorCode getForce(const EntityHandle ent,
                                    const VectorDouble3 &coords,
                                    const VectorDouble3 &normal,
                                    VectorDouble3 &force) {
      MoFEMFunctionBeginHot;
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
              "You need to implement this");
      MoFEMFunctionReturnHot(0);
    }
  };

  struct LinearVaringPresssure : public MethodForAnalyticalForce {

    LinearVaringPresssure(const VectorDouble3 &p, const double c)
        : MethodForAnalyticalForce(), linearConstants(p), pressureShift(c) {}

    MoFEMErrorCode getForce(const EntityHandle ent, const VectorDouble3 &coords,
                            const VectorDouble3 &normal, VectorDouble3 &force);

  private:
    const VectorDouble3 linearConstants;
    const double pressureShift;
  };

  /**
   * Definition of face element used for integration
   */
  struct MyTriangleFE : public MoFEM::FaceElementForcesAndSourcesCore {
    int addToRule;
    MyTriangleFE(MoFEM::Interface &m_field);
    int getRule(int order) { return 2 * order + addToRule; };
  };

  MyTriangleFE fe;
  MyTriangleFE &getLoopFe() { return fe; }

  MyTriangleFE feLhs;
  MyTriangleFE &getLoopFeLhs() { return feLhs; }

  MyTriangleFE feMatRhs;
  MyTriangleFE &getLoopFeMatRhs() { return feMatRhs; }

  MyTriangleFE feMatLhs;
  MyTriangleFE &getLoopFeMatLhs() { return feMatLhs; }

  NeummanForcesSurface(MoFEM::Interface &m_field)
      : mField(m_field), fe(m_field), feLhs(m_field), feMatRhs(m_field), feMatLhs(m_field) {}

  struct bCForce {
    ForceCubitBcData data;
    Range tRis;
  };
  std::map<int, bCForce> mapForce;

  struct bCPressure {
    PressureCubitBcData data;
    Range tRis;
  };
  std::map<int, bCPressure> mapPressure;

  boost::ptr_vector<MethodForForceScaling> methodsOp;
  boost::ptr_vector<MethodForAnalyticalForce> analyticalForceOp;

  using UserDataOperator =
      MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;
  using EntData = DataForcesAndSourcesCore::EntData;

  /// Operator for force element
  struct OpNeumannForce : public UserDataOperator {

    Vec F;
    bCForce &dAta;
    boost::ptr_vector<MethodForForceScaling> &methodsOp;

    bool hoGeometry;

    OpNeumannForce(const std::string field_name, Vec _F, bCForce &data,
                   boost::ptr_vector<MethodForForceScaling> &methods_op,
                   bool ho_geometry = false);

    VectorDouble Nf; //< Local force vector

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /// Operator for force element
  struct OpNeumannForceAnalytical : public UserDataOperator {

    OpNeumannForceAnalytical(
        const std::string field_name, Vec f, const Range tris,
        boost::ptr_vector<MethodForForceScaling> &methods_op,
        boost::shared_ptr<MethodForAnalyticalForce> &analytical_force_op,
        const bool ho_geometry = false);

    VectorDouble nF; //< Local force vector

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);

    Vec F;
    
  private:
    const Range tRis;
    boost::ptr_vector<MethodForForceScaling> &methodsOp;
    boost::shared_ptr<MethodForAnalyticalForce> analyticalForceOp;
    const bool hoGeometry;
  };

  /**
   * @brief Operator for pressure element
   *
   */
  struct OpNeumannPressure : public UserDataOperator {

    Vec F;
    bCPressure &dAta;
    boost::ptr_vector<MethodForForceScaling> &methodsOp;
    bool hoGeometry;

    int count;

    OpNeumannPressure(const std::string field_name, Vec _F, bCPressure &data,
                      boost::ptr_vector<MethodForForceScaling> &methods_op,
                      bool ho_geometry = false);

    VectorDouble Nf;

    /**
     * @brief Integrate pressure
     *
     * \f[
     * \begin{split}
     * \mathbf{t} &= p \mathbf{n} \\
     * \mathbf{f}^i &= \int_\mathcal{T} {\pmb\phi}^i \mathbf{t}
     * \textrm{d}\mathcal{T} \end{split}
     * \f]
     * where \f$p\f$ is pressure, \f$\mathbf{n}\f$ is normal, \f$\mathbf{t}\f$
     * is traction, and
     * \f$\mathbf{f}^i\f$ is local vector of external forces for ith base
     * function \f${\pmb\phi}^i\f$.
     *
     *
     * @param side
     * @param type
     * @param data
     * @return MoFEMErrorCode
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct DataAtIntegrationPts {
    vector<vector<VectorDouble>> tangent;
  };

  struct OpGetTangent : public UserDataOperator {

    boost::shared_ptr<DataAtIntegrationPts> dataAtIntegrationPts;
    OpGetTangent(const string field_name,
                 boost::shared_ptr<DataAtIntegrationPts> dataAtIntegrationPts)
        : UserDataOperator(field_name, UserDataOperator::OPROW),
          dataAtIntegrationPts(dataAtIntegrationPts) {}

    int ngp;
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpNeumannPressureLhs_dx_dX : public UserDataOperator {

    bCPressure &dAta;
    bool hoGeometry;

    Mat Aij;
    boost::shared_ptr<DataAtIntegrationPts> dataAtIntegrationPts;
    MatrixDouble NN;

    boost::shared_ptr<double> lambdaPtr;

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

    OpNeumannPressureLhs_dx_dX(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPts> dataAtIntegrationPts, Mat aij,
        bCPressure &data, boost::shared_ptr<double> lambda_ptr = nullptr,
        bool ho_geometry = false)
        : UserDataOperator(field_name_1, field_name_2,
                           UserDataOperator::OPROWCOL),
          dataAtIntegrationPts(dataAtIntegrationPts), Aij(aij), dAta(data),
          lambdaPtr(lambda_ptr), hoGeometry(ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };           

  struct DataAtIntegrationPtsMat : DataAtIntegrationPts {

    //vector<vector<VectorDouble>> tangent;

    boost::shared_ptr<MatrixDouble> hMat;
    boost::shared_ptr<MatrixDouble> FMat;
    boost::shared_ptr<MatrixDouble> HMat;
    boost::shared_ptr<VectorDouble> detHVec;
    boost::shared_ptr<MatrixDouble> invHMat;

    EntData* faceRowData;

    DataAtIntegrationPtsMat() {
      hMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      FMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      HMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      detHVec = boost::shared_ptr<VectorDouble>(new VectorDouble());
      invHMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());

      faceRowData = nullptr;
    }

    //Range forcesOnlyOnEntitiesRow;
    //Range forcesOnlyOnEntitiesCol;
  };

  struct OpCalculateDeformation
      : public MoFEM::VolumeElementForcesAndSourcesCoreOnSide::
            UserDataOperator {

    bool hoGeometry;
    boost::shared_ptr<DataAtIntegrationPtsMat> dataAtPts;

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);

    OpCalculateDeformation(
        const string field_name,
        boost::shared_ptr<DataAtIntegrationPtsMat> data_at_pts,
        bool ho_geometry = false)
        : MoFEM::VolumeElementForcesAndSourcesCoreOnSide::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          dataAtPts(data_at_pts), hoGeometry(ho_geometry) {
      doEdges = false;
      doQuads = false;
      doTris = false;
      doTets = false;
      doPrisms = false;
    };
  };

  struct OpNeumannPressureMaterialRhs_dX : public UserDataOperator {

    Vec F;
    bCPressure &dAta;
    boost::shared_ptr<double> lambdaPtr;
    bool hoGeometry;
    boost::shared_ptr<DataAtIntegrationPtsMat> dataAtPts;
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideFe;

    int count;

    std::string sideFeName; 

    VectorDouble nF;

    VectorInt rowIndices;

    int nbRows;           ///< number of dofs on rows
    int nbIntegrationPts; ///< number of integration points

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &row_data);
    MoFEMErrorCode iNtegrate(EntData &row_data);
    MoFEMErrorCode aSsemble(EntData &row_data);

    OpNeumannPressureMaterialRhs_dX(
        const string material_field,
        boost::shared_ptr<DataAtIntegrationPtsMat> data_at_pts,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe,
        std::string &side_fe_name, Vec f, bCPressure &data,
        boost::shared_ptr<double> lambda_ptr = nullptr,
        bool ho_geometry = false)
        : UserDataOperator(material_field, UserDataOperator::OPROW),
          dataAtPts(data_at_pts), sideFe(side_fe), sideFeName(side_fe_name),
          F(f), dAta(data), lambdaPtr(lambda_ptr), hoGeometry(ho_geometry){
            count = 0;
          };
  };

  struct OpNeumannPressureMaterialLhs_dX_dX : public UserDataOperator {


    Mat Aij;
    MatrixDouble NN;

    bCPressure &dAta;
    boost::shared_ptr<double> lambdaPtr;
    bool hoGeometry;
    boost::shared_ptr<DataAtIntegrationPtsMat> dataAtPts;
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideFe;

    std::string sideFeName; 

    VectorInt rowIndices;
    VectorInt colIndices;
/* 
    int nbRows;           ///< number of dofs on rows
    int nbCols;           ///< number if dof on column
    int nbIntegrationPts; ///< number of integration points */

    int row_nb_dofs;
    int col_nb_dofs;
    int nb_gauss_pts;

    int nb_base_fun_row;
    int nb_base_fun_col;

    bool diagonal_block;


    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);
    MoFEMErrorCode aSsemble(EntData &row_data, EntData &col_data);

    OpNeumannPressureMaterialLhs_dX_dX(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPtsMat> data_at_pts,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe,
        std::string &side_fe_name, Mat aij, bCPressure &data,
        boost::shared_ptr<double> lambda_ptr = nullptr,
        bool ho_geometry = false)
        : UserDataOperator(field_name_1, field_name_2,
                           UserDataOperator::OPROWCOL),
          dataAtPts(data_at_pts), sideFe(side_fe), sideFeName(side_fe_name),
          Aij(aij), dAta(data), lambdaPtr(lambda_ptr),
          hoGeometry(ho_geometry){
            sYmm = false; // This will make sure to loop over all entities
          };
  };

  struct OpNeumannPressureMaterialLhs_dX_dx : public UserDataOperator {

    Mat Aij;
    //MatrixDouble NN;

    bCPressure &dAta;
    boost::shared_ptr<double> lambdaPtr;
    bool hoGeometry;
    boost::shared_ptr<DataAtIntegrationPtsMat> dataAtPts;
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideFe;

    std::string sideFeName;

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

    OpNeumannPressureMaterialLhs_dX_dx(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPtsMat> data_at_pts,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe,
        std::string &side_fe_name, Mat aij, bCPressure &data,
        boost::shared_ptr<double> lambda_ptr = nullptr,
        bool ho_geometry = false)
        : UserDataOperator(field_name_1, field_name_2,
                           UserDataOperator::OPROWCOL),
          dataAtPts(data_at_pts), sideFe(side_fe), sideFeName(side_fe_name),
          Aij(aij), dAta(data), lambdaPtr(lambda_ptr), hoGeometry(ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  struct OpNeumannPressureMaterialVolOnSideLhs_dX_dx
      : public MoFEM::VolumeElementForcesAndSourcesCoreOnSide::
            UserDataOperator {

    Mat Aij;
    MatrixDouble NN;

    bCPressure &dAta;
    boost::shared_ptr<double> lambdaPtr;
    bool hoGeometry;
    boost::shared_ptr<DataAtIntegrationPtsMat> dataAtPts;

    //boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideFe;
    //std::string sideFeName;

    VectorInt rowIndices;
    VectorInt colIndices;

    int row_nb_dofs;
    int col_nb_dofs;
    int nb_gauss_pts;

    int nb_base_fun_row;
    int nb_base_fun_col;

    bool diagonal_block;

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);
    MoFEMErrorCode aSsemble(EntData &row_data, EntData &col_data);

    OpNeumannPressureMaterialVolOnSideLhs_dX_dx(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPtsMat> data_at_pts, Mat aij,
        bCPressure &data, boost::shared_ptr<double> lambda_ptr = nullptr,
        bool ho_geometry = false)
        : MoFEM::VolumeElementForcesAndSourcesCoreOnSide::UserDataOperator(
              field_name_1, field_name_2, UserDataOperator::OPROWCOL),
          dataAtPts(data_at_pts), Aij(aij), dAta(data), lambdaPtr(lambda_ptr),
          hoGeometry(ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  struct OpNeumannPressureMaterialVolOnSideLhs_dX_dX
      : public MoFEM::VolumeElementForcesAndSourcesCoreOnSide::
            UserDataOperator {

    Mat Aij;
    MatrixDouble NN;

    bCPressure &dAta;
    boost::shared_ptr<double> lambdaPtr;
    bool hoGeometry;
    boost::shared_ptr<DataAtIntegrationPtsMat> dataAtPts;

    // boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideFe;
    // std::string sideFeName;

    VectorInt rowIndices;
    VectorInt colIndices;

    int row_nb_dofs;
    int col_nb_dofs;
    int nb_gauss_pts;

    int nb_base_fun_row;
    int nb_base_fun_col;

    bool diagonal_block;

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);
    MoFEMErrorCode aSsemble(EntData &row_data, EntData &col_data);

    OpNeumannPressureMaterialVolOnSideLhs_dX_dX(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPtsMat> data_at_pts, Mat aij,
        bCPressure &data, boost::shared_ptr<double> lambda_ptr = nullptr,
        bool ho_geometry = false)
        : MoFEM::VolumeElementForcesAndSourcesCoreOnSide::UserDataOperator(
              field_name_1, field_name_2, UserDataOperator::OPROWCOL),
          dataAtPts(data_at_pts), Aij(aij), dAta(data), lambdaPtr(lambda_ptr),
          hoGeometry(ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  /// Operator for flux element
  struct OpNeumannFlux : public UserDataOperator {

    Vec F;
    bCPressure &dAta;
    boost::ptr_vector<MethodForForceScaling> &methodsOp;
    bool hoGeometry;

    OpNeumannFlux(const std::string field_name, Vec _F, bCPressure &data,
                  boost::ptr_vector<MethodForForceScaling> &methods_op,
                  bool ho_geometry);

    VectorDouble Nf;

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
      };

      /**
       * \brief Add operator to calculate forces on element
       * @param  field_name  Field name (f.e. TEMPERATURE)
       * @param  F           Right hand side vector
       * @param  ms_id       Set id (SideSet or BlockSet if block_set = true)
       * @param  ho_geometry Use higher order shape functions to define curved
       * geometry
       * @param  block_set   If tru get data from block set
       * @return             ErrorCode
       */
      MoFEMErrorCode addForce(const std::string field_name, Vec F, int ms_id,
                              bool ho_geometry = false, bool block_set = false);

      /**
       * \brief Add operator to calculate pressure on element
       * @param  field_name  Field name (f.e. TEMPERATURE)
       * @param  F           Right hand side vector
       * @param  ms_id       Set id (SideSet or BlockSet if block_set = true)
       * @param  ho_geometry Use higher order shape functions to define curved
       * geometry
       * @param  block_set   If tru get data from block set
       * @return             ErrorCode
       */
      MoFEMErrorCode addPressure(const std::string field_name, Vec F, int ms_id,
                                 bool ho_geometry = false,
                                 bool block_set = false);

      MoFEMErrorCode addPressure(
          const std::string field_name_1, const std::string field_name_2, Vec F,
          Mat aij, int ms_id, boost::shared_ptr<double> lambda_ptr = nullptr,
          bool ho_geometry = true, bool block_set = false);

      MoFEMErrorCode addPressureMaterial(
          const std::string field_name_1, const std::string field_name_2,
          std::string &side_fe_name,
          Vec F, Mat aij, int ms_id,
          boost::shared_ptr<double> lambda_ptr = nullptr,
          bool ho_geometry = true, bool block_set = false);

      /**
       * \brief Add operator to calculate pressure on element
       * @param  field_name  Field name (f.e. TEMPERATURE)
       * @param  F           Right hand side vector
       * @param  ms_id       Set id (SideSet or BlockSet if block_set = true)
       * @param  ho_geometry Use higher order shape functions to define curved
       * geometry
       * @param  block_set   If tru get data from block set
       * @return             ErrorCode
       */
      MoFEMErrorCode addLinearPressure(const std::string field_name, Vec F,
                                       int ms_id, bool ho_geometry = false);

      /// Add flux element operator (integration on face)
      MoFEMErrorCode addFlux(const std::string field_name, Vec F, int ms_id,
                             bool ho_geometry = false);

      /// \deprecated fixed spelling mistake
      DEPRECATED typedef MethodForAnalyticalForce MethodForAnaliticalForce;

      DEPRECATED typedef OpNeumannPressure OpNeumannPreassure;

      DEPRECATED typedef bCPressure
          bCPreassure; ///< \deprecated Do not use spelling mistake

      /// \deprecated function is deprecated because spelling mistake, use
      /// addPressure instead
      DEPRECATED MoFEMErrorCode addPreassure(const std::string field_name,
                                             Vec F, int ms_id,
                                             bool ho_geometry = false,
                                             bool block_set = false);
};

/** \brief Set of high-level function declaring elements and setting operators
 * to apply forces/fluxes \ingroup mofem_static_boundary_conditions
 */
struct MetaNeummanForces {

  /**
   * \brief Declare finite element
   *
   * Search cubit sidesets and blocksets with pressure bc and declare surface
   elemen

   * Block set has to have name “PRESSURE”. Can have name “PRESSURE_01” or any
   * other name with prefix. The first attribute  of block set is pressure
   * value.

   *
   * @param  m_field               Interface insurance
   * @param  field_name            Field name (f.e. DISPLACEMENT)
   * @param  mesh_nodals_positions Name of field on which ho-geometry is defined
   * @param  intersect_ptr         Pointer to range to interect meshset entities
   * @return                       Error code
   */
  static MoFEMErrorCode addNeumannBCElements(
      MoFEM::Interface &m_field, const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS",
      Range *intersect_ptr = NULL);

  /**
   * \brief Set operators to finite elements calculating right hand side vector

   * @param  m_field               Interface
   * @param  neumann_forces        Map of pointers to force/pressure elements
   * @param  F                     Right hand side vector
   * @param  field_name            Field name (f.e. DISPLACEMENT)
   * @param  mesh_nodals_positions Name of field on which ho-geometry is defined
   * @return                       Error code
   *
   */
  static MoFEMErrorCode setMomentumFluxOperators(
      MoFEM::Interface &m_field,
      boost::ptr_map<std::string, NeummanForcesSurface> &neumann_forces, Vec F,
      const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  static MoFEMErrorCode addNeumannFluxBCElements(
      MoFEM::Interface &m_field, const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  static MoFEMErrorCode setMassFluxOperators(
      MoFEM::Interface &m_field,
      boost::ptr_map<std::string, NeummanForcesSurface> &neumann_forces, Vec F,
      const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");
};

#endif //__SURFACE_PERSSURE_HPP__

/******************************************************************************
 * \defgroup mofem_static_boundary_conditions Pressure and force boundary
 *conditions \ingroup user_modules
 ******************************************************************************/
