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
struct NeumannForcesSurface {

  MoFEM::Interface &mField;

  using UserDataOperator =
      MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;
  using VolOnSideUserDataOperator =
      MoFEM::VolumeElementForcesAndSourcesCoreOnSide::UserDataOperator;
  using EntData = DataForcesAndSourcesCore::EntData;

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

  // FE for the right-hand side (spatial configuration)
  MyTriangleFE fe;
  MyTriangleFE &getLoopFe() { return fe; }

  // FE for the left-hand side (spatial configuration, ALE)
  MyTriangleFE feLhs;
  MyTriangleFE &getLoopFeLhs() { return feLhs; }

  // FE for the right-hand side (material configuration, ALE)
  MyTriangleFE feMatRhs;
  MyTriangleFE &getLoopFeMatRhs() { return feMatRhs; }

  // FE for the left-hand side (material configuration, ALE)
  MyTriangleFE feMatLhs;
  MyTriangleFE &getLoopFeMatLhs() { return feMatLhs; }

  struct DataAtIntegrationPts
      : public boost::enable_shared_from_this<DataAtIntegrationPts> {

    MatrixDouble tangent1;
    MatrixDouble tangent2;

    MatrixDouble hMat;
    MatrixDouble FMat;
    MatrixDouble HMat;
    VectorDouble detHVec;
    MatrixDouble invHMat;

    inline boost::shared_ptr<MatrixDouble> getSmallhMatPtr() {
      return boost::shared_ptr<MatrixDouble>(shared_from_this(), &hMat);
    }

    inline boost::shared_ptr<MatrixDouble> getHMatPtr() {
      return boost::shared_ptr<MatrixDouble>(shared_from_this(), &HMat);
    }

    EntData *faceRowData;

    DataAtIntegrationPts() {
      faceRowData = nullptr;
      arcLengthDof = nullptr;
    }

    Range forcesOnlyOnEntitiesRow;
    Range forcesOnlyOnEntitiesCol;

    // Pointer to arc length method DOF, used to scale pressure in LHS
    boost::shared_ptr<DofEntity> arcLengthDof;
  };

  boost::shared_ptr<DataAtIntegrationPts> dataAtIntegrationPts;

  NeumannForcesSurface(MoFEM::Interface &m_field)
      : mField(m_field), fe(m_field), feLhs(m_field), feMatRhs(m_field),
        feMatLhs(m_field) {
    dataAtIntegrationPts = boost::make_shared<DataAtIntegrationPts>();
  }

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
   * @brief RHS-operator for pressure element (spatial configuration)
   *
   */
  struct OpNeumannPressure : public UserDataOperator {

    Vec F;
    bCPressure &dAta;
    boost::ptr_vector<MethodForForceScaling> &methodsOp;
    bool hoGeometry;

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
     * where \f$p\f$ is pressure, \f$\mathbf{n}\f$ is normal,
     * \f$\mathbf{t}\f$ is traction, and
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

  /**
   * @brief Operator for computing tangent vectors
   *
   */
  struct OpGetTangent : public UserDataOperator {

    boost::shared_ptr<DataAtIntegrationPts> dataAtIntegrationPts;
    OpGetTangent(const string field_name,
                 boost::shared_ptr<DataAtIntegrationPts> dataAtIntegrationPts)
        : UserDataOperator(field_name, UserDataOperator::OPCOL),
          dataAtIntegrationPts(dataAtIntegrationPts) {}

    int ngp;
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /**
   * @brief LHS-operator for pressure element (spatial configuration)
   *
   * Computes linearisation of the spatial component with respect to
   * material coordinates.
   *
   */
  struct OpNeumannPressureLhs_dx_dX : public UserDataOperator {

    boost::shared_ptr<DataAtIntegrationPts> dataAtIntegrationPts;
    Mat Aij;
    bCPressure &dAta;
    bool hoGeometry;

    MatrixDouble NN;

    /**
     * @brief Compute left-hand side
     *
     * Computes linearisation of the spatial component with respect to
     * material coordinates.
     *
     * Virtual work of the surface pressure corresponding to a test function
     * of the spatial configuration \f$(\delta\mathbf{x})\f$:
     * \f[
     * \delta W^\text{spatial}_p(\mathbf{X}, \delta\mathbf{x}) =
     * \int\limits_\mathcal{T} p\,\mathbf{N}(\mathbf{X}) \cdot
     * \delta\mathbf{x}\, \textrm{d}\mathcal{T} =
     * \int\limits_{\mathcal{T}_{\xi}}
     *  p\left(\frac{\partial\mathbf{X}}{\partial\xi}\times\frac{\partial
     * \mathbf{X}} {\partial\eta}\right) \cdot \delta\mathbf{x}\,
     * \textrm{d}\xi\textrm{d}\eta, \f] where \f$p\f$ is pressure,
     * \f$\mathbf{N}\f$ is a normal to the face in the material configuration
     * and \f$\xi, \eta\f$ are coordinates in the parent space
     * \f$(\mathcal{T}_\xi)\f$.
     *
     * Linearisation with respect to a variation of material coordinates
     * \f$(\Delta\mathbf{X})\f$:
     *
     * \f[
     * \textrm{D} \delta W^\text{spatial}_p(\mathbf{X}, \delta\mathbf{x})
     * [\Delta\mathbf{X}] = \int\limits_{\mathcal{T}_{\xi}} p\left[
     * \frac{\partial\mathbf{X}}{\partial\xi} \cdot \left(\frac{\partial
     * \Delta \mathbf{X}}{\partial\eta}\times\delta\mathbf{x}\right)
     * -\frac{\partial\mathbf{X}}
     *  {\partial\eta} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\xi}\times \delta\mathbf{x}\right)\right]
     * \textrm{d}\xi\textrm{d}\eta \f]
     *
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

    OpNeumannPressureLhs_dx_dX(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPts> data_at_pts, Mat aij,
        bCPressure &data, bool ho_geometry = false)
        : UserDataOperator(field_name_1, field_name_2,
                           UserDataOperator::OPROWCOL),
          dataAtIntegrationPts(data_at_pts), Aij(aij), dAta(data),
          hoGeometry(ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  /**
   * @brief Operator for computing deformation gradients in side volumes
   *
   */
  struct OpCalculateDeformation : public VolOnSideUserDataOperator {

    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
    bool hoGeometry;

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);

    OpCalculateDeformation(const string field_name,
                           boost::shared_ptr<DataAtIntegrationPts> data_at_pts,
                           bool ho_geometry = false)
        : VolOnSideUserDataOperator(field_name, UserDataOperator::OPROW),
          dataAtPts(data_at_pts), hoGeometry(ho_geometry) {
      std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
      sYmm = false;
    };
  };

  /**
   * @brief RHS-operator for the pressure element (material configuration)
   *
   * Integrates pressure in the material configuration.
   *
   */
  struct OpNeumannPressureMaterialRhs_dX : public UserDataOperator {

    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideFe;
    std::string sideFeName;
    Vec F;
    bCPressure &dAta;
    bool hoGeometry;

    VectorDouble nF;

    VectorInt rowIndices;

    int nbRows;           ///< number of dofs on rows
    int nbIntegrationPts; ///< number of integration points

    /**
     * @brief Integrate pressure in the material configuration.
     *
     * Virtual work of the surface pressure corresponding to a test function
     * of the material configuration \f$(\delta\mathbf{X})\f$:
     *
     * \f[
     * \delta W^\text{material}_p(\mathbf{x}, \mathbf{X}, \delta\mathbf{X}) =
     * -\int\limits_\mathcal{T} p\left\{\mathbf{F}^{\intercal}\cdot
     * \mathbf{N}(\mathbf{X}) \right\} \cdot \delta\mathbf{X}\,
     * \textrm{d}\mathcal{T} =
     * -\int\limits_{\mathcal{T}_{\xi}} p\left\{\mathbf{F}^{\intercal}\cdot
     * \left(\frac{\partial\mathbf{X}}{\partial\xi}\times\frac{\partial
     * \mathbf{X}} {\partial\eta}\right) \right\} \cdot \delta\mathbf{X}\,
     * \textrm{d}\xi\textrm{d}\eta
     *  \f]
     *
     * where \f$p\f$ is pressure, \f$\mathbf{N}\f$ is a normal to the face
     * in the material configuration, \f$\xi, \eta\f$ are coordinates in the
     * parent space
     * \f$(\mathcal{T}_\xi)\f$ and \f$\mathbf{F}\f$ is the deformation gradient:
     *
     * \f[
     * \mathbf{F} = \mathbf{h}(\mathbf{x})\,\mathbf{H}(\mathbf{X})^{-1} =
     * \frac{\partial\mathbf{x}}{\partial\mathbf{\chi}}
     * \frac{\partial\mathbf{\chi}}{\partial\mathbf{X}}
     * \f]
     *
     * where \f$\mathbf{h}\f$ and \f$\mathbf{H}\f$ are the gradients of the
     * spatial and material maps, respectively, and \f$\mathbf{\chi}\f$ are
     * the reference coordinates.
     *
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &row_data);
    MoFEMErrorCode iNtegrate(EntData &row_data);
    MoFEMErrorCode aSsemble(EntData &row_data);

    OpNeumannPressureMaterialRhs_dX(
        const string material_field,
        boost::shared_ptr<DataAtIntegrationPts> data_at_pts,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe,
        std::string &side_fe_name, Vec f, bCPressure &data,
        bool ho_geometry = false)
        : UserDataOperator(material_field, UserDataOperator::OPROW),
          dataAtPts(data_at_pts), sideFe(side_fe), sideFeName(side_fe_name),
          F(f), dAta(data), hoGeometry(ho_geometry){};
  };

  /**
   * @brief Base class for LHS-operators for pressure element (material
   * configuration)
   *
   * Linearisation of the material component with respect to
   * spatial and material coordinates consists of three parts, computed
   * by operators working on the face and on the side volume:
   *
   * \f[
   * \textrm{D} \delta W^\text{material}_p(\mathbf{x}, \mathbf{X},
   * \delta\mathbf{x})
   * [\Delta\mathbf{x}, \Delta\mathbf{X}] = \textrm{D} \delta
   * W^\text{(face)}_p(\mathbf{x}, \mathbf{X}, \delta\mathbf{x})
   * [\Delta\mathbf{X}] + \textrm{D} \delta
   * W^\text{(side volume)}_p(\mathbf{x}, \mathbf{X}, \delta\mathbf{x})
   * [\Delta\mathbf{x}] + \textrm{D} \delta W^\text{(side volume)}_p
   * (\mathbf{x}, \mathbf{X}, \delta\mathbf{x}) [\Delta\mathbf{X}]
   * \f]
   *
   */
  struct OpNeumannPressureMaterialLhs : public UserDataOperator {

    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideFe;
    std::string sideFeName;
    Mat Aij;
    bCPressure &dAta;
    bool hoGeometry;

    MatrixDouble NN;
    VectorInt rowIndices;
    VectorInt colIndices;

    int row_nb_dofs;
    int col_nb_dofs;
    int nb_gauss_pts;

    int nb_base_fun_row;
    int nb_base_fun_col;

    bool diagonal_block;

    virtual MoFEMErrorCode doWork(int row_side, int col_side,
                                  EntityType row_type, EntityType col_type,
                                  DataForcesAndSourcesCore::EntData &row_data,
                                  DataForcesAndSourcesCore::EntData &col_data) {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }

    virtual MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data) {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode aSsemble(EntData &row_data, EntData &col_data);

    OpNeumannPressureMaterialLhs(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPts> data_at_pts,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe,
        std::string &side_fe_name, Mat aij, bCPressure &data,
        bool ho_geometry = false)
        : UserDataOperator(field_name_1, field_name_2,
                           UserDataOperator::OPROWCOL),
          dataAtPts(data_at_pts), sideFe(side_fe), sideFeName(side_fe_name),
          Aij(aij), dAta(data), hoGeometry(ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    }
  };

  /**
   * @brief LHS-operator for the pressure element (material configuration)
   *
   * Computes linearisation of the material component with respect to
   * material coordinates (also triggers a loop over operators
   * from the side volume).
   *
   */
  struct OpNeumannPressureMaterialLhs_dX_dX
      : public OpNeumannPressureMaterialLhs {

    /**
     * Integrates a contribution to the left-hand side and triggers a loop
     * over side volume operators.
     *
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

    /**
     * @brief Compute part of the left-hand side
     *
     * Computes the linearisation of the material component
     * with respect to a variation of material coordinates
     * \f$(\Delta\mathbf{X})\f$:
     *
     * \f[
     * \textrm{D} \delta W^\text{(face)}_p(\mathbf{x}, \mathbf{X},
     * \delta\mathbf{x})
     * [\Delta\mathbf{X}] = -\int\limits_{\mathcal{T}_{\xi}} p \,
     * \mathbf{F}^{\intercal}\cdot \left[ \frac{\partial\mathbf{X}}
     * {\partial\xi} \cdot \left(\frac{\partial\Delta
     *  \mathbf{X}}{\partial\eta}\times\delta\mathbf{x}\right)
     * -\frac{\partial\mathbf{X}}
     *  {\partial\eta} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\xi}\times \delta\mathbf{x}\right)\right]
     * \textrm{d}\xi\textrm{d}\eta
     * \f]
     *
     */
    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);

    OpNeumannPressureMaterialLhs_dX_dX(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPts> data_at_pts,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe,
        std::string &side_fe_name, Mat aij, bCPressure &data,
        bool ho_geometry = false)
        : OpNeumannPressureMaterialLhs(field_name_1, field_name_2, data_at_pts,
                                       side_fe, side_fe_name, aij, data,
                                       ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  /**
   * @brief LHS-operator for the pressure element (material configuration)
   *
   * Triggers loop over operators from the side volume
   *
   */
  struct OpNeumannPressureMaterialLhs_dX_dx
      : public OpNeumannPressureMaterialLhs {

    /*
     * Triggers loop over operators from the side volume
     *
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

    OpNeumannPressureMaterialLhs_dX_dx(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPts> data_at_pts,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe,
        std::string &side_fe_name, Mat aij, bCPressure &data,
        bool ho_geometry = false)
        : OpNeumannPressureMaterialLhs(field_name_1, field_name_2, data_at_pts,
                                       side_fe, side_fe_name, aij, data,
                                       ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  /**
   * @brief Base class for LHS-operators (material) on side volumes
   *
   */
  struct OpNeumannPressureMaterialVolOnSideLhs
      : public VolOnSideUserDataOperator {

    MatrixDouble NN;

    boost::shared_ptr<DataAtIntegrationPts> dataAtPts;
    Mat Aij;
    bCPressure &dAta;
    bool hoGeometry;

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
    virtual MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data) {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }
    MoFEMErrorCode aSsemble(EntData &row_data, EntData &col_data);

    OpNeumannPressureMaterialVolOnSideLhs(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPts> data_at_pts, Mat aij,
        bCPressure &data, bool ho_geometry = false)
        : VolOnSideUserDataOperator(field_name_1, field_name_2,
                                    UserDataOperator::OPROWCOL),
          dataAtPts(data_at_pts), Aij(aij), dAta(data),
          hoGeometry(ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    }
  };

  /**
   * @brief LHS-operator (material configuration) on the side volume
   *
   * Computes the linearisation of the material component
   * with respect to a variation of spatial coordinates on the side volume.
   */
  struct OpNeumannPressureMaterialVolOnSideLhs_dX_dx
      : public OpNeumannPressureMaterialVolOnSideLhs {

    /**
     * @brief Integrates over a face contribution from a side volume
     *
     * Computes linearisation of the material component
     * with respect to a variation of spatial coordinates:
     *
     * \f[
     * \textrm{D} \delta W^\text{(side volume)}_p(\mathbf{x}, \mathbf{X},
     * \delta\mathbf{x})
     * [\Delta\mathbf{x}] = -\int\limits_{\mathcal{T}_{\xi}} p
     * \left\{\left[
     * \frac{\partial\Delta\mathbf{x}}{\partial\mathbf{\chi}}\,\mathbf{H}^{-1}
     * \right]^{\intercal}\cdot\left(\frac{\partial\mathbf{X}}{\partial\xi}
     * \times\frac{\partial\mathbf{X}}{\partial\eta}\right)\right\}
     * \cdot \delta\mathbf{X}\, \textrm{d}\xi\textrm{d}\eta
     * \f]
     *
     */
    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);

    OpNeumannPressureMaterialVolOnSideLhs_dX_dx(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPts> data_at_pts, Mat aij,
        bCPressure &data, bool ho_geometry = false)
        : OpNeumannPressureMaterialVolOnSideLhs(
              field_name_1, field_name_2, data_at_pts, aij, data, ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  /**
   * @brief LHS-operator (material configuration) on the side volume
   *
   * Computes the linearisation of the material component
   * with respect to a variation of material coordinates on the side volume.
   *
   */
  struct OpNeumannPressureMaterialVolOnSideLhs_dX_dX
      : public OpNeumannPressureMaterialVolOnSideLhs {

    /**
     * @brief Integrates over a face contribution from a side volume
     *
     * Computes linearisation of the material component
     * with respect to a variation of material coordinates:
     *
     * \f[
     * \textrm{D} \delta W^\text{(side volume)}_p(\mathbf{x}, \mathbf{X},
     * \delta\mathbf{x})
     * [\Delta\mathbf{X}] = \int\limits_{\mathcal{T}_{\xi}} p
     * \left\{\left[
     * \mathbf{h}\,\mathbf{H}^{-1}\,\frac{\partial\Delta\mathbf{X}}
     * {\partial\mathbf{\chi}}\,\mathbf{H}^{-1}
     * \right]^{\intercal}\cdot\left(\frac{\partial\mathbf{X}}{\partial\xi}
     * \times\frac{\partial\mathbf{X}}{\partial\eta}\right)\right\}
     * \cdot \delta\mathbf{X}\, \textrm{d}\xi\textrm{d}\eta
     * \f]
     */
    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);

    OpNeumannPressureMaterialVolOnSideLhs_dX_dX(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPts> data_at_pts, Mat aij,
        bCPressure &data, bool ho_geometry = false)
        : OpNeumannPressureMaterialVolOnSideLhs(
              field_name_1, field_name_2, data_at_pts, aij, data, ho_geometry) {
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
   * @param  block_set   If true get data from block set
   * @return             ErrorCode
   */
  MoFEMErrorCode addPressure(const std::string field_name, Vec F, int ms_id,
                             bool ho_geometry = false, bool block_set = false);

  /**
   * \brief Add operator to calculate pressure on element (in ALE)
   * @param  field_name_1  Field name for spatial positions
   * @param  field_name_2  Field name for material positions
   * @param  side_fe_name  Name of the element in the side volume
   * @param  F             Right hand side vector
   * @param  aij           Tangent matrix
   * @param  ms_id         Set id (SideSet or BlockSet if block_set = true)
   * @param  ho_geometry   Use higher order shape functions to define curved
   * geometry
   * @param  block_set   If true get data from block set
   * @return             ErrorCode
   */
  MoFEMErrorCode addPressureAle(const std::string field_name_1,
                                const std::string field_name_2,
                                std::string side_fe_name, Vec F, Mat aij,
                                int ms_id, bool ho_geometry = false,
                                bool block_set = false);

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
  DEPRECATED MoFEMErrorCode addPreassure(const std::string field_name, Vec F,
                                         int ms_id, bool ho_geometry = false,
                                         bool block_set = false);
};

/** \brief Set of high-level function declaring elements and setting operators
 * to apply forces/fluxes \ingroup mofem_static_boundary_conditions
 */
struct MetaNeumannForces {

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
      boost::ptr_map<std::string, NeumannForcesSurface> &neumann_forces, Vec F,
      const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  static MoFEMErrorCode addNeumannFluxBCElements(
      MoFEM::Interface &m_field, const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  static MoFEMErrorCode setMassFluxOperators(
      MoFEM::Interface &m_field,
      boost::ptr_map<std::string, NeumannForcesSurface> &neumann_forces, Vec F,
      const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");
};

/**
 * @deprecated Do not use that name it has spelling mistake
 */
DEPRECATED typedef MetaNeumannForces MetaNeummanForces;

/**
 * @deprecated Do not use that name it has spelling mistake
 */
DEPRECATED typedef NeumannForcesSurface NeummanForcesSurface;

#endif //__SURFACE_PERSSURE_HPP__

/**
 * \defgroup mofem_static_boundary_conditions Pressure and force boundary
 * conditions
 *
 * \ingroup user_modules
 **/
