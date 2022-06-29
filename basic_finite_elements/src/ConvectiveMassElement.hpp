/** \file ConvectiveMassElement.hpp
 * \brief Operators and data structures for mass and convective mass element
 * \ingroup convective_mass_elem
 *
 */

/* Implementation of convective mass element
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

#ifndef __CONVECTIVE_MASS_ELEMENT_HPP
#define __CONVECTIVE_MASS_ELEMENT_HPP

#ifndef WITH_ADOL_C
#error "MoFEM need to be compiled with ADOL-C"
#endif

/** \brief structure grouping operators and data used for calculation of mass
 * (convective) element \ingroup convective_mass_elem \ingroup
 * nonlinear_elastic_elem
 *
 * In order to assemble matrices and right hand vectors, the loops over
 * elements, entities over that elements and finally loop over integration
 * points are executed.
 *
 * Following implementation separate those three celeries of loops and to each
 * loop attach operator.
 *
 */
struct ConvectiveMassElement {

  /// \brief  definition of volume element
  struct MyVolumeFE : public VolumeElementForcesAndSourcesCore {

    Mat A;
    Vec F;

    MyVolumeFE(MoFEM::Interface &m_field);

    /** \brief it is used to calculate nb. of Gauss integration points
     *
     * for more details pleas look
     *   Reference:
     *
     * Albert Nijenhuis, Herbert Wilf,
     * Combinatorial Algorithms for Computers and Calculators,
     * Second Edition,
     * Academic Press, 1978,
     * ISBN: 0-12-519260-6,
     * LC: QA164.N54.
     *
     * More details about algorithm
     * http://people.sc.fsu.edu/~jburkardt/cpp_src/gm_rule/gm_rule.html
     **/
    int getRule(int order);

    SmartPetscObj<Vec> V;
    double eNergy;

    MoFEMErrorCode preProcess();
    MoFEMErrorCode postProcess();
  };

  MyVolumeFE feMassRhs; ///< calculate right hand side for tetrahedral elements
  MyVolumeFE &getLoopFeMassRhs() {
    return feMassRhs;
  }                     ///< get rhs volume element
  MyVolumeFE feMassLhs; ///< calculate left hand side for tetrahedral
                        ///< elements,i.e. mass element
  MyVolumeFE &getLoopFeMassLhs() {
    return feMassLhs;
  }                        ///< get lhs volume element
  MyVolumeFE feMassAuxLhs; ///< calculate left hand side for tetrahedral
                           ///< elements for Kuu shell matrix
  MyVolumeFE &getLoopFeMassAuxLhs() {
    return feMassAuxLhs;
  } ///< get lhs volume element for Kuu shell matrix

  MyVolumeFE feVelRhs; ///< calculate right hand side for tetrahedral elements
  MyVolumeFE &getLoopFeVelRhs() { return feVelRhs; } ///< get rhs volume element
  MyVolumeFE feVelLhs; ///< calculate left hand side for tetrahedral elements
  MyVolumeFE &getLoopFeVelLhs() { return feVelLhs; } ///< get lhs volume element

  MyVolumeFE feTRhs; ///< calculate right hand side for tetrahedral elements
  MyVolumeFE &getLoopFeTRhs() { return feTRhs; } ///< get rhs volume element
  MyVolumeFE feTLhs; ///< calculate left hand side for tetrahedral elements
  MyVolumeFE &getLoopFeTLhs() { return feTLhs; } ///< get lhs volume element

  MyVolumeFE feEnergy; ///< calculate kinetic energy
  MyVolumeFE &getLoopFeEnergy() {
    return feEnergy;
  } ///< get kinetic energy element

  MoFEMErrorCode addHOOpsVol() {
    MoFEMFunctionBegin;
    auto add_ops = [&](auto &fe) {
      return MoFEM::addHOOpsVol("MESH_NODE_POSITIONS", fe, true, false, false,
                             false);
    };
    CHKERR add_ops(feMassRhs);
    CHKERR add_ops(feMassLhs);
    CHKERR add_ops(feMassAuxLhs);
    CHKERR add_ops(feVelRhs);
    CHKERR add_ops(feTRhs);
    CHKERR add_ops(feTLhs);
    CHKERR add_ops(feEnergy);
    MoFEMFunctionReturn(0);
  }

  MoFEM::Interface &mField;
  short int tAg;

  ConvectiveMassElement(MoFEM::Interface &m_field, short int tag);

  /** \brief data for calculation inertia forces
   * \ingroup user_modules
   */
  struct BlockData {
    double rho0;     ///< reference density
    VectorDouble a0; ///< constant acceleration
    Range tEts;      ///< elements in block set
  };
  std::map<int, BlockData>
      setOfBlocks; ///< maps block set id with appropriate BlockData

  /** \brief common data used by volume elements
   * \ingroup user_modules
   */
  struct CommonData {

    bool lInear;
    bool staticOnly;
    CommonData() : lInear(false), staticOnly(false) {}

    std::map<std::string, std::vector<VectorDouble>> dataAtGaussPts;
    std::map<std::string, std::vector<MatrixDouble>> gradAtGaussPts;
    string spatialPositions;
    string meshPositions;
    string spatialVelocities;
    std::vector<VectorDouble> valVel;
    std::vector<std::vector<double *>> jacVelRowPtr;
    std::vector<MatrixDouble> jacVel;
    std::vector<VectorDouble> valMass;
    std::vector<std::vector<double *>> jacMassRowPtr;
    std::vector<MatrixDouble> jacMass;
    std::vector<VectorDouble> valT;
    std::vector<std::vector<double *>> jacTRowPtr;
    std::vector<MatrixDouble> jacT;
  };
  CommonData commonData;

  boost::ptr_vector<MethodForForceScaling> methodsOp;

  struct OpGetDataAtGaussPts
      : public VolumeElementForcesAndSourcesCore::UserDataOperator {

    std::vector<VectorDouble> &valuesAtGaussPts;
    std::vector<MatrixDouble> &gradientAtGaussPts;
    const EntityType zeroAtType;

    OpGetDataAtGaussPts(const std::string field_name,
                        std::vector<VectorDouble> &values_at_gauss_pts,
                        std::vector<MatrixDouble> &gardient_at_gauss_pts);

    /** \brief operator calculating deformation gradient
     *
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data);
  };

  struct OpGetCommonDataAtGaussPts : public OpGetDataAtGaussPts {
    OpGetCommonDataAtGaussPts(const std::string field_name,
                              CommonData &common_data);
  };

  struct CommonFunctions {};

  struct OpMassJacobian
      : public VolumeElementForcesAndSourcesCore::UserDataOperator,
        CommonFunctions {

    BlockData &dAta;
    CommonData &commonData;
    int tAg;
    bool jAcobian;
    bool &lInear;
    bool fieldDisp;

    boost::ptr_vector<MethodForForceScaling> &methodsOp;

    OpMassJacobian(const std::string field_name, BlockData &data,
                   CommonData &common_data,
                   boost::ptr_vector<MethodForForceScaling> &methods_op,
                   int tag, bool linear = false);

   FTensor::Index<'i', 3> i;
   FTensor::Index<'j', 3> j;
   FTensor::Index<'k', 3> k;

   VectorBoundedArray<adouble, 3> a, dot_W, dp_dt, a_res;
   MatrixBoundedArray<adouble, 9> h, H, invH, F, g, G;
   std::vector<double> active;

   MoFEMErrorCode doWork(int row_side, EntityType row_type,
                         EntitiesFieldData::EntData &row_data);
  };

  struct OpMassRhs : public VolumeElementForcesAndSourcesCore::UserDataOperator,
                     CommonFunctions {

    BlockData &dAta;
    CommonData &commonData;

    OpMassRhs(const std::string field_name, BlockData &data,
              CommonData &common_data);

    VectorDouble nf;

    MoFEMErrorCode doWork(int row_side, EntityType row_type,
                          EntitiesFieldData::EntData &row_data);
  };

  struct OpMassLhs_dM_dv
      : public VolumeElementForcesAndSourcesCore::UserDataOperator,
        CommonFunctions {

    BlockData &dAta;
    CommonData &commonData;
    Range forcesOnlyOnEntities;

    OpMassLhs_dM_dv(const std::string vel_field, const std::string field_name,
                    BlockData &data, CommonData &common_data,
                    Range *forcesonlyonentities_ptr = NULL);

    MatrixDouble k, jac;

    virtual MoFEMErrorCode getJac(EntitiesFieldData::EntData &col_data,
                                  int gg);

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          EntitiesFieldData::EntData &row_data,
                          EntitiesFieldData::EntData &col_data);
  };

  struct OpMassLhs_dM_dx : public OpMassLhs_dM_dv {

    OpMassLhs_dM_dx(const std::string field_name, const std::string col_field,
                    BlockData &data, CommonData &common_data);

    MoFEMErrorCode getJac(EntitiesFieldData::EntData &col_data, int gg);
  };

  struct OpMassLhs_dM_dX : public OpMassLhs_dM_dv {

    OpMassLhs_dM_dX(const std::string field_name, const std::string col_field,
                    BlockData &data, CommonData &common_data);

    MoFEMErrorCode getJac(EntitiesFieldData::EntData &col_data, int gg);
  };

  struct OpEnergy : public VolumeElementForcesAndSourcesCore::UserDataOperator,
                    CommonFunctions {

    BlockData &dAta;
    CommonData &commonData;
    SmartPetscObj<Vec> V;
    bool &lInear;

    OpEnergy(const std::string field_name, BlockData &data,
             CommonData &common_data, SmartPetscObj<Vec> v);

    MatrixDouble3by3 h, H, invH, F;
    VectorDouble3 v;

    MoFEMErrorCode doWork(int row_side, EntityType row_type,
                          EntitiesFieldData::EntData &row_data);
  };

  struct OpVelocityJacobian
      : public VolumeElementForcesAndSourcesCore::UserDataOperator,
        CommonFunctions {

    BlockData &dAta;
    CommonData &commonData;
    int tAg;
    bool jAcobian, fieldDisp;

    OpVelocityJacobian(const std::string field_name, BlockData &data,
                       CommonData &common_data, int tag, bool jacobian = true);

    VectorBoundedArray<adouble, 3> a_res, v, dot_w, dot_W, dot_u;
    MatrixBoundedArray<adouble, 9> h, H, invH, F;
    adouble detH;

    std::vector<double> active;

    MoFEMErrorCode doWork(int row_side, EntityType row_type,
                          EntitiesFieldData::EntData &row_data);
  };

  struct OpVelocityRhs
      : public VolumeElementForcesAndSourcesCore::UserDataOperator,
        CommonFunctions {

    BlockData &dAta;
    CommonData &commonData;

    OpVelocityRhs(const std::string field_name, BlockData &data,
                  CommonData &common_data);

    VectorDouble nf;

    MoFEMErrorCode doWork(int row_side, EntityType row_type,
                          EntitiesFieldData::EntData &row_data);
  };

  struct OpVelocityLhs_dV_dv : public OpMassLhs_dM_dv {

    OpVelocityLhs_dV_dv(const std::string vel_field,
                        const std::string field_name, BlockData &data,
                        CommonData &common_data);

    virtual MoFEMErrorCode getJac(EntitiesFieldData::EntData &col_data,
                                  int gg);
  };

  struct OpVelocityLhs_dV_dx : public OpVelocityLhs_dV_dv {

    OpVelocityLhs_dV_dx(const std::string vel_field,
                        const std::string field_name, BlockData &data,
                        CommonData &common_data);

    virtual MoFEMErrorCode getJac(EntitiesFieldData::EntData &col_data,
                                  int gg);
  };

  struct OpVelocityLhs_dV_dX : public OpVelocityLhs_dV_dv {

    OpVelocityLhs_dV_dX(const std::string vel_field,
                        const std::string field_name, BlockData &data,
                        CommonData &common_data);

    virtual MoFEMErrorCode getJac(EntitiesFieldData::EntData &col_data,
                                  int gg);
  };

  struct OpEshelbyDynamicMaterialMomentumJacobian
      : public VolumeElementForcesAndSourcesCore::UserDataOperator,
        CommonFunctions {

    BlockData &dAta;
    CommonData &commonData;
    int tAg;
    bool jAcobian;
    bool fieldDisp;

    OpEshelbyDynamicMaterialMomentumJacobian(const std::string field_name,
                                             BlockData &data,
                                             CommonData &common_data, int tag,
                                             bool jacobian = true);

    VectorBoundedArray<adouble, 3> a, v, a_T;
    MatrixBoundedArray<adouble, 9> g, H, invH, h, F, G;
    VectorDouble active;

    MoFEMErrorCode doWork(int row_side, EntityType row_type,
                          EntitiesFieldData::EntData &row_data);
  };

  struct OpEshelbyDynamicMaterialMomentumRhs
      : public VolumeElementForcesAndSourcesCore::UserDataOperator,
        CommonFunctions {

    BlockData &dAta;
    CommonData &commonData;
    Range forcesOnlyOnEntities;

    OpEshelbyDynamicMaterialMomentumRhs(const std::string field_name,
                                        BlockData &data,
                                        CommonData &common_data,
                                        Range *forcesonlyonentities_ptr);

    VectorDouble nf;

    MoFEMErrorCode doWork(int row_side, EntityType row_type,
                          EntitiesFieldData::EntData &row_data);
  };

  struct OpEshelbyDynamicMaterialMomentumLhs_dv : public OpMassLhs_dM_dv {

    OpEshelbyDynamicMaterialMomentumLhs_dv(const std::string vel_field,
                                           const std::string field_name,
                                           BlockData &data,
                                           CommonData &common_data,
                                           Range *forcesonlyonentities_ptr);

    virtual MoFEMErrorCode getJac(EntitiesFieldData::EntData &col_data,
                                  int gg);
  };

  struct OpEshelbyDynamicMaterialMomentumLhs_dx
      : public OpEshelbyDynamicMaterialMomentumLhs_dv {

    OpEshelbyDynamicMaterialMomentumLhs_dx(const std::string vel_field,
                                           const std::string field_name,
                                           BlockData &data,
                                           CommonData &common_data,
                                           Range *forcesonlyonentities_ptr);

    virtual MoFEMErrorCode getJac(EntitiesFieldData::EntData &col_data,
                                  int gg);
  };

  struct OpEshelbyDynamicMaterialMomentumLhs_dX
      : public OpEshelbyDynamicMaterialMomentumLhs_dv {

    OpEshelbyDynamicMaterialMomentumLhs_dX(const std::string vel_field,
                                           const std::string field_name,
                                           BlockData &data,
                                           CommonData &common_data,
                                           Range *forcesonlyonentities_ptr);

    virtual MoFEMErrorCode getJac(EntitiesFieldData::EntData &col_data,
                                  int gg);
  };

  /**
   * @brief Set fields DOT_
   *
   * \note This is old solution, to keep rates calculate by TS as a fields. This
   * is not memory efficient solution.
   *
   */
  struct UpdateAndControl : public FEMethod {

    MoFEM::Interface &mField;
    TS tS;
    const std::string velocityField;
    const std::string spatialPositionField;

    int jacobianLag;
    UpdateAndControl(MoFEM::Interface &m_field, TS _ts,
                     const std::string velocity_field,
                     const std::string spatial_position_field);

    /**
     * @brief Scatter values from t_u_dt on the fields
     * 
     * @return MoFEMErrorCode 
     */
    MoFEMErrorCode preProcess();

    // This is empty fun cions does nothing
    MoFEMErrorCode postProcess();
  };

  MoFEMErrorCode setBlocks();
  
  static MoFEMErrorCode
  setBlocks(MoFEM::Interface &m_field,
            boost::shared_ptr<map<int, BlockData>> &block_sets_ptr);

  MoFEMErrorCode addConvectiveMassElement(
      string element_name, string velocity_field_name,
      string spatial_position_field_name,
      string material_position_field_name = "MESH_NODE_POSITIONS",
      bool ale = false, BitRefLevel bit = BitRefLevel());

  MoFEMErrorCode addVelocityElement(
      string element_name, string velocity_field_name,
      string spatial_position_field_name,
      string material_position_field_name = "MESH_NODE_POSITIONS",
      bool ale = false, BitRefLevel bit = BitRefLevel());

  MoFEMErrorCode addEshelbyDynamicMaterialMomentum(
      string element_name, string velocity_field_name,
      string spatial_position_field_name,
      string material_position_field_name = "MESH_NODE_POSITIONS",
      bool ale = false, BitRefLevel bit = BitRefLevel(),
      Range *intersected = NULL);

  MoFEMErrorCode setConvectiveMassOperators(
      string velocity_field_name, string spatial_position_field_name,
      string material_position_field_name = "MESH_NODE_POSITIONS",
      bool ale = false, bool linear = false);

  MoFEMErrorCode setVelocityOperators(
      string velocity_field_name, string spatial_position_field_name,
      string material_position_field_name = "MESH_NODE_POSITIONS",
      bool ale = false);

  MoFEMErrorCode setKinematicEshelbyOperators(
      string velocity_field_name, string spatial_position_field_name,
      string material_position_field_name = "MESH_NODE_POSITIONS",
      Range *forces_on_entities_ptr = NULL);

  MoFEMErrorCode setShellMatrixMassOperators(
      string velocity_field_name, string spatial_position_field_name,
      string material_position_field_name = "MESH_NODE_POSITIONS",
      bool linear = false);

  struct MatShellCtx {

    Mat K, M;
    VecScatter scatterU, scatterV;
    double ts_a; //,scale;

    bool iNitialized;
    MatShellCtx();
    virtual ~MatShellCtx();

    Mat barK;
    Vec u, v, Ku, Mv;
    MoFEMErrorCode iNit();

    MoFEMErrorCode dEstroy();

    friend MoFEMErrorCode MultOpA(Mat A, Vec x, Vec f);
    friend MoFEMErrorCode ZeroEntriesOp(Mat A);
  };

  /** \brief Mult operator for shell matrix
    *
    * \f[
    \left[
    \begin{array}{cc}
    \mathbf{M} & \mathbf{K} \\
    \mathbf{I} & -\mathbf{I}a
    \end{array}
    \right]
    \left[
    \begin{array}{c}
    \mathbf{v} \\
    \mathbf{u}
    \end{array}
    \right] =
    \left[
    \begin{array}{c}
    \mathbf{r}_u \\
    \mathbf{r}_v
    \end{array}
    \right]
    * \f]
    *
    */
  static MoFEMErrorCode MultOpA(Mat A, Vec x, Vec f) {
    MoFEMFunctionBeginHot;

    void *void_ctx;
    CHKERR MatShellGetContext(A, &void_ctx);
    MatShellCtx *ctx = (MatShellCtx *)void_ctx;
    if (!ctx->iNitialized) {
      CHKERR ctx->iNit();
    }
    CHKERR VecZeroEntries(f);
    // Mult Ku
    CHKERR VecScatterBegin(ctx->scatterU, x, ctx->u, INSERT_VALUES,
                           SCATTER_FORWARD);
    CHKERR VecScatterEnd(ctx->scatterU, x, ctx->u, INSERT_VALUES,
                         SCATTER_FORWARD);
    CHKERR MatMult(ctx->K, ctx->u, ctx->Ku);
    CHKERR VecScatterBegin(ctx->scatterU, ctx->Ku, f, INSERT_VALUES,
                           SCATTER_REVERSE);
    CHKERR VecScatterEnd(ctx->scatterU, ctx->Ku, f, INSERT_VALUES,
                         SCATTER_REVERSE);
    // Mult Mv
    CHKERR VecScatterBegin(ctx->scatterV, x, ctx->v, INSERT_VALUES,
                           SCATTER_FORWARD);
    CHKERR VecScatterEnd(ctx->scatterV, x, ctx->v, INSERT_VALUES,
                         SCATTER_FORWARD);
    CHKERR MatMult(ctx->M, ctx->v, ctx->Mv);
    CHKERR VecScatterBegin(ctx->scatterU, ctx->Mv, f, ADD_VALUES,
                           SCATTER_REVERSE);
    CHKERR VecScatterEnd(ctx->scatterU, ctx->Mv, f, ADD_VALUES,
                         SCATTER_REVERSE);
    // Velocities
    CHKERR VecAXPY(ctx->v, -ctx->ts_a, ctx->u);
    // CHKERR VecScale(ctx->v,ctx->scale);
    CHKERR VecScatterBegin(ctx->scatterV, ctx->v, f, INSERT_VALUES,
                           SCATTER_REVERSE);
    CHKERR VecScatterEnd(ctx->scatterV, ctx->v, f, INSERT_VALUES,
                         SCATTER_REVERSE);
    // Assemble
    CHKERR VecAssemblyBegin(f);
    CHKERR VecAssemblyEnd(f);
    MoFEMFunctionReturnHot(0);
  }

  static MoFEMErrorCode ZeroEntriesOp(Mat A) {
    MoFEMFunctionBeginHot;

    void *void_ctx;
    CHKERR MatShellGetContext(A, &void_ctx);
    MatShellCtx *ctx = (MatShellCtx *)void_ctx;
    CHKERR MatZeroEntries(ctx->K);
    CHKERR MatZeroEntries(ctx->M);
    MoFEMFunctionReturnHot(0);
  }

  struct PCShellCtx {

    Mat shellMat;
    bool initPC; ///< check if PC is initialized

    PCShellCtx(Mat shell_mat) : shellMat(shell_mat), initPC(false) {}

    PC pC;

    MoFEMErrorCode iNit();

    MoFEMErrorCode dEstroy();

    friend MoFEMErrorCode PCShellSetUpOp(PC pc);
    friend MoFEMErrorCode PCShellDestroy(PC pc);
    friend MoFEMErrorCode PCShellApplyOp(PC pc, Vec f, Vec x);
  };

  static MoFEMErrorCode PCShellSetUpOp(PC pc) {
    MoFEMFunctionBeginHot;

    void *void_ctx;
    CHKERR PCShellGetContext(pc, &void_ctx);
    PCShellCtx *ctx = (PCShellCtx *)void_ctx;
    CHKERR ctx->iNit();
    MatShellCtx *shell_mat_ctx;
    CHKERR MatShellGetContext(ctx->shellMat, &shell_mat_ctx);
    CHKERR PCSetFromOptions(ctx->pC);
    CHKERR PCSetOperators(ctx->pC, shell_mat_ctx->barK, shell_mat_ctx->barK);
    CHKERR PCSetUp(ctx->pC);
    MoFEMFunctionReturnHot(0);
  }

  static MoFEMErrorCode PCShellDestroy(PC pc) {
    MoFEMFunctionBeginHot;

    void *void_ctx;
    CHKERR PCShellGetContext(pc, &void_ctx);
    PCShellCtx *ctx = (PCShellCtx *)void_ctx;
    CHKERR ctx->dEstroy();
    MoFEMFunctionReturnHot(0);
  }

  /** \brief apply pre-conditioner for shell matrix
    *
    * \f[
    \left[
    \begin{array}{cc}
    \mathbf{M} & \mathbf{K} \\
    \mathbf{I} & -\mathbf{I}a
    \end{array}
    \right]
    \left[
    \begin{array}{c}
    \mathbf{v} \\
    \mathbf{u}
    \end{array}
    \right] =
    \left[
    \begin{array}{c}
    \mathbf{r}_u \\
    \mathbf{r}_v
    \end{array}
    \right]
    * \f]
    *
    * where \f$\mathbf{v} = \mathbf{r}_v + a\mathbf{u}\f$ and
    \f$\mathbf{u}=(a\mathbf{M}+\mathbf{K})^{-1}(\mathbf{r}_u -
    \mathbf{M}\mathbf{r}_v\f$.
    *
    */
  static MoFEMErrorCode PCShellApplyOp(PC pc, Vec f, Vec x) {
    MoFEMFunctionBeginHot;

    void *void_ctx;
    CHKERR PCShellGetContext(pc, &void_ctx);
    PCShellCtx *ctx = (PCShellCtx *)void_ctx;
    MatShellCtx *shell_mat_ctx;
    CHKERR MatShellGetContext(ctx->shellMat, &shell_mat_ctx);
    // forward
    CHKERR VecScatterBegin(shell_mat_ctx->scatterU, f, shell_mat_ctx->Ku,
                           INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecScatterEnd(shell_mat_ctx->scatterU, f, shell_mat_ctx->Ku,
                         INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecScatterBegin(shell_mat_ctx->scatterV, f, shell_mat_ctx->v,
                           INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecScatterEnd(shell_mat_ctx->scatterV, f, shell_mat_ctx->v,
                         INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR VecScale(shell_mat_ctx->v,1/shell_mat_ctx->scale);
    // apply pre-conditioner and calculate u
    CHKERR MatMult(shell_mat_ctx->M, shell_mat_ctx->v,
                   shell_mat_ctx->Mv);                        // Mrv
    CHKERR VecAXPY(shell_mat_ctx->Ku, -1, shell_mat_ctx->Mv); // f-Mrv
    CHKERR PCApply(ctx->pC, shell_mat_ctx->Ku,
                   shell_mat_ctx->u); // u = (aM+K)^(-1)(ru-Mrv)
    // VecView(shell_mat_ctx->u,PETSC_VIEWER_STDOUT_WORLD);
    // calculate velocities
    CHKERR VecAXPY(shell_mat_ctx->v, shell_mat_ctx->ts_a,
                   shell_mat_ctx->u); // v = v + a*u
    // VecView(shell_mat_ctx->v,PETSC_VIEWER_STDOUT_WORLD);
    // reverse
    CHKERR VecZeroEntries(x);
    CHKERR VecScatterBegin(shell_mat_ctx->scatterU, shell_mat_ctx->u, x,
                           INSERT_VALUES, SCATTER_REVERSE);
    CHKERR VecScatterEnd(shell_mat_ctx->scatterU, shell_mat_ctx->u, x,
                         INSERT_VALUES, SCATTER_REVERSE);
    CHKERR VecScatterBegin(shell_mat_ctx->scatterV, shell_mat_ctx->v, x,
                           INSERT_VALUES, SCATTER_REVERSE);
    CHKERR VecScatterEnd(shell_mat_ctx->scatterV, shell_mat_ctx->v, x,
                         INSERT_VALUES, SCATTER_REVERSE);
    CHKERR VecAssemblyBegin(x);
    CHKERR VecAssemblyEnd(x);
    MoFEMFunctionReturnHot(0);
  }

  struct ShellResidualElement : public FEMethod {
    MoFEM::Interface &mField;
    ShellResidualElement(MoFEM::Interface &m_field);

    // variables bellow need to be set by user
    MatShellCtx *shellMatCtx; ///< pointer to shell matrix

    /**
     * @brief Calculate inconsistency between approximation of velocities and
     * velocities calculated from displacements
     *
     * @return MoFEMErrorCode
     */
    MoFEMErrorCode preProcess();

  };

#ifdef __DIRICHLET_HPP__

  /** \brief blocked element/problem
   *
   * Blocked element run loops for different problem than TS problem. It is
   * used to calculate matrices of shell matrix.
   *
   */
  struct ShellMatrixElement : public FEMethod {

    MoFEM::Interface &mField;
    ShellMatrixElement(MoFEM::Interface &m_field);

    typedef std::pair<std::string, FEMethod *> PairNameFEMethodPtr;
    typedef std::vector<PairNameFEMethodPtr> LoopsToDoType;
    LoopsToDoType loopK;    ///< methods to calculate K shell matrix
    LoopsToDoType loopM;    ///< methods to calculate M shell matrix
    LoopsToDoType loopAuxM; ///< methods to calculate derivatives of inertia
                            ///< forces over displacements shell matrix

    // variables bellow need to be set by user
    string problemName;                      ///< name of shell problem
    MatShellCtx *shellMatCtx;                ///< pointer to shell matrix
    DirichletDisplacementBc *DirichletBcPtr; ///< boundary conditions

    MoFEMErrorCode preProcess();
  };

#endif //__DIRICHLET_HPP__
};

#endif //__CONVECTIVE_MASS_ELEMENT_HPP

/**
* \defgroup convective_mass_elem Mass Element
* \ingroup user_modules
**/
