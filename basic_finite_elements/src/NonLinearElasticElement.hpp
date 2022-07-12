/** \file NonLinearElasticElement.hpp
 * \ingroup nonlinear_elastic_elem
 * \brief Operators and data structures for non-linear elastic analysis
 *
 * Implementation of nonlinear elastic element.
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

#ifndef __NONLINEAR_ELASTIC_HPP
#define __NONLINEAR_ELASTIC_HPP

#ifndef WITH_ADOL_C
#error "MoFEM need to be compiled with ADOL-C"
#endif

/** \brief structure grouping operators and data used for calculation of
 * nonlinear elastic element \ingroup nonlinear_elastic_elem
 *
 * In order to assemble matrices and right hand vectors, the loops over
 * elements, entities over that elements and finally loop over integration
 * points are executed.
 *
 * Following implementation separate those three categories of loops and to each
 * loop attach operator.
 *
 */
struct NonlinearElasticElement {

  /// \brief  definition of volume element
  struct MyVolumeFE : public MoFEM::VolumeElementForcesAndSourcesCore {

    Mat A;
    Vec F;

    int addToRule;

    MyVolumeFE(MoFEM::Interface &m_field);
    virtual ~MyVolumeFE() = default;

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

  MyVolumeFE feRhs; ///< calculate right hand side for tetrahedral elements
  MyVolumeFE &getLoopFeRhs() { return feRhs; } ///< get rhs volume element
  MyVolumeFE feLhs; //< calculate left hand side for tetrahedral elements
  MyVolumeFE &getLoopFeLhs() { return feLhs; } ///< get lhs volume element

  MyVolumeFE feEnergy; ///< calculate elastic energy
  MyVolumeFE &getLoopFeEnergy() { return feEnergy; } ///< get energy fe

  MoFEM::Interface &mField;
  short int tAg;

  NonlinearElasticElement(MoFEM::Interface &m_field, short int tag);
  virtual ~NonlinearElasticElement() = default;

  template <typename TYPE> struct FunctionsToCalculatePiolaKirchhoffI;

  /** \brief data for calculation het conductivity and heat capacity elements
   * \ingroup nonlinear_elastic_elem
   */
  struct BlockData {
    int iD;
    double E;
    double PoissonRatio;
    // Eberlein Fibres stiffness properties
    double k1, k2;
    Range tEts; ///< constrains elements in block set
    boost::shared_ptr<FunctionsToCalculatePiolaKirchhoffI<adouble>>
        materialAdoublePtr;
    boost::shared_ptr<FunctionsToCalculatePiolaKirchhoffI<double>>
        materialDoublePtr;
    Range forcesOnlyOnEntitiesRow;
    Range forcesOnlyOnEntitiesCol;
  };

  std::map<int, BlockData>
      setOfBlocks; ///< maps block set id with appropriate BlockData

  /** \brief common data used by volume elements
   * \ingroup nonlinear_elastic_elem
   */
  struct CommonData {

    std::map<std::string, std::vector<VectorDouble>> dataAtGaussPts;
    std::map<std::string, std::vector<MatrixDouble>> gradAtGaussPts;
    string spatialPositions;
    string meshPositions;
    std::vector<MatrixDouble3by3> sTress;
    std::vector<MatrixDouble>
        jacStress; ///< this is simply material tangent operator

    // This part can be used to calculate stress directly from potential

    std::vector<double> eNergy;
    std::vector<VectorDouble> jacEnergy;
    std::vector<VectorDouble> hessianEnergy;
  };
  CommonData commonData;

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  /** \brief Implementation of elastic (non-linear) St. Kirchhoff equation
   * \ingroup nonlinear_elastic_elem
   */
  template <typename TYPE> struct FunctionsToCalculatePiolaKirchhoffI {

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;

    FunctionsToCalculatePiolaKirchhoffI()
        : t_F(resizeAndSet(F)), t_C(resizeAndSet(C)), t_E(resizeAndSet(E)),
          t_S(resizeAndSet(S)), t_invF(resizeAndSet(invF)),
          t_P(resizeAndSet(P)), t_sIGma(resizeAndSet(sIGma)),
          t_h(resizeAndSet(h)), t_H(resizeAndSet(H)),
          t_invH(resizeAndSet(invH)), t_sigmaCauchy(resizeAndSet(sigmaCauchy)) {
    }

    virtual ~FunctionsToCalculatePiolaKirchhoffI() = default;

    double lambda, mu;
    MatrixBoundedArray<TYPE, 9> F, C, E, S, invF, P, sIGma, h, H, invH,
        sigmaCauchy;
    FTensor::Tensor2<FTensor::PackPtr<TYPE *, 0>, 3, 3> t_F, t_C, t_E, t_S,
        t_invF, t_P, t_sIGma, t_h, t_H, t_invH, t_sigmaCauchy;

    TYPE J, eNergy, detH, detF;

    int gG;                    ///< Gauss point number
    CommonData *commonDataPtr; ///< common data shared between entities (f.e.
                               ///< field values at Gauss pts.)
    MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator
        *opPtr; ///< pointer to finite element tetrahedral operator

    MoFEMErrorCode calculateC_CauchyDeformationTensor() {
      MoFEMFunctionBeginHot;
      t_C(i, j) = t_F(k, i) * t_F(k, j);
      MoFEMFunctionReturnHot(0);
    }

    MoFEMErrorCode calculateE_GreenStrain() {
      MoFEMFunctionBeginHot;
      constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
      t_E(i, j) = 0.5 * (t_C(i, j) - t_kd(i, j));
      MoFEMFunctionReturnHot(0);
    }

    // St. Venant–Kirchhoff Material
    MoFEMErrorCode calculateS_PiolaKirchhoffII() {
      MoFEMFunctionBegin;
      constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
      t_S(i, j) = (2 * mu) * t_E(i, j) + (lambda * t_E(k, k)) * t_kd(i, j);
      MoFEMFunctionReturn(0);
    }

    /** \brief Function overload to implement user material
      *

      * Calculation of Piola Kirchhoff I is implemented by user. Tangent matrix
      * user implemented physical equation is calculated using automatic
      * differentiation.

      * \f$\mathbf{S} =
      \lambda\textrm{tr}[\mathbf{E}]\mathbf{I}+2\mu\mathbf{E}\f$

      * Notes: <br>
      * Number of actual Gauss point is accessed from variable gG. <br>
      * Access to operator data structures is available by variable opPtr. <br>
      * Access to common data is by commonDataPtr. <br>

      * \param block_data used to give access to material parameters
      * \param fe_ptr pointer to element data structures

      For details look to: <br>
      NONLINEAR CONTINUUM MECHANICS FOR FINITE ELEMENT ANALYSIS, Javier Bonet,
      Richard D. Wood

      */
    virtual MoFEMErrorCode calculateP_PiolaKirchhoffI(
        const BlockData block_data,
        boost::shared_ptr<const NumeredEntFiniteElement> fe_ptr) {
      MoFEMFunctionBegin;
      lambda = LAMBDA(block_data.E, block_data.PoissonRatio);
      mu = MU(block_data.E, block_data.PoissonRatio);
      CHKERR calculateC_CauchyDeformationTensor();
      CHKERR calculateE_GreenStrain();
      CHKERR calculateS_PiolaKirchhoffII();
      t_P(i, j) = t_F(i, k) * t_S(k, j);
      MoFEMFunctionReturn(0);
    }

    /** \brief Function overload to implement user material
      *

      * Calculation of Piola Kirchhoff I is implemented by user. Tangent matrix
      * user implemented physical equation is calculated using automatic
      * differentiation.

      * \f$\mathbf{S} =
      \lambda\textrm{tr}[\mathbf{E}]\mathbf{I}+2\mu\mathbf{E}\f$

      * Notes: <br>
      * Number of actual Gauss point is accessed from variable gG. <br>
      * Access to operator data structures is available by variable opPtr. <br>
      * Access to common data is by commonDataPtr. <br>

      * \param block_data used to give access to material parameters
      * \param fe_ptr pointer to element data structures

      For details look to: <br>
      NONLINEAR CONTINUUM MECHANICS FOR FINITE ELEMENT ANALYSIS, Javier Bonet,
      Richard D. Wood

      */
    virtual MoFEMErrorCode calculateCauchyStress(
        const BlockData block_data,
        boost::shared_ptr<const NumeredEntFiniteElement> fe_ptr) {
      MoFEMFunctionBegin;
      sigmaCauchy.resize(3, 3);
      t_sigmaCauchy(i, j) = t_P(i, k) * t_F(j, k);
      t_sigmaCauchy(i, j) /= determinantTensor3by3(t_F);
      MoFEMFunctionReturn(0);
    }

    /**
     * \brief add additional active variables
     *
     * \note This member function if used should be implement by template member
     * function Specialization, different implementation needed for TYPE=double
     * or TYPE=adouble
     *
     * More complex physical models depend on gradient of defamation and some
     * additional variables. For example can depend on temperature. This
     * function adds additional independent variables to the model.
     *
     * @param  nb_active_variables number of active variables
     * @return                     error code
     */
    virtual MoFEMErrorCode setUserActiveVariables(int &nb_active_variables) {
      MoFEMFunctionBeginHot;
      MoFEMFunctionReturnHot(0);
    }

    /**
     * \brief Add additional independent variables
     * More complex physical models depend on gradient of defamation and some
     * additional variables. For example can depend on temperature. This
     * function adds additional independent variables to the model.
     *
     * /note First 9 elements are reserved for gradient of deformation.
     * @param  activeVariables vector of deepened variables, values after index
     * 9 should be add.
     *
     * @return                 error code
     */
    virtual MoFEMErrorCode
    setUserActiveVariables(VectorDouble &activeVariables) {
      MoFEMFunctionBeginHot;
      MoFEMFunctionReturnHot(0);
    }

    /** \brief Calculate elastic energy density
     *
     * \f[\Psi =
     * \frac{1}{2}\lambda(\textrm{tr}[\mathbf{E}])^2+\mu\mathbf{E}:\mathbf{E}\f]
     */
    virtual MoFEMErrorCode calculateElasticEnergy(
        const BlockData block_data,
        boost::shared_ptr<const NumeredEntFiniteElement> fe_ptr) {
      MoFEMFunctionBegin;
      lambda = LAMBDA(block_data.E, block_data.PoissonRatio);
      mu = MU(block_data.E, block_data.PoissonRatio);
      CHKERR calculateC_CauchyDeformationTensor();
      CHKERR calculateE_GreenStrain();
      TYPE trace = 0;
      eNergy = 0;
      for (int ii = 0; ii < 3; ii++) {
        trace += E(ii, ii);
        for (int jj = 0; jj < 3; jj++) {
          TYPE e = E(ii, jj);
          eNergy += mu * e * e;
        }
      }
      eNergy += 0.5 * lambda * trace * trace;
      MoFEMFunctionReturn(0);
    }

    /** \brief Calculate Eshelby stress
     */
    virtual MoFEMErrorCode calculatesIGma_EshelbyStress(
        const BlockData block_data,
        boost::shared_ptr<const NumeredEntFiniteElement> fe_ptr) {
      MoFEMFunctionBegin;
      CHKERR calculateP_PiolaKirchhoffI(block_data, fe_ptr);
      CHKERR calculateElasticEnergy(block_data, fe_ptr);
      constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
      t_sIGma(i, j) = t_kd(i, j) * eNergy - t_F(k, i) * t_P(k, j);
      MoFEMFunctionReturn(0);
    }

    /** \brief Do operations when pre-process
     */
    virtual MoFEMErrorCode getDataOnPostProcessor(
        std::map<std::string, std::vector<VectorDouble>> &field_map,
        std::map<std::string, std::vector<MatrixDouble>> &grad_map) {
      MoFEMFunctionBeginHot;
      MoFEMFunctionReturnHot(0);
    }

    protected:
      inline static auto resizeAndSet(MatrixBoundedArray<TYPE, 9> &m) {
        m.resize(3, 3, false);
        return getFTensor2FromArray3by3(m, FTensor::Number<0>(), 0);
      };
  };

  struct OpGetDataAtGaussPts
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    std::vector<VectorDouble> &valuesAtGaussPts;
    std::vector<MatrixDouble> &gradientAtGaussPts;
    const EntityType zeroAtType;

    OpGetDataAtGaussPts(const std::string field_name,
                        std::vector<VectorDouble> &values_at_gauss_pts,
                        std::vector<MatrixDouble> &gradient_at_gauss_pts);

    /** \brief operator calculating deformation gradient
     *
     * temperature gradient is calculated multiplying derivatives of shape
     * functions by degrees of freedom
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data);
  };

  struct OpGetCommonDataAtGaussPts : public OpGetDataAtGaussPts {
    OpGetCommonDataAtGaussPts(const std::string field_name,
                              CommonData &common_data);
  };

  /**
   * \brief Operator performs automatic differentiation.
   */
  struct OpJacobianPiolaKirchhoffStress
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    BlockData &dAta; ///< Structure keeping data about problem, like material
                     ///< parameters
    CommonData &commonData; ///< Structure keeping data abut this particular
                            ///< element, e.g. gradient of deformation at
                            ///< integration points
    int tAg; //,lastId;        ///< ADOL-C tag used for recording operations
    int adlocReturnValue; ///< return value from ADOL-C, if non-zero that is
                          ///< error.
    bool jAcobian;        ///< if true Jacobian is calculated
    bool fUnction;        ///< if true stress i calculated
    bool aLe;             ///< true if arbitrary Lagrangian-Eulerian formulation
    bool fieldDisp;       ///< true if field of displacements is given, usually
                          ///< spatial positions are given.

    /**
      \brief Construct operator to calculate Piola-Kirchhoff stress or its
      derivatives over gradient deformation

      \param field_name approximation field name of spatial positions or
      displacements \param data reference to block data (what is Young modulus,
      Poisson ratio or what elements are part of the block) \param tag adol-c
      unique tag of the tape \param jacobian if true derivative of Piola Stress
      is calculated otherwise just stress is calculated \param field_disp if
      true approximation field keeps displacements not spatial positions

    */
    OpJacobianPiolaKirchhoffStress(const std::string field_name,
                                   BlockData &data, CommonData &common_data,
                                   int tag, bool jacobian, bool ale,
                                   bool field_disp);

    VectorDouble activeVariables;
    int nbActiveVariables;

    std::vector<MatrixDouble> *ptrh;
    std::vector<MatrixDouble> *ptrH;

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;

    /**
     * \brief Calculate Paola-Kirchhoff I stress
     * @return error code
     */
    virtual MoFEMErrorCode calculateStress(const int gg);

    /**
     * \brief Record ADOL-C tape
     * @return error code
     */
    virtual MoFEMErrorCode recordTag(const int gg);

    /**
     * \brief Play ADOL-C tape
     * @return error code
     */
    virtual MoFEMErrorCode playTag(const int gg);

    /**
     * \brief Cgeck if tape is recorded for given integration point
     * @param  gg integration point
     * @return    true if tag is recorded
     */
    virtual bool recordTagForIntegrationPoint(const int gg) {
      // return true;
      if (gg == 0)
        return true;
      return false;
    }

    /**
     * \brief Calculate stress or jacobian at gauss points
     *
     * @param  row_side
     * @param  row_type
     * @param  row_data
     * @return          error code
     */
    MoFEMErrorCode doWork(int row_side, EntityType row_type,
                          EntitiesFieldData::EntData &row_data);
  };

  /**
   * \brief Calculate explicit derivative of free energy
   */
  struct OpJacobianEnergy
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    BlockData &dAta;
    CommonData &commonData;

    int tAg;        ///< tape tag
    bool gRadient;  ///< if set true gradient of energy is calculated
    bool hEssian;   ///< if set true hessian of energy is calculated
    bool aLe;       ///< true if arbitrary Lagrangian-Eulerian formulation
    bool fieldDisp; ///< true if displacements instead spatial positions used

    OpJacobianEnergy(const std::string field_name, ///< field name for spatial
                                                   ///< positions or
                                                   ///< displacements
                     BlockData &data, CommonData &common_data, int tag,
                     bool gradient, bool hessian, bool ale, bool field_disp);

    VectorDouble activeVariables;
    int nbActiveVariables;

    std::vector<MatrixDouble> *ptrh;
    std::vector<MatrixDouble> *ptrH;

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;

    /**
     * \brief Check if tape is recorded for given integration point
     * @param  gg integration point
     * @return    true if tag is recorded
     */
    virtual bool recordTagForIntegrationPoint(const int gg) {
      if (gg == 0)
        return true;
      return false;
    }

    /**
     * \brief Calculate Paola-Kirchhoff I stress
     * @return error code
     */
    virtual MoFEMErrorCode calculateEnergy(const int gg);

    /**
     * \brief Record ADOL-C tape
     * @return error code
     */
    virtual MoFEMErrorCode recordTag(const int gg);

    /**
     * \brief Play ADOL-C tape
     * @return error code
     */
    virtual MoFEMErrorCode playTag(const int gg);

    MoFEMErrorCode doWork(int row_side, EntityType row_type,
                          EntitiesFieldData::EntData &row_data);
  };

  struct OpRhsPiolaKirchhoff
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    BlockData &dAta;
    CommonData &commonData;
    bool fieldDisp;
    bool aLe;

    ublas::vector<int> iNdices;
    OpRhsPiolaKirchhoff(const std::string field_name, BlockData &data,
                        CommonData &common_data);

    VectorDouble nf;
    MoFEMErrorCode doWork(int row_side, EntityType row_type,
                          EntitiesFieldData::EntData &row_data);

    virtual MoFEMErrorCode aSemble(int row_side, EntityType row_type,
                                   EntitiesFieldData::EntData &row_data);
  };

  struct OpEnergy
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    BlockData &dAta;
    CommonData &commonData;
    SmartPetscObj<Vec> ghostVec;
    bool fieldDisp;

    OpEnergy(const std::string field_name, BlockData &data,
             CommonData &common_data, SmartPetscObj<Vec> ghost_vec,
             bool field_disp);

    MoFEMErrorCode doWork(int row_side, EntityType row_type,
                          EntitiesFieldData::EntData &row_data);
  };

  struct OpLhsPiolaKirchhoff_dx
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    BlockData &dAta;
    CommonData &commonData;
    int tAg;
    bool aLe;

    ublas::vector<int> rowIndices;
    ublas::vector<int> colIndices;

    OpLhsPiolaKirchhoff_dx(const std::string vel_field,
                           const std::string field_name, BlockData &data,
                           CommonData &common_data);

    MatrixDouble k, trans_k, jac, F;

    /**
      \brief Directive of Piola Kirchhoff stress over spatial DOFs

      This project derivative \f$\frac{\partial P}{\partial F}\f$, that is
      \f[
      \frac{\partial P}{\partial x_\textrm{DOF}} =  \frac{\partial P}{\partial
      F}\frac{\partial F}{\partial x_\textrm{DOF}}, \f] where second therm
      \f$\frac{\partial F}{\partial x_\textrm{DOF}}\f$ is derivative of shape
      function

    */
    virtual MoFEMErrorCode getJac(EntitiesFieldData::EntData &col_data,
                                  int gg);

    virtual MoFEMErrorCode aSemble(int row_side, int col_side,
                                   EntityType row_type, EntityType col_type,
                                   EntitiesFieldData::EntData &row_data,
                                   EntitiesFieldData::EntData &col_data);

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          EntitiesFieldData::EntData &row_data,
                          EntitiesFieldData::EntData &col_data);
  };

  struct OpLhsPiolaKirchhoff_dX : public OpLhsPiolaKirchhoff_dx {

    OpLhsPiolaKirchhoff_dX(const std::string vel_field,
                           const std::string field_name, BlockData &data,
                           CommonData &common_data);

    /// \brief Derivative of Piola Kirchhoff stress over material DOFs
    MoFEMErrorCode getJac(EntitiesFieldData::EntData &col_data, int gg);

    MoFEMErrorCode aSemble(int row_side, int col_side, EntityType row_type,
                           EntityType col_type,
                           EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);
  };

  struct OpJacobianEshelbyStress : public OpJacobianPiolaKirchhoffStress {

    OpJacobianEshelbyStress(const std::string field_name, BlockData &data,
                            CommonData &common_data, int tag, bool jacobian,
                            bool ale);

    MoFEMErrorCode calculateStress(const int gg);
  };

  struct OpRhsEshelbyStress : public OpRhsPiolaKirchhoff {

    OpRhsEshelbyStress(const std::string field_name, BlockData &data,
                       CommonData &common_data);
  };

  /**
   * \deprecated name with spelling mistake
   */
  DEPRECATED typedef OpRhsEshelbyStress OpRhsEshelbyStrees;

  struct OpLhsEshelby_dx : public OpLhsPiolaKirchhoff_dX {

    OpLhsEshelby_dx(const std::string vel_field, const std::string field_name,
                    BlockData &data, CommonData &common_data);

    MoFEMErrorCode getJac(EntitiesFieldData::EntData &col_data, int gg);
  };

  struct OpLhsEshelby_dX : public OpLhsPiolaKirchhoff_dx {

    OpLhsEshelby_dX(const std::string vel_field, const std::string field_name,
                    BlockData &data, CommonData &common_data);

    MoFEMErrorCode getJac(EntitiesFieldData::EntData &col_data, int gg);
  };

  MoFEMErrorCode
  setBlocks(boost::shared_ptr<FunctionsToCalculatePiolaKirchhoffI<double>>
                materialDoublePtr,
            boost::shared_ptr<FunctionsToCalculatePiolaKirchhoffI<adouble>>
                materialAdoublePtr);

  MoFEMErrorCode addElement(
      const std::string element_name,
      const std::string spatial_position_field_name,
      const std::string material_position_field_name = "MESH_NODE_POSITIONS",
      const bool ale = false);

  /** \brief Set operators to calculate left hand tangent matrix and right hand
   * residual
   *
   * \param fun class needed to calculate Piola Kirchhoff I Stress tensor
   * \param spatial_position_field_name name of approximation field
   * \param material_position_field_name name of field to define geometry
   * \param ale true if arbitrary Lagrangian Eulerian formulation
   * \param field_disp true if approximation field represents displacements
   * otherwise it is field of spatial positions
   */
  MoFEMErrorCode setOperators(
      const std::string spatial_position_field_name,
      const std::string material_position_field_name = "MESH_NODE_POSITIONS",
      const bool ale = false, const bool field_disp = false);
};

#endif //__NONLINEAR_ELASTIC_HPP

/**
 * \defgroup nonlinear_elastic_elem NonLinear Elastic Element
 * \ingroup user_modules
 * \defgroup user_modules User modules
 **/
