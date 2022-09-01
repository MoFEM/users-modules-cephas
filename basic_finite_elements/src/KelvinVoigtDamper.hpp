/** \file KelvinVoigtDamper.hpp
 * \brief Implementation dashpot, i.e. damper
 * \ingroup nonlinear_elastic_elem
 *
 */



#ifndef __KELVIN_VOIGT_DAMPER_HPP__
#define __KELVIN_VOIGT_DAMPER_HPP__

#ifndef WITH_ADOL_C
#error "MoFEM need to be compiled with ADOL-C"
#endif

/** \brief Implementation of Kelvin Voigt Damper
\ingroup nonlinear_elastic_elem

*/
struct KelvinVoigtDamper {

  enum TagEvaluate { DAMPERSTRESS };

  MoFEM::Interface &mField;

  /** \brief Dumper material parameters
  \ingroup nonlinear_elastic_elem
  */
  struct BlockMaterialData {
    Range tEts;
    double vBeta; ///< Poisson ration spring alpha
    double gBeta; ///< Sheer modulus spring alpha
    bool lInear;
    BlockMaterialData() : vBeta(0), gBeta(1), lInear(false) {}
  };

  std::map<int, BlockMaterialData> blockMaterialDataMap;

  /** \brief Constitutive model functions
  \ingroup nonlinear_elastic_elem

  */
  template <typename TYPE> struct ConstitutiveEquation {

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;

    BlockMaterialData &dAta;
    bool isDisplacement;

    ConstitutiveEquation(BlockMaterialData &data, bool is_displacement = true)
        : dAta(data), isDisplacement(is_displacement) {}
    virtual ~ConstitutiveEquation() = default;

    MatrixBoundedArray<TYPE, 9> F;    ///< Gradient of deformation
    MatrixBoundedArray<TYPE, 9> FDot;         ///< Rate of gradient of deformation
    MatrixBoundedArray<TYPE, 9>
        gradientUDot; ///< Rate of gradient of displacements
    MatrixBoundedArray<TYPE, 9> engineringStrainDot;
    MatrixBoundedArray<TYPE, 9>
        dashpotCauchyStress; ///< Stress generated by spring beta
    MatrixBoundedArray<TYPE, 9> 
        dashpotFirstPiolaKirchhoffStress; ///< Stress generated by spring beta
    MatrixBoundedArray<TYPE, 9>  invF; ///< Inverse of gradient of deformation

    TYPE traceEngineeringStrainDot;
    TYPE J;                   ///< Jacobian of gradient of deformation

    /** \brief Calculate strain rate

    \f[
    \dot{\varepsilon}_{ij} = \frac{1}{2}
    \left(
    \frac{\partial v_i}{\partial X_j}
    +
    \frac{\partial v_j}{\partial X_i}
    \right)
    \f]

    */
    virtual MoFEMErrorCode calculateEngineeringStrainDot() {
      MoFEMFunctionBegin;
      gradientUDot.resize(3, 3, false);
      noalias(gradientUDot) = FDot;

      // for (int ii = 0; ii < 3; ii++)
      //   gradientUDot(ii, ii) -= 1;
        
      traceEngineeringStrainDot = 0;
      for (int ii = 0; ii < 3; ii++) {
        traceEngineeringStrainDot += gradientUDot(ii, ii);
      }
      engineringStrainDot.resize(3, 3, false);
      noalias(engineringStrainDot) = gradientUDot + trans(gradientUDot);
      engineringStrainDot *= 0.5;
      MoFEMFunctionReturn(0);
    }

    /** \brief Calculate Cauchy dashpot stress

    Calculate dashpot Cauchy stress. It has to be pull back  to reference
    configuration before use in total Lagrangian formulation.

    \f[
    \sigma^\beta_{ij} = 2G^\beta\left[
    \dot{\varepsilon}_{ij}
    + \frac{v^\beta}{1-2v^\beta}\dot{\varepsilon}_{kk}\delta_{ij}
    \right]
    \f]

    */
    virtual MoFEMErrorCode calculateDashpotCauchyStress() {
      MoFEMFunctionBegin;
      dashpotCauchyStress.resize(3, 3, false);
      double a = 2.0 * dAta.gBeta;
      double b = a * (dAta.vBeta / (1.0 - 2.0 * dAta.vBeta));
      noalias(dashpotCauchyStress) = a * engineringStrainDot;
      for (int ii = 0; ii < 3; ii++) {
        dashpotCauchyStress(ii, ii) += b * traceEngineeringStrainDot;
      }
      MoFEMFunctionReturn(0);
    }

    /** \brief Calculate First Piola-Kirchhoff Stress Dashpot stress

    \f[
    P^\beta_{ij} = J \sigma^\beta_{ik} F^{-1}_{jk}
    \f]

    */
    virtual MoFEMErrorCode calculateFirstPiolaKirchhoffStress() {
      MoFEMFunctionBegin;
      dashpotFirstPiolaKirchhoffStress.resize(3, 3, false);
      if (dAta.lInear) {
        noalias(dashpotFirstPiolaKirchhoffStress) = dashpotCauchyStress;
      } else {

        invF.resize(3, 3, false);

        auto t_dashpotFirstPiolaKirchhoffStress = getFTensor2FromArray3by3(
            dashpotFirstPiolaKirchhoffStress, FTensor::Number<0>(), 0);
        auto t_dashpotCauchyStress = getFTensor2FromArray3by3(
            dashpotCauchyStress, FTensor::Number<0>(), 0);
        auto t_F = getFTensor2FromArray3by3(F, FTensor::Number<0>(), 0);
        auto t_invF = getFTensor2FromArray3by3(invF, FTensor::Number<0>(), 0);
        if (isDisplacement) {
          t_F(0, 0) += 1;
          t_F(1, 1) += 1;
          t_F(2, 2) += 1;
        }

        J = determinantTensor3by3(t_F);
        CHKERR invertTensor3by3(t_F, J, t_invF);
        t_dashpotFirstPiolaKirchhoffStress(i, j) =
            J * (t_dashpotCauchyStress(i, k) * t_invF(j, k));
      }
      MoFEMFunctionReturn(0);
    }
  };

  typedef boost::ptr_map<int, KelvinVoigtDamper::ConstitutiveEquation<adouble>>
      ConstitutiveEquationMap;
  ConstitutiveEquationMap constitutiveEquationMap;

  /** \brief Common data for nonlinear_elastic_elem model
  \ingroup nonlinear_elastic_elem
  */
  struct CommonData {

    string spatialPositionName;
    string spatialPositionNameDot;
    string meshNodePositionName;

    std::map<std::string, std::vector<VectorDouble>> dataAtGaussPts;
    std::map<std::string, std::vector<MatrixDouble>> gradAtGaussPts;

    boost::shared_ptr<MatrixDouble> dataAtGaussTmpPtr;
    boost::shared_ptr<MatrixDouble> gradDataAtGaussTmpPtr;

    std::vector<MatrixDouble> dashpotFirstPiolaKirchhoffStress;

    std::vector<double *> jacRowPtr;
    std::vector<MatrixDouble> jacStress;

    bool recordOn;
    bool skipThis;
    std::map<int, int> nbActiveVariables, nbActiveResults;

    CommonData() : recordOn(true), skipThis(true) {
      dataAtGaussTmpPtr = boost::make_shared<MatrixDouble>();
      dataAtGaussTmpPtr->resize(3, 1);
      gradDataAtGaussTmpPtr = boost::make_shared<MatrixDouble>();
      gradDataAtGaussTmpPtr->resize(9, 1);
      meshNodePositionName = "MESH_NODE_POSITIONS";
    }
  };
  CommonData commonData;

  /// \brief definition of volume element
  struct DamperFE : public MoFEM::VolumeElementForcesAndSourcesCore {

    CommonData &commonData;
    int addToRule; ///< Takes into account HO geometry

    DamperFE(MoFEM::Interface &m_field, CommonData &common_data)
        : MoFEM::VolumeElementForcesAndSourcesCore(m_field),
          commonData(common_data), addToRule(1) {}

    int getRule(int order) { return order + addToRule; }

    MoFEMErrorCode preProcess() {

      MoFEMFunctionBegin;
      CHKERR MoFEM::VolumeElementForcesAndSourcesCore::preProcess();

      // if (ts_ctx == CTX_TSSETIFUNCTION) {

      //   CHKERR mField.getInterface<VecManager>()->setOtherLocalGhostVector(
      //       problemPtr, commonData.spatialPositionName,
      //       commonData.spatialPositionNameDot, COL, ts_u_t, INSERT_VALUES,
      //       SCATTER_REVERSE);
      // }

      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode postProcess() {

      MoFEMFunctionBegin;

      // CHKERR MoFEM::VolumeElementForcesAndSourcesCore::postProcess();

      // if (ts_ctx == CTX_TSSETIFUNCTION) {
      //   CHKERR VecAssemblyBegin(ts_F);
      //   CHKERR VecAssemblyEnd(ts_F);
      // }
      // if (ts_ctx == CTX_TSSETIJACOBIAN) {
      //   CHKERR MatAssemblyBegin(ts_B, MAT_FLUSH_ASSEMBLY);
      //   CHKERR MatAssemblyEnd(ts_B, MAT_FLUSH_ASSEMBLY);
      // }

      MoFEMFunctionReturn(0);
    }
  };

  DamperFE feRhs, feLhs;

  KelvinVoigtDamper(MoFEM::Interface &m_field)
      : mField(m_field), feRhs(m_field, commonData),
        feLhs(m_field, commonData) {}

  struct OpGetDataAtGaussPts
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    CommonData &commonData;
    bool calcVal;
    bool calcGrad;
    bool calcDot;
    EntityType zeroAtType;

    OpGetDataAtGaussPts(const std::string field_name, CommonData &common_data,
                        bool calc_val, bool calc_grad, bool calc_dot = false,
                        EntityType zero_at_type = MBVERTEX)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL),
          commonData(common_data), calcVal(calc_val), calcGrad(calc_grad),
          calcDot(calc_dot), zeroAtType(zero_at_type) {}

    /** \brief Operator field value
     *
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data) {
      MoFEMFunctionBegin;

      int nb_dofs = data.getFieldData().size();
      if (nb_dofs == 0) {
        MoFEMFunctionReturnHot(0);
      }
      int rank = data.getFieldDofs()[0]->getNbOfCoeffs();
      int nb_gauss_pts = data.getN().size1();

      string field_name = commonData.spatialPositionName;
      if (calcDot)
        field_name = commonData.spatialPositionNameDot;
      // Initialize
      if (calcVal) {
        commonData.dataAtGaussPts[field_name].resize(nb_gauss_pts);
        for (int gg = 0; gg < nb_gauss_pts; gg++) {
          commonData.dataAtGaussPts[field_name][gg].resize(rank, false);
          commonData.dataAtGaussPts[field_name][gg].clear();
        }
      }
      if (calcGrad) {
        commonData.gradAtGaussPts[field_name].resize(nb_gauss_pts);
        for (int gg = 0; gg < nb_gauss_pts; gg++) {
          commonData.gradAtGaussPts[field_name][gg].resize(rank, 3, false);
          commonData.gradAtGaussPts[field_name][gg].clear();
        }
      }

      // Zero values
      // if (type == zeroAtType) {
      //   for (int gg = 0; gg < nb_gauss_pts; gg++) {
      //     if (calcVal) {
      //       commonData.dataAtGaussPts[rowFieldName][gg].clear();
      //     }
      //     if (calcGrad) {
      //       commonData.gradAtGaussPts[rowFieldName][gg].clear();
      //     }
      //   }
      // }

      auto t_disp = getFTensor1FromMat<3>(*commonData.dataAtGaussTmpPtr);
      auto t_diff_disp =
          getFTensor2FromMat<3, 3>(*commonData.gradDataAtGaussTmpPtr);

      // Calculate values at integration points
      for (int gg = 0; gg < nb_gauss_pts; gg++) {
          for (int rr1 = 0; rr1 < rank; rr1++) {
            if (calcVal) {
              commonData.dataAtGaussPts[field_name][gg][rr1] = t_disp(rr1);
            }
            if (calcGrad) {
              for (int rr2 = 0; rr2 < 3; rr2++) {
                commonData.gradAtGaussPts[field_name][gg](rr1, rr2) =
                    t_diff_disp(rr1, rr2);
              }
            }
          }
          ++t_disp;
          ++t_diff_disp;
      }

      MoFEMFunctionReturn(0);
    }
  };

  struct OpJacobian
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    std::vector<int> tagS;
    KelvinVoigtDamper::ConstitutiveEquation<adouble> &cE;
    CommonData &commonData;

    bool calculateResidualBool;
    bool calculateJacobianBool;
    bool &recordOn;
    std::map<int, int> &nbActiveVariables;
    std::map<int, int> &nbActiveResults;

    OpJacobian(const std::string field_name, std::vector<int> tags,
               KelvinVoigtDamper::ConstitutiveEquation<adouble> &ce,
               CommonData &common_data, bool calculate_residual,
               bool calculate_jacobian)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          tagS(tags), cE(ce), commonData(common_data),
          calculateResidualBool(calculate_residual),
          calculateJacobianBool(calculate_jacobian),
          recordOn(common_data.recordOn),
          nbActiveVariables(common_data.nbActiveVariables),
          nbActiveResults(common_data.nbActiveResults) {}

    int nbGaussPts;
    VectorDouble activeVariables;

    MoFEMErrorCode recordDamperStress() {
      MoFEMFunctionBegin;

      if (tagS[DAMPERSTRESS] < 0) {
        MoFEMFunctionReturnHot(0);
      }

      cE.F.resize(3, 3, false);
      cE.FDot.resize(3, 3, false);
      MatrixDouble &F =
          (commonData.gradAtGaussPts[commonData.spatialPositionName])[0];
      MatrixDouble &F_dot =
          (commonData.gradAtGaussPts[commonData.spatialPositionNameDot])[0];
      trace_on(tagS[DAMPERSTRESS]);
      {
        // Activate gradient of defamation
        nbActiveVariables[tagS[DAMPERSTRESS]] = 0;
        for (int dd1 = 0; dd1 < 3; dd1++) {
          for (int dd2 = 0; dd2 < 3; dd2++) {
            cE.F(dd1, dd2) <<= F(dd1, dd2);
            nbActiveVariables[tagS[DAMPERSTRESS]]++;
          }
        }
        for (int dd1 = 0; dd1 < 3; dd1++) {
          for (int dd2 = 0; dd2 < 3; dd2++) {
            cE.FDot(dd1, dd2) <<= F_dot(dd1, dd2);
            nbActiveVariables[tagS[DAMPERSTRESS]]++;
          }
        }

        // Do calculations
        CHKERR cE.calculateEngineeringStrainDot();
        CHKERR cE.calculateDashpotCauchyStress();
        CHKERR cE.calculateFirstPiolaKirchhoffStress();

        // Results
        nbActiveResults[tagS[DAMPERSTRESS]] = 0;
        commonData.dashpotFirstPiolaKirchhoffStress.resize(nbGaussPts);
        commonData.dashpotFirstPiolaKirchhoffStress[0].resize(3, 3, false);
        for (int d1 = 0; d1 < 3; d1++) {
          for (int d2 = 0; d2 < 3; d2++) {
            cE.dashpotFirstPiolaKirchhoffStress(d1, d2) >>=
                (commonData.dashpotFirstPiolaKirchhoffStress[0])(d1, d2);
            nbActiveResults[tagS[DAMPERSTRESS]]++;
          }
        }
      }
      trace_off();
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode calculateFunction(TagEvaluate te, double *ptr) {
      MoFEMFunctionBegin;

      int r;
      // play recorder for values
      r = ::function(tagS[te], nbActiveResults[tagS[te]],
                     nbActiveVariables[tagS[te]], &activeVariables[0], ptr);
      if (r < 3) { // function is locally analytic
        SETERRQ1(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
                 "ADOL-C function evaluation with error r = %d", r);
      }

      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode calculateJacobian(TagEvaluate te) {
      MoFEMFunctionBegin;

      try {
        int r;
        r = jacobian(tagS[te], nbActiveResults[tagS[te]],
                     nbActiveVariables[tagS[te]], &activeVariables[0],
                     &(commonData.jacRowPtr[0]));
        if (r < 3) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
                  "ADOL-C function evaluation with error");
        }
      } catch (const std::exception &ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF, 1, ss.str().c_str());
      }
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode calculateAtIntPtsDamperStress() {
      MoFEMFunctionBegin;

      if (tagS[DAMPERSTRESS] < 0) {
        MoFEMFunctionReturnHot(0);
      }

      activeVariables.resize(nbActiveVariables[tagS[DAMPERSTRESS]], false);
      for (int gg = 0; gg < nbGaussPts; gg++) {

        MatrixDouble &F =
            (commonData.gradAtGaussPts[commonData.spatialPositionName])[gg];
        MatrixDouble &F_dot =
            (commonData.gradAtGaussPts[commonData.spatialPositionNameDot])[gg];
        int nb_active_variables = 0;

        // Activate gradient of defamation
        for (int dd1 = 0; dd1 < 3; dd1++) {
          for (int dd2 = 0; dd2 < 3; dd2++) {
            activeVariables[nb_active_variables++] = F(dd1, dd2);
          }
        }
        // Activate rate of gradient of defamation
        for (int dd1 = 0; dd1 < 3; dd1++) {
          for (int dd2 = 0; dd2 < 3; dd2++) {
            activeVariables[nb_active_variables++] = F_dot(dd1, dd2);
          }
        }

        if (nb_active_variables != nbActiveVariables[tagS[DAMPERSTRESS]]) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_IMPOSIBLE_CASE,
                  "Number of active variables does not much");
        }

        if (calculateResidualBool) {
          if (gg == 0) {
            commonData.dashpotFirstPiolaKirchhoffStress.resize(nbGaussPts);
          }
          commonData.dashpotFirstPiolaKirchhoffStress[gg].resize(3, 3, false);
          CHKERR calculateFunction(
              DAMPERSTRESS,
              &commonData.dashpotFirstPiolaKirchhoffStress[gg](0, 0));
        }

        if (calculateJacobianBool) {
          if (gg == 0) {
            commonData.jacStress.resize(nbGaussPts);
            commonData.jacRowPtr.resize(nbActiveResults[tagS[DAMPERSTRESS]]);
          }
          commonData.jacStress[gg].resize(nbActiveResults[tagS[DAMPERSTRESS]],
                                          nbActiveVariables[tagS[DAMPERSTRESS]],
                                          false);
          for (int dd = 0; dd < nbActiveResults[tagS[DAMPERSTRESS]]; dd++) {
            commonData.jacRowPtr[dd] = &commonData.jacStress[gg](dd, 0);
          }
          CHKERR calculateJacobian(DAMPERSTRESS);
        }
      }

      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode doWork(int row_side, EntityType row_type,
                          EntitiesFieldData::EntData &row_data) {
      MoFEMFunctionBegin;

      if (row_type != MBVERTEX)
        MoFEMFunctionReturnHot(0);
      nbGaussPts = row_data.getN().size1();

      commonData.skipThis = false;
      if (cE.dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
          cE.dAta.tEts.end()) {
        commonData.skipThis = true;
        MoFEMFunctionReturnHot(0);
      }

      if (recordOn) {
        CHKERR recordDamperStress();
      }
      CHKERR calculateAtIntPtsDamperStress();

      MoFEMFunctionReturn(0);
    }
  };

  struct AssembleVector
      : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {
    AssembleVector(string field_name)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW) {}

    VectorDouble nF;
    MoFEMErrorCode aSemble(int row_side, EntityType row_type,
                           EntitiesFieldData::EntData &row_data) {
      MoFEMFunctionBegin;
      int nb_dofs = row_data.getIndices().size();
      int *indices_ptr = &row_data.getIndices()[0];
      CHKERR VecSetValues(getFEMethod()->ts_F, nb_dofs, indices_ptr, &nF[0],
                          ADD_VALUES);
      MoFEMFunctionReturn(0);
    }
  };

  /** \brief Assemble internal force vector
  \ingroup nonlinear_elastic_elem

  */
  struct OpRhsStress : public AssembleVector {
    CommonData &commonData;
    OpRhsStress(CommonData &common_data)
        : AssembleVector(common_data.spatialPositionName),
          commonData(common_data) {}
    MoFEMErrorCode doWork(int row_side, EntityType row_type,
                          EntitiesFieldData::EntData &row_data) {
      MoFEMFunctionBegin;

      if (commonData.skipThis) {
        MoFEMFunctionReturnHot(0);
      }

      int nb_dofs = row_data.getIndices().size();
      if (!nb_dofs) {
        MoFEMFunctionReturnHot(0);
      }
      nF.resize(nb_dofs, false);
      nF.clear();
      int nb_gauss_pts = row_data.getN().size1();
      for (int gg = 0; gg != nb_gauss_pts; gg++) {
        const MatrixAdaptor &diffN = row_data.getDiffN(gg, nb_dofs / 3);
        const MatrixDouble &stress =
            commonData.dashpotFirstPiolaKirchhoffStress[gg];
        double val = getVolume() * getGaussPts()(3, gg);
        for (int dd = 0; dd < nb_dofs / 3; dd++) {
          for (int rr = 0; rr < 3; rr++) {
            for (int nn = 0; nn < 3; nn++) {
              nF[3 * dd + rr] += val * diffN(dd, nn) * stress(rr, nn);
            }
          }
        }
      }
      CHKERR aSemble(row_side, row_type, row_data);
      MoFEMFunctionReturn(0);
    }
  };

  struct AssembleMatrix
      : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {
    AssembleMatrix(string row_name, string col_name)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              row_name, col_name, UserDataOperator::OPROWCOL) {}

    MatrixDouble K, transK;
    MoFEMErrorCode aSemble(int row_side, int col_side, EntityType row_type,
                           EntityType col_type,
                           EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data) {
      MoFEMFunctionBegin;
      int nb_row = row_data.getIndices().size();
      int nb_col = col_data.getIndices().size();
      int *row_indices_ptr = &row_data.getIndices()[0];
      int *col_indices_ptr = &col_data.getIndices()[0];
      CHKERR MatSetValues(getFEMethod()->ts_B, nb_row, row_indices_ptr, nb_col,
                          col_indices_ptr, &K(0, 0), ADD_VALUES);
      if (sYmm) {
        // Assemble of diagonal terms
        if (row_side != col_side || row_type != col_type) {
          transK.resize(nb_col, nb_row, false);
          noalias(transK) = trans(K);
          CHKERR MatSetValues(getFEMethod()->ts_B, nb_col, col_indices_ptr,
                              nb_row, row_indices_ptr, &transK(0, 0),
                              ADD_VALUES);
        }
      }
      MoFEMFunctionReturn(0);
    }
  };

  /** \brief Assemble matrix
   */
  struct OpLhsdxdx : public AssembleMatrix {
    CommonData &commonData;
    OpLhsdxdx(CommonData &common_data)
        : AssembleMatrix(common_data.spatialPositionName,
                         common_data.spatialPositionName),
          commonData(common_data) {}
    MatrixDouble dStress_dx;
    MoFEMErrorCode get_dStress_dx(EntitiesFieldData::EntData &col_data,
                                  int gg) {
      MoFEMFunctionBegin;
      int nb_col = col_data.getIndices().size();
      dStress_dx.resize(9, nb_col, false);
      dStress_dx.clear();
      const MatrixAdaptor diffN = col_data.getDiffN(gg, nb_col / 3);
      MatrixDouble &jac_stress = commonData.jacStress[gg];
      for (int dd = 0; dd < nb_col / 3; dd++) { // DoFs in column
        for (int jj = 0; jj < 3; jj++) {        // cont. DoFs in column
          double a = diffN(dd, jj);
          for (int rr = 0; rr < 3; rr++) { // Loop over dsigma_ii/dX_rr
            for (int ii = 0; ii < 9;
                 ii++) { // ii represents components of stress tensor
              dStress_dx(ii, 3 * dd + rr) += jac_stress(ii, 3 * rr + jj) * a;
            }
          }
        }
      }
      MoFEMFunctionReturn(0);
    }
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          EntitiesFieldData::EntData &row_data,
                          EntitiesFieldData::EntData &col_data) {
      MoFEMFunctionBegin;

      if (commonData.skipThis) {
        MoFEMFunctionReturnHot(0);
      }

      int nb_row = row_data.getIndices().size();
      int nb_col = col_data.getIndices().size();
      if (nb_row == 0)
        MoFEMFunctionReturnHot(0);
      if (nb_col == 0)
        MoFEMFunctionReturnHot(0);
      K.resize(nb_row, nb_col, false);
      K.clear();
      int nb_gauss_pts = row_data.getN().size1();
      for (int gg = 0; gg != nb_gauss_pts; gg++) {
        CHKERR get_dStress_dx(col_data, gg);
        double val = getVolume() * getGaussPts()(3, gg);
        // std::cerr << dStress_dx << std::endl;
        dStress_dx *= val;
        const MatrixAdaptor &diffN = row_data.getDiffN(gg, nb_row / 3);
        { // integrate element stiffness matrix
          for (int dd1 = 0; dd1 < nb_row / 3; dd1++) {
            for (int rr1 = 0; rr1 < 3; rr1++) {
              for (int dd2 = 0; dd2 < nb_col / 3; dd2++) {
                for (int rr2 = 0; rr2 < 3; rr2++) {
                  K(3 * dd1 + rr1, 3 * dd2 + rr2) +=
                      (diffN(dd1, 0) * dStress_dx(3 * rr1 + 0, 3 * dd2 + rr2) +
                       diffN(dd1, 1) * dStress_dx(3 * rr1 + 1, 3 * dd2 + rr2) +
                       diffN(dd1, 2) * dStress_dx(3 * rr1 + 2, 3 * dd2 + rr2));
                }
              }
            }
          }
        }
      }
      // std::cerr << "G " << getNumeredEntFiniteElementPtr()->getRefEnt() <<
      // std::endl << K << std::endl;
      CHKERR aSemble(row_side, col_side, row_type, col_type, row_data,
                     col_data);

      MoFEMFunctionReturn(0);
    }
  };

  /** \brief Assemble matrix
   */
  struct OpLhsdxdot : public AssembleMatrix {
    CommonData &commonData;
    OpLhsdxdot(CommonData &common_data)
        : AssembleMatrix(common_data.spatialPositionName,
                         common_data.spatialPositionName),
          commonData(common_data) {}
    MatrixDouble dStress_dot;
    MoFEMErrorCode get_dStress_dot(EntitiesFieldData::EntData &col_data,
                                   int gg) {
      MoFEMFunctionBegin;
      int nb_col = col_data.getIndices().size();
      dStress_dot.resize(9, nb_col, false);
      dStress_dot.clear();
      const MatrixAdaptor diffN = col_data.getDiffN(gg, nb_col / 3);
      MatrixDouble &jac_stress = commonData.jacStress[gg];
      for (int dd = 0; dd < nb_col / 3; dd++) { // DoFs in column
        for (int jj = 0; jj < 3; jj++) {        // cont. DoFs in column
          double a = diffN(dd, jj);
          for (int rr = 0; rr < 3; rr++) { // Loop over dsigma_ii/dX_rr
            for (int ii = 0; ii < 9;
                 ii++) { // ii represents components of stress tensor
              dStress_dot(ii, 3 * dd + rr) +=
                  jac_stress(ii, 9 + 3 * rr + jj) * a * getFEMethod()->ts_a;
            }
          }
        }
      }
      MoFEMFunctionReturn(0);
    }
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          EntitiesFieldData::EntData &row_data,
                          EntitiesFieldData::EntData &col_data) {
      MoFEMFunctionBegin;

      if (commonData.skipThis) {
        MoFEMFunctionReturnHot(0);
      }

      int nb_row = row_data.getIndices().size();
      int nb_col = col_data.getIndices().size();
      if (nb_row == 0)
        MoFEMFunctionReturnHot(0);
      if (nb_col == 0)
        MoFEMFunctionReturnHot(0);
      K.resize(nb_row, nb_col, false);
      K.clear();
      int nb_gauss_pts = row_data.getN().size1();
      for (int gg = 0; gg != nb_gauss_pts; gg++) {
        CHKERR get_dStress_dot(col_data, gg);
        double val = getVolume() * getGaussPts()(3, gg);
        // std::cerr << dStress_dot << std::endl;
        dStress_dot *= val;
        const MatrixAdaptor &diffN = row_data.getDiffN(gg, nb_row / 3);
        { // integrate element stiffness matrix
          for (int dd1 = 0; dd1 < nb_row / 3; dd1++) {
            for (int rr1 = 0; rr1 < 3; rr1++) {
              for (int dd2 = 0; dd2 < nb_col / 3; dd2++) {
                for (int rr2 = 0; rr2 < 3; rr2++) {
                  K(3 * dd1 + rr1, 3 * dd2 + rr2) +=
                      (diffN(dd1, 0) * dStress_dot(3 * rr1 + 0, 3 * dd2 + rr2) +
                       diffN(dd1, 1) * dStress_dot(3 * rr1 + 1, 3 * dd2 + rr2) +
                       diffN(dd1, 2) * dStress_dot(3 * rr1 + 2, 3 * dd2 + rr2));
                }
              }
            }
          }
        }
      }
      // std::cerr << "G " << getNumeredEntFiniteElementPtr()->getRefEnt() <<
      // std::endl << K << std::endl;
      CHKERR aSemble(row_side, col_side, row_type, col_type, row_data,
                     col_data);

      MoFEMFunctionReturn(0);
    }
  };

  MoFEMErrorCode setBlockDataMap() {
    MoFEMFunctionBegin;

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, 6, "DAMPER") == 0) {
        std::vector<double> data;
        CHKERR it->getAttributes(data);
        if (data.size() < 2) {
          SETERRQ(PETSC_COMM_SELF, 1, "Data inconsistency");
        }
        CHKERR mField.get_moab().get_entities_by_type(
            it->meshset, MBTET, blockMaterialDataMap[it->getMeshsetId()].tEts,
            true);
        blockMaterialDataMap[it->getMeshsetId()].gBeta = data[0];
        blockMaterialDataMap[it->getMeshsetId()].vBeta = data[1];
      }
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setOperators(const int tag) {
    MoFEMFunctionBegin;

    for (auto &&fe_ptr : {&feRhs, &feLhs}) {
      // CHKERR addHOOpsVol(commonData.meshNodePositionName, *fe_ptr, true, false,
      //                    false, false);
      fe_ptr->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              commonData.spatialPositionName,
              commonData.gradDataAtGaussTmpPtr));
      fe_ptr->getOpPtrVector().push_back(new OpGetDataAtGaussPts(
          commonData.spatialPositionName, commonData, false, true, false));
      fe_ptr->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradientDot<3, 3>(
              commonData.spatialPositionName,
              commonData.gradDataAtGaussTmpPtr));
      fe_ptr->getOpPtrVector().push_back(new OpGetDataAtGaussPts(
          commonData.spatialPositionName, commonData, false, true, true));
    }
       
    // attach tags for each recorder
    std::vector<int> tags;
    tags.push_back(tag);

    ConstitutiveEquationMap::iterator mit = constitutiveEquationMap.begin();
    for (; mit != constitutiveEquationMap.end(); mit++) {
      ConstitutiveEquation<adouble> &ce =
          constitutiveEquationMap.at(mit->first);
      // Right hand side operators
      feRhs.getOpPtrVector().push_back(new OpJacobian(
          commonData.spatialPositionName, tags, ce, commonData, true, false));
      feRhs.getOpPtrVector().push_back(new OpRhsStress(commonData));

      // Left hand side operators
      feLhs.getOpPtrVector().push_back(new OpJacobian(
          commonData.spatialPositionName, tags, ce, commonData, false, true));
      feLhs.getOpPtrVector().push_back(new OpLhsdxdx(commonData));
      feLhs.getOpPtrVector().push_back(new OpLhsdxdot(commonData));
    }

    MoFEMFunctionReturn(0);
  }
};

#endif //__KELVIN_VOIGT_DAMPER_HPP__
