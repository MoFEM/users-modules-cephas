/** \file SurfaceSlidingConstrains.hpp
 * \brief Implementing surface sliding constrains
 */



#ifndef __SURFACE_SLIDING_CONSTRAINS_HPP__
#define __SURFACE_SLIDING_CONSTRAINS_HPP__

#ifdef WITH_ADOL_C

/**
 * @brief Implementation of surface sliding constrains
 *
 */
struct GenericSliding {

  GenericSliding() = default;
  virtual ~GenericSliding() = default;

  struct OpGetActiveDofsLambda
      : public MoFEM::ForcesAndSourcesCore::UserDataOperator {
    boost::shared_ptr<VectorDouble> activeVariablesPtr;
    OpGetActiveDofsLambda(const std::string field_name,
                          boost::shared_ptr<VectorDouble> &active_variables_ptr)
        : MoFEM::ForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL),
          activeVariablesPtr(active_variables_ptr) {}
    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data) {
      MoFEMFunctionBegin;
      if (type == MBVERTEX) {
        for (unsigned int dd = 0; dd != data.getFieldData().size(); ++dd)
          (*activeVariablesPtr)[dd] = data.getFieldData()[dd];
      }
      MoFEMFunctionReturn(0);
    }
  };

  template <int SizeLambda>
  struct OpGetActiveDofsPositions
      : public MoFEM::ForcesAndSourcesCore::UserDataOperator {
    boost::shared_ptr<VectorDouble> activeVariablesPtr;
    OpGetActiveDofsPositions(
        const std::string field_name,
        boost::shared_ptr<VectorDouble> &active_variables_ptr)
        : MoFEM::ForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL),
          activeVariablesPtr(active_variables_ptr) {}
    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data) {
      MoFEMFunctionBegin;
      if (type == MBVERTEX) {
        for (unsigned int dd = 0; dd != data.getFieldData().size(); ++dd)
          (*activeVariablesPtr)[SizeLambda + dd] = data.getFieldData()[dd];
      }
      MoFEMFunctionReturn(0);
    }
  };

  template <int SizeLambda, int SizePositions>
  struct OpAssembleRhs : public MoFEM::ForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<VectorDouble> resultsPtr;

    OpAssembleRhs(const std::string field_name,
                  boost::shared_ptr<VectorDouble> &results_ptr)
        : MoFEM::ForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          resultsPtr(results_ptr) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data) {
      MoFEMFunctionBegin;
      if (type != MBVERTEX)
        MoFEMFunctionReturnHot(0);
      VectorInt &indices = data.getIndices();
      int shift = 0;
      if (indices.empty())
        MoFEMFunctionReturnHot(0);
      else if (indices.size() == SizeLambda)
        shift = 0;
      else if (indices.size() == SizePositions)
        shift = SizeLambda;
      else
        SETERRQ2(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                 "Element %s: Data inconsistency nb of indices %d",
                 getFEName().c_str(), indices.size());

      CHKERR VecSetOption(getSNESf(), VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
      CHKERR VecSetValues(getSNESf(), indices.size(), &indices[0],
                          &(*resultsPtr)[shift], ADD_VALUES);
      MoFEMFunctionReturn(0);
    }
  };
  template <int SizeLambda, int SizePositions>
  struct OpAssembleLhs : public MoFEM::ForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<MatrixDouble> jacobianPtr;

    OpAssembleLhs(const std::string field_name_row,
                  const std::string field_name_col,
                  boost::shared_ptr<MatrixDouble> &jacobian_ptr)
        : MoFEM::ForcesAndSourcesCore::UserDataOperator(
              field_name_row, field_name_col, UserDataOperator::OPROWCOL),
          jacobianPtr(jacobian_ptr) {
      sYmm = false;
    }

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          EntitiesFieldData::EntData &row_data,
                          EntitiesFieldData::EntData &col_data) {
      MoFEMFunctionBegin;
      if (row_type != MBVERTEX)
        MoFEMFunctionReturnHot(0);
      if (col_type != MBVERTEX)
        MoFEMFunctionReturnHot(0);
      VectorInt &row_indices = row_data.getIndices();
      VectorInt &col_indices = col_data.getIndices();
      if (row_indices.empty() || col_indices.empty())
        MoFEMFunctionReturnHot(0);

      int shift_row = 0;
      if (row_indices.size() == SizeLambda)
        shift_row = 0;
      else if (row_indices.size() == SizePositions)
        shift_row = SizeLambda;
      else
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                "Data inconsistency");

      int shift_col = 0;
      if (col_indices.size() == SizeLambda)
        shift_col = 0;
      else if (col_indices.size() == SizePositions)
        shift_col = SizeLambda;
      else
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                "Data inconsistency");

      MatrixDouble jac(row_indices.size(), col_indices.size());
      for (unsigned int rr = 0; rr != row_indices.size(); ++rr) {
        for (unsigned int cc = 0; cc != col_indices.size(); ++cc) {
          jac(rr, cc) = (*jacobianPtr)(shift_row + rr, shift_col + cc);
          if (jac(rr, cc) != jac(rr, cc))
            SETERRQ1(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                     "Jacobian assemble not a number jac(rr,cc) = %3.4f",
                     jac(rr, cc));
        }
      }
      CHKERR MatSetValues(getSNESB(), row_indices.size(), &row_indices[0],
                          col_indices.size(), &col_indices[0], &jac(0, 0),
                          ADD_VALUES);
      MoFEMFunctionReturn(0);
    }
  };
};

/** \brief Shape preserving constrains, i.e. nodes sliding on body surface.

  Derivation and implementation of constrains preserving body surface,
  i.e. body shape and volume.

  The idea starts form observation that body shape can be globally characterized
  by constant calculated as volume over its area
  \f[
  \frac{V}{A} = C
  \f]
  Above equation expressed in integral form is
  \f[
  \int_\Omega \textrm{d}V = C \int_\Gamma \textrm{d}S
  \f]
  where notting that,
  \f[
  \frac{1}{3}
  \int_\Omega \textrm{div}[\mathbf{X}] \textrm{d}\Omega
  =
  C \int_\Gamma  \textrm{d}S
  \f]
  and applying Gauss theorem we get
  \f[
  \int_\Gamma
  \mathbf{X}\cdot \frac{\mathbf{N}}{\|\mathbf{N}\|}
  \textrm{d}\Gamma
  =
  3C \int_\Gamma  \textrm{d}S.
  \f]
  Drooping integrals on both sides, and linearizing equation, we get
  \f[
  \frac{\mathbf{N}}{\|\mathbf{N}\|} \cdot \delta \mathbf{X}
  =
  3C - \frac{\mathbf{N}}{\|\mathbf{N}\|}\cdot \mathbf{X}
  \f]
  where \f$\delta \mathbf{X}\f$ is displacement sub-inctrement. Above equation
  is a constrain if satisfied in body shape and volume is conserved. Final form
  of constrain equation is \f[ \mathcal{r} =
  \frac{\mathbf{N}}{\|\mathbf{N}\|}\cdot \mathbf{X}
  -
  \frac{\mathbf{N_0}}{\|\mathbf{N_0}\|}\cdot \mathbf{X}_0 =
  \frac{\mathbf{N}}{\|\mathbf{N}\|}\cdot (\mathbf{X}-\mathbf{X}_0)
  \f]

  In the spirit of finite element method the constrain equation is multiplied
  by shape functions and enforce using Lagrange multiplier method
  \f[
  \int_\Gamma \mathbf{N}^\mathsf{T}_\lambda
   \left(
     \frac{\mathbf{N}}{\|\mathbf{N}\|}\mathbf{N}_\mathbf{X}\cdot
     (\overline{\mathbf{X}}-\overline{\mathbf{X}}_0)
  \right)
   \|\mathbf{N}\|
  \textrm{d}\Gamma
   =
  \mathbf{0}.
  \f]
  Above equation is nonlinear, applying to it Taylor expansion, we can get form
  which can be used with Newton interactive method \f[ \begin{split}
   &\int_\Gamma \mathbf{N}^\mathsf{T}_\lambda
    \left\{
    \mathbf{N}\mathbf{N}_\mathbf{X}
    +
    \left(\mathbf{X}-\mathbf{X}_0\right) \cdot
    \left(
    \textrm{Spin}\left[\frac{\partial\mathbf{X}}{\partial\xi}\right]\cdot\mathbf{B}_\eta
    -
    \textrm{Spin}\left[\frac{\partial\mathbf{X}}{\partial\eta}\right]\cdot\mathbf{B}_\xi
    \right)
    \right\}
    \textrm{d}\Gamma
    \cdot
    \delta
    \overline{\mathbf{X}}\\
    =
    &\int_\Gamma \mathbf{N}^\mathsf{T}_\lambda
    \mathbf{N}\cdot(\mathbf{X}-\mathbf{X}_0)
    \textrm{d}\Gamma
  \end{split}.
  \f]
  Equation expressing forces on shape as result of constrains, as result
  Lagrange multiplier method have following form \f[ \begin{split}
  &\int_\Gamma
  \mathbf{N}^\mathsf{T}_\mathbf{X} \cdot \mathbf{N}
  \mathbf{N}_\lambda
  \textrm{d}\Gamma
  \cdot
  \delta\overline{\lambda}\\
  +
  &\int_\Gamma
  \lambda
  \mathbf{N}^\mathsf{T}_\mathbf{X}
  \left(
  \textrm{Spin}\left[\frac{\partial\mathbf{X}}{\partial\xi}\right]\cdot\mathbf{B}_\eta
  -
  \textrm{Spin}\left[\frac{\partial\mathbf{X}}{\partial\eta}\right]\cdot\mathbf{B}_\xi
  \right)
  \textrm{d}\Gamma
  \delta\overline{\mathbf{X}}\\
  =
  &\int_\Gamma
  \lambda
  \mathbf{N}^\mathsf{T}_\mathbf{X} \cdot \mathbf{N}
  \textrm{d}\Gamma
  \end{split}
  \f]

  Above equations are assembled into global system of equations as following
  \f[
  \left[
    \begin{array}{cc}
        \mathbf{K} + \mathbf{B} & \mathbf{C}^\mathsf{T} \\
        \mathbf{C} + \mathbf{A} & 0
    \end{array}
  \right]
  \left\{
    \begin{array}{c}
      \delta \overline{\mathbf{X}} \\
      \delta \overline{\lambda}
    \end{array}
  \right\}=
  \left[
    \begin{array}{c}
      \mathbf{f} - \mathbf{C}^\mathsf{T}\overline{\lambda} \\
      \overline{\mathbf{r}}
    \end{array}
  \right]
  \f]
  where
  \f[
  \mathbf{C}=
  \int_\Gamma
  \mathbf{N}_\lambda^\mathsf{T}
   \mathbf{N} \cdot
   \mathbf{N}_\mathbf{X}
  \textrm{d}\Gamma,
  \f]
  \f[
  \mathbf{B}=
  \int_\Gamma
  \lambda
  (\mathbf{X}-\mathbf{X}_0)\cdot
  \left(
    \textrm{Spin}\left[\frac{\partial\mathbf{X}}{\partial\xi}\right]\cdot\mathbf{B}_\eta
    -
    \textrm{Spin}\left[\frac{\partial\mathbf{X}}{\partial\eta}\right]\cdot\mathbf{B}_\xi
  \right)
  \textrm{d}\Gamma
  \f]
  and
  \f[
  \mathbf{A}=
  \int_\Gamma
  \mathbf{N}^\mathsf{T}_\lambda
  \left(\mathbf{X}-\mathbf{X}_0\right) \cdot
  \left(
  \textrm{Spin}\left[\frac{\partial\mathbf{X}}{\partial\xi}\right]\cdot\mathbf{B}_\eta
  -
  \textrm{Spin}\left[\frac{\partial\mathbf{X}}{\partial\eta}\right]\cdot\mathbf{B}_\xi
  \right).
  \f]

*/
struct SurfaceSlidingConstrains : public GenericSliding {

  /** \brief Class implemented by user to detect face orientation

  If mesh generated is with surface mesher, usually you don't have to do
  nothing, all elements on the surface have consistent orientation. In case of
  internal faces or if you do something with mesh connectivity which breaks
  orientation on the face, you have to implement method which will set
  orientation to face.

  */
  struct DriverElementOrientation {

    int elementOrientation;

    virtual MoFEMErrorCode
    getElementOrientation(MoFEM::Interface &m_field,
                          const FEMethod *fe_method_ptr) {
      MoFEMFunctionBeginHot;
      elementOrientation = 1;
      MoFEMFunctionReturnHot(0);
    }
    virtual MoFEMErrorCode getElementOrientation(MoFEM::Interface &m_field,
                                                 const EntityHandle face) {
      MoFEMFunctionBeginHot;
      elementOrientation = 1;
      MoFEMFunctionReturnHot(0);
    }
  };

  MoFEM::Interface &mField;

  struct MyTriangleFE : public MoFEM::FaceElementForcesAndSourcesCore {

    Mat B;
    Vec F;

    MyTriangleFE(MoFEM::Interface &m_field)
        : MoFEM::FaceElementForcesAndSourcesCore(m_field), B(PETSC_NULL),
          F(PETSC_NULL) {}
    int getRule(int order) { return 2 * order; };

    MoFEMErrorCode preProcess() {
      MoFEMFunctionBegin;

      CHKERR MoFEM::FaceElementForcesAndSourcesCore::preProcess();

      if (B != PETSC_NULL) {
        snes_B = B;
      }

      if (F != PETSC_NULL) {
        snes_f = F;
      }

      MoFEMFunctionReturn(0);
    }
  };

  boost::shared_ptr<MyTriangleFE> feRhsPtr, feLhsPtr;

  MyTriangleFE &feRhs;
  MyTriangleFE &getLoopFeRhs() { return feRhs; }
  MyTriangleFE &feLhs;
  MyTriangleFE &getLoopFeLhs() { return feLhs; }

  DriverElementOrientation &crackFrontOrientation;

  double aLpha;
  MoFEMErrorCode getOptions() {
    MoFEMFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "",
                             "Get surface sliding constrains element scaling",
                             "none");
    CHKERRQ(ierr);
    CHKERR PetscOptionsScalar("-surface_sliding_alpha", "scaling parameter", "",
                              aLpha, &aLpha, PETSC_NULL);
    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);
    MoFEMFunctionReturn(0);
  }

  SurfaceSlidingConstrains(MoFEM::Interface &m_field,
                           DriverElementOrientation &orientation)
      : mField(m_field), feRhsPtr(new MyTriangleFE(m_field)),
        feLhsPtr(new MyTriangleFE(m_field)), feRhs(*feRhsPtr), feLhs(*feLhsPtr),
        crackFrontOrientation(orientation), aLpha(1) {

    ierr = getOptions();
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }

  virtual ~SurfaceSlidingConstrains() = default;

  struct OpJacobian
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    const int tAg;
    boost::shared_ptr<VectorDouble> activeVariablesPtr;
    boost::shared_ptr<VectorDouble> resultsPtr;
    boost::shared_ptr<MatrixDouble> jacobianPtr;
    DriverElementOrientation &oRientation;
    bool evaluateJacobian;

    const double &aLpha;

    OpJacobian(int tag, const std::string field_name,
               boost::shared_ptr<VectorDouble> &active_variables_ptr,
               boost::shared_ptr<VectorDouble> &results_ptr,
               boost::shared_ptr<MatrixDouble> &jacobian_ptr,
               DriverElementOrientation &orientation, bool evaluate_jacobian,
               const double &alpha)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL),
          tAg(tag), activeVariablesPtr(active_variables_ptr),
          resultsPtr(results_ptr), jacobianPtr(jacobian_ptr),
          oRientation(orientation), evaluateJacobian(evaluate_jacobian),
          aLpha(alpha) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data) {
      MoFEMFunctionBegin;
      if (type != MBVERTEX)
        MoFEMFunctionReturnHot(0);

      VectorInt &indices = data.getIndices();

      trace_on(tAg);

      ublas::vector<adouble> lambda_dofs(3);
      for (int dd = 0; dd != 3; ++dd) {
        lambda_dofs[dd] <<= (*activeVariablesPtr)[dd];
      }
      ublas::vector<adouble> position_dofs(9);
      for (int dd = 0; dd != 9; ++dd) {
        position_dofs[dd] <<= (*activeVariablesPtr)[3 + dd];
      }

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      FTensor::Number<0> N0;
      FTensor::Number<1> N1;
      FTensor::Number<2> N2;

      CHKERR oRientation.getElementOrientation(getFaceFE()->mField,
                                               getFEMethod());
      int eo = oRientation.elementOrientation;

      int nb_gauss_pts = data.getN().size1();
      int nb_base_functions = data.getN().size2();
      auto t_base1 = data.getFTensor0N();
      auto t_base2 = data.getFTensor0N();
      auto t_diff_base = data.getFTensor1DiffN<2>();
      FTensor::Tensor1<adouble, 3> t_position;
      FTensor::Tensor1<adouble, 3> t_position_ksi;
      FTensor::Tensor1<adouble, 3> t_position_eta;
      adouble lambda;
      FTensor::Tensor1<adouble, 3> t_normal;
      FTensor::Tensor1<adouble, 3> t_delta;

      ublas::vector<adouble> c_vec(3);
      ublas::vector<adouble> f_vec(9);
      c_vec.clear();
      f_vec.clear();

      auto t_coord_ref = getFTensor1CoordsAtGaussPts();

      for (int gg = 0; gg != nb_gauss_pts; ++gg) {

        FTensor::Tensor1<adouble *, 3> t_position_dofs(
            &position_dofs[0], &position_dofs[1], &position_dofs[2], 3);
        FTensor::Tensor0<adouble *> t_lambda_dof(&lambda_dofs[0]);

        t_position(i) = 0;
        t_position_ksi(i) = 0;
        t_position_eta(i) = 0;
        lambda = 0;

        for (int bb = 0; bb != nb_base_functions; ++bb) {
          t_position(i) += t_base1 * t_position_dofs(i);
          t_position_ksi(i) += t_diff_base(N0) * t_position_dofs(i);
          t_position_eta(i) += t_diff_base(N1) * t_position_dofs(i);
          lambda += t_base1 * t_lambda_dof;
          ++t_base1;
          ++t_position_dofs;
          ++t_lambda_dof;
          ++t_diff_base;
        }

        t_delta(i) = t_position(i) - t_coord_ref(i);
        t_normal(k) = FTensor::cross(t_position_ksi(i), t_position_eta(j), k);

        double w = getGaussPts()(2, gg) * 0.5 * aLpha;
        adouble val;
        FTensor::Tensor0<adouble *> t_c(&c_vec[0]);
        FTensor::Tensor1<adouble *, 3> t_f(&f_vec[0], &f_vec[1], &f_vec[2], 3);

        adouble area = sqrt(t_normal(i) * t_normal(i));

        for (int bb = 0; bb != nb_base_functions; ++bb) {
          if (indices[bb] != -1) {
            val = w * eo * t_base2;
            t_c += val * t_normal(i) * t_delta(i);
            val *= lambda;
            t_f(i) += val * t_normal(i);
          }
          ++t_c;
          ++t_f;
          ++t_base2;
        }

        ++t_coord_ref;
      }

      for (int rr = 0; rr != 3; ++rr)
        c_vec[rr] >>= (*resultsPtr)[rr];

      for (int rr = 0; rr != 9; ++rr)
        f_vec(rr) >>= (*resultsPtr)[3 + rr];

      trace_off();

      if (evaluateJacobian) {
        double *jac_ptr[3 + 9];
        for (int rr = 0; rr != 3 + 9; ++rr)
          jac_ptr[rr] = &(*jacobianPtr)(rr, 0);

        // play recorder for jacobians
        int r =
            ::jacobian(tAg, 3 + 9, 3 + 9, &(*activeVariablesPtr)[0], jac_ptr);
        if (r < 0)
          SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
                  "ADOL-C function evaluation with error");
      }

      MoFEMFunctionReturn(0);
    }
  };

  MoFEMErrorCode setOperators(int tag,
                              const std::string lagrange_multipliers_field_name,
                              const std::string material_field_name,
                              const double *alpha = nullptr) {
    MoFEMFunctionBegin;

    if (alpha != nullptr) {
      aLpha = *alpha;
    }

    boost::shared_ptr<VectorDouble> active_variables_ptr(
        new VectorDouble(3 + 9));
    boost::shared_ptr<VectorDouble> results_ptr(new VectorDouble(3 + 9));
    boost::shared_ptr<MatrixDouble> jacobian_ptr(
        new MatrixDouble(3 + 9, 3 + 9));

    feRhs.getOpPtrVector().clear();
    feRhs.getOpPtrVector().push_back(new OpGetActiveDofsLambda(
        lagrange_multipliers_field_name, active_variables_ptr));
    feRhs.getOpPtrVector().push_back(new OpGetActiveDofsPositions<3>(
        material_field_name, active_variables_ptr));
    feRhs.getOpPtrVector().push_back(new OpJacobian(
        tag, lagrange_multipliers_field_name, active_variables_ptr, results_ptr,
        jacobian_ptr, crackFrontOrientation, false, aLpha));
    feRhs.getOpPtrVector().push_back(
        new OpAssembleRhs<3, 9>(lagrange_multipliers_field_name, results_ptr));
    feRhs.getOpPtrVector().push_back(
        new OpAssembleRhs<3, 9>(material_field_name, results_ptr));

    // Adding operators to calculate the left hand side
    feLhs.getOpPtrVector().clear();
    feLhs.getOpPtrVector().push_back(new OpGetActiveDofsLambda(
        lagrange_multipliers_field_name, active_variables_ptr));
    feLhs.getOpPtrVector().push_back(new OpGetActiveDofsPositions<3>(
        material_field_name, active_variables_ptr));
    feLhs.getOpPtrVector().push_back(new OpJacobian(
        tag, lagrange_multipliers_field_name, active_variables_ptr, results_ptr,
        jacobian_ptr, crackFrontOrientation, true, aLpha));
    feLhs.getOpPtrVector().push_back(new OpAssembleLhs<3, 9>(
        lagrange_multipliers_field_name, material_field_name, jacobian_ptr));
    feLhs.getOpPtrVector().push_back(new OpAssembleLhs<3, 9>(
        material_field_name, lagrange_multipliers_field_name, jacobian_ptr));
    feLhs.getOpPtrVector().push_back(new OpAssembleLhs<3, 9>(
        material_field_name, material_field_name, jacobian_ptr));

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode
  setOperatorsConstrainOnly(int tag,
                            const std::string lagrange_multipliers_field_name,
                            const std::string material_field_name) {
    MoFEMFunctionBegin;

    boost::shared_ptr<VectorDouble> active_variables_ptr(
        new VectorDouble(3 + 9));
    boost::shared_ptr<VectorDouble> results_ptr(new VectorDouble(3 + 9));
    boost::shared_ptr<MatrixDouble> jacobian_ptr(
        new MatrixDouble(3 + 9, 3 + 9));

    // Adding operators to calculate the left hand side
    feLhs.getOpPtrVector().clear();
    feLhs.getOpPtrVector().push_back(new OpGetActiveDofsLambda(
        lagrange_multipliers_field_name, active_variables_ptr));
    feLhs.getOpPtrVector().push_back(new OpGetActiveDofsPositions<3>(
        material_field_name, active_variables_ptr));
    feLhs.getOpPtrVector().push_back(new OpJacobian(
        tag, lagrange_multipliers_field_name, active_variables_ptr, results_ptr,
        jacobian_ptr, crackFrontOrientation, true, aLpha));
    feLhs.getOpPtrVector().push_back(new OpAssembleLhs<3, 9>(
        lagrange_multipliers_field_name, material_field_name, jacobian_ptr));

    MoFEMFunctionReturn(0);
  }
};

struct EdgeSlidingConstrains : public GenericSliding {

  struct CalculateEdgeBase {

    static MoFEMErrorCode createTag(moab::Interface &moab, Tag &th0, Tag &th1,
                                    Tag &th2, Tag &th3) {
      MoFEMFunctionBegin;
      double def_val[] = {0, 0, 0};
      CHKERR moab.tag_get_handle("EDGE_BASE0", 3, MB_TYPE_DOUBLE, th0,
                                 MB_TAG_CREAT | MB_TAG_SPARSE, def_val);
      CHKERR moab.tag_get_handle("EDGE_BASE1", 3, MB_TYPE_DOUBLE, th1,
                                 MB_TAG_CREAT | MB_TAG_SPARSE, def_val);
      int def_numb_val[] = {-1};
      CHKERR moab.tag_get_handle("PATCH_NUMBER", 1, MB_TYPE_INTEGER, th2,
                                 MB_TAG_CREAT | MB_TAG_SPARSE, def_numb_val);
      int def_orientation_val[] = {1};
      CHKERR moab.tag_get_handle("PATCH_ORIENTATION", 1, MB_TYPE_INTEGER, th3,
                                 MB_TAG_CREAT | MB_TAG_SPARSE,
                                 def_orientation_val);

      MoFEMFunctionReturn(0);
    }

    static MoFEMErrorCode numberSurfaces(moab::Interface &moab, Range edges,
                                         Range tris) {
      MoFEMFunctionBegin;

      auto get_edges = [&](const Range &ents) {
        Range edges;
        CHKERR moab.get_adjacencies(ents, 1, false, edges,
                                    moab::Interface::UNION);
        return edges;
      };

      auto get_face_adj = [&, get_edges](const Range &faces) {
        Range adj_faces;
        CHKERR moab.get_adjacencies(subtract(get_edges(faces), edges), 2, false,
                                    adj_faces, moab::Interface::UNION);
        return intersect(adj_faces, tris);
      };

      auto get_patch = [&, get_face_adj](const EntityHandle face) {
        Range patch_ents;
        patch_ents.insert(face);
        unsigned int nb0;
        do {
          nb0 = patch_ents.size();
          patch_ents.merge(get_face_adj(patch_ents));
        } while (nb0 != patch_ents.size());
        return patch_ents;
      };

      auto get_patches = [&]() {
        std::vector<Range> patches;
        while (!tris.empty()) {
          patches.push_back(get_patch(tris[0]));
          tris = subtract(tris, patches.back());
        }
        return patches;
      };

      Tag th0, th1, th2, th3;
      CHKERR createTag(moab, th0, th1, th2, th3);

      auto patches = get_patches();
      int pp = 0;
      for (auto patch : patches) {
        // cerr << "pp: " << pp << endl;
        // cerr << patch << endl;
        std::vector<int> tags_vals(patch.size(), pp);
        CHKERR moab.tag_set_data(th2, patch, &*tags_vals.begin());
        ++pp;
      }

      MoFEMFunctionReturn(0);
    }

    static MoFEMErrorCode setTags(
        moab::Interface &moab, Range edges, Range tris,
        bool number_pathes = true,
        boost::shared_ptr<SurfaceSlidingConstrains::DriverElementOrientation>
            surface_orientation = nullptr,
        MoFEM::Interface *m_field_ptr = nullptr) {
      MoFEMFunctionBegin;
      Tag th0, th1, th2, th3;
      CHKERR createTag(moab, th0, th1, th2, th3);
      if (number_pathes) {
        CHKERR numberSurfaces(moab, edges, tris);
      }
      for (auto edge : edges) {
        Range adj_faces;
        CHKERR moab.get_adjacencies(&edge, 1, 2, false, adj_faces);
        adj_faces = intersect(adj_faces, tris);
        if (adj_faces.size() != 2) {
          SETERRQ1(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                   "Should be 2 faces adjacent to edge but is %d",
                   adj_faces.size());
        }
        VectorDouble3 v[2] = {VectorDouble3(3), VectorDouble3(3)};
        auto get_tensor_from_vec = [](VectorDouble3 &v) {
          return FTensor::Tensor1<FTensor::PackPtr<double *, 1>, 3>(
              &v[0], &v[1], &v[2]);
        };

        FTensor::Index<'i', 3> i;
        FTensor::Index<'j', 3> j;
        FTensor::Index<'k', 3> k;

        std::array<double, 3> areas;
        auto calculate_normals = [&, get_tensor_from_vec]() {
          int ff = 0;
          for (auto face : adj_faces) {
            double &x = (v[ff])[0];
            double &y = (v[ff])[1];
            double &z = (v[ff])[2];
            moab::Util::normal(&moab, face, x, y, z);
            auto t_n = get_tensor_from_vec(v[ff]);
            areas[ff] = sqrt(t_n(i) * t_n(i));
            t_n(i) /= areas[ff];
            int orientation;
            // FIXME: this tag is empty
            CHKERR moab.tag_get_data(th3, &face, 1, &orientation);
            if (orientation == -1) {
              t_n(i) *= -1;
            }
            if (surface_orientation && m_field_ptr) {
              CHKERR surface_orientation->getElementOrientation(*m_field_ptr,
                                                                face);
              int eo = surface_orientation->elementOrientation;
              if (eo == -1) {
                t_n(i) *= -1;
              }
            }
            ++ff;
          }
        };
        calculate_normals();

        auto get_patch_number = [&]() {
          std::vector<int> p = {0, 0};
          CHKERR moab.tag_get_data(th2, adj_faces, &*p.begin());
          return p;
        };

        std::array<EntityHandle, 2> faces_handles = {adj_faces[0],
                                                     adj_faces[1]};
        auto order_normals = [&, get_patch_number]() {
          auto p = get_patch_number();
          if (p[0] < p[1]) {
            v[0].swap(v[1]);
            faces_handles[0] = adj_faces[1];
            faces_handles[1] = adj_faces[0];
          }
        };
        order_normals();

        FTensor::Tensor1<double, 3> t_cross;
        auto t_n0 = get_tensor_from_vec(v[0]);
        auto t_n1 = get_tensor_from_vec(v[1]);
        t_cross(k) = FTensor::cross(t_n0(i), t_n1(j), k);

        auto get_base_for_coplanar_faces = [&]() {
          MoFEMFunctionBegin;

          std::vector<EntityHandle> face_conn;
          CHKERR moab.get_connectivity(&faces_handles[1], 1, face_conn, true);
          std::vector<EntityHandle> edge_conn;
          CHKERR moab.get_connectivity(&edge, 1, edge_conn, true);
          std::sort(face_conn.begin(), face_conn.end());
          std::sort(edge_conn.begin(), edge_conn.end());
          int n = 0;
          for (; n != 2; ++n)
            if (face_conn[n] != edge_conn[n])
              break;
          VectorDouble3 coords_face(3);
          CHKERR moab.get_coords(&face_conn[n], 1,
                                 &*coords_face.data().begin());
          VectorDouble6 coords_edge(6);
          CHKERR moab.get_coords(&edge_conn[0], 2,
                                 &*coords_edge.data().begin());
          auto t_edge_n0 = FTensor::Tensor1<double, 3>(
              coords_edge[0], coords_edge[1], coords_edge[2]);
          auto t_edge_n1 = FTensor::Tensor1<double, 3>(
              coords_edge[3], coords_edge[4], coords_edge[5]);
          auto t_face_n = get_tensor_from_vec(coords_face);
          t_face_n(i) -= t_edge_n0(i);
          FTensor::Tensor1<double, 3> t_delta;
          t_delta(i) = t_edge_n1(i) - t_edge_n0(i);
          t_delta(i) /= sqrt(t_delta(i) * t_delta(i));
          t_n1(k) = FTensor::cross(t_n0(i), t_delta(j), k);
          if (t_n1(i) * t_face_n(i) < 0)
            t_n1(i) *= -1;

          MoFEMFunctionReturn(0);
        };

        auto get_base_for_not_planar_faces = [&]() {
          MoFEMFunctionBeginHot;
          t_n1(k) = FTensor::cross(t_n0(i), t_cross(j), k);
          t_n1(i) /= sqrt(t_n1(i) * t_n1(i));
          MoFEMFunctionReturnHot(0);
        };

        // This a case when faces adjacent to edge are coplanar !!!
        constexpr double tol = 1e-6;
        if ((t_cross(i) * t_cross(i)) < tol)
          CHKERR get_base_for_coplanar_faces();
        else
          CHKERR get_base_for_not_planar_faces();

        VectorDouble3 &v0 = v[0];
        VectorDouble3 &v1 = v[1];
        CHKERR moab.tag_set_data(th0, &edge, 1, &v0[0]);
        CHKERR moab.tag_set_data(th1, &edge, 1, &v1[0]);
      }
      MoFEMFunctionReturn(0);
    }

    static MoFEMErrorCode saveEdges(moab::Interface &moab, std::string name,
                                    Range edges, Range *faces = nullptr) {
      MoFEMFunctionBegin;
      EntityHandle meshset;
      Tag ths[4];
      CHKERR createTag(moab, ths[0], ths[1], ths[2], ths[3]);
      CHKERR moab.create_meshset(MESHSET_SET, meshset);
      CHKERR moab.add_entities(meshset, edges);
      if (faces != nullptr) {
        CHKERR moab.add_entities(meshset, *faces);
      }
      CHKERR moab.write_file(name.c_str(), "VTK", "", &meshset, 1, ths, 4);
      CHKERR moab.delete_entities(&meshset, 1);
      MoFEMFunctionReturn(0);
    }
  };

  MoFEM::Interface &mField;

  struct MyEdgeFE : public MoFEM::EdgeElementForcesAndSourcesCore {

    Mat B;
    Vec F;

    MyEdgeFE(MoFEM::Interface &m_field)
        : MoFEM::EdgeElementForcesAndSourcesCore(m_field), B(PETSC_NULL),
          F(PETSC_NULL) {}
    int getRule(int order) { return 2 * order; };

    MoFEMErrorCode preProcess() {
      MoFEMFunctionBegin;

      CHKERR MoFEM::EdgeElementForcesAndSourcesCore::preProcess();

      if (B != PETSC_NULL) {
        snes_B = B;
      }

      if (F != PETSC_NULL) {
        snes_f = F;
      }

      MoFEMFunctionReturn(0);
    }
  };

  boost::shared_ptr<MyEdgeFE> feRhsPtr, feLhsPtr;

  MyEdgeFE &feRhs;
  MyEdgeFE &getLoopFeRhs() { return feRhs; }
  MyEdgeFE &feLhs;
  MyEdgeFE &getLoopFeLhs() { return feLhs; }

  double aLpha;

  MoFEMErrorCode getOptions() {
    MoFEMFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "",
                             "Get edge sliding constrains element scaling",
                             "none");
    CHKERRQ(ierr);
    CHKERR PetscOptionsScalar("-edge_sliding_alpha", "scaling parameter", "",
                              aLpha, &aLpha, PETSC_NULL);
    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);
    MoFEMFunctionReturn(0);
  }

  EdgeSlidingConstrains(MoFEM::Interface &m_field)
      : mField(m_field), feRhsPtr(new MyEdgeFE(m_field)),
        feLhsPtr(new MyEdgeFE(m_field)), feRhs(*feRhsPtr), feLhs(*feLhsPtr),
        aLpha(1) {
    ierr = getOptions();
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }

  struct OpJacobian
      : public MoFEM::EdgeElementForcesAndSourcesCore::UserDataOperator {

    const int tAg;
    boost::shared_ptr<VectorDouble> activeVariablesPtr;
    boost::shared_ptr<VectorDouble> resultsPtr;
    boost::shared_ptr<MatrixDouble> jacobianPtr;
    bool evaluateJacobian;

    const double &aLpha;

    OpJacobian(int tag, const std::string field_name,
               boost::shared_ptr<VectorDouble> &active_variables_ptr,
               boost::shared_ptr<VectorDouble> &results_ptr,
               boost::shared_ptr<MatrixDouble> &jacobian_ptr,
               bool evaluate_jacobian, const double &alpha)
        : MoFEM::EdgeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL),
          tAg(tag), activeVariablesPtr(active_variables_ptr),
          resultsPtr(results_ptr), jacobianPtr(jacobian_ptr),
          evaluateJacobian(evaluate_jacobian), aLpha(alpha) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data) {
      MoFEMFunctionBegin;
      if (type != MBVERTEX)
        MoFEMFunctionReturnHot(0);

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 2> j;
      FTensor::Number<0> N0;
      FTensor::Number<1> N1;

      Tag th0, th1, th2, th3;
      CHKERR CalculateEdgeBase::createTag(getEdgeFE()->mField.get_moab(), th0,
                                          th1, th2, th3);
      FTensor::Tensor1<double, 3> t_edge_base0, t_edge_base1;
      EntityHandle fe_ent = getFEEntityHandle();
      CHKERR getEdgeFE()->mField.get_moab().tag_get_data(th0, &fe_ent, 1,
                                                         &t_edge_base0(0));
      CHKERR getEdgeFE()->mField.get_moab().tag_get_data(th1, &fe_ent, 1,
                                                         &t_edge_base1(0));

      VectorInt &indices = data.getIndices();

      trace_on(tAg);

      ublas::vector<adouble> lambda_dofs(4);
      for (int dd = 0; dd != 4; ++dd) {
        lambda_dofs[dd] <<= (*activeVariablesPtr)[dd];
      }
      ublas::vector<adouble> position_dofs(6);
      for (int dd = 0; dd != 6; ++dd) {
        position_dofs[dd] <<= (*activeVariablesPtr)[4 + dd];
      }

      FTensor::Tensor1<adouble *, 3> t_node0(
          &position_dofs[0], &position_dofs[1], &position_dofs[2]);
      FTensor::Tensor1<adouble *, 3> t_node1(
          &position_dofs[3], &position_dofs[4], &position_dofs[5]);

      FTensor::Tensor1<adouble, 3> t_tangent;
      t_tangent(i) = t_node1(i) - t_node0(i);
      adouble l = sqrt(t_tangent(i) * t_tangent(i));
      t_tangent(i) /= l;

      adouble t_dot0, t_dot1;
      t_dot0 = t_edge_base0(i) * t_tangent(i);
      t_dot1 = t_edge_base1(i) * t_tangent(i);

      FTensor::Tensor1<adouble, 3> t_base0, t_base1;
      t_base0(i) = t_edge_base0(i) - t_dot0 * t_tangent(i);
      t_base1(i) = t_edge_base1(i) - t_dot1 * t_tangent(i);
      t_base0(i) /= sqrt(t_base0(i) * t_base0(i));
      t_base1(i) /= sqrt(t_base1(i) * t_base1(i));

      auto t_base_fun1 = data.getFTensor0N();
      auto t_base_fun2 = data.getFTensor0N();
      FTensor::Tensor1<adouble, 3> t_position;
      FTensor::Tensor1<adouble, 2> t_lambda;
      FTensor::Tensor1<adouble, 3> t_delta;
      auto t_coord_ref = getFTensor1CoordsAtGaussPts();

      ublas::vector<adouble> c_vec(4);
      ublas::vector<adouble> f_vec(6);
      c_vec.clear();
      f_vec.clear();

      int nb_gauss_pts = data.getN().size1();
      int nb_base_functions = data.getN().size2();
      for (int gg = 0; gg != nb_gauss_pts; ++gg) {

        FTensor::Tensor1<adouble *, 3> t_position_dofs(
            &position_dofs[0], &position_dofs[1], &position_dofs[2], 3);
        FTensor::Tensor1<adouble *, 2> t_lambda_dof(&lambda_dofs[0],
                                                    &lambda_dofs[1], 2);

        t_position(i) = 0;
        t_lambda(j) = 0;
        for (int bb = 0; bb != nb_base_functions; ++bb) {
          t_position(i) += t_base_fun1 * t_position_dofs(i);
          t_lambda(j) += t_base_fun1 * t_lambda_dof(j);
          ++t_base_fun1;
          ++t_position_dofs;
          ++t_lambda_dof;
        }

        t_delta(i) = t_position(i) - t_coord_ref(i);
        adouble dot0 = t_base0(i) * t_delta(i);
        adouble dot1 = t_base1(i) * t_delta(i);

        adouble w = getGaussPts()(1, gg) * l * aLpha;
        adouble val, val1, val2;
        FTensor::Tensor1<adouble *, 2> t_c(&c_vec[0], &c_vec[1], 2);
        FTensor::Tensor1<adouble *, 3> t_f(&f_vec[0], &f_vec[1], &f_vec[2], 3);
        for (int bb = 0; bb != nb_base_functions; ++bb) {
          if (indices[2 * bb] != -1) {
            val = w * t_base_fun2;
            t_c(N0) += val * dot0;
            t_c(N1) += val * dot1;
            val1 = val * t_lambda(N0);
            val2 = val * t_lambda(N1);
            t_f(i) += val1 * t_base0(i) + val2 * t_base1(i);
          }
          ++t_c;
          ++t_f;
          ++t_base_fun2;
        }

        ++t_coord_ref;
      }

      for (int rr = 0; rr != 4; ++rr) {
        c_vec[rr] >>= (*resultsPtr)[rr];
      }
      for (int rr = 0; rr != 6; ++rr) {
        f_vec(rr) >>= (*resultsPtr)[4 + rr];
      }

      trace_off();

      if (evaluateJacobian) {
        double *jac_ptr[4 + 6];
        for (int rr = 0; rr != 4 + 6; ++rr) {
          jac_ptr[rr] = &(*jacobianPtr)(rr, 0);
        }
        // play recorder for jacobians
        int r =
            ::jacobian(tAg, 4 + 6, 4 + 6, &(*activeVariablesPtr)[0], jac_ptr);
        if (r < 0) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
                  "ADOL-C function evaluation with error");
        }
      }

      MoFEMFunctionReturn(0);
    }
  };

  MoFEMErrorCode setOperators(int tag, Range edges, Range faces,
                              const std::string lagrange_multipliers_field_name,
                              const std::string material_field_name) {
    MoFEMFunctionBegin;
    CHKERR EdgeSlidingConstrains::CalculateEdgeBase::setTags(mField.get_moab(),
                                                             edges, faces);
    CHKERR setOperators(tag, lagrange_multipliers_field_name,
                        material_field_name);
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setOperators(int tag,
                              const std::string lagrange_multipliers_field_name,
                              const std::string material_field_name,
                              const double *alpha = nullptr) {
    MoFEMFunctionBegin;

    if (alpha) {
      aLpha = *alpha;
    }

    boost::shared_ptr<VectorDouble> active_variables_ptr(
        new VectorDouble(4 + 6));
    boost::shared_ptr<VectorDouble> results_ptr(new VectorDouble(4 + 6));
    boost::shared_ptr<MatrixDouble> jacobian_ptr(
        new MatrixDouble(4 + 6, 4 + 6));

    feRhs.getOpPtrVector().clear();
    feRhs.getOpPtrVector().push_back(new OpGetActiveDofsLambda(
        lagrange_multipliers_field_name, active_variables_ptr));
    feRhs.getOpPtrVector().push_back(new OpGetActiveDofsPositions<4>(
        material_field_name, active_variables_ptr));
    feRhs.getOpPtrVector().push_back(new OpJacobian(
        tag, lagrange_multipliers_field_name, active_variables_ptr, results_ptr,
        jacobian_ptr, false, aLpha));
    feRhs.getOpPtrVector().push_back(
        new OpAssembleRhs<4, 6>(lagrange_multipliers_field_name, results_ptr));
    feRhs.getOpPtrVector().push_back(
        new OpAssembleRhs<4, 6>(material_field_name, results_ptr));

    // Adding operators to calculate the left hand side
    feLhs.getOpPtrVector().clear();
    feLhs.getOpPtrVector().push_back(new OpGetActiveDofsLambda(
        lagrange_multipliers_field_name, active_variables_ptr));
    feLhs.getOpPtrVector().push_back(new OpGetActiveDofsPositions<4>(
        material_field_name, active_variables_ptr));
    feLhs.getOpPtrVector().push_back(new OpJacobian(
        tag, lagrange_multipliers_field_name, active_variables_ptr, results_ptr,
        jacobian_ptr, true, aLpha));
    feLhs.getOpPtrVector().push_back(new OpAssembleLhs<4, 6>(
        lagrange_multipliers_field_name, material_field_name, jacobian_ptr));
    feLhs.getOpPtrVector().push_back(new OpAssembleLhs<4, 6>(
        material_field_name, lagrange_multipliers_field_name, jacobian_ptr));
    feLhs.getOpPtrVector().push_back(new OpAssembleLhs<4, 6>(
        material_field_name, material_field_name, jacobian_ptr));

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode
  setOperatorsConstrainOnly(int tag, Range edges, Range faces,
                            const std::string lagrange_multipliers_field_name,
                            const std::string material_field_name) {
    MoFEMFunctionBegin;

    CHKERR EdgeSlidingConstrains::CalculateEdgeBase::setTags(mField.get_moab(),
                                                             edges, faces);
    CHKERR setOperatorsConstrainOnly(tag, lagrange_multipliers_field_name,
                                     material_field_name);
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode
  setOperatorsConstrainOnly(int tag,
                            const std::string lagrange_multipliers_field_name,
                            const std::string material_field_name) {
    MoFEMFunctionBegin;

    boost::shared_ptr<VectorDouble> active_variables_ptr(
        new VectorDouble(4 + 6));
    boost::shared_ptr<VectorDouble> results_ptr(new VectorDouble(4 + 6));
    boost::shared_ptr<MatrixDouble> jacobian_ptr(
        new MatrixDouble(4 + 6, 4 + 6));

    // Adding operators to calculate the left hand side
    feLhs.getOpPtrVector().clear();
    feLhs.getOpPtrVector().push_back(new OpGetActiveDofsLambda(
        lagrange_multipliers_field_name, active_variables_ptr));
    feLhs.getOpPtrVector().push_back(new OpGetActiveDofsPositions<4>(
        material_field_name, active_variables_ptr));
    feLhs.getOpPtrVector().push_back(new OpJacobian(
        tag, lagrange_multipliers_field_name, active_variables_ptr, results_ptr,
        jacobian_ptr, true, aLpha));
    feLhs.getOpPtrVector().push_back(new OpAssembleLhs<4, 6>(
        lagrange_multipliers_field_name, material_field_name, jacobian_ptr));

    MoFEMFunctionReturn(0);
  }
};

#endif // WITH_ADOL_C

#endif // __SURFACE_SLIDING_CONSTRAINS_HPP__
