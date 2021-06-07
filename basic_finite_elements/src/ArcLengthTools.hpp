/** \file ArcLengthTools.cpp
 * \ingroup arc_length_control
 *
 * Implementation of arc-length control method
 *
 * FIXME: Some variables not comply with naming convention, need to be fixed.
 *
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

#ifndef __ARC_LENGTH_TOOLS_HPP__
#define __ARC_LENGTH_TOOLS_HPP__

/**
 * \brief Store variables for ArcLength analysis
 *
 * \ingroup arc_length_control

 The constrain function if given by
 \f[
 r_\lambda = f_\lambda(\mathbf{x},\lambda) - s^2
 \f]
 where \f$f_\lambda(\mathbf{x},\lambda)\f$ is some constrain function, which has
 general form given by
 \f[
 f_\lambda(\mathbf{x},\lambda) = \alpha f(\Delta\mathbf{x}) +
 \beta^2 \Delta\lambda^2 \| \mathbf{F}_{\lambda} \|^2
 \f]
 where for example \f$f(\mathbf{x})=\|\Delta\mathbf{x}\|^2\f$ is some user
 defined
 function evaluating
 increments vector of degrees of freedom  \f$\Delta\mathbf{x}\f$. The increment
 vector is
 \f[
 \Delta \mathbf{x} = \mathbf{x}-\mathbf{x}_0
 \f]
 and
 \f[
 \Delta \lambda = \lambda-\lambda_0.
 \f]

 For convenience we assume that
 \f[
 \frac{\partial f}{\partial \mathbf{x}}\Delta \mathbf{x}
 =
 \textrm{d}\mathbf{b} \Delta \mathbf{x},
 \f]
 as result linearised constrain equation takes form
 \f[
 \textrm{d}\mathbf{b} \delta \Delta x +
 D \delta \Delta\lambda - r_{\lambda} = 0
 \f]
 where
 \f[
 D = 2\beta^2 \Delta\lambda \| \mathbf{F}_{\lambda} \|^2.
 \f]

 User need to implement functions calculating \f$f(\mathbf{x},\lambda)\f$, i.e.
 function
 \f$f(\|\Delta\mathbf{x}\|^2)\f$  and its derivative,
 \f$\textrm{d}\mathbf{b}\f$.

 */
struct ArcLengthCtx {

  virtual ~ArcLengthCtx() = default;

  MoFEM::Interface &mField;

  double s;     ///< arc length radius
  double beta;  ///< force scaling factor
  double alpha; ///< displacement scaling factor

  SmartPetscObj<Vec> ghosTdLambda;
  double dLambda; ///< increment of load factor
  SmartPetscObj<Vec> ghostDiag;
  double dIag; ///< diagonal value

  double dx2;                  ///< inner_prod(dX,dX)
  double F_lambda2;            ///< inner_prod(F_lambda,F_lambda);
  double res_lambda;           ///< f_lambda - s
  SmartPetscObj<Vec> F_lambda; ///< F_lambda reference load vector
  SmartPetscObj<Vec>
      db; ///< db derivative of f(dx*dx), i.e. db = d[ f(dx*dx) ]/dx
  SmartPetscObj<Vec> xLambda; ///< solution of eq. K*xLambda = F_lambda
  SmartPetscObj<Vec> x0;      ///< displacement vector at beginning of step
  SmartPetscObj<Vec> dx;      ///< dx = x-x0

  /**
   * \brief set arc radius
   */
  MoFEMErrorCode setS(double s);

  /**
   * \brief set parameters controlling arc-length equations
   * alpha controls off diagonal therms
   * beta controls diagonal therm
   */
  MoFEMErrorCode setAlphaBeta(double alpha, double beta);

  ArcLengthCtx(MoFEM::Interface &m_field, const std::string &problem_name,
               const std::string &field_name = "LAMBDA");

  /** \brief Get global index of load factor
   */
  DofIdx getPetscGlobalDofIdx() {
    return arcDofRawPtr->getPetscGlobalDofIdx();
  };

  /** \brief Get local index of load factor
   */
  DofIdx getPetscLocalDofIdx() { return arcDofRawPtr->getPetscLocalDofIdx(); };

  /** \brief Get value of load factor
   */
  FieldData &getFieldData() { return arcDofRawPtr->getFieldData(); }

  /** \brief Get proc owning lambda dof
   */
  int getPart() { return arcDofRawPtr->getPart(); };

private:
  NumeredDofEntity *arcDofRawPtr;
};

#ifdef __SNESCTX_HPP__

/**
 * \brief It is ctx structure passed to SNES solver
 * \ingroup arc_length_control
 */
struct ArcLengthSnesCtx : public SnesCtx {
  ArcLengthCtx *arcPtrRaw;

  ArcLengthSnesCtx(MoFEM::Interface &m_field, const std::string &problem_name,
                   ArcLengthCtx *arc_ptr_raw)
      : SnesCtx(m_field, problem_name), arcPtrRaw(arc_ptr_raw) {}

  ArcLengthSnesCtx(MoFEM::Interface &m_field, const std::string &problem_name,
                   boost::shared_ptr<ArcLengthCtx> arc_ptr)
      : SnesCtx(m_field, problem_name), arcPtrRaw(arc_ptr.get()),
        arcPtr(arc_ptr) {}

private:
  boost::shared_ptr<ArcLengthCtx> arcPtr;
};

#endif //__SNESCTX_HPP__

#ifdef __TSCTX_HPP__

/**
 * \brief It is ctx structure passed to SNES solver
 * \ingroup arc_length_control
 */
struct ArcLengthTsCtx : public TsCtx {
  ArcLengthCtx *arcPtrRaw;

  /// \deprecated use constructor with shared ptr
  DEPRECATED ArcLengthTsCtx(MoFEM::Interface &m_field,
                            const std::string &problem_name,
                            ArcLengthCtx *arc_ptr_raw)
      : TsCtx(m_field, problem_name), arcPtrRaw(arc_ptr_raw) {}

  ArcLengthTsCtx(MoFEM::Interface &m_field, const std::string &problem_name,
                 boost::shared_ptr<ArcLengthCtx> arc_ptr)
      : TsCtx(m_field, problem_name), arcPtrRaw(arc_ptr.get()),
        arcPtr(arc_ptr) {}

private:
  boost::shared_ptr<ArcLengthCtx> arcPtr;
};

#endif // __TSCTX_HPP__

/** \brief shell matrix for arc-length method
 *
 * \ingroup arc_length_control

 Shell matrix which has structure:
 \f[
 \left[
  \begin{array}{cc}
   \mathbf{K} & -\mathbf{F}_\lambda \\
   \textrm{d}\mathbf{b} & D
  \end{array}
  \right]
 \left\{
 \begin{array}{c}
 \delta \Delta \mathbf{x} \\
 \delta \Delta \lambda
 \end{array}
 \right\}
 =
 \left[
  \begin{array}{c}
    -\mathbf{f}_\textrm{int} \\
    -r_\lambda
  \end{array}
  \right]
 \f]

 */
struct ArcLengthMatShell {

  SmartPetscObj<Mat> Aij;
  string problemName;
  ArcLengthCtx *arcPtrRaw; // this is for back compatibility

  ArcLengthMatShell(Mat aij, boost::shared_ptr<ArcLengthCtx> arc_ptr,
                    string problem_name);
  virtual ~ArcLengthMatShell() = default;

  /// \deprecated use constructor with shared_ptr
  DEPRECATED ArcLengthMatShell(Mat aij, ArcLengthCtx *arc_ptr_raw,
                               string problem_name);

  MoFEMErrorCode setLambda(Vec ksp_x, double *lambda, ScatterMode scattermode);
  friend MoFEMErrorCode ArcLengthMatMultShellOp(Mat A, Vec x, Vec f);

private:
  boost::shared_ptr<ArcLengthCtx> arcPtr;
};

/**
 * mult operator for Arc Length Shell Mat
 */
MoFEMErrorCode ArcLengthMatMultShellOp(Mat A, Vec x, Vec f);

/**
 * \brief structure for Arc Length pre-conditioner
 * \ingroup arc_length_control
 */
struct PCArcLengthCtx {

  SmartPetscObj<KSP> kSP;
  SmartPetscObj<PC> pC;
  SmartPetscObj<Mat> shellAij;
  SmartPetscObj<Mat> Aij;

  PCArcLengthCtx(Mat shell_Aij, Mat aij,
                 boost::shared_ptr<ArcLengthCtx> arc_ptr);

  PCArcLengthCtx(PC pc, Mat shell_Aij, Mat aij,
                 boost::shared_ptr<ArcLengthCtx> arc_ptr);

  ArcLengthCtx *arcPtrRaw; // this is for back compatibility

  /// \deprecated use with shared_ptr
  DEPRECATED PCArcLengthCtx(Mat shell_Aij, Mat aij, ArcLengthCtx *arc_ptr_raw);
  /// \deprecated use with shared_ptr
  DEPRECATED PCArcLengthCtx(PC pc, Mat shell_Aij, Mat aij,
                            ArcLengthCtx *arc_ptr_raw);

  friend MoFEMErrorCode PCApplyArcLength(PC pc, Vec pc_f, Vec pc_x);
  friend MoFEMErrorCode PCSetupArcLength(PC pc);

private:
  boost::shared_ptr<ArcLengthCtx> arcPtr;
};

/**
 * apply operator for Arc Length pre-conditioner
 * solves K*pc_x = pc_f
 * solves K*xLambda = -dF_lambda
 * solves ddlambda = ( res_lambda - db*xLambda )/( diag + db*pc_x )
 * calculate pc_x = pc_x + ddlambda*xLambda
 */
MoFEMErrorCode PCApplyArcLength(PC pc, Vec pc_f, Vec pc_x);

/**
 * set up structure for Arc Length pre-conditioner

 * it sets pre-conditioner for matrix K
 */
MoFEMErrorCode PCSetupArcLength(PC pc);

/**
 * \brief Zero F_lambda
 *
 */
struct ZeroFLmabda : public FEMethod {

  boost::shared_ptr<ArcLengthCtx> arcPtr;

  ZeroFLmabda(boost::shared_ptr<ArcLengthCtx> arc_ptr);

  MoFEMErrorCode preProcess();
};

#ifdef __DIRICHLET_HPP__

/**
 * \brief Assemble F_lambda into the right hand side
 *
 * postProcess - assembly F_lambda
 *
 */
struct AssembleFlambda : public FEMethod {

  boost::shared_ptr<ArcLengthCtx> arcPtr;

  AssembleFlambda(boost::shared_ptr<ArcLengthCtx> arc_ptr,
                  boost::shared_ptr<DirichletDisplacementBc> bc =
                      boost::shared_ptr<DirichletDisplacementBc>());

  MoFEMErrorCode preProcess();
  MoFEMErrorCode operator()();
  MoFEMErrorCode postProcess();

  inline void pushDirichletBC(boost::shared_ptr<DirichletDisplacementBc> bc) {
    bCs.push_back(bc);
  }

private:
  std::vector<boost::shared_ptr<DirichletDisplacementBc>> bCs;
};

#endif

/**
 * |brief Simple arc-length control of force
 *
 * This is added for testing, it simply control force, i.e.
 *
 * \f[
 * \lambda =  s
 * \f]
 *
 * Constructor takes one argument,
 * @param arc_ptr Pointer to arc-length CTX.
 */
struct SimpleArcLengthControl : public FEMethod {

  boost::shared_ptr<ArcLengthCtx> arcPtr;
  const bool aSsemble;

  SimpleArcLengthControl(boost::shared_ptr<ArcLengthCtx> &arc_ptr,
                         const bool assemble = false);
  ~SimpleArcLengthControl();

  MoFEMErrorCode preProcess();
  MoFEMErrorCode operator()();
  MoFEMErrorCode postProcess();

  /** \brief Calculate internal lambda
   */
  double calculateLambdaInt();

  /** \brief Calculate db
   */
  MoFEMErrorCode calculateDb();

  MoFEMErrorCode calculateDxAndDlambda(Vec x);
};

/** \brief Implementation of spherical arc-length method
  * \ingroup arc_length_control

  \f[
  \alpha \| \Delta\mathbf{x} \|^2
  + \Delta\lambda^2 \beta^2 \| \mathbf{F}_\lambda \|^2
  = s^2
  \f]

  This is particular implementation of ArcLength control, i.e. spherical arc
  length control. If beta is set to 0 and alpha is non-zero it is cylindrical
  arc-length control. Works well with general problem with non-linear
  geometry. It not guarantee dissipative loading path in case of physical
  nonlinearities.

  */
struct SphericalArcLengthControl : public FEMethod {

  ArcLengthCtx *arcPtrRaw; // this is for back compatibility

  /// \deprecated use constructor with shared_ptr
  DEPRECATED SphericalArcLengthControl(ArcLengthCtx *arc_ptr_raw);

  SphericalArcLengthControl(boost::shared_ptr<ArcLengthCtx> &arc_ptr);
  virtual ~SphericalArcLengthControl();

  MoFEMErrorCode preProcess();
  MoFEMErrorCode operator()();
  MoFEMErrorCode postProcess();

  /** \brief Calculate f_lambda(dx,lambda)

  \f[
  f_\lambda(\Delta\mathbf{x},\lambda) =
  \alpha \| \Delta\mathbf{x} \|^2
  + \Delta\lambda^2 \beta^2 \| \mathbf{F}_\lambda \|^2
  \f]

  */
  virtual double calculateLambdaInt();

  /** \brief Calculate db

  \f[
  \textrm{d}\mathbf{b} = 2 \alpha \Delta\mathbf{x}
  \f]

  */
  virtual MoFEMErrorCode calculateDb();
  virtual MoFEMErrorCode calculateDxAndDlambda(Vec x);
  virtual MoFEMErrorCode calculateInitDlambda(double *dlambda);
  virtual MoFEMErrorCode setDlambdaToX(Vec x, double dlambda);

private:
  boost::shared_ptr<ArcLengthCtx> arcPtr;
};

#endif // __ARC_LENGTH_TOOLS_HPP__

/**
  \defgroup arc_length_control Arc-Length control
  \ingroup user_modules
**/
