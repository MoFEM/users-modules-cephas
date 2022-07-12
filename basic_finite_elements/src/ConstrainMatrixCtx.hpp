/** \file ConstrainMatrixCtx.hpp
 *
 * Can be used if constrains are linear, i.e. are not function of time.
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

#ifndef __PROJECTION_MATRIX_CTX_HPP__
#define __PROJECTION_MATRIX_CTX_HPP__

/**
 * \brief structure for projection matrices
 * \ingroup projection_matrix
 *
 */
struct ConstrainMatrixCtx {

  MoFEM::Interface &mField;

  KSP kSP;
  Mat C, CT, CCT, CTC, K;
  Vec Cx, CCTm1_Cx, CT_CCTm1_Cx, CTCx;
  Vec X, Qx, KQx;
  bool initQorP, initQTKQ;
  bool createKSP;
  bool createScatter;
  bool cancelKSPMonitor;
  bool ownConstrainMatrix;

  // Scatter is created form problem_x to problem_y, or scatter is given
  // in the constructor

  VecScatter sCatter;
  string xProblem, yProblem;

  PetscLogEvent MOFEM_EVENT_projInit;
  PetscLogEvent MOFEM_EVENT_projQ;
  PetscLogEvent MOFEM_EVENT_projP;
  PetscLogEvent MOFEM_EVENT_projR;
  PetscLogEvent MOFEM_EVENT_projRT;
  PetscLogEvent MOFEM_EVENT_projCTC_QTKQ;

  /**
   * Construct data structure to build operators for projection matrices
   *
   * User need to set matrix C to make it work
   *
   * \param x_problem problem on which vector is projected
   * \param y_problem problem used to construct projection matrices
   * \param create_ksp create ksp solver otherwise  user need to set it up
   */
  ConstrainMatrixCtx(MoFEM::Interface &m_field, string x_problem,
                     string y_problem, bool create_ksp = true,
                     bool own_contrain_matrix = false);

  ConstrainMatrixCtx(MoFEM::Interface &m_field, VecScatter scatter,
                     bool create_ksp = true, bool own_contrain_matrix = false);

  virtual ~ConstrainMatrixCtx() {
    ierr = destroyQorP();
    CHKERRABORT(mField.get_comm(), ierr);
    ierr = destroyQTKQ();
    CHKERRABORT(mField.get_comm(), ierr);
    if (ownConstrainMatrix) {
      ierr = MatDestroy(&C);
      CHKERRABORT(mField.get_comm(), ierr);
    }
  };

  PetscReal rTol, absTol, dTol;
  PetscInt maxIts;

  /**
   * \brief initialize vectors and matrices for Q and P shell matrices,
   * scattering is set based on x_problem and y_problem
   *
   * \param x is a vector from problem x
   */
  MoFEMErrorCode initializeQorP(Vec x);

  /**
   * \brief initialize vectors and matrices for CTC+QTKQ shell matrices,
   * scattering is set based on x_problem and y_problem
   */
  MoFEMErrorCode initializeQTKQ();

  /**
   * \brief re-calculate CT and CCT if C matrix has been changed since
   * initialization
   */
  MoFEMErrorCode recalculateCTandCCT();

  /**
   * \brief re-calculate CTC matrix has been changed since initialization
   */
  MoFEMErrorCode recalculateCTC();

  /**
   * \brief destroy sub-matrices used for shell matrices P, Q, R, RT
   */
  MoFEMErrorCode destroyQorP();

  /**
   * \brief destroy sub-matrices used for shell matrix QTKQ
   */
  MoFEMErrorCode destroyQTKQ();

  friend MoFEMErrorCode ProjectionMatrixMultOpQ(Mat Q, Vec x, Vec f);
  friend MoFEMErrorCode ConstrainMatrixMultOpP(Mat P, Vec x, Vec f);
  friend MoFEMErrorCode ConstrainMatrixMultOpR(Mat R, Vec x, Vec f);
  friend MoFEMErrorCode ConstrainMatrixMultOpRT(Mat RT, Vec x, Vec f);
  friend MoFEMErrorCode ConstrainMatrixMultOpCTC_QTKQ(Mat CTC_QTKQ, Vec x,
                                                      Vec f);

  friend MoFEMErrorCode ConstrainMatrixDestroyOpPorQ();
  friend MoFEMErrorCode ConstrainMatrixDestroyOpQTKQ();
};

/**
  * \brief Multiplication operator for Q = I-CTC(CCT)^-1C
  *
  * \code
  * Mat Q; //for problem
  * ConstrainMatrixCtx
  projection_matrix_ctx(m_field,problem_name,contrains_problem_name);
  * CHKERR MatCreateShell(PETSC_COMM_WORLD,m,m,M,M,&projection_matrix_ctx,&Q);
  * CHKERR
  MatShellSetOperation(Q,MATOP_MULT,(void(*)(void))ProjectionMatrixMultOpQ);
  * CHKERR
  MatShellSetOperation(Q,MATOP_DESTROY,(void(*)(void))ConstrainMatrixDestroyOpPorQ);
  *
  * \endcode

  * \ingroup projection_matrix

  */
MoFEMErrorCode ProjectionMatrixMultOpQ(Mat Q, Vec x, Vec f);

/**
  * \brief Multiplication operator for P = CT(CCT)^-1C
  *
  * \code
  * Mat P; //for problem
  * ConstrainMatrixCtx
  projection_matrix_ctx(m_field,problem_name,contrains_problem_name);
  * CHKERR MatCreateShell(PETSC_COMM_WORLD,m,m,M,M,&projection_matrix_ctx,&P);
  * CHKERR
  MatShellSetOperation(P,MATOP_MULT,(void(*)(void))ConstrainMatrixMultOpP);
  *
  * \endcode

  * \ingroup projection_matrix

  */
MoFEMErrorCode ConstrainMatrixMultOpP(Mat P, Vec x, Vec f);

/**
  * \brief Multiplication operator for R = CT(CCT)^-1
  *
  * \code
  * Mat R; //for problem
  * ConstrainMatrixCtx
  projection_matrix_ctx(m_field,problem_name,contrains_problem_name);
  * CHKERR MatCreateShell(PETSC_COMM_WORLD,m,m,M,M,&projection_matrix_ctx,&R);
  * CHKERR
  MatShellSetOperation(R,MATOP_MULT,(void(*)(void))ConstrainMatrixMultOpR);
  *
  * \endcode

  * \ingroup projection_matrix

  */
MoFEMErrorCode ConstrainMatrixMultOpR(Mat R, Vec x, Vec f);

/**
  * \brief Multiplication operator for RT = (CCT)^-TC
  *
  * \code
  * Mat RT; //for problem
  * ConstrainMatrixCtx
  projection_matrix_ctx(m_field,problem_name,contrains_problem_name);
  * CHKERR MatCreateShell(PETSC_COMM_WORLD,m,m,M,M,&projection_matrix_ctx,&RT);
  * CHKERR
  MatShellSetOperation(RT,MATOP_MULT,(void(*)(void))ConstrainMatrixMultOpRT);
  *
  * \endcode

  * \ingroup projection_matrix

  */
MoFEMErrorCode ConstrainMatrixMultOpRT(Mat RT, Vec x, Vec f);

/**
  * \brief Multiplication operator for RT = (CCT)^-TC
  *
  * \code
  * Mat CTC_QTKQ; //for problem
  * ConstrainMatrixCtx
  projection_matrix_ctx(m_field,problem_name,contrains_problem_name);
  * CHKERR
  MatCreateShell(PETSC_COMM_WORLD,m,m,M,M,&projection_matrix_ctx,&CTC_QTKQ);
  * CHKERR
  MatShellSetOperation(CTC_QTKQ,MATOP_MULT,(void(*)(void))ConstrainMatrixMultOpCTC_QTKQ);
  * CHKERR
  MatShellSetOperation(CTC_QTKQ,MATOP_DESTROY,(void(*)(void))ConstrainMatrixDestroyOpQTKQ);
  *
  * \endcode
  *

  * \ingroup projection_matrix

  */
MoFEMErrorCode ConstrainMatrixMultOpCTC_QTKQ(Mat CTC_QTKQ, Vec x, Vec f);

/**
  * \brief Destroy shell matrix Q
  *
  * \code
  * Mat Q; //for problem
  * ConstrainMatrixCtx
  projection_matrix_ctx(m_field,problem_name,contrains_problem_name);
  * CHKERR MatCreateShell(PETSC_COMM_WORLD,m,m,M,M,&projection_matrix_ctx,&Q);
  * CHKERR
  MatShellSetOperation(Q,MATOP_MULT,(void(*)(void))ProjectionMatrixMultOpQ);
  * CHKERR
  MatShellSetOperation(Q,MATOP_DESTROY,(void(*)(void))ConstrainMatrixDestroyOpPorQ);
  *
  * \endcode

  * \ingroup projection_matrix

  */
MoFEMErrorCode ConstrainMatrixDestroyOpPorQ(Mat Q);

/**
  * \brief Destroy shell matrix
  *
  * \code
  * Mat CTC_QTKQ; //for problem
  * ConstrainMatrixCtx
  projection_matrix_ctx(m_field,problem_name,contrains_problem_name);
  * CHKERR MatCreateShell(PETSC_COMM_WORLD,m,m,M,M,&projection_matrix_ctx,&Q);
  * CHKERR
  MatShellSetOperation(Q,MATOP_MULT,(void(*)(void))ConstrainMatrixMultOpCTC_QTKQ);
  * CHKERR
  MatShellSetOperation(Q,MATOP_DESTROY,(void(*)(void))mat_destroy_QTKQ);
  *
  * \endcode

  * \ingroup projection_matrix

  */
MoFEMErrorCode ConstrainMatrixDestroyOpQTKQ(Mat QTKQ);

#endif // __PROJECTION_MATRIX_CTX_HPP__

/**
 \defgroup projection_matrix Constrain Projection Matrix
 \ingroup user_modules
**/
