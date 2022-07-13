/** \file ConstrainMatrixCtx.cpp
 * \brief Implementation of projection matrix
 *
 * FIXME: DESCRIPTION
 */



#include <MoFEM.hpp>
using namespace MoFEM;
#include <ConstrainMatrixCtx.hpp>

const static bool debug = false;

#define INIT_DATA_CONSTRAINMATRIXCTX                                           \
  C(PETSC_NULL), CT(PETSC_NULL), CCT(PETSC_NULL), CTC(PETSC_NULL),             \
      K(PETSC_NULL), Cx(PETSC_NULL), CCTm1_Cx(PETSC_NULL),                     \
      CT_CCTm1_Cx(PETSC_NULL), CTCx(PETSC_NULL), Qx(PETSC_NULL),               \
      KQx(PETSC_NULL), initQorP(true), initQTKQ(true), createKSP(create_ksp),  \
      createScatter(true), cancelKSPMonitor(true),                             \
      ownConstrainMatrix(own_contrain_matrix)

ConstrainMatrixCtx::ConstrainMatrixCtx(MoFEM::Interface &m_field,
                                       string x_problem, string y_problem,
                                       bool create_ksp,
                                       bool own_contrain_matrix)
    : mField(m_field), INIT_DATA_CONSTRAINMATRIXCTX, xProblem(x_problem),
      yProblem(y_problem) {
  PetscLogEventRegister("ProjectionInit", 0, &MOFEM_EVENT_projInit);
  PetscLogEventRegister("ProjectionQ", 0, &MOFEM_EVENT_projQ);
  PetscLogEventRegister("ProjectionP", 0, &MOFEM_EVENT_projP);
  PetscLogEventRegister("ProjectionR", 0, &MOFEM_EVENT_projR);
  PetscLogEventRegister("ProjectionRT", 0, &MOFEM_EVENT_projRT);
  PetscLogEventRegister("ProjectionCTC_QTKQ", 0, &MOFEM_EVENT_projCTC_QTKQ);
}

ConstrainMatrixCtx::ConstrainMatrixCtx(MoFEM::Interface &m_field,
                                       VecScatter scatter, bool create_ksp,
                                       bool own_contrain_matrix)
    : mField(m_field), INIT_DATA_CONSTRAINMATRIXCTX, sCatter(scatter) {
  PetscLogEventRegister("ProjectionInit", 0, &MOFEM_EVENT_projInit);
  PetscLogEventRegister("ProjectionQ", 0, &MOFEM_EVENT_projQ);
  PetscLogEventRegister("ProjectionP", 0, &MOFEM_EVENT_projP);
  PetscLogEventRegister("ProjectionR", 0, &MOFEM_EVENT_projR);
  PetscLogEventRegister("ProjectionRT", 0, &MOFEM_EVENT_projRT);
  PetscLogEventRegister("ProjectionCTC_QTKQ", 0, &MOFEM_EVENT_projCTC_QTKQ);
}

MoFEMErrorCode ConstrainMatrixCtx::initializeQorP(Vec x) {
  MoFEMFunctionBegin;
  if (initQorP) {
    initQorP = false;

    PetscLogEventBegin(MOFEM_EVENT_projInit, 0, 0, 0, 0);
    CHKERR MatTranspose(C, MAT_INITIAL_MATRIX, &CT);
    // need to be calculated when C is changed
    CHKERR MatTransposeMatMult(CT, CT, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CCT);
    if (createKSP) {
      CHKERR KSPCreate(mField.get_comm(), &kSP);
      // need to be recalculated when C is changed
      CHKERR KSPSetOperators(kSP, CCT, CCT);
      CHKERR KSPSetFromOptions(kSP);
      CHKERR KSPSetInitialGuessKnoll(kSP, PETSC_TRUE);
      CHKERR KSPGetTolerances(kSP, &rTol, &absTol, &dTol, &maxIts);
      CHKERR KSPSetUp(kSP);
      if (cancelKSPMonitor) {
        CHKERR KSPMonitorCancel(kSP);
      }
    }
#if PETSC_VERSION_GE(3, 5, 3)
    CHKERR MatCreateVecs(C, &X, PETSC_NULL);
    CHKERR MatCreateVecs(C, PETSC_NULL, &Cx);
    CHKERR MatCreateVecs(CCT, PETSC_NULL, &CCTm1_Cx);
#else
    CHKERR MatGetVecs(C, &X, PETSC_NULL);
    CHKERR MatGetVecs(C, PETSC_NULL, &Cx);
    CHKERR MatGetVecs(CCT, PETSC_NULL, &CCTm1_Cx);
#endif
    CHKERR VecDuplicate(X, &CT_CCTm1_Cx);
    if (createScatter) {
      CHKERR mField.getInterface<VecManager>()->vecScatterCreate(
          x, xProblem, ROW, X, yProblem, COL, &sCatter);
    }
    PetscLogEventEnd(MOFEM_EVENT_projInit, 0, 0, 0, 0);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ConstrainMatrixCtx::recalculateCTandCCT() {
  MoFEMFunctionBegin;
  if (initQorP)
    MoFEMFunctionReturnHot(0);
  CHKERR MatTranspose(C, MAT_REUSE_MATRIX, &CT);
  CHKERR MatTransposeMatMult(CT, CT, MAT_REUSE_MATRIX, PETSC_DEFAULT, &CCT);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ConstrainMatrixCtx::destroyQorP() {
  MoFEMFunctionBegin;
  if (initQorP)
    MoFEMFunctionReturnHot(0);
  CHKERR MatDestroy(&CT);
  CHKERR MatDestroy(&CCT);
  if (createKSP) {
    CHKERR KSPDestroy(&kSP);
  }
  CHKERR VecDestroy(&X);
  CHKERR VecDestroy(&Cx);
  CHKERR VecDestroy(&CCTm1_Cx);
  CHKERR VecDestroy(&CT_CCTm1_Cx);
  if (createScatter) {
    CHKERR VecScatterDestroy(&sCatter);
  }
  initQorP = true;
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ConstrainMatrixCtx::initializeQTKQ() {
  MoFEMFunctionBegin;
  if (initQTKQ) {
    initQTKQ = false;
    PetscLogEventBegin(MOFEM_EVENT_projInit, 0, 0, 0, 0);
    // need to be recalculated when C is changed
    CHKERR MatTransposeMatMult(C, C, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CTC);
    if (debug) {
      // MatView(CCT,PETSC_VIEWER_DRAW_WORLD);
      int m, n;
      MatGetSize(CCT, &m, &n);
      PetscPrintf(mField.get_comm(), "CCT size (%d,%d)\n", m, n);
      // std::string wait;
      // std::cin >> wait;
    }
#if PETSC_VERSION_GE(3, 5, 3)
    CHKERR MatCreateVecs(K, &Qx, PETSC_NULL);
    CHKERR MatCreateVecs(K, PETSC_NULL, &KQx);
    CHKERR MatCreateVecs(CTC, PETSC_NULL, &CTCx);
#else
    CHKERR MatGetVecs(K, &Qx, PETSC_NULL);
    CHKERR MatGetVecs(K, PETSC_NULL, &KQx);
    CHKERR MatGetVecs(CTC, PETSC_NULL, &CTCx);
#endif
    PetscLogEventEnd(MOFEM_EVENT_projInit, 0, 0, 0, 0);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ConstrainMatrixCtx::recalculateCTC() {
  MoFEMFunctionBegin;
  if (initQTKQ)
    MoFEMFunctionReturnHot(0);
  CHKERR MatTransposeMatMult(C, C, MAT_REUSE_MATRIX, PETSC_DEFAULT, &CTC);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ConstrainMatrixCtx::destroyQTKQ() {
  MoFEMFunctionBegin;
  if (initQTKQ)
    MoFEMFunctionReturnHot(0);
  CHKERR MatDestroy(&CTC);
  CHKERR VecDestroy(&Qx);
  CHKERR VecDestroy(&KQx);
  CHKERR VecDestroy(&CTCx);
  initQTKQ = true;
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ProjectionMatrixMultOpQ(Mat Q, Vec x, Vec f) {
  MoFEMFunctionBegin;
  void *void_ctx;
  CHKERR MatShellGetContext(Q, &void_ctx);
  ConstrainMatrixCtx *ctx = (ConstrainMatrixCtx *)void_ctx;
  PetscLogEventBegin(ctx->MOFEM_EVENT_projQ, 0, 0, 0, 0);
  CHKERR ctx->initializeQorP(x);
  CHKERR VecCopy(x, f);
  CHKERR VecScatterBegin(ctx->sCatter, x, ctx->X, INSERT_VALUES,
                         SCATTER_FORWARD);
  CHKERR VecScatterEnd(ctx->sCatter, x, ctx->X, INSERT_VALUES, SCATTER_FORWARD);
  if (debug) {
    // CHKERR VecView(ctx->X,PETSC_VIEWER_STDOUT_WORLD);
    CHKERR VecScatterBegin(ctx->sCatter, ctx->X, f, INSERT_VALUES,
                           SCATTER_REVERSE);
    CHKERR VecScatterEnd(ctx->sCatter, ctx->X, f, INSERT_VALUES,
                         SCATTER_REVERSE);
    PetscBool flg;
    CHKERR VecEqual(x, f, &flg);
    if (flg == PETSC_FALSE)
      SETERRQ(PETSC_COMM_SELF, MOFEM_IMPOSIBLE_CASE, "scatter is not working");
  }
  CHKERR MatMult(ctx->C, ctx->X, ctx->Cx);
  CHKERR KSPSolve(ctx->kSP, ctx->Cx, ctx->CCTm1_Cx);
  CHKERR MatMult(ctx->CT, ctx->CCTm1_Cx, ctx->CT_CCTm1_Cx);
  CHKERR VecScale(ctx->CT_CCTm1_Cx, -1);
  CHKERR VecScatterBegin(ctx->sCatter, ctx->CT_CCTm1_Cx, f, ADD_VALUES,
                         SCATTER_REVERSE);
  CHKERR VecScatterEnd(ctx->sCatter, ctx->CT_CCTm1_Cx, f, ADD_VALUES,
                       SCATTER_REVERSE);
  PetscLogEventEnd(ctx->MOFEM_EVENT_projQ, 0, 0, 0, 0);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ConstrainMatrixMultOpP(Mat P, Vec x, Vec f) {
  MoFEMFunctionBegin;
  void *void_ctx;
  CHKERR MatShellGetContext(P, &void_ctx);
  ConstrainMatrixCtx *ctx = (ConstrainMatrixCtx *)void_ctx;
  PetscLogEventBegin(ctx->MOFEM_EVENT_projP, 0, 0, 0, 0);
  CHKERR ctx->initializeQorP(x);
  CHKERR VecScatterBegin(ctx->sCatter, x, ctx->X, INSERT_VALUES,
                         SCATTER_FORWARD);
  CHKERR VecScatterEnd(ctx->sCatter, x, ctx->X, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR MatMult(ctx->C, ctx->X, ctx->Cx);
  CHKERR KSPSolve(ctx->kSP, ctx->Cx, ctx->CCTm1_Cx);
  CHKERR MatMult(ctx->CT, ctx->CCTm1_Cx, ctx->CT_CCTm1_Cx);
  CHKERR VecZeroEntries(f);
  CHKERR VecGhostUpdateBegin(f, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(f, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecScatterBegin(ctx->sCatter, ctx->CT_CCTm1_Cx, f, INSERT_VALUES,
                         SCATTER_REVERSE);
  CHKERR VecScatterEnd(ctx->sCatter, ctx->CT_CCTm1_Cx, f, INSERT_VALUES,
                       SCATTER_REVERSE);
  PetscLogEventEnd(ctx->MOFEM_EVENT_projP, 0, 0, 0, 0);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ConstrainMatrixMultOpR(Mat R, Vec x, Vec f) {
  MoFEMFunctionBegin;
  void *void_ctx;
  CHKERR MatShellGetContext(R, &void_ctx);
  ConstrainMatrixCtx *ctx = (ConstrainMatrixCtx *)void_ctx;
  PetscLogEventBegin(ctx->MOFEM_EVENT_projR, 0, 0, 0, 0);
  if (ctx->initQorP)
    SETERRQ(PETSC_COMM_SELF, MOFEM_IMPOSIBLE_CASE,
            "you have to call first initQorP or use Q matrix");
  CHKERR KSPSolve(ctx->kSP, x, ctx->CCTm1_Cx);
  CHKERR MatMult(ctx->CT, ctx->CCTm1_Cx, ctx->CT_CCTm1_Cx);
  CHKERR VecZeroEntries(f);
  CHKERR VecGhostUpdateBegin(f, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(f, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecScatterBegin(ctx->sCatter, ctx->CT_CCTm1_Cx, f, INSERT_VALUES,
                         SCATTER_REVERSE);
  CHKERR VecScatterEnd(ctx->sCatter, ctx->CT_CCTm1_Cx, f, INSERT_VALUES,
                       SCATTER_REVERSE);
  PetscLogEventEnd(ctx->MOFEM_EVENT_projR, 0, 0, 0, 0);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ConstrainMatrixMultOpRT(Mat RT, Vec x, Vec f) {
  MoFEMFunctionBegin;
  void *void_ctx;
  CHKERR MatShellGetContext(RT, &void_ctx);
  ConstrainMatrixCtx *ctx = (ConstrainMatrixCtx *)void_ctx;
  PetscLogEventBegin(ctx->MOFEM_EVENT_projRT, 0, 0, 0, 0);
  if (ctx->initQorP)
    SETERRQ(PETSC_COMM_SELF, MOFEM_IMPOSIBLE_CASE,
            "you have to call first initQorP or use Q matrix");
  CHKERR VecScatterBegin(ctx->sCatter, x, ctx->X, INSERT_VALUES,
                         SCATTER_FORWARD);
  CHKERR VecScatterEnd(ctx->sCatter, x, ctx->X, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR MatMult(ctx->C, ctx->X, ctx->Cx);
  CHKERR KSPSolve(ctx->kSP, ctx->Cx, f);
  PetscLogEventEnd(ctx->MOFEM_EVENT_projRT, 0, 0, 0, 0);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ConstrainMatrixMultOpCTC_QTKQ(Mat CTC_QTKQ, Vec x, Vec f) {
  MoFEMFunctionBegin;
  void *void_ctx;
  CHKERR MatShellGetContext(CTC_QTKQ, &void_ctx);
  ConstrainMatrixCtx *ctx = (ConstrainMatrixCtx *)void_ctx;
  PetscLogEventBegin(ctx->MOFEM_EVENT_projCTC_QTKQ, 0, 0, 0, 0);
  Mat Q;
  int M, N, m, n;
  CHKERR MatGetSize(ctx->K, &M, &N);
  CHKERR MatGetLocalSize(ctx->K, &m, &n);
  CHKERR MatCreateShell(ctx->mField.get_comm(), m, n, M, N, ctx, &Q);
  CHKERR MatShellSetOperation(Q, MATOP_MULT,
                              (void (*)(void))ProjectionMatrixMultOpQ);
  CHKERR ctx->initializeQTKQ();
  CHKERR MatMult(Q, x, ctx->Qx);
  CHKERR MatMult(ctx->K, ctx->Qx, ctx->KQx);
  CHKERR MatMult(Q, ctx->KQx, f);
  CHKERR VecScatterBegin(ctx->sCatter, x, ctx->X, INSERT_VALUES,
                         SCATTER_FORWARD);
  CHKERR VecScatterEnd(ctx->sCatter, x, ctx->X, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR MatMult(ctx->CTC, ctx->X, ctx->CTCx);
  CHKERR VecScatterBegin(ctx->sCatter, ctx->CTCx, f, ADD_VALUES,
                         SCATTER_REVERSE);
  CHKERR VecScatterEnd(ctx->sCatter, ctx->CTCx, f, ADD_VALUES, SCATTER_REVERSE);
  CHKERR MatDestroy(&Q);
  PetscLogEventEnd(ctx->MOFEM_EVENT_projCTC_QTKQ, 0, 0, 0, 0);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ConstrainMatrixDestroyOpPorQ(Mat Q) {
  MoFEMFunctionBegin;
  void *void_ctx;
  CHKERR MatShellGetContext(Q, &void_ctx);
  ConstrainMatrixCtx *ctx = (ConstrainMatrixCtx *)void_ctx;
  CHKERR ctx->destroyQorP();
  MoFEMFunctionReturn(0);
}
MoFEMErrorCode ConstrainMatrixDestroyOpQTKQ(Mat QTKQ) {
  MoFEMFunctionBegin;
  void *void_ctx;
  CHKERR MatShellGetContext(QTKQ, &void_ctx);
  ConstrainMatrixCtx *ctx = (ConstrainMatrixCtx *)void_ctx;
  CHKERR ctx->destroyQTKQ();
  MoFEMFunctionReturn(0);
}
