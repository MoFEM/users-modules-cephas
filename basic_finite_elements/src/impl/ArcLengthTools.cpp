/** \file ArcLengthTools.hpp

 Implementation of Arc Length element

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

#include <MoFEM.hpp>
using namespace MoFEM;
#include <ArcLengthTools.hpp>

// ********************
// Arc-length ctx class

MoFEMErrorCode ArcLengthCtx::setS(double s) {
  MoFEMFunctionBegin;
  this->s = s;
  CHKERR PetscPrintf(mField.get_comm(), "\tSet s = %6.4e\n", this->s);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ArcLengthCtx::setAlphaBeta(double alpha, double beta) {
  MoFEMFunctionBegin;
  this->alpha = alpha;
  this->beta = beta;
  CHKERR PetscPrintf(mField.get_comm(), "\tSet alpha = %6.4e beta = %6.4e\n",
                     this->alpha, this->beta);
  MoFEMFunctionReturn(0);
}

ArcLengthCtx::ArcLengthCtx(MoFEM::Interface &m_field,
                           const std::string &problem_name,
                           const std::string &field_name)
    : mField(m_field), dx2(0), F_lambda2(0), res_lambda(0) {
  ierr = m_field.getInterface<VecManager>()->vecCreateGhost(problem_name, ROW,
                                                            &F_lambda);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = VecSetOption(F_lambda, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = VecDuplicate(F_lambda, &db);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = VecDuplicate(F_lambda, &xLambda);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = VecDuplicate(F_lambda, &x0);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = VecDuplicate(F_lambda, &dx);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  ierr = VecZeroEntries(F_lambda); 
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = VecZeroEntries(db); 
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = VecZeroEntries(xLambda); 
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = VecZeroEntries(x0); 
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = VecZeroEntries(dx);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  const Problem *problem_ptr;
  ierr = m_field.get_problem(problem_name, &problem_ptr);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  boost::shared_ptr<NumeredDofEntity_multiIndex> dofs_ptr_no_const =
      problem_ptr->getNumeredDofsRows();
  NumeredDofEntityByFieldName::iterator hi_dit;
  dIt = dofs_ptr_no_const->get<FieldName_mi_tag>().lower_bound(field_name);
  hi_dit = dofs_ptr_no_const->get<FieldName_mi_tag>().upper_bound(field_name);

  if (distance(dIt, hi_dit) != 1) {
    SETERRABORT(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                "can not find unique LAMBDA (load factor)");
  }

  if ((unsigned int)mField.get_comm_rank() == (*dIt)->getPart()) {
    ierr = VecCreateGhostWithArray(mField.get_comm(), 1, 1, 0, PETSC_NULL,
                                   &dLambda, &ghosTdLambda);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = VecCreateGhostWithArray(mField.get_comm(), 1, 1, 0, PETSC_NULL,
                                   &dIag, &ghostDiag);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  } else {
    int one[] = {0};
    ierr = VecCreateGhostWithArray(mField.get_comm(), 0, 1, 1, one, &dLambda,
                                   &ghosTdLambda);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    ierr = VecCreateGhostWithArray(mField.get_comm(), 0, 1, 1, one, &dIag,
                                   &ghostDiag);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }
  dLambda = 0;
  dIag = 0;
}

ArcLengthCtx::~ArcLengthCtx() {
  ierr = VecDestroy(&F_lambda);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = VecDestroy(&db);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = VecDestroy(&xLambda);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = VecDestroy(&x0);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = VecDestroy(&dx);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = VecDestroy(&ghosTdLambda);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = VecDestroy(&ghostDiag);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
}

// ***********************
// Arc-length shell matrix

ArcLengthMatShell::ArcLengthMatShell(Mat aij, ArcLengthCtx *arc_ptr_raw,
                                     string problem_name)
    : Aij(aij), arcPtrRaw(arc_ptr_raw), problemName(problem_name) {
  ierr = PetscObjectReference((PetscObject)aij);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
}

ArcLengthMatShell::ArcLengthMatShell(Mat aij,
                                     boost::shared_ptr<ArcLengthCtx> arc_ptr,
                                     string problem_name)
    : Aij(aij), arcPtrRaw(arc_ptr.get()), problemName(problem_name),
      arcPtr(arc_ptr) {
  ierr = PetscObjectReference((PetscObject)aij);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
}

ArcLengthMatShell::~ArcLengthMatShell() {
  ierr = MatDestroy(&Aij);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
}

MoFEMErrorCode ArcLengthMatShell::setLambda(Vec ksp_x, double *lambda,
                                            ScatterMode scattermode) {
  MoFEMFunctionBegin;

  int part = arcPtrRaw->getPart();
  int rank = arcPtrRaw->mField.get_comm_rank();

  Vec lambda_ghost;
  if (rank == part) {
    CHKERR VecCreateGhostWithArray(arcPtrRaw->mField.get_comm(), 1, 1, 0,
                                   PETSC_NULL, lambda, &lambda_ghost);
  } else {
    int one[] = {0};
    CHKERR VecCreateGhostWithArray(arcPtrRaw->mField.get_comm(), 0, 1, 1, one,
                                   lambda, &lambda_ghost);
  }

  switch (scattermode) {
  case SCATTER_FORWARD: {
    int idx = arcPtrRaw->getPetscGlobalDofIdx();
    if (part == rank) {
      CHKERR VecGetValues(ksp_x, 1, &idx, lambda);
    }
    CHKERR VecGhostUpdateBegin(lambda_ghost, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(lambda_ghost, INSERT_VALUES, SCATTER_FORWARD);
  } break;
  case SCATTER_REVERSE: {
    if (part == rank) {
      PetscScalar *array;
      CHKERR VecGetArray(ksp_x, &array);
      array[arcPtrRaw->getPetscLocalDofIdx()] = *lambda;
      CHKERR VecRestoreArray(ksp_x, &array);
    }
  } break;
  default:
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "not implemented");
  }

  CHKERR VecDestroy(&lambda_ghost);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ArcLengthMatMultShellOp(Mat A, Vec x, Vec f) {
  MoFEMFunctionBegin;
  void *void_ctx;
  CHKERR MatShellGetContext(A, &void_ctx);
  ArcLengthMatShell *ctx = (ArcLengthMatShell *)void_ctx;
  CHKERR MatMult(ctx->Aij, x, f);
  double lambda;
  CHKERR ctx->setLambda(x, &lambda, SCATTER_FORWARD);
  double db_dot_x;
  CHKERR VecDot(ctx->arcPtrRaw->db, x, &db_dot_x);
  double f_lambda;
  f_lambda = ctx->arcPtrRaw->dIag * lambda + db_dot_x;
  CHKERR ctx->setLambda(f, &f_lambda, SCATTER_REVERSE);
  CHKERR VecAXPY(f, lambda, ctx->arcPtrRaw->F_lambda);
  MoFEMFunctionReturn(0);
}

// arc-length preconditioner

PCArcLengthCtx::PCArcLengthCtx(Mat shell_Aij, Mat aij, ArcLengthCtx *arc_ptr)
    : kSP(PETSC_NULL), pC(PETSC_NULL), shellAij(shell_Aij), Aij(aij),
      arcPtrRaw(arc_ptr) {
  ierr = PCCreate(PetscObjectComm((PetscObject)aij), &pC);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = KSPCreate(PetscObjectComm((PetscObject)pC), &kSP);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = KSPAppendOptionsPrefix(kSP, "arc_length_");
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
}

PCArcLengthCtx::PCArcLengthCtx(Mat shell_Aij, Mat aij,
                               boost::shared_ptr<ArcLengthCtx> &arc_ptr)
    : kSP(PETSC_NULL), shellAij(shell_Aij), Aij(aij), arcPtrRaw(arc_ptr.get()),
      arcPtr(arc_ptr) {
  ierr = PCCreate(PetscObjectComm((PetscObject)aij), &pC);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = KSPCreate(PetscObjectComm((PetscObject)pC), &kSP);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = KSPAppendOptionsPrefix(kSP, "arc_length_");
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
}

PCArcLengthCtx::PCArcLengthCtx(PC pc, Mat shell_Aij, Mat aij,
                               boost::shared_ptr<ArcLengthCtx> &arc_ptr)
    : kSP(PETSC_NULL), pC(pc), shellAij(shell_Aij), Aij(aij),
      arcPtrRaw(arc_ptr.get()), arcPtr(arc_ptr) {
  ierr = PetscObjectReference((PetscObject)pC);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = KSPCreate(PetscObjectComm((PetscObject)pC), &kSP);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = KSPAppendOptionsPrefix(kSP, "arc_length_");
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
}

PCArcLengthCtx::~PCArcLengthCtx() {
  if (kSP != PETSC_NULL) {
    ierr = KSPDestroy(&kSP);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }
  if (pC != PETSC_NULL) {
    ierr = PCDestroy(&pC);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }
}

MoFEMErrorCode PCApplyArcLength(PC pc, Vec pc_f, Vec pc_x) {
  MoFEMFunctionBegin;
  void *void_ctx;
  CHKERR PCShellGetContext(pc, &void_ctx);
  PCArcLengthCtx *ctx = (PCArcLengthCtx *)void_ctx;
  void *void_MatCtx;
  MatShellGetContext(ctx->shellAij, &void_MatCtx) ;
  ArcLengthMatShell *mat_ctx = (ArcLengthMatShell *)void_MatCtx;
  PetscBool same;
  PetscObjectTypeCompare((PetscObject)ctx->kSP, KSPPREONLY, &same);
  CHKERR KSPSetInitialGuessNonzero(ctx->kSP, PETSC_FALSE);
  CHKERR KSPSolve(ctx->kSP, pc_f, pc_x);
  if (same != PETSC_TRUE) {
    CHKERR KSPSetInitialGuessNonzero(ctx->kSP, PETSC_TRUE);
  } else {
    CHKERR KSPSetInitialGuessNonzero(ctx->kSP, PETSC_FALSE);
  }
  CHKERR KSPSolve(ctx->kSP, ctx->arcPtrRaw->F_lambda, ctx->arcPtrRaw->xLambda);
  double db_dot_pc_x, db_dot_x_lambda;
  CHKERR VecDot(ctx->arcPtrRaw->db, pc_x, &db_dot_pc_x);
  CHKERR VecDot(ctx->arcPtrRaw->db, ctx->arcPtrRaw->xLambda, &db_dot_x_lambda);
  double denominator = ctx->arcPtrRaw->dIag + db_dot_x_lambda;
  double res_lambda;
  CHKERR mat_ctx->setLambda(pc_f, &res_lambda, SCATTER_FORWARD);
  double ddlambda = (res_lambda - db_dot_pc_x) / denominator;
  if (ddlambda != ddlambda || denominator == 0) {
    double nrm2_pc_f, nrm2_db, nrm2_pc_x, nrm2_xLambda;
    CHKERR VecNorm(pc_f, NORM_2, &nrm2_pc_f);
    CHKERR VecNorm(ctx->arcPtrRaw->db, NORM_2, &nrm2_db);
    CHKERR VecNorm(pc_x, NORM_2, &nrm2_pc_x);
    CHKERR VecNorm(ctx->arcPtrRaw->xLambda, NORM_2, &nrm2_xLambda);
    std::ostringstream ss;
    ss << "problem with ddlambda=" << ddlambda << "\nres_lambda=" << res_lambda
       << "\ndenominator=" << denominator << "\ndb_dot_pc_x=" << db_dot_pc_x
       << "\ndb_dot_x_lambda=" << db_dot_x_lambda
       << "\ndiag=" << ctx->arcPtrRaw->dIag << "\nnrm2_db=" << nrm2_db
       << "\nnrm2_pc_f=" << nrm2_pc_f << "\nnrm2_pc_x=" << nrm2_pc_x
       << "\nnrm2_xLambda=" << nrm2_xLambda;
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, ss.str().c_str());
  }
  CHKERR VecAXPY(pc_x, ddlambda, ctx->arcPtrRaw->xLambda);
  CHKERR mat_ctx->setLambda(pc_x, &ddlambda, SCATTER_REVERSE);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PCSetupArcLength(PC pc) {
  MoFEMFunctionBegin;
  void *void_ctx;
  CHKERR PCShellGetContext(pc, &void_ctx);
  PCArcLengthCtx *ctx = (PCArcLengthCtx *)void_ctx;
  CHKERR PCSetFromOptions(ctx->pC);
  CHKERR PCGetOperators(pc, &ctx->shellAij, &ctx->Aij);
  CHKERR PCSetOperators(ctx->pC, ctx->Aij, ctx->Aij);
  CHKERR PCSetUp(ctx->pC);
  // SetUp PC KSP solver
  CHKERR KSPSetType(ctx->kSP, KSPPREONLY);
  CHKERR KSPSetTabLevel(ctx->kSP, 3);
  CHKERR KSPSetFromOptions(ctx->kSP);
  CHKERR KSPSetOperators(ctx->kSP, ctx->Aij, ctx->Aij);
  CHKERR KSPSetPC(ctx->kSP, ctx->pC);
  CHKERR KSPSetUp(ctx->kSP);
  MoFEMFunctionReturn(0);
}

// ***********************
// Zero F_lambda vector

ZeroFLmabda::ZeroFLmabda(boost::shared_ptr<ArcLengthCtx> arc_ptr)
    : arcPtr(arc_ptr) {}

MoFEMErrorCode ZeroFLmabda::preProcess() {
  MoFEMFunctionBegin;
  switch (snes_ctx) {
  case CTX_SNESSETFUNCTION: {
    CHKERR VecZeroEntries(arcPtr->F_lambda);
    CHKERR VecGhostUpdateBegin(arcPtr->F_lambda, INSERT_VALUES,
                               SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(arcPtr->F_lambda, INSERT_VALUES, SCATTER_FORWARD);
  } break;
  default:
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Impossible case");
  }
  MoFEMFunctionReturn(0);
}

AssembleFlambda::AssembleFlambda(boost::shared_ptr<ArcLengthCtx> arc_ptr,
                                 boost::shared_ptr<DirichletDisplacementBc> bc)
    : arcPtr(arc_ptr), bC(bc) {}

MoFEMErrorCode AssembleFlambda::preProcess() {
  MoFEMFunctionBeginHot;
  MoFEMFunctionReturnHot(0);
}
MoFEMErrorCode AssembleFlambda::operator()() {
  MoFEMFunctionBeginHot;
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode AssembleFlambda::postProcess() {
  MoFEMFunctionBegin;
  switch (snes_ctx) {
  case CTX_SNESSETFUNCTION: {
    // F_lambda
    CHKERR VecAssemblyBegin(arcPtr->F_lambda);
    CHKERR VecAssemblyEnd(arcPtr->F_lambda);
    CHKERR VecGhostUpdateBegin(arcPtr->F_lambda, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(arcPtr->F_lambda, ADD_VALUES, SCATTER_REVERSE);
    if (bC) {
      for (std::vector<int>::iterator vit = bC->dofsIndices.begin();
           vit != bC->dofsIndices.end(); vit++) {
        CHKERR VecSetValue(arcPtr->F_lambda, *vit, 0, INSERT_VALUES);
      }
      CHKERR VecAssemblyBegin(arcPtr->F_lambda);
      CHKERR VecAssemblyEnd(arcPtr->F_lambda);
    }
    CHKERR VecGhostUpdateBegin(arcPtr->F_lambda, INSERT_VALUES,
                               SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(arcPtr->F_lambda, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecDot(arcPtr->F_lambda, arcPtr->F_lambda, &arcPtr->F_lambda2);
    
    // add F_lambda
    CHKERR VecAssemblyBegin(snes_f);
    CHKERR VecAssemblyEnd(snes_f);
    Vec l_snes_f, l_f_lambda;
    CHKERR VecGhostGetLocalForm(snes_f, &l_snes_f);
    CHKERR VecGhostGetLocalForm(arcPtr->F_lambda, &l_f_lambda);
    double lambda = arcPtr->getFieldData();
    int local_lambda_idx = arcPtr->getPetscLocalDofIdx();
    {
      double *f_array, *f_lambda_array;
      CHKERR VecGetArray(l_snes_f, &f_array);
      CHKERR VecGetArray(l_f_lambda, &f_lambda_array);
      int size = problemPtr->getNbLocalDofsRow();
      f_lambda_array[local_lambda_idx] = 0;
      for(int i = 0;i!=size;++i) {
        f_array[i] += lambda * f_lambda_array[i];
      }
      CHKERR VecRestoreArray(l_snes_f, &f_array);
      CHKERR VecRestoreArray(l_f_lambda, &f_lambda_array);
    }
    CHKERR VecGhostRestoreLocalForm(snes_f, &l_snes_f);
    CHKERR VecGhostRestoreLocalForm(arcPtr->F_lambda, &l_f_lambda);

    double snes_fnorm, snes_xnorm;
    CHKERR VecNorm(snes_f, NORM_2, &snes_fnorm);
    CHKERR VecNorm(snes_x, NORM_2, &snes_xnorm);
    PetscPrintf(PETSC_COMM_WORLD,
                "\tF_lambda2 = %6.4e snes_f norm = %6.4e "
                "snes_x norm = %6.4e "
                "lambda = %6.4g\n",
                arcPtr->F_lambda2, snes_fnorm, snes_xnorm, lambda);
    if (!boost::math::isfinite(snes_fnorm)) {
      CHKERR arcPtr->mField.getInterface<Tools>()->checkVectorForNotANumber(
          problemPtr, ROW, snes_f);
    }
  } break;
  default:
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Impossible case");
  }
  MoFEMFunctionReturn(0);
}

// ************************
// Simple arc-length method

SimpleArcLengthControl::SimpleArcLengthControl(
    boost::shared_ptr<ArcLengthCtx> &arc_ptr, const bool assemble)
    : FEMethod(), arcPtr(arc_ptr), aSsemble(assemble) {}

SimpleArcLengthControl::~SimpleArcLengthControl() {}

MoFEMErrorCode SimpleArcLengthControl::preProcess() {
  MoFEMFunctionBegin;
  switch (snes_ctx) {
  case CTX_SNESSETFUNCTION: {
    if (aSsemble) {
      CHKERR VecAssemblyBegin(snes_f);
      CHKERR VecAssemblyEnd(snes_f);
    }
    CHKERR calculateDxAndDlambda(snes_x);
    CHKERR calculateDb();
  } break;
  case CTX_SNESSETJACOBIAN: {
    if (aSsemble) {
      CHKERR MatAssemblyBegin(snes_B, MAT_FLUSH_ASSEMBLY);
      CHKERR MatAssemblyEnd(snes_B, MAT_FLUSH_ASSEMBLY);
    }
  } break;
  default:
    break;
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleArcLengthControl::operator()() {
  MoFEMFunctionBegin;
  switch (snes_ctx) {
  case CTX_SNESSETFUNCTION: {
    arcPtr->res_lambda = calculateLambdaInt() - arcPtr->s;
    CHKERR VecSetValue(snes_f, arcPtr->getPetscGlobalDofIdx(),
                       arcPtr->res_lambda, ADD_VALUES);
  } break;
  case CTX_SNESSETJACOBIAN: {
    arcPtr->dIag = arcPtr->beta;
    CHKERR MatSetValue(snes_B, arcPtr->getPetscGlobalDofIdx(),
                       arcPtr->getPetscGlobalDofIdx(), 1, ADD_VALUES);
  } break;
  default:
    break;
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleArcLengthControl::postProcess() {
  MoFEMFunctionBegin;
  switch (snes_ctx) {
  case CTX_SNESSETFUNCTION: {
    if (aSsemble) {
      CHKERR VecAssemblyBegin(snes_f);
      CHKERR VecAssemblyEnd(snes_f);
    }
  } break;
  case CTX_SNESSETJACOBIAN: {
    if (aSsemble) {
      CHKERR MatAssemblyBegin(snes_B, MAT_FLUSH_ASSEMBLY);
      CHKERR MatAssemblyEnd(snes_B, MAT_FLUSH_ASSEMBLY);
    }
    CHKERR VecGhostUpdateBegin(arcPtr->ghostDiag, INSERT_VALUES,
                               SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(arcPtr->ghostDiag, INSERT_VALUES, SCATTER_FORWARD);
  } break;
  default:
    break;
  }
  MoFEMFunctionReturn(0);
}

double SimpleArcLengthControl::calculateLambdaInt() {
  return arcPtr->beta * arcPtr->dLambda;
}

MoFEMErrorCode SimpleArcLengthControl::calculateDb() {
  MoFEMFunctionBegin;
  CHKERR VecZeroEntries(arcPtr->db);
  CHKERR VecGhostUpdateBegin(arcPtr->db, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(arcPtr->db, INSERT_VALUES, SCATTER_FORWARD);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SimpleArcLengthControl::calculateDxAndDlambda(Vec x) {
  MoFEMFunctionBegin;
  // Calculate dx
  CHKERR VecGhostUpdateBegin(arcPtr->x0, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(arcPtr->x0, INSERT_VALUES, SCATTER_FORWARD);

  Vec l_x, l_x0, l_dx;
  CHKERR VecGhostGetLocalForm(x, &l_x);
  CHKERR VecGhostGetLocalForm(arcPtr->x0, &l_x0);
  CHKERR VecGhostGetLocalForm(arcPtr->dx, &l_dx);
  {
    double *x_array, *x0_array, *dx_array;
    CHKERR VecGetArray(l_x, &x_array);
    CHKERR VecGetArray(l_x0, &x0_array);
    CHKERR VecGetArray(l_dx, &dx_array);
    int size =
        problemPtr->getNbLocalDofsRow() + problemPtr->getNbGhostDofsRow();
    for (int i = 0; i != size; ++i) {
      dx_array[i] = x_array[i]-x0_array[i];
    }
    CHKERR VecRestoreArray(l_x, &x_array);
    CHKERR VecRestoreArray(l_x0, &x0_array);
    CHKERR VecRestoreArray(l_dx, &dx_array);
  }
  CHKERR VecGhostRestoreLocalForm(x, &l_x);
  CHKERR VecGhostRestoreLocalForm(arcPtr->x0, &l_x0);
  CHKERR VecGhostRestoreLocalForm(arcPtr->dx, &l_dx);

  // Calculate dlambda
  if (arcPtr->getPetscLocalDofIdx() != -1) {
    double *array;
    CHKERR VecGetArray(arcPtr->dx, &array);
    arcPtr->dLambda = array[arcPtr->getPetscLocalDofIdx()];
    array[arcPtr->getPetscLocalDofIdx()] = 0;
    CHKERR VecRestoreArray(arcPtr->dx, &array);
  }
  CHKERR VecGhostUpdateBegin(arcPtr->ghosTdLambda, INSERT_VALUES,
                             SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(arcPtr->ghosTdLambda, INSERT_VALUES,
                           SCATTER_FORWARD);

  // Calculate dx2
  double x_nrm, x0_nrm;
  CHKERR VecNorm(x, NORM_2, &x_nrm);
  CHKERR VecNorm(arcPtr->x0, NORM_2, &x0_nrm);
  CHKERR VecDot(arcPtr->dx, arcPtr->dx, &arcPtr->dx2);
  PetscPrintf(PETSC_COMM_WORLD,
              "\tx norm = %6.4e x0 norm = %6.4e dx2 = %6.4e\n", x_nrm, x0_nrm,
              arcPtr->dx2);
  MoFEMFunctionReturn(0);
}

// ***************************
// Spherical arc-length control

SphericalArcLengthControl::SphericalArcLengthControl(ArcLengthCtx *arc_ptr_raw)
    : FEMethod(), arcPtrRaw(arc_ptr_raw) {}

SphericalArcLengthControl::SphericalArcLengthControl(
    boost::shared_ptr<ArcLengthCtx> &arc_ptr)
    : FEMethod(), arcPtrRaw(arc_ptr.get()), arcPtr(arc_ptr) {}

SphericalArcLengthControl::~SphericalArcLengthControl() {}

MoFEMErrorCode SphericalArcLengthControl::preProcess() {
  MoFEMFunctionBegin;
  switch (ts_ctx) {
  case CTX_TSSETIFUNCTION: {
    snes_ctx = CTX_SNESSETFUNCTION;
    snes_x = ts_u;
    snes_f = ts_F;
    break;
  }
  case CTX_TSSETIJACOBIAN: {
    snes_ctx = CTX_SNESSETJACOBIAN;
    snes_x = ts_u;
    snes_B = ts_B;
    break;
  }
  default:
    break;
  }
  switch (snes_ctx) {
  case CTX_SNESSETFUNCTION: {
    CHKERR calculateDxAndDlambda(snes_x);
    CHKERR calculateDb();
  } break;
  case CTX_SNESSETJACOBIAN: {
  } break;
  default:
    break;
  }
  MoFEMFunctionReturn(0);
}

double SphericalArcLengthControl::calculateLambdaInt() {
  return arcPtrRaw->alpha * arcPtrRaw->dx2 +
         pow(arcPtrRaw->dLambda, 2) * pow(arcPtrRaw->beta, 2) *
             arcPtrRaw->F_lambda2;
}

MoFEMErrorCode SphericalArcLengthControl::calculateDb() {
  MoFEMFunctionBegin;
  CHKERR VecCopy(arcPtrRaw->dx, arcPtrRaw->db);
  CHKERR VecScale(arcPtrRaw->db, 2 * arcPtrRaw->alpha);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SphericalArcLengthControl::operator()() {
  MoFEMFunctionBegin;
  switch (snes_ctx) {
  case CTX_SNESSETFUNCTION: {
    arcPtrRaw->res_lambda = calculateLambdaInt() - pow(arcPtrRaw->s, 2);
    CHKERR VecSetValue(snes_f, arcPtrRaw->getPetscGlobalDofIdx(),
                       arcPtrRaw->res_lambda, ADD_VALUES);
    PetscPrintf(arcPtrRaw->mField.get_comm(), "\tres_lambda = %6.4e\n",
                arcPtrRaw->res_lambda);
  } break;
  case CTX_SNESSETJACOBIAN: {
    arcPtrRaw->dIag =
        2 * arcPtrRaw->dLambda * pow(arcPtrRaw->beta, 2) * arcPtrRaw->F_lambda2;
    CHKERR MatSetValue(snes_B, arcPtrRaw->getPetscGlobalDofIdx(),
                       arcPtrRaw->getPetscGlobalDofIdx(), 1, ADD_VALUES);
  } break;
  default:
    break;
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SphericalArcLengthControl::postProcess() {
  MoFEMFunctionBegin;
  switch (ts_ctx) {
  case CTX_TSSETIFUNCTION: {
    snes_ctx = CTX_SNESSETFUNCTION;
    snes_x = ts_u;
    snes_f = ts_F;
    break;
  }
  case CTX_TSSETIJACOBIAN: {
    snes_ctx = CTX_SNESSETJACOBIAN;
    snes_x = ts_u;
    snes_B = ts_B;
    break;
  }
  default:
    break;
  }
  switch (snes_ctx) {
  case CTX_SNESSETFUNCTION: {
    PetscPrintf(arcPtrRaw->mField.get_comm(), "\tlambda = %6.4e\n",
                arcPtrRaw->getFieldData());
  } break;
  case CTX_SNESSETJACOBIAN: {
    CHKERR VecGhostUpdateBegin(arcPtrRaw->ghostDiag, INSERT_VALUES,
                               SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(arcPtrRaw->ghostDiag, INSERT_VALUES,
                             SCATTER_FORWARD);
    CHKERR MatAssemblyBegin(snes_B, MAT_FLUSH_ASSEMBLY);
    CHKERR MatAssemblyEnd(snes_B, MAT_FLUSH_ASSEMBLY);
    PetscPrintf(arcPtrRaw->mField.get_comm(), "\tdiag = %6.4e\n",
                arcPtrRaw->dIag);
  } break;
  default:
    break;
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode SphericalArcLengthControl::calculateDxAndDlambda(Vec x) {
  MoFEMFunctionBegin;
  // dx
  ierr = VecCopy(x, arcPtrRaw->dx);
  ierr = VecAXPY(arcPtrRaw->dx, -1, arcPtrRaw->x0);
  ierr = VecGhostUpdateBegin(arcPtrRaw->dx, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRG(ierr);
  ierr = VecGhostUpdateEnd(arcPtrRaw->dx, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRG(ierr);
  // dlambda
  if (arcPtrRaw->getPetscLocalDofIdx() != -1) {
    double *array;
    ierr = VecGetArray(arcPtrRaw->dx, &array);
    CHKERRG(ierr);
    arcPtrRaw->dLambda = array[arcPtrRaw->getPetscLocalDofIdx()];
    array[arcPtrRaw->getPetscLocalDofIdx()] = 0;
    ierr = VecRestoreArray(arcPtrRaw->dx, &array);
    CHKERRG(ierr);
  }
  ierr = VecGhostUpdateBegin(arcPtrRaw->ghosTdLambda, INSERT_VALUES,
                             SCATTER_FORWARD);
  CHKERRG(ierr);
  ierr = VecGhostUpdateEnd(arcPtrRaw->ghosTdLambda, INSERT_VALUES,
                           SCATTER_FORWARD);
  CHKERRG(ierr);
  // dx2
  ierr = VecDot(arcPtrRaw->dx, arcPtrRaw->dx, &arcPtrRaw->dx2);
  CHKERRG(ierr);
  PetscPrintf(arcPtrRaw->mField.get_comm(), "\tdlambda = %6.4e dx2 = %6.4e\n",
              arcPtrRaw->dLambda, arcPtrRaw->dx2);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
SphericalArcLengthControl::calculateInitDlambda(double *dlambda) {
  MoFEMFunctionBeginHot;
  *dlambda = sqrt(pow(arcPtrRaw->s, 2) /
                  (pow(arcPtrRaw->beta, 2) * arcPtrRaw->F_lambda2));
  if (!(*dlambda == *dlambda)) {
    std::ostringstream sss;
    sss << "s " << arcPtrRaw->s << " " << arcPtrRaw->beta << " "
        << arcPtrRaw->F_lambda2;
    SETERRQ(PETSC_COMM_SELF, MOFEM_IMPOSIBLE_CASE, sss.str().c_str());
  }
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode SphericalArcLengthControl::setDlambdaToX(Vec x, double dlambda) {
  MoFEMFunctionBeginHot;
  // check if local dof idx is non zero, i.e. that lambda is accessible from
  // this processor
  if (arcPtrRaw->getPetscLocalDofIdx() != -1) {
    double *array;
    ierr = VecGetArray(x, &array);
    CHKERRG(ierr);
    double lambda_old = array[arcPtrRaw->getPetscLocalDofIdx()];
    if (!(dlambda == dlambda)) {
      std::ostringstream sss;
      sss << "s " << arcPtrRaw->s << " " << arcPtrRaw->beta << " "
          << arcPtrRaw->F_lambda2;
      SETERRQ(PETSC_COMM_SELF, 1, sss.str().c_str());
    }
    array[arcPtrRaw->getPetscLocalDofIdx()] = lambda_old + dlambda;
    PetscPrintf(arcPtrRaw->mField.get_comm(),
                "\tlambda = %6.4e, %6.4e (%6.4e)\n", lambda_old,
                array[arcPtrRaw->getPetscLocalDofIdx()], dlambda);
    ierr = VecRestoreArray(x, &array);
    CHKERRG(ierr);
  }
  MoFEMFunctionReturnHot(0);
}
