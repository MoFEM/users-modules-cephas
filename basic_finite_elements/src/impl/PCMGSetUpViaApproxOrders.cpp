/** \file PCMGSetUpViaApproxOrders.cpp
 * \brief implementation of multi-grid solver for p- adaptivity
 *
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
 * License along with MoFEM. If not, see <http://www.gnu.org/licenses/>
 */

#include <MoFEM.hpp>

#include <UnknownInterface.hpp>

using namespace MoFEM;

#include <PCMGSetUpViaApproxOrders.hpp>

#undef PETSC_VERSION_RELEASE
#define PETSC_VERSION_RELEASE 1

#if PETSC_VERSION_GE(3, 6, 0)
#include <petsc/private/petscimpl.h>
#else
#include <petsc-private/petscimpl.h>
#endif

#if PETSC_VERSION_GE(3, 6, 0)
#include <petsc/private/dmimpl.h> /*I  "petscdm.h"   I*/
// #include <petsc/private/vecimpl.h> /*I  "petscdm.h"   I*/
#else
#include <petsc-private/dmimpl.h>  /*I  "petscdm.h"   I*/
#include <petsc-private/vecimpl.h> /*I  "petscdm.h"   I*/
#endif

PCMGSubMatrixCtx::PCMGSubMatrixCtx(Mat a, IS is) : A(a), iS(is) {
  // Increase reference of petsc object (works like shared_ptr but unique for
  // PETSc)
  ierr = PetscObjectReference((PetscObject)A);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = PetscObjectReference((PetscObject)iS);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
}

PCMGSubMatrixCtx::~PCMGSubMatrixCtx() {
  ierr = MatDestroy(&A);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = ISDestroy(&iS);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
}

struct PCMGSubMatrixCtx_private : public PCMGSubMatrixCtx {
  PCMGSubMatrixCtx_private(Mat a, IS is)
      : PCMGSubMatrixCtx(a, is), isInitisalised(false) {
    PetscLogEventRegister("PCMGSubMatrixCtx_mult", 0, &MOFEM_EVENT_mult);
    PetscLogEventRegister("PCMGSubMatrixCtx_sor", 0, &MOFEM_EVENT_sor);
  }
  ~PCMGSubMatrixCtx_private() {
    if (isInitisalised) {
      ierr = VecScatterDestroy(&sCat);
      CHKERRABORT(PETSC_COMM_WORLD, ierr);
      ierr = VecDestroy(&X);
      CHKERRABORT(PETSC_COMM_WORLD, ierr);
      ierr = VecDestroy(&F);
      CHKERRABORT(PETSC_COMM_WORLD, ierr);
    }
  }
  template <InsertMode MODE>
  friend MoFEMErrorCode sub_mat_mult_generic(Mat a, Vec x, Vec f);
  friend MoFEMErrorCode sub_mat_sor(Mat mat, Vec b, PetscReal omega,
                                    MatSORType flag, PetscReal shift,
                                    PetscInt its, PetscInt lits, Vec x);

public:
  MoFEMErrorCode initData(Vec x) {
    MoFEMFunctionBegin;
    if (!isInitisalised) {
      CHKERR MatCreateVecs(A, &X, &F);
      CHKERR VecScatterCreate(X, iS, x, PETSC_NULL, &sCat);
      CHKERR VecZeroEntries(X);
      CHKERR VecZeroEntries(F);
      isInitisalised = true;
    }
    MoFEMFunctionReturn(0);
  }
  PetscLogEvent MOFEM_EVENT_mult;
  PetscLogEvent MOFEM_EVENT_sor;
  bool isInitisalised;
};

template <InsertMode MODE>
MoFEMErrorCode sub_mat_mult_generic(Mat a, Vec x, Vec f) {
  void *void_ctx;
  MoFEMFunctionBegin;
  CHKERR MatShellGetContext(a, &void_ctx);
  PCMGSubMatrixCtx_private *ctx = (PCMGSubMatrixCtx_private *)void_ctx;
  if (!ctx->isInitisalised) {
    CHKERR ctx->initData(x);
  }
  PetscLogEventBegin(ctx->MOFEM_EVENT_mult, 0, 0, 0, 0);
  CHKERR VecScatterBegin(ctx->sCat, x, ctx->X, INSERT_VALUES, SCATTER_REVERSE);
  CHKERR VecScatterEnd(ctx->sCat, x, ctx->X, INSERT_VALUES, SCATTER_REVERSE);
  CHKERR MatMult(ctx->A, ctx->X, ctx->F);
  CHKERR VecScatterBegin(ctx->sCat, ctx->F, f, MODE, SCATTER_FORWARD);
  CHKERR VecScatterEnd(ctx->sCat, ctx->F, f, MODE, SCATTER_FORWARD);
  PetscLogEventEnd(ctx->MOFEM_EVENT_mult, 0, 0, 0, 0);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode sub_mat_mult(Mat a, Vec x, Vec f) {
  return sub_mat_mult_generic<INSERT_VALUES>(a, x, f);
}

MoFEMErrorCode sub_mat_mult_add(Mat a, Vec x, Vec f) {
  return sub_mat_mult_generic<ADD_VALUES>(a, x, f);
}

MoFEMErrorCode sub_mat_sor(Mat mat, Vec b, PetscReal omega, MatSORType flag,
                           PetscReal shift, PetscInt its, PetscInt lits,
                           Vec x) {
  void *void_ctx;
  MoFEMFunctionBegin;
  CHKERR MatShellGetContext(mat, &void_ctx);
  PCMGSubMatrixCtx_private *ctx = (PCMGSubMatrixCtx_private *)void_ctx;
  if (!ctx->isInitisalised) {
    CHKERR ctx->initData(x);
  }
  PetscLogEventBegin(ctx->MOFEM_EVENT_sor, 0, 0, 0, 0);
  CHKERR VecScatterBegin(ctx->sCat, b, ctx->X, INSERT_VALUES, SCATTER_REVERSE);
  CHKERR VecScatterEnd(ctx->sCat, b, ctx->X, INSERT_VALUES, SCATTER_REVERSE);
  CHKERR MatSOR(ctx->A, ctx->X, omega, flag, shift, its, lits, ctx->F);
  CHKERR VecScatterBegin(ctx->sCat, ctx->F, x, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecScatterEnd(ctx->sCat, ctx->F, x, INSERT_VALUES, SCATTER_FORWARD);
  PetscLogEventEnd(ctx->MOFEM_EVENT_sor, 0, 0, 0, 0);
  MoFEMFunctionReturn(0);
}

DMMGViaApproxOrdersCtx::DMMGViaApproxOrdersCtx()
    : MoFEM::DMCtx(), aO(PETSC_NULL) {
  // std::cerr << "create dm\n";
}
DMMGViaApproxOrdersCtx::~DMMGViaApproxOrdersCtx() {
  ierr = destroyCoarseningIS();
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
}

MoFEMErrorCode DMMGViaApproxOrdersCtx::destroyCoarseningIS() {
  MoFEMFunctionBegin;
  for (unsigned int ii = 0; ii < coarseningIS.size(); ii++) {
    if (coarseningIS[ii])
      CHKERR ISDestroy(&coarseningIS[ii]);
  }
  for (unsigned int ii = 0; ii < kspOperators.size(); ii++) {
    if (kspOperators[ii])
      CHKERR MatDestroy(&kspOperators[ii]);
  }
  if (aO) {
    CHKERR AODestroy(&aO);
  }
  coarseningIS.clear();
  kspOperators.clear();
  shellMatrixCtxPtr.clear();
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
DMMGViaApproxOrdersCtx::query_interface(const MOFEMuuid &uuid,
                                        MoFEM::UnknownInterface **iface) const {
  MoFEMFunctionBeginHot;
  *iface = NULL;
  if (uuid == IDD_DMMGVIAAPPROXORDERSCTX) {
    *iface = static_cast<DMMGViaApproxOrdersCtx *>(
        const_cast<DMMGViaApproxOrdersCtx *>(this));
    MoFEMFunctionReturnHot(0);
  } else {
    SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY, "wrong interference");
  }

  ierr = DMCtx::query_interface(uuid, iface);
  CHKERRG(ierr);
  MoFEMFunctionReturnHot(0);
}

#define GET_DM_FIELD(DM)                                                       \
  MoFEM::UnknownInterface *iface;                                              \
  CHKERR((DMCtx *)DM->data)                                                    \
      ->query_interface(IDD_DMMGVIAAPPROXORDERSCTX, &iface);                   \
  DMMGViaApproxOrdersCtx *dm_field =                                           \
      static_cast<DMMGViaApproxOrdersCtx *>(iface);                            \
  NOT_USED(dm_field)

MoFEMErrorCode DMMGViaApproxOrdersGetCtx(DM dm, DMMGViaApproxOrdersCtx **ctx) {
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  MoFEMFunctionBeginHot;
  GET_DM_FIELD(dm);
  *ctx = dm_field;
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode DMMGViaApproxOrdersSetAO(DM dm, AO ao) {
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  MoFEMFunctionBegin;
  GET_DM_FIELD(dm);
  if (dm_field->aO) {
    // std::cerr << dm_field->aO << std::endl;
    CHKERR AODestroy(&dm_field->aO);
    // std::cerr << "destroy ao when adding\n";
  }
  dm_field->aO = ao;
  CHKERR PetscObjectReference((PetscObject)ao);
  // std::cerr << "add ao\n";
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DMMGViaApproxOrdersGetCoarseningISSize(DM dm, int *size) {
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  MoFEMFunctionBeginHot;
  GET_DM_FIELD(dm);
  *size = dm_field->coarseningIS.size();
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode DMMGViaApproxOrdersPushBackCoarseningIS(DM dm, IS is, Mat A,
                                                       Mat *subA,
                                                       bool create_sub_matrix,
                                                       bool shell_sub_a) {
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  MoFEMFunctionBegin;
  GET_DM_FIELD(dm);
  dm_field->coarseningIS.push_back(is);
  dm_field->shellMatrixCtxPtr.push_back(new PCMGSubMatrixCtx_private(A, is));
  if (is) {
    CHKERR PetscObjectReference((PetscObject)is);
  }
  if (is) {
    IS is2 = is;
    if (dm_field->aO) {
      CHKERR ISDuplicate(is, &is2);
      CHKERR ISCopy(is, is2);
      CHKERR AOApplicationToPetscIS(dm_field->aO, is2);
    }
    if (create_sub_matrix) {
      if (shell_sub_a) {
        int n, N;
        CHKERR ISGetSize(is, &N);
        CHKERR ISGetLocalSize(is, &n);
        MPI_Comm comm;
        CHKERR PetscObjectGetComm((PetscObject)A, &comm);
        CHKERR MatCreateShell(comm, n, n, N, N,
                              &(dm_field->shellMatrixCtxPtr.back()), subA);
        CHKERR MatShellSetOperation(*subA, MATOP_MULT,
                                    (void (*)(void))sub_mat_mult);
        CHKERR MatShellSetOperation(*subA, MATOP_MULT_ADD,
                                    (void (*)(void))sub_mat_mult_add);
        CHKERR MatShellSetOperation(*subA, MATOP_SOR,
                                    (void (*)(void))sub_mat_sor);
      } else {
        #if PETSC_VERSION_GE(3, 8, 0)
          CHKERR MatCreateSubMatrix(A, is2, is2, MAT_INITIAL_MATRIX, subA);
        #else
          CHKERR MatGetSubMatrix(A, is2, is2, MAT_INITIAL_MATRIX, subA);
        #endif
      }
    }
    if (dm_field->aO) {
      CHKERR ISDestroy(&is2);
    }
    dm_field->kspOperators.push_back(*subA);
    CHKERR PetscObjectReference((PetscObject)(*subA));
  } else {
    SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY, "data inconsistency");
  }
  PetscInfo(dm, "Push back IS to DMMGViaApproxOrders\n");
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DMMGViaApproxOrdersPopBackCoarseningIS(DM dm) {
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  MoFEMFunctionBegin;
  GET_DM_FIELD(dm);
  if (dm_field->coarseningIS.back()) {
    CHKERR ISDestroy(&dm_field->coarseningIS.back());
    dm_field->coarseningIS.pop_back();
  }
  if (dm_field->kspOperators.back()) {
    CHKERR MatDestroy(&dm_field->kspOperators.back());
  }
  dm_field->kspOperators.pop_back();
  PetscInfo(dm, "Pop back IS to DMMGViaApproxOrders\n");
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DMMGViaApproxOrdersClearCoarseningIS(DM dm) {
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  MoFEMFunctionBegin;
  GET_DM_FIELD(dm);
  CHKERR dm_field->destroyCoarseningIS();
  PetscInfo(dm, "Clear DMs data structures\n");
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DMMGViaApproxOrdersReplaceCoarseningIS(DM dm, IS *is_vec,
                                                      int nb_elems, Mat A,
                                                      int verb) {
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  MoFEMFunctionBegin;
  GET_DM_FIELD(dm);
  int nb_no_changed = 0;
  int nb_replaced = 0;
  int nb_deleted = 0;
  int nb_added = 0;
  std::vector<IS>::iterator it;
  it = dm_field->coarseningIS.begin();
  int ii = 0;
  for (; it != dm_field->coarseningIS.end(); it++, ii++) {
    if (ii < nb_elems) {
      PetscBool flg;
      CHKERR ISEqual(*it, is_vec[ii], &flg);
      if (!flg) {
        CHKERR ISDestroy(&*it);
        CHKERR MatDestroy(&dm_field->kspOperators[ii]);
        *it = is_vec[ii];
        CHKERR PetscObjectReference((PetscObject)is_vec[ii]);
        if (ii < nb_elems - 1) {
          IS is = is_vec[ii];
          if (dm_field->aO) {
            CHKERR ISDuplicate(is_vec[ii], &is);
            CHKERR ISCopy(is_vec[ii], is);
            CHKERR AOApplicationToPetscIS(dm_field->aO, is);
          }
          Mat subA;
          #if PETSC_VERSION_GE(3, 8, 0)
            CHKERR MatCreateSubMatrix(A, is, is, MAT_INITIAL_MATRIX, &subA);
          #else
            CHKERR MatGetSubMatrix(A, is, is, MAT_INITIAL_MATRIX, &subA);
          #endif
          CHKERR PetscObjectReference((PetscObject)subA);
          dm_field->kspOperators[ii] = subA;
          CHKERR MatDestroy(&subA);
          if (dm_field->aO) {
            CHKERR ISDestroy(&is);
          }
        } else {
          CHKERR PetscObjectReference((PetscObject)A);
          dm_field->kspOperators[ii] = A;
        }
        nb_replaced++;
      }
    } else {
      nb_no_changed++;
      continue;
    }
  }
  if (static_cast<int>(dm_field->coarseningIS.size()) < nb_elems) {
    for (; ii < nb_elems - 1; ii++) {
      Mat subA;
      CHKERR DMMGViaApproxOrdersPushBackCoarseningIS(dm, is_vec[ii], A, &subA,
                                                     true, false);
      CHKERR MatDestroy(&subA);
      nb_added++;
    }
    CHKERR DMMGViaApproxOrdersPushBackCoarseningIS(dm, is_vec[ii], A, &A, false,
                                                   false);
    nb_added++;
  } else {
    for (; ii < static_cast<int>(dm_field->coarseningIS.size()); ii++) {
      CHKERR DMMGViaApproxOrdersPopBackCoarseningIS(dm);
      nb_deleted++;
    }
  }
  MPI_Comm comm;
  CHKERR PetscObjectGetComm((PetscObject)dm, &comm);
  if (verb > 0) {
    PetscPrintf(comm,
                "DMMGViaApproxOrders nb_no_changed = %d, nb_replaced = %d, "
                "nb_added = %d, nb_deleted = %d, size = %d\n",
                nb_no_changed, nb_replaced, nb_added, nb_deleted,
                dm_field->coarseningIS.size());
  }
  PetscInfo(dm, "Replace IS to DMMGViaApproxOrders\n");
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DMMGViaApproxOrdersGetCtx(DM dm,
                                         const DMMGViaApproxOrdersCtx **ctx) {
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  MoFEMFunctionBeginHot;
  GET_DM_FIELD(dm);
  *ctx = dm_field;
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode DMRegister_MGViaApproxOrders(const char sname[]) {
  MoFEMFunctionBegin;
  CHKERR DMRegister(sname, DMCreate_MGViaApproxOrders);
  MoFEMFunctionReturn(0);
}

// MoFEMErrorCode DMDestroy_MGViaApproxOrders(DM dm) {
//   PetscValidHeaderSpecific(dm,DM_CLASSID,1);
//   MoFEMFunctionBeginHot;
//   if(!((DMMGViaApproxOrdersCtx*)dm->data)->referenceNumber) {
//     DMMGViaApproxOrdersCtx *dm_field = (DMMGViaApproxOrdersCtx*)dm->data;
//     if(dm_field->destroyProblem) {
//       if(dm_field->mField_ptr->check_problem(dm_field->problemName)) {
//         dm_field->mField_ptr->delete_problem(dm_field->problemName);
//       } // else problem has to be deleted by the user
//     }
//     cerr << "Destroy " << dm_field->problemName << endl;
//     delete (DMMGViaApproxOrdersCtx*)dm->data;
//   } else {
//     DMMGViaApproxOrdersCtx *dm_field = (DMMGViaApproxOrdersCtx*)dm->data;
//
//     cerr << "Derefrence " << dm_field->problemName << " "  <<
//     ((DMCtx*)dm->data)->referenceNumber << endl;
//     (((DMMGViaApproxOrdersCtx*)dm->data)->referenceNumber)--;
//   }
//   MoFEMFunctionReturnHot(0);
// }

static MoFEMErrorCode ksp_set_operators(KSP ksp, Mat A, Mat B, void *ctx) {
  MoFEMFunctionBeginHot;
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode DMCreate_MGViaApproxOrders(DM dm) {
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  MoFEMFunctionBegin;
  if (!dm->data) {
    dm->data = new DMMGViaApproxOrdersCtx();
  } else {
    ((DMCtx *)(dm->data))->referenceNumber++;
  }
  // cerr << "Create " << ((DMCtx*)(dm->data))->referenceNumber << endl;
  CHKERR DMSetOperators_MoFEM(dm);
  dm->ops->creatematrix = DMCreateMatrix_MGViaApproxOrders;
  dm->ops->createglobalvector = DMCreateGlobalVector_MGViaApproxOrders;
  dm->ops->coarsen = DMCoarsen_MGViaApproxOrders;
  // dm->ops->destroy = DMDestroy_MGViaApproxOrders;
  dm->ops->createinterpolation = DMCreateInterpolation_MGViaApproxOrders;
  CHKERR DMKSPSetComputeOperators(dm, ksp_set_operators, NULL);
  PetscInfo1(dm, "Create DMMGViaApproxOrders reference = %d\n",
             ((DMCtx *)(dm->data))->referenceNumber);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DMCreateMatrix_MGViaApproxOrders(DM dm, Mat *M) {

  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  MoFEMFunctionBegin;
  GET_DM_FIELD(dm);

  int leveldown = dm->leveldown;

  if (dm_field->kspOperators.empty()) {
    CHKERR DMCreateMatrix_MoFEM(dm, M);
  } else {
    MPI_Comm comm;
    CHKERR PetscObjectGetComm((PetscObject)dm, &comm);
    if (dm_field->kspOperators.empty()) {
      SETERRQ(comm, MOFEM_DATA_INCONSISTENCY,
              "data inconsistency, operator can not be set");
    }
    if (static_cast<int>(dm_field->kspOperators.size()) < leveldown) {
      SETERRQ(comm, MOFEM_DATA_INCONSISTENCY,
              "data inconsistency, no IS for that level");
    }
    *M = dm_field->kspOperators[dm_field->kspOperators.size() - 1 - leveldown];
    CHKERR PetscObjectReference((PetscObject)*M);
  }

  CHKERR MatSetDM(*M, dm);
  
  PetscInfo1(dm, "Create Matrix DMMGViaApproxOrders leveldown = %d\n",
             leveldown);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DMCoarsen_MGViaApproxOrders(DM dm, MPI_Comm comm, DM *dmc) {
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  MoFEMFunctionBegin;
  GET_DM_FIELD(dm);
  CHKERR PetscObjectGetComm((PetscObject)dm, &comm);
  CHKERR DMCreate(comm, dmc);
  (*dmc)->data = dm->data;
  DMType type;
  CHKERR DMGetType(dm, &type);
  CHKERR DMSetType(*dmc, type);
  CHKERR PetscObjectReference((PetscObject)(*dmc));
  PetscInfo1(dm, "Coarsen DMMGViaApproxOrders leveldown = %d\n", dm->leveldown);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DMCreateInterpolation_MGViaApproxOrders(DM dm1, DM dm2, Mat *mat,
                                                       Vec *vec) {
  PetscValidHeaderSpecific(dm1, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dm2, DM_CLASSID, 1);
  MoFEMFunctionBegin;

  MPI_Comm comm;
  CHKERR PetscObjectGetComm((PetscObject)dm1, &comm);

  int m, n, M, N;

  DM dm_down = dm1;
  DM dm_up = dm2;

  int dm_down_leveldown = dm_down->leveldown;
  int dm_up_leveldown = dm_up->leveldown;

  PetscInfo2(dm1,
             "Create interpolation DMMGViaApproxOrders dm1_leveldown = %d "
             "dm2_leveldown = %d\n",
             dm_down_leveldown, dm_up_leveldown);

  IS is_down, is_up;
  {
    // Coarser mesh
    GET_DM_FIELD(dm_down);
    if (static_cast<int>(dm_field->coarseningIS.size()) < dm_down_leveldown) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY, "data inconsistency");
    }
    is_down = dm_field->coarseningIS[dm_field->coarseningIS.size() - 1 -
                                     dm_down_leveldown];
    CHKERR ISGetSize(is_down, &M);
    CHKERR ISGetLocalSize(is_down, &m);
  }
  {
    // Finer mesh
    GET_DM_FIELD(dm_up);
    if (static_cast<int>(dm_field->coarseningIS.size()) < dm_up_leveldown) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY, "data inconsistency");
    }
    is_up =
        dm_field
            ->coarseningIS[dm_field->coarseningIS.size() - 1 - dm_up_leveldown];
    CHKERR ISGetSize(is_up, &N);
    CHKERR ISGetLocalSize(is_up, &n);
  }

  // is_dow rows
  // is_up columns

  CHKERR MatCreate(comm, mat);
  CHKERR MatSetSizes(*mat, m, n, M, N);
  CHKERR MatSetType(*mat, MATMPIAIJ);
  CHKERR MatMPIAIJSetPreallocation(*mat, 1, PETSC_NULL, 0, PETSC_NULL);

  // get matrix layout
  PetscLayout rmap, cmap;
  CHKERR MatGetLayouts(*mat, &rmap, &cmap);
  int rstart, rend, cstart, cend;
  CHKERR PetscLayoutGetRange(rmap, &rstart, &rend);
  CHKERR PetscLayoutGetRange(cmap, &cstart, &cend);

  // if(verb>0) {
  //   PetscSynchronizedPrintf(comm,"level %d row start %d row end
  //   %d\n",kk,rstart,rend); PetscSynchronizedPrintf(comm,"level %d col start
  //   %d col end %d\n",kk,cstart,cend);
  // }

  const int *row_indices_ptr, *col_indices_ptr;
  CHKERR ISGetIndices(is_down, &row_indices_ptr);
  CHKERR ISGetIndices(is_up, &col_indices_ptr);

  map<int, int> idx_map;
  for (int ii = 0; ii < m; ii++) {
    idx_map[row_indices_ptr[ii]] = rstart + ii;
  }

  CHKERR MatZeroEntries(*mat);
  // FIXME: Use MatCreateMPIAIJWithArrays and set array directly
  for (int jj = 0; jj < n; jj++) {
    map<int, int>::iterator mit = idx_map.find(col_indices_ptr[jj]);
    if (mit != idx_map.end()) {
      CHKERR MatSetValue(*mat, mit->second, cstart + jj, 1, INSERT_VALUES);
    }
  }

  CHKERR ISRestoreIndices(is_down, &row_indices_ptr);
  CHKERR ISRestoreIndices(is_up, &col_indices_ptr);

  CHKERR MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY);
  CHKERR MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY);

  if (vec != NULL) {
    *vec = PETSC_NULL;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DMCreateGlobalVector_MGViaApproxOrders(DM dm, Vec *g) {

  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  MoFEMFunctionBegin;
  int leveldown = dm->leveldown;
  GET_DM_FIELD(dm);
  if (dm_field->kspOperators.empty()) {
    CHKERR DMCreateGlobalVector_MoFEM(dm, g);
  } else {
#if PETSC_VERSION_GE(3, 5, 3)
    CHKERR MatCreateVecs(
        dm_field->kspOperators[dm_field->kspOperators.size() - 1 - leveldown],
        g, NULL);
#else
    CHKERR MatGetVecs(
        dm_field->kspOperators[dm_field->kspOperators.size() - 1 - leveldown],
        g, NULL);
#endif
  }
  CHKERR VecSetDM(*g, dm);

  PetscInfo1(dm, "Create global vector DMMGViaApproxOrders leveldown = %d\n",
             dm->leveldown);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PCMGSetUpViaApproxOrdersCtx::getOptions() {
  MoFEMFunctionBeginHot;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "",
                           "MOFEM Multi-Grid (Orders) pre-conditioner", "none");
  CHKERRG(ierr);

  CHKERR PetscOptionsInt("-mofem_mg_levels", "nb levels of multi-grid solver",
                         "", 2, &nbLevels, PETSC_NULL);
  CHKERR PetscOptionsInt("-mofem_mg_coarse_order",
                         "approximation order of coarse level", "", 2,
                         &coarseOrder, PETSC_NULL);
  CHKERR PetscOptionsInt("-mofem_mg_order_at_last_level", "order at last level",
                         "", 100, &orderAtLastLevel, PETSC_NULL);
  CHKERR PetscOptionsInt("-mofem_mg_verbose", "nb levels of multi-grid solver",
                         "", 0, &verboseLevel, PETSC_NULL);
  PetscBool shell_sub_a = shellSubA ? PETSC_TRUE : PETSC_FALSE;
  CHKERR PetscOptionsBool("-mofem_mg_shell_a", "use shell matrix as sub matrix",
                          "", shell_sub_a, &shell_sub_a, NULL);
  shellSubA = (shellSubA == PETSC_TRUE);

  ierr = PetscOptionsEnd();
  CHKERRG(ierr);
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode PCMGSetUpViaApproxOrdersCtx::createIsAtLevel(int kk, IS *is) {
  MoFEM::Interface *m_field_ptr;
  MoFEM::ISManager *is_manager_ptr;
  MoFEMFunctionBegin;
  // if is last level, take all remaining orders dofs, if any left
  CHKERR DMoFEMGetInterfacePtr(dM, &m_field_ptr);
  CHKERR m_field_ptr->getInterface(is_manager_ptr);
  const Problem *problem_ptr;
  CHKERR DMMoFEMGetProblemPtr(dM, &problem_ptr);
  int order_at_next_level = kk + coarseOrder;
  if (kk == nbLevels - 1) {
    int first = problem_ptr->getNumeredRowDofsPtr()
                    ->get<PetscLocalIdx_mi_tag>()
                    .find(0)
                    ->get()
                    ->getPetscGlobalDofIdx();
    CHKERR ISCreateStride(PETSC_COMM_WORLD, problem_ptr->getNbLocalDofsRow(),
                          first, 1, is);
    MoFEMFunctionReturnHot(0);
    // order_at_next_level = orderAtLastLevel;
  }
  string problem_name = problem_ptr->getName();
  CHKERR is_manager_ptr->isCreateProblemOrder(problem_name, ROW, 0,
                                              order_at_next_level, is);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PCMGSetUpViaApproxOrdersCtx::destroyIsAtLevel(int kk, IS *is) {
  MoFEMFunctionBegin;
  CHKERR ISDestroy(is);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
PCMGSetUpViaApproxOrdersCtx::buildProlongationOperator(bool use_mat_a,
                                                       int verb) {
  MoFEMFunctionBegin;
  verb = verb > verboseLevel ? verb : verboseLevel;

  MPI_Comm comm;
  CHKERR PetscObjectGetComm((PetscObject)dM, &comm);

  if (verb > QUIET) {
    PetscPrintf(comm, "set MG levels %u\n", nbLevels);
  }

  std::vector<IS> is_vec(nbLevels + 1);
  std::vector<int> is_glob_size(nbLevels + 1), is_loc_size(nbLevels + 1);

  for (int kk = 0; kk < nbLevels; kk++) {

    // get indices up to up to give approximation order
    CHKERR createIsAtLevel(kk, &is_vec[kk]);
    CHKERR ISGetSize(is_vec[kk], &is_glob_size[kk]);
    CHKERR ISGetLocalSize(is_vec[kk], &is_loc_size[kk]);

    if (verb > QUIET) {
      PetscSynchronizedPrintf(comm,
                              "Nb. dofs at level [ %d ] global %u local %d\n",
                              kk, is_glob_size[kk], is_loc_size[kk]);
    }

    // if no dofs on level kk finish here
    if (is_glob_size[kk] == 0) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "no dofs at level");
    }
  }

  for (int kk = 0; kk != nbLevels; kk++) {
    Mat subA;
    if (kk == nbLevels - 1 && use_mat_a) {
      subA = A;
      CHKERR DMMGViaApproxOrdersPushBackCoarseningIS(dM, is_vec[kk], A, &subA,
                                                     false, false);
    } else {
      if (kk > 0) {
        // Not coarse level
        CHKERR DMMGViaApproxOrdersPushBackCoarseningIS(dM, is_vec[kk], A, &subA,
                                                       true, shellSubA);
      } else {
        // Coarse lave is compressed matrix allowing for factorization when
        // needed
        CHKERR DMMGViaApproxOrdersPushBackCoarseningIS(dM, is_vec[kk], A, &subA,
                                                       true, false);
      }
      if (subA) {
        CHKERR MatDestroy(&subA);
      }
    }
  }

  for (unsigned int kk = 0; kk < is_vec.size(); kk++) {
    CHKERR destroyIsAtLevel(kk, &is_vec[kk]);
  }

  if (verb > QUIET) {
    PetscSynchronizedFlush(comm, PETSC_STDOUT);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode PCMGSetUpViaApproxOrders(PC pc, PCMGSetUpViaApproxOrdersCtx *ctx,
                                        int verb) {

  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  MoFEMFunctionBegin;

  MPI_Comm comm;
  CHKERR PetscObjectGetComm((PetscObject)pc, &comm);
  if (verb > 0) {
    PetscPrintf(comm, "Start PCMGSetUpViaApproxOrders\n");
  }

  CHKERR ctx->getOptions();
  CHKERR ctx->buildProlongationOperator(true, verb);

#if PETSC_VERSION_GE(3, 8, 0)
  CHKERR PCMGSetGalerkin(pc, PC_MG_GALERKIN_NONE);
#else
  CHKERR PCMGSetGalerkin(pc, PETSC_FALSE);
#endif

  CHKERR PCMGSetLevels(pc, ctx->nbLevels, NULL);

  if (verb > 0) {
    PetscPrintf(comm, "End PCMGSetUpViaApproxOrders\n");
  }

  MoFEMFunctionReturn(0);
}
