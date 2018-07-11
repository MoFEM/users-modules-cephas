/** \file InterfaceGapArcLengthControl.hpp
  \brief Implementation of arc-lebgth control for cohesive elements

  Arc-length in that version controls gap opening

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

namespace CohesiveElement {

struct ArcLengthIntElemFEMethod: public FEMethod {
  moab::Interface& mOab;

  boost::shared_ptr<ArcLengthCtx> arcPtr;
  
  Vec GhostLambdaInt;
  Range Faces3,Faces4;
  Range Edges3,Edges4;
  Range Nodes3,Nodes4;

  Tag thDamagedPrism;

  ArcLengthIntElemFEMethod(
    moab::Interface& moab,
    boost::shared_ptr<ArcLengthCtx> arcptr):
    FEMethod(),mOab(moab),arcPtr(arcptr) {
    PetscInt ghosts[1] = { 0 };
    ParallelComm* pcomm = ParallelComm::get_pcomm(&mOab,MYPCOMM_INDEX);
    if(pcomm->rank() == 0) {
      VecCreateGhost(PETSC_COMM_WORLD,1,1,0,ghosts,&GhostLambdaInt);
    } else {
      VecCreateGhost(PETSC_COMM_WORLD,0,1,1,ghosts,&GhostLambdaInt);
    }
    Range prisms;
    rval = mOab.get_entities_by_type(0,MBPRISM,prisms,false); MOAB_THROW(rval);
    for(Range::iterator pit = prisms.begin();pit!=prisms.end();pit++) {
      EntityHandle f3,f4;
      rval = mOab.side_element(*pit,2,3,f3); MOAB_THROW(rval);
      rval = mOab.side_element(*pit,2,4,f4); MOAB_THROW(rval);
      Faces3.insert(f3);
      Faces4.insert(f4);
    }

    rval = mOab.get_adjacencies(Faces3,1,false,Edges3); MOAB_THROW(rval);
    rval = mOab.get_adjacencies(Faces4,1,false,Edges4); MOAB_THROW(rval);
    rval = mOab.get_connectivity(Faces3,Nodes3,true); MOAB_THROW(rval);
    rval = mOab.get_connectivity(Faces4,Nodes4,true); MOAB_THROW(rval);
    //Faces3.insert(Edges3.begin(),Edges3.end());
    Faces3.insert(Nodes3.begin(),Nodes3.end());
    //Faces4.insert(Edges4.begin(),Edges4.end());
    Faces4.insert(Nodes4.begin(),Nodes4.end());

    double def_damaged = 0;
    rval = mOab.tag_get_handle(
      "DAMAGED_PRISM",1,MB_TYPE_INTEGER,thDamagedPrism,MB_TAG_CREAT|MB_TAG_SPARSE,&def_damaged); MOAB_THROW(rval);

  }
  ~ArcLengthIntElemFEMethod() {
    VecDestroy(&GhostLambdaInt);
  }

  /** \brief remove nodes of prims which are fully damaged
    *
    */
  MoFEMErrorCode remove_damaged_prisms_nodes() {
    MoFEMFunctionBeginHot;
    Range prisms;
    rval = mOab.get_entities_by_type(0,MBPRISM,prisms,false); CHKERRG(rval);
    std::vector<int> is_prism_damaged(prisms.size());
    rval = mOab.tag_get_data(thDamagedPrism,prisms,&*is_prism_damaged.begin()); CHKERRG(rval);
    Range::iterator pit = prisms.begin();
    std::vector<int>::iterator vit = is_prism_damaged.begin();
    for(;pit!=prisms.end();pit++,vit++) {
      if(*vit>0) {
        Range nodes;
        rval = mOab.get_connectivity(&*pit,1,nodes,true); CHKERRG(rval);
        for(Range::iterator nit = nodes.begin();nit!=nodes.end();nit++) {
          Faces3.erase(*nit);
          Faces4.erase(*nit);
        }
      }
    }
    MoFEMFunctionReturnHot(0);
  }

  double lambda_int;
  MoFEMErrorCode preProcess() {
    MoFEMFunctionBeginHot;
    switch(snes_ctx) {
      case CTX_SNESSETFUNCTION: {
        ierr = calculate_dx_and_dlambda(snes_x); CHKERRG(ierr);
        ierr = calculate_db(); CHKERRG(ierr);
        ierr = calculate_lambda_int(lambda_int); CHKERRG(ierr);
      }
      break;
      default:
      break;
    }
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode calculate_lambda_int(double &_lambda_int_) {
    MoFEMFunctionBeginHot;
    ParallelComm* pcomm = ParallelComm::get_pcomm(&mOab,MYPCOMM_INDEX);
    NumeredDofEntityByLocalIdx::iterator dit,hi_dit;
    dit = problemPtr->getNumeredDofsRows()->get<PetscLocalIdx_mi_tag>().lower_bound(0);
    hi_dit = problemPtr->getNumeredDofsRows()->get<PetscLocalIdx_mi_tag>().upper_bound(problemPtr->getNbLocalDofsRow());
    double *array;
    double *array_int_lambda;
    ierr = VecZeroEntries(GhostLambdaInt); CHKERRG(ierr);
    ierr = VecGhostUpdateBegin(GhostLambdaInt,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(GhostLambdaInt,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    ierr = VecGetArray(arcPtr->dx,&array); CHKERRG(ierr);
    ierr = VecGetArray(GhostLambdaInt,&array_int_lambda); CHKERRG(ierr);
    array_int_lambda[0] = 0;
    for(;dit!=hi_dit;dit++) {
      if(dit->get()->getEntType() != MBVERTEX) continue;
      if(pcomm->rank() != dit->get()->getPart()) continue;
      if(Nodes3.find(dit->get()->getEnt())!=Nodes3.end()) {
        array_int_lambda[0] += array[dit->get()->getPetscLocalDofIdx()];
      }
      if(Nodes4.find(dit->get()->getEnt())!=Nodes4.end()) {
        array_int_lambda[0] -= array[dit->get()->getPetscLocalDofIdx()];
      }
    }
    //PetscSynchronizedPrintf(PETSC_COMM_WORLD,
    //  "array_int_lambda[0] = %6.4ee\n",array_int_lambda[0]);
    ierr = VecRestoreArray(arcPtr->dx,&array); CHKERRG(ierr);
    ierr = VecRestoreArray(GhostLambdaInt,&array_int_lambda); CHKERRG(ierr);
    ierr = VecGhostUpdateBegin(GhostLambdaInt,ADD_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(GhostLambdaInt,ADD_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
    ierr = VecGhostUpdateBegin(GhostLambdaInt,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(GhostLambdaInt,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    ierr = VecGetArray(GhostLambdaInt,&array_int_lambda); CHKERRG(ierr);
    _lambda_int_ = arcPtr->alpha*array_int_lambda[0] + arcPtr->dLambda*arcPtr->beta*sqrt(arcPtr->F_lambda2);
    /*PetscSynchronizedPrintf(PETSC_COMM_WORLD,
      "array_int_lambda[0] = %6.4e arcPtr->F_lambda2 = %6.4e\n",
      array_int_lambda[0],arcPtr->F_lambda2);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);*/
    ierr = VecRestoreArray(GhostLambdaInt,&array_int_lambda); CHKERRG(ierr);
    MoFEMFunctionReturnHot(0);
  }

  virtual MoFEMErrorCode calculate_db() {
    MoFEMFunctionBeginHot;
    ierr = VecZeroEntries(arcPtr->db); CHKERRG(ierr);
    ierr = VecGhostUpdateBegin(arcPtr->db,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(arcPtr->db,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    NumeredDofEntityByLocalIdx::iterator dit,hi_dit;
    dit = problemPtr->getNumeredDofsRows()->get<PetscLocalIdx_mi_tag>().lower_bound(0);
    hi_dit = problemPtr->getNumeredDofsRows()->get<PetscLocalIdx_mi_tag>().upper_bound(
      problemPtr->getNbLocalDofsRow()+problemPtr->getNbGhostDofsRow()
    );
    double *array;
    ierr = VecGetArray(arcPtr->db,&array); CHKERRG(ierr);
    for(;dit!=hi_dit;dit++) {
      if(dit->get()->getEntType() != MBVERTEX) {
        array[dit->get()->getPetscLocalDofIdx()] = 0;
        continue;
      }
      if(Nodes3.find(dit->get()->getEnt())!=Nodes3.end()) {
        array[dit->get()->getPetscLocalDofIdx()] = +arcPtr->alpha;
      } else if(Nodes4.find(dit->get()->getEnt())!=Nodes4.end()) {
        array[dit->get()->getPetscLocalDofIdx()] = -arcPtr->alpha;
      } else array[dit->get()->getPetscLocalDofIdx()] = 0;
    }
    ierr = VecRestoreArray(arcPtr->db,&array); CHKERRG(ierr);
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode operator()() {
    MoFEMFunctionBeginHot;

    switch(snes_ctx) {
      case CTX_SNESSETFUNCTION: {
        //calculate residual for arc length row
        arcPtr->res_lambda = lambda_int - arcPtr->s;
        ierr = VecSetValue(snes_f,arcPtr->getPetscGlobalDofIdx(),arcPtr->res_lambda,ADD_VALUES); CHKERRG(ierr);
        PetscPrintf(PETSC_COMM_SELF,"\tres_lambda = %6.4e lambda_int = %6.4e s = %6.4e\n",
        arcPtr->res_lambda,lambda_int,arcPtr->s);
      }
      break;
      case CTX_SNESSETJACOBIAN: {
        //calculate diagonal therm
        arcPtr->dIag = arcPtr->beta*sqrt(arcPtr->F_lambda2);
        ierr = MatSetValue(snes_B,arcPtr->getPetscGlobalDofIdx(),arcPtr->getPetscGlobalDofIdx(),1,ADD_VALUES); CHKERRG(ierr);
      }
      break;
      default:
      break;
    }

    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBeginHot;
    switch(snes_ctx) {
      case CTX_SNESSETJACOBIAN: {
        ierr = VecGhostUpdateBegin(arcPtr->ghostDiag,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
        ierr = VecGhostUpdateEnd(arcPtr->ghostDiag,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
        PetscPrintf(PETSC_COMM_WORLD,"\tdiag = %6.4e\n",arcPtr->dIag);
        ierr = MatAssemblyBegin(snes_B,MAT_FLUSH_ASSEMBLY); CHKERRG(ierr);
        ierr = MatAssemblyEnd(snes_B,MAT_FLUSH_ASSEMBLY); CHKERRG(ierr);
      }
      break;
      default:
      break;
    }
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode calculate_dx_and_dlambda(Vec &x) {
    MoFEMFunctionBeginHot;
    //dx
    ierr = VecCopy(x,arcPtr->dx); CHKERRG(ierr);
    ierr = VecAXPY(arcPtr->dx,-1,arcPtr->x0); CHKERRG(ierr);
    //if LAMBDA dof is on this partition
    if(arcPtr->getPetscLocalDofIdx()!=-1) {
      double *array;
      ierr = VecGetArray(arcPtr->dx,&array); CHKERRG(ierr);
      arcPtr->dLambda = array[arcPtr->getPetscLocalDofIdx()];
      array[arcPtr->getPetscLocalDofIdx()] = 0;
      ierr = VecRestoreArray(arcPtr->dx,&array); CHKERRG(ierr);
    }
    //brodcast dlambda
    ierr = VecGhostUpdateBegin(arcPtr->ghosTdLambda,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(arcPtr->ghosTdLambda,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    //calculate dx2 (dot product)
    ierr = VecDot(arcPtr->dx,arcPtr->dx,&arcPtr->dx2); CHKERRG(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"\tdlambda = %6.4e dx2 = %6.4e\n",arcPtr->dLambda,arcPtr->dx2);
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode calculate_init_dlambda(double *dlambda) {
    MoFEMFunctionBeginHot;

    *dlambda = arcPtr->s/(arcPtr->beta*sqrt(arcPtr->F_lambda2));
    PetscPrintf(
      PETSC_COMM_WORLD,
      "\tInit dlambda = %6.4e s = %6.4e beta = %6.4e F_lambda2 = %6.4e\n",
      *dlambda,arcPtr->s,arcPtr->beta,arcPtr->F_lambda2
    );
    double a = *dlambda;
    if(a - a != 0) {
      std::ostringstream sss;
      sss << "s " << arcPtr->s << " " << arcPtr->beta << " " << arcPtr->F_lambda2;
      SETERRQ(PETSC_COMM_SELF,1,sss.str().c_str());
    }

    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode set_dlambda_to_x(Vec &x,double dlambda) {
    MoFEMFunctionBeginHot;

    if(arcPtr->getPetscLocalDofIdx()!=-1) {
      double *array;
      ierr = VecGetArray(x,&array); CHKERRG(ierr);
      double lambda_old = array[arcPtr->getPetscLocalDofIdx()];
      if(!(dlambda == dlambda)) {
        std::ostringstream sss;
        sss << "s " << arcPtr->s << " " << arcPtr->beta << " " << arcPtr->F_lambda2;
        SETERRQ(PETSC_COMM_SELF,1,sss.str().c_str());
      }
      array[arcPtr->getPetscLocalDofIdx()] = lambda_old + dlambda;
      PetscPrintf(PETSC_COMM_WORLD,"\tlambda = %6.4e, %6.4e (%6.4e)\n",
      lambda_old, array[arcPtr->getPetscLocalDofIdx()], dlambda);
      ierr = VecRestoreArray(x,&array); CHKERRG(ierr);
    }

    MoFEMFunctionReturnHot(0);
  }

};

}
