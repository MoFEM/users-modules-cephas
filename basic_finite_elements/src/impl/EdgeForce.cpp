
/** \file EdgeForce.cpp
  \ingroup mofem_static_boundary_conditions
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
#include <MethodForForceScaling.hpp>
#include <SurfacePressure.hpp>
#include <EdgeForce.hpp>

EdgeForce::OpEdgeForce::OpEdgeForce(
  const std::string field_name,Vec f,bCForce &data,
  boost::ptr_vector<MethodForForceScaling> &methods_op,
  bool use_snes_f
):
EdgeElementForcesAndSourcesCore::UserDataOperator(field_name,OPROW),
F(f),
dAta(data),
methodsOp(methods_op),
useSnesF(use_snes_f) {
}

MoFEMErrorCode EdgeForce::OpEdgeForce::doWork(int side,EntityType type,DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBeginHot;

  if(data.getIndices().size()==0) {
    MoFEMFunctionReturnHot(0);
  }
  EntityHandle ent = getNumeredEntFiniteElementPtr()->getEnt();
  if(dAta.eDges.find(ent)==dAta.eDges.end()) {
    MoFEMFunctionReturnHot(0);
  }

  

  // Get pointer to DOF and its rank
  const FENumeredDofEntity *dof_ptr;
  ierr = getNumeredEntFiniteElementPtr()->getRowDofsByPetscGlobalDofIdx(data.getIndices()[0],&dof_ptr); CHKERRG(ierr);
  int rank = dof_ptr->getNbOfCoeffs();

  int nb_dofs =  data.getIndices().size();

  Nf.resize(nb_dofs,false);
  Nf.clear();

  int nb_gauss_pts = data.getN().size1();
  wEights.resize(nb_gauss_pts,false);

  // This will work for fluxes and other fields with rank other than 3.
  for(int rr = 0;rr<rank;rr++) {

    // Get force value for each vector element from blockset data.
    double force;
    if(rr == 0) {
      force = dAta.data.data.value3*dAta.data.data.value1;
    } else if(rr == 1) {
      force= dAta.data.data.value4*dAta.data.data.value1;
    } else if(rr == 2) {
      force = dAta.data.data.value5*dAta.data.data.value1;
    } else {
      SETERRQ(PETSC_COMM_SELF,MOFEM_DATA_INCONSISTENCY,"data inconsistency");
    }

    // Integrate force on the line
    for(int gg = 0;gg<nb_gauss_pts;gg++) {

      if(!rr) {
        wEights[gg] = 0;
        if(getTangetAtGaussPts().size1()>0) {
          // This is if edge is curved, i.e. HO geometry
          for(int dd = 0;dd<3;dd++) {
            wEights[gg] += pow(getTangetAtGaussPts()(gg,dd),2);
          }
          wEights[gg] = sqrt(wEights[gg]);
        } else {
          wEights[gg] = getLength();
        }
        wEights[gg] *= getGaussPts()(1,gg);
      }

      cblas_daxpy(nb_dofs/rank,wEights[gg]*force,&data.getN()(gg,0),1,&Nf[rr],rank);

    }

  }

  // I time/step varying force or calculate in arc-length control. This hack
  // scale force appropriately, and is controlled for user
  ierr = MethodForForceScaling::applyScale(getFEMethod(),methodsOp,Nf); CHKERRG(ierr);

  // Assemble force into right-hand vector
  Vec myF = F;
  if(useSnesF || F == PETSC_NULL) {
    switch (getFEMethod()->ts_ctx) {
      case FEMethod::CTX_TSSETIFUNCTION: {
        const_cast<FEMethod*>(getFEMethod())->snes_ctx = FEMethod::CTX_SNESSETFUNCTION;
        const_cast<FEMethod*>(getFEMethod())->snes_x = getFEMethod()->ts_u;
        const_cast<FEMethod*>(getFEMethod())->snes_f = getFEMethod()->ts_F;
        break;
      }
      default:
      break;
    }
    myF = getFEMethod()->snes_f;
  }
  ierr = VecSetValues(
    myF,data.getIndices().size(),
    &data.getIndices()[0],&Nf[0],ADD_VALUES
  ); CHKERRG(ierr);

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode EdgeForce::addForce(const std::string field_name, Vec F,
                                   int ms_id, bool use_snes_f) {
  const CubitMeshSets *cubit_meshset_ptr;
  MeshsetsManager *mmanager_ptr;
  MoFEMFunctionBegin;
  CHKERR mField.getInterface(mmanager_ptr);
  CHKERR mmanager_ptr->getCubitMeshsetPtr(ms_id, NODESET, &cubit_meshset_ptr);
  CHKERR cubit_meshset_ptr->getBcDataStructure(mapForce[ms_id].data);
  CHKERR mField.get_moab().get_entities_by_type(
      cubit_meshset_ptr->meshset, MBEDGE, mapForce[ms_id].eDges, true);
  // Add operator for element, set data and entities operating on the data
  fe.getOpPtrVector().push_back(
      new OpEdgeForce(field_name, F, mapForce[ms_id], methodsOp, use_snes_f));
  MoFEMFunctionReturn(0);
}
