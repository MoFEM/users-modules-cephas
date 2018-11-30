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
#include <DirichletBC.hpp>

using namespace boost::numeric;

DirichletDisplacementBc::DirichletDisplacementBc(MoFEM::Interface &m_field,
                                                 const std::string &field_name,
                                                 Mat Aij, Vec X, Vec F)
    : mField(m_field), fieldName(field_name), dIag(1) {
  snes_B = Aij;
  snes_x = X;
  snes_f = F;
  ts_B = Aij;
  ts_u = X;
  ts_F = F;
};

DirichletDisplacementBc::DirichletDisplacementBc(MoFEM::Interface &m_field,
                                                 const std::string &field_name)
    : mField(m_field), fieldName(field_name), dIag(1) {
  snes_B = PETSC_NULL;
  snes_x = PETSC_NULL;
  snes_f = PETSC_NULL;
  ts_B = PETSC_NULL;
  ts_u = PETSC_NULL;
  ts_F = PETSC_NULL;
};

MoFEMErrorCode DirichletDisplacementBc::iNitalize() {
  MoFEMFunctionBegin;
  if (mapZeroRows.empty() || !methodsOp.empty()) {
    ParallelComm *pcomm =
        ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             mField, NODESET | DISPLACEMENTSET, it)) {
      DisplacementCubitBcData mydata;
      CHKERR it->getBcDataStructure(mydata);
      VectorDouble scaled_values(3);
      scaled_values[0] = mydata.data.value1;
      scaled_values[1] = mydata.data.value2;
      scaled_values[2] = mydata.data.value3;
      CHKERR MethodForForceScaling::applyScale(this, methodsOp, scaled_values);
      for (int dim = 0; dim < 3; dim++) {
        Range ents;
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), dim, ents,
                                                   true);
        if (dim > 1) {
          Range _edges;
          CHKERR mField.get_moab().get_adjacencies(ents, 1, false, _edges,
                                                   moab::Interface::UNION);
          ents.insert(_edges.begin(), _edges.end());
        }
        if (dim > 0) {
          Range _nodes;
          CHKERR mField.get_moab().get_connectivity(ents, _nodes, true);
          ents.insert(_nodes.begin(), _nodes.end());
        }
        for (Range::iterator eit = ents.begin(); eit != ents.end(); eit++) {
          for (_IT_NUMEREDDOF_ROW_BY_NAME_ENT_PART_FOR_LOOP_(
                   problemPtr, fieldName, *eit, pcomm->rank(), dof_ptr)) {
            const boost::shared_ptr<NumeredDofEntity> &dof = *dof_ptr;
            std::bitset<8> pstatus(dof->getPStatus());
            if (pstatus.test(0))
              continue; // only local
            if (dof->getEntType() == MBVERTEX) {
              if (dof->getDofCoeffIdx() == 0 && mydata.data.flag1) {
                mapZeroRows[dof->getPetscGlobalDofIdx()] = scaled_values[0];
              }
              if (dof->getDofCoeffIdx() == 1 && mydata.data.flag2) {
                mapZeroRows[dof->getPetscGlobalDofIdx()] = scaled_values[1];
              }
              if (dof->getDofCoeffIdx() == 2 && mydata.data.flag3) {
                mapZeroRows[dof->getPetscGlobalDofIdx()] = scaled_values[2];
              }
            } else {
              if (dof->getDofCoeffIdx() == 0 && mydata.data.flag1) {
                mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
              }
              if (dof->getDofCoeffIdx() == 1 && mydata.data.flag2) {
                mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
              }
              if (dof->getDofCoeffIdx() == 2 && mydata.data.flag3) {
                mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
              }
            }
          }
        }
      }
    }
    dofsIndices.resize(mapZeroRows.size());
    dofsValues.resize(mapZeroRows.size());
    int ii = 0;
    std::map<DofIdx, FieldData>::iterator mit = mapZeroRows.begin();
    for (; mit != mapZeroRows.end(); mit++, ii++) {
      dofsIndices[ii] = mit->first;
      dofsValues[ii] = mit->second;
      // std::cerr << dofsIndices[ii] << " " << dofsValues[ii] << std::endl;
    }
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DirichletDisplacementBc::preProcess() {
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
    snes_B = ts_B;
    break;
  }
  default:
    break;
  }

  CHKERR iNitalize();

  if (snes_ctx == CTX_SNESNONE && ts_ctx == CTX_TSNONE) {
    if (dofsIndices.size() > 0) {
      CHKERR VecSetValues(snes_x, dofsIndices.size(), &dofsIndices[0],
                          &dofsValues[0], INSERT_VALUES);
    }
    CHKERR VecAssemblyBegin(snes_x);
    CHKERR VecAssemblyEnd(snes_x);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DirichletDisplacementBc::postProcess() {
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
    snes_B = ts_B;
    break;
  }
  default:
    break;
  }

  if (snes_ctx == CTX_SNESNONE && ts_ctx == CTX_TSNONE) {
    if (snes_B) {
      CHKERR MatAssemblyBegin(snes_B, MAT_FINAL_ASSEMBLY);
      CHKERR MatAssemblyEnd(snes_B, MAT_FINAL_ASSEMBLY);
      CHKERR MatZeroRowsColumns(snes_B, dofsIndices.size(),
                                dofsIndices.empty() ? PETSC_NULL
                                                    : &dofsIndices[0],
                                dIag, PETSC_NULL, PETSC_NULL);
    }
    if (snes_f) {
      CHKERR VecAssemblyBegin(snes_f);
      CHKERR VecAssemblyEnd(snes_f);
      for (std::vector<int>::iterator vit = dofsIndices.begin();
           vit != dofsIndices.end(); vit++) {
        CHKERR VecSetValue(snes_f, *vit, 0, INSERT_VALUES);
      }
      CHKERR VecAssemblyBegin(snes_f);
      CHKERR VecAssemblyEnd(snes_f);
    }
  }

  switch (snes_ctx) {
  case CTX_SNESNONE:
    break;
  case CTX_SNESSETFUNCTION: {
    if (!dofsIndices.empty()) {
      dofsXValues.resize(dofsIndices.size());
      CHKERR VecGetValues(
          snes_x, dofsIndices.size(),
          dofsIndices.empty() ? PETSC_NULL : &*dofsIndices.begin(),
          dofsXValues.empty() ? PETSC_NULL : &*dofsXValues.begin());
    }
    CHKERR VecAssemblyBegin(snes_f);
    CHKERR VecAssemblyEnd(snes_f);
    if (!dofsIndices.empty()) {
      int ii = 0;
      for (std::vector<int>::iterator vit = dofsIndices.begin();
           vit != dofsIndices.end(); vit++, ii++) {
        double val = 0;
        if (!dofsXValues.empty()) {
          val += dofsXValues[ii];
          val += -mapZeroRows[*vit]; // in snes it is on the left hand side,
                                     // that way -1
          dofsXValues[ii] = val;
        }
      }
      CHKERR VecSetValues(
          snes_f, dofsIndices.size(),
          dofsIndices.empty() ? PETSC_NULL : &*dofsIndices.begin(),
          dofsXValues.empty() ? PETSC_NULL : &*dofsXValues.begin(),
          INSERT_VALUES);
    }
    CHKERR VecAssemblyBegin(snes_f);
    CHKERR VecAssemblyEnd(snes_f);
  } break;
  case CTX_SNESSETJACOBIAN: {
    CHKERR MatAssemblyBegin(snes_B, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(snes_B, MAT_FINAL_ASSEMBLY);
    CHKERR MatZeroRowsColumns(snes_B, dofsIndices.size(),
                              dofsIndices.empty() ? PETSC_NULL
                                                  : &*dofsIndices.begin(),
                              dIag, PETSC_NULL, PETSC_NULL);
  } break;
  default:
    SETERRQ(PETSC_COMM_SELF, 1, "unknown snes stage");
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DirichletSpatialPositionsBc::iNitalize() {
  MoFEMFunctionBegin;

  struct dataFromBc {
    VectorDouble scaled_values;
    VectorDouble bc_flags;
    Range bc_ents[3];
    dataFromBc() : scaled_values(3), bc_flags(3) {}

    MoFEMErrorCode getBcData(DisplacementCubitBcData &mydata,
                             const MoFEM::CubitMeshSets *it) {
      MoFEMFunctionBegin;
      // get data structure for boundary condition
      CHKERR it->getBcDataStructure(mydata);
      this->scaled_values[0] = mydata.data.value1;
      this->scaled_values[1] = mydata.data.value2;
      this->scaled_values[2] = mydata.data.value3;
      this->bc_flags[0] = mydata.data.flag1;
      this->bc_flags[1] = mydata.data.flag2;
      this->bc_flags[2] = mydata.data.flag3;

      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode getBcData(std::vector<double> &mydata,
                             const MoFEM::CubitMeshSets *it) {
      MoFEMFunctionBegin;
      CHKERR it->getAttributes(mydata);
      if (mydata.size() != 6) {
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                "six attributes are required for given BC blockset (3 values + "
                "3 flags)");
      }
      for (unsigned int ii = 0; ii < 3; ii++) {
        this->scaled_values[ii] = mydata[ii];
        this->bc_flags[ii] = mydata[ii + 3];
      }

      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode getEntitiesFromBc(MoFEM::Interface &mField,
                                     const MoFEM::CubitMeshSets *it) {
      MoFEMFunctionBegin;
      for (int dim = 0; dim < 3; dim++) {
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), dim,
                                                   bc_ents[dim], true);
        if (dim > 1) {
          Range edges;
          CHKERR mField.get_moab().get_adjacencies(
              bc_ents[dim], 1, false, edges, moab::Interface::UNION);
          bc_ents[dim].insert(edges.begin(), edges.end());
        }
        if (dim > 0) {
          Range nodes;
          CHKERR mField.get_moab().get_connectivity(bc_ents[dim], nodes, true);
          bc_ents[dim].insert(nodes.begin(), nodes.end());
        }
      }
      MoFEMFunctionReturn(0);
    }
  };

  if (mapZeroRows.empty() || !methodsOp.empty()) {
    const DofEntity_multiIndex *dofs_ptr;
    CHKERR mField.get_dofs(&dofs_ptr);
    // VectorDouble scaled_values(3);
    // sets kinetic boundary conditions by blockset.
    bool flag_cubit_disp = false;
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             mField, NODESET | DISPLACEMENTSET, it)) {
      flag_cubit_disp = true;
    }
    std::vector<dataFromBc> bcData;
    // Loop over meshsets with Dirichlet boundary condition on displacements
    if (flag_cubit_disp) {
      for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
               mField, NODESET | DISPLACEMENTSET, it)) {
        bcData.push_back(dataFromBc());
        DisplacementCubitBcData mydata;
        CHKERR bcData.back().getBcData(mydata, &(*it));
        CHKERR bcData.back().getEntitiesFromBc(mField, &(*it));
      }
    } else {
      for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
        if (it->getName().compare(0, blocksetName.length(), blocksetName) ==
            0) {
          bcData.push_back(dataFromBc());
          std::vector<double> mydata;
          CHKERR bcData.back().getBcData(mydata, &(*it));
          CHKERR bcData.back().getEntitiesFromBc(mField, &(*it));
        }
      }
    }

    for (auto &bc_it : bcData) {
      CHKERR MethodForForceScaling::applyScale(this, methodsOp,
                                               bc_it.scaled_values);
      for (int dim = 0; dim < 3; dim++) {
        for (Range::iterator eit = bc_it.bc_ents[dim].begin();
             eit != bc_it.bc_ents[dim].end(); eit++) {
          for (_IT_NUMEREDDOF_ROW_BY_NAME_ENT_PART_FOR_LOOP_(
                   problemPtr, fieldName, *eit, mField.get_comm_rank(),
                   dof_ptr)) {
            const boost::shared_ptr<NumeredDofEntity> &dof = *dof_ptr;
            if (dof->getEntType() == MBVERTEX) {
              EntityHandle node = dof->getEnt();
              cOords.resize(3);
              // find material nodal positions,
              DofEntity_multiIndex::index<
                  Composite_Name_And_Ent_mi_tag>::type::iterator mdit,
                  hi_mdit;
              mdit = dofs_ptr->get<Composite_Name_And_Ent_mi_tag>().lower_bound(
                  boost::make_tuple(materialPositions, *eit));
              hi_mdit =
                  dofs_ptr->get<Composite_Name_And_Ent_mi_tag>().upper_bound(
                      boost::make_tuple(materialPositions, *eit));
              if (mdit == hi_mdit) {
                // set coordinates form coords in MoAB
                CHKERR mField.get_moab().get_coords(&node, 1,
                                                    &*cOords.data().begin());
              } else {
                // set coordinates from field values
                for (; mdit != hi_mdit; mdit++) {
                  cOords[mdit->get()->getDofCoeffIdx()] =
                      mdit->get()->getFieldData();
                }
              }
              if (dof->getDofCoeffIdx() == 0 && bc_it.bc_flags[0]) {
                mapZeroRows[dof->getPetscGlobalDofIdx()] =
                    cOords[0] + bc_it.scaled_values[0];
              }
              if (dof->getDofCoeffIdx() == 1 && bc_it.bc_flags[1]) {
                mapZeroRows[dof->getPetscGlobalDofIdx()] =
                    cOords[1] + bc_it.scaled_values[1];
              }
              if (dof->getDofCoeffIdx() == 2 && bc_it.bc_flags[2]) {
                mapZeroRows[dof->getPetscGlobalDofIdx()] =
                    cOords[2] + bc_it.scaled_values[2];
              }
            } else {
              if (dof->getDofCoeffIdx() == 0 && bc_it.bc_flags[0]) {
                mapZeroRows[dof->getPetscGlobalDofIdx()] = dof->getFieldData();
              }
              if (dof->getDofCoeffIdx() == 1 && bc_it.bc_flags[1]) {
                mapZeroRows[dof->getPetscGlobalDofIdx()] = dof->getFieldData();
              }
              if (dof->getDofCoeffIdx() == 2 && bc_it.bc_flags[2]) {
                mapZeroRows[dof->getPetscGlobalDofIdx()] = dof->getFieldData();
              }
            }
          }
          // set boundary values to field data
          for (std::vector<std::string>::iterator fit = fixFields.begin();
               fit != fixFields.end(); fit++) {
            for (_IT_NUMEREDDOF_ROW_BY_NAME_ENT_PART_FOR_LOOP_(
                     problemPtr, *fit, *eit, mField.get_comm_rank(), dof_ptr)) {
              const boost::shared_ptr<NumeredDofEntity> &dof = *dof_ptr;
              mapZeroRows[dof->getPetscGlobalDofIdx()] = dof->getFieldData();
            }
          }
        }
      }
    }

    // set vector of values and indices
    dofsIndices.resize(mapZeroRows.size());
    dofsValues.resize(mapZeroRows.size());
    std::map<DofIdx, FieldData>::iterator mit = mapZeroRows.begin();
    for (int ii = 0; mit != mapZeroRows.end(); mit++, ii++) {
      dofsIndices[ii] = mit->first;
      dofsValues[ii] = mit->second;
    }
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DirichletTemperatureBc::iNitalize() {
  MoFEMFunctionBegin;
  if (mapZeroRows.empty() || !methodsOp.empty()) {
    ParallelComm *pcomm =
        ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             mField, NODESET | TEMPERATURESET, it)) {
      TemperatureCubitBcData mydata;
      CHKERR it->getBcDataStructure(mydata);
      VectorDouble scaled_values(1);
      scaled_values[0] = mydata.data.value1;
      CHKERR MethodForForceScaling::applyScale(this, methodsOp, scaled_values);
      for (int dim = 0; dim < 3; dim++) {
        Range ents;
        CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), dim, ents,
                                                   true);
        if (dim > 1) {
          Range _edges;
          CHKERR mField.get_moab().get_adjacencies(ents, 1, false, _edges,
                                                   moab::Interface::UNION);
          ents.insert(_edges.begin(), _edges.end());
        }
        if (dim > 0) {
          Range _nodes;
          CHKERR mField.get_moab().get_connectivity(ents, _nodes, true);
          ents.insert(_nodes.begin(), _nodes.end());
        }
        for (Range::iterator eit = ents.begin(); eit != ents.end(); eit++) {
          for (_IT_NUMEREDDOF_ROW_BY_NAME_ENT_PART_FOR_LOOP_(
                   problemPtr, fieldName, *eit, pcomm->rank(), dof_ptr)) {
            NumeredDofEntity *dof = dof_ptr->get();
            if (dof->getEntType() == MBVERTEX) {
              mapZeroRows[dof->getPetscGlobalDofIdx()] = scaled_values[0];
            } else {
              mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
            }
          }
        }
      }
    }
    dofsIndices.resize(mapZeroRows.size());
    dofsValues.resize(mapZeroRows.size());
    int ii = 0;
    std::map<DofIdx, FieldData>::iterator mit = mapZeroRows.begin();
    for (; mit != mapZeroRows.end(); mit++, ii++) {
      dofsIndices[ii] = mit->first;
      dofsValues[ii] = mit->second;
    }
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DirichletFixFieldAtEntitiesBc::iNitalize() {
  MoFEMFunctionBegin;
  if (mapZeroRows.empty()) {
    for (std::vector<std::string>::iterator fit = fieldNames.begin();
         fit != fieldNames.end(); fit++) {
      for (Range::iterator eit = eNts.begin(); eit != eNts.end(); eit++) {
        for (_IT_NUMEREDDOF_ROW_BY_NAME_ENT_PART_FOR_LOOP_(
                 problemPtr, *fit, *eit, mField.get_comm_rank(), dof_ptr)) {
          NumeredDofEntity *dof = dof_ptr->get();
          mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
        }
      }
    }
    dofsIndices.resize(mapZeroRows.size());
    dofsValues.resize(mapZeroRows.size());
    int ii = 0;
    std::map<DofIdx, FieldData>::iterator mit = mapZeroRows.begin();
    for (; mit != mapZeroRows.end(); mit++, ii++) {
      dofsIndices[ii] = mit->first;
      dofsValues[ii] = mit->second;
    }
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DirichletFixFieldAtEntitiesBc::preProcess() {
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
    snes_B = ts_B;
    break;
  }
  default:
    break;
  }

  CHKERR iNitalize();
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DirichletFixFieldAtEntitiesBc::postProcess() {
  MoFEMFunctionBegin;
  if (snes_ctx == CTX_SNESNONE && ts_ctx == CTX_TSNONE) {
    if (snes_B) {
      CHKERR MatAssemblyBegin(snes_B, MAT_FINAL_ASSEMBLY);
      CHKERR MatAssemblyEnd(snes_B, MAT_FINAL_ASSEMBLY);
      CHKERR MatZeroRowsColumns(snes_B, dofsIndices.size(),
                                dofsIndices.empty() ? PETSC_NULL
                                                    : &*dofsIndices.begin(),
                                dIag, PETSC_NULL, PETSC_NULL);
    }
    if (snes_f) {
      CHKERR VecAssemblyBegin(snes_f);
      CHKERR VecAssemblyEnd(snes_f);
      int ii = 0;
      for (std::vector<int>::iterator vit = dofsIndices.begin();
           vit != dofsIndices.end(); vit++, ii++) {
        CHKERR VecSetValue(snes_f, *vit, dofsValues[ii], INSERT_VALUES);
      }
      CHKERR VecAssemblyBegin(snes_f);
      CHKERR VecAssemblyEnd(snes_f);
    }
  }

  switch (snes_ctx) {
  case CTX_SNESNONE: {
  } break;
  case CTX_SNESSETFUNCTION: {
    CHKERR VecAssemblyBegin(snes_f);
    CHKERR VecAssemblyEnd(snes_f);
    int ii = 0;
    for (std::vector<int>::iterator vit = dofsIndices.begin();
         vit != dofsIndices.end(); vit++, ii++) {
      CHKERR VecSetValue(snes_f, *vit, dofsValues[ii], INSERT_VALUES);
    }
    CHKERR VecAssemblyBegin(snes_f);
    CHKERR VecAssemblyEnd(snes_f);
  } break;
  case CTX_SNESSETJACOBIAN: {
    CHKERR MatAssemblyBegin(snes_B, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(snes_B, MAT_FINAL_ASSEMBLY);
    CHKERR MatZeroRowsColumns(snes_B, dofsIndices.size(),
                              dofsIndices.empty() ? PETSC_NULL
                                                  : &*dofsIndices.begin(),
                              dIag, PETSC_NULL, PETSC_NULL);
  } break;
  default:
    SETERRQ(PETSC_COMM_SELF, 1, "unknown snes stage");
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DirichletSetFieldFromBlock::iNitalize() {
  MoFEMFunctionBegin;
  if (mapZeroRows.empty() || !methodsOp.empty()) {
    ParallelComm *pcomm =
        ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, blocksetName.length(), blocksetName) == 0) {
        std::vector<double> mydata;
        CHKERR it->getAttributes(mydata);
        VectorDouble scaled_values(mydata.size());
        for (unsigned int ii = 0; ii < mydata.size(); ii++) {
          scaled_values[ii] = mydata[ii];
        }
        CHKERR MethodForForceScaling::applyScale(this, methodsOp,
                                                 scaled_values);
        for (int dim = 0; dim < 3; dim++) {
          Range ents;
          CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), dim,
                                                     ents, true);
          if (dim > 1) {
            Range edges;
            CHKERR mField.get_moab().get_adjacencies(ents, 1, false, edges,
                                                     moab::Interface::UNION);
            ents.insert(edges.begin(), edges.end());
          }
          if (dim > 0) {
            Range nodes;
            CHKERR mField.get_moab().get_connectivity(ents, nodes, true);
            ents.insert(nodes.begin(), nodes.end());
          }
          for (Range::iterator eit = ents.begin(); eit != ents.end(); eit++) {
            for (_IT_NUMEREDDOF_ROW_BY_NAME_ENT_PART_FOR_LOOP_(
                     problemPtr, fieldName, *eit, pcomm->rank(), dof_ptr)) {
              NumeredDofEntity *dof = dof_ptr->get();
              if (dof->getEntType() == MBVERTEX) {
                if (dof->getDofCoeffIdx() == 0) {
                  mapZeroRows[dof->getPetscGlobalDofIdx()] = scaled_values[0];
                }
                if (dof->getDofCoeffIdx() == 1) {
                  mapZeroRows[dof->getPetscGlobalDofIdx()] = scaled_values[1];
                }
                if (dof->getDofCoeffIdx() == 2) {
                  mapZeroRows[dof->getPetscGlobalDofIdx()] = scaled_values[2];
                }
              } else {
                if (dof->getDofCoeffIdx() == 0) {
                  mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
                }
                if (dof->getDofCoeffIdx() == 1) {
                  mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
                }
                if (dof->getDofCoeffIdx() == 2) {
                  mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
                }
              }
            }
          }
        }
      }
    }
    dofsIndices.resize(mapZeroRows.size());
    dofsValues.resize(mapZeroRows.size());
    int ii = 0;
    std::map<DofIdx, FieldData>::iterator mit = mapZeroRows.begin();
    for (; mit != mapZeroRows.end(); mit++, ii++) {
      dofsIndices[ii] = mit->first;
      dofsValues[ii] = mit->second;
    }
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DirichletSetFieldFromBlockWithFlags::iNitalize() {
  MoFEMFunctionBegin;
  if (mapZeroRows.empty() || !methodsOp.empty()) {
    ParallelComm *pcomm =
        ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, blocksetName.length(), blocksetName) == 0) {
        std::vector<double> mydata;
        CHKERR it->getAttributes(mydata);
        VectorDouble scaled_values(mydata.size());
        for (unsigned int ii = 0; ii < mydata.size(); ii++) {
          scaled_values[ii] = mydata[ii];
        }

        CHKERR MethodForForceScaling::applyScale(this, methodsOp,
                                                 scaled_values);
        for (int dim = 0; dim < 3; dim++) {
          Range ents;
          CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), dim,
                                                     ents, true);
          if (dim > 1) {
            Range edges;
            CHKERR mField.get_moab().get_adjacencies(ents, 1, false, edges,
                                                     moab::Interface::UNION);
            ents.insert(edges.begin(), edges.end());
          }
          if (dim > 0) {
            Range nodes;
            CHKERR mField.get_moab().get_connectivity(ents, nodes, true);
            ents.insert(nodes.begin(), nodes.end());
          }
          for (Range::iterator eit = ents.begin(); eit != ents.end(); eit++) {
            for (_IT_NUMEREDDOF_ROW_BY_NAME_ENT_PART_FOR_LOOP_(
                     problemPtr, fieldName, *eit, pcomm->rank(), dof_ptr)) {
              NumeredDofEntity *dof = dof_ptr->get();
              if (dof->getEntType() == MBVERTEX) {
                if (dof->getDofCoeffIdx() == 0 && mydata[3]) {
                  mapZeroRows[dof->getPetscGlobalDofIdx()] = scaled_values[0];
                }
                if (dof->getDofCoeffIdx() == 1 && mydata[4]) {
                  mapZeroRows[dof->getPetscGlobalDofIdx()] = scaled_values[1];
                }
                if (dof->getDofCoeffIdx() == 2 && mydata[5]) {
                  mapZeroRows[dof->getPetscGlobalDofIdx()] = scaled_values[2];
                }
              } else {
                if (dof->getDofCoeffIdx() == 0 && mydata[3]) {
                  mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
                }
                if (dof->getDofCoeffIdx() == 1 && mydata[4]) {
                  mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
                }
                if (dof->getDofCoeffIdx() == 2 && mydata[5]) {
                  mapZeroRows[dof->getPetscGlobalDofIdx()] = 0;
                }
              }
            }
          }
        }
      }
    }
    dofsIndices.resize(mapZeroRows.size());
    dofsValues.resize(mapZeroRows.size());
    int ii = 0;
    std::map<DofIdx, FieldData>::iterator mit = mapZeroRows.begin();
    for (; mit != mapZeroRows.end(); mit++, ii++) {
      dofsIndices[ii] = mit->first;
      dofsValues[ii] = mit->second;
    }
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode Reactions::calculateReactions(Vec &internal) {

  MoFEMFunctionBegin;

  const Problem *problem_ptr;
  CHKERR mField.get_problem(problemName.c_str(), &problem_ptr);
  const double *array;
  CHKERR VecGetArrayRead(internal, &array);

  auto field_ptr = mField.get_field_structure(fieldName);
  const int nb_coefficients = field_ptr->getNbOfCoeffs();

  std::vector<int> ghosts(nb_coefficients);
  for (int g = 0; g != nb_coefficients; ++g)
    ghosts[g] = g;

  Vec v;
  CHKERR VecCreateGhost(
      mField.get_comm(), (mField.get_comm_rank() ? 0 : nb_coefficients),
      nb_coefficients, (mField.get_comm_rank() ? nb_coefficients : 0),
      &*ghosts.begin(), &v);

  for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
           mField, NODESET | DISPLACEMENTSET, it)) {

    const int id = it->getMeshsetId();
    VectorDouble &reaction_vec = reactionsMap[id];
    reaction_vec.resize(nb_coefficients);
    reaction_vec.clear();

    Range verts;
    for (int dim = 0; dim != 3; ++dim) {
      Range ents;
      CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), dim, ents,
                                                 true);
      Range nodes;
      CHKERR mField.get_moab().get_connectivity(ents, nodes, true);
      verts.insert(nodes.begin(), nodes.end());
    }

    for (auto eit : verts) {
      for (_IT_NUMEREDDOF_ROW_BY_NAME_ENT_PART_FOR_LOOP_(
               problem_ptr, fieldName, eit, mField.get_comm_rank(), dof_ptr)) {

        const boost::shared_ptr<NumeredDofEntity> &dof = *dof_ptr;
        std::bitset<8> pstatus(dof->getPStatus());
        if (pstatus.test(0))
          continue; // only local

        reaction_vec[dof->getDofCoeffIdx()] +=
            array[dof->getPetscLocalDofIdx()];
      }
    }

    double *res_array;

    CHKERR VecGetArray(v, &res_array);
    for (int dd = 0; dd != reaction_vec.size(); ++dd)
      res_array[dd] = reaction_vec[dd];
    CHKERR VecRestoreArray(v, &res_array);

    CHKERR VecGhostUpdateBegin(v, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(v, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateBegin(v, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(v, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR VecGetArray(v, &res_array);
    for (int dd = 0; dd != reaction_vec.size(); ++dd)
      reaction_vec[dd] = res_array[dd];
    CHKERR VecRestoreArray(v, &res_array);
  }
  CHKERR VecDestroy(&v);
  CHKERR VecRestoreArrayRead(internal, &array);
  MoFEMFunctionReturn(0);
}