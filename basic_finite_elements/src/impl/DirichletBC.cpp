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

static MoFEMErrorCode set_numered_dofs_on_ents(
    const Problem *problem_ptr, const FieldBitNumber bit_number, Range &ents,
    boost::function<
        MoFEMErrorCode(const boost::shared_ptr<MoFEM::NumeredDofEntity> &dof)>
        for_each_dof) {
  MoFEMFunctionBegin;

  auto &dofs_by_uid = problem_ptr->getNumeredRowDofsPtr()->get<Unique_mi_tag>();

  for (auto eit = ents.pair_begin(); eit != ents.pair_end(); ++eit) {

    auto lo_dit = dofs_by_uid.lower_bound(
        DofEntity::getLoFieldEntityUId(bit_number, eit->first));
    auto hi_dit = dofs_by_uid.upper_bound(
        DofEntity::getHiFieldEntityUId(bit_number, eit->second));

    for (; lo_dit != hi_dit; ++lo_dit) {
      auto &dof = *lo_dit;
      if (dof->getHasLocalIndex())
        CHKERR for_each_dof(dof);
    }
  }

  MoFEMFunctionReturn(0);
};

DirichletDisplacementBc::DirichletDisplacementBc(MoFEM::Interface &m_field,
                                                 const std::string &field_name,
                                                 Mat Aij, Vec X, Vec F,
                                                 string blockset_name,
                                                 bool is_partitioned)
    : mField(m_field), fieldName(field_name), blocksetName(blockset_name),
      isPartitioned(is_partitioned), dIag(1) {
  snes_B = Aij;
  snes_x = X;
  snes_f = F;
  ts_B = Aij;
  ts_u = X;
  ts_F = F;
};

DirichletDisplacementBc::DirichletDisplacementBc(MoFEM::Interface &m_field,
                                                 const std::string &field_name,
                                                 string blockset_name,
                                                 bool is_partitioned)
    : mField(m_field), fieldName(field_name), blocksetName(blockset_name),
      isPartitioned(is_partitioned), dIag(1) {
  snes_B = PETSC_NULL;
  snes_x = PETSC_NULL;
  snes_f = PETSC_NULL;
  ts_B = PETSC_NULL;
  ts_u = PETSC_NULL;
  ts_F = PETSC_NULL;
};

MoFEMErrorCode DirichletDisplacementBc::getBcDataFromSetsAndBlocks(
    std::vector<DataFromBc> &bc_data) {

  MoFEMFunctionBegin;

  // Loop over meshsets with Dirichlet boundary condition on displacements
  for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
           mField, NODESET | DISPLACEMENTSET, it)) {
    bc_data.push_back(DataFromBc());
    DisplacementCubitBcData mydata;
    CHKERR bc_data.back().getBcData(mydata, &(*it));
    CHKERR bc_data.back().getEntitiesFromBc(mField, &(*it));
  }
  // Loop over blocksets with DISPLACEMENT (default) boundary condition
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
    if (it->getName().compare(0, blocksetName.length(), blocksetName) == 0) {
      bc_data.push_back(DataFromBc());
      std::vector<double> mydata;
      CHKERR bc_data.back().getBcData(mydata, &(*it));
      CHKERR bc_data.back().getEntitiesFromBc(mField, &(*it));
    }
  }

  MoFEMFunctionReturn(0);
}

using FTensor1 = FTensor::Tensor1<double, 3>;

inline auto get_rotation_from_vector(FTensor1 &t_omega) {
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Tensor2<double, 3, 3> t_R;
  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
  t_R(i, j) = t_kd(i, j);

  const double angle = sqrt(t_omega(i) * t_omega(i));
  if (std::abs(angle) < 1e-18)
    return t_R;

  FTensor::Tensor2<double, 3, 3> t_Omega;
  t_Omega(i, j) = FTensor::levi_civita<double>(i, j, k) * t_omega(k);
  const double a = sin(angle) / angle;
  const double ss_2 = sin(angle / 2.);
  const double b = 2. * ss_2 * ss_2 / (angle * angle);
  t_R(i, j) += a * t_Omega(i, j);
  t_R(i, j) += b * t_Omega(i, k) * t_Omega(k, j);

  return t_R;
};

inline auto get_displacement(double *coords, FTensor1 t_centr,
                             FTensor1 t_normal, double theta) {
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  FTensor1 t_omega;
  FTensor1 t_coords(coords[0], coords[1], coords[2]);
  const double a = sqrt(t_normal(i) * t_normal(i));
  t_omega(i) = t_normal(i) * (theta / a);
  auto t_R = get_rotation_from_vector(t_omega);
  FTensor1 t_delta;
  t_delta(i) = t_centr(i) - t_coords(i);
  FTensor1 t_disp;
  t_disp(i) = t_delta(i) - t_R(i, j) * t_delta(j);

  VectorDouble disp_vec(3);
  for (int dd : {0, 1, 2})
    disp_vec(dd) = t_disp(dd);
  return disp_vec;
};

MoFEMErrorCode DirichletDisplacementBc::getRotationBcFromBlock(
    std::vector<DataFromBc> &bc_data) {

  MoFEMFunctionBegin;

  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
    if (it->getName().compare(0, 8, "ROTATION") == 0) {
      bc_data.push_back(DataFromBc());
      std::vector<double> mydata;
      CHKERR bc_data.back().getEntitiesFromBc(mField, &(*it));
      CHKERR it->getAttributes(mydata);
      if (mydata.size() < 6) {
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                "6 attributes are required for Rotation (3 center coords + 3 "
                "angles, (+ 3 optional) flags for xyz)");
      }
      for (int ii : {0, 1, 2}) {
        bc_data.back().bc_flags[ii] = 1;
        bc_data.back().t_centr(ii) = mydata[ii + 0];
        bc_data.back().t_normal(ii) = mydata[ii + 3];
      }
      if (mydata.size() > 8)
        for (int ii : {0, 1, 2})
          bc_data.back().bc_flags[ii] = mydata[6 + ii];

      bc_data.back().scaled_values[0] = bc_data.back().t_normal.l2();
      bc_data.back().is_rotation = true;
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DirichletDisplacementBc::calculateRotationForDof(
    MoFEM::Interface &m_field, EntityHandle ent, DataFromBc &bc_data) {
  MoFEMFunctionBegin;
  if (bc_data.is_rotation) {
    double coords[3];
    CHKERR m_field.get_moab().get_coords(&ent, 1, coords);
    bc_data.scaled_values = get_displacement(
        coords, bc_data.t_centr, bc_data.t_normal, bc_data.scaled_values[0]);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DirichletDisplacementBc::iNitialize() {
  MoFEMFunctionBegin;
  // if (!methodsOp.empty()) {
  if (1) {
    auto simple = mField.getInterface<Simple>();
    auto prb_mng = mField.getInterface<ProblemsManager>();
    auto field_ptr = mField.get_field_structure(fieldName);
    const int nb_coefficients = field_ptr->getNbOfCoeffs();

    auto remove_dofs_from_dirichlet_data = [&]() {
      MoFEMFunctionBeginHot;
      for (auto &bc_it : *bcDataPtr) {
        for (auto &ents : bc_it.bc_ents) {
          for (int i = 0; i != nb_coefficients; i++) {
            if (bc_it.bc_flags[i])
              CHKERR prb_mng->removeDofsOnEntities(problemPtr->getName(),
                                                   fieldName, ents, i, i);
          }
        }
      }

      MoFEMFunctionReturnHot(0);
    };

    auto remove_dofs_from_dirichlet_data_non_distributed = [&]() {
      MoFEMFunctionBeginHot;
      for (auto &bc_it : *bcDataPtr) {
        for (auto &ents : bc_it.bc_ents) {
          for (int i = 0; i != nb_coefficients; i++) {
            if (bc_it.bc_flags[i]) {
              CHKERR prb_mng->removeDofsOnEntitiesNotDistributed(
                  problemPtr->getName(), fieldName, ents, i, i);
            }
          }
        }
      }

      MoFEMFunctionReturnHot(0);
    };
    if (isPartitioned)
      CHKERR remove_dofs_from_dirichlet_data();
    else
      CHKERR remove_dofs_from_dirichlet_data_non_distributed();
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DirichletFixFieldAtEntitiesBc::iNitialize() {
  MoFEMFunctionBeginHot;
  auto prb_mng = mField.getInterface<ProblemsManager>();
  auto field_ptr = mField.get_field_structure(fieldName);
  const int nb_coefficients = field_ptr->getNbOfCoeffs();

  for (auto &field : fieldNames) {
    if (isPartitioned)
      CHKERR prb_mng->removeDofsOnEntities(problemPtr->getName(), field, eNts,
                                           0, nb_coefficients);
    else
      CHKERR prb_mng->removeDofsOnEntitiesNotDistributed(
          problemPtr->getName(), field, eNts, 0, nb_coefficients);
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode DirichletTemperatureBc::preProcess() {
  MoFEMFunctionBeginHot;

  auto get_bc_condition_data = [&](std::vector<DataFromBc> &bc_data) {
    MoFEMFunctionBeginHot;

    // Loop over blockset to find the block TEMPERATURE.
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, blocksetName.length(), blocksetName) == 0) {

        std::vector<double> mydata;
        CHKERR it->getAttributes(mydata);
        if (mydata.empty())
          SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                  "missing temperature attribute");
        bc_data.push_back(DataFromBc());
        CHKERR bc_data.back().getEntitiesFromBc(mField, &(*it));
        bc_data.back().scaled_values[0] = mydata[0];
        bc_data.back().bc_flags[0] = true;
      }
    }
    // look for temperature boundary conditions
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             mField, NODESET | TEMPERATURESET, it)) {
      TemperatureCubitBcData mydata;
      CHKERR it->getBcDataStructure(mydata);
      bc_data.push_back(DataFromBc());
      CHKERR bc_data.back().getEntitiesFromBc(mField, &(*it));
      bc_data.back().scaled_values[0] = mydata.data.value1;
      bc_data.back().bc_flags[0] = true;
    }

    MoFEMFunctionReturnHot(0);
  };

  if (!bcDataPtr) {
    bcDataPtr = boost::make_shared<vector<DataFromBc>>();
    CHKERR get_bc_condition_data(*bcDataPtr);
    CHKERR iNitialize();
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode DirichletDisplacementBc::preProcess() {
  MoFEMFunctionBegin;
  if (!bcDataPtr) {
    bcDataPtr = boost::make_shared<vector<DataFromBc>>();
    CHKERR getRotationBcFromBlock(*bcDataPtr);
    CHKERR getBcDataFromSetsAndBlocks(*bcDataPtr);
    CHKERR iNitialize();
  }

  for (auto &bc_it : *bcDataPtr) {
    Range all_bc_ents;
    CHKERR MethodForForceScaling::applyScale(this, methodsOp,
                                             bc_it.scaled_values);
    BcEntMethodDisp method(mField, bc_it);
    for (auto &ents : bc_it.bc_ents)
      all_bc_ents.merge(ents);
    CHKERR mField.loop_entities(fieldName, method, &all_bc_ents);
  }

  MoFEMFunctionReturnHot(0);

  if (snes_ctx == CTX_SNESNONE) {
    CHKERR VecAssemblyBegin(snes_x);
    CHKERR VecAssemblyEnd(snes_x);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DirichletSpatialPositionsBc::preProcess() {
  MoFEMFunctionBegin;
  if (!bcDataPtr) {
    bcDataPtr = boost::make_shared<vector<DataFromBc>>();
    CHKERR getRotationBcFromBlock(*bcDataPtr);
    CHKERR getBcDataFromSetsAndBlocks(*bcDataPtr);
    CHKERR iNitialize();
  }

  for (auto &bc_it : *bcDataPtr) {
    Range all_bc_ents;
    CHKERR MethodForForceScaling::applyScale(this, methodsOp,
                                             bc_it.scaled_values);
    BcEntMethodSpatial method(mField, bc_it, materialPositions);
    for (auto &ents : bc_it.bc_ents)
      all_bc_ents.merge(ents);
    CHKERR mField.loop_entities(fieldName, method, &all_bc_ents);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DirichletDisplacementBc::postProcess() {
  MoFEMFunctionBegin;

  if (snes_ctx == CTX_SNESNONE) {
    if (snes_B) {
      CHKERR MatAssemblyBegin(snes_B, MAT_FINAL_ASSEMBLY);
      CHKERR MatAssemblyEnd(snes_B, MAT_FINAL_ASSEMBLY);
    }
    if (snes_f) {
      CHKERR VecAssemblyBegin(snes_f);
      CHKERR VecAssemblyEnd(snes_f);
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DataFromBc::getBcData(DisplacementCubitBcData &mydata,
                                     const MoFEM::CubitMeshSets *it) {
  MoFEMFunctionBegin;
  // get data structure for boundary condition
  CHKERR it->getBcDataStructure(mydata);
  this->scaled_values[0] = mydata.data.value1;
  this->scaled_values[1] = mydata.data.value2;
  this->scaled_values[2] = mydata.data.value3;
  this->bc_flags[0] = (int)mydata.data.flag1;
  this->bc_flags[1] = (int)mydata.data.flag2;
  this->bc_flags[2] = (int)mydata.data.flag3;
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DataFromBc::getBcData(std::vector<double> &mydata,
                                     const MoFEM::CubitMeshSets *it) {
  MoFEMFunctionBegin;
  CHKERR it->getAttributes(mydata);
  if (mydata.size() < 6) {
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "six attributes are required for given BC blockset (3 values + "
            "3 flags)");
  }
  for (unsigned int ii = 0; ii < 3; ii++) {
    this->scaled_values[ii] = mydata[ii];
    this->bc_flags[ii] = (int)mydata[ii + 3];
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode DataFromBc::getEntitiesFromBc(MoFEM::Interface &mField,
                                             const MoFEM::CubitMeshSets *it) {
  MoFEMFunctionBegin;
  for (int dim = 0; dim < 3; dim++) {
    CHKERR it->getMeshsetIdEntitiesByDimension(mField.get_moab(), dim,
                                               bc_ents[dim], true);
    if (dim > 1) {
      Range edges;
      CHKERR mField.get_moab().get_adjacencies(bc_ents[dim], 1, false, edges,
                                               moab::Interface::UNION);
      bc_ents[1].insert(edges.begin(), edges.end());
    }
    if (dim > 0) {
      Range nodes;
      CHKERR mField.get_moab().get_connectivity(bc_ents[dim], nodes, true);
      bc_ents[0].insert(nodes.begin(), nodes.end());
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

    auto for_each_dof = [&](auto &dof) {
      MoFEMFunctionBeginHot;
      reaction_vec[dof->getDofCoeffIdx()] += array[dof->getPetscLocalDofIdx()];
      MoFEMFunctionReturnHot(0);
    };

    CHKERR set_numered_dofs_on_ents(problem_ptr,
                                    mField.get_field_bit_number(fieldName),
                                    verts, for_each_dof);

    double *res_array;

    CHKERR VecGetArray(v, &res_array);
    for (int dd = 0; dd != reaction_vec.size(); ++dd)
      res_array[dd] = reaction_vec[dd];
    CHKERR VecRestoreArray(v, &res_array);

    CHKERR VecGetArray(v, &res_array);
    for (int dd = 0; dd != reaction_vec.size(); ++dd)
      reaction_vec[dd] = res_array[dd];
    CHKERR VecRestoreArray(v, &res_array);
  }

  CHKERR VecGhostUpdateBegin(v, ADD_VALUES, SCATTER_REVERSE);
  CHKERR VecGhostUpdateEnd(v, ADD_VALUES, SCATTER_REVERSE);
  CHKERR VecGhostUpdateBegin(v, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(v, INSERT_VALUES, SCATTER_FORWARD);

  CHKERR VecDestroy(&v);
  CHKERR VecRestoreArrayRead(internal, &array);
  MoFEMFunctionReturn(0);
}
