/** \file NodeForce.cpp
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
#include <NodalForce.hpp>

using namespace boost::numeric;

NodalForce::MyFE::MyFE(MoFEM::Interface &m_field)
    : VertexElementForcesAndSourcesCore(m_field) {}

NodalForce::OpNodalForce::OpNodalForce(
    const std::string field_name, Vec _F, bCForce &data,
    boost::ptr_vector<MethodForForceScaling> &methods_op, bool use_snes_f)
    : VertexElementForcesAndSourcesCore::UserDataOperator(
          field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
      F(_F), useSnesF(use_snes_f), dAta(data), methodsOp(methods_op) {}

MoFEMErrorCode
NodalForce::OpNodalForce::doWork(int side, EntityType type,
                                 DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  EntityHandle ent = getNumeredEntFiniteElementPtr()->getEnt();
  if (dAta.nOdes.find(ent) == dAta.nOdes.end())
    MoFEMFunctionReturnHot(0);

  const auto &dof_ptr = data.getFieldDofs()[0];
  int rank = dof_ptr->getNbOfCoeffs();

  if (data.getIndices().size() != (unsigned int)rank) {
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "data inconsistency");
  }

  Nf.resize(3);
  for (int rr = 0; rr != rank; ++rr) {
    if (rr == 0) {
      Nf[0] = dAta.data.data.value3 * dAta.data.data.value1;
    } else if (rr == 1) {
      Nf[1] = dAta.data.data.value4 * dAta.data.data.value1;
    } else if (rr == 2) {
      Nf[2] = dAta.data.data.value5 * dAta.data.data.value1;
    } else {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "data inconsistency");
    }
  }

  CHKERR MethodForForceScaling::applyScale(getFEMethod(), methodsOp, Nf);

  Vec myF = F;
  if (useSnesF || F == PETSC_NULL) 
    myF = getKSPF();
  
  CHKERR VecSetValues(myF, data.getIndices().size(), &data.getIndices()[0],
                      &Nf[0], ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode NodalForce::addForce(const std::string field_name, Vec F,
                                    int ms_id, bool use_snes_f) {

  const CubitMeshSets *cubit_meshset_ptr;
  MeshsetsManager *mmanager_ptr;
  MoFEMFunctionBegin;
  CHKERR mField.getInterface(mmanager_ptr);
  CHKERR mmanager_ptr->getCubitMeshsetPtr(ms_id, NODESET, &cubit_meshset_ptr);
  CHKERR cubit_meshset_ptr->getBcDataStructure(mapForce[ms_id].data);
  CHKERR mField.get_moab().get_entities_by_type(
      cubit_meshset_ptr->meshset, MBVERTEX, mapForce[ms_id].nOdes, true);
  fe.getOpPtrVector().push_back(
      new OpNodalForce(field_name, F, mapForce[ms_id], methodsOp, use_snes_f));
  MoFEMFunctionReturn(0);
}

MetaNodalForces::TagForceScale::TagForceScale(MoFEM::Interface &m_field)
    : mField(m_field) {

  double def_scale = 1.;
  const EntityHandle root_meshset = mField.get_moab().get_root_set();
  rval = mField.get_moab().tag_get_handle(
      "_LoadFactor_Scale_", 1, MB_TYPE_DOUBLE, thScale,
      MB_TAG_CREAT | MB_TAG_EXCL | MB_TAG_MESH, &def_scale);
  if (rval == MB_ALREADY_ALLOCATED) {
    rval = mField.get_moab().tag_get_by_ptr(thScale, &root_meshset, 1,
                                            (const void **)&sCale);
    MOAB_THROW(rval);
  } else {
    MOAB_THROW(rval);
    rval =
        mField.get_moab().tag_set_data(thScale, &root_meshset, 1, &def_scale);
    MOAB_THROW(rval);
    rval = mField.get_moab().tag_get_by_ptr(thScale, &root_meshset, 1,
                                            (const void **)&sCale);
    MOAB_THROW(rval);
  }
}

MoFEMErrorCode MetaNodalForces::TagForceScale::scaleNf(const FEMethod *fe,
                                                       VectorDouble &Nf) {
  MoFEMFunctionBeginHot;
  Nf *= *sCale;
  MoFEMFunctionReturnHot(0);
}
