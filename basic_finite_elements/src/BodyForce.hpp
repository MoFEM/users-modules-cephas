/** \file BodyForce.hpp
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

#ifndef __BODY_FORCE_HPP
#define __BODY_FORCE_HPP

/** \brief Body forces elements
 * \ingroup mofem_body_forces
 */
struct BodyForceConstantField {

  MoFEM::Interface &mField;

  struct MyVolumeFE : public MoFEM::VolumeElementForcesAndSourcesCore {
    MyVolumeFE(MoFEM::Interface &m_field)
        : MoFEM::VolumeElementForcesAndSourcesCore(m_field) {}
    int getRule(int order) { return order; };
  };

  MyVolumeFE fe;
  MyVolumeFE &getLoopFe() { return fe; }

  BodyForceConstantField(MoFEM::Interface &m_field)
      : mField(m_field), fe(m_field) {}

  struct OpBodyForce
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    Vec F;
    Block_BodyForces &dAta;
    Range blockTets;
    OpBodyForce(const std::string field_name, Vec _F, Block_BodyForces &data,
                Range block_tets)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          F(_F), dAta(data), blockTets(block_tets) {}

    VectorDouble Nf;

    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data) {
      MoFEMFunctionBegin;

      if (data.getIndices().size() == 0)
        MoFEMFunctionReturnHot(0);
      if (blockTets.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
          blockTets.end())
        MoFEMFunctionReturnHot(0);

      const auto &dof_ptr = data.getFieldDofs()[0];
      int rank = dof_ptr->getNbOfCoeffs();
      int nb_row_dofs = data.getIndices().size() / rank;

      Nf.resize(data.getIndices().size());
      bzero(&*Nf.data().begin(), data.getIndices().size() * sizeof(FieldData));

      for (unsigned int gg = 0; gg < data.getN().size1(); gg++) {
        double val = getVolume() * getGaussPts()(3, gg);
        for (int rr = 0; rr < rank; rr++) {

          double acc;
          if (rr == 0) {
            acc = -dAta.data.acceleration_x;
          } else if (rr == 1) {
            acc = -dAta.data.acceleration_y;
          } else if (rr == 2) {
            acc = -dAta.data.acceleration_z;
          } else {
            SETERRQ(PETSC_COMM_SELF, 1, "data inconsistency");
          }
          acc *= dAta.data.density;
          cblas_daxpy(nb_row_dofs, val * acc, &data.getN()(gg, 0), 1, &Nf[rr],
                      rank);
        }
      }

      CHKERR VecSetValues(F, data.getIndices().size(), &data.getIndices()[0],
                           &Nf[0], ADD_VALUES);

      MoFEMFunctionReturn(0);
    }
  };

  MoFEMErrorCode addBlock(const std::string field_name, Vec F, int ms_id) {
    const CubitMeshSets *cubit_meshset_ptr;
    MeshsetsManager *mmanager_ptr;
    MoFEMFunctionBegin;
    CHKERR mField.getInterface(mmanager_ptr);
    CHKERR mmanager_ptr->getCubitMeshsetPtr(ms_id, BLOCKSET,
                                            &cubit_meshset_ptr);
    CHKERR cubit_meshset_ptr->getAttributeDataStructure(mapData[ms_id]);
    EntityHandle meshset = cubit_meshset_ptr->getMeshset();
    Range tets;
    CHKERR mField.get_moab().get_entities_by_type(meshset, MBTET, tets, true);
    fe.getOpPtrVector().push_back(
        new OpBodyForce(field_name, F, mapData[ms_id], tets));
    MoFEMFunctionReturn(0);
  }

private:
  std::map<int, Block_BodyForces> mapData;
};

/// \brief USe BodyForceConstantField
DEPRECATED typedef BodyForceConstantField BodyFroceConstantField;

#endif //__BODY_FORCE_HPP

/**
 * \defgroup mofem_body_forces Body forces elements
 * \ingroup user_modules
 */
