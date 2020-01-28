/* \file SimpleRodElement.cpp
  \brief Implementation of SimpleRod element on eDges
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

#include <SimpleRodElement.hpp>
using namespace boost::numeric;

struct BlockOptionDataSimpleRods {
  int iD;

  double simpleRodYoungModulus;
  double simpleRodSectionArea;
  double simpleRodPreStress;

  Range eDges;

  BlockOptionDataSimpleRods()
      : simpleRodYoungModulus(-1), simpleRodSectionArea(-1),
        simpleRodPreStress(-1) {}
};

struct DataAtIntegrationPtsSimpleRods {

  boost::shared_ptr<MatrixDouble> gradDispPtr =
      boost::shared_ptr<MatrixDouble>(new MatrixDouble());
  boost::shared_ptr<MatrixDouble> xAtPts =
      boost::shared_ptr<MatrixDouble>(new MatrixDouble());
  boost::shared_ptr<MatrixDouble> xInitAtPts =
      boost::shared_ptr<MatrixDouble>(new MatrixDouble());

  double simpleRodYoungModulus;
  double simpleRodSectionArea;
  double simpleRodPreStress;

  std::map<int, BlockOptionDataSimpleRods> mapSimpleRod;
  //   ~DataAtIntegrationPtsSimpleRods() {}
  DataAtIntegrationPtsSimpleRods(MoFEM::Interface &m_field) : mField(m_field) {

    ierr = setBlocks();
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }

  MoFEMErrorCode getParameters() {
    MoFEMFunctionBegin; // They will be overwritten by BlockData
    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Problem", "none");

    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode getBlockData(BlockOptionDataSimpleRods &data) {
    MoFEMFunctionBegin;

    simpleRodYoungModulus = data.simpleRodYoungModulus;
    simpleRodSectionArea = data.simpleRodSectionArea;
    simpleRodPreStress = data.simpleRodPreStress;

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setBlocks() {
    MoFEMFunctionBegin;

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 3, "ROD") == 0) {

        const int id = bit->getMeshsetId();
        mapSimpleRod[id].eDges.clear();
        CHKERR mField.get_moab().get_entities_by_type(
            bit->getMeshset(), MBEDGE, mapSimpleRod[id].eDges, true);

        std::vector<double> attributes;
        bit->getAttributes(attributes);
        if (attributes.size() < 3) {
          SETERRQ1(
              PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
              "Input mesh for ROD should have 3 attributes but there is %d",
              attributes.size());
        }
        mapSimpleRod[id].iD = id;
        mapSimpleRod[id].simpleRodYoungModulus = attributes[0];
        mapSimpleRod[id].simpleRodSectionArea = attributes[1];
        mapSimpleRod[id].simpleRodPreStress = attributes[2];

        // Print spring blocks after being read
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "\nSimple rod block %d\n", id);
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "\tYoung's modulus %3.4g\n",
                           attributes[0]);
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "\tCross-section area %3.4g\n",
                           attributes[1]);
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "\tPrestress %3.4g\n",
                           attributes[2]);
      }
    }

    MoFEMFunctionReturn(0);
  }

private:
  MoFEM::Interface &mField;
};

/** * @brief Assemble contribution of SimpleRod element to LHS *
 *
 */
struct OpSimpleRodK : MoFEM::EdgeElementForcesAndSourcesCore::UserDataOperator {

  boost::shared_ptr<DataAtIntegrationPtsSimpleRods> commonDataPtr;
  BlockOptionDataSimpleRods &dAta;

  MatrixDouble locK;
  MatrixDouble transLocK;

  OpSimpleRodK(
      boost::shared_ptr<DataAtIntegrationPtsSimpleRods> &common_data_ptr,
      BlockOptionDataSimpleRods &data, const std::string field_name)
      : MoFEM::EdgeElementForcesAndSourcesCore::UserDataOperator(
            field_name.c_str(), field_name.c_str(), OPROWCOL),
        commonDataPtr(common_data_ptr), dAta(data) {
    sYmm = false;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    // check if the edge have associated degrees of freedom
    const int row_nb_dofs = row_data.getIndices().size();
    if (!row_nb_dofs)
      MoFEMFunctionReturnHot(0);

    const int col_nb_dofs = col_data.getIndices().size();
    if (!col_nb_dofs)
      MoFEMFunctionReturnHot(0);

    if (dAta.eDges.find(getFEEntityHandle()) == dAta.eDges.end()) {
      MoFEMFunctionReturnHot(0);
    }

    CHKERR commonDataPtr->getBlockData(dAta);
    // size associated to the entity
    locK.resize(row_nb_dofs, col_nb_dofs, false);
    locK.clear();

    double tension_stiffness = commonDataPtr->simpleRodYoungModulus *
                               commonDataPtr->simpleRodSectionArea;

    VectorDouble coords;
    coords = getCoords();
    double L = getLength();
    double coeff = tension_stiffness / (L * L * L);

    double x21 = coords(3) - coords(0);
    double y21 = coords(4) - coords(1);
    double z21 = coords(5) - coords(2);

    // Calculate element matrix
    locK(0, 0) = coeff * x21 * x21;
    locK(0, 1) = coeff * x21 * y21;
    locK(0, 2) = coeff * x21 * z21;
    locK(0, 3) = -coeff * x21 * x21;
    locK(0, 4) = -coeff * x21 * y21;
    locK(0, 5) = -coeff * x21 * z21;

    locK(1, 0) = locK(0, 1);
    locK(1, 1) = coeff * y21 * y21;
    locK(1, 2) = coeff * y21 * z21;
    locK(1, 3) = -coeff * y21 * x21;
    locK(1, 4) = -coeff * y21 * y21;
    locK(1, 5) = -coeff * y21 * z21;

    locK(2, 0) = locK(0, 2);
    locK(2, 1) = locK(1, 2);
    locK(2, 2) = coeff * z21 * z21;
    locK(2, 3) = -coeff * z21 * x21;
    locK(2, 4) = -coeff * z21 * y21;
    locK(2, 5) = -coeff * z21 * z21;

    locK(3, 0) = locK(0, 3);
    locK(3, 1) = locK(1, 3);
    locK(3, 2) = locK(2, 3);
    locK(3, 3) = coeff * x21 * x21;
    locK(3, 4) = coeff * x21 * y21;
    locK(3, 5) = coeff * x21 * z21;

    locK(4, 0) = locK(0, 4);
    locK(4, 1) = locK(1, 4);
    locK(4, 2) = locK(2, 4);
    locK(4, 3) = locK(3, 4);
    locK(4, 4) = coeff * y21 * y21;
    locK(4, 5) = coeff * y21 * z21;

    locK(5, 0) = locK(0, 5);
    locK(5, 1) = locK(1, 5);
    locK(5, 2) = locK(2, 5);
    locK(5, 3) = locK(3, 5);
    locK(5, 4) = locK(4, 5);
    locK(5, 5) = coeff * z21 * z21;

    CHKERR MatSetValues(getKSPB(), row_data, col_data, &locK(0, 0), ADD_VALUES);

    MoFEMFunctionReturn(0);
  }
};

/** * @brief Add ROD pre-stress to the RHS *
 */
struct OpSimpleRodPreStress
    : MoFEM::EdgeElementForcesAndSourcesCore::UserDataOperator {

  // vector used to store force vector for each degree of freedom
  VectorDouble nF;

  boost::shared_ptr<DataAtIntegrationPtsSimpleRods> commonDataPtr;
  BlockOptionDataSimpleRods &dAta;

  OpSimpleRodPreStress(
      boost::shared_ptr<DataAtIntegrationPtsSimpleRods> &common_data_ptr,
      BlockOptionDataSimpleRods &data, const std::string field_name)
      : MoFEM::EdgeElementForcesAndSourcesCore::UserDataOperator(
            field_name.c_str(), OPROW),
        commonDataPtr(common_data_ptr), dAta(data) {}

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {

    MoFEMFunctionBegin;

    // check that the edge have associated degrees of freedom
    const int nb_dofs = data.getIndices().size();
    if (nb_dofs == 0)
      MoFEMFunctionReturnHot(0);

    if (dAta.eDges.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
        dAta.eDges.end()) {
      MoFEMFunctionReturnHot(0);
    }

    CHKERR commonDataPtr->getBlockData(dAta);

    // size of force vector associated to the entity
    // set equal to the number of degrees of freedom of associated with the
    // entity
    nF.resize(nb_dofs, false);
    nF.clear();

    double axial_force =
        commonDataPtr->simpleRodSectionArea * commonDataPtr->simpleRodPreStress;
    
    auto dir = getDirection();
    dir /= norm_2(dir);
    for (auto d : {0, 1, 2}) {
      nF(d) = -axial_force * dir[d];
      nF(d + 3) = axial_force * dir[d];
    }

    CHKERR VecSetValues(getKSPf(), data, &nF[0], ADD_VALUES);

    MoFEMFunctionReturn(0);
  }
};

MoFEMErrorCode MetaSimpleRodElement::addSimpleRodElements(
    MoFEM::Interface &m_field, const std::string field_name,
    const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;

  // Define boundary element that operates on rows, columns and data of a
  // given field
  CHKERR m_field.add_finite_element("SIMPLE_ROD", MF_ZERO);
  CHKERR m_field.modify_finite_element_add_field_row("SIMPLE_ROD", field_name);
  CHKERR m_field.modify_finite_element_add_field_col("SIMPLE_ROD", field_name);
  CHKERR m_field.modify_finite_element_add_field_data("SIMPLE_ROD", field_name);
  if (m_field.check_field(mesh_nodals_positions)) {
    CHKERR m_field.modify_finite_element_add_field_data("SIMPLE_ROD",
                                                        mesh_nodals_positions);
  }
  // Add entities to that element, here we add all eDges with ROD
  // from cubit
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
    if (bit->getName().compare(0, 3, "ROD") == 0) {
      CHKERR m_field.add_ents_to_finite_element_by_type(bit->getMeshset(),
                                                        MBEDGE, "SIMPLE_ROD");
    }
  }
  CHKERR m_field.build_finite_elements("SIMPLE_ROD");

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaSimpleRodElement::setSimpleRodOperators(
    MoFEM::Interface &m_field,
    boost::shared_ptr<EdgeElementForcesAndSourcesCore> fe_simple_rod_lhs_ptr,
    boost::shared_ptr<EdgeElementForcesAndSourcesCore> fe_simple_rod_rhs_ptr,
    const std::string field_name, const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;

  // Push operators to instances for SimpleRod elements
  // loop over blocks
  boost::shared_ptr<DataAtIntegrationPtsSimpleRods> commonDataPtr =
      boost::make_shared<DataAtIntegrationPtsSimpleRods>(m_field);
  CHKERR commonDataPtr->getParameters();

  for (auto &sitSimpleRod : commonDataPtr->mapSimpleRod) {
    fe_simple_rod_lhs_ptr->getOpPtrVector().push_back(
        new OpSimpleRodK(commonDataPtr, sitSimpleRod.second, field_name));

    fe_simple_rod_rhs_ptr->getOpPtrVector().push_back(new OpSimpleRodPreStress(
        commonDataPtr, sitSimpleRod.second, field_name));
  }

  MoFEMFunctionReturn(0);
}