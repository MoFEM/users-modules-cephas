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

  Range eDges;

  BlockOptionDataSimpleRods()
      : simpleRodYoungModulus(-1), simpleRodSectionArea(-1) {}
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

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setBlocks() {
    MoFEMFunctionBegin;

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 3, "ROD") == 0) {

        const int id = bit->getMeshsetId();
        mapSimpleRod[id].eDges.clear();
        CHKERR mField.get_moab().get_entities_by_type(bit->getMeshset(), MBEDGE,
                                                      mapSimpleRod[id].eDges, true);

        std::vector<double> attributes;
        bit->getAttributes(attributes);
        if (attributes.size() != 2) {
          SETERRQ1(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
                   "Input mesh should have 2 attributes but there is %d",
                   attributes.size());
        }
        mapSimpleRod[id].iD = id;
        mapSimpleRod[id].simpleRodYoungModulus = attributes[0];
        mapSimpleRod[id].simpleRodSectionArea = attributes[1];
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

    // check if the volumes have associated degrees of freedom
    const int row_nb_dofs = row_data.getIndices().size();
    if (!row_nb_dofs)
      MoFEMFunctionReturnHot(0);

    const int col_nb_dofs = col_data.getIndices().size();
    if (!col_nb_dofs)
      MoFEMFunctionReturnHot(0);

    if (dAta.eDges.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
        dAta.eDges.end()) {
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
    double coeff = tension_stiffness/(L*L*L);

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

    // cerr << locK.size1();
    // for (int i = 0; i != 6; i++) {
    //   for (int j = 0; j != 6; j++) {
    //     cerr << locK(i, j) << " ";
    //   }
    //   cerr << "\n";
    // }
    // cerr << "\n";

    // Add computed values of spring stiffness to the global LHS matrix
    Mat B = getFEMethod()->ksp_B != PETSC_NULL ? getFEMethod()->ksp_B
                                               : getFEMethod()->snes_B;
    CHKERR MatSetValues(B, row_nb_dofs, &*row_data.getIndices().begin(),
                        col_nb_dofs, &*col_data.getIndices().begin(),
                        &locK(0, 0), ADD_VALUES);

    // // is symmetric
    // if (row_side != col_side || row_type != col_type) {
    //   transLocK.resize(col_nb_dofs, row_nb_dofs, false);
    //   noalias(transLocK) = trans(locK);

    //   CHKERR MatSetValues(B, col_nb_dofs, &*col_data.getIndices().begin(),
    //                       row_nb_dofs, &*row_data.getIndices().begin(),
    //                       &transLocK(0, 0), ADD_VALUES);
    // }

    MoFEMFunctionReturn(0);
  }
};


MoFEMErrorCode
MetaSimpleRodElement::addSimpleRodElements(MoFEM::Interface &m_field,
                               const std::string field_name,
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
  // Add entities to that element, here we add all eDges with SIMPLE_ROD_ELEMENT
  // from cubit
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
    if (bit->getName().compare(0, 19, "SIMPLE_ROD_ELEMENT") == 0) {
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

    // fe_simple_rod_rhs_ptr->getOpPtrVector().push_back(
    //     new OpCalculateVectorFieldValues<3>(field_name, commonDataPtr->xAtPts));
    // fe_simple_rod_rhs_ptr->getOpPtrVector().push_back(
    //     new OpCalculateVectorFieldValues<3>(mesh_nodals_positions,
    //                                         commonDataPtr->xInitAtPts));
    // fe_simple_rod_rhs_ptr->getOpPtrVector().push_back(
    //     new OpSimpleRodFs(commonDataPtr, sitSimpleRod.second, field_name));
  }
  //   cerr << "commonDataPtr has been used!!! " << commonDataPtr.use_count() <<
  //   " times" << endl;
  MoFEMFunctionReturn(0);
}

/**
FTensor::Tensor2<double, 3, 3> MetaSpringBC::transformLocalToGlobal(
    FTensor::Tensor1<double, 3> t_normal_local,
    FTensor::Tensor1<double, 3> t_tangent1_local,
    FTensor::Tensor1<double, 3> t_tangent2_local,
    FTensor::Tensor2<double, 3, 3> t_spring_local) {

  // FTensor indices
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;

  // Global base vectors
  FTensor::Tensor1<double, 3> t_e1(1., 0., 0.);
  FTensor::Tensor1<double, 3> t_e2(0., 1., 0.);
  FTensor::Tensor1<double, 3> t_e3(0., 0., 1.);

  // Direction cosines
  auto get_cosine = [&](auto x, auto xp) {
    return (x(i) * xp(i)) / (sqrt(x(i) * x(i)) * sqrt(xp(i) * xp(i)));
  };

  // Transformation matrix (tensor)
  FTensor::Tensor2<double, 3, 3> t_transformation_matrix(
      get_cosine(t_e1, t_normal_local), get_cosine(t_e1, t_tangent1_local),
      get_cosine(t_e1, t_tangent2_local), get_cosine(t_e2, t_normal_local),
      get_cosine(t_e2, t_tangent1_local), get_cosine(t_e2, t_tangent2_local),
      get_cosine(t_e3, t_normal_local), get_cosine(t_e3, t_tangent1_local),
      get_cosine(t_e3, t_tangent2_local));

  // Spring stiffness in global coordinate, Q*ls*Q^T
  FTensor::Tensor2<double, 3, 3> t_spring_global;
  t_spring_global(i, j) = t_transformation_matrix(i, k) * t_spring_local(k, l) *
                          t_transformation_matrix(j, l);

  return t_spring_global;
};
*/