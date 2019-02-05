/* \file SpringElement.cpp
  \brief Implementation of spring boundary condition on triangular surfaces
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

#include <SpringElement.hpp>

using namespace boost::numeric;

MoFEMErrorCode
MetaSpringBC::addSpringElements(MoFEM::Interface &m_field,
                                const std::string field_name,
                                const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;

  // Define boundary element that operates on rows, columns and data of a
  // given field
  CHKERR m_field.add_finite_element("SPRING");
  CHKERR m_field.modify_finite_element_add_field_row("SPRING", field_name);
  CHKERR m_field.modify_finite_element_add_field_col("SPRING", field_name);
  CHKERR m_field.modify_finite_element_add_field_data("SPRING", field_name);
  if (m_field.check_field(mesh_nodals_positions)) {
    CHKERR m_field.modify_finite_element_add_field_data("SPRING",
                                                        mesh_nodals_positions);
  }
  // Add entities to that element, here we add all triangles with SPRING_BC
  // from cubit
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
    if (bit->getName().compare(0, 9, "SPRING_BC") == 0) {
      CHKERR m_field.add_ents_to_finite_element_by_type(bit->getMeshset(),
                                                        MBTRI, "SPRING");
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaSpringBC::setSpringOperators(
    MoFEM::Interface &m_field,
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ptr,
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_rhs_ptr,
    const std::string field_name, const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;

  //   Create new instances of face elements for springs
  // boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ptr(
  //     new FaceElementForcesAndSourcesCore(m_field));
  // boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_rhs_ptr(
  //     new FaceElementForcesAndSourcesCore(m_field));

  // Push operators to instances for springs
  // loop over blocks
  boost::shared_ptr<DataAtIntegrationPtsSprings> commonData =
      boost::make_shared<DataAtIntegrationPtsSprings>(m_field);
  CHKERR commonData->getParameters();

  for (auto &sitSpring : commonData->mapSpring) {
    fe_spring_lhs_ptr->getOpPtrVector().push_back(
        new OpSpringKs(*commonData, sitSpring.second));

    fe_spring_rhs_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<3>(field_name, commonData->xAtPts));
    fe_spring_rhs_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<3>(mesh_nodals_positions,
                                            commonData->xInitAtPts));
    fe_spring_rhs_ptr->getOpPtrVector().push_back(
        new OpSpringFs(commonData, sitSpring.second));
  }
  cerr << "CommonData has been used!!! " << commonData.use_count() << "times" << endl;

  MoFEMFunctionReturn(0);
}