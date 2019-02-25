/** \file SpringElements.hpp
  \brief Header file for spring element implementation
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

#ifndef __SPRINGELEMENT_HPP__
#define __SPRINGELEMENT_HPP__

/** \brief Set of functions declaring elements and setting operators
 * to apply spring boundary condition
 */
struct MetaSpringBC {

  /**
   * \brief Declare spring element
   *
   * Search cubit sidesets and blocksets with spring bc and declare surface
   * element

   * Blockset has to have name “SPRING_BC”. The first three attributes of the
   * blockset are spring stiffness value.

   *
   * @param  m_field               Interface insurance
   * @param  field_name            Field name (e.g. SPATIAL_POSITION)
   * @param  mesh_nodals_positions Name of field on which ho-geometry is defined
   * @return                       Error code
   */
  static MoFEMErrorCode addSpringElements(
      MoFEM::Interface &m_field, const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  /**
   * \brief Implementation of spring element. Set operators to calculate LHS and
   * RHS
   *
   * @param m_field               Interface insurance
   * @param fe_spring_lhs_ptr     Pointer to the FE instance for LHS
   * @param fe_spring_rhs_ptr     Pointer to the FE instance for RHS
   * @param field_name            Field name (e.g. SPATIAL_POSITION)
   * @param mesh_nodals_positions Name of field on which ho-geometry is defined
   * @return                      Error code
   */
  static MoFEMErrorCode setSpringOperators(
      MoFEM::Interface &m_field,
      boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ptr,
      boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_rhs_ptr,
      const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  /**
   * \brief Implementation of spring element. Set operators to calculate LHS and
   * RHS
   *
   * @param t_tangent1      First local tangent vector
   * @param t_tangent2      Second local tangent vector
   * @param t_normal        Local normal vector
   * @param t_spring_local    Spring stiffness in local coords
   * @return t_spring_global  Spring stiffness in global coords
  //  */
  static FTensor::Tensor2<double, 3, 3>
  transformLocalToGlobal(FTensor::Tensor1<double, 3> t_normal,
                         FTensor::Tensor1<double, 3> t_tangent1,
                         FTensor::Tensor1<double, 3> t_tangent2,
                         FTensor::Tensor2<double, 3, 3> t_spring_local);
};

#endif //__SPRINGELEMENT_HPP__