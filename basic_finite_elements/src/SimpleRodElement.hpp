/** \file SimpleRodElements.hpp
  \brief Header file for SimpleRod element implementation
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

#ifndef __SIMPLERODELEMENT_HPP__
#define __SIMPLERODELEMENT_HPP__

/** \brief Set of functions declaring elements and setting operators
 * for simple rod element
 */
struct MetaSimpleRodElement {

  /**
   * \brief Declare SimpleRod element
   *
   * Search cubit sidesets and blockset ROD and declare
   volume
   * element

   * Blockset has to have name "ROD". The first attribute of the
   * blockset is Young's modulus.

   *
   * @param  m_field               Interface insurance
   * @param  field_name            Field name (e.g. SPATIAL_POSITION)
   * @param  mesh_nodals_positions Name of field on which ho-geometry is defined
   * @return                       Error code
   */
  static MoFEMErrorCode addSimpleRodElements(
      MoFEM::Interface &m_field, const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  /**
   * \brief Implementation of SimpleRod element. Set operators to calculate LHS
   * and RHS
   *
   * @param m_field                 Interface insurance
   * @param fe_simple_rod_lhs_ptr          Pointer to the FE instance for LHS
   * @param fe_simple_rod_rhs_ptr          Pointer to the FE instance for RHS
   * @param field_name              Field name (e.g. SPATIAL_POSITION)
   * @param mesh_nodals_positions   Name of field on which ho-geometry is
   * defined
   * @return                        Error code
   */
  static MoFEMErrorCode setSimpleRodOperators(
      MoFEM::Interface &m_field,
      boost::shared_ptr<EdgeElementForcesAndSourcesCoreBase>
          fe_simple_rod_lhs_ptr,
      boost::shared_ptr<EdgeElementForcesAndSourcesCoreBase>
          fe_simple_rod_rhs_ptr,
      const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");
};

#endif //__SIMPLERODELEMENT_HPP__