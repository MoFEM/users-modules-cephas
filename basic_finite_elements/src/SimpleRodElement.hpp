/** \file SimpleRodElements.hpp
  \brief Header file for SimpleRod element implementation
*/

/* MIT License
 *
 * Copyright (c) 2022
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

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
      boost::shared_ptr<EdgeElementForcesAndSourcesCore>
          fe_simple_rod_lhs_ptr,
      boost::shared_ptr<EdgeElementForcesAndSourcesCore>
          fe_simple_rod_rhs_ptr,
      const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");
};

#endif //__SIMPLERODELEMENT_HPP__