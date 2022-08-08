/** \file SimpleRodElements.hpp
  \brief Header file for SimpleRod element implementation
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