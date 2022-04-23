
/** \file EdgeForce.hpp
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

#ifndef __EDGE_FORCE_HPP__
#define __EDGE_FORCE_HPP__

/** \brief Force on edges and lines
 */
struct EdgeForce {

  MoFEM::Interface &mField;
  EdgeForce(MoFEM::Interface &m_field) : mField(m_field), fe(m_field, 1) {}

  struct MyFE : public MoFEM::EdgeElementForcesAndSourcesCore {
    int addToRule;
    MyFE(MoFEM::Interface &m_field, int add_to_rule)
        : EdgeElementForcesAndSourcesCore(m_field), addToRule(add_to_rule) {}
    int getRule(int order) { return order + addToRule; };
  };

  MyFE fe;
  MyFE &getLoopFe() { return fe; }

  struct bCForce {
    ForceCubitBcData data;
    Range eDges;
  };
  std::map<int, bCForce> mapForce;

  boost::ptr_vector<MethodForForceScaling> methodsOp;

  struct OpEdgeForce
      : public MoFEM::EdgeElementForcesAndSourcesCore::UserDataOperator {

    Vec F;
    bCForce &dAta;
    boost::ptr_vector<MethodForForceScaling> &methodsOp;
    bool useSnesF;

    OpEdgeForce(const std::string field_name, Vec f, bCForce &data,
                boost::ptr_vector<MethodForForceScaling> &methods_op,
                bool use_snes_f = false);

    VectorDouble wEights;
    VectorDouble Nf;

    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data);
  };

  MoFEMErrorCode addForce(const std::string field_name, Vec F, int ms_id,
                          bool use_snes_f = false);
};

struct MetaEdgeForces {

  /// Add element taking information from NODESET
  static MoFEMErrorCode addElement(MoFEM::Interface &m_field,
                                   const std::string field_name,
                                   Range *intersect_ptr = NULL) {
    MoFEMFunctionBegin;
    CHKERR m_field.add_finite_element("FORCE_FE", MF_ZERO);
    CHKERR m_field.modify_finite_element_add_field_row("FORCE_FE", field_name);
    CHKERR m_field.modify_finite_element_add_field_col("FORCE_FE", field_name);
    CHKERR m_field.modify_finite_element_add_field_data("FORCE_FE", field_name);
    if (m_field.check_field("MESH_NODE_POSITIONS")) {
      CHKERR m_field.modify_finite_element_add_field_data(
          "FORCE_FE", "MESH_NODE_POSITIONS");
    }
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                    it)) {
      Range tris;
      CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI, tris,
                                                     true);
      Range edges;
      CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBEDGE, edges,
                                                     true);
      Range tris_edges;
      CHKERR m_field.get_moab().get_adjacencies(tris, 1, false, tris_edges,
                                                moab::Interface::UNION);
      edges = subtract(edges, tris_edges);
      if (intersect_ptr) {
        edges = intersect(edges, *intersect_ptr);
      }
      CHKERR m_field.add_ents_to_finite_element_by_type(edges, MBEDGE,
                                                        "FORCE_FE");
    }
    MoFEMFunctionReturn(0);
  }

  /// Set integration point operators
  static MoFEMErrorCode
  setOperators(MoFEM::Interface &m_field,
               boost::ptr_map<std::string, EdgeForce> &edge_forces, Vec F,
               const std::string field_name,
               std::string mesh_node_positions = "MESH_NODE_POSITIONS") {
    MoFEMFunctionBegin;
    string fe_name = "FORCE_FE";
    edge_forces.insert(fe_name, new EdgeForce(m_field));
    if (m_field.check_field(mesh_node_positions)) {
      auto &fe = edge_forces.at(fe_name).getLoopFe();
      fe.getOpPtrVector().push_back(
          new OpGetHOTangentsOnEdge(mesh_node_positions));
    }
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                    it)) {
      CHKERR edge_forces.at(fe_name).addForce(field_name, F,
                                              it->getMeshsetId());
    }
    MoFEMFunctionReturn(0);
  }
};

#endif //__EDGE_FORCE_HPP__
