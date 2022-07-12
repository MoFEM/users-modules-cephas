/** \file NodalForce.hpp
  \ingroup mofem_static_boundary_conditions
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

#ifndef __NODAL_FORCES_HPP__
#define __NODAL_FORCES_HPP__

/** \brief Force applied to nodes
 * \ingroup mofem_static_boundary_conditions
 */
struct NodalForce {

  MoFEM::Interface &mField;
  NodalForce(MoFEM::Interface &m_field) : mField(m_field), fe(m_field) {}

  struct MyFE : public MoFEM::VertexElementForcesAndSourcesCore {
    MyFE(MoFEM::Interface &m_field);
  };

  MyFE fe;
  MyFE &getLoopFe() { return fe; }

  struct bCForce {
    ForceCubitBcData data;
    Range nOdes;
  };
  std::map<int, bCForce> mapForce;

  boost::ptr_vector<MethodForForceScaling> methodsOp;

  /// \brief Operator to assemble nodal force into right hand side vector
  struct OpNodalForce
      : public MoFEM::VertexElementForcesAndSourcesCore::UserDataOperator {

    Vec F;
    bool useSnesF;
    bCForce &dAta;
    boost::ptr_vector<MethodForForceScaling> &methodsOp;

    OpNodalForce(const std::string field_name, Vec _F, bCForce &data,
                 boost::ptr_vector<MethodForForceScaling> &methods_op,
                 bool use_snes_f = false);

    VectorDouble Nf;

    /** Executed for each entity on element, i.e. in this case Vertex element
      has only one entity, that is vertex
    */
    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data);
  };

  MoFEMErrorCode addForce(const std::string field_name, Vec F, int ms_id,
                          bool use_snes_f = false);
};

struct MetaNodalForces {

  /** \brief Scale force based on tag value "_LoadFactor_Scale_"

    This is obsolete, is kept to have back compatibility with fracture code.

  */
  struct TagForceScale : public MethodForForceScaling {
    MoFEM::Interface &mField;
    double *sCale;
    Tag thScale;

    TagForceScale(MoFEM::Interface &m_field);
    MoFEMErrorCode scaleNf(const FEMethod *fe, VectorDouble &Nf);
  };

  /**
   * \brief Scale force based on some DOF value
   *
   * That dof usually will be associated with dof used for arc-length control
   *
   */
  struct DofForceScale : public MethodForForceScaling {
    boost::shared_ptr<DofEntity> dOf;
    DofForceScale(boost::shared_ptr<DofEntity> dof) : dOf(dof) {}
    MoFEMErrorCode scaleNf(const FEMethod *fe, VectorDouble &Nf) {
      MoFEMFunctionBeginHot;
      Nf *= dOf->getFieldData();
      MoFEMFunctionReturnHot(0);
    }
  };

  /// Add element taking information from NODESET
  static MoFEMErrorCode addElement(MoFEM::Interface &m_field,
                                   const std::string field_name,
                                   Range *intersect_ptr = NULL) {
    MoFEMFunctionBegin;
    CHKERR m_field.add_finite_element("FORCE_FE", MF_ZERO);
    CHKERR m_field.modify_finite_element_add_field_row("FORCE_FE", field_name);
    CHKERR m_field.modify_finite_element_add_field_col("FORCE_FE", field_name);
    CHKERR m_field.modify_finite_element_add_field_data("FORCE_FE", field_name);
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                    it)) {
      Range tris;
      CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI, tris,
                                                     true);
      Range edges;
      CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBEDGE, edges,
                                                     true);
      Range tris_nodes;
      CHKERR m_field.get_moab().get_connectivity(tris, tris_nodes);
      Range edges_nodes;
      CHKERR m_field.get_moab().get_connectivity(edges, edges_nodes);
      Range nodes;
      CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBVERTEX,
                                                     nodes, true);
      nodes = subtract(nodes, tris_nodes);
      nodes = subtract(nodes, edges_nodes);
      if (intersect_ptr) {
        nodes = intersect(nodes, *intersect_ptr);
      }
      CHKERR m_field.add_ents_to_finite_element_by_type(nodes, MBVERTEX,
                                                        "FORCE_FE");
    }
    MoFEMFunctionReturn(0);
  }

  /// Set integration point operators
  static MoFEMErrorCode
  setOperators(MoFEM::Interface &m_field,
               boost::ptr_map<std::string, NodalForce> &nodal_forces, Vec F,
               const std::string field_name) {
    MoFEMFunctionBegin;
    string fe_name = "FORCE_FE";
    nodal_forces.insert(fe_name, new NodalForce(m_field));
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                    it)) {
      CHKERR nodal_forces.at(fe_name).addForce(field_name, F,
                                               it->getMeshsetId());
    }
    MoFEMFunctionReturn(0);
  }
};

#endif //__NODAL_FORCES_HPP__
