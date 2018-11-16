/* \file SurfacePressure.hpp
  \brief Implementation of pressure and forces on triangles surface

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

#ifndef __SURFACE_PERSSURE_HPP__
#define __SURFACE_PERSSURE_HPP__

/** \brief Finite element and operators to apply force/pressures applied to
 * surfaces \ingroup mofem_static_boundary_conditions
 */
struct NeummanForcesSurface {

  MoFEM::Interface &mField;

  /**
   * \brief Analytical force method
   */
  struct MethodForAnalyticalForce {

    virtual ~MethodForAnalyticalForce() {}

    /**
     * User implemented analytical force
     * @param  coords coordinates of integration point
     * @param  normal normal at integration point
     * @param  force  returned force
     * @return        error code
     */
    virtual MoFEMErrorCode getForce(const EntityHandle ent,
                                    const VectorDouble3 &coords,
                                    const VectorDouble3 &normal,
                                    VectorDouble3 &force) {
      MoFEMFunctionBeginHot;
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
              "You need to implement this");
      MoFEMFunctionReturnHot(0);
    }
  };

  struct LinearVaringPresssure : public MethodForAnalyticalForce {

    LinearVaringPresssure(const VectorDouble3 &p, const double c)
        : MethodForAnalyticalForce(), linearConstants(p), pressureShift(c) {}

    MoFEMErrorCode getForce(const EntityHandle ent, const VectorDouble3 &coords,
                            const VectorDouble3 &normal, VectorDouble3 &force);

  private:
    const VectorDouble3 linearConstants;
    const double pressureShift;
  };

  /**
   * Definition of face element used for integration
   */
  struct MyTriangleFE : public MoFEM::FaceElementForcesAndSourcesCore {
    int addToRule;
    MyTriangleFE(MoFEM::Interface &m_field);
    int getRule(int order) { return 2 * order + addToRule; };
  };

  MyTriangleFE fe;
  MyTriangleFE &getLoopFe() { return fe; }

  NeummanForcesSurface(MoFEM::Interface &m_field)
      : mField(m_field), fe(m_field) {}

  struct bCForce {
    ForceCubitBcData data;
    Range tRis;
  };
  std::map<int, bCForce> mapForce;

  struct bCPressure {
    PressureCubitBcData data;
    Range tRis;
  };
  std::map<int, bCPressure> mapPressure;

  boost::ptr_vector<MethodForForceScaling> methodsOp;
  boost::ptr_vector<MethodForAnalyticalForce> analyticalForceOp;

  /// Operator for force element
  struct OpNeumannForce
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    Vec F;
    bCForce &dAta;
    boost::ptr_vector<MethodForForceScaling> &methodsOp;

    bool hoGeometry;

    OpNeumannForce(const std::string field_name, Vec _F, bCForce &data,
                   boost::ptr_vector<MethodForForceScaling> &methods_op,
                   bool ho_geometry = false);

    VectorDouble Nf; //< Local force vector

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /// Operator for force element
  struct OpNeumannForceAnalytical
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    Vec F;
    const Range tRis;
    boost::ptr_vector<MethodForForceScaling> &methodsOp;
    boost::ptr_vector<MethodForAnalyticalForce> &analyticalForceOp;

    bool hoGeometry;

    OpNeumannForceAnalytical(
        const std::string field_name, Vec f, const Range tris,
        boost::ptr_vector<MethodForForceScaling> &methods_op,
        boost::ptr_vector<MethodForAnalyticalForce> &analytical_force_op,
        bool ho_geometry = false);

    VectorDouble Nf; //< Local force vector

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /**
   * @brief Operator for pressure element
   *
   */
  struct OpNeumannPressure
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    Vec F;
    bCPressure &dAta;
    boost::ptr_vector<MethodForForceScaling> &methodsOp;
    bool hoGeometry;

    OpNeumannPressure(const std::string field_name, Vec _F, bCPressure &data,
                      boost::ptr_vector<MethodForForceScaling> &methods_op,
                      bool ho_geometry = false);

    VectorDouble Nf;

    /**
     * @brief Integrate pressure
     *
     * \f[
     * \begin{split}
     * \mathbf{t} &= p \mathbf{n} \\
     * \mathbf{f}^i &= \int_\mathcal{T} {\pmb\phi}^i \mathbf{t}
     * \textrm{d}\mathcal{T} \end{split}
     * \f]
     * where \f$p\f$ is pressure, \f$\mathbf{n}\f$ is normal, \f$\mathbf{t}\f$
     * is traction, and
     * \f$\mathbf{f}^i\f$ is local vector of external forces for ith base
     * function \f${\pmb\phi}^i\f$.
     *
     *
     * @param side
     * @param type
     * @param data
     * @return MoFEMErrorCode
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /// Operator for flux element
  struct OpNeumannFlux
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    Vec F;
    bCPressure &dAta;
    boost::ptr_vector<MethodForForceScaling> &methodsOp;
    bool hoGeometry;

    OpNeumannFlux(const std::string field_name, Vec _F, bCPressure &data,
                  boost::ptr_vector<MethodForForceScaling> &methods_op,
                  bool ho_geometry);

    VectorDouble Nf;

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /**
   * \brief Add operator to calculate forces on element
   * @param  field_name  Field name (f.e. TEMPERATURE)
   * @param  F           Right hand side vector
   * @param  ms_id       Set id (SideSet or BlockSet if block_set = true)
   * @param  ho_geometry Use higher order shape functions to define curved
   * geometry
   * @param  block_set   If tru get data from block set
   * @return             ErrorCode
   */
  MoFEMErrorCode addForce(const std::string field_name, Vec F, int ms_id,
                          bool ho_geometry = false, bool block_set = false);

  /**
   * \brief Add operator to calculate pressure on element
   * @param  field_name  Field name (f.e. TEMPERATURE)
   * @param  F           Right hand side vector
   * @param  ms_id       Set id (SideSet or BlockSet if block_set = true)
   * @param  ho_geometry Use higher order shape functions to define curved
   * geometry
   * @param  block_set   If tru get data from block set
   * @return             ErrorCode
   */
  MoFEMErrorCode addPressure(const std::string field_name, Vec F, int ms_id,
                             bool ho_geometry = false, bool block_set = false);

  /// Add flux element operator (integration on face)
  MoFEMErrorCode addFlux(const std::string field_name, Vec F, int ms_id,
                         bool ho_geometry = false);

  /// \deprecated fixed spelling mistake
  DEPRECATED typedef MethodForAnalyticalForce MethodForAnaliticalForce;

  DEPRECATED typedef OpNeumannPressure OpNeumannPreassure;

  DEPRECATED typedef bCPressure
      bCPreassure; ///< \deprecated Do not use spelling mistake

  /// \deprecated function is deprecated because spelling mistake, use
  /// addPressure instead
  DEPRECATED MoFEMErrorCode addPreassure(const std::string field_name, Vec F,
                                         int ms_id, bool ho_geometry = false,
                                         bool block_set = false);
};

/** \brief Set of high-level function declaring elements and setting operators
 * to apply forces/fluxes \ingroup mofem_static_boundary_conditions
 */
struct MetaNeummanForces {

  /**
   * \brief Declare finite element
   *
   * Search cubit sidesets and blocksets with pressure bc and declare surface
   elemen

   * Block set has to have name “PRESSURE”. Can have name “PRESSURE_01” or any
   * other name with prefix. The first attribute  of block set is pressure
   * value.

   *
   * @param  m_field               Interface insurance
   * @param  field_name            Field name (f.e. DISPLACEMENT)
   * @param  mesh_nodals_positions Name of field on which ho-geometry is defined
   * @param  intersect_ptr         Pointer to range to interect meshset entities
   * @return                       Error code
   */
  static MoFEMErrorCode addNeumannBCElements(
      MoFEM::Interface &m_field, const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS",
      Range *intersect_ptr = NULL) {
    MoFEMFunctionBegin;

    // Define boundary element that operates on rows, columns and data of a
    // given field
    CHKERR m_field.add_finite_element("FORCE_FE", MF_ZERO);
    CHKERR m_field.modify_finite_element_add_field_row("FORCE_FE", field_name);
    CHKERR m_field.modify_finite_element_add_field_col("FORCE_FE", field_name);
    CHKERR m_field.modify_finite_element_add_field_data("FORCE_FE", field_name);
    if (m_field.check_field(mesh_nodals_positions)) {
      CHKERR m_field.modify_finite_element_add_field_data(
          "FORCE_FE", mesh_nodals_positions);
    }
    // Add entities to that element, here we add all triangles with FORCESET
    // from cubit
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                    it)) {
      Range tris;
      CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI, tris,
                                                     true);
      if (intersect_ptr) {
        tris = intersect(tris, *intersect_ptr);
      }
      CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI,
                                                        "FORCE_FE");
    }

    CHKERR m_field.add_finite_element("PRESSURE_FE", MF_ZERO);
    CHKERR m_field.modify_finite_element_add_field_row("PRESSURE_FE",
                                                       field_name);
    CHKERR m_field.modify_finite_element_add_field_col("PRESSURE_FE",
                                                       field_name);
    CHKERR m_field.modify_finite_element_add_field_data("PRESSURE_FE",
                                                        field_name);
    if (m_field.check_field(mesh_nodals_positions)) {
      ierr = m_field.modify_finite_element_add_field_data(
          "PRESSURE_FE", mesh_nodals_positions);
    }

    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             m_field, SIDESET | PRESSURESET, it)) {
      Range tris;
      CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI, tris,
                                                     true);
      if (intersect_ptr) {
        tris = intersect(tris, *intersect_ptr);
      }
      CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI,
                                                        "PRESSURE_FE");
    }

    // Reading forces from BLOCKSET

    const string block_set_force_name("FORCE");
    // search for block named FORCE and add its attributes to FORCE_FE element
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
      if (it->getName().compare(0, block_set_force_name.length(),
                                block_set_force_name) == 0) {
        std::vector<double> mydata;
        CHKERR it->getAttributes(mydata);
        VectorDouble force(mydata.size());
        for (unsigned int ii = 0; ii < mydata.size(); ii++) {
          force[ii] = mydata[ii];
        }
        if (force.empty()) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Force not given");
        }
        Range tris;
        CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI, tris,
                                                       true);
        if (intersect_ptr) {
          tris = intersect(tris, *intersect_ptr);
        }
        CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI,
                                                          "FORCE_FE");
      }
    }
    // search for block named PRESSURE and add its attributes to PRESSURE_FE
    // element
    const string block_set_pressure_name("PRESSURE");
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
      if (it->getName().compare(0, block_set_pressure_name.length(),
                                block_set_pressure_name) == 0) {
        std::vector<double> mydata;
        CHKERR it->getAttributes(mydata);
        VectorDouble pressure(mydata.size());
        for (unsigned int ii = 0; ii < mydata.size(); ii++) {
          pressure[ii] = mydata[ii];
        }
        if (pressure.empty()) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                  "Pressure not given");
        }
        Range tris;
        CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI, tris,
                                                       true);
        if (intersect_ptr) {
          tris = intersect(tris, *intersect_ptr);
        }
        // cerr << tris << endl;
        CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI,
                                                          "PRESSURE_FE");
      }
    }

    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Set operators to finite elements calculating right hand side vector

   * @param  m_field               Interface
   * @param  neumann_forces        Map of pointers to force/pressure elements
   * @param  F                     Right hand side vector
   * @param  field_name            Field name (f.e. DISPLACEMENT)
   * @param  mesh_nodals_positions Name of field on which ho-geometry is defined
   * @return                       Error code
   *
   */
  static MoFEMErrorCode setMomentumFluxOperators(
      MoFEM::Interface &m_field,
      boost::ptr_map<std::string, NeummanForcesSurface> &neumann_forces, Vec F,
      const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS") {
    MoFEMFunctionBegin;

    string fe_name;
    fe_name = "FORCE_FE";
    neumann_forces.insert(fe_name, new NeummanForcesSurface(m_field));
    bool ho_geometry = m_field.check_field(mesh_nodals_positions);
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                    it)) {
      CHKERR neumann_forces.at(fe_name).addForce(
          field_name, F, it->getMeshsetId(), ho_geometry, false);
    }
    // Reading forces from BLOCKSET
    const string block_set_force_name("FORCE");
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
      if (it->getName().compare(0, block_set_force_name.length(),
                                block_set_force_name) == 0) {
        CHKERR neumann_forces.at(fe_name).addForce(
            field_name, F, it->getMeshsetId(), ho_geometry, true);
      }
    }

    fe_name = "PRESSURE_FE";
    neumann_forces.insert(fe_name, new NeummanForcesSurface(m_field));
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             m_field, SIDESET | PRESSURESET, it)) {
      CHKERR neumann_forces.at(fe_name).addPressure(
          field_name, F, it->getMeshsetId(), ho_geometry, false);
    }
    // Reading pressures from BLOCKSET
    const string block_set_pressure_name("PRESSURE");
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
      if (it->getName().compare(0, block_set_pressure_name.length(),
                                block_set_pressure_name) == 0) {
        CHKERR neumann_forces.at(fe_name).addPressure(
            field_name, F, it->getMeshsetId(), ho_geometry, true);
      }
    }

    MoFEMFunctionReturn(0);
  }

  static MoFEMErrorCode addNeumannFluxBCElements(
      MoFEM::Interface &m_field, const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS") {
    MoFEMFunctionBegin;

    CHKERR m_field.add_finite_element("FLUX_FE", MF_ZERO);
    CHKERR m_field.modify_finite_element_add_field_row("FLUX_FE", field_name);
    CHKERR m_field.modify_finite_element_add_field_col("FLUX_FE", field_name);
    CHKERR m_field.modify_finite_element_add_field_data("FLUX_FE", field_name);
    if (m_field.check_field(mesh_nodals_positions)) {
      ierr = m_field.modify_finite_element_add_field_data(
          "FLUX_FE", mesh_nodals_positions);
    }

    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             m_field, SIDESET | PRESSURESET, it)) {
      Range tris;
      CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI, tris,
                                                     true);
      CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI, "FLUX_FE");
    }

    MoFEMFunctionReturn(0);
  }

  static MoFEMErrorCode setMassFluxOperators(
      MoFEM::Interface &m_field,
      boost::ptr_map<std::string, NeummanForcesSurface> &neumann_forces, Vec F,
      const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS") {
    MoFEMFunctionBegin;

    string fe_name;
    fe_name = "FLUX_FE";
    neumann_forces.insert(fe_name, new NeummanForcesSurface(m_field));
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             m_field, SIDESET | PRESSURESET, it)) {
      bool ho_geometry = m_field.check_field(mesh_nodals_positions);
      CHKERR neumann_forces.at(fe_name).addFlux(
          field_name, F, it->getMeshsetId(), ho_geometry);
    }
    MoFEMFunctionReturn(0);
  }
};

#endif //__SURFACE_PERSSURE_HPP__

/***************************************************************************/ /**
                                                                               * \defgroup mofem_static_boundary_conditions Pressure and force boundary conditions
                                                                               * \ingroup user_modules
                                                                               ******************************************************************************/
