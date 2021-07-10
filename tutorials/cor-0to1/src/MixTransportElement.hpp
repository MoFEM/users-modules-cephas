/** \file MixTransportElement.hpp
 * \brief Mix implementation of transport element
 *
 * \ingroup mofem_mix_transport_elem
 *
 */

/*
 * This file is part of MoFEM.
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

#ifndef _MIX_TRANPORT_ELEMENT_HPP_
#define _MIX_TRANPORT_ELEMENT_HPP_

namespace MixTransport {

/** \brief Mix transport problem
  \ingroup mofem_mix_transport_elem

  Note to solve this system you need to use direct solver or proper
  preconditioner for saddle problem.

  It is based on \cite arnold2006differential \cite arnold2012mixed \cite
  reviartthomas1996
  <https://www.researchgate.net/profile/Richard_Falk/publication/226454406_Differential_Complexes_and_Stability_of_Finite_Element_Methods_I._The_de_Rham_Complex/links/02e7e5214f0426ff77000000.pdf>

  General problem have form,
  \f[
  \mathbf{A} \boldsymbol\sigma + \textrm{grad}[u] = \mathbf{0} \; \textrm{on} \;
  \Omega \\ \textrm{div}[\boldsymbol\sigma] = f \; \textrm{on} \; \Omega \f]

*/
struct MixTransportElement {

  MoFEM::Interface &mField;

  /**
   * \brief definition of volume element

   * It is used to calculate volume integrals. On volume element we set-up
   * operators to calculate components of matrix and vector.

   */
  struct MyVolumeFE : public MoFEM::VolumeElementForcesAndSourcesCore {
    MyVolumeFE(MoFEM::Interface &m_field)
        : MoFEM::VolumeElementForcesAndSourcesCore(m_field) {}
    int getRule(int order) { return 2 * order + 1; };
  };

  MyVolumeFE feVol; ///< Instance of volume element

  /** \brief definition of surface element

    * It is used to calculate surface integrals. On volume element are operators
    * evaluating natural boundary conditions.

    */
  struct MyTriFE : public MoFEM::FaceElementForcesAndSourcesCore {
    MyTriFE(MoFEM::Interface &m_field)
        : MoFEM::FaceElementForcesAndSourcesCore(m_field) {}
    int getRule(int order) { return 2 * order + 1; };
  };

  MyTriFE feTri; ///< Instance of surface element

  /**
   * \brief construction of this data structure
   */
  MixTransportElement(MoFEM::Interface &m_field)
      : mField(m_field), feVol(m_field), feTri(m_field){};

  /**
   * \brief destructor
   */
  virtual ~MixTransportElement() {}

  VectorDouble valuesAtGaussPts; ///< values at integration points on element
  MatrixDouble
      valuesGradientAtGaussPts; ///< gradients at integration points on element
  VectorDouble
      divergenceAtGaussPts; ///< divergence at integration points on element
  MatrixDouble fluxesAtGaussPts; ///< fluxes at integration points on element

  set<int> bcIndices;

  /**
   * \brief get dof indices where essential boundary conditions are applied
   * @param  is indices
   * @return    error code
   */
  MoFEMErrorCode getDirichletBCIndices(IS *is) {
    MoFEMFunctionBegin;
    std::vector<int> ids;
    ids.insert(ids.begin(), bcIndices.begin(), bcIndices.end());
    IS is_local;
    CHKERR ISCreateGeneral(mField.get_comm(), ids.size(),
                           ids.empty() ? PETSC_NULL : &ids[0],
                           PETSC_COPY_VALUES, &is_local);
    CHKERR ISAllGather(is_local, is);
    CHKERR ISDestroy(&is_local);
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief set source term
   * @param  ent  handle to entity on which function is evaluated
   * @param  x    coord
   * @param  y    coord
   * @param  z    coord
   * @param  flux reference to source term set by function
   * @return      error code
   */
  virtual MoFEMErrorCode getSource(const EntityHandle ent, const double x,
                                   const double y, const double z,
                                   double &flux) {
    MoFEMFunctionBeginHot;
    flux = 0;
    MoFEMFunctionReturnHot(0);
  }

  /**
   * \brief natural (Dirichlet) boundary conditions (set values)
   * @param  ent   handle to finite element entity
   * @param  x     coord
   * @param  y     coord
   * @param  z     coord
   * @param  value reference to value set by function
   * @return       error code
   */
  virtual MoFEMErrorCode getResistivity(const EntityHandle ent, const double x,
                                        const double y, const double z,
                                        MatrixDouble3by3 &inv_k) {
    MoFEMFunctionBeginHot;
    inv_k.clear();
    for (int dd = 0; dd < 3; dd++) {
      inv_k(dd, dd) = 1;
    }
    MoFEMFunctionReturnHot(0);
  }

  /**
   * \brief evaluate natural (Dirichlet) boundary conditions
   * @param  ent   entity on which bc is evaluated
   * @param  x     coordinate
   * @param  y     coordinate
   * @param  z     coordinate
   * @param  value vale
   * @return       error code
   */
  virtual MoFEMErrorCode getBcOnValues(const EntityHandle ent, const int gg,
                                       const double x, const double y,
                                       const double z, double &value) {
    MoFEMFunctionBeginHot;
    value = 0;
    MoFEMFunctionReturnHot(0);
  }

  /**
   * \brief essential (Neumann) boundary condition (set fluxes)
   * @param  ent  handle to finite element entity
   * @param  x    coord
   * @param  y    coord
   * @param  z    coord
   * @param  flux reference to flux which is set by function
   * @return      [description]
   */
  virtual MoFEMErrorCode getBcOnFluxes(const EntityHandle ent, const double x,
                                       const double y, const double z,
                                       double &flux) {
    MoFEMFunctionBeginHot;
    flux = 0;
    MoFEMFunctionReturnHot(0);
  }

  /** \brief data for evaluation of het conductivity and heat capacity elements
   * \ingroup mofem_thermal_elem
   */
  struct BlockData {
    double cOnductivity;
    double cApacity;
    Range tEts; ///< constatins elements in block set
  };
  std::map<int, BlockData>
      setOfBlocks; ///< maps block set id with appropriate BlockData

  /**
   * \brief Add fields to database
   * @param  values name of the fields
   * @param  fluxes name of filed for fluxes
   * @param  order  order of approximation
   * @return        error code
   */
  MoFEMErrorCode addFields(const std::string &values, const std::string &fluxes,
                           const int order) {
    MoFEMFunctionBegin;
    // Fields
    CHKERR mField.add_field(fluxes, HDIV, DEMKOWICZ_JACOBI_BASE, 1);
    CHKERR mField.add_field(values, L2, AINSWORTH_LEGENDRE_BASE, 1);

    // meshset consisting all entities in mesh
    EntityHandle root_set = mField.get_moab().get_root_set();
    // add entities to field
    CHKERR mField.add_ents_to_field_by_type(root_set, MBTET, fluxes);
    CHKERR mField.add_ents_to_field_by_type(root_set, MBTET, values);
    CHKERR mField.set_field_order(root_set, MBTET, fluxes, order + 1);
    CHKERR mField.set_field_order(root_set, MBTRI, fluxes, order + 1);
    CHKERR mField.set_field_order(root_set, MBTET, values, order);
    MoFEMFunctionReturn(0);
  }

  /// \brief add finite elements
  MoFEMErrorCode addFiniteElements(
      const std::string &fluxes_name, const std::string &values_name) {
    MoFEMFunctionBegin;

    // Set up volume element operators. Operators are used to calculate
    // components of stiffness matrix & right hand side, in essence are used to
    // do volume integrals over tetrahedral in this case.

    // Define element "MIX". Note that this element will work with fluxes_name
    // and values_name. This reflect bilinear form for the problem
    CHKERR mField.add_finite_element("MIX", MF_ZERO);
    CHKERR mField.modify_finite_element_add_field_row("MIX", fluxes_name);
    CHKERR mField.modify_finite_element_add_field_col("MIX", fluxes_name);
    CHKERR mField.modify_finite_element_add_field_row("MIX", values_name);
    CHKERR mField.modify_finite_element_add_field_col("MIX", values_name);
    CHKERR mField.modify_finite_element_add_field_data("MIX", fluxes_name);
    CHKERR mField.modify_finite_element_add_field_data("MIX", values_name);

    // Define finite element to integrate over skeleton, we need that to
    // evaluate error
    CHKERR mField.add_finite_element("MIX_SKELETON", MF_ZERO);
    CHKERR mField.modify_finite_element_add_field_row("MIX_SKELETON",
                                                      fluxes_name);
    CHKERR mField.modify_finite_element_add_field_col("MIX_SKELETON",
                                                      fluxes_name);
    CHKERR mField.modify_finite_element_add_field_data("MIX_SKELETON",
                                                       fluxes_name);

    // Look for all BLOCKSET which are MAT_THERMALSET, takes entities from those
    // BLOCKSETS and add them to "MIX" finite element. In addition get data form
    // that meshset and set conductivity which is used to calculate fluxes from
    // gradients of concentration or gradient of temperature, depending how you
    // interpret variables.
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             mField, BLOCKSET | MAT_THERMALSET, it)) {

      Mat_Thermal temp_data;
      CHKERR it->getAttributeDataStructure(temp_data);
      setOfBlocks[it->getMeshsetId()].cOnductivity =
          temp_data.data.Conductivity;
      setOfBlocks[it->getMeshsetId()].cApacity = temp_data.data.HeatCapacity;
      CHKERR mField.get_moab().get_entities_by_type(
          it->meshset, MBTET, setOfBlocks[it->getMeshsetId()].tEts, true);
      CHKERR mField.add_ents_to_finite_element_by_type(
          setOfBlocks[it->getMeshsetId()].tEts, MBTET, "MIX");

      Range skeleton;
      CHKERR mField.get_moab().get_adjacencies(
          setOfBlocks[it->getMeshsetId()].tEts, 2, false, skeleton,
          moab::Interface::UNION);
      CHKERR mField.add_ents_to_finite_element_by_type(skeleton, MBTRI,
                                                       "MIX_SKELETON");
    }

    // Define element to integrate natural boundary conditions, i.e. set values.
    CHKERR mField.add_finite_element("MIX_BCVALUE", MF_ZERO);
    CHKERR mField.modify_finite_element_add_field_row("MIX_BCVALUE",
                                                      fluxes_name);
    CHKERR mField.modify_finite_element_add_field_col("MIX_BCVALUE",
                                                      fluxes_name);
    CHKERR mField.modify_finite_element_add_field_data("MIX_BCVALUE",
                                                       fluxes_name);
    CHKERR mField.modify_finite_element_add_field_data("MIX_BCVALUE",
                                                       values_name);

    // Define element to apply essential boundary conditions.
    CHKERR mField.add_finite_element("MIX_BCFLUX", MF_ZERO);
    CHKERR mField.modify_finite_element_add_field_row("MIX_BCFLUX",
                                                      fluxes_name);
    CHKERR mField.modify_finite_element_add_field_col("MIX_BCFLUX",
                                                      fluxes_name);
    CHKERR mField.modify_finite_element_add_field_data("MIX_BCFLUX",
                                                       fluxes_name);
    CHKERR mField.modify_finite_element_add_field_data("MIX_BCFLUX",
                                                       values_name);
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Build problem
   * @param  ref_level mesh refinement on which mesh problem you like to built.
   * @return           error code
   */
  MoFEMErrorCode buildProblem(BitRefLevel &ref_level) {
    MoFEMFunctionBegin;
    // build field
    CHKERR mField.build_fields();
    // get tetrahedrons which has been build previously and now in so called
    // garbage bit level
    Range done_tets;
    CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        BitRefLevel().set(0), BitRefLevel().set(), MBTET, done_tets);
    CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        BitRefLevel().set(BITREFLEVEL_SIZE - 1), BitRefLevel().set(), MBTET,
        done_tets);
    // get tetrahedrons which belong to problem bit level
    Range ref_tets;
    CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        ref_level, BitRefLevel().set(), MBTET, ref_tets);
    ref_tets = subtract(ref_tets, done_tets);
    CHKERR mField.build_finite_elements("MIX", &ref_tets, 2);
    // get triangles which has been build previously and now in so called
    // garbage bit level
    Range done_faces;
    CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        BitRefLevel().set(0), BitRefLevel().set(), MBTRI, done_faces);
    CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        BitRefLevel().set(BITREFLEVEL_SIZE - 1), BitRefLevel().set(), MBTRI,
        done_faces);
    // get triangles which belong to problem bit level
    Range ref_faces;
    CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        ref_level, BitRefLevel().set(), MBTRI, ref_faces);
    ref_faces = subtract(ref_faces, done_faces);
    // build finite elements structures
    CHKERR mField.build_finite_elements("MIX_BCFLUX", &ref_faces, 2);
    CHKERR mField.build_finite_elements("MIX_BCVALUE", &ref_faces, 2);
    CHKERR mField.build_finite_elements("MIX_SKELETON", &ref_faces, 2);
    // Build adjacencies of degrees of freedom and elements
    CHKERR mField.build_adjacencies(ref_level);
    // Define problem
    CHKERR mField.add_problem("MIX", MF_ZERO);
    // set refinement level for problem
    CHKERR mField.modify_problem_ref_level_set_bit("MIX", ref_level);
    // Add element to problem
    CHKERR mField.modify_problem_add_finite_element("MIX", "MIX");
    CHKERR mField.modify_problem_add_finite_element("MIX", "MIX_SKELETON");
    // Boundary conditions
    CHKERR mField.modify_problem_add_finite_element("MIX", "MIX_BCFLUX");
    CHKERR mField.modify_problem_add_finite_element("MIX", "MIX_BCVALUE");
    // build problem

    ProblemsManager *prb_mng_ptr;
    CHKERR mField.getInterface(prb_mng_ptr);
    CHKERR prb_mng_ptr->buildProblem("MIX", true);
    // mesh partitioning
    // partition
    CHKERR prb_mng_ptr->partitionProblem("MIX");
    CHKERR prb_mng_ptr->partitionFiniteElements("MIX");
    // what are ghost nodes, see Petsc Manual
    CHKERR prb_mng_ptr->partitionGhostDofs("MIX");
    MoFEMFunctionReturn(0);
  }

  struct OpPostProc
      : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {
    moab::Interface &postProcMesh;
    std::vector<EntityHandle> &mapGaussPts;
    OpPostProc(moab::Interface &post_proc_mesh,
               std::vector<EntityHandle> &map_gauss_pts)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              "VALUES", UserDataOperator::OPCOL),
          postProcMesh(post_proc_mesh), mapGaussPts(map_gauss_pts) {}
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      if (type != MBTET)
        MoFEMFunctionReturnHot(0);
      EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
      Tag th_error_flux;
      CHKERR getVolumeFE()->mField.get_moab().tag_get_handle("ERROR_FLUX",
                                                             th_error_flux);
      double *error_flux_ptr;
      CHKERR getVolumeFE()->mField.get_moab().tag_get_by_ptr(
          th_error_flux, &fe_ent, 1, (const void **)&error_flux_ptr);

      Tag th_error_div;
      CHKERR getVolumeFE()->mField.get_moab().tag_get_handle("ERROR_DIV",
                                                             th_error_div);
      double *error_div_ptr;
      CHKERR getVolumeFE()->mField.get_moab().tag_get_by_ptr(
          th_error_div, &fe_ent, 1, (const void **)&error_div_ptr);

      Tag th_error_jump;
      CHKERR getVolumeFE()->mField.get_moab().tag_get_handle("ERROR_JUMP",
                                                             th_error_jump);
      double *error_jump_ptr;
      CHKERR getVolumeFE()->mField.get_moab().tag_get_by_ptr(
          th_error_jump, &fe_ent, 1, (const void **)&error_jump_ptr);

      {
        double def_val = 0;
        Tag th_error_flux;
        CHKERR postProcMesh.tag_get_handle(
            "ERROR_FLUX", 1, MB_TYPE_DOUBLE, th_error_flux,
            MB_TAG_CREAT | MB_TAG_SPARSE, &def_val);
        for (vector<EntityHandle>::iterator vit = mapGaussPts.begin();
             vit != mapGaussPts.end(); vit++) {
          CHKERR postProcMesh.tag_set_data(th_error_flux, &*vit, 1,
                                           error_flux_ptr);
        }

        Tag th_error_div;
        CHKERR postProcMesh.tag_get_handle(
            "ERROR_DIV", 1, MB_TYPE_DOUBLE, th_error_div,
            MB_TAG_CREAT | MB_TAG_SPARSE, &def_val);
        for (vector<EntityHandle>::iterator vit = mapGaussPts.begin();
             vit != mapGaussPts.end(); vit++) {
          CHKERR postProcMesh.tag_set_data(th_error_div, &*vit, 1,
                                           error_div_ptr);
        }

        Tag th_error_jump;
        CHKERR postProcMesh.tag_get_handle(
            "ERROR_JUMP", 1, MB_TYPE_DOUBLE, th_error_jump,
            MB_TAG_CREAT | MB_TAG_SPARSE, &def_val);
        for (vector<EntityHandle>::iterator vit = mapGaussPts.begin();
             vit != mapGaussPts.end(); vit++) {
          CHKERR postProcMesh.tag_set_data(th_error_jump, &*vit, 1,
                                           error_jump_ptr);
        }
      }
      MoFEMFunctionReturn(0);
    }
  };

  /**
   * \brief Post process results
   * @return error code
   */
  MoFEMErrorCode postProc(const string out_file) {
    MoFEMFunctionBegin;
    PostProcVolumeOnRefinedMesh post_proc(mField);
    CHKERR post_proc.generateReferenceElementMesh();
    CHKERR post_proc.addFieldValuesPostProc("VALUES");
    CHKERR post_proc.addFieldValuesGradientPostProc("VALUES");
    CHKERR post_proc.addFieldValuesPostProc("FLUXES");
    // CHKERR post_proc.addHdivFunctionsPostProc("FLUXES");
    post_proc.getOpPtrVector().push_back(
        new OpPostProc(post_proc.postProcMesh, post_proc.mapGaussPts));
    CHKERR mField.loop_finite_elements("MIX", "MIX", post_proc);
    CHKERR post_proc.writeFile(out_file.c_str());
    MoFEMFunctionReturn(0);
  }

  Vec D, D0, F;
  Mat Aij;

  /// \brief create matrices
  MoFEMErrorCode createMatrices() {
    MoFEMFunctionBegin;
    CHKERR mField.getInterface<MatrixManager>()
        ->createMPIAIJWithArrays<PetscGlobalIdx_mi_tag>("MIX", &Aij);
    CHKERR mField.getInterface<VecManager>()->vecCreateGhost("MIX", COL, &D);
    CHKERR mField.getInterface<VecManager>()->vecCreateGhost("MIX", COL, &D0);
    CHKERR mField.getInterface<VecManager>()->vecCreateGhost("MIX", ROW, &F);
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief solve problem
   * @return error code
   */
  MoFEMErrorCode solveLinearProblem() {
    MoFEMFunctionBegin;
    CHKERR MatZeroEntries(Aij);
    CHKERR VecZeroEntries(F);
    CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecZeroEntries(D0);
    CHKERR VecGhostUpdateBegin(D0, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D0, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecZeroEntries(D);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR mField.getInterface<VecManager>()->setGlobalGhostVector(
        "MIX", COL, D, INSERT_VALUES, SCATTER_REVERSE);

    // Calculate essential boundary conditions

    // clear essential bc indices, it could have dofs from other mesh refinement
    bcIndices.clear();
    // clear operator, just in case if some other operators are left on this
    // element
    feTri.getOpPtrVector().clear();
    // set operator to calculate essential boundary conditions
    feTri.getOpPtrVector().push_back(new OpEvaluateBcOnFluxes(*this, "FLUXES"));
    CHKERR mField.loop_finite_elements("MIX", "MIX_BCFLUX", feTri);
    CHKERR VecGhostUpdateBegin(D0, INSERT_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(D0, INSERT_VALUES, SCATTER_REVERSE);
    CHKERR VecAssemblyBegin(D0);
    CHKERR VecAssemblyEnd(D0);

    // set operators to calculate matrix and right hand side vectors
    feVol.getOpPtrVector().clear();
    feVol.getOpPtrVector().push_back(new OpL2Source(*this, "VALUES", F));
    feVol.getOpPtrVector().push_back(
        new OpFluxDivergenceAtGaussPts(*this, "FLUXES"));
    feVol.getOpPtrVector().push_back(new OpValuesAtGaussPts(*this, "VALUES"));
    feVol.getOpPtrVector().push_back(
        new OpDivTauU_HdivL2(*this, "FLUXES", "VALUES", F));
    feVol.getOpPtrVector().push_back(
        new OpTauDotSigma_HdivHdiv(*this, "FLUXES", Aij, F));
    feVol.getOpPtrVector().push_back(
        new OpVDivSigma_L2Hdiv(*this, "VALUES", "FLUXES", Aij, F));
    CHKERR mField.loop_finite_elements("MIX", "MIX", feVol);
    ;

    // calculate right hand side for natural boundary conditions
    feTri.getOpPtrVector().clear();
    feTri.getOpPtrVector().push_back(new OpRhsBcOnValues(*this, "FLUXES", F));
    CHKERR mField.loop_finite_elements("MIX", "MIX_BCVALUE", feTri);
    ;

    // assemble matrices
    CHKERR MatAssemblyBegin(Aij, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(Aij, MAT_FINAL_ASSEMBLY);
    CHKERR VecGhostUpdateBegin(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecAssemblyBegin(F);
    CHKERR VecAssemblyEnd(F);

    {
      double nrm2_F;
      CHKERR VecNorm(F, NORM_2, &nrm2_F);
      PetscPrintf(PETSC_COMM_WORLD, "nrm2_F = %6.4e\n", nrm2_F);
    }

    // CHKERR MatMultAdd(Aij,D0,F,F); ;

    // for ksp solver vector is moved into rhs side
    // for snes it is left ond the left
    CHKERR VecScale(F, -1);

    IS essential_bc_ids;
    CHKERR getDirichletBCIndices(&essential_bc_ids);
    CHKERR MatZeroRowsColumnsIS(Aij, essential_bc_ids, 1, D0, F);
    CHKERR ISDestroy(&essential_bc_ids);

    // {
    //   double norm;
    //   CHKERR MatNorm(Aij,NORM_FROBENIUS,&norm); ;
    //   PetscPrintf(PETSC_COMM_WORLD,"mat norm = %6.4e\n",norm);
    // }

    {
      double nrm2_F;
      CHKERR VecNorm(F, NORM_2, &nrm2_F);
      PetscPrintf(PETSC_COMM_WORLD, "With BC nrm2_F = %6.4e\n", nrm2_F);
    }

    // MatView(Aij,PETSC_VIEWER_DRAW_WORLD);
    // MatView(Aij,PETSC_VIEWER_STDOUT_WORLD);
    // std::string wait;
    // std::cin >> wait;

    // MatView(Aij,PETSC_VIEWER_DRAW_WORLD);
    // std::cin >> wait;

    // Solve
    KSP solver;
    CHKERR KSPCreate(PETSC_COMM_WORLD, &solver);
    CHKERR KSPSetOperators(solver, Aij, Aij);
    CHKERR KSPSetFromOptions(solver);
    CHKERR KSPSetUp(solver);
    CHKERR KSPSolve(solver, F, D);
    CHKERR KSPDestroy(&solver);

    {
      double nrm2_D;
      CHKERR VecNorm(D, NORM_2, &nrm2_D);
      ;
      PetscPrintf(PETSC_COMM_WORLD, "nrm2_D = %6.4e\n", nrm2_D);
    }
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

    // copy data form vector on mesh
    CHKERR mField.getInterface<VecManager>()->setGlobalGhostVector(
        "MIX", COL, D, INSERT_VALUES, SCATTER_REVERSE);

    MoFEMFunctionReturn(0);
  }

  /// \brief calculate residual
  MoFEMErrorCode calculateResidual() {
    MoFEMFunctionBegin;
    CHKERR VecZeroEntries(F);
    CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecAssemblyBegin(F);
    CHKERR VecAssemblyEnd(F);
    // calculate residuals
    feVol.getOpPtrVector().clear();
    feVol.getOpPtrVector().push_back(new OpL2Source(*this, "VALUES", F));
    feVol.getOpPtrVector().push_back(
        new OpFluxDivergenceAtGaussPts(*this, "FLUXES"));
    feVol.getOpPtrVector().push_back(new OpValuesAtGaussPts(*this, "VALUES"));
    feVol.getOpPtrVector().push_back(
        new OpDivTauU_HdivL2(*this, "FLUXES", "VALUES", F));
    feVol.getOpPtrVector().push_back(
        new OpTauDotSigma_HdivHdiv(*this, "FLUXES", PETSC_NULL, F));
    feVol.getOpPtrVector().push_back(
        new OpVDivSigma_L2Hdiv(*this, "VALUES", "FLUXES", PETSC_NULL, F));
    CHKERR mField.loop_finite_elements("MIX", "MIX", feVol);
    feTri.getOpPtrVector().clear();
    feTri.getOpPtrVector().push_back(new OpRhsBcOnValues(*this, "FLUXES", F));
    CHKERR mField.loop_finite_elements("MIX", "MIX_BCVALUE", feTri);
    CHKERR VecGhostUpdateBegin(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecAssemblyBegin(F);
    CHKERR VecAssemblyEnd(F);
    // CHKERR VecAXPY(F,-1.,D0);
    // CHKERR MatZeroRowsIS(Aij,essential_bc_ids,1,PETSC_NULL,F);
    {
      std::vector<int> ids;
      ids.insert(ids.begin(), bcIndices.begin(), bcIndices.end());
      std::vector<double> vals(ids.size(), 0);
      CHKERR VecSetValues(F, ids.size(), &*ids.begin(), &*vals.begin(),
                          INSERT_VALUES);
      CHKERR VecAssemblyBegin(F);
      CHKERR VecAssemblyEnd(F);
    }
    {
      double nrm2_F;
      CHKERR VecNorm(F, NORM_2, &nrm2_F);
      PetscPrintf(PETSC_COMM_WORLD, "nrm2_F = %6.4e\n", nrm2_F);
      const double eps = 1e-8;
      if (nrm2_F > eps) {
        // SETERRQ(PETSC_COMM_SELF,MOFEM_ATOM_TEST_INVALID,"problem with
        // residual");
      }
    }
    MoFEMFunctionReturn(0);
  }

  /** \brief Calculate error on elements

    \todo this functions runs serial, in future should be parallel and work on
    distributed meshes.

  */
  MoFEMErrorCode evaluateError() {
    MoFEMFunctionBegin;
    errorMap.clear();
    sumErrorFlux = 0;
    sumErrorDiv = 0;
    sumErrorJump = 0;
    feTri.getOpPtrVector().clear();
    feTri.getOpPtrVector().push_back(new OpSkeleton(*this, "FLUXES"));
    CHKERR mField.loop_finite_elements("MIX", "MIX_SKELETON", feTri, 0,
                                       mField.get_comm_size());
    feVol.getOpPtrVector().clear();
    feVol.getOpPtrVector().push_back(
        new OpFluxDivergenceAtGaussPts(*this, "FLUXES"));
    feVol.getOpPtrVector().push_back(
        new OpValuesGradientAtGaussPts(*this, "VALUES"));
    feVol.getOpPtrVector().push_back(new OpError(*this, "VALUES"));
    CHKERR mField.loop_finite_elements("MIX", "MIX", feVol, 0,
                                       mField.get_comm_size());
    const Problem *problem_ptr;
    CHKERR mField.get_problem("MIX", &problem_ptr);
    PetscPrintf(mField.get_comm(),
                "Nb dofs %d error flux^2 = %6.4e error div^2 = %6.4e error "
                "jump^2 = %6.4e error tot^2 = %6.4e\n",
                problem_ptr->getNbDofsRow(), sumErrorFlux, sumErrorDiv,
                sumErrorJump, sumErrorFlux + sumErrorDiv + sumErrorJump);
    MoFEMFunctionReturn(0);
  }

  /// \brief destroy matrices
  MoFEMErrorCode destroyMatrices() {
    MoFEMFunctionBegin;
    CHKERR MatDestroy(&Aij);
    ;
    CHKERR VecDestroy(&D);
    ;
    CHKERR VecDestroy(&D0);
    ;
    CHKERR VecDestroy(&F);
    ;
    MoFEMFunctionReturn(0);
  }

  /**
  \brief Assemble \f$\int_\mathcal{T} \mathbf{A} \boldsymbol\sigma \cdot
  \boldsymbol\tau \textrm{d}\mathcal{T}\f$

  \ingroup mofem_mix_transport_elem
  */
  struct OpTauDotSigma_HdivHdiv
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    MixTransportElement &cTx;
    Mat Aij;
    Vec F;

    OpTauDotSigma_HdivHdiv(MixTransportElement &ctx,
                           const std::string flux_name, Mat aij, Vec f)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              flux_name, flux_name,
              UserDataOperator::OPROWCOL | UserDataOperator::OPCOL),
          cTx(ctx), Aij(aij), F(f) {
      sYmm = true;
    }

    MatrixDouble NN, transNN;
    MatrixDouble3by3 invK;
    VectorDouble Nf;

    /**
     * \brief Assemble matrix
     * @param  row_side local index of row entity on element
     * @param  col_side local index of col entity on element
     * @param  row_type type of row entity, f.e. MBVERTEX, MBEDGE, or MBTET
     * @param  col_type type of col entity, f.e. MBVERTEX, MBEDGE, or MBTET
     * @param  row_data data for row
     * @param  col_data data for col
     * @return          error code
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data) {
      MoFEMFunctionBegin;
      if (Aij == PETSC_NULL)
        MoFEMFunctionReturnHot(0);
      if (row_data.getIndices().size() == 0)
        MoFEMFunctionReturnHot(0);
      if (col_data.getIndices().size() == 0)
        MoFEMFunctionReturnHot(0);
      EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
      int nb_row = row_data.getIndices().size();
      int nb_col = col_data.getIndices().size();
      NN.resize(nb_row, nb_col, false);
      NN.clear();
      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      invK.resize(3, 3, false);
      // get access to resistivity data by tensor rank 2
      FTensor::Tensor2<double *, 3, 3> t_inv_k(
          &invK(0, 0), &invK(0, 1), &invK(0, 2), &invK(1, 0), &invK(1, 1),
          &invK(1, 2), &invK(2, 0), &invK(2, 1), &invK(2, 2));
      // Get base functions
      auto t_n_hdiv_row = row_data.getFTensor1N<3>();
      FTensor::Tensor1<double, 3> t_row;
      int nb_gauss_pts = row_data.getN().size1();
      for (int gg = 0; gg != nb_gauss_pts; gg++) {
        // get integration weight and multiply by element volume
        double w = getGaussPts()(3, gg) * getVolume();
        const double x = getCoordsAtGaussPts()(gg, 0);
        const double y = getCoordsAtGaussPts()(gg, 1);
        const double z = getCoordsAtGaussPts()(gg, 2);
        // calculate receptivity (invers of conductivity)
        CHKERR cTx.getResistivity(fe_ent, x, y, z, invK);
        for (int kk = 0; kk != nb_row; kk++) {
          FTensor::Tensor1<const double *, 3> t_n_hdiv_col(
              &col_data.getVectorN<3>(gg)(0, HVEC0),
              &col_data.getVectorN<3>(gg)(0, HVEC1),
              &col_data.getVectorN<3>(gg)(0, HVEC2), 3);
          t_row(j) = w * t_n_hdiv_row(i) * t_inv_k(i, j);
          for (int ll = 0; ll != nb_col; ll++) {
            NN(kk, ll) += t_row(j) * t_n_hdiv_col(j);
            ++t_n_hdiv_col;
          }
          ++t_n_hdiv_row;
        }
      }
      Mat a = (Aij != PETSC_NULL) ? Aij : getFEMethod()->ts_B;
      CHKERR MatSetValues(a, nb_row, &row_data.getIndices()[0], nb_col,
                          &col_data.getIndices()[0], &NN(0, 0), ADD_VALUES);
      // matrix is symmetric, assemble other part
      if (row_side != col_side || row_type != col_type) {
        transNN.resize(nb_col, nb_row);
        noalias(transNN) = trans(NN);
        CHKERR MatSetValues(a, nb_col, &col_data.getIndices()[0], nb_row,
                            &row_data.getIndices()[0], &transNN(0, 0),
                            ADD_VALUES);
      }

      MoFEMFunctionReturn(0);
    }

    /**
     * \brief Assemble matrix
     * @param  side local index of row entity on element
     * @param  type type of row entity, f.e. MBVERTEX, MBEDGE, or MBTET
     * @param  data data for row
     * @return          error code
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      if (F == PETSC_NULL)
        MoFEMFunctionReturnHot(0);
      int nb_row = data.getIndices().size();
      if (nb_row == 0)
        MoFEMFunctionReturnHot(0);

      EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
      Nf.resize(nb_row);
      Nf.clear();
      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      invK.resize(3, 3, false);
      Nf.resize(nb_row);
      Nf.clear();
      // get access to resistivity data by tensor rank 2
      FTensor::Tensor2<double *, 3, 3> t_inv_k(
          &invK(0, 0), &invK(0, 1), &invK(0, 2), &invK(1, 0), &invK(1, 1),
          &invK(1, 2), &invK(2, 0), &invK(2, 1), &invK(2, 2));
      // get base functions
      auto t_n_hdiv = data.getFTensor1N<3>();
      auto t_flux = getFTensor1FromMat<3>(cTx.fluxesAtGaussPts);
      int nb_gauss_pts = data.getN().size1();
      for (int gg = 0; gg < nb_gauss_pts; gg++) {
        double w = getGaussPts()(3, gg) * getVolume();
        const double x = getCoordsAtGaussPts()(gg, 0);
        const double y = getCoordsAtGaussPts()(gg, 1);
        const double z = getCoordsAtGaussPts()(gg, 2);
        CHKERR cTx.getResistivity(fe_ent, x, y, z, invK);
        for (int ll = 0; ll != nb_row; ll++) {
          Nf[ll] += w * t_n_hdiv(i) * t_inv_k(i, j) * t_flux(j);
          ++t_n_hdiv;
        }
        ++t_flux;
      }
      CHKERR VecSetValues(F, nb_row, &data.getIndices()[0], &Nf[0], ADD_VALUES);

      MoFEMFunctionReturn(0);
    }
  };

  /** \brief Assemble \f$ \int_\mathcal{T} u \textrm{div}[\boldsymbol\tau]
   * \textrm{d}\mathcal{T} \f$
   */
  struct OpDivTauU_HdivL2
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    MixTransportElement &cTx;
    Vec F;

    OpDivTauU_HdivL2(MixTransportElement &ctx, const std::string flux_name_row,
                     string val_name_col, Vec f)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              flux_name_row, val_name_col, UserDataOperator::OPROW),
          cTx(ctx), F(f) {
      // this operator is not symmetric setting this variable makes element
      // operator to loop over element entities (sub-simplices) without
      // assumption that off-diagonal matrices are symmetric.
      sYmm = false;
    }

    VectorDouble divVec, Nf;

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      if (data.getFieldData().size() == 0)
        MoFEMFunctionReturnHot(0);
      int nb_row = data.getIndices().size();
      Nf.resize(nb_row);
      Nf.clear();
      divVec.resize(data.getN().size2() / 3, 0);
      if (divVec.size() != data.getIndices().size()) {
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                "data inconsistency");
      }
      int nb_gauss_pts = data.getN().size1();

      FTensor::Index<'i', 3> i;
      auto t_base_diff_hdiv = data.getFTensor2DiffN<3, 3>();

      int gg = 0;
      for (; gg < nb_gauss_pts; gg++) {
        double w = getGaussPts()(3, gg) * getVolume();
        for(auto &v : divVec) {
          v = t_base_diff_hdiv(i, i);
          ++t_base_diff_hdiv;
        }
        noalias(Nf) -= w * divVec * cTx.valuesAtGaussPts[gg];
      }
      CHKERR VecSetValues(F, nb_row, &data.getIndices()[0], &Nf[0], ADD_VALUES);
      ;

      MoFEMFunctionReturn(0);
    }
  };

  /** \brief \f$ \int_\mathcal{T} \textrm{div}[\boldsymbol\sigma] v
   * \textrm{d}\mathcal{T} \f$
   */
  struct OpVDivSigma_L2Hdiv
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    MixTransportElement &cTx;
    Mat Aij;
    Vec F;

    /**
     * \brief Constructor
     */
    OpVDivSigma_L2Hdiv(MixTransportElement &ctx, const std::string val_name_row,
                       string flux_name_col, Mat aij, Vec f)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              val_name_row, flux_name_col,
              UserDataOperator::OPROW | UserDataOperator::OPROWCOL),
          cTx(ctx), Aij(aij), F(f) {

      // this operator is not symmetric setting this variable makes element
      // operator to loop over element entities without
      // assumption that off-diagonal matrices are symmetric.
      sYmm = false;
    }

    MatrixDouble NN, transNN;
    VectorDouble divVec, Nf;

    /**
     * \brief Do calculations
     * @param  row_side local index of entity on row
     * @param  col_side local index of entity on column
     * @param  row_type type of row entity
     * @param  col_type type of col entity
     * @param  row_data row data structure carrying information about base
     * functions, DOFs indices, etc.
     * @param  col_data column data structure carrying information about base
     * functions, DOFs indices, etc.
     * @return          error code
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data) {
      MoFEMFunctionBegin;
      if (Aij == PETSC_NULL)
        MoFEMFunctionReturnHot(0);
      if (row_data.getFieldData().size() == 0)
        MoFEMFunctionReturnHot(0);
      if (col_data.getFieldData().size() == 0)
        MoFEMFunctionReturnHot(0);
      int nb_row = row_data.getFieldData().size();
      int nb_col = col_data.getFieldData().size();
      NN.resize(nb_row, nb_col);
      NN.clear();
      divVec.resize(nb_col, false);

      FTensor::Index<'i', 3> i;
      auto t_base_diff_hdiv = col_data.getFTensor2DiffN<3, 3>();

      int nb_gauss_pts = row_data.getN().size1();
      for (int gg = 0; gg < nb_gauss_pts; gg++) {
        double w = getGaussPts()(3, gg) * getVolume();
        for (auto &v : divVec) {
          v = t_base_diff_hdiv(i, i);
          ++t_base_diff_hdiv;
        }
        noalias(NN) += w * outer_prod(row_data.getN(gg), divVec);
      }
      CHKERR MatSetValues(Aij, nb_row, &row_data.getIndices()[0], nb_col,
                          &col_data.getIndices()[0], &NN(0, 0), ADD_VALUES);
      transNN.resize(nb_col, nb_row);
      ublas::noalias(transNN) = -trans(NN);
      CHKERR MatSetValues(Aij, nb_col, &col_data.getIndices()[0], nb_row,
                          &row_data.getIndices()[0], &transNN(0, 0),
                          ADD_VALUES);
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      if (data.getIndices().size() == 0)
        MoFEMFunctionReturnHot(0);
      if (data.getIndices().size() != data.getN().size2()) {
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                "data inconsistency");
      }
      int nb_row = data.getIndices().size();
      Nf.resize(nb_row);
      Nf.clear();
      int nb_gauss_pts = data.getN().size1();
      for (int gg = 0; gg < nb_gauss_pts; gg++) {
        double w = getGaussPts()(3, gg) * getVolume();
        noalias(Nf) += w * data.getN(gg) * cTx.divergenceAtGaussPts[gg];
      }
      CHKERR VecSetValues(F, nb_row, &data.getIndices()[0], &Nf[0], ADD_VALUES);
      ;
      MoFEMFunctionReturn(0);
    }
  };

  /** \brief Calculate source therms, i.e. \f$\int_\mathcal{T} f v
   * \textrm{d}\mathcal{T}\f$
   */
  struct OpL2Source
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {
    MixTransportElement &cTx;
    Vec F;

    OpL2Source(MixTransportElement &ctx, const std::string val_name, Vec f)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              val_name, UserDataOperator::OPROW),
          cTx(ctx), F(f) {}

    VectorDouble Nf;
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      if (data.getFieldData().size() == 0)
        MoFEMFunctionReturnHot(0);
      EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
      int nb_row = data.getFieldData().size();
      Nf.resize(nb_row, false);
      Nf.clear();
      int nb_gauss_pts = data.getN().size1();
      for (int gg = 0; gg < nb_gauss_pts; gg++) {
        double w = getGaussPts()(3, gg) * getVolume();
        double x, y, z;
        if (getHOCoordsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
          x = getHOCoordsAtGaussPts()(gg, 0);
          y = getHOCoordsAtGaussPts()(gg, 1);
          z = getHOCoordsAtGaussPts()(gg, 2);
        } else {
          x = getCoordsAtGaussPts()(gg, 0);
          y = getCoordsAtGaussPts()(gg, 1);
          z = getCoordsAtGaussPts()(gg, 2);
        }
        double flux = 0;
        CHKERR cTx.getSource(fe_ent, x, y, z, flux);
        ;
        noalias(Nf) += w * data.getN(gg) * flux;
      }
      CHKERR VecSetValues(F, nb_row, &data.getIndices()[0], &Nf[0], ADD_VALUES);
      ;
      MoFEMFunctionReturn(0);
    }
  };

  /**
   * \brief calculate \f$ \int_\mathcal{S} {\boldsymbol\tau} \cdot \mathbf{n}u
   \textrm{d}\mathcal{S} \f$

   * This terms comes from differentiation by parts. Note that in this Dirichlet
   * boundary conditions are natural.

   */
  struct OpRhsBcOnValues
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    MixTransportElement &cTx;
    Vec F;

    /**
     * \brief Constructor
     */
    OpRhsBcOnValues(MixTransportElement &ctx, const std::string fluxes_name,
                    Vec f)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              fluxes_name, UserDataOperator::OPROW),
          cTx(ctx), F(f) {}

    VectorDouble nF;

    /**
     * \brief Integrate boundary condition
     * @param  side local index of entity
     * @param  type type of entity
     * @param  data data on entity
     * @return      error code
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      if (data.getFieldData().size() == 0)
        MoFEMFunctionReturnHot(0);
      EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
      nF.resize(data.getIndices().size());
      nF.clear();
      int nb_gauss_pts = data.getN().size1();
      for (int gg = 0; gg < nb_gauss_pts; gg++) {
        double x, y, z;
        if (getNormalsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
          x = getHOCoordsAtGaussPts()(gg, 0);
          y = getHOCoordsAtGaussPts()(gg, 1);
          z = getHOCoordsAtGaussPts()(gg, 2);
        } else {
          x = getCoordsAtGaussPts()(gg, 0);
          y = getCoordsAtGaussPts()(gg, 1);
          z = getCoordsAtGaussPts()(gg, 2);
        }
        double value;
        CHKERR cTx.getBcOnValues(fe_ent, gg, x, y, z, value);
        ;
        double w = getGaussPts()(2, gg) * 0.5;
        if (getNormalsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
          noalias(nF) +=
              w * prod(data.getVectorN<3>(gg), getNormalsAtGaussPts(gg)) *
              value;
        } else {
          noalias(nF) += w * prod(data.getVectorN<3>(gg), getNormal()) * value;
        }
      }
      Vec f = (F != PETSC_NULL) ? F : getFEMethod()->ts_F;
      CHKERR VecSetValues(f, data.getIndices().size(), &data.getIndices()[0],
                          &nF[0], ADD_VALUES);

      MoFEMFunctionReturn(0);
    }
  };

  /**
   * \brief Evaluate boundary conditions on fluxes.
   *
   * Note that Neumann boundary conditions here are essential. So it is opposite
   * what you find in displacement finite element method.
   *

   * Here we have to solve for degrees of freedom on boundary such base
   functions
   * approximate flux.

   *
   */
  struct OpEvaluateBcOnFluxes
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {
    MixTransportElement &cTx;
    OpEvaluateBcOnFluxes(MixTransportElement &ctx, const std::string flux_name)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              flux_name, UserDataOperator::OPROW),
          cTx(ctx) {}

    MatrixDouble NN;
    VectorDouble Nf;
    FTensor::Index<'i', 3> i;

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      if (data.getFieldData().size() == 0)
        MoFEMFunctionReturnHot(0);
      EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
      int nb_dofs = data.getFieldData().size();
      int nb_gauss_pts = data.getN().size1();
      if (3 * nb_dofs != static_cast<int>(data.getN().size2())) {
        SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                "wrong number of dofs");
      }
      NN.resize(nb_dofs, nb_dofs);
      Nf.resize(nb_dofs);
      NN.clear();
      Nf.clear();

      // Get normal vector. Note that when higher order geometry is set, then
      // face element could be curved, i.e. normal can be different at each
      // integration point.
      double *normal_ptr;
      if (getNormalsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
        // HO geometry
        normal_ptr = &getNormalsAtGaussPts(0)[0];
      } else {
        // Linear geometry, i.e. constant normal on face
        normal_ptr = &getNormal()[0];
      }
      // set tensor from pointer
      FTensor::Tensor1<const double *, 3> t_normal(normal_ptr, &normal_ptr[1],
                                                   &normal_ptr[2], 3);
      // get base functions
      auto t_n_hdiv_row = data.getFTensor1N<3>();

      double nrm2 = 0;

      // loop over integration points
      for (int gg = 0; gg < nb_gauss_pts; gg++) {

        // get integration point coordinates
        double x, y, z;
        if (getNormalsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
          x = getHOCoordsAtGaussPts()(gg, 0);
          y = getHOCoordsAtGaussPts()(gg, 1);
          z = getHOCoordsAtGaussPts()(gg, 2);
        } else {
          x = getCoordsAtGaussPts()(gg, 0);
          y = getCoordsAtGaussPts()(gg, 1);
          z = getCoordsAtGaussPts()(gg, 2);
        }

        // get flux on fece for given element handle and coordinates
        double flux;
        CHKERR cTx.getBcOnFluxes(fe_ent, x, y, z, flux);
        ;
        // get weight for integration rule
        double w = getGaussPts()(2, gg);
        if (gg == 0) {
          nrm2 = sqrt(t_normal(i) * t_normal(i));
        }

        // set tensor of rank 0 to matrix NN elements
        // loop over base functions on rows and columns
        for (int ll = 0; ll != nb_dofs; ll++) {
          // get column on shape functions
          FTensor::Tensor1<const double *, 3> t_n_hdiv_col(
              &data.getVectorN<3>(gg)(0, HVEC0),
              &data.getVectorN<3>(gg)(0, HVEC1),
              &data.getVectorN<3>(gg)(0, HVEC2), 3);
          for (int kk = 0; kk <= ll; kk++) {
            NN(ll, kk) += w * t_n_hdiv_row(i) * t_n_hdiv_col(i);
            ++t_n_hdiv_col;
          }
          // right hand side
          Nf[ll] += w * t_n_hdiv_row(i) * t_normal(i) * flux / nrm2;
          ++t_n_hdiv_row;
        }

        // If HO geometry increment t_normal to next integration point
        if (getNormalsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
          ++t_normal;
          nrm2 = sqrt(t_normal(i) * t_normal(i));
        }
      }
      // get global dofs indices on element
      cTx.bcIndices.insert(data.getIndices().begin(), data.getIndices().end());
      // factor matrix
      cholesky_decompose(NN);
      // solve local problem
      cholesky_solve(NN, Nf, ublas::lower());

      // set solution to vector
      CHKERR VecSetOption(cTx.D0, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
      CHKERR VecSetValues(cTx.D0, nb_dofs, &*data.getIndices().begin(),
                          &*Nf.begin(), INSERT_VALUES);
      for (int dd = 0; dd != nb_dofs; ++dd)
        data.getFieldDofs()[dd]->getFieldData() = Nf[dd];

      MoFEMFunctionReturn(0);
    }
  };

  /**
   * \brief Calculate values at integration points
   */
  struct OpValuesAtGaussPts
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    MixTransportElement &cTx;

    OpValuesAtGaussPts(MixTransportElement &ctx,
                       const std::string val_name = "VALUES")
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              val_name, UserDataOperator::OPROW),
          cTx(ctx) {}

    virtual ~OpValuesAtGaussPts() {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      if (data.getFieldData().size() == 0)
        MoFEMFunctionReturnHot(0);

      int nb_gauss_pts = data.getN().size1();
      cTx.valuesAtGaussPts.resize(nb_gauss_pts);
      for (int gg = 0; gg < nb_gauss_pts; gg++) {
        cTx.valuesAtGaussPts[gg] =
            inner_prod(trans(data.getN(gg)), data.getFieldData());
      }

      MoFEMFunctionReturn(0);
    }
  };

  /**
   * \brief Calculate gradients of values at integration points
   */
  struct OpValuesGradientAtGaussPts
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    MixTransportElement &cTx;

    OpValuesGradientAtGaussPts(MixTransportElement &ctx,
                               const std::string val_name = "VALUES")
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              val_name, UserDataOperator::OPROW),
          cTx(ctx) {}
    virtual ~OpValuesGradientAtGaussPts() {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      if (data.getFieldData().size() == 0)
        MoFEMFunctionReturnHot(0);
      int nb_gauss_pts = data.getDiffN().size1();
      cTx.valuesGradientAtGaussPts.resize(3, nb_gauss_pts);
      for (int gg = 0; gg < nb_gauss_pts; gg++) {
        ublas::matrix_column<MatrixDouble> values_grad_at_gauss_pts(
            cTx.valuesGradientAtGaussPts, gg);
        noalias(values_grad_at_gauss_pts) =
            prod(trans(data.getDiffN(gg)), data.getFieldData());
      }
      MoFEMFunctionReturn(0);
    }
  };

  /**
   * \brief calculate flux at integration point
   */
  struct OpFluxDivergenceAtGaussPts
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    MixTransportElement &cTx;

    OpFluxDivergenceAtGaussPts(MixTransportElement &ctx,
                               const std::string field_name)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          cTx(ctx) {}

    VectorDouble divVec;
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;

      if (data.getFieldData().size() == 0)
        MoFEMFunctionReturnHot(0);
      int nb_gauss_pts = data.getDiffN().size1();
      int nb_dofs = data.getFieldData().size();
      cTx.fluxesAtGaussPts.resize(3, nb_gauss_pts);
      cTx.divergenceAtGaussPts.resize(nb_gauss_pts);
      if (type == MBTRI && side == 0) {
        cTx.divergenceAtGaussPts.clear();
        cTx.fluxesAtGaussPts.clear();
      }
      divVec.resize(nb_dofs);

      FTensor::Index<'i', 3> i;
      auto t_base_diff_hdiv = data.getFTensor2DiffN<3, 3>();

      for (int gg = 0; gg < nb_gauss_pts; gg++) {
        for(auto &v : divVec) {
          v = t_base_diff_hdiv(i, i);
          ++t_base_diff_hdiv;
        }
        cTx.divergenceAtGaussPts[gg] += inner_prod(divVec, data.getFieldData());
        ublas::matrix_column<MatrixDouble> flux_at_gauss_pt(
            cTx.fluxesAtGaussPts, gg);
        flux_at_gauss_pt +=
            prod(trans(data.getVectorN<3>(gg)), data.getFieldData());
      }
      MoFEMFunctionReturn(0);
    }
  };

  map<double, EntityHandle> errorMap;
  double sumErrorFlux;
  double sumErrorDiv;
  double sumErrorJump;

  /** \brief calculate error evaluator
   */
  struct OpError
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    MixTransportElement &cTx;

    OpError(MixTransportElement &ctx, const std::string field_name)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          cTx(ctx) {}
    virtual ~OpError() {}

    VectorDouble deltaFlux;
    MatrixDouble3by3 invK;

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      if (type != MBTET)
        MoFEMFunctionReturnHot(0);
      invK.resize(3, 3, false);
      int nb_gauss_pts = data.getN().size1();
      EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
      double def_val = 0;
      Tag th_error_flux, th_error_div;
      CHKERR cTx.mField.get_moab().tag_get_handle(
          "ERROR_FLUX", 1, MB_TYPE_DOUBLE, th_error_flux,
          MB_TAG_CREAT | MB_TAG_SPARSE, &def_val);
      double *error_flux_ptr;
      CHKERR cTx.mField.get_moab().tag_get_by_ptr(
          th_error_flux, &fe_ent, 1, (const void **)&error_flux_ptr);

      CHKERR cTx.mField.get_moab().tag_get_handle(
          "ERROR_DIV", 1, MB_TYPE_DOUBLE, th_error_div,
          MB_TAG_CREAT | MB_TAG_SPARSE, &def_val);
      double *error_div_ptr;
      CHKERR cTx.mField.get_moab().tag_get_by_ptr(
          th_error_div, &fe_ent, 1, (const void **)&error_div_ptr);

      Tag th_error_jump;
      CHKERR cTx.mField.get_moab().tag_get_handle(
          "ERROR_JUMP", 1, MB_TYPE_DOUBLE, th_error_jump,
          MB_TAG_CREAT | MB_TAG_SPARSE, &def_val);
      double *error_jump_ptr;
      CHKERR cTx.mField.get_moab().tag_get_by_ptr(
          th_error_jump, &fe_ent, 1, (const void **)&error_jump_ptr);
      *error_jump_ptr = 0;

      /// characteristic size of the element
      const double h = pow(getVolume() * 12 / sqrt(2), (double)1 / 3);

      for (int ff = 0; ff != 4; ff++) {
        EntityHandle face;
        CHKERR cTx.mField.get_moab().side_element(fe_ent, 2, ff, face);
        double *error_face_jump_ptr;
        CHKERR cTx.mField.get_moab().tag_get_by_ptr(
            th_error_jump, &face, 1, (const void **)&error_face_jump_ptr);
        *error_face_jump_ptr = (1 / sqrt(h)) * sqrt(*error_face_jump_ptr);
        *error_face_jump_ptr = pow(*error_face_jump_ptr, 2);
        *error_jump_ptr += *error_face_jump_ptr;
      }

      *error_flux_ptr = 0;
      *error_div_ptr = 0;
      deltaFlux.resize(3, false);
      for (int gg = 0; gg < nb_gauss_pts; gg++) {
        double w = getGaussPts()(3, gg) * getVolume();
        double x, y, z;
        if (getHOCoordsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
          x = getHOCoordsAtGaussPts()(gg, 0);
          y = getHOCoordsAtGaussPts()(gg, 1);
          z = getHOCoordsAtGaussPts()(gg, 2);
        } else {
          x = getCoordsAtGaussPts()(gg, 0);
          y = getCoordsAtGaussPts()(gg, 1);
          z = getCoordsAtGaussPts()(gg, 2);
        }
        double flux;
        CHKERR cTx.getSource(fe_ent, x, y, z, flux);
        ;
        *error_div_ptr += w * pow(cTx.divergenceAtGaussPts[gg] - flux, 2);
        CHKERR cTx.getResistivity(fe_ent, x, y, z, invK);
        ;
        ublas::matrix_column<MatrixDouble> flux_at_gauss_pt(
            cTx.fluxesAtGaussPts, gg);
        ublas::matrix_column<MatrixDouble> values_grad_at_gauss_pts(
            cTx.valuesGradientAtGaussPts, gg);
        noalias(deltaFlux) =
            prod(invK, flux_at_gauss_pt) + values_grad_at_gauss_pts;
        *error_flux_ptr += w * inner_prod(deltaFlux, deltaFlux);
      }
      *error_div_ptr = h * sqrt(*error_div_ptr);
      *error_div_ptr = pow(*error_div_ptr, 2);
      cTx.errorMap[sqrt(*error_flux_ptr + *error_div_ptr + *error_jump_ptr)] =
          fe_ent;
      // Sum/Integrate all errors
      cTx.sumErrorFlux += *error_flux_ptr * getVolume();
      cTx.sumErrorDiv += *error_div_ptr * getVolume();
      // FIXME: Summation should be while skeleton is calculated
      cTx.sumErrorJump +=
          *error_jump_ptr * getVolume(); /// FIXME: this need to be fixed
      MoFEMFunctionReturn(0);
    }
  };

  /**
   * \brief calculate jump on entities
   */
  struct OpSkeleton
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    /**
     * \brief volume element to get values from adjacent tets to face
     */
    VolumeElementForcesAndSourcesCoreOnSide volSideFe;

    /** store values at integration point, key of the map is sense of face in
     * respect to adjacent tetrahedra
     */
    map<int, VectorDouble> valMap;

    /**
     * \brief calculate values on adjacent tetrahedra to face
     */
    struct OpVolSide
        : public VolumeElementForcesAndSourcesCoreOnSide::UserDataOperator {
      map<int, VectorDouble> &valMap;
      OpVolSide(map<int, VectorDouble> &val_map)
          : VolumeElementForcesAndSourcesCoreOnSide::UserDataOperator(
                "VALUES", UserDataOperator::OPROW),
            valMap(val_map) {}
      MoFEMErrorCode doWork(int side, EntityType type,
                            DataForcesAndSourcesCore::EntData &data) {
        MoFEMFunctionBegin;
        if (data.getFieldData().size() == 0)
          MoFEMFunctionReturnHot(0);
        int nb_gauss_pts = data.getN().size1();
        valMap[getFaceSense()].resize(nb_gauss_pts);
        for (int gg = 0; gg < nb_gauss_pts; gg++) {
          valMap[getFaceSense()][gg] =
              inner_prod(trans(data.getN(gg)), data.getFieldData());
        }
        MoFEMFunctionReturn(0);
      }
    };

    MixTransportElement &cTx;

    OpSkeleton(MixTransportElement &ctx, const std::string flux_name)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              flux_name, UserDataOperator::OPROW),
          volSideFe(ctx.mField), cTx(ctx) {
      volSideFe.getOpPtrVector().push_back(new OpSkeleton::OpVolSide(valMap));
    }

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;

      if (type == MBTRI) {
        EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();

        double def_val = 0;
        Tag th_error_jump;
        CHKERR cTx.mField.get_moab().tag_get_handle(
            "ERROR_JUMP", 1, MB_TYPE_DOUBLE, th_error_jump,
            MB_TAG_CREAT | MB_TAG_SPARSE, &def_val);
        double *error_jump_ptr;
        CHKERR cTx.mField.get_moab().tag_get_by_ptr(
            th_error_jump, &fe_ent, 1, (const void **)&error_jump_ptr);
        *error_jump_ptr = 0;

        // check if this is essential boundary condition
        EntityHandle essential_bc_meshset =
            cTx.mField.get_finite_element_meshset("MIX_BCFLUX");
        if (cTx.mField.get_moab().contains_entities(essential_bc_meshset,
                                                    &fe_ent, 1)) {
          // essential bc, np jump then, exit and go to next face
          MoFEMFunctionReturnHot(0);
        }

        // calculate values form adjacent tets
        valMap.clear();
        CHKERR loopSideVolumes("MIX", volSideFe);
        ;

        int nb_gauss_pts = data.getN().size1();

        // it is only one face, so it has to be bc natural boundary condition
        if (valMap.size() == 1) {
          if (static_cast<int>(valMap.begin()->second.size()) != nb_gauss_pts) {
            SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                    "wrong number of integration points");
          }
          for (int gg = 0; gg != nb_gauss_pts; gg++) {
            double x, y, z;
            if (static_cast<int>(getNormalsAtGaussPts().size1()) ==
                nb_gauss_pts) {
              x = getHOCoordsAtGaussPts()(gg, 0);
              y = getHOCoordsAtGaussPts()(gg, 1);
              z = getHOCoordsAtGaussPts()(gg, 2);
            } else {
              x = getCoordsAtGaussPts()(gg, 0);
              y = getCoordsAtGaussPts()(gg, 1);
              z = getCoordsAtGaussPts()(gg, 2);
            }
            double value;
            CHKERR cTx.getBcOnValues(fe_ent, gg, x, y, z, value);
            ;
            double w = getGaussPts()(2, gg);
            if (static_cast<int>(getNormalsAtGaussPts().size1()) ==
                nb_gauss_pts) {
              w *= norm_2(getNormalsAtGaussPts(gg)) * 0.5;
            } else {
              w *= getArea();
            }
            *error_jump_ptr += w * pow(value - valMap.begin()->second[gg], 2);
          }
        } else if (valMap.size() == 2) {
          for (int gg = 0; gg != nb_gauss_pts; gg++) {
            double w = getGaussPts()(2, gg);
            if (getNormalsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
              w *= norm_2(getNormalsAtGaussPts(gg)) * 0.5;
            } else {
              w *= getArea();
            }
            double delta = valMap.at(1)[gg] - valMap.at(-1)[gg];
            *error_jump_ptr += w * pow(delta, 2);
          }
        } else {
          SETERRQ1(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                   "data inconsistency, wrong number of neighbors "
                   "valMap.size() = %d",
                   valMap.size());
        }
      }

      MoFEMFunctionReturn(0);
    }
  };
};

} // namespace MixTransport

#endif //_MIX_TRANPORT_ELEMENT_HPP_

/***************************************************************************/ /**
                                                                               * \defgroup mofem_mix_transport_elem Mix transport element
                                                                               * \ingroup user_modules
                                                                               ******************************************************************************/
