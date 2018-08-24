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

  Note to solve this system you need to use direct solver or proper preconditioner
  for saddle problem.

  It is based on \cite arnold2006differential \cite arnold2012mixed \cite reviartthomas1996
  <https://www.researchgate.net/profile/Richard_Falk/publication/226454406_Differential_Complexes_and_Stability_of_Finite_Element_Methods_I._The_de_Rham_Complex/links/02e7e5214f0426ff77000000.pdf>

  General problem have form,
  \f[
  \mathbf{A} \boldsymbol\sigma + \textrm{grad}[u] = \mathbf{0} \; \textrm{on} \; \Omega \\
  \textrm{div}[\boldsymbol\sigma] = f \; \textrm{on} \; \Omega
  \f]

*/
struct MixTransportElement {

  MoFEM::Interface &mField;

  /**
   * \brief definition of volume element

   * It is used to calculate volume integrals. On volume element we set-up
   * operators to calculate components of matrix and vector.

   */
  struct MyVolumeFE: public MoFEM::VolumeElementForcesAndSourcesCore {
    MyVolumeFE(MoFEM::Interface &m_field): MoFEM::VolumeElementForcesAndSourcesCore(m_field) {}
    int getRule(int order) { return 2*order+1; };
  };

  MyVolumeFE feVol;   ///< Instance of volume element

  /** \brief definition of surface element

    * It is used to calculate surface integrals. On volume element are operators
    * evaluating natural boundary conditions.

    */
  struct MyTriFE: public MoFEM::FaceElementForcesAndSourcesCore {
    MyTriFE(MoFEM::Interface &m_field): MoFEM::FaceElementForcesAndSourcesCore(m_field) {}
    int getRule(int order) { return 2*order+1; };
  };

  MyTriFE feTri;   ///< Instance of surface element

  /**
   * \brief construction of this data structure
   */
  MixTransportElement(MoFEM::Interface &m_field):
  mField(m_field),
  feVol(m_field),
  feTri(m_field) {};

  /**
   * \brief destructor
   */
  virtual ~MixTransportElement() {}

  VectorDouble valuesAtGaussPts;                          ///< values at integration points on element
  MatrixDouble valuesGradientAtGaussPts;                  ///< gradients at integration points on element
  VectorDouble divergenceAtGaussPts;                      ///< divergence at integration points on element
  MatrixDouble fluxesAtGaussPts;                          ///< fluxes at integration points on element

  set<int> bcIndices;

  /**
   * \brief get dof indices where essential boundary conditions are applied
   * @param  is indices
   * @return    error code
   */
  MoFEMErrorCode getDirichletBCIndices(IS *is) {
    MoFEMFunctionBeginHot;
    std::vector<int> ids;
    ids.insert(ids.begin(),bcIndices.begin(),bcIndices.end());
    IS is_local;
    ierr = ISCreateGeneral(
      mField.get_comm(),ids.size(),ids.empty()?PETSC_NULL:&ids[0],PETSC_COPY_VALUES,&is_local
    ); CHKERRG(ierr);
    ierr = ISAllGather(is_local,is); CHKERRG(ierr);
    ierr = ISDestroy(&is_local); CHKERRG(ierr);
    MoFEMFunctionReturnHot(0);
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
  virtual MoFEMErrorCode getSource(
    const EntityHandle ent,
    const double x,const double y,const double z,
    double &flux
  ) {
    MoFEMFunctionBeginHot;
    flux  = 0;
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
  virtual MoFEMErrorCode getResistivity(
    const EntityHandle ent,
    const double x,const double y,const double z,
    MatrixDouble3by3& inv_k
  ) {
    MoFEMFunctionBeginHot;
    inv_k.clear();
    for(int dd = 0;dd<3;dd++) {
      inv_k(dd,dd) = 1;
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
  virtual MoFEMErrorCode getBcOnValues(
    const EntityHandle ent,
    const int gg,
    const double x,const double y,const double z,
    double &value) {
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
  virtual MoFEMErrorCode getBcOnFluxes(
    const EntityHandle ent,
    const double x,const double y,const double z,
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
  std::map<int,BlockData> setOfBlocks; ///< maps block set id with appropriate BlockData

  /**
   * \brief Add fields to database
   * @param  values name of the fields
   * @param  fluxes name of filed for fluxes
   * @param  order  order of approximation
   * @return        error code
   */
  MoFEMErrorCode addFields(const std::string &values,const std::string &fluxes,const int order) {

    MoFEMFunctionBeginHot;
    //Fields
    ierr = mField.add_field(fluxes,HDIV,DEMKOWICZ_JACOBI_BASE,1); CHKERRG(ierr);
    ierr = mField.add_field(values,L2,AINSWORTH_LEGENDRE_BASE,1); CHKERRG(ierr);

    //meshset consisting all entities in mesh
    EntityHandle root_set = mField.get_moab().get_root_set();
    //add entities to field
    ierr = mField.add_ents_to_field_by_type(root_set,MBTET,fluxes); CHKERRG(ierr);
    ierr = mField.add_ents_to_field_by_type(root_set,MBTET,values); CHKERRG(ierr);
    ierr = mField.set_field_order(root_set,MBTET,fluxes,order+1); CHKERRG(ierr);
    ierr = mField.set_field_order(root_set,MBTRI,fluxes,order+1); CHKERRG(ierr);
    ierr = mField.set_field_order(root_set,MBTET,values,order); CHKERRG(ierr);


    MoFEMFunctionReturnHot(0);
  }

  /// \brief add finite elements
  MoFEMErrorCode addFiniteElements(
    const std::string &fluxes_name,
    const std::string &values_name,
    const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS"
  ) {
    MoFEMFunctionBeginHot;




    // Set up volume element operators. Operators are used to calculate components
    // of stiffness matrix & right hand side, in essence are used to do volume integrals over
    // tetrahedral in this case.

    // Define element "MIX". Note that this element will work with fluxes_name and
    // values_name. This reflect bilinear form for the problem
    ierr = mField.add_finite_element("MIX",MF_ZERO); CHKERRG(ierr);
    ierr = mField.modify_finite_element_add_field_row("MIX",fluxes_name); CHKERRG(ierr);
    ierr = mField.modify_finite_element_add_field_col("MIX",fluxes_name); CHKERRG(ierr);
    ierr = mField.modify_finite_element_add_field_row("MIX",values_name); CHKERRG(ierr);
    ierr = mField.modify_finite_element_add_field_col("MIX",values_name); CHKERRG(ierr);
    ierr = mField.modify_finite_element_add_field_data("MIX",fluxes_name); CHKERRG(ierr);
    ierr = mField.modify_finite_element_add_field_data("MIX",values_name); CHKERRG(ierr);

    // Define finite element to integrate over skeleton, we need that to evaluate error
    ierr = mField.add_finite_element("MIX_SKELETON",MF_ZERO); CHKERRG(ierr);
    ierr = mField.modify_finite_element_add_field_row("MIX_SKELETON",fluxes_name); CHKERRG(ierr);
    ierr = mField.modify_finite_element_add_field_col("MIX_SKELETON",fluxes_name); CHKERRG(ierr);
    ierr = mField.modify_finite_element_add_field_data("MIX_SKELETON",fluxes_name); CHKERRG(ierr);

    // In some cases you like to use HO geometry to describe shape of the bode, curved edges and faces, for
    // example body is a sphere. HO geometry is approximated by a field,  which can be hierarchical, so shape of
    // the edges could be given by polynomial of arbitrary order.
    //
    // Check if field "mesh_nodals_positions" is defined, and if it is add that field to data of finite
    // element. MoFEM will use that that to calculate Jacobian as result that geometry in nonlinear.
    if(mField.check_field(mesh_nodals_positions)) {
      ierr = mField.modify_finite_element_add_field_data("MIX",mesh_nodals_positions); CHKERRG(ierr);
      ierr = mField.modify_finite_element_add_field_data("MIX_SKELETON",mesh_nodals_positions); CHKERRG(ierr);
    }
    // Look for all BLOCKSET which are MAT_THERMALSET, takes entities from those BLOCKSETS
    // and add them to "MIX" finite element. In addition get data form that meshset
    // and set conductivity which is used to calculate fluxes from gradients of concentration
    // or gradient of temperature, depending how you interpret variables.
    for(_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(mField,BLOCKSET|MAT_THERMALSET,it)) {

      // cerr << *it << endl;

      Mat_Thermal temp_data;
      ierr = it->getAttributeDataStructure(temp_data); CHKERRG(ierr);
      setOfBlocks[it->getMeshsetId()].cOnductivity = temp_data.data.Conductivity;
      setOfBlocks[it->getMeshsetId()].cApacity = temp_data.data.HeatCapacity;
      rval = mField.get_moab().get_entities_by_type(
        it->meshset,MBTET,setOfBlocks[it->getMeshsetId()].tEts,true
      ); CHKERRG(rval);
      ierr = mField.add_ents_to_finite_element_by_type(
        setOfBlocks[it->getMeshsetId()].tEts,MBTET,"MIX"
      ); CHKERRG(ierr);

      Range skeleton;
      rval = mField.get_moab().get_adjacencies(
        setOfBlocks[it->getMeshsetId()].tEts,2,false,skeleton,moab::Interface::UNION
      );
      ierr = mField.add_ents_to_finite_element_by_type(
        skeleton,MBTRI,"MIX_SKELETON"
      ); CHKERRG(ierr);

    }

    // Define element to integrate natural boundary conditions, i.e. set values.
    ierr = mField.add_finite_element("MIX_BCVALUE",MF_ZERO); CHKERRG(ierr);
    ierr = mField.modify_finite_element_add_field_row("MIX_BCVALUE",fluxes_name); CHKERRG(ierr);
    ierr = mField.modify_finite_element_add_field_col("MIX_BCVALUE",fluxes_name); CHKERRG(ierr);
    ierr = mField.modify_finite_element_add_field_data("MIX_BCVALUE",fluxes_name); CHKERRG(ierr);
    ierr = mField.modify_finite_element_add_field_data("MIX_BCVALUE",values_name); CHKERRG(ierr);
    if(mField.check_field(mesh_nodals_positions)) {
      ierr = mField.modify_finite_element_add_field_data("MIX_BCVALUE",mesh_nodals_positions); CHKERRG(ierr);
    }

    // Define element to apply essential boundary conditions.
    ierr = mField.add_finite_element("MIX_BCFLUX",MF_ZERO); CHKERRG(ierr);
    ierr = mField.modify_finite_element_add_field_row("MIX_BCFLUX",fluxes_name); CHKERRG(ierr);
    ierr = mField.modify_finite_element_add_field_col("MIX_BCFLUX",fluxes_name); CHKERRG(ierr);
    ierr = mField.modify_finite_element_add_field_data("MIX_BCFLUX",fluxes_name); CHKERRG(ierr);
    ierr = mField.modify_finite_element_add_field_data("MIX_BCFLUX",values_name); CHKERRG(ierr);
    if(mField.check_field(mesh_nodals_positions)) {
      ierr = mField.modify_finite_element_add_field_data("MIX_BCFLUX",mesh_nodals_positions); CHKERRG(ierr);
    }

    MoFEMFunctionReturnHot(0);
  }

  /**
   * \brief Build problem
   * @param  ref_level mesh refinement on which mesh problem you like to built.
   * @return           error code
   */
  MoFEMErrorCode buildProblem(BitRefLevel &ref_level) {

    MoFEMFunctionBeginHot;
    //build field
    ierr = mField.build_fields(); CHKERRG(ierr);
    // get tetrahedrons which has been build previously and now in so called garbage bit level
    Range done_tets;
    ierr = mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
      BitRefLevel().set(0),BitRefLevel().set(),MBTET,done_tets
    ); CHKERRG(ierr);
    ierr = mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
      BitRefLevel().set(BITREFLEVEL_SIZE-1),BitRefLevel().set(),MBTET,done_tets
    ); CHKERRG(ierr);
    // get tetrahedrons which belong to problem bit level
    Range ref_tets;
    ierr = mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
      ref_level,BitRefLevel().set(),MBTET,ref_tets
    ); CHKERRG(ierr);
    ref_tets = subtract(ref_tets,done_tets);
    ierr = mField.build_finite_elements("MIX",&ref_tets,2); CHKERRG(ierr);
    // get triangles which has been build previously and now in so called garbage bit level
    Range done_faces;
    ierr = mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
      BitRefLevel().set(0),BitRefLevel().set(),MBTRI,done_faces
    ); CHKERRG(ierr);
    ierr = mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
      BitRefLevel().set(BITREFLEVEL_SIZE-1),BitRefLevel().set(),MBTRI,done_faces
    ); CHKERRG(ierr);
    // get triangles which belong to problem bit level
    Range ref_faces;
    ierr = mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
      ref_level,BitRefLevel().set(),MBTRI,ref_faces
    ); CHKERRG(ierr);
    ref_faces = subtract(ref_faces,done_faces);
    //build finite elements structures
    ierr = mField.build_finite_elements("MIX_BCFLUX",&ref_faces,2); CHKERRG(ierr);
    ierr = mField.build_finite_elements("MIX_BCVALUE",&ref_faces,2); CHKERRG(ierr);
    ierr = mField.build_finite_elements("MIX_SKELETON",&ref_faces,2); CHKERRG(ierr);
    //Build adjacencies of degrees of freedom and elements
    ierr = mField.build_adjacencies(ref_level); CHKERRG(ierr);
    //Define problem
    ierr = mField.add_problem("MIX",MF_ZERO); CHKERRG(ierr);
    //set refinement level for problem
    ierr = mField.modify_problem_ref_level_set_bit("MIX",ref_level); CHKERRG(ierr);
    // Add element to problem
    ierr = mField.modify_problem_add_finite_element("MIX","MIX"); CHKERRG(ierr);
    ierr = mField.modify_problem_add_finite_element("MIX","MIX_SKELETON"); CHKERRG(ierr);
    // Boundary conditions
    ierr = mField.modify_problem_add_finite_element("MIX","MIX_BCFLUX"); CHKERRG(ierr);
    ierr = mField.modify_problem_add_finite_element("MIX","MIX_BCVALUE"); CHKERRG(ierr);
    //build problem

    ProblemsManager *prb_mng_ptr;
    ierr = mField.getInterface(prb_mng_ptr); CHKERRG(ierr);
    ierr = prb_mng_ptr->buildProblem("MIX",true); CHKERRG(ierr);
    //mesh partitioning
    //partition
    ierr = prb_mng_ptr->partitionProblem("MIX"); CHKERRG(ierr);
    ierr = prb_mng_ptr->partitionFiniteElements("MIX"); CHKERRG(ierr);
    //what are ghost nodes, see Petsc Manual
    ierr = prb_mng_ptr->partitionGhostDofs("MIX"); CHKERRG(ierr);
    MoFEMFunctionReturnHot(0);
  }

  struct OpPostProc: MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {
    moab::Interface &postProcMesh;
    std::vector<EntityHandle> &mapGaussPts;
    OpPostProc(
      moab::Interface &post_proc_mesh,
      std::vector<EntityHandle> &map_gauss_pts
    ):
    MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator("VALUES",UserDataOperator::OPCOL),
    postProcMesh(post_proc_mesh),
    mapGaussPts(map_gauss_pts) {
    }
    MoFEMErrorCode doWork(
      int side,
      EntityType type,
      DataForcesAndSourcesCore::EntData &data
    ) {
      MoFEMFunctionBeginHot;
      if(type != MBTET) MoFEMFunctionReturnHot(0);
      EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
      Tag th_error_flux;
      rval = getVolumeFE()->mField.get_moab().tag_get_handle("ERROR_FLUX",th_error_flux); CHKERRG(rval);
      double* error_flux_ptr;
      rval = getVolumeFE()->mField.get_moab().tag_get_by_ptr(
        th_error_flux,&fe_ent,1,(const void**)&error_flux_ptr
      ); CHKERRG(rval);

      Tag th_error_div;
      rval = getVolumeFE()->mField.get_moab().tag_get_handle("ERROR_DIV",th_error_div); CHKERRG(rval);
      double* error_div_ptr;
      rval = getVolumeFE()->mField.get_moab().tag_get_by_ptr(
        th_error_div,&fe_ent,1,(const void**)&error_div_ptr
      ); CHKERRG(rval);

      Tag th_error_jump;
      rval = getVolumeFE()->mField.get_moab().tag_get_handle("ERROR_JUMP",th_error_jump); CHKERRG(rval);
      double* error_jump_ptr;
      rval = getVolumeFE()->mField.get_moab().tag_get_by_ptr(
        th_error_jump,&fe_ent,1,(const void**)&error_jump_ptr
      ); CHKERRG(rval);

      {
        double def_val = 0;
        Tag th_error_flux;
        rval = postProcMesh.tag_get_handle(
          "ERROR_FLUX",1,MB_TYPE_DOUBLE,th_error_flux,MB_TAG_CREAT|MB_TAG_SPARSE,&def_val
        ); CHKERRG(rval);
        for(vector<EntityHandle>::iterator vit = mapGaussPts.begin();vit!=mapGaussPts.end();vit++) {
          rval = postProcMesh.tag_set_data(th_error_flux,&*vit,1,error_flux_ptr); CHKERRG(rval);
        }

        Tag th_error_div;
        rval = postProcMesh.tag_get_handle(
          "ERROR_DIV",1,MB_TYPE_DOUBLE,th_error_div,MB_TAG_CREAT|MB_TAG_SPARSE,&def_val
        ); CHKERRG(rval);
        for(vector<EntityHandle>::iterator vit = mapGaussPts.begin();vit!=mapGaussPts.end();vit++) {
          rval = postProcMesh.tag_set_data(th_error_div,&*vit,1,error_div_ptr); CHKERRG(rval);
        }

        Tag th_error_jump;
        rval = postProcMesh.tag_get_handle(
          "ERROR_JUMP",1,MB_TYPE_DOUBLE,th_error_jump,MB_TAG_CREAT|MB_TAG_SPARSE,&def_val
        ); CHKERRG(rval);
        for(vector<EntityHandle>::iterator vit = mapGaussPts.begin();vit!=mapGaussPts.end();vit++) {
          rval = postProcMesh.tag_set_data(th_error_jump,&*vit,1,error_jump_ptr); CHKERRG(rval);
        }

      }
      MoFEMFunctionReturnHot(0);
    }
  };

  /**
   * \brief Post process results
   * @return error code
   */
  MoFEMErrorCode postProc(const string out_file) {

    MoFEMFunctionBeginHot;
    PostProcVolumeOnRefinedMesh post_proc(mField);
    ierr = post_proc.generateReferenceElementMesh(); CHKERRG(ierr);
    ierr = post_proc.addFieldValuesPostProc("VALUES"); CHKERRG(ierr);
    ierr = post_proc.addFieldValuesGradientPostProc("VALUES"); CHKERRG(ierr);
    ierr = post_proc.addFieldValuesPostProc("FLUXES"); CHKERRG(ierr);
    // ierr = post_proc.addHdivFunctionsPostProc("FLUXES"); CHKERRG(ierr);
    post_proc.getOpPtrVector().push_back(new OpPostProc(post_proc.postProcMesh,post_proc.mapGaussPts));
    ierr = mField.loop_finite_elements("MIX","MIX",post_proc);  CHKERRG(ierr);
    ierr = post_proc.writeFile(out_file.c_str()); CHKERRG(ierr);
    MoFEMFunctionReturnHot(0);
  }

  Vec D,D0,F;
  Mat Aij;

  /// \brief create matrices
  MoFEMErrorCode createMatrices() {
    MoFEMFunctionBeginHot;
    ierr = mField.MatCreateMPIAIJWithArrays("MIX",&Aij); CHKERRG(ierr);
    ierr = mField.getInterface<VecManager>()->vecCreateGhost("MIX",COL,&D); CHKERRG(ierr);
    ierr = mField.getInterface<VecManager>()->vecCreateGhost("MIX",COL,&D0); CHKERRG(ierr);
    ierr = mField.getInterface<VecManager>()->vecCreateGhost("MIX",ROW,&F); CHKERRG(ierr);
    MoFEMFunctionReturnHot(0);
  }

  /**
   * \brief solve problem
   * @return error code
   */
  MoFEMErrorCode solveLinearProblem() {

    MoFEMFunctionBeginHot;

    ierr = MatZeroEntries(Aij); CHKERRG(ierr);
    ierr = VecZeroEntries(F); CHKERRG(ierr);
    ierr = VecGhostUpdateBegin(F,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(F,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    ierr = VecZeroEntries(D0); CHKERRG(ierr);
    ierr = VecGhostUpdateBegin(D0,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(D0,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    ierr = VecZeroEntries(D); CHKERRG(ierr);
    ierr = VecGhostUpdateBegin(D,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(D,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);

    ierr = mField.getInterface<VecManager>()->setGlobalGhostVector("MIX",COL,D,INSERT_VALUES,SCATTER_REVERSE); CHKERRG(ierr);

    // Calculate essential boundary conditions

    // clear essential bc indices, it could have dofs from other mesh refinement
    bcIndices.clear();
    // clear operator, just in case if some other operators are left on this element
    feTri.getOpPtrVector().clear();
    // set operator to calculate essential boundary conditions
    feTri.getOpPtrVector().push_back(new OpEvaluateBcOnFluxes(*this,"FLUXES"));
    ierr = mField.loop_finite_elements("MIX","MIX_BCFLUX",feTri); CHKERRG(ierr);
    ierr = VecGhostUpdateBegin(D0,INSERT_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(D0,INSERT_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
    ierr = VecAssemblyBegin(D0); CHKERRG(ierr);
    ierr = VecAssemblyEnd(D0); CHKERRG(ierr);

    // set operators to calculate matrix and right hand side vectors
    feVol.getOpPtrVector().clear();
    feVol.getOpPtrVector().push_back(new OpL2Source(*this,"VALUES",F));
    feVol.getOpPtrVector().push_back(new OpFluxDivergenceAtGaussPts(*this,"FLUXES"));
    feVol.getOpPtrVector().push_back(new OpValuesAtGaussPts(*this,"VALUES"));
    feVol.getOpPtrVector().push_back(new OpDivTauU_HdivL2(*this,"FLUXES","VALUES",F));
    feVol.getOpPtrVector().push_back(new OpTauDotSigma_HdivHdiv(*this,"FLUXES",Aij,F));
    feVol.getOpPtrVector().push_back(new OpVDivSigma_L2Hdiv(*this,"VALUES","FLUXES",Aij,F));
    ierr = mField.loop_finite_elements("MIX","MIX",feVol); CHKERRG(ierr);

    // calculate right hand side for natural boundary conditions
    feTri.getOpPtrVector().clear();
    feTri.getOpPtrVector().push_back(new OpRhsBcOnValues(*this,"FLUXES",F));
    ierr = mField.loop_finite_elements("MIX","MIX_BCVALUE",feTri); CHKERRG(ierr);

    // assemble matrices
    ierr = MatAssemblyBegin(Aij,MAT_FINAL_ASSEMBLY); CHKERRG(ierr);
    ierr = MatAssemblyEnd(Aij,MAT_FINAL_ASSEMBLY); CHKERRG(ierr);
    ierr = VecGhostUpdateBegin(F,ADD_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(F,ADD_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
    ierr = VecAssemblyBegin(F); CHKERRG(ierr);
    ierr = VecAssemblyEnd(F); CHKERRG(ierr);

    {
      double nrm2_F;
      ierr = VecNorm(F,NORM_2,&nrm2_F); CHKERRG(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"nrm2_F = %6.4e\n",nrm2_F);
    }

    // ierr = MatMultAdd(Aij,D0,F,F); CHKERRG(ierr);

    // for ksp solver vector is moved into rhs side
    // for snes it is left ond the left
    ierr = VecScale(F,-1); CHKERRG(ierr);

    IS essential_bc_ids;
    ierr = getDirichletBCIndices(&essential_bc_ids); CHKERRG(ierr);
    ierr = MatZeroRowsColumnsIS(Aij,essential_bc_ids,1,D0,F); CHKERRG(ierr);
    ierr = ISDestroy(&essential_bc_ids); CHKERRG(ierr);

    // {
    //   double norm;
    //   ierr = MatNorm(Aij,NORM_FROBENIUS,&norm); CHKERRG(ierr);
    //   PetscPrintf(PETSC_COMM_WORLD,"mat norm = %6.4e\n",norm);
    // }

    {
      double nrm2_F;
      ierr = VecNorm(F,NORM_2,&nrm2_F); CHKERRG(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"With BC nrm2_F = %6.4e\n",nrm2_F);
    }

    //MatView(Aij,PETSC_VIEWER_DRAW_WORLD);
    //MatView(Aij,PETSC_VIEWER_STDOUT_WORLD);
    // std::string wait;
    //std::cin >> wait;

    // MatView(Aij,PETSC_VIEWER_DRAW_WORLD);
    // std::cin >> wait;

    // Solve
    KSP solver;
    ierr = KSPCreate(PETSC_COMM_WORLD,&solver); CHKERRG(ierr);
    ierr = KSPSetOperators(solver,Aij,Aij); CHKERRG(ierr);
    ierr = KSPSetFromOptions(solver); CHKERRG(ierr);
    ierr = KSPSetUp(solver); CHKERRG(ierr);
    ierr = KSPSolve(solver,F,D); CHKERRG(ierr);
    ierr = KSPDestroy(&solver); CHKERRG(ierr);

    {
      double nrm2_D;
      ierr = VecNorm(D,NORM_2,&nrm2_D); CHKERRG(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"nrm2_D = %6.4e\n",nrm2_D);
    }
    ierr = VecGhostUpdateBegin(D,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(D,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);


    // copy data form vector on mesh
    ierr = mField.getInterface<VecManager>()->setGlobalGhostVector("MIX",COL,D,INSERT_VALUES,SCATTER_REVERSE); CHKERRG(ierr);

    MoFEMFunctionReturnHot(0);
  }

  /// \brief calculate residual
  MoFEMErrorCode calculateResidual() {

    MoFEMFunctionBeginHot;
    ierr = VecZeroEntries(F); CHKERRG(ierr);
    ierr = VecGhostUpdateBegin(F,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(F,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    ierr = VecAssemblyBegin(F); CHKERRG(ierr);
    ierr = VecAssemblyEnd(F); CHKERRG(ierr);
    //calculate residuals
    feVol.getOpPtrVector().clear();
    feVol.getOpPtrVector().push_back(new OpL2Source(*this,"VALUES",F));
    feVol.getOpPtrVector().push_back(new OpFluxDivergenceAtGaussPts(*this,"FLUXES"));
    feVol.getOpPtrVector().push_back(new OpValuesAtGaussPts(*this,"VALUES"));
    feVol.getOpPtrVector().push_back(new OpDivTauU_HdivL2(*this,"FLUXES","VALUES",F));
    feVol.getOpPtrVector().push_back(new OpTauDotSigma_HdivHdiv(*this,"FLUXES",PETSC_NULL,F));
    feVol.getOpPtrVector().push_back(new OpVDivSigma_L2Hdiv(*this,"VALUES","FLUXES",PETSC_NULL,F));
    ierr = mField.loop_finite_elements("MIX","MIX",feVol); CHKERRG(ierr);
    feTri.getOpPtrVector().clear();
    feTri.getOpPtrVector().push_back(new OpRhsBcOnValues(*this,"FLUXES",F));
    ierr = mField.loop_finite_elements("MIX","MIX_BCVALUE",feTri); CHKERRG(ierr);
    ierr = VecGhostUpdateBegin(F,ADD_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(F,ADD_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
    ierr = VecAssemblyBegin(F); CHKERRG(ierr);
    ierr = VecAssemblyEnd(F); CHKERRG(ierr);
    // ierr = VecAXPY(F,-1.,D0); CHKERRG(ierr);
    // ierr = MatZeroRowsIS(Aij,essential_bc_ids,1,PETSC_NULL,F); CHKERRG(ierr);
    {
      std::vector<int> ids;
      ids.insert(ids.begin(),bcIndices.begin(),bcIndices.end());
      std::vector<double> vals(ids.size(),0);
      ierr = VecSetValues(F,ids.size(),&*ids.begin(),&*vals.begin(),INSERT_VALUES); CHKERRG(ierr);
      ierr = VecAssemblyBegin(F); CHKERRG(ierr);
      ierr = VecAssemblyEnd(F); CHKERRG(ierr);
    }
    {
      double nrm2_F;
      ierr = VecNorm(F,NORM_2,&nrm2_F); CHKERRG(ierr);
      PetscPrintf(PETSC_COMM_WORLD,"nrm2_F = %6.4e\n",nrm2_F);
      const double eps = 1e-8;
      if(nrm2_F > eps) {
        //SETERRQ(PETSC_COMM_SELF,MOFEM_ATOM_TEST_INVALID,"problem with residual");
      }
    }
    MoFEMFunctionReturnHot(0);
  }

  /** \brief Calculate error on elements

    \todo this functions runs serial, in future should be parallel and work on
    distributed meshes.

  */
  MoFEMErrorCode evaluateError() {

    MoFEMFunctionBeginHot;
    errorMap.clear();
    sumErrorFlux = 0;
    sumErrorDiv = 0;
    sumErrorJump = 0;
    feTri.getOpPtrVector().clear();
    feTri.getOpPtrVector().push_back(new OpSkeleton(*this,"FLUXES"));
    ierr = mField.loop_finite_elements("MIX","MIX_SKELETON",feTri,0,mField.get_comm_size()); CHKERRG(ierr);
    feVol.getOpPtrVector().clear();
    feVol.getOpPtrVector().push_back(new OpFluxDivergenceAtGaussPts(*this,"FLUXES"));
    feVol.getOpPtrVector().push_back(new OpValuesGradientAtGaussPts(*this,"VALUES"));
    feVol.getOpPtrVector().push_back(new OpError(*this,"VALUES"));
    ierr = mField.loop_finite_elements("MIX","MIX",feVol,0,mField.get_comm_size()); CHKERRG(ierr);
    const Problem *problem_ptr;
    ierr = mField.get_problem("MIX",&problem_ptr); CHKERRG(ierr);
    PetscPrintf(
      mField.get_comm(),
      "Nb dofs %d error flux^2 = %6.4e error div^2 = %6.4e error jump^2 = %6.4e error tot^2 = %6.4e\n",
      problem_ptr->getNbDofsRow(),
      sumErrorFlux,sumErrorDiv,sumErrorJump,sumErrorFlux+sumErrorDiv+sumErrorJump
    );
    MoFEMFunctionReturnHot(0);
  }

  /// \brief destroy matrices
  MoFEMErrorCode destroyMatrices() {
    MoFEMFunctionBeginHot;
    ierr = MatDestroy(&Aij); CHKERRG(ierr);
    ierr = VecDestroy(&D); CHKERRG(ierr);
    ierr = VecDestroy(&D0); CHKERRG(ierr);
    ierr = VecDestroy(&F); CHKERRG(ierr);
    MoFEMFunctionReturnHot(0);
  }


  /**
  \brief Assemble \f$\int_\mathcal{T} \mathbf{A} \boldsymbol\sigma \cdot \boldsymbol\tau \textrm{d}\mathcal{T}\f$

  \ingroup mofem_mix_transport_elem
  */
  struct OpTauDotSigma_HdivHdiv: public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    MixTransportElement &cTx;
    Mat Aij;
    Vec F;

    OpTauDotSigma_HdivHdiv(
      MixTransportElement &ctx,
      const std::string flux_name,
      Mat aij,Vec f
    ):
    MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
      flux_name,flux_name,
      UserDataOperator::OPROWCOL|UserDataOperator::OPCOL
    ),
    cTx(ctx),
    Aij(aij),
    F(f) {
      sYmm = true;
    }

    MatrixDouble NN,transNN;
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
    MoFEMErrorCode doWork(
      int row_side,int col_side,
      EntityType row_type,EntityType col_type,
      DataForcesAndSourcesCore::EntData &row_data,
      DataForcesAndSourcesCore::EntData &col_data
    ) {

      MoFEMFunctionBeginHot;
      try {
        if(Aij == PETSC_NULL) MoFEMFunctionReturnHot(0);
        if(row_data.getIndices().size()==0) MoFEMFunctionReturnHot(0);
        if(col_data.getIndices().size()==0) MoFEMFunctionReturnHot(0);
        EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
        int nb_row = row_data.getIndices().size();
        int nb_col = col_data.getIndices().size();
        NN.resize(nb_row,nb_col,false);
        NN.clear();
        FTensor::Index<'i',3> i;
        FTensor::Index<'j',3> j;
        invK.resize(3,3,false);
        // get access to resistivity data by tensor rank 2
        FTensor::Tensor2<double*,3,3> t_inv_k(
          &invK(0,0),&invK(0,1),&invK(0,2),
          &invK(1,0),&invK(1,1),&invK(1,2),
          &invK(2,0),&invK(2,1),&invK(2,2)
        );
        // Get base functions
        auto t_n_hdiv_row = row_data.getFTensor1HdivN<3>();
        FTensor::Tensor1<double,3> t_row;
        int nb_gauss_pts = row_data.getHdivN().size1();
        for(int gg = 0;gg!=nb_gauss_pts;gg++) {
          // get integration weight and multiply by element volume
          double w = getGaussPts()(3,gg)*getVolume();
          // in case that HO geometry is defined that below take into account that
          // edges of element are curved
          if(getHoGaussPtsDetJac().size()>0) {
            w *= getHoGaussPtsDetJac()(gg);
          }
          const double x = getCoordsAtGaussPts()(gg,0);
          const double y = getCoordsAtGaussPts()(gg,1);
          const double z = getCoordsAtGaussPts()(gg,2);
          // calculate receptivity (invers of conductivity)
          ierr = cTx.getResistivity(fe_ent,x,y,z,invK); CHKERRG(ierr);
          for(int kk = 0;kk!=nb_row;kk++) {
            FTensor::Tensor1<const double*,3> t_n_hdiv_col(
              &col_data.getHdivN(gg)(0,HDIV0),
              &col_data.getHdivN(gg)(0,HDIV1),
              &col_data.getHdivN(gg)(0,HDIV2),3
            );
            t_row(j) = w*t_n_hdiv_row(i)*t_inv_k(i,j);
            for(int ll = 0;ll!=nb_col;ll++) {
              NN(kk,ll) += t_row(j)*t_n_hdiv_col(j);
              ++t_n_hdiv_col;
            }
            ++t_n_hdiv_row;
          }
        }
        Mat a = (Aij!=PETSC_NULL) ? Aij : getFEMethod()->ts_B;
        ierr = MatSetValues(
          a,
          nb_row,&row_data.getIndices()[0],
          nb_col,&col_data.getIndices()[0],
          &NN(0,0),ADD_VALUES
        ); CHKERRG(ierr);
        // matrix is symmetric, assemble other part
        if(row_side != col_side || row_type != col_type) {
          transNN.resize(nb_col,nb_row);
          noalias(transNN) = trans(NN);
          ierr = MatSetValues(
            a,
            nb_col,&col_data.getIndices()[0],
            nb_row,&row_data.getIndices()[0],
            &transNN(0,0),ADD_VALUES
          ); CHKERRG(ierr);
        }
      } catch (const std::exception& ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,MOFEM_STD_EXCEPTION_THROW,ss.str().c_str());
      }

      MoFEMFunctionReturnHot(0);
    }

    /**
     * \brief Assemble matrix
     * @param  side local index of row entity on element
     * @param  type type of row entity, f.e. MBVERTEX, MBEDGE, or MBTET
     * @param  data data for row
     * @return          error code
     */
    MoFEMErrorCode doWork(
      int side,EntityType type,DataForcesAndSourcesCore::EntData &data
    ) {

      MoFEMFunctionBeginHot;
      try {
        if(F==PETSC_NULL) MoFEMFunctionReturnHot(0);
        int nb_row = data.getIndices().size();
        if(nb_row==0) MoFEMFunctionReturnHot(0);

        EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
        // cerr << data.getIndices() << endl;
        // cerr << data.getHdivN() << endl;
        Nf.resize(nb_row);
        Nf.clear();
        FTensor::Index<'i',3> i;
        FTensor::Index<'j',3> j;
        invK.resize(3,3,false);
        Nf.resize(nb_row);
        Nf.clear();
        // get access to resistivity data by tensor rank 2
        FTensor::Tensor2<double*,3,3> t_inv_k(
          &invK(0,0),&invK(0,1),&invK(0,2),
          &invK(1,0),&invK(1,1),&invK(1,2),
          &invK(2,0),&invK(2,1),&invK(2,2)
        );
        // get base functions
        auto t_n_hdiv = data.getFTensor1HdivN<3>();
        auto t_flux = getFTensor1FromMat<3>(cTx.fluxesAtGaussPts);
        int nb_gauss_pts = data.getHdivN().size1();
        for(int gg = 0;gg<nb_gauss_pts;gg++) {
          double w = getGaussPts()(3,gg)*getVolume();
          if(getHoGaussPtsDetJac().size()>0) {
            w *= getHoGaussPtsDetJac()(gg);
          }
          const double x = getCoordsAtGaussPts()(gg,0);
          const double y = getCoordsAtGaussPts()(gg,1);
          const double z = getCoordsAtGaussPts()(gg,2);
          ierr = cTx.getResistivity(fe_ent,x,y,z,invK); CHKERRG(ierr);
          for(int ll = 0;ll!=nb_row;ll++) {
            Nf[ll] += w*t_n_hdiv(i)*t_inv_k(i,j)*t_flux(j);
            ++t_n_hdiv;
          }
          ++t_flux;
        }
        ierr = VecSetValues(
          F,nb_row,&data.getIndices()[0],&Nf[0],ADD_VALUES
        ); CHKERRG(ierr);

      } catch (const std::exception& ex) {
        cerr << data.getFieldData() << endl;
        cerr << data.getIndices() << endl;
        cerr << data.getN() << endl;
        cerr << data.getDiffN() << endl;
        std::ostringstream ss;
        ss << "throw in method:"
        << " type: " << type
        << " side: " << side << " "
        << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,MOFEM_STD_EXCEPTION_THROW,ss.str().c_str());
      }
      MoFEMFunctionReturnHot(0);
    }

  };

  /** \brief Assemble \f$ \int_\mathcal{T} u \textrm{div}[\boldsymbol\tau] \textrm{d}\mathcal{T} \f$
    */
  struct OpDivTauU_HdivL2: public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    MixTransportElement &cTx;
    Vec F;

    OpDivTauU_HdivL2(
      MixTransportElement &ctx,
      const std::string flux_name_row,string val_name_col,Vec f
    ):
    MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
      flux_name_row,val_name_col,UserDataOperator::OPROW
    ),
    cTx(ctx),
    F(f) {
      //this operator is not symmetric setting this variable makes element
      //operator to loop over element entities (sub-simplices) without
      //assumption that off-diagonal matrices are symmetric.
      sYmm = false;
    }

    VectorDouble divVec,Nf;

    MoFEMErrorCode doWork(
      int side,EntityType type,
      DataForcesAndSourcesCore::EntData &data
    ) {

      MoFEMFunctionBeginHot;
      try {
        if(data.getFieldData().size()==0) MoFEMFunctionReturnHot(0);
        int nb_row = data.getIndices().size();
        Nf.resize(nb_row);
        Nf.clear();
        divVec.resize(data.getHdivN().size2()/3,0);
        if(divVec.size()!=data.getIndices().size()) {
          SETERRQ(PETSC_COMM_SELF,MOFEM_DATA_INCONSISTENCY,"data inconsistency");
        }
        int nb_gauss_pts = data.getN().size1();
        int gg = 0;
        for(;gg<nb_gauss_pts;gg++) {
          double w = getGaussPts()(3,gg)*getVolume();
          if(getHoGaussPtsDetJac().size()>0) {
            w *= getHoGaussPtsDetJac()(gg);
          }
          ierr = getDivergenceOfHDivBaseFunctions(side,type,data,gg,divVec); CHKERRG(ierr);
          noalias(Nf) -= w*divVec*cTx.valuesAtGaussPts[gg];
        }
        ierr = VecSetValues(
          F,nb_row,&data.getIndices()[0],
          &Nf[0],ADD_VALUES
        ); CHKERRG(ierr);
      } catch (const std::exception& ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,MOFEM_STD_EXCEPTION_THROW,ss.str().c_str());
      }
      MoFEMFunctionReturnHot(0);
    }

  };

  /** \brief \f$ \int_\mathcal{T} \textrm{div}[\boldsymbol\sigma] v \textrm{d}\mathcal{T} \f$
    */
  struct OpVDivSigma_L2Hdiv: public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    MixTransportElement &cTx;
    Mat Aij;
    Vec F;

    /**
     * \brief Constructor
     */
    OpVDivSigma_L2Hdiv(
      MixTransportElement &ctx,
      const std::string val_name_row,string flux_name_col,Mat aij,Vec f
    ):
    MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
      val_name_row,flux_name_col,
      UserDataOperator::OPROW|UserDataOperator::OPROWCOL
    ),
    cTx(ctx),
    Aij(aij),
    F(f) {

      //this operator is not symmetric setting this variable makes element
      //operator to loop over element entities without
      //assumption that off-diagonal matrices are symmetric.
      sYmm = false;

    }

    MatrixDouble NN,transNN;
    VectorDouble divVec,Nf;

    /**
     * \brief Do calculations
     * @param  row_side local index of entity on row
     * @param  col_side local index of entity on column
     * @param  row_type type of row entity
     * @param  col_type type of col entity
     * @param  row_data row data structure carrying information about base functions, DOFs indices, etc.
     * @param  col_data column data structure carrying information about base functions, DOFs indices, etc.
     * @return          error code
     */
    MoFEMErrorCode doWork(
      int row_side,int col_side,
      EntityType row_type,EntityType col_type,
      DataForcesAndSourcesCore::EntData &row_data,
      DataForcesAndSourcesCore::EntData &col_data
    ) {
      MoFEMFunctionBeginHot;
      try {
        if(Aij == PETSC_NULL) MoFEMFunctionReturnHot(0);
        if(row_data.getFieldData().size()==0) MoFEMFunctionReturnHot(0);
        if(col_data.getFieldData().size()==0) MoFEMFunctionReturnHot(0);
        int nb_row = row_data.getFieldData().size();
        int nb_col = col_data.getFieldData().size();
        NN.resize(nb_row,nb_col);
        NN.clear();
        divVec.resize(nb_col,false);
        int nb_gauss_pts = row_data.getHdivN().size1();
        for(int gg = 0;gg<nb_gauss_pts;gg++) {
          double w = getGaussPts()(3,gg)*getVolume();
          if(getHoGaussPtsDetJac().size()>0) {
            w *= getHoGaussPtsDetJac()(gg);
          }
          ierr = getDivergenceOfHDivBaseFunctions(
            col_side,col_type,col_data,gg,divVec
          ); CHKERRG(ierr);
          noalias(NN) += w*outer_prod(row_data.getN(gg),divVec);
        }
        ierr = MatSetValues(
          Aij,
          nb_row,&row_data.getIndices()[0],
          nb_col,&col_data.getIndices()[0],
          &NN(0,0),ADD_VALUES
        ); CHKERRG(ierr);
        transNN.resize(nb_col,nb_row);
        ublas::noalias(transNN) = -trans(NN);
        ierr = MatSetValues(
          Aij,
          nb_col,&col_data.getIndices()[0],
          nb_row,&row_data.getIndices()[0],
          &transNN(0,0),ADD_VALUES
        ); CHKERRG(ierr);
      } catch (const std::exception& ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,MOFEM_STD_EXCEPTION_THROW,ss.str().c_str());
      }
      MoFEMFunctionReturnHot(0);
    }

    MoFEMErrorCode doWork(
      int side,EntityType type,
      DataForcesAndSourcesCore::EntData &data
    ) {

      MoFEMFunctionBeginHot;
      try {
        if(data.getIndices().size()==0) MoFEMFunctionReturnHot(0);
        if(data.getIndices().size()!=data.getN().size2()) {
          SETERRQ(PETSC_COMM_SELF,MOFEM_DATA_INCONSISTENCY,"data inconsistency");
        }
        int nb_row = data.getIndices().size();
        Nf.resize(nb_row);
        Nf.clear();
        int nb_gauss_pts = data.getN().size1();
        for(int gg = 0;gg<nb_gauss_pts;gg++) {
          double w = getGaussPts()(3,gg)*getVolume();
          if(getHoGaussPtsDetJac().size()>0) {
            w *= getHoGaussPtsDetJac()(gg);
          }
          noalias(Nf) += w*data.getN(gg)*cTx.divergenceAtGaussPts[gg];
        }
        ierr = VecSetValues(
          F,nb_row,&data.getIndices()[0],
          &Nf[0],ADD_VALUES
        ); CHKERRG(ierr);
      } catch (const std::exception& ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
      }
      MoFEMFunctionReturnHot(0);
    }

  };

  /** \brief Calculate source therms, i.e. \f$\int_\mathcal{T} f v \textrm{d}\mathcal{T}\f$
  */
  struct OpL2Source: public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    MixTransportElement &cTx;
    Vec F;

    OpL2Source(
      MixTransportElement &ctx,
      const std::string val_name,
      Vec f
    ):
    MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(val_name,UserDataOperator::OPROW),
    cTx(ctx),
    F(f) {}

    VectorDouble Nf;
    MoFEMErrorCode doWork(
      int side,EntityType type,DataForcesAndSourcesCore::EntData &data
    ) {

      MoFEMFunctionBeginHot;
      try {
        if(data.getFieldData().size()==0) MoFEMFunctionReturnHot(0);
        EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
        int nb_row = data.getFieldData().size();
        Nf.resize(nb_row,false);
        Nf.clear();
        int nb_gauss_pts = data.getHdivN().size1();
        for(int gg = 0;gg<nb_gauss_pts;gg++) {
          double w = getGaussPts()(3,gg)*getVolume();
          if(getHoGaussPtsDetJac().size()>0) {
            w *= getHoGaussPtsDetJac()(gg);
          }
          double x,y,z;
          if(getHoCoordsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
            x = getHoCoordsAtGaussPts()(gg,0);
            y = getHoCoordsAtGaussPts()(gg,1);
            z = getHoCoordsAtGaussPts()(gg,2);
          } else {
            x = getCoordsAtGaussPts()(gg,0);
            y = getCoordsAtGaussPts()(gg,1);
            z = getCoordsAtGaussPts()(gg,2);
         }
          double flux = 0;
          ierr = cTx.getSource(fe_ent,x,y,z,flux); CHKERRG(ierr);
          noalias(Nf) += w*data.getN(gg)*flux;
        }
        ierr = VecSetValues(F,nb_row,&data.getIndices()[0],&Nf[0],ADD_VALUES); CHKERRG(ierr);
      } catch (const std::exception& ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
      }

      MoFEMFunctionReturnHot(0);
    }

  };

  /**
   * \brief calculate \f$ \int_\mathcal{S} {\boldsymbol\tau} \cdot \mathbf{n}u \textrm{d}\mathcal{S} \f$

   * This terms comes from differentiation by parts. Note that in this Dirichlet
   * boundary conditions are natural.

   */
  struct OpRhsBcOnValues: public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    MixTransportElement &cTx;
    Vec F;

    /**
     * \brief Constructor
     */
    OpRhsBcOnValues(
      MixTransportElement &ctx,const std::string fluxes_name,Vec f
    ):
    MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(fluxes_name,UserDataOperator::OPROW),
    cTx(ctx),
    F(f) {}

    VectorDouble nF;

    /**
     * \brief Integrate boundary condition
     * @param  side local index of entity
     * @param  type type of entity
     * @param  data data on entity
     * @return      error code
     */
    MoFEMErrorCode doWork(
      int side,EntityType type,DataForcesAndSourcesCore::EntData &data
    ) {
      MoFEMFunctionBeginHot;
      try {
        if(data.getFieldData().size()==0) MoFEMFunctionReturnHot(0);
        EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
        nF.resize(data.getIndices().size());
        nF.clear();
        int nb_gauss_pts = data.getHdivN().size1();
        for(int gg = 0;gg<nb_gauss_pts;gg++) {
          double x,y,z;
          if(getNormalsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
            x = getHoCoordsAtGaussPts()(gg,0);
            y = getHoCoordsAtGaussPts()(gg,1);
            z = getHoCoordsAtGaussPts()(gg,2);
          } else {
            x = getCoordsAtGaussPts()(gg,0);
            y = getCoordsAtGaussPts()(gg,1);
            z = getCoordsAtGaussPts()(gg,2);
         }
          double value;
          ierr = cTx.getBcOnValues(fe_ent,gg,x,y,z,value); CHKERRG(ierr);
          double w = getGaussPts()(2,gg)*0.5;
          if(getNormalsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
            noalias(nF) += w*prod(data.getHdivN(gg),getNormalsAtGaussPts(gg))*value;
          } else {
            noalias(nF) += w*prod(data.getHdivN(gg),getNormal())*value;
          }
        }
        Vec f = (F!=PETSC_NULL) ? F : getFEMethod()->ts_F;
        ierr = VecSetValues(
          f,data.getIndices().size(),&data.getIndices()[0],&nF[0],ADD_VALUES
        ); CHKERRG(ierr);
      } catch (const std::exception& ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,MOFEM_STD_EXCEPTION_THROW,ss.str().c_str());
      }
      MoFEMFunctionReturnHot(0);
    }
  };

  /**
   * \brief Evaluate boundary conditions on fluxes.
   *
   * Note that Neumann boundary conditions here are essential. So it is opposite
   * what you find in displacement finite element method.
   *

   * Here we have to solve for degrees of freedom on boundary such base functions
   * approximate flux.

   *
   */
  struct OpEvaluateBcOnFluxes: public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {
    MixTransportElement &cTx;
    OpEvaluateBcOnFluxes(
      MixTransportElement &ctx,const std::string flux_name
    ):
    MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(flux_name,UserDataOperator::OPROW),
    cTx(ctx) {
    }

    MatrixDouble NN;
    VectorDouble Nf;
    FTensor::Index<'i',3> i;

    MoFEMErrorCode doWork(int side,EntityType type,DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBeginHot;
      try {
        if(data.getFieldData().size()==0) MoFEMFunctionReturnHot(0);
        EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
        int nb_dofs = data.getFieldData().size();
        int nb_gauss_pts = data.getHdivN().size1();
        if(3*nb_dofs!=static_cast<int>(data.getHdivN().size2())) {
          SETERRQ(PETSC_COMM_WORLD,MOFEM_DATA_INCONSISTENCY,"wrong number of dofs");
        }
        NN.resize(nb_dofs,nb_dofs);
        Nf.resize(nb_dofs);
        NN.clear();
        Nf.clear();

        // Get normal vector. Note that when higher order geometry is set, then
        // face element could be curved, i.e. normal can be different at each integration
        // point.
        double *normal_ptr;
        if(getNormalsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
          // HO geometry
          normal_ptr = &getNormalsAtGaussPts(0)[0];
        } else {
          // Linear geometry, i.e. constant normal on face
          normal_ptr = &getNormal()[0];
        }
        // set tensor from pointer
        FTensor::Tensor1<const double*,3> t_normal(normal_ptr,&normal_ptr[1],&normal_ptr[2],3);
        // get base functions
        auto t_n_hdiv_row = data.getFTensor1HdivN<3>();

        double nrm2 = 0;

        // loop over integration points
        for(int gg = 0;gg<nb_gauss_pts;gg++) {

          // get integration point coordinates
          double x,y,z;
          if(getNormalsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
            x = getHoCoordsAtGaussPts()(gg,0);
            y = getHoCoordsAtGaussPts()(gg,1);
            z = getHoCoordsAtGaussPts()(gg,2);
          } else {
            x = getCoordsAtGaussPts()(gg,0);
            y = getCoordsAtGaussPts()(gg,1);
            z = getCoordsAtGaussPts()(gg,2);
          }

          // get flux on fece for given element handle and coordinates
          double flux;
          ierr = cTx.getBcOnFluxes(fe_ent,x,y,z,flux); CHKERRG(ierr);
          // get weight for integration rule
          double w = getGaussPts()(2,gg);
          if(gg == 0) {
            nrm2  = sqrt(t_normal(i)*t_normal(i));
          }

          // set tensor of rank 0 to matrix NN elements
          // loop over base functions on rows and columns
          for(int ll = 0;ll!=nb_dofs;ll++) {
            // get column on shape functions
            FTensor::Tensor1<const double*,3> t_n_hdiv_col(
              &data.getHdivN(gg)(0,HDIV0),
              &data.getHdivN(gg)(0,HDIV1),
              &data.getHdivN(gg)(0,HDIV2),3
            );
            for(int kk = 0;kk<=ll;kk++) {
              NN(ll,kk) += w*t_n_hdiv_row(i)*t_n_hdiv_col(i);
              ++t_n_hdiv_col;
            }
            // right hand side
            Nf[ll] += w*t_n_hdiv_row(i)*t_normal(i)*flux/nrm2;
            ++t_n_hdiv_row;
          }

          // If HO geometry increment t_normal to next integration point
          if(getNormalsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
            ++t_normal;
            nrm2  = sqrt(t_normal(i)*t_normal(i));
          }

        }
        // get global dofs indices on element
        cTx.bcIndices.insert(data.getIndices().begin(),data.getIndices().end());
        // factor matrix
        cholesky_decompose(NN);
        // solve local problem
        cholesky_solve(NN,Nf,ublas::lower());

        // cerr << Nf << endl;
        // cerr << data.getIndices() << endl;

        // set solution to vector
        ierr = VecSetValues(
          cTx.D0,nb_dofs,&*data.getIndices().begin(),
          &*Nf.begin(),INSERT_VALUES
        ); CHKERRG(ierr);

      } catch (const std::exception& ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
      }

      MoFEMFunctionReturnHot(0);
    }

  };

  /**
   * \brief Calculate values at integration points
   */
  struct OpValuesAtGaussPts: public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    MixTransportElement &cTx;

    OpValuesAtGaussPts(
      MixTransportElement &ctx,
      const std::string val_name = "VALUES"
    ):
    MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(val_name,UserDataOperator::OPROW),
    cTx(ctx) {}

    virtual ~OpValuesAtGaussPts() {}

    MoFEMErrorCode doWork(int side,EntityType type,DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBeginHot;

      try {

        if(data.getFieldData().size() == 0)  MoFEMFunctionReturnHot(0);

        int nb_gauss_pts = data.getN().size1();
        cTx.valuesAtGaussPts.resize(nb_gauss_pts);
        for(int gg = 0;gg<nb_gauss_pts;gg++) {
          cTx.valuesAtGaussPts[gg] = inner_prod( trans(data.getN(gg)), data.getFieldData() );
        }

      } catch (const std::exception& ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
      }

      MoFEMFunctionReturnHot(0);
    }

  };

  /**
   * \brief Calculate gradients of values at integration points
   */
  struct OpValuesGradientAtGaussPts: public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    MixTransportElement &cTx;

    OpValuesGradientAtGaussPts(
      MixTransportElement &ctx,
      const std::string val_name = "VALUES"
    ):
    MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(val_name,UserDataOperator::OPROW),
    cTx(ctx) {}
    virtual ~OpValuesGradientAtGaussPts() {}

    MoFEMErrorCode doWork(int side,EntityType type,DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBeginHot;
      try {
        if(data.getFieldData().size() == 0)  MoFEMFunctionReturnHot(0);
        int nb_gauss_pts = data.getDiffN().size1();
        // cerr << data.getDiffN() << endl;
        cTx.valuesGradientAtGaussPts.resize(3,nb_gauss_pts);
        for(int gg = 0;gg<nb_gauss_pts;gg++) {
          ublas::matrix_column<MatrixDouble> values_grad_at_gauss_pts(cTx.valuesGradientAtGaussPts,gg);
          noalias(values_grad_at_gauss_pts) = prod( trans(data.getDiffN(gg)), data.getFieldData() );
        }
      } catch (const std::exception& ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
      }

      MoFEMFunctionReturnHot(0);
    }

  };

  /**
   * \brief calculate flux at integration point
   */
  struct OpFluxDivergenceAtGaussPts: public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    MixTransportElement &cTx;

    OpFluxDivergenceAtGaussPts(
      MixTransportElement &ctx,
      const std::string field_name
    ):
    MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(field_name,UserDataOperator::OPROW),
    cTx(ctx) {}

    VectorDouble divVec;
    MoFEMErrorCode doWork(int side,EntityType type,DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBeginHot;

      try {
        if(data.getFieldData().size() == 0)  MoFEMFunctionReturnHot(0);
        int nb_gauss_pts = data.getDiffN().size1();
        int nb_dofs = data.getFieldData().size();
        cTx.fluxesAtGaussPts.resize(3,nb_gauss_pts);
        cTx.divergenceAtGaussPts.resize(nb_gauss_pts);
        if(type == MBTRI && side == 0) {
          cTx.divergenceAtGaussPts.clear();
          cTx.fluxesAtGaussPts.clear();
        }
        divVec.resize(nb_dofs);
        for(int gg = 0;gg<nb_gauss_pts;gg++) {
          ierr = getDivergenceOfHDivBaseFunctions(side,type,data,gg,divVec); CHKERRG(ierr);
          cTx.divergenceAtGaussPts[gg] += inner_prod(divVec,data.getFieldData());
          ublas::matrix_column<MatrixDouble> flux_at_gauss_pt(cTx.fluxesAtGaussPts,gg);
          flux_at_gauss_pt += prod(trans(data.getHdivN(gg)),data.getFieldData());
        }
      } catch (const std::exception& ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
      }
      MoFEMFunctionReturnHot(0);
    }

  };

  map<double,EntityHandle> errorMap;
  double sumErrorFlux;
  double sumErrorDiv;
  double sumErrorJump;

  /** \brief calculate error evaluator
    */
  struct OpError: public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    MixTransportElement &cTx;

    OpError(
      MixTransportElement &ctx,
      const std::string field_name):
      MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(field_name,UserDataOperator::OPROW),
      cTx(ctx) {}
    virtual ~OpError() {}

    VectorDouble deltaFlux;
    MatrixDouble3by3 invK;

    MoFEMErrorCode doWork(
      int side,EntityType type,DataForcesAndSourcesCore::EntData &data
    ) {


      MoFEMFunctionBeginHot;
      try {
        if(type != MBTET) MoFEMFunctionReturnHot(0);
        invK.resize(3,3,false);
        int nb_gauss_pts = data.getN().size1();
        EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();
        double def_val = 0;
        Tag th_error_flux,th_error_div;
        rval = cTx.mField.get_moab().tag_get_handle(
          "ERROR_FLUX",1,MB_TYPE_DOUBLE,th_error_flux,MB_TAG_CREAT|MB_TAG_SPARSE,&def_val
        ); CHKERRG(rval);
        double* error_flux_ptr;
        rval = cTx.mField.get_moab().tag_get_by_ptr(
          th_error_flux,&fe_ent,1,(const void**)&error_flux_ptr
        ); CHKERRG(rval);

        rval = cTx.mField.get_moab().tag_get_handle(
          "ERROR_DIV",1,MB_TYPE_DOUBLE,th_error_div,MB_TAG_CREAT|MB_TAG_SPARSE,&def_val
        ); CHKERRG(rval);
        double* error_div_ptr;
        rval = cTx.mField.get_moab().tag_get_by_ptr(
          th_error_div,&fe_ent,1,(const void**)&error_div_ptr
        ); CHKERRG(rval);

        Tag th_error_jump;
        rval = cTx.mField.get_moab().tag_get_handle(
          "ERROR_JUMP",1,MB_TYPE_DOUBLE,th_error_jump,MB_TAG_CREAT|MB_TAG_SPARSE,&def_val
        ); CHKERRG(rval);
        double* error_jump_ptr;
        rval = cTx.mField.get_moab().tag_get_by_ptr(
          th_error_jump,&fe_ent,1,(const void**)&error_jump_ptr
        ); CHKERRG(rval);
        *error_jump_ptr = 0;

        /// characteristic size of the element
        const double h = pow(getVolume()*12/sqrt(2),(double)1/3);

        for(int ff = 0;ff!=4;ff++) {
          EntityHandle face;
          rval = cTx.mField.get_moab().side_element(fe_ent,2,ff,face); CHKERRG(rval);
          double* error_face_jump_ptr;
          rval = cTx.mField.get_moab().tag_get_by_ptr(
            th_error_jump,&face,1,(const void**)&error_face_jump_ptr
          ); CHKERRG(rval);
          *error_face_jump_ptr = (1/sqrt(h))*sqrt(*error_face_jump_ptr);
          *error_face_jump_ptr = pow(*error_face_jump_ptr,2);
          *error_jump_ptr += *error_face_jump_ptr;
        }

        *error_flux_ptr = 0;
        *error_div_ptr = 0;
        deltaFlux.resize(3,false);
        for(int gg = 0;gg<nb_gauss_pts;gg++) {
          double w = getGaussPts()(3,gg)*getVolume();
          if(getHoGaussPtsDetJac().size()>0) {
            w *= getHoGaussPtsDetJac()(gg);
          }
          double x,y,z;
          if(getHoCoordsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
            x = getHoCoordsAtGaussPts()(gg,0);
            y = getHoCoordsAtGaussPts()(gg,1);
            z = getHoCoordsAtGaussPts()(gg,2);
          } else {
            x = getCoordsAtGaussPts()(gg,0);
            y = getCoordsAtGaussPts()(gg,1);
            z = getCoordsAtGaussPts()(gg,2);
          }
          double flux;
          ierr = cTx.getSource(fe_ent,x,y,z,flux); CHKERRG(ierr);
          *error_div_ptr += w*pow(cTx.divergenceAtGaussPts[gg]-flux,2);
          ierr = cTx.getResistivity(fe_ent,x,y,z,invK); CHKERRG(ierr);
          ublas::matrix_column<MatrixDouble> flux_at_gauss_pt(cTx.fluxesAtGaussPts,gg);
          ublas::matrix_column<MatrixDouble> values_grad_at_gauss_pts(cTx.valuesGradientAtGaussPts,gg);
          noalias(deltaFlux) = prod(invK,flux_at_gauss_pt)+values_grad_at_gauss_pts;
          *error_flux_ptr += w*inner_prod(deltaFlux,deltaFlux);
        }
        *error_div_ptr = h*sqrt(*error_div_ptr);
        *error_div_ptr = pow(*error_div_ptr,2);
        cTx.errorMap[sqrt(*error_flux_ptr+*error_div_ptr+*error_jump_ptr)] = fe_ent;
        // Sum/Integrate all errors
        cTx.sumErrorFlux += *error_flux_ptr*getVolume();
        cTx.sumErrorDiv += *error_div_ptr*getVolume();
        // FIXME: Summation should be while skeleton is calculated
        cTx.sumErrorJump += *error_jump_ptr*getVolume(); /// FIXME: this need to be fixed
      } catch (const std::exception& ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
      }

      MoFEMFunctionReturnHot(0);
    }

  };

  /**
   * \brief calculate jump on entities
   */
  struct OpSkeleton: public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    /**
     * \brief volume element to get values from adjacent tets to face
     */
    VolumeElementForcesAndSourcesCoreOnSide volSideFe;

    /** store values at integration point, key of the map is sense of face in
     * respect to adjacent tetrahedra
     */
    map<int,VectorDouble> valMap;

    /**
     * \brief calculate values on adjacent tetrahedra to face
     */
    struct OpVolSide: public VolumeElementForcesAndSourcesCoreOnSide::UserDataOperator {
      map<int,VectorDouble> &valMap;
      OpVolSide(map<int,VectorDouble> &val_map):
      VolumeElementForcesAndSourcesCoreOnSide::UserDataOperator("VALUES",UserDataOperator::OPROW),
      valMap(val_map) {
      }
      MoFEMErrorCode doWork(int side, EntityType type,DataForcesAndSourcesCore::EntData &data) {
        MoFEMFunctionBeginHot;
        try {
          if(data.getFieldData().size() == 0)  MoFEMFunctionReturnHot(0);
          int nb_gauss_pts = data.getN().size1();
          valMap[getFaceSense()].resize(nb_gauss_pts);
          for(int gg = 0;gg<nb_gauss_pts;gg++) {
            valMap[getFaceSense()][gg] = inner_prod(trans(data.getN(gg)),data.getFieldData());
          }
        } catch (const std::exception& ex) {
          std::ostringstream ss;
          ss << "throw in method: " << ex.what() << std::endl;
          SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
        }
        MoFEMFunctionReturnHot(0);
      }
    };

    MixTransportElement &cTx;

    OpSkeleton(
      MixTransportElement &ctx,const std::string flux_name
    ):
    MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(flux_name,UserDataOperator::OPROW),
    volSideFe(ctx.mField),
    cTx(ctx) {
      volSideFe.getOpPtrVector().push_back(new OpSkeleton::OpVolSide(valMap));
    }

    MoFEMErrorCode doWork(int side,EntityType type,DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBeginHot;
      try {

        if(type == MBTRI) {
          EntityHandle fe_ent = getNumeredEntFiniteElementPtr()->getEnt();

          double def_val = 0;
          Tag th_error_jump;
          rval = cTx.mField.get_moab().tag_get_handle(
            "ERROR_JUMP",1,MB_TYPE_DOUBLE,th_error_jump,MB_TAG_CREAT|MB_TAG_SPARSE,&def_val
          ); CHKERRG(rval);
          double* error_jump_ptr;
          rval = cTx.mField.get_moab().tag_get_by_ptr(
            th_error_jump,&fe_ent,1,(const void**)&error_jump_ptr
          ); CHKERRG(rval);
          *error_jump_ptr = 0;

          // check if this is essential boundary condition
          EntityHandle essential_bc_meshset = cTx.mField.get_finite_element_meshset("MIX_BCFLUX");
          if(cTx.mField.get_moab().contains_entities(essential_bc_meshset,&fe_ent,1)) {
            // essential bc, np jump then, exit and go to next face
            MoFEMFunctionReturnHot(0);
          }

          // calculate values form adjacent tets
          valMap.clear();
          ierr = loopSideVolumes("MIX",volSideFe); CHKERRG(ierr);

          int nb_gauss_pts = data.getHdivN().size1();

          // it is only one face, so it has to be bc natural boundary condition
          if(valMap.size()==1) {
            if(static_cast<int>(valMap.begin()->second.size())!=nb_gauss_pts) {
              SETERRQ(PETSC_COMM_WORLD,MOFEM_DATA_INCONSISTENCY,"wrong number of integration points");
            }
            for(int gg = 0;gg!=nb_gauss_pts;gg++) {
              double x,y,z;
              if(static_cast<int>(getNormalsAtGaussPts().size1()) == nb_gauss_pts) {
                x = getHoCoordsAtGaussPts()(gg,0);
                y = getHoCoordsAtGaussPts()(gg,1);
                z = getHoCoordsAtGaussPts()(gg,2);
              } else {
                x = getCoordsAtGaussPts()(gg,0);
                y = getCoordsAtGaussPts()(gg,1);
                z = getCoordsAtGaussPts()(gg,2);
              }
              double value;
              ierr = cTx.getBcOnValues(fe_ent,gg,x,y,z,value); CHKERRG(ierr);
              double w = getGaussPts()(2,gg);
              if(static_cast<int>(getNormalsAtGaussPts().size1()) == nb_gauss_pts) {
                w *= norm_2(getNormalsAtGaussPts(gg))*0.5;
              } else {
                w *= getArea();
              }
              *error_jump_ptr += w*pow(value-valMap.begin()->second[gg],2);
            }
          } else if(valMap.size()==2) {
            for(int gg = 0;gg!=nb_gauss_pts;gg++) {
              double w = getGaussPts()(2,gg);
              if(getNormalsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
                w *= norm_2(getNormalsAtGaussPts(gg))*0.5;
              } else {
                w *= getArea();
              }
              double delta = valMap.at(1)[gg]-valMap.at(-1)[gg];
              *error_jump_ptr += w*pow(delta,2);
            }
          } else {
            SETERRQ1(
              PETSC_COMM_WORLD,MOFEM_DATA_INCONSISTENCY,
              "data inconsistency, wrong number of neighbors valMap.size() = %d",
              valMap.size()
            );
          }
        }

      } catch (const std::out_of_range& ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,MOFEM_STD_EXCEPTION_THROW,ss.str().c_str());
      } catch (const std::exception& ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF,MOFEM_STD_EXCEPTION_THROW,ss.str().c_str());
      }

      MoFEMFunctionReturnHot(0);
    }

  };

};

}

#endif //_MIX_TRANPORT_ELEMENT_HPP_

/***************************************************************************//**
 * \defgroup mofem_mix_transport_elem Mix transport element
 * \ingroup user_modules
 ******************************************************************************/
