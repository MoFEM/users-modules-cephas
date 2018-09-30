/** \file PostProcOnRefMesh.hpp
 * \brief Post-process fields on refined mesh
 *
 * Create refined mesh, without enforcing continuity between element. Calculate
 * field values on nodes of that mesh.
 *
 * \ingroup mofem_fs_post_proc
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

#ifndef __POSTPROC_ON_REF_MESH_HPP
#define __POSTPROC_ON_REF_MESH_HPP

/** \brief Set of operators and data structures used for post-processing

This set of functions works that for given problem a new MoAB instance is crated
only used for post-processing. For each post-processed element  in the problem
an element entity in post-processing mesh is created. Post-processed elements do
not share nodes and any others entities between them, such that discontinuities
between element could be shown.

Post-processed entities could be represented by ho-elements, for example 10 node
tetrahedrons. Moreover each element could be refined such that higher order
polynomials  could be well represented.

* \ingroup mofem_fs_post_proc

*/
struct PostProcCommonOnRefMesh {

  struct CommonData {
    std::map<std::string, std::vector<VectorDouble>> fieldMap;
    std::map<std::string, std::vector<MatrixDouble>> gradMap;
  };

  struct CommonDataForVolume : CommonData {
    Range tEts;
  };

  /**
   * \brief operator to post-process (save gradients on refined post-processing
   * mesh) field gradient \ingroup mofem_fs_post_proc
   *
   * \todo Implamentation of setting values to fieldMap for Hcurl and Hdiv not
   * implemented
   *
   */
  struct OpGetFieldValues
      : public MoFEM::ForcesAndSourcesCore::UserDataOperator {

    moab::Interface &postProcMesh;
    std::vector<EntityHandle> &mapGaussPts;
    CommonData &commonData;
    const std::string tagName;
    Vec V;

    OpGetFieldValues(moab::Interface &post_proc_mesh,
                     std::vector<EntityHandle> &map_gauss_pts,
                     const std::string field_name, const std::string tag_name,
                     CommonData &common_data, Vec v = PETSC_NULL)
        : MoFEM::ForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL),
          postProcMesh(post_proc_mesh), mapGaussPts(map_gauss_pts),
          commonData(common_data), tagName(tag_name), V(v) {}

    VectorDouble vAlues;
    VectorDouble *vAluesPtr;

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /**
   * \brief operator to post-process (save gradients on refined post-processing
   * mesh) field gradient \ingroup mofem_fs_post_proc
   *
   * \todo Implementation for Hdiv and Hcurl to be implemented
   *
   */
  struct OpGetFieldGradientValues
      : public MoFEM::ForcesAndSourcesCore::UserDataOperator {

    moab::Interface &postProcMesh;
    std::vector<EntityHandle> &mapGaussPts;
    CommonData &commonData;
    const std::string tagName;
    Vec V;

    OpGetFieldGradientValues(moab::Interface &post_proc_mesh,
                             std::vector<EntityHandle> &map_gauss_pts,
                             const std::string field_name,
                             const std::string tag_name,
                             CommonData &common_data, Vec v = PETSC_NULL)
        : MoFEM::ForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL),
          postProcMesh(post_proc_mesh), mapGaussPts(map_gauss_pts),
          commonData(common_data), tagName(tag_name), V(v) {}

    VectorDouble vAlues;
    VectorDouble *vAluesPtr;

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };
};

/**
 * \brief Generic post-processing class
 *
 * Generate refined mesh and save data on vertices
 *
 * \ingroup mofem_fs_post_proc
 */
template <class ELEMENT> struct PostProcTemplateOnRefineMesh : public ELEMENT {

  moab::Core coreMesh;
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> mapGaussPts;

  PostProcTemplateOnRefineMesh(MoFEM::Interface &m_field)
      : ELEMENT(m_field), postProcMesh(coreMesh) {}

  virtual PostProcCommonOnRefMesh::CommonData &getCommonData() {
    THROW_MESSAGE("not implemented");
  }

  /** \brief Add operator to post-process L2, H1, Hdiv, Hcurl field value

    \param field_name
    \param v If vector is given, values from vector are used to set tags on mesh

    Note:
    Name of the tag to store values on post-process mesh is the same as field
    name

    * \ingroup mofem_fs_post_proc

  */
  MoFEMErrorCode addFieldValuesPostProc(const std::string field_name,
                                        Vec v = PETSC_NULL) {
    MoFEMFunctionBeginHot;
    ELEMENT::getOpPtrVector().push_back(
        new PostProcCommonOnRefMesh::OpGetFieldValues(postProcMesh, mapGaussPts,
                                                      field_name, field_name,
                                                      getCommonData(), v));
    MoFEMFunctionReturnHot(0);
  }

  /** \brief Add operator to post-process L2 or H1 field value

    \param field_name
    \param tag_name to store results on post-process mesh
    \param v If vector is given, values from vector are used to set tags on mesh

    * \ingroup mofem_fs_post_proc

  */
  MoFEMErrorCode addFieldValuesPostProc(const std::string field_name,
                                        const std::string tag_name,
                                        Vec v = PETSC_NULL) {
    MoFEMFunctionBeginHot;
    ELEMENT::getOpPtrVector().push_back(
        new PostProcCommonOnRefMesh::OpGetFieldValues(postProcMesh, mapGaussPts,
                                                      field_name, tag_name,
                                                      getCommonData(), v));
    MoFEMFunctionReturnHot(0);
  }

  /** \brief Add operator to post-process L2 or H1 field gradient

    \param field_name
    \param v If vector is given, values from vector are used to set tags on mesh

    Note:
    Name of the tag to store values on post-process mesh is the same as field
    name

    * \ingroup mofem_fs_post_proc

  */
  MoFEMErrorCode addFieldValuesGradientPostProc(const std::string field_name,
                                                Vec v = PETSC_NULL) {
    MoFEMFunctionBeginHot;
    ELEMENT::getOpPtrVector().push_back(
        new PostProcCommonOnRefMesh::OpGetFieldGradientValues(
            postProcMesh, mapGaussPts, field_name, field_name + "_GRAD",
            getCommonData(), v));
    MoFEMFunctionReturnHot(0);
  }

  /** \brief Add operator to post-process L2 or H1 field gradient

    \param field_name
    \param tag_name to store results on post-process mesh
    \param v If vector is given, values from vector are used to set tags on mesh

    * \ingroup mofem_fs_post_proc

  */
  MoFEMErrorCode addFieldValuesGradientPostProc(const std::string field_name,
                                                const std::string tag_name,
                                                Vec v = PETSC_NULL) {
    MoFEMFunctionBeginHot;
    ELEMENT::getOpPtrVector().push_back(
        new PostProcCommonOnRefMesh::OpGetFieldGradientValues(
            postProcMesh, mapGaussPts, field_name, tag_name, getCommonData(),
            v));
    MoFEMFunctionReturnHot(0);
  }

  /**
   * \brief wrote results in (MOAB) format, use "file_name.h5m"
   * @param  file_name file name (should always end with .h5m)
   * @return           error code

   * \ingroup mofem_fs_post_proc

   */
  MoFEMErrorCode writeFile(const std::string file_name) {
    MoFEMFunctionBeginHot;
    // #ifdef MOAB_HDF5_PARALLEL
    rval = postProcMesh.write_file(file_name.c_str(), "MOAB",
                                   "PARALLEL=WRITE_PART");
    CHKERRG(rval);
    // #else
    //  #warning "No parallel HDF5, not most efficient way of writing files"
    //  if(mField.get_comm_rank()==0) {
    //    rval = postProcMesh.write_file(file_name.c_str(),"MOAB","");
    //    CHKERRG(rval);
    //  }
    // #endif
    MoFEMFunctionReturnHot(0);
  }
};

template <class VOLUME_ELEMENT>
struct PostProcTemplateVolumeOnRefinedMesh
    : public PostProcTemplateOnRefineMesh<VOLUME_ELEMENT> {

  typedef PostProcTemplateOnRefineMesh<VOLUME_ELEMENT> T;

  bool tenNodesPostProcTets;
  int nbOfRefLevels;

  PostProcTemplateVolumeOnRefinedMesh(MoFEM::Interface &m_field,
                                      bool ten_nodes_post_proc_tets = true,
                                      int nb_ref_levels = -1)
      : PostProcTemplateOnRefineMesh<VOLUME_ELEMENT>(m_field),
        tenNodesPostProcTets(ten_nodes_post_proc_tets),
        nbOfRefLevels(nb_ref_levels) {}

  virtual ~PostProcTemplateVolumeOnRefinedMesh() {
    moab::Interface &moab = T::coreMesh;
    ParallelComm *pcomm_post_proc_mesh =
        ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm_post_proc_mesh != NULL) {
      delete pcomm_post_proc_mesh;
    }
  }

  MatrixDouble shapeFunctions;
  MatrixDouble coordsAtGaussPts;
  ublas::matrix<int> refTets;
  MatrixDouble gaussPts_FirstOrder;

  // Gauss pts set on refined mesh
  int getRule(int order) { return -1; };

  typedef PostProcCommonOnRefMesh::CommonDataForVolume CommonData;
  CommonData commonData;

  virtual PostProcCommonOnRefMesh::CommonData &getCommonData() {
    return commonData;
  }

  /** \brief Generate reference mesh on single element

  Each element is subdivided on smaller elements, i.e. a reference mesh on
  single element is created. Nodes of such reference mesh are used as
  integration points at which field values are calculated and to
  each node a "moab" tag is attached to store those values.

  */
  MoFEMErrorCode generateReferenceElementMesh() {
    MoFEMFunctionBeginHot;

    int max_level = 0;
    if (nbOfRefLevels == -1) {
      PetscBool flg = PETSC_TRUE;
      PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_max_post_proc_ref_level",
                         &max_level, &flg);
    } else {
      max_level = nbOfRefLevels;
    }

    double base_coords[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};

    moab::Core core_ref;
    moab::Interface &moab_ref = core_ref;

    EntityHandle nodes[4];
    for (int nn = 0; nn < 4; nn++) {
      rval = moab_ref.create_vertex(&base_coords[3 * nn], nodes[nn]);
      CHKERRG(rval);
    }
    EntityHandle tet;
    rval = moab_ref.create_element(MBTET, nodes, 4, tet);
    CHKERRG(rval);

    MoFEM::Core m_core_ref(moab_ref, PETSC_COMM_SELF, -2);
    MoFEM::Interface &m_field_ref = m_core_ref;

    ierr = m_field_ref.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, BitRefLevel().set(0));
    CHKERRG(ierr);

    for (int ll = 0; ll < max_level; ll++) {
      PetscPrintf(T::mField.get_comm(), "Refine Level %d\n", ll);
      Range edges;
      ierr = m_field_ref.getInterface<BitRefManager>()
                 ->getEntitiesByTypeAndRefLevel(
                     BitRefLevel().set(ll), BitRefLevel().set(), MBEDGE, edges);
      CHKERRG(ierr);
      Range tets;
      ierr = m_field_ref.getInterface<BitRefManager>()
                 ->getEntitiesByTypeAndRefLevel(
                     BitRefLevel().set(ll), BitRefLevel(ll).set(), MBTET, tets);
      CHKERRG(ierr);
      // refine mesh
      MeshRefinement *m_ref;
      ierr = m_field_ref.getInterface(m_ref);
      CHKERRG(ierr);
      ierr = m_ref->add_verices_in_the_middel_of_edges(
          edges, BitRefLevel().set(ll + 1));
      CHKERRG(ierr);
      ierr = m_ref->refine_TET(tets, BitRefLevel().set(ll + 1));
      CHKERRG(ierr);
    }

    Range tets;
    ierr =
        m_field_ref.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
            BitRefLevel().set(max_level), BitRefLevel().set(max_level), MBTET,
            tets);
    CHKERRG(ierr);

    if (tenNodesPostProcTets) {
      // Range edges;
      // rval = moab_ref.get_adjacencies(tets,1,true,edges); CHKERRG(rval);
      EntityHandle meshset;
      rval = moab_ref.create_meshset(MESHSET_SET, meshset);
      CHKERRG(rval);
      rval = moab_ref.add_entities(meshset, tets);
      CHKERRG(rval);
      rval = moab_ref.convert_entities(meshset, true, false, false);
      CHKERRG(rval);
      rval = moab_ref.delete_entities(&meshset, 1);
      CHKERRG(rval);
    }

    Range elem_nodes;
    rval = moab_ref.get_connectivity(tets, elem_nodes, false);
    CHKERRG(rval);
    // ierr = m_field_ref.get_entities_by_type_and_ref_level(
    //   BitRefLevel().set(max_level),BitRefLevel().set(),MBVERTEX,elem_nodes
    // ); CHKERRG(ierr);

    std::map<EntityHandle, int> little_map;
    gaussPts_FirstOrder.resize(elem_nodes.size(), 4, 0);
    Range::iterator nit = elem_nodes.begin();
    for (int gg = 0; nit != elem_nodes.end(); nit++, gg++) {
      rval = moab_ref.get_coords(&*nit, 1, &gaussPts_FirstOrder(gg, 0));
      CHKERRG(rval);
      little_map[*nit] = gg;
    }
    T::gaussPts = gaussPts_FirstOrder;
    T::gaussPts = trans(T::gaussPts);

    Range::iterator tit = tets.begin();
    for (int tt = 0; tit != tets.end(); tit++, tt++) {
      const EntityHandle *conn;
      int num_nodes;
      rval = moab_ref.get_connectivity(*tit, conn, num_nodes, false);
      CHKERRG(rval);
      if (tt == 0) {
        // Ref tets has number of rows equal to number of tets on element,
        // columns are number of gauss points
        refTets.resize(tets.size(), num_nodes);
      }
      for (int nn = 0; nn < num_nodes; nn++) {
        refTets(tt, nn) = little_map[conn[nn]];
      }
    }

    shapeFunctions.resize(elem_nodes.size(), 4);
    ierr =
        ShapeMBTET(&*shapeFunctions.data().begin(), &T::gaussPts(0, 0),
                   &T::gaussPts(1, 0), &T::gaussPts(2, 0), elem_nodes.size());
    CHKERRG(ierr);

    // EntityHandle meshset;
    // rval = moab_ref.create_meshset(MESHSET_SET|MESHSET_TRACK_OWNER,meshset);
    // CHKERRG(rval); rval = moab_ref.add_entities(meshset,tets); CHKERRG(rval);
    // rval =
    // moab_ref.write_file("test_reference_mesh.vtk","VTK","",&meshset,1);
    // CHKERRG(rval);
    // moab_ref.list_entities(tets);

    MoFEMFunctionReturnHot(0);
  }

  /** \brief Set integration points

  If reference mesh is generated on single elements. This function maps
  reference coordinates into physical coordinates and create element
  on post-processing mesh.

  */
  MoFEMErrorCode setGaussPts(int order) {
    MoFEMFunctionBeginHot;

    try {

      Tag th;
      int def_in_the_loop = -1;
      rval = T::postProcMesh.tag_get_handle(
          "NB_IN_THE_LOOP", 1, MB_TYPE_INTEGER, th,
          MB_TAG_CREAT | MB_TAG_SPARSE, &def_in_the_loop);
      CHKERRG(rval);

      T::mapGaussPts.resize(gaussPts_FirstOrder.size1());
      for (unsigned int gg = 0; gg < gaussPts_FirstOrder.size1(); gg++) {
        rval = T::postProcMesh.create_vertex(&gaussPts_FirstOrder(gg, 0),
                                             T::mapGaussPts[gg]);
        CHKERRG(rval);
      }

      commonData.tEts.clear();
      for (unsigned int tt = 0; tt < refTets.size1(); tt++) {
        int num_nodes = refTets.size2();
        EntityHandle conn[num_nodes];
        for (int nn = 0; nn != num_nodes; nn++) {
          conn[nn] = T::mapGaussPts[refTets(tt, nn)];
        }
        EntityHandle tet;
        rval = T::postProcMesh.create_element(MBTET, conn, num_nodes, tet);
        CHKERRG(rval);
        int n_in_loop = T::nInTheLoop;
        rval = T::postProcMesh.tag_set_data(th, &tet, 1, &n_in_loop);
        CHKERRG(rval);
        commonData.tEts.insert(tet);
      }

      EntityHandle fe_ent = T::numeredEntFiniteElementPtr->getEnt();
      T::coords.resize(12, false);
      {
        const EntityHandle *conn;
        int num_nodes;
        T::mField.get_moab().get_connectivity(fe_ent, conn, num_nodes, true);
        // coords.resize(3*num_nodes,false);
        rval = T::mField.get_moab().get_coords(conn, num_nodes, &T::coords[0]);
        CHKERRG(rval);
      }

      Range nodes;
      rval = T::postProcMesh.get_connectivity(commonData.tEts, nodes, false);
      CHKERRG(rval);

      coordsAtGaussPts.resize(nodes.size(), 3, false);
      for (unsigned int gg = 0; gg < nodes.size(); gg++) {
        for (int dd = 0; dd < 3; dd++) {
          coordsAtGaussPts(gg, dd) =
              cblas_ddot(4, &shapeFunctions(gg, 0), 1, &T::coords[dd], 3);
        }
      }

      T::mapGaussPts.resize(nodes.size());
      Range::iterator nit = nodes.begin();
      for (int gg = 0; nit != nodes.end(); nit++, gg++) {
        rval = T::postProcMesh.set_coords(&*nit, 1, &coordsAtGaussPts(gg, 0));
        CHKERRG(rval);
        T::mapGaussPts[gg] = *nit;
      }

    } catch (std::exception &ex) {
      std::ostringstream ss;
      ss << "throw in method: " << ex.what() << " at line " << __LINE__
         << " in file " << __FILE__;
      SETERRQ(PETSC_COMM_SELF, MOFEM_STD_EXCEPTION_THROW, ss.str().c_str());
    }

    MoFEMFunctionReturnHot(0);
  }

  /** \brief Clear operators list

  Clear operators list, user can use the same mesh instance to post-process
  different problem or the same problem with different set of post-processed
  fields.

  */
  MoFEMErrorCode clearOperators() {
    MoFEMFunctionBeginHot;
    T::getOpPtrVector().clear();
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode preProcess() {
    MoFEMFunctionBeginHot;

    moab::Interface &moab = T::coreMesh;
    ParallelComm *pcomm_post_proc_mesh =
        ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm_post_proc_mesh != NULL) {
      delete pcomm_post_proc_mesh;
    }
    rval = T::postProcMesh.delete_mesh();
    CHKERRG(rval);
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBeginHot;

    moab::Interface &moab = T::coreMesh;
    ParallelComm *pcomm =
        ParallelComm::get_pcomm(&T::mField.get_moab(), MYPCOMM_INDEX);
    ParallelComm *pcomm_post_proc_mesh =
        ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm_post_proc_mesh == NULL) {
      pcomm_post_proc_mesh = new ParallelComm(&moab, T::mField.get_comm());
    }

    Range edges;
    rval = T::postProcMesh.get_entities_by_type(0, MBEDGE, edges, false);
    CHKERRG(rval);
    rval = T::postProcMesh.delete_entities(edges);
    CHKERRG(rval);
    Range tris;
    rval = T::postProcMesh.get_entities_by_type(0, MBTRI, tris, false);
    CHKERRG(rval);
    rval = T::postProcMesh.delete_entities(tris);
    CHKERRG(rval);

    Range tets;
    rval = T::postProcMesh.get_entities_by_type(0, MBTET, tets, false);
    CHKERRG(rval);

    // std::cerr << "total tets size " << tets.size() << std::endl;

    int rank = pcomm->rank();
    Range::iterator tit = tets.begin();
    for (; tit != tets.end(); tit++) {
      rval = T::postProcMesh.tag_set_data(pcomm_post_proc_mesh->part_tag(),
                                          &*tit, 1, &rank);
      CHKERRG(rval);
    }

    rval = pcomm_post_proc_mesh->resolve_shared_ents(0);
    CHKERRG(rval);

    // #ifndef MOAB_HDF5_PARALLEL
    // #warning "No parallel HDF5, not most efficient way of writing files"
    // for(int r = 0;r<pcomm_post_proc_mesh->size();r++) {
    //   // FIXME make better communication send only to proc 0
    //   rval = pcomm_post_proc_mesh->broadcast_entities(r,tets,false,true);
    //   CHKERRG(rval);
    // }
    // #endif //

    MoFEMFunctionReturnHot(0);
  }

  /** \brief Add operator to post-process Hdiv field
   */
  MoFEMErrorCode addHdivFunctionsPostProc(const std::string field_name) {
    MoFEMFunctionBeginHot;
    T::getOpPtrVector().push_back(
        new OpHdivFunctions(T::postProcMesh, T::mapGaussPts, field_name));
    MoFEMFunctionReturnHot(0);
  }

  struct OpHdivFunctions : public VOLUME_ELEMENT::UserDataOperator {

    moab::Interface &postProcMesh;
    std::vector<EntityHandle> &mapGaussPts;

    OpHdivFunctions(moab::Interface &post_proc_mesh,
                    std::vector<EntityHandle> &map_gauss_pts,
                    const std::string field_name)
        : VOLUME_ELEMENT::UserDataOperator(field_name,
                                           T::UserDataOperator::OPCOL),
          postProcMesh(post_proc_mesh), mapGaussPts(map_gauss_pts) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBeginHot;

      if (data.getIndices().size() == 0)
        MoFEMFunctionReturnHot(0);

      std::vector<Tag> th;
      th.resize(data.getFieldData().size());

      double def_VAL[9] = {0, 0, 0};

      switch (type) {
      case MBTRI:
        for (unsigned int dd = 0; dd < data.getN().size2() / 3; dd++) {
          std::ostringstream ss;
          ss << "HDIV_FACE_" << side << "_" << dd;
          rval = postProcMesh.tag_get_handle(
              ss.str().c_str(), 3, MB_TYPE_DOUBLE, th[dd],
              MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
          CHKERRG(rval);
        }
        break;
      case MBTET:
        for (unsigned int dd = 0; dd < data.getN().size2() / 3; dd++) {
          std::ostringstream ss;
          ss << "HDIV_TET_" << dd;
          rval = postProcMesh.tag_get_handle(
              ss.str().c_str(), 3, MB_TYPE_DOUBLE, th[dd],
              MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
          CHKERRG(rval);
        }
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                "data inconsistency");
      }

      for (unsigned int gg = 0; gg < data.getN().size1(); gg++) {
        for (unsigned int dd = 0; dd < data.getN().size2() / 3; dd++) {
          ierr = postProcMesh.tag_set_data(th[dd], &mapGaussPts[gg], 1,
                                           &data.getVectorN<3>(gg)(dd, 0));
          CHKERRG(ierr);
        }
      }

      MoFEMFunctionReturnHot(0);
    }
  };
};

/** \brief Post processing
 * \ingroup mofem_fs_post_proc
 */
struct PostProcVolumeOnRefinedMesh
    : public PostProcTemplateVolumeOnRefinedMesh<
          MoFEM::VolumeElementForcesAndSourcesCore> {

  PostProcVolumeOnRefinedMesh(MoFEM::Interface &m_field,
                              bool ten_nodes_post_proc_tets = true,
                              int nb_ref_levels = -1)
      : PostProcTemplateVolumeOnRefinedMesh<
            MoFEM::VolumeElementForcesAndSourcesCore>(m_field) {}
};

// /** \deprecated Use PostPocOnRefinedMesh instead
// */
// DEPRECATED typedef PostProcVolumeOnRefinedMesh PostPocOnRefinedMesh;

/**
 *  \brief Postprocess on prism
 *
 * \ingroup mofem_fs_post_proc
 */
struct PostProcFatPrismOnRefinedMesh
    : public PostProcTemplateOnRefineMesh<
          MoFEM::FatPrismElementForcesAndSourcesCore> {

  bool tenNodesPostProcTets;

  PostProcFatPrismOnRefinedMesh(MoFEM::Interface &m_field,
                                bool ten_nodes_post_proc_tets = true)
      : PostProcTemplateOnRefineMesh<
            MoFEM::FatPrismElementForcesAndSourcesCore>(m_field),
        tenNodesPostProcTets(ten_nodes_post_proc_tets) {}

  virtual ~PostProcFatPrismOnRefinedMesh() {
    ParallelComm *pcomm_post_proc_mesh =
        ParallelComm::get_pcomm(&postProcMesh, MYPCOMM_INDEX);
    if (pcomm_post_proc_mesh != NULL) {
      delete pcomm_post_proc_mesh;
    }
  }

  int getRuleTrianglesOnly(int order) { return -1; };
  int getRuleThroughThickness(int order) { return -1; };

  struct PointsMap3D {
    const int kSi;
    const int eTa;
    const int zEta;
    int nN;
    PointsMap3D(const int ksi, const int eta, const int zeta, const int nn)
        : kSi(ksi), eTa(eta), zEta(zeta), nN(nn) {}
  };

  typedef multi_index_container<
      PointsMap3D,
      indexed_by<ordered_unique<composite_key<
          PointsMap3D, member<PointsMap3D, const int, &PointsMap3D::kSi>,
          member<PointsMap3D, const int, &PointsMap3D::eTa>,
          member<PointsMap3D, const int, &PointsMap3D::zEta>>>>>
      PointsMap3D_multiIndex;

  PointsMap3D_multiIndex pointsMap;

  MoFEMErrorCode setGaussPtsTrianglesOnly(int order_triangles_only);
  MoFEMErrorCode setGaussPtsThroughThickness(int order_thickness);
  MoFEMErrorCode generateReferenceElementMesh();

  std::map<EntityHandle, EntityHandle> elementsMap;

  MoFEMErrorCode preProcess();
  MoFEMErrorCode postProcess();

  struct CommonData : PostProcCommonOnRefMesh::CommonData {};
  CommonData commonData;

  virtual PostProcCommonOnRefMesh::CommonData &getCommonData() {
    return commonData;
  }
};

/**
 * \brief Postprocess on face
 *
 * \ingroup mofem_fs_post_proc
 */
struct PostProcFaceOnRefinedMesh : public PostProcTemplateOnRefineMesh<
                                       MoFEM::FaceElementForcesAndSourcesCore> {

  bool sixNodePostProcTris;

  PostProcFaceOnRefinedMesh(MoFEM::Interface &m_field,
                            bool six_node_post_proc_tris = true)
      : PostProcTemplateOnRefineMesh<MoFEM::FaceElementForcesAndSourcesCore>(
            m_field),
        sixNodePostProcTris(six_node_post_proc_tris) {}

  virtual ~PostProcFaceOnRefinedMesh() {
    ParallelComm *pcomm_post_proc_mesh =
        ParallelComm::get_pcomm(&postProcMesh, MYPCOMM_INDEX);
    if (pcomm_post_proc_mesh != NULL) {
      delete pcomm_post_proc_mesh;
    }
  }

  // Gauss pts set on refined mesh
  int getRule(int order) { return -1; };

  MoFEMErrorCode generateReferenceElementMesh();
  MoFEMErrorCode setGaussPts(int order);

  std::map<EntityHandle, EntityHandle> elementsMap;

  MoFEMErrorCode preProcess();
  MoFEMErrorCode postProcess();

  struct CommonData : PostProcCommonOnRefMesh::CommonData {};
  CommonData commonData;

  virtual PostProcCommonOnRefMesh::CommonData &getCommonData() {
    return commonData;
  }
};

#endif //__POSTPROC_ON_REF_MESH_HPP

/***************************************************************************/ /**
 * \defgroup mofem_fs_post_proc Post Process
 * \ingroup user_modules
******************************************************************************/
