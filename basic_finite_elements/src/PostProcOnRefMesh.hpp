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
    MoFEMFunctionBegin;

    auto get_nb_of_ref_levels_from_options = [this] {
      if (nbOfRefLevels == -1) {
        int max_level = 0;
        PetscBool flg = PETSC_TRUE;
        PetscOptionsGetInt(PETSC_NULL, PETSC_NULL,
                           "-my_max_post_proc_ref_level", &max_level, &flg);
        return max_level;
      } else {
        return nbOfRefLevels;
      }
      return 0;
    };
    const int max_level = get_nb_of_ref_levels_from_options();

    moab::Core core_ref;
    moab::Interface &moab_ref = core_ref;
    
    auto create_reference_element = [&moab_ref]() {
      MoFEMFunctionBegin;
      const double base_coords[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
      EntityHandle nodes[4];
      for (int nn = 0; nn < 4; nn++) {
        CHKERR moab_ref.create_vertex(&base_coords[3 * nn], nodes[nn]);
      }
      EntityHandle tet;
      CHKERR moab_ref.create_element(MBTET, nodes, 4, tet);
      MoFEMFunctionReturn(0);
    };

    MoFEM::Core m_core_ref(moab_ref, PETSC_COMM_SELF, -2);
    MoFEM::Interface &m_field_ref = m_core_ref;

    auto refine_ref_tetrahedron = [this, &m_field_ref, max_level]() {
      MoFEMFunctionBegin;
      // seed ref mofem database by setting bit ref level to reference
      // tetrahedron
      CHKERR m_field_ref.getInterface<BitRefManager>()->setBitRefLevelByDim(
          0, 3, BitRefLevel().set(0));
      for (int ll = 0; ll != max_level; ++ll) {
        PetscPrintf(T::mField.get_comm(), "Refine Level %d\n", ll);
        Range edges;
        CHKERR m_field_ref.getInterface<BitRefManager>()
            ->getEntitiesByTypeAndRefLevel(BitRefLevel().set(ll),
                                           BitRefLevel().set(), MBEDGE, edges);
        Range tets;
        CHKERR m_field_ref.getInterface<BitRefManager>()
            ->getEntitiesByTypeAndRefLevel(BitRefLevel().set(ll),
                                           BitRefLevel(ll).set(), MBTET, tets);
        // refine mesh
        MeshRefinement *m_ref;
        CHKERR m_field_ref.getInterface(m_ref);
        CHKERR m_ref->add_vertices_in_the_middel_of_edges(
            edges, BitRefLevel().set(ll + 1));
        CHKERR m_ref->refine_TET(tets, BitRefLevel().set(ll + 1));
      }
      MoFEMFunctionReturn(0);
    };

    auto get_ref_gauss_pts_and_shape_functions = [this, max_level, &moab_ref,
                                                  &m_field_ref]() {
      MoFEMFunctionBegin;
      for (int ll = 0; ll != max_level + 1; ++ll) {
        Range tets;
        CHKERR m_field_ref.getInterface<BitRefManager>()
            ->getEntitiesByTypeAndRefLevel(BitRefLevel().set(ll),
                                           BitRefLevel().set(ll), MBTET, tets);
        if (tenNodesPostProcTets) {
          EntityHandle meshset;
          CHKERR moab_ref.create_meshset(MESHSET_SET, meshset);
          CHKERR moab_ref.add_entities(meshset, tets);
          CHKERR moab_ref.convert_entities(meshset, true, false, false);
          CHKERR moab_ref.delete_entities(&meshset, 1);
        }
        Range elem_nodes;
        CHKERR moab_ref.get_connectivity(tets, elem_nodes, false);

        auto &gauss_pts = levelGaussPtsOnRefMesh[ll];
        gauss_pts.resize(elem_nodes.size(), 4, false);
        std::map<EntityHandle, int> little_map;
        Range::iterator nit = elem_nodes.begin();
        for (int gg = 0; nit != elem_nodes.end(); nit++, gg++) {
          CHKERR moab_ref.get_coords(&*nit, 1, &gauss_pts(gg, 0));
          little_map[*nit] = gg;
        }
        gauss_pts = trans(gauss_pts);

        auto &ref_tets = levelRefTets[ll];
        Range::iterator tit = tets.begin();
        for (int tt = 0; tit != tets.end(); ++tit, ++tt) {
          const EntityHandle *conn;
          int num_nodes;
          CHKERR moab_ref.get_connectivity(*tit, conn, num_nodes, false);
          if (tt == 0) {
            // Ref tets has number of rows equal to number of tets on element,
            // columns are number of gauss points
            ref_tets.resize(tets.size(), num_nodes);
          }
          for (int nn = 0; nn != num_nodes; ++nn) {
            ref_tets(tt, nn) = little_map[conn[nn]];
          }
        }

        auto &shape_functions = levelShapeFunctions[ll];
        shape_functions.resize(elem_nodes.size(), 4);
        CHKERR ShapeMBTET(&*shape_functions.data().begin(), &gauss_pts(0, 0),
                          &gauss_pts(1, 0), &gauss_pts(2, 0),
                          elem_nodes.size());
      }
      MoFEMFunctionReturn(0);
    };

    levelRefTets.resize(max_level + 1);
    levelGaussPtsOnRefMesh.resize(max_level + 1);
    levelShapeFunctions.resize(max_level + 1);

    CHKERR create_reference_element();
    CHKERR refine_ref_tetrahedron();
    CHKERR get_ref_gauss_pts_and_shape_functions();

    MoFEMFunctionReturn(0);
  }

  /** \brief Set integration points

  If reference mesh is generated on single elements. This function maps
  reference coordinates into physical coordinates and create element
  on post-processing mesh.

  */
  MoFEMErrorCode setGaussPts(int order) {
    MoFEMFunctionBegin;

    auto get_element_max_dofs_order = [this]() {
      int max_order = 0;
      auto &dofs_multi_index = *this->dataPtr;
      for (auto &dof : dofs_multi_index) {
        const int dof_order = dof->getDofOrder();
        max_order = (max_order < dof_order) ? dof_order : max_order;
      };
      return max_order;
    };

    const int dof_max_order = get_element_max_dofs_order();
    int level = (dof_max_order > 0) ? (dof_max_order - 1) / 2 : 0;
    if (level > levelGaussPtsOnRefMesh.size() - 1) {
      level = levelGaussPtsOnRefMesh.size() - 1;
    }

    auto &level_ref_gauss_pts = levelGaussPtsOnRefMesh[level];
    auto &level_ref_tets = levelRefTets[level];
    auto &shape_functions = levelShapeFunctions[level];
    T::gaussPts.resize(level_ref_gauss_pts.size1(),
                       level_ref_gauss_pts.size2(), false);
    noalias(T::gaussPts) = level_ref_gauss_pts;

    ReadUtilIface *iface;
    CHKERR T::postProcMesh.query_interface(iface);

    const int num_nodes = level_ref_gauss_pts.size2();
    std::vector<double *> arrays;
    EntityHandle startv;
    CHKERR iface->get_node_coords(3, num_nodes, 0, startv, arrays);
    T::mapGaussPts.resize(level_ref_gauss_pts.size2());
    for (int gg = 0; gg != num_nodes; ++gg)
      T::mapGaussPts[gg] = startv + gg;


    Tag th;
    int def_in_the_loop = -1;
    CHKERR T::postProcMesh.tag_get_handle("NB_IN_THE_LOOP", 1, MB_TYPE_INTEGER,
                                          th, MB_TAG_CREAT | MB_TAG_SPARSE,
                                          &def_in_the_loop);

    commonData.tEts.clear();
    const int num_el = level_ref_tets.size1();
    const int num_nodes_on_ele = level_ref_tets.size2();
    EntityHandle starte;
    EntityHandle *conn;
    CHKERR iface->get_element_connect(num_el, num_nodes_on_ele, MBTET, 0,
                                      starte, conn);
    for (unsigned int tt = 0; tt != level_ref_tets.size1(); ++tt) {
      for (int nn = 0; nn != num_nodes_on_ele; ++nn) 
        conn[num_nodes_on_ele * tt + nn] =
            T::mapGaussPts[level_ref_tets(tt, nn)];
    }
    CHKERR iface->update_adjacencies(starte, num_el, num_nodes_on_ele, conn);
    commonData.tEts = Range(starte, starte + num_el - 1);
    CHKERR T::postProcMesh.tag_clear_data(th, commonData.tEts,
                                          &(T::nInTheLoop));

    EntityHandle fe_ent = T::numeredEntFiniteElementPtr->getEnt();
    T::coords.resize(12, false);
    {
      const EntityHandle *conn;
      int num_nodes;
      T::mField.get_moab().get_connectivity(fe_ent, conn, num_nodes, true);
      CHKERR T::mField.get_moab().get_coords(conn, num_nodes, &T::coords[0]);
    }

    FTensor::Index<'i', 3> i;
    FTensor::Tensor0<FTensor::PackPtr<double *, 1>> t_n(
        &*shape_functions.data().begin());
    FTensor::Tensor1<FTensor::PackPtr<double *, 1>, 3> t_coords(
        arrays[0], arrays[1], arrays[2]);
    const double *t_coords_ele_x = &T::coords[0];
    const double *t_coords_ele_y = &T::coords[1];
    const double *t_coords_ele_z = &T::coords[2];
    for (unsigned int gg = 0; gg != num_nodes; ++gg) {
      FTensor::Tensor1<FTensor::PackPtr<const double *, 3>, 3> t_ele_coords(
          t_coords_ele_x, t_coords_ele_y, t_coords_ele_z);
      t_coords(i) = 0;
      for (int nn = 0; nn != 4; ++nn) {
        t_coords(i) += t_n * t_ele_coords(i);
        ++t_ele_coords;
        ++t_n;
      }
      ++t_coords;
    }

    MoFEMFunctionReturn(0);
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
      CHKERR T::postProcMesh.delete_mesh();
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
      CHKERR T::postProcMesh.get_entities_by_type(0, MBEDGE, edges, false);
      CHKERR T::postProcMesh.delete_entities(edges);
      Range tris;
      CHKERR T::postProcMesh.get_entities_by_type(0, MBTRI, tris, false);
      CHKERR T::postProcMesh.delete_entities(tris);

      Range tets;
      CHKERR T::postProcMesh.get_entities_by_type(0, MBTET, tets, false);

      int rank = pcomm->rank();
      Range::iterator tit = tets.begin();
      for (; tit != tets.end(); tit++) {
        CHKERR T::postProcMesh.tag_set_data(pcomm_post_proc_mesh->part_tag(),
                                            &*tit, 1, &rank);
      }

      CHKERR pcomm_post_proc_mesh->resolve_shared_ents(0);

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
        MoFEMFunctionBegin;

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
            CHKERR postProcMesh.tag_get_handle(
                ss.str().c_str(), 3, MB_TYPE_DOUBLE, th[dd],
                MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
          }
          break;
        case MBTET:
          for (unsigned int dd = 0; dd < data.getN().size2() / 3; dd++) {
            std::ostringstream ss;
            ss << "HDIV_TET_" << dd;
            CHKERR postProcMesh.tag_get_handle(
                ss.str().c_str(), 3, MB_TYPE_DOUBLE, th[dd],
                MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
          }
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                  "data inconsistency");
        }

        for (unsigned int gg = 0; gg < data.getN().size1(); gg++) {
          for (unsigned int dd = 0; dd < data.getN().size2() / 3; dd++) {
            CHKERR postProcMesh.tag_set_data(th[dd], &mapGaussPts[gg], 1,
                                             &data.getVectorN<3>(gg)(dd, 0));
          }
        }

        MoFEMFunctionReturn(0);
      }
    };

  private:
    std::vector<MatrixDouble> levelShapeFunctions;
    std::vector<MatrixDouble> levelGaussPtsOnRefMesh;
    std::vector<ublas::matrix<int>> levelRefTets;
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
