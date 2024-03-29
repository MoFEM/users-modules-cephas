/** \file PostProcOnRefMesh.hpp
 * \brief Post-process fields on refined mesh
 *
 * Create refined mesh, without enforcing continuity between element. Calculate
 * field values on nodes of that mesh.
 *
 * \ingroup mofem_fs_post_proc
 */



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
                          EntitiesFieldData::EntData &data);
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
    int spaceDim;

    OpGetFieldGradientValues(moab::Interface &post_proc_mesh,
                             std::vector<EntityHandle> &map_gauss_pts,
                             const std::string field_name,
                             const std::string tag_name,
                             CommonData &common_data, Vec v = PETSC_NULL,
                             int space_dim = 3)
        : MoFEM::ForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL),
          postProcMesh(post_proc_mesh), mapGaussPts(map_gauss_pts),
          commonData(common_data), tagName(tag_name), V(v),
          spaceDim(space_dim) {}

    VectorDouble vAlues;
    VectorDouble *vAluesPtr;

    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data);
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
  boost::shared_ptr<WrapMPIComm> wrapRefMeshComm;

  std::vector<EntityHandle> mapGaussPts;

  PostProcTemplateOnRefineMesh(MoFEM::Interface &m_field)
      : ELEMENT(m_field), postProcMesh(coreMesh) {}

  virtual ~PostProcTemplateOnRefineMesh() {
    ParallelComm *pcomm_post_proc_mesh =
        ParallelComm::get_pcomm(&postProcMesh, MYPCOMM_INDEX);
    if (pcomm_post_proc_mesh != NULL)
      delete pcomm_post_proc_mesh;
  }

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

  /** \brief Add operator to post-process L2 or H1 field gradient

  \param field_name
  \param space_dim the dimension of the problem
  \param v If vector is given, values from vector are used to set tags on mesh

  * \ingroup mofem_fs_post_proc

*/
  MoFEMErrorCode addFieldValuesGradientPostProc(const std::string field_name,
                                                int space_dim,
                                                Vec v = PETSC_NULL) {
    MoFEMFunctionBeginHot;
    ELEMENT::getOpPtrVector().push_back(
        new PostProcCommonOnRefMesh::OpGetFieldGradientValues(
            postProcMesh, mapGaussPts, field_name, field_name + "_GRAD",
            getCommonData(), v, space_dim));
    MoFEMFunctionReturnHot(0);
  }

  /**
   * \brief wrote results in (MOAB) format, use "file_name.h5m"
   * @param  file_name file name (should always end with .h5m)
   * @return           error code

   * \ingroup mofem_fs_post_proc

   */
  MoFEMErrorCode writeFile(const std::string file_name,
                           const char *file_type = "MOAB",
                           const char *file_options = "PARALLEL=WRITE_PART") {
    MoFEMFunctionBegin;
    ParallelComm *pcomm_post_proc_mesh =
        ParallelComm::get_pcomm(&postProcMesh, MYPCOMM_INDEX);
    if (pcomm_post_proc_mesh == NULL)
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "ParallelComm not allocated");
    CHKERR postProcMesh.write_file(file_name.c_str(), file_type, file_options);
    MoFEMFunctionReturn(0);
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

    auto generate_for_hex = [&]() {
      MoFEMFunctionBegin;

      moab::Core core_ref;
      moab::Interface &moab_ref = core_ref;

      auto create_reference_element = [&moab_ref]() {
        MoFEMFunctionBegin;
        constexpr double base_coords[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
                                          0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
        EntityHandle nodes[8];
        for (int nn = 0; nn < 8; nn++)
          CHKERR moab_ref.create_vertex(&base_coords[3 * nn], nodes[nn]);
        EntityHandle hex;
        CHKERR moab_ref.create_element(MBHEX, nodes, 8, hex);
        MoFEMFunctionReturn(0);
      };

      auto add_ho_nodes = [&]() {
        MoFEMFunctionBegin;
        Range hexes;
        CHKERR moab_ref.get_entities_by_type(0, MBHEX, hexes, true);
        EntityHandle meshset;
        CHKERR moab_ref.create_meshset(MESHSET_SET, meshset);
        CHKERR moab_ref.add_entities(meshset, hexes);
        CHKERR moab_ref.convert_entities(meshset, true, true, true);
        CHKERR moab_ref.delete_entities(&meshset, 1);
        MoFEMFunctionReturn(0);
      };

      auto set_gauss_pts = [&](std::map<EntityHandle, int> &little_map) {
        MoFEMFunctionBegin;
        Range hexes;
        CHKERR moab_ref.get_entities_by_type(0, MBHEX, hexes, true);
        Range hexes_nodes;
        CHKERR moab_ref.get_connectivity(hexes, hexes_nodes, false);
        auto &gauss_pts = levelGaussPtsOnRefMeshHexes[0];
        gauss_pts.resize(hexes_nodes.size(), 4, false);
        size_t gg = 0;
        for (auto node : hexes_nodes) {
          CHKERR moab_ref.get_coords(&node, 1, &gauss_pts(gg, 0));
          little_map[node] = gg;
          ++gg;
        }
        gauss_pts = trans(gauss_pts);
        MoFEMFunctionReturn(0);
      };

      auto set_ref_hexes = [&](std::map<EntityHandle, int> &little_map) {
        MoFEMFunctionBegin;
        Range hexes;
        CHKERR moab_ref.get_entities_by_type(0, MBHEX, hexes, true);
        size_t hh = 0;
        auto &ref_hexes = levelRefHexes[0];
        for (auto hex : hexes) {
          const EntityHandle *conn;
          int num_nodes;
          CHKERR moab_ref.get_connectivity(hex, conn, num_nodes, false);
          if (ref_hexes.size2() != num_nodes) {
            ref_hexes.resize(hexes.size(), num_nodes);
          }
          for (int nn = 0; nn != num_nodes; ++nn) {
            ref_hexes(hh, nn) = little_map[conn[nn]];
          }
          ++hh;
        }
        MoFEMFunctionReturn(0);
      };

      auto set_shape_functions = [&]() {
        MoFEMFunctionBegin;
        auto &gauss_pts = levelGaussPtsOnRefMeshHexes[0];
        auto &shape_functions = levelShapeFunctionsHexes[0];
        const auto nb_gauss_pts = gauss_pts.size2();
        shape_functions.resize(nb_gauss_pts, 8);
        for (int gg = 0; gg != nb_gauss_pts; ++gg) {
          const double ksi = gauss_pts(0, gg);
          const double zeta = gauss_pts(1, gg);
          const double eta = gauss_pts(2, gg);
          shape_functions(gg, 0) = N_MBHEX0(ksi, zeta, eta);
          shape_functions(gg, 1) = N_MBHEX1(ksi, zeta, eta);
          shape_functions(gg, 2) = N_MBHEX2(ksi, zeta, eta);
          shape_functions(gg, 3) = N_MBHEX3(ksi, zeta, eta);
          shape_functions(gg, 4) = N_MBHEX4(ksi, zeta, eta);
          shape_functions(gg, 5) = N_MBHEX5(ksi, zeta, eta);
          shape_functions(gg, 6) = N_MBHEX6(ksi, zeta, eta);
          shape_functions(gg, 7) = N_MBHEX7(ksi, zeta, eta);
        }
        MoFEMFunctionReturn(0);
      };

      levelRefHexes.resize(1);
      levelGaussPtsOnRefMeshHexes.resize(1);
      levelShapeFunctionsHexes.resize(1);

      CHKERR create_reference_element();
      if (tenNodesPostProcTets)
        CHKERR add_ho_nodes();
      std::map<EntityHandle, int> little_map;
      CHKERR set_gauss_pts(little_map);
      CHKERR set_ref_hexes(little_map);
      CHKERR set_shape_functions();

      MoFEMFunctionReturn(0);
    };

    auto generate_for_tet = [&]() {
      MoFEMFunctionBegin;

      const int max_level = get_nb_of_ref_levels_from_options();

      moab::Core core_ref;
      moab::Interface &moab_ref = core_ref;

      auto create_reference_element = [&moab_ref]() {
        MoFEMFunctionBegin;
        constexpr double base_coords[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
        EntityHandle nodes[4];
        for (int nn = 0; nn < 4; nn++) {
          CHKERR
          moab_ref.create_vertex(&base_coords[3 * nn], nodes[nn]);
        }
        EntityHandle tet;
        CHKERR moab_ref.create_element(MBTET, nodes, 4, tet);
        MoFEMFunctionReturn(0);
      };

      MoFEM::CoreTmp<-1> m_core_ref(moab_ref, PETSC_COMM_SELF, -2);
      MoFEM::Interface &m_field_ref = m_core_ref;

      auto refine_ref_tetrahedron = [this, &m_field_ref, max_level]() {
        MoFEMFunctionBegin;
        // seed ref mofem database by setting bit ref level to reference
        // tetrahedron
        CHKERR
        m_field_ref.getInterface<BitRefManager>()->setBitRefLevelByDim(
            0, 3, BitRefLevel().set(0));
        for (int ll = 0; ll != max_level; ++ll) {
          MOFEM_TAG_AND_LOG_C("WORLD", Sev::verbose, "PostProc",
                              "Refine Level %d", ll);
          Range edges;
          CHKERR m_field_ref.getInterface<BitRefManager>()
              ->getEntitiesByTypeAndRefLevel(
                  BitRefLevel().set(ll), BitRefLevel().set(), MBEDGE, edges);
          Range tets;
          CHKERR m_field_ref.getInterface<BitRefManager>()
              ->getEntitiesByTypeAndRefLevel(
                  BitRefLevel().set(ll), BitRefLevel(ll).set(), MBTET, tets);
          // refine mesh
          MeshRefinement *m_ref;
          CHKERR m_field_ref.getInterface(m_ref);
          CHKERR m_ref->addVerticesInTheMiddleOfEdges(
              edges, BitRefLevel().set(ll + 1));
          CHKERR m_ref->refineTets(tets, BitRefLevel().set(ll + 1));
        }
        MoFEMFunctionReturn(0);
      };

      auto get_ref_gauss_pts_and_shape_functions = [this, max_level, &moab_ref,
                                                    &m_field_ref]() {
        MoFEMFunctionBegin;
        for (int ll = 0; ll != max_level + 1; ++ll) {
          Range tets;
          CHKERR
          m_field_ref.getInterface<BitRefManager>()
              ->getEntitiesByTypeAndRefLevel(
                  BitRefLevel().set(ll), BitRefLevel().set(ll), MBTET, tets);
          if (tenNodesPostProcTets) {
            EntityHandle meshset;
            CHKERR moab_ref.create_meshset(MESHSET_SET, meshset);
            CHKERR moab_ref.add_entities(meshset, tets);
            CHKERR moab_ref.convert_entities(meshset, true, false, false);
            CHKERR moab_ref.delete_entities(&meshset, 1);
          }
          Range elem_nodes;
          CHKERR moab_ref.get_connectivity(tets, elem_nodes, false);

          auto &gauss_pts = levelGaussPtsOnRefMeshTets[ll];
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
              ref_tets.resize(tets.size(), num_nodes);
            }
            for (int nn = 0; nn != num_nodes; ++nn) {
              ref_tets(tt, nn) = little_map[conn[nn]];
            }
          }

          auto &shape_functions = levelShapeFunctionsTets[ll];
          shape_functions.resize(elem_nodes.size(), 4);
          CHKERR ShapeMBTET(&*shape_functions.data().begin(), &gauss_pts(0, 0),
                            &gauss_pts(1, 0), &gauss_pts(2, 0),
                            elem_nodes.size());
        }
        MoFEMFunctionReturn(0);
      };

      levelRefTets.resize(max_level + 1);
      levelGaussPtsOnRefMeshTets.resize(max_level + 1);
      levelShapeFunctionsTets.resize(max_level + 1);

      CHKERR create_reference_element();
      CHKERR refine_ref_tetrahedron();
      CHKERR get_ref_gauss_pts_and_shape_functions();

      MoFEMFunctionReturn(0);
    };

    CHKERR generate_for_hex();
    CHKERR generate_for_tet();

    MoFEMFunctionReturn(0);
  }

  size_t getMaxLevel() const {
    auto get_element_max_dofs_order = [&]() {
      int max_order = 0;
      auto dofs_vec = this->getDataVectorDofsPtr();
      for (auto &dof : *dofs_vec) {
        const int dof_order = dof->getDofOrder();
        max_order = (max_order < dof_order) ? dof_order : max_order;
      };
      return max_order;
    };
    const auto dof_max_order = get_element_max_dofs_order();
    return (dof_max_order > 0) ? (dof_max_order - 1) / 2 : 0;
  };

  /** \brief Set integration points

  If reference mesh is generated on single elements. This function maps
  reference coordinates into physical coordinates and create element
  on post-processing mesh.

  */
  MoFEMErrorCode setGaussPts(int order) {
    MoFEMFunctionBegin;

    auto type = type_from_handle(this->getFEEntityHandle());

    auto set_gauss_pts = [&](auto &level_gauss_pts_on_ref_mesh, auto &level_ref,
                             auto &level_shape_functions,

                             auto start_vert_handle, auto start_ele_handle,
                             auto &verts_array, auto &conn, auto &ver_count,
                             auto &ele_count

                         ) {
      MoFEMFunctionBegin;

      auto level =
          std::min(getMaxLevel(), level_gauss_pts_on_ref_mesh.size() - 1);

      auto &level_ref_gauss_pts = level_gauss_pts_on_ref_mesh[level];
      auto &level_ref_ele = level_ref[level];
      auto &shape_functions = level_shape_functions[level];
      T::gaussPts.resize(level_ref_gauss_pts.size1(),
                         level_ref_gauss_pts.size2(), false);
      noalias(T::gaussPts) = level_ref_gauss_pts;

      EntityHandle fe_ent = T::numeredEntFiniteElementPtr->getEnt();
      {
        const EntityHandle *conn;
        int num_nodes;
        T::mField.get_moab().get_connectivity(fe_ent, conn, num_nodes, true);
        T::coords.resize(3 * num_nodes, false);
        CHKERR T::mField.get_moab().get_coords(conn, num_nodes, &T::coords[0]);
      }

      const int num_nodes = level_ref_gauss_pts.size2();
      T::mapGaussPts.resize(level_ref_gauss_pts.size2());

      FTensor::Index<'i', 3> i;
      FTensor::Tensor0<FTensor::PackPtr<double *, 1>> t_n(
          &*shape_functions.data().begin());
      FTensor::Tensor1<FTensor::PackPtr<double *, 1>, 3> t_coords(
          &verts_array[0][ver_count], &verts_array[1][ver_count],
          &verts_array[2][ver_count]);
      for (int gg = 0; gg != num_nodes; ++gg, ++ver_count) {

        T::mapGaussPts[gg] = start_vert_handle + ver_count;

        auto set_float_precision = [](const double x) {
          if (std::abs(x) < std::numeric_limits<float>::epsilon())
            return 0.;
          else
            return x;
        };

        t_coords(i) = 0;
        auto t_ele_coords = getFTensor1FromArray<3, 3>(T::coords);
        for (int nn = 0; nn != CN::VerticesPerEntity(type); ++nn) {
          t_coords(i) += t_n * t_ele_coords(i);
          ++t_ele_coords;
          ++t_n;
        }

        for (auto ii : {0, 1, 2})
          t_coords(ii) = set_float_precision(t_coords(ii));

        ++t_coords;
      }

      Tag th;
      int def_in_the_loop = -1;
      CHKERR T::postProcMesh.tag_get_handle(
          "NB_IN_THE_LOOP", 1, MB_TYPE_INTEGER, th,
          MB_TAG_CREAT | MB_TAG_SPARSE, &def_in_the_loop);

      commonData.tEts.clear();
      const int num_el = level_ref_ele.size1();
      const int num_nodes_on_ele = level_ref_ele.size2();
      auto start_e = start_ele_handle + ele_count;
      commonData.tEts = Range(start_e, start_e + num_el - 1);
      for (auto tt = 0; tt != level_ref_ele.size1(); ++tt, ++ele_count) {
        for (int nn = 0; nn != num_nodes_on_ele; ++nn) {
          conn[num_nodes_on_ele * ele_count + nn] =
              T::mapGaussPts[level_ref_ele(tt, nn)];
        }
      }

      const int n_in_the_loop = T::nInTheLoop;
      CHKERR T::postProcMesh.tag_clear_data(th, commonData.tEts,
                                            &n_in_the_loop);

      MoFEMFunctionReturn(0);
    };

    switch (type) {
    case MBTET:
      return set_gauss_pts(levelGaussPtsOnRefMeshTets, levelRefTets,
                           levelShapeFunctionsTets,

                           startingVertTetHandle, startingEleTetHandle,
                           verticesOnTetArrays, tetConn, countVertTet, countTet

      );
    case MBHEX:
      return set_gauss_pts(levelGaussPtsOnRefMeshHexes, levelRefHexes,
                           levelShapeFunctionsHexes,

                           startingVertHexHandle, startingEleHexHandle,
                           verticesOnHexArrays, hexConn, countVertHex, countHex

      );
    default:
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
              "Element type not implemented");
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
    if (pcomm_post_proc_mesh != NULL)
      delete pcomm_post_proc_mesh;

    CHKERR T::postProcMesh.delete_mesh();

    auto alloc_vertices_and_elements_on_post_proc_mesh = [&]() {
      MoFEMFunctionBegin;

      auto fe_ptr = this->problemPtr->numeredFiniteElementsPtr;

      auto miit =
          fe_ptr->template get<Composite_Name_And_Part_mi_tag>().lower_bound(
              boost::make_tuple(this->getFEName(), this->getLoFERank()));
      auto hi_miit =
          fe_ptr->template get<Composite_Name_And_Part_mi_tag>().upper_bound(
              boost::make_tuple(this->getFEName(), this->getHiFERank()));

      const int number_of_ents_in_the_loop = this->getLoopSize();
      if (std::distance(miit, hi_miit) != number_of_ents_in_the_loop) {
        SETERRQ(this->mField.get_comm(), MOFEM_DATA_INCONSISTENCY,
                "Wrong size of indicices. Inconsistent size number of iterated "
                "elements iterated by problem and from range.");
      }

      int nb_tet_vertices = 0;
      int nb_tets = 0;
      int nb_hex_vertices = 0;
      int nb_hexes = 0;

      for (; miit != hi_miit; ++miit) {
        auto type = (*miit)->getEntType();

        // Set pointer to element. So that getDataVectorDofsPtr in getMaxLevel
        // can work
        this->numeredEntFiniteElementPtr = *miit;

        bool add = true;
        if (this->exeTestHook) {
          add = this->exeTestHook(this);
        }

        if (add) {

          auto level = getMaxLevel();

          switch (type) {
          case MBTET:
            level = std::min(level, levelGaussPtsOnRefMeshTets.size() - 1);
            nb_tet_vertices += levelGaussPtsOnRefMeshTets[level].size2();
            nb_tets += levelRefTets[level].size1();
            break;
          case MBHEX:
            level = std::min(level, levelGaussPtsOnRefMeshHexes.size() - 1);
            nb_hex_vertices += levelGaussPtsOnRefMeshHexes[level].size2();
            nb_hexes += levelRefHexes[level].size1();
            break;
          default:
            SETERRQ(this->mField.get_comm(), MOFEM_DATA_INCONSISTENCY,
                    "Element type not implemented");
            break;
          }
        }
      }

      ReadUtilIface *iface;
      CHKERR this->postProcMesh.query_interface(iface);

      if (nb_tets) {
        CHKERR iface->get_node_coords(
            3, nb_tet_vertices, 0, startingVertTetHandle, verticesOnTetArrays);
        CHKERR iface->get_element_connect(nb_tets, levelRefTets[0].size2(),
                                          MBTET, 0, startingEleTetHandle,
                                          tetConn);
      }

      if (nb_hexes) {
        CHKERR iface->get_node_coords(
            3, nb_hex_vertices, 0, startingVertHexHandle, verticesOnHexArrays);
        CHKERR iface->get_element_connect(nb_hexes, levelRefHexes[0].size2(),
                                          MBHEX, 0, startingEleHexHandle,
                                          hexConn);
      }

      countTet = 0;
      countVertTet = 0;
      countHex = 0;
      countVertHex = 0;

      MoFEMFunctionReturn(0);
    };

    CHKERR alloc_vertices_and_elements_on_post_proc_mesh();

    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBeginHot;

    auto update_elements = [&]() {
      MoFEMFunctionBegin;
      ReadUtilIface *iface;
      CHKERR this->postProcMesh.query_interface(iface);

      if (countTet) {
        MOFEM_TAG_AND_LOG("SELF", Sev::noisy, "PostProc")
            << "Update tets " << countTet;

        MOFEM_TAG_AND_LOG("SELF", Sev::noisy, "PostProc")
            << "Nb nodes on tets " << levelRefTets[0].size2();

        CHKERR iface->update_adjacencies(startingEleTetHandle, countTet,
                                         levelRefTets[0].size2(), tetConn);
      }
      if (countHex) {
        MOFEM_TAG_AND_LOG("SELF", Sev::noisy, "PostProc")
            << "Update hexes " << countHex;
        CHKERR iface->update_adjacencies(startingEleHexHandle, countHex,
                                         levelRefHexes[0].size2(), hexConn);
      }
      MoFEMFunctionReturn(0);
    };

    auto resolve_shared_ents = [&]() {
      MoFEMFunctionBegin;
      ParallelComm *pcomm_post_proc_mesh =
          ParallelComm::get_pcomm(&(T::postProcMesh), MYPCOMM_INDEX);
      if (pcomm_post_proc_mesh == NULL) {
        // T::wrapRefMeshComm =
            // boost::make_shared<WrapMPIComm>(T::mField.get_comm(), false);
        pcomm_post_proc_mesh = new ParallelComm(
            &(T::postProcMesh),
            PETSC_COMM_WORLD /*(T::wrapRefMeshComm)->get_comm()*/);
      }

      Range edges;
      CHKERR T::postProcMesh.get_entities_by_type(0, MBEDGE, edges, false);
      CHKERR T::postProcMesh.delete_entities(edges);
      Range faces;
      CHKERR T::postProcMesh.get_entities_by_dimension(0, 2, faces, false);
      CHKERR T::postProcMesh.delete_entities(faces);

      Range ents;
      CHKERR T::postProcMesh.get_entities_by_dimension(0, 3, ents, false);

      int rank = T::mField.get_comm_rank();
      CHKERR T::postProcMesh.tag_clear_data(pcomm_post_proc_mesh->part_tag(),
                                            ents, &rank);

      CHKERR pcomm_post_proc_mesh->resolve_shared_ents(0);

      MoFEMFunctionReturn(0);
    };

    CHKERR resolve_shared_ents();
    CHKERR update_elements();

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
                          EntitiesFieldData::EntData &data) {
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
  std::vector<MatrixDouble> levelShapeFunctionsTets;
  std::vector<MatrixDouble> levelGaussPtsOnRefMeshTets;
  std::vector<ublas::matrix<int>> levelRefTets;
  std::vector<MatrixDouble> levelShapeFunctionsHexes;
  std::vector<MatrixDouble> levelGaussPtsOnRefMeshHexes;
  std::vector<ublas::matrix<int>> levelRefHexes;

  EntityHandle startingVertTetHandle;
  std::vector<double *> verticesOnTetArrays;
  EntityHandle startingEleTetHandle;
  EntityHandle *tetConn;
  int countTet;
  int countVertTet;

  EntityHandle startingVertHexHandle;
  std::vector<double *> verticesOnHexArrays;
  EntityHandle startingEleHexHandle;
  EntityHandle *hexConn;
  int countHex;
  int countVertHex;
};

/** \brief Post processing
 * \ingroup mofem_fs_post_proc
 */
struct PostProcVolumeOnRefinedMesh
    : public PostProcTemplateVolumeOnRefinedMesh<
          MoFEM::VolumeElementForcesAndSourcesCore> {

  using PostProcTemplateVolumeOnRefinedMesh<
      MoFEM::VolumeElementForcesAndSourcesCore>::
      PostProcTemplateVolumeOnRefinedMesh;
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

  // PointsMap3D_multiIndex pointsMap;

  MoFEMErrorCode setGaussPtsTrianglesOnly(int order_triangles_only);
  MoFEMErrorCode setGaussPtsThroughThickness(int order_thickness);
  MoFEMErrorCode generateReferenceElementMesh();

  // std::map<EntityHandle, EntityHandle> elementsMap;
  std::map<EntityHandle, std::vector<EntityHandle>> elementsMap;
  std::map<EntityHandle, std::vector<PointsMap3D_multiIndex>>
      pointsMapVectorMap;

  MoFEMErrorCode preProcess();
  MoFEMErrorCode postProcess();

  struct CommonData : PostProcCommonOnRefMesh::CommonDataForVolume {};
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
        sixNodePostProcTris(six_node_post_proc_tris), counterTris(0),
        counterQuads(0) {}

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

  template <int RANK>
  struct OpGetFieldValuesOnSkinImpl
      : public FaceElementForcesAndSourcesCore::UserDataOperator {

    moab::Interface &postProcMesh;
    std::vector<EntityHandle> &mapGaussPts;
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideOpFe;
    const std::string feVolName;
    boost::shared_ptr<MatrixDouble> matPtr;
    const std::string tagName;
    const std::string fieldName;
    const bool saveOnTag;

    OpGetFieldValuesOnSkinImpl(
        moab::Interface &post_proc_mesh,
        std::vector<EntityHandle> &map_gauss_pts, const std::string field_name,
        const std::string tag_name,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe,
        const std::string vol_fe_name, boost::shared_ptr<MatrixDouble> mat_ptr,
        bool save_on_tag)
        : FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL),
          postProcMesh(post_proc_mesh), mapGaussPts(map_gauss_pts),
          sideOpFe(side_fe), feVolName(vol_fe_name), matPtr(mat_ptr),
          tagName(tag_name), fieldName(field_name), saveOnTag(save_on_tag) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data);
  };

  typedef struct OpGetFieldValuesOnSkinImpl<3> OpGetFieldGradientValuesOnSkin;
  typedef struct OpGetFieldValuesOnSkinImpl<1> OpGetFieldValuesOnSkin;

  MoFEMErrorCode addFieldValuesGradientPostProcOnSkin(
      const std::string field_name, const std::string vol_fe_name = "dFE",
      boost::shared_ptr<MatrixDouble> grad_mat_ptr = nullptr,
      bool save_on_tag = true);

  MoFEMErrorCode addFieldValuesPostProcOnSkin(
      const std::string field_name, const std::string vol_fe_name = "dFE",
      boost::shared_ptr<MatrixDouble> mat_ptr = nullptr,
      bool save_on_tag = true);

private:
  MatrixDouble gaussPtsTri; ///<  Gauss points coordinates on reference triangle
  MatrixDouble gaussPtsQuad; ///< Gauss points coordinates on reference quad

  EntityHandle *triConn;  ///< Connectivity for created tri elements
  EntityHandle *quadConn; ///< Connectivity for created quad elements
  EntityHandle
      startingVertTriHandle; ///< Starting handle for vertices on triangles
  EntityHandle
      startingVertQuadHandle; ///< Starting handle for vertices on quads
  std::vector<double *>
      verticesOnTriArrays; /// pointers to memory allocated by MoAB for
                           /// storing X, Y, and Z coordinates
  std::vector<double *>
      verticesOnQuadArrays; /// pointers to memory allocated by MoAB for
                            /// storing X, Y, and Z coordinates

  EntityHandle
      startingEleTriHandle; ///< Starting handle for triangles post proc
  EntityHandle startingEleQuadHandle; ///< Starting handle for quads post proc

  int numberOfTriangles; ///< Number of triangles to  create
  int numberOfQuads;     ///< NUmber of quads to create
  int counterTris;
  int counterQuads;
};

/**
 * @brief Postprocess 2d face elements
 *
 */
struct PostProcFaceOnRefinedMeshFor2D : public PostProcFaceOnRefinedMesh {

  using PostProcFaceOnRefinedMesh::PostProcFaceOnRefinedMesh;
};

using EdgeEleBasePostProc = MoFEM::EdgeElementForcesAndSourcesCore;

/**
 * \brief Postprocess on edge
 *
 * \ingroup mofem_fs_post_proc
 */
struct PostProcEdgeOnRefinedMesh
    : public PostProcTemplateOnRefineMesh<EdgeEleBasePostProc> {

  bool sixNodePostProcTris;

  PostProcEdgeOnRefinedMesh(MoFEM::Interface &m_field,
                            bool six_node_post_proc_tris = true)
      : PostProcTemplateOnRefineMesh<EdgeEleBasePostProc>(m_field),
        sixNodePostProcTris(six_node_post_proc_tris) {}

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

/**
 * @brief Post post-proc data at points from hash maps
 *
 * @tparam DIM1 dimension of vector in data_map_vec  and first column of
 * data_map_may
 * @tparam DIM2 dimension of second column in data_map_mat
 */
template <int DIM1, int DIM2>
struct OpPostProcMap : public ForcesAndSourcesCore::UserDataOperator {

  using DataMapVec = std::map<std::string, boost::shared_ptr<VectorDouble>>;
  using DataMapMat = std::map<std::string, boost::shared_ptr<MatrixDouble>>;

  /**
   * @brief Construct a new OpPostProcMap object
   *
   * @param post_proc_mesh postprocessing mesh
   * @param map_gauss_pts map of gauss points to nodes of postprocessing mesh
   * @param data_map_scalar hash map of scalar values (string is name of the
   * tag)
   * @param data_map_vec hash map of vector values
   * @param data_map_mat hash map of second order tensor values
   * @param data_symm_map_mat hash map of symmetric second order tensor values
   */
  OpPostProcMap(moab::Interface &post_proc_mesh,
                std::vector<EntityHandle> &map_gauss_pts,
                DataMapVec data_map_scalar, DataMapMat data_map_vec,
                DataMapMat data_map_mat, DataMapMat data_symm_map_mat)
      : ForcesAndSourcesCore::UserDataOperator(
            NOSPACE, ForcesAndSourcesCore::UserDataOperator::OPSPACE),
        postProcMesh(post_proc_mesh), mapGaussPts(map_gauss_pts),
        dataMapScalar(data_map_scalar), dataMapVec(data_map_vec),
        dataMapMat(data_map_mat), dataMapSymmMat(data_symm_map_mat) {
    // Operator is only executed for vertices
    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
  }
  MoFEMErrorCode doWork(int side, EntityType type,
                        EntitiesFieldData::EntData &data);

private:
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;
  DataMapVec dataMapScalar;
  DataMapMat dataMapVec;
  DataMapMat dataMapMat;
  DataMapMat dataMapSymmMat;
};

template <int DIM1, int DIM2>
MoFEMErrorCode
OpPostProcMap<DIM1, DIM2>::doWork(int side, EntityType type,
                                  EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  std::array<double, 9> def;
  std::fill(def.begin(), def.end(), 0);

  auto get_tag = [&](const std::string name, size_t size) {
    Tag th;
    CHKERR postProcMesh.tag_get_handle(name.c_str(), size, MB_TYPE_DOUBLE, th,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       def.data());
    return th;
  };

  MatrixDouble3by3 mat(3, 3);

  auto set_vector_3d = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != DIM1; ++r)
      mat(0, r) = t(r);
    return mat;
  };

  auto set_matrix_3d = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != DIM1; ++r)
      for (size_t c = 0; c != DIM2; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_matrix_symm_3d = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != DIM1; ++r)
      for (size_t c = 0; c != DIM1; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_scalar = [&](auto t) -> MatrixDouble3by3 & {
    mat.clear();
    mat(0, 0) = t;
    return mat;
  };

  auto set_float_precision = [](const double x) {
    if (std::abs(x) < std::numeric_limits<float>::epsilon())
      return 0.;
    else
      return x;
  };

  auto set_tag = [&](auto th, auto gg, MatrixDouble3by3 &mat) {
    for (auto &v : mat.data())
      v = set_float_precision(v);
    return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
                                     &*mat.data().begin());
  };

  for (auto &m : dataMapScalar) {
    auto th = get_tag(m.first, 1);
    auto t_scl = getFTensor0FromVec(*m.second);
    auto nb_integration_pts = getGaussPts().size2();
    size_t gg = 0;
    for (int gg = 0; gg != nb_integration_pts; ++gg) {
      CHKERR set_tag(th, gg, set_scalar(t_scl));
      ++t_scl;
    }
  }

  for (auto &m : dataMapVec) {
    auto th = get_tag(m.first, 3);
    auto t_vec = getFTensor1FromMat<DIM1>(*m.second);
    auto nb_integration_pts = getGaussPts().size2();
    size_t gg = 0;
    for (int gg = 0; gg != nb_integration_pts; ++gg) {
      CHKERR set_tag(th, gg, set_vector_3d(t_vec));
      ++t_vec;
    }
  }

  for (auto &m : dataMapMat) {
    auto th = get_tag(m.first, 9);
    auto t_mat = getFTensor2FromMat<DIM1, DIM2>(*m.second);
    auto nb_integration_pts = getGaussPts().size2();
    size_t gg = 0;
    for (int gg = 0; gg != nb_integration_pts; ++gg) {
      CHKERR set_tag(th, gg, set_matrix_3d(t_mat));
      ++t_mat;
    }
  }

  for (auto &m : dataMapSymmMat) {
    auto th = get_tag(m.first, 9);
    auto t_mat = getFTensor2SymmetricFromMat<DIM1>(*m.second);
    auto nb_integration_pts = getGaussPts().size2();
    size_t gg = 0;
    for (int gg = 0; gg != nb_integration_pts; ++gg) {
      CHKERR set_tag(th, gg, set_matrix_symm_3d(t_mat));
      ++t_mat;
    }
  }

  MoFEMFunctionReturn(0);
}

#endif //__POSTPROC_ON_REF_MESH_HPP

/**
 * \defgroup mofem_fs_post_proc Post Process
 * \ingroup user_modules
 **/
