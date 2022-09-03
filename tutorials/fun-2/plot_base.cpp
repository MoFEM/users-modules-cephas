/**
 * \file plot_base.cpp
 * \example plot_base.cpp
 *
 * Utility for plotting base functions for different spaces, polynomial bases
 */

#include <MoFEM.hpp>

using namespace MoFEM;

static char help[] = "...\n\n";

#include <BasicFiniteElements.hpp>

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = FaceElementForcesAndSourcesCore;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = VolumeElementForcesAndSourcesCore;
};

constexpr int SPACE_DIM =
    EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh

using EntData = EntitiesFieldData::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

struct MyPostProc : public PostProcEle {
  using PostProcEle::PostProcEle;

  MoFEMErrorCode generateReferenceElementMesh();
  MoFEMErrorCode setGaussPts(int order);

  MoFEMErrorCode preProcess();
  MoFEMErrorCode postProcess();

protected:
  ublas::matrix<int> refEleMap;
  MatrixDouble shapeFunctions;
};

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;
  Simple *simpleInterface;

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();
  MoFEMErrorCode checkResults();

  FieldApproximationBase base;
  FieldSpace space;
};

//! [Run programme]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR setIntegrationRules();
  CHKERR createCommonData();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR outputResults();
  CHKERR checkResults();
  MoFEMFunctionReturn(0);
}
//! [Run programme]

//! [Read mesh]
MoFEMErrorCode Example::readMesh() {
  MoFEMFunctionBegin;

  PetscBool load_file = PETSC_FALSE;
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-load_file", &load_file,
                             PETSC_NULL);

  if (load_file == PETSC_FALSE) {

    auto &moab = mField.get_moab();

    if (SPACE_DIM == 3) {

      // create one tet
      double tet_coords[] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
      EntityHandle nodes[4];
      for (int nn = 0; nn < 4; nn++) {
        CHKERR moab.create_vertex(&tet_coords[3 * nn], nodes[nn]);
      }
      EntityHandle tet;
      CHKERR moab.create_element(MBTET, nodes, 4, tet);
      Range adj;
      for (auto d : {1, 2})
        CHKERR moab.get_adjacencies(&tet, 1, d, true, adj);
    }

    if (SPACE_DIM == 2) {

      // create one triangle
      double tri_coords[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
      EntityHandle nodes[3];
      for (int nn = 0; nn < 3; nn++) {
        CHKERR moab.create_vertex(&tri_coords[3 * nn], nodes[nn]);
      }
      EntityHandle tri;
      CHKERR moab.create_element(MBTRI, nodes, 3, tri);
      Range adj;
      CHKERR moab.get_adjacencies(&tri, 1, 1, true, adj);
    }

    CHKERR mField.rebuild_database();
    CHKERR mField.getInterface(simpleInterface);
    simpleInterface->setDim(SPACE_DIM);

    // Add all elements to database
    CHKERR mField.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, SPACE_DIM, simpleInterface->getBitRefLevel());

  } else {

    CHKERR mField.getInterface(simpleInterface);
    CHKERR simpleInterface->getOptions();
    CHKERR simpleInterface->loadFile();
  }

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
  MoFEMFunctionBegin;
  // Add field

  // Declare elements
  enum bases { AINSWORTH, AINSWORTH_LOBATTO, DEMKOWICZ, BERNSTEIN, LASBASETOP };
  const char *list_bases[] = {"ainsworth", "ainsworth_lobatto", "demkowicz",
                              "bernstein"};

  PetscBool flg;
  PetscInt choice_base_value = AINSWORTH;
  CHKERR PetscOptionsGetEList(PETSC_NULL, NULL, "-base", list_bases, LASBASETOP,
                              &choice_base_value, &flg);
  if (flg != PETSC_TRUE)
    SETERRQ(PETSC_COMM_SELF, MOFEM_IMPOSIBLE_CASE, "base not set");
  base = AINSWORTH_LEGENDRE_BASE;
  if (choice_base_value == AINSWORTH)
    base = AINSWORTH_LEGENDRE_BASE;
  if (choice_base_value == AINSWORTH_LOBATTO)
    base = AINSWORTH_LOBATTO_BASE;
  else if (choice_base_value == DEMKOWICZ)
    base = DEMKOWICZ_JACOBI_BASE;
  else if (choice_base_value == BERNSTEIN)
    base = AINSWORTH_BERNSTEIN_BEZIER_BASE;

  enum spaces { H1SPACE, L2SPACE, HCURLSPACE, HDIVSPACE, LASBASETSPACE };
  const char *list_spaces[] = {"h1", "l2", "hcurl", "hdiv"};
  PetscInt choice_space_value = H1SPACE;
  CHKERR PetscOptionsGetEList(PETSC_NULL, NULL, "-space", list_spaces,
                              LASBASETSPACE, &choice_space_value, &flg);
  if (flg != PETSC_TRUE)
    SETERRQ(PETSC_COMM_SELF, MOFEM_IMPOSIBLE_CASE, "space not set");
  space = H1;
  if (choice_space_value == H1SPACE)
    space = H1;
  else if (choice_space_value == L2SPACE)
    space = L2;
  else if (choice_space_value == HCURLSPACE)
    space = HCURL;
  else if (choice_space_value == HDIVSPACE)
    space = HDIV;

  CHKERR simpleInterface->addDomainField("U", space, base, 1);

  int order = 3;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simpleInterface->setFieldOrder("U", order);
  CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Set integration rule]
MoFEMErrorCode Example::setIntegrationRules() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}
//! [Set integration rule]

//! [Create common data]
MoFEMErrorCode Example::createCommonData() { return 0; }
//! [Create common data]

//! [Boundary condition]
MoFEMErrorCode Example::boundaryCondition() { return 0; }
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystem() { return 0; }
//! [Push operators to pipeline]

//! [Solve]
MoFEMErrorCode Example::solveSystem() { return 0; }

//! [Solve]
MoFEMErrorCode Example::outputResults() {
  MoFEMFunctionBegin;

  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  pipeline_mng->getDomainLhsFE().reset();

  auto post_proc_fe = boost::make_shared<MyPostProc>(mField);
  post_proc_fe->generateReferenceElementMesh();
  pipeline_mng->getDomainRhsFE() = post_proc_fe;

  if (SPACE_DIM == 2) {
    if (space == HCURL) {
      auto jac_ptr = boost::make_shared<MatrixDouble>();
      post_proc_fe->getOpPtrVector().push_back(
          new OpCalculateHOJacForFace(jac_ptr));
      post_proc_fe->getOpPtrVector().push_back(new OpMakeHdivFromHcurl());
      post_proc_fe->getOpPtrVector().push_back(
          new OpSetContravariantPiolaTransformOnFace2D(jac_ptr));
    }
  }

  switch (space) {
  case H1:
  case L2:

  {

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    auto u_ptr = boost::make_shared<VectorDouble>();
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("U", u_ptr));
    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(

            post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

            {{"U", u_ptr}},

            {},

            {},

            {}

            )

    );
  } break;
  case HCURL:
  case HDIV:

  {
    using OpPPMap = OpPostProcMapInMoab<3, 3>;

    auto u_ptr = boost::make_shared<MatrixDouble>();
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateHVecVectorField<3>("U", u_ptr));

    post_proc_fe->getOpPtrVector().push_back(

        new OpPPMap(

            post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

            {},

            {{"U", u_ptr}},

            {},

            {}

            )

    );
  } break;
  default:
    break;
  }

  auto scale_tag_val = [&]() {
    MoFEMFunctionBegin;
    auto &post_proc_mesh = post_proc_fe->getPostProcMesh();
    Range nodes;
    CHKERR post_proc_mesh.get_entities_by_type(0, MBVERTEX, nodes);
    Tag th;
    CHKERR post_proc_mesh.tag_get_handle("U", th);
    int length;
    CHKERR post_proc_mesh.tag_get_length(th, length);
    std::vector<double> data(nodes.size() * length);
    CHKERR post_proc_mesh.tag_get_data(th, nodes, &*data.begin());
    double max_v = 0;
    for (int i = 0; i != nodes.size(); ++i) {
      double v = 0;
      for (int d = 0; d != length; ++d)
        v += pow(data[length * i + d], 2);
      v = std::sqrt(v);
      max_v = std::max(max_v, v);
    }
    for (auto &v : data)
      v /= max_v;
    CHKERR post_proc_mesh.tag_set_data(th, nodes, &*data.begin());
    MoFEMFunctionReturn(0);
  };

  size_t nb = 0;
  auto dofs_ptr = mField.get_dofs();

  for (auto dof_ptr : (*dofs_ptr)) {
    MOFEM_LOG("PLOTBASE", Sev::verbose) << *dof_ptr;
    auto &val = const_cast<double &>(dof_ptr->getFieldData());
    val = 1;
    CHKERR pipeline_mng->loopFiniteElements();
    CHKERR scale_tag_val();
    CHKERR post_proc_fe->writeFile(
        "out_base_dof_" + boost::lexical_cast<std::string>(nb) + ".h5m");
    CHKERR post_proc_fe->getPostProcMesh().delete_mesh();
    val = 0;
    ++nb;
  };

  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

//! [Check results]
MoFEMErrorCode Example::checkResults() { return 0; }
//! [Check results]

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    //! [Register MoFEM discrete manager in PETSc]
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);
    //! [Register MoFEM discrete manager in PETSc

    // Add logging channel for example
    auto core_log = logging::core::get();
    core_log->add_sink(
        LogManager::createSink(LogManager::getStrmWorld(), "PLOTBASE"));
    LogManager::setLog("PLOTBASE");
    MOFEM_LOG_TAG("PLOTBASE", "plotbase");

    //! [Create MoAB]
    moab::Core mb_instance;              ///< mesh database
    moab::Interface &moab = mb_instance; ///< mesh database interface
    //! [Create MoAB]

    //! [Create MoFEM]
    MoFEM::Core core(moab);           ///< finite element database
    MoFEM::Interface &m_field = core; ///< finite element database insterface
    //! [Create MoFEM]

    //! [Example]
    Example ex(m_field);
    CHKERR ex.runProblem();
    //! [Example]
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}

MoFEMErrorCode MyPostProc::generateReferenceElementMesh() {
  MoFEMFunctionBegin;
  moab::Core core_ref;
  moab::Interface &moab_ref = core_ref;

  char ref_mesh_file_name[255];

  if (SPACE_DIM == 2)
    strcpy(ref_mesh_file_name, "ref_mesh2d.h5m");
  else if (SPACE_DIM == 3)
    strcpy(ref_mesh_file_name, "ref_mesh3d.h5m");
  else
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
            "Dimension not implemented");

  CHKERR PetscOptionsGetString(PETSC_NULL, "", "-ref_file", ref_mesh_file_name,
                               255, PETSC_NULL);
  CHKERR moab_ref.load_file(ref_mesh_file_name, 0, "");

  // Get elements
  Range elems;
  CHKERR moab_ref.get_entities_by_dimension(0, SPACE_DIM, elems);

  // Add mid-nodes on edges
  EntityHandle meshset;
  CHKERR moab_ref.create_meshset(MESHSET_SET, meshset);
  CHKERR moab_ref.add_entities(meshset, elems);
  CHKERR moab_ref.convert_entities(meshset, true, false, false);
  CHKERR moab_ref.delete_entities(&meshset, 1);

  // Get nodes on the mesh
  Range elem_nodes;
  CHKERR moab_ref.get_connectivity(elems, elem_nodes, false);

  // Map node entity and Gauss pint number
  std::map<EntityHandle, int> nodes_pts_map;

  // Set gauss points coordinates from the reference mesh
  gaussPts.resize(SPACE_DIM + 1, elem_nodes.size(), false);
  gaussPts.clear();
  Range::iterator nit = elem_nodes.begin();
  for (int gg = 0; nit != elem_nodes.end(); nit++, gg++) {
    double coords[3];
    CHKERR moab_ref.get_coords(&*nit, 1, coords);
    for (auto d : {0, 1, 2})
      gaussPts(d, gg) = coords[d];
    nodes_pts_map[*nit] = gg;
  }

  if (SPACE_DIM == 2) {
    // Set size of adjacency matrix (note ho order nodes 3 nodes and 3 nodes on
    // edges)
    refEleMap.resize(elems.size(), 3 + 3);
  } else if (SPACE_DIM == 3) {
    refEleMap.resize(elems.size(), 4 + 6);
  }

  // Set adjacency matrix
  Range::iterator tit = elems.begin();
  for (int tt = 0; tit != elems.end(); ++tit, ++tt) {
    const EntityHandle *conn;
    int num_nodes;
    CHKERR moab_ref.get_connectivity(*tit, conn, num_nodes, false);
    for (int nn = 0; nn != num_nodes; ++nn) {
      refEleMap(tt, nn) = nodes_pts_map[conn[nn]];
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MyPostProc::setGaussPts(int order) {
  MoFEMFunctionBegin;

  const int num_nodes = gaussPts.size2();

  // Calculate shape functions

  switch (numeredEntFiniteElementPtr->getEntType()) {
  case MBTRI:
    shapeFunctions.resize(num_nodes, 3);
    CHKERR Tools::shapeFunMBTRI(&*shapeFunctions.data().begin(),
                                &gaussPts(0, 0), &gaussPts(1, 0), num_nodes);
    break;
  case MBQUAD: {
    shapeFunctions.resize(num_nodes, 4);
    for (int gg = 0; gg != num_nodes; gg++) {
      double ksi = gaussPts(0, gg);
      double eta = gaussPts(1, gg);
      shapeFunctions(gg, 0) = N_MBQUAD0(ksi, eta);
      shapeFunctions(gg, 1) = N_MBQUAD1(ksi, eta);
      shapeFunctions(gg, 2) = N_MBQUAD2(ksi, eta);
      shapeFunctions(gg, 3) = N_MBQUAD3(ksi, eta);
    }
  } break;
  case MBTET: {
    shapeFunctions.resize(num_nodes, 8);
    CHKERR Tools::shapeFunMBTET(&*shapeFunctions.data().begin(),
                                &gaussPts(0, 0), &gaussPts(1, 0),
                                &gaussPts(2, 0), num_nodes);
  } break;
  case MBHEX: {
    shapeFunctions.resize(num_nodes, 8);
    for (int gg = 0; gg != num_nodes; gg++) {
      double ksi = gaussPts(0, gg);
      double eta = gaussPts(1, gg);
      double zeta = gaussPts(2, gg);
      shapeFunctions(gg, 0) = N_MBHEX0(ksi, eta, zeta);
      shapeFunctions(gg, 1) = N_MBHEX1(ksi, eta, zeta);
      shapeFunctions(gg, 2) = N_MBHEX2(ksi, eta, zeta);
      shapeFunctions(gg, 3) = N_MBHEX3(ksi, eta, zeta);
      shapeFunctions(gg, 4) = N_MBHEX4(ksi, eta, zeta);
      shapeFunctions(gg, 5) = N_MBHEX5(ksi, eta, zeta);
      shapeFunctions(gg, 6) = N_MBHEX6(ksi, eta, zeta);
      shapeFunctions(gg, 7) = N_MBHEX7(ksi, eta, zeta);
    }
  } break;
  default:
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
            "Not implemented element type");
  }

  // Create physical nodes

  // MoAB interface allowing for creating nodes and elements in the bulk
  ReadUtilIface *iface;
  CHKERR postProcMesh.query_interface(iface);

  std::vector<double *> arrays; /// pointers to memory allocated by MoAB for
                                /// storing X, Y, and Z coordinates
  EntityHandle startv;          // Starting handle for vertex
  // Allocate memory for num_nodes, and return starting handle, and access to
  // memort.
  CHKERR iface->get_node_coords(3, num_nodes, 0, startv, arrays);

  mapGaussPts.resize(gaussPts.size2());
  for (int gg = 0; gg != num_nodes; ++gg)
    mapGaussPts[gg] = startv + gg;

  Tag th;
  int def_in_the_loop = -1;
  CHKERR postProcMesh.tag_get_handle("NB_IN_THE_LOOP", 1, MB_TYPE_INTEGER, th,
                                     MB_TAG_CREAT | MB_TAG_SPARSE,
                                     &def_in_the_loop);

  // Create physical elements

  const int num_el = refEleMap.size1();
  const int num_nodes_on_ele = refEleMap.size2();

  EntityHandle starte; // Starting handle to first created element
  EntityHandle *conn;  // Access to MOAB memory with connectivity of elements

  // Create tris/tets in the bulk in MoAB database
  if (SPACE_DIM == 2)
    CHKERR iface->get_element_connect(num_el, num_nodes_on_ele, MBTRI, 0,
                                      starte, conn);
  else if (SPACE_DIM == 3)
    CHKERR iface->get_element_connect(num_el, num_nodes_on_ele, MBTET, 0,
                                      starte, conn);
  else
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
            "Dimension not implemented");

  // At this point elements (memory for elements) is allocated, at code bellow
  // actual connectivity of elements is set.
  for (unsigned int tt = 0; tt != refEleMap.size1(); ++tt) {
    for (int nn = 0; nn != num_nodes_on_ele; ++nn)
      conn[num_nodes_on_ele * tt + nn] = mapGaussPts[refEleMap(tt, nn)];
  }

  // Finalise elements creation. At that point MOAB updates adjacency tables,
  // and elements are ready to use.
  CHKERR iface->update_adjacencies(starte, num_el, num_nodes_on_ele, conn);

  auto physical_elements = Range(starte, starte + num_el - 1);
  CHKERR postProcMesh.tag_clear_data(th, physical_elements, &(nInTheLoop));

  EntityHandle fe_ent = numeredEntFiniteElementPtr->getEnt();
  int fe_num_nodes;
  {
    const EntityHandle *conn;
    mField.get_moab().get_connectivity(fe_ent, conn, fe_num_nodes, true);
    coords.resize(3 * fe_num_nodes, false);
    CHKERR mField.get_moab().get_coords(conn, fe_num_nodes, &coords[0]);
  }

  // Set physical coordinates to physical nodes
  FTensor::Index<'i', 3> i;
  FTensor::Tensor0<FTensor::PackPtr<double *, 1>> t_n(
      &*shapeFunctions.data().begin());

  FTensor::Tensor1<FTensor::PackPtr<double *, 1>, 3> t_coords(
      arrays[0], arrays[1], arrays[2]);
  const double *t_coords_ele_x = &coords[0];
  const double *t_coords_ele_y = &coords[1];
  const double *t_coords_ele_z = &coords[2];
  for (int gg = 0; gg != num_nodes; ++gg) {
    FTensor::Tensor1<FTensor::PackPtr<const double *, 3>, 3> t_ele_coords(
        t_coords_ele_x, t_coords_ele_y, t_coords_ele_z);
    t_coords(i) = 0;
    for (int nn = 0; nn != fe_num_nodes; ++nn) {
      t_coords(i) += t_n * t_ele_coords(i);
      for (auto ii : {0, 1, 2})
        if (std::abs(t_coords(ii)) < std::numeric_limits<float>::epsilon())
          t_coords(ii) = 0;
      ++t_ele_coords;
      ++t_n;
    }
    ++t_coords;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MyPostProc::preProcess() {
  MoFEMFunctionBegin;
  moab::Interface &moab = coreMesh;
  ParallelComm *pcomm_post_proc_mesh =
      ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
  if (pcomm_post_proc_mesh != NULL)
    delete pcomm_post_proc_mesh;
  MoFEMFunctionReturn(0);
};

MoFEMErrorCode MyPostProc::postProcess() {
  MoFEMFunctionBeginHot;

  auto resolve_shared_ents = [&]() {
    MoFEMFunctionBegin;

    ParallelComm *pcomm_post_proc_mesh =
        ParallelComm::get_pcomm(&(postProcMesh), MYPCOMM_INDEX);
    if (pcomm_post_proc_mesh == NULL) {
      // wrapRefMeshComm =
      // boost::make_shared<WrapMPIComm>(T::mField.get_comm(), false);
      pcomm_post_proc_mesh = new ParallelComm(
          &(postProcMesh),
          PETSC_COMM_WORLD /*(T::wrapRefMeshComm)->get_comm()*/);
    }

    CHKERR pcomm_post_proc_mesh->resolve_shared_ents(0);

    MoFEMFunctionReturn(0);
  };

  CHKERR resolve_shared_ents();

  MoFEMFunctionReturnHot(0);
}