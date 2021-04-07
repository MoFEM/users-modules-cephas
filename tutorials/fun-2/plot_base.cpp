/**
 * \file approximation.cpp
 * \example approximation.cpp
 *
 * Using PipelineManager interface calculate the divergence of base functions,
 * and integral of flux on the boundary. Since the h-div space is used, volume
 * integral and boundary integral should give the same result.
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

#include <MoFEM.hpp>

using namespace MoFEM;

static char help[] = "...\n\n";

#include <BasicFiniteElements.hpp>

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = FaceElementForcesAndSourcesCoreBase;
  using DomainEleOp = DomainEle::UserDataOperator;
  using PostProcEle = PostProcFaceOnRefinedMesh;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = VolumeElementForcesAndSourcesCoreBase;
  using DomainEleOp = DomainEle::UserDataOperator;
  using PostProcEle = PostProcVolumeOnRefinedMesh;
};

constexpr int SPACE_DIM =
    EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh

using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = ElementsAndOps<SPACE_DIM>::DomainEleOp;
using PostProcEle = ElementsAndOps<SPACE_DIM>::PostProcEle;

struct MyPostProc : public PostProcEle {
  using PostProcEle::PostProcEle;

  MoFEMErrorCode generateReferenceElementMesh();
  MoFEMErrorCode setGaussPts(int order);

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

  //! [Approximated function]
  static double approxFunction(const double x, const double y, const double z) {
    return sin(x * 10.) * cos(y * 10.);
  }
  //! [Approximated function]

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode setIntegrationRules();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();
  MoFEMErrorCode checkResults();
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
    for(auto d : {1,2})
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

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
  MoFEMFunctionBegin;
  // Add field

  const FieldApproximationBase base = AINSWORTH_LEGENDRE_BASE;
  CHKERR simpleInterface->addDomainField("U", H1, base, 1);

  constexpr int order = 4;
  CHKERR simpleInterface->setFieldOrder("U", order);
  CHKERR simpleInterface->setUp();

  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Set integration rule]
MoFEMErrorCode Example::setIntegrationRules() {
  MoFEMFunctionBegin;

  auto rule = [](int, int, int p) -> int { return -1; };

  PipelineManager *pipeline_mng = mField.getInterface<PipelineManager>();
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(rule);

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
  post_proc_fe->addFieldValuesPostProc("U");
  pipeline_mng->getDomainRhsFE() = post_proc_fe;

  Range edges;
  CHKERR mField.get_moab().get_entities_by_type(0, MBEDGE, edges, true);

  CHKERR mField.getInterface<FieldBlas>()->setField(0, MBEDGE, edges, "U");

  size_t nb = 0;
  auto dofs_ptr = mField.get_dofs();
  for (auto dof_ptr : (*dofs_ptr)) {
    MOFEM_LOG("PLOTBASE", Sev::verbose) << *dof_ptr;
    auto &val = const_cast<double &>(dof_ptr->getFieldData());
    val = 1;
    CHKERR pipeline_mng->loopFiniteElements();
    CHKERR post_proc_fe->writeFile(
        "out_base_dof_" + boost::lexical_cast<std::string>(nb) + ".h5m");
    CHKERR post_proc_fe->postProcMesh.delete_mesh();
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

  CHKERR MoFEM::Core::Finalize();
}

MoFEMErrorCode MyPostProc::generateReferenceElementMesh() {
  MoFEMFunctionBegin;
  moab::Core core_ref;
  moab::Interface &moab_ref = core_ref;

  // Load mesh
  if (SPACE_DIM == 2)
    CHKERR moab_ref.load_file("ref_mesh2d.h5m", 0, "");
  else if (SPACE_DIM == 3)
    CHKERR moab_ref.load_file("ref_mesh3d.h5m", 0, "");
  else
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
            "Dimension not implemented");

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
    for(auto d :{0,1,2})
      gaussPts(d, gg) = coords[d];
    nodes_pts_map[*nit] = gg;
  }

  // Set size of adjacency matrix (note ho order nodes 3 nodes and 3 nodes on
  // edges)
  refEleMap.resize(elems.size(), 2 * (SPACE_DIM + 1));

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

  // Calculate sheape functions
  shapeFunctions.resize(elem_nodes.size(), SPACE_DIM + 1);
  if (SPACE_DIM == 2)
    CHKERR Tools::shapeFunMBTRI(&*shapeFunctions.data().begin(),
                                &gaussPts(0, 0), &gaussPts(1, 0),
                                elem_nodes.size());

  if (SPACE_DIM == 3)
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
            "Not yet simplemented version for 3d");

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MyPostProc::setGaussPts(int order) {
  MoFEMFunctionBegin;

  // Create physical nodes
  ReadUtilIface *iface;
  CHKERR postProcMesh.query_interface(iface);

  const int num_nodes = gaussPts.size2();
  std::vector<double *> arrays;
  EntityHandle startv;
  CHKERR iface->get_node_coords(3, num_nodes, 0, startv, arrays);
  mapGaussPts.resize(gaussPts.size2());
  for (int gg = 0; gg != num_nodes; ++gg)
    mapGaussPts[gg] = startv + gg;

  Tag th;
  int def_in_the_loop = -1;
  CHKERR postProcMesh.tag_get_handle("NB_IN_THE_LOOP", 1, MB_TYPE_INTEGER,
                                        th, MB_TAG_CREAT | MB_TAG_SPARSE,
                                        &def_in_the_loop);

  // Create physical elements

  const int num_el = refEleMap.size1();
  const int num_nodes_on_ele = refEleMap.size2();
  
  EntityHandle starte;
  EntityHandle *conn;

  if (SPACE_DIM == 2)
    CHKERR iface->get_element_connect(num_el, num_nodes_on_ele, MBTRI, 0,
                                      starte, conn);
  else if (SPACE_DIM == 3)
    CHKERR iface->get_element_connect(num_el, num_nodes_on_ele, MBTET, 0,
                                      starte, conn);
  else
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
            "Dimension not implemented");

  for (unsigned int tt = 0; tt != refEleMap.size1(); ++tt) {
    for (int nn = 0; nn != num_nodes_on_ele; ++nn)
      conn[num_nodes_on_ele * tt + nn] = mapGaussPts[refEleMap(tt, nn)];
  }
  CHKERR iface->update_adjacencies(starte, num_el, num_nodes_on_ele, conn);
  auto physical_elements = Range(starte, starte + num_el - 1);
  CHKERR postProcMesh.tag_clear_data(th, physical_elements, &(nInTheLoop));

  EntityHandle fe_ent = numeredEntFiniteElementPtr->getEnt();
  coords.resize(12, false);
  {
    const EntityHandle *conn;
    int num_nodes;
    mField.get_moab().get_connectivity(fe_ent, conn, num_nodes, true);
    CHKERR mField.get_moab().get_coords(conn, num_nodes, &coords[0]);
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
    for (int nn = 0; nn != SPACE_DIM + 1; ++nn) {
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
