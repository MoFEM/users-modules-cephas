/** \file mesh_cut.cpp
 * \brief test for mesh cut interface
 * \example mesh_cut.cpp
 *
 * \ingroup mesh_cut
 */



#include <MoFEM.hpp>

using namespace MoFEM;

static char help[] = "mesh cutting\n\n";

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    PetscBool flg_myfile = PETSC_TRUE;
    char mesh_file_name[255];
    int surface_side_set = 200;
    PetscBool flg_vol_block_set;
    int vol_block_set = 1;
    int edges_block_set = 1;
    int vertex_block_set = 2;
    PetscBool flg_shift;
    double shift[] = {0, 0, 0};
    int nmax = 3;
    int fraction_level = 2;
    PetscBool squash_bits = PETSC_TRUE;
    PetscBool set_coords = PETSC_TRUE;
    PetscBool output_vtk = PETSC_TRUE;
    int create_surface_side_set = 201;
    PetscBool flg_create_surface_side_set;
    int nb_ref_cut = 0;
    int nb_ref_trim = 0;
    PetscBool flg_tol;
    double tol[] = {1e-2, 2e-1, 2e-1};
    int nmax_tol = 3;
    PetscBool debug = PETSC_FALSE;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Mesh cut options", "none");
    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_myfile);
    CHKERR PetscOptionsInt("-surface_side_set", "surface side set", "",
                           surface_side_set, &surface_side_set, PETSC_NULL);
    CHKERR PetscOptionsInt("-vol_block_set", "volume side set", "",
                           vol_block_set, &vol_block_set, &flg_vol_block_set);
    CHKERR PetscOptionsInt("-edges_block_set", "edges block set", "",
                           edges_block_set, &edges_block_set, PETSC_NULL);
    CHKERR PetscOptionsInt("-vertex_block_set", "vertex block set", "",
                           vertex_block_set, &vertex_block_set, PETSC_NULL);
    CHKERR PetscOptionsRealArray("-shift", "shift surface by vector", "", shift,
                                 &nmax, &flg_shift);
    CHKERR PetscOptionsInt("-fraction_level", "fraction of merges merged", "",
                           fraction_level, &fraction_level, PETSC_NULL);
    CHKERR PetscOptionsBool("-squash_bits", "true to squash bits at the end",
                            "", squash_bits, &squash_bits, PETSC_NULL);
    CHKERR PetscOptionsBool("-set_coords", "true to set coords at the end", "",
                            set_coords, &set_coords, PETSC_NULL);
    CHKERR PetscOptionsBool("-output_vtk", "if true outout vtk file", "",
                            output_vtk, &output_vtk, PETSC_NULL);
    CHKERR PetscOptionsInt("-create_side_set", "crete side set", "",
                           create_surface_side_set, &create_surface_side_set,
                           &flg_create_surface_side_set);
    CHKERR PetscOptionsInt("-nb_ref_cut", "nb refinements befor cut", "",
                           nb_ref_cut, &nb_ref_cut, PETSC_NULL);
    CHKERR PetscOptionsInt("-nb_ref_trim", "nb refinements befor trim", "",
                           nb_ref_trim, &nb_ref_trim, PETSC_NULL);
    CHKERR PetscOptionsRealArray(
        "-tol", "tolerances tolCut, tolCutClose, tolTrim, tolTrimClose", "",
        tol, &nmax_tol, &flg_tol);
    CHKERR PetscOptionsBool("-debug", "debug (produces many VTK files)", "",
                            debug, &debug, PETSC_NULL);
    ierr = PetscOptionsEnd();
    CHKERRG(ierr);

    if (flg_myfile != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
              "*** ERROR -my_file (MESH FILE NEEDED)");
    }
    if (flg_shift && nmax != 3) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA, "three values expected");
    }

    if (flg_tol && nmax_tol != 3)
      SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA, "four values expected");

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    MoFEM::Core core(moab);
    MoFEM::CoreInterface &m_field =
        *(core.getInterface<MoFEM::CoreInterface>());

    // get cut mesh interface
    CutMeshInterface *cut_mesh;
    CHKERR m_field.getInterface(cut_mesh);
    // get meshset manager interface
    MeshsetsManager *meshset_manager;
    CHKERR m_field.getInterface(meshset_manager);
    // get bit ref manager interface
    BitRefManager *bit_ref_manager;
    CHKERR m_field.getInterface(bit_ref_manager);

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(
             (core.getInterface<MeshsetsManager &, 0>()), SIDESET, it)) {
      cout << *it << endl;
    }

    CHKERR bit_ref_manager->setBitRefLevelByType(0, MBTET,
                                                 BitRefLevel().set(0));

    // get surface entities form side set
    Range surface;
    if (meshset_manager->checkMeshset(surface_side_set, SIDESET))
      CHKERR meshset_manager->getEntitiesByDimension(surface_side_set, SIDESET,
                                                     2, surface, true);

    if (surface.empty())
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "No surface to cut");

    // Set surface entities. If surface entities are from existing side set,
    // copy those entities and do other geometrical transformations, like shift
    // scale or streach, rotate.
    if (meshset_manager->checkMeshset(surface_side_set, SIDESET))
      CHKERR cut_mesh->copySurface(surface, NULL, shift);
    else
      SETERRQ1(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
               "Side set not found %d", surface_side_set);

    // Get geometric corner nodes and corner edges
    Range fixed_edges, corner_nodes;
    if (meshset_manager->checkMeshset(edges_block_set, BLOCKSET)) 
      CHKERR meshset_manager->getEntitiesByDimension(edges_block_set, BLOCKSET,
                                                     1, fixed_edges, true);
    if (meshset_manager->checkMeshset(edges_block_set, SIDESET))
      CHKERR meshset_manager->getEntitiesByDimension(edges_block_set, SIDESET,
                                                     1, fixed_edges, true);
    if (meshset_manager->checkMeshset(vertex_block_set, BLOCKSET)) 
      CHKERR meshset_manager->getEntitiesByDimension(
          vertex_block_set, BLOCKSET, 0, corner_nodes, true);
    

    Range tets;
    if (flg_vol_block_set) {
      if (meshset_manager->checkMeshset(vol_block_set, BLOCKSET))
        CHKERR meshset_manager->getEntitiesByDimension(vol_block_set, BLOCKSET,
                                                       3, tets, true);
      else
        SETERRQ1(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                 "Block set %d not found", vol_block_set);
    } else
      CHKERR moab.get_entities_by_dimension(0, 3, tets, false);

    CHKERR cut_mesh->setVolume(tets);
    CHKERR cut_mesh->makeFront(true);

    // Build tree, it is used to ask geometrical queries, i.e. to find edges to
    // cut or trim.
    CHKERR cut_mesh->buildTree();

    // Refine mesh
    CHKERR cut_mesh->refineMesh(0, nb_ref_cut, nb_ref_trim, &fixed_edges,
                                VERBOSE, false);
    auto shift_after_ref = [&]() {
      MoFEMFunctionBegin;
      BitRefLevel mask;
      mask.set(0);
      for (int ll = 1; ll != nb_ref_cut + nb_ref_trim + 1; ++ll)
        mask.set(ll);
      CHKERR core.getInterface<BitRefManager>()->shiftRightBitRef(
          nb_ref_cut + nb_ref_trim, mask, VERBOSE);
      MoFEMFunctionReturn(0);
    };
    CHKERR shift_after_ref();

    // Create tag storing nodal positions
    double def_position[] = {0, 0, 0};
    Tag th;
    CHKERR moab.tag_get_handle("POSITION", 3, MB_TYPE_DOUBLE, th,
                               MB_TAG_CREAT | MB_TAG_SPARSE, def_position);
    // Set tag values with coordinates of nodes
    CHKERR cut_mesh->setTagData(th);

    // Cut mesh, trim surface and merge bad edges
    int first_bit = 1;
    CHKERR cut_mesh->cutTrimAndMerge(first_bit, fraction_level, th, tol[0],
                                     tol[1], tol[2], fixed_edges, corner_nodes,
                                     true, false);

    // Set coordinates for tag data
    if (set_coords)
      CHKERR cut_mesh->setCoords(th);

    // Add tets from last level to block
    if (flg_vol_block_set) {
      EntityHandle meshset;
      CHKERR meshset_manager->getMeshset(vol_block_set, BLOCKSET, meshset);
      CHKERR bit_ref_manager->getEntitiesByTypeAndRefLevel(
          BitRefLevel().set(first_bit), BitRefLevel().set(), MBTET, meshset);
    }

    // Shift bits
    if (squash_bits) {
      BitRefLevel shift_mask;
      for (int ll = 0; ll != first_bit; ++ll)
        shift_mask.set(ll);
      CHKERR core.getInterface<BitRefManager>()->shiftRightBitRef(
          first_bit - 1, shift_mask, VERBOSE);
    }

    Range surface_verts;
    CHKERR moab.get_connectivity(cut_mesh->getSurface(), surface_verts);
    Range surface_edges;
    CHKERR moab.get_adjacencies(cut_mesh->getSurface(), 1, false, surface_edges,
                                moab::Interface::UNION);
    CHKERR moab.delete_entities(cut_mesh->getSurface());
    CHKERR moab.delete_entities(surface_edges);
    CHKERR moab.delete_entities(surface_verts);

    if (flg_create_surface_side_set) {
      // Check is meshset is there
      if (!core.getInterface<MeshsetsManager>()->checkMeshset(
              create_surface_side_set, SIDESET))
        CHKERR meshset_manager->addMeshset(SIDESET, create_surface_side_set);
      else
        MOFEM_LOG_C("WORLD", Sev::warning,
                  "Warring >>> sideset %d is on the mesh",
                  create_surface_side_set);

      CHKERR meshset_manager->addEntitiesToMeshset(
          SIDESET, create_surface_side_set, cut_mesh->getMergedSurfaces());
    }

    CHKERR moab.write_file("out.h5m");

    if (output_vtk) {
      EntityHandle meshset;
      CHKERR moab.create_meshset(MESHSET_SET, meshset);
      if (flg_vol_block_set) {
        Range ents;
        meshset_manager->getEntitiesByDimension(vol_block_set, BLOCKSET, 3,
                                                ents, true);
        CHKERR moab.add_entities(meshset, ents);
      } else {
        BitRefLevel bit = BitRefLevel().set(0);
        CHKERR bit_ref_manager->getEntitiesByTypeAndRefLevel(
            bit, BitRefLevel().set(), MBTET, meshset);
      }
      CHKERR moab.add_entities(meshset, cut_mesh->getMergedSurfaces());
      CHKERR moab.write_file("out.vtk", "VTK", "", &meshset, 1);
      CHKERR moab.delete_entities(&meshset, 1);
    }
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}