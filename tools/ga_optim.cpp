/** \file ga_optim.cpp
 * \brief genetic algoritm optimisation for mesh cutting
 * \example ga_optim.cpp
 *
 * \ingroup ga_optim
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

#include <random>

using namespace MoFEM;
#include <ga_optim.hpp>

static char help[] = "mesh cutting\n\n";

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    PetscBool flg_myfile = PETSC_TRUE;
    char mesh_file_name[255];
    int surface_side_set = 200;
    PetscBool flg_vol_block_set;
    int vol_block_set = 1;
    int edges_block_set = 2;
    int vertex_block_set = 3;
    PetscBool flg_shift;
    double shift[] = {0, 0, 0};
    int nmax = 3;
    int fraction_level = 2;
    PetscBool squash_bits = PETSC_TRUE;
    PetscBool set_coords = PETSC_TRUE;
    PetscBool output_vtk = PETSC_TRUE;
    int create_surface_side_set = 201;
    PetscBool flg_create_surface_side_set;

    // optimize params

    int popMax = 10;
    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Mesh cut options", "none");
    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_myfile);
    CHKERR PetscOptionsInt("-surface_side_set", "surface side set", "",
                           surface_side_set, &surface_side_set, PETSC_NULL);
    CHKERR PetscOptionsInt("-vol_block_set", "volume side set", "",
                           vol_block_set, &vol_block_set, &flg_vol_block_set);
    CHKERR PetscOptionsInt("-edges_block_set", "edges side set", "",
                           edges_block_set, &edges_block_set, PETSC_NULL);
    CHKERR PetscOptionsInt("-vertex_block_set", "vertex side set", "",
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
    CHKERR PetscOptionsInt("-pop_max", "max size of the population", "",
                           popMax, &popMax,
                           &flg_create_surface_side_set);
    ierr = PetscOptionsEnd();
    CHKERRG(ierr);

    if (flg_myfile != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
              "*** ERROR -my_file (MESH FILE NEEDED)");
    }
    if (flg_shift && nmax != 3) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
              "three values expected");
    }

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    const char *option;
    option = ""; //"PARALLEL=BCAST";//;DEBUG_IO";
    std::cout.setstate(std::ios_base::failbit);
    CHKERR moab.load_file(mesh_file_name, 0, option);
    std::cout.clear();
    MoFEM::Core core(moab,PETSC_COMM_WORLD,0);
    MoFEM::CoreInterface &m_field =
        *(core.getInterface<MoFEM::CoreInterface>());

    // get cut mesh interface
    CutWithGA *cut_mesh = new CutWithGA(core);
    // CHKERR m_field.getInterface(cut_mesh);
    // get meshset manager interface
    MeshsetsManager *meshset_manager;
    CHKERR m_field.getInterface(meshset_manager);
    // get bit ref manager interface
    BitRefManager *bit_ref_manager;
    CHKERR m_field.getInterface(bit_ref_manager);


  Range tets;
  if (flg_vol_block_set) {
    if (meshset_manager->checkMeshset(vol_block_set, BLOCKSET)) {
      CHKERR meshset_manager->getEntitiesByDimension(vol_block_set, BLOCKSET, 3,
                                                     tets, true);
    } else {
      SETERRQ1(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
               "Block set %d not found", vol_block_set);
    }
  } else {
    CHKERR moab.get_entities_by_dimension(0, 3, tets, false);
  }
  CHKERR cut_mesh->setVolume(tets);

    BitRefLevel bit_level0;
    bit_level0.set(0);
    CHKERR bit_ref_manager->setBitRefLevelByType(0, MBTET, bit_level0);

    // get surface entities form side set
    Range surface;
    if (meshset_manager->checkMeshset(surface_side_set, SIDESET)) {
      CHKERR meshset_manager->getEntitiesByDimension(surface_side_set, SIDESET,
                                                     2, surface, true);
    }
    if (surface.empty()) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "No surface to cut");
    }

    Range all_edges;
  CHKERR moab.get_adjacencies(tets, 1, true, all_edges, moab::Interface::UNION);
    double minEdgeL = 1e16;
    FTensor::Index<'i', 3> i;
    // Store edge nodes coordinates in FTensor
    double edge_node_coords[6];
    FTensor::Tensor1<double *, 3> t_node_edge[2] = {
        FTensor::Tensor1<double *, 3>(edge_node_coords, &edge_node_coords[1],
                                      &edge_node_coords[2]),
        FTensor::Tensor1<double *, 3>(
            &edge_node_coords[3], &edge_node_coords[4], &edge_node_coords[5])};
    for (auto edge : all_edges) {
      int num_nodes;
      const EntityHandle *conn;
      CHKERR moab.get_connectivity(edge, conn, num_nodes, true);
      CHKERR moab.get_coords(conn, num_nodes, edge_node_coords);
      t_node_edge[0](i) -= t_node_edge[1](i);
      double l = sqrt(t_node_edge[0](i) * t_node_edge[0](i));
      minEdgeL = (minEdgeL > l) ? l : minEdgeL;
    }

    // PetscPrintf(PETSC_COMM_WORLD, "Min edge length = %6.4g\n", minEdgeL);

    // Build tree, it is used to ask geometrical queries, i.e. to find edges
    // to cut or trim.
    

    BitRefLevel bit_level1; // Cut level
    bit_level1.set(1);
    BitRefLevel bit_level2; // Trim level
    bit_level2.set(2);
    BitRefLevel bit_level3; // Merge level
    bit_level3.set(3);
    BitRefLevel bit_level4; // TetGen level
    bit_level4.set(4);

    // Create tag storing nodal positions
    double def_position[] = {0, 0, 0};
    Tag th;

/*
   
    */
    // string target = "MoFEM is the best Finite Element Code! :) ";
    // for(auto str : target){
    //   cout << int(str) << " " << str << endl;
    // }


    const vector<double> target(4, 0.);
    int popmax = popMax;
    double mutation_rate = 0.01;

    // for(int i = 0; i != 255; i++){
    //         std::cout << i << " " << char(i) << std::endl;
    //     }
    // Range fixed_edges, corner_nodes;
    // shared_ptr<DataForCut> data_cut_ptr = make_shared<DataForCut>(
    //     cut_mesh, fraction_level, bit_level0, bit_level1, bit_level2,
    //     bit_level3, th, fixed_edges, corner_nodes, minEdgeL);

    double ave = 0;
        //New population
    Population population(target, /*data_cut_ptr, */popmax, mutation_rate);
    // calculate fitness
    population.calcFitness();
    while (true) {

      // next generation
      population.generateNewPopulation();

      double fit = 0;
      auto best = population.getBest(fit);
      if (population.getGenerations() % 1 == 0) {
        system("clear");
        cout << "best Phrase: " << best << "\n";
        cout << "best fit: " << fit << "\n";
        cout << "total generations: " << population.getGenerations() << "\n";
        cout << "total population: " << population.getAverageFitness() << "\n";
        cout << "mutation rate: " << mutation_rate * 100 << "% \n";
      }
      if (population.finished() || population.getGenerations() > 100) {
        cout << "FINISHED" << endl;
        ave += population.getGenerations();
        break;
      }
    }

    return 0;
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();
}