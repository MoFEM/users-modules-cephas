/** \file insert_prism_interface.cpp
 * \example insert_prism_interface.cpp

*/

/* MoFEM is free software: you can redistribute it and/or modify it under
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

#include <BasicFiniteElements.hpp>

#include <boost/program_options.hpp>
using namespace std;
namespace po = boost::program_options;
#include <Hooke.hpp>
using namespace boost::numeric;

//#include <SimpleContact.hpp>

using namespace MoFEM;

static char help[] = "\n";
int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    PetscBool output_vtk = PETSC_TRUE;
    PetscBool flg = PETSC_TRUE;
    char mesh_file_name[255];
    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Split sides options",
                             "none");
    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg);
    CHKERR PetscOptionsBool("-output_vtk", "if true outout vtk file", "",
                            output_vtk, &output_vtk, PETSC_NULL);
    ierr = PetscOptionsEnd();
    CHKERRG(ierr);

    if (flg != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);
    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    PrismInterface *interface;
    CHKERR m_field.getInterface(interface);

    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, BitRefLevel().set(0));
    std::vector<BitRefLevel> bit_levels;
    bit_levels.push_back(BitRefLevel().set(0));

    Range master_tris;

    int ll = 1;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, SIDESET, cit)) {
      if (cit->getName().compare(0, 11, "INT_CONTACT") == 0) {
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "Insert %s (id: %d)\n",
                           cit->getName().c_str(), cit->getMeshsetId());
        EntityHandle cubit_meshset = cit->getMeshset();
        Range tris;
        CHKERR moab.get_entities_by_type(cubit_meshset, MBTRI, tris, true);
        master_tris.merge(tris);

        {
          // get tet enties form back bit_level
          EntityHandle ref_level_meshset = 0;
          CHKERR moab.create_meshset(MESHSET_SET, ref_level_meshset);
          CHKERR m_field.getInterface<BitRefManager>()
              ->getEntitiesByTypeAndRefLevel(bit_levels.back(),
                                             BitRefLevel().set(), MBTET,
                                             ref_level_meshset);
          CHKERR m_field.getInterface<BitRefManager>()
              ->getEntitiesByTypeAndRefLevel(bit_levels.back(),
                                             BitRefLevel().set(), MBPRISM,
                                             ref_level_meshset);
          Range ref_level_tets;
          CHKERR moab.get_entities_by_handle(ref_level_meshset, ref_level_tets,
                                             true);
          ref_level_tets.print();
          // get faces and tets to split
          CHKERR interface->getSides(cubit_meshset, bit_levels.back(), true, 0);
          // set new bit level
          bit_levels.push_back(BitRefLevel().set(ll++));
          // split faces and tets
          CHKERR interface->splitSides(ref_level_meshset, bit_levels.back(),
                                       cubit_meshset, true, true, 0);
          // clean meshsets
          CHKERR moab.delete_entities(&ref_level_meshset, 1);
        }
        // update cubit meshsets
        for (_IT_CUBITMESHSETS_FOR_LOOP_(m_field, ciit)) {
          EntityHandle cubit_meshset = ciit->meshset;
          CHKERR m_field.getInterface<BitRefManager>()
              ->updateMeshsetByEntitiesChildren(cubit_meshset,
                                                bit_levels.back(),
                                                cubit_meshset, MBVERTEX, true);
          CHKERR m_field.getInterface<BitRefManager>()
              ->updateMeshsetByEntitiesChildren(cubit_meshset,
                                                bit_levels.back(),
                                                cubit_meshset, MBEDGE, true);
          CHKERR m_field.getInterface<BitRefManager>()
              ->updateMeshsetByEntitiesChildren(
                  cubit_meshset, bit_levels.back(), cubit_meshset, MBTRI, true);
          CHKERR m_field.getInterface<BitRefManager>()
              ->updateMeshsetByEntitiesChildren(
                  cubit_meshset, bit_levels.back(), cubit_meshset, MBTET, true);
        }
      }
    }

    Range tets_back_bit_level;
    CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByRefLevel(
        bit_levels.back(), BitRefLevel().set(), tets_back_bit_level);

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, cit)) {

      EntityHandle cubit_meshset = cit->getMeshset();

      BlockSetAttributes mydata;
      CHKERR cit->getAttributeDataStructure(mydata);
      std::cout << mydata << std::endl;

      Range tets;
      CHKERR moab.get_entities_by_type(cubit_meshset, MBTET, tets, true);
      tets = intersect(tets_back_bit_level, tets);
      Range nodes;
      CHKERR moab.get_connectivity(tets, nodes, true);

      for (Range::iterator nit = nodes.begin(); nit != nodes.end(); nit++) {
        double coords[3];
        CHKERR moab.get_coords(&*nit, 1, coords);
        coords[0] += mydata.data.User1;
        coords[1] += mydata.data.User2;
        coords[2] += mydata.data.User3;
        CHKERR moab.set_coords(&*nit, 1, coords);
      }
    }

    EntityHandle out_meshset_tet;
    EntityHandle out_meshset_prism;
    EntityHandle out_meshset_tets_and_prism;

    CHKERR moab.create_meshset(MESHSET_SET, out_meshset_tet);
    CHKERR moab.create_meshset(MESHSET_SET, out_meshset_prism);
    CHKERR moab.create_meshset(MESHSET_SET, out_meshset_tets_and_prism);

    CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        bit_levels.back(), BitRefLevel().set(), MBTET, out_meshset_tet);
    Range tets;
    CHKERR moab.get_entities_by_handle(out_meshset_tet, tets, true);
    tets.print();

    CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        bit_levels.back(), BitRefLevel().set(), MBPRISM, out_meshset_prism);
    Range prisms;
    CHKERR moab.get_entities_by_handle(out_meshset_prism, prisms);

    CHKERR moab.add_entities(out_meshset_tets_and_prism, tets);
    CHKERR moab.add_entities(out_meshset_tets_and_prism, prisms);

    Range tris;
    CHKERR moab.get_adjacencies(prisms, 2, false, tris, moab::Interface::UNION);
    tris = tris.subset_by_type(MBTRI);

    Range slave_tris;
    slave_tris = subtract(tris, master_tris);
    //slave_tris.print();

    CHKERR moab.write_file("out_tet.vtk", "VTK", "", &out_meshset_tet, 1);
    CHKERR moab.write_file("out_prism.vtk", "VTK", "", &out_meshset_prism, 1);
    CHKERR moab.write_file("out_tets_and_prisms.vtk", "VTK", "",
                           &out_meshset_tets_and_prism, 1);

    CHKERR moab.write_file("out.h5m", "MOAB", "", &out_meshset_tets_and_prism, 1);
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();

  return 0;
}