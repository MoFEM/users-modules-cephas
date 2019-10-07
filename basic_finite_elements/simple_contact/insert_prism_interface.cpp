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

    PetscBool flg = PETSC_TRUE;
    char mesh_file_name[255];
#if PETSC_VERSION_GE(3, 6, 4)
    CHKERR PetscOptionsGetString(PETSC_NULL, "", "-my_file", mesh_file_name,
                                 255, &flg);
#else
    CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-my_file",
                                 mesh_file_name, 255, &flg);
#endif
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

    int ll = 1;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, SIDESET, cit)) {
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "Insert Interface %d\n",
                         cit->getMeshsetId());
      EntityHandle cubit_meshset = cit->getMeshset();
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
        // get faces and test to split
        CHKERR interface->getSides(cubit_meshset, bit_levels.back(), true, 0);
        // set new bit level
        bit_levels.push_back(BitRefLevel().set(ll++));
        // split faces and
        CHKERR interface->splitSides(ref_level_meshset, bit_levels.back(),
                                     cubit_meshset, true, true, 0);
        // clean meshsets
        CHKERR moab.delete_entities(&ref_level_meshset, 1);
      }
      // update cubit meshsets
      for (_IT_CUBITMESHSETS_FOR_LOOP_(m_field, ciit)) {
        EntityHandle cubit_meshset = ciit->meshset;
        CHKERR m_field.getInterface<BitRefManager>()
            ->updateMeshsetByEntitiesChildren(cubit_meshset, bit_levels.back(),
                                              cubit_meshset, MBVERTEX, true);
        CHKERR m_field.getInterface<BitRefManager>()
            ->updateMeshsetByEntitiesChildren(cubit_meshset, bit_levels.back(),
                                              cubit_meshset, MBEDGE, true);
        CHKERR m_field.getInterface<BitRefManager>()
            ->updateMeshsetByEntitiesChildren(cubit_meshset, bit_levels.back(),
                                              cubit_meshset, MBTRI, true);
        CHKERR m_field.getInterface<BitRefManager>()
            ->updateMeshsetByEntitiesChildren(cubit_meshset, bit_levels.back(),
                                              cubit_meshset, MBTET, true);
      }
    }

    // add fields
    CHKERR m_field.add_field("H1FIELD_SCALAR", H1, AINSWORTH_LEGENDRE_BASE, 1);

    // add finite elements
    CHKERR m_field.add_finite_element("ELEM_SCALAR");
    CHKERR m_field.modify_finite_element_add_field_row("ELEM_SCALAR",
                                                       "H1FIELD_SCALAR");
    CHKERR m_field.modify_finite_element_add_field_col("ELEM_SCALAR",
                                                       "H1FIELD_SCALAR");
    CHKERR m_field.modify_finite_element_add_field_data("ELEM_SCALAR",
                                                        "H1FIELD_SCALAR");
    // finite element interface
    CHKERR m_field.add_finite_element("INTERFACE");
    CHKERR m_field.modify_finite_element_add_field_row("INTERFACE",
                                                       "H1FIELD_SCALAR");
    CHKERR m_field.modify_finite_element_add_field_col("INTERFACE",
                                                       "H1FIELD_SCALAR");
    CHKERR m_field.modify_finite_element_add_field_data("INTERFACE",
                                                        "H1FIELD_SCALAR");

    // add ents to field and set app. order
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "H1FIELD_SCALAR");
    CHKERR m_field.set_field_order(0, MBVERTEX, "H1FIELD_SCALAR", 1);

    // add finite elements entities
    // all TETS and PRIMS are added to finite elements, for testin purposes.
    // in some practical applications to save memory, you would like to add
    // elements from particular refinement level (see:
    // m_field.add_ents_to_finite_element_by_bit_ref(...)
    CHKERR m_field.add_ents_to_finite_element_by_type(0, MBTET, "ELEM_SCALAR",
                                                      true);
    CHKERR m_field.add_ents_to_finite_element_by_type(0, MBPRISM, "INTERFACE",
                                                      true);

    // add problems
    // set problem for all last two levels, only for testing purposes
    for (int lll = ll - 2; lll < ll; lll++) {
      std::stringstream problem_name;
      problem_name << "PROBLEM_SCALAR_" << lll;
      CHKERR m_field.add_problem(problem_name.str());
      // define problems and finite elements
      CHKERR m_field.modify_problem_add_finite_element(problem_name.str(),
                                                       "ELEM_SCALAR");
      CHKERR m_field.modify_problem_add_finite_element(problem_name.str(),
                                                       "INTERFACE");
    }

    // set problem level
    for (int lll = ll - 2; lll < ll; lll++) {
      std::stringstream problem_name;
      problem_name << "PROBLEM_SCALAR_" << lll;
      std::stringstream message;
      message << "set problem problem < " << problem_name.str()
              << " > bit level " << bit_levels[lll] << std::endl;
      CHKERR PetscPrintf(PETSC_COMM_WORLD, message.str().c_str());
      CHKERR m_field.modify_problem_ref_level_add_bit(problem_name.str(),
                                                      bit_levels[lll]);
    }

    // build fields
    CHKERR m_field.build_fields();
    // build finite elements
    CHKERR m_field.build_finite_elements();
    // build adjacencies
    // Its build adjacencies for all elements in database,
    // for practical applications consider to build adjacencies
    // only for refinemnt levels which you use for calculations
    CHKERR m_field.build_adjacencies(BitRefLevel().set());

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

    ProblemsManager *prb_mng_ptr;
    CHKERR m_field.getInterface(prb_mng_ptr);

    // partition
    for (int lll = ll - 2; lll < ll; lll++) {
      // build problem
      std::stringstream problem_name;
      problem_name << "PROBLEM_SCALAR_" << lll;
      CHKERR prb_mng_ptr->buildProblem(problem_name.str(), true);
      CHKERR prb_mng_ptr->partitionProblem(problem_name.str());
      CHKERR prb_mng_ptr->partitionFiniteElements(problem_name.str());
      CHKERR prb_mng_ptr->partitionGhostDofs(problem_name.str());
    }

    std::ofstream myfile;
    myfile.open("mesh_insert_interface.txt");

    EntityHandle out_meshset_tet;
    CHKERR moab.create_meshset(MESHSET_SET, out_meshset_tet);

    CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        bit_levels.back(), BitRefLevel().set(), MBTET, out_meshset_tet);
    Range tets;
    CHKERR moab.get_entities_by_handle(out_meshset_tet, tets, true);
    for (Range::iterator tit = tets.begin(); tit != tets.end(); tit++) {
      int num_nodes;
      const EntityHandle *conn;
      CHKERR moab.get_connectivity(*tit, conn, num_nodes, true);

      for (int nn = 0; nn < num_nodes; nn++) {
        std::cout << conn[nn] << " ";
        myfile << conn[nn] << " ";
      }
      std::cout << std::endl;
      myfile << std::endl;
    }
    EntityHandle out_meshset_prism;
    CHKERR moab.create_meshset(MESHSET_SET, out_meshset_prism);
    CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        bit_levels.back(), BitRefLevel().set(), MBPRISM, out_meshset_prism);
    Range prisms;
    CHKERR moab.get_entities_by_handle(out_meshset_prism, prisms);
    for (Range::iterator pit = prisms.begin(); pit != prisms.end(); pit++) {
      int num_nodes;
      const EntityHandle *conn;
      CHKERR moab.get_connectivity(*pit, conn, num_nodes, true);

      for (int nn = 0; nn < num_nodes; nn++) {
        std::cout << conn[nn] << " ";
        myfile << conn[nn] << " ";
      }
      std::cout << std::endl;
      myfile << std::endl;
    }
    myfile.close();

    CHKERR moab.write_file("out_tet.vtk", "VTK", "", &out_meshset_tet, 1);
    CHKERR moab.write_file("out_prism.vtk", "VTK", "", &out_meshset_prism, 1);

    EntityHandle out_meshset_tets_and_prism;
    CHKERR moab.create_meshset(MESHSET_SET, out_meshset_tets_and_prism);
    CHKERR moab.add_entities(out_meshset_tets_and_prism, tets);
    CHKERR moab.add_entities(out_meshset_tets_and_prism, prisms);
    CHKERR moab.write_file("out_tets_and_prisms.vtk", "VTK", "",
                           &out_meshset_tets_and_prism, 1);

    EntityHandle out_meshset_tris;
    CHKERR moab.create_meshset(MESHSET_SET, out_meshset_tris);
    Range tris;
    CHKERR moab.get_adjacencies(prisms, 2, false, tris, moab::Interface::UNION);
    std::cerr << tris.size() << " : " << prisms.size() << std::endl;
    CHKERR moab.add_entities(out_meshset_tris, tris);
    CHKERR moab.write_file("out_tris.vtk", "VTK", "", &out_meshset_tris, 1);
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();

  return 0;
}