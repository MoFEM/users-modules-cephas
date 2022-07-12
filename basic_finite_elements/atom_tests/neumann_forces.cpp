/* MIT License
 *
 * Copyright (c) 2022
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <BasicFiniteElements.hpp>
using namespace MoFEM;

namespace bio = boost::iostreams;
using bio::stream;
using bio::tee_device;

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscBool flg = PETSC_TRUE;
    char mesh_file_name[255];
    CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-my_file",
                                 mesh_file_name, 255, &flg);
    if (flg != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    // Create MoFEM (Joseph) database
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    // set entitities bit level
    BitRefLevel bit_level0;
    bit_level0.set(0);
    EntityHandle meshset_level0;
    CHKERR moab.create_meshset(MESHSET_SET, meshset_level0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_level0);

    // Fields
    CHKERR m_field.add_field("DISPLACEMENT", H1, AINSWORTH_LEGENDRE_BASE, 3);
    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3);

    // Problem
    CHKERR m_field.add_problem("TEST_PROBLEM");
    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("TEST_PROBLEM", bit_level0);

    // meshset consisting all entities in mesh
    EntityHandle root_set = moab.get_root_set();
    // add entities to field
    CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET, "DISPLACEMENT");
    CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET,
                                             "MESH_NODE_POSITIONS");

    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                    it)) {

      std::ostringstream fe_name;
      fe_name << "FORCE_FE_" << it->getMeshsetId();
      CHKERR m_field.add_finite_element(fe_name.str());
      CHKERR m_field.modify_finite_element_add_field_row(fe_name.str(),
                                                         "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_col(fe_name.str(),
                                                         "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_data(fe_name.str(),
                                                          "DISPLACEMENT");
      CHKERR m_field.modify_problem_add_finite_element("TEST_PROBLEM",
                                                       fe_name.str());

      Range tris;
      CHKERR moab.get_entities_by_type(it->meshset, MBTRI, tris, true);
      CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI,
                                                        fe_name.str());
    }

    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             m_field, SIDESET | PRESSURESET, it)) {

      std::ostringstream fe_name;
      fe_name << "PRESSURE_FE_" << it->getMeshsetId();
      CHKERR m_field.add_finite_element(fe_name.str());
      CHKERR m_field.modify_finite_element_add_field_row(fe_name.str(),
                                                         "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_col(fe_name.str(),
                                                         "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_data(fe_name.str(),
                                                          "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_data(
          fe_name.str(), "MESH_NODE_POSITIONS");
      CHKERR m_field.modify_problem_add_finite_element("TEST_PROBLEM",
                                                       fe_name.str());

      Range tris;
      CHKERR moab.get_entities_by_type(it->meshset, MBTRI, tris, true);
      CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI,
                                                        fe_name.str());
    }

    // set app. order
    // see Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes
    // (Mark Ainsworth & Joe Coyle)
    int order = 2;
    CHKERR m_field.set_field_order(root_set, MBTET, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(root_set, MBTRI, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(root_set, MBEDGE, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(root_set, MBVERTEX, "DISPLACEMENT", 1);
    CHKERR m_field.set_field_order(root_set, MBTET, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(root_set, MBTRI, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(root_set, MBEDGE, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(root_set, MBVERTEX, "MESH_NODE_POSITIONS",
                                   1);

    /****/
    // build database
    // build field
    CHKERR m_field.build_fields();
    // set FIELD1 from positions of 10 node tets
    Projection10NodeCoordsOnField ent_method(m_field, "MESH_NODE_POSITIONS");
    CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method);

    // build finite elemnts
    CHKERR m_field.build_finite_elements();
    // build adjacencies
    CHKERR m_field.build_adjacencies(bit_level0);
    // build problem
    ProblemsManager *prb_mng_ptr;
    CHKERR m_field.getInterface(prb_mng_ptr);
    CHKERR prb_mng_ptr->buildProblem("TEST_PROBLEM", true);

    /****/
    // mesh partitioning
    // partition
    CHKERR prb_mng_ptr->partitionSimpleProblem("TEST_PROBLEM");
    CHKERR prb_mng_ptr->partitionFiniteElements("TEST_PROBLEM");
    // what are ghost nodes, see Petsc Manual
    CHKERR prb_mng_ptr->partitionGhostDofs("TEST_PROBLEM");

    Vec F;
    CHKERR m_field.getInterface<VecManager>()->vecCreateGhost("TEST_PROBLEM",
                                                              ROW, &F);

    typedef tee_device<std::ostream, std::ofstream> TeeDevice;
    typedef stream<TeeDevice> TeeStream;
    std::ostringstream txt_name;
    txt_name << "forces_and_sources_" << mesh_file_name << ".txt";
    std::ofstream ofs(txt_name.str().c_str());
    TeeDevice my_tee(std::cout, ofs);
    TeeStream my_split(my_tee);

    CHKERR VecZeroEntries(F);
    boost::ptr_map<std::string, NeumannForcesSurface> neumann_forces;
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                    it)) {
      std::ostringstream fe_name;
      fe_name << "FORCE_FE_" << it->getMeshsetId();
      string fe_name_str = fe_name.str();
      neumann_forces.insert(fe_name_str, new NeumannForcesSurface(m_field));
      CHKERR addHOOpsFace3D(
          "MESH_NODE_POSITIONS", neumann_forces.at(fe_name_str).getLoopFe(),
          false, false);
      neumann_forces.at(fe_name_str)
          .addForce("DISPLACEMENT", F, it->getMeshsetId());
      ForceCubitBcData data;
      CHKERR it->getBcDataStructure(data);
      my_split << *it << std::endl;
      my_split << data << std::endl;
    }
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             m_field, SIDESET | PRESSURESET, it)) {
      std::ostringstream fe_name;
      fe_name << "PRESSURE_FE_" << it->getMeshsetId();
      string fe_name_str = fe_name.str();
      neumann_forces.insert(fe_name_str, new NeumannForcesSurface(m_field));
      CHKERR addHOOpsFace3D(
          "MESH_NODE_POSITIONS", neumann_forces.at(fe_name_str).getLoopFe(),
          false, false);
      neumann_forces.at(fe_name_str)
          .addPressure("DISPLACEMENT", F, it->getMeshsetId());
      PressureCubitBcData data;
      CHKERR it->getBcDataStructure(data);
      my_split << *it << std::endl;
      my_split << data << std::endl;
    }
    boost::ptr_map<std::string, NeumannForcesSurface>::iterator mit =
        neumann_forces.begin();
    for (; mit != neumann_forces.end(); mit++) {
      CHKERR m_field.loop_finite_elements("TEST_PROBLEM", mit->first,
                                          mit->second->getLoopFe());
    }
    CHKERR VecAssemblyBegin(F);
    CHKERR VecAssemblyEnd(F);

    CHKERR m_field.getInterface<VecManager>()->setGlobalGhostVector(
        "TEST_PROBLEM", ROW, F, INSERT_VALUES, SCATTER_REVERSE);

    const double eps = 1e-4;

    const Problem *problemPtr;
    CHKERR m_field.get_problem("TEST_PROBLEM", &problemPtr);
    for (_IT_NUMEREDDOF_ROW_FOR_LOOP_(problemPtr, dit)) {

      my_split.precision(3);
      my_split.setf(std::ios::fixed);
      double val = fabs(dit->get()->getFieldData()) < eps
                       ? 0.0
                       : dit->get()->getFieldData();
      my_split << dit->get()->getPetscGlobalDofIdx() << " " << val << std::endl;
    }

    double sum = 0;
    CHKERR VecSum(F, &sum);
    sum = fabs(sum) < eps ? 0.0 : sum;
    my_split << std::endl
             << "Sum : " << std::setprecision(3) << sum << std::endl;

    CHKERR VecDestroy(&F);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
