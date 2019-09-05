/** \file prisms_elements_from_surface.cpp
  \example prisms_elements_from_surface.cpp
  \brief Adding prims on the surface

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

namespace bio = boost::iostreams;
using bio::stream;
using bio::tee_device;

using namespace MoFEM;

static char help[] = "...\n\n";
static int debug = 1;

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // Read parameters from line command
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

    // Read mesh to MOAB
    const char *option;
    option = ""; //"PARALLEL=BCAST;";//;DEBUG_IO";
    CHKERR moab.load_file(mesh_file_name, 0, option);
    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    // Create MoFEM (Joseph) databas
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    PrismsFromSurfaceInterface *prisms_from_surface_interface;
    CHKERR m_field.getInterface(prisms_from_surface_interface);

    Range tris;
    CHKERR moab.get_entities_by_type(0, MBTRI, tris, false);
    Range prisms;
    CHKERR prisms_from_surface_interface->createPrisms(tris, prisms);
    prisms_from_surface_interface->createdVertices.clear();
    Range add_prims_layer;
    CHKERR prisms_from_surface_interface->createPrismsFromPrisms(
        prisms, true, add_prims_layer);
    prisms.merge(add_prims_layer);

    EntityHandle meshset;
    CHKERR moab.create_meshset(MESHSET_SET | MESHSET_TRACK_OWNER, meshset);
    CHKERR moab.add_entities(meshset, prisms);

    BitRefLevel bit_level0;
    bit_level0.set(0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        meshset, 3, bit_level0);
    CHKERR prisms_from_surface_interface->seedPrismsEntities(prisms,
                                                             bit_level0);

    // Fields
    CHKERR m_field.add_field("FIELD1", H1, AINSWORTH_LEGENDRE_BASE, 1);
    CHKERR m_field.add_ents_to_field_by_type(meshset, MBPRISM, "FIELD1", 10);

    CHKERR m_field.set_field_order(0, MBVERTEX, "FIELD1", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "FIELD1", 3, 10);
    CHKERR m_field.build_fields(10);

    const DofEntity_multiIndex *dofs_ptr;
    CHKERR m_field.get_dofs(&dofs_ptr);
    PetscPrintf(PETSC_COMM_WORLD, "dofs_ptr.size() = %d\n", dofs_ptr->size());
    if (dofs_ptr->size() != 887) {
      SETERRQ1(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
               "data inconsistency 323!=%d", dofs_ptr->size());
    }

    CHKERR m_field.set_field_order(0, MBQUAD, "FIELD1", 4, 10);
    CHKERR m_field.set_field_order(0, MBPRISM, "FIELD1", 6, 10);
    CHKERR m_field.build_fields(10);

    PetscPrintf(PETSC_COMM_WORLD, "dofs_ptr.size() = %d\n", dofs_ptr->size());
    if (dofs_ptr->size() != 1207) {
      SETERRQ1(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
               "data inconsistency 483!=%d", dofs_ptr->size());
    }

    CHKERR m_field.set_field_order(0, MBTRI, "FIELD1", 3);
    CHKERR m_field.build_fields();

    if (debug) {
      CHKERR moab.write_file("prism_mesh.vtk", "VTK", "", &meshset, 1);
    }

    // FE
    CHKERR m_field.add_finite_element("TEST_FE1");

    // Define rows/cols and element data
    CHKERR m_field.modify_finite_element_add_field_row("TEST_FE1", "FIELD1");
    CHKERR m_field.modify_finite_element_add_field_col("TEST_FE1", "FIELD1");
    CHKERR m_field.modify_finite_element_add_field_data("TEST_FE1", "FIELD1");

    CHKERR m_field.add_ents_to_finite_element_by_type(prisms, MBPRISM,
                                                      "TEST_FE1");

    // build finite elemnts
    CHKERR m_field.build_finite_elements();
    // //build adjacencies
    CHKERR m_field.build_adjacencies(bit_level0);

    // Problem
    CHKERR m_field.add_problem("TEST_PROBLEM");

    // set finite elements for problem
    CHKERR m_field.modify_problem_add_finite_element("TEST_PROBLEM",
                                                     "TEST_FE1");
    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("TEST_PROBLEM", bit_level0);

    // build problem
    ProblemsManager *prb_mng_ptr;
    CHKERR m_field.getInterface(prb_mng_ptr);
    CHKERR prb_mng_ptr->buildProblem("TEST_PROBLEM", true);
    // partition
    CHKERR prb_mng_ptr->partitionSimpleProblem("TEST_PROBLEM");
    CHKERR prb_mng_ptr->partitionFiniteElements("TEST_PROBLEM");
    // what are ghost nodes, see Petsc Manual
    CHKERR prb_mng_ptr->partitionGhostDofs("TEST_PROBLEM");

    typedef tee_device<std::ostream, std::ofstream> TeeDevice;
    typedef stream<TeeDevice> TeeStream;

    std::ofstream ofs("prisms_elements_from_surface.txt");
    TeeDevice my_tee(std::cout, ofs);
    TeeStream my_split(my_tee);

    struct MyOp : public FatPrismElementForcesAndSourcesCore::UserDataOperator {

      TeeStream &mySplit;
      MyOp(TeeStream &mySplit, const char type)
          : FatPrismElementForcesAndSourcesCore::UserDataOperator(
                "FIELD1", "FIELD1", type),
            mySplit(mySplit) {}

      MoFEMErrorCode doWork(int side, EntityType type,
                            DataForcesAndSourcesCore::EntData &data) {
        MoFEMFunctionBeginHot;

        // if(data.getFieldData().empty()) MoFEMFunctionReturnHot(0);

        // const double eps = 1e-4;
        // for(
        //   DoubleAllocator::iterator it = getNormal().data().begin();
        //   it!=getNormal().data().end();it++
        // ) {
        //   *it = fabs(*it)<eps ? 0.0 : *it;
        // }
        // for(
        //   DoubleAllocator::iterator it =
        //   getNormalsAtGaussPtF3().data().begin();
        //   it!=getNormalsAtGaussPtF3().data().end();it++
        // ) {
        //   *it = fabs(*it)<eps ? 0.0 : *it;
        // }
        // for(
        //   DoubleAllocator::iterator it =
        //   getTangent1AtGaussPtF3().data().begin();
        //   it!=getTangent1AtGaussPtF3().data().end();it++
        // ) {
        //   *it = fabs(*it)<eps ? 0.0 : *it;
        // }
        // for(
        //   DoubleAllocator::iterator it =
        //   getTangent2AtGaussPtF3().data().begin();
        //   it!=getTangent2AtGaussPtF3().data().end();it++
        // ) {
        //   *it = fabs(*it)<eps ? 0.0 : *it;
        // }
        // for(
        //   DoubleAllocator::iterator it =
        //   getNormalsAtGaussPtF4().data().begin();
        //   it!=getNormalsAtGaussPtF4().data().end();it++
        // ) {
        //   *it = fabs(*it)<eps ? 0.0 : *it;
        // }
        // for(
        //   DoubleAllocator::iterator it =
        //   getTangent1AtGaussPtF4().data().begin();
        //   it!=getTangent1AtGaussPtF4().data().end();it++
        // ) {
        //   *it = fabs(*it)<eps ? 0.0 : *it;
        // }
        // for(
        //   DoubleAllocator::iterator it =
        //   getTangent2AtGaussPtF4().data().begin();
        //   it!=getTangent2AtGaussPtF4().data().end();it++
        // ) {
        //   *it = fabs(*it)<eps ? 0.0 : *it;
        // }
        //
        mySplit << "NH1" << std::endl;
        mySplit << "side: " << side << " type: " << type << std::endl;
        data.getN() *= 1e4;
        data.getDiffN() *= 1e4;
        mySplit << data << std::endl;
        mySplit << "getTroughThicknessDataStructure" << std::endl;
        mySplit << getTroughThicknessDataStructure().dataOnEntities[type][side]
                << std::endl;

        // mySplit << "integration pts " << getGaussPts() << std::endl;
        // mySplit << "coords at integration pts " << getCoordsAtGaussPts() <<
        // std::endl;
        mySplit << std::endl << std::endl;

        // mySplit << std::setprecision(3) << getCoords() << std::endl;
        // mySplit << std::setprecision(3) << getCoordsAtGaussPts() <<
        // std::endl; mySplit << std::setprecision(3) << getArea(0) <<
        // std::endl; mySplit << std::setprecision(3) << getArea(1) <<
        // std::endl; mySplit << std::setprecision(3) << "normal F3 " <<
        // getNormalF3() << std::endl; mySplit << std::setprecision(3) <<
        // "normal F4 " << getNormalF4() << std::endl; mySplit <<
        // std::setprecision(3) << "normal at Gauss pt F3 " <<
        // getNormalsAtGaussPtF3() << std::endl; mySplit << std::setprecision(3)
        // << getTangent1AtGaussPtF3() << std::endl; mySplit <<
        // std::setprecision(3) << getTangent2AtGaussPtF3() << std::endl;
        // mySplit << std::setprecision(3) << "normal at Gauss pt F4 " <<
        // getNormalsAtGaussPtF4() << std::endl; mySplit << std::setprecision(3)
        // << getTangent1AtGaussPtF4() << std::endl; mySplit <<
        // std::setprecision(3) << getTangent2AtGaussPtF4() << std::endl;
        MoFEMFunctionReturnHot(0);
      }

      MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                            EntityType col_type,
                            DataForcesAndSourcesCore::EntData &row_data,
                            DataForcesAndSourcesCore::EntData &col_data) {
        MoFEMFunctionBeginHot;

        // if(row_data.getFieldData().empty()) MoFEMFunctionReturnHot(0);
        //
        // mySplit << "NH1NH1" << std::endl;
        // mySplit << "row side: " << row_side << " row_type: " << row_type <<
        // std::endl; mySplit << row_data << std::endl; mySplit << "NH1NH1" <<
        // std::endl; mySplit << "col side: " << col_side << " col_type: " <<
        // col_type << std::endl; mySplit << row_data << std::endl;

        MoFEMFunctionReturnHot(0);
      }
    };

    FatPrismElementForcesAndSourcesCore fe1(m_field);
    fe1.getOpPtrVector().push_back(
        new MyOp(my_split, ForcesAndSourcesCore::UserDataOperator::OPROW));
    // fe1.getOpPtrVector().push_back(new
    // MyOp(my_split,ForcesAndSourcesCore::UserDataOperator::OPROWCOL));
    CHKERR m_field.loop_finite_elements("TEST_PROBLEM", "TEST_FE1", fe1);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();
  return 0;
}
