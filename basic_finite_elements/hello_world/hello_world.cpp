/**
 * \file hello_world.cpp
 * \ingroup mofem_simple_interface
 * \example hello_world.cpp
 *
 * Prints basic information about users data operator.
 * See more details in \ref hello_world_tut1
 *
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

#include <BasicFiniteElements.hpp>
using namespace MoFEM;

static char help[] = "...\n\n";

static map<EntityType, std::string> type_name;

struct OpRow : public ForcesAndSourcesCore::UserDataOperator {
  OpRow(const std::string &field_name)
      : ForcesAndSourcesCore::UserDataOperator(field_name, field_name, OPROW) {}
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;
    if (type == MBVERTEX) {
      // get number of evaluated element in the loop
      std::cout << std::endl
                << "**** " << getNinTheLoop() << " **** " << std::endl;
      std::cout << "**** Operators **** " << std::endl;
    }
    std::cout << "Hello Operator OpRow:"
              << " field name " << rowFieldName << " side " << side << " type "
              << type_name[type] << " nb dofs on entity "
              << data.getIndices().size() << std::endl;
    MoFEMFunctionReturn(0);
  }
};

struct OpRowCol : public ForcesAndSourcesCore::UserDataOperator {
  OpRowCol(const std::string row_field, const std::string col_field,
           const bool symm)
      : ForcesAndSourcesCore::UserDataOperator(row_field, col_field, OPROWCOL,
                                               symm) {}
  virtual MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                                EntityType col_type,
                                DataForcesAndSourcesCore::EntData &row_data,
                                DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;
    std::cout << "Hello Operator OpRowCol:"
              << " row field name " << rowFieldName << " row side " << row_side
              << " row type " << type_name[row_type] << " nb dofs on row entity"
              << row_data.getIndices().size() << " : "
              << " col field name " << colFieldName << " col side " << col_side
              << " col type " << type_name[col_type] << " nb dofs on col entity"
              << col_data.getIndices().size() << std::endl;
    MoFEMFunctionReturn(0);
  }
};

struct OpVolume : public VolumeElementForcesAndSourcesCore::UserDataOperator {
  OpVolume(const std::string &field_name)
      : VolumeElementForcesAndSourcesCore::UserDataOperator(field_name,
                                                            field_name, OPROW) {
  }
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;
    if (type == MBVERTEX) {
      std::cout << "Hello Operator OpVolume:"
                << " volume " << getVolume() << endl;
    }
    MoFEMFunctionReturn(0);
  }
};

struct OpFace : public FaceElementForcesAndSourcesCore::UserDataOperator {
  OpFace(const std::string &field_name)
      : FaceElementForcesAndSourcesCore::UserDataOperator(field_name, OPROW) {}
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;
    if (type == MBVERTEX) {
      std::cout << "Hello Operator OpFace:"
                << " normal " << getNormal() << endl;
    }
    MoFEMFunctionReturn(0);
  }
};

struct OpFaceSide : public FaceElementForcesAndSourcesCore::UserDataOperator {
  boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> &feSidePtr;
  OpFaceSide(
      const std::string &field_name,
      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> &fe_side_ptr)
      : FaceElementForcesAndSourcesCore::UserDataOperator(field_name, OPROW),
        feSidePtr(fe_side_ptr) {}
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {

    MoFEMFunctionBegin;
    if (type == MBVERTEX) {
      std::cout << "Hello Operator OpSideFace" << endl;
      CHKERR loopSideVolumes("dFE", *feSidePtr);
    }
    MoFEMFunctionReturn(0);
  }
};

struct OpVolumeSide
    : public VolumeElementForcesAndSourcesCoreOnSide::UserDataOperator {
  OpVolumeSide(const std::string &field_name)
      : VolumeElementForcesAndSourcesCoreOnSide::UserDataOperator(
            field_name, field_name, OPROW) {}
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;
    if (type == MBVERTEX) {
      std::cout << "Hello Operator OpVolumeSide:"
                << " volume " << getVolume() << " normal " << getNormal()
                << endl;
    }
    MoFEMFunctionReturn(0);
  }
};

int main(int argc, char *argv[]) {

  // initialize petsc
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    type_name[MBVERTEX] = "Vertex";
    type_name[MBEDGE] = "Edge";
    type_name[MBTRI] = "Triangle";
    type_name[MBTET] = "Tetrahedra";

    // Register DM Manager
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    // Create MoAB database
    moab::Core moab_core;
    moab::Interface &moab = moab_core;

    // Create MoFEM database and link it to MoAB
    MoFEM::Core mofem_core(moab);
    MoFEM::Interface &m_field = mofem_core;

    // Simple interface
    Simple *simple_interface;
    CHKERR m_field.getInterface(simple_interface);

    // get options from command line
    CHKERR simple_interface->getOptions();
    // load mesh file
    CHKERR simple_interface->loadFile();
    // add fields
    CHKERR simple_interface->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE,
                                            3);
    CHKERR simple_interface->addBoundaryField("L", H1, AINSWORTH_LEGENDRE_BASE,
                                              3);
    CHKERR simple_interface->addSkeletonField("S", H1, AINSWORTH_LEGENDRE_BASE,
                                              3);
    // set fields order
    CHKERR simple_interface->setFieldOrder("U", 4);
    CHKERR simple_interface->setFieldOrder("L", 3);
    CHKERR simple_interface->setFieldOrder("S", 3);
    // setup problem
    CHKERR simple_interface->setUp();
    // create elements instances
    boost::shared_ptr<ForcesAndSourcesCore> domain_fe(
        new VolumeElementForcesAndSourcesCore(m_field));
    boost::shared_ptr<ForcesAndSourcesCore> boundary_fe(
        new FaceElementForcesAndSourcesCore(m_field));
    boost::shared_ptr<ForcesAndSourcesCore> skeleton_fe(
        new FaceElementForcesAndSourcesCore(m_field));
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe(
        new VolumeElementForcesAndSourcesCoreOnSide(m_field));
    // set operator to the volume element instance
    domain_fe->getOpPtrVector().push_back(new OpRow("U"));
    domain_fe->getOpPtrVector().push_back(new OpRowCol("U", "U", true));
    domain_fe->getOpPtrVector().push_back(new OpVolume("U"));
    // set operator to the face element instance
    boundary_fe->getOpPtrVector().push_back(new OpRow("L"));
    boundary_fe->getOpPtrVector().push_back(new OpRowCol("U", "L", false));
    boundary_fe->getOpPtrVector().push_back(new OpFace("L"));
    // set operator to the face element on skeleton instance
    skeleton_fe->getOpPtrVector().push_back(new OpRow("S"));
    skeleton_fe->getOpPtrVector().push_back(new OpFaceSide("S", side_fe));
    // set operator to the volume on side of the skeleton face
    side_fe->getOpPtrVector().push_back(new OpVolumeSide("U"));
    DM dm;
    // get dm
    CHKERR simple_interface->getDM(&dm);
    // iterate domain elements and execute element instance with operator on
    // mesh entities
    CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
                                    domain_fe);
    // iterate boundary elements and execute element instance with operator on
    // mesh entities
    CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getBoundaryFEName(),
                                    boundary_fe);
    // iterate skeleton elements and execute element instance with operator on
    // mesh entities
    CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getSkeletonFEName(),
                                    skeleton_fe);
    // destroy dm
    CHKERR DMDestroy(&dm);
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}
