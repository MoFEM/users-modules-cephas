/**
 * \file hello_world.cpp
 * \ingroup mofem_simple_interface
 * \example hello_world.cpp
 *
 * Prints basic information about users data operator.
 * See more details in \ref hello_world_tut1
 *
 */

#include <BasicFiniteElements.hpp>
using namespace MoFEM;

static char help[] = "...\n\n";

static map<EntityType, std::string> type_name;

#define HelloFunctionBegin                                                     \
  MoFEMFunctionBegin;                                                          \
  MOFEM_LOG_CHANNEL("SYNC");                                                   \
  MOFEM_LOG_FUNCTION();                                                        \
  MOFEM_LOG_TAG("WORLD", "HelloWorld");

struct OpRow : public ForcesAndSourcesCore::UserDataOperator {
  OpRow(const std::string &field_name)
      : ForcesAndSourcesCore::UserDataOperator(field_name, field_name, OPROW) {}
  MoFEMErrorCode doWork(int side, EntityType type,
                        EntitiesFieldData::EntData &data) {
    HelloFunctionBegin;
    if (type == MBVERTEX) {
      // get number of evaluated element in the loop
      MOFEM_LOG("SYNC", Sev::inform) << "**** " << getNinTheLoop() << " ****";
      MOFEM_LOG("SYNC", Sev::inform) << "**** Operators ****";
    }
    MOFEM_LOG("SYNC", Sev::inform)
        << "Hello Operator OpRow:"
        << " field name " << rowFieldName << " side " << side << " type "
        << type_name[type] << " nb dofs on entity " << data.getIndices().size();
    MoFEMFunctionReturn(0);
  }
};

struct OpRowCol : public ForcesAndSourcesCore::UserDataOperator {
  OpRowCol(const std::string row_field, const std::string col_field,
           const bool symm)
      : ForcesAndSourcesCore::UserDataOperator(row_field, col_field, OPROWCOL,
                                               symm) {}
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        EntitiesFieldData::EntData &row_data,
                        EntitiesFieldData::EntData &col_data) {
    HelloFunctionBegin;
    MOFEM_LOG("SYNC", Sev::inform)
        << "Hello Operator OpRowCol:"
        << " row field name " << rowFieldName << " row side " << row_side
        << " row type " << type_name[row_type] << " nb dofs on row entity"
        << row_data.getIndices().size() << " : "
        << " col field name " << colFieldName << " col side " << col_side
        << " col type " << type_name[col_type] << " nb dofs on col entity"
        << col_data.getIndices().size();
    MoFEMFunctionReturn(0);
  }
};

struct OpVolume : public VolumeElementForcesAndSourcesCore::UserDataOperator {
  OpVolume(const std::string &field_name)
      : VolumeElementForcesAndSourcesCore::UserDataOperator(field_name,
                                                            field_name, OPROW) {
  }
  MoFEMErrorCode doWork(int side, EntityType type,
                        EntitiesFieldData::EntData &data) {
    HelloFunctionBegin;
    if (type == MBVERTEX) {
      MOFEM_LOG("SYNC", Sev::inform) << "Hello Operator OpVolume:"
                                     << " volume " << getVolume();
    }
    MoFEMFunctionReturn(0);
  }
};

struct OpFace : public FaceElementForcesAndSourcesCore::UserDataOperator {
  OpFace(const std::string &field_name)
      : FaceElementForcesAndSourcesCore::UserDataOperator(field_name, OPROW) {}
  MoFEMErrorCode doWork(int side, EntityType type,
                        EntitiesFieldData::EntData &data) {
    HelloFunctionBegin;
    if (type == MBVERTEX) {
      MOFEM_LOG("SYNC", Sev::inform) << "Hello Operator OpFace:"
                                     << " normal " << getNormal();
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
                        EntitiesFieldData::EntData &data) {

    HelloFunctionBegin;
    if (type == MBVERTEX) {
      MOFEM_LOG("SYNC", Sev::inform) << "Hello Operator OpSideFace";
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
                        EntitiesFieldData::EntData &data) {
    HelloFunctionBegin;
    if (type == MBVERTEX) {
      MOFEM_LOG("SYNC", Sev::inform)
          << "Hello Operator OpVolumeSide:"
          << " volume " << getVolume() << " normal " << getNormal();
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
    Simple *simple;
    CHKERR m_field.getInterface(simple);

    // get options from command line
    CHKERR simple->getOptions();
    // load mesh file
    CHKERR simple->loadFile();
    // add fields
    CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, 3);
    CHKERR simple->addBoundaryField("L", H1, AINSWORTH_LEGENDRE_BASE, 3);
    CHKERR simple->addSkeletonField("S", H1, AINSWORTH_LEGENDRE_BASE, 3);
    // set fields order
    CHKERR simple->setFieldOrder("U", 4);
    CHKERR simple->setFieldOrder("L", 3);
    CHKERR simple->setFieldOrder("S", 3);

    // setup problem
    CHKERR simple->setUp();

    auto pipeline_mng = m_field.getInterface<PipelineManager>();

    //! [set operator to the volume element instance]
    pipeline_mng->getOpDomainRhsPipeline().push_back(new OpRow("U"));
    pipeline_mng->getOpDomainRhsPipeline().push_back(new OpVolume("U"));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpRowCol("U", "U", true));
    //! [set operator to the volume element instance]

    //! [set operator to the face element instance]
    pipeline_mng->getOpBoundaryRhsPipeline().push_back(new OpRow("L"));
    pipeline_mng->getOpBoundaryRhsPipeline().push_back(new OpFace("L"));
    pipeline_mng->getOpBoundaryLhsPipeline().push_back(
        new OpRowCol("U", "L", false));
    //! [set operator to the face element instance]

    //! [create skeleton side element and push operator to it]
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe(
        new VolumeElementForcesAndSourcesCoreOnSide(m_field));
    side_fe->getOpPtrVector().push_back(new OpVolumeSide("U"));
    //! [create skeleton side element and push operator to it]

    //! [set operator to the face element on skeleton instance]
    pipeline_mng->getOpSkeletonRhsPipeline().push_back(new OpRow("S"));
    pipeline_mng->getOpSkeletonRhsPipeline().push_back(
        new OpFaceSide("S", side_fe));
    //! [set operator to the face element on skeleton instance]

    //! [executing finite elements]
    CHKERR pipeline_mng->loopFiniteElements();
    //! [executing finite elements] 

    MOFEM_LOG_SYNCHRONISE(m_field.get_comm());
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}
