/**
 * \file level_set.cpp
 * \example level_set.cpp
 *
 *
 * Calculate level set for initally given surface
 */

#include <BasicFiniteElements.hpp>
using namespace MoFEM;

static char help[] = "...\n\n";

struct CalculateDistantFromSurface {

  CalculateDistantFromSurface(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode operator()(const Range &surface) {
    MoFEMFunctionBegin;
    treeSurfPtr = boost::shared_ptr<OrientedBoxTreeTool>(
        new OrientedBoxTreeTool(&mField.get_moab(), "ROOTSETSURF", true));
    CHKERR treeSurfPtr->build(surface, rootSetSurf);
    DistanceFromVertex surf_dist(treeSurfPtr, rootSetSurf);
    CHKERR mField.loop_entities("PHI", surf_dist);
    MoFEMFunctionReturn(0);
  }

private:
  MoFEM::Interface &mField;
  boost::shared_ptr<OrientedBoxTreeTool> treeSurfPtr;
  EntityHandle rootSetSurf;

  struct DistanceFromVertex: public EntityMethod {

    DistanceFromVertex(boost::shared_ptr<OrientedBoxTreeTool> &tree,
                       EntityHandle root)
        : EntityMethod(), treeSurfPtr(tree), rootSetSurf(root) {}

    MoFEMErrorCode operator()() {
      MoFEMFunctionBegin;
      EntityHandle vert = entPtr->getEnt();
      VectorDouble3 coords(3);
      CHKERR entPtr->getBasicDataPtr()->moab.get_coords(&vert, 1,
                                                        &*coords.begin());
      VectorDouble3 point_out(3);
      EntityHandle facets_out;
      CHKERR treeSurfPtr->closest_to_location(&coords[0], rootSetSurf,
                                              &point_out[0], facets_out);
      VectorDouble3 delta = point_out - coords;
      entPtr->getEntFieldData()[0] = norm_2(delta);
      MoFEMFunctionReturn(0);
    };

  private:
    boost::shared_ptr<OrientedBoxTreeTool> treeSurfPtr;
    EntityHandle rootSetSurf;
  };


};

int main(int argc, char *argv[]) {

  // initialize petsc
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    PetscBool flg_myfile = PETSC_TRUE;
    char mesh_file_name[255];
    int surface_side_set = 200;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Level set", "none");
    CHKERR PetscOptionsInt("-surface_side_set", "surface side set", "",
                           surface_side_set, &surface_side_set, PETSC_NULL);
    ierr = PetscOptionsEnd();
    CHKERRG(ierr);

    if (flg_myfile != PETSC_TRUE)
      SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
              "*** ERROR -my_file (MESH FILE NEEDED)");

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByType(
        0, MBTET, BitRefLevel().set());

    // get surface entities form side set
    Range surface;
    if (m_field.getInterface<MeshsetsManager>()->checkMeshset(surface_side_set,
                                                            SIDESET))
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          surface_side_set, SIDESET, 2, surface, true);
    if (surface.empty())
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "No surface to cut");

    // Simple interface
    Simple *simple_interface;
    CHKERR m_field.getInterface(simple_interface);
    CHKERR simple_interface->getOptions();
    CHKERR simple_interface->loadFile(); 

    // add ields
    CHKERR simple_interface->addDomainField("PHI", H1, AINSWORTH_LEGENDRE_BASE,
                                            1);
   // set fields order
    CHKERR simple_interface->setFieldOrder("PHI", 1);
    // setup problem
    CHKERR simple_interface->setUp();

    boost::shared_ptr<PostProcVolumeOnRefinedMesh> post_proc_volume =
        boost::shared_ptr<PostProcVolumeOnRefinedMesh>(
            new PostProcVolumeOnRefinedMesh(m_field));
    boost::shared_ptr<ForcesAndSourcesCore> null;

    post_proc_volume->generateReferenceElementMesh();
    post_proc_volume->addFieldValuesPostProc("PHI");
    post_proc_volume->addFieldValuesGradientPostProc("PHI");

    auto dm = simple_interface->getDM();

    CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
                                    post_proc_volume);
    CHKERR post_proc_volume->writeFile("out_level.h5m");

  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}
