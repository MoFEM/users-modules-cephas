/** \file navier_stokes.cpp
 * \example navier_stokes.cpp
 *
 * Example of viscous fluid flow problem
 *
 */

#include <BasicFiniteElements.hpp>

using PostProcVol = PostProcBrokenMeshInMoab<VolumeElementForcesAndSourcesCore>;
using PostProcFace = PostProcBrokenMeshInMoab<FaceElementForcesAndSourcesCore>;

using namespace boost::numeric;
using namespace MoFEM;
using namespace std;

static char help[] = "Navier-Stokes Example\n";

double NavierStokesElement::LoadScale::lambda = 1;

//! [Example Navier Stokes]
struct NavierStokesExample {

  NavierStokesExample(MoFEM::Interface &m_field) : mField(m_field) {
    commonData = boost::make_shared<NavierStokesElement::CommonData>(m_field);
  }
  ~NavierStokesExample() {
    CHKERR SNESDestroy(&snes);
    CHKERR DMDestroy(&dM);
  }

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  PetscBool isPartitioned = PETSC_FALSE;

  int orderVelocity = 2; // default approximation orderVelocity
  int orderPressure = 1; // default approximation orderPressure
  int numHoLevels = 0;
  PetscBool isDiscontPressure = PETSC_FALSE;
  PetscBool isHoGeometry = PETSC_TRUE;

  double pressureScale = 1.0;
  double lengthScale = 1.0;
  double velocityScale = 1.0;
  double reNumber = 1.0;
  double density;
  double viscosity;
  PetscBool isStokesFlow = PETSC_FALSE;

  int numSteps = 1;
  double lambdaStep = 1.0;
  double lambda = 0.0;
  int step;

  Range solidFaces;
  BitRefLevel bitLevel;

  DM dM;
  SNES snes;

  boost::shared_ptr<NavierStokesElement::CommonData> commonData;

  SmartPetscObj<Vec> D;
  SmartPetscObj<Vec> F;
  SmartPetscObj<Mat> Aij;

  boost::shared_ptr<VolumeElementForcesAndSourcesCore> feLhsPtr;
  boost::shared_ptr<VolumeElementForcesAndSourcesCore> feRhsPtr;

  boost::shared_ptr<FaceElementForcesAndSourcesCore> feDragPtr;
  boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> feDragSidePtr;

  boost::shared_ptr<DirichletDisplacementBc> dirichletBcPtr;
  boost::ptr_map<std::string, NeumannForcesSurface> neumannForces;

  boost::shared_ptr<PostProcVol> fePostProcPtr;
  boost::shared_ptr<PostProcFace> fePostProcDragPtr;

  MoFEMErrorCode readInput();
  MoFEMErrorCode findBlocks();
  MoFEMErrorCode setupFields();
  MoFEMErrorCode defineFiniteElements();
  MoFEMErrorCode setupDiscreteManager();
  MoFEMErrorCode setupAlgebraicStructures();
  MoFEMErrorCode setupElementInstances();
  MoFEMErrorCode setupSNES();
  MoFEMErrorCode solveProblem();
  MoFEMErrorCode postProcess();
};
//! [Example Navier Stokes]

//! [Run problem]
MoFEMErrorCode NavierStokesExample::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readInput();
  CHKERR findBlocks();
  CHKERR setupFields();
  CHKERR defineFiniteElements();
  CHKERR setupDiscreteManager();
  CHKERR setupAlgebraicStructures();
  CHKERR setupElementInstances();
  CHKERR setupSNES();
  CHKERR solveProblem();
  MoFEMFunctionReturn(0);
}
//! [Run problem]

//! [Read input]
MoFEMErrorCode NavierStokesExample::readInput() {
  MoFEMFunctionBegin;
  char mesh_file_name[255];
  PetscBool flg_mesh_file;

  CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "NAVIER_STOKES problem",
                           "none");

  CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                            mesh_file_name, 255, &flg_mesh_file);
  CHKERR PetscOptionsBool("-my_is_partitioned", "is partitioned?", "",
                          isPartitioned, &isPartitioned, PETSC_NULL);

  CHKERR PetscOptionsInt("-my_order_u", "approximation orderVelocity", "",
                         orderVelocity, &orderVelocity, PETSC_NULL);
  CHKERR PetscOptionsInt("-my_order_p", "approximation orderPressure", "",
                         orderPressure, &orderPressure, PETSC_NULL);
  CHKERR PetscOptionsInt("-my_num_ho_levels",
                         "number of higher order boundary levels", "",
                         numHoLevels, &numHoLevels, PETSC_NULL);
  CHKERR PetscOptionsBool("-my_discont_pressure", "discontinuous pressure", "",
                          isDiscontPressure, &isDiscontPressure, PETSC_NULL);
  CHKERR PetscOptionsBool("-my_ho_geometry", "use second order geometry", "",
                          isHoGeometry, &isHoGeometry, PETSC_NULL);

  CHKERR PetscOptionsScalar("-my_length_scale", "length scale", "", lengthScale,
                            &lengthScale, PETSC_NULL);
  CHKERR PetscOptionsScalar("-my_velocity_scale", "velocity scale", "",
                            velocityScale, &velocityScale, PETSC_NULL);
  CHKERR PetscOptionsBool("-my_stokes_flow", "stokes flow", "", isStokesFlow,
                          &isStokesFlow, PETSC_NULL);

  CHKERR PetscOptionsInt("-my_step_num", "number of steps", "", numSteps,
                         &numSteps, PETSC_NULL);

  ierr = PetscOptionsEnd();

  if (flg_mesh_file != PETSC_TRUE) {
    SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
  }

  // Read whole mesh or part of it if partitioned
  if (isPartitioned == PETSC_TRUE) {
    // This is a case of distributed mesh and algebra. In that case each
    // processor keeps only part of the problem.
    const char *option;
    option = "PARALLEL=READ_PART;"
             "PARALLEL_RESOLVE_SHARED_ENTS;"
             "PARTITION=PARALLEL_PARTITION;";
    CHKERR mField.get_moab().load_file(mesh_file_name, 0, option);
  } else {
    // In this case, we have distributed algebra, i.e. assembly of vectors and
    // matrices is in parallel, but whole mesh is stored on all processors.
    // snes and matrix scales well, however problem set-up of problem is
    // not fully parallel.
    const char *option;
    option = "";
    CHKERR mField.get_moab().load_file(mesh_file_name, 0, option);
  }
  CHKERR mField.rebuild_database();

  bitLevel.set(0);
  CHKERR mField.getInterface<BitRefManager>()->setBitRefLevelByDim(0, 3,
                                                                   bitLevel);

  MoFEMFunctionReturn(0);
}
//! [Read input]

//! [Find blocks]
MoFEMErrorCode NavierStokesExample::findBlocks() {
  MoFEMFunctionBegin;

  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
    if (bit->getName().compare(0, 9, "MAT_FLUID") == 0) {
      const int id = bit->getMeshsetId();
      CHKERR mField.get_moab().get_entities_by_dimension(
          bit->getMeshset(), 3, commonData->setOfBlocksData[id].eNts, true);

      std::vector<double> attributes;
      bit->getAttributes(attributes);
      if (attributes.size() < 2) {
        SETERRQ1(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                 "should be at least 2 attributes but is %d",
                 attributes.size());
      }
      commonData->setOfBlocksData[id].iD = id;
      commonData->setOfBlocksData[id].fluidViscosity = attributes[0];
      commonData->setOfBlocksData[id].fluidDensity = attributes[1];

      viscosity = commonData->setOfBlocksData[id].fluidViscosity;
      density = commonData->setOfBlocksData[id].fluidDensity;
    }
  }

  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
    if (bit->getName().compare(0, 9, "INT_SOLID") == 0) {
      Range tets, tet;
      const int id = bit->getMeshsetId();
      CHKERR mField.get_moab().get_entities_by_type(
          bit->getMeshset(), MBTRI, commonData->setOfFacesData[id].eNts, true);
      solidFaces.merge(commonData->setOfFacesData[id].eNts);
      CHKERR mField.get_moab().get_adjacencies(
          commonData->setOfFacesData[id].eNts, 3, true, tets,
          moab::Interface::UNION);
      tet = Range(tets.front(), tets.front());
      for (auto &bit : commonData->setOfBlocksData) {
        if (bit.second.eNts.contains(tet)) {
          commonData->setOfFacesData[id].fluidViscosity =
              bit.second.fluidViscosity;
          commonData->setOfFacesData[id].fluidDensity = bit.second.fluidDensity;
          commonData->setOfFacesData[id].iD = id;
          break;
        }
      }
      if (commonData->setOfFacesData[id].fluidViscosity < 0) {
        SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                "Cannot find a fluid block adjacent to a given solid face");
      }
    }
  }

  MoFEMFunctionReturn(0);
}
//! [Find blocks]

//! [Setup fields]
MoFEMErrorCode NavierStokesExample::setupFields() {
  MoFEMFunctionBegin;

  CHKERR mField.add_field("VELOCITY", H1, AINSWORTH_LEGENDRE_BASE, 3);
  if (isDiscontPressure) {
    CHKERR mField.add_field("PRESSURE", L2, AINSWORTH_LEGENDRE_BASE, 1);
  } else {
    CHKERR mField.add_field("PRESSURE", H1, AINSWORTH_LEGENDRE_BASE, 1);
  }
  CHKERR mField.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                          3);

  CHKERR mField.add_ents_to_field_by_dim(0, 3, "VELOCITY");
  CHKERR mField.add_ents_to_field_by_dim(0, 3, "PRESSURE");
  CHKERR mField.add_ents_to_field_by_dim(0, 3, "MESH_NODE_POSITIONS");

  CHKERR mField.set_field_order(0, MBVERTEX, "VELOCITY", 1);
  CHKERR mField.set_field_order(0, MBEDGE, "VELOCITY", orderVelocity);
  CHKERR mField.set_field_order(0, MBTRI, "VELOCITY", orderVelocity);
  CHKERR mField.set_field_order(0, MBTET, "VELOCITY", orderVelocity);

  if (!isDiscontPressure) {
    CHKERR mField.set_field_order(0, MBVERTEX, "PRESSURE", 1);
    CHKERR mField.set_field_order(0, MBEDGE, "PRESSURE", orderPressure);
    CHKERR mField.set_field_order(0, MBTRI, "PRESSURE", orderPressure);
  }
  CHKERR mField.set_field_order(0, MBTET, "PRESSURE", orderPressure);

  if (numHoLevels > 0) {
    std::vector<Range> levels(numHoLevels);
    Range ents;
    ents.merge(solidFaces);
    for (int ll = 0; ll != numHoLevels; ll++) {
      Range verts;
      CHKERR mField.get_moab().get_connectivity(ents, verts, true);
      for (auto d : {1, 2, 3}) {
        CHKERR mField.get_moab().get_adjacencies(verts, d, false, ents,
                                                 moab::Interface::UNION);
      }
      levels[ll] = subtract(ents, ents.subset_by_type(MBVERTEX));
    }
    for (int ll = numHoLevels - 1; ll >= 1; ll--) {
      levels[ll] = subtract(levels[ll], levels[ll - 1]);
    }

    int add_order = 1;
    for (int ll = numHoLevels - 1; ll >= 0; ll--) {
      if (isPartitioned)
        CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(
            levels[ll]);

      CHKERR mField.set_field_order(levels[ll], "VELOCITY",
                                    orderVelocity + add_order);
      if (!isDiscontPressure)
        CHKERR mField.set_field_order(levels[ll], "PRESSURE",
                                      orderPressure + add_order);
      else
        CHKERR mField.set_field_order(levels[ll].subset_by_type(MBTET),
                                      "PRESSURE", orderPressure + add_order);
      ++add_order;
    }
  }

  CHKERR mField.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);
  // Set 2nd order of approximation of geometry
  if (isHoGeometry) {
    Range ents, edges;
    CHKERR mField.get_moab().get_entities_by_dimension(0, 3, ents);
    CHKERR mField.get_moab().get_adjacencies(ents, 1, false, edges,
                                             moab::Interface::UNION);
    if (isPartitioned)
      CHKERR mField.getInterface<CommInterface>()->synchroniseEntities(edges);
    CHKERR mField.set_field_order(edges, "MESH_NODE_POSITIONS", 2);
  }

  if (isPartitioned) {
    CHKERR mField.getInterface<CommInterface>()->synchroniseFieldEntities(
        "VELOCITY");
    CHKERR mField.getInterface<CommInterface>()->synchroniseFieldEntities(
        "PRESSURE");
    CHKERR mField.getInterface<CommInterface>()->synchroniseFieldEntities(
        "MESH_NODE_POSITIONS");
  }

  CHKERR mField.build_fields();

  Projection10NodeCoordsOnField ent_method_material(mField,
                                                    "MESH_NODE_POSITIONS");
  CHKERR mField.loop_dofs("MESH_NODE_POSITIONS", ent_method_material);

  MoFEMFunctionReturn(0);
}
//! [Setup fields]

//! [Define finite elements]
MoFEMErrorCode NavierStokesExample::defineFiniteElements() {
  MoFEMFunctionBegin;

  // Add finite element (this defines element, declaration comes later)
  CHKERR NavierStokesElement::addElement(mField, "NAVIER_STOKES", "VELOCITY",
                                         "PRESSURE", "MESH_NODE_POSITIONS");

  CHKERR NavierStokesElement::addElement(mField, "DRAG", "VELOCITY", "PRESSURE",
                                         "MESH_NODE_POSITIONS", 2, &solidFaces);

  // setup elements for loading
  CHKERR MetaNeumannForces::addNeumannBCElements(mField, "VELOCITY");

  // build finite elements
  CHKERR mField.build_finite_elements();
  // build adjacencies between elements and degrees of freedom
  CHKERR mField.build_adjacencies(bitLevel);

  MoFEMFunctionReturn(0);
}
//! [Define finite elements]

//! [Setup discrete manager]
MoFEMErrorCode NavierStokesExample::setupDiscreteManager() {
  MoFEMFunctionBegin;
  DMType dm_name = "DM_NAVIER_STOKES";
  // Register DM problem
  CHKERR DMRegister_MoFEM(dm_name);
  CHKERR DMCreate(PETSC_COMM_WORLD, &dM);
  CHKERR DMSetType(dM, dm_name);
  // Create DM instance
  CHKERR DMMoFEMCreateMoFEM(dM, &mField, dm_name, bitLevel);
  // Configure DM form line command options s
  CHKERR DMSetFromOptions(dM);
  // Add elements to dM (only one here)
  CHKERR DMMoFEMAddElement(dM, "NAVIER_STOKES");
  CHKERR DMMoFEMAddElement(dM, "DRAG");
  if (mField.check_finite_element("PRESSURE_FE"))
    CHKERR DMMoFEMAddElement(dM, "PRESSURE_FE");
  CHKERR DMMoFEMSetIsPartitioned(dM, isPartitioned);
  // setup the DM
  CHKERR DMSetUp(dM);
  MoFEMFunctionReturn(0);
}
//! [Setup discrete manager]

//! [Setup algebraic structures]
MoFEMErrorCode NavierStokesExample::setupAlgebraicStructures() {
  MoFEMFunctionBegin;

  D = smartCreateDMVector(dM);
  F = smartVectorDuplicate(D);
  Aij = smartCreateDMMatrix(dM);

  CHKERR VecZeroEntries(F);
  CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);

  CHKERR VecZeroEntries(D);
  CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

  CHKERR MatSetOption(Aij, MAT_SPD, PETSC_TRUE);
  CHKERR MatZeroEntries(Aij);

  MoFEMFunctionReturn(0);
}
//! [Setup algebraic structures]

//! [Setup element instances]
MoFEMErrorCode NavierStokesExample::setupElementInstances() {
  MoFEMFunctionBegin;

  feLhsPtr = boost::shared_ptr<VolumeElementForcesAndSourcesCore>(
      new VolumeElementForcesAndSourcesCore(mField));
  feRhsPtr = boost::shared_ptr<VolumeElementForcesAndSourcesCore>(
      new VolumeElementForcesAndSourcesCore(mField));

  feLhsPtr->getRuleHook = NavierStokesElement::VolRule();
  feRhsPtr->getRuleHook = NavierStokesElement::VolRule();
  CHKERR addHOOpsVol("MESH_NODE_POSITIONS", *feLhsPtr, true, false, false,
                     true);
  CHKERR addHOOpsVol("MESH_NODE_POSITIONS", *feRhsPtr, true, false, false,
                     true);

  feDragPtr = boost::shared_ptr<FaceElementForcesAndSourcesCore>(
      new FaceElementForcesAndSourcesCore(mField));
  feDragSidePtr = boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide>(
      new VolumeElementForcesAndSourcesCoreOnSide(mField));

  feDragPtr->getRuleHook = NavierStokesElement::FaceRule();
  CHKERR addHOOpsVol("MESH_NODE_POSITIONS", *feDragSidePtr, true, false, false,
                     true);

  if (isStokesFlow) {
    CHKERR NavierStokesElement::setStokesOperators(
        feRhsPtr, feLhsPtr, "VELOCITY", "PRESSURE", commonData);
  } else {
    CHKERR NavierStokesElement::setNavierStokesOperators(
        feRhsPtr, feLhsPtr, "VELOCITY", "PRESSURE", commonData);
  }

  NavierStokesElement::setCalcDragOperators(feDragPtr, feDragSidePtr,
                                            "NAVIER_STOKES", "VELOCITY",
                                            "PRESSURE", commonData);

  dirichletBcPtr = boost::shared_ptr<DirichletDisplacementBc>(
      new DirichletDisplacementBc(mField, "VELOCITY", Aij, D, F));
  dirichletBcPtr->methodsOp.push_back(new NavierStokesElement::LoadScale());
  // dirichletBcPtr->snes_ctx = FEMethod::CTX_SNESNONE;
  dirichletBcPtr->snes_x = D;

  // Assemble pressure and traction forces
  CHKERR MetaNeumannForces::setMomentumFluxOperators(mField, neumannForces,
                                                     NULL, "VELOCITY");

  // for postprocessing:
  fePostProcPtr = boost::make_shared<PostProcVol>(mField);
  CHKERR addHOOpsVol("MESH_NODE_POSITIONS", *fePostProcPtr, true, false, false,
                     true);

  auto v_ptr = boost::make_shared<MatrixDouble>();
  auto grad_ptr = boost::make_shared<MatrixDouble>();
  auto pos_ptr = boost::make_shared<MatrixDouble>();
  auto p_ptr = boost::make_shared<VectorDouble>();

  fePostProcPtr->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<3>("VELOCITY", v_ptr));
  fePostProcPtr->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<3, 3>("VELOCITY", grad_ptr));
  fePostProcPtr->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues("PRESSURE", p_ptr));
  fePostProcPtr->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<3>("MESH_NODE_POSITIONS", pos_ptr));

  using OpPPMap = OpPostProcMapInMoab<3, 3>;

  fePostProcPtr->getOpPtrVector().push_back(

      new OpPPMap(

          fePostProcPtr->getPostProcMesh(), fePostProcPtr->getMapGaussPts(),

          {{"PRESSURE", p_ptr}},

          {{"VELOCITY", v_ptr}, {"MESH_NODE_POSITIONS", pos_ptr}},

          {{"VELOCITY_GRAD", grad_ptr}},

          {}

          )

  );

  fePostProcDragPtr = boost::make_shared<PostProcFace>(mField);
  fePostProcDragPtr->getOpPtrVector().push_back(
      new OpCalculateVectorFieldValues<3>("MESH_NODE_POSITIONS", pos_ptr));
  fePostProcDragPtr->getOpPtrVector().push_back(

      new OpPPMap(

          fePostProcDragPtr->getPostProcMesh(),
          fePostProcDragPtr->getMapGaussPts(),

          {},

          {{"MESH_NODE_POSITIONS", pos_ptr}},

          {},

          {}

          )

  );

  CHKERR NavierStokesElement::setPostProcDragOperators(
      fePostProcDragPtr, feDragSidePtr, "NAVIER_STOKES", "VELOCITY", "PRESSURE",
      commonData);

  MoFEMFunctionReturn(0);
}
//! [Setup element instances]

//! [Setup SNES]
MoFEMErrorCode NavierStokesExample::setupSNES() {
  MoFEMFunctionBegin;

  boost::ptr_map<std::string, NeumannForcesSurface>::iterator mit =
      neumannForces.begin();
  for (; mit != neumannForces.end(); mit++) {
    CHKERR DMMoFEMSNESSetFunction(dM, mit->first.c_str(),
                                  &mit->second->getLoopFe(), NULL, NULL);
  }

  boost::shared_ptr<FEMethod> null_fe;
  CHKERR DMMoFEMSNESSetFunction(dM, "NAVIER_STOKES", feRhsPtr, null_fe,
                                null_fe);
  CHKERR DMMoFEMSNESSetFunction(dM, DM_NO_ELEMENT, null_fe, null_fe,
                                dirichletBcPtr);

  CHKERR DMMoFEMSNESSetJacobian(dM, DM_NO_ELEMENT, null_fe, dirichletBcPtr,
                                null_fe);
  CHKERR DMMoFEMSNESSetJacobian(dM, "NAVIER_STOKES", feLhsPtr, null_fe,
                                null_fe);
  CHKERR DMMoFEMSNESSetJacobian(dM, DM_NO_ELEMENT, null_fe, null_fe,
                                dirichletBcPtr);

  SnesCtx *snes_ctx;
  // create snes nonlinear solver
  CHKERR SNESCreate(PETSC_COMM_WORLD, &snes);
  CHKERR DMMoFEMGetSnesCtx(dM, &snes_ctx);
  CHKERR SNESSetFunction(snes, F, SnesRhs, snes_ctx);
  CHKERR SNESSetJacobian(snes, Aij, Aij, SnesMat, snes_ctx);
  CHKERR SNESSetFromOptions(snes);

  MoFEMFunctionReturn(0);
}
//! [Setup SNES]

//! [Solve problem]
MoFEMErrorCode NavierStokesExample::solveProblem() {
  MoFEMFunctionBegin;

  auto scale_problem = [&](double U, double L, double P) {
    MoFEMFunctionBegin;
    CHKERR mField.getInterface<FieldBlas>()->fieldScale(L,
                                                        "MESH_NODE_POSITIONS");

    ProjectionFieldOn10NodeTet ent_method_on_10nodeTet(
        mField, "MESH_NODE_POSITIONS", true, true);
    CHKERR mField.loop_dofs("MESH_NODE_POSITIONS", ent_method_on_10nodeTet);
    ent_method_on_10nodeTet.setNodes = false;
    CHKERR mField.loop_dofs("MESH_NODE_POSITIONS", ent_method_on_10nodeTet);

    CHKERR mField.getInterface<FieldBlas>()->fieldScale(U, "VELOCITY");
    CHKERR mField.getInterface<FieldBlas>()->fieldScale(P, "PRESSURE");
    MoFEMFunctionReturn(0);
  };

  pressureScale = viscosity * velocityScale / lengthScale;
  NavierStokesElement::LoadScale::lambda = 1.0 / velocityScale;
  CHKERR scale_problem(1.0 / velocityScale, 1.0 / lengthScale,
                       1.0 / pressureScale);

  step = 0;

  CHKERR PetscPrintf(PETSC_COMM_WORLD, "Viscosity: %6.4e\n", viscosity);
  CHKERR PetscPrintf(PETSC_COMM_WORLD, "Density: %6.4e\n", density);
  CHKERR PetscPrintf(PETSC_COMM_WORLD, "Length scale: %6.4e\n", lengthScale);
  CHKERR PetscPrintf(PETSC_COMM_WORLD, "Velocity scale: %6.4e\n",
                     velocityScale);
  CHKERR PetscPrintf(PETSC_COMM_WORLD, "Pressure scale: %6.4e\n",
                     pressureScale);
  if (isStokesFlow) {
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "Re number: 0 (Stokes flow)\n");
  } else {
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "Re number: %6.4e\n",
                       density / viscosity * velocityScale * lengthScale);
  }

  lambdaStep = 1.0 / numSteps;

  while (lambda < 1.0 - 1e-12) {

    lambda += lambdaStep;

    if (isStokesFlow) {
      reNumber = 0.0;
      for (auto &bit : commonData->setOfBlocksData) {
        bit.second.inertiaCoef = 0.0;
        bit.second.viscousCoef = 1.0;
      }
    } else {
      reNumber = density / viscosity * velocityScale * lengthScale * lambda;
      for (auto &bit : commonData->setOfBlocksData) {
        bit.second.inertiaCoef = reNumber;
        bit.second.viscousCoef = 1.0;
      }
    }

    CHKERR PetscPrintf(
        PETSC_COMM_WORLD,
        "Step: %d | Lambda: %6.4e | Inc: %6.4e | Re number: %6.4e \n", step,
        lambda, lambdaStep, reNumber);

    CHKERR DMoFEMPreProcessFiniteElements(dM, dirichletBcPtr.get());

    CHKERR VecAssemblyBegin(D);
    CHKERR VecAssemblyEnd(D);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR SNESSolve(snes, PETSC_NULL, D);

    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToGlobalVector(dM, D, INSERT_VALUES, SCATTER_REVERSE);

    CHKERR scale_problem(velocityScale, lengthScale, pressureScale);

    CHKERR postProcess();

    CHKERR scale_problem(1.0 / velocityScale, 1.0 / lengthScale,
                         1.0 / pressureScale);

    step++;
  }

  MoFEMFunctionReturn(0);
}
//! [Solve problem]

//! [Post process]
MoFEMErrorCode NavierStokesExample::postProcess() {
  MoFEMFunctionBegin;

  string out_file_name;

  CHKERR DMoFEMLoopFiniteElements(dM, "NAVIER_STOKES", fePostProcPtr);
  {
    std::ostringstream stm;
    stm << "out_" << step << ".h5m";
    out_file_name = stm.str();
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "Write file %s\n",
                       out_file_name.c_str());
    CHKERR fePostProcPtr->writeFile(out_file_name.c_str());
  }

  CHKERR VecZeroEntries(commonData->pressureDragForceVec);
  CHKERR VecZeroEntries(commonData->shearDragForceVec);
  CHKERR VecZeroEntries(commonData->totalDragForceVec);
  CHKERR DMoFEMLoopFiniteElements(dM, "DRAG", feDragPtr);

  auto get_vec_data = [&](auto vec, std::array<double, 3> &data) {
    MoFEMFunctionBegin;
    CHKERR VecAssemblyBegin(vec);
    CHKERR VecAssemblyEnd(vec);
    const double *array;
    CHKERR VecGetArrayRead(vec, &array);
    if (mField.get_comm_rank() == 0) {
      for (int i : {0, 1, 2})
        data[i] = array[i];
    }
    CHKERR VecRestoreArrayRead(vec, &array);
    MoFEMFunctionReturn(0);
  };

  std::array<double, 3> pressure_drag;
  std::array<double, 3> shear_drag;
  std::array<double, 3> total_drag;

  CHKERR get_vec_data(commonData->pressureDragForceVec, pressure_drag);
  CHKERR get_vec_data(commonData->shearDragForceVec, shear_drag);
  CHKERR get_vec_data(commonData->totalDragForceVec, total_drag);

  if (mField.get_comm_rank() == 0) {
    MOFEM_LOG_C("SELF", Sev::inform,
                "Pressure drag force: [ %6.4e, %6.4e, %6.4e ]",
                pressure_drag[0], pressure_drag[1], pressure_drag[2]);
    MOFEM_LOG_C("SELF", Sev::inform,
                "Shear drag force: [ %6.4e, %6.4e, %6.4e ]", shear_drag[0],
                shear_drag[1], shear_drag[2]);
    MOFEM_LOG_C("SELF", Sev::inform,
                "Total drag force: [ %6.4e, %6.4e, %6.4e ]", total_drag[0],
                total_drag[1], total_drag[2]);
  }

  CHKERR DMoFEMLoopFiniteElements(dM, "DRAG", fePostProcDragPtr);
  {
    std::ostringstream stm;
    stm << "out_drag_" << step << ".h5m";
    out_file_name = stm.str();
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "out file %s\n",
                       out_file_name.c_str());
    CHKERR fePostProcDragPtr->writeFile(out_file_name);
  }
  MoFEMFunctionReturn(0);
}
//! [Post process]

//! [Main function]
int main(int argc, char *argv[]) {

  const char param_file[] = "param_file.petsc";
  // Initialise MoFEM
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  try {

    // Create mesh database
    moab::Core mb_instance;              // create database
    moab::Interface &moab = mb_instance; // create interface to database

    // Create MoFEM database and link it to MoAB
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    NavierStokesExample ex(m_field);
    CHKERR ex.runProblem();
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();
  return 0;
}
//! [Main function]