/** \file mesh_smoothing.cpp
 * \brief Improve mesh quality using Volume-length quality measure with barrier
 * \example mesh_smoothing.cpp
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

#include <MoFEM.hpp>
using namespace MoFEM;

#include <BasicFiniteElements.hpp>

#include <Smoother.hpp>
#include <SurfaceSlidingConstrains.hpp>
#include <VolumeLengthQuality.hpp>

using namespace MoFEM;

static char help[] = "mesh smoothing\n\n";

PetscBool flg_myfile = PETSC_TRUE;
char mesh_file_name[255];
int edges_block_set = 1;
int vertex_block_set = 2;
PetscBool output_vtk = PETSC_TRUE;
double tol = 0.1;
double gamma_factor = 0.8;

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Mesh cut options", "none");
    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_myfile);
    CHKERR PetscOptionsInt("-edges_block_set", "edges side set", "",
                           edges_block_set, &edges_block_set, PETSC_NULL);
    CHKERR PetscOptionsInt("-vertex_block_set", "vertex side set", "",
                           vertex_block_set, &vertex_block_set, PETSC_NULL);
    CHKERR PetscOptionsBool("-output_vtk", "if true outout vtk file", "",
                            output_vtk, &output_vtk, PETSC_NULL);
    CHKERR PetscOptionsScalar("-quality_reduction_tol", "",
                              "Tolerance of quality reduction", tol, &tol,
                              PETSC_NULL);
    CHKERR PetscOptionsScalar("-gamma_factor", "",
                              "Gamma factor", gamma_factor, &gamma_factor,
                              PETSC_NULL);
    ierr = PetscOptionsEnd();
    CHKERRG(ierr);

    // Create MoAB database
    moab::Core moab_core;              // create database
    moab::Interface &moab = moab_core; // create interface to database
    // Create MoFEM database and link it to MoAB
    MoFEM::Core mofem_core(moab);           // create database
    MoFEM::Interface &m_field = mofem_core; // create interface to database
    // Register DM Manager
    CHKERR DMRegister_MoFEM("DMMOFEM"); // register MoFEM DM in PETSc

    // Get simple interface is simplified version enabling quick and
    // easy construction of problem.
    Simple *simple_interface;
    // Query interface and get pointer to Simple interface
    CHKERR m_field.getInterface(simple_interface);

    // Build problem with simple interface
    {
      // Get options for simple interface from command line
      CHKERR simple_interface->getOptions();
      // Load mesh file to database
      CHKERR simple_interface->loadFile();

      // Add domain filed "U" in space H1 and Legendre base, Ainsworth recipe is
      // used to construct base functions.
      CHKERR simple_interface->addDomainField("MESH_NODE_POSITIONS", H1,
                                              AINSWORTH_LEGENDRE_BASE, 3);
      // Add Lagrange multiplier field on body boundary
      CHKERR simple_interface->addBoundaryField("LAMBDA_SURFACE", H1,
                                                AINSWORTH_LEGENDRE_BASE, 1);

      // Set fields order domain and boundary fields.
      CHKERR simple_interface->setFieldOrder("MESH_NODE_POSITIONS",
                                             1); // to approximate function
      CHKERR simple_interface->setFieldOrder("LAMBDA_SURFACE",
                                             1); // to Lagrange multipliers

      simple_interface->getDomainFEName() = "SMOOTHING";
      simple_interface->getBoundaryFEName() = "SURFACE_SLIDING";

      // Other fields and finite elements added to database directly
      {
        if (m_field.getInterface<MeshsetsManager>()->checkMeshset(
                edges_block_set, BLOCKSET)) {
          // Declare approximation fields
          CHKERR m_field.add_field("LAMBDA_EDGE", H1, AINSWORTH_LOBATTO_BASE, 2,
                                   MB_TAG_SPARSE, MF_ZERO);
          Range edges;
          CHKERR m_field.getInterface<MeshsetsManager>()
              ->getEntitiesByDimension(edges_block_set, BLOCKSET, 1, edges,
                                       true);
          CHKERR m_field.add_ents_to_field_by_type(edges, MBEDGE,
                                                   "LAMBDA_EDGE");
          CHKERR m_field.getInterface<CommInterface>()
              ->synchroniseFieldEntities("LAMBDA_EDGE");
          CHKERR m_field.set_field_order(0, MBVERTEX, "LAMBDA_EDGE", 1);

          CHKERR m_field.add_finite_element("EDGE_SLIDING");
          CHKERR m_field.modify_finite_element_add_field_row(
              "EDGE_SLIDING", "MESH_NODE_POSITIONS");
          CHKERR m_field.modify_finite_element_add_field_col(
              "EDGE_SLIDING", "MESH_NODE_POSITIONS");
          CHKERR m_field.modify_finite_element_add_field_data(
              "EDGE_SLIDING", "MESH_NODE_POSITIONS");
          CHKERR m_field.modify_finite_element_add_field_row("EDGE_SLIDING",
                                                             "LAMBDA_EDGE");
          CHKERR m_field.modify_finite_element_add_field_col("EDGE_SLIDING",
                                                             "LAMBDA_EDGE");
          CHKERR m_field.modify_finite_element_add_field_data("EDGE_SLIDING",
                                                              "LAMBDA_EDGE");
          CHKERR m_field.add_ents_to_finite_element_by_type(edges, MBEDGE,
                                                           "EDGE_SLIDING");
          simple_interface->getOtherFiniteElements().push_back("EDGE_SLIDING");
        }
      }

      CHKERR simple_interface->defineFiniteElements();
      CHKERR simple_interface->defineProblem();
      CHKERR simple_interface->buildFields();
      // Remove vertices form LAMBDA_SURFACE which are on the edges
      if (m_field.getInterface<MeshsetsManager>()->checkMeshset(
              edges_block_set, BLOCKSET)) {
        Range edges;
        CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
            edges_block_set, BLOCKSET, 1, edges, true);
        Range verts;
        CHKERR m_field.get_moab().get_connectivity(edges, verts, true);
        CHKERR m_field.remove_ents_from_field("LAMBDA_SURFACE", verts);
      }
      CHKERR simple_interface->buildFiniteElements();
      CHKERR simple_interface->buildProblem();
    }

    struct ElementsAndOperators {

      MoFEM::Interface &mField;
      Vec minQualityVec;
      double *minQualityPtr;

      ElementsAndOperators(MoFEM::Interface &m_field) : mField(m_field) {
        ierr = VecCreateMPI(PETSC_COMM_WORLD, 1, m_field.get_comm_size(),
                            &minQualityVec);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecGetArray(minQualityVec, &minQualityPtr);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
      }

      virtual ~ElementsAndOperators() {
        ierr = VecRestoreArray(minQualityVec, &minQualityPtr);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecDestroy(&minQualityVec);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
      }

      double getMinQuality() const { return *minQualityPtr; }

      enum Tags {
        SMOOTHING_TAG = 1,
        SURFACE_CONSTRAINS_TAG,
        EDGE_CONSTRAINS_TAG
      };

      boost::shared_ptr<Smoother> smootherFe;
      boost::shared_ptr<Smoother::MyVolumeFE>
          feSmootherRhs; ///< Integrate smoothing operators
      boost::shared_ptr<Smoother::MyVolumeFE>
          feSmootherLhs; ///< Integrate smoothing operators
      boost::shared_ptr<VolumeLengthQuality<double> > volumeLengthDouble;
      boost::shared_ptr<VolumeLengthQuality<adouble> > volumeLengthAdouble;

      boost::shared_ptr<SurfaceSlidingConstrains> surfaceConstrain;
      boost::shared_ptr<SurfaceSlidingConstrains::DriverElementOrientation>
          skinOrientation;
      boost::shared_ptr<EdgeSlidingConstrains> edgeConstrain;

      boost::shared_ptr<DirichletFixFieldAtEntitiesBc> fixMaterialEnts;

      boost::shared_ptr<MoFEM::VolumeElementForcesAndSourcesCore> minQualityFe;

      struct MinQualityOp : MoFEM::ForcesAndSourcesCore::UserDataOperator {
        double *minQualityPtr;
        MinQualityOp(double *min_quality_ptr)
            : MoFEM::ForcesAndSourcesCore::UserDataOperator(
                  "MESH_NODE_POSITIONS", UserDataOperator::OPROW),
              minQualityPtr(min_quality_ptr) {}
        MoFEMErrorCode doWork(int side, EntityType type,
                              EntitiesFieldData::EntData &data) {
          MoFEMFunctionBeginHot;
          if (type != MBVERTEX)
            MoFEMFunctionReturnHot(0);
          double q = Tools::volumeLengthQuality(&*data.getFieldData().begin());
          *minQualityPtr = fmin(*minQualityPtr, q);
          MoFEMFunctionReturnHot(0);
        }
      };

      MoFEMErrorCode createSmoothingFE() {
        MoFEMFunctionBegin;
        smootherFe = boost::shared_ptr<Smoother>(new Smoother(mField));
        volumeLengthAdouble = boost::shared_ptr<VolumeLengthQuality<adouble> >(
            new VolumeLengthQuality<adouble>(BARRIER_AND_QUALITY, 1, 0));
        volumeLengthDouble = boost::shared_ptr<VolumeLengthQuality<double> >(
            new VolumeLengthQuality<double>(BARRIER_AND_QUALITY, 1, 0));

        Range tets;
        CHKERR mField.get_moab().get_entities_by_type(0, MBTET, tets);
        smootherFe->setOfBlocks[0].tEts.merge(tets);

        smootherFe->setOfBlocks[0].materialDoublePtr = volumeLengthDouble;
        smootherFe->setOfBlocks[0].materialAdoublePtr = volumeLengthAdouble;

        // set element data
        smootherFe->commonData.spatialPositions = "MESH_NODE_POSITIONS";
        smootherFe->commonData.meshPositions = "NONE";

        smootherFe->feRhs.meshPositionsFieldName = "NONE";
        smootherFe->feLhs.meshPositionsFieldName = "NONE";
        smootherFe->feRhs.addToRule = 0;
        smootherFe->feLhs.addToRule = 0;

        feSmootherRhs = smootherFe->feRhsPtr;
        feSmootherLhs = smootherFe->feLhsPtr;

        // Smoother right hand side
        feSmootherRhs->getOpPtrVector().push_back(
            new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
                "MESH_NODE_POSITIONS", smootherFe->commonData));
        feSmootherRhs->getOpPtrVector().push_back(
            new Smoother::OpJacobianSmoother(
                "MESH_NODE_POSITIONS", smootherFe->setOfBlocks.at(0),
                smootherFe->commonData, SMOOTHING_TAG, false));
        feSmootherRhs->getOpPtrVector().push_back(new Smoother::OpRhsSmoother(
            "MESH_NODE_POSITIONS", smootherFe->setOfBlocks[0],
            smootherFe->commonData, smootherFe->smootherData));

        // Smoother left hand side
        feSmootherLhs->getOpPtrVector().push_back(
            new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
                "MESH_NODE_POSITIONS", smootherFe->commonData));
        feSmootherLhs->getOpPtrVector().push_back(
            new Smoother::OpJacobianSmoother(
                "MESH_NODE_POSITIONS", smootherFe->setOfBlocks.at(0),
                smootherFe->commonData, SMOOTHING_TAG, true));
        feSmootherLhs->getOpPtrVector().push_back(new Smoother::OpLhsSmoother(
            "MESH_NODE_POSITIONS", "MESH_NODE_POSITIONS",
            smootherFe->setOfBlocks.at(0), smootherFe->commonData,
            smootherFe->smootherData, "LAMBDA_CRACKFRONT_AREA_TANGENT"));

        minQualityFe =
            boost::shared_ptr<MoFEM::VolumeElementForcesAndSourcesCore>(
                new MoFEM::VolumeElementForcesAndSourcesCore(mField));
        minQualityFe->getOpPtrVector().push_back(
            new MinQualityOp(minQualityPtr));

        Range fixed_vertex;
        if (mField.getInterface<MeshsetsManager>()->checkMeshset(
                vertex_block_set, BLOCKSET)) {
          CHKERR mField.getInterface<MeshsetsManager>()->getEntitiesByDimension(
              vertex_block_set, BLOCKSET, 0, fixed_vertex, true);
        }
        fixMaterialEnts = boost::shared_ptr<DirichletFixFieldAtEntitiesBc>(
            new DirichletFixFieldAtEntitiesBc(mField, "MESH_NODE_POSITIONS",
                                              fixed_vertex));
        fixMaterialEnts->fieldNames.push_back("LAMBDA_SURFACE");
        fixMaterialEnts->fieldNames.push_back("LAMBDA_EDGE");

        MoFEMFunctionReturn(0);
      }

      MoFEMErrorCode createConstrians() {
        MoFEMFunctionBegin;
        skinOrientation = boost::shared_ptr<
            SurfaceSlidingConstrains::DriverElementOrientation>(
            new SurfaceSlidingConstrains::DriverElementOrientation());
        surfaceConstrain = boost::shared_ptr<SurfaceSlidingConstrains>(
            skinOrientation,
            new SurfaceSlidingConstrains(mField, *skinOrientation));
        surfaceConstrain->setOperators(SURFACE_CONSTRAINS_TAG, "LAMBDA_SURFACE",
                                       "MESH_NODE_POSITIONS");

        if (mField.getInterface<MeshsetsManager>()->checkMeshset(
                edges_block_set, BLOCKSET)) {
          Range edges;
          CHKERR mField.getInterface<MeshsetsManager>()
              ->getEntitiesByDimension(edges_block_set, BLOCKSET, 1, edges,
                                       true);

          Range tets;
          CHKERR mField.get_moab().get_entities_by_type(0, MBTET, tets);
          Skinner skin(&mField.get_moab());
          Range skin_faces; // skin faces from 3d ents
          CHKERR skin.find_skin(0, tets, false, skin_faces);

          edgeConstrain = boost::shared_ptr<EdgeSlidingConstrains>(
              new EdgeSlidingConstrains(mField));
          CHKERR edgeConstrain->setOperators(EDGE_CONSTRAINS_TAG, edges,
                                             skin_faces, "LAMBDA_EDGE",
                                             "MESH_NODE_POSITIONS");

          // CHKERR EdgeSlidingConstrains::CalculateEdgeBase::saveEdges(
              // mField.get_moab(), "out_edges.vtk", edges);

        }
        MoFEMFunctionReturn(0);
      }

      MoFEMErrorCode addFEtoDM(DM dm) {
        MoFEMFunctionBegin;
        boost::shared_ptr<ForcesAndSourcesCore> null;

        CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, null, fixMaterialEnts,
                                      null);
        CHKERR DMMoFEMSNESSetFunction(dm, "SMOOTHING", feSmootherRhs, null,
                                      null);
        CHKERR DMMoFEMSNESSetFunction(dm, "SURFACE_SLIDING",
                                      surfaceConstrain->feRhsPtr, null, null);
        CHKERR DMMoFEMSNESSetFunction(dm, "EDGE_SLIDING",
                                      edgeConstrain->feRhsPtr, null, null);
        CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, null, null,
                                      fixMaterialEnts);

        CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, null, fixMaterialEnts,
                                      null);
        CHKERR DMMoFEMSNESSetJacobian(dm, "SMOOTHING", feSmootherLhs, null,
                                      null);
        CHKERR DMMoFEMSNESSetJacobian(dm, "SURFACE_SLIDING",
                                      surfaceConstrain->feLhsPtr, null, null);
        CHKERR DMMoFEMSNESSetJacobian(dm, "EDGE_SLIDING",
                                      edgeConstrain->feLhsPtr, null, null);
        CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, null, null,
                                      fixMaterialEnts);

        // MoFEM::SnesCtx *snes_ctx;
        // DMMoFEMGetSnesCtx(dm,&snes_ctx);
        // snes_ctx->vErify = true;

        MoFEMFunctionReturn(0);
      }

      MoFEMErrorCode calcuteMinQuality(DM dm) {
        MoFEMFunctionBegin;
        *minQualityPtr = 1;
        CHKERR DMoFEMLoopFiniteElements(dm, "SMOOTHING", minQualityFe.get());
        CHKERR VecMin(minQualityVec, PETSC_NULL, minQualityPtr);
        MoFEMFunctionReturn(0);
      }
    };

    ElementsAndOperators elements_and_operators(m_field);
    CHKERR elements_and_operators.createSmoothingFE();
    CHKERR elements_and_operators.createConstrians();

    DM dm;
    CHKERR simple_interface->getDM(&dm);
    CHKERR elements_and_operators.addFEtoDM(dm);

    struct Solve {

      MoFEMErrorCode operator()(DM dm) const {
        MoFEMFunctionBegin;

        // Create the right hand side vector and vector of unknowns
        Vec F, D;
        CHKERR DMCreateGlobalVector(dm, &F);
        // Create unknown vector by creating duplicate copy of F vector. only
        // structure is duplicated no values.
        CHKERR VecDuplicate(F, &D);

        CHKERR zeroLambdaFields(dm);
        CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_FORWARD);

        // Create solver and link it to DM
        SNES solver;
        CHKERR SNESCreate(PETSC_COMM_WORLD, &solver);
        CHKERR SNESSetFromOptions(solver);
        CHKERR SNESSetDM(solver, dm);
        // Set-up solver, is type of solver and pre-conditioners
        CHKERR SNESSetUp(solver);
        // At solution process, KSP solver using DM creates matrices, Calculate
        // values of the left hand side and the right hand side vector. then
        // solves system of equations. Results are stored in vector D.
        CHKERR SNESSolve(solver, F, D);

        // Scatter solution on the mesh. Stores unknown vector on field on the
        // mesh.
        CHKERR DMoFEMMeshToGlobalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
        // Clean data. Solver and vector are not needed any more.
        CHKERR SNESDestroy(&solver);
        CHKERR VecDestroy(&D);
        CHKERR VecDestroy(&F);

        MoFEMFunctionReturn(0);
      }

      MoFEMErrorCode setCoordsFromField(DM dm) const {
        MoFEMFunctionBegin;
        MoFEM::Interface *m_field_ptr;
        CHKERR DMoFEMGetInterfacePtr(dm, &m_field_ptr);
        for (_IT_GET_ENT_FIELD_BY_NAME_FOR_LOOP_(*m_field_ptr,
                                                 "MESH_NODE_POSITIONS", it)) {
          if (it->get()->getEntType() != MBVERTEX)
            continue;
          VectorDouble3 coords(3);
          for(int dd = 0;dd!=3;++dd)
            coords[dd] = it->get()->getEntFieldData()[dd];
          EntityHandle ent = it->get()->getEnt();
          CHKERR m_field_ptr->get_moab().set_coords(&ent, 1, &*coords.begin());
        }
        MoFEMFunctionReturn(0);
      }

      MoFEMErrorCode setFieldFromCoords(DM dm) const {
        MoFEMFunctionBegin;
        MoFEM::Interface *m_field_ptr;
        CHKERR DMoFEMGetInterfacePtr(dm, &m_field_ptr);
        for (_IT_GET_ENT_FIELD_BY_NAME_FOR_LOOP_(*m_field_ptr,
                                                 "MESH_NODE_POSITIONS", it)) {
          if (it->get()->getEntType() != MBVERTEX)
            continue;
          EntityHandle ent = it->get()->getEnt();
          VectorDouble3 coords(3);
          CHKERR m_field_ptr->get_moab().get_coords(&ent, 1, &*coords.begin());
          for(int dd = 0;dd!=3;++dd) 
            it->get()->getEntFieldData()[dd] = coords[dd];
        }
        MoFEMFunctionReturn(0);
      }

    private:
      MoFEMErrorCode zeroLambdaFields(DM dm) const {
        MoFEMFunctionBegin;
        MoFEM::Interface *m_field_ptr;
        CHKERR DMoFEMGetInterfacePtr(dm, &m_field_ptr);
        CHKERR m_field_ptr->getInterface<FieldBlas>()->setField(
            0, MBVERTEX, "LAMBDA_SURFACE");
        MoFEMFunctionReturn(0);
      }

    };

    Solve solve;
    CHKERR solve.setFieldFromCoords(dm);

    CHKERR elements_and_operators.calcuteMinQuality(dm);
    double min_quality = elements_and_operators.getMinQuality();
    PetscPrintf(PETSC_COMM_WORLD, "Min quality = %4.3f\n", min_quality);

    double gamma = min_quality > 0 ? gamma_factor * min_quality
                                   : min_quality / gamma_factor;
    elements_and_operators.volumeLengthDouble->gAmma = gamma;
    elements_and_operators.volumeLengthAdouble->gAmma = gamma;

    double min_quality_p, eps;
    do {

      min_quality_p = min_quality;

      CHKERR solve(dm);

      CHKERR solve.setCoordsFromField(dm);
      CHKERR elements_and_operators.calcuteMinQuality(dm);
      min_quality = elements_and_operators.getMinQuality();

      eps = (min_quality - min_quality_p) / min_quality;
      PetscPrintf(PETSC_COMM_WORLD, "Min quality = %4.3f eps = %4.3f\n",
                  min_quality, eps);

      double gamma = min_quality > 0 ? gamma_factor * min_quality
                                     : min_quality / gamma_factor;
      elements_and_operators.volumeLengthDouble->gAmma = gamma;
      elements_and_operators.volumeLengthAdouble->gAmma = gamma;

    } while (eps > tol);

    // if (m_field.getInterface<MeshsetsManager>()->checkMeshset(edges_block_set,
    //                                                           BLOCKSET)) {
    //   Range edges;
    //   CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
    //       edges_block_set, BLOCKSET, 1, edges, true);

    //   Range tets;
    //   CHKERR moab.get_entities_by_type(0,MBTET,tets);
    //   Skinner skin(&moab);
    //   Range skin_faces; // skin faces from 3d ents
    //   CHKERR skin.find_skin(0, tets, false, skin_faces);

    //   CHKERR EdgeSlidingConstrains::CalculateEdgeBase::setTags(moab, edges,
    //                                                          skin_faces);
    //   CHKERR EdgeSlidingConstrains::CalculateEdgeBase::saveEdges(
    //       moab, "out_edges.vtk", edges);
    // }

    if (output_vtk)
      CHKERR m_field.getInterface<BitRefManager>()->writeBitLevelByType(
          BitRefLevel().set(0), BitRefLevel().set(), MBTET, "out.vtk", "VTK",
          "");
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();
}
