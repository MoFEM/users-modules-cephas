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

using FaceEle = FaceElementForcesAndSourcesCore;
using OpFaceEle = FaceElementForcesAndSourcesCore::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;

struct CalculateDistantFromSurface {

  CalculateDistantFromSurface(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode operator()(const Range &surface) {
    MoFEMFunctionBegin;
    treeSurfPtr = boost::shared_ptr<OrientedBoxTreeTool>(
        new OrientedBoxTreeTool(&mField.get_moab(), "ROOTSETSURF", true));
    CHKERR treeSurfPtr->build(surface, rootSetSurf);
    DistanceFromVertex surf_dist(treeSurfPtr, rootSetSurf);
    CHKERR mField.loop_entities("PHI0", surf_dist);
    CHKERR mField.getInterface<FieldBlas>()->fieldCopy(1, "PHI0", "PHI");
    MoFEMFunctionReturn(0);
  }

private:
  MoFEM::Interface &mField;
  boost::shared_ptr<OrientedBoxTreeTool> treeSurfPtr;
  EntityHandle rootSetSurf;

  struct DistanceFromVertex : public EntityMethod {

    DistanceFromVertex(boost::shared_ptr<OrientedBoxTreeTool> &tree,
                       EntityHandle root)
        : EntityMethod(), treeSurfPtr(tree), rootSetSurf(root) {}

    MoFEMErrorCode preProcess() { return 0; }

    MoFEMErrorCode operator()() {
      MoFEMFunctionBegin;
      if (entPtr->getEntType() == MBVERTEX) {
        EntityHandle vert = entPtr->getEnt();
        VectorDouble3 coords(3);
        CHKERR entPtr->getBasicDataPtr()->moab.get_coords(&vert, 1,
                                                          &*coords.begin());
        // VectorDouble3 point_out(3);
        // EntityHandle facets_out;
        // CHKERR treeSurfPtr->closest_to_location(&coords[0], rootSetSurf,
        //                                         &point_out[0], facets_out);
        // VectorDouble3 n(3);
        // Util::normal(&entPtr->getBasicDataPtr()->moab, facets_out, n[0], n[1],
        //              n[2]);
        // n /= norm_2(n);
        // VectorDouble3 delta = point_out - coords;
        entPtr->getEntFieldData()[0] = inner_prod(coords, coords) - 9.;
      }
      MoFEMFunctionReturn(0);
    };

    MoFEMErrorCode postProcess() { return 0; }

  private:
    boost::shared_ptr<OrientedBoxTreeTool> treeSurfPtr;
    EntityHandle rootSetSurf;
  };
};

struct CommonData {

  MatrixDouble grad;
  VectorDouble val;

  SmartPetscObj<Mat> M;
  SmartPetscObj<KSP> ksp;
};

struct OpCalMass : OpFaceEle {
  OpCalMass(boost::shared_ptr<CommonData> &data)
      : OpFaceEle("PHI", "PHI", OpFaceEle::OPROWCOL), commonData(data) {}
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;
    FTensor::Index<'i', 3> i;
    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();
    if (nb_row_dofs && nb_col_dofs) {
      const int nb_integration_pts = getGaussPts().size2();
      mat.resize(nb_row_dofs, nb_col_dofs, false);
      mat.clear();
      auto t_row_base = row_data.getFTensor0N();
      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getVolume();
      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = t_w * vol;
        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          auto t_col_base = col_data.getFTensor0N(gg, 0);
          for (int cc = 0; cc != nb_col_dofs; ++cc) {
            mat(rr, cc) += a * t_row_base * t_col_base;
            ++t_col_base;
          }
          ++t_row_base;
        }
        ++t_w;
      }
      CHKERR MatSetValues(
          commonData->M, nb_row_dofs, &*row_data.getIndices().begin(),
          nb_col_dofs, &*col_data.getIndices().begin(), &mat(0, 0), ADD_VALUES);
    }
    MoFEMFunctionReturn(0);
  }

private:
  MatrixDouble mat;
  boost::shared_ptr<CommonData> commonData;
};

struct OpAssemble : OpFaceEle {
  OpAssemble(boost::shared_ptr<CommonData> &data)
      : OpFaceEle("PHI", OpFaceEle::OPROW), commonData(data) {
    doEdges = false;
    doTris = false;
    doTets = false;
  }
  VectorDouble vecF;
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    const int nb_dofs = data.getIndices().size();
    if (nb_dofs) {
      vecF.resize(nb_dofs, false);
      vecF.clear();

      const int nb_integration_pts = getGaussPts().size2();
      auto t_val = getFTensor0FromVec(commonData->val);
      auto t_grad = getFTensor1FromMat<3>(commonData->grad);
      auto t_base = data.getFTensor0N();
      auto t_diff_base = data.getFTensor1DiffN<3>();
      auto t_w = getFTensor0IntegrationWeight();
      FTensor::Index<'i', 3> i;

      const double vol = getVolume();
      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        const double f = -a * t_val * (1 - t_val);
        for (int rr = 0; rr != nb_dofs; ++rr) {
          const double b = f * t_base + a * t_diff_base(i) * t_grad(i);
          vecF[rr] += b;
          ++t_base;
          ++t_diff_base;
        }

        ++t_val;
        ++t_grad;
        ++t_w;
      }

      CHKERR VecSetValues(getFEMethod()->ts_F, nb_dofs,
                          &*data.getIndices().begin(), &*vecF.begin(),
                          ADD_VALUES);
    }
    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<CommonData> commonData;
};

int main(int argc, char *argv[]) {

  // initialize petsc
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    int surface_side_set = 200;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Level set", "none");
    CHKERR PetscOptionsInt("-surface_side_set", "surface side set", "",
                           surface_side_set, &surface_side_set, PETSC_NULL);
    ierr = PetscOptionsEnd();
    CHKERRG(ierr);

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    // Register DM Manager
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    // Simple interface
    Simple *simple_interface;
    CHKERR m_field.getInterface(simple_interface);
    CHKERR simple_interface->getOptions();
    CHKERR simple_interface->loadFile("");

    // add fields
    CHKERR simple_interface->addDomainField("PHI", H1, AINSWORTH_LEGENDRE_BASE,
                                            1);
    CHKERR simple_interface->addDataField("PHI0", H1, AINSWORTH_LEGENDRE_BASE,
                                          1);
    // set fields order
    CHKERR simple_interface->setFieldOrder("PHI", 1);
    CHKERR simple_interface->setFieldOrder("PHI0", 1);
    // setup problem
    CHKERR simple_interface->setUp(PETSC_FALSE);

    // get surface entities form side set
    Range surface;
    if (m_field.getInterface<MeshsetsManager>()->checkMeshset(surface_side_set,
                                                              SIDESET))
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          surface_side_set, SIDESET, 2, surface, true);
    if (surface.empty())
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "No surface");

    boost::shared_ptr<CommonData> data(new CommonData());
    auto val_ptr = boost::shared_ptr<VectorDouble>(data, &data->val);
    auto grad_ptr = boost::shared_ptr<MatrixDouble>(data, &data->grad);

    boost::shared_ptr<FaceEle> vol_ele(new FaceEle(m_field));
    vol_ele->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("PHI0", val_ptr));
    vol_ele->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<3>("PHI", grad_ptr));
    vol_ele->getOpPtrVector().push_back(new OpAssemble(data));
    boost::shared_ptr<FaceEle> vol_mass_ele(new FaceEle(m_field));
    vol_mass_ele->getOpPtrVector().push_back(new OpCalMass(data));

    auto vol_rule = [](int, int, int p) -> int { return 2 * p; };
    vol_ele->getRuleHook = vol_rule;
    vol_mass_ele->getRuleHook = vol_rule;

    boost::shared_ptr<PostProcVolumeOnRefinedMesh> post_proc_volume =
        boost::shared_ptr<PostProcVolumeOnRefinedMesh>(
            new PostProcVolumeOnRefinedMesh(m_field));
    boost::shared_ptr<ForcesAndSourcesCore> null;

    post_proc_volume->generateReferenceElementMesh();
    post_proc_volume->addFieldValuesPostProc("PHI");
    post_proc_volume->addFieldValuesGradientPostProc("PHI");
    post_proc_volume->addFieldValuesPostProc("PHI0");
    post_proc_volume->addFieldValuesGradientPostProc("PHI0");

    auto dm = simple_interface->getDM();

    CHKERR CalculateDistantFromSurface(m_field)(surface);

    auto ts = createTS(m_field.get_comm());
    CHKERR TSSetType(ts, TSRK);
    // CHKERR TSSetType(ts, TSEULER);
    CHKERR DMMoFEMTSSetRHSFunction(dm, simple_interface->getDomainFEName(),
                                   vol_ele, null, null);

    SmartPetscObj<Vec> X;
    CHKERR DMCreateGlobalVector_MoFEM(dm, X);
    CHKERR DMCreateMatrix_MoFEM(dm, data->M);

    CHKERR MatZeroEntries(data->M);
    CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
                                    vol_mass_ele);
    CHKERR MatAssemblyBegin(data->M, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(data->M, MAT_FINAL_ASSEMBLY);

    data->ksp = createKSP(m_field.get_comm());
    CHKERR KSPSetOperators(data->ksp, data->M, data->M);
    CHKERR KSPSetFromOptions(data->ksp);
    CHKERR KSPSetUp(data->ksp);

    auto solve_mass = [&]() {
      MoFEMFunctionBegin;

      if (vol_ele->vecAssembleSwitch) {
        CHKERR VecGhostUpdateBegin(vol_ele->ts_F, ADD_VALUES, SCATTER_REVERSE);
        CHKERR VecGhostUpdateEnd(vol_ele->ts_F, ADD_VALUES, SCATTER_REVERSE);
        CHKERR VecAssemblyBegin(vol_ele->ts_F);
        CHKERR VecAssemblyEnd(vol_ele->ts_F);
        *vol_ele->vecAssembleSwitch = false;
      }

      CHKERR KSPSolve(data->ksp, vol_ele->ts_F, vol_ele->ts_F);

      MoFEMFunctionReturn(0);
    };
    vol_ele->postProcessHook = solve_mass;

    CHKERR DMoFEMMeshToLocalVector(dm, X, INSERT_VALUES, SCATTER_FORWARD);

    double ftime = 1;
    CHKERR TSSetDM(ts, dm);
    CHKERR TSSetDuration(ts, PETSC_DEFAULT, ftime);
    CHKERR TSSetSolution(ts, X);
    CHKERR TSSetFromOptions(ts);
    CHKERR TSSolve(ts, X);

    CHKERR VecGhostUpdateBegin(X, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(X, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToGlobalVector(dm, X, INSERT_VALUES, SCATTER_REVERSE);
    CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
                                    post_proc_volume);
    CHKERR post_proc_volume->writeFile("out_level.h5m");
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();

  return 0;
}
