/** 
 * \file nonlinear_dynamics.cpp
 * \example nonlinear_dynamics.cpp
 * \ingroup nonlinear_elastic_ele
 *
 * \brief Non-linear elastic dynamics.

 \note For block solver is only for settings, some features are not implemented
 for this part.

 \note This is implementation where first order ODE is solved, and displacements
 and velocities are independently approximated. User can set lowe approximation
 order to velocities. However this method is inefficient comparing to method
 using Alpha method sor second order ODEs. Look into tutorials to see how to
 implement dynamic problem for TS type alpha2


 */

/*
 * This file is part of MoFEM.
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

#include <boost/program_options.hpp>
using namespace std;
namespace po = boost::program_options;
#include <ElasticMaterials.hpp>
#include <SurfacePressureComplexForLazy.hpp>

static char help[] = "...\n\n";

struct MonitorPostProc : public FEMethod {

  MoFEM::Interface &mField;
  PostProcVolumeOnRefinedMesh postProc;
  std::map<int, NonlinearElasticElement::BlockData> &setOfBlocks;
  NonlinearElasticElement::MyVolumeFE
      &feElasticEnergy; ///< calculate elastic energy
  ConvectiveMassElement::MyVolumeFE
      &feKineticEnergy; ///< calculate elastic energy

  bool iNit;

  int pRT;
  int *step;

  MonitorPostProc(
      MoFEM::Interface &m_field,
      std::map<int, NonlinearElasticElement::BlockData> &set_of_blocks,
      NonlinearElasticElement::MyVolumeFE &fe_elastic_energy,
      ConvectiveMassElement::MyVolumeFE &fe_kinetic_energy)
      : FEMethod(), mField(m_field), postProc(m_field),
        setOfBlocks(set_of_blocks), feElasticEnergy(fe_elastic_energy),
        feKineticEnergy(fe_kinetic_energy), iNit(false) {

    double def_t_val = 0;
    const EntityHandle root_meshset = mField.get_moab().get_root_set();

    Tag th_step;
    rval = m_field.get_moab().tag_get_handle(
        "_TsStep_", 1, MB_TYPE_INTEGER, th_step,
        MB_TAG_CREAT | MB_TAG_EXCL | MB_TAG_MESH, &def_t_val);
    if (rval == MB_ALREADY_ALLOCATED) {
      rval = m_field.get_moab().tag_get_by_ptr(th_step, &root_meshset, 1,
                                               (const void **)&step);
      MOAB_THROW(rval);
    } else {
      rval = m_field.get_moab().tag_set_data(th_step, &root_meshset, 1,
                                             &def_t_val);
      MOAB_THROW(rval);
      rval = m_field.get_moab().tag_get_by_ptr(th_step, &root_meshset, 1,
                                               (const void **)&step);
      MOAB_THROW(rval);
    }

    PetscBool flg = PETSC_TRUE;
    ierr = PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_output_prt", &pRT,
                              &flg);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    if (flg != PETSC_TRUE) {
      pRT = 10;
    }
  }

  MoFEMErrorCode preProcess() {
    MoFEMFunctionBegin;

    if (!iNit) {
      CHKERR postProc.generateReferenceElementMesh();
      if(mField.check_field("MESH_NODE_POSITIONS"))
        CHKERR addHOOpsVol("MESH_NODE_POSITIONS", postProc, true, false, false,
                        false);
      CHKERR postProc.addFieldValuesPostProc("DISPLACEMENT");
      CHKERR postProc.addFieldValuesPostProc("VELOCITY");
      CHKERR postProc.addFieldValuesPostProc("MESH_NODE_POSITIONS");
      CHKERR postProc.addFieldValuesGradientPostProc("DISPLACEMENT");

      std::map<int, NonlinearElasticElement::BlockData>::iterator sit =
          setOfBlocks.begin();
      for (; sit != setOfBlocks.end(); sit++) {
        postProc.getOpPtrVector().push_back(new PostProcStress(
            postProc.postProcMesh, postProc.mapGaussPts, "DISPLACEMENT",
            sit->second, postProc.commonData, true));
      }

      iNit = true;
    }

    if ((*step) % pRT == 0) {
      CHKERR mField.loop_finite_elements("DYNAMICS", "MASS_ELEMENT", postProc);
      std::ostringstream sss;
      sss << "out_values_" << (*step) << ".h5m";
      CHKERR postProc.writeFile(sss.str().c_str());
    }

    feElasticEnergy.ts_ctx = TSMethod::CTX_TSNONE;
    feElasticEnergy.snes_ctx = SnesMethod::CTX_SNESNONE;
    CHKERR mField.loop_finite_elements("DYNAMICS", "ELASTIC", feElasticEnergy);
    feKineticEnergy.ts_ctx = TSMethod::CTX_TSNONE;
    CHKERR mField.loop_finite_elements("DYNAMICS", "MASS_ELEMENT",
                                       feKineticEnergy);
    double E = feElasticEnergy.eNergy;
    double T = feKineticEnergy.eNergy;
    MOFEM_LOG_C(
        "DYNAMIC", Sev::inform,
        "%d Time %3.2e Elastic energy %3.2e Kinetic Energy %3.2e Total %3.2e\n",
        ts_step, ts_t, E, T, E + T);

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode operator()() {
    MoFEMFunctionBeginHot;
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBeginHot;
    MoFEMFunctionReturnHot(0);
  }
};

struct MonitorRestart : public FEMethod {

  double *time;
  int *step;
  MoFEM::Interface &mField;
  int pRT;

  MonitorRestart(MoFEM::Interface &m_field, TS ts) : mField(m_field) {
    double def_t_val = 0;

    const EntityHandle root_meshset = mField.get_moab().get_root_set();

    Tag th_time;
    rval = m_field.get_moab().tag_get_handle(
        "_TsTime_", 1, MB_TYPE_DOUBLE, th_time,
        MB_TAG_CREAT | MB_TAG_EXCL | MB_TAG_MESH, &def_t_val);
    if (rval == MB_ALREADY_ALLOCATED) {
      rval = m_field.get_moab().tag_get_by_ptr(th_time, &root_meshset, 1,
                                               (const void **)&time);
      MOAB_THROW(rval);
      ierr = TSSetTime(ts, *time);
      CHKERRABORT(PETSC_COMM_WORLD, ierr);
    } else {
      rval = m_field.get_moab().tag_set_data(th_time, &root_meshset, 1,
                                             &def_t_val);
      MOAB_THROW(rval);
      rval = m_field.get_moab().tag_get_by_ptr(th_time, &root_meshset, 1,
                                               (const void **)&time);
      MOAB_THROW(rval);
    }
    Tag th_step;
    rval = m_field.get_moab().tag_get_handle(
        "_TsStep_", 1, MB_TYPE_INTEGER, th_step,
        MB_TAG_CREAT | MB_TAG_EXCL | MB_TAG_MESH, &def_t_val);
    if (rval == MB_ALREADY_ALLOCATED) {
      rval = m_field.get_moab().tag_get_by_ptr(th_step, &root_meshset, 1,
                                               (const void **)&step);
      MOAB_THROW(rval);
    } else {
      rval = m_field.get_moab().tag_set_data(th_step, &root_meshset, 1,
                                             &def_t_val);
      MOAB_THROW(rval);
      rval = m_field.get_moab().tag_get_by_ptr(th_step, &root_meshset, 1,
                                               (const void **)&step);
      MOAB_THROW(rval);
    }

    PetscBool flg = PETSC_TRUE;
    ierr = PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_output_prt", &pRT,
                              &flg);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    if (flg != PETSC_TRUE) {
      pRT = 10;
    }
  }

  MoFEMErrorCode preProcess() {
    MoFEMFunctionBeginHot;

    (*time) = ts_t;
    // if(pRT>0) {
    //   if((*step)%pRT==0) {
    //     std::ostringstream ss;
    //     ss << "restart_" << (*step) << ".h5m";
    //     CHKERR
    //     mField.get_moab().write_file(ss.str().c_str()/*,"MOAB","PARALLEL=WRITE_PART"*/);
    //
    //   }
    // }
    (*step)++;
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode operator()() {
    MoFEMFunctionBeginHot;
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBeginHot;
    MoFEMFunctionReturnHot(0);
  }
};

// See file users_modules/elasticity/TimeForceScale.hpp
#include <TimeForceScale.hpp>

int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_type mumps \n"
                                 "-ksp_atol 1e-10 \n"
                                 "-ksp_rtol 1e-10 \n"
                                 "-snes_monitor \n"
                                 "-snes_type newtonls \n"
                                 "-snes_linesearch_type basic \n"
                                 "-snes_max_it 100 \n"
                                 "-snes_atol 1e-7 \n"
                                 "-snes_rtol 1e-7 \n"
                                 "-ts_monitor \n"
                                 "-ts_type alpha \n";

  string param_file = "param_file.petsc";
  if (!static_cast<bool>(ifstream(param_file))) {
    std::ofstream file(param_file.c_str(), std::ios::ate);
    if (file.is_open()) {
      file << default_options;
      file.close();
    }
  }

  MoFEM::Core::Initialize(&argc, &argv, param_file.c_str(), help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "DYNAMIC"));
  LogManager::setLog("DYNAMIC");
  MOFEM_LOG_TAG("DYNAMIC", "dynamic");

  try {

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;

    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    auto moab_comm_wrap =
        boost::make_shared<WrapMPIComm>(PETSC_COMM_WORLD, false);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, moab_comm_wrap->get_comm());

    FieldApproximationBase base = NOBASE;
    char mesh_file_name[255];
    PetscBool is_partitioned = PETSC_FALSE;
    PetscBool linear = PETSC_TRUE;
    PetscInt disp_order = 1;
    PetscInt vel_order = 1;
    PetscBool is_solve_at_time_zero = PETSC_FALSE;

    auto read_command_line_parameters = [&]() {
      MoFEMFunctionBegin;
      PetscBool flg = PETSC_TRUE;
      CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-my_file",
                                   mesh_file_name, 255, &flg);
      if (flg != PETSC_TRUE)
        SETERRQ(PETSC_COMM_SELF, 1, "Error -my_file (mesh file needed)");

      // use this if your mesh is partitioned and you run code on parts,
      // you can solve very big problems
      CHKERR PetscOptionsGetBool(PETSC_NULL, PETSC_NULL, "-my_is_partitioned",
                                 &is_partitioned, &flg);

      CHKERR PetscOptionsGetBool(PETSC_NULL, PETSC_NULL, "-is_linear", &linear,
                                 PETSC_NULL);

      enum bases { LEGENDRE, LOBATTO, BERNSTEIN_BEZIER, LASBASETOP };
      const char *list_bases[] = {"legendre", "lobatto", "bernstein_bezier"};
      PetscInt choice_base_value = BERNSTEIN_BEZIER;
      CHKERR PetscOptionsGetEList(PETSC_NULL, NULL, "-base", list_bases,
                                  LASBASETOP, &choice_base_value, PETSC_NULL);
      if (choice_base_value == LEGENDRE)
        base = AINSWORTH_LEGENDRE_BASE;
      else if (choice_base_value == LOBATTO)
        base = AINSWORTH_LOBATTO_BASE;
      else if (choice_base_value == BERNSTEIN_BEZIER)
        base = AINSWORTH_BERNSTEIN_BEZIER_BASE;

      CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_disp_order",
                                &disp_order, &flg);
      if (flg != PETSC_TRUE)
        disp_order = 1;

      CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_vel_order",
                                &vel_order, &flg);
      if (flg != PETSC_TRUE)
        vel_order = disp_order;

      CHKERR PetscOptionsGetBool(PETSC_NULL, PETSC_NULL,
                                 "-my_solve_at_time_zero",
                                 &is_solve_at_time_zero, &flg);

      MoFEMFunctionReturn(0);
    };

    auto read_mesh = [&]() {
      MoFEMFunctionBegin;
      if (is_partitioned == PETSC_TRUE) {
        // Read mesh to MOAB
        const char *option;
        option = "PARALLEL=BCAST_DELETE;"
                 "PARALLEL_RESOLVE_SHARED_ENTS;"
                 "PARTITION=PARALLEL_PARTITION;";
        CHKERR moab.load_file(mesh_file_name, 0, option);
      } else {
        const char *option;
        option = "";
        CHKERR moab.load_file(mesh_file_name, 0, option);
      }
      MoFEMFunctionReturn(0);
    };

    CHKERR read_command_line_parameters();
    CHKERR read_mesh();

    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    // ref meshset ref level 0
    BitRefLevel bit_level0;
    bit_level0.set(0);
    EntityHandle meshset_level0;
    CHKERR moab.create_meshset(MESHSET_SET, meshset_level0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_level0);
    CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByRefLevel(
        bit_level0, BitRefLevel().set(), meshset_level0);

    // Fields
    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3, MB_TAG_SPARSE, MF_ZERO);
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "MESH_NODE_POSITIONS");
    CHKERR m_field.set_field_order(0, MBTET, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(0, MBTRI, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);

    bool check_if_spatial_field_exist = m_field.check_field("DISPLACEMENT");
    CHKERR m_field.add_field("DISPLACEMENT", H1, base, 3, MB_TAG_SPARSE,
                             MF_ZERO);
    // add entities (by tets) to the field
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "DISPLACEMENT");

    // set app. order
    CHKERR m_field.set_field_order(0, MBTET, "DISPLACEMENT", disp_order);
    CHKERR m_field.set_field_order(0, MBTRI, "DISPLACEMENT", disp_order);
    CHKERR m_field.set_field_order(0, MBEDGE, "DISPLACEMENT", disp_order);
    if (base == AINSWORTH_BERNSTEIN_BEZIER_BASE)
      CHKERR m_field.set_field_order(0, MBVERTEX, "DISPLACEMENT", disp_order);
    else
      CHKERR m_field.set_field_order(0, MBVERTEX, "DISPLACEMENT", 1);

    // Add nodal force element
    CHKERR MetaNeumannForces::addNeumannBCElements(m_field, "DISPLACEMENT");
    CHKERR MetaEdgeForces::addElement(m_field, "DISPLACEMENT");
    CHKERR MetaNodalForces::addElement(m_field, "DISPLACEMENT");
    // Add fluid pressure finite elements
    FluidPressure fluid_pressure_fe(m_field);
    fluid_pressure_fe.addNeumannFluidPressureBCElements("DISPLACEMENT");
    CHKERR addHOOpsFace3D("MESH_NODE_POSITIONS", fluid_pressure_fe.getLoopFe(),
                          false, false);
    fluid_pressure_fe.setNeumannFluidPressureFiniteElementOperators(
        "DISPLACEMENT", PETSC_NULL, false, true);

    // Velocity
    CHKERR m_field.add_field("VELOCITY", H1, base, 3, MB_TAG_SPARSE, MF_ZERO);
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "VELOCITY");

    CHKERR m_field.set_field_order(0, MBTET, "VELOCITY", vel_order);
    CHKERR m_field.set_field_order(0, MBTRI, "VELOCITY", vel_order);
    CHKERR m_field.set_field_order(0, MBEDGE, "VELOCITY", vel_order);
    if (base == AINSWORTH_BERNSTEIN_BEZIER_BASE)
      CHKERR m_field.set_field_order(0, MBVERTEX, "VELOCITY", vel_order);
    else
      CHKERR m_field.set_field_order(0, MBVERTEX, "VELOCITY", 1);

    CHKERR m_field.add_field("DOT_DISPLACEMENT", H1, base, 3, MB_TAG_SPARSE,
                             MF_ZERO);
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "DOT_DISPLACEMENT");
    CHKERR m_field.set_field_order(0, MBTET, "DOT_DISPLACEMENT", disp_order);
    CHKERR m_field.set_field_order(0, MBTRI, "DOT_DISPLACEMENT", disp_order);
    CHKERR m_field.set_field_order(0, MBEDGE, "DOT_DISPLACEMENT", disp_order);
    if (base == AINSWORTH_BERNSTEIN_BEZIER_BASE)
      CHKERR m_field.set_field_order(0, MBVERTEX, "DOT_DISPLACEMENT",
                                     disp_order);
    else
      CHKERR m_field.set_field_order(0, MBVERTEX, "DOT_DISPLACEMENT", 1);

    CHKERR m_field.add_field("DOT_VELOCITY", H1, base, 3, MB_TAG_SPARSE,
                             MF_ZERO);
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "DOT_VELOCITY");
    CHKERR m_field.set_field_order(0, MBTET, "DOT_VELOCITY", vel_order);
    CHKERR m_field.set_field_order(0, MBTRI, "DOT_VELOCITY", vel_order);
    CHKERR m_field.set_field_order(0, MBEDGE, "DOT_VELOCITY", vel_order);
    if (base == AINSWORTH_BERNSTEIN_BEZIER_BASE)
      CHKERR m_field.set_field_order(0, MBVERTEX, "DOT_VELOCITY", disp_order);
    else
      CHKERR m_field.set_field_order(0, MBVERTEX, "DOT_VELOCITY", 1);

    // Set material model and mass element
    NonlinearElasticElement elastic(m_field, 2);
    ElasticMaterials elastic_materials(m_field);
    CHKERR elastic_materials.setBlocks(elastic.setOfBlocks);
    // NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<adouble>
    // st_venant_kirchhoff_material_adouble;
    // NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<double>
    // st_venant_kirchhoff_material_double;  CHKERR
    // elastic.setBlocks(&st_venant_kirchhoff_material_double,&st_venant_kirchhoff_material_adouble);
    CHKERR elastic.addElement("ELASTIC", "DISPLACEMENT");
    CHKERR addHOOpsVol("MESH_NODE_POSITIONS", elastic.getLoopFeRhs(), true, false,
                    false, false);
    CHKERR addHOOpsVol("MESH_NODE_POSITIONS", elastic.getLoopFeLhs(), true, false,
                    false, false);
    CHKERR addHOOpsVol("MESH_NODE_POSITIONS", elastic.getLoopFeEnergy(), true,
                    false, false, false);
    CHKERR elastic.setOperators("DISPLACEMENT", "MESH_NODE_POSITIONS", false,
                                true);

    // set mass element
    ConvectiveMassElement inertia(m_field, 1);
    // CHKERR inertia.setBlocks();
    CHKERR elastic_materials.setBlocks(inertia.setOfBlocks);
    CHKERR inertia.addConvectiveMassElement("MASS_ELEMENT", "VELOCITY",
                                            "DISPLACEMENT");
    CHKERR inertia.addHOOpsVol();
    CHKERR inertia.addVelocityElement("VELOCITY_ELEMENT", "VELOCITY",
                                      "DISPLACEMENT");

    // Add possibility to load accelerogram
    {
      string name = "-my_accelerogram";
      char time_file_name[255];
      PetscBool flg;
      CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL, name.c_str(),
                                   time_file_name, 255, &flg);
      if (flg == PETSC_TRUE) {
        inertia.methodsOp.push_back(new TimeAccelerogram(name));
      }
    }

    // damper element
    KelvinVoigtDamper damper(m_field);
    CHKERR elastic_materials.setBlocks(damper.blockMaterialDataMap);
    {
      KelvinVoigtDamper::CommonData &common_data = damper.commonData;
      common_data.spatialPositionName = "DISPLACEMENT";
      common_data.spatialPositionNameDot = "DOT_DISPLACEMENT";
      CHKERR m_field.add_finite_element("DAMPER", MF_ZERO);
      CHKERR m_field.modify_finite_element_add_field_row("DAMPER",
                                                         "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_col("DAMPER",
                                                         "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_data("DAMPER",
                                                          "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_data("DAMPER",
                                                          "DOT_DISPLACEMENT");
      if (m_field.check_field("MESH_NODE_POSITIONS")) {
        CHKERR m_field.modify_finite_element_add_field_data(
            "DAMPER", "MESH_NODE_POSITIONS");
      }
      std::map<int, KelvinVoigtDamper::BlockMaterialData>::iterator bit =
          damper.blockMaterialDataMap.begin();
      for (; bit != damper.blockMaterialDataMap.end(); bit++) {
        bit->second.lInear = linear;
        int id = bit->first;
        KelvinVoigtDamper::BlockMaterialData &material_data = bit->second;
        damper.constitutiveEquationMap.insert(
            id, new KelvinVoigtDamper::ConstitutiveEquation<adouble>(
                    material_data));
        CHKERR m_field.add_ents_to_finite_element_by_type(bit->second.tEts,
                                                          MBTET, "DAMPER");
      }
      CHKERR damper.setOperators(3);
    }

    MonitorPostProc post_proc(m_field, elastic.setOfBlocks,
                              elastic.getLoopFeEnergy(),
                              inertia.getLoopFeEnergy());

    // elastic and mass element calculated in Kuu shell matrix problem. To
    // calculate Mass element, velocity field is needed.
    CHKERR m_field.modify_finite_element_add_field_data("ELASTIC", "VELOCITY");
    CHKERR m_field.modify_finite_element_add_field_data("ELASTIC",
                                                        "DOT_DISPLACEMENT");
    CHKERR m_field.modify_finite_element_add_field_data("ELASTIC",
                                                        "DOT_VELOCITY");

    // build field
    CHKERR m_field.build_fields();
    // CHKERR m_field.list_dofs_by_field_name("DISPLACEMENT");

    // 10 node tets
    if (!check_if_spatial_field_exist) {
      Projection10NodeCoordsOnField ent_method_material(m_field,
                                                        "MESH_NODE_POSITIONS");
      CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method_material);
    }

    // build finite elements
    CHKERR m_field.build_finite_elements();
    // build adjacencies
    CHKERR m_field.build_adjacencies(bit_level0);

    // define problems
    {
      CHKERR m_field.add_problem("Kuu", MF_ZERO);
      CHKERR m_field.modify_problem_add_finite_element("Kuu", "ELASTIC");
      CHKERR m_field.modify_problem_add_finite_element("Kuu", "PRESSURE_FE");
      CHKERR m_field.modify_problem_add_finite_element("Kuu", "FORCE_FE");
      CHKERR m_field.modify_problem_add_finite_element("Kuu",
                                                       "FLUID_PRESSURE_FE");
      CHKERR m_field.modify_problem_ref_level_add_bit("Kuu", bit_level0);

      ProblemsManager *prb_mng_ptr;
      CHKERR m_field.getInterface(prb_mng_ptr);
      if (is_partitioned) {
        CHKERR prb_mng_ptr->buildProblemOnDistributedMesh("Kuu", true);
        CHKERR prb_mng_ptr->partitionFiniteElements("Kuu", true, 0,
                                                    pcomm->size());
      } else {
        CHKERR prb_mng_ptr->buildProblem("Kuu", true);
        CHKERR prb_mng_ptr->partitionProblem("Kuu");
        CHKERR prb_mng_ptr->partitionFiniteElements("Kuu");
      }
      CHKERR prb_mng_ptr->partitionGhostDofs("Kuu");
    }

    CHKERR m_field.add_problem("DYNAMICS", MF_ZERO);
    // set finite elements for problems
    CHKERR m_field.modify_problem_add_finite_element("DYNAMICS", "ELASTIC");
    CHKERR m_field.modify_problem_add_finite_element("DYNAMICS", "DAMPER");
    CHKERR m_field.modify_problem_add_finite_element("DYNAMICS", "PRESSURE_FE");
    CHKERR m_field.modify_problem_add_finite_element("DYNAMICS", "FORCE_FE");
    CHKERR m_field.modify_problem_add_finite_element("DYNAMICS",
                                                     "FLUID_PRESSURE_FE");
    CHKERR m_field.modify_problem_add_finite_element("DYNAMICS",
                                                     "MASS_ELEMENT");
    CHKERR m_field.modify_problem_add_finite_element("DYNAMICS",
                                                     "VELOCITY_ELEMENT");
    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("DYNAMICS", bit_level0);

    ProblemsManager *prb_mng_ptr;
    CHKERR m_field.getInterface(prb_mng_ptr);
    if (is_partitioned) {
      CHKERR prb_mng_ptr->buildProblemOnDistributedMesh("DYNAMICS", true);
      CHKERR prb_mng_ptr->partitionFiniteElements("DYNAMICS", true, 0,
                                                  pcomm->size());
    } else {
      CHKERR prb_mng_ptr->buildProblem("DYNAMICS", true);
      CHKERR prb_mng_ptr->partitionProblem("DYNAMICS");
      CHKERR prb_mng_ptr->partitionFiniteElements("DYNAMICS");
    }
    CHKERR prb_mng_ptr->partitionGhostDofs("DYNAMICS");

    Vec F;
    CHKERR m_field.getInterface<VecManager>()->vecCreateGhost("DYNAMICS", COL,
                                                              &F);
    Vec D;
    CHKERR VecDuplicate(F, &D);

    // create tS
    TS ts;
    CHKERR TSCreate(PETSC_COMM_WORLD, &ts);
    CHKERR TSSetType(ts, TSBEULER);

    // shell matrix
    ConvectiveMassElement::MatShellCtx *shellAij_ctx =
        new ConvectiveMassElement::MatShellCtx();
    CHKERR m_field.getInterface<MatrixManager>()
        ->createMPIAIJWithArrays<PetscGlobalIdx_mi_tag>("Kuu",
                                                        &shellAij_ctx->K);
    CHKERR MatDuplicate(shellAij_ctx->K, MAT_DO_NOT_COPY_VALUES,
                        &shellAij_ctx->M);
    CHKERR shellAij_ctx->iNit();
    CHKERR m_field.getInterface<VecManager>()->vecScatterCreate(
        D, "DYNAMICS", COL, shellAij_ctx->u, "Kuu", COL,
        &shellAij_ctx->scatterU);
    CHKERR m_field.getInterface<VecManager>()->vecScatterCreate(
        D, "DYNAMICS", "VELOCITY", COL, shellAij_ctx->v, "Kuu", "DISPLACEMENT",
        COL, &shellAij_ctx->scatterV);
    Mat shell_Aij;
    const Problem *problem_ptr;
    CHKERR m_field.get_problem("DYNAMICS", &problem_ptr);
    CHKERR MatCreateShell(
        PETSC_COMM_WORLD, problem_ptr->getNbLocalDofsRow(),
        problem_ptr->getNbLocalDofsCol(), problem_ptr->getNbDofsRow(),
        problem_ptr->getNbDofsRow(), (void *)shellAij_ctx, &shell_Aij);
    CHKERR MatShellSetOperation(shell_Aij, MATOP_MULT,
                                (void (*)(void))ConvectiveMassElement::MultOpA);
    CHKERR MatShellSetOperation(
        shell_Aij, MATOP_ZERO_ENTRIES,
        (void (*)(void))ConvectiveMassElement::ZeroEntriesOp);
    // blocked problem
    ConvectiveMassElement::ShellMatrixElement shell_matrix_element(m_field);
    DirichletDisplacementBc shell_dirichlet_bc(
        m_field, "DISPLACEMENT", shellAij_ctx->barK, PETSC_NULL, PETSC_NULL);
    DirichletDisplacementBc my_dirichlet_bc(m_field, "DISPLACEMENT", PETSC_NULL,
                                            D, F);
    shell_matrix_element.problemName = "Kuu";
    shell_matrix_element.shellMatCtx = shellAij_ctx;
    shell_matrix_element.DirichletBcPtr = &shell_dirichlet_bc;
    shell_matrix_element.loopK.push_back(
        ConvectiveMassElement::ShellMatrixElement::PairNameFEMethodPtr(
            "ELASTIC", &elastic.getLoopFeLhs()));
    // damper
    shell_matrix_element.loopK.push_back(
        ConvectiveMassElement::ShellMatrixElement::PairNameFEMethodPtr(
            "ELASTIC", &damper.feLhs));

    CHKERR inertia.setShellMatrixMassOperators("VELOCITY", "DISPLACEMENT",
                                               "MESH_NODE_POSITIONS", linear);
    // element name "ELASTIC" is used, therefore M matrix is assembled as K
    // matrix. This is added to M is shell matrix. M matrix is a derivative of
    // inertia forces over spatial velocities
    shell_matrix_element.loopM.push_back(
        ConvectiveMassElement::ShellMatrixElement::PairNameFEMethodPtr(
            "ELASTIC", &inertia.getLoopFeMassLhs()));
    // this calculate derivatives of inertia forces over spatial positions and
    // add this to shell K matrix
    shell_matrix_element.loopAuxM.push_back(
        ConvectiveMassElement::ShellMatrixElement::PairNameFEMethodPtr(
            "ELASTIC", &inertia.getLoopFeMassAuxLhs()));

    // Element to calculate shell matrix residual
    ConvectiveMassElement::ShellResidualElement shell_matrix_residual(m_field);
    shell_matrix_residual.shellMatCtx = shellAij_ctx;

    // surface pressure
    boost::ptr_map<std::string, NeumannForcesSurface> surface_forces;
    {
      string fe_name_str = "FORCE_FE";
      surface_forces.insert(fe_name_str, new NeumannForcesSurface(m_field));
      CHKERR addHOOpsFace3D("MESH_NODE_POSITIONS",
                            surface_forces.at(fe_name_str).getLoopFe(), false,
                            false);
      for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,
                                                      NODESET | FORCESET, it)) {
        CHKERR surface_forces.at(fe_name_str)
            .addForce("DISPLACEMENT", PETSC_NULL, it->getMeshsetId(), true);
        surface_forces.at(fe_name_str)
            .methodsOp.push_back(new TimeForceScale());
      }
    }

    boost::ptr_map<std::string, NeumannForcesSurface> surface_pressure;
    {
      string fe_name_str = "PRESSURE_FE";
      surface_pressure.insert(fe_name_str, new NeumannForcesSurface(m_field));
      CHKERR addHOOpsFace3D("MESH_NODE_POSITIONS",
                            surface_pressure.at(fe_name_str).getLoopFe(), false,
                            false);
      for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
               m_field, SIDESET | PRESSURESET, it)) {
        CHKERR surface_pressure.at(fe_name_str)
            .addPressure("DISPLACEMENT", PETSC_NULL, it->getMeshsetId(), true);
        surface_pressure.at(fe_name_str)
            .methodsOp.push_back(new TimeForceScale());
      }
    }

    // edge forces
    boost::ptr_map<std::string, EdgeForce> edge_forces;
    {
      string fe_name_str = "FORCE_FE";
      edge_forces.insert(fe_name_str, new EdgeForce(m_field));
      for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,
                                                      NODESET | FORCESET, it)) {
        CHKERR edge_forces.at(fe_name_str)
            .addForce("DISPLACEMENT", PETSC_NULL, it->getMeshsetId(), true);
        edge_forces.at(fe_name_str).methodsOp.push_back(new TimeForceScale());
      }
    }

    // nodal forces
    boost::ptr_map<std::string, NodalForce> nodal_forces;
    {
      string fe_name_str = "FORCE_FE";
      nodal_forces.insert(fe_name_str, new NodalForce(m_field));
      for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,
                                                      NODESET | FORCESET, it)) {
        CHKERR nodal_forces.at(fe_name_str)
            .addForce("DISPLACEMENT", F, it->getMeshsetId(), true);
        nodal_forces.at(fe_name_str).methodsOp.push_back(new TimeForceScale());
      }
    }

    MonitorRestart monitor_restart(m_field, ts);
    ConvectiveMassElement::UpdateAndControl update_and_control(
        m_field, ts, "VELOCITY", "DISPLACEMENT");

    // TS
    TsCtx ts_ctx(m_field, "DYNAMICS");

    // right hand side
    // preprocess
    ts_ctx.getPreProcessIFunction().push_back(&update_and_control);
    ts_ctx.getPreProcessIFunction().push_back(&my_dirichlet_bc);
    // fe looops
    TsCtx::FEMethodsSequence &loops_to_do_Rhs =
        ts_ctx.getLoopsIFunction();

    auto add_static_rhs = [&](auto &loops_to_do_Rhs) {
      MoFEMFunctionBegin;
      loops_to_do_Rhs.push_back(
          PairNameFEMethodPtr("ELASTIC", &elastic.getLoopFeRhs()));
      for (auto fit = surface_forces.begin(); fit != surface_forces.end();
           fit++) {
        loops_to_do_Rhs.push_back(
            PairNameFEMethodPtr(fit->first, &fit->second->getLoopFe()));
      }
      for (auto fit = surface_pressure.begin(); fit != surface_pressure.end();
           fit++) {
        loops_to_do_Rhs.push_back(
            PairNameFEMethodPtr(fit->first, &fit->second->getLoopFe()));
      }
      for (auto fit = edge_forces.begin(); fit != edge_forces.end(); fit++) {
        loops_to_do_Rhs.push_back(
            PairNameFEMethodPtr(fit->first, &fit->second->getLoopFe()));
      }
      for (auto fit = nodal_forces.begin(); fit != nodal_forces.end(); fit++) {
        loops_to_do_Rhs.push_back(
            PairNameFEMethodPtr(fit->first, &fit->second->getLoopFe()));
      }
      loops_to_do_Rhs.push_back(PairNameFEMethodPtr(
          "FLUID_PRESSURE_FE", &fluid_pressure_fe.getLoopFe()));
      MoFEMFunctionReturn(0);
    };

    CHKERR add_static_rhs(loops_to_do_Rhs);

    loops_to_do_Rhs.push_back(PairNameFEMethodPtr("DAMPER", &damper.feRhs));
    loops_to_do_Rhs.push_back(
        PairNameFEMethodPtr("MASS_ELEMENT", &inertia.getLoopFeMassRhs()));

    ts_ctx.getPreProcessIFunction().push_back(&shell_matrix_residual);

    // postproc
    ts_ctx.getPostProcessIFunction().push_back(&my_dirichlet_bc);
    ts_ctx.getPostProcessIFunction().push_back(&shell_matrix_residual);

    // left hand side
    // preprocess
    ts_ctx.getPreProcessIJacobian().push_back(&update_and_control);
    ts_ctx.getPreProcessIJacobian().push_back(&shell_matrix_element);
    ts_ctx.getPostProcessIJacobian().push_back(&update_and_control);
    // monitor
    TsCtx::FEMethodsSequence &loopsMonitor =
        ts_ctx.getLoopsMonitor();
    loopsMonitor.push_back(
        TsCtx::PairNameFEMethodPtr("MASS_ELEMENT", &post_proc));
    loopsMonitor.push_back(
        TsCtx::PairNameFEMethodPtr("MASS_ELEMENT", &monitor_restart));

    CHKERR TSSetIFunction(ts, F, TsSetIFunction, &ts_ctx);
    CHKERR TSSetIJacobian(ts, shell_Aij, shell_Aij, TsSetIJacobian, &ts_ctx);

    CHKERR TSMonitorSet(ts, TsMonitorSet, &ts_ctx, PETSC_NULL);

    double ftime = 1;
    CHKERR TSSetDuration(ts, PETSC_DEFAULT, ftime);
    CHKERR TSSetSolution(ts, D);
    CHKERR TSSetFromOptions(ts);
    // shell matrix pre-conditioner
    SNES snes;
    CHKERR TSGetSNES(ts, &snes);
    // CHKERR SNESSetFromOptions(snes);
    KSP ksp;
    CHKERR SNESGetKSP(snes, &ksp);
    CHKERR KSPSetFromOptions(ksp);
    PC pc;
    CHKERR KSPGetPC(ksp, &pc);
    CHKERR PCSetType(pc, PCSHELL);
    ConvectiveMassElement::PCShellCtx pc_shell_ctx(shell_Aij);
    CHKERR PCShellSetContext(pc, (void *)&pc_shell_ctx);
    CHKERR PCShellSetApply(pc, ConvectiveMassElement::PCShellApplyOp);
    CHKERR PCShellSetSetUp(pc, ConvectiveMassElement::PCShellSetUpOp);
    CHKERR PCShellSetDestroy(pc, ConvectiveMassElement::PCShellDestroy);

    CHKERR VecZeroEntries(D);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR m_field.getInterface<VecManager>()->setGlobalGhostVector(
        "DYNAMICS", COL, D, INSERT_VALUES, SCATTER_REVERSE);

    // Solve problem at time Zero
    if (is_solve_at_time_zero) {

      Mat Aij = shellAij_ctx->K;
      Vec F;
      CHKERR m_field.getInterface<VecManager>()->vecCreateGhost("Kuu", COL, &F);
      Vec D;
      CHKERR VecDuplicate(F, &D);
      
      // Set vector for Kuu problem from the mesh data
      CHKERR m_field.getInterface<VecManager>()->setLocalGhostVector(
          "Kuu", COL, D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

      SnesCtx snes_ctx(m_field, "Kuu");

      SNES snes;
      CHKERR SNESCreate(PETSC_COMM_WORLD, &snes);
      CHKERR SNESSetApplicationContext(snes, &snes_ctx);
      CHKERR SNESSetFunction(snes, F, SnesRhs, &snes_ctx);
      CHKERR SNESSetJacobian(snes, Aij, Aij, SnesMat, &snes_ctx);
      CHKERR SNESSetFromOptions(snes);

      DirichletDisplacementBc my_dirichlet_bc(m_field, "DISPLACEMENT",
                                              PETSC_NULL, D, F);

      SnesCtx::FEMethodsSequence &loops_to_do_Rhs =
          snes_ctx.get_loops_to_do_Rhs();
      snes_ctx.get_preProcess_to_do_Rhs().push_back(&my_dirichlet_bc);
      fluid_pressure_fe.getLoopFe().ts_t = 0;
      CHKERR add_static_rhs(loops_to_do_Rhs);
      snes_ctx.get_postProcess_to_do_Rhs().push_back(&my_dirichlet_bc);

      SnesCtx::FEMethodsSequence &loops_to_do_Mat =
          snes_ctx.get_loops_to_do_Mat();
      snes_ctx.get_preProcess_to_do_Mat().push_back(&my_dirichlet_bc);
      loops_to_do_Mat.push_back(
          SnesCtx::PairNameFEMethodPtr("ELASTIC", &elastic.getLoopFeLhs()));
      snes_ctx.get_postProcess_to_do_Mat().push_back(&my_dirichlet_bc);

      CHKERR m_field.getInterface<FieldBlas>()->fieldScale(0, "VELOCITY");
      CHKERR m_field.getInterface<FieldBlas>()->fieldScale(0,
                                                           "DOT_DISPLACEMENT");
      CHKERR m_field.getInterface<FieldBlas>()->fieldScale(0, "DOT_VELOCITY");

      CHKERR m_field.getInterface<VecManager>()->setLocalGhostVector(
          "Kuu", COL, D, INSERT_VALUES, SCATTER_FORWARD);

      CHKERR SNESSolve(snes, PETSC_NULL, D);
      int its;
      CHKERR SNESGetIterationNumber(snes, &its);
      MOFEM_LOG_C("DYNAMIC", Sev::inform, "number of Newton iterations = %d\n",
                  its);

      // Set data on the mesh
      CHKERR m_field.getInterface<VecManager>()->setGlobalGhostVector(
          "Kuu", COL, D, INSERT_VALUES, SCATTER_REVERSE);

      CHKERR VecDestroy(&F);
      CHKERR VecDestroy(&D);
      CHKERR SNESDestroy(&snes);
    }

    if (is_solve_at_time_zero) {
      CHKERR m_field.getInterface<VecManager>()->setLocalGhostVector(
          "DYNAMICS", COL, D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR TSSetSolution(ts, D);
    }

#if PETSC_VERSION_GE(3, 7, 0)
    CHKERR TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);
#endif
    CHKERR TSSolve(ts, D);
    CHKERR TSGetTime(ts, &ftime);

    PetscInt steps, snesfails, rejects, nonlinits, linits;
    CHKERR TSGetTimeStepNumber(ts, &steps);
    CHKERR TSGetSNESFailures(ts, &snesfails);
    CHKERR TSGetStepRejections(ts, &rejects);
    CHKERR TSGetSNESIterations(ts, &nonlinits);
    CHKERR TSGetKSPIterations(ts, &linits);
    MOFEM_LOG_C("DYNAMIC", Sev::inform,
                "steps %d (%d rejected, %D SNES fails), ftime %g, nonlinits "
                "%d, linits %D\n",
                steps, rejects, snesfails, ftime, nonlinits, linits);
    CHKERR TSDestroy(&ts);

    CHKERR VecDestroy(&F);
    CHKERR VecDestroy(&D);
    CHKERR MatDestroy(&shellAij_ctx->K);
    CHKERR MatDestroy(&shellAij_ctx->M);
    CHKERR VecScatterDestroy(&shellAij_ctx->scatterU);
    CHKERR VecScatterDestroy(&shellAij_ctx->scatterV);
    CHKERR MatDestroy(&shell_Aij);
    delete shellAij_ctx;
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
