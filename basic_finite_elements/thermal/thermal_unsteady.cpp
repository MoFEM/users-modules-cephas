/** \file thermal_unsteady.cpp
 \ingroup mofem_thermal_elem
 \brief Example of thermal unsteady analyze.

 TODO:
 \todo Make it work in distributed meshes with multigird solver. At the moment
 it is not working efficient as can.
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

#include <boost/program_options.hpp>
using namespace std;
namespace po = boost::program_options;

#ifdef __GROUND_SURFACE_TEMPERATURE_HPP

  #include <GenericClimateModel.hpp>
  #include <GroundSurfaceTemperature.hpp>

  #include <time.h>
  extern "C" {
    #include <spa.h>
  }
  #include <CrudeClimateModel.hpp>

#endif // __GROUND_SURFACE_TEMPERATURE_HPP

static char help[] =
  "-my_file mesh file\n"
  "-order set approx. order to all blocks\n"
  "-my_block_config set block data\n"
  "-my_ground_analysis_data data for crude climate model\n"
  "\n";

struct BlockOptionData {
  int oRder;
  double cOnductivity;
  double cApacity;
  double initTemp;
  BlockOptionData():
    oRder(-1),
    cOnductivity(-1),
    cApacity(-1),
    initTemp(0) {}
};

struct MonitorPostProc: public FEMethod {

  MoFEM::Interface &mField;
  PostProcVolumeOnRefinedMesh postProc;

  bool iNit;
  int pRT;

  MonitorPostProc(MoFEM::Interface &m_field):
    FEMethod(),mField(m_field),postProc(m_field),iNit(false) {
    
    PetscBool flg = PETSC_TRUE;
    CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_output_prt", &pRT,
                              &flg);
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
    if(flg!=PETSC_TRUE) {
      pRT = 1;
    }
  }

  MoFEMErrorCode preProcess() {
    MoFEMFunctionBeginHot;
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode operator()() {
    MoFEMFunctionBeginHot;
    MoFEMFunctionReturnHot(0);
  }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;
    
    if(!iNit) {
      CHKERR postProc.generateReferenceElementMesh(); 
      CHKERR postProc.addFieldValuesPostProc("TEMP"); 
      CHKERR postProc.addFieldValuesPostProc("TEMP_RATE"); 
      CHKERR postProc.addFieldValuesGradientPostProc("TEMP"); 
      CHKERR postProc.addFieldValuesPostProc("MESH_NODE_POSITIONS"); 
      iNit = true;
    }
    int step;
    CHKERR TSGetTimeStepNumber(ts,&step); 
    
    if((step)%pRT==0) {
      CHKERR mField.loop_finite_elements("DMTHERMAL","THERMAL_FE",postProc); 
      std::ostringstream sss;
      sss << "out_thermal_" << step << ".h5m";
      CHKERR postProc.writeFile(sss.str().c_str()); 
    }
    MoFEMFunctionReturn(0);
  }

};


int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_type mumps \n"
                                 "-ksp_monitor \n"
                                 "-snes_type newtonls \n"
                                 "-snes_linesearch_type basic \n"
                                 "-snes_max_it 100 \n"
                                 "-snes_atol 1e-8 \n"
                                 "-snes_rtol 1e-8 \n"
                                 "-snes_monitor \n"
                                 "-ts_monitor \n"
                                 "-ts_type beuler \n"
                                 "-ts_exact_final_time stepover \n";

  string param_file = "param_file.petsc";
  if (!static_cast<bool>(ifstream(param_file))) {
    std::ofstream file(param_file.c_str(), std::ios::ate);
    if (file.is_open()) {
      file << default_options;
      file.close();
    }
  }

  MoFEM::Core::Initialize(&argc, &argv, param_file.c_str(), help);
  
  try {

  PetscBool flg = PETSC_TRUE;
  char mesh_file_name[255];
  CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-my_file",
                               mesh_file_name, 255, &flg);
  if(flg != PETSC_TRUE) {
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_FOUND,
            "*** ERROR -my_file (MESH FILE NEEDED)");
  }

  char time_data_file_for_ground_surface[255];
  PetscBool ground_temperature_analysis = PETSC_FALSE;
  CHKERR PetscOptionsGetString(PETSC_NULL,PETSC_NULL,"-my_ground_analysis_data",
    time_data_file_for_ground_surface,255,&ground_temperature_analysis); 
  if(ground_temperature_analysis) {
#ifndef WITH_ADOL_C
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_INSTALLED,
            "*** ERROR to do ground thermal analysis MoFEM need to be compiled "
            "with ADOL-C");
#endif // WITH_ADOL_C
  }

  //create MoAB database
  moab::Core mb_instance;
  moab::Interface& moab = mb_instance;
  const char *option;
  option = "";
  CHKERR moab.load_file(mesh_file_name, 0, option); 
  //create MoFEM  database
  MoFEM::Core core(moab);
  MoFEM::Interface& m_field = core;

  DMType dm_name = "DMTHERMAL";
  CHKERR DMRegister_MoFEM(dm_name);
  // create dm instance
  DM dm;
  CHKERR DMCreate(PETSC_COMM_WORLD, &dm);
  CHKERR DMSetType(dm, dm_name);

  //set entities bit level (this allow to set refinement levels for h-adaptivity)
  //only one level is used in this example
  BitRefLevel bit_level0;
  bit_level0.set(0);
  CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(0, 3,
                                                                    bit_level0);

  //Fields H1 space rank 1
  CHKERR m_field.add_field("TEMP", H1, AINSWORTH_LEGENDRE_BASE, 1,
                           MB_TAG_SPARSE, MF_ZERO);
  CHKERR m_field.add_field("TEMP_RATE", H1, AINSWORTH_LEGENDRE_BASE, 1,
                           MB_TAG_SPARSE, MF_ZERO);

  //Add field H1 space rank 3 to approximate geometry using hierarchical basis
  //For 10 node tets, before use, geometry is projected on that field (see below)
  CHKERR m_field.add_field(
    "MESH_NODE_POSITIONS",H1,AINSWORTH_LEGENDRE_BASE,3,MB_TAG_SPARSE,MF_ZERO
  ); 

  //meshset consisting all entities in mesh
  EntityHandle root_set = moab.get_root_set();
  //add entities to field (root_mesh, i.e. on all mesh etities fields are approx.)
  CHKERR m_field.add_ents_to_field_by_type(root_set,MBTET,"TEMP"); 
  CHKERR m_field.add_ents_to_field_by_type(root_set,MBTET,"TEMP_RATE"); 

  int order;
  CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_order", &order, &flg);
  
  if (flg != PETSC_TRUE) {
    order = 1;
  }
  // set app. order
  // see Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes
  // (Mark Ainsworth & Joe Coyle)  for simplicity of example to all entities is
  // applied the same order
  CHKERR m_field.set_field_order(root_set,MBTET,"TEMP",order); 
  CHKERR m_field.set_field_order(root_set,MBTRI,"TEMP",order); 
  CHKERR m_field.set_field_order(root_set,MBEDGE,"TEMP",order); 
  CHKERR m_field.set_field_order(root_set,MBVERTEX,"TEMP",1); 

  CHKERR m_field.set_field_order(root_set,MBTET,"TEMP_RATE",order); 
  CHKERR m_field.set_field_order(root_set,MBTRI,"TEMP_RATE",order); 
  CHKERR m_field.set_field_order(root_set,MBEDGE,"TEMP_RATE",order); 
  CHKERR m_field.set_field_order(root_set,MBVERTEX,"TEMP_RATE",1); 

  //geometry approximation is set to 2nd oreder
  CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET,
                                           "MESH_NODE_POSITIONS");
  CHKERR m_field.set_field_order(0,MBTET,"MESH_NODE_POSITIONS",2); 
  CHKERR m_field.set_field_order(0,MBTRI,"MESH_NODE_POSITIONS",2); 
  CHKERR m_field.set_field_order(0,MBEDGE,"MESH_NODE_POSITIONS",2); 
  CHKERR m_field.set_field_order(0,MBVERTEX,"MESH_NODE_POSITIONS",1); 

  // configure blocks by parsing config file
  // it allow to set approximation order for each block independently
  PetscBool block_config;
  char block_config_file[255];
  CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-my_block_config",
                               block_config_file, 255, &block_config);
  std::map<int,BlockOptionData> block_data;
  bool solar_radiation = false;
  if (block_config) {
    try {
      ifstream ini_file(block_config_file);
      // std::cerr << block_config_file << std::endl;
      po::variables_map vm;
      po::options_description config_file_options;
      for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {

        std::ostringstream str_order;
        str_order << "block_" << it->getMeshsetId() << ".temperature_order";
        config_file_options.add_options()(
            str_order.str().c_str(),
            po::value<int>(&block_data[it->getMeshsetId()].oRder)
                ->default_value(order));

        std::ostringstream str_cond;
        str_cond << "block_" << it->getMeshsetId() << ".heat_conductivity";
        config_file_options.add_options()(
            str_cond.str().c_str(),
            po::value<double>(&block_data[it->getMeshsetId()].cOnductivity)
                ->default_value(-1));

        std::ostringstream str_capa;
        str_capa << "block_" << it->getMeshsetId() << ".heat_capacity";
        config_file_options.add_options()(
            str_capa.str().c_str(),
            po::value<double>(&block_data[it->getMeshsetId()].cApacity)
                ->default_value(-1));

        std::ostringstream str_init_temp;
        str_init_temp << "block_" << it->getMeshsetId()
                      << ".initial_temperature";
        config_file_options.add_options()(
            str_init_temp.str().c_str(),
            po::value<double>(&block_data[it->getMeshsetId()].initTemp)
                ->default_value(0));
      }
      config_file_options.add_options()(
          "climate_model.solar_radiation",
          po::value<bool>(&solar_radiation)->default_value(false));

      po::parsed_options parsed =
          parse_config_file(ini_file, config_file_options, true);
      store(parsed,vm);
      po::notify(vm);

      for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
        if (block_data[it->getMeshsetId()].oRder == -1)
          continue;
        if (block_data[it->getMeshsetId()].oRder == order)
          continue;
        PetscPrintf(PETSC_COMM_WORLD, "Set block %d oRder to %d\n",
                    it->getMeshsetId(), block_data[it->getMeshsetId()].oRder);
        Range block_ents;
        CHKERR moab.get_entities_by_handle(it->meshset, block_ents, true);
        Range ents_to_set_order;
        CHKERR moab.get_adjacencies(block_ents, 3, false, ents_to_set_order,
                                    moab::Interface::UNION);
        ents_to_set_order = ents_to_set_order.subset_by_type(MBTET);
        CHKERR moab.get_adjacencies(block_ents, 2, false, ents_to_set_order,
                                    moab::Interface::UNION);
        CHKERR moab.get_adjacencies(block_ents, 1, false, ents_to_set_order,
                                    moab::Interface::UNION);
        CHKERR m_field.set_field_order(ents_to_set_order, "TEMP",
                                       block_data[it->getMeshsetId()].oRder);
        CHKERR m_field.set_field_order(ents_to_set_order, "TEMP_RATE",
                                       block_data[it->getMeshsetId()].oRder);
      }
      std::vector<std::string> additional_parameters;
      additional_parameters =
          collect_unrecognized(parsed.options, po::include_positional);
      for (std::vector<std::string>::iterator vit =
               additional_parameters.begin();
           vit != additional_parameters.end(); vit++) {
        CHKERR PetscPrintf(PETSC_COMM_WORLD,
                           "** WARRING Unrecognised option %s\n", vit->c_str());
      }

    } catch (const std::exception& ex) {
      std::ostringstream ss;
      ss << ex.what() << std::endl;
      SETERRQ(PETSC_COMM_SELF,MOFEM_STD_EXCEPTION_THROW,ss.str().c_str());
    }
  }

  // this default class to calculate thermal elements
  ThermalElement thermal_elements(m_field);
  CHKERR thermal_elements.addThermalElements("TEMP");
  CHKERR thermal_elements.addThermalFluxElement("TEMP");
  CHKERR thermal_elements.addThermalConvectionElement("TEMP");
  CHKERR thermal_elements.addThermalRadiationElement("TEMP");
  // add rate of temperature to data field of finite element
  CHKERR m_field.modify_finite_element_add_field_data("THERMAL_FE",
                                                      "TEMP_RATE");
  // and temperature element default element operators at integration (gauss)
  // points
  CHKERR thermal_elements.setTimeSteppingProblem("TEMP", "TEMP_RATE");

  // set block material data from option file
  std::map<int, ThermalElement::BlockData>::iterator mit;
  mit = thermal_elements.setOfBlocks.begin();
  for (; mit != thermal_elements.setOfBlocks.end(); mit++) {
    // std::cerr << mit->first << std::endl;
    // std::cerr << block_data[mit->first].cOnductivity  << " " <<
    // block_data[mit->first].cApacity << std::endl;
    if (block_data[mit->first].cOnductivity != -1) {
      PetscPrintf(PETSC_COMM_WORLD, "Set block %d heat conductivity to %3.2e\n",
                  mit->first, block_data[mit->first].cOnductivity);
      for (int dd = 0; dd < 3; dd++) {
        mit->second.cOnductivity_mat(dd, dd) =
            block_data[mit->first].cOnductivity;
      }
    }
    if (block_data[mit->first].cApacity != -1) {
      PetscPrintf(PETSC_COMM_WORLD, "Set block %d heat capacity to %3.2e\n",
                  mit->first, block_data[mit->first].cApacity);
      mit->second.cApacity = block_data[mit->first].cApacity;
    }
  }

#ifdef __GROUND_SURFACE_TEMPERATURE_HPP
  GroundSurfaceTemperature ground_surface(m_field);
  CrudeClimateModel time_data(time_data_file_for_ground_surface);
  GroundSurfaceTemperature::PreProcess exectuteGenericClimateModel(&time_data);
  if (ground_temperature_analysis) {
    CHKERR ground_surface.addSurfaces("TEMP");
    CHKERR ground_surface.setOperators(&time_data, "TEMP");
  }
#endif //__GROUND_SURFACE_TEMPERATURE_HPP

  //build database, i.e. declare dofs, elements and adjacencies

  // build field
  CHKERR m_field.build_fields();
  // project 10 node tet approximation of geometry on hierarchical basis
  Projection10NodeCoordsOnField ent_method_material(m_field,
                                                    "MESH_NODE_POSITIONS");
  CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method_material);

  // set initial temperature from Cubit blocksets
  mit = thermal_elements.setOfBlocks.begin();
  for (; mit != thermal_elements.setOfBlocks.end(); mit++) {
    if (mit->second.initTemp != 0) {
      Range vertices;
      CHKERR moab.get_connectivity(mit->second.tEts, vertices, true);
      CHKERR m_field.getInterface<FieldBlas>()->setField(
          mit->second.initTemp, MBVERTEX, vertices, "TEMP");
    }
  }

  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
    if (block_data[it->getMeshsetId()].initTemp != 0) {
      Range block_ents;
      CHKERR moab.get_entities_by_handle(it->meshset, block_ents, true);
      Range vertices;
      CHKERR moab.get_connectivity(block_ents, vertices, true);
      CHKERR m_field.getInterface<FieldBlas>()->setField(
          block_data[it->getMeshsetId()].initTemp, MBVERTEX, vertices, "TEMP");
    }
  }

  // build finite elemnts
  CHKERR m_field.build_finite_elements();
  // build adjacencies
  CHKERR m_field.build_adjacencies(bit_level0);

  // delete old temperature recorded series
  SeriesRecorder *recorder_ptr;
  CHKERR m_field.getInterface(recorder_ptr);
  if (recorder_ptr->check_series("THEMP_SERIES")) {
    /*for(_IT_SERIES_STEPS_BY_NAME_FOR_LOOP_(recorder_ptr,"THEMP_SERIES",sit)) {
      CHKERR
    recorder_ptr->load_series_data("THEMP_SERIES",sit->get_step_number());
    }*/
    CHKERR recorder_ptr->delete_recorder_series("THEMP_SERIES");
  }

  // set dm data structure which created mofem data structures
  CHKERR DMMoFEMCreateMoFEM(dm, &m_field, dm_name, bit_level0);
  CHKERR DMSetFromOptions(dm);
  // add elements to dm
  CHKERR DMMoFEMAddElement(dm, "THERMAL_FE");
  CHKERR DMMoFEMAddElement(dm, "THERMAL_FLUX_FE");
  CHKERR DMMoFEMAddElement(dm, "THERMAL_CONVECTION_FE");
  CHKERR DMMoFEMAddElement(dm, "THERMAL_RADIATION_FE");
#ifdef __GROUND_SURFACE_TEMPERATURE_HPP
  if (ground_temperature_analysis) {
    CHKERR DMMoFEMAddElement(dm, "GROUND_SURFACE_FE");
  }
#endif //__GROUND_SURFACE_TEMPERATURE_HPP

  CHKERR DMSetUp(dm);

  // create matrices
  Vec T, F;
  CHKERR DMCreateGlobalVector_MoFEM(dm, &T);
  CHKERR VecDuplicate(T, &F);
  Mat A;
  CHKERR DMCreateMatrix_MoFEM(dm, &A);

  DirichletTemperatureBc dirichlet_bc(m_field, "TEMP", A, T, F);
  ThermalElement::UpdateAndControl update_velocities(m_field, "TEMP",
                                                     "TEMP_RATE");
  ThermalElement::TimeSeriesMonitor monitor(m_field, "THEMP_SERIES", "TEMP");
  MonitorPostProc post_proc(m_field);

  // Initialize data with values save of on the field
  CHKERR VecZeroEntries(T);
  CHKERR DMoFEMMeshToLocalVector(dm, T, INSERT_VALUES, SCATTER_FORWARD);
  CHKERR DMoFEMPreProcessFiniteElements(dm, &dirichlet_bc);
  CHKERR DMoFEMMeshToGlobalVector(dm, T, INSERT_VALUES, SCATTER_REVERSE);

  // preprocess
  CHKERR DMMoFEMTSSetIFunction(dm, DM_NO_ELEMENT, NULL, &update_velocities,
                               NULL);
  CHKERR DMMoFEMTSSetIFunction(dm, DM_NO_ELEMENT, NULL, &dirichlet_bc, NULL);
  CHKERR DMMoFEMTSSetIJacobian(dm, DM_NO_ELEMENT, NULL, &dirichlet_bc, NULL);
#ifdef __GROUND_SURFACE_TEMPERATURE_HPP
  CHKERR DMMoFEMTSSetIFunction(dm, DM_NO_ELEMENT, NULL,
                               &exectuteGenericClimateModel, NULL);
  { // add preprocessor, calculating angle on which sun ray on the surface
    if (solar_radiation) {
      boost::ptr_vector<
          GroundSurfaceTemperature::SolarRadiationPreProcessor>::iterator it,
          hi_it;
      it    = ground_surface.preProcessShade.begin();
      hi_it = ground_surface.preProcessShade.end();
      for (; it != hi_it; it++) {
        CHKERR DMMoFEMTSSetIFunction(dm, DM_NO_ELEMENT, NULL, &*it, NULL);
      }
    }
  }
#endif //__GROUND_SURFACE_TEMPERATURE_HPP

  // loops rhs
  CHKERR DMMoFEMTSSetIFunction(dm, "THERMAL_FE", &thermal_elements.feRhs, NULL,
                               NULL);
  CHKERR DMMoFEMTSSetIFunction(dm, "THERMAL_FLUX_FE", &thermal_elements.feFlux,
                               NULL, NULL);
  CHKERR DMMoFEMTSSetIFunction(dm, "THERMAL_CONVECTION_FE",
                               &thermal_elements.feConvectionRhs, NULL, NULL);
  CHKERR DMMoFEMTSSetIFunction(dm, "THERMAL_RADIATION_FE",
                               &thermal_elements.feRadiationRhs, NULL, NULL);
#ifdef __GROUND_SURFACE_TEMPERATURE_HPP
  if (ground_temperature_analysis) {
    CHKERR DMMoFEMTSSetIFunction(dm, "GROUND_SURFACE_FE",
                                 &ground_surface.getFeGroundSurfaceRhs(), NULL,
                                 NULL);
  }
#endif //__GROUND_SURFACE_TEMPERATURE_HPP

  // loops lhs
  CHKERR DMMoFEMTSSetIJacobian(dm, "THERMAL_FE", &thermal_elements.feLhs, NULL,
                               NULL);
  CHKERR DMMoFEMTSSetIJacobian(dm, "THERMAL_CONVECTION_FE",
                               &thermal_elements.feConvectionLhs, NULL, NULL);
  CHKERR DMMoFEMTSSetIJacobian(dm, "THERMAL_RADIATION_FE",
                               &thermal_elements.feRadiationLhs, NULL, NULL);
#ifdef __GROUND_SURFACE_TEMPERATURE_HPP
  if (ground_temperature_analysis) {
    CHKERR DMMoFEMTSSetIJacobian(dm, "GROUND_SURFACE_FE",
                                 &ground_surface.getFeGroundSurfaceLhs(), NULL,
                                 NULL);
  }
#endif //__GROUND_SURFACE_TEMPERATURE_HPP

  //postprocess
  CHKERR DMMoFEMTSSetIFunction(dm,DM_NO_ELEMENT,NULL,NULL,&dirichlet_bc); 
  CHKERR DMMoFEMTSSetIJacobian(dm,DM_NO_ELEMENT,NULL,NULL,&dirichlet_bc); 

  TsCtx *ts_ctx;
  DMMoFEMGetTsCtx(dm,&ts_ctx);
  //add monitor operator
  ts_ctx->get_postProcess_to_do_Monitor().push_back(&monitor);
  ts_ctx->get_postProcess_to_do_Monitor().push_back(&post_proc);

  //create time solver
  TS ts;
  CHKERR TSCreate(PETSC_COMM_WORLD,&ts); 
  CHKERR TSSetType(ts,TSBEULER); 

  CHKERR TSSetIFunction(ts,F,PETSC_NULL,PETSC_NULL); 
  CHKERR TSSetIJacobian(ts,A,A,PETSC_NULL,PETSC_NULL); 
  //add monitor to TS solver
  CHKERR TSMonitorSet(ts,TsMonitorSet,ts_ctx,PETSC_NULL);  // !!!

  CHKERR recorder_ptr->add_series_recorder("THEMP_SERIES"); 
  //start to record
  CHKERR recorder_ptr->initialize_series_recorder("THEMP_SERIES"); 

  double ftime = 1;
  CHKERR TSSetDuration(ts,PETSC_DEFAULT,ftime); 
  CHKERR TSSetFromOptions(ts); 
  CHKERR TSSetDM(ts,dm); 

  CHKERR TSSolve(ts,T); 
  CHKERR TSGetTime(ts,&ftime); 

  //end recoder
  CHKERR recorder_ptr->finalize_series_recorder("THEMP_SERIES"); 

  PetscInt steps,snesfails,rejects,nonlinits,linits;
  CHKERR TSGetTimeStepNumber(ts,&steps); 
  CHKERR TSGetSNESFailures(ts,&snesfails); 
  CHKERR TSGetStepRejections(ts,&rejects); 
  CHKERR TSGetSNESIterations(ts,&nonlinits); 
  CHKERR TSGetKSPIterations(ts,&linits);

  PetscPrintf(PETSC_COMM_WORLD,
              "steps %D (%D rejected, %D SNES fails), ftime %g, nonlinits %D, "
              "linits %D\n",
              steps, rejects, snesfails, ftime, nonlinits, linits);

  // save solution, if boundary conditions are defined you can use that file in
  // mechanical problem to calculate thermal stresses
  PetscBool is_partitioned = PETSC_FALSE;
  CHKERR PetscOptionsGetBool(PETSC_NULL, PETSC_NULL, "-dm_is_partitioned",
                             &is_partitioned, PETSC_NULL);
  if (is_partitioned) {
    CHKERR moab.write_file("solution.h5m");
  } else {
    if (m_field.get_comm_rank() == 0) {
      CHKERR moab.write_file("solution.h5m");
    }
  }

  CHKERR MatDestroy(&A);
  CHKERR VecDestroy(&T);
  CHKERR VecDestroy(&F);

  CHKERR TSDestroy(&ts);

  }
  CATCH_ERRORS;

  return MoFEM::Core::Finalize();
}
