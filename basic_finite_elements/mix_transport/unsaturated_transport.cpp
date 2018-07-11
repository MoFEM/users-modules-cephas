/** \file unsaturated_transport.cpp
\brief Implementation of unsaturated flow problem
\example unsaturated_transport.cpp

\ingroup mofem_mix_transport_elem
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

#include <boost/program_options.hpp>
using namespace std;
namespace po = boost::program_options;

#include <BasicFiniteElements.hpp>
#include <MixTransportElement.hpp>
#include <UnsaturatedFlow.hpp>
#include <MaterialUnsaturatedFlow.hpp>

#ifndef WITH_ADOL_C
  #error "MoFEM need to be compiled with ADOL-C"
#endif

using namespace MoFEM;
using namespace MixTransport;
static char help[] = "...\n\n";

double GenericMaterial::ePsilon0 = 0;
double GenericMaterial::ePsilon1 = 0;

map<std::string,CommonMaterialData::RegisterHook> RegisterMaterials::mapOfRegistredMaterials;

int main(int argc, char *argv[]) {

  PetscInitialize(&argc,&argv,(char *)0,help);

  // Register DM Manager
  ierr = DMRegister_MoFEM("DMMOFEM"); CHKERRG(ierr); // register MoFEM DM in PETSc
  // Register materials
  ierr = RegisterMaterials()(); CHKERRG(ierr);

  PetscBool test_mat = PETSC_FALSE;
  ierr = PetscOptionsGetBool(
    PETSC_NULL,"","-test_mat",&test_mat,PETSC_NULL
  ); CHKERRG(ierr);

  if(test_mat==PETSC_TRUE) {
    CommonMaterialData data;
    // Testing van_genuchten
    MaterialVanGenuchten van_genuchten(data);
    van_genuchten.printMatParameters(-1,"testing");
    van_genuchten.printTheta(-1e-16,-1e4,-1e-12,"hTheta");
    van_genuchten.printKappa(-1e-16,-1,-1e-12,"hK");
    van_genuchten.printC(-1e-16,-1,-1e-12,"hC");
    ierr = PetscFinalize(); CHKERRG(ierr);
    return 0;
  }

  try {

    moab::Core mb_instance;
    moab::Interface& moab = mb_instance;

    // get file name form command line
    PetscBool flg_mesh_file_name = PETSC_TRUE;
    char mesh_file_name[255];
    PetscBool flg_conf_file_name = PETSC_TRUE;
    char conf_file_name[255];

    int order = 1;

    ierr = PetscOptionsBegin(
      PETSC_COMM_WORLD,"",
      "Unsaturated flow options","none"
    ); CHKERRG(ierr);
    ierr = PetscOptionsString(
      "-my_file",
      "mesh file name","","mesh.h5m",mesh_file_name,255,&flg_mesh_file_name
    ); CHKERRG(ierr);
    ierr = PetscOptionsString(
      "-configure",
      "material and bc configuration file name","", "unsaturated.cfg",
      conf_file_name,255,&flg_conf_file_name
    ); CHKERRG(ierr);
    ierr = PetscOptionsInt(
      "-my_order",
      "default approximation order","",
      order,&order,PETSC_NULL
    ); CHKERRG(ierr);
    ierr = PetscOptionsEnd(); CHKERRG(ierr);

    if(flg_mesh_file_name != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF,MOFEM_INVALID_DATA,"File name not set (e.g. -my_name mesh.h5m)");
    }
    if(flg_conf_file_name != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF,MOFEM_INVALID_DATA,"File name not set (e.g. -config unsaturated.cfg)");
    }

    // Create MOAB communicator
    MPI_Comm moab_comm_world;
    MPI_Comm_dup(PETSC_COMM_WORLD,&moab_comm_world);
    ParallelComm* pcomm = ParallelComm::get_pcomm(&moab,MYPCOMM_INDEX);
    if(pcomm == NULL) pcomm =  new ParallelComm(&moab,moab_comm_world);

    const char *option;
    option = "PARALLEL=READ_PART;"
    "PARALLEL_RESOLVE_SHARED_ENTS;"
    "PARTITION=PARALLEL_PARTITION;";
    rval = moab.load_file(mesh_file_name, 0, option); CHKERRG(rval);

    // Create mofem interface
    MoFEM::Core core(moab);
    MoFEM::Interface& m_field = core;

    // Add meshsets with material and boundary conditions
    MeshsetsManager *meshsets_manager_ptr;
    ierr = m_field.getInterface(meshsets_manager_ptr); CHKERRG(ierr);
    ierr = meshsets_manager_ptr->setMeshsetFromFile(); CHKERRG(ierr);

    PetscPrintf(PETSC_COMM_WORLD,"Read meshsets and added meshsets from bc.cfg\n");
    for(_IT_CUBITMESHSETS_FOR_LOOP_(m_field,it)) {
      PetscPrintf(
        PETSC_COMM_WORLD,
        "%s",static_cast<std::ostringstream&>(std::ostringstream().seekp(0) << *it << endl).str().c_str()
      );
    }

    // Set entities bit level
    ierr = m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(0,3,BitRefLevel().set(0)); CHKERRG(ierr);

    UnsaturatedFlowElement uf(m_field);

    ifstream ini_file(conf_file_name);
    po::variables_map vm;
    po::options_description config_file_options;
    Range domain_ents,bc_boundary_ents;

    map<int,CommonMaterialData> material_blocks;
    map<int,double> head_blocks;
    map<int,double> flux_blocks;
    // Set material blocks
    for(_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field,BLOCKSET,it)) {
      if(it->getName().compare(0,4,"SOIL")!=0) continue;
      // get block id
      const int block_id = it->getMeshsetId();
      std::string block_name = "mat_block_"+boost::lexical_cast<std::string>(block_id);
      material_blocks[block_id].blockId = block_id;
      material_blocks[block_id].addOptions(config_file_options,block_name);
    }
    for(_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field,BLOCKSET,it)) {
      if(it->getName().compare(0,4,"HEAD")!=0) continue;
      // get block id
      const int block_id = it->getMeshsetId();
      std::string block_name = "head_block_"+boost::lexical_cast<std::string>(block_id);
      config_file_options.add_options()
      ((block_name+".head").c_str(),po::value<double>(&head_blocks[block_id])->default_value(0));
    }
    for(_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field,BLOCKSET,it)) {
      if(it->getName().compare(0,4,"FLUX")!=0) continue;
      // get block id
      const int block_id = it->getMeshsetId();
      std::string block_name = "flux_block_"+boost::lexical_cast<std::string>(block_id);
      config_file_options.add_options()
      ((block_name+".flux").c_str(),po::value<double>(&head_blocks[block_id])->default_value(0));
    }
    po::parsed_options parsed = parse_config_file(ini_file,config_file_options,true);
    store(parsed,vm);
    po::notify(vm);
    for(_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field,BLOCKSET,it)) {
      if(it->getName().compare(0,4,"SOIL")!=0) continue;
      const int block_id = it->getMeshsetId();
      uf.dMatMap[block_id] =
      RegisterMaterials::mapOfRegistredMaterials.
      at(material_blocks[block_id].matName)(material_blocks.at(block_id));
      if(!uf.dMatMap.at(block_id)) {
        SETERRQ(PETSC_COMM_WORLD,MOFEM_DATA_INCONSISTENCY,"Material block not set");
      }
      // get block test
      rval = m_field.get_moab().get_entities_by_type(
        it->meshset,MBTET,uf.dMatMap.at(block_id)->tEts,true
      ); CHKERRG(rval);
      domain_ents.merge(uf.dMatMap.at(block_id)->tEts);
      uf.dMatMap.at(block_id)->printMatParameters(block_id,"Read material");
    }
    std::vector<std::string> additional_parameters;
    additional_parameters = collect_unrecognized(parsed.options,po::include_positional);
    for(std::vector<std::string>::iterator vit = additional_parameters.begin();
    vit!=additional_parameters.end();vit++) {
      ierr = PetscPrintf(m_field.get_comm(),"** WARNING Unrecognized option %s\n",vit->c_str()); CHKERRG(ierr);
    }
    // Set capillary pressure bc data
    for(_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field,BLOCKSET,it)) {
      if(it->getName().compare(0,4,"HEAD")!=0) continue;
      // get block id
      const int block_id = it->getMeshsetId();
      // create block data instance
      uf.bcValueMap[block_id] = boost::shared_ptr<UnsaturatedFlowElement::BcData>(
        new UnsaturatedFlowElement::BcData()
      );
      // get bc value
      std::vector<double> attributes;
      ierr = it->getAttributes(attributes); CHKERRG(ierr);
      uf.bcValueMap[block_id]->fixValue = attributes[0];
      std::string block_name = "head_block_"+boost::lexical_cast<std::string>(block_id);
      if(vm.count((block_name)+".head")) {
        uf.bcValueMap[block_id]->fixValue = head_blocks[block_id];
      }
      // cerr << uf.bcValueMap[block_id]->fixValue  << endl;
      // ierr = it->printAttributes(std::cout); CHKERRG(ierr);
      // get faces in the block
      rval = m_field.get_moab().get_entities_by_type(
        it->meshset,MBTRI,uf.bcValueMap[block_id]->eNts,true
      ); CHKERRG(rval);
      bc_boundary_ents.merge(uf.bcValueMap[block_id]->eNts);
    }

    int max_flux_id = 0;
    // Set water flux bc data
    for(_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field,BLOCKSET,it)) {
      if(it->getName().compare(0,4,"FLUX")!=0) continue;
      // get block id
      const int block_id = it->getMeshsetId();
      // create block data instance
      uf.bcFluxMap[block_id] = boost::shared_ptr<UnsaturatedFlowElement::BcData>(
        new UnsaturatedFlowElement::BcData()
      );
      // get bc value
      std::vector<double> attributes;
      ierr = it->getAttributes(attributes); CHKERRG(ierr);
      // ierr = it->printAttributes(std::cout); CHKERRG(ierr);
      uf.bcFluxMap[block_id]->fixValue = attributes[0];
      std::string block_name = "flux_block_"+boost::lexical_cast<std::string>(block_id);
      if(vm.count((block_name)+".flux")) {
        uf.bcValueMap[block_id]->fixValue = head_blocks[block_id];
      }
      // get faces in the block
      rval = m_field.get_moab().get_entities_by_type(
        it->meshset,MBTRI,uf.bcFluxMap[block_id]->eNts,true
      ); CHKERRG(rval);
      bc_boundary_ents.merge(uf.bcFluxMap[block_id]->eNts);
      max_flux_id = max_flux_id>block_id ? max_flux_id : block_id+1;
    }

    // Add zero flux on the rest of the boundary
    Range zero_flux_ents;
    {
      Skinner skin(&m_field.get_moab());
      Range domain_skin;
      rval = skin.find_skin(0,domain_ents,false,domain_skin); CHKERRG(rval);
      // filter not owned entities, those are not on boundary
      rval = pcomm->filter_pstatus(
        domain_skin,PSTATUS_SHARED|PSTATUS_MULTISHARED,PSTATUS_NOT,-1,&zero_flux_ents
      ); CHKERRG(rval);
      zero_flux_ents = subtract(zero_flux_ents,bc_boundary_ents);
      uf.bcFluxMap[max_flux_id] = boost::shared_ptr<UnsaturatedFlowElement::BcData>(
        new UnsaturatedFlowElement::BcData()
      );
      uf.bcFluxMap[max_flux_id]->fixValue = 0;
      uf.bcFluxMap[max_flux_id]->eNts = zero_flux_ents;
    }

    ierr = uf.addFields("VALUES","FLUXES",order); CHKERRG(ierr);
    ierr = uf.addFiniteElements("FLUXES","VALUES"); CHKERRG(ierr);
    ierr = m_field.add_ents_to_finite_element_by_type(
      zero_flux_ents,MBTRI,"MIX_BCFLUX"
    ); CHKERRG(ierr);
    ierr = uf.buildProblem(); CHKERRG(ierr);
    ierr = uf.createMatrices(); CHKERRG(ierr);
    ierr = uf.setFiniteElements(); CHKERRG(ierr);
    ierr = uf.calculateEssentialBc(); CHKERRG(ierr);
    ierr = uf.calculateInitialPc(); CHKERRG(ierr);
    ierr = uf.solveProblem(); CHKERRG(ierr);
    ierr = uf.destroyMatrices(); CHKERRG(ierr);

    MPI_Comm_free(&moab_comm_world);

  } catch (MoFEMException const &e) {
    SETERRQ(PETSC_COMM_SELF,e.errorCode,e.errorMessage);
  }

  ierr = PetscFinalize(); CHKERRG(ierr);

  return 0;
}
