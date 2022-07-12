/** \file ElasticMaterials.hpp
 * \ingroup nonlinear_elastic_elem
 * \brief Elastic materials
 */

/* MIT License
 *
 * Copyright (c) 2022
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef __ELASTICMATERIALS_HPP__
#define __ELASTICMATERIALS_HPP__

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <Hooke.hpp>
#include <NeoHookean.hpp>

#define MAT_KIRCHHOFF "KIRCHHOFF"
#define MAT_HOOKE "HOOKE"
#define MAT_NEOHOOKEAN "NEOHOOKEAN"

/** \brief Manage setting parameters and constitutive equations for
 * nonlinear/linear elastic materials \ingroup nonlinear_elastic_elem
 */
struct ElasticMaterials {

  MoFEM::Interface &mField;
  std::string defMaterial; ///< default material, if block is set to elastic,
                           ///< this material is used as default
  std::string configFile;  //< default name of config file

  bool iNitialized; ///< true if class is initialized

  ElasticMaterials(MoFEM::Interface &m_field,
                   std::string def_material = MAT_KIRCHHOFF)
      : mField(m_field), defMaterial(def_material),
        configFile("elastic_material.in"), iNitialized(false) {}

  virtual ~ElasticMaterials() {}

  std::map<std::string,
           boost::shared_ptr<NonlinearElasticElement::
                                 FunctionsToCalculatePiolaKirchhoffI<adouble>>>
      aDoubleMaterialModel; ///< Hash map of materials for evaluation with
                            ///< adouble, i.e. ADOL-C

  std::map<
      std::string,
      boost::shared_ptr<
          NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<double>>>
      doubleMaterialModel; ///< Hash map of materials for evaluation with double

  /**
   * Structure for material parameters for block
   */
  struct BlockOptionData {
    string mAterial;
    int oRder;
    double yOung;
    double pOisson;
    double dEnsity;
    double dashG;
    double dashPoisson;
    double aX, aY, aZ;
    BlockOptionData()
        : mAterial(MAT_KIRCHHOFF), oRder(-1), yOung(-1), pOisson(-2),
          dEnsity(-1), dashG(-1), dashPoisson(-1), aX(0), aY(0), aZ(0) {}
  };

  std::map<int, BlockOptionData> blockData; ///< Material parameters on blocks

  PetscBool isConfigFileSet; ///< True if config file is set from command line
  po::variables_map vM;

  /**
   * Initialize  model parameters
   * @return [description]
   */
  virtual MoFEMErrorCode iNit() {
    MoFEMFunctionBeginHot;
    // add new material below
    string mat_name;
    aDoubleMaterialModel[MAT_KIRCHHOFF] =
        boost::make_shared<NonlinearElasticElement::
                               FunctionsToCalculatePiolaKirchhoffI<adouble>>();
    doubleMaterialModel[MAT_KIRCHHOFF] = boost::make_shared<
        NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<double>>();
    aDoubleMaterialModel[MAT_HOOKE] = boost::make_shared<Hooke<adouble>>();
    doubleMaterialModel[MAT_HOOKE] = boost::make_shared<Hooke<double>>();
    aDoubleMaterialModel[MAT_NEOHOOKEAN] =
        boost::make_shared<NeoHookean<adouble>>();
    doubleMaterialModel[MAT_NEOHOOKEAN] =
        boost::make_shared<NeoHookean<double>>();
    std::ostringstream avilable_materials;
    avilable_materials << "set elastic material < ";
    std::map<std::string,
             boost::shared_ptr<
                 NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<
                     double>>>::iterator mit;
    mit = doubleMaterialModel.begin();
    for (; mit != doubleMaterialModel.end(); mit++) {
      avilable_materials << mit->first << " ";
    }
    avilable_materials << ">";

    CHKERR PetscOptionsBegin(mField.get_comm(), "",
                             "Elastic Materials Configuration", "none");
    char default_material[255];
    PetscBool def_mat_set;
    CHKERR PetscOptionsString(
        "-default_material", avilable_materials.str().c_str(), "",
        defMaterial.c_str(), default_material, 255, &def_mat_set);
    if (def_mat_set) {
      defMaterial = default_material;
      if (aDoubleMaterialModel.find(defMaterial) ==
          aDoubleMaterialModel.end()) {
        SETERRQ1(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
                 "material <%s> not implemented", default_material);
      }
    }
    char config_file[255];
    CHKERR PetscOptionsString("-elastic_material_configuration",
                              "elastic materials configure file name", "",
                              configFile.c_str(), config_file, 255,
                              &isConfigFileSet);
    if (isConfigFileSet) {
      configFile = config_file;
    }
    ierr = PetscOptionsEnd();
    CHKERRG(ierr);
    MoFEMFunctionReturnHot(0);
  }

  /** \brief read Elastic materials declaration for blocks and meshsets

    File parameters:
    \code
    [block_1]
    displacemet_order = 1/2 .. N
    material = KIRCHHOFF/HOOKE/NEOHOOKEAN
    young_modulus = 1
    poisson_ratio = 0.25
    density = 1
    a_x = 0
    a_y = 0
    a_z = 10
    \endcode

    To read material configuration file you need to use option:
    \code
    -elastic_material_configuration name_of_config_file
    \endcode

    */
  virtual MoFEMErrorCode readConfigFile() {
    MoFEMFunctionBegin;

    ifstream file(configFile.c_str());
    if (isConfigFileSet) {
      if (!file.good()) {
        SETERRQ1(PETSC_COMM_SELF, MOFEM_NOT_FOUND, "file < %s > not found",
                 configFile.c_str());
      }
    } else {
      if (!file.good()) {
        MoFEMFunctionReturnHot(0);
      }
    }

    po::options_description config_file_options;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {

      std::ostringstream str_order;
      str_order << "block_" << it->getMeshsetId() << ".displacemet_order";
      config_file_options.add_options()(
          str_order.str().c_str(),
          po::value<int>(&blockData[it->getMeshsetId()].oRder)
              ->default_value(-1));

      std::ostringstream str_material;
      str_material << "block_" << it->getMeshsetId() << ".material";
      config_file_options.add_options()(
          str_material.str().c_str(),
          po::value<std::string>(&blockData[it->getMeshsetId()].mAterial)
              ->default_value(defMaterial));

      std::ostringstream str_ym;
      str_ym << "block_" << it->getMeshsetId() << ".young_modulus";
      config_file_options.add_options()(
          str_ym.str().c_str(),
          po::value<double>(&blockData[it->getMeshsetId()].yOung)
              ->default_value(-1));

      std::ostringstream str_pr;
      str_pr << "block_" << it->getMeshsetId() << ".poisson_ratio";
      config_file_options.add_options()(
          str_pr.str().c_str(),
          po::value<double>(&blockData[it->getMeshsetId()].pOisson)
              ->default_value(-2));

      std::ostringstream str_density;
      str_density << "block_" << it->getMeshsetId() << ".density";
      config_file_options.add_options()(
          str_density.str().c_str(),
          po::value<double>(&blockData[it->getMeshsetId()].dEnsity)
              ->default_value(-1));

      std::ostringstream str_dashG;
      str_dashG << "block_" << it->getMeshsetId() << ".dashG";
      config_file_options.add_options()(
          str_dashG.str().c_str(),
          po::value<double>(&blockData[it->getMeshsetId()].dashG)
              ->default_value(-1));

      std::ostringstream str_dashPoisson;
      str_dashPoisson << "block_" << it->getMeshsetId() << ".dashPoisson";
      config_file_options.add_options()(
          str_dashPoisson.str().c_str(),
          po::value<double>(&blockData[it->getMeshsetId()].dashPoisson)
              ->default_value(-2));

      std::ostringstream str_ax;
      str_ax << "block_" << it->getMeshsetId() << ".a_x";
      config_file_options.add_options()(
          str_ax.str().c_str(),
          po::value<double>(&blockData[it->getMeshsetId()].aX)
              ->default_value(0));

      std::ostringstream str_ay;
      str_ay << "block_" << it->getMeshsetId() << ".a_y";
      config_file_options.add_options()(
          str_ay.str().c_str(),
          po::value<double>(&blockData[it->getMeshsetId()].aY)
              ->default_value(0));

      std::ostringstream str_az;
      str_az << "block_" << it->getMeshsetId() << ".a_z";
      config_file_options.add_options()(
          str_az.str().c_str(),
          po::value<double>(&blockData[it->getMeshsetId()].aZ)
              ->default_value(0));
    }
    po::parsed_options parsed =
        parse_config_file(file, config_file_options, true);
    store(parsed, vM);
    po::notify(vM);
    std::vector<std::string> additional_parameters;
    additional_parameters =
        collect_unrecognized(parsed.options, po::include_positional);
    for (std::vector<std::string>::iterator vit = additional_parameters.begin();
         vit != additional_parameters.end(); vit++) {
      CHKERR PetscPrintf(PETSC_COMM_WORLD,
                         "** WARNING Unrecognized option %s\n", vit->c_str());
    }

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setBlocksOrder() {
    MoFEMFunctionBeginHot;

    // set app. order
    PetscBool flg = PETSC_TRUE;
    PetscInt disp_order;
    CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-order", &disp_order,
                              &flg);
    if (flg != PETSC_TRUE) {
      disp_order = 1;
    }
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (blockData[it->getMeshsetId()].oRder == -1)
        continue;
      if (blockData[it->getMeshsetId()].oRder == disp_order)
        continue;
      PetscPrintf(mField.get_comm(), "Set block %d oRder to %d\n",
                  it->getMeshsetId(), blockData[it->getMeshsetId()].oRder);
      Range block_ents;
      rval = mField.get_moab().get_entities_by_handle(it->meshset, block_ents,
                                                      true);
      CHKERRG(rval);
      Range ents_to_set_order;
      CHKERR mField.get_moab().get_adjacencies(
          block_ents, 3, false, ents_to_set_order, moab::Interface::UNION);
      ents_to_set_order = ents_to_set_order.subset_by_type(MBTET);
      CHKERR mField.get_moab().get_adjacencies(
          block_ents, 2, false, ents_to_set_order, moab::Interface::UNION);
      CHKERR mField.get_moab().get_adjacencies(
          block_ents, 1, false, ents_to_set_order, moab::Interface::UNION);
      if (mField.check_field("DISPLACEMENT")) {
        CHKERR mField.set_field_order(ents_to_set_order, "DISPLACEMENT",
                                      blockData[it->getMeshsetId()].oRder);
      }
      if (mField.check_field("SPATIAL_POSITION")) {
        CHKERR mField.set_field_order(ents_to_set_order, "SPATIAL_POSITION",
                                      blockData[it->getMeshsetId()].oRder);
      }
      if (mField.check_field("DOT_SPATIAL_POSITION")) {
        CHKERR mField.set_field_order(ents_to_set_order, "DOT_SPATIAL_POSITION",
                                      blockData[it->getMeshsetId()].oRder);
      }
      if (mField.check_field("SPATIAL_VELOCITY")) {
        CHKERR mField.set_field_order(ents_to_set_order, "SPATIAL_VELOCITY",
                                      blockData[it->getMeshsetId()].oRder);
      }
    }
    MoFEMFunctionReturnHot(0);
  }

#ifdef __NONLINEAR_ELASTIC_HPP

  virtual MoFEMErrorCode
  setBlocks(std::map<int, NonlinearElasticElement::BlockData> &set_of_blocks) {
    MoFEMFunctionBegin;

    if (!iNitialized) {
      CHKERR iNit();
      CHKERR readConfigFile();
      CHKERR setBlocksOrder();
      iNitialized = true;
    }
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             mField, BLOCKSET | MAT_ELASTICSET, it)) {
      int id = it->getMeshsetId();
      Mat_Elastic mydata;
      CHKERR it->getAttributeDataStructure(mydata);
      EntityHandle meshset = it->getMeshset();
      CHKERR mField.get_moab().get_entities_by_type(
          meshset, MBTET, set_of_blocks[id].tEts, true);
      set_of_blocks[id].iD = id;
      set_of_blocks[id].E = mydata.data.Young;
      set_of_blocks[id].PoissonRatio = mydata.data.Poisson;
      if (blockData[id].yOung >= 0)
        set_of_blocks[id].E = blockData[id].yOung;
      if (blockData[id].pOisson >= -1)
        set_of_blocks[id].PoissonRatio = blockData[id].pOisson;

      blockData[id].mAterial = defMaterial;
      set_of_blocks[id].materialDoublePtr = doubleMaterialModel.at(defMaterial);
      set_of_blocks[id].materialAdoublePtr =
          aDoubleMaterialModel.at(defMaterial);

      PetscPrintf(mField.get_comm(),
                  "Block Id %d Young Modulus %3.2g Poisson Ration %3.2f "
                  "Material model %s Nb. of elements %d\n",
                  id, set_of_blocks[id].E, set_of_blocks[id].PoissonRatio,
                  blockData[id].mAterial.c_str(),
                  set_of_blocks[id].tEts.size());

      if (blockData[id].mAterial.compare(MAT_KIRCHHOFF) == 0) {
        set_of_blocks[id].materialDoublePtr =
            doubleMaterialModel.at(MAT_KIRCHHOFF);
        set_of_blocks[id].materialAdoublePtr =
            aDoubleMaterialModel.at(MAT_KIRCHHOFF);
      } else if (blockData[id].mAterial.compare(MAT_HOOKE) == 0) {
        set_of_blocks[id].materialDoublePtr = doubleMaterialModel.at(MAT_HOOKE);
        set_of_blocks[id].materialAdoublePtr =
            aDoubleMaterialModel.at(MAT_HOOKE);
      } else if (blockData[id].mAterial.compare(MAT_NEOHOOKEAN) == 0) {
        set_of_blocks[id].materialDoublePtr =
            doubleMaterialModel.at(MAT_NEOHOOKEAN);
        set_of_blocks[id].materialAdoublePtr =
            aDoubleMaterialModel.at(MAT_NEOHOOKEAN);
      } else {
        SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
                "field with that space is not implemented");
      }
    }
    MoFEMFunctionReturn(0);
  }

#endif //__NONLINEAR_ELASTIC_HPP

#ifdef __CONVECTIVE_MASS_ELEMENT_HPP

  MoFEMErrorCode
  setBlocks(std::map<int, ConvectiveMassElement::BlockData> &set_of_blocks) {
    MoFEMFunctionBeginHot;

    if (!iNitialized) {
      CHKERR iNit();
      CHKERR readConfigFile();
      CHKERR setBlocksOrder();
      iNitialized = true;
    }
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             mField, BLOCKSET | BODYFORCESSET, it)) {
      int id = it->getMeshsetId();
      EntityHandle meshset = it->getMeshset();
      CHKERR mField.get_moab().get_entities_by_type(
          meshset, MBTET, set_of_blocks[id].tEts, true);
      Block_BodyForces mydata;
      CHKERR it->getAttributeDataStructure(mydata);
      set_of_blocks[id].rho0 = mydata.data.density;
      set_of_blocks[id].a0.resize(3);
      set_of_blocks[id].a0[0] = mydata.data.acceleration_x;
      set_of_blocks[id].a0[1] = mydata.data.acceleration_y;
      set_of_blocks[id].a0[2] = mydata.data.acceleration_z;
      if (blockData[id].dEnsity >= 0) {
        set_of_blocks[id].rho0 = blockData[id].dEnsity;
        std::ostringstream str_ax;
        str_ax << "block_" << it->getMeshsetId() << ".a_x";
        std::ostringstream str_ay;
        str_ay << "block_" << it->getMeshsetId() << ".a_y";
        std::ostringstream str_az;
        str_az << "block_" << it->getMeshsetId() << ".a_z";
        if (vM.count(str_ax.str().c_str())) {
          set_of_blocks[id].a0[0] = blockData[id].aX;
        }
        if (vM.count(str_ay.str().c_str())) {
          set_of_blocks[id].a0[1] = blockData[id].aY;
        }
        if (vM.count(str_az.str().c_str())) {
          set_of_blocks[id].a0[2] = blockData[id].aZ;
        }
      }
      PetscPrintf(mField.get_comm(),
                  "Block Id %d Density %3.2g a_x = %3.2g a_y = %3.2g a_z = "
                  "%3.2g Nb. of elements %d\n",
                  id, set_of_blocks[id].rho0, set_of_blocks[id].a0[0],
                  set_of_blocks[id].a0[1], set_of_blocks[id].a0[2],
                  set_of_blocks[id].tEts.size());
    }

    MoFEMFunctionReturnHot(0);
  }

#endif //__CONVECTIVE_MASS_ELEMENT_HPP

#ifdef __KELVIN_VOIGT_DAMPER_HPP__

  MoFEMErrorCode setBlocks(
      std::map<int, KelvinVoigtDamper::BlockMaterialData> &set_of_blocks) {
    MoFEMFunctionBeginHot;

    if (!iNitialized) {
      CHKERR iNit();
      CHKERR readConfigFile();
      CHKERR setBlocksOrder();
      iNitialized = true;
    }

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      bool set = false;
      int id = it->getMeshsetId();
      EntityHandle meshset = it->getMeshset();
      if (it->getName().compare(0, 6, "DAMPER") == 0) {
        set = true;
        std::vector<double> data;
        CHKERR it->getAttributes(data);
        if (data.size() < 2) {
          SETERRQ(PETSC_COMM_SELF, 1, "Data inconsistency");
        }
        rval = mField.get_moab().get_entities_by_type(
            it->meshset, MBTET, set_of_blocks[it->getMeshsetId()].tEts, true);
        CHKERRG(rval);
        set_of_blocks[it->getMeshsetId()].gBeta = data[0];
        set_of_blocks[it->getMeshsetId()].vBeta = data[1];
      }
      if (blockData[id].dashG > 0) {
        set = true;
        Range tEts;
        rval =
            mField.get_moab().get_entities_by_type(meshset, MBTET, tEts, true);
        CHKERRG(rval);
        if (tEts.empty())
          continue;
        set_of_blocks[it->getMeshsetId()].tEts = tEts;
        set_of_blocks[it->getMeshsetId()].gBeta = blockData[id].dashG;
        set_of_blocks[it->getMeshsetId()].vBeta = blockData[id].dashPoisson;
      }
      if (set) {
        PetscPrintf(
            mField.get_comm(),
            "Block Id %d Damper Shear Modulus = %3.2g Poisson ratio = %3.2g\n",
            id, set_of_blocks[id].gBeta, set_of_blocks[id].vBeta);
      }
    }

    MoFEMFunctionReturnHot(0);
  }

#endif //__KELVIN_VOIGT_DAMPER_HPP__
};

#endif //__ELASTICMATERIALS_HPP__
