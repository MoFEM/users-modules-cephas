/* \file Dirichlet.hpp
 * \brief Implementation of Dirichlet boundary conditions
 * \ingroup Dirichlet_bc
 *
 * Structures and method in this file erase rows and column, set value on
 * matrix diagonal and on the right hand side vector to enforce boundary
 * condition.
 *
 * Current implementation is suboptimal, classes name too long. Need to
 * rethinking and improved, more elegant and more efficient implementation.
 *
 */

/* Notes:

 DirichletSetFieldFromBlock implemented by Zahur Ullah
 (Zahur.Ullah@glasgow.ac.uk)

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

#ifndef __DIRICHLET_HPP__
#define __DIRICHLET_HPP__

using namespace boost::numeric;

/** \brief Data from Cubit blocksets
 * \ingroup Dirichlet_bc
 */
struct DataFromBc {
  VectorDouble scaled_values;
  VectorDouble initial_values;
  VectorInt bc_flags;
  Range bc_ents[3];

  // for rotation
  bool is_rotation;
  FTensor::Tensor1<double, 3> t_normal;
  FTensor::Tensor1<double, 3> t_centr;
  double theta;

  DataFromBc()
      : scaled_values(3), initial_values(3), bc_flags(3), is_rotation(false) {}

  MoFEMErrorCode getBcData(DisplacementCubitBcData &mydata,
                           const MoFEM::CubitMeshSets *it);
  MoFEMErrorCode getBcData(std::vector<double> &mydata,
                           const MoFEM::CubitMeshSets *it);
  MoFEMErrorCode getEntitiesFromBc(MoFEM::Interface &mField,
                                   const MoFEM::CubitMeshSets *it);
};

/** \brief Set Dirichlet boundary conditions on displacements
 * \ingroup Dirichlet_bc
 */
struct DirichletDisplacementBc : public MoFEM::FEMethod {

  MoFEM::Interface &mField;
  const std::string fieldName; ///< field name to set Dirichlet BC
  double dIag;                 ///< diagonal value set on zeroed column and rows

  DirichletDisplacementBc(MoFEM::Interface &m_field,
                          const std::string &field_name, Mat Aij, Vec X, Vec F,
                          string blockset_name = "DISPLACEMENT");
  DirichletDisplacementBc(MoFEM::Interface &m_field,
                          const std::string &field_name,
                          string blockset_name = "DISPLACEMENT");

  std::map<DofIdx, FieldData> mapZeroRows;
  std::vector<int> dofsIndices;
  std::vector<double> dofsValues;
  std::vector<double> dofsXValues;
  const std::string blocksetName;

  boost::ptr_vector<MethodForForceScaling> methodsOp;
  virtual MoFEMErrorCode iNitialize();

  MoFEMErrorCode preProcess();
  MoFEMErrorCode operator()() { return 0; }
  MoFEMErrorCode postProcess();
  /**
   * @brief Get the Bc Data From Sets And Blocks object
   *  Use DISPLACEMENT blockset name (default)
   *  with 6 atributes:
   *  1,2,3 are values of displacements x,y,z
   *  4,5,6 are flags for x,y,z (0 or 1)
   * @param bc_data
   * @return MoFEMErrorCode
   */
  MoFEMErrorCode getBcDataFromSetsAndBlocks(std::vector<DataFromBc> &bc_data);
  /**
   * @brief Get the Rotation Bc From Block object
   *  Use ROTATION blockset name
   *  with 7 atributes:
   *  1,2,3 are x,y,z coords of the center of rotation
   *  4,5,6 are are angular velocities in x,y,z
   * @param bc_data
   * @return MoFEMErrorCode
   */
  MoFEMErrorCode getRotationBcFromBlock(std::vector<DataFromBc> &bc_data);

  /**
   * @brief Calculate displacements from rotation for particular dof
   * @param dof
   * @param bc_data
   * @return MoFEMErrorCode
   */
  MoFEMErrorCode calculateRotationForDof(VectorDouble3 &coords,
                                                DataFromBc &bc_data);
  MoFEMErrorCode calculateRotationForDof(EntityHandle ent,
                                                DataFromBc &bc_data);
  MoFEMErrorCode applyScaleBcData(DataFromBc &bc_data);
};

struct BcEntMethodDisp : public MoFEM::EntityMethod {
  DirichletDisplacementBc *dirichletBcPtr;
  DataFromBc &dataFromDirichletBc;
  BcEntMethodDisp(DirichletDisplacementBc *dirichlet_bc_ptr,
                  DataFromBc &data_from_dirichlet_bc)
      : dirichletBcPtr(dirichlet_bc_ptr), dataFromDirichletBc(data_from_dirichlet_bc) {}

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode postProcess() { return 0; }
  MoFEMErrorCode operator()() {
    MoFEMFunctionBegin;
    auto &mField = dirichletBcPtr->mField;
    auto &bc_it = dataFromDirichletBc;

    EntityHandle v = entPtr->getEnt();
    int coeff = fieldPtr->getNbOfCoeffs();
    CHKERR dirichletBcPtr->calculateRotationForDof(v, bc_it);
    for (int i = 0; i != coeff; i++) {
      if (bc_it.bc_flags[i]) {
        if (entPtr->getEntType() == MBVERTEX) {
          entPtr->getEntFieldData()[i] = bc_it.scaled_values[i];
        } else if (!entPtr->getEntFieldData().empty()) {
          entPtr->getEntFieldData()[i] = 0;
        }
      }
    }

    MoFEMFunctionReturn(0);
  }
};

struct BcEntMethodSpatial : public BcEntMethodDisp {
  // using BcEntMethodDisp::BcEntMethodDisp;
  string materialPositions;
  BcEntMethodSpatial(DirichletDisplacementBc *dirichlet_bc_ptr,
                     DataFromBc &data_from_dirichlet_bc,
                     string material_positions)
      : BcEntMethodDisp(dirichlet_bc_ptr, data_from_dirichlet_bc),
        materialPositions(material_positions) {}

  MoFEMErrorCode operator()() {
    MoFEMFunctionBegin;
    EntityHandle ent = entPtr->getEnt();
    auto &mField = dirichletBcPtr->mField;
    auto &bc_it = dataFromDirichletBc;
    EntityHandle v = entPtr->getEnt();

    const FieldEntity_multiIndex *field_ents;
    CHKERR mField.get_field_ents(&field_ents);
    auto &field_ents_by_uid = field_ents->get<Unique_mi_tag>();

    auto get_coords = [&]() {
      VectorDouble3 coords(3);
      if (entPtr->getEntType() == MBVERTEX) {
        auto eit =
            field_ents_by_uid.find(FieldEntity::getLocalUniqueIdCalculate(
                mField.get_field_bit_number(materialPositions), ent));
        if (eit != field_ents_by_uid.end())
          noalias(coords) = (*eit)->getEntFieldData();
        else
          CHKERR mField.get_moab().get_coords(&ent, 1, &*coords.data().begin());
      }
      return coords;
    };

    int coeff = fieldPtr->getNbOfCoeffs();
    auto coords = get_coords();

    CHKERR dirichletBcPtr->calculateRotationForDof(v, bc_it);
    for (int i = 0; i != coeff; i++) {
      if (bc_it.bc_flags[i]) {
        if (entPtr->getEntType() == MBVERTEX) {
          entPtr->getEntFieldData()[i] = coords(i) + bc_it.scaled_values[i];
        } else if (!entPtr->getEntFieldData().empty()) {
          entPtr->getEntFieldData()[i] = 0;
        }
      }
    }

    MoFEMFunctionReturn(0);
  }
};

/// \deprecated use DirichletDisplacementBc
DEPRECATED typedef DirichletDisplacementBc DisplacementBCFEMethodPreAndPostProc;

/** \brief Set Dirichlet boundary conditions on spatial displacements
 * \ingroup Dirichlet_bc
 */
struct DirichletSpatialPositionsBc : public DirichletDisplacementBc {

  DirichletSpatialPositionsBc(
      MoFEM::Interface &m_field, const std::string &field_name, Mat aij, Vec x,
      Vec f, const std::string material_positions = "MESH_NODE_POSITIONS",
      const std::string blockset_name = "DISPLACEMENT")
      : DirichletDisplacementBc(m_field, field_name, aij, x, f, blockset_name),
        materialPositions(material_positions) {}

  DirichletSpatialPositionsBc(
      MoFEM::Interface &m_field, const std::string &field_name,
      const std::string material_positions = "MESH_NODE_POSITIONS",
      const std::string blockset_name = "DISPLACEMENT")
      : DirichletDisplacementBc(m_field, field_name, blockset_name),
        materialPositions(material_positions) {}

  std::string materialPositions; ///< name of the field with reference material
                                 ///< positions
  std::vector<std::string> fixFields; ///<

  VectorDouble cOords;
  MoFEMErrorCode iNitialize();
};

/// \deprecated use DirichletSpatialPositionsBc
DEPRECATED typedef DirichletSpatialPositionsBc
    SpatialPositionsBCFEMethodPreAndPostProc;

struct DirichletTemperatureBc : public DirichletDisplacementBc {

  DirichletTemperatureBc(MoFEM::Interface &m_field,
                         const std::string &field_name, Mat aij, Vec x, Vec f)
      : DirichletDisplacementBc(m_field, field_name, aij, x, f) {}

  DirichletTemperatureBc(MoFEM::Interface &m_field,
                         const std::string &field_name)
      : DirichletDisplacementBc(m_field, field_name) {}

  MoFEMErrorCode iNitialize();
};

/// \deprecated use DirichletTemperatureBc
DEPRECATED typedef DirichletTemperatureBc TemperatureBCFEMethodPreAndPostProc;

/** \brief Fix dofs on entities
 * \ingroup Dirichlet_bc
 */
struct DirichletFixFieldAtEntitiesBc : public DirichletDisplacementBc {

  Range eNts;
  std::vector<std::string> fieldNames;
  DirichletFixFieldAtEntitiesBc(MoFEM::Interface &m_field,
                                const std::string field_name, Mat aij, Vec x,
                                Vec f, Range &ents)
      : DirichletDisplacementBc(m_field, field_name, aij, x, f), eNts(ents) {
    fieldNames.push_back(fieldName);
  }

  DirichletFixFieldAtEntitiesBc(MoFEM::Interface &m_field,
                                const std::string field_name, Range &ents)
      : DirichletDisplacementBc(m_field, field_name), eNts(ents) {
    fieldNames.push_back(fieldName);
  }

  MoFEMErrorCode iNitialize();
  MoFEMErrorCode preProcess();
  MoFEMErrorCode postProcess();
};

/** \brief Set Dirichlet boundary conditions on displacements by removing dofs
 * \ingroup Dirichlet_bc
 */
struct DirichletDisplacementRemoveDofsBc : public DirichletDisplacementBc {

  boost::shared_ptr<vector<DataFromBc>> bcDataPtr;
  bool isPartitioned;
  string problemName;

  DirichletDisplacementRemoveDofsBc(MoFEM::Interface &m_field,
                                    const std::string &field_name,
                                    const std::string &problem_name,
                                    string blockset_name = "DISPLACEMENT",
                                    bool is_partitioned = false)
      : DirichletDisplacementBc(m_field, field_name, blockset_name),
        problemName(problem_name), isPartitioned(is_partitioned) {}

  MoFEMErrorCode iNitialize();

  boost::shared_ptr<EntityMethod> getEntMethodPtr(DataFromBc &data) {
    return boost::make_shared<BcEntMethodDisp>(this, data);
  }

  MoFEMErrorCode preProcess();
  MoFEMErrorCode operator()() { return 0; }
  MoFEMErrorCode postProcess() { return 0; }
};

/** \brief Set Dirichlet boundary conditions on spatial positions  by removing dofs
 * \ingroup Dirichlet_bc
 */
struct DirichletSpatialRemoveDofsBc : public DirichletDisplacementRemoveDofsBc {

  std::string materialPositions;

  DirichletSpatialRemoveDofsBc(
      MoFEM::Interface &m_field, const std::string &field_name,
      const std::string &problem_name,
      const std::string material_positions = "MESH_NODE_POSITIONS",
      string blockset_name = "DISPLACEMENT", bool is_partitioned = false)
      : DirichletDisplacementRemoveDofsBc(m_field, field_name, problem_name, blockset_name,
                                          is_partitioned),
        materialPositions(material_positions) {}

  boost::shared_ptr<EntityMethod> getEntMethodPtr(DataFromBc &data) {
    return boost::make_shared<BcEntMethodSpatial>(this, data,
                                                  materialPositions);
  }
};

/// \deprecated use DirichletFixFieldAtEntitiesBc
DEPRECATED typedef DirichletFixFieldAtEntitiesBc FixBcAtEntities;

/**
 * \brief Add boundary conditions form block set having 6 attributes
 *
 * First 3 values are magnitudes of dofs e.g. in x,y,z direction and next 3 are
 flags, respectively.
 * If flag is false ( = 0), particular dof is not taken into account.
    Usage in Cubit for displacement:
     block 1 tri 28 32
     block 1 name "DISPLACEMENT_1"
     block 1 attribute count 6
     block 1 attribute index 1 97  # any value
     block 1 attribute index 2 0
     block 1 attribute index 3 0
     block 1 attribute index 4 0  # flag for x dir
     block 1 attribute index 5 1  # flag for y dir
     block 1 attribute index 6 1  # flag for z dir
 This means that we set zero displacement on y and z dir and on x set
 direction freely. (value 97 is irrelevant because flag for 1 value is 0
 (false)) It can be usefull if we want to set boundary conditions directly to
 triangles e.g, since standard boundary conditions in Cubit allow only using
 nodeset or surface which might not work with mesh based on facet engine (e.g.
 STL file)
 */
struct DirichletSetFieldFromBlockWithFlags : public DirichletDisplacementBc {

  DirichletSetFieldFromBlockWithFlags(MoFEM::Interface &m_field,
                                      const std::string &field_name,
                                      const std::string &blockset_name, Mat aij,
                                      Vec x, Vec f)
      : DirichletDisplacementBc(m_field, field_name, aij, x, f, blockset_name) {
  }

  DirichletSetFieldFromBlockWithFlags(MoFEM::Interface &m_field,
                                      const std::string &field_name,
                                      const std::string &blockset_name)
      : DirichletDisplacementBc(m_field, field_name, blockset_name) {}
};

/// \deprecated use DirichletSetFieldFromBlockWithFlags
DEPRECATED typedef DirichletSetFieldFromBlockWithFlags
    DirichletBCFromBlockSetFEMethodPreAndPostProcWithFlags;

/// \deprecated use DirichletSetFieldFromBlockWithFlags
DEPRECATED typedef DirichletSetFieldFromBlockWithFlags
    DirichletSetFieldFromBlock;

/// \deprecated use DirichletSetFieldFromBlockWithFlags
DEPRECATED typedef DirichletSetFieldFromBlockWithFlags
    DirichletBCFromBlockSetFEMethodPreAndPostProc;
/**
 * @brief calculate reactions from vector of internal forces on meshsets
 *
 * example usage
 *
 * \code
      Vec F_int;
      DMCreateGlobalVector_MoFEM(dm, &F_int);

      feRhs->snes_ctx = FEMethod::CTX_SNESSETFUNCTION;
      feRhs->snes_f = F_int;
      DMoFEMLoopFiniteElements(dm, "ELASTIC", feRhs);

      VecAssemblyBegin(F_int);
      VecAssemblyEnd(F_int);
      VecGhostUpdateBegin(F_int, INSERT_VALUES, SCATTER_FORWARD);
      VecGhostUpdateEnd(F_int, INSERT_VALUES, SCATTER_FORWARD);

      Reactions my_react(m_field, "DM_ELASTIC", "U");
      my_react.calculateReactions(F_int);
      int fix_nodes_meshset_id = 1;
      cout << my_react.getReactionsFromSet(fix_nodes_meshset_id) << endl;

* \endcode
 */
struct Reactions {

  Reactions(MoFEM::Interface &m_field, string problem_name, string field_name)
      : mField(m_field), problemName(problem_name), fieldName(field_name) {}

  typedef std::map<int, VectorDouble> ReactionsMap;
  MoFEM::Interface &mField;
  /**
   * @brief Get the Reactions Map
   *
   * @return const ReactionsMap&
   */
  inline const ReactionsMap &getReactionsMap() const { return reactionsMap; }
  /**
   * @brief Get the Reactions at specified meshset id
   *
   * @param id meshset id (from Cubit)
   * @return const VectorDouble&
   */
  inline const VectorDouble &getReactionsFromSet(const int &id) const {
    return reactionsMap.at(id);
  }
  /**
   * @brief calculate reactions from a given vector
   *
   * @param internal forces vector
   * @return MoFEMErrorCode
   */
  MoFEMErrorCode calculateReactions(Vec &internal);

private:
  std::string problemName;
  std::string fieldName;
  ReactionsMap reactionsMap;
};

#endif //__DIRICHLET_HPP__

/**
 * \defgroup Dirichlet_bc Dirichlet boundary conditions
 * \ingroup user_modules
 **/
