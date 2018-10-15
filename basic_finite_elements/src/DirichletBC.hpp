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

/** \brief Set Dirichlet boundary conditions on displacements
  * \ingroup Dirichlet_bc
  */
struct DirichletDisplacementBc: public MoFEM::FEMethod {

  MoFEM::Interface& mField;
  const std::string fieldName;			///< field name to set Dirichlet BC
  double dIag;					      ///< diagonal value set on zeroed column and rows

  DirichletDisplacementBc(
    MoFEM::Interface& m_field,const std::string &field_name,
    Mat Aij,Vec X,Vec F
  );
  DirichletDisplacementBc(
    MoFEM::Interface& m_field,const std::string &field_name
  );

  std::map<DofIdx,FieldData> mapZeroRows;
  std::vector<int> dofsIndices;
  std::vector<double> dofsValues;
  std::vector<double> dofsXValues;
  virtual MoFEMErrorCode iNitalize();

  MoFEMErrorCode preProcess();
  MoFEMErrorCode postProcess();

  boost::ptr_vector<MethodForForceScaling> methodsOp;

};

/// \deprecated use DirichletDisplacementBc
DEPRECATED typedef DirichletDisplacementBc DisplacementBCFEMethodPreAndPostProc;

/** \brief Set Dirichlet boundary conditions on spatial displacements
  * \ingroup Dirichlet_bc
  */
struct DirichletSpatialPositionsBc: public DirichletDisplacementBc {

  DirichletSpatialPositionsBc(
    MoFEM::Interface& m_field,
    const std::string &field_name,
    Mat aij,Vec x,Vec f,
    const std::string material_positions = "MESH_NODE_POSITIONS",
    const std::string blockset_name = "DISPLACEMENT"

  ):
  DirichletDisplacementBc(m_field,field_name,aij,x,f),
  materialPositions(material_positions),
  blocksetName(blockset_name) {
  }

  DirichletSpatialPositionsBc(
    MoFEM::Interface& m_field,
    const std::string &field_name,
    const std::string material_positions = "MESH_NODE_POSITIONS",
    const std::string blockset_name = "DISPLACEMENT"
  ):
  DirichletDisplacementBc(m_field,field_name),
  materialPositions(material_positions),
  blocksetName(blockset_name) {
  }

  std::string materialPositions;        ///< name of the field with reference material positions
  std::vector<std::string> fixFields;   ///<
  const std::string blocksetName;

  VectorDouble cOords;
  MoFEMErrorCode iNitalize();

};

/// \deprecated use DirichletSpatialPositionsBc
DEPRECATED typedef DirichletSpatialPositionsBc SpatialPositionsBCFEMethodPreAndPostProc;

struct DirichletTemperatureBc: public DirichletDisplacementBc {

  DirichletTemperatureBc(
    MoFEM::Interface& m_field,const std::string &field_name,Mat aij,Vec x,Vec f):
    DirichletDisplacementBc(m_field,field_name,aij,x,f) {}

  DirichletTemperatureBc(
    MoFEM::Interface& m_field,const std::string &field_name):
    DirichletDisplacementBc(m_field,field_name) {}

  MoFEMErrorCode iNitalize();

};

/// \deprecated use DirichletTemperatureBc
DEPRECATED typedef DirichletTemperatureBc TemperatureBCFEMethodPreAndPostProc;

/** \brief Fix dofs on entities
  * \ingroup Dirichlet_bc
  */
struct DirichletFixFieldAtEntitiesBc: public DirichletDisplacementBc {

  Range eNts;
  std::vector<std::string> fieldNames;
  DirichletFixFieldAtEntitiesBc(
    MoFEM::Interface& m_field,
    const std::string &field_name,
    Mat aij,Vec x,Vec f,
    Range &ents
  ):
  DirichletDisplacementBc(m_field,field_name,aij,x,f),
  eNts(ents) {
    fieldNames.push_back(fieldName);
  }

  DirichletFixFieldAtEntitiesBc(
    MoFEM::Interface& m_field,const std::string &field_name,Range &ents
  ):
  DirichletDisplacementBc(m_field,field_name),eNts(ents) {
    fieldNames.push_back(fieldName);
  }

  MoFEMErrorCode iNitalize();
  MoFEMErrorCode preProcess();
  MoFEMErrorCode postProcess();

};

/// \deprecated use DirichletFixFieldAtEntitiesBc
DEPRECATED typedef DirichletFixFieldAtEntitiesBc FixBcAtEntities;

/** \brief Blockset boundary conditions
  * \ingroup Dirichlet_bc
  *
  * Implementation of generalized Dirichlet Boundary Conditions from CUBIT Blockset
  * (or not using CUBIT building boundary conditions, e.g. Temperature or Displacements etc).
  * It can work for any Problem rank (1,2,3)
  *
  *    Usage in Cubit for displacement:
       block 1 surface 12
       block 1 name "DISPLACEMENT_1"
       block 1 attribute count 3
       block 1 attribute index 1 0       # value for x direction
       block 1 attribute index 2 2       # value for y direction
       block 1 attribute index 3 0       # value for z direction

      With above command we set displacement of 2 on y-direction and constrain x,z direction (0 displacement)
  *
**/
struct DirichletSetFieldFromBlock: public DirichletDisplacementBc {

  const std::string blocksetName;
  DirichletSetFieldFromBlock(
    MoFEM::Interface& m_field,
    const std::string &field_name,
    const std::string &blockset_name,
    Mat aij,Vec x,Vec f
  ):
  DirichletDisplacementBc(m_field,field_name,aij,x,f),
  blocksetName(blockset_name) {
  }

  DirichletSetFieldFromBlock(
    MoFEM::Interface& m_field,const std::string &field_name,const std::string &blockset_name
  ):
  DirichletDisplacementBc(m_field,field_name),
  blocksetName(blockset_name) {
  }

  MoFEMErrorCode iNitalize();

};

/// \deprecated use DirichletSetFieldFromBlock
DEPRECATED typedef DirichletSetFieldFromBlock DirichletBCFromBlockSetFEMethodPreAndPostProc;

/**
 * \brief Add boundary conditions form block set having 6 attributes
 *
 * First 3 values are magnitudes of dofs e.g. in x,y,z direction and next 3 are flags, respectively.
 * If flag is false ( = 0), particular dof is not taken into account.
    Usage in Cubit for displacement:
     block 1 tri 28 32
     block 1 name "DISPLACEMENT_1"
     block 1 attribute count 6
     block 1 attribute index 1 97    # any value (Cubit doesnt allow for blank attributes)
     block 1 attribute index 2 0
     block 1 attribute index 3 0
     block 1 attribute index 4 0       # flag for x direction
     block 1 attribute index 5 1       # flag for y direction
     block 1 attribute index 6 1       # flag for z direction
    This means that we set zero displacement on y and z direction and on x set direction freely.
    (value 97 is irrelevant because flag for 1 value is 0 (false))
    It can be usefull if we want to set boundary conditions directly to triangles e.g,
    since standard boundary conditions in Cubit allow only using nodeset or surface
    which might not work with mesh based on facet engine (e.g. STL file)
 */
struct DirichletSetFieldFromBlockWithFlags: public DirichletDisplacementBc {

  const std::string blocksetName;
  DirichletSetFieldFromBlockWithFlags(
    MoFEM::Interface& m_field,const std::string &field_name,const std::string &blockset_name,Mat aij,Vec x,Vec f
  ):
  DirichletDisplacementBc(m_field,field_name,aij,x,f),
  blocksetName(blockset_name) {
  }

  DirichletSetFieldFromBlockWithFlags(
    MoFEM::Interface& m_field,const std::string &field_name,const std::string &blockset_name
  ):
  DirichletDisplacementBc(m_field,field_name),
  blocksetName(blockset_name) {
  }

  MoFEMErrorCode iNitalize();

};

/// \deprecated use DirichletSetFieldFromBlockWithFlags
DEPRECATED typedef DirichletSetFieldFromBlockWithFlags DirichletBCFromBlockSetFEMethodPreAndPostProcWithFlags;

/**
 * @brief calculate reactions from vector of internal forces on a given meshset id
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

      Reactions my_react(m_field, "DM_ELASTIC", "U", F_int);
      my_react.calculateReactions();
      int fix_nodes_meshset_id = 1;
      cout << my_react.getReactionsFromSet(fix_nodes_meshset_id) << endl;

* \endcode
 */
struct Reactions {

  Reactions(MoFEM::Interface &m_field, string problem_name, string field_name,
            Vec &f_internal)
      : mField(m_field), problemName(problem_name), fieldName(field_name),
        fInternal(f_internal) {}

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
   * @return MoFEMErrorCode
   */
  MoFEMErrorCode calculateReactions();

private:
  std::string problemName;
  std::string fieldName;
  ReactionsMap reactionsMap;
  Vec fInternal;
};

#endif //__DIRICHLET_HPP__

/***************************************************************************//**
 * \defgroup Dirichlet_bc Dirichlet boundary conditions
 * \ingroup user_modules
 ******************************************************************************/
