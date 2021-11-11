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

#ifdef WITH_MODULE_MORTAR_CONTACT

namespace MMortarContactFunctions {

MoFEMErrorCode SetContactStructures(MoFEM::Interface &mField) {
  MoFEMFunctionBegin;

  MortarContactInterface contact(mField, "U", "MESH_NODE_POSITIONS");

  MoFEMFunctionReturn(0);
}
} // namespace MMortarContactFunctions

#endif

// namespace MMortarContactFunctions