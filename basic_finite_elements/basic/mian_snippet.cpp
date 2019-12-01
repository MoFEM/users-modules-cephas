/**
 * \file main_snippet.cpp
 * \example main_snippet.cpp
 *
 * Using Basic interface calculate the divergence of base functions, and
 * integral of flux on the boundary. Since the h-div space is used, volume
 * integral and boundary integral should give the same result.
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

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    // Create MoAB
    moab::Core mb_instance;              ///< database
    moab::Interface &moab = mb_instance; ///< interface

    // Create MoFEM
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    Simple *simple_interface = m_field.getInterface<Simple>();
    CHKERR simple_interface->getOptions();
    CHKERR simple_interface->loadFile("");

    CHKERR simple_interface->addDomainField("FIELD", H1,
                                             AINSWORTH_LEGENDRE_BASE, 1);
    CHKERR simple_interface->addBoundaryField("FIELD", H1,
                                              AINSWORTH_LEGENDRE_BASE, 1);

    constexpr int order = 2;
    CHKERR simple_interface->setFieldOrder("FIELD", order);
    CHKERR simple_interface->setUp();

    Basic *basic_interface = m_field.getInterface<Basic>();

    auto integration_rule = [](int, int, int p_data) { return 2 * p_data; };
    CHKERR basic_interface->setDomainRhsIntegrationRule(integration_rule);
    CHKERR basic_interface->setBoundaryRhsIntegrationRule(integration_rule);

  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}

