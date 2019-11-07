#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <EP_Operators.hpp>

using namespace MoFEM;
using namespace ElecPhys;

static char help[] = "...\n\n";

struct ElectroPhysioProblem {
public:
  ElectroPhysioProblem(){}


};

int main(int argc, char *argv[]) {
for(int i = 1; i < 20; ++i){
  if((i+3) % 4 == 0)
    cout << i << ", " << "u" << endl;
}

return 0;
}