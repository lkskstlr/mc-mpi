// Specializes RmaComm<T> to needed types
// See following explanations:
// https://isocpp.org/wiki/faq/templates#separate-template-class-defn-from-decl
// https://stackoverflow.com/questions/115703/storing-c-template-function-definitions-in-a-cpp-file
#include "rma_comm.cpp"

template class RmaComm<int>;