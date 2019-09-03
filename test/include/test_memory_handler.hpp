#ifndef RPP_TEST_MEMORY_HANDLER_H
#define RPP_TEST_MEMORY_HANDLER_H


#ifdef HIP_COMPILE

#include <hip/hip_runtime.h>

template <typename T>
void test_create_input(T inPtr; unsigned int rows, unsigned int cols, unsigned int chns)
{

}


#elif defined (OCL_COMPILE)


     // TO BE ADDED


#endif //backend




#endif //RPP_TEST_MEMORY_HANDLER_H