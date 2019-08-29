#ifndef RPPI_VALIDATE_OPERATIONS_FUNCTIONS
#define RPPI_VALIDATE_OPERATIONS_FUNCTIONS
#include <iostream>
#include <stdlib.h>
#include <rppdefs.h>


inline void validate_image_size(RppiSize imgSize){
    if(!(imgSize.width >= 0) || !(imgSize.height >= 0)){
       // std::cerr<<"\nImage width and height should be positive "<<std::endl;
        exit(0);
    }
}
inline void validate_float_range_b(Rpp32f min, Rpp32f max, Rpp32f *value) {
    if( !(*value <= max) || !(*value >= min)){
       // std::cerr<<"\nOut of bounds: "<<*value<<std::endl;
        //std::cerr<<"\nValue should be between "<<min<<" and "<<max<<std::endl;
        *value = max;
        //std::cerr<<"\nSetting the value to "<<max<<std::endl;
    }
}

inline void validate_float_range(Rpp32f min, Rpp32f max, Rpp32f *value) {
    if( !(*value <= max) || !(*value >= min)){
        //std::cerr<<"\nOut of bounds: "<<*value<<std::endl;
        //std::cerr<<"\nValue should be between "<<min<<" and "<<max<<std::endl;
        *value = max;
        //std::cerr<<"\nSetting the value to "<<max<<std::endl;
    }
}

inline void validate_double_range(Rpp64f min, Rpp64f max, Rpp64f *value) {
    if( !(*value <= max) || !(*value >= min)){
        //std::cerr<<"\nOut of bounds: "<<*value<<std::endl;
        //std::cerr<<"\nValue should be between "<<min<<" and "<<max<<std::endl;
        *value = max;
        //std::cerr<<"\nSetting the value to "<<max<<std::endl;
    }
}

inline void validate_int_range(Rpp32s min, Rpp32s max, Rpp32s *value) {
    if( !(*value <= max) || !(*value >= min)){
        //std::cerr<<"\nOut of bounds: "<<*value<<std::endl;
        //std::cerr<<"\nValue should be between "<<min<<" and "<<max<<std::endl;
        *value = max;
        //std::cerr<<"\nSetting the value to "<<max<<std::endl;
    }
}
inline void validate_unsigned_int_range(Rpp32u min, Rpp32u max, Rpp32u *value) {
    if( !(*value <= max) || !(*value >= min)){
        //std::cerr<<"\nOut of bounds: "<<*value<<std::endl;
        //std::cerr<<"\nValue should be between "<<min<<" and "<<max<<std::endl;
        *value = max;
        //std::cerr<<"\nSetting the value to "<<max<<std::endl;
    }
}

inline void validate_int_max(Rpp32s max, Rpp32s *value) {
    if( !(*value <= max) ){
       //std::cerr<<"\nOut of bounds: "<<*value<<std::endl;
       //std::cerr<<"\nValue should be less than "<<max<<std::endl;
       *value = max;
       //std::cerr<<"\nSetting the value to "<<max<<std::endl;
    }
}

inline void validate_unsigned_int_max(Rpp32u max, Rpp32u *value) {
    if( !(*value <= max) ){
       //std::cerr<<"\nOut of bounds: "<<*value<<std::endl;
       //std::cerr<<"\nValue should be less than "<<max<<std::endl;
       *value = max;
       //std::cerr<<"\nSetting the value to "<<max<<std::endl;
    }
}

inline void validate_int_min(Rpp32s min, Rpp32s *value) {
    if( !(*value >= min) ){
       //std::cerr<<"\nOut of bounds: "<<*value<<std::endl;
       //std::cerr<<"\nValue should be greater than "<<min<<std::endl;
       *value = min;
       //std::cerr<<"\nSetting the value to "<<min<<std::endl;
    }
}
inline void validate_unsigned_int_min(Rpp32u min, Rpp32u *value) {
    if( !(*value >= min) ){
       //std::cerr<<"\nOut of bounds: "<<*value<<std::endl;
       //std::cerr<<"\nValue should be greater than "<<min<<std::endl;
       *value = min;
       //std::cerr<<"\nSetting the value to "<<min<<std::endl;
    }
}
inline void validate_float_max(Rpp32f max, Rpp32f *value) {
    if( !(*value <= max) ){
       //std::cerr<<"\nOut of bounds: "<<*value<<std::endl;
       //std::cerr<<"\nValue should be less than "<<max<<std::endl;
       *value = max;
       //std::cerr<<"\nSetting the value to "<<max<<std::endl;
    }
}

inline void validate_float_min(Rpp32f min, Rpp32f *value) {
    if( !(*value >= min) ){
       //std::cerr<<"\nOut of bounds: "<<*value<<std::endl;
       //std::cerr<<"\nValue should be greater than "<<min<<std::endl;
       *value = min;
       //std::cerr<<"\nSetting the value to "<<min<<std::endl;
    }
}

inline void validate_affine_matrix(Rpp32f* affine){
    if((affine[0] * affine[4] - affine[1] * affine[3]) == 0){
        //std::cerr<<"\n Affine matrix is not valid--\n Identity matrix is considered instead"
                                    << std::endl;
        affine[0] = 1;
        affine[1] = 0;
        affine[3] = 0;
        affine[4] = 1;
    }
}
#endif