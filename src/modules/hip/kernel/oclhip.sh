 find . -type f -name "*.cpp" -exec sed -i'' -e 's/\<__global\>//g' {} +
 find . -type f -name "*.cpp" -exec sed -i'' -e 's/get_global_id(0)/hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x/g' {} +
 find . -type f -name "*.cpp" -exec sed -i'' -e 's/get_global_id(1)/hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y/g' {} +
 find . -type f -name "*.cpp" -exec sed -i'' -e 's/get_global_id(2)/hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z/g' {} +
 find . -type f -name "*.cpp" -exec sed -i'' -e 's/__kernel/extern "C" __global__/g' {} +
 find . -type f -name "*.cpp" -exec sed -i'' -e 's/get_local_id(0)/hipThreadIdx_x/g' {} +
 find . -type f -name "*.cpp" -exec sed -i'' -e 's/get_local_id(1)/hipThreadIdx_y/g' {} +
 find . -type f -name "*.cpp" -exec sed -i'' -e 's/get_local_id(2)/hipThreadIdx_z/g' {} +
 find . -type f -name "*.cpp" -exec sed -i'' -e 's/get_group_id(0)/hipBlockIdx_x/g' {} +
 find . -type f -name "*.cpp" -exec sed -i'' -e 's/get_group_id(1)/hipBlockIdx_y/g' {} +
 find . -type f -name "*.cpp" -exec sed -i'' -e 's/get_group_id(2)/hipBlockIdx_z/g' {} +
 find . -type f -name "*.cpp" -exec sed -i'' -e 's/get_local_size(0)/hipBlockDim_x/g' {} +
 find . -type f -name "*.cpp" -exec sed -i'' -e 's/get_local_size(1)/hipBlockDim_y/g' {} +
 find . -type f -name "*.cpp" -exec sed -i'' -e 's/get_local_size(2)/hipBlockDim_z/g' {} +
 find . -type f -name "*.cpp" -exec sed -i'' -e 's/get_num_groups(0)/hipGridDim_x/g' {} +
 find . -type f -name "*.cpp" -exec sed -i'' -e 's/get_num_groups(1)/hipGridDim_y/g' {} +
 find . -type f -name "*.cpp" -exec sed -i'' -e 's/get_num_groups(2)/hipGridDim_z/g' {} +
