import Support_Code as sc
import subprocess
import csv
import shlex, subprocess
import os

def process_csv(csv_reader,numline):
    shell_text = ""
    shell_script = open(sc.code_folder + "shell.sh" ,"a+")
    for count,row in enumerate(csv_reader):
        funcName = row[2] 
        writeFuncName = funcName
        if (row[0] == '3'):
            writeFuncName = writeFuncName + "_pkd3"
        else:
            writeFuncName = writeFuncName + "_pln1"
        if(row[1] == '0'):
            writeFuncName = writeFuncName + "_host"
        elif(row[1] == '1'):
            writeFuncName = writeFuncName + "_ocl"
        else:
            writeFuncName = writeFuncName + "_hip"
        os.mkdir(sc.code_folder + writeFuncName)
        os.mkdir(sc.code_folder + writeFuncName + "/build")
        cm = open(sc.code_folder + writeFuncName + "/CMakeLists.txt", "a+")
        cmake = sc.cmake
        if(row[1] == '0'):
            cmake = cmake + "\n\n" + """set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOCL_COMPILE=1 -DRPP_BACKEND_OPENCL=1 ")"""
        elif(row[1] == '1'):
            cmake = cmake + "\n\n" + """set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOCL_COMPILE=1 -DRPP_BACKEND_OPENCL=1 ")"""
        elif(row[1] == '2'):
            cmake = cmake + "\n\n" + """set(COMPILER_FOR_HIP /opt/rocm/bin/hipcc)"""
            cmake = cmake + "\n\n" + """set(CMAKE_CXX_COMPILER ${COMPILER_FOR_HIP})"""
            cmake = cmake + "\n\n" + """set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHIP_COMPILE=1 -DRPP_BACKEND_HIP=1 -fopenmp -std=c++14") """
        shell_text = shell_text + "cd\ncd " + sc.code_folder + writeFuncName + "/build\ncmake ..\nmake\n"
        row_len = len(row)
        srcD1 = ""
        srcD2 = ""
        srcS1 = ""
        srcS2 = ""
        if(row[3] == '2'):
            srcD1 = srcD1 + row[row_len - 5]
            srcD2 = srcD2 + row[row_len - 4]
            srcS1 = srcS1 + row[row_len - 3]
            srcS2 = srcS2 + row[row_len - 2]
        else:
            srcD1 = srcD1 + row[row_len - 3]
            srcS1 = srcS1 + row[row_len - 2]
        dst = row[row_len - 1]
        # FILE
        for file_name in sc.files:
            if(funcName in sc.non_roi_functions) and ("ROI" in file_name):
                continue
            code = ""
            code = code + sc.header
            if(row[1]) != '2':
                code=code+"#include <CL/cl.hpp>\n"
            if(row[1] == '2'):
                code = code + '#include "hip/hip_runtime_api.h"\n'
                code = code + sc.hip_stater
            code = code + "int G_IP_CHANNEL = " + row[0] + ";\n"
            code = code + "int G_MODE = " + row[1] + ";\n"
            if ("BatchD" in file_name) or ("BatchP" in file_name):
                code = code + """char src[1000] = {"RELACECHAR"};\n"""
                code = code.replace("RELACECHAR",srcD1)
                if(row[3] == '2'):
                    code = code + """char src_second[1000] = {"RELACECHAR"};\n """
                    code = code.replace("RELACECHAR",srcD2)
            else:
                code = code + """char src[1000] = {"RELACECHAR"};\n"""
                code = code.replace("RELACECHAR",srcS1)
                if(row[3] == '2'):
                    code = code + """char src_second[1000] = {"RELACECHAR"};\n """
                    code = code.replace("RELACECHAR",srcS2)
            code = code + """char dst[1000] = {"RELACECHAR"};\n """
            code = code.replace("RELACECHAR",dst)
            code = code + """char funcName[1000] = {"RELACECHAR"};\n"""
            code = code.replace("RELACECHAR",funcName)
            code = code + """char funcType[1000] = {"RELACECHAR"};\n"""
            code = code.replace("RELACECHAR",file_name)
            code = code + sc.main
            if (row[3] == '2'):
                if("Single" in file_name) or ("ROI" == file_name):
                    code = code + sc.double_image_2
                else:
                    code = code + sc.double_image_1
            else:
                if("Single" in file_name) or ("ROI" == file_name):
                    code = code + sc.single_image_2
                else:
                    code = code + sc.single_image_1
            if("ROID" in file_name):
                code = code + sc.roi_buffer
            if ("BatchP" in file_name):
                code = code + sc.padding_ioBuffer
            else:
                code = code + sc.non_padding_ioBuffer
                if("ROIS" in file_name):
                    code += sc.rois_patch
            if ("ROID" in file_name):
                if ("ROID_C" in file_name):
                    code = code + sc.center_roi
                else:
                    code = code + sc.different_roi
            if ("Single" == file_name) or ("ROI" == file_name):
                code = code + sc.single1
                if(row[3] == '2'):
                    code = code + sc.single3
                else:
                    code = code + sc.single2
            elif ("BatchP" in file_name):
                code = code + sc.padding_io_ending
                if(row[3] == '2'):
                    code = code + sc.double_image_padding
                else:
                    code = code + sc.single_image_padding
            else:
                code = code + sc.non_padding_io_ending
                if(row[3] == '2'):
                    code = code + sc.double_image_non_padding
                else:
                    code = code + sc.single_image_non_padding
            if ("ROIS" in file_name):
                if ("ROIS_C" in file_name):
                    code = code + sc.same_center_roi
                else:
                    code = code + sc.same_roi
            if ("ROI" == file_name):
                code = code + sc.roi_patch
                code = code + sc.same_roi
            if (row[4] != '0'):
                i = 5
                for x in range(int(row[4])):
                    if ("BatchDD" in file_name) or ("BatchSD" in file_name) or ("BatchPD" in file_name):
                        code = code + '\t' + row[i] + ' min' + row[i+1] + ' = ' + row[i+2] + ', max' + row[i+1] + ' = ' + row[i+3] + ', ' + row[i+1] + '[images];\n'
                        i = i + 4 
                    else:
                        code = code + '\t' + row[i] + ' min' + row[i+1] + ' = ' + row[i+2] + ', max' + row[i+1] + ' = ' + row[i+3] + ', ' + row[i+1] + ';\n'
                        i = i + 4 
                i = 5
                if ("BatchDD" in file_name) or ("BatchSD" in file_name) or ("BatchPD" in file_name):
                    code = code + '\t' + 'for(i = 0 ; i < images ; i++)\n\t{\n'
                    for x in range(int(row[4])):
                        # if (int(row[i+3]) - int(row[i+2]) != 0):
                        if (float(row[i+3]) - float(row[i+2]) > 0.5):
                            if(row[i+1] == "kernelSize"):
                                code = code + '\t\t' + row[i+1] + '[i] = ((max' + row[i+1] + ' - min' + row[i+1] + ') / images) * i + min' + row[i+1] + ';\n'
                                code = code + '\t\t' + row[i+1] + '[i] -= (' + row[i+1] + '[i] % 2);\n'
                            else:
                                code = code + '\t\t' + row[i+1] + '[i] = ((max' + row[i+1] + ' - min' + row[i+1] + ') / images) * i + min' + row[i+1] + ';\n' 
                        else:
                            if(row[i+1] == "kernelSize"):
                                code = code + '\t\t' + row[i+1] + '[i] = min' + row[i+1] + ';\n' 
                                code = code + '\t\t' + row[i+1] + '[i] -= (' + row[i+1] + '[i] % 2);\n'
                            else:
                                code = code + '\t\t' + row[i+1] + '[i] = min' + row[i+1] + ';\n' 
                        i = i + 4
                    code = code + '\t}\n'
                else:
                    for x in range(int(row[4])):
                        # if (int(row[i+3]) - int(row[i+2]) != 0):
                        if (float(row[i+3]) - float(row[i+2]) > 0.5):
                            if(row[i+1] == "kernelSize"):
                                code = code + '\t' + row[i+1] + ' = ('+ row[i] +') ((rand() % (int) (max' + row[i+1] + ' - min' + row[i+1] + ')) + min' + row[i+1] + ');\n' 
                                code = code + '\t' + row[i+1] + ' -= (' + row[i+1] + ' % 2);\n'
                            else:
                                code = code + '\t' + row[i+1] + ' = ('+ row[i] +') ((rand() % (int) (max' + row[i+1] + ' - min' + row[i+1] + ')) + min' + row[i+1] + ');\n' 
                        else:
                            if(row[i+1] == "kernelSize"):
                                code = code + '\t' + row[i+1] + ' = min' + row[i+1] + ';\n'
                                code = code + '\t' + row[i+1] + ' -= (' + row[i+1] + ' % 2);\n'
                            else:
                                code = code + '\t' + row[i+1] + ' = min' + row[i+1] + ';\n' 
                        i = i + 4
            if (row[1] == '1'):
                if (row[3] == '1'):
                    code = code + sc.ocl_single
                else:
                    code = code + sc.ocl_double
            elif (row[1] == '2'):
                if (row[3] == '1'):
                    code = code + sc.hip_single
                else:
                    code = code + sc.hip_double
            else:
                code = code + sc.host_timing
            code = code + "\trppi_" + funcName + "_u8_"
            if(row[0] == '3'):
                code = code + "pkd3"
            else :
                code = code + "pln1"
            if ("Batch" in file_name):
                code = code + "_batch"
                if ("BatchSD" in file_name):
                    code = code + "SD"
                elif ("BatchSS" in file_name):
                    code = code + "SS"
                elif ("BatchDD" in file_name):
                    code = code + "DD"
                elif ("BatchDS" in file_name):
                    code = code + "DS"
                elif ("BatchPD" in file_name):
                    code = code + "PD"
                elif ("BatchPS" in file_name):
                    code = code + "PS"
            if ("ROI" in file_name):
                code = code + "_ROI"
                if ("ROIS" in file_name):
                    code = code + "S"
                elif ("ROID" in file_name):
                    code = code + "D"
            if (row[1] == '0'):
                code = code + "_host(input, "
                if(row[3] == '2'):
                    code = code + "input_second, "
            else:
                code = code + "_gpu(d_input, "
                if(row[3] == '2'):
                    code = code + "d_input_second, "
            if ("BatchD" in file_name):
                code = code + "srcSize, "
            elif ("BatchP" in file_name):
                code = code + "srcSize, maxSize, "
            else:
                code = code + "srcSize[0], "
            if (row[1] == '0'):
                code = code + "output, "
            else:
                code = code + "d_output, "
            if (row[4] != '0'):
                i = 5
                for x in range(int(row[4])):    
                    code = code + row[i+1] + ', ' 
                    i = i + 4
            if ("ROI" in file_name):
                code = code + "roiPoints, "
            if ("Batch" in file_name):
                code = code + "noOfImages, "
            code = code + "handle);\n"
            code = code + sc.timing_end_1 + file_name + sc.timing_end_2
            if (row[1] == '0'):
                code = code + sc.host_copy
            elif (row[1] == '1'):
                code = code + sc.ocl_copy
            else:
                code = code + sc.hip_copy
            if("BatchP" in file_name):
                code = code + sc.write_image_padding
            else:
                code = code + sc.write_image
            code = code + sc.free_mem
            if (row[3] == '2'):
                code = code + "\tfree(input_second);\n"
            code = code + "\t return 0; \n}"
            fileDir = "/" + file_name + ".cpp"
            fileptr = open(sc.code_folder + writeFuncName + fileDir,"a+")
            fileptr.write(code)
            shell_text = shell_text + "./" + file_name + "\n"
            index_cmake = cmake.index('set(')
            cmake = cmake[:index_cmake] + sc.IP_cmake_1 + file_name + " " + file_name + ".cpp" + sc.IP_cmake_2 + cmake[index_cmake:]
            if(row[1] == '2'):
                cmake = cmake + sc.IP_cmake_3_hip + file_name + sc.IP_cmake_4_hip
            else:
                cmake = cmake + sc.IP_cmake_3 + file_name + sc.IP_cmake_4        
        cm.write(cmake)
    shell_script.write(shell_text)
    s = subprocess.Popen("cd " + sc.code_folder + ";chmod +x shell.sh;./shell.sh",stdin = None, stdout = None, stderr = None, close_fds = True, shell = True)

with open(sc.csv_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    f = open(sc.csv_name)
    numline = len(f.readlines())
    print (numline)
    process_csv(csv_reader,numline)
