#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <iostream>
#include <experimental/filesystem>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <algorithm>
#include <sstream>

using namespace std;
namespace fs = std::experimental::filesystem;

int main()
{
    string folderPath = "/dockerx/rpp/utilities/test_suite/HOST/temp/";

    // find all subfolders
    vector<string> entryList;
    string fullPath = folderPath;
    auto subDir = opendir(folderPath.c_str());
    if (!subDir)
    {
        std::cerr << "ERROR: Failed opening the directory at " << folderPath << std::endl;
        exit(0);
    }

    struct dirent* entity;
    while ((entity = readdir(subDir)) != nullptr)
    {
        string entryName(entity->d_name);
        if (entryName == "." || entryName == "..")
            continue;
        entryList.push_back(entryName);
    }
    closedir(subDir);

    for (unsigned dirCount = 0; dirCount < entryList.size(); ++dirCount)
    {
        string subFolderPath = folderPath + entryList[dirCount];
        std::vector<string> imageNamesPath, imageNames;
        auto srcDir = opendir(subFolderPath.c_str());
        struct dirent* entity;
        std::string fileName = " ";
        if (srcDir == nullptr)
            std::cerr << "\n ERROR: Failed opening the directory at " << subFolderPath;

        std::vector<float> outBuf; // Use float instead of unsigned char
        std::vector<string> entryNames;
        while ((entity = readdir(srcDir)) != nullptr)
        {
            string entryName(entity->d_name);
            entryNames.push_back(entryName);
        }
        std::sort(entryNames.begin(), entryNames.end());

        for (int i = 0; i < entryNames.size(); i++)
        {
            if (entryNames[i] == "." || entryNames[i] == "..")
                continue;
            fileName = entryNames[i];
            std::string filePath = subFolderPath;
            filePath.append("/");
            filePath.append(entryNames[i]);
            cout << "filePath: " << filePath << endl;

            ifstream file(filePath);
            string line, word;
            int index = 0;

            // Load the reference output values from files and store in vector
            if (file.is_open())
            {
                while (getline(file, line))
                {
                    stringstream str(line);
                    while (getline(str, word, ','))
                    {
                        outBuf.push_back(stof(word)); // Store floats instead of unsigned char
                        index++;
                    }
                }
            }
            else
            {
                cout << "Could not open the reference output. Please check the path specified\n";
                return 0;
            }

            imageNamesPath.push_back(filePath);
            imageNames.push_back(entryNames[i]);
        }

        if (imageNames.empty())
            std::cerr << "\n Did not load any file from " << subFolderPath;

        // Write the output buffer as floats to a binary file
        std::ofstream refFile;
        refFile.open(entryList[dirCount] + ".bin", std::ios::out | std::ios::binary);

        for (size_t i = 0; i < outBuf.size(); i++)
            refFile.write(reinterpret_cast<char*>(&outBuf[i]), sizeof(float));
        refFile.close();
        outBuf.clear();
    }

    // Read from binary file and write the float values to a CSV file
    // std::fstream refFile;
    // refFile.open("concat.bin", std::ios::in | std::ios::binary);

    // std::ofstream outFile;
    // outFile.open("1d_output.csv");
    // float ch;
    // while (refFile.read(reinterpret_cast<char*>(&ch), sizeof(float)))
    //     outFile << static_cast<float>(ch) << ",";
    // outFile.close();

    return 0;
}
