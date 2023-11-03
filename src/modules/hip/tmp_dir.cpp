#include <chrono>
#include "filesystem.h"
#include "rpp/tmp_dir.hpp"
#include "rpp/errors.hpp"
#include "rpp/logger.hpp"

namespace rpp {

void SystemCmd(std::string cmd)
{
#ifndef NDEBUG
    RPP_LOG_I(cmd);
#endif
// We shouldn't call system commands
#ifdef RPP_USE_CLANG_TIDY
    (void)cmd;
#else
    if(std::system(cmd.c_str()) != 0)
        RPP_THROW("Can't execute " + cmd);
#endif
}

TmpDir::TmpDir(std::string prefix)
    : path(fs::temp_directory_path() / ("rpp-" + prefix + "-" +
            std::to_string(std::chrono::system_clock::now().time_since_epoch().count())))
{
    fs::create_directories(this->path);
}

void TmpDir::Execute(std::string exe, std::string args)
{
    // std::cout<<"Invoking Execute routine\n";
    std::string cd  = "cd " + this->path.string() + "; ";
    std::string cmd = cd + exe + " " + args; // + " > /dev/null";
    SystemCmd(cmd);
    // std::cout<<"Done with Execute routine"<<cmd.c_str()<<std::endl;
}

TmpDir::~TmpDir() { fs::remove_all(this->path); }

} // namespace rpp
