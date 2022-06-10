#include <boost/filesystem.hpp>

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
    : path(boost::filesystem::temp_directory_path() /
           boost::filesystem::unique_path("rpp-" + prefix + "-%%%%-%%%%-%%%%-%%%%"))
{
    boost::filesystem::create_directories(this->path);
}

void TmpDir::Execute(std::string exe, std::string args)
{
    // std::cout<<"Invoking Execute routine\n";
    std::string cd  = "cd " + this->path.string() + "; ";
    std::string cmd = cd + exe + " " + args; // + " > /dev/null";
    SystemCmd(cmd);
    // std::cout<<"Done with Execute routine"<<cmd.c_str()<<std::endl;
}

TmpDir::~TmpDir() { boost::filesystem::remove_all(this->path); }

} // namespace rpp
