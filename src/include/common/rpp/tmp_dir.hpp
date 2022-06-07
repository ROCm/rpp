#ifndef GUARD_RPP_TMP_DIR_HPP
#define GUARD_RPP_TMP_DIR_HPP

#include <string>
#include <boost/filesystem/path.hpp>

namespace rpp {

void SystemCmd(std::string cmd);

struct TmpDir
{
    boost::filesystem::path path;
    TmpDir(std::string prefix);

    TmpDir(TmpDir const&) = delete;
    TmpDir& operator=(TmpDir const&) = delete;

    void Execute(std::string exe, std::string args);

    ~TmpDir();
};

} // namespace rpp

#endif
