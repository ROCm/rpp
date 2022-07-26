#ifndef GUARD_RPP_WRITE_FILE_HPP
#define GUARD_RPP_WRITE_FILE_HPP

#include <fstream>
#include <boost/filesystem.hpp>

#include "rpp/manage_ptr.hpp"

namespace rpp {

using FilePtr = RPP_MANAGE_PTR(FILE*, std::fclose);

inline void WriteFile(const std::string& content, const boost::filesystem::path& name)
{
    FilePtr f{std::fopen(name.string().c_str(), "w")};
    if(std::fwrite(content.c_str(), 1, content.size(), f.get()) != content.size())
        RPP_THROW("Failed to write to src file");
}
} // namespace rpp

#endif
