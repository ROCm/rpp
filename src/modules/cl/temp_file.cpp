#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem.hpp>

#include "rpp/temp_file.hpp"
#include "rpp/errors.hpp"

namespace rpp {
TempFile::TempFile(const std::string& path_template) : name(path_template), dir("tmp")
{
    if(!std::ofstream{this->Path(), std::ios_base::out | std::ios_base::in | std::ios_base::trunc}
            .good())
    {
        RPP_THROW("Failed to create temp file: " + this->Path());
    }
}

} // namespace rpp
