#include <rpp/load_file.hpp>
#include <sstream>
#include <fstream>

namespace rpp {

std::string LoadFile(const std::string& s)
{
    std::ifstream t(s);
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

} // namespace rpp
