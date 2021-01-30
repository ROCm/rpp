// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean staticLibrary=false) {
    project.paths.construct_build_prefix()
        
    project.paths.build_command = './install -c'

    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'

    String installPackage = ""
    String osInfo = ""
    String cmake = ""
    String centos7 = ""
    String sles = ""
    String update = ""

    if (platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('sles'))
    {
        osInfo = 'cat /etc/os-release && uname -r'
        update = 'sudo yum -y update'
        installPackage = 'sudo yum install -y boost-devel clang'
        cmake = 'cmake3'
        if (platform.jenkinsLabel.contains('centos7'))
        {
          centos7 = 'scl enable devtoolset-7 bash'
        }
        if (platform.jenkinsLabel.contains('sles'))
        {
          sles = 'yum install -y sudo'
        }
    }
    else
    {
        osInfo = 'cat /etc/lsb-release && uname -r'
        update = 'sudo apt -y update'
        installPackage = 'sudo apt install -y unzip cmake libboost-all-dev clang'
        cmake = 'cmake'
    }

    def command = """#!/usr/bin/env bash
                set -x
                ${osInfo}
                ${sles}
                ${update}
                ${centos7}
                echo Install RPP Prerequisites
                mkdir -p rpp-deps && cd rpp-deps
                ${installPackage}
                wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip
                unzip half-1.12.0.zip -d half-files
                sudo cp half-files/include/half.hpp /usr/local/include/
                cd ../
                echo Build RPP - ${buildTypeDir}
                cd ${project.paths.project_build_prefix}
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                ${cmake} -DBACKEND=OCL ${buildTypeArg} ../..
                make -j\$(nproc)
                sudo make install
                sudo make package
                """
    
    platform.runCommand(this, command)
}

/*@Override
def runTestCommand (platform, project) {
//TBD
}*/

def runPackageCommand(platform, project) {
    def packageHelper = platform.makePackage(platform.jenkinsLabel, "${project.paths.project_build_prefix}/build/release")
        
    platform.runCommand(this, packageHelper[0])
    platform.archiveArtifacts(this, packageHelper[1])
}

return this
