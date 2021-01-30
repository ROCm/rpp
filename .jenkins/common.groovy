// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean staticLibrary=false) {
    project.paths.construct_build_prefix()

    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'

    String installPackage = ""
    String osInfo = ""
    String cmake = ""
    String centos7 = ""
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
            update = 'sudo zypper -y update'
            cmake = 'cmake'
            installPackage = 'sudo zypper install -y boost-devel clang'
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

def runTestCommand (platform, project) {

    def command = """#!/usr/bin/env bash
                set -x
                ldd -v /opt/rocm/rpp/lib/libamd_rpp.so
                """

    platform.runCommand(this, command)
    // Unit tests - TBD
}

def runPackageCommand(platform, project) {

    def packageHelper = platform.makePackage(platform.jenkinsLabel, "${project.paths.project_build_prefix}/build/release")

    String packageType = ""
    String packageInfo = ""

    if (platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('sles'))
    {
        packageType = 'rpm'
        packageInfo = 'rpm -qlp'
    }
    else
    {
        packageType = 'deb'
        packageInfo = 'dpkg -c'
    }

    def command = """#!/usr/bin/env bash
                set -x
                export HOME=/home/jenkins
                echo Make RPP Package
                cd ${project.paths.project_build_prefix}/build/release
                sudo make package
                mkdir -p package
                mv *.${packageType} package/
                ${packageInfo} package/*.${packageType}
                """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, packageHelper[1])
}

return this
