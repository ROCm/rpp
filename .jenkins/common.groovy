// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean staticLibrary=false) {
    project.paths.construct_build_prefix()
        
    project.paths.build_command = './install -c'
    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'
    String cmake = platform.jenkinsLabel.contains('centos') ? 'cmake3' : 'cmake'
    String packageInstaller = platform.jenkinsLabel.contains('centos') ? 'yum' : 'apt-get'
    String packages = platform.jenkinsLabel.contains('centos') ? 'boost-devel clang' : 'libboost-all-dev clang'
    String sudo = auxiliary.sudo(platform.jenkinsLabel)

    def command = """#!/usr/bin/env bash
                set -x
                mkdir -p rpp-deps && cd rpp-deps
                sudo ${packageInstaller} install -y ${packages}
                wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip
                unzip half-1.12.0.zip -d half-files
                sudo cp half-files/include/half.hpp /usr/local/include/
                cd ../
                cd ${project.paths.project_build_prefix}
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                ${cmake} -DBACKEND=OCL ${buildTypeArg} ../..
                make -j\$(nproc)
                make package
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
