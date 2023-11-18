// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean staticLibrary=false) {
    project.paths.construct_build_prefix()

    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'
    String backend = 'HIP'
    String enableSCL = 'echo build-rpp'
    String enableAudioTesting = 'echo audio-tests-not-supported'
    String enableVoxelTesting = 'echo voxel-tests-not-supported'

    if (platform.jenkinsLabel.contains('centos')) {
        backend = 'CPU'
        enableSCL = 'source scl_source enable llvm-toolset-7'
    }

    if (platform.jenkinsLabel.contains('ubuntu')) {
        enableAudioTesting = 'sudo apt-get install libsndfile1-dev'
        enableVoxelTesting = '(git clone https://github.com/NIFTI-Imaging/nifti_clib.git; cd nifti_clib; mkdir build; cd build; cmake ../; sudo make -j$nproc install)'
        if (platform.jenkinsLabel.contains('ubuntu20')) {
            backend = 'OCL'
        }
    }
    else if (platform.jenkinsLabel.contains('rhel')) {
        enableAudioTesting = 'sudo yum install libsndfile-devel'
        enableVoxelTesting = '(git clone https://github.com/NIFTI-Imaging/nifti_clib.git; cd nifti_clib; mkdir build; cd build; cmake ../; sudo make -j$nproc install)'
    }
    

    def command = """#!/usr/bin/env bash
                set -x
                wget https://sourceforge.net/projects/half/files/half/1.12.0/half-1.12.0.zip
                unzip half-1.12.0.zip -d half-files
                sudo mkdir -p /usr/local/include/half
                sudo cp half-files/include/half.hpp /usr/local/include/half
                echo Build RPP - ${buildTypeDir}
                cd ${project.paths.project_build_prefix}
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                ${enableSCL}
                ${enableAudioTesting}
                ${enableVoxelTesting}
                cmake -DBACKEND=${backend} ${buildTypeArg} ../..
                make -j\$(nproc)
                sudo make install
                make test ARGS="-VV"
                sudo make package
                """

    platform.runCommand(this, command)
}

def runTestCommand (platform, project) {

    def command = """#!/usr/bin/env bash
                set -x
                ldd -v /opt/rocm/lib/librpp.so
                """

    platform.runCommand(this, command)
// Unit tests - TBD
}

def runPackageCommand(platform, project) {

    def packageHelper = platform.makePackage(platform.jenkinsLabel, "${project.paths.project_build_prefix}/build/release")

    String packageType = ""
    String packageInfo = ""

    if (platform.jenkinsLabel.contains('centos') ||
        platform.jenkinsLabel.contains('rhel') ||
        platform.jenkinsLabel.contains('sles')) {
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
                mv rpp*.${packageType} package/
                ${packageInfo} package/rpp-dev*.${packageType}
                ${packageInfo} package/rpp_*.${packageType}
                """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, packageHelper[1])
}

return this
