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
    else if (platform.jenkinsLabel.contains('ubuntu')) {
        enableAudioTesting = 'sudo apt-get install -y libsndfile1-dev'
        enableVoxelTesting = '(git clone https://github.com/NIFTI-Imaging/nifti_clib.git; cd nifti_clib; mkdir build; cd build; cmake ../; sudo make -j$nproc install)'
        if (platform.jenkinsLabel.contains('ubuntu20')) {
            backend = 'OCL'
        }
    }
    else if (platform.jenkinsLabel.contains('rhel')) {
        enableAudioTesting = 'sudo yum install -y libsndfile-devel'
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

    String packageType = ''
    String packageInfo = ''
    String packageDetail = ''
    String osType = ''
    String packageRunTime = ''

    if (platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('rhel') || platform.jenkinsLabel.contains('sles')) {
        packageType = 'rpm'
        packageInfo = 'rpm -qlp'
        packageDetail = 'rpm -qi'
        packageRunTime = 'rpp-*'

        if (platform.jenkinsLabel.contains('sles')) {
            osType = 'sles'
        }
        else if (platform.jenkinsLabel.contains('centos7')) {
            osType = 'centos7'
        }
        else if (platform.jenkinsLabel.contains('rhel8')) {
            osType = 'rhel8'
        }
        else if (platform.jenkinsLabel.contains('rhel9')) {
            osType = 'rhel9'
        }
    }
    else
    {
        packageType = 'deb'
        packageInfo = 'dpkg -c'
        packageDetail = 'dpkg -I'
        packageRunTime = 'rpp_*'

        if (platform.jenkinsLabel.contains('ubuntu20')) {
            osType = 'ubuntu20'
        }
        else if (platform.jenkinsLabel.contains('ubuntu22')) {
            osType = 'ubuntu22'
        }
    }

    def command = """#!/usr/bin/env bash
                set -x
                export HOME=/home/jenkins
                echo Make RPP Package
                cd ${project.paths.project_build_prefix}/build/release
                sudo make package
                mkdir -p package
                mv rpp-dev*.${packageType} package/${osType}-rpp-dev.${packageType}
                mv ${packageRunTime}.${packageType} package/${osType}-rpp.${packageType}
                mv Testing/Temporary/LastTest.log ${osType}-LastTest.log
                mv Testing/Temporary/LastTestsFailed.log ${osType}-LastTestsFailed.log
                ${packageDetail} package/${osType}-rpp-dev.${packageType}
                ${packageDetail} package/${osType}-rpp.${packageType}
                ${packageInfo} package/${osType}-rpp-dev.${packageType}
                ${packageInfo} package/${osType}-rpp.${packageType}
                """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, packageHelper[1])
}


return this
