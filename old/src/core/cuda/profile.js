/// <reference path="../../ref.ts" />
define(["require", "exports"], function(require, exports) {
    var KB = 1024;
    var M = KB * KB;

    var FermiArchitecture = (function () {
        function FermiArchitecture() {
            this.maxGridDimensions = 3;
            this.warpSize = 32;
            this.maxXGridDimension = Math.pow(2.0, 31.0) - 1;
            this.maxYGridDimension = Math.pow(2.0, 31.0) - 1;
            this.maxZGridDimension = 65535;
            this.maxBlockDimensions = 3;
            this.maxXBlockDimension = 1024;
            this.maxYBlockDimension = 1024;
            this.maxZBlockDimension = 64;
            this.maxThreadsPerBlock = 1024;
            this.numResigersPerThread = 64 * KB;
            this.maxResidentBlocksPerSM = 16;
            this.maxResidentWarpsPerSM = 64;
            this.maxSharedMemoryPerSM = 48 * KB;
            this.numSharedMemoryBanks = 32;
            this.localMemorySize = 512 * KB;
            this.constantMemorySize = 64 * KB;
            this.maxNumInstructions = 512 * M;
            this.numWarpSchedulers = 2;
        }
        return FermiArchitecture;
    })();
    exports.FermiArchitecture = FermiArchitecture;

    exports.ComputeCapabilityMap = undefined;

    if (exports.ComputeCapabilityMap !== undefined) {
        exports.ComputeCapabilityMap = new Map();
        exports.ComputeCapabilityMap[2.0] = new FermiArchitecture();
    }
});
//# sourceMappingURL=profile.js.map
