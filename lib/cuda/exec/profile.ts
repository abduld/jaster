
/// <reference path="../../ref.ts" />

module lib.cuda.exec {

    export interface GPUArchitecture {
        maxGridDimensions : number;
        warpSize : number;
        maxXGridDimension : number;
        maxYGridDimension : number;
        maxZGridDimension : number;
        maxBlockDimensions : number;
        maxXBlockDimension : number;
        maxYBlockDimension : number;
        maxZBlockDimension : number;
        maxThreadsPerBlock : number;
        numResigersPerThread : number;
        maxResidentBlocksPerSM : number;
        maxResidentWarpsPerSM : number;
        maxSharedMemoryPerSM : number;
        numSharedMemoryBanks : number;
        localMemorySize : number;
        constantMemorySize : number;
        maxNumInstructions : number;
        numWarpSchedulers : number;
    }

    var KB:number = 1024;
    var M:number = KB * KB;

    export class FermiArchitecture implements GPUArchitecture {
        maxGridDimensions:number = 3;
        warpSize:number = 32;
        maxXGridDimension:number = Math.pow(2.0, 31.0) - 1;
        maxYGridDimension:number = Math.pow(2.0, 31.0) - 1;
        maxZGridDimension:number = 65535;
        maxBlockDimensions:number = 3;
        maxXBlockDimension:number = 1024;
        maxYBlockDimension:number = 1024;
        maxZBlockDimension:number = 64;
        maxThreadsPerBlock:number = 1024;
        numResigersPerThread:number = 64 * KB;
        maxResidentBlocksPerSM:number = 16;
        maxResidentWarpsPerSM:number = 64;
        maxSharedMemoryPerSM:number = 48 * KB;
        numSharedMemoryBanks:number = 32;
        localMemorySize:number = 512 * KB;
        constantMemorySize:number = 64 * KB;
        maxNumInstructions:number = 512 * M;
        numWarpSchedulers:number = 2;
    }


    export var ComputeCapabilityMap:Map<number, GPUArchitecture> = undefined;

    if (ComputeCapabilityMap !== undefined) {
        ComputeCapabilityMap = new Map<number, GPUArchitecture>();
        ComputeCapabilityMap[2.0] = new FermiArchitecture();
    }
}