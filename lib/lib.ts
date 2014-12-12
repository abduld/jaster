

/// <reference path="memory/memory.ts" />
/// <reference path="cuda/exec/profile.ts" />

module lib {
    export interface StateInterface {
        type : string;
        model: typeof lib.cuda.exec.FermiArchitecture;
        globalMemory :lib.memory.GlobalMemoryManager;
        hostMemory :lib.memory.HostMemoryManager;
        id: string;
    }
    export function init() : StateInterface {
        return {
            type: "GlobalState",
            model: lib.cuda.exec.FermiArchitecture,
            globalMemory : new lib.memory.GlobalMemoryManager(),
            hostMemory: new lib.memory.HostMemoryManager(),
            id: lib.utils.guuid()
        }
    }

    export function chceckEvent(state, worker, functionStack) {
        return false;
    }
    export function handleEvent(state, worker, functionStack) {
        return false;
    }


    export interface CUDAReference {
        type: string;
        id: string;
    }
    export interface CReference {
        type: string;
        id: string;
    }
    export module cuda {
        export function cudaMalloc(state: StateInterface, ref : CUDAReference, byteCount : number, args : string[]) {

        }
    }

    export module c {
        export function malloc(state: StateInterface, byteCount : number, args : string[]) : CReference {
            return {
                type: "CReference",
                id: lib.utils.guuid()
            }
        }
    }
}