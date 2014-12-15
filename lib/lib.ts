

/// <reference path="memory/memory.ts" />
/// <reference path="cuda/exec/profile.ts" />
/// <reference path="parallel/workerpool.ts" />
/// <reference path="wb/wb.ts" />

module lib {
    export interface StateInterface {
        type: string;
        mpnum : number;
        model: typeof lib.cuda.exec.FermiArchitecture;
        globalMemory: lib.memory.GlobalMemoryManager;
        hostMemory: lib.memory.HostMemoryManager;
        threadPool: typeof lib.parallel.WorkerPool;
        id: string;
    }
    export function init(mpnum : number): StateInterface {
        parallel.funsToProcess = [];
        parallel.funsMask = [];
        return {
            type: "GlobalState",
            mpnum : mpnum,
            model: lib.cuda.exec.FermiArchitecture,
            globalMemory: new lib.memory.GlobalMemoryManager(),
            hostMemory: new lib.memory.HostMemoryManager(),
            threadPool: lib.parallel.WorkerPool,
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
        mem: lib.c.memory.Reference;
        args: string[]
    }
    export interface CReference {
        type: string;
        id: string;
        stack?: any;
        mem: lib.c.memory.Reference;
        args: string[]
    }
    export module cuda {
        export function cudaMalloc(state: StateInterface, stack, ref: any, byteCount: number, args: string[]) {
            if (ref.type === "Identifier") {
                ref = {
                    type: "CUDAReference",
                    id: ref.id,
                    mem: state.globalMemory.malloc(byteCount),
                    args: args
                };
                stack[ref.id] = ref;
                return ref;
            }
            lib.utils.assert.ok(ref.mem.type === "CUDAReference");
          return {
                type: "CUDAReference",
                id: lib.utils.guuid(),
                mem: state.globalMemory.malloc(byteCount),
                args: args
            }
        }
        export function cudaMemcpy(state: StateInterface, stack, trg, src, size, direction) {
            switch (direction) {
                case "cudaMemcpyHostToDevice":
                case "cudaMemcpyDeviceToHost":
                    break;
                default:
                    console.log("Invalid direction " + direction);
            }
            var trgbuf = new Int8Array(trg.mem.data.buffer, 0, size);
            var srcbuf = new Int8Array(src.mem.data.buffer, 0, size);
            trgbuf.set(srcbuf);

        }
        export function cudaMemset(state: StateInterface, stack, trg, val, size) {
          var trgbuf = new Int8Array(trg.mem.data.buffer, 0, size);
          _.each(_.range(size), (idx : number) => trgbuf[idx] = val);
        }

        export function cudaThreadSynchronize(state, stack) {
            return lib.parallel.synchronize(state, stack);
        }


        export function cudaDeviceSynchronize(state, stack) {
          return cudaThreadSynchronize(state, stack);
        }

        export function getElement(state, stack, ref, idx) {
            return lib.memory.getElement(state, stack, ref, idx);
        }

        export function setElement(state, stack, ref, idx, val) {
            var res = lib.memory.setElement(state, stack, ref, idx, val);
            //console.log("Set element " + idx + " to " + val + " element[ " + idx + "] = " + getElement(state, stack, ref, idx));
        }
    }

    export module m {
      export function ceil(state, arg) {
        return Math.ceil(arg);
      }
      export function floor(state, arg) {
        return Math.floor(arg);
      }
    }
    export module c {

        export function getElement(state, stack, ref, idx) {
            return lib.memory.getElement(state, stack, ref, idx);
        }

        export function setElement(state, stack, ref, idx, val) {
            return lib.memory.setElement(state, stack, ref, idx, val);
        }
        export function malloc(state: StateInterface, stack, byteCount: number, args: string[]): CReference {
            return {
                type: "CReference",
                id: lib.utils.guuid(),
                mem: state.hostMemory.malloc(byteCount),
                args: args
            }
        }

        export function free(state: StateInterface, stack, ref) {
            if (_.isString(stack[ref.id])) {
                delete stack[ref.id];
            }
        }
    }

    export module parallel {
        export var funsToProcess: any[] = [];
        export var funsMask : boolean[] = [];
        export var finished: boolean = false;
        export function scheduleThread(state: StateInterface, fun: Function) {
          var len = funsMask.length;
          funsMask.push(false);
            funsToProcess.push(fun());
            return;
        }

        function sleepFor( sleepDuration ){
          var now = new Date().getTime();
        while(new Date().getTime() < now + sleepDuration){ /* do nothing */ }
      }

        export function synchronize(state: StateInterface, stack) {
            return Q.all(_.map(_.shuffle(lib.parallel.funsToProcess), Q.fcall));
        }
    }

    export function setType(stack, name, type) {
        if (_.isUndefined(stack["types"])) {
            stack["types"] = {}
      }
        stack["types"][name] = type;
    }

    export function cudaReference(state, stack, name): any {
        var ref;
        if (_.isUndefined(stack[name]) && stack.types[name] !== "ReferenceType" && stack.types[name] !== "CUDAReferenceType"
            && stack.types[name] !== "CReference" && stack.types[name] !== "CUDAReference") {
            return {
                type: "Identifier",
                id: name,
                stack: stack,
                state: state
            };
        }
        lib.utils.assert.ok(stack[name].type === "CUDAReference");
        ref = stack[name];
      return ref
    }

    export function reference(state, stack, name): any {
        var ref;
        if (name === "argv") {
            return {};
        } else if (_.isObject(name) && _.contains(["CUDAReference", "CReference", "Identifier"], name.type)) {
            return name;
        } else if (_.isUndefined(stack[name]) && stack.types[name] !== "ReferenceType" && stack.types[name] !== "CUDAReferenceType"
            && stack.types[name] !== "CReference" && stack.types[name] !== "CUDAReference") {

              return {
                type: "Identifier",
                id: name,
                stack: stack,
                state: state
            }

      } else if (!_.isUndefined(stack[name]) && _.isNumber(stack[name])) {
            return stack[name];
        }
        if (stack[name].type === "CUDAReference") {
            return stack[name];
        }
        lib.utils.assert.ok(stack[name].type === "CReference");
        ref = stack[name];
      return ref;
    }

    export function setReference(state, stack, to, from) {
        if (to.type === "Identifier") {
            stack[to.id] = from;
        }
    }

    export function checkEvent(state, stack) {
        return false;
    }

    export function initWorker(state) {
      return {
            type: "Worker",
            state: state
        }
    }
}
