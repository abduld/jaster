

/// <reference path="memory/memory.ts" />
/// <reference path="cuda/exec/profile.ts" />
/// <reference path="parallel/workerpool.ts" />
/// <reference path="wb/wb.ts" />

module lib {
    export interface StateInterface {
        type: string;
        model: typeof lib.cuda.exec.FermiArchitecture;
        globalMemory: lib.memory.GlobalMemoryManager;
        hostMemory: lib.memory.HostMemoryManager;
        threadPool: typeof lib.parallel.WorkerPool;
        id: string;
    }
    export function init(): StateInterface {
        return {
            type: "GlobalState",
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
          lib.utils.assert.ok(ref.mem.type === "CUDAReference");
          return {
            type: "CUDAReference",
            id: lib.utils.guuid(),
            mem: state.globalMemory.malloc(byteCount),
            args: args
          }
        }
    }

    export module c {
        export function malloc(state: StateInterface, stack, byteCount: number, args: string[]): CReference {
            return {
                type: "CReference",
                id: lib.utils.guuid(),
                mem: state.hostMemory.malloc(byteCount),
                args: args
            }
        }
    }

    export module parallel {
        export function scheduleThread(state: StateInterface, fun: Function) {
          console.log("todo scheduleThread");
          return ;
        }
    }

    export function setType(stack, name, type) {
        if (_.isUndefined(stack["types"])) {
            stack["types"] = {}
      }
        stack["types"][name] = type;
    }

    export function cudaReference(state, stack, name) : any {
      var ref;
      if (_.isUndefined(stack[name]) && stack.types[name] !== "ReferenceType") {
        return {
          type: "Identifier",
          id: name,
          stack: stack,
          state: state
        };
      }
      lib.utils.assert.ok(stack[name].type === "CUDAReference");
      ref = stack[name];
      return {
        type: ref.type,
        stack: stack,
        state: state,
        id: ref.id,
        mem: ref.mem,
        args: ref.args
      }
    }

    export function reference(state, stack, name) : any {
      var ref;
      if (name === "argv") {
          return {};
      } else if (_.isUndefined(stack[name])&& stack.types[name] !== "ReferenceType") {

              return {
                  type: "Identifier",
                  id: name,
                  stack: stack,
                  state: state
              }

      } else if (!_.isUndefined(stack[name]) && _.isNumber(stack[name])) {
            return stack[name];
        }
      lib.utils.assert.ok(stack[name].type === "CReference");
      ref = stack[name];
      return {
        type: ref.type,
        stack: stack,
        state: state,
        id: ref.id,
        mem: ref.mem,
        args: ref.args
      }
    }

    export function checkEvent(state, stack) {
        return false;
    }
}
