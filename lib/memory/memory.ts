/// <reference path="../ref.ts" />
/// <reference path="../c/memory/memory.ts" />

module lib {
    export module memory {
        export module detail {
            export import MemoryManager = lib.c.memory.MemoryManager;
        }
        export import AddressSpace = lib.c.memory.AddressSpace;
        export class HostMemoryManager extends detail.MemoryManager {
            constructor() {
                super(AddressSpace.Host);
            }
        }
        export class GlobalMemoryManager extends detail.MemoryManager {
            constructor() {
                super(AddressSpace.Global);
            }
        }
        export class SharedMemoryManager extends detail.MemoryManager {
            constructor() {
                super(AddressSpace.Shared);
            }
        }
        import CLiteralKind =  lib.c.memory.CLiteralKind;

var typeStringToCLiteralKind : {[key: string]: CLiteralKind;} = {
  char: CLiteralKind.Int8,
  int8: CLiteralKind.Int8,
  uint: CLiteralKind.Uint32,
  unsigned: CLiteralKind.Uint32,
  "unsigned int": CLiteralKind.Uint32,
  int: CLiteralKind.Int32,
  int64_t : CLiteralKind.Int64,
  float: CLiteralKind.Float,
  double: CLiteralKind.Double
};
        export function getElement(state, stack, ref, idx) {
          var typ;
          var mm;
          if (ref.addressSpace === "Global") {
            mm = state.globalMemory;
          } else {
            mm = state.hostMemory;
          }
          typ = state["types"][ref.name];
          if (typ != "ReferenceType") {
            console.log("Invalid type " + JSON.stringify(typ));
          }
          typ = typ.kind;
          if (typ == "ReferenceType") {
            console.log("Unexpected type " + JSON.stringify(typ));
          }
          typ = typeStringToCLiteralKind[typ.bases[0]];
          return ref.getElement(idx, typ);
        }
        export function setElement(state, stack, ref, idx, data) {

        }
    }
}
