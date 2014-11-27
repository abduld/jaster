/// <reference path="../ref.ts" />

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
    }
}
