/// <reference path="utils.ts" />

import utils = require("./utils");

module System {
    "use strict";
    enum AddressSpace {
        Shared,
        Global,
        Host
    };
    export class MemoryObject {
        public id: string;
        public data: DataView;
        public addressSpace : AddressSpace;
        constructor(id: string, addressSpace : AddressSpace, data: DataView) {
            this.id = id;
            this.addressSpace = addressSpace;
            this.data = data;
        }
        get(idx: number): any {
            return this.data[idx];
        }
        set(idx: number, val: any): any {
            return this.data[idx] = val;
        }
    };
    export class MemoryManager {
        private addressSpace : AddressSpace;
	//private memmap: Map<string, MemoryObject> = new Map<string, MemoryObject>();
        private MB: number = 1024;
        private TOTALMEMORY : number;
        private memory : ArrayBuffer;
        private memoryOffset: number = 0;

        constructor(addressSpace : AddressSpace) {
            this.TOTALMEMORY = 10 * this.MB;
            this.addressSpace = addressSpace;
            this.memory = new ArrayBuffer(this.TOTALMEMORY);
        }

        public malloc(n: number): MemoryObject {
            var buffer = new MemoryObject(
                utils.guuid(),
                this.addressSpace,
                new DataView(this.memory, this.memoryOffset, this.memoryOffset + n)
            );
	    //this.memmap.set(buffer.id, buffer);
            this.memoryOffset += n;
            return buffer;
        }
        public free(mem : MemoryObject): void {
            mem = undefined;
        }
        public ref(obj) {
	        return "todo";
        }
        public deref(mem) {
            return mem[0];
        }
    }
    export class HostMemoryManager extends MemoryManager {
        constructor() {
            super(AddressSpace.Host);
        }
    }
    export class GlobalMemoryManager extends MemoryManager {
        constructor() {
            super(AddressSpace.Global);
        }
    }
    export class SharedMemoryManager extends MemoryManager {
        constructor() {
            super(AddressSpace.Shared);
        }
    }
}

export = System;

