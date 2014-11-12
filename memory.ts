
module System {
    var guuid = (): string => {
        var s4 = (): string =>
            Math.floor((1 + Math.random()) * 0x10000)
                .toString(16)
                .substring(1);
        return s4() + s4() + '-' + s4() + '-' + s4() + '-' +
            s4() + '-' + s4() + s4() + s4();
    };
    export class MemoryObject {
        public id: string;
        public data: DataView;
        constructor(id: string, data: DataView) {
            this.id = id;
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
        private memmap: Map<string, MemoryObject> = new Map<string, MemoryObject>();
        private MB: number = 1024;
        private TOTAL_MEMORY;
        private memory;
        private memory_offset: number = 0;

        constructor() {
            this.TOTAL_MEMORY = 10 * this.MB;
            this.memory = new ArrayBuffer(this.TOTAL_MEMORY);
        }

        malloc(n: number): MemoryObject {
            var buffer = new MemoryObject(
                guuid(),
                new DataView(this.memory, this.memory_offset, this.memory_offset + n)
            );
            this.memmap.set(buffer.id, buffer);
            this.memory_offset += n;
            return buffer;
        }

        free(mem : MemoryObject): void {
            mem = undefined;
        }

        ref(obj) {
	        return "todo"
        }

        deref(mem) {
            return mem[0];
        }

    }

    export class HostMemoryManager extends MemoryManager {
        constructor() {
            super();
        }
    }
    export class GlobalMemoryManager extends MemoryManager {
        constructor() {
            super();
        }
    }

    export class SharedMemoryManager extends MemoryManager {
        constructor() {
            super();
        }
    }

}



