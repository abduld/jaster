/// <reference path="./type/type.ts" />

module lib {
    export module c {
        import type = lib.c.type;

        var sizeof_: { [key: string]: number; } = {
            float: 4,
            int32: 4,
            int: 4,
            int64: 8,
            double: 8,
            float32: 4,
            float64: 8,
            int16: 2,
            int8: 2,
            char: 2,
            uint8: 1,
            uint16: 2,
            uint32: 4,
            uint64: 8
        };
        export function sizeof(state, stack, typ: string) {
            return sizeof_[typ];
        }

        export function makeReference(state, stack, name, data) {
            if (_.contains(["CReference", "CUDAReference"], data.type)) {
                return _.extend(data, { name: name });
            }
            var hostMem: lib.memory.HostMemoryManager = state.hostMemory;
            var typ = stack.types[name].kind.bases[0];
            var elemSize = sizeof(state, stack, typ);
            var mem = hostMem.malloc(data.length * elemSize);
            var ref = {
                type: "CReference",
                id: name,
                mem: mem,
                args: []
            };

            _.each(data, (elem, idx) => setElement(state, stack, ref, idx, elem));
            return ref;
        }
    }
}
