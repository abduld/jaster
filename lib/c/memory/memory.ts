/// <reference path='../../utils/utils.ts' />
/// <reference path='../../ast/importer/cena.ts' />

module lib.c {
    export module memory {
        export enum AddressSpace {
            Shared,
            Global,
            Host
        }
        ;
        export import CLiteral = lib.c.type.detail.CLiteral;
        export import CLiteralKind = lib.c.type.detail.CLiteralKind;
        export class Reference {
            name: string;
            id: string;
            data: DataView;
            addressSpace: AddressSpace;
            KIND: CLiteralKind;

            constructor(id: string, addressSpace: AddressSpace, data: DataView) {
                this.id = id;
                this.addressSpace = addressSpace;
                this.data = data;
                this.KIND = CLiteralKind.Int8;
            }

            getElement(idx: number, kind?: CLiteralKind): any {

                if (_.isUndefined(kind)) {
                    kind = this.KIND;
                }
                if (lib.ast.importer.cena.FakeMath) {
                switch (kind) {
                    case CLiteralKind.Int8:
                        return (this.data.getInt8(idx));
                    case CLiteralKind.Int16:
                        return (this.data.getInt16(idx));
                    case CLiteralKind.Int32:
                        return (this.data.getInt32(idx));
                    case CLiteralKind.Int64:
                        return [this.data.getInt32(2 * idx), this.data.getInt32(2 * idx + 1)];
                    case CLiteralKind.Uint8:
                        return (this.data.getUint8(idx));
                    case CLiteralKind.Uint16:
                        return (this.data.getUint16(idx));
                    case CLiteralKind.Uint32:
                        return (this.data.getUint32(idx));
                }
              } else {

                switch (kind) {
                  case CLiteralKind.Int8:
                  return new lib.c.type.Int8(this.data.getInt8(idx));
                  case CLiteralKind.Int16:
                  return new lib.c.type.Int16(this.data.getInt16(idx));
                  case CLiteralKind.Int32:
                  return new lib.c.type.Int32(this.data.getInt32(idx));
                  case CLiteralKind.Int64:
                  return new lib.c.type.Int64(this.data.getInt32(2 * idx), this.data.getInt32(2 * idx + 1));
                  case CLiteralKind.Uint8:
                  return new lib.c.type.Uint8(this.data.getUint8(idx));
                  case CLiteralKind.Uint16:
                  return new lib.c.type.Uint16(this.data.getUint16(idx));
                  case CLiteralKind.Uint32:
                  return new lib.c.type.Uint32(this.data.getUint32(idx));
                }
              }
            }

            setElement(idx: number, val: number, kind?: CLiteralKind): CLiteral;
            setElement(idx: number, val: CLiteral, kind?: CLiteralKind): CLiteral;
            setElement(idx: number, val: any, kind?: CLiteralKind): any {
                if (val instanceof lib.c.type.Int64) {
                    var i64: lib.c.type.Int64 = utils.castTo<lib.c.type.Int64>(val);
                    this.data.setInt32(2 * idx, i64.getHigh());
                    this.data.setInt32(2 * idx + 1, i64.getLow());
                    return this.getElement(idx);
                } else if (val instanceof Object) {
                    var tmp: CLiteral = utils.castTo<CLiteral>(val);
                    val = tmp.getValue()[0];
                }

                if (_.isUndefined(kind)) {
                    kind = this.KIND;
                }
                switch (kind) {
                    case CLiteralKind.Int8:
                        this.data.setInt8(idx, val);
                        break;
                    case CLiteralKind.Int16:
                        this.data.setInt16(idx, val);
                        break;
                    case CLiteralKind.Int32:
                        this.data.setInt32(idx, val);
                        break;
                    case CLiteralKind.Int64:
                        this.data.setInt32(2 * idx, 0);
                        this.data.setInt32(2 * idx + 1, val);
                        break;
                    case CLiteralKind.Uint8:
                        this.data.setUint8(idx, val);
                        break;
                    case CLiteralKind.Uint16:
                        this.data.setUint16(idx, val);
                        break;
                    case CLiteralKind.Uint32:
                        this.data.setUint32(idx, val);
                        break;
                }
                return this.getElement(idx);
            }

            ref(): Reference {
                return new Reference(
                    utils.guuid(),
                    this.addressSpace,
                    new DataView(this.data.buffer, 0, 1)
                    );
            }

            deref(): CLiteral {
                return this.getElement(0);
            }
        }

        var KB: number = 1024;
        var MB: number = KB*KB;
        export class MemoryManager {
            private addressSpace: AddressSpace;
            //private memmap: Map<string, MemoryObject> = new Map<string, MemoryObject>();
            private TOTAL_MEMORY: number;
            private memory: ArrayBuffer;
            private memoryOffset: number = 0;

            constructor(addressSpace: AddressSpace) {
                this.TOTAL_MEMORY = 10 * MB;
                this.addressSpace = addressSpace;
                this.memory = new ArrayBuffer(this.TOTAL_MEMORY);
            }

            public malloc(n: number): Reference {
                var buffer = new Reference(
                    utils.guuid(),
                    this.addressSpace,
                    new DataView(this.memory, this.memoryOffset, this.memoryOffset + n)
                    );
                //this.memmap.set(buffer.id, buffer);
                this.memoryOffset += n;
                return buffer;
            }

            public free(mem: Reference): void {
                mem = undefined;
            }

            public ref(obj) {
                return "todo";
            }

            public deref(mem) {
                return mem[0];
            }
        }
    }
}
