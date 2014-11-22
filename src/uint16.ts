
/// <reference path="ref.ts" />
module Core {
    export class Uint16 implements CNumber, IntegerTraits, UnsignedIntegerTraits {
        private value_:Uint8Array;
        public MAX_VALUE:number = 65535;
        public MIN_VALUE:number = 0;
        public KIND:CNumberKind = CNumberKind.Uint8;

        is_integer : () => boolean;
        is_exact : () => boolean;
        has_infinity : () => boolean;
        is_modulo : () => boolean;
        is_signed : () => boolean;
        min = () => new Uint16(this.MIN_VALUE);
        max = () => new Uint16(this.MAX_VALUE);
        lowest = () => new Uint16(this.MIN_VALUE);
        highest = () => new Uint16(this.MAX_VALUE);
        infinity = () => new Uint16(0);

        constructor(n?:number) {
            this.value_ = new Uint8Array(1);
            if (n) {
                this.value_[0] = n;
            } else {
                this.value_[0] = rand(this.MIN_VALUE, this.MAX_VALUE);
            }
        }

        public getValue():Int8Array {
            return this.value_;
        }

        public add(other:CNumber):CNumber {
            if (other.KIND <= this.KIND) {
                return new Uint16(this.value_[0] + other.getValue()[0]);
            }
            var typ = CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] + other.getValue()[0]);
        }

        public addTo(other:CNumber):CNumber {
            this.value_[0] += other.getValue()[0];
            return this;
        }

        public sub(other:CNumber):CNumber {
            if (other.KIND <= this.KIND) {
                return new Uint16(this.value_[0] - other.getValue()[0]);
            }
            var typ = CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] - other.getValue()[0]);
        }

        public subFrom(other:CNumber):CNumber {
            this.value_[0] -= other.getValue()[0];
            return this;
        }

        public mul(other:CNumber):CNumber {
            if (other.KIND <= this.KIND) {
                return new Uint16(this.value_[0] * other.getValue()[0]);
            }
            var typ = CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] * other.getValue()[0]);
        }

        public mulBy(other:CNumber):CNumber {
            this.value_[0] *= other.getValue()[0];
            return this;
        }

        public div(other:CNumber):CNumber {
            if (other.KIND <= this.KIND) {
                return new Uint16(this.value_[0] / other.getValue()[0]);
            }
            var typ = CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] / other.getValue()[0]);
        }

        public divBy(other:CNumber):CNumber {
            this.value_[0] /= other.getValue()[0];
            return this;
        }

        public negate():CNumber {
            return new Uint16(-this.value_[0]);
        }
        public value() : number {
            return this.value_[0];
        }
    }
    CNumberKindMap.set(CNumberKind.Uint16, Uint16);
    applyMixins(Uint16, [IntegerTraits, UnsignedIntegerTraits]);
}