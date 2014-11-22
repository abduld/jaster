
/// <reference path="ref.ts" />

module Core {

    export class Int8 implements CNumber, IntegerTraits, SignedIntegerTraits {
        private value_:Int8Array;
        public MAX_VALUE:number = 128;
        public MIN_VALUE:number = -128;
        public KIND:CNumberKind = CNumberKind.Int8;

        is_integer : () => boolean;
        is_exact : () => boolean;
        has_infinity : () => boolean;
        is_modulo : () => boolean;
        is_signed : () => boolean;
        min = () => new Int8(this.MIN_VALUE);
        max = () => new Int8(this.MAX_VALUE);
        lowest = () => new Int8(this.MIN_VALUE);
        highest = () => new Int8(this.MAX_VALUE);
        infinity = () => new Int8(0);

        constructor(n?:number) {
            this.value_ = new Int8Array(1);
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
                return new Int8(this.value_[0] + other.getValue()[0]);
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
                return new Int8(this.value_[0] - other.getValue()[0]);
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
                return new Int8(this.value_[0] * other.getValue()[0]);
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
                return new Int8(this.value_[0] / other.getValue()[0]);
            }
            var typ = CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] / other.getValue()[0]);
        }

        public divBy(other:CNumber):CNumber {
            this.value_[0] /= other.getValue()[0];
            return this;
        }

        public negate():CNumber {
            return new Int8(-this.value_[0]);
        }
        public value() : number {
            return this.value_[0];
        }
    }

    CNumberKindMap.set(CNumberKind.Int8, Int8);
    applyMixins(Int8, [IntegerTraits, SignedIntegerTraits]);
}

export = Core