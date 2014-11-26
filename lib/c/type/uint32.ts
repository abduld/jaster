
/// <reference path='../../utils/utils.ts' />
/// <reference path='detail.ts' />


module lib.c.type {
    import utils = lib.utils;
    export module detail {
        export class Uint32 implements CLiteral, IntegerTraits, UnsignedIntegerTraits {
            private value_: Uint32Array;
            MAX_VALUE: number = 4294967295;
            MIN_VALUE: number = 0;
            KIND: CLiteralKind = CLiteralKind.Uint32;

            is_integer: () => boolean;
            is_exact: () => boolean;
            has_infinity: () => boolean;
            is_modulo: () => boolean;
            is_signed: () => boolean;
            min = () => new Uint32(this.MIN_VALUE);
            max = () => new Uint32(this.MAX_VALUE);
            lowest = () => new Uint32(this.MIN_VALUE);
            highest = () => new Uint32(this.MAX_VALUE);
            infinity = () => new Uint32(0);

            constructor(n?: number) {
                this.value_ = new Uint32Array(1);
                if (n) {
                    this.value_[0] = n;
                } else {
                    this.value_[0] = utils.rand(this.MIN_VALUE, this.MAX_VALUE);
                }
            }

            getValue(): Int32Array {
                return this.value_;
            }

            add(other: CLiteral): CLiteral {
                if (other.KIND <= this.KIND) {
                    return utils.castTo<CLiteral>(new Uint32(this.value_[0] + other.getValue()[0]));
                }
                var typ = CLiteralKindMap.get(other.KIND);
                return new typ(this.value_[0] + other.getValue()[0]);
            }

            addTo(other: CLiteral): CLiteral {
                this.value_[0] += other.getValue()[0];
                return this;
            }

            sub(other: CLiteral): CLiteral {
                if (other.KIND <= this.KIND) {
                    return utils.castTo<CLiteral>(new Uint32(this.value_[0] - other.getValue()[0]));
                }
                var typ = CLiteralKindMap.get(other.KIND);
                return new typ(this.value_[0] - other.getValue()[0]);
            }

            subFrom(other: CLiteral): CLiteral {
                this.value_[0] -= other.getValue()[0];
                return this;
            }

            mul(other: CLiteral): CLiteral {
                if (other.KIND <= this.KIND) {
                    return utils.castTo<CLiteral>(new Uint32(this.value_[0] * other.getValue()[0]));
                }
                var typ = CLiteralKindMap.get(other.KIND);
                return new typ(this.value_[0] * other.getValue()[0]);
            }

            mulBy(other: CLiteral): CLiteral {
                this.value_[0] *= other.getValue()[0];
                return this;
            }

            div(other: CLiteral): CLiteral {
                if (other.KIND <= this.KIND) {
                    return utils.castTo<CLiteral>(new Uint32(this.value_[0] / other.getValue()[0]));
                }
                var typ = CLiteralKindMap.get(other.KIND);
                return new typ(this.value_[0] / other.getValue()[0]);
            }

            divBy(other: CLiteral): CLiteral {
                this.value_[0] /= other.getValue()[0];
                return this;
            }

            negate(): CLiteral {
                return utils.castTo<CLiteral>(new Uint32(-this.value_[0]));
            }
            value(): number {
                return this.value_[0];
            }
        }
        CLiteralKindMap.set(CLiteralKind.Uint32, Uint32);
        utils.applyMixins(Uint32, [IntegerTraits, UnsignedIntegerTraits]);
    }
    export import Uint32 = detail.Uint32;
}