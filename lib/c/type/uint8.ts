
/// <reference path='../../utils/utils.ts' />
/// <reference path='detail.ts' />


module lib.c.type {
    export module detail {
        import utils = lib.utils;
        export class Uint8 implements CLiteral, IntegerTraits, UnsignedIntegerTraits {
            private value_: Uint8Array;
            MAX_VALUE: number = 255;
            MIN_VALUE: number = 0;
            KIND: CLiteralKind = CLiteralKind.Uint8;

            is_integer: () => boolean;
            is_exact: () => boolean;
            has_infinity: () => boolean;
            is_modulo: () => boolean;
            is_signed: () => boolean;
            min = () => new Uint8(this.MIN_VALUE);
            max = () => new Uint8(this.MAX_VALUE);
            lowest = () => new Uint8(this.MIN_VALUE);
            highest = () => new Uint8(this.MAX_VALUE);
            infinity = () => new Uint8(0);

            constructor(n?: number) {
                this.value_ = new Uint8Array(1);
                if (n) {
                    this.value_[0] = n;
                } else {
                    this.value_[0] = utils.rand(this.MIN_VALUE, this.MAX_VALUE);
                }
            }

            getValue(): Int8Array {
                return this.value_;
            }

            add(other: CLiteral): CLiteral {
                if (other.KIND <= this.KIND) {
                    return utils.castTo<CLiteral>(new Uint8(this.value_[0] + other.getValue()[0]));
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
                    return utils.castTo<CLiteral>(new Uint8(this.value_[0] - other.getValue()[0]));
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
                    return utils.castTo<CLiteral>(new Uint8(this.value_[0] * other.getValue()[0]));
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
                    return utils.castTo<CLiteral>(new Uint8(this.value_[0] / other.getValue()[0]));
                }
                var typ = CLiteralKindMap.get(other.KIND);
                return new typ(this.value_[0] / other.getValue()[0]);
            }

            divBy(other: CLiteral): CLiteral {
                this.value_[0] /= other.getValue()[0];
                return this;
            }

            negate(): CLiteral {
                return utils.castTo<CLiteral>(new Uint8(-this.value_[0]));
            }
            value(): number {
                return this.value_[0];
            }
        }
        CLiteralKindMap.set(CLiteralKind.Uint8, Uint8);
        utils.applyMixins(Uint8, [IntegerTraits, UnsignedIntegerTraits]);
    }
    export import Uint8 = detail.Uint8;
}