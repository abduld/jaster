
/// <reference path='../../utils/utils.ts' />
/// <reference path='detail.ts' />


module lib.c.type {
    import utils = lib.utils;
    export module detail {
        export class Uint16 implements CLiteral, IntegerTraits, UnsignedIntegerTraits {
            private value_: Uint16Array;
            MAX_VALUE: number = 65535;
            MIN_VALUE: number = 0;
            KIND: CLiteralKind = CLiteralKind.Uint16;

            is_integer: () => boolean;
            is_exact: () => boolean;
            has_infinity: () => boolean;
            is_modulo: () => boolean;
            is_signed: () => boolean;
            min = () => new Uint16(this.MIN_VALUE);
            max = () => new Uint16(this.MAX_VALUE);
            lowest = () => new Uint16(this.MIN_VALUE);
            highest = () => new Uint16(this.MAX_VALUE);
            infinity = () => new Uint16(0);

            constructor(n?: number) {
                this.value_ = new Uint16Array(1);
                if (n) {
                    this.value_[0] = n;
                } else {
                    this.value_[0] = utils.rand(this.MIN_VALUE, this.MAX_VALUE);
                }
            }

            getValue(): Int16Array {
                return this.value_;
            }

            add(other: CLiteral): CLiteral {
                if (other.KIND <= this.KIND) {
                    return utils.castTo<CLiteral>(new Uint16(this.value_[0] + other.getValue()[0]));
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
                    return utils.castTo<CLiteral>(new Uint16(this.value_[0] - other.getValue()[0]));
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
                    return utils.castTo<CLiteral>(new Uint16(this.value_[0] * other.getValue()[0]));
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
                    return utils.castTo<CLiteral>(new Uint16(this.value_[0] / other.getValue()[0]));
                }
                var typ = CLiteralKindMap.get(other.KIND);
                return new typ(this.value_[0] / other.getValue()[0]);
            }

            divBy(other: CLiteral): CLiteral {
                this.value_[0] /= other.getValue()[0];
                return this;
            }

            negate(): CLiteral {
                return utils.castTo<CLiteral>(new Uint16(-this.value_[0]));
            }
            value(): number {
                return this.value_[0];
            }
        }
        CLiteralKindMap.set(CLiteralKind.Uint16, Uint16);
        utils.applyMixins(Uint16, [IntegerTraits, UnsignedIntegerTraits]);
    }
    export import Uint16 = detail.Uint16;
}