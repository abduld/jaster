
/// <reference path='../../utils/utils.ts' />
/// <reference path='detail.ts' />


module lib.c.type {
    import utils = lib.utils;
    export module detail {
        export class Int32 implements CLiteral, IntegerTraits, SignedIntegerTraits {
            private value_: Int32Array;
            MAX_VALUE: number = 2147483648;
            MIN_VALUE: number = -2147483648;
            KIND: CLiteralKind = CLiteralKind.Int32;

            is_integer: () => boolean;
            is_exact: () => boolean;
            has_infinity: () => boolean;
            is_modulo: () => boolean;
            is_signed: () => boolean;
            min = () => new Int32(this.MIN_VALUE);
            max = () => new Int32(this.MAX_VALUE);
            lowest = () => new Int32(this.MIN_VALUE);
            highest = () => new Int32(this.MAX_VALUE);
            infinity = () => new Int32(0);

            constructor(n?: number) {
                this.value_ = new Int32Array(1);
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
                    return utils.castTo<CLiteral>(new Int32(this.value_[0] + other.getValue()[0]));
                }
                var typ = CLiteralKindMap.get(other.KIND);
                return new typ(this.value_[0] + other.getValue()[0]);
            }

            addTo(other: CLiteral): CLiteral {
                this.value_[0] += other.getValue()[0];
                return utils.castTo<CLiteral>(this);
            }

            sub(other: CLiteral): CLiteral {
                if (other.KIND <= this.KIND) {
                    return utils.castTo<CLiteral>(new Int32(this.value_[0] - other.getValue()[0]));
                }
                var typ = CLiteralKindMap.get(other.KIND);
                return new typ(this.value_[0] - other.getValue()[0]);
            }

            subFrom(other: CLiteral): CLiteral {
                this.value_[0] -= other.getValue()[0];
                return utils.castTo<CLiteral>(this);
            }

            mul(other: CLiteral): CLiteral {
                if (other.KIND <= this.KIND) {
                    return utils.castTo<CLiteral>(new Int32(this.value_[0] * other.getValue()[0]));
                }
                var typ = CLiteralKindMap.get(other.KIND);
                return new typ(this.value_[0] * other.getValue()[0]);
            }

            mulBy(other: CLiteral): CLiteral {
                this.value_[0] *= other.getValue()[0];
                return utils.castTo<CLiteral>(this);
            }

            div(other: CLiteral): CLiteral {
                if (other.KIND <= this.KIND) {
                    return utils.castTo<CLiteral>(new Int32(this.value_[0] / other.getValue()[0]));
                }
                var typ = CLiteralKindMap.get(other.KIND);
                return new typ(this.value_[0] / other.getValue()[0]);
            }

            divBy(other: CLiteral): CLiteral {
                this.value_[0] /= other.getValue()[0];
                return utils.castTo<CLiteral>(this);
            }

            negate(): CLiteral {
                return utils.castTo<CLiteral>(new Int32(-this.value_[0]));
            }
            value(): number {
                return this.value_[0];
            }
        }

        CLiteralKindMap.set(CLiteralKind.Int32, Int32);
        utils.applyMixins(Int32, [IntegerTraits, SignedIntegerTraits]);
    }
    export import Int32 = detail.Int32;
}
