/// <reference path='../../utils/utils.ts' />
/// <reference path='detail.ts' />


module lib.c.type {
    import utils = lib.utils;
    export module detail {
        export class Int16 implements CLiteral, IntegerTraits, SignedIntegerTraits {
            private value_:Int16Array;
            MAX_VALUE:number = 3276;
            MIN_VALUE:number = -3276;
            KIND:CLiteralKind = CLiteralKind.Int16;

            is_integer:() => boolean;
            is_exact:() => boolean;
            has_infinity:() => boolean;
            is_modulo:() => boolean;
            is_signed:() => boolean;
            min = () => new Int16(this.MIN_VALUE);
            max = () => new Int16(this.MAX_VALUE);
            lowest = () => new Int16(this.MIN_VALUE);
            highest = () => new Int16(this.MAX_VALUE);
            infinity = () => new Int16(0);

            constructor(n?:number) {
                this.value_ = new Int16Array(1);
                if (n) {
                    this.value_[0] = n;
                } else {
                    this.value_[0] = utils.rand(this.MIN_VALUE, this.MAX_VALUE);
                }
            }

            getValue():Int16Array {
                return this.value_;
            }

            add(other:CLiteral):CLiteral {
                if (other.KIND <= this.KIND) {
                    return utils.castTo<CLiteral>(new Int16(this.value_[0] + other.getValue()[0]));
                }
                var typ = CLiteralKindMap.get(other.KIND);
                return new typ(this.value_[0] + other.getValue()[0]);
            }

            addTo(other:CLiteral):CLiteral {
                this.value_[0] += other.getValue()[0];
                return utils.castTo<CLiteral>(this);
            }

            sub(other:CLiteral):CLiteral {
                if (other.KIND <= this.KIND) {
                    return utils.castTo<CLiteral>(new Int16(this.value_[0] - other.getValue()[0]));
                }
                var typ = CLiteralKindMap.get(other.KIND);
                return new typ(this.value_[0] - other.getValue()[0]);
            }

            subFrom(other:CLiteral):CLiteral {
                this.value_[0] -= other.getValue()[0];
                return utils.castTo<CLiteral>(this);
            }

            mul(other:CLiteral):CLiteral {
                if (other.KIND <= this.KIND) {
                    return utils.castTo<CLiteral>(new Int16(this.value_[0] * other.getValue()[0]));
                }
                var typ = CLiteralKindMap.get(other.KIND);
                return new typ(this.value_[0] * other.getValue()[0]);
            }

            mulBy(other:CLiteral):CLiteral {
                this.value_[0] *= other.getValue()[0];
                return utils.castTo<CLiteral>(this);
            }

            div(other:CLiteral):CLiteral {
                if (other.KIND <= this.KIND) {
                    return utils.castTo<CLiteral>(new Int16(this.value_[0] / other.getValue()[0]));
                }
                var typ = CLiteralKindMap.get(other.KIND);
                return new typ(this.value_[0] / other.getValue()[0]);
            }

            divBy(other:CLiteral):CLiteral {
                this.value_[0] /= other.getValue()[0];
                return utils.castTo<CLiteral>(this);
            }

            negate():CLiteral {
                return utils.castTo<CLiteral>(new Int16(-this.value_[0]));
            }

            value():number {
                return this.value_[0];
            }
        }

        CLiteralKindMap.set(CLiteralKind.Int16, Int16);
        utils.applyMixins(Int16, [IntegerTraits, SignedIntegerTraits]);
    }
    export import Int16 = detail.Int16;
}