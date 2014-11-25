/// <reference path="../../ref.ts" />

import numerics = require("./numerics");
import integer = require("./integer");
import utils = require("./../../utils/utils");

class Int8 implements numerics.CNumber, integer.IntegerTraits, integer.SignedIntegerTraits {
    private value_:Int8Array;
    static MAX_VALUE:number = 128;
    static MIN_VALUE:number = -128;
    static KIND:numerics.CNumberKind = numerics.CNumberKind.Int8;

    static is_integer : () => boolean;
    static is_exact : () => boolean;
    static has_infinity : () => boolean;
    static is_modulo : () => boolean;
    static is_signed : () => boolean;
    static min = () => new Int8(this.MIN_VALUE);
    static max = () => new Int8(this.MAX_VALUE);
    static lowest = () => new Int8(this.MIN_VALUE);
    static highest = () => new Int8(this.MAX_VALUE);
    static infinity = () => new Int8(0);

    constructor(n?:number) {
        this.value_ = new Int8Array(1);
        if (n) {
            this.value_[0] = n;
        } else {
            this.value_[0] = utils.rand(this.MIN_VALUE, this.MAX_VALUE);
        }
    }

    getValue():Int8Array {
        return this.value_;
    }

    add(other:numerics.CNumber):numerics.CNumber {
        if (other.KIND <= this.KIND) {
            return new Int8(this.value_[0] + other.getValue()[0]);
        }
        var typ = numerics.CNumberKindMap.get(other.KIND);
        return new typ(this.value_[0] + other.getValue()[0]);
    }

    addTo(other:numerics.CNumber):numerics.CNumber {
        this.value_[0] += other.getValue()[0];
        return this;
    }

    sub(other:numerics.CNumber):numerics.CNumber {
        if (other.KIND <= this.KIND) {
            return new Int8(this.value_[0] - other.getValue()[0]);
        }
        var typ = numerics.CNumberKindMap.get(other.KIND);
        return new typ(this.value_[0] - other.getValue()[0]);
    }

    subFrom(other:numerics.CNumber):numerics.CNumber {
        this.value_[0] -= other.getValue()[0];
        return this;
    }

    mul(other:numerics.CNumber):numerics.CNumber {
        if (other.KIND <= this.KIND) {
            return new Int8(this.value_[0] * other.getValue()[0]);
        }
        var typ = numerics.CNumberKindMap.get(other.KIND);
        return new typ(this.value_[0] * other.getValue()[0]);
    }

    mulBy(other:numerics.CNumber):numerics.CNumber {
        this.value_[0] *= other.getValue()[0];
        return this;
    }

    div(other:numerics.CNumber):numerics.CNumber {
        if (other.KIND <= this.KIND) {
            return new Int8(this.value_[0] / other.getValue()[0]);
        }
        var typ = numerics.CNumberKindMap.get(other.KIND);
        return new typ(this.value_[0] / other.getValue()[0]);
    }

    divBy(other:numerics.CNumber):numerics.CNumber {
        this.value_[0] /= other.getValue()[0];
        return this;
    }

    negate():numerics.CNumber {
        return new Int8(-this.value_[0]);
    }
    value() : number {
        return this.value_[0];
    }
}

numerics.CNumberKindMap.set(numerics.CNumberKind.Int8, Int8);
utils.applyMixins(Int8, [integer.IntegerTraits, integer.SignedIntegerTraits]);
export = Int8;