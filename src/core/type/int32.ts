/// <reference path="../../ref.ts" />

import numerics = require("./numerics");
import integer = require("./integer");
import utils = require("./../../utils/utils");

class Int32 implements numerics.CNumber, integer.IntegerTraits, integer.SignedIntegerTraits {
    private value_:Int32Array;
    public MAX_VALUE:number = 2147483648;
    public MIN_VALUE:number = -2147483648;
    public KIND:numerics.CNumberKind = numerics.CNumberKind.Int32;

    is_integer : () => boolean;
    is_exact : () => boolean;
    has_infinity : () => boolean;
    is_modulo : () => boolean;
    is_signed : () => boolean;
    min = () => new Int32(this.MIN_VALUE);
    max = () => new Int32(this.MAX_VALUE);
    lowest = () => new Int32(this.MIN_VALUE);
    highest = () => new Int32(this.MAX_VALUE);
    infinity = () => new Int32(0);

    constructor(n?:number) {
        this.value_ = new Int8Array(1);
        if (n) {
            this.value_[0] = n;
        } else {
            this.value_[0] = utils.rand(this.MIN_VALUE, this.MAX_VALUE);
        }
    }

    public getValue():Int32Array {
        return this.value_;
    }

    public add(other:numerics.CNumber):numerics.CNumber {
        if (other.KIND <= this.KIND) {
            return new Int32(this.value_[0] + other.getValue()[0]);
        }
        var typ = numerics.CNumberKindMap.get(other.KIND);
        return new typ(this.value_[0] + other.getValue()[0]);
    }

    public addTo(other:numerics.CNumber):numerics.CNumber {
        this.value_[0] += other.getValue()[0];
        return this;
    }

    public sub(other:numerics.CNumber):numerics.CNumber {
        if (other.KIND <= this.KIND) {
            return new Int32(this.value_[0] - other.getValue()[0]);
        }
        var typ = numerics.CNumberKindMap.get(other.KIND);
        return new typ(this.value_[0] - other.getValue()[0]);
    }

    public subFrom(other:numerics.CNumber):numerics.CNumber {
        this.value_[0] -= other.getValue()[0];
        return this;
    }

    public mul(other:numerics.CNumber):numerics.CNumber {
        if (other.KIND <= this.KIND) {
            return new Int32(this.value_[0] * other.getValue()[0]);
        }
        var typ = numerics.CNumberKindMap.get(other.KIND);
        return new typ(this.value_[0] * other.getValue()[0]);
    }

    public mulBy(other:numerics.CNumber):numerics.CNumber {
        this.value_[0] *= other.getValue()[0];
        return this;
    }

    public div(other:numerics.CNumber):numerics.CNumber {
        if (other.KIND <= this.KIND) {
            return new Int32(this.value_[0] / other.getValue()[0]);
        }
        var typ = numerics.CNumberKindMap.get(other.KIND);
        return new typ(this.value_[0] / other.getValue()[0]);
    }

    public divBy(other:numerics.CNumber):numerics.CNumber {
        this.value_[0] /= other.getValue()[0];
        return this;
    }

    public negate():numerics.CNumber {
        return new Int32(-this.value_[0]);
    }
    public value() : number {
        return this.value_[0];
    }
}
numerics.CNumberKindMap.set(numerics.CNumberKind.Int32, Int32);
utils.applyMixins(Int32, [integer.IntegerTraits, integer.SignedIntegerTraits]);

export = Int32;