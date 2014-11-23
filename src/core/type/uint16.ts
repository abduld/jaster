
/// <reference path="../../ref.ts" />

import numerics = require("./numerics");
import integer = require("./integer");
import utils = require("./../../utils/utils");

class Uint16 implements numerics.CNumber, integer.IntegerTraits, integer.UnsignedIntegerTraits {
    private value_:Uint16Array;
    public MAX_VALUE:number = 65535;
    public MIN_VALUE:number = 0;
    public KIND:numerics.CNumberKind = numerics.CNumberKind.Uint8;

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
            this.value_[0] = utils.rand(this.MIN_VALUE, this.MAX_VALUE);
        }
    }

    public getValue():Int8Array {
        return this.value_;
    }

    public add(other:numerics.CNumber):numerics.CNumber {
        if (other.KIND <= this.KIND) {
            return new Uint16(this.value_[0] + other.getValue()[0]);
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
            return new Uint16(this.value_[0] - other.getValue()[0]);
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
            return new Uint16(this.value_[0] * other.getValue()[0]);
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
            return new Uint16(this.value_[0] / other.getValue()[0]);
        }
        var typ = numerics.CNumberKindMap.get(other.KIND);
        return new typ(this.value_[0] / other.getValue()[0]);
    }

    public divBy(other:numerics.CNumber):numerics.CNumber {
        this.value_[0] /= other.getValue()[0];
        return this;
    }

    public negate():numerics.CNumber {
        return new Uint16(-this.value_[0]);
    }
    public value() : number {
        return this.value_[0];
    }
}
numerics.CNumberKindMap.set(numerics.CNumberKind.Uint16, Uint16);
utils.applyMixins(Uint16, [integer.IntegerTraits, integer.UnsignedIntegerTraits]);
export = Uint16;