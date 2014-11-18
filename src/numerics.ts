
export enum CNumberKind {
    Int8 = 10,
    Uint8 = 11,
    Int32 = 20,
    Uint32 = 21,
    Int64= 30,
    Float = 22,
    Double = 32
}
var rand = function(min : number, max :number) : number {
    return min + Math.random() * (max - min);
}

var CNumberKindMap : Map<CNumberKind, any> = new Map<CNumberKind, any>();
CNumberKindMap.set(CNumberKind.Int8, Int8);
CNumberKindMap.set(CNumberKind.Uint8, Uint8);

export interface CNumber {
    MAX_VALUE : number;
    MIN_VALUE : number;
    KIND : CNumberKind;
    getValue() : ArrayBufferView;
    add(n : CNumber) : CNumber;
    sub(n : CNumber) : CNumber;
    mul(n : CNumber) : CNumber;
    div(n : CNumber) : CNumber;
    negate() : CNumber;
}
export class Int8 implements CNumber {
    private value : Int8Array;
    public MAX_VALUE : number = 128;
    public MIN_VALUE : number = -128;
    public KIND : CNumberKind = CNumberKind.Int8;
    constructor(n? : number) {
        this.value = new Int8Array(1);
        if (n) {
            this.value[0] = n;
        } else {
            this.value[0] = rand(this.MIN_VALUE, this.MAX_VALUE);
        }
    }
    public getValue() : Int8Array {
        return this.value;
    }
    public add(other : CNumber) : CNumber {
        if (other.KIND <= this.KIND) {
            return new Int8(this.value[0] + other.getValue()[0]);
        }
        var typ = CNumberKindMap.get(other.KIND);
        return new typ(this.value[0] + other.getValue()[0]);
    }
    public addTo(other : CNumber) : CNumber {
        this.value[0] += other.getValue()[0];
        return this;
    }
    public sub(other : CNumber) : CNumber {
        if (other.KIND <= this.KIND) {
            return new Int8(this.value[0] - other.getValue()[0]);
        }
        var typ = CNumberKindMap.get(other.KIND);
        return new typ(this.value[0] - other.getValue()[0]);
    }
    public subFrom(other : CNumber) : CNumber {
        this.value[0] -= other.getValue()[0];
        return this;
    }
    public mul(other : CNumber) : CNumber {
        if (other.KIND <= this.KIND) {
            return new Int8(this.value[0] * other.getValue()[0]);
        }
        var typ = CNumberKindMap.get(other.KIND);
        return new typ(this.value[0] * other.getValue()[0]);
    }
    public mulBy(other : CNumber) : CNumber {
        this.value[0] *= other.getValue()[0];
        return this;
    }
    public div(other : CNumber) : CNumber {
        if (other.KIND <= this.KIND) {
            return new Int8(this.value[0] / other.getValue()[0]);
        }
        var typ = CNumberKindMap.get(other.KIND);
        return new typ(this.value[0] / other.getValue()[0]);
    }
    public divBy(other : CNumber) : CNumber {
        this.value[0] /= other.getValue()[0];
        return this;
    }
    public negate() : CNumber {
        return new Int8(-this.value[0]);
    }
}

export class Uint8 implements CNumber {
    private value : Uint8Array;
    public MAX_VALUE : number = 255;
    public MIN_VALUE : number = 0;
    public KIND : CNumberKind = CNumberKind.Uint8;
    constructor(n? : number) {
        this.value = new Uint8Array(1);
        if (n) {
            this.value[0] = n;
        } else {
            this.value[0] = rand(this.MIN_VALUE, this.MAX_VALUE);
        }
    }
    public getValue() : Int8Array {
        return this.value;
    }
    public add(other : CNumber) : CNumber {
        if (other.KIND <= this.KIND) {
            return new Uint8(this.value[0] + other.getValue()[0]);
        }
        var typ = CNumberKindMap.get(other.KIND);
        return new typ(this.value[0] + other.getValue()[0]);
    }
    public addTo(other : CNumber) : CNumber {
        this.value[0] += other.getValue()[0];
        return this;
    }
    public sub(other : CNumber) : CNumber {
        if (other.KIND <= this.KIND) {
            return new Uint8(this.value[0] - other.getValue()[0]);
        }
        var typ = CNumberKindMap.get(other.KIND);
        return new typ(this.value[0] - other.getValue()[0]);
    }
    public subFrom(other : CNumber) : CNumber {
        this.value[0] -= other.getValue()[0];
        return this;
    }
    public mul(other : CNumber) : CNumber {
        if (other.KIND <= this.KIND) {
            return new Uint8(this.value[0] * other.getValue()[0]);
        }
        var typ = CNumberKindMap.get(other.KIND);
        return new typ(this.value[0] * other.getValue()[0]);
    }
    public mulBy(other : CNumber) : CNumber {
        this.value[0] *= other.getValue()[0];
        return this;
    }
    public div(other : CNumber) : CNumber {
        if (other.KIND <= this.KIND) {
            return new Uint8(this.value[0] / other.getValue()[0]);
        }
        var typ = CNumberKindMap.get(other.KIND);
        return new typ(this.value[0] / other.getValue()[0]);
    }
    public divBy(other : CNumber) : CNumber {
        this.value[0] /= other.getValue()[0];
        return this;
    }
    public negate() : CNumber {
        return new Uint8(-this.value[0]);
    }
}


export class Int32 implements CNumber {
    private value : Int32Array;
    public MAX_VALUE : number = 2147483648;
    public MIN_VALUE : number = -2147483648;
    public KIND : CNumberKind = CNumberKind.Int32;
    constructor(n? : number) {
        this.value = new Int8Array(1);
        if (n) {
            this.value[0] = n;
        } else {
            this.value[0] = rand(this.MIN_VALUE, this.MAX_VALUE);
        }
    }
    public getValue() : Int32Array {
        return this.value;
    }
    public add(other : CNumber) : CNumber {
        if (other.KIND <= this.KIND) {
            return new Int32(this.value[0] + other.getValue()[0]);
        }
        var typ = CNumberKindMap.get(other.KIND);
        return new typ(this.value[0] + other.getValue()[0]);
    }
    public addTo(other : CNumber) : CNumber {
        this.value[0] += other.getValue()[0];
        return this;
    }
    public sub(other : CNumber) : CNumber {
        if (other.KIND <= this.KIND) {
            return new Int32(this.value[0] - other.getValue()[0]);
        }
        var typ = CNumberKindMap.get(other.KIND);
        return new typ(this.value[0] - other.getValue()[0]);
    }
    public subFrom(other : CNumber) : CNumber {
        this.value[0] -= other.getValue()[0];
        return this;
    }
    public mul(other : CNumber) : CNumber {
        if (other.KIND <= this.KIND) {
            return new Int32(this.value[0] * other.getValue()[0]);
        }
        var typ = CNumberKindMap.get(other.KIND);
        return new typ(this.value[0] * other.getValue()[0]);
    }
    public mulBy(other : CNumber) : CNumber {
        this.value[0] *= other.getValue()[0];
        return this;
    }
    public div(other : CNumber) : CNumber {
        if (other.KIND <= this.KIND) {
            return new Int32(this.value[0] / other.getValue()[0]);
        }
        var typ = CNumberKindMap.get(other.KIND);
        return new typ(this.value[0] / other.getValue()[0]);
    }
    public divBy(other : CNumber) : CNumber {
        this.value[0] /= other.getValue()[0];
        return this;
    }
    public negate() : CNumber {
        return new Int32(-this.value[0]);
    }
}

export class Uint32 implements CNumber {
    private value : Uint32Array;
    public MAX_VALUE : number = 4294967295;
    public MIN_VALUE : number = 0;
    public KIND : CNumberKind = CNumberKind.Uint8;
    constructor(n? : number) {
        this.value = new Uint8Array(1);
        if (n) {
            this.value[0] = n;
        } else {
            this.value[0] = rand(this.MIN_VALUE, this.MAX_VALUE);
        }
    }
    public getValue() : Int8Array {
        return this.value;
    }
    public add(other : CNumber) : CNumber {
        if (other.KIND <= this.KIND) {
            return new Uint32(this.value[0] + other.getValue()[0]);
        }
        var typ = CNumberKindMap.get(other.KIND);
        return new typ(this.value[0] + other.getValue()[0]);
    }
    public addTo(other : CNumber) : CNumber {
        this.value[0] += other.getValue()[0];
        return this;
    }
    public sub(other : CNumber) : CNumber {
        if (other.KIND <= this.KIND) {
            return new Uint32(this.value[0] - other.getValue()[0]);
        }
        var typ = CNumberKindMap.get(other.KIND);
        return new typ(this.value[0] - other.getValue()[0]);
    }
    public subFrom(other : CNumber) : CNumber {
        this.value[0] -= other.getValue()[0];
        return this;
    }
    public mul(other : CNumber) : CNumber {
        if (other.KIND <= this.KIND) {
            return new Uint32(this.value[0] * other.getValue()[0]);
        }
        var typ = CNumberKindMap.get(other.KIND);
        return new typ(this.value[0] * other.getValue()[0]);
    }
    public mulBy(other : CNumber) : CNumber {
        this.value[0] *= other.getValue()[0];
        return this;
    }
    public div(other : CNumber) : CNumber {
        if (other.KIND <= this.KIND) {
            return new Uint32(this.value[0] / other.getValue()[0]);
        }
        var typ = CNumberKindMap.get(other.KIND);
        return new typ(this.value[0] / other.getValue()[0]);
    }
    public divBy(other : CNumber) : CNumber {
        this.value[0] /= other.getValue()[0];
        return this;
    }
    public negate() : CNumber {
        return new Uint32(-this.value[0]);
    }
}

// We now hit the problem of numerical representation
// this needs to be reddone, but this will serve as a template
export class Int64 implements CNumber {
    private value : Int32Array;
    public MAX_VALUE : number = NaN;
    public MIN_VALUE : number = NaN;
    public KIND : CNumberKind = CNumberKind.Int32;
    constructor(low ? : number, high? : number) {
        this.value = new Int32Array(2);
        if (low && high) {
            this.value[0] = low;
            this.value[1] = high;
        } else {
            this.value[0] = (new Int32()).getValue()[0];
            this.value[1] = (new Int32()).getValue()[0];
        }
    }
    getLow() : number {
        return this.value[0];
    }
    getHigh() : number {
        return this.value[1];
    }
    public getValue() : Int8Array {
        return this.value;
    }
    // lifted from
    // http://docs.closure-library.googlecode.com/git/local_closure_goog_math_long.js.source.html
    public add(other : CNumber) : CNumber {
        if (other.KIND === this.KIND) {
            var o = <Int64> other;
            var a48 : number = this.getHigh() >>> 16;
            var a32 : number = this.getHigh() & 0xFFFF;
            var a16 : number = this.getLow() >>> 16;
            var a00 : number = this.getLow() & 0xFFFF;

            var b48 : number = o.getHigh() >>> 16;
            var b32 : number = o.getHigh() & 0xFFFF;
            var b16 : number = o.getLow() >>> 16;
            var b00 : number= o.getLow() & 0xFFFF;

            var c48 : number = 0, c32 : number = 0;
            var c16 : number = 0, c00 : number = 0;
            c00 += a00 + b00;
            c16 += c00 >>> 16;
            c00 &= 0xFFFF;
            c16 += a16 + b16;
            c32 += c16 >>> 16;
            c16 &= 0xFFFF;
            c32 += a32 + b32;
            c48 += c32 >>> 16;
            c32 &= 0xFFFF;
            c48 += a48 + b48;
            c48 &= 0xFFFF;

            return new Int64((c16 << 16) | c00, (c48 << 16) | c32);
        }
        var low : Uint32 = <Uint32> (new Uint32(this.getLow())).add(other.getValue()[0]);
        var high : Uint32 = <Uint32> (new Uint32(this.getHigh())).add(new Uint32(low.getValue()[0] >> 31));
        return new Int64(low.getValue()[0] & 0x7FFFFFFF, high.getValue()[0]);
    }
    public addTo(other : CNumber) : CNumber {
        this.value = <Int32Array> this.add(other).getValue();
        return this;
    }
    public sub(other : CNumber) : CNumber {
        return this.add(other.negate());
    }
    public subFrom(other : CNumber) : CNumber {
        this.value = <Int32Array> this.sub(other).getValue();
        return this;
    }
    public mul(other : CNumber) : CNumber {
        throw "Unimplemented";
        return new Int64(0, 0);
    }
    public mulBy(other : CNumber) : CNumber {
        throw "Unimplemented";
        return this;
    }
    public div(other : CNumber) : CNumber {
        throw "Unimplemented";
        return new Int64(0, 0);
    }
    public divBy(other : CNumber) : CNumber {
        throw "Unimplemented";
        return this;
    }
    public negate() : CNumber {
        return new Int64(-this.getLow(), -this.getHigh());
    }
}
