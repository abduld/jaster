/// <reference path="../../ref.ts" />
export enum CNumberKind {
    Int8 = 10,
    Uint8 = 11,
    Int16 = 20,
    Uint16 = 21,
    Int32 = 30,
    Uint32 = 31,
    Int64 = 40,
    Float = 52,
    Double = 62
}

export var CNumberKindMap:Map<CNumberKind, any> = null;
if (CNumberKindMap === null) {
    CNumberKindMap = new Map<CNumberKind, any>();
}

export interface CNumber {
    static MAX_VALUE : number;
    static MIN_VALUE : number;
    static KIND : CNumberKind;
    static is_signed() : boolean;
    static is_integer() : boolean;
    static is_exact() : boolean;
    static has_infinity() : boolean;
    static is_modulo() : boolean;
    static min() : CNumber;
    static max() : CNumber;
    static lowest() : CNumber;
    static highest() : CNumber;
    static infinity() : CNumber;
    getValue() : ArrayBufferView;
    add(n:CNumber) : CNumber;
    sub(n:CNumber) : CNumber;
    mul(n:CNumber) : CNumber;
    div(n:CNumber) : CNumber;
    negate() : CNumber;
}


import int8_ = require("./int8");
import uint8_ = require("./uint8");
import int16_ = require("./int16");
import uint16_ = require("./uint16");
import int32_ = require("./int32");
import uint32_ = require("./uint32");
import int64_ = require("./int64");

export var int8 = int8_;
export var uint8 = uint8_;
export var int16 = int16_;
export var uint16 = uint16_;
export var int32 = int32_;
export var uint32 = uint32_;
export var int64 = int64_;
