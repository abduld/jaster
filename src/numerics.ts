
/// <reference path="ref.ts" />
module Core {
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

    export var CNumberKindMap:Map<CNumberKind, any> = new Map<CNumberKind, any>();

    export interface CNumber {
        MAX_VALUE : number;
        MIN_VALUE : number;
        KIND : CNumberKind;
        is_signed() : boolean;
        is_integer() : boolean;
        is_exact() : boolean;
        has_infinity() : boolean;
        is_modulo() : boolean;
        min() : CNumber;
        max() : CNumber;
        lowest() : CNumber;
        highest() : CNumber;
        infinity() : CNumber;
        getValue() : ArrayBufferView;
        add(n:CNumber) : CNumber;
        sub(n:CNumber) : CNumber;
        mul(n:CNumber) : CNumber;
        div(n:CNumber) : CNumber;
        negate() : CNumber;
    }
}
export = Core;