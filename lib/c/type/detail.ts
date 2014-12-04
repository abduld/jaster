/// <reference path='../../utils/mixin.ts' />

module lib.c.type.detail {

    export class IntegerTraits {
        is_integer = () => true;
        is_exact = () => true;
        has_infinity = () => false;
        is_modulo = () => true;
    }
    export class SignedIntegerTraits {
        is_signed = () => true;
    }
    export class UnsignedIntegerTraits {
        is_signed = () => false;
    }

    export enum CLiteralKind {
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

    export var CLiteralKindMap:Map<CLiteralKind, any> = null;
    if (CLiteralKindMap === null) {
        CLiteralKindMap = new Map<CLiteralKind, any>();
    }

    export interface CLiteral {
        MAX_VALUE: number;
        MIN_VALUE: number;
        KIND: CLiteralKind;
        is_signed(): boolean;
        is_integer(): boolean;
        is_exact(): boolean;
        has_infinity(): boolean;
        is_modulo(): boolean;
        min(): CLiteral;
        max(): CLiteral;
        lowest(): CLiteral;
        highest(): CLiteral;
        infinity(): CLiteral;
        getValue(): ArrayBufferView;
        add(n:CLiteral): CLiteral;
        sub(n:CLiteral): CLiteral;
        mul(n:CLiteral): CLiteral;
        div(n:CLiteral): CLiteral;
        negate(): CLiteral;
    }
}