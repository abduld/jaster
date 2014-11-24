/// <reference path="../../ref.ts" />
/// <reference path="numerics.ts" />
/// <reference path="int8.ts" />
/// <reference path="uint8.ts" />
/// <reference path="int16.ts" />
/// <reference path="uint16.ts" />
/// <reference path="int32.ts" />
/// <reference path="uint32.ts" />
/// <reference path="int64.ts" />
/// <reference path="uint64.ts" />

export class IntegerTraits {
    is_integer = () => true;
    is_exact = () => true;
    has_infinity = () => false;
    is_modulo = () => true;
}
export class SignedIntegerTraits  {
    is_signed = () => true;
}
export class UnsignedIntegerTraits  {
    is_signed = () => false;
}