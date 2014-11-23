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
define(["require", "exports"], function (require, exports) {
    var IntegerTraits = (function () {
        function IntegerTraits() {
            this.is_integer = function () { return true; };
            this.is_exact = function () { return true; };
            this.has_infinity = function () { return false; };
            this.is_modulo = function () { return true; };
        }
        return IntegerTraits;
    })();
    exports.IntegerTraits = IntegerTraits;
    var SignedIntegerTraits = (function () {
        function SignedIntegerTraits() {
            this.is_signed = function () { return true; };
        }
        return SignedIntegerTraits;
    })();
    exports.SignedIntegerTraits = SignedIntegerTraits;
    var UnsignedIntegerTraits = (function () {
        function UnsignedIntegerTraits() {
            this.is_signed = function () { return false; };
        }
        return UnsignedIntegerTraits;
    })();
    exports.UnsignedIntegerTraits = UnsignedIntegerTraits;
});
//# sourceMappingURL=integer.js.map