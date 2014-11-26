/// <reference path="../../ref.ts" />
define(["require", "exports", "./numerics", "./integer", "./../../utils/utils"], function (require, exports, numerics, integer, utils) {
    var CNumberKind = numerics.CNumberKind;
    var CNumberKindMap = numerics.CNumberKindMap;
    var Int16 = (function () {
        function Int16(n) {
            this._value = new Uint8Array(1);
            if (n) {
                this._value[0] = n;
            }
            else {
                this._value[0] = utils.rand(Int16.MIN_VALUE, Int16.MAX_VALUE);
            }
        }
        Int16.prototype.getValue = function () {
            return this._value;
        };
        Int16.prototype.add = function (other) {
            if (other.KIND <= Int16.KIND) {
                return utils.castTo(new Int16(this._value[0] + other.getValue()[0]));
            }
            var typ = CNumberKindMap.get(other.KIND);
            return new typ(this._value[0] + other.getValue()[0]);
        };
        Int16.prototype.addTo = function (other) {
            this._value[0] += other.getValue()[0];
            return utils.castTo;
        };
        Int16.prototype.sub = function (other) {
            if (other.KIND <= Int16.KIND) {
                return new Int16(this._value[0] - other.getValue()[0]);
            }
            var typ = CNumberKindMap.get(other.KIND);
            return new typ(this._value[0] - other.getValue()[0]);
        };
        Int16.prototype.subFrom = function (other) {
            this._value[0] -= other.getValue()[0];
            return this;
        };
        Int16.prototype.mul = function (other) {
            if (other.KIND <= Int16.KIND) {
                return new Int16(this._value[0] * other.getValue()[0]);
            }
            var typ = CNumberKindMap.get(other.KIND);
            return new typ(this._value[0] * other.getValue()[0]);
        };
        Int16.prototype.mulBy = function (other) {
            this._value[0] *= other.getValue()[0];
            return this;
        };
        Int16.prototype.div = function (other) {
            if (other.KIND <= Int16.KIND) {
                return new Int16(this._value[0] / other.getValue()[0]);
            }
            var typ = CNumberKindMap.get(other.KIND);
            return new typ(this._value[0] / other.getValue()[0]);
        };
        Int16.prototype.divBy = function (other) {
            this._value[0] /= other.getValue()[0];
            return this;
        };
        Int16.prototype.negate = function () {
            return new Int16(-this._value[0]);
        };
        Int16.prototype.value = function () {
            return this._value[0];
        };
        Int16.MAX_VALUE = 32767;
        Int16.MIN_VALUE = -32767;
        Int16.KIND = 10 /* Int8 */;
        Int16.min = function () { return new Int16(Int16.MIN_VALUE); };
        Int16.max = function () { return new Int16(Int16.MAX_VALUE); };
        Int16.lowest = function () { return new Int16(Int16.MIN_VALUE); };
        Int16.highest = function () { return new Int16(Int16.MAX_VALUE); };
        Int16.infinity = function () { return new Int16(0); };
        return Int16;
    })();
    CNumberKindMap.set(20 /* Int16 */, Int16);
    utils.applyMixins(Int16, [integer.IntegerTraits, integer.SignedIntegerTraits]);
    return Int16;
});
