/// <reference path="../../ref.ts" />
define(["require", "exports", "./numerics", "./integer", "./../../utils/utils"], function (require, exports, numerics, integer, utils) {
    var MAX_VALUE = 255;
    var MIN_VALUE = 0;
    var Uint8 = (function () {
        function Uint8(n) {
            this.value_ = new Uint8Array(1);
            if (n) {
                this.value_[0] = n;
            }
            else {
                this.value_[0] = utils.rand(this.MIN_VALUE, this.MAX_VALUE);
            }
        }
        Uint8.prototype.getValue = function () {
            return this.value_;
        };
        Uint8.prototype.add = function (other) {
            if (other.KIND <= this.KIND) {
                return (new Uint8(this.value_[0] + other.getValue()[0]));
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] + other.getValue()[0]);
        };
        Uint8.prototype.addTo = function (other) {
            this.value_[0] += other.getValue()[0];
            return this;
        };
        Uint8.prototype.sub = function (other) {
            if (other.KIND <= this.KIND) {
                return new Uint8(this.value_[0] - other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] - other.getValue()[0]);
        };
        Uint8.prototype.subFrom = function (other) {
            this.value_[0] -= other.getValue()[0];
            return this;
        };
        Uint8.prototype.mul = function (other) {
            if (other.KIND <= this.KIND) {
                return new Uint8(this.value_[0] * other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] * other.getValue()[0]);
        };
        Uint8.prototype.mulBy = function (other) {
            this.value_[0] *= other.getValue()[0];
            return this;
        };
        Uint8.prototype.div = function (other) {
            if (other.KIND <= this.KIND) {
                return new Uint8(this.value_[0] / other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] / other.getValue()[0]);
        };
        Uint8.prototype.divBy = function (other) {
            this.value_[0] /= other.getValue()[0];
            return this;
        };
        Uint8.prototype.negate = function () {
            return new Uint8(-this.value_[0]);
        };
        Uint8.prototype.value = function () {
            return this.value_[0];
        };
        Uint8.KIND = 11 /* Uint8 */;
        Uint8.MAX_VALUE = MAX_VALUE;
        Uint8.MIN_VALUE = MIN_VALUE;
        Uint8.min = function () { return new Uint8(MIN_VALUE); };
        Uint8.max = function () { return new Uint8(MAX_VALUE); };
        Uint8.lowest = function () { return new Uint8(MIN_VALUE); };
        Uint8.highest = function () { return new Uint8(MAX_VALUE); };
        Uint8.infinity = function () { return new Uint8(0); };
        return Uint8;
    })();
    numerics.CNumberKindMap.set(11 /* Uint8 */, Uint8);
    utils.applyMixins(Uint8, [integer.IntegerTraits, integer.UnsignedIntegerTraits]);
    return Uint8;
});
