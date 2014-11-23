/// <reference path="../../ref.ts" />
define(["require", "exports", "numerics", "integer", "../../utils/utils"], function (require, exports, numerics, integer, utils) {
    var Int8 = (function () {
        function Int8(n) {
            var _this = this;
            this.MAX_VALUE = 128;
            this.MIN_VALUE = -128;
            this.KIND = 10 /* Int8 */;
            this.min = function () { return new Int8(_this.MIN_VALUE); };
            this.max = function () { return new Int8(_this.MAX_VALUE); };
            this.lowest = function () { return new Int8(_this.MIN_VALUE); };
            this.highest = function () { return new Int8(_this.MAX_VALUE); };
            this.infinity = function () { return new Int8(0); };
            this.value_ = new Int8Array(1);
            if (n) {
                this.value_[0] = n;
            }
            else {
                this.value_[0] = utils.rand(this.MIN_VALUE, this.MAX_VALUE);
            }
        }
        Int8.prototype.getValue = function () {
            return this.value_;
        };
        Int8.prototype.add = function (other) {
            if (other.KIND <= this.KIND) {
                return new Int8(this.value_[0] + other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] + other.getValue()[0]);
        };
        Int8.prototype.addTo = function (other) {
            this.value_[0] += other.getValue()[0];
            return this;
        };
        Int8.prototype.sub = function (other) {
            if (other.KIND <= this.KIND) {
                return new Int8(this.value_[0] - other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] - other.getValue()[0]);
        };
        Int8.prototype.subFrom = function (other) {
            this.value_[0] -= other.getValue()[0];
            return this;
        };
        Int8.prototype.mul = function (other) {
            if (other.KIND <= this.KIND) {
                return new Int8(this.value_[0] * other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] * other.getValue()[0]);
        };
        Int8.prototype.mulBy = function (other) {
            this.value_[0] *= other.getValue()[0];
            return this;
        };
        Int8.prototype.div = function (other) {
            if (other.KIND <= this.KIND) {
                return new Int8(this.value_[0] / other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] / other.getValue()[0]);
        };
        Int8.prototype.divBy = function (other) {
            this.value_[0] /= other.getValue()[0];
            return this;
        };
        Int8.prototype.negate = function () {
            return new Int8(-this.value_[0]);
        };
        Int8.prototype.value = function () {
            return this.value_[0];
        };
        return Int8;
    })();
    numerics.CNumberKindMap.set(10 /* Int8 */, Int8);
    utils.applyMixins(Int8, [integer.IntegerTraits, integer.SignedIntegerTraits]);
    return Int8;
});
//# sourceMappingURL=int8.js.map