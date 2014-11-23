/// <reference path="../../ref.ts" />
define(["require", "exports", "numerics", "integer", "../../utils/utils"], function (require, exports, numerics, integer, utils) {
    var Int32 = (function () {
        function Int32(n) {
            var _this = this;
            this.MAX_VALUE = 2147483648;
            this.MIN_VALUE = -2147483648;
            this.KIND = 30 /* Int32 */;
            this.min = function () { return new Int32(_this.MIN_VALUE); };
            this.max = function () { return new Int32(_this.MAX_VALUE); };
            this.lowest = function () { return new Int32(_this.MIN_VALUE); };
            this.highest = function () { return new Int32(_this.MAX_VALUE); };
            this.infinity = function () { return new Int32(0); };
            this.value_ = new Int8Array(1);
            if (n) {
                this.value_[0] = n;
            }
            else {
                this.value_[0] = utils.rand(this.MIN_VALUE, this.MAX_VALUE);
            }
        }
        Int32.prototype.getValue = function () {
            return this.value_;
        };
        Int32.prototype.add = function (other) {
            if (other.KIND <= this.KIND) {
                return new Int32(this.value_[0] + other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] + other.getValue()[0]);
        };
        Int32.prototype.addTo = function (other) {
            this.value_[0] += other.getValue()[0];
            return this;
        };
        Int32.prototype.sub = function (other) {
            if (other.KIND <= this.KIND) {
                return new Int32(this.value_[0] - other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] - other.getValue()[0]);
        };
        Int32.prototype.subFrom = function (other) {
            this.value_[0] -= other.getValue()[0];
            return this;
        };
        Int32.prototype.mul = function (other) {
            if (other.KIND <= this.KIND) {
                return new Int32(this.value_[0] * other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] * other.getValue()[0]);
        };
        Int32.prototype.mulBy = function (other) {
            this.value_[0] *= other.getValue()[0];
            return this;
        };
        Int32.prototype.div = function (other) {
            if (other.KIND <= this.KIND) {
                return new Int32(this.value_[0] / other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] / other.getValue()[0]);
        };
        Int32.prototype.divBy = function (other) {
            this.value_[0] /= other.getValue()[0];
            return this;
        };
        Int32.prototype.negate = function () {
            return new Int32(-this.value_[0]);
        };
        Int32.prototype.value = function () {
            return this.value_[0];
        };
        return Int32;
    })();
    numerics.CNumberKindMap.set(30 /* Int32 */, Int32);
    utils.applyMixins(Int32, [integer.IntegerTraits, integer.SignedIntegerTraits]);
    return Int32;
});
//# sourceMappingURL=int32.js.map