/// <reference path="../../ref.ts" />
define(["require", "exports", "./numerics", "./integer", "./../../utils/utils"], function(require, exports, numerics, integer, utils) {
    var Int16 = (function () {
        function Int16(n) {
            var _this = this;
            this.MAX_VALUE = 32767;
            this.MIN_VALUE = -32767;
            this.KIND = 10 /* Int8 */;
            this.min = function () {
                return new Int16(_this.MIN_VALUE);
            };
            this.max = function () {
                return new Int16(_this.MAX_VALUE);
            };
            this.lowest = function () {
                return new Int16(_this.MIN_VALUE);
            };
            this.highest = function () {
                return new Int16(_this.MAX_VALUE);
            };
            this.infinity = function () {
                return new Int16(0);
            };
            this.value_ = new Uint8Array(1);
            if (n) {
                this.value_[0] = n;
            } else {
                this.value_[0] = utils.rand(this.MIN_VALUE, this.MAX_VALUE);
            }
        }
        Int16.prototype.getValue = function () {
            return this.value_;
        };

        Int16.prototype.add = function (other) {
            if (other.KIND <= this.KIND) {
                return new Int16(this.value_[0] + other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] + other.getValue()[0]);
        };

        Int16.prototype.addTo = function (other) {
            this.value_[0] += other.getValue()[0];
            return this;
        };

        Int16.prototype.sub = function (other) {
            if (other.KIND <= this.KIND) {
                return new Int16(this.value_[0] - other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] - other.getValue()[0]);
        };

        Int16.prototype.subFrom = function (other) {
            this.value_[0] -= other.getValue()[0];
            return this;
        };

        Int16.prototype.mul = function (other) {
            if (other.KIND <= this.KIND) {
                return new Int16(this.value_[0] * other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] * other.getValue()[0]);
        };

        Int16.prototype.mulBy = function (other) {
            this.value_[0] *= other.getValue()[0];
            return this;
        };

        Int16.prototype.div = function (other) {
            if (other.KIND <= this.KIND) {
                return new Int16(this.value_[0] / other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] / other.getValue()[0]);
        };

        Int16.prototype.divBy = function (other) {
            this.value_[0] /= other.getValue()[0];
            return this;
        };

        Int16.prototype.negate = function () {
            return new Int16(-this.value_[0]);
        };
        Int16.prototype.value = function () {
            return this.value_[0];
        };
        return Int16;
    })();

    numerics.CNumberKindMap.set(20 /* Int16 */, Int16);
    utils.applyMixins(Int16, [integer.IntegerTraits, integer.SignedIntegerTraits]);

    
    return Int16;
});
//# sourceMappingURL=int16.js.map
