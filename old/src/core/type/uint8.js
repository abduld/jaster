/// <reference path="../../ref.ts" />
define(["require", "exports", "./numerics", "./integer", "./../../utils/utils"], function(require, exports, numerics, integer, utils) {
    var Uint8 = (function () {
        function Uint8(n) {
            var _this = this;
            this.MAX_VALUE = 255;
            this.MIN_VALUE = 0;
            this.KIND = 11 /* Uint8 */;
            this.min = function () {
                return new Uint8(_this.MIN_VALUE);
            };
            this.max = function () {
                return new Uint8(_this.MAX_VALUE);
            };
            this.lowest = function () {
                return new Uint8(_this.MIN_VALUE);
            };
            this.highest = function () {
                return new Uint8(_this.MAX_VALUE);
            };
            this.infinity = function () {
                return new Uint8(0);
            };
            this.value_ = new Uint8Array(1);
            if (n) {
                this.value_[0] = n;
            } else {
                this.value_[0] = utils.rand(this.MIN_VALUE, this.MAX_VALUE);
            }
        }
        Uint8.prototype.getValue = function () {
            return this.value_;
        };

        Uint8.prototype.add = function (other) {
            if (other.KIND <= this.KIND) {
                return new Uint8(this.value_[0] + other.getValue()[0]);
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
        return Uint8;
    })();
    numerics.CNumberKindMap.set(11 /* Uint8 */, Uint8);
    utils.applyMixins(Uint8, [integer.IntegerTraits, integer.UnsignedIntegerTraits]);
    
    return Uint8;
});
//# sourceMappingURL=uint8.js.map
