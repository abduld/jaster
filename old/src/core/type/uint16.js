/// <reference path="../../ref.ts" />
define(["require", "exports", "./numerics", "./integer", "./../../utils/utils"], function(require, exports, numerics, integer, utils) {
    var Uint16 = (function () {
        function Uint16(n) {
            var _this = this;
            this.MAX_VALUE = 65535;
            this.MIN_VALUE = 0;
            this.KIND = 11 /* Uint8 */;
            this.min = function () {
                return new Uint16(_this.MIN_VALUE);
            };
            this.max = function () {
                return new Uint16(_this.MAX_VALUE);
            };
            this.lowest = function () {
                return new Uint16(_this.MIN_VALUE);
            };
            this.highest = function () {
                return new Uint16(_this.MAX_VALUE);
            };
            this.infinity = function () {
                return new Uint16(0);
            };
            this.value_ = new Uint8Array(1);
            if (n) {
                this.value_[0] = n;
            } else {
                this.value_[0] = utils.rand(this.MIN_VALUE, this.MAX_VALUE);
            }
        }
        Uint16.prototype.getValue = function () {
            return this.value_;
        };

        Uint16.prototype.add = function (other) {
            if (other.KIND <= this.KIND) {
                return new Uint16(this.value_[0] + other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] + other.getValue()[0]);
        };

        Uint16.prototype.addTo = function (other) {
            this.value_[0] += other.getValue()[0];
            return this;
        };

        Uint16.prototype.sub = function (other) {
            if (other.KIND <= this.KIND) {
                return new Uint16(this.value_[0] - other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] - other.getValue()[0]);
        };

        Uint16.prototype.subFrom = function (other) {
            this.value_[0] -= other.getValue()[0];
            return this;
        };

        Uint16.prototype.mul = function (other) {
            if (other.KIND <= this.KIND) {
                return new Uint16(this.value_[0] * other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] * other.getValue()[0]);
        };

        Uint16.prototype.mulBy = function (other) {
            this.value_[0] *= other.getValue()[0];
            return this;
        };

        Uint16.prototype.div = function (other) {
            if (other.KIND <= this.KIND) {
                return new Uint16(this.value_[0] / other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] / other.getValue()[0]);
        };

        Uint16.prototype.divBy = function (other) {
            this.value_[0] /= other.getValue()[0];
            return this;
        };

        Uint16.prototype.negate = function () {
            return new Uint16(-this.value_[0]);
        };
        Uint16.prototype.value = function () {
            return this.value_[0];
        };
        return Uint16;
    })();
    numerics.CNumberKindMap.set(21 /* Uint16 */, Uint16);
    utils.applyMixins(Uint16, [integer.IntegerTraits, integer.UnsignedIntegerTraits]);
    
    return Uint16;
});
//# sourceMappingURL=uint16.js.map
