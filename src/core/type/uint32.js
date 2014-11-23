/// <reference path="../../ref.ts" />
define(["require", "exports", "numerics", "integer", "../../utils/utils"], function (require, exports, numerics, integer, utils) {
    var Uint32 = (function () {
        function Uint32(n) {
            var _this = this;
            this.MAX_VALUE = 4294967295;
            this.MIN_VALUE = 0;
            this.KIND = 11 /* Uint8 */;
            this.min = function () { return new Uint32(_this.MIN_VALUE); };
            this.max = function () { return new Uint32(_this.MAX_VALUE); };
            this.lowest = function () { return new Uint32(_this.MIN_VALUE); };
            this.highest = function () { return new Uint32(_this.MAX_VALUE); };
            this.infinity = function () { return new Uint32(0); };
            this.value_ = new Uint8Array(1);
            if (n) {
                this.value_[0] = n;
            }
            else {
                this.value_[0] = utils.rand(this.MIN_VALUE, this.MAX_VALUE);
            }
        }
        Uint32.prototype.getValue = function () {
            return this.value_;
        };
        Uint32.prototype.add = function (other) {
            if (other.KIND <= this.KIND) {
                return new Uint32(this.value_[0] + other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] + other.getValue()[0]);
        };
        Uint32.prototype.addTo = function (other) {
            this.value_[0] += other.getValue()[0];
            return this;
        };
        Uint32.prototype.sub = function (other) {
            if (other.KIND <= this.KIND) {
                return new Uint32(this.value_[0] - other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] - other.getValue()[0]);
        };
        Uint32.prototype.subFrom = function (other) {
            this.value_[0] -= other.getValue()[0];
            return this;
        };
        Uint32.prototype.mul = function (other) {
            if (other.KIND <= this.KIND) {
                return new Uint32(this.value_[0] * other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] * other.getValue()[0]);
        };
        Uint32.prototype.mulBy = function (other) {
            this.value_[0] *= other.getValue()[0];
            return this;
        };
        Uint32.prototype.div = function (other) {
            if (other.KIND <= this.KIND) {
                return new Uint32(this.value_[0] / other.getValue()[0]);
            }
            var typ = numerics.CNumberKindMap.get(other.KIND);
            return new typ(this.value_[0] / other.getValue()[0]);
        };
        Uint32.prototype.divBy = function (other) {
            this.value_[0] /= other.getValue()[0];
            return this;
        };
        Uint32.prototype.negate = function () {
            return new Uint32(-this.value_[0]);
        };
        Uint32.prototype.value = function () {
            return this.value_[0];
        };
        return Uint32;
    })();
    numerics.CNumberKindMap.set(31 /* Uint32 */, Uint32);
    utils.applyMixins(Uint32, [integer.IntegerTraits, integer.UnsignedIntegerTraits]);
    return Uint32;
});
//# sourceMappingURL=uint32.js.map