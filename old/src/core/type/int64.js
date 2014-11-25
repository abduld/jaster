/// <reference path="../../ref.ts" />
define(["require", "exports", "./numerics", "./integer", "./../../utils/utils", "./int32", "./uint32"], function(require, exports, numerics, integer, utils, Int32, Uint32) {
    // We now hit the problem of numerical representation
    // this needs to be reddone, but this will serve as a template
    var Int64 = (function () {
        function Int64(low, high) {
            var _this = this;
            this.MAX_VALUE = NaN;
            this.MIN_VALUE = NaN;
            this.KIND = 30 /* Int32 */;
            this.min = function () {
                return new Int64(_this.MIN_VALUE);
            };
            this.max = function () {
                return new Int64(_this.MAX_VALUE);
            };
            this.lowest = function () {
                return new Int64(_this.MIN_VALUE);
            };
            this.highest = function () {
                return new Int64(_this.MAX_VALUE);
            };
            this.infinity = function () {
                return new Int64(0);
            };
            this.value = new Int32Array(2);
            if (low && high) {
                this.value[0] = low;
                this.value[1] = high;
            } else {
                this.value[0] = (new Int32()).getValue()[0];
                this.value[1] = (new Int32()).getValue()[0];
            }
        }
        Int64.prototype.getLow = function () {
            return this.value[0];
        };

        Int64.prototype.getHigh = function () {
            return this.value[1];
        };

        Int64.prototype.getValue = function () {
            return this.value;
        };

        // lifted from
        // http://docs.closure-library.googlecode.com/git/local_closure_goog_math_long.js.source.html
        Int64.prototype.add = function (other) {
            if (other.KIND === this.KIND) {
                var o = other;
                var a48 = this.getHigh() >>> 16;
                var a32 = this.getHigh() & 0xFFFF;
                var a16 = this.getLow() >>> 16;
                var a00 = this.getLow() & 0xFFFF;

                var b48 = o.getHigh() >>> 16;
                var b32 = o.getHigh() & 0xFFFF;
                var b16 = o.getLow() >>> 16;
                var b00 = o.getLow() & 0xFFFF;

                var c48 = 0, c32 = 0;
                var c16 = 0, c00 = 0;
                c00 += a00 + b00;
                c16 += c00 >>> 16;
                c00 &= 0xFFFF;
                c16 += a16 + b16;
                c32 += c16 >>> 16;
                c16 &= 0xFFFF;
                c32 += a32 + b32;
                c48 += c32 >>> 16;
                c32 &= 0xFFFF;
                c48 += a48 + b48;
                c48 &= 0xFFFF;

                return new Int64((c16 << 16) | c00, (c48 << 16) | c32);
            }
            var low = new Uint32(((new Uint32(this.getLow())).add(other.getValue()[0])).getValue()[0]);
            var high = new Uint32(((new Uint32(this.getHigh())).add(new Uint32(low.getValue()[0] >> 31))).getValue()[0]);
            return new Int64(low.getValue()[0] & 0x7FFFFFFF, high.getValue()[0]);
        };

        Int64.prototype.addTo = function (other) {
            this.value = this.add(other).getValue();
            return this;
        };

        Int64.prototype.sub = function (other) {
            return this.add(other.negate());
        };

        Int64.prototype.subFrom = function (other) {
            this.value = this.sub(other).getValue();
            return this;
        };

        Int64.prototype.mul = function (other) {
            throw "Unimplemented";
            return new Int64(0, 0);
        };

        Int64.prototype.mulBy = function (other) {
            throw "Unimplemented";
            return this;
        };

        Int64.prototype.div = function (other) {
            throw "Unimplemented";
            return new Int64(0, 0);
        };

        Int64.prototype.divBy = function (other) {
            throw "Unimplemented";
            return this;
        };

        Int64.prototype.negate = function () {
            return new Int64(-this.getLow(), -this.getHigh());
        };
        return Int64;
    })();
    numerics.CNumberKindMap.set(40 /* Int64 */, Int64);
    utils.applyMixins(Int64, [integer.IntegerTraits, integer.SignedIntegerTraits]);

    
    return Int64;
});
//# sourceMappingURL=int64.js.map
