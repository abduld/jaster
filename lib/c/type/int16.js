/// <reference path='../../utils/utils.ts' />
/// <reference path='detail.ts' />
var lib;
(function (lib) {
    var c;
    (function (c) {
        var type;
        (function (type) {
            var utils = lib.utils;
            var detail;
            (function (detail) {
                var Int16 = (function () {
                    function Int16(n) {
                        var _this = this;
                        this.MAX_VALUE = 3276;
                        this.MIN_VALUE = -3276;
                        this.KIND = 20 /* Int16 */;
                        this.min = function () { return new Int16(_this.MIN_VALUE); };
                        this.max = function () { return new Int16(_this.MAX_VALUE); };
                        this.lowest = function () { return new Int16(_this.MIN_VALUE); };
                        this.highest = function () { return new Int16(_this.MAX_VALUE); };
                        this.infinity = function () { return new Int16(0); };
                        this.value_ = new Int16Array(1);
                        if (n) {
                            this.value_[0] = n;
                        }
                        else {
                            this.value_[0] = utils.rand(this.MIN_VALUE, this.MAX_VALUE);
                        }
                    }
                    Int16.prototype.getValue = function () {
                        return this.value_;
                    };
                    Int16.prototype.add = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Int16(this.value_[0] + other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] + other.getValue()[0]);
                    };
                    Int16.prototype.addTo = function (other) {
                        this.value_[0] += other.getValue()[0];
                        return utils.castTo(this);
                    };
                    Int16.prototype.sub = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Int16(this.value_[0] - other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] - other.getValue()[0]);
                    };
                    Int16.prototype.subFrom = function (other) {
                        this.value_[0] -= other.getValue()[0];
                        return utils.castTo(this);
                    };
                    Int16.prototype.mul = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Int16(this.value_[0] * other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] * other.getValue()[0]);
                    };
                    Int16.prototype.mulBy = function (other) {
                        this.value_[0] *= other.getValue()[0];
                        return utils.castTo(this);
                    };
                    Int16.prototype.div = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Int16(this.value_[0] / other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] / other.getValue()[0]);
                    };
                    Int16.prototype.divBy = function (other) {
                        this.value_[0] /= other.getValue()[0];
                        return utils.castTo(this);
                    };
                    Int16.prototype.negate = function () {
                        return utils.castTo(new Int16(-this.value_[0]));
                    };
                    Int16.prototype.value = function () {
                        return this.value_[0];
                    };
                    return Int16;
                })();
                detail.Int16 = Int16;
                detail.CLiteralKindMap.set(20 /* Int16 */, Int16);
                utils.applyMixins(Int16, [detail.IntegerTraits, detail.SignedIntegerTraits]);
            })(detail = type.detail || (type.detail = {}));
            type.Int16 = detail.Int16;
        })(type = c.type || (c.type = {}));
    })(c = lib.c || (lib.c = {}));
})(lib || (lib = {}));
