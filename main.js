var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var detail;
        (function (detail) {
            var LogType;
            (function (LogType) {
                LogType[LogType["Debug"] = 0] = "Debug";
                LogType[LogType["Trace"] = 1] = "Trace";
                LogType[LogType["Warn"] = 2] = "Warn";
                LogType[LogType["Error"] = 3] = "Error";
                LogType[LogType["Fatal"] = 4] = "Fatal";
            })(LogType || (LogType = {}));
            var Logger = (function () {
                function Logger(level) {
                    if (level) {
                        this._level = level;
                    }
                    else {
                        this._level = 0 /* Debug */;
                    }
                }
                Logger.prototype._go = function (msg, type) {
                    var color = {
                        "LogType.Debug": '\033[39m',
                        "LogType.Trace": '\033[39m',
                        "LogType.Warn": '\033[33m',
                        "LogType.Error": '\033[33m',
                        "LogType.Fatal": '\033[31m'
                    };
                    if (type >= this._level) {
                        console[type](color[type.toString()] + msg + color["LogType.Debug"]);
                    }
                };
                Logger.prototype.debug = function (msg) {
                    this._go(msg, 0 /* Debug */);
                };
                Logger.prototype.trace = function (msg) {
                    this._go(msg, 1 /* Trace */);
                };
                Logger.prototype.warn = function (msg) {
                    this._go(msg, 2 /* Warn */);
                };
                Logger.prototype.error = function (msg) {
                    this._go(msg, 3 /* Error */);
                };
                Logger.prototype.fatal = function (msg) {
                    this._go(msg, 4 /* Fatal */);
                };
                return Logger;
            })();
            detail.Logger = Logger;
        })(detail = utils.detail || (utils.detail = {}));
        utils.logger = new detail.Logger();
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
/// <reference path="logger.ts" />
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var detail;
        (function (detail) {
            var Logger = lib.utils.detail.Logger;
            var logger = new Logger();
            function assert(res, msg) {
                if (!res) {
                    logger.error('FAIL: ' + msg);
                }
                else {
                    logger.debug('Pass: ' + msg);
                }
            }
            detail.assert = assert;
        })(detail = utils.detail || (utils.detail = {}));
        utils.assert = detail.assert;
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        function rand(min, max) {
            return min + Math.random() * (max - min);
        }
        utils.rand = rand;
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        function applyMixins(derivedCtor, baseCtors) {
            baseCtors.forEach(function (baseCtor) {
                Object.getOwnPropertyNames(baseCtor.prototype).forEach(function (name) {
                    derivedCtor.prototype[name] = baseCtor.prototype[name];
                });
            });
        }
        utils.applyMixins = applyMixins;
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
/// <reference path='../../utils/mixin.ts' />
var lib;
(function (lib) {
    var c;
    (function (c) {
        var type;
        (function (type) {
            var detail;
            (function (detail) {
                var IntegerTraits = (function () {
                    function IntegerTraits() {
                        this.is_integer = function () { return true; };
                        this.is_exact = function () { return true; };
                        this.has_infinity = function () { return false; };
                        this.is_modulo = function () { return true; };
                    }
                    return IntegerTraits;
                })();
                detail.IntegerTraits = IntegerTraits;
                var SignedIntegerTraits = (function () {
                    function SignedIntegerTraits() {
                        this.is_signed = function () { return true; };
                    }
                    return SignedIntegerTraits;
                })();
                detail.SignedIntegerTraits = SignedIntegerTraits;
                var UnsignedIntegerTraits = (function () {
                    function UnsignedIntegerTraits() {
                        this.is_signed = function () { return false; };
                    }
                    return UnsignedIntegerTraits;
                })();
                detail.UnsignedIntegerTraits = UnsignedIntegerTraits;
                (function (CLiteralKind) {
                    CLiteralKind[CLiteralKind["Int8"] = 10] = "Int8";
                    CLiteralKind[CLiteralKind["Uint8"] = 11] = "Uint8";
                    CLiteralKind[CLiteralKind["Int16"] = 20] = "Int16";
                    CLiteralKind[CLiteralKind["Uint16"] = 21] = "Uint16";
                    CLiteralKind[CLiteralKind["Int32"] = 30] = "Int32";
                    CLiteralKind[CLiteralKind["Uint32"] = 31] = "Uint32";
                    CLiteralKind[CLiteralKind["Int64"] = 40] = "Int64";
                    CLiteralKind[CLiteralKind["Float"] = 52] = "Float";
                    CLiteralKind[CLiteralKind["Double"] = 62] = "Double";
                })(detail.CLiteralKind || (detail.CLiteralKind = {}));
                var CLiteralKind = detail.CLiteralKind;
                detail.CLiteralKindMap = null;
                if (detail.CLiteralKindMap === null) {
                    detail.CLiteralKindMap = new Map();
                }
            })(detail = type.detail || (type.detail = {}));
        })(type = c.type || (c.type = {}));
    })(c = lib.c || (lib.c = {}));
})(lib || (lib = {}));
/// <reference path='../../utils/utils.ts' />
/// <reference path='detail.ts' />
var lib;
(function (lib) {
    var c;
    (function (c) {
        var type;
        (function (type) {
            var detail;
            (function (detail) {
                var utils = lib.utils;
                var Uint8 = (function () {
                    function Uint8(n) {
                        var _this = this;
                        this.MAX_VALUE = 255;
                        this.MIN_VALUE = 0;
                        this.KIND = 11 /* Uint8 */;
                        this.min = function () { return new Uint8(_this.MIN_VALUE); };
                        this.max = function () { return new Uint8(_this.MAX_VALUE); };
                        this.lowest = function () { return new Uint8(_this.MIN_VALUE); };
                        this.highest = function () { return new Uint8(_this.MAX_VALUE); };
                        this.infinity = function () { return new Uint8(0); };
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
                            return utils.castTo(new Uint8(this.value_[0] + other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] + other.getValue()[0]);
                    };
                    Uint8.prototype.addTo = function (other) {
                        this.value_[0] += other.getValue()[0];
                        return this;
                    };
                    Uint8.prototype.sub = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Uint8(this.value_[0] - other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] - other.getValue()[0]);
                    };
                    Uint8.prototype.subFrom = function (other) {
                        this.value_[0] -= other.getValue()[0];
                        return this;
                    };
                    Uint8.prototype.mul = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Uint8(this.value_[0] * other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] * other.getValue()[0]);
                    };
                    Uint8.prototype.mulBy = function (other) {
                        this.value_[0] *= other.getValue()[0];
                        return this;
                    };
                    Uint8.prototype.div = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Uint8(this.value_[0] / other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] / other.getValue()[0]);
                    };
                    Uint8.prototype.divBy = function (other) {
                        this.value_[0] /= other.getValue()[0];
                        return this;
                    };
                    Uint8.prototype.negate = function () {
                        return utils.castTo(new Uint8(-this.value_[0]));
                    };
                    Uint8.prototype.value = function () {
                        return this.value_[0];
                    };
                    return Uint8;
                })();
                detail.Uint8 = Uint8;
                detail.CLiteralKindMap.set(11 /* Uint8 */, Uint8);
                utils.applyMixins(Uint8, [detail.IntegerTraits, detail.UnsignedIntegerTraits]);
            })(detail = type.detail || (type.detail = {}));
            type.Uint8 = detail.Uint8;
        })(type = c.type || (c.type = {}));
    })(c = lib.c || (lib.c = {}));
})(lib || (lib = {}));
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
                var Uint16 = (function () {
                    function Uint16(n) {
                        var _this = this;
                        this.MAX_VALUE = 65535;
                        this.MIN_VALUE = 0;
                        this.KIND = 21 /* Uint16 */;
                        this.min = function () { return new Uint16(_this.MIN_VALUE); };
                        this.max = function () { return new Uint16(_this.MAX_VALUE); };
                        this.lowest = function () { return new Uint16(_this.MIN_VALUE); };
                        this.highest = function () { return new Uint16(_this.MAX_VALUE); };
                        this.infinity = function () { return new Uint16(0); };
                        this.value_ = new Uint16Array(1);
                        if (n) {
                            this.value_[0] = n;
                        }
                        else {
                            this.value_[0] = utils.rand(this.MIN_VALUE, this.MAX_VALUE);
                        }
                    }
                    Uint16.prototype.getValue = function () {
                        return this.value_;
                    };
                    Uint16.prototype.add = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Uint16(this.value_[0] + other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] + other.getValue()[0]);
                    };
                    Uint16.prototype.addTo = function (other) {
                        this.value_[0] += other.getValue()[0];
                        return this;
                    };
                    Uint16.prototype.sub = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Uint16(this.value_[0] - other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] - other.getValue()[0]);
                    };
                    Uint16.prototype.subFrom = function (other) {
                        this.value_[0] -= other.getValue()[0];
                        return this;
                    };
                    Uint16.prototype.mul = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Uint16(this.value_[0] * other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] * other.getValue()[0]);
                    };
                    Uint16.prototype.mulBy = function (other) {
                        this.value_[0] *= other.getValue()[0];
                        return this;
                    };
                    Uint16.prototype.div = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Uint16(this.value_[0] / other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] / other.getValue()[0]);
                    };
                    Uint16.prototype.divBy = function (other) {
                        this.value_[0] /= other.getValue()[0];
                        return this;
                    };
                    Uint16.prototype.negate = function () {
                        return utils.castTo(new Uint16(-this.value_[0]));
                    };
                    Uint16.prototype.value = function () {
                        return this.value_[0];
                    };
                    return Uint16;
                })();
                detail.Uint16 = Uint16;
                detail.CLiteralKindMap.set(21 /* Uint16 */, Uint16);
                utils.applyMixins(Uint16, [detail.IntegerTraits, detail.UnsignedIntegerTraits]);
            })(detail = type.detail || (type.detail = {}));
            type.Uint16 = detail.Uint16;
        })(type = c.type || (c.type = {}));
    })(c = lib.c || (lib.c = {}));
})(lib || (lib = {}));
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
                        this.value_ = new Int32Array(1);
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
                            return utils.castTo(new Int32(this.value_[0] + other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] + other.getValue()[0]);
                    };
                    Int32.prototype.addTo = function (other) {
                        this.value_[0] += other.getValue()[0];
                        return utils.castTo(this);
                    };
                    Int32.prototype.sub = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Int32(this.value_[0] - other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] - other.getValue()[0]);
                    };
                    Int32.prototype.subFrom = function (other) {
                        this.value_[0] -= other.getValue()[0];
                        return utils.castTo(this);
                    };
                    Int32.prototype.mul = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Int32(this.value_[0] * other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] * other.getValue()[0]);
                    };
                    Int32.prototype.mulBy = function (other) {
                        this.value_[0] *= other.getValue()[0];
                        return utils.castTo(this);
                    };
                    Int32.prototype.div = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Int32(this.value_[0] / other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] / other.getValue()[0]);
                    };
                    Int32.prototype.divBy = function (other) {
                        this.value_[0] /= other.getValue()[0];
                        return utils.castTo(this);
                    };
                    Int32.prototype.negate = function () {
                        return utils.castTo(new Int32(-this.value_[0]));
                    };
                    Int32.prototype.value = function () {
                        return this.value_[0];
                    };
                    return Int32;
                })();
                detail.Int32 = Int32;
                detail.CLiteralKindMap.set(30 /* Int32 */, Int32);
                utils.applyMixins(Int32, [detail.IntegerTraits, detail.SignedIntegerTraits]);
            })(detail = type.detail || (type.detail = {}));
            type.Int32 = detail.Int32;
        })(type = c.type || (c.type = {}));
    })(c = lib.c || (lib.c = {}));
})(lib || (lib = {}));
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
                var Uint32 = (function () {
                    function Uint32(n) {
                        var _this = this;
                        this.MAX_VALUE = 4294967295;
                        this.MIN_VALUE = 0;
                        this.KIND = 31 /* Uint32 */;
                        this.min = function () { return new Uint32(_this.MIN_VALUE); };
                        this.max = function () { return new Uint32(_this.MAX_VALUE); };
                        this.lowest = function () { return new Uint32(_this.MIN_VALUE); };
                        this.highest = function () { return new Uint32(_this.MAX_VALUE); };
                        this.infinity = function () { return new Uint32(0); };
                        this.value_ = new Uint32Array(1);
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
                            return utils.castTo(new Uint32(this.value_[0] + other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] + other.getValue()[0]);
                    };
                    Uint32.prototype.addTo = function (other) {
                        this.value_[0] += other.getValue()[0];
                        return this;
                    };
                    Uint32.prototype.sub = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Uint32(this.value_[0] - other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] - other.getValue()[0]);
                    };
                    Uint32.prototype.subFrom = function (other) {
                        this.value_[0] -= other.getValue()[0];
                        return this;
                    };
                    Uint32.prototype.mul = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Uint32(this.value_[0] * other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] * other.getValue()[0]);
                    };
                    Uint32.prototype.mulBy = function (other) {
                        this.value_[0] *= other.getValue()[0];
                        return this;
                    };
                    Uint32.prototype.div = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Uint32(this.value_[0] / other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] / other.getValue()[0]);
                    };
                    Uint32.prototype.divBy = function (other) {
                        this.value_[0] /= other.getValue()[0];
                        return this;
                    };
                    Uint32.prototype.negate = function () {
                        return utils.castTo(new Uint32(-this.value_[0]));
                    };
                    Uint32.prototype.value = function () {
                        return this.value_[0];
                    };
                    return Uint32;
                })();
                detail.Uint32 = Uint32;
                detail.CLiteralKindMap.set(31 /* Uint32 */, Uint32);
                utils.applyMixins(Uint32, [detail.IntegerTraits, detail.UnsignedIntegerTraits]);
            })(detail = type.detail || (type.detail = {}));
            type.Uint32 = detail.Uint32;
        })(type = c.type || (c.type = {}));
    })(c = lib.c || (lib.c = {}));
})(lib || (lib = {}));
/// <reference path='../../utils/utils.ts' />
/// <reference path='detail.ts' />
/// <reference path='int32.ts' />
/// <reference path='uint32.ts' />
// We now hit the problem of numerical representation
// this needs to be reddone, but this will serve as a template
var lib;
(function (lib) {
    var c;
    (function (c) {
        var type;
        (function (type) {
            var utils = lib.utils;
            var detail;
            (function (detail) {
                var Int64 = (function () {
                    function Int64(low, high) {
                        var _this = this;
                        this.MAX_VALUE = NaN;
                        this.MIN_VALUE = NaN;
                        this.KIND = 30 /* Int32 */;
                        this.min = function () { return new Int64(_this.MIN_VALUE); };
                        this.max = function () { return new Int64(_this.MAX_VALUE); };
                        this.lowest = function () { return new Int64(_this.MIN_VALUE); };
                        this.highest = function () { return new Int64(_this.MAX_VALUE); };
                        this.infinity = function () { return new Int64(0); };
                        this.value = new Int32Array(2);
                        if (low && high) {
                            this.value[0] = low;
                            this.value[1] = high;
                        }
                        else {
                            this.value[0] = (new detail.Int32()).getValue()[0];
                            this.value[1] = (new detail.Int32()).getValue()[0];
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
                        var low = new detail.Uint32(((new detail.Uint32(this.getLow())).add(other.getValue()[0])).getValue()[0]);
                        var high = new detail.Uint32(((new detail.Uint32(this.getHigh())).add(new detail.Uint32(low.getValue()[0] >> 31))).getValue()[0]);
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
                detail.Int64 = Int64;
                detail.CLiteralKindMap.set(40 /* Int64 */, Int64);
                utils.applyMixins(Int64, [detail.IntegerTraits, detail.SignedIntegerTraits]);
            })(detail = type.detail || (type.detail = {}));
            type.Int64 = detail.Int64;
        })(type = c.type || (c.type = {}));
    })(c = lib.c || (lib.c = {}));
})(lib || (lib = {}));
/// <reference path='../../utils/utils.ts' />
/// <reference path="int8.ts" />
/// <reference path="uint8.ts" />
/// <reference path="int16.ts" />
/// <reference path="uint16.ts" />
/// <reference path="int32.ts" />
/// <reference path="uint32.ts" />
/// <reference path="int64.ts" />
/// <reference path="uint64.ts" />
/// <reference path="./type/type.ts" />
var lib;
(function (lib) {
    var cuda;
    (function (cuda) {
        (function (Status) {
            Status[Status["Running"] = 0] = "Running";
            Status[Status["Idle"] = 1] = "Idle";
            Status[Status["Complete"] = 2] = "Complete";
            Status[Status["Stopped"] = 3] = "Stopped";
            Status[Status["Waiting"] = 4] = "Waiting";
        })(cuda.Status || (cuda.Status = {}));
        var Status = cuda.Status;
        var Dim3 = (function () {
            function Dim3(x, y, z) {
                if (y === void 0) { y = 1; }
                if (z === void 0) { z = 1; }
                this.x = x;
                this.y = y;
                this.z = z;
            }
            Dim3.prototype.flattenedLength = function () {
                return this.x * this.y * this.z;
            };
            Dim3.prototype.dimension = function () {
                if (this.z == 1) {
                    if (this.y == 1) {
                        return 1;
                    }
                    else {
                        return 2;
                    }
                }
                else {
                    return 3;
                }
            };
            return Dim3;
        })();
        cuda.Dim3 = Dim3;
    })(cuda = lib.cuda || (lib.cuda = {}));
})(lib || (lib = {}));
/// <reference path="../ref.ts" />
var lib;
(function (lib) {
    var parallel;
    (function (parallel) {
        (function (WorkerStatus) {
            WorkerStatus[WorkerStatus["Paused"] = 0] = "Paused";
            WorkerStatus[WorkerStatus["Idle"] = 1] = "Idle";
            WorkerStatus[WorkerStatus["Busy"] = 2] = "Busy";
            WorkerStatus[WorkerStatus["Cancel"] = 3] = "Cancel";
        })(parallel.WorkerStatus || (parallel.WorkerStatus = {}));
        var WorkerStatus = parallel.WorkerStatus;
        ;
        var INIT_PAUSE_LENGTH = 100; // milliseconds;
        var ParallelWorker = (function () {
            function ParallelWorker(fun, port) {
                this.timeout_handle = -1;
                this.id = lib.utils.guuid();
                this.status = 1 /* Idle */;
                this.master_port = port;
                this.chan = new MessageChannel();
                // Build a worker from an anonymous function body
                var blobURL = URL.createObjectURL(new Blob(['(', fun.toString(), ')()'], { type: 'application/javascript' }));
                this.worker = new Worker(blobURL);
                // Won't be needing this anymore
                URL.revokeObjectURL(blobURL);
            }
            ParallelWorker.prototype.run0 = function (init, end, inc) {
                var iter = init;
                if (this.status === 0 /* Paused */) {
                    this.pause_length *= 2;
                    setTimeout(this.run0, this.pause_length, [init, end, inc]);
                    return false;
                }
                if (this.timeout_handle !== -1) {
                    clearTimeout(this.timeout_handle);
                }
                while (iter < end) {
                    this.fun(iter);
                    if (this.status === 3 /* Cancel */) {
                        break;
                    }
                    else if (this.status === 0 /* Paused */) {
                        setTimeout(this.run0, this.pause_length, [iter + inc, end, inc]);
                        return false;
                    }
                    iter += inc;
                }
                this.status = 1 /* Idle */;
            };
            ParallelWorker.prototype.run = function (fun, start_idx, end_idx, inc) {
                this.fun = fun;
                this.pause_length = INIT_PAUSE_LENGTH;
                this.status = 2 /* Busy */;
                if (inc) {
                    return this.run0(start_idx, end_idx, inc);
                }
                else {
                    return this.run0(start_idx, end_idx, 1);
                }
            };
            ParallelWorker.prototype.cancel = function () {
                this.status = 3 /* Cancel */;
            };
            ParallelWorker.prototype.pause = function () {
                this.status = 0 /* Paused */;
            };
            return ParallelWorker;
        })();
        parallel.ParallelWorker = ParallelWorker;
    })(parallel = lib.parallel || (lib.parallel = {}));
})(lib || (lib = {}));
/// <reference path="../ref.ts" />
/// based on https://github.com/broofa/node-uuid/blob/master/uuid.js
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var detail;
        (function (detail) {
            var randArray = new Uint8Array(16);
            var makeRandom = function () {
                for (var i = 0, r; i < 16; i++) {
                    if ((i & 0x03) === 0)
                        r = Math.random() * 0x100000000;
                    randArray[i] = r >>> ((i & 0x03) << 3) & 0xff;
                }
                return randArray;
            };
            // Maps for number <-> hex string conversion
            var byteToHex = [];
            var hexToByte = {};
            for (var i = 0; i < 256; i++) {
                byteToHex[i] = (i + 0x100).toString(16).substr(1);
                hexToByte[byteToHex[i]] = i;
            }
            // **`unparse()` - Convert UUID byte array (ala parse()) into a string*
            function unparse(buf) {
                var i = 0, bth = byteToHex;
                return bth[buf[i++]] + bth[buf[i++]] + bth[buf[i++]] + bth[buf[i++]] + '-' + bth[buf[i++]] + bth[buf[i++]] + '-' + bth[buf[i++]] + bth[buf[i++]] + '-' + bth[buf[i++]] + bth[buf[i++]] + '-' + bth[buf[i++]] + bth[buf[i++]] + bth[buf[i++]] + bth[buf[i++]] + bth[buf[i++]] + bth[buf[i++]];
            }
            function guuid() {
                var rnds = makeRandom();
                rnds[6] = (rnds[6] & 0x0f) | 0x40;
                rnds[8] = (rnds[8] & 0x3f) | 0x80;
                return unparse(rnds);
            }
            detail.guuid = guuid;
        })(detail = utils.detail || (utils.detail = {}));
        utils.guuid = detail.guuid;
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var detail;
        (function (detail) {
            (function (ErrorCode) {
                ErrorCode[ErrorCode["Success"] = 0] = "Success";
                ErrorCode[ErrorCode["MemoryOverflow"] = 1] = "MemoryOverflow";
                ErrorCode[ErrorCode["IntegerOverflow"] = 2] = "IntegerOverflow";
                ErrorCode[ErrorCode["Unknown"] = 3] = "Unknown";
            })(detail.ErrorCode || (detail.ErrorCode = {}));
            var ErrorCode = detail.ErrorCode;
        })(detail = utils.detail || (utils.detail = {}));
        ;
        var Error = (function () {
            function Error(code) {
                if (code) {
                    this.code = code;
                }
                else {
                    this.code = 0 /* Success */;
                }
            }
            return Error;
        })();
        utils.Error = Error;
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        function castTo(arg) {
            return arg;
        }
        utils.castTo = castTo;
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
/// <reference path="./logger.ts" />
/// <reference path="./assert.ts" />
/// <reference path="./rand.ts" />
/// <reference path="./mixin.ts" />
/// <reference path="./guuid.ts" />
/// <reference path="./exception.ts" />
/// <reference path="./error.ts" />
/// <reference path="./timer.ts" />
/// <reference path="./testing.ts" />
/// <reference path="./castTo.ts" />
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
                            return utils.castTo(new Int8(this.value_[0] + other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] + other.getValue()[0]);
                    };
                    Int8.prototype.addTo = function (other) {
                        this.value_[0] += other.getValue()[0];
                        return utils.castTo(this);
                    };
                    Int8.prototype.sub = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Int8(this.value_[0] - other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] - other.getValue()[0]);
                    };
                    Int8.prototype.subFrom = function (other) {
                        this.value_[0] -= other.getValue()[0];
                        return utils.castTo(this);
                    };
                    Int8.prototype.mul = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Int8(this.value_[0] * other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] * other.getValue()[0]);
                    };
                    Int8.prototype.mulBy = function (other) {
                        this.value_[0] *= other.getValue()[0];
                        return utils.castTo(this);
                    };
                    Int8.prototype.div = function (other) {
                        if (other.KIND <= this.KIND) {
                            return utils.castTo(new Int8(this.value_[0] / other.getValue()[0]));
                        }
                        var typ = detail.CLiteralKindMap.get(other.KIND);
                        return new typ(this.value_[0] / other.getValue()[0]);
                    };
                    Int8.prototype.divBy = function (other) {
                        this.value_[0] /= other.getValue()[0];
                        return utils.castTo(this);
                    };
                    Int8.prototype.negate = function () {
                        return utils.castTo(new Int8(-this.value_[0]));
                    };
                    Int8.prototype.value = function () {
                        return this.value_[0];
                    };
                    return Int8;
                })();
                detail.Int8 = Int8;
                detail.CLiteralKindMap.set(10 /* Int8 */, Int8);
                utils.applyMixins(Int8, [detail.IntegerTraits, detail.SignedIntegerTraits]);
            })(detail = type.detail || (type.detail = {}));
            type.Int8 = detail.Int8;
        })(type = c.type || (c.type = {}));
    })(c = lib.c || (lib.c = {}));
})(lib || (lib = {}));
/// <reference path="./lib/utils/utils.ts" />
/// <reference path="./lib/c/c.ts" />
var app;
(function (app) {
    var Greeter = (function () {
        function Greeter(element) {
            this.element = element;
            this.element.innerHTML += "The time is: ";
            this.span = document.createElement('span');
            this.element.appendChild(this.span);
            this.span.innerText = new Date().toUTCString();
        }
        Greeter.prototype.start = function () {
            var _this = this;
            this.timerToken = setInterval(function () { return _this.span.innerHTML = new Date().toUTCString(); }, 500);
        };
        Greeter.prototype.stop = function () {
            lib.utils.assert(1 == 1, "test");
            clearTimeout(this.timerToken);
        };
        return Greeter;
    })();
    app.Greeter = Greeter;
})(app || (app = {}));
window.onload = function () {
    var el = document.getElementById('content');
    var greeter = new app.Greeter(el);
    greeter.start();
};
/// <reference path='../../utils/utils.ts' />
var lib;
(function (lib) {
    var c;
    (function (c) {
        var memory;
        (function (memory) {
            (function (AddressSpace) {
                AddressSpace[AddressSpace["Shared"] = 0] = "Shared";
                AddressSpace[AddressSpace["Global"] = 1] = "Global";
                AddressSpace[AddressSpace["Host"] = 2] = "Host";
            })(memory.AddressSpace || (memory.AddressSpace = {}));
            var AddressSpace = memory.AddressSpace;
            ;
            var CLiteralKind = lib.c.type.detail.CLiteralKind;
            var Reference = (function () {
                function Reference(id, addressSpace, data) {
                    this.id = id;
                    this.addressSpace = addressSpace;
                    this.data = data;
                    this.KIND = 10 /* Int8 */;
                }
                Reference.prototype.get = function (idx) {
                    switch (this.KIND) {
                        case 10 /* Int8 */:
                            return new lib.c.type.Int8(this.data.getInt8(idx));
                        case 20 /* Int16 */:
                            return new lib.c.type.Int16(this.data.getInt16(idx));
                        case 30 /* Int32 */:
                            return new lib.c.type.Int32(this.data.getInt32(idx));
                        case 40 /* Int64 */:
                            return new lib.c.type.Int64(this.data.getInt32(2 * idx), this.data.getInt32(2 * idx + 1));
                        case 11 /* Uint8 */:
                            return new lib.c.type.Uint8(this.data.getUint8(idx));
                        case 21 /* Uint16 */:
                            return new lib.c.type.Uint16(this.data.getUint16(idx));
                        case 31 /* Uint32 */:
                            return new lib.c.type.Uint32(this.data.getUint32(idx));
                    }
                };
                Reference.prototype.set = function (idx, val) {
                    if (val instanceof lib.c.type.Int64) {
                        var i64 = lib.utils.castTo(val);
                        this.data.setInt32(2 * idx, i64.getHigh());
                        this.data.setInt32(2 * idx + 1, i64.getLow());
                        return this.get(idx);
                    }
                    else if (val instanceof Object) {
                        var tmp = lib.utils.castTo(val);
                        val = tmp.getValue()[0];
                    }
                    switch (this.KIND) {
                        case 10 /* Int8 */:
                            this.data.setInt8(idx, val);
                            break;
                        case 20 /* Int16 */:
                            this.data.setInt16(idx, val);
                            break;
                        case 30 /* Int32 */:
                            this.data.setInt32(idx, val);
                            break;
                        case 40 /* Int64 */:
                            this.data.setInt32(2 * idx, 0);
                            this.data.setInt32(2 * idx + 1, val);
                            break;
                        case 11 /* Uint8 */:
                            this.data.setUint8(idx, val);
                            break;
                        case 21 /* Uint16 */:
                            this.data.setUint16(idx, val);
                            break;
                        case 31 /* Uint32 */:
                            this.data.setUint32(idx, val);
                            break;
                    }
                    return this.get(idx);
                };
                Reference.prototype.ref = function () {
                    return new Reference(lib.utils.guuid(), this.addressSpace, new DataView(this.data.buffer, 0, 1));
                };
                Reference.prototype.deref = function () {
                    return this.get(0);
                };
                return Reference;
            })();
            memory.Reference = Reference;
            var MB = 1024;
            var MemoryManager = (function () {
                function MemoryManager(addressSpace) {
                    this.memoryOffset = 0;
                    this.TOTAL_MEMORY = 10 * MB;
                    this.addressSpace = addressSpace;
                    this.memory = new ArrayBuffer(this.TOTAL_MEMORY);
                }
                MemoryManager.prototype.malloc = function (n) {
                    var buffer = new Reference(lib.utils.guuid(), this.addressSpace, new DataView(this.memory, this.memoryOffset, this.memoryOffset + n));
                    //this.memmap.set(buffer.id, buffer);
                    this.memoryOffset += n;
                    return buffer;
                };
                MemoryManager.prototype.free = function (mem) {
                    mem = undefined;
                };
                MemoryManager.prototype.ref = function (obj) {
                    return "todo";
                };
                MemoryManager.prototype.deref = function (mem) {
                    return mem[0];
                };
                return MemoryManager;
            })();
            memory.MemoryManager = MemoryManager;
        })(memory = c.memory || (c.memory = {}));
    })(c = lib.c || (lib.c = {}));
})(lib || (lib = {}));
/// <reference path="../../ref.ts" />
var lib;
(function (lib) {
    var cuda;
    (function (cuda) {
        var exec;
        (function (exec) {
            var Barrier = (function () {
                function Barrier(dim) {
                    this.mask = new Array(dim.flattenedLength());
                }
                return Barrier;
            })();
            exec.Barrier = Barrier;
        })(exec = cuda.exec || (cuda.exec = {}));
    })(cuda = lib.cuda || (lib.cuda = {}));
})(lib || (lib = {}));
/// <reference path="../../ref.ts" />
var lib;
(function (lib) {
    var cuda;
    (function (cuda) {
        var exec;
        (function (exec) {
            var Block = (function () {
                function Block(grid, blockIdx, fun, args) {
                    this.blockIdx = new lib.cuda.Dim3(0);
                    this.blockDim = new lib.cuda.Dim3(0);
                    this.gridIdx = new lib.cuda.Dim3(0);
                    this.gridDim = new lib.cuda.Dim3(0);
                    this.threads = null;
                    this.barriers = null;
                    this.fun = undefined;
                    this.args = [];
                    this.status = 1 /* Idle */;
                    this.grid = grid;
                    this.blockIdx = blockIdx;
                    this.gridIdx = grid.gridIdx;
                    this.gridDim = grid.gridDim;
                    this.args = args;
                    this.fun = fun;
                }
                return Block;
            })();
            exec.Block = Block;
        })(exec = cuda.exec || (cuda.exec = {}));
    })(cuda = lib.cuda || (lib.cuda = {}));
})(lib || (lib = {}));
/// <reference path="../../ref.ts" />
var lib;
(function (lib) {
    var cuda;
    (function (cuda) {
        var exec;
        (function (exec) {
            var Grid = (function () {
                function Grid() {
                    this.gridIdx = new lib.utils.Dim3(0);
                    this.gridDim = new lib.utils.Dim3(0);
                    this.blocks = null;
                }
                return Grid;
            })();
            exec.Grid = Grid;
        })(exec = cuda.exec || (cuda.exec = {}));
    })(cuda = lib.cuda || (lib.cuda = {}));
})(lib || (lib = {}));
/// <reference path="../../ref.ts" />
var lib;
(function (lib) {
    var cuda;
    (function (cuda) {
        var exec;
        (function (exec) {
            var KB = 1024;
            var M = KB * KB;
            var FermiArchitecture = (function () {
                function FermiArchitecture() {
                    this.maxGridDimensions = 3;
                    this.warpSize = 32;
                    this.maxXGridDimension = Math.pow(2.0, 31.0) - 1;
                    this.maxYGridDimension = Math.pow(2.0, 31.0) - 1;
                    this.maxZGridDimension = 65535;
                    this.maxBlockDimensions = 3;
                    this.maxXBlockDimension = 1024;
                    this.maxYBlockDimension = 1024;
                    this.maxZBlockDimension = 64;
                    this.maxThreadsPerBlock = 1024;
                    this.numResigersPerThread = 64 * KB;
                    this.maxResidentBlocksPerSM = 16;
                    this.maxResidentWarpsPerSM = 64;
                    this.maxSharedMemoryPerSM = 48 * KB;
                    this.numSharedMemoryBanks = 32;
                    this.localMemorySize = 512 * KB;
                    this.constantMemorySize = 64 * KB;
                    this.maxNumInstructions = 512 * M;
                    this.numWarpSchedulers = 2;
                }
                return FermiArchitecture;
            })();
            exec.FermiArchitecture = FermiArchitecture;
            exec.ComputeCapabilityMap = undefined;
            if (exec.ComputeCapabilityMap !== undefined) {
                exec.ComputeCapabilityMap = new Map();
                exec.ComputeCapabilityMap[2.0] = new FermiArchitecture();
            }
        })(exec = cuda.exec || (cuda.exec = {}));
    })(cuda = lib.cuda || (lib.cuda = {}));
})(lib || (lib = {}));
/// <reference path="../../ref.ts" />
var lib;
(function (lib) {
    var cuda;
    (function (cuda) {
        var exec;
        (function (exec) {
            var Thread = (function () {
                function Thread(block, threadIdx, fun, args) {
                    this.error = new lib.utils.Error();
                    this.threadIdx = new lib.cuda.Dim3(0);
                    this.blockIdx = new lib.cuda.Dim3(0);
                    this.blockDim = new lib.cuda.Dim3(0);
                    this.gridIdx = new lib.cuda.Dim3(0);
                    this.gridDim = new lib.cuda.Dim3(0);
                    this.fun = undefined;
                    this.args = [];
                    this.status = 1 /* Idle */;
                    this.block = block;
                    this.blockIdx = block.blockIdx;
                    this.gridIdx = block.gridIdx;
                    this.gridDim = block.gridDim;
                    this.threadIdx = threadIdx;
                    this.args = args;
                    this.fun = fun;
                }
                Thread.prototype.run = function () {
                    var res;
                    this.status = 0 /* Running */;
                    try {
                        res = this.fun.apply(this, this.args);
                    }
                    catch (err) {
                        res = err.code;
                    }
                    this.status = 2 /* Complete */;
                    return res;
                };
                Thread.prototype.terminate = function () {
                    this.status = 3 /* Stopped */;
                };
                return Thread;
            })();
            exec.Thread = Thread;
        })(exec = cuda.exec || (cuda.exec = {}));
    })(cuda = lib.cuda || (lib.cuda = {}));
})(lib || (lib = {}));
/// <reference path="../../ref.ts" />
var lib;
(function (lib) {
    var cuda;
    (function (cuda) {
        var exec;
        (function (exec) {
            var Warp = (function () {
                function Warp() {
                    this.id = lib.utils.guuid();
                    this.thread = null;
                }
                return Warp;
            })();
            exec.Warp = Warp;
        })(exec = cuda.exec || (cuda.exec = {}));
    })(cuda = lib.cuda || (lib.cuda = {}));
})(lib || (lib = {}));
/// <reference path="../ref.ts" />
var System;
(function (System) {
    "use strict";
})(System || (System = {}));
/// <reference path="../ref.ts" />
var __extends = this.__extends || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    __.prototype = b.prototype;
    d.prototype = new __();
};
var lib;
(function (lib) {
    var memory;
    (function (memory) {
        var detail;
        (function (detail) {
            detail.MemoryManager = lib.c.memory.MemoryManager;
        })(detail = memory.detail || (memory.detail = {}));
        memory.AddressSpace = lib.c.memory.AddressSpace;
        var HostMemoryManager = (function (_super) {
            __extends(HostMemoryManager, _super);
            function HostMemoryManager() {
                _super.call(this, 2 /* Host */);
            }
            return HostMemoryManager;
        })(detail.MemoryManager);
        memory.HostMemoryManager = HostMemoryManager;
        var GlobalMemoryManager = (function (_super) {
            __extends(GlobalMemoryManager, _super);
            function GlobalMemoryManager() {
                _super.call(this, 1 /* Global */);
            }
            return GlobalMemoryManager;
        })(detail.MemoryManager);
        memory.GlobalMemoryManager = GlobalMemoryManager;
        var SharedMemoryManager = (function (_super) {
            __extends(SharedMemoryManager, _super);
            function SharedMemoryManager() {
                _super.call(this, 0 /* Shared */);
            }
            return SharedMemoryManager;
        })(detail.MemoryManager);
        memory.SharedMemoryManager = SharedMemoryManager;
    })(memory = lib.memory || (lib.memory = {}));
})(lib || (lib = {}));
/// <reference path="../ref.ts" />
var lib;
(function (lib) {
    var parallel;
    (function (parallel) {
        var Thread = (function () {
            function Thread(id) {
                this.id = id;
            }
            return Thread;
        })();
        parallel.Thread = Thread;
    })(parallel = lib.parallel || (lib.parallel = {}));
})(lib || (lib = {}));
/// <reference path="../ref.ts" />
var lib;
(function (lib) {
    var parallel;
    (function (parallel) {
        var WorkerPool = (function () {
            function WorkerPool(num_workers) {
                this.num_workers = num_workers;
                this.workers = new Array(num_workers);
            }
            return WorkerPool;
        })();
        parallel.WorkerPool = WorkerPool;
    })(parallel = lib.parallel || (lib.parallel = {}));
})(lib || (lib = {}));
