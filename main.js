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
                        console[type](color["" + type] + msg + color["LogType.Debug"]);
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
            function assertUnreachable(msg) {
                var location = new utils.Error().stack.split('\n')[1];
                throw new utils.Error("Reached unreachable location " + location + msg);
            }
            detail.assertUnreachable = assertUnreachable;
            function error(message) {
                console.error(message);
                throw new utils.Error(message);
            }
            detail.error = error;
            function assertNotImplemented(condition, message) {
                if (!condition) {
                    error("notImplemented: " + message);
                }
            }
            detail.assertNotImplemented = assertNotImplemented;
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
                ErrorCode[ErrorCode["Message"] = 4] = "Message";
            })(detail.ErrorCode || (detail.ErrorCode = {}));
            var ErrorCode = detail.ErrorCode;
        })(detail = utils.detail || (utils.detail = {}));
        ;
        var Error = (function () {
            function Error(arg) {
                if (arg) {
                    if (utils.isString(arg)) {
                        this.message = arg;
                        this.code = 4 /* Message */;
                    }
                    else {
                        this.code = arg;
                        this.message = arg.toString();
                    }
                }
                else {
                    this.code = 0 /* Success */;
                    this.message = "Success";
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
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        utils.isCommonJS = false;
        utils.isNode = false;
        /**
         * Various constant values. Enum'd so they are inlined by the TypeScript
         * compiler.
         */
        (function (constant) {
            constant[constant["INT_MAX"] = Math.pow(2, 31) - 1] = "INT_MAX";
            constant[constant["INT_MIN"] = -constant.INT_MAX - 1] = "INT_MIN";
            constant[constant["FLOAT_POS_INFINITY"] = Math.pow(2, 128)] = "FLOAT_POS_INFINITY";
            constant[constant["FLOAT_NEG_INFINITY"] = -1 * constant.FLOAT_POS_INFINITY] = "FLOAT_NEG_INFINITY";
            constant[constant["FLOAT_POS_INFINITY_AS_INT"] = 0x7F800000] = "FLOAT_POS_INFINITY_AS_INT";
            constant[constant["FLOAT_NEG_INFINITY_AS_INT"] = -8388608] = "FLOAT_NEG_INFINITY_AS_INT";
            // We use the JavaScript NaN as our NaN value, and convert it to
            // a NaN value in the SNaN range when an int equivalent is requested.
            constant[constant["FLOAT_NaN_AS_INT"] = 0x7fc00000] = "FLOAT_NaN_AS_INT";
        })(utils.constant || (utils.constant = {}));
        var constant = utils.constant;
        /*jshint evil: true */
        var getGlobal = new Function('return this;');
        /*jshint evil: false */
        utils.globals = getGlobal();
        utils.global_isFinite = utils.globals.isFinite;
        var _slice = Array.prototype.slice;
        var _indexOf = String.prototype.indexOf;
        utils._toString = Object.prototype.toString;
        var _hasOwnProperty = Object.prototype.hasOwnProperty;
        var Symbol = utils.globals.Symbol || {};
        function isSymbol(sym) {
            /*jshint notypeof: true */
            return typeof utils.globals.Symbol === 'function' && typeof sym === 'symbol';
            /*jshint notypeof: false */
        }
        utils.isSymbol = isSymbol;
        ;
        function isString(value) {
            return typeof value === "string";
        }
        utils.isString = isString;
        function isFunction(value) {
            return typeof value === "function";
        }
        utils.isFunction = isFunction;
        function isNumber(value) {
            return typeof value === "number";
        }
        utils.isNumber = isNumber;
        function isInteger(value) {
            return (value | 0) === value;
        }
        utils.isInteger = isInteger;
        function isArray(value) {
            return value instanceof Array;
        }
        utils.isArray = isArray;
        function isNumberOrString(value) {
            return typeof value === "number" || typeof value === "string";
        }
        utils.isNumberOrString = isNumberOrString;
        function isObject(value) {
            return typeof value === "object" || typeof value === 'function';
        }
        utils.isObject = isObject;
        function toNumber(x) {
            return +x;
        }
        utils.toNumber = toNumber;
        function float2int(a) {
            if (a > constant.INT_MAX) {
                return constant.INT_MAX;
            }
            else if (a < constant.INT_MIN) {
                return constant.INT_MIN;
            }
            else {
                return a | 0;
            }
        }
        utils.float2int = float2int;
        function isNumericString(value) {
            // ECMAScript 5.1 - 9.8.1 Note 1, this expression is true for all
            // numbers x other than -0.
            return String(Number(value)) === value;
        }
        utils.isNumericString = isNumericString;
        function isNullOrUndefined(value) {
            return value == undefined;
        }
        utils.isNullOrUndefined = isNullOrUndefined;
        function backtrace() {
            try {
                throw new utils.Error();
            }
            catch (e) {
                return e.stack ? e.stack.split('\n').slice(2).join('\n') : '';
            }
        }
        utils.backtrace = backtrace;
        function getTicks() {
            return performance.now();
        }
        utils.getTicks = getTicks;
        // Creates and initializes *JavaScript* array to *val* in each element slot.
        // Like memset, but for arrays.
        function arrayset(len, val) {
            var array = new Array(len);
            for (var i = 0; i < len; i++) {
                array[i] = val;
            }
            return array;
        }
        utils.arrayset = arrayset;
        // taken directly from https://github.com/ljharb/is-arguments/blob/master/index.js
        // can be replaced with require('is-arguments') if we ever use a build process instead
        function isArguments(value) {
            var str = utils._toString.call(value);
            var result = str === '[object Arguments]';
            if (!result) {
                result = str !== '[object Array]' && value !== null && typeof value === 'object' && typeof value.length === 'number' && value.length >= 0 && utils._toString.call(value.callee) === '[object Function]';
            }
            return result;
        }
        utils.isArguments = isArguments;
        ;
    })(utils = lib.utils || (lib.utils = {}));
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
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var sourcemap;
        (function (sourcemap) {
            var utils;
            (function (utils) {
                /**
                 * This is a helper function for getting values from parameter/options
                 * objects.
                 *
                 * @param args The object we are extracting values from
                 * @param name The name of the property we are getting.
                 * @param defaultValue An optional value to return if the property is missing
                 * from the object. If this is not specified and the property is missing, an
                 * error will be thrown.
                 */
                function getArg(aArgs, aName, aDefaultValue) {
                    if (aName in aArgs) {
                        return aArgs[aName];
                    }
                    else if (arguments.length === 3) {
                        return aDefaultValue;
                    }
                    else {
                        throw new Error('"' + aName + '" is a required argument.');
                    }
                }
                utils.getArg = getArg;
                var urlRegexp = /^(?:([\w+\-.]+):)?\/\/(?:(\w+:\w+)@)?([\w.]*)(?::(\d+))?(\S*)$/;
                var dataUrlRegexp = /^data:.+\,.+$/;
                function urlParse(aUrl) {
                    var match = aUrl.match(urlRegexp);
                    if (!match) {
                        return null;
                    }
                    return {
                        scheme: match[1],
                        auth: match[2],
                        host: match[3],
                        port: match[4],
                        path: match[5]
                    };
                }
                utils.urlParse = urlParse;
                function urlGenerate(aParsedUrl) {
                    var url = '';
                    if (aParsedUrl.scheme) {
                        url += aParsedUrl.scheme + ':';
                    }
                    url += '//';
                    if (aParsedUrl.auth) {
                        url += aParsedUrl.auth + '@';
                    }
                    if (aParsedUrl.host) {
                        url += aParsedUrl.host;
                    }
                    if (aParsedUrl.port) {
                        url += ":" + aParsedUrl.port;
                    }
                    if (aParsedUrl.path) {
                        url += aParsedUrl.path;
                    }
                    return url;
                }
                utils.urlGenerate = urlGenerate;
                /**
                 * Normalizes a path, or the path portion of a URL:
                 *
                 * - Replaces consequtive slashes with one slash.
                 * - Removes unnecessary '.' parts.
                 * - Removes unnecessary '<dir>/..' parts.
                 *
                 * Based on code in the Node.js 'path' core module.
                 *
                 * @param aPath The path or url to normalize.
                 */
                function normalize(aPath) {
                    var path = aPath;
                    var url = urlParse(aPath);
                    if (url) {
                        if (!url.path) {
                            return aPath;
                        }
                        path = url.path;
                    }
                    var isAbsolute = (path.charAt(0) === '/');
                    var parts = path.split(/\/+/);
                    for (var part, up = 0, i = parts.length - 1; i >= 0; i--) {
                        part = parts[i];
                        if (part === '.') {
                            parts.splice(i, 1);
                        }
                        else if (part === '..') {
                            up++;
                        }
                        else if (up > 0) {
                            if (part === '') {
                                // The first part is blank if the path is absolute. Trying to go
                                // above the root is a no-op. Therefore we can remove all '..' parts
                                // directly after the root.
                                parts.splice(i + 1, up);
                                up = 0;
                            }
                            else {
                                parts.splice(i, 2);
                                up--;
                            }
                        }
                    }
                    path = parts.join('/');
                    if (path === '') {
                        path = isAbsolute ? '/' : '.';
                    }
                    if (url) {
                        url.path = path;
                        return urlGenerate(url);
                    }
                    return path;
                }
                utils.normalize = normalize;
                /**
                 * Joins two paths/URLs.
                 *
                 * @param aRoot The root path or URL.
                 * @param aPath The path or URL to be joined with the root.
                 *
                 * - If aPath is a URL or a data URI, aPath is returned, unless aPath is a
                 *   scheme-relative URL: Then the scheme of aRoot, if any, is prepended
                 *   first.
                 * - Otherwise aPath is a path. If aRoot is a URL, then its path portion
                 *   is updated with the result and aRoot is returned. Otherwise the result
                 *   is returned.
                 *   - If aPath is absolute, the result is aPath.
                 *   - Otherwise the two paths are joined with a slash.
                 * - Joining for example 'http://' and 'www.example.com' is also supported.
                 */
                function join(aRoot, aPath) {
                    if (aRoot === "") {
                        aRoot = ".";
                    }
                    if (aPath === "") {
                        aPath = ".";
                    }
                    var aPathUrl = urlParse(aPath);
                    var aRootUrl = urlParse(aRoot);
                    if (aRootUrl) {
                        aRoot = aRootUrl.path || '/';
                    }
                    // `join(foo, '//www.example.org')`
                    if (aPathUrl && !aPathUrl.scheme) {
                        if (aRootUrl) {
                            aPathUrl.scheme = aRootUrl.scheme;
                        }
                        return urlGenerate(aPathUrl);
                    }
                    if (aPathUrl || aPath.match(dataUrlRegexp)) {
                        return aPath;
                    }
                    // `join('http://', 'www.example.com')`
                    if (aRootUrl && !aRootUrl.host && !aRootUrl.path) {
                        aRootUrl.host = aPath;
                        return urlGenerate(aRootUrl);
                    }
                    var joined = aPath.charAt(0) === '/' ? aPath : normalize(aRoot.replace(/\/+$/, '') + '/' + aPath);
                    if (aRootUrl) {
                        aRootUrl.path = joined;
                        return urlGenerate(aRootUrl);
                    }
                    return joined;
                }
                utils.join = join;
                /**
                 * Make a path relative to a URL or another path.
                 *
                 * @param aRoot The root path or URL.
                 * @param aPath The path or URL to be made relative to aRoot.
                 */
                function relative(aRoot, aPath) {
                    if (aRoot === "") {
                        aRoot = ".";
                    }
                    aRoot = aRoot.replace(/\/$/, '');
                    // XXX: It is possible to remove this block, and the tests still pass!
                    var url = urlParse(aRoot);
                    if (aPath.charAt(0) == "/" && url && url.path == "/") {
                        return aPath.slice(1);
                    }
                    return aPath.indexOf(aRoot + '/') === 0 ? aPath.substr(aRoot.length + 1) : aPath;
                }
                utils.relative = relative;
                /**
                 * Because behavior goes wacky when you set `__proto__` on objects, we
                 * have to prefix all the strings in our set with an arbitrary character.
                 *
                 * See https://github.com/mozilla/source-map/pull/31 and
                 * https://github.com/mozilla/source-map/issues/30
                 *
                 * @param String aStr
                 */
                function toSetString(aStr) {
                    return '$' + aStr;
                }
                utils.toSetString = toSetString;
                function fromSetString(aStr) {
                    return aStr.substr(1);
                }
                utils.fromSetString = fromSetString;
                function strcmp(aStr1, aStr2) {
                    var s1 = aStr1 || "";
                    var s2 = aStr2 || "";
                    var d1 = lib.utils.castTo(s1 > s2);
                    var d2 = lib.utils.castTo(s1 < s2);
                    return d1 - d2;
                }
                /**
                 * Comparator between two mappings where the original positions are compared.
                 *
                 * Optionally pass in `true` as `onlyCompareGenerated` to consider two
                 * mappings with the same original source/line/column, but different generated
                 * line and column the same. Useful when searching for a mapping with a
                 * stubbed out mapping.
                 */
                function compareByOriginalPositions(mappingA, mappingB, onlyCompareOriginal) {
                    var cmp;
                    cmp = strcmp(mappingA.source, mappingB.source);
                    if (cmp) {
                        return cmp;
                    }
                    cmp = mappingA.originalLine - mappingB.originalLine;
                    if (cmp) {
                        return cmp;
                    }
                    cmp = mappingA.originalColumn - mappingB.originalColumn;
                    if (cmp || onlyCompareOriginal) {
                        return cmp;
                    }
                    cmp = strcmp(mappingA.name, mappingB.name);
                    if (cmp) {
                        return cmp;
                    }
                    cmp = mappingA.generatedLine - mappingB.generatedLine;
                    if (cmp) {
                        return cmp;
                    }
                    return mappingA.generatedColumn - mappingB.generatedColumn;
                }
                utils.compareByOriginalPositions = compareByOriginalPositions;
                ;
                /**
                 * Comparator between two mappings where the generated positions are
                 * compared.
                 *
                 * Optionally pass in `true` as `onlyCompareGenerated` to consider two
                 * mappings with the same generated line and column, but different
                 * source/name/original line and column the same. Useful when searching for a
                 * mapping with a stubbed out mapping.
                 */
                function compareByGeneratedPositions(mappingA, mappingB, onlyCompareGenerated) {
                    var cmp;
                    cmp = mappingA.generatedLine - mappingB.generatedLine;
                    if (cmp) {
                        return cmp;
                    }
                    cmp = mappingA.generatedColumn - mappingB.generatedColumn;
                    if (cmp || onlyCompareGenerated) {
                        return cmp;
                    }
                    cmp = strcmp(mappingA.source, mappingB.source);
                    if (cmp) {
                        return cmp;
                    }
                    cmp = mappingA.originalLine - mappingB.originalLine;
                    if (cmp) {
                        return cmp;
                    }
                    cmp = mappingA.originalColumn - mappingB.originalColumn;
                    if (cmp) {
                        return cmp;
                    }
                    return strcmp(mappingA.name, mappingB.name);
                }
                utils.compareByGeneratedPositions = compareByGeneratedPositions;
                ;
            })(utils = sourcemap.utils || (sourcemap.utils = {}));
        })(sourcemap = ast.sourcemap || (ast.sourcemap = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/* -*- Mode: js; js-indent-level: 2; -*- */
/*
 * Copyright 2011 Mozilla Foundation and contributors
 * Licensed under the New BSD license. See LICENSE or:
 * http://opensource.org/licenses/BSD-3-Clause
 */
/// <reference path="utils.ts" />
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var sourcemap;
        (function (sourcemap) {
            var search = lib.utils.search;
            var ArraySet = lib.utils.ArraySet;
            var base64VLQ = lib.utils.vlq;
            var utils = lib.ast.sourcemap.utils;
            /**
             * A SourceMapConsumer instance represents a parsed source map which we can
             * query for information about the original file positions by giving it a file
             * position in the generated source.
             *
             * The only parameter is the raw source map (either as a JSON string, or
             * already parsed to an object). According to the spec, source maps have the
             * following attributes:
             *
             *   - version: Which version of the source map spec this map is following.
             *   - sources: An array of URLs to the original source files.
             *   - names: An array of identifiers which can be referrenced by individual mappings.
             *   - sourceRoot: Optional. The URL root from which all sources are relative.
             *   - sourcesContent: Optional. An array of contents of the original source files.
             *   - mappings: A string of base64 VLQs which contain the actual mappings.
             *   - file: Optional. The generated file this source map is associated with.
             *
             * Here is an example source map, taken from the source map spec[0]:
             *
             *     {
             *       version : 3,
             *       file: "out.js",
             *       sourceRoot : "",
             *       sources: ["foo.js", "bar.js"],
             *       names: ["src", "maps", "are", "fun"],
             *       mappings: "AA,AB;;ABCDE;"
             *     }
             *
             * [0]: https://docs.google.com/document/d/1U1RGAehQwRypUTovF1KRlpiOFze0b-_2gc6fAH0KY0k/edit?pli=1#
             */
            var SourceMapConsumer = (function () {
                function SourceMapConsumer(aSourceMap) {
                    /**
                     * The version of the source mapping spec that we are consuming.
                     */
                    this._version = 3;
                    // `__generatedMappings` and `__originalMappings` are arrays that hold the
                    // parsed mapping coordinates from the source map's "mappings" attribute. They
                    // are lazily instantiated, accessed via the `_generatedMappings` and
                    // `_originalMappings` getters respectively, and we only parse the mappings
                    // and create these arrays once queried for a source location. We jump through
                    // these hoops because there can be many thousands of mappings, and parsing
                    // them is expensive, so we only want to do it if we must.
                    //
                    // Each object in the arrays is of the form:
                    //
                    //     {
                    //       generatedLine: The line number in the generated code,
                    //       generatedColumn: The column number in the generated code,
                    //       source: The path to the original source file that generated this
                    //               chunk of code,
                    //       originalLine: The line number in the original source that
                    //                     corresponds to this chunk of generated code,
                    //       originalColumn: The column number in the original source that
                    //                       corresponds to this chunk of generated code,
                    //       name: The name of the original symbol which generated this chunk of
                    //             code.
                    //     }
                    //
                    // All properties except for `generatedLine` and `generatedColumn` can be
                    // `null`.
                    //
                    // `_generatedMappings` is ordered by the generated positions.
                    //
                    // `_originalMappings` is ordered by the original positions.
                    this.__generatedMappings = null;
                    this.__originalMappings = null;
                    /**
                     * Returns the original source, line, and column information for the generated
                     * source's line and column positions provided. The only argument is an object
                     * with the following properties:
                     *
                     *   - line: The line number in the generated source.
                     *   - column: The column number in the generated source.
                     *
                     * and an object is returned with the following properties:
                     *
                     *   - source: The original source file, or null.
                     *   - line: The line number in the original source, or null.
                     *   - column: The column number in the original source, or null.
                     *   - name: The original identifier, or null.
                     */
                    this.originalPositionFor = function SourceMapConsumer_originalPositionFor(aArgs) {
                        var needle = {
                            generatedLine: utils.getArg(aArgs, 'line'),
                            generatedColumn: utils.getArg(aArgs, 'column')
                        };
                        var index = this._findMapping(needle, this._generatedMappings, "generatedLine", "generatedColumn", utils.compareByGeneratedPositions);
                        if (index >= 0) {
                            var mapping = this._generatedMappings[index];
                            if (mapping.generatedLine === needle.generatedLine) {
                                var source = utils.getArg(mapping, 'source', null);
                                if (source != null && this.sourceRoot != null) {
                                    source = utils.join(this.sourceRoot, source);
                                }
                                return {
                                    source: source,
                                    line: utils.getArg(mapping, 'originalLine', null),
                                    column: utils.getArg(mapping, 'originalColumn', null),
                                    name: utils.getArg(mapping, 'name', null)
                                };
                            }
                        }
                        return {
                            source: null,
                            line: null,
                            column: null,
                            name: null
                        };
                    };
                    var sourceMap = aSourceMap;
                    if (typeof aSourceMap === 'string') {
                        sourceMap = JSON.parse(aSourceMap.replace(/^\)\]\}'/, ''));
                    }
                    var version = utils.getArg(sourceMap, 'version');
                    var sources = utils.getArg(sourceMap, 'sources');
                    // Sass 3.3 leaves out the 'names' array, so we deviate from the spec (which
                    // requires the array) to play nice here.
                    var names = utils.getArg(sourceMap, 'names', []);
                    var sourceRoot = utils.getArg(sourceMap, 'sourceRoot', null);
                    var sourcesContent = utils.getArg(sourceMap, 'sourcesContent', null);
                    var mappings = utils.getArg(sourceMap, 'mappings');
                    var file = utils.getArg(sourceMap, 'file', null);
                    // Once again, Sass deviates from the spec and supplies the version as a
                    // string rather than a number, so we use loose equality checking here.
                    if (version != this._version) {
                        throw new Error('Unsupported version: ' + version);
                    }
                    // Pass `true` below to allow duplicate names and sources. While source maps
                    // are intended to be compressed and deduplicated, the TypeScript compiler
                    // sometimes generates source maps with duplicates in them. See Github issue
                    // #72 and bugzil.la/889492.
                    this._names = ArraySet.fromArray(names, true);
                    this._sources = ArraySet.fromArray(sources, true);
                    this._sourceRoot = sourceRoot;
                    this._sourcesContent = sourcesContent;
                    this._mappings = mappings;
                    this._file = file;
                }
                Object.defineProperty(SourceMapConsumer.prototype, "file", {
                    get: function () {
                        return this._file;
                    },
                    set: function (val) {
                        this._file = val;
                    },
                    enumerable: true,
                    configurable: true
                });
                Object.defineProperty(SourceMapConsumer.prototype, "sourceRoot", {
                    get: function () {
                        return this._sourceRoot;
                    },
                    set: function (val) {
                        this._sourceRoot = val;
                    },
                    enumerable: true,
                    configurable: true
                });
                Object.defineProperty(SourceMapConsumer.prototype, "sources", {
                    /**
                     * The list of original sources.
                     */
                    get: function () {
                        var _this = this;
                        return this._sources.toArray().map(function (s) { return _this._sourceRoot != null ? utils.join(_this._sourceRoot, s) : s; }, this);
                    },
                    set: function (val) {
                        this._sources = val;
                    },
                    enumerable: true,
                    configurable: true
                });
                Object.defineProperty(SourceMapConsumer.prototype, "names", {
                    get: function () {
                        return this._names;
                    },
                    set: function (val) {
                        this._names = val;
                    },
                    enumerable: true,
                    configurable: true
                });
                Object.defineProperty(SourceMapConsumer.prototype, "mappings", {
                    get: function () {
                        return this._mappings;
                    },
                    set: function (val) {
                        this._mappings = val;
                    },
                    enumerable: true,
                    configurable: true
                });
                Object.defineProperty(SourceMapConsumer.prototype, "sourcesContent", {
                    get: function () {
                        return this._sourcesContent;
                    },
                    set: function (val) {
                        this._sourcesContent = val;
                    },
                    enumerable: true,
                    configurable: true
                });
                /**
                 * Create a SourceMapConsumer from a SourceMapGenerator.
                 *
                 * @param SourceMapGenerator aSourceMap
                 *        The source map that will be consumed.
                 * @returns SourceMapConsumer
                 */
                SourceMapConsumer.fromSourceMap = function (aSourceMap) {
                    var smc = new SourceMapConsumer();
                    smc.names(ArraySet.fromArray(aSourceMap._names.toArray(), true));
                    smc.sources = ArraySet.fromArray(aSourceMap._sources.toArray(), true);
                    smc.sourceRoot = aSourceMap._sourceRoot;
                    smc.sourcesContent = aSourceMap._generateSourcesContent(smc.sources.toArray(), smc.sourceRoot);
                    smc.file = aSourceMap._file;
                    smc._generatedMappings = aSourceMap._mappings.slice().sort(utils.compareByGeneratedPositions);
                    smc._originalMappings = aSourceMap._mappings.slice().sort(utils.compareByOriginalPositions);
                    return smc;
                };
                Object.defineProperty(SourceMapConsumer.prototype, "_generatedMappings", {
                    get: function () {
                        if (!this.__generatedMappings) {
                            this.__generatedMappings = [];
                            this.__originalMappings = [];
                            this._parseMappings(this._mappings, this.sourceRoot);
                        }
                        return this.__generatedMappings;
                    },
                    enumerable: true,
                    configurable: true
                });
                Object.defineProperty(SourceMapConsumer.prototype, "_originalMappings", {
                    get: function () {
                        if (!this.__originalMappings) {
                            this.__generatedMappings = [];
                            this.__originalMappings = [];
                            this._parseMappings(this._mappings, this.sourceRoot);
                        }
                        return this.__originalMappings;
                    },
                    enumerable: true,
                    configurable: true
                });
                SourceMapConsumer.prototype._nextCharIsMappingSeparator = function (aStr) {
                    var c = aStr.charAt(0);
                    return c === ";" || c === ",";
                };
                /**
                 * Parse the mappings in a string in to a data structure which we can easily
                 * query (the ordered arrays in the `this.__generatedMappings` and
                 * `this.__originalMappings` properties).
                 */
                SourceMapConsumer.prototype._parseMappings = function (aStr, aSourceRoot) {
                    var generatedLine = 1;
                    var previousGeneratedColumn = 0;
                    var previousOriginalLine = 0;
                    var previousOriginalColumn = 0;
                    var previousSource = 0;
                    var previousName = 0;
                    var str = aStr;
                    var temp = lib.utils.castTo({});
                    var mapping;
                    while (str.length > 0) {
                        if (str.charAt(0) === ';') {
                            generatedLine++;
                            str = str.slice(1);
                            previousGeneratedColumn = 0;
                        }
                        else if (str.charAt(0) === ',') {
                            str = str.slice(1);
                        }
                        else {
                            mapping = {};
                            mapping.generatedLine = generatedLine;
                            // Generated column.
                            base64VLQ.decode(str, temp);
                            mapping.generatedColumn = previousGeneratedColumn + temp.value;
                            previousGeneratedColumn = mapping.generatedColumn;
                            str = temp.rest;
                            if (str.length > 0 && !this._nextCharIsMappingSeparator(str)) {
                                // Original source.
                                base64VLQ.decode(str, temp);
                                mapping.source = this._sources.at(previousSource + temp.value);
                                previousSource += temp.value;
                                str = temp.rest;
                                if (str.length === 0 || this._nextCharIsMappingSeparator(str)) {
                                    throw new Error('Found a source, but no line and column');
                                }
                                // Original line.
                                base64VLQ.decode(str, temp);
                                mapping.originalLine = previousOriginalLine + temp.value;
                                previousOriginalLine = mapping.originalLine;
                                // Lines are stored 0-based
                                mapping.originalLine += 1;
                                str = temp.rest;
                                if (str.length === 0 || this._nextCharIsMappingSeparator(str)) {
                                    throw new Error('Found a source and line, but no column');
                                }
                                // Original column.
                                base64VLQ.decode(str, temp);
                                mapping.originalColumn = previousOriginalColumn + temp.value;
                                previousOriginalColumn = mapping.originalColumn;
                                str = temp.rest;
                                if (str.length > 0 && !this._nextCharIsMappingSeparator(str)) {
                                    // Original name.
                                    base64VLQ.decode(str, temp);
                                    mapping.name = this._names.at(previousName + temp.value);
                                    previousName += temp.value;
                                    str = temp.rest;
                                }
                            }
                            this.__generatedMappings.push(mapping);
                            if (typeof mapping.originalLine === 'number') {
                                this.__originalMappings.push(mapping);
                            }
                        }
                    }
                    this.__generatedMappings.sort(utils.compareByGeneratedPositions);
                    this.__originalMappings.sort(utils.compareByOriginalPositions);
                };
                /**
                 * Find the mapping that best matches the hypothetical "needle" mapping that
                 * we are searching for in the given "haystack" of mappings.
                 */
                SourceMapConsumer.prototype._findMapping = function (aNeedle, aMappings, aLineName, aColumnName, aComparator) {
                    // To return the position we are searching for, we must first find the
                    // mapping for the given position and then return the opposite position it
                    // points to. Because the mappings are sorted, we can use binary search to
                    // find the best mapping.
                    if (aNeedle[aLineName] <= 0) {
                        throw new TypeError('Line must be greater than or equal to 1, got ' + aNeedle[aLineName]);
                    }
                    if (aNeedle[aColumnName] < 0) {
                        throw new TypeError('Column must be greater than or equal to 0, got ' + aNeedle[aColumnName]);
                    }
                    return search(aNeedle, aMappings, aComparator);
                };
                /**
                 * Returns the original source content. The only argument is the url of the
                 * original source file. Returns null if no original source content is
                 * availible.
                 */
                SourceMapConsumer.prototype.sourceContentFor = function (aSource) {
                    if (!this.sourcesContent) {
                        return null;
                    }
                    if (this.sourceRoot != null) {
                        aSource = utils.relative(this.sourceRoot, aSource);
                    }
                    if (this._sources.has(aSource)) {
                        return this.sourcesContent[this._sources.indexOf(aSource)];
                    }
                    var url;
                    if (this.sourceRoot != null && (url = utils.urlParse(this.sourceRoot))) {
                        // XXX: file:// URIs and absolute paths lead to unexpected behavior for
                        // many users. We can help them out when they expect file:// URIs to
                        // behave like it would if they were running a local HTTP server. See
                        // https://bugzilla.mozilla.org/show_bug.cgi?id=885597.
                        var fileUriAbsPath = aSource.replace(/^file:\/\//, "");
                        if (url.scheme == "file" && this._sources.has(fileUriAbsPath)) {
                            return this.sourcesContent[this._sources.indexOf(fileUriAbsPath)];
                        }
                        if ((!url.path || url.path == "/") && this._sources.has("/" + aSource)) {
                            return this.sourcesContent[this._sources.indexOf("/" + aSource)];
                        }
                    }
                    throw new Error('"' + aSource + '" is not in the SourceMap.');
                };
                /**
                 * Returns the generated line and column information for the original source,
                 * line, and column positions provided. The only argument is an object with
                 * the following properties:
                 *
                 *   - source: The filename of the original source.
                 *   - line: The line number in the original source.
                 *   - column: The column number in the original source.
                 *
                 * and an object is returned with the following properties:
                 *
                 *   - line: The line number in the generated source, or null.
                 *   - column: The column number in the generated source, or null.
                 */
                SourceMapConsumer.prototype.generatedPositionFor = function (aArgs) {
                    var needle = {
                        source: utils.getArg(aArgs, 'source'),
                        originalLine: utils.getArg(aArgs, 'line'),
                        originalColumn: utils.getArg(aArgs, 'column')
                    };
                    if (this.sourceRoot != null) {
                        needle.source = utils.relative(this.sourceRoot, needle.source);
                    }
                    var index = this._findMapping(needle, this._originalMappings, "originalLine", "originalColumn", utils.compareByOriginalPositions);
                    if (index >= 0) {
                        var mapping = this._originalMappings[index];
                        return {
                            line: utils.getArg(mapping, 'generatedLine', null),
                            column: utils.getArg(mapping, 'generatedColumn', null)
                        };
                    }
                    return {
                        line: null,
                        column: null
                    };
                };
                /**
                 * Returns all generated line and column information for the original source
                 * and line provided. The only argument is an object with the following
                 * properties:
                 *
                 *   - source: The filename of the original source.
                 *   - line: The line number in the original source.
                 *
                 * and an array of objects is returned, each with the following properties:
                 *
                 *   - line: The line number in the generated source, or null.
                 *   - column: The column number in the generated source, or null.
                 */
                SourceMapConsumer.prototype.allGeneratedPositionsFor = function (aArgs) {
                    // When there is no exact match, SourceMapConsumer.prototype._findMapping
                    // returns the index of the closest mapping less than the needle. By
                    // setting needle.originalColumn to Infinity, we thus find the last
                    // mapping for the given line, provided such a mapping exists.
                    var needle = {
                        source: utils.getArg(aArgs, 'source'),
                        originalLine: utils.getArg(aArgs, 'line'),
                        originalColumn: Infinity
                    };
                    if (this.sourceRoot != null) {
                        needle.source = utils.relative(this.sourceRoot, needle.source);
                    }
                    var mappings = [];
                    var index = this._findMapping(needle, this._originalMappings, "originalLine", "originalColumn", utils.compareByOriginalPositions);
                    if (index >= 0) {
                        var mapping = this._originalMappings[index];
                        while (mapping && mapping.originalLine === needle.originalLine) {
                            mappings.push({
                                line: utils.getArg(mapping, 'generatedLine', null),
                                column: utils.getArg(mapping, 'generatedColumn', null)
                            });
                            mapping = this._originalMappings[--index];
                        }
                    }
                    return mappings.reverse();
                };
                /**
                 * Iterate over each mapping between an original source/line/column and a
                 * generated line/column in this source map.
                 *
                 * @param Function aCallback
                 *        The function that is called with each mapping.
                 * @param Object aContext
                 *        Optional. If specified, this object will be the value of `this` every
                 *        time that `aCallback` is called.
                 * @param aOrder
                 *        Either `SourceMapConsumer.GENERATED_ORDER` or
                 *        `SourceMapConsumer.ORIGINAL_ORDER`. Specifies whether you want to
                 *        iterate over the mappings sorted by the generated file's line/column
                 *        order or the original's source/line/column order, respectively. Defaults to
                 *        `SourceMapConsumer.GENERATED_ORDER`.
                 */
                SourceMapConsumer.prototype.eachMapping = function (aCallback, aContext, aOrder) {
                    var context = aContext || null;
                    var order = aOrder || SourceMapConsumer.GENERATED_ORDER;
                    var mappings;
                    switch (order) {
                        case SourceMapConsumer.GENERATED_ORDER:
                            mappings = this._generatedMappings;
                            break;
                        case SourceMapConsumer.ORIGINAL_ORDER:
                            mappings = this._originalMappings;
                            break;
                        default:
                            throw new Error("Unknown order of iteration.");
                    }
                    var sourceRoot = this.sourceRoot;
                    mappings.map(function (mapping) {
                        var source = mapping.source;
                        if (source != null && sourceRoot != null) {
                            source = utils.join(sourceRoot, source);
                        }
                        return {
                            source: source,
                            generatedLine: mapping.generatedLine,
                            generatedColumn: mapping.generatedColumn,
                            originalLine: mapping.originalLine,
                            originalColumn: mapping.originalColumn,
                            name: mapping.name
                        };
                    }).forEach(aCallback, context);
                };
                SourceMapConsumer.GENERATED_ORDER = 1;
                SourceMapConsumer.ORIGINAL_ORDER = 2;
                return SourceMapConsumer;
            })();
            sourcemap.SourceMapConsumer = SourceMapConsumer;
        })(sourcemap = ast.sourcemap || (ast.sourcemap = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/* -*- Mode: js; js-indent-level: 2; -*- */
/*
 * Copyright 2011 Mozilla Foundation and contributors
 * Licensed under the New BSD license. See LICENSE or:
 * http://opensource.org/licenses/BSD-3-Clause
 */
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var sourcemap;
        (function (sourcemap) {
            var ArraySet = lib.utils.ArraySet;
            var base64VLQ = lib.utils.vlq;
            var util = lib.ast.sourcemap.utils;
            /**
             * An instance of the SourceMapGenerator represents a source map which is
             * being built incrementally. You may pass an object with the following
             * properties:
             *
             *   - file: The filename of the generated source.
             *   - sourceRoot: A root for all relative URLs in this source map.
             */
            var SourceMapGenerator = (function () {
                function SourceMapGenerator(aArgs) {
                    this._version = 3;
                    if (!aArgs) {
                        aArgs = {};
                    }
                    this._file = util.getArg(aArgs, 'file', null);
                    this._sourceRoot = util.getArg(aArgs, 'sourceRoot', null);
                    this._sources = new ArraySet();
                    this._names = new ArraySet();
                    this._mappings = [];
                    this._sourcesContents = null;
                }
                Object.defineProperty(SourceMapGenerator.prototype, "file", {
                    get: function () {
                        return this._file;
                    },
                    set: function (val) {
                        this._file = val;
                    },
                    enumerable: true,
                    configurable: true
                });
                Object.defineProperty(SourceMapGenerator.prototype, "sourceRoot", {
                    get: function () {
                        return this._sourceRoot;
                    },
                    set: function (val) {
                        this._sourceRoot = val;
                    },
                    enumerable: true,
                    configurable: true
                });
                Object.defineProperty(SourceMapGenerator.prototype, "sources", {
                    get: function () {
                        return this._sources;
                    },
                    set: function (val) {
                        this._sources = val;
                    },
                    enumerable: true,
                    configurable: true
                });
                Object.defineProperty(SourceMapGenerator.prototype, "names", {
                    get: function () {
                        return this._names;
                    },
                    set: function (val) {
                        this._names = val;
                    },
                    enumerable: true,
                    configurable: true
                });
                Object.defineProperty(SourceMapGenerator.prototype, "mappings", {
                    get: function () {
                        return this._mappings;
                    },
                    set: function (val) {
                        this._mappings = val;
                    },
                    enumerable: true,
                    configurable: true
                });
                Object.defineProperty(SourceMapGenerator.prototype, "sourcesContents", {
                    get: function () {
                        return this._sourcesContents;
                    },
                    set: function (val) {
                        this._sourcesContents = val;
                    },
                    enumerable: true,
                    configurable: true
                });
                /**
                 * Creates a new SourceMapGenerator based on a SourceMapConsumer
                 *
                 * @param aSourceMapConsumer The SourceMap.
                 */
                SourceMapGenerator.fromSourceMap = function (aSourceMapConsumer) {
                    var sourceRoot = aSourceMapConsumer.sourceRoot;
                    var generator = new SourceMapGenerator({
                        file: aSourceMapConsumer.file,
                        sourceRoot: sourceRoot
                    });
                    aSourceMapConsumer.eachMapping(function (mapping) {
                        var newMapping = {
                            generated: {
                                line: mapping.generatedLine,
                                column: mapping.generatedColumn
                            }
                        };
                        if (mapping.source != null) {
                            newMapping.source = mapping.source;
                            if (sourceRoot != null) {
                                newMapping.source = util.relative(sourceRoot, newMapping.source);
                            }
                            newMapping.original = {
                                line: mapping.originalLine,
                                column: mapping.originalColumn
                            };
                            if (mapping.name != null) {
                                newMapping.name = mapping.name;
                            }
                        }
                        generator.addMapping(newMapping);
                    });
                    aSourceMapConsumer.sources.forEach(function (sourceFile) {
                        var content = aSourceMapConsumer.sourceContentFor(sourceFile);
                        if (content != null) {
                            generator.setSourceContent(sourceFile, content);
                        }
                    });
                    return generator;
                };
                /**
                 * Add a single mapping from original source line and column to the generated
                 * source's line and column for this source map being created. The mapping
                 * object should have the following properties:
                 *
                 *   - generated: An object with the generated line and column positions.
                 *   - original: An object with the original line and column positions.
                 *   - source: The original source file (relative to the sourceRoot).
                 *   - name: An optional original token name for this mapping.
                 */
                SourceMapGenerator.prototype.addMapping = function (aArgs) {
                    var generated = util.getArg(aArgs, 'generated');
                    var original = util.getArg(aArgs, 'original', null);
                    var source = util.getArg(aArgs, 'source', null);
                    var name = util.getArg(aArgs, 'name', null);
                    this._validateMapping(generated, original, source, name);
                    if (source != null && !this._sources.has(source)) {
                        this._sources.add(source);
                    }
                    if (name != null && !this._names.has(name)) {
                        this._names.add(name);
                    }
                    this._mappings.push({
                        generatedLine: generated.line,
                        generatedColumn: generated.column,
                        originalLine: original != null && original.line,
                        originalColumn: original != null && original.column,
                        source: source,
                        name: name
                    });
                };
                /**
                 * Set the source content for a source file.
                 */
                SourceMapGenerator.prototype.setSourceContent = function (aSourceFile, aSourceContent) {
                    var source = aSourceFile;
                    if (this._sourceRoot != null) {
                        source = util.relative(this._sourceRoot, source);
                    }
                    if (aSourceContent != null) {
                        // Add the source content to the _sourcesContents map.
                        // Create a new _sourcesContents map if the property is null.
                        if (!this._sourcesContents) {
                            this._sourcesContents = {};
                        }
                        this._sourcesContents[util.toSetString(source)] = aSourceContent;
                    }
                    else if (this._sourcesContents) {
                        // Remove the source file from the _sourcesContents map.
                        // If the _sourcesContents map is empty, set the property to null.
                        delete this._sourcesContents[util.toSetString(source)];
                        if (Object.keys(this._sourcesContents).length === 0) {
                            this._sourcesContents = null;
                        }
                    }
                };
                /**
                 * Applies the mappings of a sub-source-map for a specific source file to the
                 * source map being generated. Each mapping to the supplied source file is
                 * rewritten using the supplied source map. Note: The resolution for the
                 * resulting mappings is the minimium of this map and the supplied map.
                 *
                 * @param aSourceMapConsumer The source map to be applied.
                 * @param aSourceFile Optional. The filename of the source file.
                 *        If omitted, SourceMapConsumer's file property will be used.
                 * @param aSourceMapPath Optional. The dirname of the path to the source map
                 *        to be applied. If relative, it is relative to the SourceMapConsumer.
                 *        This parameter is needed when the two source maps aren't in the same
                 *        directory, and the source map to be applied contains relative source
                 *        paths. If so, those relative source paths need to be rewritten
                 *        relative to the SourceMapGenerator.
                 */
                SourceMapGenerator.prototype.applySourceMap = function (aSourceMapConsumer, aSourceFile, aSourceMapPath) {
                    var sourceFile = aSourceFile;
                    // If aSourceFile is omitted, we will use the file property of the SourceMap
                    if (aSourceFile == null) {
                        if (aSourceMapConsumer.file == null) {
                            throw new Error('applySourceMap requires either an explicit source file, ' + 'or the source map\'s "file" property. Both were omitted.');
                        }
                        sourceFile = aSourceMapConsumer.file;
                    }
                    var sourceRoot = this._sourceRoot;
                    // Make "sourceFile" relative if an absolute Url is passed.
                    if (sourceRoot != null) {
                        sourceFile = util.relative(sourceRoot, sourceFile);
                    }
                    // Applying the SourceMap can add and remove items from the sources and
                    // the names array.
                    var newSources = new ArraySet();
                    var newNames = new ArraySet();
                    // Find mappings for the "sourceFile"
                    this._mappings.forEach(function (mapping) {
                        if (mapping.source === sourceFile && mapping.originalLine != null) {
                            // Check if it can be mapped by the source map, then update the mapping.
                            var original = aSourceMapConsumer.originalPositionFor({
                                line: mapping.originalLine,
                                column: mapping.originalColumn
                            });
                            if (original.source != null) {
                                // Copy mapping
                                mapping.source = original.source;
                                if (aSourceMapPath != null) {
                                    mapping.source = util.join(aSourceMapPath, mapping.source);
                                }
                                if (sourceRoot != null) {
                                    mapping.source = util.relative(sourceRoot, mapping.source);
                                }
                                mapping.originalLine = original.line;
                                mapping.originalColumn = original.column;
                                if (original.name != null) {
                                    mapping.name = original.name;
                                }
                            }
                        }
                        var source = mapping.source;
                        if (source != null && !newSources.has(source)) {
                            newSources.add(source);
                        }
                        var name = mapping.name;
                        if (name != null && !newNames.has(name)) {
                            newNames.add(name);
                        }
                    }, this);
                    this._sources = newSources;
                    this._names = newNames;
                    // Copy sourcesContents of applied map.
                    aSourceMapConsumer.sources.forEach(function (sourceFile) {
                        var content = aSourceMapConsumer.sourceContentFor(sourceFile);
                        if (content != null) {
                            if (aSourceMapPath != null) {
                                sourceFile = util.join(aSourceMapPath, sourceFile);
                            }
                            if (sourceRoot != null) {
                                sourceFile = util.relative(sourceRoot, sourceFile);
                            }
                            this.setSourceContent(sourceFile, content);
                        }
                    }, this);
                };
                /**
                 * A mapping can have one of the three levels of data:
                 *
                 *   1. Just the generated position.
                 *   2. The Generated position, original position, and original source.
                 *   3. Generated and original position, original source, as well as a name
                 *      token.
                 *
                 * To maintain consistency, we validate that any new mapping being added falls
                 * in to one of these categories.
                 */
                SourceMapGenerator.prototype._validateMapping = function (aGenerated, aOriginal, aSource, aName) {
                    if (aGenerated && 'line' in aGenerated && 'column' in aGenerated && aGenerated.line > 0 && aGenerated.column >= 0 && !aOriginal && !aSource && !aName) {
                        // Case 1.
                        return;
                    }
                    else if (aGenerated && 'line' in aGenerated && 'column' in aGenerated && aOriginal && 'line' in aOriginal && 'column' in aOriginal && aGenerated.line > 0 && aGenerated.column >= 0 && aOriginal.line > 0 && aOriginal.column >= 0 && aSource) {
                        // Cases 2 and 3.
                        return;
                    }
                    else {
                        throw new Error('Invalid mapping: ' + JSON.stringify({
                            generated: aGenerated,
                            source: aSource,
                            original: aOriginal,
                            name: aName
                        }));
                    }
                };
                /**
                 * Serialize the accumulated mappings in to the stream of base 64 VLQs
                 * specified by the source map format.
                 */
                SourceMapGenerator.prototype._serializeMappings = function () {
                    var previousGeneratedColumn = 0;
                    var previousGeneratedLine = 1;
                    var previousOriginalColumn = 0;
                    var previousOriginalLine = 0;
                    var previousName = 0;
                    var previousSource = 0;
                    var result = '';
                    var mapping;
                    // The mappings must be guaranteed to be in sorted order before we start
                    // serializing them or else the generated line numbers (which are defined
                    // via the ';' separators) will be all messed up. Note: it might be more
                    // performant to maintain the sorting as we insert them, rather than as we
                    // serialize them, but the big O is the same either way.
                    this._mappings.sort(sourcemap.utils.compareByGeneratedPositions);
                    for (var i = 0, len = this._mappings.length; i < len; i++) {
                        mapping = this._mappings[i];
                        if (mapping.generatedLine !== previousGeneratedLine) {
                            previousGeneratedColumn = 0;
                            while (mapping.generatedLine !== previousGeneratedLine) {
                                result += ';';
                                previousGeneratedLine++;
                            }
                        }
                        else {
                            if (i > 0) {
                                if (!sourcemap.utils.compareByGeneratedPositions(mapping, this._mappings[i - 1])) {
                                    continue;
                                }
                                result += ',';
                            }
                        }
                        result += base64VLQ.encode(mapping.generatedColumn - previousGeneratedColumn);
                        previousGeneratedColumn = mapping.generatedColumn;
                        if (mapping.source != null) {
                            result += base64VLQ.encode(this._sources.indexOf(mapping.source) - previousSource);
                            previousSource = this._sources.indexOf(mapping.source);
                            // lines are stored 0-based in SourceMap spec version 3
                            result += base64VLQ.encode(mapping.originalLine - 1 - previousOriginalLine);
                            previousOriginalLine = mapping.originalLine - 1;
                            result += base64VLQ.encode(mapping.originalColumn - previousOriginalColumn);
                            previousOriginalColumn = mapping.originalColumn;
                            if (mapping.name != null) {
                                result += base64VLQ.encode(this._names.indexOf(mapping.name) - previousName);
                                previousName = this._names.indexOf(mapping.name);
                            }
                        }
                    }
                    return result;
                };
                SourceMapGenerator.prototype._generateSourcesContent = function (aSources, aSourceRoot) {
                    return aSources.map(function (source) {
                        if (!this._sourcesContents) {
                            return null;
                        }
                        if (aSourceRoot != null) {
                            source = util.relative(aSourceRoot, source);
                        }
                        var key = util.toSetString(source);
                        return Object.prototype.hasOwnProperty.call(this._sourcesContents, key) ? this._sourcesContents[key] : null;
                    }, this);
                };
                /**
                 * Externalize the source map.
                 */
                SourceMapGenerator.prototype.toJSON = function () {
                    var map = {
                        version: this._version,
                        sources: this._sources.toArray(),
                        names: this._names.toArray(),
                        mappings: this._serializeMappings()
                    };
                    if (this._file != null) {
                        map.file = this._file;
                    }
                    if (this._sourceRoot != null) {
                        map.sourceRoot = this._sourceRoot;
                    }
                    if (this._sourcesContents) {
                        map.sourcesContent = this._generateSourcesContent(map.sources, map.sourceRoot);
                    }
                    return map;
                };
                /**
                 * Render the source map being generated to a string.
                 */
                SourceMapGenerator.prototype.toString = function () {
                    return JSON.stringify(this);
                };
                return SourceMapGenerator;
            })();
            sourcemap.SourceMapGenerator = SourceMapGenerator;
        })(sourcemap = ast.sourcemap || (ast.sourcemap = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/* -*- Mode: js; js-indent-level: 2; -*- */
/*
 * Copyright 2011 Mozilla Foundation and contributors
 * Licensed under the New BSD license. See LICENSE or:
 * http://opensource.org/licenses/BSD-3-Clause
 */
/// <reference path="utils.ts" />
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var sourcemap;
        (function (sourcemap) {
            var utils = lib.ast.sourcemap.utils;
            // Matches a Windows-style `\r\n` newline or a `\n` newline used by all other
            // operating systems these days (capturing the result).
            var REGEX_NEWLINE = /(\r?\n)/;
            // Matches a Windows-style newline, or any character.
            var REGEX_CHARACTER = /\r\n|[\s\S]/g;
            /**
             * SourceNodes provide a way to abstract over interpolating/concatenating
             * snippets of generated JavaScript source code while maintaining the line and
             * column information associated with the original source code.
             *
             * @param aLine The original line number.
             * @param aColumn The original column number.
             * @param aSource The original source's filename.
             * @param aChunks Optional. An array of strings which are snippets of
             *        generated JS, or other SourceNodes.
             * @param aName The original identifier.
             */
            var SourceNode = (function () {
                function SourceNode(aLine, aColumn, aSource, aChunks, aName) {
                    this.children = [];
                    this.sourceContents = {};
                    this.line = aLine == null ? null : aLine;
                    this.column = aColumn == null ? null : aColumn;
                    this.source = aSource == null ? null : aSource;
                    this.name = aName == null ? null : aName;
                    if (aChunks != null)
                        this.add(aChunks);
                }
                /**
                 * Creates a SourceNode from generated code and a SourceMapConsumer.
                 *
                 * @param aGeneratedCode The generated code
                 * @param aSourceMapConsumer The SourceMap for the generated code
                 * @param aRelativePath Optional. The path that relative sources in the
                 *        SourceMapConsumer should be relative to.
                 */
                SourceNode.fromStringWithSourceMap = function (aGeneratedCode, aSourceMapConsumer, aRelativePath) {
                    // The SourceNode we want to fill with the generated code
                    // and the SourceMap
                    var node = new SourceNode();
                    // All even indices of this array are one line of the generated code,
                    // while all odd indices are the newlines between two adjacent lines
                    // (since `REGEX_NEWLINE` captures its match).
                    // Processed fragments are removed from this array, by calling `shiftNextLine`.
                    var remainingLines = aGeneratedCode.split(REGEX_NEWLINE);
                    var shiftNextLine = function () {
                        var lineContents = remainingLines.shift();
                        // The last line of a file might not have a newline.
                        var newLine = remainingLines.shift() || "";
                        return lineContents + newLine;
                    };
                    // We need to remember the position of "remainingLines"
                    var lastGeneratedLine = 1, lastGeneratedColumn = 0;
                    // The generate SourceNodes we need a code range.
                    // To extract it current and last mapping is used.
                    // Here we store the last mapping.
                    var lastMapping = null;
                    aSourceMapConsumer.eachMapping(function (mapping) {
                        if (lastMapping !== null) {
                            var code;
                            // We add the code from "lastMapping" to "mapping":
                            // First check if there is a new line in between.
                            if (lastGeneratedLine < mapping.generatedLine) {
                                code = "";
                                // Associate first line with "lastMapping"
                                addMappingWithCode(lastMapping, shiftNextLine());
                                lastGeneratedLine++;
                                lastGeneratedColumn = 0;
                            }
                            else {
                                // There is no new line in between.
                                // Associate the code between "lastGeneratedColumn" and
                                // "mapping.generatedColumn" with "lastMapping"
                                var nextLine = remainingLines[0];
                                code = nextLine.substr(0, mapping.generatedColumn - lastGeneratedColumn);
                                remainingLines[0] = nextLine.substr(mapping.generatedColumn - lastGeneratedColumn);
                                lastGeneratedColumn = mapping.generatedColumn;
                                addMappingWithCode(lastMapping, code);
                                // No more remaining code, continue
                                lastMapping = mapping;
                                return;
                            }
                        }
                        while (lastGeneratedLine < mapping.generatedLine) {
                            node.add(shiftNextLine());
                            lastGeneratedLine++;
                        }
                        if (lastGeneratedColumn < mapping.generatedColumn) {
                            var nextLine = remainingLines[0];
                            node.add(nextLine.substr(0, mapping.generatedColumn));
                            remainingLines[0] = nextLine.substr(mapping.generatedColumn);
                            lastGeneratedColumn = mapping.generatedColumn;
                        }
                        lastMapping = mapping;
                    }, this);
                    // We have processed all mappings.
                    if (remainingLines.length > 0) {
                        if (lastMapping) {
                            // Associate the remaining code in the current line with "lastMapping"
                            addMappingWithCode(lastMapping, shiftNextLine());
                        }
                        // and add the remaining lines without any mapping
                        node.add(remainingLines.join(""));
                    }
                    // Copy sourcesContent into SourceNode
                    aSourceMapConsumer.sources.forEach(function (sourceFile) {
                        var content = aSourceMapConsumer.sourceContentFor(sourceFile);
                        if (content != null) {
                            if (aRelativePath != null) {
                                sourceFile = utils.join(aRelativePath, sourceFile);
                            }
                            node.setSourceContent(sourceFile, content);
                        }
                    });
                    return node;
                    function addMappingWithCode(mapping, code) {
                        if (mapping === null || mapping.source === undefined) {
                            node.add(code);
                        }
                        else {
                            var source = aRelativePath ? utils.join(aRelativePath, mapping.source) : mapping.source;
                            node.add(new SourceNode(mapping.originalLine, mapping.originalColumn, source, code, mapping.name));
                        }
                    }
                };
                /**
                 * Add a chunk of generated JS to this source node.
                 *
                 * @param aChunk A string snippet of generated JS code, another instance of
                 *        SourceNode, or an array where each member is one of those things.
                 */
                SourceNode.prototype.add = function (aChunk) {
                    if (Array.isArray(aChunk)) {
                        aChunk.forEach(function (chunk) {
                            this.add(chunk);
                        }, this);
                    }
                    else if (aChunk instanceof SourceNode || typeof aChunk === "string") {
                        if (aChunk) {
                            this.children.push(aChunk);
                        }
                    }
                    else {
                        throw new TypeError("Expected a SourceNode, string, or an array of SourceNodes and strings. Got " + aChunk);
                    }
                    return this;
                };
                /**
                 * Add a chunk of generated JS to the beginning of this source node.
                 *
                 * @param aChunk A string snippet of generated JS code, another instance of
                 *        SourceNode, or an array where each member is one of those things.
                 */
                SourceNode.prototype.prepend = function (aChunk) {
                    if (Array.isArray(aChunk)) {
                        for (var i = aChunk.length - 1; i >= 0; i--) {
                            this.prepend(aChunk[i]);
                        }
                    }
                    else if (aChunk instanceof SourceNode || typeof aChunk === "string") {
                        this.children.unshift(aChunk);
                    }
                    else {
                        throw new TypeError("Expected a SourceNode, string, or an array of SourceNodes and strings. Got " + aChunk);
                    }
                    return this;
                };
                /**
                 * Walk over the tree of JS snippets in this node and its children. The
                 * walking function is called once for each snippet of JS and is passed that
                 * snippet and the its original associated source's line/column location.
                 *
                 * @param aFn The traversal function.
                 */
                SourceNode.prototype.walk = function (aFn) {
                    var chunk;
                    for (var i = 0, len = this.children.length; i < len; i++) {
                        chunk = this.children[i];
                        if (chunk instanceof SourceNode) {
                            chunk.walk(aFn);
                        }
                        else {
                            if (chunk !== '') {
                                aFn(chunk, {
                                    source: this.source,
                                    line: this.line,
                                    column: this.column,
                                    name: this.name
                                });
                            }
                        }
                    }
                };
                /**
                 * Like `String.prototype.join` except for SourceNodes. Inserts `aStr` between
                 * each of `this.children`.
                 *
                 * @param aSep The separator.
                 */
                SourceNode.prototype.join = function (aSep) {
                    var newChildren;
                    var i;
                    var len = this.children.length;
                    if (len > 0) {
                        newChildren = [];
                        for (i = 0; i < len - 1; i++) {
                            newChildren.push(this.children[i]);
                            newChildren.push(aSep);
                        }
                        newChildren.push(this.children[i]);
                        this.children = newChildren;
                    }
                    return this;
                };
                /**
                 * Call String.prototype.replace on the very right-most source snippet. Useful
                 * for trimming whitespace from the end of a source node, etc.
                 *
                 * @param aPattern The pattern to replace.
                 * @param aReplacement The thing to replace the pattern with.
                 */
                SourceNode.prototype.replaceRight = function (aPattern, aReplacement) {
                    var lastChild = this.children[this.children.length - 1];
                    if (lastChild instanceof SourceNode) {
                        lastChild.replaceRight(aPattern, aReplacement);
                    }
                    else if (typeof lastChild === 'string') {
                        this.children[this.children.length - 1] = lastChild.replace(aPattern, aReplacement);
                    }
                    else {
                        this.children.push(''.replace(aPattern, aReplacement));
                    }
                    return this;
                };
                /**
                 * Set the source content for a source file. This will be added to the SourceMapGenerator
                 * in the sourcesContent field.
                 *
                 * @param aSourceFile The filename of the source file
                 * @param aSourceContent The content of the source file
                 */
                SourceNode.prototype.setSourceContent = function (aSourceFile, aSourceContent) {
                    this.sourceContents[utils.toSetString(aSourceFile)] = aSourceContent;
                };
                /**
                 * Walk over the tree of SourceNodes. The walking function is called for each
                 * source file content and is passed the filename and source content.
                 *
                 * @param aFn The traversal function.
                 */
                SourceNode.prototype.walkSourceContents = function (aFn) {
                    for (var i = 0, len = this.children.length; i < len; i++) {
                        if (this.children[i] instanceof SourceNode) {
                            this.children[i].walkSourceContents(aFn);
                        }
                    }
                    var sources = Object.keys(this.sourceContents);
                    for (var i = 0, len = sources.length; i < len; i++) {
                        aFn(utils.fromSetString(sources[i]), this.sourceContents[sources[i]]);
                    }
                };
                /**
                 * Return the string representation of this source node. Walks over the tree
                 * and concatenates all the various snippets together to one string.
                 */
                SourceNode.prototype.toString = function () {
                    var str = "";
                    this.walk(function (chunk) {
                        str += chunk;
                    });
                    return str;
                };
                /**
                 * Returns the string representation of this source node along with a source
                 * map.
                 */
                SourceNode.prototype.toStringWithSourceMap = function (aArgs) {
                    var generated = {
                        code: "",
                        line: 1,
                        column: 0
                    };
                    var map = new sourcemap.SourceMapGenerator(aArgs);
                    var sourceMappingActive = false;
                    var lastOriginalSource = null;
                    var lastOriginalLine = null;
                    var lastOriginalColumn = null;
                    var lastOriginalName = null;
                    this.walk(function (chunk, original) {
                        generated.code += chunk;
                        if (original.source !== null && original.line !== null && original.column !== null) {
                            if (lastOriginalSource !== original.source || lastOriginalLine !== original.line || lastOriginalColumn !== original.column || lastOriginalName !== original.name) {
                                map.addMapping({
                                    source: original.source,
                                    original: {
                                        line: original.line,
                                        column: original.column
                                    },
                                    generated: {
                                        line: generated.line,
                                        column: generated.column
                                    },
                                    name: original.name
                                });
                            }
                            lastOriginalSource = original.source;
                            lastOriginalLine = original.line;
                            lastOriginalColumn = original.column;
                            lastOriginalName = original.name;
                            sourceMappingActive = true;
                        }
                        else if (sourceMappingActive) {
                            map.addMapping({
                                generated: {
                                    line: generated.line,
                                    column: generated.column
                                }
                            });
                            lastOriginalSource = null;
                            sourceMappingActive = false;
                        }
                        chunk.match(REGEX_CHARACTER).forEach(function (ch, idx, array) {
                            if (REGEX_NEWLINE.test(ch)) {
                                generated.line++;
                                generated.column = 0;
                                // Mappings end at eol
                                if (idx + 1 === array.length) {
                                    lastOriginalSource = null;
                                    sourceMappingActive = false;
                                }
                                else if (sourceMappingActive) {
                                    map.addMapping({
                                        source: original.source,
                                        original: {
                                            line: original.line,
                                            column: original.column
                                        },
                                        generated: {
                                            line: generated.line,
                                            column: generated.column
                                        },
                                        name: original.name
                                    });
                                }
                            }
                            else {
                                generated.column += ch.length;
                            }
                        });
                    });
                    this.walkSourceContents(function (sourceFile, sourceContent) {
                        map.setSourceContent(sourceFile, sourceContent);
                    });
                    return { code: generated.code, map: map };
                };
                return SourceNode;
            })();
            sourcemap.SourceNode = SourceNode;
        })(sourcemap = ast.sourcemap || (ast.sourcemap = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/// <reference path="utils.ts" />
/// <reference path="consumer.ts" />
/// <reference path="gen.ts" />
/// <reference path="node.ts" />
/*
 Copyright (C) 2012-2014 Yusuke Suzuki <utatane.tea@gmail.com>
 Copyright (C) 2014 Ivan Nikulin <ifaaan@gmail.com>
 Copyright (C) 2012-2013 Michael Ficarra <escodegen.copyright@michael.ficarra.me>
 Copyright (C) 2012-2013 Mathias Bynens <mathias@qiwi.be>
 Copyright (C) 2013 Irakli Gozalishvili <rfobic@gmail.com>
 Copyright (C) 2012 Robert Gust-Bardon <donate@robert.gust-bardon.org>
 Copyright (C) 2012 John Freeman <jfreeman08@gmail.com>
 Copyright (C) 2011-2012 Ariya Hidayat <ariya.hidayat@gmail.com>
 Copyright (C) 2012 Joost-Wim Boekesteijn <joost-wim@boekesteijn.nl>
 Copyright (C) 2012 Kris Kowal <kris.kowal@cixar.com>
 Copyright (C) 2012 Arpad Borsos <arpad.borsos@googlemail.com>
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// from https://github.com/estools/escodegen
/// <reference path="sourcemap/sourcemap.ts" /> 
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var gen;
        (function (gen) {
            var esutils = lib.ast.utils;
            var Syntax, Precedence, BinaryPrecedence, isArray, base, indent, json, renumber, hexadecimal, quotes, escapeless, newline, space, parentheses, semicolons, safeConcatenation, directive, extra, parse, sourceMap, FORMAT_MINIFY, FORMAT_DEFAULTS;
            var SourceNode = lib.ast.sourcemap.SourceNode;
            Syntax = {
                AssignmentExpression: 'AssignmentExpression',
                ArrayExpression: 'ArrayExpression',
                ArrayPattern: 'ArrayPattern',
                ArrowFunctionExpression: 'ArrowFunctionExpression',
                BlockStatement: 'BlockStatement',
                BinaryExpression: 'BinaryExpression',
                BreakStatement: 'BreakStatement',
                CallExpression: 'CallExpression',
                CatchClause: 'CatchClause',
                ClassBody: 'ClassBody',
                ClassDeclaration: 'ClassDeclaration',
                ClassExpression: 'ClassExpression',
                ComprehensionBlock: 'ComprehensionBlock',
                ComprehensionExpression: 'ComprehensionExpression',
                ConditionalExpression: 'ConditionalExpression',
                ContinueStatement: 'ContinueStatement',
                DirectiveStatement: 'DirectiveStatement',
                DoWhileStatement: 'DoWhileStatement',
                DebuggerStatement: 'DebuggerStatement',
                EmptyStatement: 'EmptyStatement',
                ExportBatchSpecifier: 'ExportBatchSpecifier',
                ExportDeclaration: 'ExportDeclaration',
                ExportSpecifier: 'ExportSpecifier',
                ExpressionStatement: 'ExpressionStatement',
                ForStatement: 'ForStatement',
                ForInStatement: 'ForInStatement',
                ForOfStatement: 'ForOfStatement',
                FunctionDeclaration: 'FunctionDeclaration',
                FunctionExpression: 'FunctionExpression',
                GeneratorExpression: 'GeneratorExpression',
                Identifier: 'Identifier',
                IfStatement: 'IfStatement',
                ImportDeclaration: 'ImportDeclaration',
                ImportDefaultSpecifier: 'ImportDefaultSpecifier',
                ImportNamespaceSpecifier: 'ImportNamespaceSpecifier',
                ImportSpecifier: 'ImportSpecifier',
                Literal: 'Literal',
                LabeledStatement: 'LabeledStatement',
                LogicalExpression: 'LogicalExpression',
                MemberExpression: 'MemberExpression',
                MethodDefinition: 'MethodDefinition',
                ModuleSpecifier: 'ModuleSpecifier',
                NewExpression: 'NewExpression',
                ObjectExpression: 'ObjectExpression',
                ObjectPattern: 'ObjectPattern',
                Program: 'Program',
                Property: 'Property',
                ReturnStatement: 'ReturnStatement',
                SequenceExpression: 'SequenceExpression',
                SpreadElement: 'SpreadElement',
                SwitchStatement: 'SwitchStatement',
                SwitchCase: 'SwitchCase',
                TaggedTemplateExpression: 'TaggedTemplateExpression',
                TemplateElement: 'TemplateElement',
                TemplateLiteral: 'TemplateLiteral',
                ThisExpression: 'ThisExpression',
                ThrowStatement: 'ThrowStatement',
                TryStatement: 'TryStatement',
                UnaryExpression: 'UnaryExpression',
                UpdateExpression: 'UpdateExpression',
                VariableDeclaration: 'VariableDeclaration',
                VariableDeclarator: 'VariableDeclarator',
                WhileStatement: 'WhileStatement',
                WithStatement: 'WithStatement',
                YieldExpression: 'YieldExpression'
            };
            // Generation is done by generateExpression.
            function isExpression(node) {
                return CodeGenerator.Expression.hasOwnProperty(node.type);
            }
            // Generation is done by generateStatement.
            function isStatement(node) {
                return CodeGenerator.Statement.hasOwnProperty(node.type);
            }
            Precedence = {
                Sequence: 0,
                Yield: 1,
                Assignment: 1,
                Conditional: 2,
                ArrowFunction: 2,
                LogicalOR: 3,
                LogicalAND: 4,
                BitwiseOR: 5,
                BitwiseXOR: 6,
                BitwiseAND: 7,
                Equality: 8,
                Relational: 9,
                BitwiseSHIFT: 10,
                Additive: 11,
                Multiplicative: 12,
                Unary: 13,
                Postfix: 14,
                Call: 15,
                New: 16,
                TaggedTemplate: 17,
                Member: 18,
                Primary: 19
            };
            BinaryPrecedence = {
                '||': Precedence.LogicalOR,
                '&&': Precedence.LogicalAND,
                '|': Precedence.BitwiseOR,
                '^': Precedence.BitwiseXOR,
                '&': Precedence.BitwiseAND,
                '==': Precedence.Equality,
                '!=': Precedence.Equality,
                '===': Precedence.Equality,
                '!==': Precedence.Equality,
                'is': Precedence.Equality,
                'isnt': Precedence.Equality,
                '<': Precedence.Relational,
                '>': Precedence.Relational,
                '<=': Precedence.Relational,
                '>=': Precedence.Relational,
                'in': Precedence.Relational,
                'instanceof': Precedence.Relational,
                '<<': Precedence.BitwiseSHIFT,
                '>>': Precedence.BitwiseSHIFT,
                '>>>': Precedence.BitwiseSHIFT,
                '+': Precedence.Additive,
                '-': Precedence.Additive,
                '*': Precedence.Multiplicative,
                '%': Precedence.Multiplicative,
                '/': Precedence.Multiplicative
            };
            //Flags
            var F_ALLOW_IN = 1, F_ALLOW_CALL = 1 << 1, F_ALLOW_UNPARATH_NEW = 1 << 2, F_FUNC_BODY = 1 << 3, F_DIRECTIVE_CTX = 1 << 4, F_SEMICOLON_OPT = 1 << 5;
            //Expression flag sets
            //NOTE: Flag order:
            // F_ALLOW_IN
            // F_ALLOW_CALL
            // F_ALLOW_UNPARATH_NEW
            var E_FTT = F_ALLOW_CALL | F_ALLOW_UNPARATH_NEW, E_TTF = F_ALLOW_IN | F_ALLOW_CALL, E_TTT = F_ALLOW_IN | F_ALLOW_CALL | F_ALLOW_UNPARATH_NEW, E_TFF = F_ALLOW_IN, E_FFT = F_ALLOW_UNPARATH_NEW, E_TFT = F_ALLOW_IN | F_ALLOW_UNPARATH_NEW;
            //Statement flag sets
            //NOTE: Flag order:
            // F_ALLOW_IN
            // F_FUNC_BODY
            // F_DIRECTIVE_CTX
            // F_SEMICOLON_OPT
            var S_TFFF = F_ALLOW_IN, S_TFFT = F_ALLOW_IN | F_SEMICOLON_OPT, S_FFFF = 0x00, S_TFTF = F_ALLOW_IN | F_DIRECTIVE_CTX, S_TTFF = F_ALLOW_IN | F_FUNC_BODY;
            function getDefaultOptions() {
                // default options
                return {
                    indent: null,
                    base: null,
                    parse: null,
                    comment: false,
                    format: {
                        indent: {
                            style: '    ',
                            base: 0,
                            adjustMultilineComment: false
                        },
                        newline: '\n',
                        space: ' ',
                        json: false,
                        renumber: false,
                        hexadecimal: false,
                        quotes: 'single',
                        escapeless: false,
                        compact: false,
                        parentheses: true,
                        semicolons: true,
                        safeConcatenation: false
                    },
                    moz: {
                        comprehensionExpressionStartsWithAssignment: false,
                        starlessGenerator: false
                    },
                    sourceMap: null,
                    sourceMapRoot: null,
                    sourceMapWithCode: false,
                    directive: false,
                    raw: true,
                    verbatim: null
                };
            }
            function stringRepeat(str, num) {
                var result = '';
                for (num |= 0; num > 0; num >>>= 1, str += str) {
                    if (num & 1) {
                        result += str;
                    }
                }
                return result;
            }
            isArray = Array.isArray;
            if (!isArray) {
                isArray = function isArray(array) {
                    return Object.prototype.toString.call(array) === '[object Array]';
                };
            }
            function hasLineTerminator(str) {
                return (/[\r\n]/g).test(str);
            }
            function endsWithLineTerminator(str) {
                var len = str.length;
                return len && esutils.isLineTerminator(str.charCodeAt(len - 1));
            }
            function merge(target, override) {
                var key;
                for (key in override) {
                    if (override.hasOwnProperty(key)) {
                        target[key] = override[key];
                    }
                }
                return target;
            }
            function updateDeeply(target, override) {
                var key, val;
                function isHashObject(target) {
                    return typeof target === 'object' && target instanceof Object && !(target instanceof RegExp);
                }
                for (key in override) {
                    if (override.hasOwnProperty(key)) {
                        val = override[key];
                        if (isHashObject(val)) {
                            if (isHashObject(target[key])) {
                                updateDeeply(target[key], val);
                            }
                            else {
                                target[key] = updateDeeply({}, val);
                            }
                        }
                        else {
                            target[key] = val;
                        }
                    }
                }
                return target;
            }
            function generateNumber(value) {
                var result, point, temp, exponent, pos;
                if (value !== value) {
                    throw new Error('Numeric literal whose value is NaN');
                }
                if (value < 0 || (value === 0 && 1 / value < 0)) {
                    throw new Error('Numeric literal whose value is negative');
                }
                if (value === 1 / 0) {
                    return json ? 'null' : renumber ? '1e400' : '1e+400';
                }
                result = '' + value;
                if (!renumber || result.length < 3) {
                    return result;
                }
                point = result.indexOf('.');
                if (!json && result.charCodeAt(0) === 0x30 && point === 1) {
                    point = 0;
                    result = result.slice(1);
                }
                temp = result;
                result = result.replace('e+', 'e');
                exponent = 0;
                if ((pos = temp.indexOf('e')) > 0) {
                    exponent = +temp.slice(pos + 1);
                    temp = temp.slice(0, pos);
                }
                if (point >= 0) {
                    exponent -= temp.length - point - 1;
                    temp = +(temp.slice(0, point) + temp.slice(point + 1)) + '';
                }
                pos = 0;
                while (temp.charCodeAt(temp.length + pos - 1) === 0x30) {
                    --pos;
                }
                if (pos !== 0) {
                    exponent -= pos;
                    temp = temp.slice(0, pos);
                }
                if (exponent !== 0) {
                    temp += 'e' + exponent;
                }
                if ((temp.length < result.length || (hexadecimal && value > 1e12 && Math.floor(value) === value && (temp = '0x' + value.toString(16)).length < result.length)) && +temp === value) {
                    result = temp;
                }
                return result;
            }
            // Generate valid RegExp expression.
            // This function is based on https://github.com/Constellation/iv Engine
            function escapeRegExpCharacter(ch, previousIsBackslash) {
                // not handling '\' and handling \u2028 or \u2029 to unicode escape sequence
                if ((ch & ~1) === 0x2028) {
                    return (previousIsBackslash ? 'u' : '\\u') + ((ch === 0x2028) ? '2028' : '2029');
                }
                else if (ch === 10 || ch === 13) {
                    return (previousIsBackslash ? '' : '\\') + ((ch === 10) ? 'n' : 'r');
                }
                return String.fromCharCode(ch);
            }
            function generateRegExp(reg) {
                var match, result, flags, i, iz, ch, characterInBrack, previousIsBackslash;
                result = reg.toString();
                if (reg.source) {
                    // extract flag from toString result
                    match = result.match(/\/([^/]*)$/);
                    if (!match) {
                        return result;
                    }
                    flags = match[1];
                    result = '';
                    characterInBrack = false;
                    previousIsBackslash = false;
                    for (i = 0, iz = reg.source.length; i < iz; ++i) {
                        ch = reg.source.charCodeAt(i);
                        if (!previousIsBackslash) {
                            if (characterInBrack) {
                                if (ch === 93) {
                                    characterInBrack = false;
                                }
                            }
                            else {
                                if (ch === 47) {
                                    result += '\\';
                                }
                                else if (ch === 91) {
                                    characterInBrack = true;
                                }
                            }
                            result += escapeRegExpCharacter(ch, previousIsBackslash);
                            previousIsBackslash = ch === 92; // \
                        }
                        else {
                            // if new RegExp("\\\n') is provided, create /\n/
                            result += escapeRegExpCharacter(ch, previousIsBackslash);
                            // prevent like /\\[/]/
                            previousIsBackslash = false;
                        }
                    }
                    return '/' + result + '/' + flags;
                }
                return result;
            }
            function escapeAllowedCharacter(code, next) {
                var hex;
                if (code === 0x08) {
                    return '\\b';
                }
                if (code === 0x0C) {
                    return '\\f';
                }
                if (code === 0x09) {
                    return '\\t';
                }
                hex = code.toString(16).toUpperCase();
                if (json || code > 0xFF) {
                    return '\\u' + '0000'.slice(hex.length) + hex;
                }
                else if (code === 0x0000 && !esutils.isDecimalDigit(next)) {
                    return '\\0';
                }
                else if (code === 0x000B) {
                    return '\\x0B';
                }
                else {
                    return '\\x' + '00'.slice(hex.length) + hex;
                }
            }
            function escapeDisallowedCharacter(code) {
                if (code === 0x5C) {
                    return '\\\\';
                }
                if (code === 0x0A) {
                    return '\\n';
                }
                if (code === 0x0D) {
                    return '\\r';
                }
                if (code === 0x2028) {
                    return '\\u2028';
                }
                if (code === 0x2029) {
                    return '\\u2029';
                }
                throw new Error('Incorrectly classified character');
            }
            function escapeDirective(str) {
                var i, iz, code, quote;
                quote = quotes === 'double' ? '"' : '\'';
                for (i = 0, iz = str.length; i < iz; ++i) {
                    code = str.charCodeAt(i);
                    if (code === 0x27) {
                        quote = '"';
                        break;
                    }
                    else if (code === 0x22) {
                        quote = '\'';
                        break;
                    }
                    else if (code === 0x5C) {
                        ++i;
                    }
                }
                return quote + str + quote;
            }
            function escapeString(str) {
                var result = '', i, len, code, singleQuotes = 0, doubleQuotes = 0, single, quote;
                for (i = 0, len = str.length; i < len; ++i) {
                    code = str.charCodeAt(i);
                    if (code === 0x27) {
                        ++singleQuotes;
                    }
                    else if (code === 0x22) {
                        ++doubleQuotes;
                    }
                    else if (code === 0x2F && json) {
                        result += '\\';
                    }
                    else if (esutils.isLineTerminator(code) || code === 0x5C) {
                        result += escapeDisallowedCharacter(code);
                        continue;
                    }
                    else if ((json && code < 0x20) || !(json || escapeless || (code >= 0x20 && code <= 0x7E))) {
                        result += escapeAllowedCharacter(code, str.charCodeAt(i + 1));
                        continue;
                    }
                    result += String.fromCharCode(code);
                }
                single = !(quotes === 'double' || (quotes === 'auto' && doubleQuotes < singleQuotes));
                quote = single ? '\'' : '"';
                if (!(single ? singleQuotes : doubleQuotes)) {
                    return quote + result + quote;
                }
                str = result;
                result = quote;
                for (i = 0, len = str.length; i < len; ++i) {
                    code = str.charCodeAt(i);
                    if ((code === 0x27 && single) || (code === 0x22 && !single)) {
                        result += '\\';
                    }
                    result += String.fromCharCode(code);
                }
                return result + quote;
            }
            /**
             * flatten an array to a string, where the array can contain
             * either strings or nested arrays
             */
            function flattenToString(arr) {
                var i, iz, elem, result = '';
                for (i = 0, iz = arr.length; i < iz; ++i) {
                    elem = arr[i];
                    result += isArray(elem) ? flattenToString(elem) : elem;
                }
                return result;
            }
            /**
             * convert generated to a SourceNode when source maps are enabled.
             */
            function toSourceNodeWhenNeeded(generated, node) {
                if (!sourceMap) {
                    // with no source maps, generated is either an
                    // array or a string.  if an array, flatten it.
                    // if a string, just return it
                    if (isArray(generated)) {
                        return flattenToString(generated);
                    }
                    else {
                        return generated;
                    }
                }
                if (node == null) {
                    if (generated instanceof SourceNode) {
                        return generated;
                    }
                    else {
                        node = {};
                    }
                }
                if (node.loc == null) {
                    return new SourceNode(null, null, sourceMap, generated, node.name || null);
                }
                return new SourceNode(node.loc.start.line, node.loc.start.column, (sourceMap === true ? node.loc.source || null : sourceMap), generated, node.name || null);
            }
            function noEmptySpace() {
                return (space) ? space : ' ';
            }
            function join(left, right) {
                var leftSource, rightSource, leftCharCode, rightCharCode;
                leftSource = toSourceNodeWhenNeeded(left).toString();
                if (leftSource.length === 0) {
                    return [right];
                }
                rightSource = toSourceNodeWhenNeeded(right).toString();
                if (rightSource.length === 0) {
                    return [left];
                }
                leftCharCode = leftSource.charCodeAt(leftSource.length - 1);
                rightCharCode = rightSource.charCodeAt(0);
                if ((leftCharCode === 0x2B || leftCharCode === 0x2D) && leftCharCode === rightCharCode || esutils.isIdentifierPart(leftCharCode) && esutils.isIdentifierPart(rightCharCode) || leftCharCode === 0x2F && rightCharCode === 0x69) {
                    return [left, noEmptySpace(), right];
                }
                else if (esutils.isWhiteSpace(leftCharCode) || esutils.isLineTerminator(leftCharCode) || esutils.isWhiteSpace(rightCharCode) || esutils.isLineTerminator(rightCharCode)) {
                    return [left, right];
                }
                return [left, space, right];
            }
            function addIndent(stmt) {
                return [base, stmt];
            }
            function withIndent(fn) {
                var previousBase;
                previousBase = base;
                base += indent;
                fn(base);
                base = previousBase;
            }
            function calculateSpaces(str) {
                var i;
                for (i = str.length - 1; i >= 0; --i) {
                    if (esutils.isLineTerminator(str.charCodeAt(i))) {
                        break;
                    }
                }
                return (str.length - 1) - i;
            }
            function adjustMultilineComment(value, specialBase) {
                var array, i, len, line, j, spaces, previousBase, sn;
                array = value.split(/\r\n|[\r\n]/);
                spaces = Number.MAX_VALUE;
                for (i = 1, len = array.length; i < len; ++i) {
                    line = array[i];
                    j = 0;
                    while (j < line.length && esutils.isWhiteSpace(line.charCodeAt(j))) {
                        ++j;
                    }
                    if (spaces > j) {
                        spaces = j;
                    }
                }
                if (typeof specialBase !== 'undefined') {
                    // pattern like
                    // {
                    //   var t = 20;  /*
                    //                 * this is comment
                    //                 */
                    // }
                    previousBase = base;
                    if (array[1][spaces] === '*') {
                        specialBase += ' ';
                    }
                    base = specialBase;
                }
                else {
                    if (spaces & 1) {
                        // /*
                        //  *
                        //  */
                        // If spaces are odd number, above pattern is considered.
                        // We waste 1 space.
                        --spaces;
                    }
                    previousBase = base;
                }
                for (i = 1, len = array.length; i < len; ++i) {
                    sn = toSourceNodeWhenNeeded(addIndent(array[i].slice(spaces)));
                    array[i] = sourceMap ? sn.join('') : sn;
                }
                base = previousBase;
                return array.join('\n');
            }
            function generateComment(comment, specialBase) {
                if (comment.type === 'Line') {
                    if (endsWithLineTerminator(comment.value)) {
                        return '//' + comment.value;
                    }
                    else {
                        // Always use LineTerminator
                        return '//' + comment.value + '\n';
                    }
                }
                if (extra.format.indent.adjustMultilineComment && /[\n\r]/.test(comment.value)) {
                    return adjustMultilineComment('/*' + comment.value + '*/', specialBase);
                }
                return '/*' + comment.value + '*/';
            }
            function addComments(stmt, result) {
                var i, len, comment, save, tailingToStatement, specialBase, fragment;
                if (stmt.leadingComments && stmt.leadingComments.length > 0) {
                    save = result;
                    comment = stmt.leadingComments[0];
                    result = [];
                    if (safeConcatenation && stmt.type === Syntax.Program && stmt.body.length === 0) {
                        result.push('\n');
                    }
                    result.push(generateComment(comment));
                    if (!endsWithLineTerminator(toSourceNodeWhenNeeded(result).toString())) {
                        result.push('\n');
                    }
                    for (i = 1, len = stmt.leadingComments.length; i < len; ++i) {
                        comment = stmt.leadingComments[i];
                        fragment = [generateComment(comment)];
                        if (!endsWithLineTerminator(toSourceNodeWhenNeeded(fragment).toString())) {
                            fragment.push('\n');
                        }
                        result.push(addIndent(fragment));
                    }
                    result.push(addIndent(save));
                }
                if (stmt.trailingComments) {
                    tailingToStatement = !endsWithLineTerminator(toSourceNodeWhenNeeded(result).toString());
                    specialBase = stringRepeat(' ', calculateSpaces(toSourceNodeWhenNeeded([base, result, indent]).toString()));
                    for (i = 0, len = stmt.trailingComments.length; i < len; ++i) {
                        comment = stmt.trailingComments[i];
                        if (tailingToStatement) {
                            // We assume target like following script
                            //
                            // var t = 20;  /**
                            //               * This is comment of t
                            //               */
                            if (i === 0) {
                                // first case
                                result = [result, indent];
                            }
                            else {
                                result = [result, specialBase];
                            }
                            result.push(generateComment(comment, specialBase));
                        }
                        else {
                            result = [result, addIndent(generateComment(comment))];
                        }
                        if (i !== len - 1 && !endsWithLineTerminator(toSourceNodeWhenNeeded(result).toString())) {
                            result = [result, '\n'];
                        }
                    }
                }
                return result;
            }
            function parenthesize(text, current, should) {
                if (current < should) {
                    return ['(', text, ')'];
                }
                return text;
            }
            function generateVerbatimString(string) {
                var i, iz, result;
                result = string.split(/\r\n|\n/);
                for (i = 1, iz = result.length; i < iz; i++) {
                    result[i] = newline + base + result[i];
                }
                return result;
            }
            function generateVerbatim(expr, precedence) {
                var verbatim, result, prec;
                verbatim = expr[extra.verbatim];
                if (typeof verbatim === 'string') {
                    result = parenthesize(generateVerbatimString(verbatim), Precedence.Sequence, precedence);
                }
                else {
                    // verbatim is object
                    result = generateVerbatimString(verbatim.content);
                    prec = (verbatim.precedence != null) ? verbatim.precedence : Precedence.Sequence;
                    result = parenthesize(result, prec, precedence);
                }
                return toSourceNodeWhenNeeded(result, expr);
            }
            var CodeGenerator = (function () {
                function CodeGenerator() {
                }
                // Helpers.
                CodeGenerator.prototype.maybeBlock = function (stmt, flags) {
                    var result, noLeadingComment, that = this;
                    noLeadingComment = !extra.comment || !stmt.leadingComments;
                    if (stmt.type === Syntax.BlockStatement && noLeadingComment) {
                        return [space, this.generateStatement(stmt, flags)];
                    }
                    if (stmt.type === Syntax.EmptyStatement && noLeadingComment) {
                        return ';';
                    }
                    withIndent(function () {
                        result = [
                            newline,
                            addIndent(that.generateStatement(stmt, flags))
                        ];
                    });
                    return result;
                };
                CodeGenerator.prototype.maybeBlockSuffix = function (stmt, result) {
                    var ends = endsWithLineTerminator(toSourceNodeWhenNeeded(result).toString());
                    if (stmt.type === Syntax.BlockStatement && (!extra.comment || !stmt.leadingComments) && !ends) {
                        return [result, space];
                    }
                    if (ends) {
                        return [result, base];
                    }
                    return [result, newline, base];
                };
                CodeGenerator.prototype.generateIdentifier = function (node) {
                    return toSourceNodeWhenNeeded(node.name, node);
                };
                CodeGenerator.prototype.generatePattern = function (node, precedence, flags) {
                    if (node.type === Syntax.Identifier) {
                        return this.generateIdentifier(node);
                    }
                    return this.generateExpression(node, precedence, flags);
                };
                CodeGenerator.prototype.generateFunctionParams = function (node) {
                    var i, iz, result, hasDefault;
                    hasDefault = false;
                    if (node.type === Syntax.ArrowFunctionExpression && !node.rest && (!node.defaults || node.defaults.length === 0) && node.params.length === 1 && node.params[0].type === Syntax.Identifier) {
                        // arg => { } case
                        result = [this.generateIdentifier(node.params[0])];
                    }
                    else {
                        result = ['('];
                        if (node.defaults) {
                            hasDefault = true;
                        }
                        for (i = 0, iz = node.params.length; i < iz; ++i) {
                            if (hasDefault && node.defaults[i]) {
                                // Handle default values.
                                result.push(this.generateAssignment(node.params[i], node.defaults[i], '=', Precedence.Assignment, E_TTT));
                            }
                            else {
                                result.push(this.generatePattern(node.params[i], Precedence.Assignment, E_TTT));
                            }
                            if (i + 1 < iz) {
                                result.push(',' + space);
                            }
                        }
                        if (node.rest) {
                            if (node.params.length) {
                                result.push(',' + space);
                            }
                            result.push('...');
                            result.push(this.generateIdentifier(node.rest));
                        }
                        result.push(')');
                    }
                    return result;
                };
                CodeGenerator.prototype.generateFunctionBody = function (node) {
                    var result, expr;
                    result = this.generateFunctionParams(node);
                    if (node.type === Syntax.ArrowFunctionExpression) {
                        result.push(space);
                        result.push('=>');
                    }
                    if (node.expression) {
                        result.push(space);
                        expr = this.generateExpression(node.body, Precedence.Assignment, E_TTT);
                        if (expr.toString().charAt(0) === '{') {
                            expr = ['(', expr, ')'];
                        }
                        result.push(expr);
                    }
                    else {
                        result.push(this.maybeBlock(node.body, S_TTFF));
                    }
                    return result;
                };
                CodeGenerator.prototype.generateIterationForStatement = function (operator, stmt, flags) {
                    var result = ['for' + space + '('], that = this;
                    withIndent(function () {
                        if (stmt.left.type === Syntax.VariableDeclaration) {
                            withIndent(function () {
                                result.push(stmt.left.kind + noEmptySpace());
                                result.push(that.generateStatement(stmt.left.declarations[0], S_FFFF));
                            });
                        }
                        else {
                            result.push(that.generateExpression(stmt.left, Precedence.Call, E_TTT));
                        }
                        result = join(result, operator);
                        result = [join(result, that.generateExpression(stmt.right, Precedence.Sequence, E_TTT)), ')'];
                    });
                    result.push(this.maybeBlock(stmt.body, flags));
                    return result;
                };
                CodeGenerator.prototype.generatePropertyKey = function (expr, computed) {
                    var result = [];
                    if (computed) {
                        result.push('[');
                    }
                    result.push(this.generateExpression(expr, Precedence.Sequence, E_TTT));
                    if (computed) {
                        result.push(']');
                    }
                    return result;
                };
                CodeGenerator.prototype.generateAssignment = function (left, right, operator, precedence, flags) {
                    if (Precedence.Assignment < precedence) {
                        flags |= F_ALLOW_IN;
                    }
                    return parenthesize([
                        this.generateExpression(left, Precedence.Call, flags),
                        space + operator + space,
                        this.generateExpression(right, Precedence.Assignment, flags)
                    ], Precedence.Assignment, precedence);
                };
                CodeGenerator.prototype.semicolon = function (flags) {
                    if (!semicolons && flags & F_SEMICOLON_OPT) {
                        return '';
                    }
                    return ';';
                };
                CodeGenerator.prototype.generateExpression = function (expr, precedence, flags) {
                    var result, type;
                    type = expr.type || Syntax.Property;
                    if (extra.verbatim && expr.hasOwnProperty(extra.verbatim)) {
                        return generateVerbatim(expr, precedence);
                    }
                    result = this[type](expr, precedence, flags);
                    if (extra.comment) {
                        result = addComments(expr, result);
                    }
                    return toSourceNodeWhenNeeded(result, expr);
                };
                CodeGenerator.prototype.generateStatement = function (stmt, flags) {
                    var result, fragment;
                    result = this[stmt.type](stmt, flags);
                    // Attach comments
                    if (extra.comment) {
                        result = addComments(stmt, result);
                    }
                    fragment = toSourceNodeWhenNeeded(result).toString();
                    if (stmt.type === Syntax.Program && !safeConcatenation && newline === '' && fragment.charAt(fragment.length - 1) === '\n') {
                        result = sourceMap ? toSourceNodeWhenNeeded(result).replaceRight(/\s+$/, '') : fragment.replace(/\s+$/, '');
                    }
                    return toSourceNodeWhenNeeded(result, stmt);
                };
                CodeGenerator.prototype.generateInternal = function (node) {
                    var codegen;
                    codegen = new CodeGenerator();
                    if (isStatement(node)) {
                        return codegen.generateStatement(node, S_TFFF);
                    }
                    if (isExpression(node)) {
                        return codegen.generateExpression(node, Precedence.Sequence, E_TTT);
                    }
                    throw new Error('Unknown node type: ' + node.type);
                };
                // Statements.
                CodeGenerator.Statement = {
                    BlockStatement: function (stmt, flags) {
                        var result = ['{', newline], that = this;
                        withIndent(function () {
                            var i, iz, fragment, bodyFlags;
                            bodyFlags = S_TFFF;
                            if (flags & F_FUNC_BODY) {
                                bodyFlags |= F_DIRECTIVE_CTX;
                            }
                            for (i = 0, iz = stmt.body.length; i < iz; ++i) {
                                if (i === iz - 1) {
                                    bodyFlags |= F_SEMICOLON_OPT;
                                }
                                fragment = addIndent(that.generateStatement(stmt.body[i], bodyFlags));
                                result.push(fragment);
                                if (!endsWithLineTerminator(toSourceNodeWhenNeeded(fragment).toString())) {
                                    result.push(newline);
                                }
                            }
                        });
                        result.push(addIndent('}'));
                        return result;
                    },
                    BreakStatement: function (stmt, flags) {
                        if (stmt.label) {
                            return 'break ' + stmt.label.name + this.semicolon(flags);
                        }
                        return 'break' + this.semicolon(flags);
                    },
                    ContinueStatement: function (stmt, flags) {
                        if (stmt.label) {
                            return 'continue ' + stmt.label.name + this.semicolon(flags);
                        }
                        return 'continue' + this.semicolon(flags);
                    },
                    ClassBody: function (stmt, flags) {
                        var result = ['{', newline], that = this;
                        withIndent(function (indent) {
                            var i, iz;
                            for (i = 0, iz = stmt.body.length; i < iz; ++i) {
                                result.push(indent);
                                result.push(that.generateExpression(stmt.body[i], Precedence.Sequence, E_TTT));
                                if (i + 1 < iz) {
                                    result.push(newline);
                                }
                            }
                        });
                        if (!endsWithLineTerminator(toSourceNodeWhenNeeded(result).toString())) {
                            result.push(newline);
                        }
                        result.push(base);
                        result.push('}');
                        return result;
                    },
                    ClassDeclaration: function (stmt, flags) {
                        var result, fragment;
                        result = ['class ' + stmt.id.name];
                        if (stmt.superClass) {
                            fragment = join('extends', this.generateExpression(stmt.superClass, Precedence.Assignment, E_TTT));
                            result = join(result, fragment);
                        }
                        result.push(space);
                        result.push(this.generateStatement(stmt.body, S_TFFT));
                        return result;
                    },
                    DirectiveStatement: function (stmt, flags) {
                        if (extra.raw && stmt.raw) {
                            return stmt.raw + this.semicolon(flags);
                        }
                        return escapeDirective(stmt.directive) + this.semicolon(flags);
                    },
                    DoWhileStatement: function (stmt, flags) {
                        // Because `do 42 while (cond)` is Syntax Error. We need semicolon.
                        var result = join('do', this.maybeBlock(stmt.body, S_TFFF));
                        result = this.maybeBlockSuffix(stmt.body, result);
                        return join(result, [
                            'while' + space + '(',
                            this.generateExpression(stmt.test, Precedence.Sequence, E_TTT),
                            ')' + this.semicolon(flags)
                        ]);
                    },
                    CatchClause: function (stmt, flags) {
                        var result, that = this;
                        withIndent(function () {
                            var guard;
                            result = [
                                'catch' + space + '(',
                                that.generateExpression(stmt.param, Precedence.Sequence, E_TTT),
                                ')'
                            ];
                            if (stmt.guard) {
                                guard = that.generateExpression(stmt.guard, Precedence.Sequence, E_TTT);
                                result.splice(2, 0, ' if ', guard);
                            }
                        });
                        result.push(this.maybeBlock(stmt.body, S_TFFF));
                        return result;
                    },
                    DebuggerStatement: function (stmt, flags) {
                        return 'debugger' + this.semicolon(flags);
                    },
                    EmptyStatement: function (stmt, flags) {
                        return ';';
                    },
                    ExportDeclaration: function (stmt, flags) {
                        var result = ['export'], bodyFlags, that = this;
                        bodyFlags = (flags & F_SEMICOLON_OPT) ? S_TFFT : S_TFFF;
                        // export default HoistableDeclaration[Default]
                        // export default AssignmentExpression[In] ;
                        if (stmt['default']) {
                            result = join(result, 'default');
                            if (isStatement(stmt.declaration)) {
                                result = join(result, this.generateStatement(stmt.declaration, bodyFlags));
                            }
                            else {
                                result = join(result, this.generateExpression(stmt.declaration, Precedence.Assignment, E_TTT) + this.semicolon(flags));
                            }
                            return result;
                        }
                        // export VariableStatement
                        // export Declaration[Default]
                        if (stmt.declaration) {
                            return join(result, this.generateStatement(stmt.declaration, bodyFlags));
                        }
                        // export * FromClause ;
                        // export ExportClause[NoReference] FromClause ;
                        // export ExportClause ;
                        if (stmt.specifiers) {
                            if (stmt.specifiers.length === 0) {
                                result = join(result, '{' + space + '}');
                            }
                            else if (stmt.specifiers[0].type === Syntax.ExportBatchSpecifier) {
                                result = join(result, this.generateExpression(stmt.specifiers[0], Precedence.Sequence, E_TTT));
                            }
                            else {
                                result = join(result, '{');
                                withIndent(function (indent) {
                                    var i, iz;
                                    result.push(newline);
                                    for (i = 0, iz = stmt.specifiers.length; i < iz; ++i) {
                                        result.push(indent);
                                        result.push(that.generateExpression(stmt.specifiers[i], Precedence.Sequence, E_TTT));
                                        if (i + 1 < iz) {
                                            result.push(',' + newline);
                                        }
                                    }
                                });
                                if (!endsWithLineTerminator(toSourceNodeWhenNeeded(result).toString())) {
                                    result.push(newline);
                                }
                                result.push(base + '}');
                            }
                            if (stmt.source) {
                                result = join(result, [
                                    'from' + space,
                                    this.generateExpression(stmt.source, Precedence.Sequence, E_TTT),
                                    this.semicolon(flags)
                                ]);
                            }
                            else {
                                result.push(this.semicolon(flags));
                            }
                        }
                        return result;
                    },
                    ExpressionStatement: function (stmt, flags) {
                        var result, fragment;
                        result = [this.generateExpression(stmt.expression, Precedence.Sequence, E_TTT)];
                        // 12.4 '{', 'function', 'class' is not allowed in this position.
                        // wrap expression with parentheses
                        fragment = toSourceNodeWhenNeeded(result).toString();
                        if (fragment.charAt(0) === '{' || (fragment.slice(0, 5) === 'class' && ' {'.indexOf(fragment.charAt(5)) >= 0) || (fragment.slice(0, 8) === 'function' && '* ('.indexOf(fragment.charAt(8)) >= 0) || (directive && (flags & F_DIRECTIVE_CTX) && stmt.expression.type === Syntax.Literal && typeof stmt.expression.value === 'string')) {
                            result = ['(', result, ')' + this.semicolon(flags)];
                        }
                        else {
                            result.push(this.semicolon(flags));
                        }
                        return result;
                    },
                    ImportDeclaration: function (stmt, flags) {
                        // ES6: 15.2.1 valid import declarations:
                        //     - import ImportClause FromClause ;
                        //     - import ModuleSpecifier ;
                        var result, cursor, that = this;
                        // If no ImportClause is present,
                        // this should be `import ModuleSpecifier` so skip `from`
                        // ModuleSpecifier is StringLiteral.
                        if (stmt.specifiers.length === 0) {
                            // import ModuleSpecifier ;
                            return [
                                'import',
                                space,
                                this.generateExpression(stmt.source, Precedence.Sequence, E_TTT),
                                this.semicolon(flags)
                            ];
                        }
                        // import ImportClause FromClause ;
                        result = [
                            'import'
                        ];
                        cursor = 0;
                        // ImportedBinding
                        if (stmt.specifiers[cursor].type === Syntax.ImportDefaultSpecifier) {
                            result = join(result, [
                                this.generateExpression(stmt.specifiers[cursor], Precedence.Sequence, E_TTT)
                            ]);
                            ++cursor;
                        }
                        if (stmt.specifiers[cursor]) {
                            if (cursor !== 0) {
                                result.push(',');
                            }
                            if (stmt.specifiers[cursor].type === Syntax.ImportNamespaceSpecifier) {
                                // NameSpaceImport
                                result = join(result, [
                                    space,
                                    this.generateExpression(stmt.specifiers[cursor], Precedence.Sequence, E_TTT)
                                ]);
                            }
                            else {
                                // NamedImports
                                result.push(space + '{');
                                if ((stmt.specifiers.length - cursor) === 1) {
                                    // import { ... } from "...";
                                    result.push(space);
                                    result.push(this.generateExpression(stmt.specifiers[cursor], Precedence.Sequence, E_TTT));
                                    result.push(space + '}' + space);
                                }
                                else {
                                    // import {
                                    //    ...,
                                    //    ...,
                                    // } from "...";
                                    withIndent(function (indent) {
                                        var i, iz;
                                        result.push(newline);
                                        for (i = cursor, iz = stmt.specifiers.length; i < iz; ++i) {
                                            result.push(indent);
                                            result.push(that.generateExpression(stmt.specifiers[i], Precedence.Sequence, E_TTT));
                                            if (i + 1 < iz) {
                                                result.push(',' + newline);
                                            }
                                        }
                                    });
                                    if (!endsWithLineTerminator(toSourceNodeWhenNeeded(result).toString())) {
                                        result.push(newline);
                                    }
                                    result.push(base + '}' + space);
                                }
                            }
                        }
                        result = join(result, [
                            'from' + space,
                            this.generateExpression(stmt.source, Precedence.Sequence, E_TTT),
                            this.semicolon(flags)
                        ]);
                        return result;
                    },
                    VariableDeclarator: function (stmt, flags) {
                        var itemFlags = (flags & F_ALLOW_IN) ? E_TTT : E_FTT;
                        if (stmt.init) {
                            return [
                                this.generateExpression(stmt.id, Precedence.Assignment, itemFlags),
                                space,
                                '=',
                                space,
                                this.generateExpression(stmt.init, Precedence.Assignment, itemFlags)
                            ];
                        }
                        return this.generatePattern(stmt.id, Precedence.Assignment, itemFlags);
                    },
                    VariableDeclaration: function (stmt, flags) {
                        // VariableDeclarator is typed as Statement,
                        // but joined with comma (not LineTerminator).
                        // So if comment is attached to target node, we should specialize.
                        var result, i, iz, node, bodyFlags, that = this;
                        result = [stmt.kind];
                        bodyFlags = (flags & F_ALLOW_IN) ? S_TFFF : S_FFFF;
                        function block() {
                            node = stmt.declarations[0];
                            if (extra.comment && node.leadingComments) {
                                result.push('\n');
                                result.push(addIndent(that.generateStatement(node, bodyFlags)));
                            }
                            else {
                                result.push(noEmptySpace());
                                result.push(that.generateStatement(node, bodyFlags));
                            }
                            for (i = 1, iz = stmt.declarations.length; i < iz; ++i) {
                                node = stmt.declarations[i];
                                if (extra.comment && node.leadingComments) {
                                    result.push(',' + newline);
                                    result.push(addIndent(that.generateStatement(node, bodyFlags)));
                                }
                                else {
                                    result.push(',' + space);
                                    result.push(that.generateStatement(node, bodyFlags));
                                }
                            }
                        }
                        if (stmt.declarations.length > 1) {
                            withIndent(block);
                        }
                        else {
                            block();
                        }
                        result.push(this.semicolon(flags));
                        return result;
                    },
                    ThrowStatement: function (stmt, flags) {
                        return [join('throw', this.generateExpression(stmt.argument, Precedence.Sequence, E_TTT)), this.semicolon(flags)];
                    },
                    TryStatement: function (stmt, flags) {
                        var result, i, iz, guardedHandlers;
                        result = ['try', this.maybeBlock(stmt.block, S_TFFF)];
                        result = this.maybeBlockSuffix(stmt.block, result);
                        if (stmt.handlers) {
                            for (i = 0, iz = stmt.handlers.length; i < iz; ++i) {
                                result = join(result, this.generateStatement(stmt.handlers[i], S_TFFF));
                                if (stmt.finalizer || i + 1 !== iz) {
                                    result = this.maybeBlockSuffix(stmt.handlers[i].body, result);
                                }
                            }
                        }
                        else {
                            guardedHandlers = stmt.guardedHandlers || [];
                            for (i = 0, iz = guardedHandlers.length; i < iz; ++i) {
                                result = join(result, this.generateStatement(guardedHandlers[i], S_TFFF));
                                if (stmt.finalizer || i + 1 !== iz) {
                                    result = this.maybeBlockSuffix(guardedHandlers[i].body, result);
                                }
                            }
                            // new interface
                            if (stmt.handler) {
                                if (isArray(stmt.handler)) {
                                    for (i = 0, iz = stmt.handler.length; i < iz; ++i) {
                                        result = join(result, this.generateStatement(stmt.handler[i], S_TFFF));
                                        if (stmt.finalizer || i + 1 !== iz) {
                                            result = this.maybeBlockSuffix(stmt.handler[i].body, result);
                                        }
                                    }
                                }
                                else {
                                    result = join(result, this.generateStatement(stmt.handler, S_TFFF));
                                    if (stmt.finalizer) {
                                        result = this.maybeBlockSuffix(stmt.handler.body, result);
                                    }
                                }
                            }
                        }
                        if (stmt.finalizer) {
                            result = join(result, ['finally', this.maybeBlock(stmt.finalizer, S_TFFF)]);
                        }
                        return result;
                    },
                    SwitchStatement: function (stmt, flags) {
                        var result, fragment, i, iz, bodyFlags, that = this;
                        withIndent(function () {
                            result = [
                                'switch' + space + '(',
                                that.generateExpression(stmt.discriminant, Precedence.Sequence, E_TTT),
                                ')' + space + '{' + newline
                            ];
                        });
                        if (stmt.cases) {
                            bodyFlags = S_TFFF;
                            for (i = 0, iz = stmt.cases.length; i < iz; ++i) {
                                if (i === iz - 1) {
                                    bodyFlags |= F_SEMICOLON_OPT;
                                }
                                fragment = addIndent(this.generateStatement(stmt.cases[i], bodyFlags));
                                result.push(fragment);
                                if (!endsWithLineTerminator(toSourceNodeWhenNeeded(fragment).toString())) {
                                    result.push(newline);
                                }
                            }
                        }
                        result.push(addIndent('}'));
                        return result;
                    },
                    SwitchCase: function (stmt, flags) {
                        var result, fragment, i, iz, bodyFlags, that = this;
                        withIndent(function () {
                            if (stmt.test) {
                                result = [
                                    join('case', that.generateExpression(stmt.test, Precedence.Sequence, E_TTT)),
                                    ':'
                                ];
                            }
                            else {
                                result = ['default:'];
                            }
                            i = 0;
                            iz = stmt.consequent.length;
                            if (iz && stmt.consequent[0].type === Syntax.BlockStatement) {
                                fragment = that.maybeBlock(stmt.consequent[0], S_TFFF);
                                result.push(fragment);
                                i = 1;
                            }
                            if (i !== iz && !endsWithLineTerminator(toSourceNodeWhenNeeded(result).toString())) {
                                result.push(newline);
                            }
                            bodyFlags = S_TFFF;
                            for (; i < iz; ++i) {
                                if (i === iz - 1 && flags & F_SEMICOLON_OPT) {
                                    bodyFlags |= F_SEMICOLON_OPT;
                                }
                                fragment = addIndent(that.generateStatement(stmt.consequent[i], bodyFlags));
                                result.push(fragment);
                                if (i + 1 !== iz && !endsWithLineTerminator(toSourceNodeWhenNeeded(fragment).toString())) {
                                    result.push(newline);
                                }
                            }
                        });
                        return result;
                    },
                    IfStatement: function (stmt, flags) {
                        var result, bodyFlags, semicolonOptional, that = this;
                        withIndent(function () {
                            result = [
                                'if' + space + '(',
                                that.generateExpression(stmt.test, Precedence.Sequence, E_TTT),
                                ')'
                            ];
                        });
                        semicolonOptional = flags & F_SEMICOLON_OPT;
                        bodyFlags = S_TFFF;
                        if (semicolonOptional) {
                            bodyFlags |= F_SEMICOLON_OPT;
                        }
                        if (stmt.alternate) {
                            result.push(this.maybeBlock(stmt.consequent, S_TFFF));
                            result = this.maybeBlockSuffix(stmt.consequent, result);
                            if (stmt.alternate.type === Syntax.IfStatement) {
                                result = join(result, ['else ', this.generateStatement(stmt.alternate, bodyFlags)]);
                            }
                            else {
                                result = join(result, join('else', this.maybeBlock(stmt.alternate, bodyFlags)));
                            }
                        }
                        else {
                            result.push(this.maybeBlock(stmt.consequent, bodyFlags));
                        }
                        return result;
                    },
                    ForStatement: function (stmt, flags) {
                        var result, that = this;
                        withIndent(function () {
                            result = ['for' + space + '('];
                            if (stmt.init) {
                                if (stmt.init.type === Syntax.VariableDeclaration) {
                                    result.push(that.generateStatement(stmt.init, S_FFFF));
                                }
                                else {
                                    // F_ALLOW_IN becomes false.
                                    result.push(that.generateExpression(stmt.init, Precedence.Sequence, E_FTT));
                                    result.push(';');
                                }
                            }
                            else {
                                result.push(';');
                            }
                            if (stmt.test) {
                                result.push(space);
                                result.push(that.generateExpression(stmt.test, Precedence.Sequence, E_TTT));
                                result.push(';');
                            }
                            else {
                                result.push(';');
                            }
                            if (stmt.update) {
                                result.push(space);
                                result.push(that.generateExpression(stmt.update, Precedence.Sequence, E_TTT));
                                result.push(')');
                            }
                            else {
                                result.push(')');
                            }
                        });
                        result.push(this.maybeBlock(stmt.body, flags & F_SEMICOLON_OPT ? S_TFFT : S_TFFF));
                        return result;
                    },
                    ForInStatement: function (stmt, flags) {
                        return this.generateIterationForStatement('in', stmt, flags & F_SEMICOLON_OPT ? S_TFFT : S_TFFF);
                    },
                    ForOfStatement: function (stmt, flags) {
                        return this.generateIterationForStatement('of', stmt, flags & F_SEMICOLON_OPT ? S_TFFT : S_TFFF);
                    },
                    LabeledStatement: function (stmt, flags) {
                        return [stmt.label.name + ':', this.maybeBlock(stmt.body, flags & F_SEMICOLON_OPT ? S_TFFT : S_TFFF)];
                    },
                    Program: function (stmt, flags) {
                        var result, fragment, i, iz, bodyFlags;
                        iz = stmt.body.length;
                        result = [safeConcatenation && iz > 0 ? '\n' : ''];
                        bodyFlags = S_TFTF;
                        for (i = 0; i < iz; ++i) {
                            if (!safeConcatenation && i === iz - 1) {
                                bodyFlags |= F_SEMICOLON_OPT;
                            }
                            fragment = addIndent(this.generateStatement(stmt.body[i], bodyFlags));
                            result.push(fragment);
                            if (i + 1 < iz && !endsWithLineTerminator(toSourceNodeWhenNeeded(fragment).toString())) {
                                result.push(newline);
                            }
                        }
                        return result;
                    },
                    FunctionDeclaration: function (stmt, flags) {
                        var isGenerator = stmt.generator && !extra.moz.starlessGenerator;
                        return [
                            (isGenerator ? 'function*' : 'function'),
                            (isGenerator ? space : noEmptySpace()),
                            this.generateIdentifier(stmt.id),
                            this.generateFunctionBody(stmt)
                        ];
                    },
                    ReturnStatement: function (stmt, flags) {
                        if (stmt.argument) {
                            return [join('return', this.generateExpression(stmt.argument, Precedence.Sequence, E_TTT)), this.semicolon(flags)];
                        }
                        return ['return' + this.semicolon(flags)];
                    },
                    WhileStatement: function (stmt, flags) {
                        var result, that = this;
                        withIndent(function () {
                            result = [
                                'while' + space + '(',
                                that.generateExpression(stmt.test, Precedence.Sequence, E_TTT),
                                ')'
                            ];
                        });
                        result.push(this.maybeBlock(stmt.body, flags & F_SEMICOLON_OPT ? S_TFFT : S_TFFF));
                        return result;
                    },
                    WithStatement: function (stmt, flags) {
                        var result, that = this;
                        withIndent(function () {
                            result = [
                                'with' + space + '(',
                                that.generateExpression(stmt.object, Precedence.Sequence, E_TTT),
                                ')'
                            ];
                        });
                        result.push(this.maybeBlock(stmt.body, flags & F_SEMICOLON_OPT ? S_TFFT : S_TFFF));
                        return result;
                    }
                };
                // Expressions.
                CodeGenerator.Expression = {
                    SequenceExpression: function (expr, precedence, flags) {
                        var result, i, iz;
                        if (Precedence.Sequence < precedence) {
                            flags |= F_ALLOW_IN;
                        }
                        result = [];
                        for (i = 0, iz = expr.expressions.length; i < iz; ++i) {
                            result.push(this.generateExpression(expr.expressions[i], Precedence.Assignment, flags));
                            if (i + 1 < iz) {
                                result.push(',' + space);
                            }
                        }
                        return parenthesize(result, Precedence.Sequence, precedence);
                    },
                    AssignmentExpression: function (expr, precedence, flags) {
                        return this.generateAssignment(expr.left, expr.right, expr.operator, precedence, flags);
                    },
                    ArrowFunctionExpression: function (expr, precedence, flags) {
                        return parenthesize(this.generateFunctionBody(expr), Precedence.ArrowFunction, precedence);
                    },
                    ConditionalExpression: function (expr, precedence, flags) {
                        if (Precedence.Conditional < precedence) {
                            flags |= F_ALLOW_IN;
                        }
                        return parenthesize([
                            this.generateExpression(expr.test, Precedence.LogicalOR, flags),
                            space + '?' + space,
                            this.generateExpression(expr.consequent, Precedence.Assignment, flags),
                            space + ':' + space,
                            this.generateExpression(expr.alternate, Precedence.Assignment, flags)
                        ], Precedence.Conditional, precedence);
                    },
                    LogicalExpression: function (expr, precedence, flags) {
                        return this.BinaryExpression(expr, precedence, flags);
                    },
                    BinaryExpression: function (expr, precedence, flags) {
                        var result, currentPrecedence, fragment, leftSource;
                        currentPrecedence = BinaryPrecedence[expr.operator];
                        if (currentPrecedence < precedence) {
                            flags |= F_ALLOW_IN;
                        }
                        fragment = this.generateExpression(expr.left, currentPrecedence, flags);
                        leftSource = fragment.toString();
                        if (leftSource.charCodeAt(leftSource.length - 1) === 0x2F && esutils.isIdentifierPart(expr.operator.charCodeAt(0))) {
                            result = [fragment, noEmptySpace(), expr.operator];
                        }
                        else {
                            result = join(fragment, expr.operator);
                        }
                        fragment = this.generateExpression(expr.right, currentPrecedence + 1, flags);
                        if (expr.operator === '/' && fragment.toString().charAt(0) === '/' || expr.operator.slice(-1) === '<' && fragment.toString().slice(0, 3) === '!--') {
                            // If '/' concats with '/' or `<` concats with `!--`, it is interpreted as comment start
                            result.push(noEmptySpace());
                            result.push(fragment);
                        }
                        else {
                            result = join(result, fragment);
                        }
                        if (expr.operator === 'in' && !(flags & F_ALLOW_IN)) {
                            return ['(', result, ')'];
                        }
                        return parenthesize(result, currentPrecedence, precedence);
                    },
                    CallExpression: function (expr, precedence, flags) {
                        var result, i, iz;
                        // F_ALLOW_UNPARATH_NEW becomes false.
                        result = [this.generateExpression(expr.callee, Precedence.Call, E_TTF)];
                        result.push('(');
                        for (i = 0, iz = expr['arguments'].length; i < iz; ++i) {
                            result.push(this.generateExpression(expr['arguments'][i], Precedence.Assignment, E_TTT));
                            if (i + 1 < iz) {
                                result.push(',' + space);
                            }
                        }
                        result.push(')');
                        if (!(flags & F_ALLOW_CALL)) {
                            return ['(', result, ')'];
                        }
                        return parenthesize(result, Precedence.Call, precedence);
                    },
                    NewExpression: function (expr, precedence, flags) {
                        var result, length, i, iz, itemFlags;
                        length = expr['arguments'].length;
                        // F_ALLOW_CALL becomes false.
                        // F_ALLOW_UNPARATH_NEW may become false.
                        itemFlags = (flags & F_ALLOW_UNPARATH_NEW && !parentheses && length === 0) ? E_TFT : E_TFF;
                        result = join('new', this.generateExpression(expr.callee, Precedence.New, itemFlags));
                        if (!(flags & F_ALLOW_UNPARATH_NEW) || parentheses || length > 0) {
                            result.push('(');
                            for (i = 0, iz = length; i < iz; ++i) {
                                result.push(this.generateExpression(expr['arguments'][i], Precedence.Assignment, E_TTT));
                                if (i + 1 < iz) {
                                    result.push(',' + space);
                                }
                            }
                            result.push(')');
                        }
                        return parenthesize(result, Precedence.New, precedence);
                    },
                    MemberExpression: function (expr, precedence, flags) {
                        var result, fragment;
                        // F_ALLOW_UNPARATH_NEW becomes false.
                        result = [this.generateExpression(expr.object, Precedence.Call, (flags & F_ALLOW_CALL) ? E_TTF : E_TFF)];
                        if (expr.computed) {
                            result.push('[');
                            result.push(this.generateExpression(expr.property, Precedence.Sequence, flags & F_ALLOW_CALL ? E_TTT : E_TFT));
                            result.push(']');
                        }
                        else {
                            if (expr.object.type === Syntax.Literal && typeof expr.object.value === 'number') {
                                fragment = toSourceNodeWhenNeeded(result).toString();
                                // When the following conditions are all true,
                                //   1. No floating point
                                //   2. Don't have exponents
                                //   3. The last character is a decimal digit
                                //   4. Not hexadecimal OR octal number literal
                                // we should add a floating point.
                                if (fragment.indexOf('.') < 0 && !/[eExX]/.test(fragment) && esutils.isDecimalDigit(fragment.charCodeAt(fragment.length - 1)) && !(fragment.length >= 2 && fragment.charCodeAt(0) === 48)) {
                                    result.push('.');
                                }
                            }
                            result.push('.');
                            result.push(this.generateIdentifier(expr.property));
                        }
                        return parenthesize(result, Precedence.Member, precedence);
                    },
                    UnaryExpression: function (expr, precedence, flags) {
                        var result, fragment, rightCharCode, leftSource, leftCharCode;
                        fragment = this.generateExpression(expr.argument, Precedence.Unary, E_TTT);
                        if (space === '') {
                            result = join(expr.operator, fragment);
                        }
                        else {
                            result = [expr.operator];
                            if (expr.operator.length > 2) {
                                // delete, void, typeof
                                // get `typeof []`, not `typeof[]`
                                result = join(result, fragment);
                            }
                            else {
                                // Prevent inserting spaces between operator and argument if it is unnecessary
                                // like, `!cond`
                                leftSource = toSourceNodeWhenNeeded(result).toString();
                                leftCharCode = leftSource.charCodeAt(leftSource.length - 1);
                                rightCharCode = fragment.toString().charCodeAt(0);
                                if (((leftCharCode === 0x2B || leftCharCode === 0x2D) && leftCharCode === rightCharCode) || (esutils.isIdentifierPart(leftCharCode) && esutils.isIdentifierPart(rightCharCode))) {
                                    result.push(noEmptySpace());
                                    result.push(fragment);
                                }
                                else {
                                    result.push(fragment);
                                }
                            }
                        }
                        return parenthesize(result, Precedence.Unary, precedence);
                    },
                    YieldExpression: function (expr, precedence, flags) {
                        var result;
                        if (expr.delegate) {
                            result = 'yield*';
                        }
                        else {
                            result = 'yield';
                        }
                        if (expr.argument) {
                            result = join(result, this.generateExpression(expr.argument, Precedence.Yield, E_TTT));
                        }
                        return parenthesize(result, Precedence.Yield, precedence);
                    },
                    UpdateExpression: function (expr, precedence, flags) {
                        if (expr.prefix) {
                            return parenthesize([
                                expr.operator,
                                this.generateExpression(expr.argument, Precedence.Unary, E_TTT)
                            ], Precedence.Unary, precedence);
                        }
                        return parenthesize([
                            this.generateExpression(expr.argument, Precedence.Postfix, E_TTT),
                            expr.operator
                        ], Precedence.Postfix, precedence);
                    },
                    FunctionExpression: function (expr, precedence, flags) {
                        var result, isGenerator;
                        isGenerator = expr.generator && !extra.moz.starlessGenerator;
                        result = isGenerator ? 'function*' : 'function';
                        if (expr.id) {
                            return [result, (isGenerator) ? space : noEmptySpace(), this.generateIdentifier(expr.id), this.generateFunctionBody(expr)];
                        }
                        return [result + space, this.generateFunctionBody(expr)];
                    },
                    ExportBatchSpecifier: function (expr, precedence, flags) {
                        return '*';
                    },
                    ArrayPattern: function (expr, precedence, flags) {
                        return this.ArrayExpression(expr, precedence, flags);
                    },
                    ArrayExpression: function (expr, precedence, flags) {
                        var result, multiline, that = this;
                        if (!expr.elements.length) {
                            return '[]';
                        }
                        multiline = expr.elements.length > 1;
                        result = ['[', multiline ? newline : ''];
                        withIndent(function (indent) {
                            var i, iz;
                            for (i = 0, iz = expr.elements.length; i < iz; ++i) {
                                if (!expr.elements[i]) {
                                    if (multiline) {
                                        result.push(indent);
                                    }
                                    if (i + 1 === iz) {
                                        result.push(',');
                                    }
                                }
                                else {
                                    result.push(multiline ? indent : '');
                                    result.push(that.generateExpression(expr.elements[i], Precedence.Assignment, E_TTT));
                                }
                                if (i + 1 < iz) {
                                    result.push(',' + (multiline ? newline : space));
                                }
                            }
                        });
                        if (multiline && !endsWithLineTerminator(toSourceNodeWhenNeeded(result).toString())) {
                            result.push(newline);
                        }
                        result.push(multiline ? base : '');
                        result.push(']');
                        return result;
                    },
                    ClassExpression: function (expr, precedence, flags) {
                        var result, fragment;
                        result = ['class'];
                        if (expr.id) {
                            result = join(result, this.generateExpression(expr.id, Precedence.Sequence, E_TTT));
                        }
                        if (expr.superClass) {
                            fragment = join('extends', this.generateExpression(expr.superClass, Precedence.Assignment, E_TTT));
                            result = join(result, fragment);
                        }
                        result.push(space);
                        result.push(this.generateStatement(expr.body, S_TFFT));
                        return result;
                    },
                    MethodDefinition: function (expr, precedence, flags) {
                        var result, fragment;
                        if (expr['static']) {
                            result = ['static' + space];
                        }
                        else {
                            result = [];
                        }
                        if (expr.kind === 'get' || expr.kind === 'set') {
                            result = join(result, [
                                join(expr.kind, this.generatePropertyKey(expr.key, expr.computed)),
                                this.generateFunctionBody(expr.value)
                            ]);
                        }
                        else {
                            fragment = [
                                this.generatePropertyKey(expr.key, expr.computed),
                                this.generateFunctionBody(expr.value)
                            ];
                            if (expr.value.generator) {
                                result.push('*');
                                result.push(fragment);
                            }
                            else {
                                result = join(result, fragment);
                            }
                        }
                        return result;
                    },
                    Property: function (expr, precedence, flags) {
                        var result;
                        if (expr.kind === 'get' || expr.kind === 'set') {
                            return [
                                expr.kind,
                                noEmptySpace(),
                                this.generatePropertyKey(expr.key, expr.computed),
                                this.generateFunctionBody(expr.value)
                            ];
                        }
                        if (expr.shorthand) {
                            return this.generatePropertyKey(expr.key, expr.computed);
                        }
                        if (expr.method) {
                            result = [];
                            if (expr.value.generator) {
                                result.push('*');
                            }
                            result.push(this.generatePropertyKey(expr.key, expr.computed));
                            result.push(this.generateFunctionBody(expr.value));
                            return result;
                        }
                        return [
                            this.generatePropertyKey(expr.key, expr.computed),
                            ':' + space,
                            this.generateExpression(expr.value, Precedence.Assignment, E_TTT)
                        ];
                    },
                    ObjectExpression: function (expr, precedence, flags) {
                        var multiline, result, fragment, that = this;
                        if (!expr.properties.length) {
                            return '{}';
                        }
                        multiline = expr.properties.length > 1;
                        withIndent(function () {
                            fragment = that.generateExpression(expr.properties[0], Precedence.Sequence, E_TTT);
                        });
                        if (!multiline) {
                            // issues 4
                            // Do not transform from
                            //   dejavu.Class.declare({
                            //       method2: function () {}
                            //   });
                            // to
                            //   dejavu.Class.declare({method2: function () {
                            //       }});
                            if (!hasLineTerminator(toSourceNodeWhenNeeded(fragment).toString())) {
                                return ['{', space, fragment, space, '}'];
                            }
                        }
                        withIndent(function (indent) {
                            var i, iz;
                            result = ['{', newline, indent, fragment];
                            if (multiline) {
                                result.push(',' + newline);
                                for (i = 1, iz = expr.properties.length; i < iz; ++i) {
                                    result.push(indent);
                                    result.push(that.generateExpression(expr.properties[i], Precedence.Sequence, E_TTT));
                                    if (i + 1 < iz) {
                                        result.push(',' + newline);
                                    }
                                }
                            }
                        });
                        if (!endsWithLineTerminator(toSourceNodeWhenNeeded(result).toString())) {
                            result.push(newline);
                        }
                        result.push(base);
                        result.push('}');
                        return result;
                    },
                    ObjectPattern: function (expr, precedence, flags) {
                        var result, i, iz, multiline, property, that = this;
                        if (!expr.properties.length) {
                            return '{}';
                        }
                        multiline = false;
                        if (expr.properties.length === 1) {
                            property = expr.properties[0];
                            if (property.value.type !== Syntax.Identifier) {
                                multiline = true;
                            }
                        }
                        else {
                            for (i = 0, iz = expr.properties.length; i < iz; ++i) {
                                property = expr.properties[i];
                                if (!property.shorthand) {
                                    multiline = true;
                                    break;
                                }
                            }
                        }
                        result = ['{', multiline ? newline : ''];
                        withIndent(function (indent) {
                            var i, iz;
                            for (i = 0, iz = expr.properties.length; i < iz; ++i) {
                                result.push(multiline ? indent : '');
                                result.push(that.generateExpression(expr.properties[i], Precedence.Sequence, E_TTT));
                                if (i + 1 < iz) {
                                    result.push(',' + (multiline ? newline : space));
                                }
                            }
                        });
                        if (multiline && !endsWithLineTerminator(toSourceNodeWhenNeeded(result).toString())) {
                            result.push(newline);
                        }
                        result.push(multiline ? base : '');
                        result.push('}');
                        return result;
                    },
                    ThisExpression: function (expr, precedence, flags) {
                        return 'this';
                    },
                    Identifier: function (expr, precedence, flags) {
                        return this.generateIdentifier(expr);
                    },
                    ImportDefaultSpecifier: function (expr, precedence, flags) {
                        return this.generateIdentifier(expr.id);
                    },
                    ImportNamespaceSpecifier: function (expr, precedence, flags) {
                        var result = ['*'];
                        if (expr.id) {
                            result.push(space + 'as' + noEmptySpace() + this.generateIdentifier(expr.id));
                        }
                        return result;
                    },
                    ImportSpecifier: function (expr, precedence, flags) {
                        return this.ExportSpecifier(expr, precedence, flags);
                    },
                    ExportSpecifier: function (expr, precedence, flags) {
                        var result = [expr.id.name];
                        if (expr.name) {
                            result.push(noEmptySpace() + 'as' + noEmptySpace() + this.generateIdentifier(expr.name));
                        }
                        return result;
                    },
                    Literal: function (expr, precedence, flags) {
                        var raw;
                        if (expr.hasOwnProperty('raw') && parse && extra.raw) {
                            try {
                                raw = parse(expr.raw).body[0].expression;
                                if (raw.type === Syntax.Literal) {
                                    if (raw.value === expr.value) {
                                        return expr.raw;
                                    }
                                }
                            }
                            catch (e) {
                            }
                        }
                        if (expr.value === null) {
                            return 'null';
                        }
                        if (typeof expr.value === 'string') {
                            return escapeString(expr.value);
                        }
                        if (typeof expr.value === 'number') {
                            return generateNumber(expr.value);
                        }
                        if (typeof expr.value === 'boolean') {
                            return expr.value ? 'true' : 'false';
                        }
                        return generateRegExp(expr.value);
                    },
                    GeneratorExpression: function (expr, precedence, flags) {
                        return this.ComprehensionExpression(expr, precedence, flags);
                    },
                    ComprehensionExpression: function (expr, precedence, flags) {
                        // GeneratorExpression should be parenthesized with (...), ComprehensionExpression with [...]
                        // Due to https://bugzilla.mozilla.org/show_bug.cgi?id=883468 position of expr.body can differ in Spidermonkey and ES6
                        var result, i, iz, fragment, that = this;
                        result = (expr.type === Syntax.GeneratorExpression) ? ['('] : ['['];
                        if (extra.moz.comprehensionExpressionStartsWithAssignment) {
                            fragment = this.generateExpression(expr.body, Precedence.Assignment, E_TTT);
                            result.push(fragment);
                        }
                        if (expr.blocks) {
                            withIndent(function () {
                                for (i = 0, iz = expr.blocks.length; i < iz; ++i) {
                                    fragment = that.generateExpression(expr.blocks[i], Precedence.Sequence, E_TTT);
                                    if (i > 0 || extra.moz.comprehensionExpressionStartsWithAssignment) {
                                        result = join(result, fragment);
                                    }
                                    else {
                                        result.push(fragment);
                                    }
                                }
                            });
                        }
                        if (expr.filter) {
                            result = join(result, 'if' + space);
                            fragment = this.generateExpression(expr.filter, Precedence.Sequence, E_TTT);
                            result = join(result, ['(', fragment, ')']);
                        }
                        if (!extra.moz.comprehensionExpressionStartsWithAssignment) {
                            fragment = this.generateExpression(expr.body, Precedence.Assignment, E_TTT);
                            result = join(result, fragment);
                        }
                        result.push((expr.type === Syntax.GeneratorExpression) ? ')' : ']');
                        return result;
                    },
                    ComprehensionBlock: function (expr, precedence, flags) {
                        var fragment;
                        if (expr.left.type === Syntax.VariableDeclaration) {
                            fragment = [
                                expr.left.kind,
                                noEmptySpace(),
                                this.generateStatement(expr.left.declarations[0], S_FFFF)
                            ];
                        }
                        else {
                            fragment = this.generateExpression(expr.left, Precedence.Call, E_TTT);
                        }
                        fragment = join(fragment, expr.of ? 'of' : 'in');
                        fragment = join(fragment, this.generateExpression(expr.right, Precedence.Sequence, E_TTT));
                        return ['for' + space + '(', fragment, ')'];
                    },
                    SpreadElement: function (expr, precedence, flags) {
                        return [
                            '...',
                            this.generateExpression(expr.argument, Precedence.Assignment, E_TTT)
                        ];
                    },
                    TaggedTemplateExpression: function (expr, precedence, flags) {
                        var itemFlags = E_TTF;
                        if (!(flags & F_ALLOW_CALL)) {
                            itemFlags = E_TFF;
                        }
                        var result = [
                            this.generateExpression(expr.tag, Precedence.Call, itemFlags),
                            this.generateExpression(expr.quasi, Precedence.Primary, E_FFT)
                        ];
                        return parenthesize(result, Precedence.TaggedTemplate, precedence);
                    },
                    TemplateElement: function (expr, precedence, flags) {
                        // Don't use "cooked". Since tagged template can use raw template
                        // representation. So if we do so, it breaks the script semantics.
                        return expr.value.raw;
                    },
                    TemplateLiteral: function (expr, precedence, flags) {
                        var result, i, iz;
                        result = ['`'];
                        for (i = 0, iz = expr.quasis.length; i < iz; ++i) {
                            result.push(this.generateExpression(expr.quasis[i], Precedence.Primary, E_TTT));
                            if (i + 1 < iz) {
                                result.push('${' + space);
                                result.push(this.generateExpression(expr.expressions[i], Precedence.Sequence, E_TTT));
                                result.push(space + '}');
                            }
                        }
                        result.push('`');
                        return result;
                    },
                    ModuleSpecifier: function (expr, precedence, flags) {
                        return this.Literal(expr, precedence, flags);
                    }
                };
                return CodeGenerator;
            })();
            merge(CodeGenerator.prototype, CodeGenerator.Statement);
            merge(CodeGenerator.prototype, CodeGenerator.Expression);
            function generate(node, options) {
                var defaultOptions = getDefaultOptions(), result, pair;
                if (options != null) {
                    // Obsolete options
                    //
                    //   `options.indent`
                    //   `options.base`
                    //
                    // Instead of them, we can use `option.format.indent`.
                    if (typeof options.indent === 'string') {
                        defaultOptions.format.indent.style = options.indent;
                    }
                    if (typeof options.base === 'number') {
                        defaultOptions.format.indent.base = options.base;
                    }
                    options = updateDeeply(defaultOptions, options);
                    indent = options.format.indent.style;
                    if (typeof options.base === 'string') {
                        base = options.base;
                    }
                    else {
                        base = stringRepeat(indent, options.format.indent.base);
                    }
                }
                else {
                    options = defaultOptions;
                    indent = options.format.indent.style;
                    base = stringRepeat(indent, options.format.indent.base);
                }
                json = options.format.json;
                renumber = options.format.renumber;
                hexadecimal = json ? false : options.format.hexadecimal;
                quotes = json ? 'double' : options.format.quotes;
                escapeless = options.format.escapeless;
                newline = options.format.newline;
                space = options.format.space;
                if (options.format.compact) {
                    newline = space = indent = base = '';
                }
                parentheses = options.format.parentheses;
                semicolons = options.format.semicolons;
                safeConcatenation = options.format.safeConcatenation;
                directive = options.directive;
                parse = json ? null : options.parse;
                sourceMap = options.sourceMap;
                extra = options;
                result = this.generateInternal(node);
                if (!sourceMap) {
                    pair = { code: result.toString(), map: null };
                    return options.sourceMapWithCode ? pair : pair.code;
                }
                pair = result.toStringWithSourceMap({
                    file: options.file,
                    sourceRoot: options.sourceMapRoot
                });
                if (options.sourceContent) {
                    pair.map.setSourceContent(options.sourceMap, options.sourceContent);
                }
                if (options.sourceMapWithCode) {
                    return pair;
                }
                return pair.map.toString();
            }
        })(gen = ast.gen || (ast.gen = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/*
 Copyright (C) 2012-2013 Yusuke Suzuki <utatane.tea@gmail.com>
 Copyright (C) 2012 Ariya Hidayat <ariya.hidayat@gmail.com>

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/*jslint vars:false, bitwise:true*/
/*jshint indent:4*/
/*global exports:true, define:true*/
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var traverse;
        (function (_traverse) {
            var Syntax, isArray, VisitorOption, VisitorKeys, objectCreate, objectKeys, BREAK, SKIP, REMOVE;
            function ignoreJSHintError() {
            }
            isArray = Array.isArray;
            if (!isArray) {
                isArray = function isArray(array) {
                    return Object.prototype.toString.call(array) === '[object Array]';
                };
            }
            function deepCopy(obj) {
                var ret = {}, key, val;
                for (key in obj) {
                    if (obj.hasOwnProperty(key)) {
                        val = obj[key];
                        if (typeof val === 'object' && val !== null) {
                            ret[key] = deepCopy(val);
                        }
                        else {
                            ret[key] = val;
                        }
                    }
                }
                return ret;
            }
            function shallowCopy(obj) {
                var ret = {}, key;
                for (key in obj) {
                    if (obj.hasOwnProperty(key)) {
                        ret[key] = obj[key];
                    }
                }
                return ret;
            }
            // based on LLVM libc++ upper_bound / lower_bound
            // MIT License
            function upperBound(array, func) {
                var diff, len, i, current;
                len = array.length;
                i = 0;
                while (len) {
                    diff = len >>> 1;
                    current = i + diff;
                    if (func(array[current])) {
                        len = diff;
                    }
                    else {
                        i = current + 1;
                        len -= diff + 1;
                    }
                }
                return i;
            }
            function lowerBound(array, func) {
                var diff, len, i, current;
                len = array.length;
                i = 0;
                while (len) {
                    diff = len >>> 1;
                    current = i + diff;
                    if (func(array[current])) {
                        i = current + 1;
                        len -= diff + 1;
                    }
                    else {
                        len = diff;
                    }
                }
                return i;
            }
            objectCreate = Object.create || (function () {
                function F() {
                }
                return function (o) {
                    F.prototype = o;
                    return new F();
                };
            })();
            objectKeys = Object.keys || function (o) {
                var keys = [], key;
                for (key in o) {
                    keys.push(key);
                }
                return keys;
            };
            function extend(to, from) {
                objectKeys(from).forEach(function (key) {
                    to[key] = from[key];
                });
                return to;
            }
            Syntax = {
                AssignmentExpression: 'AssignmentExpression',
                ArrayExpression: 'ArrayExpression',
                ArrayPattern: 'ArrayPattern',
                ArrowFunctionExpression: 'ArrowFunctionExpression',
                BlockStatement: 'BlockStatement',
                BinaryExpression: 'BinaryExpression',
                BreakStatement: 'BreakStatement',
                CallExpression: 'CallExpression',
                CatchClause: 'CatchClause',
                ClassBody: 'ClassBody',
                ClassDeclaration: 'ClassDeclaration',
                ClassExpression: 'ClassExpression',
                ComprehensionBlock: 'ComprehensionBlock',
                ComprehensionExpression: 'ComprehensionExpression',
                ConditionalExpression: 'ConditionalExpression',
                ContinueStatement: 'ContinueStatement',
                DebuggerStatement: 'DebuggerStatement',
                DirectiveStatement: 'DirectiveStatement',
                DoWhileStatement: 'DoWhileStatement',
                EmptyStatement: 'EmptyStatement',
                ExportBatchSpecifier: 'ExportBatchSpecifier',
                ExportDeclaration: 'ExportDeclaration',
                ExportSpecifier: 'ExportSpecifier',
                ExpressionStatement: 'ExpressionStatement',
                ForStatement: 'ForStatement',
                ForInStatement: 'ForInStatement',
                ForOfStatement: 'ForOfStatement',
                FunctionDeclaration: 'FunctionDeclaration',
                FunctionExpression: 'FunctionExpression',
                GeneratorExpression: 'GeneratorExpression',
                Identifier: 'Identifier',
                IfStatement: 'IfStatement',
                ImportDeclaration: 'ImportDeclaration',
                ImportDefaultSpecifier: 'ImportDefaultSpecifier',
                ImportNamespaceSpecifier: 'ImportNamespaceSpecifier',
                ImportSpecifier: 'ImportSpecifier',
                Literal: 'Literal',
                LabeledStatement: 'LabeledStatement',
                LogicalExpression: 'LogicalExpression',
                MemberExpression: 'MemberExpression',
                MethodDefinition: 'MethodDefinition',
                ModuleSpecifier: 'ModuleSpecifier',
                NewExpression: 'NewExpression',
                ObjectExpression: 'ObjectExpression',
                ObjectPattern: 'ObjectPattern',
                Program: 'Program',
                Property: 'Property',
                ReturnStatement: 'ReturnStatement',
                SequenceExpression: 'SequenceExpression',
                SpreadElement: 'SpreadElement',
                SwitchStatement: 'SwitchStatement',
                SwitchCase: 'SwitchCase',
                TaggedTemplateExpression: 'TaggedTemplateExpression',
                TemplateElement: 'TemplateElement',
                TemplateLiteral: 'TemplateLiteral',
                ThisExpression: 'ThisExpression',
                ThrowStatement: 'ThrowStatement',
                TryStatement: 'TryStatement',
                UnaryExpression: 'UnaryExpression',
                UpdateExpression: 'UpdateExpression',
                VariableDeclaration: 'VariableDeclaration',
                VariableDeclarator: 'VariableDeclarator',
                WhileStatement: 'WhileStatement',
                WithStatement: 'WithStatement',
                YieldExpression: 'YieldExpression'
            };
            VisitorKeys = {
                AssignmentExpression: ['left', 'right'],
                ArrayExpression: ['elements'],
                ArrayPattern: ['elements'],
                ArrowFunctionExpression: ['params', 'defaults', 'rest', 'body'],
                BlockStatement: ['body'],
                BinaryExpression: ['left', 'right'],
                BreakStatement: ['label'],
                CallExpression: ['callee', 'arguments'],
                CatchClause: ['param', 'body'],
                ClassBody: ['body'],
                ClassDeclaration: ['id', 'body', 'superClass'],
                ClassExpression: ['id', 'body', 'superClass'],
                ComprehensionBlock: ['left', 'right'],
                ComprehensionExpression: ['blocks', 'filter', 'body'],
                ConditionalExpression: ['test', 'consequent', 'alternate'],
                ContinueStatement: ['label'],
                DebuggerStatement: [],
                DirectiveStatement: [],
                DoWhileStatement: ['body', 'test'],
                EmptyStatement: [],
                ExportBatchSpecifier: [],
                ExportDeclaration: ['declaration', 'specifiers', 'source'],
                ExportSpecifier: ['id', 'name'],
                ExpressionStatement: ['expression'],
                ForStatement: ['init', 'test', 'update', 'body'],
                ForInStatement: ['left', 'right', 'body'],
                ForOfStatement: ['left', 'right', 'body'],
                FunctionDeclaration: ['id', 'params', 'defaults', 'rest', 'body'],
                FunctionExpression: ['id', 'params', 'defaults', 'rest', 'body'],
                GeneratorExpression: ['blocks', 'filter', 'body'],
                Identifier: [],
                IfStatement: ['test', 'consequent', 'alternate'],
                ImportDeclaration: ['specifiers', 'source'],
                ImportDefaultSpecifier: ['id'],
                ImportNamespaceSpecifier: ['id'],
                ImportSpecifier: ['id', 'name'],
                Literal: [],
                LabeledStatement: ['label', 'body'],
                LogicalExpression: ['left', 'right'],
                MemberExpression: ['object', 'property'],
                MethodDefinition: ['key', 'value'],
                ModuleSpecifier: [],
                NewExpression: ['callee', 'arguments'],
                ObjectExpression: ['properties'],
                ObjectPattern: ['properties'],
                Program: ['body'],
                Property: ['key', 'value'],
                ReturnStatement: ['argument'],
                SequenceExpression: ['expressions'],
                SpreadElement: ['argument'],
                SwitchStatement: ['discriminant', 'cases'],
                SwitchCase: ['test', 'consequent'],
                TaggedTemplateExpression: ['tag', 'quasi'],
                TemplateElement: [],
                TemplateLiteral: ['quasis', 'expressions'],
                ThisExpression: [],
                ThrowStatement: ['argument'],
                TryStatement: ['block', 'handlers', 'handler', 'guardedHandlers', 'finalizer'],
                UnaryExpression: ['argument'],
                UpdateExpression: ['argument'],
                VariableDeclaration: ['declarations'],
                VariableDeclarator: ['id', 'init'],
                WhileStatement: ['test', 'body'],
                WithStatement: ['object', 'body'],
                YieldExpression: ['argument']
            };
            // unique id
            BREAK = {};
            SKIP = {};
            REMOVE = {};
            VisitorOption = {
                Break: BREAK,
                Skip: SKIP,
                Remove: REMOVE
            };
            function Reference(parent, key) {
                this.parent = parent;
                this.key = key;
            }
            Reference.prototype.replace = function replace(node) {
                this.parent[this.key] = node;
            };
            Reference.prototype.remove = function remove() {
                if (isArray(this.parent)) {
                    this.parent.splice(this.key, 1);
                    return true;
                }
                else {
                    this.replace(null);
                    return false;
                }
            };
            function Element(node, path, wrap, ref) {
                this.node = node;
                this.path = path;
                this.wrap = wrap;
                this.ref = ref;
            }
            function Controller() {
            }
            // API:
            // return property path array from root to current node
            Controller.prototype.path = function path() {
                var i, iz, j, jz, result, element;
                function addToPath(result, path) {
                    if (isArray(path)) {
                        for (j = 0, jz = path.length; j < jz; ++j) {
                            result.push(path[j]);
                        }
                    }
                    else {
                        result.push(path);
                    }
                }
                // root node
                if (!this.__current.path) {
                    return null;
                }
                // first node is sentinel, second node is root element
                result = [];
                for (i = 2, iz = this.__leavelist.length; i < iz; ++i) {
                    element = this.__leavelist[i];
                    addToPath(result, element.path);
                }
                addToPath(result, this.__current.path);
                return result;
            };
            // API:
            // return type of current node
            Controller.prototype.type = function () {
                var node = this.current();
                return node.type || this.__current.wrap;
            };
            // API:
            // return array of parent elements
            Controller.prototype.parents = function parents() {
                var i, iz, result;
                // first node is sentinel
                result = [];
                for (i = 1, iz = this.__leavelist.length; i < iz; ++i) {
                    result.push(this.__leavelist[i].node);
                }
                return result;
            };
            // API:
            // return current node
            Controller.prototype.current = function current() {
                return this.__current.node;
            };
            Controller.prototype.__execute = function __execute(callback, element) {
                var previous, result;
                result = undefined;
                previous = this.__current;
                this.__current = element;
                this.__state = null;
                if (callback) {
                    result = callback.call(this, element.node, this.__leavelist[this.__leavelist.length - 1].node);
                }
                this.__current = previous;
                return result;
            };
            // API:
            // notify control skip / break
            Controller.prototype.notify = function notify(flag) {
                this.__state = flag;
            };
            // API:
            // skip child nodes of current node
            Controller.prototype.skip = function () {
                this.notify(SKIP);
            };
            // API:
            // break traversals
            Controller.prototype['break'] = function () {
                this.notify(BREAK);
            };
            // API:
            // remove node
            Controller.prototype.remove = function () {
                this.notify(REMOVE);
            };
            Controller.prototype.__initialize = function (root, visitor) {
                this.visitor = visitor;
                this.root = root;
                this.__worklist = [];
                this.__leavelist = [];
                this.__current = null;
                this.__state = null;
                this.__fallback = visitor.fallback === 'iteration';
                this.__keys = VisitorKeys;
                if (visitor.keys) {
                    this.__keys = extend(objectCreate(this.__keys), visitor.keys);
                }
            };
            function isNode(node) {
                if (node == null) {
                    return false;
                }
                return typeof node === 'object' && typeof node.type === 'string';
            }
            function isProperty(nodeType, key) {
                return (nodeType === Syntax.ObjectExpression || nodeType === Syntax.ObjectPattern) && 'properties' === key;
            }
            Controller.prototype.traverse = function traverse(root, visitor) {
                var worklist, leavelist, element, node, nodeType, ret, key, current, current2, candidates, candidate, sentinel;
                this.__initialize(root, visitor);
                sentinel = {};
                // reference
                worklist = this.__worklist;
                leavelist = this.__leavelist;
                // initialize
                worklist.push(new Element(root, null, null, null));
                leavelist.push(new Element(null, null, null, null));
                while (worklist.length) {
                    element = worklist.pop();
                    if (element === sentinel) {
                        element = leavelist.pop();
                        ret = this.__execute(visitor.leave, element);
                        if (this.__state === BREAK || ret === BREAK) {
                            return;
                        }
                        continue;
                    }
                    if (element.node) {
                        ret = this.__execute(visitor.enter, element);
                        if (this.__state === BREAK || ret === BREAK) {
                            return;
                        }
                        worklist.push(sentinel);
                        leavelist.push(element);
                        if (this.__state === SKIP || ret === SKIP) {
                            continue;
                        }
                        node = element.node;
                        nodeType = element.wrap || node.type;
                        candidates = this.__keys[nodeType];
                        if (!candidates) {
                            if (this.__fallback) {
                                candidates = objectKeys(node);
                            }
                            else {
                                throw new Error('Unknown node type ' + nodeType + '.');
                            }
                        }
                        current = candidates.length;
                        while ((current -= 1) >= 0) {
                            key = candidates[current];
                            candidate = node[key];
                            if (!candidate) {
                                continue;
                            }
                            if (isArray(candidate)) {
                                current2 = candidate.length;
                                while ((current2 -= 1) >= 0) {
                                    if (!candidate[current2]) {
                                        continue;
                                    }
                                    if (isProperty(nodeType, candidates[current])) {
                                        element = new Element(candidate[current2], [key, current2], 'Property', null);
                                    }
                                    else if (isNode(candidate[current2])) {
                                        element = new Element(candidate[current2], [key, current2], null, null);
                                    }
                                    else {
                                        continue;
                                    }
                                    worklist.push(element);
                                }
                            }
                            else if (isNode(candidate)) {
                                worklist.push(new Element(candidate, key, null, null));
                            }
                        }
                    }
                }
            };
            Controller.prototype.replace = function replace(root, visitor) {
                function removeElem(element) {
                    var i, key, nextElem, parent;
                    if (element.ref.remove()) {
                        // When the reference is an element of an array.
                        key = element.ref.key;
                        parent = element.ref.parent;
                        // If removed from array, then decrease following items' keys.
                        i = worklist.length;
                        while (i--) {
                            nextElem = worklist[i];
                            if (nextElem.ref && nextElem.ref.parent === parent) {
                                if (nextElem.ref.key < key) {
                                    break;
                                }
                                --nextElem.ref.key;
                            }
                        }
                    }
                }
                var worklist, leavelist, node, nodeType, target, element, current, current2, candidates, candidate, sentinel, outer, key;
                this.__initialize(root, visitor);
                sentinel = {};
                // reference
                worklist = this.__worklist;
                leavelist = this.__leavelist;
                // initialize
                outer = {
                    root: root
                };
                element = new Element(root, null, null, new Reference(outer, 'root'));
                worklist.push(element);
                leavelist.push(element);
                while (worklist.length) {
                    element = worklist.pop();
                    if (element === sentinel) {
                        element = leavelist.pop();
                        target = this.__execute(visitor.leave, element);
                        // node may be replaced with null,
                        // so distinguish between undefined and null in this place
                        if (target !== undefined && target !== BREAK && target !== SKIP && target !== REMOVE) {
                            // replace
                            element.ref.replace(target);
                        }
                        if (this.__state === REMOVE || target === REMOVE) {
                            removeElem(element);
                        }
                        if (this.__state === BREAK || target === BREAK) {
                            return outer.root;
                        }
                        continue;
                    }
                    target = this.__execute(visitor.enter, element);
                    // node may be replaced with null,
                    // so distinguish between undefined and null in this place
                    if (target !== undefined && target !== BREAK && target !== SKIP && target !== REMOVE) {
                        // replace
                        element.ref.replace(target);
                        element.node = target;
                    }
                    if (this.__state === REMOVE || target === REMOVE) {
                        removeElem(element);
                        element.node = null;
                    }
                    if (this.__state === BREAK || target === BREAK) {
                        return outer.root;
                    }
                    // node may be null
                    node = element.node;
                    if (!node) {
                        continue;
                    }
                    worklist.push(sentinel);
                    leavelist.push(element);
                    if (this.__state === SKIP || target === SKIP) {
                        continue;
                    }
                    nodeType = element.wrap || node.type;
                    candidates = this.__keys[nodeType];
                    if (!candidates) {
                        if (this.__fallback) {
                            candidates = objectKeys(node);
                        }
                        else {
                            throw new Error('Unknown node type ' + nodeType + '.');
                        }
                    }
                    current = candidates.length;
                    while ((current -= 1) >= 0) {
                        key = candidates[current];
                        candidate = node[key];
                        if (!candidate) {
                            continue;
                        }
                        if (isArray(candidate)) {
                            current2 = candidate.length;
                            while ((current2 -= 1) >= 0) {
                                if (!candidate[current2]) {
                                    continue;
                                }
                                if (isProperty(nodeType, candidates[current])) {
                                    element = new Element(candidate[current2], [key, current2], 'Property', new Reference(candidate, current2));
                                }
                                else if (isNode(candidate[current2])) {
                                    element = new Element(candidate[current2], [key, current2], null, new Reference(candidate, current2));
                                }
                                else {
                                    continue;
                                }
                                worklist.push(element);
                            }
                        }
                        else if (isNode(candidate)) {
                            worklist.push(new Element(candidate, key, null, new Reference(node, key)));
                        }
                    }
                }
                return outer.root;
            };
            function traverse(root, visitor) {
                var controller = new Controller();
                return controller.traverse(root, visitor);
            }
            function replace(root, visitor) {
                var controller = new Controller();
                return controller.replace(root, visitor);
            }
            function extendCommentRange(comment, tokens) {
                var target;
                target = upperBound(tokens, function search(token) {
                    return token.range[0] > comment.range[0];
                });
                comment.extendedRange = [comment.range[0], comment.range[1]];
                if (target !== tokens.length) {
                    comment.extendedRange[1] = tokens[target].range[0];
                }
                target -= 1;
                if (target >= 0) {
                    comment.extendedRange[0] = tokens[target].range[1];
                }
                return comment;
            }
            function attachComments(tree, providedComments, tokens) {
                // At first, we should calculate extended comment ranges.
                var comments = [], comment, len, i, cursor;
                if (!tree.range) {
                    throw new Error('attachComments needs range information');
                }
                // tokens array is empty, we attach comments to tree as 'leadingComments'
                if (!tokens.length) {
                    if (providedComments.length) {
                        for (i = 0, len = providedComments.length; i < len; i += 1) {
                            comment = deepCopy(providedComments[i]);
                            comment.extendedRange = [0, tree.range[0]];
                            comments.push(comment);
                        }
                        tree.leadingComments = comments;
                    }
                    return tree;
                }
                for (i = 0, len = providedComments.length; i < len; i += 1) {
                    comments.push(extendCommentRange(deepCopy(providedComments[i]), tokens));
                }
                // This is based on John Freeman's implementation.
                cursor = 0;
                traverse(tree, {
                    enter: function (node) {
                        var comment;
                        while (cursor < comments.length) {
                            comment = comments[cursor];
                            if (comment.extendedRange[1] > node.range[0]) {
                                break;
                            }
                            if (comment.extendedRange[1] === node.range[0]) {
                                if (!node.leadingComments) {
                                    node.leadingComments = [];
                                }
                                node.leadingComments.push(comment);
                                comments.splice(cursor, 1);
                            }
                            else {
                                cursor += 1;
                            }
                        }
                        // already out of owned node
                        if (cursor === comments.length) {
                            return VisitorOption.Break;
                        }
                        if (comments[cursor].extendedRange[0] > node.range[1]) {
                            return VisitorOption.Skip;
                        }
                    }
                });
                cursor = 0;
                traverse(tree, {
                    leave: function (node) {
                        var comment;
                        while (cursor < comments.length) {
                            comment = comments[cursor];
                            if (node.range[1] < comment.extendedRange[0]) {
                                break;
                            }
                            if (node.range[1] === comment.extendedRange[0]) {
                                if (!node.trailingComments) {
                                    node.trailingComments = [];
                                }
                                node.trailingComments.push(comment);
                                comments.splice(cursor, 1);
                            }
                            else {
                                cursor += 1;
                            }
                        }
                        // already out of owned node
                        if (cursor === comments.length) {
                            return VisitorOption.Break;
                        }
                        if (comments[cursor].extendedRange[0] > node.range[1]) {
                            return VisitorOption.Skip;
                        }
                    }
                });
                return tree;
            }
        })(traverse = ast.traverse || (ast.traverse = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/*
 Copyright (C) 2013 Yusuke Suzuki <utatane.tea@gmail.com>
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// from https://github.com/estools/esutils
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var utils;
        (function (utils) {
            var Regex, NON_ASCII_WHITESPACES;
            // See `tools/generate-identifier-regex.js`.
            Regex = {
                NonAsciiIdentifierStart: new RegExp('[\xAA\xB5\xBA\xC0-\xD6\xD8-\xF6\xF8-\u02C1\u02C6-\u02D1\u02E0-\u02E4\u02EC\u02EE\u0370-\u0374\u0376\u0377\u037A-\u037D\u0386\u0388-\u038A\u038C\u038E-\u03A1\u03A3-\u03F5\u03F7-\u0481\u048A-\u0527\u0531-\u0556\u0559\u0561-\u0587\u05D0-\u05EA\u05F0-\u05F2\u0620-\u064A\u066E\u066F\u0671-\u06D3\u06D5\u06E5\u06E6\u06EE\u06EF\u06FA-\u06FC\u06FF\u0710\u0712-\u072F\u074D-\u07A5\u07B1\u07CA-\u07EA\u07F4\u07F5\u07FA\u0800-\u0815\u081A\u0824\u0828\u0840-\u0858\u08A0\u08A2-\u08AC\u0904-\u0939\u093D\u0950\u0958-\u0961\u0971-\u0977\u0979-\u097F\u0985-\u098C\u098F\u0990\u0993-\u09A8\u09AA-\u09B0\u09B2\u09B6-\u09B9\u09BD\u09CE\u09DC\u09DD\u09DF-\u09E1\u09F0\u09F1\u0A05-\u0A0A\u0A0F\u0A10\u0A13-\u0A28\u0A2A-\u0A30\u0A32\u0A33\u0A35\u0A36\u0A38\u0A39\u0A59-\u0A5C\u0A5E\u0A72-\u0A74\u0A85-\u0A8D\u0A8F-\u0A91\u0A93-\u0AA8\u0AAA-\u0AB0\u0AB2\u0AB3\u0AB5-\u0AB9\u0ABD\u0AD0\u0AE0\u0AE1\u0B05-\u0B0C\u0B0F\u0B10\u0B13-\u0B28\u0B2A-\u0B30\u0B32\u0B33\u0B35-\u0B39\u0B3D\u0B5C\u0B5D\u0B5F-\u0B61\u0B71\u0B83\u0B85-\u0B8A\u0B8E-\u0B90\u0B92-\u0B95\u0B99\u0B9A\u0B9C\u0B9E\u0B9F\u0BA3\u0BA4\u0BA8-\u0BAA\u0BAE-\u0BB9\u0BD0\u0C05-\u0C0C\u0C0E-\u0C10\u0C12-\u0C28\u0C2A-\u0C33\u0C35-\u0C39\u0C3D\u0C58\u0C59\u0C60\u0C61\u0C85-\u0C8C\u0C8E-\u0C90\u0C92-\u0CA8\u0CAA-\u0CB3\u0CB5-\u0CB9\u0CBD\u0CDE\u0CE0\u0CE1\u0CF1\u0CF2\u0D05-\u0D0C\u0D0E-\u0D10\u0D12-\u0D3A\u0D3D\u0D4E\u0D60\u0D61\u0D7A-\u0D7F\u0D85-\u0D96\u0D9A-\u0DB1\u0DB3-\u0DBB\u0DBD\u0DC0-\u0DC6\u0E01-\u0E30\u0E32\u0E33\u0E40-\u0E46\u0E81\u0E82\u0E84\u0E87\u0E88\u0E8A\u0E8D\u0E94-\u0E97\u0E99-\u0E9F\u0EA1-\u0EA3\u0EA5\u0EA7\u0EAA\u0EAB\u0EAD-\u0EB0\u0EB2\u0EB3\u0EBD\u0EC0-\u0EC4\u0EC6\u0EDC-\u0EDF\u0F00\u0F40-\u0F47\u0F49-\u0F6C\u0F88-\u0F8C\u1000-\u102A\u103F\u1050-\u1055\u105A-\u105D\u1061\u1065\u1066\u106E-\u1070\u1075-\u1081\u108E\u10A0-\u10C5\u10C7\u10CD\u10D0-\u10FA\u10FC-\u1248\u124A-\u124D\u1250-\u1256\u1258\u125A-\u125D\u1260-\u1288\u128A-\u128D\u1290-\u12B0\u12B2-\u12B5\u12B8-\u12BE\u12C0\u12C2-\u12C5\u12C8-\u12D6\u12D8-\u1310\u1312-\u1315\u1318-\u135A\u1380-\u138F\u13A0-\u13F4\u1401-\u166C\u166F-\u167F\u1681-\u169A\u16A0-\u16EA\u16EE-\u16F0\u1700-\u170C\u170E-\u1711\u1720-\u1731\u1740-\u1751\u1760-\u176C\u176E-\u1770\u1780-\u17B3\u17D7\u17DC\u1820-\u1877\u1880-\u18A8\u18AA\u18B0-\u18F5\u1900-\u191C\u1950-\u196D\u1970-\u1974\u1980-\u19AB\u19C1-\u19C7\u1A00-\u1A16\u1A20-\u1A54\u1AA7\u1B05-\u1B33\u1B45-\u1B4B\u1B83-\u1BA0\u1BAE\u1BAF\u1BBA-\u1BE5\u1C00-\u1C23\u1C4D-\u1C4F\u1C5A-\u1C7D\u1CE9-\u1CEC\u1CEE-\u1CF1\u1CF5\u1CF6\u1D00-\u1DBF\u1E00-\u1F15\u1F18-\u1F1D\u1F20-\u1F45\u1F48-\u1F4D\u1F50-\u1F57\u1F59\u1F5B\u1F5D\u1F5F-\u1F7D\u1F80-\u1FB4\u1FB6-\u1FBC\u1FBE\u1FC2-\u1FC4\u1FC6-\u1FCC\u1FD0-\u1FD3\u1FD6-\u1FDB\u1FE0-\u1FEC\u1FF2-\u1FF4\u1FF6-\u1FFC\u2071\u207F\u2090-\u209C\u2102\u2107\u210A-\u2113\u2115\u2119-\u211D\u2124\u2126\u2128\u212A-\u212D\u212F-\u2139\u213C-\u213F\u2145-\u2149\u214E\u2160-\u2188\u2C00-\u2C2E\u2C30-\u2C5E\u2C60-\u2CE4\u2CEB-\u2CEE\u2CF2\u2CF3\u2D00-\u2D25\u2D27\u2D2D\u2D30-\u2D67\u2D6F\u2D80-\u2D96\u2DA0-\u2DA6\u2DA8-\u2DAE\u2DB0-\u2DB6\u2DB8-\u2DBE\u2DC0-\u2DC6\u2DC8-\u2DCE\u2DD0-\u2DD6\u2DD8-\u2DDE\u2E2F\u3005-\u3007\u3021-\u3029\u3031-\u3035\u3038-\u303C\u3041-\u3096\u309D-\u309F\u30A1-\u30FA\u30FC-\u30FF\u3105-\u312D\u3131-\u318E\u31A0-\u31BA\u31F0-\u31FF\u3400-\u4DB5\u4E00-\u9FCC\uA000-\uA48C\uA4D0-\uA4FD\uA500-\uA60C\uA610-\uA61F\uA62A\uA62B\uA640-\uA66E\uA67F-\uA697\uA6A0-\uA6EF\uA717-\uA71F\uA722-\uA788\uA78B-\uA78E\uA790-\uA793\uA7A0-\uA7AA\uA7F8-\uA801\uA803-\uA805\uA807-\uA80A\uA80C-\uA822\uA840-\uA873\uA882-\uA8B3\uA8F2-\uA8F7\uA8FB\uA90A-\uA925\uA930-\uA946\uA960-\uA97C\uA984-\uA9B2\uA9CF\uAA00-\uAA28\uAA40-\uAA42\uAA44-\uAA4B\uAA60-\uAA76\uAA7A\uAA80-\uAAAF\uAAB1\uAAB5\uAAB6\uAAB9-\uAABD\uAAC0\uAAC2\uAADB-\uAADD\uAAE0-\uAAEA\uAAF2-\uAAF4\uAB01-\uAB06\uAB09-\uAB0E\uAB11-\uAB16\uAB20-\uAB26\uAB28-\uAB2E\uABC0-\uABE2\uAC00-\uD7A3\uD7B0-\uD7C6\uD7CB-\uD7FB\uF900-\uFA6D\uFA70-\uFAD9\uFB00-\uFB06\uFB13-\uFB17\uFB1D\uFB1F-\uFB28\uFB2A-\uFB36\uFB38-\uFB3C\uFB3E\uFB40\uFB41\uFB43\uFB44\uFB46-\uFBB1\uFBD3-\uFD3D\uFD50-\uFD8F\uFD92-\uFDC7\uFDF0-\uFDFB\uFE70-\uFE74\uFE76-\uFEFC\uFF21-\uFF3A\uFF41-\uFF5A\uFF66-\uFFBE\uFFC2-\uFFC7\uFFCA-\uFFCF\uFFD2-\uFFD7\uFFDA-\uFFDC]'),
                NonAsciiIdentifierPart: new RegExp('[\xAA\xB5\xBA\xC0-\xD6\xD8-\xF6\xF8-\u02C1\u02C6-\u02D1\u02E0-\u02E4\u02EC\u02EE\u0300-\u0374\u0376\u0377\u037A-\u037D\u0386\u0388-\u038A\u038C\u038E-\u03A1\u03A3-\u03F5\u03F7-\u0481\u0483-\u0487\u048A-\u0527\u0531-\u0556\u0559\u0561-\u0587\u0591-\u05BD\u05BF\u05C1\u05C2\u05C4\u05C5\u05C7\u05D0-\u05EA\u05F0-\u05F2\u0610-\u061A\u0620-\u0669\u066E-\u06D3\u06D5-\u06DC\u06DF-\u06E8\u06EA-\u06FC\u06FF\u0710-\u074A\u074D-\u07B1\u07C0-\u07F5\u07FA\u0800-\u082D\u0840-\u085B\u08A0\u08A2-\u08AC\u08E4-\u08FE\u0900-\u0963\u0966-\u096F\u0971-\u0977\u0979-\u097F\u0981-\u0983\u0985-\u098C\u098F\u0990\u0993-\u09A8\u09AA-\u09B0\u09B2\u09B6-\u09B9\u09BC-\u09C4\u09C7\u09C8\u09CB-\u09CE\u09D7\u09DC\u09DD\u09DF-\u09E3\u09E6-\u09F1\u0A01-\u0A03\u0A05-\u0A0A\u0A0F\u0A10\u0A13-\u0A28\u0A2A-\u0A30\u0A32\u0A33\u0A35\u0A36\u0A38\u0A39\u0A3C\u0A3E-\u0A42\u0A47\u0A48\u0A4B-\u0A4D\u0A51\u0A59-\u0A5C\u0A5E\u0A66-\u0A75\u0A81-\u0A83\u0A85-\u0A8D\u0A8F-\u0A91\u0A93-\u0AA8\u0AAA-\u0AB0\u0AB2\u0AB3\u0AB5-\u0AB9\u0ABC-\u0AC5\u0AC7-\u0AC9\u0ACB-\u0ACD\u0AD0\u0AE0-\u0AE3\u0AE6-\u0AEF\u0B01-\u0B03\u0B05-\u0B0C\u0B0F\u0B10\u0B13-\u0B28\u0B2A-\u0B30\u0B32\u0B33\u0B35-\u0B39\u0B3C-\u0B44\u0B47\u0B48\u0B4B-\u0B4D\u0B56\u0B57\u0B5C\u0B5D\u0B5F-\u0B63\u0B66-\u0B6F\u0B71\u0B82\u0B83\u0B85-\u0B8A\u0B8E-\u0B90\u0B92-\u0B95\u0B99\u0B9A\u0B9C\u0B9E\u0B9F\u0BA3\u0BA4\u0BA8-\u0BAA\u0BAE-\u0BB9\u0BBE-\u0BC2\u0BC6-\u0BC8\u0BCA-\u0BCD\u0BD0\u0BD7\u0BE6-\u0BEF\u0C01-\u0C03\u0C05-\u0C0C\u0C0E-\u0C10\u0C12-\u0C28\u0C2A-\u0C33\u0C35-\u0C39\u0C3D-\u0C44\u0C46-\u0C48\u0C4A-\u0C4D\u0C55\u0C56\u0C58\u0C59\u0C60-\u0C63\u0C66-\u0C6F\u0C82\u0C83\u0C85-\u0C8C\u0C8E-\u0C90\u0C92-\u0CA8\u0CAA-\u0CB3\u0CB5-\u0CB9\u0CBC-\u0CC4\u0CC6-\u0CC8\u0CCA-\u0CCD\u0CD5\u0CD6\u0CDE\u0CE0-\u0CE3\u0CE6-\u0CEF\u0CF1\u0CF2\u0D02\u0D03\u0D05-\u0D0C\u0D0E-\u0D10\u0D12-\u0D3A\u0D3D-\u0D44\u0D46-\u0D48\u0D4A-\u0D4E\u0D57\u0D60-\u0D63\u0D66-\u0D6F\u0D7A-\u0D7F\u0D82\u0D83\u0D85-\u0D96\u0D9A-\u0DB1\u0DB3-\u0DBB\u0DBD\u0DC0-\u0DC6\u0DCA\u0DCF-\u0DD4\u0DD6\u0DD8-\u0DDF\u0DF2\u0DF3\u0E01-\u0E3A\u0E40-\u0E4E\u0E50-\u0E59\u0E81\u0E82\u0E84\u0E87\u0E88\u0E8A\u0E8D\u0E94-\u0E97\u0E99-\u0E9F\u0EA1-\u0EA3\u0EA5\u0EA7\u0EAA\u0EAB\u0EAD-\u0EB9\u0EBB-\u0EBD\u0EC0-\u0EC4\u0EC6\u0EC8-\u0ECD\u0ED0-\u0ED9\u0EDC-\u0EDF\u0F00\u0F18\u0F19\u0F20-\u0F29\u0F35\u0F37\u0F39\u0F3E-\u0F47\u0F49-\u0F6C\u0F71-\u0F84\u0F86-\u0F97\u0F99-\u0FBC\u0FC6\u1000-\u1049\u1050-\u109D\u10A0-\u10C5\u10C7\u10CD\u10D0-\u10FA\u10FC-\u1248\u124A-\u124D\u1250-\u1256\u1258\u125A-\u125D\u1260-\u1288\u128A-\u128D\u1290-\u12B0\u12B2-\u12B5\u12B8-\u12BE\u12C0\u12C2-\u12C5\u12C8-\u12D6\u12D8-\u1310\u1312-\u1315\u1318-\u135A\u135D-\u135F\u1380-\u138F\u13A0-\u13F4\u1401-\u166C\u166F-\u167F\u1681-\u169A\u16A0-\u16EA\u16EE-\u16F0\u1700-\u170C\u170E-\u1714\u1720-\u1734\u1740-\u1753\u1760-\u176C\u176E-\u1770\u1772\u1773\u1780-\u17D3\u17D7\u17DC\u17DD\u17E0-\u17E9\u180B-\u180D\u1810-\u1819\u1820-\u1877\u1880-\u18AA\u18B0-\u18F5\u1900-\u191C\u1920-\u192B\u1930-\u193B\u1946-\u196D\u1970-\u1974\u1980-\u19AB\u19B0-\u19C9\u19D0-\u19D9\u1A00-\u1A1B\u1A20-\u1A5E\u1A60-\u1A7C\u1A7F-\u1A89\u1A90-\u1A99\u1AA7\u1B00-\u1B4B\u1B50-\u1B59\u1B6B-\u1B73\u1B80-\u1BF3\u1C00-\u1C37\u1C40-\u1C49\u1C4D-\u1C7D\u1CD0-\u1CD2\u1CD4-\u1CF6\u1D00-\u1DE6\u1DFC-\u1F15\u1F18-\u1F1D\u1F20-\u1F45\u1F48-\u1F4D\u1F50-\u1F57\u1F59\u1F5B\u1F5D\u1F5F-\u1F7D\u1F80-\u1FB4\u1FB6-\u1FBC\u1FBE\u1FC2-\u1FC4\u1FC6-\u1FCC\u1FD0-\u1FD3\u1FD6-\u1FDB\u1FE0-\u1FEC\u1FF2-\u1FF4\u1FF6-\u1FFC\u200C\u200D\u203F\u2040\u2054\u2071\u207F\u2090-\u209C\u20D0-\u20DC\u20E1\u20E5-\u20F0\u2102\u2107\u210A-\u2113\u2115\u2119-\u211D\u2124\u2126\u2128\u212A-\u212D\u212F-\u2139\u213C-\u213F\u2145-\u2149\u214E\u2160-\u2188\u2C00-\u2C2E\u2C30-\u2C5E\u2C60-\u2CE4\u2CEB-\u2CF3\u2D00-\u2D25\u2D27\u2D2D\u2D30-\u2D67\u2D6F\u2D7F-\u2D96\u2DA0-\u2DA6\u2DA8-\u2DAE\u2DB0-\u2DB6\u2DB8-\u2DBE\u2DC0-\u2DC6\u2DC8-\u2DCE\u2DD0-\u2DD6\u2DD8-\u2DDE\u2DE0-\u2DFF\u2E2F\u3005-\u3007\u3021-\u302F\u3031-\u3035\u3038-\u303C\u3041-\u3096\u3099\u309A\u309D-\u309F\u30A1-\u30FA\u30FC-\u30FF\u3105-\u312D\u3131-\u318E\u31A0-\u31BA\u31F0-\u31FF\u3400-\u4DB5\u4E00-\u9FCC\uA000-\uA48C\uA4D0-\uA4FD\uA500-\uA60C\uA610-\uA62B\uA640-\uA66F\uA674-\uA67D\uA67F-\uA697\uA69F-\uA6F1\uA717-\uA71F\uA722-\uA788\uA78B-\uA78E\uA790-\uA793\uA7A0-\uA7AA\uA7F8-\uA827\uA840-\uA873\uA880-\uA8C4\uA8D0-\uA8D9\uA8E0-\uA8F7\uA8FB\uA900-\uA92D\uA930-\uA953\uA960-\uA97C\uA980-\uA9C0\uA9CF-\uA9D9\uAA00-\uAA36\uAA40-\uAA4D\uAA50-\uAA59\uAA60-\uAA76\uAA7A\uAA7B\uAA80-\uAAC2\uAADB-\uAADD\uAAE0-\uAAEF\uAAF2-\uAAF6\uAB01-\uAB06\uAB09-\uAB0E\uAB11-\uAB16\uAB20-\uAB26\uAB28-\uAB2E\uABC0-\uABEA\uABEC\uABED\uABF0-\uABF9\uAC00-\uD7A3\uD7B0-\uD7C6\uD7CB-\uD7FB\uF900-\uFA6D\uFA70-\uFAD9\uFB00-\uFB06\uFB13-\uFB17\uFB1D-\uFB28\uFB2A-\uFB36\uFB38-\uFB3C\uFB3E\uFB40\uFB41\uFB43\uFB44\uFB46-\uFBB1\uFBD3-\uFD3D\uFD50-\uFD8F\uFD92-\uFDC7\uFDF0-\uFDFB\uFE00-\uFE0F\uFE20-\uFE26\uFE33\uFE34\uFE4D-\uFE4F\uFE70-\uFE74\uFE76-\uFEFC\uFF10-\uFF19\uFF21-\uFF3A\uFF3F\uFF41-\uFF5A\uFF66-\uFFBE\uFFC2-\uFFC7\uFFCA-\uFFCF\uFFD2-\uFFD7\uFFDA-\uFFDC]')
            };
            function isDecimalDigit(ch) {
                return (ch >= 48 && ch <= 57); // 0..9
            }
            utils.isDecimalDigit = isDecimalDigit;
            function isHexDigit(ch) {
                return isDecimalDigit(ch) || (97 <= ch && ch <= 102) || (65 <= ch && ch <= 70); // A..F
            }
            utils.isHexDigit = isHexDigit;
            function isOctalDigit(ch) {
                return (ch >= 48 && ch <= 55); // 0..7
            }
            utils.isOctalDigit = isOctalDigit;
            // 7.2 White Space
            NON_ASCII_WHITESPACES = [
                0x1680,
                0x180E,
                0x2000,
                0x2001,
                0x2002,
                0x2003,
                0x2004,
                0x2005,
                0x2006,
                0x2007,
                0x2008,
                0x2009,
                0x200A,
                0x202F,
                0x205F,
                0x3000,
                0xFEFF
            ];
            function isWhiteSpace(ch) {
                return (ch === 0x20) || (ch === 0x09) || (ch === 0x0B) || (ch === 0x0C) || (ch === 0xA0) || (ch >= 0x1680 && NON_ASCII_WHITESPACES.indexOf(ch) >= 0);
            }
            utils.isWhiteSpace = isWhiteSpace;
            // 7.3 Line Terminators
            function isLineTerminator(ch) {
                return (ch === 0x0A) || (ch === 0x0D) || (ch === 0x2028) || (ch === 0x2029);
            }
            utils.isLineTerminator = isLineTerminator;
            // 7.6 Identifier Names and Identifiers
            function isIdentifierStart(ch) {
                return (ch >= 97 && ch <= 122) || (ch >= 65 && ch <= 90) || (ch === 36) || (ch === 95) || (ch === 92) || ((ch >= 0x80) && Regex.NonAsciiIdentifierStart.test(String.fromCharCode(ch)));
            }
            utils.isIdentifierStart = isIdentifierStart;
            function isIdentifierPart(ch) {
                return (ch >= 97 && ch <= 122) || (ch >= 65 && ch <= 90) || (ch >= 48 && ch <= 57) || (ch === 36) || (ch === 95) || (ch === 92) || ((ch >= 0x80) && Regex.NonAsciiIdentifierPart.test(String.fromCharCode(ch)));
            }
            utils.isIdentifierPart = isIdentifierPart;
            function isExpression(node) {
                if (node == null) {
                    return false;
                }
                switch (node.type) {
                    case 'ArrayExpression':
                    case 'AssignmentExpression':
                    case 'BinaryExpression':
                    case 'CallExpression':
                    case 'ConditionalExpression':
                    case 'FunctionExpression':
                    case 'Identifier':
                    case 'Literal':
                    case 'LogicalExpression':
                    case 'MemberExpression':
                    case 'NewExpression':
                    case 'ObjectExpression':
                    case 'SequenceExpression':
                    case 'ThisExpression':
                    case 'UnaryExpression':
                    case 'UpdateExpression':
                        return true;
                }
                return false;
            }
            utils.isExpression = isExpression;
            function isIterationStatement(node) {
                if (node == null) {
                    return false;
                }
                switch (node.type) {
                    case 'DoWhileStatement':
                    case 'ForInStatement':
                    case 'ForStatement':
                    case 'WhileStatement':
                        return true;
                }
                return false;
            }
            utils.isIterationStatement = isIterationStatement;
            function isStatement(node) {
                if (node == null) {
                    return false;
                }
                switch (node.type) {
                    case 'BlockStatement':
                    case 'BreakStatement':
                    case 'ContinueStatement':
                    case 'DebuggerStatement':
                    case 'DoWhileStatement':
                    case 'EmptyStatement':
                    case 'ExpressionStatement':
                    case 'ForInStatement':
                    case 'ForStatement':
                    case 'IfStatement':
                    case 'LabeledStatement':
                    case 'ReturnStatement':
                    case 'SwitchStatement':
                    case 'ThrowStatement':
                    case 'TryStatement':
                    case 'VariableDeclaration':
                    case 'WhileStatement':
                    case 'WithStatement':
                        return true;
                }
                return false;
            }
            utils.isStatement = isStatement;
            function isSourceElement(node) {
                return isStatement(node) || node != null && node.type === 'FunctionDeclaration';
            }
            utils.isSourceElement = isSourceElement;
            function trailingStatement(node) {
                switch (node.type) {
                    case 'IfStatement':
                        if (node.alternate != null) {
                            return node.alternate;
                        }
                        return node.consequent;
                    case 'LabeledStatement':
                    case 'ForStatement':
                    case 'ForInStatement':
                    case 'WhileStatement':
                    case 'WithStatement':
                        return node.body;
                }
                return null;
            }
            utils.trailingStatement = trailingStatement;
            function isProblematicIfStatement(node) {
                var current;
                if (node.type !== 'IfStatement') {
                    return false;
                }
                if (node.alternate == null) {
                    return false;
                }
                current = node.consequent;
                do {
                    if (current.type === 'IfStatement') {
                        if (current.alternate == null) {
                            return true;
                        }
                    }
                    current = trailingStatement(current);
                } while (current);
                return false;
            }
            utils.isProblematicIfStatement = isProblematicIfStatement;
            function isStrictModeReservedWordES6(id) {
                switch (id) {
                    case 'implements':
                    case 'interface':
                    case 'package':
                    case 'private':
                    case 'protected':
                    case 'public':
                    case 'static':
                    case 'let':
                        return true;
                    default:
                        return false;
                }
            }
            utils.isStrictModeReservedWordES6 = isStrictModeReservedWordES6;
            function isKeywordES5(id, strict) {
                // yield should not be treated as keyword under non-strict mode.
                if (!strict && id === 'yield') {
                    return false;
                }
                return isKeywordES6(id, strict);
            }
            utils.isKeywordES5 = isKeywordES5;
            function isKeywordES6(id, strict) {
                if (strict && isStrictModeReservedWordES6(id)) {
                    return true;
                }
                switch (id.length) {
                    case 2:
                        return (id === 'if') || (id === 'in') || (id === 'do');
                    case 3:
                        return (id === 'var') || (id === 'for') || (id === 'new') || (id === 'try');
                    case 4:
                        return (id === 'this') || (id === 'else') || (id === 'case') || (id === 'void') || (id === 'with') || (id === 'enum');
                    case 5:
                        return (id === 'while') || (id === 'break') || (id === 'catch') || (id === 'throw') || (id === 'const') || (id === 'yield') || (id === 'class') || (id === 'super');
                    case 6:
                        return (id === 'return') || (id === 'typeof') || (id === 'delete') || (id === 'switch') || (id === 'export') || (id === 'import');
                    case 7:
                        return (id === 'default') || (id === 'finally') || (id === 'extends');
                    case 8:
                        return (id === 'export function') || (id === 'continue') || (id === 'debugger');
                    case 10:
                        return (id === 'instanceof');
                    default:
                        return false;
                }
            }
            utils.isKeywordES6 = isKeywordES6;
            function isReservedWordES5(id, strict) {
                return id === 'null' || id === 'true' || id === 'false' || isKeywordES5(id, strict);
            }
            utils.isReservedWordES5 = isReservedWordES5;
            function isReservedWordES6(id, strict) {
                return id === 'null' || id === 'true' || id === 'false' || isKeywordES6(id, strict);
            }
            utils.isReservedWordES6 = isReservedWordES6;
            function isRestrictedWord(id) {
                return id === 'eval' || id === 'arguments';
            }
            utils.isRestrictedWord = isRestrictedWord;
            function isIdentifierName(id) {
                var i, iz, ch;
                if (id.length === 0) {
                    return false;
                }
                ch = id.charCodeAt(0);
                if (!isIdentifierStart(ch) || ch === 92) {
                    return false;
                }
                for (i = 1, iz = id.length; i < iz; ++i) {
                    ch = id.charCodeAt(i);
                    if (!isIdentifierPart(ch) || ch === 92) {
                        return false;
                    }
                }
                return true;
            }
            utils.isIdentifierName = isIdentifierName;
            function isIdentifierES5(id, strict) {
                return isIdentifierName(id) && !isReservedWordES5(id, strict);
            }
            utils.isIdentifierES5 = isIdentifierES5;
            function isIdentifierES6(id, strict) {
                return isIdentifierName(id) && !isReservedWordES6(id, strict);
            }
            utils.isIdentifierES6 = isIdentifierES6;
        })(utils = ast.utils || (ast.utils = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
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
                function Barrier(lineNumber, thread) {
                    this.lineNumber = lineNumber;
                    this.thread = thread;
                }
                return Barrier;
            })();
            exec.Barrier = Barrier;
            var BarrierGroup = (function () {
                function BarrierGroup(dim) {
                    this.mask = new Array(dim.flattenedLength());
                }
                return BarrierGroup;
            })();
            exec.BarrierGroup = BarrierGroup;
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
                    this.gridIdx = new lib.cuda.Dim3(0);
                    this.gridDim = new lib.cuda.Dim3(0);
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
/* -*- Mode: js; js-indent-level: 2; -*- */
/*
 * Copyright 2011 Mozilla Foundation and contributors
 * Licensed under the New BSD license. See LICENSE or:
 * http://opensource.org/licenses/BSD-3-Clause
 */
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var internal;
        (function (internal) {
            /**
             * Because behavior goes wacky when you set `__proto__` on objects, we
             * have to prefix all the strings in our set with an arbitrary character.
             *
             * See https://github.com/mozilla/source-map/pull/31 and
             * https://github.com/mozilla/source-map/issues/30
             *
             * @param String aStr
             */
            function toSetString(aStr) {
                return '$' + aStr;
            }
            internal.toSetString = toSetString;
            function fromSetString(aStr) {
                return aStr.substr(1);
            }
            internal.fromSetString = fromSetString;
        })(internal || (internal = {}));
        /**
         * A data structure which is a combination of an array and a set. Adding a new
         * member is O(1), testing for membership is O(1), and finding the index of an
         * element is O(1). Removing elements from the set is not supported. Only
         * strings are supported for membership.
         */
        var ArraySet = (function () {
            function ArraySet() {
                this._array = [];
                this._set = {};
                this._array = [];
                this._set = {};
            }
            /**
             * Static method for creating ArraySet instances from an existing array.
             */
            ArraySet.fromArray = function (aArray, aAllowDuplicates) {
                var set = new ArraySet();
                for (var i = 0, len = aArray.length; i < len; i++) {
                    set.add(aArray[i], aAllowDuplicates);
                }
                return set;
            };
            /**
             * Add the given string to this set.
             *
             * @param String aStr
             */
            ArraySet.prototype.add = function (aStr, aAllowDuplicates) {
                var isDuplicate = this.has(aStr);
                var idx = this._array.length;
                if (!isDuplicate || aAllowDuplicates) {
                    this._array.push(aStr);
                }
                if (!isDuplicate) {
                    this._set[internal.toSetString(aStr)] = idx;
                }
            };
            /**
             * Is the given string a member of this set?
             *
             * @param String aStr
             */
            ArraySet.prototype.has = function (aStr) {
                return Object.prototype.hasOwnProperty.call(this._set, internal.toSetString(aStr));
            };
            /**
             * What is the index of the given string in the array?
             *
             * @param String aStr
             */
            ArraySet.prototype.indexOf = function (aStr) {
                if (this.has(aStr)) {
                    return this._set[internal.toSetString(aStr)];
                }
                throw new utils.Error('"' + aStr + '" is not in the set.');
            };
            /**
             * What is the element at the given index?
             *
             * @param Number aIdx
             */
            ArraySet.prototype.at = function (aIdx) {
                if (aIdx >= 0 && aIdx < this._array.length) {
                    return this._array[aIdx];
                }
                throw new utils.Error('No element indexed by ' + aIdx);
            };
            /**
             * Returns the array representation of this set (which has the proper indices
             * indicated by indexOf). Note that this is a copy of the internal array used
             * for storing the members so that no one can mess with internal state.
             */
            ArraySet.prototype.toArray = function () {
                return this._array.slice();
            };
            return ArraySet;
        })();
        utils.ArraySet = ArraySet;
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var base64;
        (function (base64) {
            var charToIntMap = {};
            var intToCharMap = {};
            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'.split('').forEach(function (ch, index) {
                charToIntMap[ch] = index;
                intToCharMap[index] = ch;
            });
            /**
             * Encode an integer in the range of 0 to 63 to a single base 64 digit.
             */
            function encode(aNumber) {
                if (aNumber in intToCharMap) {
                    return intToCharMap[aNumber];
                }
                throw new TypeError("Must be between 0 and 63: " + aNumber);
            }
            base64.encode = encode;
            ;
            /**
             * Decode a single base 64 digit to an integer.
             */
            function decode(aChar) {
                if (aChar in charToIntMap) {
                    return charToIntMap[aChar];
                }
                throw new TypeError("Not a valid base 64 digit: " + aChar);
            }
            base64.decode = decode;
            ;
        })(base64 = utils.base64 || (utils.base64 = {}));
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        /**
         * Characters used on the terminal to format output.
         * Maps each type of formatting to its [beginning, end] characters as an array.
         *
         * Modified from colors.js.
         * @url https://github.com/Marak/colors.js
         */
        utils.FormatChars = {
            // styles
            BOLD: ['\x1B[1m', '\x1B[22m'],
            ITALICS: ['\x1B[3m', '\x1B[23m'],
            UNDERLINE: ['\x1B[4m', '\x1B[24m'],
            INVERSE: ['\x1B[7m', '\x1B[27m'],
            STRIKETHROUGH: ['\x1B[9m', '\x1B[29m'],
            // text colors
            // grayscale
            WHITE: ['\x1B[37m', '\x1B[39m'],
            GREY: ['\x1B[90m', '\x1B[39m'],
            BLACK: ['\x1B[30m', '\x1B[39m'],
            // colors
            BLUE: ['\x1B[34m', '\x1B[39m'],
            CYAN: ['\x1B[36m', '\x1B[39m'],
            GREEN: ['\x1B[32m', '\x1B[39m'],
            MAGENTA: ['\x1B[35m', '\x1B[39m'],
            RED: ['\x1B[31m', '\x1B[39m'],
            YELLOW: ['\x1B[33m', '\x1B[39m'],
            // background colors
            // grayscale
            WHITE_BG: ['\x1B[47m', '\x1B[49m'],
            GREY_BG: ['\x1B[49;5;8m', '\x1B[49m'],
            BLACK_BG: ['\x1B[40m', '\x1B[49m'],
            // colors
            BLUE_BG: ['\x1B[44m', '\x1B[49m'],
            CYAN_BG: ['\x1B[46m', '\x1B[49m'],
            GREEN_BG: ['\x1B[42m', '\x1B[49m'],
            MAGENTA_BG: ['\x1B[45m', '\x1B[49m'],
            RED_BG: ['\x1B[41m', '\x1B[49m'],
            YELLOW_BG: ['\x1B[43m', '\x1B[49m']
        };
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var functional;
        (function (functional) {
            function forEach(array, callback) {
                var result;
                if (array) {
                    for (var i = 0, len = array.length; i < len; i++) {
                        if (result = callback(array[i])) {
                            break;
                        }
                    }
                }
                return result;
            }
            functional.forEach = forEach;
            function contains(array, value) {
                if (array) {
                    for (var i = 0, len = array.length; i < len; i++) {
                        if (array[i] === value) {
                            return true;
                        }
                    }
                }
                return false;
            }
            functional.contains = contains;
            function indexOf(array, value) {
                if (array) {
                    for (var i = 0, len = array.length; i < len; i++) {
                        if (array[i] === value) {
                            return i;
                        }
                    }
                }
                return -1;
            }
            functional.indexOf = indexOf;
            function countWhere(array, predicate) {
                var count = 0;
                if (array) {
                    for (var i = 0, len = array.length; i < len; i++) {
                        if (predicate(array[i])) {
                            count++;
                        }
                    }
                }
                return count;
            }
            functional.countWhere = countWhere;
            function filter(array, f) {
                if (array) {
                    var result = [];
                    for (var i = 0, len = array.length; i < len; i++) {
                        var item = array[i];
                        if (f(item)) {
                            result.push(item);
                        }
                    }
                }
                return result;
            }
            functional.filter = filter;
            function map(array, f) {
                if (array) {
                    var result = [];
                    for (var i = 0, len = array.length; i < len; i++) {
                        result.push(f(array[i]));
                    }
                }
                return result;
            }
            functional.map = map;
            function concatenate(array1, array2) {
                if (!array2 || !array2.length)
                    return array1;
                if (!array1 || !array1.length)
                    return array2;
                return array1.concat(array2);
            }
            functional.concatenate = concatenate;
            function deduplicate(array) {
                if (array) {
                    var result = [];
                    for (var i = 0, len = array.length; i < len; i++) {
                        var item = array[i];
                        if (!contains(result, item))
                            result.push(item);
                    }
                }
                return result;
            }
            functional.deduplicate = deduplicate;
            function sum(array, prop) {
                var result = 0;
                for (var i = 0; i < array.length; i++) {
                    result += array[i][prop];
                }
                return result;
            }
            functional.sum = sum;
            /**
             * Returns the last element of an array if non-empty, undefined otherwise.
             */
            function lastOrUndefined(array) {
                if (array.length === 0) {
                    return undefined;
                }
                return array[array.length - 1];
            }
            functional.lastOrUndefined = lastOrUndefined;
            function binarySearch(array, value) {
                var low = 0;
                var high = array.length - 1;
                while (low <= high) {
                    var middle = low + ((high - low) >> 1);
                    var midValue = array[middle];
                    if (midValue === value) {
                        return middle;
                    }
                    else if (midValue > value) {
                        high = middle - 1;
                    }
                    else {
                        low = middle + 1;
                    }
                }
                return ~low;
            }
            functional.binarySearch = binarySearch;
            var hasOwnProperty = Object.prototype.hasOwnProperty;
            function hasProperty(map, key) {
                return hasOwnProperty.call(map, key);
            }
            functional.hasProperty = hasProperty;
            function getProperty(map, key) {
                return hasOwnProperty.call(map, key) ? map[key] : undefined;
            }
            functional.getProperty = getProperty;
            function isEmpty(map) {
                for (var id in map) {
                    if (hasProperty(map, id)) {
                        return false;
                    }
                }
                return true;
            }
            functional.isEmpty = isEmpty;
            function clone(object) {
                var result = {};
                for (var id in object) {
                    result[id] = object[id];
                }
                return result;
            }
            functional.clone = clone;
            function forEachValue(map, callback) {
                var result;
                for (var id in map) {
                    if (result = callback(map[id]))
                        break;
                }
                return result;
            }
            functional.forEachValue = forEachValue;
            function forEachKey(map, callback) {
                var result;
                for (var id in map) {
                    if (result = callback(id))
                        break;
                }
                return result;
            }
            functional.forEachKey = forEachKey;
            function lookUp(map, key) {
                return hasProperty(map, key) ? map[key] : undefined;
            }
            functional.lookUp = lookUp;
            function mapToArray(map) {
                var result = [];
                for (var id in map) {
                    result.push(map[id]);
                }
                return result;
            }
            functional.mapToArray = mapToArray;
        })(functional = utils.functional || (utils.functional = {}));
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
/*
 Copyright (C) 2012 Yusuke Suzuki <utatane.tea@gmail.com>
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/*global module:true*/
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var Map = (function () {
            function Map() {
                this.__data = {};
            }
            Map.prototype.get = function (key) {
                key = '$' + key;
                if (this.__data.hasOwnProperty(key)) {
                    return this.__data[key];
                }
            };
            Map.prototype.has = function (key) {
                key = '$' + key;
                return this.__data.hasOwnProperty(key);
            };
            Map.prototype.set = function (key, val) {
                key = '$' + key;
                this.__data[key] = val;
            };
            Map.prototype.delete = function (key) {
                key = '$' + key;
                return delete this.__data[key];
            };
            Map.prototype.clear = function () {
                this.__data = {};
            };
            Map.prototype.forEach = function (callback, thisArg) {
                var real, key;
                for (real in this.__data) {
                    if (this.__data.hasOwnProperty(real)) {
                        key = real.substring(1);
                        callback.call(thisArg, this.__data[real], key, this);
                    }
                }
            };
            Map.prototype.keys = function () {
                var real, result;
                result = [];
                for (real in this.__data) {
                    if (this.__data.hasOwnProperty(real)) {
                        result.push(real.substring(1));
                    }
                }
                return result;
            };
            Map.prototype.values = function () {
                var real, result;
                result = [];
                for (real in this.__data) {
                    if (this.__data.hasOwnProperty(real)) {
                        result.push(this.__data[real]);
                    }
                }
                return result;
            };
            Map.prototype.items = function () {
                var real, result;
                result = [];
                for (real in this.__data) {
                    if (this.__data.hasOwnProperty(real)) {
                        result.push([real.substring(1), this.__data[real]]);
                    }
                }
                return result;
            };
            return Map;
        })();
        utils.Map = Map;
        if (!utils.isNullOrUndefined(utils.globals.Map)) {
            Map = utils.globals.Map;
        }
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
/* -*- Mode: js; js-indent-level: 2; -*- */
/*
 * Copyright 2011 Mozilla Foundation and contributors
 * Licensed under the New BSD license. See LICENSE or:
 * http://opensource.org/licenses/BSD-3-Clause
 */
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        /**
         * Recursive implementation of binary search.
         *
         * @param aLow Indices here and lower do not contain the needle.
         * @param aHigh Indices here and higher do not contain the needle.
         * @param aNeedle The element being searched for.
         * @param aHaystack The non-empty array being searched.
         * @param aCompare Function which takes two elements and returns -1, 0, or 1.
         */
        function recursiveSearch(aLow, aHigh, aNeedle, aHaystack, aCompare) {
            // This function terminates when one of the following is true:
            //
            //   1. We find the exact element we are looking for.
            //
            //   2. We did not find the exact element, but we can return the index of
            //      the next closest element that is less than that element.
            //
            //   3. We did not find the exact element, and there is no next-closest
            //      element which is less than the one we are searching for, so we
            //      return -1.
            var mid = Math.floor((aHigh - aLow) / 2) + aLow;
            var cmp = aCompare(aNeedle, aHaystack[mid], true);
            if (cmp === 0) {
                // Found the element we are looking for.
                return mid;
            }
            else if (cmp > 0) {
                // aHaystack[mid] is greater than our needle.
                if (aHigh - mid > 1) {
                    // The element is in the upper half.
                    return recursiveSearch(mid, aHigh, aNeedle, aHaystack, aCompare);
                }
                // We did not find an exact match, return the next closest one
                // (termination case 2).
                return mid;
            }
            else {
                // aHaystack[mid] is less than our needle.
                if (mid - aLow > 1) {
                    // The element is in the lower half.
                    return recursiveSearch(aLow, mid, aNeedle, aHaystack, aCompare);
                }
                // The exact needle element was not found in this haystack. Determine if
                // we are in termination case (2) or (3) and return the appropriate thing.
                return aLow < 0 ? -1 : aLow;
            }
        }
        /**
         * This is an implementation of binary search which will always try and return
         * the index of next lowest value checked if there is no exact hit. This is
         * because mappings between original and generated line/col pairs are single
         * points, and there is an implicit region between each of them, so a miss
         * just means that you aren't on the very start of a region.
         *
         * @param aNeedle The element you are looking for.
         * @param aHaystack The array that is being searched.
         * @param aCompare A function which takes the needle and an element in the
         *     array and returns -1, 0, or 1 depending on whether the needle is less
         *     than, equal to, or greater than the element, respectively.
         */
        function search(aNeedle, aHaystack, aCompare) {
            if (aHaystack.length === 0) {
                return -1;
            }
            return recursiveSearch(-1, aHaystack.length, aNeedle, aHaystack, aCompare);
        }
        utils.search = search;
        ;
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var parallel;
        (function (parallel) {
            var Semaphore = (function () {
                function Semaphore(n) {
                    this.n = n;
                    this._queue = [];
                    this._curr = n;
                }
                Semaphore.prototype.enter = function (fn) {
                    if (this._curr > 0) {
                        --this._curr;
                    }
                    else {
                        this._queue.push(fn);
                    }
                };
                Semaphore.prototype.leave = function () {
                    if (this._queue.length > 0) {
                        var fn = this._queue.pop();
                        fn();
                    }
                    else {
                        ++this._curr;
                    }
                };
                return Semaphore;
            })();
            parallel.Semaphore = Semaphore;
        })(parallel = utils.parallel || (utils.parallel = {}));
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
/*
 Copyright (c) 2012 Barnesandnoble.com, llc, Donavon West, and Domenic Denicola

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 */
/// <reference path="utils.ts" />
// from https://github.com/YuzuJS/setImmediate
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var nextHandle = 1; // Spec says greater than zero
        var tasksByHandle = {};
        var currentlyRunningATask = false;
        var global = utils.globals;
        var doc = global.document;
        var setImmediate;
        function addFromSetImmediateArguments(args) {
            tasksByHandle[nextHandle] = partiallyApplied.apply(undefined, args);
            return nextHandle++;
        }
        // This function accepts the same arguments as setImmediate, but
        // returns a function that requires no arguments.
        function partiallyApplied(handler) {
            var args = [].slice.call(arguments, 1);
            return function () {
                if (typeof handler === "function") {
                    handler.apply(undefined, args);
                }
                else {
                    var f = new Function("" + handler);
                    f();
                }
            };
        }
        function runIfPresent(handle) {
            // From the spec: "Wait until any invocations of this algorithm started before this one have completed."
            // So if we're currently running a task, we'll need to delay this invocation.
            if (currentlyRunningATask) {
                // Delay by doing a setTimeout. setImmediate was tried instead, but in Firefox 7 it generated a
                // "too much recursion" error.
                setTimeout(partiallyApplied(runIfPresent, handle), 0);
            }
            else {
                var task = tasksByHandle[handle];
                if (task) {
                    currentlyRunningATask = true;
                    try {
                        task();
                    }
                    finally {
                        clearImmediate(handle);
                        currentlyRunningATask = false;
                    }
                }
            }
        }
        function clearImmediate(handle) {
            delete tasksByHandle[handle];
        }
        function installNextTickImplementation() {
            utils.setImmediate = function () {
                var handle = addFromSetImmediateArguments(arguments);
                process.nextTick(partiallyApplied(runIfPresent, handle));
                return handle;
            };
        }
        function canUsePostMessage() {
            // The test against `importScripts` prevents this implementation from being installed inside a web worker,
            // where `global.postMessage` means something completely different and can't be used for this purpose.
            if (global.postMessage && !global.importScripts) {
                var postMessageIsAsynchronous = true;
                var oldOnMessage = global.onmessage;
                global.onmessage = function () {
                    postMessageIsAsynchronous = false;
                };
                global.postMessage("", "*");
                global.onmessage = oldOnMessage;
                return postMessageIsAsynchronous;
            }
        }
        function installPostMessageImplementation() {
            // Installs an event handler on `global` for the `message` event: see
            // * https://developer.mozilla.org/en/DOM/window.postMessage
            // * http://www.whatwg.org/specs/web-apps/current-work/multipage/comms.html#crossDocumentMessages
            var messagePrefix = "setImmediate$" + Math.random() + "$";
            var onGlobalMessage = function (event) {
                if (event.source === global && typeof event.data === "string" && event.data.indexOf(messagePrefix) === 0) {
                    runIfPresent(+event.data.slice(messagePrefix.length));
                }
            };
            if (global.addEventListener) {
                global.addEventListener("message", onGlobalMessage, false);
            }
            else {
                global.attachEvent("onmessage", onGlobalMessage);
            }
            utils.setImmediate = function () {
                var handle = addFromSetImmediateArguments(arguments);
                global.postMessage(messagePrefix + handle, "*");
                return handle;
            };
        }
        function installMessageChannelImplementation() {
            var channel = new MessageChannel();
            channel.port1.onmessage = function (event) {
                var handle = event.data;
                runIfPresent(handle);
            };
            utils.setImmediate = function () {
                var handle = addFromSetImmediateArguments(arguments);
                channel.port2.postMessage(handle);
                return handle;
            };
        }
        function installReadyStateChangeImplementation() {
            var html = doc.documentElement;
            utils.setImmediate = function () {
                var handle = addFromSetImmediateArguments(arguments);
                // Create a <script> element; its readystatechange event will be fired asynchronously once it is inserted
                // into the document. Do so, thus queuing up the task. Remember to clean up once it's been called.
                var script = doc.createElement("script");
                script.onreadystatechange = function () {
                    runIfPresent(handle);
                    script.onreadystatechange = null;
                    html.removeChild(script);
                    script = null;
                };
                html.appendChild(script);
                return handle;
            };
        }
        function installSetTimeoutImplementation() {
            utils.setImmediate = function () {
                var handle = addFromSetImmediateArguments(arguments);
                setTimeout(partiallyApplied(runIfPresent, handle), 0);
                return handle;
            };
        }
        // If supported, we should attach to the prototype of global, since that is where setTimeout et al. live.
        var attachTo = Object.getPrototypeOf && Object.getPrototypeOf(global);
        attachTo = attachTo && attachTo.setTimeout ? attachTo : global;
        // Don't get fooled by e.g. browserify environments.
        if ({}.toString.call(global.process) === "[object process]") {
            // For Node.js before 0.9
            installNextTickImplementation();
        }
        else if (canUsePostMessage()) {
            // For non-IE10 modern browsers
            installPostMessageImplementation();
        }
        else if (global.MessageChannel) {
            // For web workers, where supported
            installMessageChannelImplementation();
        }
        else if (doc && "onreadystatechange" in doc.createElement("script")) {
            // For IE 6–8
            installReadyStateChangeImplementation();
        }
        else {
            // For older browsers
            installSetTimeoutImplementation();
        }
        utils.setImmediate = utils.setImmediate;
        utils.clearImmediate = clearImmediate;
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
/// <reference path="utils.ts" />
// copied from https://github.com/paulmillr/es6-shim/blob/master/es6-shim.js
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        function toInt32(x) {
            return x >> 0;
        }
        utils.toInt32 = toInt32;
        function toUint32(x) {
            return x >>> 0;
        }
        utils.toUint32 = toUint32;
        function toInteger(value) {
            var number = +value;
            if (math.isNaN(number)) {
                return 0;
            }
            if (number === 0 || !math.isFinite(number)) {
                return number;
            }
            return (number > 0 ? 1 : -1) * Math.floor(Math.abs(number));
        }
        utils.toInteger = toInteger;
        var numberConversion;
        (function (numberConversion) {
            // from https://github.com/inexorabletash/polyfill/blob/master/typedarray.js#L176-L266
            // with permission and license, per https://twitter.com/inexorabletash/status/372206509540659200
            function roundToEven(n) {
                var w = Math.floor(n), f = n - w;
                if (f < 0.5) {
                    return w;
                }
                if (f > 0.5) {
                    return w + 1;
                }
                return w % 2 ? w + 1 : w;
            }
            function packIEEE754(v, ebits, fbits) {
                var bias = (1 << (ebits - 1)) - 1, s, e, f, i, bits, str, bytes;
                // Compute sign, exponent, fraction
                if (v !== v) {
                    // NaN
                    // http://dev.w3.org/2006/webapi/WebIDL/#es-type-mapping
                    e = (1 << ebits) - 1;
                    f = Math.pow(2, fbits - 1);
                    s = 0;
                }
                else if (v === Infinity || v === -Infinity) {
                    e = (1 << ebits) - 1;
                    f = 0;
                    s = (v < 0) ? 1 : 0;
                }
                else if (v === 0) {
                    e = 0;
                    f = 0;
                    s = (1 / v === -Infinity) ? 1 : 0;
                }
                else {
                    s = v < 0;
                    v = Math.abs(v);
                    if (v >= Math.pow(2, 1 - bias)) {
                        e = Math.min(Math.floor(Math.log(v) / Math.LN2), 1023);
                        f = roundToEven(v / Math.pow(2, e) * Math.pow(2, fbits));
                        if (f / Math.pow(2, fbits) >= 2) {
                            e = e + 1;
                            f = 1;
                        }
                        if (e > bias) {
                            // Overflow
                            e = (1 << ebits) - 1;
                            f = 0;
                        }
                        else {
                            // Normal
                            e = e + bias;
                            f = f - Math.pow(2, fbits);
                        }
                    }
                    else {
                        // Subnormal
                        e = 0;
                        f = roundToEven(v / Math.pow(2, 1 - bias - fbits));
                    }
                }
                // Pack sign, exponent, fraction
                bits = [];
                for (i = fbits; i; i -= 1) {
                    bits.push(f % 2 ? 1 : 0);
                    f = Math.floor(f / 2);
                }
                for (i = ebits; i; i -= 1) {
                    bits.push(e % 2 ? 1 : 0);
                    e = Math.floor(e / 2);
                }
                bits.push(s ? 1 : 0);
                bits.reverse();
                str = bits.join('');
                // Bits to bytes
                bytes = [];
                while (str.length) {
                    bytes.push(parseInt(str.slice(0, 8), 2));
                    str = str.slice(8);
                }
                return bytes;
            }
            function unpackIEEE754(bytes, ebits, fbits) {
                // Bytes to bits
                var bits = [], i, j, b, str, bias, s, e, f;
                for (i = bytes.length; i; i -= 1) {
                    b = bytes[i - 1];
                    for (j = 8; j; j -= 1) {
                        bits.push(b % 2 ? 1 : 0);
                        b = b >> 1;
                    }
                }
                bits.reverse();
                str = bits.join('');
                // Unpack sign, exponent, fraction
                bias = (1 << (ebits - 1)) - 1;
                s = parseInt(str.slice(0, 1), 2) ? -1 : 1;
                e = parseInt(str.slice(1, 1 + ebits), 2);
                f = parseInt(str.slice(1 + ebits), 2);
                // Produce number
                if (e === (1 << ebits) - 1) {
                    return f !== 0 ? NaN : s * Infinity;
                }
                else if (e > 0) {
                    // Normalized
                    return s * Math.pow(2, e - bias) * (1 + f / Math.pow(2, fbits));
                }
                else if (f !== 0) {
                    // Denormalized
                    return s * Math.pow(2, -(bias - 1)) * (f / Math.pow(2, fbits));
                }
                else {
                    return s < 0 ? -0 : 0;
                }
            }
            function unpackFloat64(b) {
                return unpackIEEE754(b, 11, 52);
            }
            function packFloat64(v) {
                return packIEEE754(v, 11, 52);
            }
            function unpackFloat32(b) {
                return unpackIEEE754(b, 8, 23);
            }
            function packFloat32(v) {
                return packIEEE754(v, 8, 23);
            }
            numberConversion.toFloat32 = function (num) {
                return unpackFloat32(packFloat32(num));
            };
            if (typeof Float32Array !== 'undefined') {
                var float32array = new Float32Array(1);
                numberConversion.toFloat32 = function (num) {
                    float32array[0] = num;
                    return float32array[0];
                };
            }
        })(numberConversion || (numberConversion = {}));
        utils.toFloat32 = numberConversion.toFloat32;
        function isCallableWithoutNew(func) {
            try {
                func();
            }
            catch (e) {
                return false;
            }
            return true;
        }
        utils.isCallableWithoutNew = isCallableWithoutNew;
        ;
        function isCallable(x) {
            return typeof x === 'function' && utils._toString.call(x) === '[object Function]';
        }
        utils.isCallable = isCallable;
        var arePropertyDescriptorsSupported = function () {
            try {
                Object.defineProperty({}, 'x', {});
                return true;
            }
            catch (e) {
                return false;
            }
        };
        var supportsDescriptors = !!Object.defineProperty && arePropertyDescriptorsSupported();
        var defineProperty = function (object, name, value, force) {
            if (!force && name in object) {
                return;
            }
            if (supportsDescriptors) {
                Object.defineProperty(object, name, {
                    configurable: true,
                    enumerable: false,
                    writable: true,
                    value: value
                });
            }
            else {
                object[name] = value;
            }
        };
        // Define configurable, writable and non-enumerable props
        // if they don’t exist.
        utils.defineProperties = function (object, map) {
            Object.keys(map).forEach(function (name) {
                var method = map[name];
                defineProperty(object, name, method, false);
            });
        };
        var math;
        (function (math) {
            function isNaN(value) {
                // NaN !== NaN, but they are identical.
                // NaNs are the only non-reflexive value, i.e., if x !== x,
                // then x is NaN.
                // isNaN is broken: it converts its argument to number, so
                // isNaN('foo') => true
                return value !== value;
            }
            math.isNaN = isNaN;
            function acosh(value) {
                value = Number(value);
                if (isNaN(value) || value < 1) {
                    return NaN;
                }
                if (value === 1) {
                    return 0;
                }
                if (value === Infinity) {
                    return value;
                }
                return Math.log(value + Math.sqrt(value * value - 1));
            }
            math.acosh = acosh;
            function asinh(value) {
                value = Number(value);
                if (value === 0 || !utils.global_isFinite(value)) {
                    return value;
                }
                return value < 0 ? -asinh(-value) : Math.log(value + Math.sqrt(value * value + 1));
            }
            math.asinh = asinh;
            function atanh(value) {
                value = Number(value);
                if (isNaN(value) || value < -1 || value > 1) {
                    return NaN;
                }
                if (value === -1) {
                    return -Infinity;
                }
                if (value === 1) {
                    return Infinity;
                }
                if (value === 0) {
                    return value;
                }
                return 0.5 * Math.log((1 + value) / (1 - value));
            }
            math.atanh = atanh;
            function cbrt(value) {
                value = Number(value);
                if (value === 0) {
                    return value;
                }
                var negate = value < 0, result;
                if (negate) {
                    value = -value;
                }
                result = Math.pow(value, 1 / 3);
                return negate ? -result : result;
            }
            math.cbrt = cbrt;
            function clz32(value) {
                // See https://bugs.ecmascript.org/show_bug.cgi?id=2465
                value = Number(value);
                var number = utils.toUint32(value);
                if (number === 0) {
                    return 32;
                }
                return 32 - (number).toString(2).length;
            }
            math.clz32 = clz32;
            function cosh(value) {
                value = Number(value);
                if (value === 0) {
                    return 1;
                } // +0 or -0
                if (isNaN(value)) {
                    return NaN;
                }
                if (!utils.global_isFinite(value)) {
                    return Infinity;
                }
                if (value < 0) {
                    value = -value;
                }
                if (value > 21) {
                    return Math.exp(value) / 2;
                }
                return (Math.exp(value) + Math.exp(-value)) / 2;
            }
            math.cosh = cosh;
            function expm1(value) {
                value = Number(value);
                if (value === -Infinity) {
                    return -1;
                }
                if (!utils.global_isFinite(value) || value === 0) {
                    return value;
                }
                return Math.exp(value) - 1;
            }
            math.expm1 = expm1;
            function hypot(x, y) {
                var anyNaN = false;
                var allZero = true;
                var anyInfinity = false;
                var numbers = [];
                Array.prototype.every.call(arguments, function (arg) {
                    var num = Number(arg);
                    if (isNaN(num)) {
                        anyNaN = true;
                    }
                    else if (num === Infinity || num === -Infinity) {
                        anyInfinity = true;
                    }
                    else if (num !== 0) {
                        allZero = false;
                    }
                    if (anyInfinity) {
                        return false;
                    }
                    else if (!anyNaN) {
                        numbers.push(Math.abs(num));
                    }
                    return true;
                });
                if (anyInfinity) {
                    return Infinity;
                }
                if (anyNaN) {
                    return NaN;
                }
                if (allZero) {
                    return 0;
                }
                numbers.sort(function (a, b) {
                    return b - a;
                });
                var largest = numbers[0];
                var divided = numbers.map(function (number) {
                    return number / largest;
                });
                var sum = divided.reduce(function (sum, number) {
                    return sum += number * number;
                }, 0);
                return largest * Math.sqrt(sum);
            }
            math.hypot = hypot;
            function log2(value) {
                return Math.log(value) * Math.LOG2E;
            }
            math.log2 = log2;
            function log10(value) {
                return Math.log(value) * Math.LOG10E;
            }
            math.log10 = log10;
            function log1p(value) {
                value = Number(value);
                if (value < -1 || isNaN(value)) {
                    return NaN;
                }
                if (value === 0 || value === Infinity) {
                    return value;
                }
                if (value === -1) {
                    return -Infinity;
                }
                var result = 0;
                var n = 50;
                if (value < 0 || value > 1) {
                    return Math.log(1 + value);
                }
                for (var i = 1; i < n; i++) {
                    if ((i % 2) === 0) {
                        result -= Math.pow(value, i) / i;
                    }
                    else {
                        result += Math.pow(value, i) / i;
                    }
                }
                return result;
            }
            math.log1p = log1p;
            function sign(value) {
                var number = +value;
                if (number === 0) {
                    return number;
                }
                if (isNaN(number)) {
                    return number;
                }
                return number < 0 ? -1 : 1;
            }
            math.sign = sign;
            function sinh(value) {
                value = Number(value);
                if (!utils.global_isFinite(value) || value === 0) {
                    return value;
                }
                return (Math.exp(value) - Math.exp(-value)) / 2;
            }
            math.sinh = sinh;
            function tanh(value) {
                value = Number(value);
                if (isNaN(value) || value === 0) {
                    return value;
                }
                if (value === Infinity) {
                    return 1;
                }
                if (value === -Infinity) {
                    return -1;
                }
                return (Math.exp(value) - Math.exp(-value)) / (Math.exp(value) + Math.exp(-value));
            }
            math.tanh = tanh;
            function trunc(value) {
                var number = Number(value);
                return number < 0 ? -Math.floor(-number) : Math.floor(number);
            }
            math.trunc = trunc;
            function imul(x, y) {
                // taken from https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/imul
                x = utils.toUint32(x);
                y = utils.toUint32(y);
                var ah = (x >>> 16) & 0xffff;
                var al = x & 0xffff;
                var bh = (y >>> 16) & 0xffff;
                var bl = y & 0xffff;
                // the shift by 0 fixes the sign on the high part
                // the final |0 converts the unsigned value into a signed value
                return ((al * bl) + (((ah * bl + al * bh) << 16) >>> 0) | 0);
            }
            math.imul = imul;
            function fround(x) {
                if (x === 0 || x === Infinity || x === -Infinity || isNaN(x)) {
                    return x;
                }
                var num = Number(x);
                return numberConversion.toFloat32(num);
            }
            math.fround = fround;
            var maxSafeInteger = Math.pow(2, 53) - 1;
            var MAX_SAFE_INTEGER = maxSafeInteger;
            var MIN_SAFE_INTEGER = -maxSafeInteger;
            var EPSILON = 2.220446049250313e-16;
            function isFinite(value) {
                return typeof value === 'number' && utils.global_isFinite(value);
            }
            math.isFinite = isFinite;
            function isInteger(value) {
                return isFinite(value) && utils.toInteger(value) === value;
            }
            math.isInteger = isInteger;
            function isSafeInteger(value) {
                return isInteger(value) && Math.abs(value) <= MAX_SAFE_INTEGER;
            }
            math.isSafeInteger = isSafeInteger;
        })(math = utils.math || (utils.math = {}));
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
/* -*- Mode: js; js-indent-level: 2; -*- */
/*
 * Copyright 2011 Mozilla Foundation and contributors
 * Licensed under the New BSD license. See LICENSE or:
 * http://opensource.org/licenses/BSD-3-Clause
 *
 * Based on the Base 64 VLQ implementation in Closure Compiler:
 * https://code.google.com/p/closure-compiler/source/browse/trunk/src/com/google/debugging/sourcemap/Base64VLQ.java
 *
 * Copyright 2011 The Closure Compiler Authors. All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *  * Neither the name of Google Inc. nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var vlq;
        (function (_vlq) {
            var base64 = lib.utils.base64;
            // A single base 64 digit can contain 6 bits of data. For the base 64 variable
            // length quantities we use in the source map spec, the first bit is the sign,
            // the next four bits are the actual value, and the 6th bit is the
            // continuation bit. The continuation bit tells us whether there are more
            // digits in this value following this digit.
            //
            //   Continuation
            //   |    Sign
            //   |    |
            //   V    V
            //   101011
            var VLQ_BASE_SHIFT = 5;
            // binary: 100000
            var VLQ_BASE = 1 << VLQ_BASE_SHIFT;
            // binary: 011111
            var VLQ_BASE_MASK = VLQ_BASE - 1;
            // binary: 100000
            var VLQ_CONTINUATION_BIT = VLQ_BASE;
            /**
             * Converts from a two-complement value to a value where the sign bit is
             * is placed in the least significant bit.  For example, as decimals:
             *   1 becomes 2 (10 binary), -1 becomes 3 (11 binary)
             *   2 becomes 4 (100 binary), -2 becomes 5 (101 binary)
             */
            function toVLQSigned(aValue) {
                return aValue < 0 ? ((-aValue) << 1) + 1 : (aValue << 1) + 0;
            }
            /**
             * Converts to a two-complement value from a value where the sign bit is
             * is placed in the least significant bit.  For example, as decimals:
             *   2 (10 binary) becomes 1, 3 (11 binary) becomes -1
             *   4 (100 binary) becomes 2, 5 (101 binary) becomes -2
             */
            function fromVLQSigned(aValue) {
                var isNegative = (aValue & 1) === 1;
                var shifted = aValue >> 1;
                return isNegative ? -shifted : shifted;
            }
            /**
             * Returns the base 64 VLQ encoded value.
             */
            function encode(aValue) {
                var encoded = "";
                var digit;
                var vlq = toVLQSigned(aValue);
                do {
                    digit = vlq & VLQ_BASE_MASK;
                    vlq >>>= VLQ_BASE_SHIFT;
                    if (vlq > 0) {
                        // There are still more digits in this value, so we must make sure the
                        // continuation bit is marked.
                        digit |= VLQ_CONTINUATION_BIT;
                    }
                    encoded += base64.encode(digit);
                } while (vlq > 0);
                return encoded;
            }
            _vlq.encode = encode;
            ;
            /**
             * Decodes the next base 64 VLQ value from the given string and returns the
             * value and the rest of the string via the out parameter.
             */
            function decode(aStr, aOutParam) {
                var i = 0;
                var strLen = aStr.length;
                var result = 0;
                var shift = 0;
                var continuation, digit;
                do {
                    if (i >= strLen) {
                        throw new utils.Error("Expected more digits in base 64 VLQ value.");
                    }
                    digit = base64.decode(aStr.charAt(i++));
                    continuation = !!(digit & VLQ_CONTINUATION_BIT);
                    digit &= VLQ_BASE_MASK;
                    result = result + (digit << shift);
                    shift += VLQ_BASE_SHIFT;
                } while (continuation);
                aOutParam.value = fromVLQSigned(result);
                aOutParam.rest = aStr.slice(i);
            }
            _vlq.decode = decode;
            ;
        })(vlq = utils.vlq || (utils.vlq = {}));
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var worker;
    (function (worker) {
        var detail;
        (function (detail) {
            if (lib.utils.isNode) {
                process.once('message', function (code) {
                    eval(JSON.parse(code).data);
                });
            }
            else {
                self.onmessage = function (code) {
                    eval(code.data);
                };
            }
        })(detail = worker.detail || (worker.detail = {}));
    })(worker = lib.worker || (lib.worker = {}));
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
