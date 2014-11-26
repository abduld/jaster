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
                Logger.prototype.go = function (msg, type) {
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
                    this.go(msg, 0 /* Debug */);
                };
                Logger.prototype.trace = function (msg) {
                    this.go(msg, 1 /* Trace */);
                };
                Logger.prototype.warn = function (msg) {
                    this.go(msg, 2 /* Warn */);
                };
                Logger.prototype.error = function (msg) {
                    this.go(msg, 3 /* Error */);
                };
                Logger.prototype.fatal = function (msg) {
                    this.go(msg, 4 /* Fatal */);
                };
                return Logger;
            })();
            detail.Logger = Logger;
        })(detail = utils.detail || (utils.detail = {}));
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
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var detail;
        (function (detail) {
            function rand(min, max) {
                return min + Math.random() * (max - min);
            }
            detail.rand = rand;
        })(detail = utils.detail || (utils.detail = {}));
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
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
/// <reference path="./int8.ts" />
/// <reference path="./type/type.ts" />
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
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var detail;
        (function (detail) {
            function castTo(arg) {
                return arg;
            }
            detail.castTo = castTo;
        })(detail = utils.detail || (utils.detail = {}));
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
/// <reference path="./logger.ts" />
/// <reference path="./assert.ts" />
/// <reference path="./rand.ts" />
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
        utils.logger = new lib.utils.detail.Logger();
        utils.assert = lib.utils.detail.assert;
        utils.guuid = lib.utils.detail.guuid;
        utils.rand = lib.utils.detail.rand;
        utils.castTo = lib.utils.detail.castTo;
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
        })(type = c.type || (c.type = {}));
    })(c = lib.c || (lib.c = {}));
})(lib || (lib = {}));
