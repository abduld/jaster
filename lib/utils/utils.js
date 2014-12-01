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
/// <reference path="./freeze.ts" />
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
        function isUndefined(value) {
            return typeof value == 'undefined';
        }
        utils.isUndefined = isUndefined;
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
//# sourceMappingURL=utils.js.map