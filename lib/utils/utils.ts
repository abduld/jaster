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
/// <reference path="./search.ts" />
/// <reference path="./arrayset.ts" />
/// <reference path="./base64.ts" />
/// <reference path="./vlq.ts" />
/// <reference path="./shim.ts" />
/// <reference path="./semaphore.ts" />
/// <reference path="./setImmediate.ts" />
/// <reference path="./hash.ts" />
/// <reference path="./internal.ts" />
/// <reference path="./functional.ts" />
/// <reference path="./format.ts" />


module lib {

    export module utils {

        export var isCommonJS:boolean = false;
        export var isNode:boolean = false;
        /**
         * Various constant values. Enum'd so they are inlined by the TypeScript
         * compiler.
         */
        export enum constant {
            INT_MAX = Math.pow(2, 31) - 1,
            INT_MIN = -INT_MAX - 1,
            FLOAT_POS_INFINITY = Math.pow(2, 128),
            FLOAT_NEG_INFINITY = -1 * FLOAT_POS_INFINITY,
            FLOAT_POS_INFINITY_AS_INT = 0x7F800000,
            FLOAT_NEG_INFINITY_AS_INT = -8388608,
            // We use the JavaScript NaN as our NaN value, and convert it to
            // a NaN value in the SNaN range when an int equivalent is requested.
            FLOAT_NaN_AS_INT = 0x7fc00000
        }


        /*jshint evil: true */
        var getGlobal = new Function('return this;');
        /*jshint evil: false */

        export var globals = getGlobal();
        export var global_isFinite = globals.isFinite;
        var _slice = Array.prototype.slice;
        var _indexOf = String.prototype.indexOf;
        export var _toString = Object.prototype.toString;
        var _hasOwnProperty = Object.prototype.hasOwnProperty;

        var Symbol = globals.Symbol || {};

        export function isSymbol(sym) {
            /*jshint notypeof: true */
            return typeof globals.Symbol === 'function' && typeof sym === 'symbol';
            /*jshint notypeof: false */
        };

        export function isString(value):boolean {
            return typeof value === "string";
        }

        export function isFunction(value):boolean {
            return typeof value === "function";
        }

        export function isNumber(value):boolean {
            return typeof value === "number";
        }

        export function isInteger(value):boolean {
            return (value | 0) === value;
        }

        export function isArray(value):boolean {
            var f = Array.isArray || function (xs) {
                return Object.prototype.toString.call(xs) === '[object Array]';
            };
            return f(value);
        }

        export function isNumberOrString(value):boolean {
            return typeof value === "number" || typeof value === "string";
        }

        export function isObject(value):boolean {
            return typeof value === "object" || typeof value === 'function';
        }

        export function isUndefined(value) {
            return typeof value == 'undefined';
        }

        export function toNumber(x):number {
            return +x;
        }

        export function float2int(a:number):number {
            if (a > constant.INT_MAX) {
                return constant.INT_MAX;
            } else if (a < constant.INT_MIN) {
                return constant.INT_MIN;
            } else {
                return a | 0;
            }
        }

        export function isNumericString(value:string):boolean {
            // ECMAScript 5.1 - 9.8.1 Note 1, this expression is true for all
            // numbers x other than -0.
            return String(Number(value)) === value;
        }

        export function isNullOrUndefined(value) {
            return value == undefined;
        }

        export function backtrace() {
            //return "Uncomment Debug.backtrace();";
            try {
                throw new Error();
            } catch (e) {
                return e.stack ? e.stack.split('\n').slice(2).join('\n') : '';
            }
        }

        export function getTicks():number {
            return performance.now();
        }

        // Creates and initializes *JavaScript* array to *val* in each element slot.
        // Like memset, but for arrays.
        export function arrayset<T>(len:number, val:T):T[] {
            var array = new Array(len);
            for (var i = 0; i < len; i++) {
                array[i] = val;
            }
            return array;
        }

        // taken directly from https://github.com/ljharb/is-arguments/blob/master/index.js
        // can be replaced with require('is-arguments') if we ever use a build process instead
        export function isArguments(value) {
            var str = _toString.call(value);
            var result = str === '[object Arguments]';
            if (!result) {
                result = str !== '[object Array]' &&
                value !== null &&
                typeof value === 'object' &&
                typeof value.length === 'number' &&
                value.length >= 0 &&
                _toString.call(value.callee) === '[object Function]';
            }
            return result;
        };
    }
}
