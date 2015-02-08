/// <reference path="utils.ts" />
// copied from https://github.com/paulmillr/es6-shim/blob/master/es6-shim.js
module lib {
    export module utils {

        export function toInt32(x: any): number {
            return x >> 0;
        }

        export function toUint32(x: any): number {
            return x >>> 0;
        }

        export function toInteger(value: any): number {
            var number = +value;
            if (math.isNaN(number)) {
                return 0;
            }
            if (number === 0 || !math.isFinite(number)) {
                return number;
            }
            return (number > 0 ? 1 : -1) * Math.floor(Math.abs(number));
        }

        export module numberConversion {
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
                var bias = (1 << (ebits - 1)) - 1,
                    s, e, f,
                    i, bits, str, bytes;

                // Compute sign, exponent, fraction
                if (v !== v) {
                    // NaN
                    // http://dev.w3.org/2006/webapi/WebIDL/#es-type-mapping
                    e = (1 << ebits) - 1;
                    f = Math.pow(2, fbits - 1);
                    s = 0;
                } else if (v === Infinity || v === -Infinity) {
                    e = (1 << ebits) - 1;
                    f = 0;
                    s = (v < 0) ? 1 : 0;
                } else if (v === 0) {
                    e = 0;
                    f = 0;
                    s = (1 / v === -Infinity) ? 1 : 0;
                } else {
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
                        } else {
                            // Normal
                            e = e + bias;
                            f = f - Math.pow(2, fbits);
                        }
                    } else {
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
                var bits = [], i, j, b, str,
                    bias, s, e, f;

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
                } else if (e > 0) {
                    // Normalized
                    return s * Math.pow(2, e - bias) * (1 + f / Math.pow(2, fbits));
                } else if (f !== 0) {
                    // Denormalized
                    return s * Math.pow(2, -(bias - 1)) * (f / Math.pow(2, fbits));
                } else {
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

            export var toFloat32 = function(num: number) {
                return unpackFloat32(packFloat32(num));
            }
            if (typeof Float32Array !== 'undefined') {
                var float32array = new Float32Array(1);
                toFloat32 = function(num) {
                    float32array[0] = num;
                    return float32array[0];
                };
            }
        }
        export import toFloat32 = numberConversion.toFloat32;
        export function isCallableWithoutNew(func: Function) {
            try {
                func();
            }
            catch (e) {
                return false;
            }
            return true;
        };
        export function isCallable(x) {
            return typeof x === 'function' &&
                // some versions of IE say that typeof /abc/ === 'function'
                _toString.call(x) === '[object Function]';
        }


        var arePropertyDescriptorsSupported = function() {
            try {
                Object.defineProperty({}, 'x', {});
                return true;
            } catch (e) { /* this is IE 8. */
                return false;
            }
        };
        var supportsDescriptors = !!Object.defineProperty && arePropertyDescriptorsSupported();
        var defineProperty = function(object, name, value, force) {
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
            } else {
                object[name] = value;
            }
        };
        // Define configurable, writable and non-enumerable props
        // if they don’t exist.
        export var defineProperties = function(object, map) {
            Object.keys(map).forEach(function(name) {
                var method = map[name];
                defineProperty(object, name, method, false);
            });
        };

        export module math {
            export function isNaN(value: number): boolean {
                // NaN !== NaN, but they are identical.
                // NaNs are the only non-reflexive value, i.e., if x !== x,
                // then x is NaN.
                // isNaN is broken: it converts its argument to number, so
                // isNaN('foo') => true
                return value !== value;
            }

            export function acosh(value: number): number {
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

            export function asinh(value: number): number {
                value = Number(value);
                if (value === 0 || !global_isFinite(value)) {
                    return value;
                }
                return value < 0 ? -asinh(-value) : Math.log(value + Math.sqrt(value * value + 1));
            }

            export function atanh(value: number): number {
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

            export function cbrt(value: number): number {
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

            export function clz32(value: number): number {
                // See https://bugs.ecmascript.org/show_bug.cgi?id=2465
                value = Number(value);
                var number = utils.toUint32(value);
                if (number === 0) {
                    return 32;
                }
                return 32 - (number).toString(2).length;
            }

            export function cosh(value: number): number {
                value = Number(value);
                if (value === 0) {
                    return 1;
                } // +0 or -0
                if (isNaN(value)) {
                    return NaN;
                }
                if (!global_isFinite(value)) {
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

            export function expm1(value: number): number {
                value = Number(value);
                if (value === -Infinity) {
                    return -1;
                }
                if (!global_isFinite(value) || value === 0) {
                    return value;
                }
                return Math.exp(value) - 1;
            }

            export function hypot(x: number, y: number): number {
                var anyNaN = false;
                var allZero = true;
                var anyInfinity = false;
                var numbers = [];
                Array.prototype.every.call(arguments, function(arg) {
                    var num = Number(arg);
                    if (isNaN(num)) {
                        anyNaN = true;
                    } else if (num === Infinity || num === -Infinity) {
                        anyInfinity = true;
                    } else if (num !== 0) {
                        allZero = false;
                    }
                    if (anyInfinity) {
                        return false;
                    } else if (!anyNaN) {
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

                numbers.sort(function(a, b) {
                    return b - a;
                });
                var largest = numbers[0];
                var divided = numbers.map(function(number) {
                    return number / largest;
                });
                var sum = divided.reduce(function(sum, number) {
                    return sum += number * number;
                }, 0);
                return largest * Math.sqrt(sum);
            }

            export function log2(value: number): number {
                return Math.log(value) * Math.LOG2E;
            }

            export function log10(value: number): number {
                return Math.log(value) * Math.LOG10E;
            }

            export function log1p(value: number): number {
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
                    } else {
                        result += Math.pow(value, i) / i;
                    }
                }

                return result;
            }

            export function sign(value: number): number {
                var number = +value;
                if (number === 0) {
                    return number;
                }
                if (isNaN(number)) {
                    return number;
                }
                return number < 0 ? -1 : 1;
            }

            export function sinh(value: number): number {
                value = Number(value);
                if (!global_isFinite(value) || value === 0) {
                    return value;
                }
                return (Math.exp(value) - Math.exp(-value)) / 2;
            }

            export function tanh(value: number) {
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

            export function trunc(value: number) {
                var number = Number(value);
                return number < 0 ? -Math.floor(-number) : Math.floor(number);
            }

            export function imul(x: number, y: number): number {
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

            export function fround(x: number) {
                if (x === 0 || x === Infinity || x === -Infinity || isNaN(x)) {
                    return x;
                }
                var num = Number(x);
                return numberConversion.toFloat32(num);
            }

            var maxSafeInteger = Math.pow(2, 53) - 1;
            var MAX_SAFE_INTEGER = maxSafeInteger;
            var MIN_SAFE_INTEGER = -maxSafeInteger;
            var EPSILON = 2.220446049250313e-16;

            export function isFinite(value: number): boolean {
                return typeof value === 'number' && global_isFinite(value);
            }

            export function isInteger(value: number): boolean {
                return isFinite(value) &&
                    utils.toInteger(value) === value;
            }

            export function isSafeInteger(value: number): boolean {
                return isInteger(value) && Math.abs(value) <= MAX_SAFE_INTEGER;
            }
        }
    }
}