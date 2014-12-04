/*
 This is free and unencumbered software released into the public domain.

 Anyone is free to copy, modify, publish, use, compile, sell, or
 distribute this software, either in source code form or as a compiled
 binary, for any purpose, commercial or non-commercial, and by any
 means.

 In jurisdictions that recognize copyright laws, the author or authors
 of this software dedicate any and all copyright interest in the
 software to the public domain. We make this dedication for the benefit
 of the public at large and to the detriment of our heirs and
 successors. We intend this dedication to be an overt act of
 relinquishment in perpetuity of all present and future rights to this
 software under copyright law.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 OTHER DEALINGS IN THE SOFTWARE.

 For more information, please refer to <http://unlicense.org>
 */

// from https://github.com/stacktracejs/error-stack-parser
module lib.utils {
    export module stacktrace {

        // ES5 Polyfills
        // See https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Function/bind
        if (!Function.prototype.bind) {
            Function.prototype.bind = function (oThis) {
                if (typeof this !== 'function') {
                    throw new TypeError('Function.prototype.bind - what is trying to be bound is not callable');
                }

                var aArgs = Array.prototype.slice.call(arguments, 1);
                var fToBind = this;
                var NoOp = function () {
                };
                var fBound = function () {
                    return fToBind.apply(this instanceof NoOp && oThis ? this : oThis,
                        aArgs.concat(Array.prototype.slice.call(arguments)));
                };

                NoOp.prototype = this.prototype;
                fBound.prototype = new NoOp();

                return fBound;
            };
        }

        // See https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/map
        if (!Array.prototype.map) {
            Array.prototype.map = function (callback, thisArg) {
                if (this === void 0 || this === null) {
                    throw new TypeError("this is null or not defined");
                }
                var O = Object(this);
                var len = O.length >>> 0;
                var T;
                if (typeof callback !== "function") {
                    throw new TypeError(callback + " is not a function");
                }
                if (arguments.length > 1) {
                    T = thisArg;
                }

                var A = new Array(len);
                var k = 0;

                while (k < len) {
                    var kValue, mappedValue;
                    if (k in O) {
                        kValue = O[k];
                        mappedValue = callback.call(T, kValue, k, O);
                        A[k] = mappedValue;
                    }
                    k++;
                }

                return A;
            };
        }

        // See https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/filter
        if (!Array.prototype.filter) {
            Array.prototype.filter = function (callback/*, thisArg*/) {
                if (this === void 0 || this === null) {
                    throw new TypeError("this is null or not defined");
                }

                var t = Object(this);
                var len = t.length >>> 0;
                if (typeof callback !== "function") {
                    throw new TypeError(callback + " is not a function");
                }

                var res = [];
                var thisArg = arguments.length >= 2 ? arguments[1] : void 0;
                for (var i = 0; i < len; i++) {
                    if (i in t) {
                        var val = t[i];
                        if (callback.call(thisArg, val, i, t)) {
                            res.push(val);
                        }
                    }
                }

                return res;
            };
        }

        var FIREFOX_SAFARI_STACK_REGEXP = /\S+\:\d+/;
        var CHROME_IE_STACK_REGEXP = /\s+at /;
        var factory = function ErrorStackParser(StackFrame) {
            return {
                /**
                 * Given an Error object, extract the most information from it.
                 * @param error {Error}
                 * @return Array[StackFrame]
                 */
                parse: function ErrorStackParser$$parse(error) {
                    if (typeof error.stacktrace !== 'undefined' || typeof error['opera#sourceloc'] !== 'undefined') {
                        return this.parseOpera(error);
                    } else if (error.stack.match(CHROME_IE_STACK_REGEXP)) {
                        return this.parseV8OrIE(error);
                    } else if (error.stack.match(FIREFOX_SAFARI_STACK_REGEXP)) {
                        return this.parseFFOrSafari(error);
                    } else {
                        throw new Error('Cannot parse given Error object');
                    }
                },

                /**
                 * Separate line and column numbers from a URL-like string.
                 * @param urlLike String
                 * @return Array[String]
                 */
                extractLocation: function ErrorStackParser$$extractLocation(urlLike) {
                    var locationParts = urlLike.split(':');
                    var lastNumber = locationParts.pop();
                    var possibleNumber = locationParts[locationParts.length - 1];
                    if (!isNaN(parseFloat(possibleNumber)) && isFinite(possibleNumber)) {
                        var lineNumber = locationParts.pop();
                        return [locationParts.join(':'), lineNumber, lastNumber];
                    } else {
                        return [locationParts.join(':'), lastNumber, undefined];
                    }
                },

                parseV8OrIE: function ErrorStackParser$$parseV8OrIE(error) {
                    return error.stack.split('\n').slice(1).map(function (line) {
                        var tokens = line.replace(/^\s+/, '').split(/\s+/).slice(1);
                        var locationParts = this.extractLocation(tokens.pop().replace(/[\(\)\s]/g, ''));
                        var functionName = (!tokens[0] || tokens[0] === 'Anonymous') ? undefined : tokens[0];
                        return new StackFrame(functionName, undefined, locationParts[0], locationParts[1], locationParts[2]);
                    }.bind(this));
                },

                parseFFOrSafari: function ErrorStackParser$$parseFFOrSafari(error) {
                    return error.stack.split('\n').filter(function (line) {
                        return !!line.match(FIREFOX_SAFARI_STACK_REGEXP);
                    }.bind(this)).map(function (line) {
                        var tokens = line.split('@');
                        var locationParts = this.extractLocation(tokens.pop());
                        var functionName = tokens.shift() || undefined;
                        return new StackFrame(functionName, undefined, locationParts[0], locationParts[1], locationParts[2]);
                    }.bind(this));
                },

                parseOpera: function ErrorStackParser$$parseOpera(e) {
                    if (!e.stacktrace || (e.message.indexOf('\n') > -1 &&
                        e.message.split('\n').length > e.stacktrace.split('\n').length)) {
                        return this.parseOpera9(e);
                    } else if (!e.stack) {
                        return this.parseOpera10a(e);
                    } else if (e.stacktrace.indexOf("called from line") < 0) {
                        return this.parseOpera10b(e);
                    } else {
                        return this.parseOpera11(e);
                    }
                },

                parseOpera9: function ErrorStackParser$$parseOpera9(e) {
                    var lineRE = /Line (\d+).*script (?:in )?(\S+)/i;
                    var lines = e.message.split('\n');
                    var result = [];

                    for (var i = 2, len = lines.length; i < len; i += 2) {
                        var match = lineRE.exec(lines[i]);
                        if (match) {
                            result.push(new StackFrame(undefined, undefined, match[2], match[1]));
                        }
                    }

                    return result;
                },

                parseOpera10a: function ErrorStackParser$$parseOpera10a(e) {
                    var lineRE = /Line (\d+).*script (?:in )?(\S+)(?:: In function (\S+))?$/i;
                    var lines = e.stacktrace.split('\n');
                    var result = [];

                    for (var i = 0, len = lines.length; i < len; i += 2) {
                        var match = lineRE.exec(lines[i]);
                        if (match) {
                            result.push(new StackFrame(match[3] || undefined, undefined, match[2], match[1]));
                        }
                    }

                    return result;
                },

                // Opera 10.65+ Error.stack very similar to FF/Safari
                parseOpera11: function ErrorStackParser$$parseOpera11(error) {
                    return error.stack.split('\n').filter(function (line) {
                        return !!line.match(FIREFOX_SAFARI_STACK_REGEXP);
                    }.bind(this)).map(function (line) {
                        var tokens = line.split('@');
                        var locationParts = this.extractLocation(tokens.pop());
                        var functionCall = (tokens.shift() || '');
                        var functionName = functionCall.replace(/<anonymous function: (\w+)>/, '$1').replace(/\([^\)]*\)/, '') || undefined;
                        var argsRaw = functionCall.replace(/^[^\(]+\(([^\)]*)\)$/, '$1') || undefined;
                        var args = (argsRaw === undefined || argsRaw === '[arguments not available]') ? undefined : argsRaw.split(',');
                        return new StackFrame(functionName, args, locationParts[0], locationParts[1], locationParts[2]);
                    }.bind(this));
                }
            }
        };
        export var ErrorStackParser = factory(lib.utils.stacktrace.StackFrame);
    }
}