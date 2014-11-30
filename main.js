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
                        switch (type) {
                            case 0 /* Debug */:
                            case 1 /* Trace */:
                                console.info(msg);
                                break;
                            case 2 /* Warn */:
                                console.warn(msg);
                                break;
                            case 3 /* Error */:
                            case 4 /* Fatal */:
                            default:
                                debugger;
                                console.error(msg);
                                break;
                        }
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
                    debugger;
                    if (msg) {
                        logger.error('FAIL: ' + msg);
                    }
                    else {
                        logger.error('FAIL: ');
                    }
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
        var assert;
        (function (assert) {
            var _assert = lib.utils.detail.assert;
            function ok(cond, msg) {
                return _assert(cond, msg);
            }
            assert.ok = ok;
            function fail(cond, msg) {
                return _assert(!cond, msg);
            }
            assert.fail = fail;
            function strictEqual(a, b, msg) {
                return _assert(a === b, msg);
            }
            assert.strictEqual = strictEqual;
            function notStrictEqual(a, b, msg) {
                return fail(a === b, msg);
            }
            assert.notStrictEqual = notStrictEqual;
            function deepEqual(a, b, msg) {
                ok(a === b, msg);
                var aprops = Object.getOwnPropertyNames(a);
                var bprops = Object.getOwnPropertyNames(b);
                ok(aprops === bprops, msg);
                aprops.forEach(function (prop) {
                    ok(a.hasOwnProperty(prop), msg);
                    ok(b.hasOwnProperty(prop), msg);
                    deepEqual(a[prop], b[prop], msg);
                });
            }
            assert.deepEqual = deepEqual;
        })(assert = utils.assert || (utils.assert = {}));
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
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        function freeze(o) {
            Object.freeze(o);
            Object.getOwnPropertyNames(o).forEach(function (prop) {
                if (o.hasOwnProperty(prop) && o[prop] !== null && (typeof o[prop] === "object" || typeof o[prop] === "function") && !Object.isFrozen(o[prop])) {
                    freeze(o[prop]);
                }
            });
            return o;
        }
        utils.freeze = freeze;
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
/// <reference path="../../utils/utils.ts" />
/// < reference path="shared.ts" />
/// < reference path="equiv.ts" />
/// < reference path="path.ts" />
/// < reference path="node-path.ts" />
/// < reference path="path-visitor.ts" />
/// < reference path="scope.ts" />
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var types;
        (function (_types) {
            var assert = lib.utils.assert;
            var Ap = Array.prototype;
            var slice = Ap.slice;
            var map = Ap.map;
            var each = Ap.forEach;
            var Op = Object.prototype;
            var objToStr = Op.toString;
            var funObjStr = objToStr.call(function () {
            });
            var strObjStr = objToStr.call("");
            var hasOwn = Op.hasOwnProperty;
            // A type is an object with a .check method that takes a value and returns
            // true or false according to whether the value matches the type.
            var Type = (function () {
                function Type(check, name) {
                    var self = this;
                    assert.ok(self instanceof Type);
                    // Unfortunately we can't elegantly reuse isFunction and isString,
                    // here, because this code is executed while defining those types.
                    assert.strictEqual(objToStr.call(check), funObjStr, check + " is not a function");
                    // The `name` parameter can be either a function or a string.
                    var nameObjStr = objToStr.call(name);
                    assert.ok(nameObjStr === funObjStr || nameObjStr === strObjStr, name + " is neither a function nor a string");
                    Object.defineProperties(self, {
                        name: { value: name },
                        check: {
                            value: function (value, deep) {
                                var result = check.call(self, value, deep);
                                if (!result && deep && objToStr.call(deep) === funObjStr)
                                    deep(self, value);
                                return result;
                            }
                        }
                    });
                }
                // Like .check, except that failure triggers an AssertionError.
                Type.prototype.assert = function (value, deep) {
                    if (!this.check(value, deep)) {
                        var str = shallowStringify(value);
                        assert.ok(false, str + " does not match type " + this);
                        return false;
                    }
                    return true;
                };
                Type.prototype.toString = function () {
                    var name = this.name;
                    if (isString.check(name))
                        return name;
                    if (isFunction.check(name))
                        return name.call(this) + "";
                    return name + " type";
                };
                // Returns a type that matches the given value iff any of type1, type2,
                // etc. match the value.
                Type.or = function () {
                    var args = [];
                    for (var _i = 0; _i < arguments.length; _i++) {
                        args[_i - 0] = arguments[_i];
                    }
                    var types = [];
                    var len = arguments.length;
                    for (var i = 0; i < len; ++i)
                        types.push(toType(args[i]));
                    return new Type(function (value, deep) {
                        for (var i = 0; i < len; ++i)
                            if (types[i].check(value, deep))
                                return true;
                        return false;
                    }, function () {
                        return types.join(" | ");
                    });
                };
                Type.fromArray = function (arr) {
                    assert.ok(isArray.check(arr));
                    assert.strictEqual(arr.length, 1, "only one element type is permitted for typed arrays");
                    return toType(arr[0]).arrayOf();
                };
                Type.prototype.arrayOf = function () {
                    var elemType = this;
                    return new Type(function (value, deep) {
                        return isArray.check(value) && value.every(function (elem) {
                            return elemType.check(elem, deep);
                        });
                    }, function () {
                        return "[" + elemType + "]";
                    });
                };
                Type.fromObject = function (obj) {
                    var fields = Object.keys(obj).map(function (name) {
                        return new Field(name, obj[name]);
                    });
                    return new Type(function (value, deep) {
                        return isObject.check(value) && fields.every(function (field) {
                            return field.type.check(value[field.name], deep);
                        });
                    }, function () {
                        return "{ " + fields.join(", ") + " }";
                    });
                };
                // Define a type whose name is registered in a namespace (the defCache) so
                // that future definitions will return the same type given the same name.
                // In particular, this system allows for circular and forward definitions.
                // The Def object d returned from Type.def may be used to configure the
                // type d.type by calling methods such as d.bases, d.build, and d.field.
                Type.def = function (typeName) {
                    isString.assert(typeName);
                    return hasOwn.call(defCache, typeName) ? defCache[typeName] : defCache[typeName] = new Def(typeName);
                };
                return Type;
            })();
            _types.Type = Type;
            _types.builtInTypes = {};
            function defBuiltInType(example, name) {
                var objStr = objToStr.call(example);
                Object.defineProperty(_types.builtInTypes, name, {
                    enumerable: true,
                    value: new Type(function (value) {
                        return objToStr.call(value) === objStr;
                    }, name)
                });
                return _types.builtInTypes[name];
            }
            // These types check the underlying [[Class]] attribute of the given
            // value, rather than using the problematic typeof operator. Note however
            // that no subtyping is considered; so, for instance, isObject.check
            // returns false for [], /./, new Date, and null.
            var isString = defBuiltInType("", "string");
            var isFunction = defBuiltInType(function () {
            }, "function");
            var isArray = defBuiltInType([], "array");
            var isObject = defBuiltInType({}, "object");
            var isRegExp = defBuiltInType(/./, "RegExp");
            var isDate = defBuiltInType(new Date, "Date");
            var isNumber = defBuiltInType(3, "number");
            var isBoolean = defBuiltInType(true, "boolean");
            var isNull = defBuiltInType(null, "null");
            var isUndefined = defBuiltInType(void 0, "undefined");
            // There are a number of idiomatic ways of expressing types, so this
            // function serves to coerce them all to actual Type objects. Note that
            // providing the name argument is not necessary in most cases.
            function toType(from, name) {
                // The toType function should of course be idempotent.
                if (from instanceof Type)
                    return from;
                // The Def type is used as a helper for constructing compound
                // interface types for AST nodes.
                if (from instanceof Def)
                    return from.type;
                // Support [ElemType] syntax.
                if (isArray.check(from))
                    return Type.fromArray(from);
                // Support { someField: FieldType, ... } syntax.
                if (isObject.check(from))
                    return Type.fromObject(from);
                // If isFunction.check(from), assume that from is a binary predicate
                // function we can use to define the type.
                if (isFunction.check(from))
                    return new Type(from, name);
                // As a last resort, toType returns a type that matches any value that
                // is === from. This is primarily useful for literal values like
                // toType(null), but it has the additional advantage of allowing
                // toType to be a total function.
                return new Type(function (value) {
                    return value === from;
                }, isUndefined.check(name) ? function () {
                    return from + "";
                } : name);
            }
            var Field = (function () {
                function Field(name, type, defaultFn, hidden) {
                    var self = this;
                    assert.ok(self instanceof Field);
                    isString.assert(name);
                    type = toType(type);
                    if (isFunction.check(defaultFn)) {
                        this.defaultFn = defaultFn;
                    }
                    this.name = name;
                    this.type = type;
                    this.hidden = !!hidden;
                }
                Field.prototype.toString = function () {
                    return JSON.stringify(this.name) + ": " + this.type;
                };
                Field.prototype.getValue = function (obj) {
                    var value = this.name;
                    if (!isUndefined.check(value))
                        return value;
                    if (this.defaultFn && this.defaultFn.value)
                        value = this.defaultFn.value.call(obj);
                    return value;
                };
                return Field;
            })();
            function shallowStringify(value) {
                if (isObject.check(value))
                    return "{" + Object.keys(value).map(function (key) {
                        return key + ": " + value[key];
                    }).join(", ") + "}";
                if (isArray.check(value))
                    return "[" + value.map(shallowStringify).join(", ") + "]";
                return JSON.stringify(value);
            }
            // In order to return the same Def instance every time Type.def is called
            // with a particular name, those instances need to be stored in a cache.
            var defCache = Object.create(null);
            var Def = (function () {
                function Def(typeName) {
                    this.finalized = false;
                    this.buildable = false;
                    var self = this;
                    assert.ok(self instanceof Def);
                    Object.defineProperties(self, {
                        typeName: { value: typeName },
                        baseNames: { value: [] },
                        ownFields: { value: Object.create(null) },
                        // These two are populated during finalization.
                        allSupertypes: { value: Object.create(null) },
                        supertypeList: { value: [] },
                        allFields: { value: Object.create(null) },
                        fieldNames: { value: [] },
                        type: {
                            value: new Type(function (value, deep) {
                                return self.check(value, deep);
                            }, typeName)
                        }
                    });
                }
                Def.fromValue = function (value) {
                    if (value && typeof value === "object") {
                        var type = value.type;
                        if (typeof type === "string" && hasOwn.call(defCache, type)) {
                            var d = defCache[type];
                            if (d.finalized) {
                                return d;
                            }
                        }
                    }
                    return null;
                };
                Def.prototype.isSupertypeOf = function (that) {
                    if (that instanceof Def) {
                        assert.strictEqual(this.finalized, true);
                        assert.strictEqual(that.finalized, true);
                        return hasOwn.call(that.allSupertypes, this.typeName);
                    }
                    else {
                        assert.ok(false, that + " is not a Def");
                    }
                };
                Def.prototype.checkAllFields = function (value, deep) {
                    var allFields = this.allFields;
                    assert.strictEqual(this.finalized, true);
                    function checkFieldByName(name) {
                        var field = allFields[name];
                        var type = field.type;
                        var child = field.getValue(value);
                        return type.check(child, deep);
                    }
                    return isObject.check(value) && Object.keys(allFields).every(checkFieldByName);
                };
                Def.prototype.check = function (value, deep) {
                    assert.strictEqual(this.finalized, true, "prematurely checking unfinalized type " + this.typeName);
                    // A Def type can only match an object value.
                    if (!isObject.check(value))
                        return false;
                    var vDef = Def.fromValue(value);
                    if (!vDef) {
                        // If we couldn't infer the Def associated with the given value,
                        // and we expected it to be a SourceLocation or a Position, it was
                        // probably just missing a "type" field (because Esprima does not
                        // assign a type property to such nodes). Be optimistic and let
                        // this.checkAllFields make the final decision.
                        if (this.typeName === "SourceLocation" || this.typeName === "Position") {
                            return this.checkAllFields(value, deep);
                        }
                        // Calling this.checkAllFields for any other type of node is both
                        // bad for performance and way too forgiving.
                        return false;
                    }
                    // If checking deeply and vDef === this, then we only need to call
                    // checkAllFields once. Calling checkAllFields is too strict when deep
                    // is false, because then we only care about this.isSupertypeOf(vDef).
                    if (deep && vDef === this)
                        return this.checkAllFields(value, deep);
                    // In most cases we rely exclusively on isSupertypeOf to make O(1)
                    // subtyping determinations. This suffices in most situations outside
                    // of unit tests, since interface conformance is checked whenever new
                    // instances are created using builder functions.
                    if (!this.isSupertypeOf(vDef))
                        return false;
                    // The exception is when deep is true; then, we recursively check all
                    // fields.
                    if (!deep)
                        return true;
                    // Use the more specific Def (vDef) to perform the deep check, but
                    // shallow-check fields defined by the less specific Def (this).
                    return vDef.checkAllFields(value, deep) && this.checkAllFields(value, false);
                };
                Def.prototype.bases = function () {
                    var args = [];
                    for (var _i = 0; _i < arguments.length; _i++) {
                        args[_i - 0] = arguments[_i];
                    }
                    var bases = this.baseNames;
                    assert.strictEqual(this.finalized, false);
                    each.call(args, function (baseName) {
                        isString.assert(baseName);
                        // This indexOf lookup may be O(n), but the typical number of base
                        // names is very small, and indexOf is a native Array method.
                        if (bases.indexOf(baseName) < 0)
                            bases.push(baseName);
                    });
                    return this; // For chaining.
                };
                Def.prototype.finalize = function () {
                    // It's not an error to finalize a type more than once, but only the
                    // first call to .finalize does anything.
                    if (!this.finalized) {
                        var allFields = this.allFields;
                        var allSupertypes = this.allSupertypes;
                        this.baseNames.forEach(function (name) {
                            var def = defCache[name];
                            if (lib.utils.isUndefined(def)) {
                                return;
                            }
                            def.finalize();
                            extend(allFields, def.allFields);
                            extend(allSupertypes, def.allSupertypes);
                        });
                        // TODO Warn if fields are overridden with incompatible types.
                        extend(allFields, this.ownFields);
                        allSupertypes[this.typeName] = this;
                        this.fieldNames.length = 0;
                        for (var fieldName in allFields) {
                            if (hasOwn.call(allFields, fieldName) && !allFields[fieldName].hidden) {
                                this.fieldNames.push(fieldName);
                            }
                        }
                        // Types are exported only once they have been finalized.
                        Object.defineProperty(_types.namedTypes, this.typeName, {
                            enumerable: true,
                            value: this.type
                        });
                        Object.defineProperty(this, "finalized", { value: true });
                        // A linearization of the inheritance hierarchy.
                        populateSupertypeList(this.typeName, this.supertypeList);
                    }
                };
                // Calling the .build method of a Def simultaneously marks the type as
                // buildable (by defining builders[getBuilderName(typeName)]) and
                // specifies the order of arguments that should be passed to the builder
                // function to create an instance of the type.
                Def.prototype.build = function () {
                    var args = [];
                    for (var _i = 0; _i < arguments.length; _i++) {
                        args[_i - 0] = arguments[_i];
                    }
                    var self = this;
                    // Calling Def.prototype.build multiple times has the effect of merely
                    // redefining this property.
                    Object.defineProperty(self, "buildParams", {
                        value: slice.call(args),
                        writable: false,
                        enumerable: false,
                        configurable: true
                    });
                    assert.strictEqual(self.finalized, false);
                    isString.arrayOf().assert(self.buildParams);
                    if (self.buildable) {
                        // If this Def is already buildable, update self.buildParams and
                        // continue using the old builder function.
                        return self;
                    }
                    // Every buildable type will have its "type" field filled in
                    // automatically. This includes types that are not subtypes of Node,
                    // like SourceLocation, but that seems harmless (TODO?).
                    self.field("type", self.typeName, function () {
                        return self.typeName;
                    });
                    // Override Dp.buildable for this Def instance.
                    Object.defineProperty(self, "buildable", { value: true });
                    Object.defineProperty(_types.builders, getBuilderName(self.typeName), {
                        enumerable: true,
                        value: function () {
                            var args = [];
                            for (var _i = 0; _i < arguments.length; _i++) {
                                args[_i - 0] = arguments[_i];
                            }
                            var argc = args.length;
                            var built = Object.create(nodePrototype);
                            assert.ok(self.finalized, "attempting to instantiate unfinalized type " + self.typeName);
                            function add(param, i) {
                                if (hasOwn.call(built, param))
                                    return;
                                var all = self.allFields;
                                assert.ok(hasOwn.call(all, param), param);
                                var field = all[param];
                                var type = field.type;
                                var value;
                                if (isNumber.check(i) && i < argc) {
                                    value = args[i];
                                }
                                else if (!lib.utils.isUndefined(field.defaultFn)) {
                                    // Expose the partially-built object to the default
                                    // function as its `this` object.
                                    value = field.defaultFn.call(built);
                                }
                                else {
                                    var message = "no value or default function given for field " + JSON.stringify(param) + " of " + self.typeName + "(" + self.buildParams.map(function (name) {
                                        return all[name];
                                    }).join(", ") + ")";
                                    assert.ok(false, message);
                                }
                                if (!type.check(value)) {
                                    assert.ok(false, shallowStringify(value) + " does not match field " + field + " of type " + self.typeName);
                                }
                                // TODO Could attach getters and setters here to enforce
                                // dynamic type safety.
                                built[param] = value;
                            }
                            self.buildParams.forEach(function (param, i) {
                                add(param, i);
                            });
                            Object.keys(self.allFields).forEach(function (param) {
                                add(param); // Use the default value.
                            });
                            // Make sure that the "type" field was filled automatically.
                            assert.strictEqual(built.type, self.typeName);
                            return built;
                        }
                    });
                    return self; // For chaining.
                };
                // The reason fields are specified using .field(...) instead of an object
                // literal syntax is somewhat subtle: the object literal syntax would
                // support only one key and one value, but with .field(...) we can pass
                // any number of arguments to specify the field.
                Def.prototype.field = function (name, type, defaultFn, hidden) {
                    assert.strictEqual(this.finalized, false);
                    this.ownFields[name] = new Field(name, type, defaultFn, hidden);
                    return this; // For chaining.
                };
                return Def;
            })();
            _types.Def = Def;
            // Note that the list returned by this function is a copy of the internal
            // supertypeList, *without* the typeName itself as the first element.
            function getSupertypeNames(typeName) {
                assert.ok(hasOwn.call(defCache, typeName));
                var d = defCache[typeName];
                assert.strictEqual(d.finalized, true);
                return d.supertypeList.slice(1);
            }
            _types.getSupertypeNames = getSupertypeNames;
            ;
            // Returns an object mapping from every known type in the defCache to the
            // most specific supertype whose name is an own property of the candidates
            // object.
            function computeSupertypeLookupTable(candidates) {
                var table = {};
                var typeNames = Object.keys(defCache);
                var typeNameCount = typeNames.length;
                for (var i = 0; i < typeNameCount; ++i) {
                    var typeName = typeNames[i];
                    var d = defCache[typeName];
                    assert.strictEqual(d.finalized, true);
                    for (var j = 0; j < d.supertypeList.length; ++j) {
                        var superTypeName = d.supertypeList[j];
                        if (hasOwn.call(candidates, superTypeName)) {
                            table[typeName] = superTypeName;
                            break;
                        }
                    }
                }
                return table;
            }
            _types.computeSupertypeLookupTable = computeSupertypeLookupTable;
            ;
            _types.builders = {};
            // This object is used as prototype for any node created by a builder.
            var nodePrototype = {};
            // Call this function to define a new method to be shared by all AST
            // nodes. The replaced method (if any) is returned for easy wrapping.
            function defineMethod(name, func) {
                var old = nodePrototype[name];
                // Pass undefined as func to delete nodePrototype[name].
                if (isUndefined.check(func)) {
                    delete nodePrototype[name];
                }
                else {
                    isFunction.assert(func);
                    Object.defineProperty(nodePrototype, name, {
                        enumerable: true,
                        configurable: true,
                        value: func
                    });
                }
                return old;
            }
            _types.defineMethod = defineMethod;
            ;
            function getBuilderName(typeName) {
                return typeName.replace(/^[A-Z]+/, function (upperCasePrefix) {
                    var len = upperCasePrefix.length;
                    switch (len) {
                        case 0:
                            return "";
                        case 1:
                            return upperCasePrefix.toLowerCase();
                        default:
                            // If there's more than one initial capital letter, lower-case
                            // all but the last one, so that XMLDefaultDeclaration (for
                            // example) becomes xmlDefaultDeclaration.
                            return upperCasePrefix.slice(0, len - 1).toLowerCase() + upperCasePrefix.charAt(len - 1);
                    }
                });
            }
            _types.namedTypes = {};
            // Like Object.keys, but aware of what fields each AST type should have.
            function getFieldNames(object) {
                var d = Def.fromValue(object);
                if (d) {
                    return d.fieldNames.slice(0);
                }
                if ("type" in object) {
                    assert.ok(false, "did not recognize object of type " + JSON.stringify(object.type));
                }
                return Object.keys(object);
            }
            _types.getFieldNames = getFieldNames;
            // Get the value of an object property, taking object.type and default
            // functions into account.
            function getFieldValue(object, fieldName) {
                var d = Def.fromValue(object);
                if (d) {
                    var field = d.allFields[fieldName];
                    if (field) {
                        return field.getValue(object);
                    }
                }
                return object[fieldName];
            }
            _types.getFieldValue = getFieldValue;
            // Iterate over all defined fields of an object, including those missing
            // or undefined, passing each field name and effective value (as returned
            // by getFieldValue) to the callback. If the object has no corresponding
            // Def, the callback will never be called.
            function eachField(object, callback, context) {
                getFieldNames(object).forEach(function (name) {
                    callback.call(this, name, getFieldValue(object, name));
                }, context);
            }
            _types.eachField = eachField;
            ;
            // Similar to eachField, except that iteration stops as soon as the
            // callback returns a truthy value. Like Array.prototype.some, the final
            // result is either true or false to indicates whether the callback
            // returned true for any element or not.
            function someField(object, callback, context) {
                return getFieldNames(object).some(function (name) {
                    return callback.call(this, name, getFieldValue(object, name));
                }, context);
            }
            _types.someField = someField;
            ;
            function populateSupertypeList(typeName, list) {
                list.length = 0;
                list.push(typeName);
                var lastSeen = Object.create(null);
                for (var pos = 0; pos < list.length; ++pos) {
                    typeName = list[pos];
                    var d = defCache[typeName];
                    assert.strictEqual(d.finalized, true);
                    // If we saw typeName earlier in the breadth-first traversal,
                    // delete the last-seen occurrence.
                    if (hasOwn.call(lastSeen, typeName)) {
                        delete list[lastSeen[typeName]];
                    }
                    // Record the new index of the last-seen occurrence of typeName.
                    lastSeen[typeName] = pos;
                    // Enqueue the base names of this type.
                    list.push.apply(list, d.baseNames);
                }
                for (var to = 0, from = to, len = list.length; from < len; ++from) {
                    if (hasOwn.call(list, from)) {
                        list[to++] = list[from];
                    }
                }
                list.length = to;
            }
            function extend(into, from) {
                Object.keys(from).forEach(function (name) {
                    into[name] = from[name];
                });
                return into;
            }
            ;
            function finalize() {
                Object.keys(defCache).forEach(function (name) {
                    var n = defCache[name];
                    if (!lib.utils.isUndefined(n)) {
                        n.finalize();
                    }
                });
            }
            _types.finalize = finalize;
            ;
            var shared;
            (function (shared) {
                var builtin = types.builtInTypes;
                var isNumber = builtin["number"];
                // An example of constructing a new type with arbitrary constraints from
                // an existing type.
                function geq(than) {
                    return new Type(function (value) {
                        return isNumber.check(value) && value >= than;
                    }, isNumber + " >= " + than);
                }
                shared.geq = geq;
                ;
                // Default value-returning functions that may optionally be passed as a
                // third argument to Def.prototype.field.
                shared.defaults = {
                    // Functions were used because (among other reasons) that's the most
                    // elegant way to allow for the emptyArray one always to give a new
                    // array instance.
                    "null": function () {
                        return null;
                    },
                    "emptyArray": function () {
                        return [];
                    },
                    "false": function () {
                        return false;
                    },
                    "true": function () {
                        return true;
                    },
                    "undefined": function () {
                    }
                };
                var naiveIsPrimitive = Type.or(builtin["string"], builtin["number"], builtin["boolean"], builtin["null"], builtin["undefined"]);
                shared.isPrimitive = new Type(function (value) {
                    if (value === null)
                        return true;
                    var type = typeof value;
                    return !(type === "object" || type === "function");
                }, naiveIsPrimitive.toString());
            })(shared = _types.shared || (_types.shared = {}));
            _types.geq = shared.geq;
            _types.isPrimitive = shared.isPrimitive;
            _types.defaults = shared.defaults;
        })(types = ast.types || (ast.types = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/*
 Copyright (c) 2014 Ben Newman <bn@cs.stanford.edu>

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
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
// from https://github.com/benjamn/private
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var recast;
        (function (recast) {
            var priv;
            (function (priv) {
                var originalObject = Object;
                var originalDefProp = Object.defineProperty;
                var originalCreate = Object.create;
                function defProp(obj, name, value) {
                    if (originalDefProp)
                        try {
                            originalDefProp.call(originalObject, obj, name, { value: value });
                        }
                        catch (definePropertyIsBrokenInIE8) {
                            obj[name] = value;
                        }
                    else {
                        obj[name] = value;
                    }
                }
                // For functions that will be invoked using .call or .apply, we need to
                // define those methods on the function objects themselves, rather than
                // inheriting them from Function.prototype, so that a malicious or clumsy
                // third party cannot interfere with the functionality of this module by
                // redefining Function.prototype.call or .apply.
                function makeSafeToCall(fun) {
                    if (fun) {
                        defProp(fun, "call", fun.call);
                        defProp(fun, "apply", fun.apply);
                    }
                    return fun;
                }
                makeSafeToCall(originalDefProp);
                makeSafeToCall(originalCreate);
                var hasOwn = makeSafeToCall(Object.prototype.hasOwnProperty);
                var numToStr = makeSafeToCall(Number.prototype.toString);
                var strSlice = makeSafeToCall(String.prototype.slice);
                var cloner = function () {
                };
                function create(prototype) {
                    if (originalCreate) {
                        return originalCreate.call(originalObject, prototype);
                    }
                    cloner.prototype = prototype || null;
                    return new cloner;
                }
                var rand = Math.random;
                var uniqueKeys = create(null);
                function makeUniqueKey() {
                    do
                        var uniqueKey = internString(strSlice.call(numToStr.call(rand(), 36), 2));
                    while (hasOwn.call(uniqueKeys, uniqueKey));
                    return uniqueKeys[uniqueKey] = uniqueKey;
                }
                priv.makeUniqueKey = makeUniqueKey;
                function internString(str) {
                    var obj = {};
                    obj[str] = true;
                    return Object.keys(obj)[0];
                }
                // Object.getOwnPropertyNames is the only way to enumerate non-enumerable
                // properties, so if we wrap it to ignore our secret keys, there should be
                // no way (except guessing) to access those properties.
                var originalGetOPNs = Object.getOwnPropertyNames;
                Object.getOwnPropertyNames = function getOwnPropertyNames(object) {
                    for (var names = originalGetOPNs(object), src = 0, dst = 0, len = names.length; src < len; ++src) {
                        if (!hasOwn.call(uniqueKeys, names[src])) {
                            if (src > dst) {
                                names[dst] = names[src];
                            }
                            ++dst;
                        }
                    }
                    names.length = dst;
                    return names;
                };
                function defaultCreatorFn(object) {
                    return create(null);
                }
                function makeAccessor(secretCreatorFn) {
                    var brand = makeUniqueKey();
                    var passkey = create(null);
                    secretCreatorFn = secretCreatorFn || defaultCreatorFn;
                    function register(object) {
                        var secret; // Created lazily.
                        function vault(key, forget) {
                            // Only code that has access to the passkey can retrieve (or forget)
                            // the secret object.
                            if (key === passkey) {
                                return forget ? secret = null : secret || (secret = secretCreatorFn(object));
                            }
                        }
                        defProp(object, brand, vault);
                    }
                    function accessor(object) {
                        if (!hasOwn.call(object, brand))
                            register(object);
                        return object[brand](passkey);
                    }
                    return accessor;
                }
                priv.makeAccessor = makeAccessor;
            })(priv = recast.priv || (recast.priv = {}));
        })(recast = ast.recast || (ast.recast = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/// <reference path="recast.ts" />
/// <reference path="../../../Scripts/typings/esprima/esprima.d.ts" />
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var recast;
        (function (recast) {
            var defaults = {
                // If you want to use a different branch of esprima, or any other
                // module that supports a .parse function, pass that module object to
                // recast.parse as options.esprima.
                esprima: esprima,
                // Number of spaces the pretty-printer should use per tab for
                // indentation. If you do not pass this option explicitly, it will be
                // (quite reliably!) inferred from the original code.
                tabWidth: 4,
                // If you really want the pretty-printer to use tabs instead of
                // spaces, make this option true.
                useTabs: false,
                // The reprinting code leaves leading whitespace untouched unless it
                // has to reindent a line, or you pass false for this option.
                reuseWhitespace: true,
                // Some of the pretty-printer code (such as that for printing function
                // parameter lists) makes a valiant attempt to prevent really long
                // lines. You can adjust the limit by changing this option; however,
                // there is no guarantee that line length will fit inside this limit.
                wrapColumn: 74,
                // Pass a string as options.sourceFileName to recast.parse to tell the
                // reprinter to keep track of reused code so that it can construct a
                // source map automatically.
                sourceFileName: null,
                // Pass a string as options.sourceMapName to recast.print, and
                // (provided you passed options.sourceFileName earlier) the
                // PrintResult of recast.print will have a .map property for the
                // generated source map.
                sourceMapName: null,
                // If provided, this option will be passed along to the source map
                // generator as a root directory for relative source file paths.
                sourceRoot: null,
                // If you provide a source map that was generated from a previous call
                // to recast.print as options.inputSourceMap, the old source map will
                // be composed with the new source map.
                inputSourceMap: null,
                // If you want esprima to generate .range information (recast only
                // uses .loc internally), pass true for this option.
                range: false,
                // If you want esprima not to throw exceptions when it encounters
                // non-fatal errors, keep this option true.
                tolerant: true
            }, hasOwn = defaults.hasOwnProperty;
            // Copy options and fill in default values.
            function normalize(options) {
                options = options || defaults;
                function get(key) {
                    return hasOwn.call(options, key) ? options[key] : defaults[key];
                }
                return {
                    tabWidth: +get("tabWidth"),
                    useTabs: !!get("useTabs"),
                    reuseWhitespace: !!get("reuseWhitespace"),
                    wrapColumn: Math.max(get("wrapColumn"), 0),
                    sourceFileName: get("sourceFileName"),
                    sourceMapName: get("sourceMapName"),
                    sourceRoot: get("sourceRoot"),
                    inputSourceMap: get("inputSourceMap"),
                    esprima: get("esprima"),
                    range: get("range"),
                    tolerant: get("tolerant")
                };
            }
            recast.normalize = normalize;
        })(recast = ast.recast || (ast.recast = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/// <reference path="../types/types.ts" />
/// <reference path="private.ts" />
/// <reference path="options.ts" />
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var recast;
        (function (recast) {
            function print(node, options) {
                return new recast.Printer(options).print(node);
            }
            recast.print = print;
            function prettyPrint(node, options) {
                return new recast.Printer(options).printGenerically(node);
            }
            recast.prettyPrint = prettyPrint;
            function defaultWriteback(output) {
                console.log(output);
            }
            function run(code, transformer, options) {
                var writeback = options && options.writeback || defaultWriteback;
                transformer(recast.parse(code, options), function (node) {
                    writeback(print(node, options).code);
                });
            }
            recast.run = run;
        })(recast = ast.recast || (ast.recast = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/// <reference path="sourcemap/sourcemap.ts" />
/// <reference path="types/types.ts" />
/// <reference path="recast/recast.ts" />
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
/// <reference path="./lib/ref.ts" />
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
            var b = lib.ast.types.builders;
            b["identifier"]("foo");
        };
        Greeter.prototype.stop = function () {
            lib.utils.assert.ok(1 == 1, "test");
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
/// <reference path="recast.ts" />
var lib;
(function (lib) {
    var ast;
    (function (_ast) {
        var recast;
        (function (recast) {
            var types = lib.ast.types;
            var assert = lib.utils.assert;
            var n = types.namedTypes;
            var isArray = types.builtInTypes["array"];
            var isObject = types.builtInTypes["object"];
            var childNodesCacheKey = lib.ast.recast.priv.makeUniqueKey();
            // TODO Move a non-caching implementation of this function into ast-types,
            // and implement a caching wrapper function here.
            function getSortedChildNodes(node, resultArray) {
                if (!node) {
                    return;
                }
                if (resultArray) {
                    if (n["Node"].check(node)) {
                        for (var i = resultArray.length - 1; i >= 0; --i) {
                            if (recast.comparePos(resultArray[i].loc.end, node.loc.start) <= 0) {
                                break;
                            }
                        }
                        resultArray.splice(i + 1, 0, node);
                        return;
                    }
                }
                else if (node[childNodesCacheKey]) {
                    return node[childNodesCacheKey];
                }
                var names;
                if (isArray.check(node)) {
                    names = Object.keys(node);
                }
                else if (isObject.check(node)) {
                    names = types.getFieldNames(node);
                }
                else {
                    return;
                }
                if (!resultArray) {
                    Object.defineProperty(node, childNodesCacheKey, {
                        value: resultArray = [],
                        enumerable: false
                    });
                }
                for (var i = 0, nameCount = names.length; i < nameCount; ++i) {
                    getSortedChildNodes(node[names[i]], resultArray);
                }
                return resultArray;
            }
            // As efficiently as possible, decorate the comment object with
            // .precedingNode, .enclosingNode, and/or .followingNode properties, at
            // least one of which is guaranteed to be defined.
            function decorateComment(node, comment) {
                var childNodes = getSortedChildNodes(node);
                // Time to dust off the old binary search robes and wizard hat.
                var left = 0, right = childNodes.length;
                while (left < right) {
                    var middle = (left + right) >> 1;
                    var child = childNodes[middle];
                    if (recast.comparePos(child.loc.start, comment.loc.start) <= 0 && recast.comparePos(comment.loc.end, child.loc.end) <= 0) {
                        // The comment is completely contained by this child node.
                        decorateComment(comment.enclosingNode = child, comment);
                        return; // Abandon the binary search at this level.
                    }
                    if (recast.comparePos(child.loc.end, comment.loc.start) <= 0) {
                        // This child node falls completely before the comment.
                        // Because we will never consider this node or any nodes
                        // before it again, this node must be the closest preceding
                        // node we have encountered so far.
                        var precedingNode = child;
                        left = middle + 1;
                        continue;
                    }
                    if (recast.comparePos(comment.loc.end, child.loc.start) <= 0) {
                        // This child node falls completely after the comment.
                        // Because we will never consider this node or any nodes after
                        // it again, this node must be the closest following node we
                        // have encountered so far.
                        var followingNode = child;
                        right = middle;
                        continue;
                    }
                    throw new Error("Comment location overlaps with node location");
                }
                if (precedingNode) {
                    comment.precedingNode = precedingNode;
                }
                if (followingNode) {
                    comment.followingNode = followingNode;
                }
            }
            function add(ast, lines) {
                var comments = ast.comments;
                if (!isArray.check(comments)) {
                    return;
                }
                delete ast.comments;
                var tiesToBreak = [];
                comments.forEach(function (comment) {
                    comment.loc.lines = lines;
                    decorateComment(ast, comment);
                    var pn = comment.precedingNode;
                    var en = comment.enclosingNode;
                    var fn = comment.followingNode;
                    if (pn && fn) {
                        var tieCount = tiesToBreak.length;
                        if (tieCount > 0) {
                            var lastTie = tiesToBreak[tieCount - 1];
                            assert.strictEqual(lastTie.precedingNode === comment.precedingNode, lastTie.followingNode === comment.followingNode);
                            if (lastTie.followingNode !== comment.followingNode) {
                                breakTies(tiesToBreak, lines);
                            }
                        }
                        tiesToBreak.push(comment);
                    }
                    else if (pn) {
                        // No contest: we have a trailing comment.
                        breakTies(tiesToBreak, lines);
                        Comments.forNode(pn).addTrailing(comment);
                    }
                    else if (fn) {
                        // No contest: we have a leading comment.
                        breakTies(tiesToBreak, lines);
                        Comments.forNode(fn).addLeading(comment);
                    }
                    else if (en) {
                        // The enclosing node has no child nodes at all, so what we
                        // have here is a dangling comment, e.g. [/* crickets */].
                        breakTies(tiesToBreak, lines);
                        Comments.forNode(en).addDangling(comment);
                    }
                    else {
                        throw new Error("AST contains no nodes at all?");
                    }
                });
                breakTies(tiesToBreak, lines);
            }
            recast.add = add;
            ;
            function breakTies(tiesToBreak, lines) {
                var tieCount = tiesToBreak.length;
                if (tieCount === 0) {
                    return;
                }
                var pn = tiesToBreak[0].precedingNode;
                var fn = tiesToBreak[0].followingNode;
                var gapEndPos = fn.loc.start;
                for (var indexOfFirstLeadingComment = tieCount; indexOfFirstLeadingComment > 0; --indexOfFirstLeadingComment) {
                    var comment = tiesToBreak[indexOfFirstLeadingComment - 1];
                    assert.strictEqual(comment.precedingNode, pn);
                    assert.strictEqual(comment.followingNode, fn);
                    var gap = lines.sliceString(comment.loc.end, gapEndPos);
                    if (/\S/.test(gap)) {
                        break;
                    }
                    gapEndPos = comment.loc.start;
                }
                while (indexOfFirstLeadingComment <= tieCount && (comment = tiesToBreak[indexOfFirstLeadingComment]) && comment.type === "Line" && comment.loc.start.column > fn.loc.start.column) {
                    ++indexOfFirstLeadingComment;
                }
                tiesToBreak.forEach(function (comment, i) {
                    if (i < indexOfFirstLeadingComment) {
                        Comments.forNode(pn).addTrailing(comment);
                    }
                    else {
                        Comments.forNode(fn).addLeading(comment);
                    }
                });
                tiesToBreak.length = 0;
            }
            var Comments = (function () {
                function Comments() {
                    assert.ok(this instanceof Comments);
                    this.leading = [];
                    this.dangling = [];
                    this.trailing = [];
                }
                Comments.forNode = function (node) {
                    var comments = node.comments;
                    if (!comments) {
                        Object.defineProperty(node, "comments", {
                            value: comments = new Comments,
                            enumerable: false
                        });
                    }
                    return comments;
                };
                Comments.prototype.forEach = function (callback, context) {
                    this.leading.forEach(callback, context);
                    // this.dangling.forEach(callback, context);
                    this.trailing.forEach(callback, context);
                };
                Comments.prototype.addLeading = function (comment) {
                    this.leading.push(comment);
                };
                Comments.prototype.addDangling = function (comment) {
                    this.dangling.push(comment);
                };
                Comments.prototype.addTrailing = function (comment) {
                    comment.trailing = true;
                    if (comment.type === "Block") {
                        this.trailing.push(comment);
                    }
                    else {
                        this.leading.push(comment);
                    }
                };
                return Comments;
            })();
            recast.Comments = Comments;
            /**
             * @param {Object} options - Options object that configures printing.
             */
            function printLeadingComment(comment, options) {
                var loc = comment.loc;
                var lines = loc && loc.lines;
                var parts = [];
                if (comment.type === "Block") {
                    parts.push("/*", recast.fromString(comment.value, options), "*/");
                }
                else if (comment.type === "Line") {
                    parts.push("//", recast.fromString(comment.value, options));
                }
                else
                    assert.fail(comment.type);
                if (comment.trailing) {
                    // When we print trailing comments as leading comments, we don't
                    // want to bring any trailing spaces along.
                    parts.push("\n");
                }
                else if (lines instanceof recast.Lines) {
                    var trailingSpace = lines.slice(loc.end, lines.skipSpaces(loc.end));
                    if (trailingSpace.length === 1) {
                        // If the trailing space contains no newlines, then we want to
                        // preserve it exactly as we found it.
                        parts.push(trailingSpace);
                    }
                    else {
                        // If the trailing space contains newlines, then replace it
                        // with just that many newlines, with all other spaces removed.
                        parts.push(new Array(trailingSpace.length).join("\n"));
                    }
                }
                else {
                    parts.push("\n");
                }
                var marg = loc ? loc.start.column : 0;
                return recast.concat(parts).stripMargin(marg);
            }
            /**
             * @param {Object} options - Options object that configures printing.
             */
            function printTrailingComment(comment, options) {
                var loc = comment.loc;
                var lines = loc && loc.lines;
                var parts = [];
                if (lines instanceof recast.Lines) {
                    var fromPos = lines.skipSpaces(loc.start, true) || lines.firstPos();
                    var leadingSpace = lines.slice(fromPos, loc.start);
                    if (leadingSpace.length === 1) {
                        // If the leading space contains no newlines, then we want to
                        // preserve it exactly as we found it.
                        parts.push(leadingSpace);
                    }
                    else {
                        // If the leading space contains newlines, then replace it
                        // with just that many newlines, sans all other spaces.
                        parts.push(new Array(leadingSpace.length).join("\n"));
                    }
                }
                if (comment.type === "Block") {
                    parts.push("/*", recast.fromString(comment.value, options), "*/");
                }
                else if (comment.type === "Line") {
                    parts.push("//", recast.fromString(comment.value, options), "\n");
                }
                else
                    assert.fail(comment.type);
                return recast.concat(parts).stripMargin(loc ? loc.start.column : 0, true);
            }
            /**
             * @param {Object} options - Options object that configures printing.
             */
            function printComments(comments, innerLines, options) {
                if (innerLines) {
                    assert.ok(innerLines instanceof recast.Lines);
                }
                else {
                    innerLines = recast.fromString("");
                }
                if (!comments || !(comments.leading.length + comments.trailing.length)) {
                    return innerLines;
                }
                var parts = [];
                comments.leading.forEach(function (comment) {
                    parts.push(printLeadingComment(comment, options));
                });
                parts.push(innerLines);
                comments.trailing.forEach(function (comment) {
                    assert.strictEqual(comment.type, "Block");
                    parts.push(printTrailingComment(comment, options));
                });
                return recast.concat(parts);
            }
            recast.printComments = printComments;
        })(recast = _ast.recast || (_ast.recast = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/// <reference path="recast.ts" />
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var recast;
        (function (recast) {
            var types = lib.ast.types;
            var assert = lib.utils.assert;
            var n = types.namedTypes;
            var isArray = types.builtInTypes["array"];
            var isObject = types.builtInTypes["object"];
            var isString = types.builtInTypes["string"];
            var sourceMap = lib.ast.sourcemap;
            var secretKey = recast.priv.makeUniqueKey();
            // Goals:
            // 1. Minimize new string creation.
            // 2. Keep (de)identation O(lines) time.
            // 3. Permit negative indentations.
            // 4. Enforce immutability.
            // 5. No newline characters.
            function getSecret(lines) {
                return lines[secretKey];
            }
            var Lines = (function () {
                function Lines(infos, sourceFileName) {
                    assert.ok(this instanceof Lines);
                    assert.ok(infos.length > 0);
                    if (sourceFileName) {
                        isString.assert(sourceFileName);
                    }
                    else {
                        sourceFileName = null;
                    }
                    Object.defineProperty(this, secretKey, {
                        value: {
                            infos: infos,
                            mappings: [],
                            name: sourceFileName,
                            cachedSourceMap: null
                        }
                    });
                    if (sourceFileName) {
                        getSecret(this).mappings.push(new recast.Mapping(this, {
                            start: this.firstPos(),
                            end: this.lastPos()
                        }));
                    }
                }
                Object.defineProperty(Lines.prototype, "length", {
                    // These properties used to be assigned to each new object in the Lines
                    // constructor, but we can more efficiently stuff them into the secret and
                    // let these lazy accessors compute their values on-the-fly.
                    get: function () {
                        return getSecret(this).infos.length;
                    },
                    enumerable: true,
                    configurable: true
                });
                Object.defineProperty(Lines.prototype, "name", {
                    get: function () {
                        return getSecret(this).name;
                    },
                    enumerable: true,
                    configurable: true
                });
                Lines.prototype.toString = function (options) {
                    return this.sliceString(this.firstPos(), this.lastPos(), options);
                };
                Lines.prototype.getSourceMap = function (sourceMapName, sourceRoot) {
                    if (!sourceMapName) {
                        // Although we could make up a name or generate an anonymous
                        // source map, instead we assume that any consumer who does not
                        // provide a name does not actually want a source map.
                        return null;
                    }
                    var targetLines = this;
                    function updateJSON(json) {
                        json = json || {};
                        isString.assert(sourceMapName);
                        json.file = sourceMapName;
                        if (sourceRoot) {
                            isString.assert(sourceRoot);
                            json.sourceRoot = sourceRoot;
                        }
                        return json;
                    }
                    var secret = getSecret(targetLines);
                    if (secret.cachedSourceMap) {
                        // Since Lines objects are immutable, we can reuse any source map
                        // that was previously generated. Nevertheless, we return a new
                        // JSON object here to protect the cached source map from outside
                        // modification.
                        return updateJSON(secret.cachedSourceMap.toJSON());
                    }
                    var smg = new sourceMap.SourceMapGenerator(updateJSON());
                    var sourcesToContents = {};
                    secret.mappings.forEach(function (mapping) {
                        var sourceCursor = mapping.sourceLines.skipSpaces(mapping.sourceLoc.start) || mapping.sourceLines.lastPos();
                        var targetCursor = targetLines.skipSpaces(mapping.targetLoc.start) || targetLines.lastPos();
                        while (recast.comparePos(sourceCursor, mapping.sourceLoc.end) < 0 && recast.comparePos(targetCursor, mapping.targetLoc.end) < 0) {
                            var sourceChar = mapping.sourceLines.charAt(sourceCursor);
                            var targetChar = targetLines.charAt(targetCursor);
                            assert.strictEqual(sourceChar, targetChar);
                            var sourceName = mapping.sourceLines.name;
                            // Add mappings one character at a time for maximum resolution.
                            smg.addMapping({
                                source: sourceName,
                                original: {
                                    line: sourceCursor.line,
                                    column: sourceCursor.column
                                },
                                generated: {
                                    line: targetCursor.line,
                                    column: targetCursor.column
                                }
                            });
                            if (!hasOwn.call(sourcesToContents, sourceName)) {
                                var sourceContent = mapping.sourceLines.toString();
                                smg.setSourceContent(sourceName, sourceContent);
                                sourcesToContents[sourceName] = sourceContent;
                            }
                            targetLines.nextPos(targetCursor, true);
                            mapping.sourceLines.nextPos(sourceCursor, true);
                        }
                    });
                    secret.cachedSourceMap = smg;
                    return smg.toJSON();
                };
                Lines.prototype.bootstrapCharAt = function (pos) {
                    assert.strictEqual(typeof pos, "object");
                    assert.strictEqual(typeof pos.line, "number");
                    assert.strictEqual(typeof pos.column, "number");
                    var line = pos.line, column = pos.column, strings = this.toString().split("\n"), string = strings[line - 1];
                    if (typeof string === "undefined")
                        return "";
                    if (column === string.length && line < strings.length)
                        return "\n";
                    if (column >= string.length)
                        return "";
                    return string.charAt(column);
                };
                Lines.prototype.charAt = function (pos) {
                    assert.strictEqual(typeof pos, "object");
                    assert.strictEqual(typeof pos.line, "number");
                    assert.strictEqual(typeof pos.column, "number");
                    var line = pos.line, column = pos.column, secret = getSecret(this), infos = secret.infos, info = infos[line - 1], c = column;
                    if (typeof info === "undefined" || c < 0)
                        return "";
                    var indent = this.getIndentAt(line);
                    if (c < indent)
                        return " ";
                    c += info.sliceStart - indent;
                    if (c === info.sliceEnd && line < this.length)
                        return "\n";
                    if (c >= info.sliceEnd)
                        return "";
                    return info.line.charAt(c);
                };
                Lines.prototype.stripMargin = function (width, skipFirstLine) {
                    if (width === 0)
                        return this;
                    assert.ok(width > 0, "negative margin: " + width);
                    if (skipFirstLine && this.length === 1)
                        return this;
                    var secret = getSecret(this);
                    var lines = new Lines(secret.infos.map(function (info, i) {
                        if (info.line && (i > 0 || !skipFirstLine)) {
                            info = copyLineInfo(info);
                            info.indent = Math.max(0, info.indent - width);
                        }
                        return info;
                    }));
                    if (secret.mappings.length > 0) {
                        var newMappings = getSecret(lines).mappings;
                        assert.strictEqual(newMappings.length, 0);
                        secret.mappings.forEach(function (mapping) {
                            newMappings.push(mapping.indent(width, skipFirstLine, true));
                        });
                    }
                    return lines;
                };
                Lines.prototype.indent = function (by) {
                    if (by === 0)
                        return this;
                    var secret = getSecret(this);
                    var lines = new Lines(secret.infos.map(function (info) {
                        if (info.line) {
                            info = copyLineInfo(info);
                            info.indent += by;
                        }
                        return info;
                    }));
                    if (secret.mappings.length > 0) {
                        var newMappings = getSecret(lines).mappings;
                        assert.strictEqual(newMappings.length, 0);
                        secret.mappings.forEach(function (mapping) {
                            newMappings.push(mapping.indent(by));
                        });
                    }
                    return lines;
                };
                Lines.prototype.indentTail = function (by) {
                    if (by === 0)
                        return this;
                    if (this.length < 2)
                        return this;
                    var secret = getSecret(this);
                    var lines = new Lines(secret.infos.map(function (info, i) {
                        if (i > 0 && info.line) {
                            info = copyLineInfo(info);
                            info.indent += by;
                        }
                        return info;
                    }));
                    if (secret.mappings.length > 0) {
                        var newMappings = getSecret(lines).mappings;
                        assert.strictEqual(newMappings.length, 0);
                        secret.mappings.forEach(function (mapping) {
                            newMappings.push(mapping.indent(by, true));
                        });
                    }
                    return lines;
                };
                Lines.prototype.getIndentAt = function (line) {
                    assert.ok(line >= 1, "no line " + line + " (line numbers start from 1)");
                    var secret = getSecret(this), info = secret.infos[line - 1];
                    return Math.max(info.indent, 0);
                };
                Lines.prototype.guessTabWidth = function () {
                    var secret = getSecret(this);
                    if (hasOwn.call(secret, "cachedTabWidth")) {
                        return secret.cachedTabWidth;
                    }
                    var counts = []; // Sparse array.
                    var lastIndent = 0;
                    for (var line = 1, last = this.length; line <= last; ++line) {
                        var info = secret.infos[line - 1];
                        var sliced = info.line.slice(info.sliceStart, info.sliceEnd);
                        // Whitespace-only lines don't tell us much about the likely tab
                        // width of this code.
                        if (isOnlyWhitespace(sliced)) {
                            continue;
                        }
                        var diff = Math.abs(info.indent - lastIndent);
                        counts[diff] = ~~counts[diff] + 1;
                        lastIndent = info.indent;
                    }
                    var maxCount = -1;
                    var result = 2;
                    for (var tabWidth = 1; tabWidth < counts.length; tabWidth += 1) {
                        if (hasOwn.call(counts, tabWidth) && counts[tabWidth] > maxCount) {
                            maxCount = counts[tabWidth];
                            result = tabWidth;
                        }
                    }
                    return secret.cachedTabWidth = result;
                };
                Lines.prototype.isOnlyWhitespace = function () {
                    return isOnlyWhitespace(this.toString());
                };
                Lines.prototype.isPrecededOnlyByWhitespace = function (pos) {
                    var secret = getSecret(this);
                    var info = secret.infos[pos.line - 1];
                    var indent = Math.max(info.indent, 0);
                    var diff = pos.column - indent;
                    if (diff <= 0) {
                        // If pos.column does not exceed the indentation amount, then
                        // there must be only whitespace before it.
                        return true;
                    }
                    var start = info.sliceStart;
                    var end = Math.min(start + diff, info.sliceEnd);
                    var prefix = info.line.slice(start, end);
                    return isOnlyWhitespace(prefix);
                };
                Lines.prototype.getLineLength = function (line) {
                    var secret = getSecret(this), info = secret.infos[line - 1];
                    return this.getIndentAt(line) + info.sliceEnd - info.sliceStart;
                };
                Lines.prototype.nextPos = function (pos, skipSpaces) {
                    var l = Math.max(pos.line, 0), c = Math.max(pos.column, 0);
                    if (c < this.getLineLength(l)) {
                        pos.column += 1;
                        return skipSpaces ? !!this.skipSpaces(pos, false, true) : true;
                    }
                    if (l < this.length) {
                        pos.line += 1;
                        pos.column = 0;
                        return skipSpaces ? !!this.skipSpaces(pos, false, true) : true;
                    }
                    return false;
                };
                Lines.prototype.prevPos = function (pos, skipSpaces) {
                    var l = pos.line, c = pos.column;
                    if (c < 1) {
                        l -= 1;
                        if (l < 1)
                            return false;
                        c = this.getLineLength(l);
                    }
                    else {
                        c = Math.min(c - 1, this.getLineLength(l));
                    }
                    pos.line = l;
                    pos.column = c;
                    return skipSpaces ? !!this.skipSpaces(pos, true, true) : true;
                };
                Lines.prototype.firstPos = function () {
                    // Trivial, but provided for completeness.
                    return { line: 1, column: 0 };
                };
                Lines.prototype.lastPos = function () {
                    return {
                        line: this.length,
                        column: this.getLineLength(this.length)
                    };
                };
                Lines.prototype.skipSpaces = function (pos, backward, modifyInPlace) {
                    if (pos) {
                        pos = modifyInPlace ? pos : {
                            line: pos.line,
                            column: pos.column
                        };
                    }
                    else if (backward) {
                        pos = this.lastPos();
                    }
                    else {
                        pos = this.firstPos();
                    }
                    if (backward) {
                        while (this.prevPos(pos)) {
                            if (!isOnlyWhitespace(this.charAt(pos)) && this.nextPos(pos)) {
                                return pos;
                            }
                        }
                        return null;
                    }
                    else {
                        while (isOnlyWhitespace(this.charAt(pos))) {
                            if (!this.nextPos(pos)) {
                                return null;
                            }
                        }
                        return pos;
                    }
                };
                Lines.prototype.trimLeft = function () {
                    var pos = this.skipSpaces(this.firstPos(), false, true);
                    return pos ? this.slice(pos) : emptyLines;
                };
                Lines.prototype.trimRight = function () {
                    var pos = this.skipSpaces(this.lastPos(), true, true);
                    return pos ? this.slice(this.firstPos(), pos) : emptyLines;
                };
                Lines.prototype.trim = function () {
                    var start = this.skipSpaces(this.firstPos(), false, true);
                    if (start === null)
                        return emptyLines;
                    var end = this.skipSpaces(this.lastPos(), true, true);
                    assert.notStrictEqual(end, null);
                    return this.slice(start, end);
                };
                Lines.prototype.eachPos = function (callback, startPos, skipSpaces) {
                    var pos = this.firstPos();
                    if (startPos) {
                        pos.line = startPos.line, pos.column = startPos.column;
                    }
                    if (skipSpaces && !this.skipSpaces(pos, false, true)) {
                        return; // Encountered nothing but spaces.
                    }
                    do
                        callback.call(this, pos);
                    while (this.nextPos(pos, skipSpaces));
                };
                Lines.prototype.bootstrapSlice = function (start, end) {
                    var strings = this.toString().split("\n").slice(start.line - 1, end.line);
                    strings.push(strings.pop().slice(0, end.column));
                    strings[0] = strings[0].slice(start.column);
                    return fromString(strings.join("\n"));
                };
                Lines.prototype.slice = function (start, end) {
                    if (!end) {
                        if (!start) {
                            // The client seems to want a copy of this Lines object, but
                            // Lines objects are immutable, so it's perfectly adequate to
                            // return the same object.
                            return this;
                        }
                        // Slice to the end if no end position was provided.
                        end = this.lastPos();
                    }
                    var secret = getSecret(this);
                    var sliced = secret.infos.slice(start.line - 1, end.line);
                    if (start.line === end.line) {
                        sliced[0] = sliceInfo(sliced[0], start.column, end.column);
                    }
                    else {
                        assert.ok(start.line < end.line);
                        sliced[0] = sliceInfo(sliced[0], start.column);
                        sliced.push(sliceInfo(sliced.pop(), 0, end.column));
                    }
                    var lines = new Lines(sliced);
                    if (secret.mappings.length > 0) {
                        var newMappings = getSecret(lines).mappings;
                        assert.strictEqual(newMappings.length, 0);
                        secret.mappings.forEach(function (mapping) {
                            var sliced = mapping.slice(this, start, end);
                            if (sliced) {
                                newMappings.push(sliced);
                            }
                        }, this);
                    }
                    return lines;
                };
                Lines.prototype.bootstrapSliceString = function (start, end, options) {
                    return this.slice(start, end).toString(options);
                };
                Lines.prototype.sliceString = function (start, end, options) {
                    if (!end) {
                        if (!start) {
                            // The client seems to want a copy of this Lines object, but
                            // Lines objects are immutable, so it's perfectly adequate to
                            // return the same object.
                            return this;
                        }
                        // Slice to the end if no end position was provided.
                        end = this.lastPos();
                    }
                    options = recast.normalize(options);
                    var infos = getSecret(this).infos;
                    var parts = [];
                    var tabWidth = options.tabWidth;
                    for (var line = start.line; line <= end.line; ++line) {
                        var info = infos[line - 1];
                        if (line === start.line) {
                            if (line === end.line) {
                                info = sliceInfo(info, start.column, end.column);
                            }
                            else {
                                info = sliceInfo(info, start.column);
                            }
                        }
                        else if (line === end.line) {
                            info = sliceInfo(info, 0, end.column);
                        }
                        var indent = Math.max(info.indent, 0);
                        var before = info.line.slice(0, info.sliceStart);
                        if (options.reuseWhitespace && isOnlyWhitespace(before) && countSpaces(before, options.tabWidth) === indent) {
                            // Reuse original spaces if the indentation is correct.
                            parts.push(info.line.slice(0, info.sliceEnd));
                            continue;
                        }
                        var tabs = 0;
                        var spaces = indent;
                        if (options.useTabs) {
                            tabs = Math.floor(indent / tabWidth);
                            spaces -= tabs * tabWidth;
                        }
                        var result = "";
                        if (tabs > 0) {
                            result += new Array(tabs + 1).join("\t");
                        }
                        if (spaces > 0) {
                            result += new Array(spaces + 1).join(" ");
                        }
                        result += info.line.slice(info.sliceStart, info.sliceEnd);
                        parts.push(result);
                    }
                    return parts.join("\n");
                };
                Lines.prototype.isEmpty = function () {
                    return this.length < 2 && this.getLineLength(1) < 1;
                };
                Lines.prototype.join = function (elements) {
                    var separator = this;
                    var separatorSecret = getSecret(separator);
                    var infos = [];
                    var mappings = [];
                    var prevInfo;
                    function appendSecret(secret) {
                        if (secret === null)
                            return;
                        if (prevInfo) {
                            var info = secret.infos[0];
                            var indent = new Array(info.indent + 1).join(" ");
                            var prevLine = infos.length;
                            var prevColumn = Math.max(prevInfo.indent, 0) + prevInfo.sliceEnd - prevInfo.sliceStart;
                            prevInfo.line = prevInfo.line.slice(0, prevInfo.sliceEnd) + indent + info.line.slice(info.sliceStart, info.sliceEnd);
                            prevInfo.sliceEnd = prevInfo.line.length;
                            if (secret.mappings.length > 0) {
                                secret.mappings.forEach(function (mapping) {
                                    mappings.push(mapping.add(prevLine, prevColumn));
                                });
                            }
                        }
                        else if (secret.mappings.length > 0) {
                            mappings.push.apply(mappings, secret.mappings);
                        }
                        secret.infos.forEach(function (info, i) {
                            if (!prevInfo || i > 0) {
                                prevInfo = copyLineInfo(info);
                                infos.push(prevInfo);
                            }
                        });
                    }
                    function appendWithSeparator(secret, i) {
                        if (i > 0)
                            appendSecret(separatorSecret);
                        appendSecret(secret);
                    }
                    elements.map(function (elem) {
                        var lines = fromString(elem);
                        if (lines.isEmpty())
                            return null;
                        return getSecret(lines);
                    }).forEach(separator.isEmpty() ? appendSecret : appendWithSeparator);
                    if (infos.length < 1)
                        return emptyLines;
                    var lines = new Lines(infos);
                    getSecret(lines).mappings = mappings;
                    return lines;
                };
                Lines.prototype.concat = function (other) {
                    var args = arguments, list = [this];
                    list.push.apply(list, args);
                    assert.strictEqual(list.length, args.length + 1);
                    return emptyLines.join(list);
                };
                return Lines;
            })();
            recast.Lines = Lines;
            function sliceInfo(info, startCol, endCol) {
                var sliceStart = info.sliceStart;
                var sliceEnd = info.sliceEnd;
                var indent = Math.max(info.indent, 0);
                var lineLength = indent + sliceEnd - sliceStart;
                if (typeof endCol === "undefined") {
                    endCol = lineLength;
                }
                startCol = Math.max(startCol, 0);
                endCol = Math.min(endCol, lineLength);
                endCol = Math.max(endCol, startCol);
                if (endCol < indent) {
                    indent = endCol;
                    sliceEnd = sliceStart;
                }
                else {
                    sliceEnd -= lineLength - endCol;
                }
                lineLength = endCol;
                lineLength -= startCol;
                if (startCol < indent) {
                    indent -= startCol;
                }
                else {
                    startCol -= indent;
                    indent = 0;
                    sliceStart += startCol;
                }
                assert.ok(indent >= 0);
                assert.ok(sliceStart <= sliceEnd);
                assert.strictEqual(lineLength, indent + sliceEnd - sliceStart);
                if (info.indent === indent && info.sliceStart === sliceStart && info.sliceEnd === sliceEnd) {
                    return info;
                }
                return {
                    line: info.line,
                    indent: indent,
                    sliceStart: sliceStart,
                    sliceEnd: sliceEnd
                };
            }
            function concat(elements) {
                return emptyLines.join(elements);
            }
            recast.concat = concat;
            function copyLineInfo(info) {
                return {
                    line: info.line,
                    indent: info.indent,
                    sliceStart: info.sliceStart,
                    sliceEnd: info.sliceEnd
                };
            }
            var fromStringCache = {};
            var hasOwn = fromStringCache.hasOwnProperty;
            var maxCacheKeyLen = 10;
            function countSpaces(spaces, tabWidth) {
                var count = 0;
                var len = spaces.length;
                for (var i = 0; i < len; ++i) {
                    var ch = spaces.charAt(i);
                    if (ch === " ") {
                        count += 1;
                    }
                    else if (ch === "\t") {
                        assert.strictEqual(typeof tabWidth, "number");
                        assert.ok(tabWidth > 0);
                        var next = Math.ceil(count / tabWidth) * tabWidth;
                        if (next === count) {
                            count += tabWidth;
                        }
                        else {
                            count = next;
                        }
                    }
                    else if (ch === "\r") {
                    }
                    else {
                        assert.fail("unexpected whitespace character", ch);
                    }
                }
                return count;
            }
            recast.countSpaces = countSpaces;
            var leadingSpaceExp = /^\s*/;
            /**
             * @param {Object} options - Options object that configures printing.
             */
            function fromString(str, options) {
                if (str instanceof Lines)
                    return str;
                str += "";
                var tabWidth = options && options.tabWidth;
                var tabless = str.indexOf("\t") < 0;
                var cacheable = !options && tabless && (str.length <= maxCacheKeyLen);
                assert.ok(tabWidth || tabless, "No tab width specified but encountered tabs in str\n" + str);
                if (cacheable && hasOwn.call(fromStringCache, str))
                    return fromStringCache[str];
                var lines = new Lines(str.split("\n").map(function (line) {
                    var spaces = leadingSpaceExp.exec(line)[0];
                    return {
                        line: line,
                        indent: countSpaces(spaces, tabWidth),
                        sliceStart: spaces.length,
                        sliceEnd: line.length
                    };
                }), recast.normalize(options).sourceFileName);
                if (cacheable)
                    fromStringCache[str] = lines;
                return lines;
            }
            recast.fromString = fromString;
            function isOnlyWhitespace(string) {
                return !/\S/.test(string);
            }
            // The emptyLines object needs to be created all the way down here so that
            // Lines.prototype will be fully populated.
            var emptyLines = fromString("");
        })(recast = ast.recast || (ast.recast = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/// <reference path="recast.ts" />
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var recast;
        (function (recast) {
            var types = lib.ast.types;
            var assert = lib.utils.assert;
            var n = types.namedTypes;
            var isArray = types.builtInTypes["array"];
            var isObject = types.builtInTypes["object"];
            var isString = types.builtInTypes["string"];
            var isNumber = types.builtInTypes["number"];
            var SourceLocation = types.namedTypes["SourceLocation"];
            var Position = types.namedTypes["Position"];
            var Mapping = (function () {
                function Mapping(sourceLines, sourceLoc, targetLoc) {
                    assert.ok(this instanceof Mapping);
                    assert.ok(sourceLines instanceof recast.Lines);
                    SourceLocation.assert(sourceLoc);
                    if (targetLoc) {
                        // In certain cases it's possible for targetLoc.{start,end}.column
                        // values to be negative, which technically makes them no longer
                        // valid SourceLocation nodes, so we need to be more forgiving.
                        assert.ok(isNumber.check(targetLoc.start.line) && isNumber.check(targetLoc.start.column) && isNumber.check(targetLoc.end.line) && isNumber.check(targetLoc.end.column));
                    }
                    else {
                        // Assume identity mapping if no targetLoc specified.
                        targetLoc = sourceLoc;
                    }
                    Object.defineProperties(this, {
                        sourceLines: { value: sourceLines },
                        sourceLoc: { value: sourceLoc },
                        targetLoc: { value: targetLoc }
                    });
                }
                Mapping.prototype.slice = function (lines, start, end) {
                    assert.ok(lines instanceof recast.Lines);
                    Position.assert(start);
                    if (end) {
                        Position.assert(end);
                    }
                    else {
                        end = lines.lastPos();
                    }
                    var sourceLines = this.sourceLines;
                    var sourceLoc = this.sourceLoc;
                    var targetLoc = this.targetLoc;
                    function skip(name) {
                        var sourceFromPos = sourceLoc[name];
                        var targetFromPos = targetLoc[name];
                        var targetToPos = start;
                        if (name === "end") {
                            targetToPos = end;
                        }
                        else {
                            assert.strictEqual(name, "start");
                        }
                        return skipChars(sourceLines, sourceFromPos, lines, targetFromPos, targetToPos);
                    }
                    if (recast.comparePos(start, targetLoc.start) <= 0) {
                        if (recast.comparePos(targetLoc.end, end) <= 0) {
                            targetLoc = {
                                start: subtractPos(targetLoc.start, start.line, start.column),
                                end: subtractPos(targetLoc.end, start.line, start.column)
                            };
                        }
                        else if (recast.comparePos(end, targetLoc.start) <= 0) {
                            return null;
                        }
                        else {
                            sourceLoc = {
                                start: sourceLoc.start,
                                end: skip("end")
                            };
                            targetLoc = {
                                start: subtractPos(targetLoc.start, start.line, start.column),
                                end: subtractPos(end, start.line, start.column)
                            };
                        }
                    }
                    else {
                        if (recast.comparePos(targetLoc.end, start) <= 0) {
                            return null;
                        }
                        if (recast.comparePos(targetLoc.end, end) <= 0) {
                            sourceLoc = {
                                start: skip("start"),
                                end: sourceLoc.end
                            };
                            targetLoc = {
                                // Same as subtractPos(start, start.line, start.column):
                                start: { line: 1, column: 0 },
                                end: subtractPos(targetLoc.end, start.line, start.column)
                            };
                        }
                        else {
                            sourceLoc = {
                                start: skip("start"),
                                end: skip("end")
                            };
                            targetLoc = {
                                // Same as subtractPos(start, start.line, start.column):
                                start: { line: 1, column: 0 },
                                end: subtractPos(end, start.line, start.column)
                            };
                        }
                    }
                    return new Mapping(this.sourceLines, sourceLoc, targetLoc);
                };
                Mapping.prototype.add = function (line, column) {
                    return new Mapping(this.sourceLines, this.sourceLoc, {
                        start: addPos(this.targetLoc.start, line, column),
                        end: addPos(this.targetLoc.end, line, column)
                    });
                };
                Mapping.prototype.subtract = function (line, column) {
                    return new Mapping(this.sourceLines, this.sourceLoc, {
                        start: subtractPos(this.targetLoc.start, line, column),
                        end: subtractPos(this.targetLoc.end, line, column)
                    });
                };
                Mapping.prototype.indent = function (by, skipFirstLine, noNegativeColumns) {
                    if (by === 0) {
                        return this;
                    }
                    var targetLoc = this.targetLoc;
                    var startLine = targetLoc.start.line;
                    var endLine = targetLoc.end.line;
                    if (skipFirstLine && startLine === 1 && endLine === 1) {
                        return this;
                    }
                    targetLoc = {
                        start: targetLoc.start,
                        end: targetLoc.end
                    };
                    if (!skipFirstLine || startLine > 1) {
                        var startColumn = targetLoc.start.column + by;
                        targetLoc.start = {
                            line: startLine,
                            column: noNegativeColumns ? Math.max(0, startColumn) : startColumn
                        };
                    }
                    if (!skipFirstLine || endLine > 1) {
                        var endColumn = targetLoc.end.column + by;
                        targetLoc.end = {
                            line: endLine,
                            column: noNegativeColumns ? Math.max(0, endColumn) : endColumn
                        };
                    }
                    return new Mapping(this.sourceLines, this.sourceLoc, targetLoc);
                };
                return Mapping;
            })();
            recast.Mapping = Mapping;
            function addPos(toPos, line, column) {
                return {
                    line: toPos.line + line - 1,
                    column: (toPos.line === 1) ? toPos.column + column : toPos.column
                };
            }
            function subtractPos(fromPos, line, column) {
                return {
                    line: fromPos.line - line + 1,
                    column: (fromPos.line === line) ? fromPos.column - column : fromPos.column
                };
            }
            function skipChars(sourceLines, sourceFromPos, targetLines, targetFromPos, targetToPos) {
                assert.ok(sourceLines instanceof recast.Lines);
                assert.ok(targetLines instanceof recast.Lines);
                Position.assert(sourceFromPos);
                Position.assert(targetFromPos);
                Position.assert(targetToPos);
                var targetComparison = recast.comparePos(targetFromPos, targetToPos);
                if (targetComparison === 0) {
                    // Trivial case: no characters to skip.
                    return sourceFromPos;
                }
                if (targetComparison < 0) {
                    // Skipping forward.
                    var sourceCursor = sourceLines.skipSpaces(sourceFromPos);
                    var targetCursor = targetLines.skipSpaces(targetFromPos);
                    var lineDiff = targetToPos.line - targetCursor.line;
                    sourceCursor.line += lineDiff;
                    targetCursor.line += lineDiff;
                    if (lineDiff > 0) {
                        // If jumping to later lines, reset columns to the beginnings
                        // of those lines.
                        sourceCursor.column = 0;
                        targetCursor.column = 0;
                    }
                    else {
                        assert.strictEqual(lineDiff, 0);
                    }
                    while (recast.comparePos(targetCursor, targetToPos) < 0 && targetLines.nextPos(targetCursor, true)) {
                        assert.ok(sourceLines.nextPos(sourceCursor, true));
                        assert.strictEqual(sourceLines.charAt(sourceCursor), targetLines.charAt(targetCursor));
                    }
                }
                else {
                    // Skipping backward.
                    var sourceCursor = sourceLines.skipSpaces(sourceFromPos, true);
                    var targetCursor = targetLines.skipSpaces(targetFromPos, true);
                    var lineDiff = targetToPos.line - targetCursor.line;
                    sourceCursor.line += lineDiff;
                    targetCursor.line += lineDiff;
                    if (lineDiff < 0) {
                        // If jumping to earlier lines, reset columns to the ends of
                        // those lines.
                        sourceCursor.column = sourceLines.getLineLength(sourceCursor.line);
                        targetCursor.column = targetLines.getLineLength(targetCursor.line);
                    }
                    else {
                        assert.strictEqual(lineDiff, 0);
                    }
                    while (recast.comparePos(targetToPos, targetCursor) < 0 && targetLines.prevPos(targetCursor, true)) {
                        assert.ok(sourceLines.prevPos(sourceCursor, true));
                        assert.strictEqual(sourceLines.charAt(sourceCursor), targetLines.charAt(targetCursor));
                    }
                }
                return sourceCursor;
            }
        })(recast = ast.recast || (ast.recast = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var recast;
        (function (recast) {
            var types = lib.ast.types;
            var assert = lib.utils.assert;
            var n = types.namedTypes;
            var isArray = types.builtInTypes["array"];
            var isObject = types.builtInTypes["object"];
            var isString = types.builtInTypes["string"];
            var isFunction = types.builtInTypes["function"];
            var b = types.builders;
            function parse(source, options) {
                options = recast.normalize(options);
                var lines = recast.fromString(source, options);
                var sourceWithoutTabs = lines.toString({
                    tabWidth: options.tabWidth,
                    reuseWhitespace: false,
                    useTabs: false
                });
                var pure = options.esprima.parse(sourceWithoutTabs, {
                    loc: true,
                    range: options.range,
                    comment: true,
                    tolerant: options.tolerant
                });
                recast.add(pure, lines);
                // In order to ensure we reprint leading and trailing program
                // comments, wrap the original Program node with a File node.
                pure = b["file"](pure);
                pure.loc = {
                    lines: lines,
                    indent: 0,
                    start: lines.firstPos(),
                    end: lines.lastPos()
                };
                // Return a copy of the original AST so that any changes made may be
                // compared to the original.
                return new TreeCopier(lines).copy(pure);
            }
            recast.parse = parse;
            ;
            var TreeCopier = (function () {
                function TreeCopier(lines) {
                    assert.ok(this instanceof TreeCopier);
                    this.lines = lines;
                    this.indent = 0;
                }
                TreeCopier.prototype.copy = function (node) {
                    if (isArray.check(node)) {
                        return node.map(this.copy, this);
                    }
                    if (!isObject.check(node)) {
                        return node;
                    }
                    if ((n["MethodDefinition"] && n["MethodDefinition"].check(node)) || (n["Property"].check(node) && (node["method"] || node["shorthand"]))) {
                        // If the node is a MethodDefinition or a .method or .shorthand
                        // Property, then the location information stored in
                        // node.value.loc is very likely untrustworthy (just the {body}
                        // part of a method, or nothing in the case of shorthand
                        // properties), so we null out that information to prevent
                        // accidental reuse of bogus source code during reprinting.
                        node.value.loc = null;
                    }
                    var copy = Object.create(Object.getPrototypeOf(node), {
                        original: {
                            value: node,
                            configurable: false,
                            enumerable: false,
                            writable: true
                        }
                    });
                    var loc = node.loc;
                    var oldIndent = this.indent;
                    var newIndent = oldIndent;
                    if (loc) {
                        if (loc.start.line < 1) {
                            loc.start.line = 1;
                        }
                        if (loc.end.line < 1) {
                            loc.end.line = 1;
                        }
                        if (this.lines.isPrecededOnlyByWhitespace(loc.start)) {
                            newIndent = this.indent = loc.start.column;
                        }
                        loc.lines = this.lines;
                        loc.indent = newIndent;
                    }
                    var keys = Object.keys(node);
                    var keyCount = keys.length;
                    for (var i = 0; i < keyCount; ++i) {
                        var key = keys[i];
                        if (key === "loc") {
                            copy[key] = node[key];
                        }
                        else if (key === "comments") {
                        }
                        else {
                            copy[key] = this.copy(node[key]);
                        }
                    }
                    this.indent = oldIndent;
                    if (node.comments) {
                        Object.defineProperty(copy, "comments", {
                            value: node.comments,
                            enumerable: false
                        });
                    }
                    return copy;
                };
                return TreeCopier;
            })();
            recast.TreeCopier = TreeCopier;
        })(recast = ast.recast || (ast.recast = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/// <reference path="recast.ts" />
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var recast;
        (function (recast) {
            var types = lib.ast.types;
            var assert = lib.utils.assert;
            var NodePath = types.NodePath;
            var n = types.namedTypes;
            var isArray = types.builtInTypes["array"];
            var isObject = types.builtInTypes["object"];
            var isString = types.builtInTypes["string"];
            var isFunction = types.builtInTypes["function"];
            var b = types.builders;
            var Node = types.namedTypes["Node"];
            var Expression = types.namedTypes["Expression"];
            var SourceLocation = types.namedTypes["SourceLocation"];
            function Patcher(lines) {
                assert.ok(this instanceof Patcher);
                assert.ok(lines instanceof recast.Lines);
                var self = this, replacements = [];
                self.replace = function (loc, lines) {
                    if (isString.check(lines))
                        lines = recast.fromString(lines);
                    replacements.push({
                        lines: lines,
                        start: loc.start,
                        end: loc.end
                    });
                };
                self.get = function (loc) {
                    // If no location is provided, return the complete Lines object.
                    loc = loc || {
                        start: { line: 1, column: 0 },
                        end: {
                            line: lines.length,
                            column: lines.getLineLength(lines.length)
                        }
                    };
                    var sliceFrom = loc.start, toConcat = [];
                    function pushSlice(from, to) {
                        assert.ok(recast.comparePos(from, to) <= 0);
                        toConcat.push(lines.slice(from, to));
                    }
                    replacements.sort(function (a, b) {
                        return recast.comparePos(a.start, b.start);
                    }).forEach(function (rep) {
                        if (recast.comparePos(sliceFrom, rep.start) > 0) {
                        }
                        else {
                            pushSlice(sliceFrom, rep.start);
                            toConcat.push(rep.lines);
                            sliceFrom = rep.end;
                        }
                    });
                    pushSlice(sliceFrom, loc.end);
                    return recast.concat(toConcat);
                };
            }
            recast.Patcher = Patcher;
            function getReprinter(path) {
                assert.ok(path instanceof NodePath);
                // Make sure that this path refers specifically to a Node, rather than
                // some non-Node subproperty of a Node.
                var node = path.value;
                if (!Node.check(node))
                    return;
                var orig = node.original;
                var origLoc = orig && orig.loc;
                var lines = origLoc && origLoc.lines;
                var reprints = [];
                if (!lines || !findReprints(path, reprints))
                    return;
                return function (print) {
                    var patcher = new Patcher(lines);
                    reprints.forEach(function (reprint) {
                        var old = reprint.oldPath.value;
                        SourceLocation.assert(old.loc, true);
                        patcher.replace(old.loc, print(reprint.newPath).indentTail(old.loc.indent));
                    });
                    return patcher.get(origLoc).indentTail(-orig.loc.indent);
                };
            }
            recast.getReprinter = getReprinter;
            ;
            function findReprints(newPath, reprints) {
                var newNode = newPath.value;
                Node.assert(newNode);
                var oldNode = newNode.original;
                Node.assert(oldNode);
                assert.deepEqual(reprints, []);
                if (newNode.type !== oldNode.type) {
                    return false;
                }
                var oldPath = new NodePath(oldNode);
                var canReprint = findChildReprints(newPath, oldPath, reprints);
                if (!canReprint) {
                    // Make absolutely sure the calling code does not attempt to reprint
                    // any nodes.
                    reprints.length = 0;
                }
                return canReprint;
            }
            function findAnyReprints(newPath, oldPath, reprints) {
                var newNode = newPath.value;
                var oldNode = oldPath.value;
                if (newNode === oldNode)
                    return true;
                if (isArray.check(newNode))
                    return findArrayReprints(newPath, oldPath, reprints);
                if (isObject.check(newNode))
                    return findObjectReprints(newPath, oldPath, reprints);
                return false;
            }
            function findArrayReprints(newPath, oldPath, reprints) {
                var newNode = newPath.value;
                var oldNode = oldPath.value;
                isArray.assert(newNode);
                var len = newNode.length;
                if (!(isArray.check(oldNode) && oldNode.length === len))
                    return false;
                for (var i = 0; i < len; ++i)
                    if (!findAnyReprints(newPath.get(i), oldPath.get(i), reprints))
                        return false;
                return true;
            }
            function findObjectReprints(newPath, oldPath, reprints) {
                var newNode = newPath.value;
                isObject.assert(newNode);
                if (newNode.original === null) {
                    // If newNode.original node was set to null, reprint the node.
                    return false;
                }
                var oldNode = oldPath.value;
                if (!isObject.check(oldNode))
                    return false;
                if (Node.check(newNode)) {
                    if (!Node.check(oldNode)) {
                        return false;
                    }
                    if (!oldNode.loc) {
                        // If we have no .loc information for oldNode, then we won't
                        // be able to reprint it.
                        return false;
                    }
                    // Here we need to decide whether the reprinted code for newNode
                    // is appropriate for patching into the location of oldNode.
                    if (newNode.type === oldNode.type) {
                        var childReprints = [];
                        if (findChildReprints(newPath, oldPath, childReprints)) {
                            reprints.push.apply(reprints, childReprints);
                        }
                        else {
                            reprints.push({
                                newPath: newPath,
                                oldPath: oldPath
                            });
                        }
                        return true;
                    }
                    if (Expression.check(newNode) && Expression.check(oldNode)) {
                        // If both nodes are subtypes of Expression, then we should be
                        // able to fill the location occupied by the old node with
                        // code printed for the new node with no ill consequences.
                        reprints.push({
                            newPath: newPath,
                            oldPath: oldPath
                        });
                        return true;
                    }
                    // The nodes have different types, and at least one of the types
                    // is not a subtype of the Expression type, so we cannot safely
                    // assume the nodes are syntactically interchangeable.
                    return false;
                }
                return findChildReprints(newPath, oldPath, reprints);
            }
            // This object is reused in hasOpeningParen and hasClosingParen to avoid
            // having to allocate a temporary object.
            var reusablePos = { line: 1, column: 0 };
            function hasOpeningParen(oldPath) {
                var oldNode = oldPath.value;
                var loc = oldNode.loc;
                var lines = loc && loc.lines;
                if (lines) {
                    var pos = reusablePos;
                    pos.line = loc.start.line;
                    pos.column = loc.start.column;
                    while (lines.prevPos(pos)) {
                        var ch = lines.charAt(pos);
                        if (ch === "(") {
                            var rootPath = oldPath;
                            while (rootPath.parentPath)
                                rootPath = rootPath.parentPath;
                            // If we found an opening parenthesis but it occurred before
                            // the start of the original subtree for this reprinting, then
                            // we must not return true for hasOpeningParen(oldPath).
                            return recast.comparePos(rootPath.value.loc.start, pos) <= 0;
                        }
                        if (ch !== " ") {
                            return false;
                        }
                    }
                }
                return false;
            }
            function hasClosingParen(oldPath) {
                var oldNode = oldPath.value;
                var loc = oldNode.loc;
                var lines = loc && loc.lines;
                if (lines) {
                    var pos = reusablePos;
                    pos.line = loc.end.line;
                    pos.column = loc.end.column;
                    do {
                        var ch = lines.charAt(pos);
                        if (ch === ")") {
                            var rootPath = oldPath;
                            while (rootPath.parentPath)
                                rootPath = rootPath.parentPath;
                            // If we found a closing parenthesis but it occurred after the
                            // end of the original subtree for this reprinting, then we
                            // must not return true for hasClosingParen(oldPath).
                            return recast.comparePos(pos, rootPath.value.loc.end) <= 0;
                        }
                        if (ch !== " ") {
                            return false;
                        }
                    } while (lines.nextPos(pos));
                }
                return false;
            }
            function hasParens(oldPath) {
                // This logic can technically be fooled if the node has parentheses
                // but there are comments intervening between the parentheses and the
                // node. In such cases the node will be harmlessly wrapped in an
                // additional layer of parentheses.
                return hasOpeningParen(oldPath) && hasClosingParen(oldPath);
            }
            function findChildReprints(newPath, oldPath, reprints) {
                var newNode = newPath.value;
                var oldNode = oldPath.value;
                isObject.assert(newNode);
                isObject.assert(oldNode);
                if (newNode.original === null) {
                    // If newNode.original node was set to null, reprint the node.
                    return false;
                }
                // If this type of node cannot come lexically first in its enclosing
                // statement (e.g. a function expression or object literal), and it
                // seems to be doing so, then the only way we can ignore this problem
                // and save ourselves from falling back to the pretty printer is if an
                // opening parenthesis happens to precede the node.  For example,
                // (function(){ ... }()); does not need to be reprinted, even though
                // the FunctionExpression comes lexically first in the enclosing
                // ExpressionStatement and fails the hasParens test, because the
                // parent CallExpression passes the hasParens test. If we relied on
                // the path.needsParens() && !hasParens(oldNode) check below, the
                // absence of a closing parenthesis after the FunctionExpression would
                // trigger pretty-printing unnecessarily.
                if (!newPath.canBeFirstInStatement() && newPath.firstInStatement() && !hasOpeningParen(oldPath))
                    return false;
                // If this node needs parentheses and will not be wrapped with
                // parentheses when reprinted, then return false to skip reprinting
                // and let it be printed generically.
                if (newPath.needsParens(true) && !hasParens(oldPath)) {
                    return false;
                }
                for (var k in recast.getUnionOfKeys(newNode, oldNode)) {
                    if (k === "loc")
                        continue;
                    if (!findAnyReprints(newPath.get(k), oldPath.get(k), reprints))
                        return false;
                }
                return true;
            }
        })(recast = ast.recast || (ast.recast = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/// <reference path="recast.ts" />
var lib;
(function (lib) {
    var ast;
    (function (_ast) {
        var recast;
        (function (recast) {
            var types = lib.ast.types;
            var assert = lib.utils.assert;
            var namedTypes = types.namedTypes;
            var isString = types.builtInTypes["string"];
            var isObject = types.builtInTypes["object"];
            var NodePath = types.NodePath;
            var PrintResult = (function () {
                function PrintResult(code, sourceMap) {
                    assert.ok(this instanceof PrintResult);
                    isString.assert(code);
                    this.code = code;
                    if (sourceMap) {
                        isObject.assert(sourceMap);
                        this.map = sourceMap;
                    }
                }
                return PrintResult;
            })();
            recast.PrintResult = PrintResult;
            var PRp = PrintResult.prototype;
            var warnedAboutToString = false;
            PRp.toString = function () {
                if (!warnedAboutToString) {
                    console.warn("Deprecation warning: recast.print now returns an object with " + "a .code property. You appear to be treating the object as a " + "string, which might still work but is strongly discouraged.");
                    warnedAboutToString = true;
                }
                return this.code;
            };
            var emptyPrintResult = new PrintResult("");
            function Printer(originalOptions) {
                assert.ok(this instanceof Printer);
                var explicitTabWidth = originalOptions && originalOptions.tabWidth;
                var options = recast.normalize(originalOptions);
                assert.notStrictEqual(options, originalOptions);
                // It's common for client code to pass the same options into both
                // recast.parse and recast.print, but the Printer doesn't need (and
                // can be confused by) options.sourceFileName, so we null it out.
                options.sourceFileName = null;
                function printWithComments(path) {
                    assert.ok(path instanceof NodePath);
                    return recast.printComments(path.node.comments, print(path), options);
                }
                function print(path, includeComments) {
                    if (includeComments)
                        return printWithComments(path);
                    assert.ok(path instanceof NodePath);
                    if (!explicitTabWidth) {
                        var oldTabWidth = options.tabWidth;
                        var orig = path.node.original;
                        var origLoc = orig && orig.loc;
                        var origLines = origLoc && origLoc.lines;
                        if (origLines) {
                            options.tabWidth = origLines.guessTabWidth();
                            var lines = maybeReprint(path);
                            options.tabWidth = oldTabWidth;
                            return lines;
                        }
                    }
                    return maybeReprint(path);
                }
                function maybeReprint(path) {
                    var reprinter = recast.getReprinter(path);
                    if (reprinter)
                        return maybeAddParens(path, reprinter(maybeReprint));
                    return printRootGenerically(path);
                }
                // Print the root node generically, but then resume reprinting its
                // children non-generically.
                function printRootGenerically(path) {
                    return genericPrint(path, options, printWithComments);
                }
                // Print the entire AST generically.
                function printGenerically(path) {
                    return genericPrint(path, options, printGenerically);
                }
                this.print = function (ast) {
                    if (!ast) {
                        return emptyPrintResult;
                    }
                    var path = ast instanceof NodePath ? ast : new NodePath(ast);
                    var lines = print(path, true);
                    return new PrintResult(lines.toString(options), recast.composeSourceMaps(options.inputSourceMap, lines.getSourceMap(options.sourceMapName, options.sourceRoot)));
                };
                this.printGenerically = function (ast) {
                    if (!ast) {
                        return emptyPrintResult;
                    }
                    var path = ast instanceof NodePath ? ast : new NodePath(ast);
                    var oldReuseWhitespace = options.reuseWhitespace;
                    // Do not reuse whitespace (or anything else, for that matter)
                    // when printing generically.
                    options.reuseWhitespace = false;
                    var pr = new PrintResult(printGenerically(path).toString(options));
                    options.reuseWhitespace = oldReuseWhitespace;
                    return pr;
                };
            }
            recast.Printer = Printer;
            function maybeAddParens(path, lines) {
                return path.needsParens() ? recast.concat(["(", lines, ")"]) : lines;
            }
            function genericPrint(path, options, printPath) {
                assert.ok(path instanceof NodePath);
                return maybeAddParens(path, genericPrintNoParens(path, options, printPath));
            }
            function genericPrintNoParens(path, options, print) {
                var n = path.value;
                if (!n) {
                    return recast.fromString("");
                }
                if (typeof n === "string") {
                    return recast.fromString(n, options);
                }
                namedTypes["Node"].assert(n);
                switch (n.type) {
                    case "File":
                        path = path.get("program");
                        n = path.node;
                        namedTypes["Program"].assert(n);
                    case "Program":
                        return maybeAddSemicolon(printStatementSequence(path.get("body"), options, print));
                    case "EmptyStatement":
                        return recast.fromString("");
                    case "ExpressionStatement":
                        return recast.concat([print(path.get("expression")), ";"]);
                    case "BinaryExpression":
                    case "LogicalExpression":
                    case "AssignmentExpression":
                        return recast.fromString(" ").join([
                            print(path.get("left")),
                            n.operator,
                            print(path.get("right"))
                        ]);
                    case "MemberExpression":
                        {
                            var parts = [print(path.get("object"))];
                            if (n.computed)
                                parts.push("[", print(path.get("property")), "]");
                            else
                                parts.push(".", print(path.get("property")));
                            return recast.concat(parts);
                        }
                    case "Path":
                        return recast.fromString(".").join(n.body);
                    case "Identifier":
                        return recast.fromString(n.name, options);
                    case "SpreadElement":
                    case "SpreadElementPattern":
                    case "SpreadProperty":
                    case "SpreadPropertyPattern":
                        return recast.concat(["...", print(path.get("argument"))]);
                    case "FunctionDeclaration":
                    case "FunctionExpression":
                        {
                            var parts = [];
                            if (n.async)
                                parts.push("async ");
                            parts.push("function");
                            if (n.generator)
                                parts.push("*");
                            if (n.id)
                                parts.push(" ", print(path.get("id")));
                            parts.push("(", printFunctionParams(path, options, print), ") ", print(path.get("body")));
                            return recast.concat(parts);
                        }
                    case "ArrowFunctionExpression":
                        {
                            var parts = [];
                            if (n.async)
                                parts.push("async ");
                            if (n.params.length === 1) {
                                parts.push(print(path.get("params", 0)));
                            }
                            else {
                                parts.push("(", printFunctionParams(path, options, print), ")");
                            }
                            parts.push(" => ", print(path.get("body")));
                            return recast.concat(parts);
                        }
                    case "MethodDefinition":
                        {
                            var parts = [];
                            if (n.static) {
                                parts.push("static ");
                            }
                            parts.push(printMethod(n.kind, path.get("key"), path.get("value"), options, print));
                            return recast.concat(parts);
                        }
                    case "YieldExpression":
                        {
                            var parts = ["yield"];
                            if (n.delegate)
                                parts.push("*");
                            if (n.argument)
                                parts.push(" ", print(path.get("argument")));
                            return recast.concat(parts);
                        }
                    case "AwaitExpression":
                        {
                            var parts = ["await"];
                            if (n.all)
                                parts.push("*");
                            if (n.argument)
                                parts.push(" ", print(path.get("argument")));
                            return recast.concat(parts);
                        }
                    case "ModuleDeclaration":
                        var parts = ["module", print(path.get("id"))];
                        if (n.source) {
                            assert.ok(!n.body);
                            parts.push("from", print(path.get("source")));
                        }
                        else {
                            parts.push(print(path.get("body")));
                        }
                        return recast.fromString(" ").join(parts);
                    case "ImportSpecifier":
                    case "ExportSpecifier":
                        var parts = [print(path.get("id"))];
                        if (n.name)
                            parts.push(" as ", print(path.get("name")));
                        return recast.concat(parts);
                    case "ExportBatchSpecifier":
                        return recast.fromString("*");
                    case "ImportNamespaceSpecifier":
                        return recast.concat(["* as ", print(path.get("id"))]);
                    case "ImportDefaultSpecifier":
                        return print(path.get("id"));
                    case "ExportDeclaration":
                        var parts = ["export"];
                        if (n["default"]) {
                            parts.push(" default");
                        }
                        else if (n.specifiers && n.specifiers.length > 0) {
                            if (n.specifiers.length === 1 && n.specifiers[0].type === "ExportBatchSpecifier") {
                                parts.push(" *");
                            }
                            else {
                                parts.push(" { ", recast.fromString(", ").join(path.get("specifiers").map(print)), " }");
                            }
                            if (n.source)
                                parts.push(" from ", print(path.get("source")));
                            parts.push(";");
                            return recast.concat(parts);
                        }
                        if (n.declaration) {
                            if (!namedTypes["Node"].check(n.declaration)) {
                                console.log(JSON.stringify(n, null, 2));
                            }
                            var decLines = print(path.get("declaration"));
                            parts.push(" ", decLines);
                            if (lastNonSpaceCharacter(decLines) !== ";") {
                                parts.push(";");
                            }
                        }
                        return recast.concat(parts);
                    case "ImportDeclaration":
                        var parts = ["import "];
                        if (n.specifiers && n.specifiers.length > 0) {
                            var foundImportSpecifier = false;
                            path.get("specifiers").each(function (sp) {
                                if (sp.name > 0) {
                                    parts.push(", ");
                                }
                                if (namedTypes["ImportDefaultSpecifier"].check(sp.value) || namedTypes["ImportNamespaceSpecifier"].check(sp.value)) {
                                    assert.strictEqual(foundImportSpecifier, false);
                                }
                                else {
                                    namedTypes["ImportSpecifier"].assert(sp.value);
                                    if (!foundImportSpecifier) {
                                        foundImportSpecifier = true;
                                        parts.push("{");
                                    }
                                }
                                parts.push(print(sp));
                            });
                            if (foundImportSpecifier) {
                                parts.push("}");
                            }
                            parts.push(" from ");
                        }
                        parts.push(print(path.get("source")), ";");
                        return recast.concat(parts);
                    case "BlockStatement":
                        var naked = printStatementSequence(path.get("body"), options, print);
                        if (naked.isEmpty())
                            return recast.fromString("{}");
                        return recast.concat([
                            "{\n",
                            naked.indent(options.tabWidth),
                            "\n}"
                        ]);
                    case "ReturnStatement":
                        var parts = ["return"];
                        if (n.argument) {
                            var argLines = print(path.get("argument"));
                            if (argLines.length > 1 && namedTypes["XJSElement"] && namedTypes["XJSElement"].check(n.argument)) {
                                parts.push(" (\n", argLines.indent(options.tabWidth), "\n)");
                            }
                            else {
                                parts.push(" ", argLines);
                            }
                        }
                        parts.push(";");
                        return recast.concat(parts);
                    case "CallExpression":
                        return recast.concat([
                            print(path.get("callee")),
                            printArgumentsList(path, options, print)
                        ]);
                    case "ObjectExpression":
                    case "ObjectPattern":
                        {
                            var allowBreak = false, len = n.properties.length, parts = [len > 0 ? "{\n" : "{"];
                            path.get("properties").map(function (childPath) {
                                var prop = childPath.value;
                                var i = childPath.name;
                                var lines = print(childPath).indent(options.tabWidth);
                                var multiLine = lines.length > 1;
                                if (multiLine && allowBreak) {
                                    // Similar to the logic for BlockStatement.
                                    parts.push("\n");
                                }
                                parts.push(lines);
                                if (i < len - 1) {
                                    // Add an extra line break if the previous object property
                                    // had a multi-line value.
                                    parts.push(multiLine ? ",\n\n" : ",\n");
                                    allowBreak = !multiLine;
                                }
                            });
                            parts.push(len > 0 ? "\n}" : "}");
                            return recast.concat(parts);
                        }
                    case "PropertyPattern":
                        return recast.concat([
                            print(path.get("key")),
                            ": ",
                            print(path.get("pattern"))
                        ]);
                    case "Property":
                        if (n.method || n.kind === "get" || n.kind === "set") {
                            return printMethod(n.kind, path.get("key"), path.get("value"), options, print);
                        }
                        if (path.node.shorthand) {
                            return print(path.get("key"));
                        }
                        else {
                            return recast.concat([
                                print(path.get("key")),
                                ": ",
                                print(path.get("value"))
                            ]);
                        }
                    case "ArrayExpression":
                    case "ArrayPattern":
                        var elems = n.elements, len = elems.length, parts = ["["];
                        path.get("elements").each(function (elemPath) {
                            var elem = elemPath.value;
                            if (!elem) {
                                // If the array expression ends with a hole, that hole
                                // will be ignored by the interpreter, but if it ends with
                                // two (or more) holes, we need to write out two (or more)
                                // commas so that the resulting code is interpreted with
                                // both (all) of the holes.
                                parts.push(",");
                            }
                            else {
                                var i = elemPath.name;
                                if (i > 0)
                                    parts.push(" ");
                                parts.push(print(elemPath));
                                if (i < len - 1)
                                    parts.push(",");
                            }
                        });
                        parts.push("]");
                        return recast.concat(parts);
                    case "SequenceExpression":
                        return recast.fromString(", ").join(path.get("expressions").map(print));
                    case "ThisExpression":
                        return recast.fromString("this");
                    case "Literal":
                        if (typeof n.value !== "string")
                            return recast.fromString(n.value, options);
                    case "ModuleSpecifier":
                        // A ModuleSpecifier is a string-valued Literal.
                        return recast.fromString(nodeStr(n), options);
                    case "UnaryExpression":
                        var parts = [n.operator];
                        if (/[a-z]$/.test(n.operator))
                            parts.push(" ");
                        parts.push(print(path.get("argument")));
                        return recast.concat(parts);
                    case "UpdateExpression":
                        var parts = [
                            print(path.get("argument")),
                            n.operator
                        ];
                        if (n.prefix)
                            parts.reverse();
                        return recast.concat(parts);
                    case "ConditionalExpression":
                        return recast.concat([
                            "(",
                            print(path.get("test")),
                            " ? ",
                            print(path.get("consequent")),
                            " : ",
                            print(path.get("alternate")),
                            ")"
                        ]);
                    case "NewExpression":
                        var parts = ["new ", print(path.get("callee"))];
                        var args = n.arguments;
                        if (args) {
                            parts.push(printArgumentsList(path, options, print));
                        }
                        return recast.concat(parts);
                    case "VariableDeclaration":
                        var parts = [n.kind, " "];
                        var maxLen = 0;
                        var printed = path.get("declarations").map(function (childPath) {
                            var lines = print(childPath);
                            maxLen = Math.max(lines.length, maxLen);
                            return lines;
                        });
                        if (maxLen === 1) {
                            parts.push(recast.fromString(", ").join(printed));
                        }
                        else if (printed.length > 1) {
                            parts.push(recast.fromString(",\n").join(printed).indentTail(n.kind.length + 1));
                        }
                        else {
                            parts.push(printed[0]);
                        }
                        // We generally want to terminate all variable declarations with a
                        // semicolon, except when they are children of for loops.
                        var parentNode = path.parent && path.parent.node;
                        if (!namedTypes["ForStatement"].check(parentNode) && !namedTypes["ForStatement"].check(parentNode) && !(namedTypes["ForStatement"] && namedTypes["ForOfStatement"].check(parentNode))) {
                            parts.push(";");
                        }
                        return recast.concat(parts);
                    case "VariableDeclarator":
                        return n.init ? recast.fromString(" = ").join([
                            print(path.get("id")),
                            print(path.get("init"))
                        ]) : print(path.get("id"));
                    case "WithStatement":
                        return recast.concat([
                            "with (",
                            print(path.get("object")),
                            ") ",
                            print(path.get("body"))
                        ]);
                    case "IfStatement":
                        var con = adjustClause(print(path.get("consequent")), options), parts = ["if (", print(path.get("test")), ")", con];
                        if (n.alternate)
                            parts.push(endsWithBrace(con) ? " else" : "\nelse", adjustClause(print(path.get("alternate")), options));
                        return recast.concat(parts);
                    case "ForStatement":
                        // TODO Get the for (;;) case right.
                        var init = print(path.get("init")), sep = init.length > 1 ? ";\n" : "; ", forParen = "for (", indented = recast.fromString(sep).join([
                            init,
                            print(path.get("test")),
                            print(path.get("update"))
                        ]).indentTail(forParen.length), head = recast.concat([forParen, indented, ")"]), clause = adjustClause(print(path.get("body")), options), parts = [head];
                        if (head.length > 1) {
                            parts.push("\n");
                            clause = clause.trimLeft();
                        }
                        parts.push(clause);
                        return recast.concat(parts);
                    case "WhileStatement":
                        return recast.concat([
                            "while (",
                            print(path.get("test")),
                            ")",
                            adjustClause(print(path.get("body")), options)
                        ]);
                    case "ForInStatement":
                        // Note: esprima can't actually parse "for each (".
                        return recast.concat([
                            n.each ? "for each (" : "for (",
                            print(path.get("left")),
                            " in ",
                            print(path.get("right")),
                            ")",
                            adjustClause(print(path.get("body")), options)
                        ]);
                    case "ForOfStatement":
                        return recast.concat([
                            "for (",
                            print(path.get("left")),
                            " of ",
                            print(path.get("right")),
                            ")",
                            adjustClause(print(path.get("body")), options)
                        ]);
                    case "DoWhileStatement":
                        {
                            var doBody = recast.concat([
                                "do",
                                adjustClause(print(path.get("body")), options)
                            ]), parts = [doBody];
                            if (endsWithBrace(doBody))
                                parts.push(" while");
                            else
                                parts.push("\nwhile");
                            parts.push(" (", print(path.get("test")), ");");
                            return recast.concat(parts);
                        }
                    case "BreakStatement":
                        {
                            var parts = ["break"];
                            if (n.label)
                                parts.push(" ", print(path.get("label")));
                            parts.push(";");
                            return recast.concat(parts);
                        }
                    case "ContinueStatement":
                        var parts = ["continue"];
                        if (n.label)
                            parts.push(" ", print(path.get("label")));
                        parts.push(";");
                        return recast.concat(parts);
                    case "LabeledStatement":
                        return recast.concat([
                            print(path.get("label")),
                            ":\n",
                            print(path.get("body"))
                        ]);
                    case "TryStatement":
                        var parts = [
                            "try ",
                            print(path.get("block"))
                        ];
                        path.get("handlers").each(function (handler) {
                            parts.push(" ", print(handler));
                        });
                        if (n.finalizer)
                            parts.push(" finally ", print(path.get("finalizer")));
                        return recast.concat(parts);
                    case "CatchClause":
                        var parts = ["catch (", print(path.get("param"))];
                        if (n.guard)
                            // Note: esprima does not recognize conditional catch clauses.
                            parts.push(" if ", print(path.get("guard")));
                        parts.push(") ", print(path.get("body")));
                        return recast.concat(parts);
                    case "ThrowStatement":
                        return recast.concat([
                            "throw ",
                            print(path.get("argument")),
                            ";"
                        ]);
                    case "SwitchStatement":
                        return recast.concat([
                            "switch (",
                            print(path.get("discriminant")),
                            ") {\n",
                            recast.fromString("\n").join(path.get("cases").map(print)),
                            "\n}"
                        ]);
                    case "SwitchCase":
                        var parts = [];
                        if (n.test)
                            parts.push("case ", print(path.get("test")), ":");
                        else
                            parts.push("default:");
                        if (n.consequent.length > 0) {
                            parts.push("\n", printStatementSequence(path.get("consequent"), options, print).indent(options.tabWidth));
                        }
                        return recast.concat(parts);
                    case "DebuggerStatement":
                        return recast.fromString("debugger;");
                    case "XJSAttribute":
                        var parts = [print(path.get("name"))];
                        if (n.value)
                            parts.push("=", print(path.get("value")));
                        return recast.concat(parts);
                    case "XJSIdentifier":
                        return recast.fromString(n.name, options);
                    case "XJSNamespacedName":
                        return recast.fromString(":").join([
                            print(path.get("namespace")),
                            print(path.get("name"))
                        ]);
                    case "XJSMemberExpression":
                        return recast.fromString(".").join([
                            print(path.get("object")),
                            print(path.get("property"))
                        ]);
                    case "XJSSpreadAttribute":
                        return recast.concat(["{...", print(path.get("argument")), "}"]);
                    case "XJSExpressionContainer":
                        return recast.concat(["{", print(path.get("expression")), "}"]);
                    case "XJSElement":
                        var openingLines = print(path.get("openingElement"));
                        if (n.openingElement.selfClosing) {
                            assert.ok(!n.closingElement);
                            return openingLines;
                        }
                        var childLines = recast.concat(path.get("children").map(function (childPath) {
                            var child = childPath.value;
                            if (namedTypes["Literal"].check(child) && typeof child.value === "string") {
                                if (/\S/.test(child.value)) {
                                    return child.value.replace(/^\s+|\s+$/g, "");
                                }
                                else if (/\n/.test(child.value)) {
                                    return "\n";
                                }
                            }
                            return print(childPath);
                        })).indentTail(options.tabWidth);
                        var closingLines = print(path.get("closingElement"));
                        return recast.concat([
                            openingLines,
                            childLines,
                            closingLines
                        ]);
                    case "XJSOpeningElement":
                        var parts = ["<", print(path.get("name"))];
                        var attrParts = [];
                        path.get("attributes").each(function (attrPath) {
                            attrParts.push(" ", print(attrPath));
                        });
                        var attrLines = recast.concat(attrParts);
                        var needLineWrap = (attrLines.length > 1 || attrLines.getLineLength(1) > options.wrapColumn);
                        if (needLineWrap) {
                            attrParts.forEach(function (part, i) {
                                if (part === " ") {
                                    assert.strictEqual(i % 2, 0);
                                    attrParts[i] = "\n";
                                }
                            });
                            attrLines = recast.concat(attrParts).indentTail(options.tabWidth);
                        }
                        parts.push(attrLines, n.selfClosing ? " />" : ">");
                        return recast.concat(parts);
                    case "XJSClosingElement":
                        return recast.concat(["</", print(path.get("name")), ">"]);
                    case "XJSText":
                        return recast.fromString(n.value, options);
                    case "XJSEmptyExpression":
                        return recast.fromString("");
                    case "TypeAnnotatedIdentifier":
                        var parts = [
                            print(path.get("annotation")),
                            " ",
                            print(path.get("identifier"))
                        ];
                        return recast.concat(parts);
                    case "ClassBody":
                        if (n.body.length === 0) {
                            return recast.fromString("{}");
                        }
                        return recast.concat([
                            "{\n",
                            printStatementSequence(path.get("body"), options, print).indent(options.tabWidth),
                            "\n}"
                        ]);
                    case "ClassPropertyDefinition":
                        {
                            var parts = ["static ", print(path.get("definition"))];
                            if (!namedTypes["MethodDefinition"].check(n.definition))
                                parts.push(";");
                            return recast.concat(parts);
                        }
                    case "ClassProperty":
                        return recast.concat([print(path.get("id")), ";"]);
                    case "ClassDeclaration":
                    case "ClassExpression":
                        {
                            var parts = ["class"];
                            if (n.id)
                                parts.push(" ", print(path.get("id")));
                            if (n.superClass)
                                parts.push(" extends ", print(path.get("superClass")));
                            parts.push(" ", print(path.get("body")));
                            return recast.concat(parts);
                        }
                    case "Node":
                    case "Printable":
                    case "SourceLocation":
                    case "Position":
                    case "Statement":
                    case "Function":
                    case "Pattern":
                    case "Expression":
                    case "Declaration":
                    case "Specifier":
                    case "NamedSpecifier":
                    case "Block":
                    case "Line":
                        throw new Error("unprintable type: " + JSON.stringify(n.type));
                    case "ClassHeritage":
                    case "ComprehensionBlock":
                    case "ComprehensionExpression":
                    case "Glob":
                    case "TaggedTemplateExpression":
                    case "TemplateElement":
                    case "TemplateLiteral":
                    case "GeneratorExpression":
                    case "LetStatement":
                    case "LetExpression":
                    case "GraphExpression":
                    case "GraphIndexExpression":
                    case "AnyTypeAnnotation":
                    case "BooleanTypeAnnotation":
                    case "ClassImplements":
                    case "DeclareClass":
                    case "DeclareFunction":
                    case "DeclareModule":
                    case "DeclareVariable":
                    case "FunctionTypeAnnotation":
                    case "FunctionTypeParam":
                    case "GenericTypeAnnotation":
                    case "InterfaceDeclaration":
                    case "InterfaceExtends":
                    case "IntersectionTypeAnnotation":
                    case "MemberTypeAnnotation":
                    case "NullableTypeAnnotation":
                    case "NumberTypeAnnotation":
                    case "ObjectTypeAnnotation":
                    case "ObjectTypeCallProperty":
                    case "ObjectTypeIndexer":
                    case "ObjectTypeProperty":
                    case "QualifiedTypeIdentifier":
                    case "StringLiteralTypeAnnotation":
                    case "StringTypeAnnotation":
                    case "TupleTypeAnnotation":
                    case "Type":
                    case "TypeAlias":
                    case "TypeAnnotation":
                    case "TypeParameterDeclaration":
                    case "TypeParameterInstantiation":
                    case "TypeofTypeAnnotation":
                    case "UnionTypeAnnotation":
                    case "VoidTypeAnnotation":
                    case "XMLDefaultDeclaration":
                    case "XMLAnyName":
                    case "XMLQualifiedIdentifier":
                    case "XMLFunctionQualifiedIdentifier":
                    case "XMLAttributeSelector":
                    case "XMLFilterExpression":
                    case "XML":
                    case "XMLElement":
                    case "XMLList":
                    case "XMLEscape":
                    case "XMLText":
                    case "XMLStartTag":
                    case "XMLEndTag":
                    case "XMLPointTag":
                    case "XMLName":
                    case "XMLAttribute":
                    case "XMLCdata":
                    case "XMLComment":
                    case "XMLProcessingInstruction":
                    default:
                        debugger;
                        throw new Error("unknown type: " + JSON.stringify(n.type));
                }
                return undefined;
            }
            function printStatementSequence(path, options, print) {
                var inClassBody = path.parent && namedTypes["ClassBody"] && namedTypes["ClassBody"].check(path.parent.node);
                var filtered = path.filter(function (stmtPath) {
                    var stmt = stmtPath.value;
                    // Just in case the AST has been modified to contain falsy
                    // "statements," it's safer simply to skip them.
                    if (!stmt)
                        return false;
                    // Skip printing EmptyStatement nodes to avoid leaving stray
                    // semicolons lying around.
                    if (stmt.type === "EmptyStatement")
                        return false;
                    if (!inClassBody) {
                        namedTypes["Statement"].assert(stmt);
                    }
                    return true;
                });
                var prevTrailingSpace = null;
                var len = filtered.length;
                var parts = [];
                filtered.forEach(function (stmtPath, i) {
                    var printed = print(stmtPath);
                    var stmt = stmtPath.value;
                    var needSemicolon = true;
                    var multiLine = printed.length > 1;
                    var notFirst = i > 0;
                    var notLast = i < len - 1;
                    var leadingSpace;
                    var trailingSpace;
                    if (inClassBody) {
                        var stmt = stmtPath.value;
                        if (namedTypes["MethodDefinition"].check(stmt) || (namedTypes["ClassPropertyDefinition"].check(stmt) && namedTypes["MethodDefinition"].check(stmt.definition))) {
                            needSemicolon = false;
                        }
                    }
                    if (needSemicolon) {
                        // Try to add a semicolon to anything that isn't a method in a
                        // class body.
                        printed = maybeAddSemicolon(printed);
                    }
                    var orig = options.reuseWhitespace && stmt.original;
                    var trueLoc = orig && getTrueLoc(orig);
                    var lines = trueLoc && trueLoc.lines;
                    if (notFirst) {
                        if (lines) {
                            var beforeStart = lines.skipSpaces(trueLoc.start, true);
                            var beforeStartLine = beforeStart ? beforeStart.line : 1;
                            var leadingGap = trueLoc.start.line - beforeStartLine;
                            leadingSpace = Array(leadingGap + 1).join("\n");
                        }
                        else {
                            leadingSpace = multiLine ? "\n\n" : "\n";
                        }
                    }
                    else {
                        leadingSpace = "";
                    }
                    if (notLast) {
                        if (lines) {
                            var afterEnd = lines.skipSpaces(trueLoc.end);
                            var afterEndLine = afterEnd ? afterEnd.line : lines.length;
                            var trailingGap = afterEndLine - trueLoc.end.line;
                            trailingSpace = Array(trailingGap + 1).join("\n");
                        }
                        else {
                            trailingSpace = multiLine ? "\n\n" : "\n";
                        }
                    }
                    else {
                        trailingSpace = "";
                    }
                    parts.push(maxSpace(prevTrailingSpace, leadingSpace), printed);
                    if (notLast) {
                        prevTrailingSpace = trailingSpace;
                    }
                    else if (trailingSpace) {
                        parts.push(trailingSpace);
                    }
                });
                return recast.concat(parts);
            }
            function getTrueLoc(node) {
                if (!node.comments) {
                    // If the node has no comments, regard node.loc as true.
                    return node.loc;
                }
                var start = node.loc.start;
                var end = node.loc.end;
                // If the node has any comments, their locations might contribute to
                // the true start/end positions of the node.
                node.comments.forEach(function (comment) {
                    if (comment.loc) {
                        if (recast.comparePos(comment.loc.start, start) < 0) {
                            start = comment.loc.start;
                        }
                        if (recast.comparePos(end, comment.loc.end) < 0) {
                            end = comment.loc.end;
                        }
                    }
                });
                return {
                    lines: node.loc.lines,
                    start: start,
                    end: end
                };
            }
            function maxSpace(s1, s2) {
                if (!s1 && !s2) {
                    return recast.fromString("");
                }
                if (!s1) {
                    return recast.fromString(s2);
                }
                if (!s2) {
                    return recast.fromString(s1);
                }
                var spaceLines1 = recast.fromString(s1);
                var spaceLines2 = recast.fromString(s2);
                if (spaceLines2.length > spaceLines1.length) {
                    return spaceLines2;
                }
                return spaceLines1;
            }
            function printMethod(kind, keyPath, valuePath, options, print) {
                var parts = [];
                var key = keyPath.value;
                var value = valuePath.value;
                namedTypes["FunctionExpression"].assert(value);
                if (value.async) {
                    parts.push("async ");
                }
                if (!kind || kind === "init") {
                    if (value.generator) {
                        parts.push("*");
                    }
                }
                else {
                    assert.ok(kind === "get" || kind === "set");
                    parts.push(kind, " ");
                }
                parts.push(print(keyPath), "(", printFunctionParams(valuePath, options, print), ") ", print(valuePath.get("body")));
                return recast.concat(parts);
            }
            function printArgumentsList(path, options, print) {
                var printed = path.get("arguments").map(print);
                var joined = recast.fromString(", ").join(printed);
                if (joined.getLineLength(1) > options.wrapColumn) {
                    joined = recast.fromString(",\n").join(printed);
                    return recast.concat(["(\n", joined.indent(options.tabWidth), "\n)"]);
                }
                return recast.concat(["(", joined, ")"]);
            }
            function printFunctionParams(path, options, print) {
                var fun = path.node;
                namedTypes["Function"].assert(fun);
                var params = path.get("params");
                var defaults = path.get("defaults");
                var printed = params.map(defaults.value ? function (param) {
                    var p = print(param);
                    var d = defaults.get(param.name);
                    return d.value ? recast.concat([p, "=", print(d)]) : p;
                } : print);
                if (fun.rest) {
                    printed.push(recast.concat(["...", print(path.get("rest"))]));
                }
                var joined = recast.fromString(", ").join(printed);
                if (joined.length > 1 || joined.getLineLength(1) > options.wrapColumn) {
                    joined = recast.fromString(",\n").join(printed);
                    return recast.concat(["\n", joined.indent(options.tabWidth)]);
                }
                return joined;
            }
            function adjustClause(clause, options) {
                if (clause.length > 1)
                    return recast.concat([" ", clause]);
                return recast.concat([
                    "\n",
                    maybeAddSemicolon(clause).indent(options.tabWidth)
                ]);
            }
            function lastNonSpaceCharacter(lines) {
                var pos = lines.lastPos();
                do {
                    var ch = lines.charAt(pos);
                    if (/\S/.test(ch))
                        return ch;
                } while (lines.prevPos(pos));
            }
            function endsWithBrace(lines) {
                return lastNonSpaceCharacter(lines) === "}";
            }
            function nodeStr(n) {
                namedTypes["Literal"].assert(n);
                isString.assert(n.value);
                return JSON.stringify(n.value);
            }
            function maybeAddSemicolon(lines) {
                var eoc = lastNonSpaceCharacter(lines);
                if (!eoc || "\n};".indexOf(eoc) < 0)
                    return recast.concat([lines, ";"]);
                return lines;
            }
        })(recast = _ast.recast || (_ast.recast = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var types;
        (function (types) {
            var definitions;
            (function (definitions) {
                var types = lib.ast.types;
                var Type = types.Type;
                var def = Type.def;
                var or = Type.or;
                var builtin = types.builtInTypes;
                var isString = builtin["string"];
                var isNumber = builtin["number"];
                var isBoolean = builtin["boolean"];
                var isRegExp = builtin["RegExp"];
                debugger;
                var defaults = types.defaults;
                var geq = types.geq;
                var core;
                (function (core) {
                    // Abstract supertype of all syntactic entities that are allowed to have a
                    // .loc field.
                    def("Printable").field("loc", or(def("SourceLocation"), null), defaults["null"], true);
                    def("Node").bases("Printable").field("type", isString);
                    def("SourceLocation").build("start", "end", "source").field("start", def("Position")).field("end", def("Position")).field("source", or(isString, null), defaults["null"]);
                    def("Position").build("line", "column").field("line", geq(1)).field("column", geq(0));
                    def("Program").bases("Node").build("body").field("body", [def("Statement")]).field("comments", or([or(def("Block"), def("Line"))], null), defaults["null"], true);
                    def("Function").bases("Node").field("id", or(def("Identifier"), null), defaults["null"]).field("params", [def("Pattern")]).field("body", or(def("BlockStatement"), def("Expression")));
                    def("Statement").bases("Node");
                    // The empty .build() here means that an EmptyStatement can be constructed
                    // (i.e. it's not abstract) but that it needs no arguments.
                    def("EmptyStatement").bases("Statement").build();
                    def("BlockStatement").bases("Statement").build("body").field("body", [def("Statement")]);
                    // TODO Figure out how to silently coerce Expressions to
                    // ExpressionStatements where a Statement was expected.
                    def("ExpressionStatement").bases("Statement").build("expression").field("expression", def("Expression"));
                    def("IfStatement").bases("Statement").build("test", "consequent", "alternate").field("test", def("Expression")).field("consequent", def("Statement")).field("alternate", or(def("Statement"), null), defaults["null"]);
                    def("LabeledStatement").bases("Statement").build("label", "body").field("label", def("Identifier")).field("body", def("Statement"));
                    def("BreakStatement").bases("Statement").build("label").field("label", or(def("Identifier"), null), defaults["null"]);
                    def("ContinueStatement").bases("Statement").build("label").field("label", or(def("Identifier"), null), defaults["null"]);
                    def("WithStatement").bases("Statement").build("object", "body").field("object", def("Expression")).field("body", def("Statement"));
                    def("SwitchStatement").bases("Statement").build("discriminant", "cases", "lexical").field("discriminant", def("Expression")).field("cases", [def("SwitchCase")]).field("lexical", isBoolean, defaults["false"]);
                    def("ReturnStatement").bases("Statement").build("argument").field("argument", or(def("Expression"), null));
                    def("ThrowStatement").bases("Statement").build("argument").field("argument", def("Expression"));
                    def("TryStatement").bases("Statement").build("block", "handler", "finalizer").field("block", def("BlockStatement")).field("handler", or(def("CatchClause"), null), function () {
                        return this["handlers"] && this["handlers"][0] || null;
                    }).field("handlers", [def("CatchClause")], function () {
                        return this.handler ? [this.handler] : [];
                    }, true).field("guardedHandlers", [def("CatchClause")], defaults.emptyArray).field("finalizer", or(def("BlockStatement"), null), defaults["null"]);
                    def("CatchClause").bases("Node").build("param", "guard", "body").field("param", def("Pattern")).field("guard", or(def("Expression"), null), defaults["null"]).field("body", def("BlockStatement"));
                    def("WhileStatement").bases("Statement").build("test", "body").field("test", def("Expression")).field("body", def("Statement"));
                    def("DoWhileStatement").bases("Statement").build("body", "test").field("body", def("Statement")).field("test", def("Expression"));
                    def("ForStatement").bases("Statement").build("init", "test", "update", "body").field("init", or(def("VariableDeclaration"), def("Expression"), null)).field("test", or(def("Expression"), null)).field("update", or(def("Expression"), null)).field("body", def("Statement"));
                    def("ForInStatement").bases("Statement").build("left", "right", "body", "each").field("left", or(def("VariableDeclaration"), def("Expression"))).field("right", def("Expression")).field("body", def("Statement")).field("each", isBoolean);
                    def("DebuggerStatement").bases("Statement").build();
                    def("Declaration").bases("Statement");
                    def("FunctionDeclaration").bases("Function", "Declaration").build("id", "params", "body").field("id", def("Identifier"));
                    def("FunctionExpression").bases("Function", "Expression").build("id", "params", "body");
                    def("VariableDeclaration").bases("Declaration").build("kind", "declarations").field("kind", or("var", "let", "const")).field("declarations", [or(def("VariableDeclarator"), def("Identifier"))]);
                    def("VariableDeclarator").bases("Node").build("id", "init").field("id", def("Pattern")).field("init", or(def("Expression"), null));
                    // TODO Are all Expressions really Patterns?
                    def("Expression").bases("Node", "Pattern");
                    def("ThisExpression").bases("Expression").build();
                    def("ArrayExpression").bases("Expression").build("elements").field("elements", [or(def("Expression"), null)]);
                    def("ObjectExpression").bases("Expression").build("properties").field("properties", [def("Property")]);
                    // TODO Not in the Mozilla Parser API, but used by Esprima.
                    def("Property").bases("Node").build("kind", "key", "value").field("kind", or("init", "get", "set")).field("key", or(def("Literal"), def("Identifier"))).field("value", def("Expression"));
                    def("SequenceExpression").bases("Expression").build("expressions").field("expressions", [def("Expression")]);
                    var UnaryOperator = or("-", "+", "!", "~", "typeof", "void", "delete");
                    def("UnaryExpression").bases("Expression").build("operator", "argument", "prefix").field("operator", UnaryOperator).field("argument", def("Expression")).field("prefix", isBoolean, defaults["true"]);
                    var BinaryOperator = or("==", "!=", "===", "!==", "<", "<=", ">", ">=", "<<", ">>", ">>>", "+", "-", "*", "/", "%", "&", "|", "^", "in", "instanceof", "..");
                    def("BinaryExpression").bases("Expression").build("operator", "left", "right").field("operator", BinaryOperator).field("left", def("Expression")).field("right", def("Expression"));
                    var AssignmentOperator = or("=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", ">>>=", "|=", "^=", "&=");
                    def("AssignmentExpression").bases("Expression").build("operator", "left", "right").field("operator", AssignmentOperator).field("left", def("Pattern")).field("right", def("Expression"));
                    var UpdateOperator = or("++", "--");
                    def("UpdateExpression").bases("Expression").build("operator", "argument", "prefix").field("operator", UpdateOperator).field("argument", def("Expression")).field("prefix", isBoolean);
                    var LogicalOperator = or("||", "&&");
                    def("LogicalExpression").bases("Expression").build("operator", "left", "right").field("operator", LogicalOperator).field("left", def("Expression")).field("right", def("Expression"));
                    def("ConditionalExpression").bases("Expression").build("test", "consequent", "alternate").field("test", def("Expression")).field("consequent", def("Expression")).field("alternate", def("Expression"));
                    def("NewExpression").bases("Expression").build("callee", "arguments").field("callee", def("Expression")).field("arguments", [def("Expression")]);
                    def("CallExpression").bases("Expression").build("callee", "arguments").field("callee", def("Expression")).field("arguments", [def("Expression")]);
                    def("MemberExpression").bases("Expression").build("object", "property", "computed").field("object", def("Expression")).field("property", or(def("Identifier"), def("Expression"))).field("computed", isBoolean);
                    def("Pattern").bases("Node");
                    def("ObjectPattern").bases("Pattern").build("properties").field("properties", [def("PropertyPattern")]);
                    def("PropertyPattern").bases("Pattern").build("key", "pattern").field("key", or(def("Literal"), def("Identifier"))).field("pattern", def("Pattern"));
                    def("ArrayPattern").bases("Pattern").build("elements").field("elements", [or(def("Pattern"), null)]);
                    def("SwitchCase").bases("Node").build("test", "consequent").field("test", or(def("Expression"), null)).field("consequent", [def("Statement")]);
                    def("Identifier").bases("Node", "Expression", "Pattern").build("name").field("name", isString);
                    def("Literal").bases("Node", "Expression").build("value").field("value", or(isString, isBoolean, null, isNumber, isRegExp));
                    // Block comment. Not a Node.
                    def("Block").bases("Printable").build("loc", "value").field("value", isString);
                    // Single line comment. Not a Node.
                    def("Line").bases("Printable").build("loc", "value").field("value", isString);
                })(core = definitions.core || (definitions.core = {}));
                definitions.coredefs = true;
            })(definitions = types.definitions || (types.definitions = {}));
        })(types = ast.types || (ast.types = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var types;
        (function (types) {
            var definitions;
            (function (definitions) {
                var types = lib.ast.types;
                var def = types.Type.def;
                var or = types.Type.or;
                var builtin = types.builtInTypes;
                var isString = builtin["string"];
                var isBoolean = builtin["boolean"];
                var defaults = types.defaults;
                var geq = types.geq;
                // Note that none of these types are buildable because the Mozilla Parser
                // API doesn't specify any builder functions, and nobody uses E4X anymore.
                def("XMLDefaultDeclaration").bases("Declaration").field("namespace", def("Expression"));
                def("XMLAnyName").bases("Expression");
                def("XMLQualifiedIdentifier").bases("Expression").field("left", or(def("Identifier"), def("XMLAnyName"))).field("right", or(def("Identifier"), def("Expression"))).field("computed", isBoolean);
                def("XMLFunctionQualifiedIdentifier").bases("Expression").field("right", or(def("Identifier"), def("Expression"))).field("computed", isBoolean);
                def("XMLAttributeSelector").bases("Expression").field("attribute", def("Expression"));
                def("XMLFilterExpression").bases("Expression").field("left", def("Expression")).field("right", def("Expression"));
                def("XMLElement").bases("XML", "Expression").field("contents", [def("XML")]);
                def("XMLList").bases("XML", "Expression").field("contents", [def("XML")]);
                def("XML").bases("Node");
                def("XMLEscape").bases("XML").field("expression", def("Expression"));
                def("XMLText").bases("XML").field("text", isString);
                def("XMLStartTag").bases("XML").field("contents", [def("XML")]);
                def("XMLEndTag").bases("XML").field("contents", [def("XML")]);
                def("XMLPointTag").bases("XML").field("contents", [def("XML")]);
                def("XMLName").bases("XML").field("contents", or(isString, [def("XML")]));
                def("XMLAttribute").bases("XML").field("value", isString);
                def("XMLCdata").bases("XML").field("contents", isString);
                def("XMLComment").bases("XML").field("contents", isString);
                def("XMLProcessingInstruction").bases("XML").field("target", isString).field("contents", or(isString, null));
            })(definitions = types.definitions || (types.definitions = {}));
        })(types = ast.types || (ast.types = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var types;
        (function (types) {
            var definitions;
            (function (definitions) {
                var types = lib.ast.types;
                var def = types.Type.def;
                var or = types.Type.or;
                var builtin = types.builtInTypes;
                var isString = builtin["string"];
                var isBoolean = builtin["boolean"];
                var defaults = types.defaults;
                var geq = types.geq;
                def("Function").field("generator", isBoolean, defaults["false"]).field("expression", isBoolean, defaults["false"]).field("defaults", [or(def("Expression"), null)], defaults.emptyArray).field("rest", or(def("Identifier"), null), defaults["null"]);
                def("FunctionDeclaration").build("id", "params", "body", "generator", "expression");
                def("FunctionExpression").build("id", "params", "body", "generator", "expression");
                // TODO The Parser API calls this ArrowExpression, but Esprima uses
                // ArrowFunctionExpression.
                def("ArrowFunctionExpression").bases("Function", "Expression").build("params", "body", "expression").field("id", null, defaults["null"]).field("generator", false);
                def("YieldExpression").bases("Expression").build("argument", "delegate").field("argument", or(def("Expression"), null)).field("delegate", isBoolean, defaults["false"]);
                def("GeneratorExpression").bases("Expression").build("body", "blocks", "filter").field("body", def("Expression")).field("blocks", [def("ComprehensionBlock")]).field("filter", or(def("Expression"), null));
                def("ComprehensionExpression").bases("Expression").build("body", "blocks", "filter").field("body", def("Expression")).field("blocks", [def("ComprehensionBlock")]).field("filter", or(def("Expression"), null));
                def("ComprehensionBlock").bases("Node").build("left", "right", "each").field("left", def("Pattern")).field("right", def("Expression")).field("each", isBoolean);
                def("ModuleSpecifier").bases("Literal").build("value").field("value", isString);
                def("Property").field("key", or(def("Literal"), def("Identifier"), def("Expression"))).field("method", isBoolean, defaults["false"]).field("shorthand", isBoolean, defaults["false"]).field("computed", isBoolean, defaults["false"]);
                def("PropertyPattern").field("key", or(def("Literal"), def("Identifier"), def("Expression"))).field("computed", isBoolean, defaults["false"]);
                def("MethodDefinition").bases("Declaration").build("kind", "key", "value").field("kind", or("init", "get", "set", "")).field("key", or(def("Literal"), def("Identifier"), def("Expression"))).field("value", def("Function")).field("computed", isBoolean, defaults["false"]);
                def("SpreadElement").bases("Node").build("argument").field("argument", def("Expression"));
                def("ArrayExpression").field("elements", [or(def("Expression"), def("SpreadElement"), null)]);
                def("NewExpression").field("arguments", [or(def("Expression"), def("SpreadElement"))]);
                def("CallExpression").field("arguments", [or(def("Expression"), def("SpreadElement"))]);
                def("SpreadElementPattern").bases("Pattern").build("argument").field("argument", def("Pattern"));
                var ClassBodyElement = or(def("MethodDefinition"), def("VariableDeclarator"), def("ClassPropertyDefinition"), def("ClassProperty"));
                def("ClassProperty").bases("Declaration").build("key").field("key", or(def("Literal"), def("Identifier"), def("Expression"))).field("computed", isBoolean, defaults["false"]);
                def("ClassPropertyDefinition").bases("Declaration").build("definition").field("definition", ClassBodyElement);
                def("ClassBody").bases("Declaration").build("body").field("body", [ClassBodyElement]);
                def("ClassDeclaration").bases("Declaration").build("id", "body", "superClass").field("id", def("Identifier")).field("body", def("ClassBody")).field("superClass", or(def("Expression"), null), defaults["null"]);
                def("ClassExpression").bases("Expression").build("id", "body", "superClass").field("id", or(def("Identifier"), null), defaults["null"]).field("body", def("ClassBody")).field("superClass", or(def("Expression"), null), defaults["null"]).field("implements", [def("ClassImplements")], defaults.emptyArray);
                def("ClassImplements").bases("Node").build("id").field("id", def("Identifier")).field("superClass", or(def("Expression"), null), defaults["null"]);
                // Specifier and NamedSpecifier are abstract non-standard types that I
                // introduced for definitional convenience.
                def("Specifier").bases("Node");
                def("NamedSpecifier").bases("Specifier").field("id", def("Identifier")).field("name", or(def("Identifier"), null), defaults["null"]);
                // Like NamedSpecifier, except type:"ExportSpecifier" and buildable.
                // export {<id [as name]>} [from ...];
                def("ExportSpecifier").bases("NamedSpecifier").build("id", "name");
                // export <*> from ...;
                def("ExportBatchSpecifier").bases("Specifier").build();
                // Like NamedSpecifier, except type:"ImportSpecifier" and buildable.
                // import {<id [as name]>} from ...;
                def("ImportSpecifier").bases("NamedSpecifier").build("id", "name");
                // import <* as id> from ...;
                def("ImportNamespaceSpecifier").bases("Specifier").build("id").field("id", def("Identifier"));
                // import <id> from ...;
                def("ImportDefaultSpecifier").bases("Specifier").build("id").field("id", def("Identifier"));
                def("ExportDeclaration").bases("Declaration").build("default", "declaration", "specifiers", "source").field("default", isBoolean).field("declaration", or(def("Declaration"), def("Expression"), null)).field("specifiers", [or(def("ExportSpecifier"), def("ExportBatchSpecifier"))], defaults.emptyArray).field("source", or(def("ModuleSpecifier"), null), defaults["null"]);
                def("ImportDeclaration").bases("Declaration").build("specifiers", "source").field("specifiers", [or(def("ImportSpecifier"), def("ImportNamespaceSpecifier"), def("ImportDefaultSpecifier"))], defaults.emptyArray).field("source", def("ModuleSpecifier"));
                def("TaggedTemplateExpression").bases("Expression").field("tag", def("Expression")).field("quasi", def("TemplateLiteral"));
                def("TemplateLiteral").bases("Expression").build("quasis", "expressions").field("quasis", [def("TemplateElement")]).field("expressions", [def("Expression")]);
                def("TemplateElement").bases("Node").build("value", "tail").field("value", { "cooked": isString, "raw": isString }).field("tail", isBoolean);
            })(definitions = types.definitions || (types.definitions = {}));
        })(types = ast.types || (ast.types = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var types;
        (function (types) {
            var definitions;
            (function (definitions) {
                var types = lib.ast.types;
                var def = types.Type.def;
                var or = types.Type.or;
                var builtin = types.builtInTypes;
                var isString = builtin["string"];
                var isBoolean = builtin["boolean"];
                var defaults = types.defaults;
                var geq = types.geq;
                def("Function").field("async", isBoolean, defaults["false"]);
                def("SpreadProperty").bases("Node").build("argument").field("argument", def("Expression"));
                def("ObjectExpression").field("properties", [or(def("Property"), def("SpreadProperty"))]);
                def("SpreadPropertyPattern").bases("Pattern").build("argument").field("argument", def("Pattern"));
                def("ObjectPattern").field("properties", [or(def("PropertyPattern"), def("SpreadPropertyPattern"))]);
                def("AwaitExpression").bases("Expression").build("argument", "all").field("argument", or(def("Expression"), null)).field("all", isBoolean, defaults["false"]);
            })(definitions = types.definitions || (types.definitions = {}));
        })(types = ast.types || (ast.types = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var types;
        (function (types) {
            var definitions;
            (function (definitions) {
                var types = lib.ast.types;
                var def = types.Type.def;
                var or = types.Type.or;
                var builtin = types.builtInTypes;
                var isString = builtin["string"];
                var isBoolean = builtin["boolean"];
                var defaults = types.defaults;
                var geq = types.geq;
                def("XJSAttribute").bases("Node").build("name", "value").field("name", or(def("XJSIdentifier"), def("XJSNamespacedName"))).field("value", or(def("Literal"), def("XJSExpressionContainer"), null), defaults["null"]);
                def("XJSIdentifier").bases("Node").build("name").field("name", isString);
                def("XJSNamespacedName").bases("Node").build("namespace", "name").field("namespace", def("XJSIdentifier")).field("name", def("XJSIdentifier"));
                def("XJSMemberExpression").bases("MemberExpression").build("object", "property").field("object", or(def("XJSIdentifier"), def("XJSMemberExpression"))).field("property", def("XJSIdentifier")).field("computed", isBoolean, defaults.false);
                var XJSElementName = or(def("XJSIdentifier"), def("XJSNamespacedName"), def("XJSMemberExpression"));
                def("XJSSpreadAttribute").bases("Node").build("argument").field("argument", def("Expression"));
                var XJSAttributes = [or(def("XJSAttribute"), def("XJSSpreadAttribute"))];
                def("XJSExpressionContainer").bases("Expression").build("expression").field("expression", def("Expression"));
                /*
                 def("XJSElement")
                 .bases("Expression")
                 .build("openingElement", "closingElement", "children")
                 .field("openingElement", def("XJSOpeningElement"))
                 .field("closingElement", or(def("XJSClosingElement"), null), defaults["null"])
                 .field("children", [or(
                 def("XJSElement"),
                 def("XJSExpressionContainer"),
                 def("XJSText"),
                 def("Literal") // TODO Esprima should return XJSText instead.
                 )], defaults.emptyArray)
                 .field("name", XJSElementName, function() {
                 // Little-known fact: the `this` object inside a default function
                 // is none other than the partially-built object itself, and any
                 // fields initialized directly from builder function arguments
                 // (like openingElement, closingElement, and children) are
                 // guaranteed to be available.
                 return this.openingElement.name;
                 })
                 .field("selfClosing", isBoolean, function() {
                 return this.openingElement.selfClosing;
                 })
                 .field("attributes", XJSAttributes, function() {
                 return this.openingElement.attributes;
                 });
                 */
                def("XJSOpeningElement").bases("Node").build("name", "attributes", "selfClosing").field("name", XJSElementName).field("attributes", XJSAttributes, defaults.emptyArray).field("selfClosing", isBoolean, defaults["false"]);
                def("XJSClosingElement").bases("Node").build("name").field("name", XJSElementName);
                def("XJSText").bases("Literal").build("value").field("value", isString);
                def("XJSEmptyExpression").bases("Expression").build();
                // Type Annotations
                def("Type").bases("Node");
                def("AnyTypeAnnotation").bases("Type");
                def("VoidTypeAnnotation").bases("Type");
                def("NumberTypeAnnotation").bases("Type");
                def("StringTypeAnnotation").bases("Type");
                def("StringLiteralTypeAnnotation").bases("Type").build("value", "raw").field("value", isString).field("raw", isString);
                def("BooleanTypeAnnotation").bases("Type");
                def("TypeAnnotation").bases("Node").build("typeAnnotation").field("typeAnnotation", def("Type"));
                def("NullableTypeAnnotation").bases("Type").build("typeAnnotation").field("typeAnnotation", def("Type"));
                def("FunctionTypeAnnotation").bases("Type").build("params", "returnType", "rest", "typeParameters").field("params", [def("FunctionTypeParam")]).field("returnType", def("Type")).field("rest", or(def("FunctionTypeParam"), null)).field("typeParameters", or(def("TypeParameterDeclaration"), null));
                def("FunctionTypeParam").bases("Node").build("name", "typeAnnotation", "optional").field("name", def("Identifier")).field("typeAnnotation", def("Type")).field("optional", isBoolean);
                def("ObjectTypeAnnotation").bases("Type").build("properties").field("properties", [def("ObjectTypeProperty")]).field("indexers", [def("ObjectTypeIndexer")], defaults.emptyArray).field("callProperties", [def("ObjectTypeCallProperty")], defaults.emptyArray);
                def("ObjectTypeProperty").bases("Node").build("key", "value", "optional").field("key", or(def("Literal"), def("Identifier"))).field("value", def("Type")).field("optional", isBoolean);
                def("ObjectTypeIndexer").bases("Node").build("id", "key", "value").field("id", def("Identifier")).field("key", def("Type")).field("value", def("Type"));
                def("ObjectTypeCallProperty").bases("Node").build("value").field("value", def("FunctionTypeAnnotation")).field("static", isBoolean, false);
                def("QualifiedTypeIdentifier").bases("Node").build("qualification", "id").field("qualification", or(def("Identifier"), def("QualifiedTypeIdentifier"))).field("id", def("Identifier"));
                def("GenericTypeAnnotation").bases("Type").build("id", "typeParameters").field("id", or(def("Identifier"), def("QualifiedTypeIdentifier"))).field("typeParameters", or(def("TypeParameterInstantiation"), null));
                def("MemberTypeAnnotation").bases("Type").build("object", "property").field("object", def("Identifier")).field("property", or(def("MemberTypeAnnotation"), def("GenericTypeAnnotation")));
                def("UnionTypeAnnotation").bases("Type").build("types").field("types", [def("Type")]);
                def("IntersectionTypeAnnotation").bases("Type").build("types").field("types", [def("Type")]);
                def("TypeofTypeAnnotation").bases("Type").build("argument").field("argument", def("Type"));
                def("Identifier").field("typeAnnotation", or(def("TypeAnnotation"), null), defaults["null"]);
                def("TypeParameterDeclaration").bases("Node").build("params").field("params", [def("Identifier")]);
                def("TypeParameterInstantiation").bases("Node").build("params").field("params", [def("Type")]);
                def("Function").field("returnType", or(def("TypeAnnotation"), null), defaults["null"]).field("typeParameters", or(def("TypeParameterDeclaration"), null), defaults["null"]);
                def("ClassProperty").build("key", "typeAnnotation").field("typeAnnotation", def("TypeAnnotation")).field("static", isBoolean, false);
                def("ClassImplements").field("typeParameters", or(def("TypeParameterInstantiation"), null), defaults["null"]);
                def("InterfaceDeclaration").bases("Statement").build("id", "body", "extends").field("id", def("Identifier")).field("typeParameters", or(def("TypeParameterDeclaration"), null), defaults["null"]).field("body", def("ObjectTypeAnnotation")).field("extends", [def("InterfaceExtends")]);
                def("InterfaceExtends").bases("Node").build("id").field("id", def("Identifier")).field("typeParameters", or(def("TypeParameterInstantiation"), null));
                def("TypeAlias").bases("Statement").build("id", "typeParameters", "right").field("id", def("Identifier")).field("typeParameters", or(def("TypeParameterDeclaration"), null)).field("right", def("Type"));
                def("TupleTypeAnnotation").bases("Type").build("types").field("types", [def("Type")]);
                def("DeclareVariable").bases("Statement").build("id").field("id", def("Identifier"));
                def("DeclareFunction").bases("Statement").build("id").field("id", def("Identifier"));
                def("DeclareClass").bases("InterfaceDeclaration").build("id");
                def("DeclareModule").bases("Statement").build("id", "body").field("id", or(def("Identifier"), def("Literal"))).field("body", def("BlockStatement"));
            })(definitions = types.definitions || (types.definitions = {}));
        })(types = ast.types || (ast.types = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var types;
        (function (types) {
            var definitions;
            (function (definitions) {
                var types = lib.ast.types;
                var def = types.Type.def;
                var or = types.Type.or;
                var geq = types.geq;
                def("ForOfStatement").bases("Statement").build("left", "right", "body").field("left", or(def("VariableDeclaration"), def("Expression"))).field("right", def("Expression")).field("body", def("Statement"));
                def("LetStatement").bases("Statement").build("head", "body").field("head", [def("VariableDeclarator")]).field("body", def("Statement"));
                def("LetExpression").bases("Expression").build("head", "body").field("head", [def("VariableDeclarator")]).field("body", def("Expression"));
                def("GraphExpression").bases("Expression").build("index", "expression").field("index", geq(0)).field("expression", def("Literal"));
                def("GraphIndexExpression").bases("Expression").build("index").field("index", geq(0));
            })(definitions = types.definitions || (types.definitions = {}));
        })(types = ast.types || (ast.types = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/// <reference path="../types/def/core.ts" />
/// <reference path="../types/def/e4x.ts" />
/// <reference path="../types/def/es6.ts" />
/// <reference path="../types/def/es7.ts" />
/// <reference path="../types/def/fb-harmony.ts" />
/// <reference path="../types/def/mozilla.ts" />
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var recast;
        (function (recast) {
            recast.types = lib.ast.types;
            var def = recast.types.Type.def;
            def("File").bases("Node").build("program").field("program", def("Program"));
            recast.types.finalize();
        })(recast = ast.recast || (ast.recast = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/// <reference path="recast.ts" />
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var recast;
        (function (recast) {
            var sourceMap = lib.ast.sourcemap;
            var SourceMapConsumer = sourceMap.SourceMapConsumer;
            var SourceMapGenerator = sourceMap.SourceMapGenerator;
            var hasOwn = Object.prototype.hasOwnProperty;
            function getUnionOfKeys() {
                var args = [];
                for (var _i = 0; _i < arguments.length; _i++) {
                    args[_i - 0] = arguments[_i];
                }
                var result = {};
                var argc = args.length;
                for (var i = 0; i < argc; ++i) {
                    var keys = Object.keys(args[i]);
                    var keyCount = keys.length;
                    for (var j = 0; j < keyCount; ++j) {
                        result[keys[j]] = true;
                    }
                }
                return result;
            }
            recast.getUnionOfKeys = getUnionOfKeys;
            function comparePos(pos1, pos2) {
                return (pos1.line - pos2.line) || (pos1.column - pos2.column);
            }
            recast.comparePos = comparePos;
            function composeSourceMaps(formerMap, latterMap) {
                if (formerMap) {
                    if (!latterMap) {
                        return formerMap;
                    }
                }
                else {
                    return latterMap || null;
                }
                var smcFormer = new SourceMapConsumer(formerMap);
                var smcLatter = new SourceMapConsumer(latterMap);
                var smg = new SourceMapGenerator({
                    file: latterMap.file,
                    sourceRoot: latterMap.sourceRoot
                });
                var sourcesToContents = {};
                smcLatter.eachMapping(function (mapping) {
                    var origPos = smcFormer.originalPositionFor({
                        line: mapping.originalLine,
                        column: mapping.originalColumn
                    });
                    var sourceName = origPos.source;
                    smg.addMapping({
                        source: sourceName,
                        original: {
                            line: origPos.line,
                            column: origPos.column
                        },
                        generated: {
                            line: mapping.generatedLine,
                            column: mapping.generatedColumn
                        },
                        name: mapping.name
                    });
                    var sourceContent = smcFormer.sourceContentFor(sourceName);
                    if (sourceContent && !hasOwn.call(sourcesToContents, sourceName)) {
                        sourcesToContents[sourceName] = sourceContent;
                        smg.setSourceContent(sourceName, sourceContent);
                    }
                });
                return smg.toJSON();
            }
            recast.composeSourceMaps = composeSourceMaps;
        })(recast = ast.recast || (ast.recast = {}));
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
            var Controller = (function () {
                function Controller() {
                }
                // API:
                // return property path array from root to current node
                Controller.prototype.path = function () {
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
                Controller.prototype.parents = function () {
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
                Controller.prototype.current = function () {
                    return this.__current.node;
                };
                Controller.prototype.__execute = function (callback, element) {
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
                Controller.prototype.notify = function (flag) {
                    this.__state = flag;
                };
                // API:
                // skip child nodes of current node
                Controller.prototype.skip = function () {
                    this.notify(SKIP);
                };
                // API:
                // break traversals
                Controller.prototype["break"] = function () {
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
                Controller.prototype.traverse = function (root, visitor) {
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
                Controller.prototype.replace = function (root, visitor) {
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
                return Controller;
            })();
            function isNode(node) {
                if (node == null) {
                    return false;
                }
                return typeof node === 'object' && typeof node.type === 'string';
            }
            function isProperty(nodeType, key) {
                return (nodeType === Syntax.ObjectExpression || nodeType === Syntax.ObjectPattern) && 'properties' === key;
            }
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
/// <referench path="core.ts" />
/// <referench path="e4x.ts" />
/// <referench path="es6.ts" />
/// <referench path="es7.ts" />
/// <referench path="fb-harmony.ts" />
/// <referench path="mozilla.ts" />
/// <reference path="types.ts" />
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var types;
        (function (types) {
            var assert = lib.utils.assert;
            var getFieldNames = getFieldNames;
            var getFieldValue = getFieldValue;
            var isArray = types.builtInTypes["array"];
            var isObject = types.builtInTypes["object"];
            var isDate = types.builtInTypes["Date"];
            var isRegExp = types.builtInTypes["RegExp"];
            var hasOwn = Object.prototype.hasOwnProperty;
            function _astNodesAreEquivalent(a, b, problemPath) {
                if (isArray.check(problemPath)) {
                    problemPath.length = 0;
                }
                else {
                    problemPath = null;
                }
                return areEquivalent(a, b, problemPath);
            }
            var astNodesAreEquivalent = (function () {
                function astNodesAreEquivalent() {
                }
                astNodesAreEquivalent.assert = function (a, b) {
                    var problemPath = [];
                    if (!_astNodesAreEquivalent(a, b, problemPath)) {
                        if (problemPath.length === 0) {
                            assert.strictEqual(a, b);
                        }
                        else {
                            assert.ok(false, "Nodes differ in the following path: " + problemPath.map(subscriptForProperty).join(""));
                        }
                    }
                };
                return astNodesAreEquivalent;
            })();
            types.astNodesAreEquivalent = astNodesAreEquivalent;
            function subscriptForProperty(property) {
                if (/[_$a-z][_$a-z0-9]*/i.test(property)) {
                    return "." + property;
                }
                return "[" + JSON.stringify(property) + "]";
            }
            function areEquivalent(a, b, problemPath) {
                if (a === b) {
                    return true;
                }
                if (isArray.check(a)) {
                    return arraysAreEquivalent(a, b, problemPath);
                }
                if (isObject.check(a)) {
                    return objectsAreEquivalent(a, b, problemPath);
                }
                if (isDate.check(a)) {
                    return isDate.check(b) && (+a === +b);
                }
                if (isRegExp.check(a)) {
                    return isRegExp.check(b) && (a.source === b.source && a.global === b.global && a.multiline === b.multiline && a.ignoreCase === b.ignoreCase);
                }
                return a == b;
            }
            function arraysAreEquivalent(a, b, problemPath) {
                isArray.assert(a);
                var aLength = a.length;
                if (!isArray.check(b) || b.length !== aLength) {
                    if (problemPath) {
                        problemPath.push("length");
                    }
                    return false;
                }
                for (var i = 0; i < aLength; ++i) {
                    if (problemPath) {
                        problemPath.push(i);
                    }
                    if (i in a !== i in b) {
                        return false;
                    }
                    if (!areEquivalent(a[i], b[i], problemPath)) {
                        return false;
                    }
                    if (problemPath) {
                        assert.strictEqual(problemPath.pop(), i);
                    }
                }
                return true;
            }
            function objectsAreEquivalent(a, b, problemPath) {
                isObject.assert(a);
                if (!isObject.check(b)) {
                    return false;
                }
                // Fast path for a common property of AST nodes.
                if (a.type !== b.type) {
                    if (problemPath) {
                        problemPath.push("type");
                    }
                    return false;
                }
                var aNames = getFieldNames(a);
                var aNameCount = aNames.length;
                var bNames = getFieldNames(b);
                var bNameCount = bNames.length;
                if (aNameCount === bNameCount) {
                    for (var i = 0; i < aNameCount; ++i) {
                        var name = aNames[i];
                        var aChild = getFieldValue(a, name);
                        var bChild = getFieldValue(b, name);
                        if (problemPath) {
                            problemPath.push(name);
                        }
                        if (!areEquivalent(aChild, bChild, problemPath)) {
                            return false;
                        }
                        if (problemPath) {
                            assert.strictEqual(problemPath.pop(), name);
                        }
                    }
                    return true;
                }
                if (!problemPath) {
                    return false;
                }
                // Since aNameCount !== bNameCount, we need to find some name that's
                // missing in aNames but present in bNames, or vice-versa.
                var seenNames = Object.create(null);
                for (i = 0; i < aNameCount; ++i) {
                    seenNames[aNames[i]] = true;
                }
                for (i = 0; i < bNameCount; ++i) {
                    name = bNames[i];
                    if (!hasOwn.call(seenNames, name)) {
                        problemPath.push(name);
                        return false;
                    }
                    delete seenNames[name];
                }
                for (name in seenNames) {
                    problemPath.push(name);
                    break;
                }
                return false;
            }
        })(types = ast.types || (ast.types = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/// <reference path="types.ts" />
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var types;
        (function (types) {
            var assert = lib.utils.assert;
            var Op = Object.prototype;
            var hasOwn = Op.hasOwnProperty;
            var isArray = types.builtInTypes["array"];
            var isNumber = types.builtInTypes["number"];
            var Ap = Array.prototype;
            var slice = Array.prototype.slice;
            var map = Array.prototype.map;
            var Path = (function () {
                function Path(value, parentPath, name) {
                    assert.ok(this instanceof Path);
                    if (parentPath) {
                        assert.ok(parentPath instanceof Path);
                    }
                    else {
                        parentPath = null;
                        name = null;
                    }
                    // The value encapsulated by this Path, generally equal to
                    // parentPath.value[name] if we have a parentPath.
                    this.value = value;
                    // The immediate parent Path of this Path.
                    this.parentPath = parentPath;
                    // The name of the property of parentPath.value through which this
                    // Path's value was reached.
                    this.name = name;
                    // Calling path.get("child") multiple times always returns the same
                    // child Path object, for both performance and consistency reasons.
                    this.__childCache = null;
                }
                // This method is designed to be overridden by subclasses that need to
                // handle missing properties, etc.
                Path.prototype.getValueProperty = function (name) {
                    return this.value[name];
                };
                Path.prototype.get = function (name) {
                    var path = this;
                    var names = arguments;
                    var count = names.length;
                    for (var i = 0; i < count; ++i) {
                        path = getChildPath(path, names[i]);
                    }
                    return path;
                };
                Path.prototype.each = function (callback, context) {
                    var childPaths = [];
                    var len = this.value.length;
                    var i = 0;
                    for (var i = 0; i < len; ++i) {
                        if (hasOwn.call(this.value, i)) {
                            childPaths[i] = this.get(i);
                        }
                    }
                    // Invoke the callback on just the original child paths, regardless of
                    // any modifications made to the array by the callback. I chose these
                    // semantics over cleverly invoking the callback on new elements because
                    // this way is much easier to reason about.
                    context = context || this;
                    for (i = 0; i < len; ++i) {
                        if (hasOwn.call(childPaths, i)) {
                            callback.call(context, childPaths[i]);
                        }
                    }
                };
                Path.prototype.map = function (callback, context) {
                    var result = [];
                    this.each(function (childPath) {
                        result.push(callback.call(this, childPath));
                    }, context);
                    return result;
                };
                Path.prototype.filter = function (callback, context) {
                    var result = [];
                    this.each(function (childPath) {
                        if (callback.call(this, childPath)) {
                            result.push(childPath);
                        }
                    }, context);
                    return result;
                };
                Path.prototype.shift = function () {
                    var move = getMoves(this, -1);
                    var result = this.value.shift();
                    move();
                    return result;
                };
                Path.prototype.unshift = function (node) {
                    var move = getMoves(this, arguments.length);
                    var result = this.value.unshift.apply(this.value, arguments);
                    move();
                    return result;
                };
                Path.prototype.push = function (node) {
                    isArray.assert(this.value);
                    delete getChildCache(this).length;
                    return this.value.push.apply(this.value, arguments);
                };
                Path.prototype.pop = function () {
                    isArray.assert(this.value);
                    var cache = getChildCache(this);
                    delete cache[this.value.length - 1];
                    delete cache.length;
                    return this.value.pop();
                };
                Path.prototype.insertAt = function (index, node) {
                    var argc = arguments.length;
                    var move = getMoves(this, argc - 1, index);
                    if (move === emptyMoves) {
                        return this;
                    }
                    index = Math.max(index, 0);
                    for (var i = 1; i < argc; ++i) {
                        this.value[index + i - 1] = arguments[i];
                    }
                    move();
                    return this;
                };
                Path.prototype.insertBefore = function (node) {
                    var pp = this.parentPath;
                    var argc = arguments.length;
                    var insertAtArgs = [this.name];
                    for (var i = 0; i < argc; ++i) {
                        insertAtArgs.push(arguments[i]);
                    }
                    return pp.insertAt.apply(pp, insertAtArgs);
                };
                Path.prototype.insertAfter = function (node) {
                    var pp = this.parentPath;
                    var argc = arguments.length;
                    var insertAtArgs = [this.name + 1];
                    for (var i = 0; i < argc; ++i) {
                        insertAtArgs.push(arguments[i]);
                    }
                    return pp.insertAt.apply(pp, insertAtArgs);
                };
                Path.prototype.replace = function (replacement) {
                    var results = [];
                    var parentValue = this.parentPath.value;
                    var parentCache = getChildCache(this.parentPath);
                    var count = arguments.length;
                    repairRelationshipWithParent(this);
                    if (isArray.check(parentValue)) {
                        var originalLength = parentValue.length;
                        var move = getMoves(this.parentPath, count - 1, this.name + 1);
                        var spliceArgs = [this.name, 1];
                        for (var i = 0; i < count; ++i) {
                            spliceArgs.push(arguments[i]);
                        }
                        var splicedOut = parentValue.splice.apply(parentValue, spliceArgs);
                        assert.strictEqual(splicedOut[0], this.value);
                        assert.strictEqual(parentValue.length, originalLength - 1 + count);
                        move();
                        if (count === 0) {
                            delete this.value;
                            delete parentCache[this.name];
                            this.__childCache = null;
                        }
                        else {
                            assert.strictEqual(parentValue[this.name], replacement);
                            if (this.value !== replacement) {
                                this.value = replacement;
                                this.__childCache = null;
                            }
                            for (i = 0; i < count; ++i) {
                                results.push(this.parentPath.get(this.name + i));
                            }
                            assert.strictEqual(results[0], this);
                        }
                    }
                    else if (count === 1) {
                        if (this.value !== replacement) {
                            this.__childCache = null;
                        }
                        this.value = parentValue[this.name] = replacement;
                        results.push(this);
                    }
                    else if (count === 0) {
                        delete parentValue[this.name];
                        delete this.value;
                        this.__childCache = null;
                    }
                    else {
                        assert.ok(false, "Could not replace path");
                    }
                    return results;
                };
                return Path;
            })();
            types.Path = Path;
            function getChildCache(path) {
                // Lazily create the child cache. This also cheapens cache
                // invalidation, since you can just reset path.__childCache to null.
                return path.__childCache || (path.__childCache = Object.create(null));
            }
            function getChildPath(path, name) {
                var cache = getChildCache(path);
                var actualChildValue = path.getValueProperty(name);
                var childPath = cache[name];
                if (!hasOwn.call(cache, name) || childPath.value !== actualChildValue) {
                    childPath = cache[name] = new path.constructor(actualChildValue, path, name);
                }
                return childPath;
            }
            function emptyMoves() {
            }
            function getMoves(path, offset, start, end) {
                isArray.assert(path.value);
                if (offset === 0) {
                    return emptyMoves;
                }
                var length = path.value.length;
                if (length < 1) {
                    return emptyMoves;
                }
                var argc = arguments.length;
                if (argc === 2) {
                    start = 0;
                    end = length;
                }
                else if (argc === 3) {
                    start = Math.max(start, 0);
                    end = length;
                }
                else {
                    start = Math.max(start, 0);
                    end = Math.min(end, length);
                }
                isNumber.assert(start);
                isNumber.assert(end);
                var moves = Object.create(null);
                var cache = getChildCache(path);
                for (var i = start; i < end; ++i) {
                    if (hasOwn.call(path.value, i)) {
                        var childPath = path.get(i);
                        assert.strictEqual(childPath.name, i);
                        var newIndex = i + offset;
                        childPath.name = newIndex;
                        moves[newIndex] = childPath;
                        delete cache[i];
                    }
                }
                delete cache.length;
                return function () {
                    for (var newIndex in moves) {
                        var childPath = moves[newIndex];
                        assert.strictEqual(childPath.name, +newIndex);
                        cache[newIndex] = childPath;
                        path.value[newIndex] = childPath.value;
                    }
                };
            }
            function repairRelationshipWithParent(path) {
                assert.ok(path instanceof Path);
                var pp = path.parentPath;
                if (!pp) {
                    // Orphan paths have no relationship to repair.
                    return path;
                }
                var parentValue = pp.value;
                var parentCache = getChildCache(pp);
                // Make sure parentCache[path.name] is populated.
                if (parentValue[path.name] === path.value) {
                    parentCache[path.name] = path;
                }
                else if (isArray.check(parentValue)) {
                    // Something caused path.name to become out of date, so attempt to
                    // recover by searching for path.value in parentValue.
                    var i = parentValue.indexOf(path.value);
                    if (i >= 0) {
                        parentCache[path.name = i] = path;
                    }
                }
                else {
                    // If path.value disagrees with parentValue[path.name], and
                    // path.name is not an array index, let path.value become the new
                    // parentValue[path.name] and update parentCache accordingly.
                    parentValue[path.name] = path.value;
                    parentCache[path.name] = path;
                }
                assert.strictEqual(parentValue[path.name], path.value);
                assert.strictEqual(path.parentPath.get(path.name), path);
                return path;
            }
        })(types = ast.types || (ast.types = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
var __extends = this.__extends || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    __.prototype = b.prototype;
    d.prototype = new __();
};
/// <reference path="path.ts" />
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var types;
        (function (types) {
            var assert = lib.utils.assert;
            var n = types.namedTypes;
            var b = types.builders;
            var isNumber = types.builtInTypes["number"];
            var isArray = types.builtInTypes["array"];
            var NodePath = (function (_super) {
                __extends(NodePath, _super);
                function NodePath(value, parentPath, name) {
                    assert.ok(this instanceof NodePath);
                    _super.call(this, value, parentPath, name);
                }
                NodePath.prototype.replace = function () {
                    delete this.node;
                    delete this.parent;
                    delete this.scope;
                    return types.Path.prototype.replace.apply(this, arguments);
                };
                NodePath.prototype.prune = function () {
                    var remainingNodePath = this.parent;
                    this.replace();
                    return cleanUpNodesAfterPrune(remainingNodePath);
                };
                // The value of the first ancestor Path whose value is a Node.
                NodePath.prototype._computeNode = function () {
                    var value = this.value;
                    if (n["Node"].check(value)) {
                        return value;
                    }
                    var pp = this.parentPath;
                    return pp && pp.node || null;
                };
                // The first ancestor Path whose value is a Node distinct from this.node.
                NodePath.prototype._computeParent = function () {
                    var value = this.value;
                    var pp = this.parentPath;
                    if (!n["Node"].check(value)) {
                        while (pp && !n["Node"].check(pp.value)) {
                            pp = pp.parentPath;
                        }
                        if (pp) {
                            pp = pp.parentPath;
                        }
                    }
                    while (pp && !n["Node"].check(pp.value)) {
                        pp = pp.parentPath;
                    }
                    return pp || null;
                };
                // The closest enclosing scope that governs this node.
                NodePath.prototype._computeScope = function () {
                    var value = this.value;
                    var pp = this.parentPath;
                    var scope = pp && pp.scope;
                    if (n["Node"].check(value) && types.Scope.isEstablishedBy(value)) {
                        scope = new types.Scope(this, scope);
                    }
                    return scope || null;
                };
                NodePath.prototype.getValueProperty = function (name) {
                    return types.getFieldValue(this.value, name);
                };
                NodePath.prototype.canBeFirstInStatement = function () {
                    var node = this.node;
                    return !n["FunctionExpression"].check(node) && !n["ObjectExpression"].check(node);
                };
                NodePath.prototype.firstInStatement = function () {
                    return firstInStatement(this);
                };
                /**
                 * Determine whether this.node needs to be wrapped in parentheses in order
                 * for a parser to reproduce the same local AST structure.
                 *
                 * For instance, in the expression `(1 + 2) * 3`, the BinaryExpression
                 * whose operator is "+" needs parentheses, because `1 + 2 * 3` would
                 * parse differently.
                 *
                 * If assumeExpressionContext === true, we don't worry about edge cases
                 * like an anonymous FunctionExpression appearing lexically first in its
                 * enclosing statement and thus needing parentheses to avoid being parsed
                 * as a FunctionDeclaration with a missing name.
                 */
                NodePath.prototype.needsParens = function (assumeExpressionContext) {
                    var pp = this.parentPath;
                    if (!pp) {
                        return false;
                    }
                    var node = this.value;
                    // Only expressions need parentheses.
                    if (!n["Expression"].check(node)) {
                        return false;
                    }
                    // Identifiers never need parentheses.
                    if (node.type === "Identifier") {
                        return false;
                    }
                    while (!n["Node"].check(pp.value)) {
                        pp = pp.parentPath;
                        if (!pp) {
                            return false;
                        }
                    }
                    var parent = pp.value;
                    switch (node.type) {
                        case "UnaryExpression":
                        case "SpreadElement":
                        case "SpreadProperty":
                            return parent.type === "MemberExpression" && this.name === "object" && parent.object === node;
                        case "BinaryExpression":
                        case "LogicalExpression":
                            switch (parent.type) {
                                case "CallExpression":
                                    return this.name === "callee" && parent.callee === node;
                                case "UnaryExpression":
                                case "SpreadElement":
                                case "SpreadProperty":
                                    return true;
                                case "MemberExpression":
                                    return this.name === "object" && parent.object === node;
                                case "BinaryExpression":
                                case "LogicalExpression":
                                    var po = parent.operator;
                                    var pp = PRECEDENCE[po];
                                    var no = node.operator;
                                    var np = PRECEDENCE[no];
                                    if (pp > np) {
                                        return true;
                                    }
                                    if (pp === np && this.name === "right") {
                                        assert.strictEqual(parent.right, node);
                                        return true;
                                    }
                                default:
                                    return false;
                            }
                        case "SequenceExpression":
                            switch (parent.type) {
                                case "ForStatement":
                                    // Although parentheses wouldn't hurt around sequence
                                    // expressions in the head of for loops, traditional style
                                    // dictates that e.g. i++, j++ should not be wrapped with
                                    // parentheses.
                                    return false;
                                case "ExpressionStatement":
                                    return this.name !== "expression";
                                default:
                                    // Otherwise err on the side of overparenthesization, adding
                                    // explicit exceptions above if this proves overzealous.
                                    return true;
                            }
                        case "YieldExpression":
                            switch (parent.type) {
                                case "BinaryExpression":
                                case "LogicalExpression":
                                case "UnaryExpression":
                                case "SpreadElement":
                                case "SpreadProperty":
                                case "CallExpression":
                                case "MemberExpression":
                                case "NewExpression":
                                case "ConditionalExpression":
                                case "YieldExpression":
                                    return true;
                                default:
                                    return false;
                            }
                        case "Literal":
                            return parent.type === "MemberExpression" && isNumber.check(node.value) && this.name === "object" && parent.object === node;
                        case "AssignmentExpression":
                        case "ConditionalExpression":
                            switch (parent.type) {
                                case "UnaryExpression":
                                case "SpreadElement":
                                case "SpreadProperty":
                                case "BinaryExpression":
                                case "LogicalExpression":
                                    return true;
                                case "CallExpression":
                                    return this.name === "callee" && parent.callee === node;
                                case "ConditionalExpression":
                                    return this.name === "test" && parent.test === node;
                                case "MemberExpression":
                                    return this.name === "object" && parent.object === node;
                                default:
                                    return false;
                            }
                        default:
                            if (parent.type === "NewExpression" && this.name === "callee" && parent.callee === node) {
                                return containsCallExpression(node);
                            }
                    }
                    if (assumeExpressionContext !== true && !this.canBeFirstInStatement() && this.firstInStatement())
                        return true;
                    return false;
                };
                return NodePath;
            })(types.Path);
            types.NodePath = NodePath;
            Object.defineProperties(NodePath.prototype, {
                node: {
                    get: function () {
                        Object.defineProperty(this, "node", {
                            configurable: true,
                            value: this._computeNode()
                        });
                        return this.node;
                    }
                },
                parent: {
                    get: function () {
                        Object.defineProperty(this, "parent", {
                            configurable: true,
                            value: this._computeParent()
                        });
                        return this.parent;
                    }
                },
                scope: {
                    get: function () {
                        Object.defineProperty(this, "scope", {
                            configurable: true,
                            value: this._computeScope()
                        });
                        return this.scope;
                    }
                }
            });
            function isBinary(node) {
                return n["BinaryExpression"].check(node) || n["LogicalExpression"].check(node);
            }
            function isUnaryLike(node) {
                return n["UnaryExpression"].check(node) || (n["SpreadElement"] && n["SpreadElement"].check(node));
            }
            var PRECEDENCE = {};
            [["||"], ["&&"], ["|"], ["^"], ["&"], ["==", "===", "!=", "!=="], ["<", ">", "<=", ">=", "in", "instanceof"], [">>", "<<", ">>>"], ["+", "-"], ["*", "/", "%"]].forEach(function (tier, i) {
                tier.forEach(function (op) {
                    PRECEDENCE[op] = i;
                });
            });
            function containsCallExpression(node) {
                if (n["CallExpression"].check(node)) {
                    return true;
                }
                if (isArray.check(node)) {
                    return node.some(containsCallExpression);
                }
                if (n["Node"].check(node)) {
                    return types.someField(node, function (name, child) {
                        return containsCallExpression(child);
                    });
                }
                return false;
            }
            function firstInStatement(path) {
                for (var node, parent; path.parent; path = path.parent) {
                    node = path.node;
                    parent = path.parent.node;
                    if (n["BlockStatement"].check(parent) && path.parent.name === "body" && path.name === 0) {
                        assert.strictEqual(parent.body[0], node);
                        return true;
                    }
                    if (n["ExpressionStatement"].check(parent) && path.name === "expression") {
                        assert.strictEqual(parent.expression, node);
                        return true;
                    }
                    if (n["SequenceExpression"].check(parent) && path.parent.name === "expressions" && path.name === 0) {
                        assert.strictEqual(parent.expressions[0], node);
                        continue;
                    }
                    if (n["CallExpression"].check(parent) && path.name === "callee") {
                        assert.strictEqual(parent.callee, node);
                        continue;
                    }
                    if (n["MemberExpression"].check(parent) && path.name === "object") {
                        assert.strictEqual(parent.object, node);
                        continue;
                    }
                    if (n["ConditionalExpression"].check(parent) && path.name === "test") {
                        assert.strictEqual(parent.test, node);
                        continue;
                    }
                    if (isBinary(parent) && path.name === "left") {
                        assert.strictEqual(parent.left, node);
                        continue;
                    }
                    if (n["UnaryExpression"].check(parent) && !parent.prefix && path.name === "argument") {
                        assert.strictEqual(parent.argument, node);
                        continue;
                    }
                    return false;
                }
                return true;
            }
            /**
             * Pruning certain nodes will result in empty or incomplete nodes, here we clean those nodes up.
             */
            function cleanUpNodesAfterPrune(remainingNodePath) {
                if (n["VariableDeclaration"].check(remainingNodePath.node)) {
                    var declarations = remainingNodePath.get('declarations').value;
                    if (!declarations || declarations.length === 0) {
                        return remainingNodePath.prune();
                    }
                }
                else if (n["ExpressionStatement"].check(remainingNodePath.node)) {
                    if (!remainingNodePath.get('expression').value) {
                        return remainingNodePath.prune();
                    }
                }
                else if (n["IfStatement"].check(remainingNodePath.node)) {
                    cleanUpIfStatementAfterPrune(remainingNodePath);
                }
                return remainingNodePath;
            }
            function cleanUpIfStatementAfterPrune(ifStatement) {
                var testExpression = ifStatement.get('test').value;
                var alternate = ifStatement.get('alternate').value;
                var consequent = ifStatement.get('consequent').value;
                if (!consequent && !alternate) {
                    var testExpressionStatement = b["expressionStatement"](testExpression);
                    ifStatement.replace(testExpressionStatement);
                }
                else if (!consequent && alternate) {
                    var negatedTestExpression = b["unaryExpression"]('!', testExpression, true);
                    if (n["UnaryExpression"].check(testExpression) && testExpression.operator === '!') {
                        negatedTestExpression = testExpression.argument;
                    }
                    ifStatement.get("test").replace(negatedTestExpression);
                    ifStatement.get("consequent").replace(alternate);
                    ifStatement.get("alternate").replace();
                }
            }
        })(types = ast.types || (ast.types = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/// <reference path="types.ts" />
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var types;
        (function (types) {
            var Node = esprima.Syntax.Node;
            var assert = lib.utils.assert;
            var n = types.namedTypes;
            var b = types.builders;
            var isNumber = types.builtInTypes["number"];
            var isArray = types.builtInTypes["array"];
            var Node = types.namedTypes["Node"];
            var isObject = types.builtInTypes["object"];
            var isFunction = types.builtInTypes["function"];
            var hasOwn = Object.prototype.hasOwnProperty;
            var undefined;
            var PathVisitor = (function () {
                function PathVisitor() {
                    assert.ok(this instanceof PathVisitor);
                    this._reusableContextStack = [];
                    this._methodNameTable = computeMethodNameTable(this);
                    this.Context = makeContextConstructor(this);
                }
                PathVisitor.fromMethodsObject = function (methods) {
                    if (methods instanceof PathVisitor) {
                        return methods;
                    }
                    if (!isObject.check(methods)) {
                        // An empty visitor?
                        return new PathVisitor;
                    }
                    function Visitor() {
                        assert.ok(this instanceof Visitor);
                        PathVisitor.call(this);
                    }
                    var Vp = Visitor.prototype = Object.create(PathVisitor.prototype);
                    Vp.constructor = Visitor;
                    extend(Vp, methods);
                    extend(Visitor, PathVisitor);
                    return new Visitor;
                };
                PathVisitor.visit = function (node, methods) {
                    var visitor = PathVisitor.fromMethodsObject(methods);
                    if (node instanceof types.NodePath) {
                        visitor.visit(node);
                        return node.value;
                    }
                    var rootPath = new types.NodePath({ root: node });
                    visitor.visit(rootPath.get("root"));
                    return rootPath.value.root;
                };
                PathVisitor.prototype.visit = function (path) {
                    if (this instanceof this.Context) {
                        // If we somehow end up calling context.visit, then we need to
                        // re-invoke the .visit method against context.visitor.
                        return this.visitor.visit(path);
                    }
                    assert.ok(path instanceof types.NodePath);
                    var value = path.value;
                    var methodName = Node.check(value) && this._methodNameTable[value.type];
                    if (methodName) {
                        var context = this.acquireContext(path);
                        try {
                            context.invokeVisitorMethod(methodName);
                        }
                        finally {
                            this.releaseContext(context);
                        }
                    }
                    else {
                        // If there was no visitor method to call, visit the children of
                        // this node generically.
                        visitChildren(path, this);
                    }
                };
                PathVisitor.prototype.acquireContext = function (path) {
                    if (this._reusableContextStack.length === 0) {
                        return new this.Context(path);
                    }
                    return this._reusableContextStack.pop().reset(path);
                };
                PathVisitor.prototype.releaseContext = function (context) {
                    assert.ok(context instanceof this.Context);
                    this._reusableContextStack.push(context);
                    context.currentPath = null;
                };
                return PathVisitor;
            })();
            types.PathVisitor = PathVisitor;
            function computeMethodNameTable(visitor) {
                var typeNames = Object.create(null);
                for (var methodName in visitor) {
                    if (/^visit[A-Z]/.test(methodName)) {
                        typeNames[methodName.slice("visit".length)] = true;
                    }
                }
                var supertypeTable = types.computeSupertypeLookupTable(typeNames);
                var methodNameTable = Object.create(null);
                typeNames = Object.keys(supertypeTable);
                var typeNameCount = typeNames.length;
                for (var i = 0; i < typeNameCount; ++i) {
                    var typeName = typeNames[i];
                    methodName = "visit" + supertypeTable[typeName];
                    if (isFunction.check(visitor[methodName])) {
                        methodNameTable[typeName] = methodName;
                    }
                }
                return methodNameTable;
            }
            function extend(target, source) {
                for (var property in source) {
                    if (hasOwn.call(source, property)) {
                        target[property] = source[property];
                    }
                }
                return target;
            }
            function visitChildren(path, visitor) {
                assert.ok(path instanceof types.NodePath);
                assert.ok(visitor instanceof PathVisitor);
                var value = path.value;
                if (isArray.check(value)) {
                    path.each(visitor.visit, visitor);
                }
                else if (!isObject.check(value)) {
                }
                else {
                    var childNames = types.getFieldNames(value);
                    var childCount = childNames.length;
                    var childPaths = [];
                    for (var i = 0; i < childCount; ++i) {
                        var childName = childNames[i];
                        if (!hasOwn.call(value, childName)) {
                            value[childName] = types.getFieldValue(value, childName);
                        }
                        childPaths.push(path.get(childName));
                    }
                    for (var i = 0; i < childCount; ++i) {
                        visitor.visit(childPaths[i]);
                    }
                }
            }
            function makeContextConstructor(visitor) {
                function Context(path) {
                    assert.ok(this instanceof Context);
                    assert.ok(this instanceof PathVisitor);
                    assert.ok(path instanceof types.NodePath);
                    Object.defineProperty(this, "visitor", {
                        value: visitor,
                        writable: false,
                        enumerable: true,
                        configurable: false
                    });
                    this.currentPath = path;
                    this.needToCallTraverse = true;
                    Object.seal(this);
                }
                assert.ok(visitor instanceof PathVisitor);
                // Note that the visitor object is the prototype of Context.prototype,
                // so all visitor methods are inherited by context objects.
                var Cp = Context.prototype = Object.create(visitor);
                Cp.constructor = Context;
                extend(Cp, sharedContextProtoMethods);
                return Context;
            }
            // Every PathVisitor has a different this.Context constructor and
            // this.Context.prototype object, but those prototypes can all use the
            // same reset, invokeVisitorMethod, and traverse function objects.
            var sharedContextProtoMethods = Object.create(null);
            sharedContextProtoMethods.reset = function reset(path) {
                assert.ok(this instanceof this.Context);
                assert.ok(path instanceof types.NodePath);
                this.currentPath = path;
                this.needToCallTraverse = true;
                return this;
            };
            sharedContextProtoMethods.invokeVisitorMethod = function invokeVisitorMethod(methodName) {
                assert.ok(this instanceof this.Context);
                assert.ok(this.currentPath instanceof types.NodePath);
                var result = this.visitor[methodName].call(this, this.currentPath);
                if (result === false) {
                    // Visitor methods return false to indicate that they have handled
                    // their own traversal needs, and we should not complain if
                    // this.needToCallTraverse is still true.
                    this.needToCallTraverse = false;
                }
                else if (result !== undefined) {
                    // Any other non-undefined value returned from the visitor method
                    // is interpreted as a replacement value.
                    this.currentPath = this.currentPath.replace(result)[0];
                    if (this.needToCallTraverse) {
                        // If this.traverse still hasn't been called, visit the
                        // children of the replacement node.
                        this.traverse(this.currentPath);
                    }
                }
                assert.strictEqual(this.needToCallTraverse, false, "Must either call this.traverse or return false in " + methodName);
            };
            sharedContextProtoMethods.traverse = function traverse(path, newVisitor) {
                assert.ok(this instanceof this.Context);
                assert.ok(path instanceof types.NodePath);
                assert.ok(this.currentPath instanceof types.NodePath);
                this.needToCallTraverse = false;
                visitChildren(path, PathVisitor.fromMethodsObject(newVisitor || this.visitor));
            };
        })(types = ast.types || (ast.types = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/// <reference path="types.ts" />
var lib;
(function (lib) {
    var ast;
    (function (ast) {
        var types;
        (function (types) {
            var assert = lib.utils.assert;
            var namedTypes = types.namedTypes;
            var Node = namedTypes["Node"];
            var Expression = namedTypes["Expression"];
            var isArray = types.builtInTypes["array"];
            var hasOwn = Object.prototype.hasOwnProperty;
            var b = types.builders;
            var Scope = (function () {
                function Scope(path, parentScope) {
                    // Will be overridden after an instance lazily calls scanScope.
                    this.didScan = false;
                    assert.ok(this instanceof Scope);
                    assert.ok(path instanceof require("./node-path"));
                    ScopeType.assert(path.value);
                    var depth;
                    if (parentScope) {
                        assert.ok(parentScope instanceof Scope);
                        depth = parentScope.depth + 1;
                    }
                    else {
                        parentScope = null;
                        depth = 0;
                    }
                    Object.defineProperties(this, {
                        path: { value: path },
                        node: { value: path.value },
                        isGlobal: { value: !parentScope, enumerable: true },
                        depth: { value: depth },
                        parent: { value: parentScope },
                        bindings: { value: {} }
                    });
                }
                Scope.isEstablishedBy = function (node) {
                    return ScopeType.check(node);
                };
                Scope.prototype.declares = function (name) {
                    this.scan();
                    return hasOwn.call(this.bindings, name);
                };
                Scope.prototype.declareTemporary = function (prefix) {
                    if (prefix) {
                        assert.ok(/^[a-z$_]/i.test(prefix), prefix);
                    }
                    else {
                        prefix = "t$";
                    }
                    // Include this.depth in the name to make sure the name does not
                    // collide with any variables in nested/enclosing scopes.
                    prefix += this.depth.toString(36) + "$";
                    this.scan();
                    var index = 0;
                    while (this.declares(prefix + index)) {
                        ++index;
                    }
                    var name = prefix + index;
                    return this.bindings[name] = types.builders["identifier"](name);
                };
                Scope.prototype.injectTemporary = function (identifier, init) {
                    identifier || (identifier = this.declareTemporary());
                    var bodyPath = this.path.get("body");
                    if (namedTypes["BlockStatement"].check(bodyPath.value)) {
                        bodyPath = bodyPath.get("body");
                    }
                    bodyPath.unshift(b["variableDeclaration"]("var", [b["variableDeclarator"](identifier, init || null)]));
                    return identifier;
                };
                Scope.prototype.scan = function (force) {
                    if (force || !this.didScan) {
                        for (var name in this.bindings) {
                            // Empty out this.bindings, just in cases.
                            delete this.bindings[name];
                        }
                        scanScope(this.path, this.bindings);
                        this.didScan = true;
                    }
                };
                Scope.prototype.getBindings = function () {
                    this.scan();
                    return this.bindings;
                };
                Scope.prototype.lookup = function (name) {
                    for (var scope = this; scope; scope = scope.parent)
                        if (scope.declares(name))
                            break;
                    return scope;
                };
                Scope.prototype.getGlobalScope = function () {
                    var scope = this;
                    while (!scope.isGlobal)
                        scope = scope.parent;
                    return scope;
                };
                return Scope;
            })();
            types.Scope = Scope;
            var scopeTypes = [
                namedTypes["Program"],
                namedTypes["Function"],
                namedTypes["CatchClause"]
            ];
            var ScopeType = types.Type.or.apply(types.Type, scopeTypes);
            function scanScope(path, bindings) {
                var node = path.value;
                ScopeType.assert(node);
                if (namedTypes["CatchClause"].check(node)) {
                    // A catch clause establishes a new scope but the only variable
                    // bound in that scope is the catch parameter. Any other
                    // declarations create bindings in the outer scope.
                    addPattern(path.get("param"), bindings);
                }
                else {
                    recursiveScanScope(path, bindings);
                }
            }
            function recursiveScanScope(path, bindings) {
                var node = path.value;
                if (path.parent && namedTypes["FunctionExpression"].check(path.parent.node) && path.parent.node.id) {
                    addPattern(path.parent.get("id"), bindings);
                }
                if (!node) {
                }
                else if (isArray.check(node)) {
                    path.each(function (childPath) {
                        recursiveScanChild(childPath, bindings);
                    });
                }
                else if (namedTypes["Function"].check(node)) {
                    path.get("params").each(function (paramPath) {
                        addPattern(paramPath, bindings);
                    });
                    recursiveScanChild(path.get("body"), bindings);
                }
                else if (namedTypes["VariableDeclarator"].check(node)) {
                    addPattern(path.get("id"), bindings);
                    recursiveScanChild(path.get("init"), bindings);
                }
                else if (node.type === "ImportSpecifier" || node.type === "ImportNamespaceSpecifier" || node.type === "ImportDefaultSpecifier") {
                    addPattern(node.name ? path.get("name") : path.get("id"), bindings);
                }
                else if (Node.check(node) && !Expression.check(node)) {
                    types.eachField(node, function (name, child) {
                        var childPath = path.get(name);
                        assert.strictEqual(childPath.value, child);
                        recursiveScanChild(childPath, bindings);
                    });
                }
            }
            function recursiveScanChild(path, bindings) {
                var node = path.value;
                if (!node || Expression.check(node)) {
                }
                else if (namedTypes["FunctionDeclaration"].check(node)) {
                    addPattern(path.get("id"), bindings);
                }
                else if (namedTypes["ClassDeclaration"] && namedTypes["ClassDeclaration"].check(node)) {
                    addPattern(path.get("id"), bindings);
                }
                else if (ScopeType.check(node)) {
                    if (namedTypes["CatchClause"].check(node)) {
                        var catchParamName = node.param.name;
                        var hadBinding = hasOwn.call(bindings, catchParamName);
                        // Any declarations that occur inside the catch body that do
                        // not have the same name as the catch parameter should count
                        // as bindings in the outer scope.
                        recursiveScanScope(path.get("body"), bindings);
                        // If a new binding matching the catch parameter name was
                        // created while scanning the catch body, ignore it because it
                        // actually refers to the catch parameter and not the outer
                        // scope that we're currently scanning.
                        if (!hadBinding) {
                            delete bindings[catchParamName];
                        }
                    }
                }
                else {
                    recursiveScanScope(path, bindings);
                }
            }
            function addPattern(patternPath, bindings) {
                var pattern = patternPath.value;
                namedTypes["Pattern"].assert(pattern);
                if (namedTypes["Identifier"].check(pattern)) {
                    if (hasOwn.call(bindings, pattern.name)) {
                        bindings[pattern.name].push(patternPath);
                    }
                    else {
                        bindings[pattern.name] = [patternPath];
                    }
                }
                else if (namedTypes["SpreadElement"] && namedTypes["SpreadElement"].check(pattern)) {
                    addPattern(patternPath.get("argument"), bindings);
                }
            }
        })(types = ast.types || (ast.types = {}));
    })(ast = lib.ast || (lib.ast = {}));
})(lib || (lib = {}));
/// <reference path="types.ts" />
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
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var stacktrace;
        (function (stacktrace) {
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
                        return fToBind.apply(this instanceof NoOp && oThis ? this : oThis, aArgs.concat(Array.prototype.slice.call(arguments)));
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
                Array.prototype.filter = function (callback /*, thisArg*/) {
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
                        }
                        else if (error.stack.match(CHROME_IE_STACK_REGEXP)) {
                            return this.parseV8OrIE(error);
                        }
                        else if (error.stack.match(FIREFOX_SAFARI_STACK_REGEXP)) {
                            return this.parseFFOrSafari(error);
                        }
                        else {
                            throw new utils.Error('Cannot parse given Error object');
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
                        }
                        else {
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
                        if (!e.stacktrace || (e.message.indexOf('\n') > -1 && e.message.split('\n').length > e.stacktrace.split('\n').length)) {
                            return this.parseOpera9(e);
                        }
                        else if (!e.stack) {
                            return this.parseOpera10a(e);
                        }
                        else if (e.stacktrace.indexOf("called from line") < 0) {
                            return this.parseOpera10b(e);
                        }
                        else {
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
                };
            };
            stacktrace.ErrorStackParser = factory(lib.utils.stacktrace.StackFrame);
        })(stacktrace = utils.stacktrace || (utils.stacktrace = {}));
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
/*This is free and unencumbered software released into the public domain.

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
// from https://github.com/stacktracejs/stackframe
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var stacktrace;
        (function (stacktrace) {
            function _isNumber(n) {
                return !isNaN(parseFloat(n)) && isFinite(n);
            }
            var StackFrame = (function () {
                function StackFrame(functionName, args, fileName, lineNumber, columnNumber) {
                    if (functionName !== undefined) {
                        this.setFunctionName(functionName);
                    }
                    if (args !== undefined) {
                        this.setArgs(args);
                    }
                    if (fileName !== undefined) {
                        this.setFileName(fileName);
                    }
                    if (lineNumber !== undefined) {
                        this.setLineNumber(lineNumber);
                    }
                    if (columnNumber !== undefined) {
                        this.setColumnNumber(columnNumber);
                    }
                }
                StackFrame.prototype.getFunctionName = function () {
                    return this.functionName;
                };
                StackFrame.prototype.setFunctionName = function (v) {
                    this.functionName = String(v);
                };
                StackFrame.prototype.getArgs = function () {
                    return this.args;
                };
                StackFrame.prototype.setArgs = function (v) {
                    if (Object.prototype.toString.call(v) !== '[object Array]') {
                        throw new TypeError('Args must be an Array');
                    }
                    this.args = v;
                };
                // NOTE: Property name may be misleading as it includes the path,
                // but it somewhat mirrors V8's JavaScriptStackTraceApi
                // https://code.google.com/p/v8/wiki/JavaScriptStackTraceApi
                StackFrame.prototype.getFileName = function () {
                    return this.fileName;
                };
                StackFrame.prototype.setFileName = function (v) {
                    this.fileName = String(v);
                };
                StackFrame.prototype.getLineNumber = function () {
                    return this.lineNumber;
                };
                StackFrame.prototype.setLineNumber = function (v) {
                    if (!_isNumber(v)) {
                        throw new TypeError('Line Number must be a Number');
                    }
                    this.lineNumber = Number(v);
                };
                StackFrame.prototype.getColumnNumber = function () {
                    return this.columnNumber;
                };
                StackFrame.prototype.setColumnNumber = function (v) {
                    if (!_isNumber(v)) {
                        throw new TypeError('Column Number must be a Number');
                    }
                    this.columnNumber = Number(v);
                };
                StackFrame.prototype.toString = function () {
                    var functionName = this.getFunctionName() || '{anonymous}';
                    var args = '(' + (this.getArgs() || []).join(',') + ')';
                    var fileName = this.getFileName() ? ('@' + this.getFileName()) : '';
                    var lineNumber = _isNumber(this.getLineNumber()) ? (':' + this.getLineNumber()) : '';
                    var columnNumber = _isNumber(this.getColumnNumber()) ? (':' + this.getColumnNumber()) : '';
                    return functionName + args + fileName + lineNumber + columnNumber;
                };
                return StackFrame;
            })();
            stacktrace.StackFrame = StackFrame;
        })(stacktrace = utils.stacktrace || (utils.stacktrace = {}));
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
/*This is free and unencumbered software released into the public domain.

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
// from https://github.com/stacktracejs/stacktrace-gps
/// <reference path="../../../Scripts/typings/es6-promise/es6-promise.d.ts" />
var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var stacktrace;
        (function (stacktrace) {
            var SourceMap = lib.ast.sourcemap;
            /**
             * Create XHR or equivalent object for this environment.
             * @returns XMLHttpRequest, XDomainRequest or ActiveXObject
             * @private
             */
            function _createXMLHTTPObject() {
                var xmlhttp;
                var XMLHttpFactories = [
                    function () {
                        return new XMLHttpRequest();
                    },
                    function () {
                        return new XDomainRequest();
                    },
                    function () {
                        return new ActiveXObject('Msxml2.XMLHTTP');
                    },
                    function () {
                        return new ActiveXObject('Msxml3.XMLHTTP');
                    },
                    function () {
                        return new ActiveXObject('Microsoft.XMLHTTP');
                    }
                ];
                for (var i = 0; i < XMLHttpFactories.length; i++) {
                    try {
                        xmlhttp = XMLHttpFactories[i]();
                        return xmlhttp;
                    }
                    catch (e) {
                    }
                }
            }
            /**
             * Make a X-Domain request to url and callback.
             *
             * @param url [String]
             * @param callback [Function] to callback on completion
             * @param errback [Function] to callback on error
             */
            function _xdr(url, callback, errback) {
                var req = _createXMLHTTPObject();
                if (!req) {
                    errback(new utils.Error('X-Domain request failed because no form of XHR is supported'));
                }
                req.open('get', url);
                req.onerror = errback;
                if (typeof XMLHttpRequest === 'function' || typeof ActiveXObject === 'function') {
                    req.onreadystatechange = function onreadystatechange() {
                        if (req.readyState === 4) {
                            if (req.status >= 200 && req.status < 400) {
                                return callback(req.responseText);
                            }
                            else {
                                errback(new utils.Error('Unable to retrieve ' + url));
                            }
                        }
                    };
                    req.send();
                }
                else {
                    req.onload = function onload() {
                        callback(req.responseText);
                    };
                    // Avoid bug with concurrent requests in XDomainRequest API
                    setTimeout(req.send, 0);
                }
            }
            function _findFunctionName(source, lineNumber, columnNumber) {
                // function {name}({args}) m[1]=name m[2]=args
                var reFunctionDeclaration = /function\s+([^(]*?)\s*\(([^)]*)\)/;
                // {name} = function ({args}) TODO args capture
                var reFunctionExpression = /['"]?([$_A-Za-z][$_A-Za-z0-9]*)['"]?\s*[:=]\s*function\b/;
                // {name} = eval()
                var reFunctionEvaluation = /['"]?([$_A-Za-z][$_A-Za-z0-9]*)['"]?\s*[:=]\s*(?:eval|new Function)\b/;
                var lines = source.split("\n");
                // Walk backwards in the source lines until we find the line which matches one of the patterns above
                var code = '', line, maxLines = Math.min(lineNumber, 20), m, commentPos;
                for (var i = 0; i < maxLines; ++i) {
                    // lineNo is 1-based, source[] is 0-based
                    line = lines[lineNumber - i - 1];
                    commentPos = line.indexOf('//');
                    if (commentPos >= 0) {
                        line = line.substr(0, commentPos);
                    }
                    if (line) {
                        code = line + code;
                        m = reFunctionExpression.exec(code);
                        if (m && m[1]) {
                            return m[1];
                        }
                        m = reFunctionDeclaration.exec(code);
                        if (m && m[1]) {
                            //return m[1] + "(" + (m[2] || "") + ")";
                            return m[1];
                        }
                        m = reFunctionEvaluation.exec(code);
                        if (m && m[1]) {
                            return m[1];
                        }
                    }
                }
                return undefined;
            }
            function _ensureSupportedEnvironment() {
                if (typeof Object.defineProperty !== 'function' || typeof Object.create !== 'function') {
                    throw new utils.Error('Unable to consume source maps in older browsers');
                }
            }
            function _ensureStackFrameIsLegit(stackframe) {
                if (typeof stackframe !== 'object') {
                    throw new TypeError('Given StackFrame is not an object');
                }
                else if (typeof stackframe.fileName !== 'string') {
                    throw new TypeError('Given file name is not a String');
                }
                else if (typeof stackframe.lineNumber !== 'number' || stackframe.lineNumber % 1 !== 0 || stackframe.lineNumber < 1) {
                    throw new TypeError('Given line number must be a positive integer');
                }
                else if (typeof stackframe.columnNumber !== 'number' || stackframe.columnNumber % 1 !== 0 || stackframe.columnNumber < 0) {
                    throw new TypeError('Given column number must be a non-negative integer');
                }
                return true;
            }
            function _findSourceMappingURL(source) {
                var m = /\/\/[#@] ?sourceMappingURL=([^\s'"]+)$/.exec(source);
                if (m && m[1]) {
                    return m[1];
                }
                else {
                    throw new utils.Error('sourceMappingURL not found');
                }
            }
            function _newLocationInfoFromSourceMap(rawSourceMap, lineNumber, columnNumber) {
                return new SourceMap.SourceMapConsumer(rawSourceMap).originalPositionFor({ line: lineNumber, column: columnNumber });
            }
            var factory = function (SourceMap, ES6Promise) {
                return function StackTraceGPS(opts) {
                    this.sourceCache = {};
                    this._get = function _get(location) {
                        return new Promise(function (resolve, reject) {
                            if (this.sourceCache[location]) {
                                resolve(this.sourceCache[location]);
                            }
                            else {
                                _xdr(location, function (source) {
                                    this.sourceCache[location] = source;
                                    resolve(source);
                                }.bind(this), reject);
                            }
                        }.bind(this));
                    };
                    /**
                     * Given location information for a Function definition, return the function name.
                     *
                     * @param stackframe - {StackFrame}-like object
                     *      {fileName: 'path/to/file.js', lineNumber: 100, columnNumber: 5}
                     */
                    this.findFunctionName = function StackTraceGPS$$findFunctionName(stackframe) {
                        return new Promise(function (resolve, reject) {
                            _ensureStackFrameIsLegit(stackframe);
                            this._get(stackframe.fileName).then(function getSourceCallback(source) {
                                resolve(_findFunctionName(source, stackframe.lineNumber, stackframe.columnNumber));
                            }, reject);
                        }.bind(this));
                    };
                    /**
                     * Given a StackFrame, seek source-mapped location and return source-mapped location.
                     *
                     * @param stackframe - {StackFrame}-like object
                     *      {fileName: 'path/to/file.js', lineNumber: 100, columnNumber: 5}
                     */
                    this.getMappedLocation = function StackTraceGPS$$sourceMap(stackframe) {
                        return new Promise(function (resolve, reject) {
                            _ensureSupportedEnvironment();
                            _ensureStackFrameIsLegit(stackframe);
                            this._get(stackframe.fileName).then(function (source) {
                                this._get(_findSourceMappingURL(source)).then(function (map) {
                                    var lineNumber = stackframe.lineNumber;
                                    var columnNumber = stackframe.columnNumber;
                                    resolve(_newLocationInfoFromSourceMap(map, lineNumber, columnNumber));
                                }, reject)['catch'](reject);
                            }.bind(this), reject)['catch'](reject);
                        }.bind(this));
                    };
                };
            };
            stacktrace.StackTraceGPS = factory(SourceMap, Promise);
        })(stacktrace = utils.stacktrace || (utils.stacktrace = {}));
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
            self.onmessage = function (code) {
                eval(code.data);
            };
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
//# sourceMappingURL=main.js.map