
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

module lib.utils {
    export module stacktrace {

        import SourceMap = lib.ast.sourcemap;

        /**
         * Create XHR or equivalent object for this environment.
         * @returns XMLHttpRequest, XDomainRequest or ActiveXObject
         * @private
         */
        function _createXMLHTTPObject() : XMLHttpRequest {
            var xmlhttp: XMLHttpRequest;
            var XMLHttpFactories = [
                function () {
                    return new XMLHttpRequest();
                }, function () {
                    return new XDomainRequest();
                }, function () {
                    return new ActiveXObject('Msxml2.XMLHTTP');
                }, function () {
                    return new ActiveXObject('Msxml3.XMLHTTP');
                }, function () {
                    return new ActiveXObject('Microsoft.XMLHTTP');
                }
            ];
            for (var i = 0; i < XMLHttpFactories.length; i++) {
                try {
                    xmlhttp = XMLHttpFactories[i]();
                    return xmlhttp;
                } catch (e) {
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
                errback(new Error('X-Domain request failed because no form of XHR is supported'));
            }

            req.open('get', url);
            req.onerror = errback;

            if (typeof XMLHttpRequest === 'function' || typeof ActiveXObject === 'function') {
                req.onreadystatechange = function onreadystatechange() {
                    if (req.readyState === 4) {
                        if (req.status >= 200 && req.status < 400) {
                            return callback(req.responseText);
                        } else {
                            errback(new Error('Unable to retrieve ' + url));
                        }
                    }
                };
                req.send();
            } else {
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
                throw new Error('Unable to consume source maps in older browsers');
            }
        }

        function _ensureStackFrameIsLegit(stackframe) {
            if (typeof stackframe !== 'object') {
                throw new TypeError('Given StackFrame is not an object');
            } else if (typeof stackframe.fileName !== 'string') {
                throw new TypeError('Given file name is not a String');
            } else if (typeof stackframe.lineNumber !== 'number' || stackframe.lineNumber % 1 !== 0 || stackframe.lineNumber < 1) {
                throw new TypeError('Given line number must be a positive integer');
            } else if (typeof stackframe.columnNumber !== 'number' || stackframe.columnNumber % 1 !== 0 || stackframe.columnNumber < 0) {
                throw new TypeError('Given column number must be a non-negative integer');
            }
            return true;
        }

        function _findSourceMappingURL(source) {
            var m = /\/\/[#@] ?sourceMappingURL=([^\s'"]+)$/.exec(source);
            if (m && m[1]) {
                return m[1];
            } else {
                throw new Error('sourceMappingURL not found');
            }
        }

        function _newLocationInfoFromSourceMap(rawSourceMap, lineNumber, columnNumber) {
            return new SourceMap.SourceMapConsumer(rawSourceMap)
                .originalPositionFor({ line: lineNumber, column: columnNumber });
        }

        var factory = function (SourceMap, ES6Promise) {
            return function StackTraceGPS(opts) {
                this.sourceCache = {};

                this._get = function _get(location) {
                    return new Promise(function (resolve, reject) {
                        if (this.sourceCache[location]) {
                            resolve(this.sourceCache[location]);
                        } else {
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
        }
        export var StackTraceGPS = factory(SourceMap, Promise);
    }
}