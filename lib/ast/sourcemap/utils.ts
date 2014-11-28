﻿

module lib.ast.sourcemap {
    export module utils {

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
        export function getArg(aArgs, aName, aDefaultValue?) {
            if (aName in aArgs) {
                return aArgs[aName];
            } else if (arguments.length === 3) {
                return aDefaultValue;
            } else {
                throw new Error('"' + aName + '" is a required argument.');
            }
        }


        var urlRegexp = /^(?:([\w+\-.]+):)?\/\/(?:(\w+:\w+)@)?([\w.]*)(?::(\d+))?(\S*)$/;
        var dataUrlRegexp = /^data:.+\,.+$/;

        export function urlParse(aUrl) {
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


        export function urlGenerate(aParsedUrl) {
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
                url += ":" + aParsedUrl.port
            }
            if (aParsedUrl.path) {
                url += aParsedUrl.path;
            }
            return url;
        }


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
        export function normalize(aPath) {
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
                } else if (part === '..') {
                    up++;
                } else if (up > 0) {
                    if (part === '') {
                        // The first part is blank if the path is absolute. Trying to go
                        // above the root is a no-op. Therefore we can remove all '..' parts
                        // directly after the root.
                        parts.splice(i + 1, up);
                        up = 0;
                    } else {
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
        export function join(aRoot, aPath) {
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

            var joined = aPath.charAt(0) === '/'
                ? aPath
                : normalize(aRoot.replace(/\/+$/, '') + '/' + aPath);

            if (aRootUrl) {
                aRootUrl.path = joined;
                return urlGenerate(aRootUrl);
            }
            return joined;
        }


        /**
         * Make a path relative to a URL or another path.
         *
         * @param aRoot The root path or URL.
         * @param aPath The path or URL to be made relative to aRoot.
         */
        export function relative(aRoot, aPath) {
            if (aRoot === "") {
                aRoot = ".";
            }

            aRoot = aRoot.replace(/\/$/, '');

            // XXX: It is possible to remove this block, and the tests still pass!
            var url = urlParse(aRoot);
            if (aPath.charAt(0) == "/" && url && url.path == "/") {
                return aPath.slice(1);
            }

            return aPath.indexOf(aRoot + '/') === 0
                ? aPath.substr(aRoot.length + 1)
                : aPath;
        }


        /**
         * Because behavior goes wacky when you set `__proto__` on objects, we
         * have to prefix all the strings in our set with an arbitrary character.
         *
         * See https://github.com/mozilla/source-map/pull/31 and
         * https://github.com/mozilla/source-map/issues/30
         *
         * @param String aStr
         */
        export function toSetString(aStr) {
            return '$' + aStr;
        }


        export function fromSetString(aStr) {
            return aStr.substr(1);
        }


        function strcmp(aStr1, aStr2) {
            var s1 = aStr1 || "";
            var s2 = aStr2 || "";
            var d1: number = lib.utils.castTo<number>(s1 > s2);
            var d2: number = lib.utils.castTo<number>(s1 < s2);
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
        export function compareByOriginalPositions(mappingA, mappingB, onlyCompareOriginal) {
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
        };

        /**
         * Comparator between two mappings where the generated positions are
         * compared.
         *
         * Optionally pass in `true` as `onlyCompareGenerated` to consider two
         * mappings with the same generated line and column, but different
         * source/name/original line and column the same. Useful when searching for a
         * mapping with a stubbed out mapping.
         */
        export function compareByGeneratedPositions(mappingA, mappingB, onlyCompareGenerated?) {
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
        };
    }
}
