/*
 The MIT License (MIT)

 Copyright (c) <year> <copyright holders>

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */
// from https://github.com/substack/node-falafel
module lib.ast {
    export module importer {

        import parse = lib.ast.esprima.parse;
        var objectKeys = Object.keys || function (obj) {
                var keys = [];
                for (var key in obj) keys.push(key);
                return keys;
            };
        var forEach = function (xs, fn) {
            if (xs.forEach) return xs.forEach(fn);
            for (var i = 0; i < xs.length; i++) {
                fn.call(xs, xs[i], i, xs);
            }
        };

        import isArray = lib.utils.isArray;
        export function flafel(src, opts, fn) {
            if (typeof opts === 'function') {
                fn = opts;
                opts = {};
            }
            if (typeof src === 'object') {
                opts = src;
                src = opts.source;
                delete opts.source;
            }
            src = src === undefined ? opts.source : src;
            opts.range = true;
            if (typeof src !== 'string') src = String(src);

            var ast = parse(src, opts);

            var result = {
                chunks : src.split(''),
                toString : function () { return result.chunks.join('') },
                inspect : function () { return result.toString() }
            };
            var index = 0;

            (function walk (node, parent) {
                insertHelpers(node, parent, result.chunks);

                forEach(objectKeys(node), function (key) {
                    if (key === 'parent') return;

                    var child = node[key];
                    if (isArray(child)) {
                        forEach(child, function (c) {
                            if (c && typeof c.type === 'string') {
                                walk(c, node);
                            }
                        });
                    }
                    else if (child && typeof child.type === 'string') {
                        insertHelpers(child, node, result.chunks);
                        walk(child, node);
                    }
                });
                fn(node);
            })(ast, undefined);

            return result;
        };

        function insertHelpers (node, parent, chunks) {
            if (!node.range) return;

            node.parent = parent;

            node.source = function () {
                return chunks.slice(
                    node.range[0], node.range[1]
                ).join('');
            };

            if (node.update && typeof node.update === 'object') {
                var prev = node.update;
                forEach(objectKeys(prev), function (key) {
                    update[key] = prev[key];
                });
                node.update = update;
            }
            else {
                node.update = update;
            }

            function update (s) {
                chunks[node.range[0]] = s;
                for (var i = node.range[0] + 1; i < node.range[1]; i++) {
                    chunks[i] = '';
                }
            };
        }
    }
}
