module lib.ast.recast {
    import types = lib.ast.types;
    var n = types.namedTypes;
    var isArray:types.Type = types.builtInTypes["array"];
    var isObject:types.Type = types.builtInTypes["object"];
    var isString:types.Type = types.builtInTypes["string"];
    var isFunction:types.Type = types.builtInTypes["function"];
    var sourceMap = require("source-map");
    var b = types.builders;

    export function parse(source, options) {
        options = normalize(options);

        var lines = fromString(source, options);

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

        add(pure, lines);

        // In order to ensure we reprint leading and trailing program
        // comments, wrap the original Program node with a File node.
        pure = b.file(pure);
        pure.loc = {
            lines: lines,
            indent: 0,
            start: lines.firstPos(),
            end: lines.lastPos()
        };

        // Return a copy of the original AST so that any changes made may be
        // compared to the original.
        return new TreeCopier(lines).copy(pure);
    };

    export class TreeCopier {
        lines:Lines;
        indent:number;

        constructor(lines) {
            assert.ok(this instanceof TreeCopier);
            this.lines = lines;
            this.indent = 0;
        }

        copy(node) {
            if (isArray.check(node)) {
                return node.map(this.copy, this);
            }

            if (!isObject.check(node)) {
                return node;
            }

            if ((n.MethodDefinition && n.MethodDefinition.check(node)) ||
                (n.Property.check(node) && (node.method || node.shorthand))) {
                // If the node is a MethodDefinition or a .method or .shorthand
                // Property, then the location information stored in
                // node.value.loc is very likely untrustworthy (just the {body}
                // part of a method, or nothing in the case of shorthand
                // properties), so we null out that information to prevent
                // accidental reuse of bogus source code during reprinting.
                node.value.loc = null;
            }

            var copy = Object.create(Object.getPrototypeOf(node), {
                original: { // Provide a link from the copy to the original.
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
                } else if (key === "comments") {
                    // Handled below.
                } else {
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
        }
    }
}