/// <reference path="recast.ts" />
/// <reference path="../types/node-path.ts" />
module lib.ast.recast {
    import types = lib.ast.types;
    import assert = lib.utils.assert;
    import NodePath = types.NodePath;
    import sourceMap = lib.ast.sourcemap;
    var n = types.namedTypes;
    var isArray: types.Type = types.builtInTypes["array"];
    var isObject: types.Type = types.builtInTypes["object"];
    var isString: types.Type = types.builtInTypes["string"];
    var isFunction: types.Type = types.builtInTypes["function"];
    var b = types.builders;
    var Node = types.namedTypes["Node"];
    var Expression = types.namedTypes["Expression"];
    var SourceLocation = types.namedTypes["SourceLocation"];

    export function Patcher(lines) {
        assert.ok(this instanceof Patcher);
        assert.ok(lines instanceof Lines);

        var self = this,
            replacements = [];

        self.replace = function(loc, lines) {
            if (isString.check(lines))
                lines = fromString(lines);

            replacements.push({
                lines: lines,
                start: loc.start,
                end: loc.end
            });
        };

        self.get = function(loc) {
            // If no location is provided, return the complete Lines object.
            loc = loc || {
                start: { line: 1, column: 0 },
                end: {
                    line: lines.length,
                    column: lines.getLineLength(lines.length)
                }
            };

            var sliceFrom = loc.start,
                toConcat = [];

            function pushSlice(from, to) {
                assert.ok(comparePos(from, to) <= 0);
                toConcat.push(lines.slice(from, to));
            }

            replacements.sort(function(a, b) {
                return comparePos(a.start, b.start);
            }).forEach(function(rep) {
                    if (comparePos(sliceFrom, rep.start) > 0) {
                        // Ignore nested replacement ranges.
                    } else {
                        pushSlice(sliceFrom, rep.start);
                        toConcat.push(rep.lines);
                        sliceFrom = rep.end;
                    }
                });

            pushSlice(sliceFrom, loc.end);

            return concat(toConcat);
        }
    }

    export function getReprinter(path) {
        //assert.ok(path instanceof lib.ast.types.NodePath);

        // Make sure that this path refers specifically to a Node, rather than
        // some non-Node subproperty of a Node.
        var node = path.value;
        if (!lib.ast.types.namedTypes["Node"].check(node))
            return;

        var orig = node.original;
        var origLoc = orig && orig.loc;
        var lines = origLoc && origLoc.lines;
        var reprints = [];

        if (!lines || !findReprints(path, reprints))
            return;

        return function(print) {
            var patcher = new Patcher(lines);

            reprints.forEach(function(reprint) {
                var old = reprint.oldPath.value;
                SourceLocation.assert(old.loc, true);
                patcher.replace(
                    old.loc,
                    print(reprint.newPath).indentTail(old.loc.indent)
                    );
            });

            return patcher.get(origLoc).indentTail(-orig.loc.indent);
        };
    };

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

        if (!(isArray.check(oldNode) &&
            oldNode.length === len))
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
                } else {
                    reprints.push({
                        newPath: newPath,
                        oldPath: oldPath
                    });
                }

                return true;
            }

            if (Expression.check(newNode) &&
                Expression.check(oldNode)) {

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
                    return comparePos(rootPath.value.loc.start, pos) <= 0;
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
                    return comparePos(pos, rootPath.value.loc.end) <= 0;
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
        if (!newPath.canBeFirstInStatement() &&
            newPath.firstInStatement() && !hasOpeningParen(oldPath))
            return false;

        // If this node needs parentheses and will not be wrapped with
        // parentheses when reprinted, then return false to skip reprinting
        // and let it be printed generically.
        if (newPath.needsParens(true) && !hasParens(oldPath)) {
            return false;
        }

        for (var k in getUnionOfKeys(newNode, oldNode)) {
            if (k === "loc")
                continue;

            if (!findAnyReprints(newPath.get(k), oldPath.get(k), reprints))
                return false;
        }

        return true;
    }
}
