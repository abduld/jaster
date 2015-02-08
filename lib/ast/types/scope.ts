/// <reference path="types.ts" />

module lib.ast {
    export module types {
        import assert = lib.utils.assert;
        var namedTypes = types.namedTypes;
        var Node = namedTypes["Node"];
        var Expression = namedTypes["Expression"];
        var isArray = types.builtInTypes["array"];
        var hasOwn = Object.prototype.hasOwnProperty;
        var b = types.builders;

        export class Scope {
            path: any;
            node: any;
            isGlobal: boolean;
            depth: any;
            parent: any;
            bindings: any;

            constructor(path, parentScope) {
                assert.ok(this instanceof Scope);
                ScopeType.assert(path.value);

                var depth;

                if (parentScope) {
                    assert.ok(parentScope instanceof Scope);
                    depth = parentScope.depth + 1;
                } else {
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

            static isEstablishedBy(node) {
                return ScopeType.check(node);
            }

            // Will be overridden after an instance lazily calls scanScope.
            didScan: boolean = false;


            declares(name) {
                this.scan();
                return hasOwn.call(this.bindings, name);
            }

            declareTemporary(prefix?) {
                if (prefix) {
                    assert.ok(/^[a-z$_]/i.test(prefix), prefix);
                } else {
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
            }

            injectTemporary(identifier, init) {
                identifier || (identifier = this.declareTemporary());

                var bodyPath = this.path.get("body");
                if (namedTypes["BlockStatement"].check(bodyPath.value)) {
                    bodyPath = bodyPath.get("body");
                }

                bodyPath.unshift(
                    b["variableDeclaration"](
                        "var",
                        [b["variableDeclarator"](identifier, init || null)]
                        )
                    );

                return identifier;
            }

            scan(force?) {
                if (force || !this.didScan) {
                    for (var name in this.bindings) {
                        // Empty out this.bindings, just in cases.
                        delete this.bindings[name];
                    }
                    scanScope(this.path, this.bindings);
                    this.didScan = true;
                }
            }

            getBindings() {
                this.scan();
                return this.bindings;
            }

            lookup(name) {
                for (var scope = this; scope; scope = scope.parent)
                    if (scope.declares(name))
                        break;
                return scope;
            }

            getGlobalScope() {
                var scope = this;
                while (!scope.isGlobal)
                    scope = scope.parent;
                return scope;
            }
        }
        var scopeTypes = [
            // Program nodes introduce global scopes.
            namedTypes["Program"],

            // Function is the supertype of FunctionExpression,
            // FunctionDeclaration, ArrowExpression, etc.
            namedTypes["Function"],

            // In case you didn't know, the caught parameter shadows any variable
            // of the same name in an outer scope.
            namedTypes["CatchClause"]
        ];

        var ScopeType = Type.or.apply(Type, scopeTypes);


        function scanScope(path, bindings) {
            var node = path.value;
            ScopeType.assert(node);

            if (namedTypes["CatchClause"].check(node)) {
                // A catch clause establishes a new scope but the only variable
                // bound in that scope is the catch parameter. Any other
                // declarations create bindings in the outer scope.
                addPattern(path.get("param"), bindings);

            } else {
                recursiveScanScope(path, bindings);
            }
        }

        function recursiveScanScope(path, bindings) {
            var node = path.value;

            if (path.parent &&
                namedTypes["FunctionExpression"].check(path.parent.node) &&
                path.parent.node.id) {
                addPattern(path.parent.get("id"), bindings);
            }

            if (!node) {
                // None of the remaining cases matter if node is falsy.

            } else if (isArray.check(node)) {
                path.each(function(childPath) {
                    recursiveScanChild(childPath, bindings);
                });

            } else if (namedTypes["Function"].check(node)) {
                path.get("params").each(function(paramPath) {
                    addPattern(paramPath, bindings);
                });

                recursiveScanChild(path.get("body"), bindings);

            } else if (namedTypes["VariableDeclarator"].check(node)) {
                addPattern(path.get("id"), bindings);
                recursiveScanChild(path.get("init"), bindings);

            } else if (node.type === "ImportSpecifier" ||
                node.type === "ImportNamespaceSpecifier" ||
                node.type === "ImportDefaultSpecifier") {
                addPattern(
                    node.name ? path.get("name") : path.get("id"),
                    bindings
                    );

            } else if (Node.check(node) && !Expression.check(node)) {
                types.eachField(node, function(name, child) {
                    var childPath = path.get(name);
                    assert.strictEqual(childPath.value, child);
                    recursiveScanChild(childPath, bindings);
                });
            }
        }

        function recursiveScanChild(path, bindings) {
            var node = path.value;

            if (!node || Expression.check(node)) {
                // Ignore falsy values and Expressions.

            } else if (namedTypes["FunctionDeclaration"].check(node)) {
                addPattern(path.get("id"), bindings);

            } else if (namedTypes["ClassDeclaration"] &&
                namedTypes["ClassDeclaration"].check(node)) {
                addPattern(path.get("id"), bindings);

            } else if (ScopeType.check(node)) {
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

            } else {
                recursiveScanScope(path, bindings);
            }
        }

        function addPattern(patternPath, bindings) {
            var pattern = patternPath.value;
            namedTypes["Pattern"].assert(pattern);

            if (namedTypes["Identifier"].check(pattern)) {
                if (hasOwn.call(bindings, pattern.name)) {
                    bindings[pattern.name].push(patternPath);
                } else {
                    bindings[pattern.name] = [patternPath];
                }

            } else if (namedTypes["SpreadElement"] &&
                namedTypes["SpreadElement"].check(pattern)) {
                addPattern(patternPath.get("argument"), bindings);
            }
        }


    }
}
