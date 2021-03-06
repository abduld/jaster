module lib.ast {
    export module scope {
        export module detail {
            var currentScope,
                globalScope,
                scopes,
                options;

            import estraverse = lib.ast.traverse;
            import esutils = lib.ast.utils;

            import Map = lib.utils.Map;
            import assert = lib.utils.assert;

            import Syntax = estraverse.Syntax;

            function defaultOptions() {
                return {
                    optimistic: false,
                    directive: false,
                    ecmaVersion: 5
                };
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
                            } else {
                                target[key] = updateDeeply({}, val);
                            }
                        } else {
                            target[key] = val;
                        }
                    }
                }
                return target;
            }

            /**
             * A Reference represents a single occurrence of an identifier in code.
             * @class Reference
             */
            export function Reference(ident, scope, flag, writeExpr, maybeImplicitGlobal, partial) {
                /**
                 * Identifier syntax node.
                 * @member {esprima#Identifier} Reference#identifier
                 */
                this.identifier = ident;
                /**
                 * Reference to the enclosing Scope.
                 * @member {Scope} Reference#from
                 */
                this.from = scope;
                /**
                 * Whether the reference comes from a dynamic scope (such as 'eval',
                 * 'with', etc.), and may be trapped by dynamic scopes.
                 * @member {boolean} Reference#tainted
                 */
                this.tainted = false;
                /**
                 * The variable this reference is resolved with.
                 * @member {Variable} Reference#resolved
                 */
                this.resolved = null;
                /**
                 * The read-write mode of the reference. (Value is one of {@link
                    * Reference.READ}, {@link Reference.RW}, {@link Reference.WRITE}).
                 * @member {number} Reference#flag
                 * @private
                 */
                this.flag = flag;
                if (this.isWrite()) {
                    /**
                     * If reference is writeable, this is the tree being written to it.
                     * @member {esprima#Node} Reference#writeExpr
                     */
                    this.writeExpr = writeExpr;
                    /**
                     * Whether the Reference might refer to a partial value of writeExpr.
                     * @member {boolean} Reference#partial
                     */
                    this.partial = partial;
                }
                this.__maybeImplicitGlobal = maybeImplicitGlobal;
            }

            /**
             * @constant Reference.READ
             * @private
             */
            Reference.READ = 0x1;
            /**
             * @constant Reference.WRITE
             * @private
             */
            Reference.WRITE = 0x2;
            /**
             * @constant Reference.RW
             * @private
             */
            Reference.RW = Reference.READ | Reference.WRITE;

            /**
             * Whether the reference is static.
             * @method Reference#isStatic
             * @return {boolean}
             */
            Reference.prototype.isStatic = function isStatic() {
                return !this.tainted && this.resolved && this.resolved.scope.isStatic();
            };

            /**
             * Whether the reference is writeable.
             * @method Reference#isWrite
             * @return {boolean}
             */
            Reference.prototype.isWrite = function isWrite() {
                return !!(this.flag & Reference.WRITE);
            };

            /**
             * Whether the reference is readable.
             * @method Reference#isRead
             * @return {boolean}
             */
            Reference.prototype.isRead = function isRead() {
                return !!(this.flag & Reference.READ);
            };

            /**
             * Whether the reference is read-only.
             * @method Reference#isReadOnly
             * @return {boolean}
             */
            Reference.prototype.isReadOnly = function isReadOnly() {
                return this.flag === Reference.READ;
            };

            /**
             * Whether the reference is write-only.
             * @method Reference#isWriteOnly
             * @return {boolean}
             */
            Reference.prototype.isWriteOnly = function isWriteOnly() {
                return this.flag === Reference.WRITE;
            };

            /**
             * Whether the reference is read-write.
             * @method Reference#isReadWrite
             * @return {boolean}
             */
            Reference.prototype.isReadWrite = function isReadWrite() {
                return this.flag === Reference.RW;
            };

            /**
             * A Variable represents a locally scoped identifier. These include arguments to
             * functions.
             * @class Variable
             */
            export function Variable(name, scope) {
                /**
                 * The variable name, as given in the source code.
                 * @member {String} Variable#name
                 */
                this.name = name;
                /**
                 * List of defining occurrences of this variable (like in 'var ...'
                 * statements or as parameter), as AST nodes.
                 * @member {esprima.Identifier[]} Variable#identifiers
                 */
                this.identifiers = [];
                /**
                 * List of {@link Reference|references} of this variable (excluding parameter entries)
                 * in its defining scope and all nested scopes. For defining
                 * occurrences only see {@link Variable#defs}.
                 * @member {Reference[]} Variable#references
                 */
                this.references = [];

                /**
                 * List of defining occurrences of this variable (like in 'var ...'
                 * statements or as parameter), as custom objects.
                 * @typedef {Object} DefEntry
                 * @property {String} DefEntry.type - the type of the occurrence (e.g.
                 *      "Parameter", "Variable", ...)
                 * @property {esprima.Identifier} DefEntry.name - the identifier AST node of the occurrence
                 * @property {esprima.Node} DefEntry.node - the enclosing node of the
                 *      identifier
                 * @property {esprima.Node} [DefEntry.parent] - the enclosing statement
                 *      node of the identifier
                 * @member {DefEntry[]} Variable#defs
                 */
                this.defs = [];

                this.tainted = false;
                /**
                 * Whether this is a stack variable.
                 * @member {boolean} Variable#stack
                 */
                this.stack = true;
                /**
                 * Reference to the enclosing Scope.
                 * @member {Scope} Variable#scope
                 */
                this.scope = scope;
            }

            Variable.CatchClause = 'CatchClause';
            Variable.Parameter = 'Parameter';
            Variable.FunctionName = 'FunctionName';
            Variable.ClassName = 'ClassName';
            Variable.Variable = 'Variable';
            Variable.TDZ = 'TDZ';
            Variable.ImplicitGlobalVariable = 'ImplicitGlobalVariable';

            function isMethodDefinition(block, parent) {
                // Check
                // + Class MethodDefinition
                // + Object MethodDefiniton
                // cases.
                if (block.type !== Syntax.FunctionExpression) {
                    return false;
                }

                if (!parent) {
                    return false;
                }

                if (parent.type === Syntax.MethodDefinition && block === parent.value) {
                    return true;
                }
                if (parent.type === Syntax.Property && parent.method && block === parent.value) {
                    return true;
                }

                return false;
            }

            function isStrictScope(scope, block, parent) {
                var body, i, iz, stmt, expr;

                // When upper scope is exists and strict, inner scope is also strict.
                if (scope.upper && scope.upper.isStrict) {
                    return true;
                }

                // ArrowFunctionExpression's scope is always strict scope.
                if (block.type === Syntax.ArrowFunctionExpression) {
                    return true;
                }

                if (parent) {
                    if (isMethodDefinition(block, parent)) {
                        return true;
                    }
                }

                if (scope.type === 'class') {
                    return true;
                } else if (scope.type === 'function') {
                    body = block.body;
                } else if (scope.type === 'global') {
                    body = block;
                } else {
                    return false;
                }

                // Search 'use strict' directive.
                if (options.directive) {
                    for (i = 0, iz = body.body.length; i < iz; ++i) {
                        stmt = body.body[i];
                        if (stmt.type !== 'DirectiveStatement') {
                            break;
                        }
                        if (stmt.raw === '"use strict"' || stmt.raw === '\'use strict\'') {
                            return true;
                        }
                    }
                } else {
                    for (i = 0, iz = body.body.length; i < iz; ++i) {
                        stmt = body.body[i];
                        if (stmt.type !== Syntax.ExpressionStatement) {
                            break;
                        }
                        expr = stmt.expression;
                        if (expr.type !== Syntax.Literal || typeof expr.value !== 'string') {
                            break;
                        }
                        if (expr.raw != null) {
                            if (expr.raw === '"use strict"' || expr.raw === '\'use strict\'') {
                                return true;
                            }
                        } else {
                            if (expr.value === 'use strict') {
                                return true;
                            }
                        }
                    }
                }
                return false;
            }

            /* Special Scope types. */
            var SCOPE_NORMAL = 0,
                SCOPE_FUNCTION_EXPRESSION_NAME = 1,
                SCOPE_TDZ = 2;

            /**
             * @class Scope
             */
            export function Scope(scopeManager, block, parent, scopeType) {
                /**
                 * One of 'catch', 'with', 'function', 'global' or 'block'.
                 * @member {String} Scope#type
                 */
                this.type =
                (scopeType === SCOPE_TDZ) ? 'TDZ' :
                (block.type === Syntax.BlockStatement) ? 'block' :
                (block.type === Syntax.FunctionExpression || block.type === Syntax.FunctionDeclaration || block.type === Syntax.ArrowFunctionExpression) ? 'function' :
                (block.type === Syntax.CatchClause) ? 'catch' :
                (block.type === Syntax.ForInStatement || block.type === Syntax.ForOfStatement || block.type === Syntax.ForStatement) ? 'for' :
                (block.type === Syntax.WithStatement) ? 'with' :
                (block.type === Syntax.ClassBody) ? 'class' : 'global';
                /**
                 * The scoped {@link Variable}s of this scope, as <code>{ Variable.name
         * : Variable }</code>.
                 * @member {Map} Scope#set
                 */
                this.set = new Map();
                /**
                 * The tainted variables of this scope, as <code>{ Variable.name :
         * boolean }</code>.
                 * @member {Map} Scope#taints */
                this.taints = new Map();
                /**
                 * Generally, through the lexical scoping of JS you can always know
                 * which variable an identifier in the source code refers to. There are
                 * a few exceptions to this rule. With 'global' and 'with' scopes you
                 * can only decide at runtime which variable a reference refers to.
                 * Moreover, if 'eval()' is used in a scope, it might introduce new
                 * bindings in this or its prarent scopes.
                 * All those scopes are considered 'dynamic'.
                 * @member {boolean} Scope#dynamic
                 */
                this.dynamic = this.type === 'global' || this.type === 'with';
                /**
                 * A reference to the scope-defining syntax node.
                 * @member {esprima.Node} Scope#block
                 */
                this.block = block;
                /**
                 * The {@link Reference|references} that are not resolved with this scope.
                 * @member {Reference[]} Scope#through
                 */
                this.through = [];
                /**
                 * The scoped {@link Variable}s of this scope. In the case of a
                 * 'function' scope this includes the automatic argument <em>arguments</em> as
                 * its first element, as well as all further formal arguments.
                 * @member {Variable[]} Scope#variables
                 */
                this.variables = [];
                /**
                 * Any variable {@link Reference|reference} found in this scope. This
                 * includes occurrences of local variables as well as variables from
                 * parent scopes (including the global scope). For local variables
                 * this also includes defining occurrences (like in a 'var' statement).
                 * In a 'function' scope this does not include the occurrences of the
                 * formal parameter in the parameter list.
                 * @member {Reference[]} Scope#references
                 */
                this.references = [];

                /**
                 * For 'global' and 'function' scopes, this is a self-reference. For
                 * other scope types this is the <em>variableScope</em> value of the
                 * parent scope.
                 * @member {Scope} Scope#variableScope
                 */
                this.variableScope =
                (this.type === 'global' || this.type === 'function') ? this : currentScope.variableScope;
                /**
                 * Whether this scope is created by a FunctionExpression.
                 * @member {boolean} Scope#functionExpressionScope
                 */
                this.functionExpressionScope = false;
                /**
                 * Whether this is a scope that contains an 'eval()' invocation.
                 * @member {boolean} Scope#directCallToEvalScope
                 */
                this.directCallToEvalScope = false;
                /**
                 * @member {boolean} Scope#thisFound
                 */
                this.thisFound = false;

                this.__left = [];

                if (scopeType === SCOPE_FUNCTION_EXPRESSION_NAME) {
                    this.__define(block.id, {
                        type: Variable.FunctionName,
                        name: block.id,
                        node: block
                    });
                    this.functionExpressionScope = true;
                } else {
                    if (this.type === 'function') {
                        this.__defineArguments();
                    }

                    if (block.type === Syntax.FunctionExpression && block.id && !isMethodDefinition(block, parent)) {
                        scopeManager.__nestFunctionExpressionNameScope(block, parent);
                    }
                }

                /**
                 * Reference to the parent {@link Scope|scope}.
                 * @member {Scope} Scope#upper
                 */
                this.upper = currentScope;
                /**
                 * Whether 'use strict' is in effect in this scope.
                 * @member {boolean} Scope#isStrict
                 */
                this.isStrict = isStrictScope(this, block, parent);

                /**
                 * List of nested {@link Scope}s.
                 * @member {Scope[]} Scope#childScopes
                 */
                this.childScopes = [];
                if (currentScope) {
                    currentScope.childScopes.push(this);
                }


                // RAII
                currentScope = this;
                if (this.type === 'global') {
                    globalScope = this;
                    globalScope.implicit = {
                        set: new Map(),
                        variables: [],
                        /**
                         * List of {@link Reference}s that are left to be resolved (i.e. which
                         * need to be linked to the variable they refer to).
                         * @member {Reference[]} Scope#implicit#left
                         */
                        left: []
                    };
                }
                scopes.push(this);
            }

            Scope.prototype.__close = function __close() {
                var i, iz, ref, current, implicit, info;

                // Because if this is global environment, upper is null
                if (!this.dynamic || options.optimistic) {
                    // static resolve
                    for (i = 0, iz = this.__left.length; i < iz; ++i) {
                        ref = this.__left[i];
                        if (!this.__resolve(ref)) {
                            this.__delegateToUpperScope(ref);
                        }
                    }
                } else {
                    // this is "global" / "with" / "function with eval" environment
                    if (this.type === 'with') {
                        for (i = 0, iz = this.__left.length; i < iz; ++i) {
                            ref = this.__left[i];
                            ref.tainted = true;
                            this.__delegateToUpperScope(ref);
                        }
                    } else {
                        for (i = 0, iz = this.__left.length; i < iz; ++i) {
                            // notify all names are through to global
                            ref = this.__left[i];
                            current = this;
                            do {
                                current.through.push(ref);
                                current = current.upper;
                            } while (current);
                        }
                    }
                }

                if (this.type === 'global') {
                    implicit = [];
                    for (i = 0, iz = this.__left.length; i < iz; ++i) {
                        ref = this.__left[i];
                        if (ref.__maybeImplicitGlobal && !this.set.has(ref.identifier.name)) {
                            implicit.push(ref.__maybeImplicitGlobal);
                        }
                    }

                    // create an implicit global variable from assignment expression
                    for (i = 0, iz = implicit.length; i < iz; ++i) {
                        info = implicit[i];
                        this.__defineImplicit(info.pattern, {
                            type: Variable.ImplicitGlobalVariable,
                            name: info.pattern,
                            node: info.node
                        });
                    }

                    this.implicit.left = this.__left;
                }

                this.__left = null;
                currentScope = this.upper;
            };

            Scope.prototype.__resolve = function __resolve(ref) {
                var variable, name;
                name = ref.identifier.name;
                if (this.set.has(name)) {
                    variable = this.set.get(name);
                    variable.references.push(ref);
                    variable.stack = variable.stack && ref.from.variableScope === this.variableScope;
                    if (ref.tainted) {
                        variable.tainted = true;
                        this.taints.set(variable.name, true);
                    }
                    ref.resolved = variable;
                    return true;
                }
                return false;
            };

            Scope.prototype.__delegateToUpperScope = function __delegateToUpperScope(ref) {
                if (this.upper) {
                    this.upper.__left.push(ref);
                }
                this.through.push(ref);
            };

            Scope.prototype.__defineGeneric = function(name, set, variables, node, info) {
                var variable;

                variable = set.get(name);
                if (!variable) {
                    variable = new Variable(name, this);
                    set.set(name, variable);
                    variables.push(variable);
                }

                if (info) {
                    variable.defs.push(info);
                }
                if (node) {
                    variable.identifiers.push(node);
                }
            };

            Scope.prototype.__defineArguments = function() {
                this.__defineGeneric('arguments', this.set, this.variables);
                this.taints.set('arguments', true);
            };

            Scope.prototype.__defineImplicit = function(node, info) {
                if (node && node.type === Syntax.Identifier) {
                    this.__defineGeneric(node.name, this.implicit.set, this.implicit.variables, node, info);
                }
            };

            Scope.prototype.__define = function(node, info) {
                if (node && node.type === Syntax.Identifier) {
                    this.__defineGeneric(node.name, this.set, this.variables, node, info);
                }
            };

            Scope.prototype.__referencing = function __referencing(node, assign, writeExpr, maybeImplicitGlobal, partial) {
                var ref;
                // because Array element may be null
                if (node && node.type === Syntax.Identifier) {
                    ref = new Reference(node, this, assign || Reference.READ, writeExpr, maybeImplicitGlobal, !!partial);
                    this.references.push(ref);
                    this.__left.push(ref);
                }
            };

            Scope.prototype.__detectEval = function __detectEval() {
                var current;
                current = this;
                this.directCallToEvalScope = true;
                do {
                    current.dynamic = true;
                    current = current.upper;
                } while (current);
            };

            Scope.prototype.__detectThis = function __detectThis() {
                this.thisFound = true;
            };

            Scope.prototype.__isClosed = function isClosed() {
                return this.__left === null;
            };

            // API Scope#resolve(name)
            // returns resolved reference
            Scope.prototype.resolve = function resolve(ident) {
                var ref, i, iz;
                assert(this.__isClosed(), 'scope should be closed');
                assert(ident.type === Syntax.Identifier, 'target should be identifier');
                for (i = 0, iz = this.references.length; i < iz; ++i) {
                    ref = this.references[i];
                    if (ref.identifier === ident) {
                        return ref;
                    }
                }
                return null;
            };

            // API Scope#isStatic
            // returns this scope is static
            Scope.prototype.isStatic = function isStatic() {
                return !this.dynamic;
            };

            // API Scope#isArgumentsMaterialized
            // return this scope has materialized arguments
            Scope.prototype.isArgumentsMaterialized = function isArgumentsMaterialized() {
                // TODO(Constellation)
                // We can more aggressive on this condition like this.
                //
                // function t() {
                //     // arguments of t is always hidden.
                //     function arguments() {
                //     }
                // }
                var variable;

                // This is not function scope
                if (this.type !== 'function') {
                    return true;
                }

                if (!this.isStatic()) {
                    return true;
                }

                variable = this.set.get('arguments');
                assert(variable, 'always have arguments variable');
                return variable.tainted || variable.references.length !== 0;
            };

            // API Scope#isThisMaterialized
            // return this scope has materialized `this` reference
            Scope.prototype.isThisMaterialized = function isThisMaterialized() {
                // This is not function scope
                if (this.type !== 'function') {
                    return true;
                }
                if (!this.isStatic()) {
                    return true;
                }
                return this.thisFound;
            };

            Scope.mangledName = '__$escope$__';

            Scope.prototype.attach = function attach() {
                if (!this.functionExpressionScope) {
                    this.block[Scope.mangledName] = this;
                }
            };

            Scope.prototype.detach = function detach() {
                if (!this.functionExpressionScope) {
                    delete this.block[Scope.mangledName];
                }
            };

            Scope.prototype.isUsedName = function(name) {
                if (this.set.has(name)) {
                    return true;
                }
                for (var i = 0, iz = this.through.length; i < iz; ++i) {
                    if (this.through[i].identifier.name === name) {
                        return true;
                    }
                }
                return false;
            };

            /**
             * @class ScopeManager
             */
            export function ScopeManager(scopes, options) {
                this.scopes = scopes;
                this.attached = false;
                this.__options = options;
            }

            // Returns appropliate scope for this node
            ScopeManager.prototype.__get = function __get(node) {
                var i, iz, scope;
                if (this.attached) {
                    return node[Scope.mangledName] || null;
                }

                for (i = 0, iz = this.scopes.length; i < iz; ++i) {
                    scope = this.scopes[i];
                    if (!scope.functionExpressionScope) {
                        if (scope.block === node) {
                            return scope;
                        }
                    }
                }
                return null;
            };

            ScopeManager.prototype.acquire = function acquire(node) {
                return this.__get(node);
            };

            ScopeManager.prototype.release = function release(node) {
                var scope = this.__get(node);
                if (scope) {
                    scope = scope.upper;
                    while (scope) {
                        if (!scope.functionExpressionScope) {
                            return scope;
                        }
                        scope = scope.upper;
                    }
                }
                return null;
            };

            ScopeManager.prototype.attach = function attach() {
                var i, iz;
                for (i = 0, iz = this.scopes.length; i < iz; ++i) {
                    this.scopes[i].attach();
                }
                this.attached = true;
            };

            ScopeManager.prototype.detach = function detach() {
                var i, iz;
                for (i = 0, iz = this.scopes.length; i < iz; ++i) {
                    this.scopes[i].detach();
                }
                this.attached = false;
            };

            ScopeManager.prototype.__nestScope = function(node, parent) {
                return new Scope(this, node, parent, SCOPE_NORMAL);
            };

            ScopeManager.prototype.__nestTDZScope = function(node, iterationNode) {
                return new Scope(this, node, iterationNode, SCOPE_TDZ);
            };

            ScopeManager.prototype.__nestFunctionExpressionNameScope = function(node, parent) {
                return new Scope(this, node, parent, SCOPE_FUNCTION_EXPRESSION_NAME);
            };

            ScopeManager.prototype.__isES6 = function() {
                return this.__options.ecmaVersion >= 6;
            };

            function traverseIdentifierInPattern(rootPattern, callback) {
                estraverse.traverse(rootPattern, {
                    enter: function(pattern, parent) {
                        var i, iz, element, property;

                        switch (pattern.type) {
                            case Syntax.Identifier:
                                // Toplevel identifier.
                                if (parent === null) {
                                    callback(pattern, true);
                                }
                                break;

                            case Syntax.SpreadElement:
                                if (pattern.argument.type === Syntax.Identifier) {
                                    callback(pattern.argument, false);
                                }
                                break;

                            case Syntax.ObjectPattern:
                                for (i = 0, iz = pattern.properties.length; i < iz; ++i) {
                                    property = pattern.properties[i];
                                    if (property.shorthand) {
                                        callback(property.key, false);
                                        continue;
                                    }
                                    if (property.value.type === Syntax.Identifier) {
                                        callback(property.value, false);
                                        continue;
                                    }
                                }
                                break;

                            case Syntax.ArrayPattern:
                                for (i = 0, iz = pattern.elements.length; i < iz; ++i) {
                                    element = pattern.elements[i];
                                    if (element && element.type === Syntax.Identifier) {
                                        callback(element, false);
                                    }
                                }
                                break;
                        }
                    }
                });
            }

            function doVariableDeclaration(variableTargetScope, type, node, index) {
                var decl, init;

                decl = node.declarations[index];
                init = decl.init;
                // FIXME: Don't consider initializer with complex patterns.
                // Such as,
                // var [a, b, c = 20] = array;
                traverseIdentifierInPattern(decl.id, function(pattern, toplevel) {
                    variableTargetScope.__define(pattern, {
                        type: type,
                        name: pattern,
                        node: decl,
                        index: index,
                        kind: node.kind,
                        parent: node
                    });

                    if (init) {
                        currentScope.__referencing(pattern, Reference.WRITE, init, null, !toplevel);
                    }
                });
            }

            function materializeTDZScope(scopeManager, node, iterationNode) {
                // http://people.mozilla.org/~jorendorff/es6-draft.html#sec-runtime-semantics-forin-div-ofexpressionevaluation-abstract-operation
                // TDZ scope hides the declaration's names.
                scopeManager.__nestTDZScope(node, iterationNode);
                doVariableDeclaration(currentScope, Variable.TDZ, iterationNode.left, 0);
                currentScope.__referencing(node);
            }

            function materializeIterationScope(scopeManager, node) {
                // Generate iteration scope for upper ForIn/ForOf Statements.
                // parent node for __nestScope is only necessary to
                // distinguish MethodDefinition.
                var letOrConstDecl;
                scopeManager.__nestScope(node, null);
                letOrConstDecl = node.left;
                doVariableDeclaration(currentScope, Variable.Variable, letOrConstDecl, 0);
                traverseIdentifierInPattern(letOrConstDecl.declarations[0].id, function(pattern) {
                    currentScope.__referencing(pattern, Reference.WRITE, node.right, null, true);
                });
            }

            /**
             * Main interface function. Takes an Esprima syntax tree and returns the
             * analyzed scopes.
             * @function analyze
             * @param {esprima.Tree} tree
             * @param {Object} providedOptions - Options that tailor the scope analysis
             * @param {boolean} [providedOptions.optimistic=false] - the optimistic flag
             * @param {boolean} [providedOptions.directive=false]- the directive flag
             * @param {boolean} [providedOptions.ignoreEval=false]- whether to check 'eval()' calls
             * @return {ScopeManager}
             */
            export function analyze(tree, providedOptions) {
                var resultScopes, scopeManager;

                options = updateDeeply(defaultOptions(), providedOptions);
                resultScopes = scopes = [];
                currentScope = null;
                globalScope = null;

                scopeManager = new ScopeManager(resultScopes, options);

                // attach scope and collect / resolve names
                estraverse.traverse(tree, {
                    enter: function enter(node, parent) {
                        var i, iz, decl, variableTargetScope;

                        // Special path for ForIn/ForOf Statement block scopes.
                        if (parent &&
                            (parent.type === Syntax.ForInStatement || parent.type === Syntax.ForOfStatement) &&
                            parent.left.type === Syntax.VariableDeclaration &&
                            parent.left.kind !== 'var') {
                            // Construct TDZ scope.
                            if (parent.right === node) {
                                materializeTDZScope(scopeManager, node, parent);
                            }
                            if (parent.body === node) {
                                materializeIterationScope(scopeManager, parent);
                            }
                        }

                        switch (this.type()) {
                            case Syntax.AssignmentExpression:
                                if (node.operator === '=') {
                                    traverseIdentifierInPattern(node.left, function(pattern, toplevel) {
                                        var maybeImplicitGlobal = null;
                                        if (!currentScope.isStrict) {
                                            maybeImplicitGlobal = {
                                                pattern: pattern,
                                                node: node
                                            };
                                        }
                                        currentScope.__referencing(pattern, Reference.WRITE, node.right, maybeImplicitGlobal, !toplevel);
                                    });
                                } else {
                                    currentScope.__referencing(node.left, Reference.RW, node.right);
                                }
                                currentScope.__referencing(node.right);
                                break;

                            case Syntax.ArrayExpression:
                                for (i = 0, iz = node.elements.length; i < iz; ++i) {
                                    currentScope.__referencing(node.elements[i]);
                                }
                                break;

                            case Syntax.BlockStatement:
                                if (scopeManager.__isES6()) {
                                    if (!parent ||
                                        parent.type !== Syntax.FunctionExpression &&
                                        parent.type !== Syntax.FunctionDeclaration &&
                                        parent.type !== Syntax.ArrowFunctionExpression) {
                                        scopeManager.__nestScope(node, parent);
                                    }
                                }
                                break;

                            case Syntax.BinaryExpression:
                                currentScope.__referencing(node.left);
                                currentScope.__referencing(node.right);
                                break;

                            case Syntax.BreakStatement:
                                break;

                            case Syntax.CallExpression:
                                currentScope.__referencing(node.callee);
                                for (i = 0, iz = node['arguments'].length; i < iz; ++i) {
                                    currentScope.__referencing(node['arguments'][i]);
                                }

                                // Check this is direct call to eval
                                if (!options.ignoreEval && node.callee.type === Syntax.Identifier && node.callee.name === 'eval') {
                                    // NOTE: This should be `variableScope`. Since direct eval call always creates Lexical environment and
                                    // let / const should be enclosed into it. Only VariableDeclaration affects on the caller's environment.
                                    currentScope.variableScope.__detectEval();
                                }
                                break;

                            case Syntax.CatchClause:
                                scopeManager.__nestScope(node, parent);
                                traverseIdentifierInPattern(node.param, function(pattern) {
                                    currentScope.__define(pattern, {
                                        type: Variable.CatchClause,
                                        name: node.param,
                                        node: node
                                    });
                                });
                                break;

                            case Syntax.ClassDeclaration:
                                // Outer block scope.
                                currentScope.__define(node.id, {
                                    type: Variable.ClassName,
                                    name: node.id,
                                    node: node
                                });
                                currentScope.__referencing(node.superClass);
                                break;

                            case Syntax.ClassBody:
                                // ClassBody scope.
                                scopeManager.__nestScope(node, parent);
                                if (parent && parent.id) {
                                    currentScope.__define(parent.id, {
                                        type: Variable.ClassName,
                                        name: node.id,
                                        node: node
                                    });
                                }
                                break;

                            case Syntax.ClassExpression:
                                currentScope.__referencing(node.superClass);
                                break;

                            case Syntax.ConditionalExpression:
                                currentScope.__referencing(node.test);
                                currentScope.__referencing(node.consequent);
                                currentScope.__referencing(node.alternate);
                                break;

                            case Syntax.ContinueStatement:
                                break;

                            case Syntax.DirectiveStatement:
                                break;

                            case Syntax.DoWhileStatement:
                                currentScope.__referencing(node.test);
                                break;

                            case Syntax.DebuggerStatement:
                                break;

                            case Syntax.EmptyStatement:
                                break;

                            case Syntax.ExpressionStatement:
                                currentScope.__referencing(node.expression);
                                break;

                            case Syntax.ForStatement:
                                if (node.init && node.init.type === Syntax.VariableDeclaration && node.init.kind !== 'var') {
                                    // Create ForStatement declaration.
                                    // NOTE: In ES6, ForStatement dynamically generates
                                    // per iteration environment. However, escope is
                                    // a static analyzer, we only generate one scope for ForStatement.
                                    scopeManager.__nestScope(node, parent);
                                }
                                currentScope.__referencing(node.init);
                                currentScope.__referencing(node.test);
                                currentScope.__referencing(node.update);
                                break;

                            case Syntax.ForOfStatement:
                            case Syntax.ForInStatement:
                                if (node.left.type === Syntax.VariableDeclaration) {
                                    if (node.left.kind !== 'var') {
                                        // LetOrConst Declarations are specially handled.
                                        break;
                                    }
                                    traverseIdentifierInPattern(node.left.declarations[0].id, function(pattern) {
                                        currentScope.__referencing(pattern, Reference.WRITE, node.right, null, true);
                                    });
                                } else {
                                    traverseIdentifierInPattern(node.left, function(pattern) {
                                        var maybeImplicitGlobal = null;
                                        if (!currentScope.isStrict) {
                                            maybeImplicitGlobal = {
                                                pattern: pattern,
                                                node: node
                                            };
                                        }
                                        currentScope.__referencing(pattern, Reference.WRITE, node.right, maybeImplicitGlobal, true);
                                    });
                                }
                                currentScope.__referencing(node.right);
                                break;

                            case Syntax.FunctionDeclaration:
                                // FunctionDeclaration name is defined in upper scope
                                // NOTE: Not referring variableScope. It is intended.
                                // Since
                                //  in ES5, FunctionDeclaration should be in FunctionBody.
                                //  in ES6, FunctionDeclaration should be block scoped.
                                currentScope.__define(node.id, {
                                    type: Variable.FunctionName,
                                    name: node.id,
                                    node: node
                                });
                            // falls through

                            case Syntax.FunctionExpression:
                            case Syntax.ArrowFunctionExpression:
                                // id is defined in upper scope
                                scopeManager.__nestScope(node, parent);

                                for (i = 0, iz = node.params.length; i < iz; ++i) {
                                    traverseIdentifierInPattern(node.params[i], function(pattern) {
                                        currentScope.__define(pattern, {
                                            type: Variable.Parameter,
                                            name: pattern,
                                            node: node,
                                            index: i
                                        });
                                    });
                                }
                                break;

                            case Syntax.Identifier:
                                break;

                            case Syntax.IfStatement:
                                currentScope.__referencing(node.test);
                                break;

                            case Syntax.Literal:
                                break;

                            case Syntax.LabeledStatement:
                                break;

                            case Syntax.LogicalExpression:
                                currentScope.__referencing(node.left);
                                currentScope.__referencing(node.right);
                                break;

                            case Syntax.MemberExpression:
                                currentScope.__referencing(node.object);
                                if (node.computed) {
                                    currentScope.__referencing(node.property);
                                }
                                break;

                            case Syntax.NewExpression:
                                currentScope.__referencing(node.callee);
                                for (i = 0, iz = node['arguments'].length; i < iz; ++i) {
                                    currentScope.__referencing(node['arguments'][i]);
                                }
                                break;

                            case Syntax.ObjectExpression:
                                break;

                            case Syntax.Program:
                                scopeManager.__nestScope(node, parent);
                                break;

                            case Syntax.Property:
                                // Don't referencing variables when the parent type is ObjectPattern.
                                if (parent.type !== Syntax.ObjectExpression) {
                                    break;
                                }
                                if (node.computed) {
                                    currentScope.__referencing(node.key);
                                }
                                if (node.kind === 'init') {
                                    currentScope.__referencing(node.value);
                                }
                                break;

                            case Syntax.MethodDefinition:
                                if (node.computed) {
                                    currentScope.__referencing(node.key);
                                }
                                break;

                            case Syntax.ReturnStatement:
                                currentScope.__referencing(node.argument);
                                break;

                            case Syntax.SequenceExpression:
                                for (i = 0, iz = node.expressions.length; i < iz; ++i) {
                                    currentScope.__referencing(node.expressions[i]);
                                }
                                break;

                            case Syntax.SwitchStatement:
                                currentScope.__referencing(node.discriminant);
                                break;

                            case Syntax.SwitchCase:
                                currentScope.__referencing(node.test);
                                break;

                            case Syntax.TaggedTemplateExpression:
                                currentScope.__referencing(node.tag);
                                break;

                            case Syntax.TemplateLiteral:
                                for (i = 0, iz = node.expressions.length; i < iz; ++i) {
                                    currentScope.__referencing(node.expressions[i]);
                                }
                                break;

                            case Syntax.ThisExpression:
                                currentScope.variableScope.__detectThis();
                                break;

                            case Syntax.ThrowStatement:
                                currentScope.__referencing(node.argument);
                                break;

                            case Syntax.TryStatement:
                                break;

                            case Syntax.UnaryExpression:
                                currentScope.__referencing(node.argument);
                                break;

                            case Syntax.UpdateExpression:
                                currentScope.__referencing(node.argument, Reference.RW, null);
                                break;

                            case Syntax.VariableDeclaration:
                                if (node.kind !== 'var' && parent) {
                                    if (parent.type === Syntax.ForInStatement || parent.type === Syntax.ForOfStatement) {
                                        // e.g.
                                        //    for (let i in abc);
                                        // In this case, they are specially handled in ForIn/ForOf statements.
                                        break;
                                    }
                                }

                                variableTargetScope = (node.kind === 'var') ? currentScope.variableScope : currentScope;
                                for (i = 0, iz = node.declarations.length; i < iz; ++i) {
                                    decl = node.declarations[i];
                                    doVariableDeclaration(variableTargetScope, Variable.Variable, node, i);
                                    if (decl.init) {
                                        currentScope.__referencing(decl.init);
                                    }
                                }
                                break;

                            case Syntax.VariableDeclarator:
                                break;

                            case Syntax.WhileStatement:
                                currentScope.__referencing(node.test);
                                break;

                            case Syntax.WithStatement:
                                currentScope.__referencing(node.object);
                                // Then nest scope for WithStatement.
                                scopeManager.__nestScope(node, parent);
                                break;
                        }
                    },

                    leave: function leave(node) {
                        while (currentScope && node === currentScope.block) {
                            currentScope.__close();
                        }
                    }
                });

                assert(currentScope === null);
                globalScope = null;
                scopes = null;
                options = null;

                return scopeManager;
            }

        }

        /** @name module:escope.version */
        export var version = '2.0.0-dev';
        /** @name module:escope.Reference */
        export import Reference = detail.Reference;
        /** @name module:escope.Variable */
        export import Variable = detail.Variable;
        /** @name module:escope.Scope */
        export import Scope = detail.Scope;
        /** @name module:escope.ScopeManager */
        export import ScopeManager = detail.ScopeManager;
        /** @name module:escope.analyze */
        export import analyze = detail.analyze;
    }
}