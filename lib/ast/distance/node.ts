module lib.ast {
    export module distance {
        import esprima = lib.ast.esprima;
        import Syntax = esprima.Syntax;
        import castTo = lib.utils.castTo;
        import hash = lib.utils.hash;

        var leftMostHash:Map<string, Syntax.Node> = new Map<string, Syntax.Node>();

        export function leftMost(node:typeof undefined):Syntax.Node;
        export function leftMost(node:Array<Node>):Syntax.Node;
        export function leftMost(node:Syntax.Node):Syntax.Node;
        export function leftMost(node:any):Syntax.Node {
            var h = hash(node);
            if (leftMostHash.has(h)) {
                return leftMostHash.get(h);
            } else {
                var res:Syntax.Node = function () {
                    if (lib.utils.isUndefined(node)) {
                        return node;
                    } else if (lib.utils.isArray(node)) {
                        var arry:Array<any> = castTo<Array<any>>(node);
                        if (arry.length > 0) {
                            return leftMost(arry[0]);
                        } else {
                            return undefined;
                        }
                    } else if (node instanceof Syntax.Literal) {
                        return node;
                    } else if (node instanceof Syntax.Identifier) {
                        return node;
                    } else if (node instanceof Syntax.CatchClause) {
                        return leftMost(castTo<Syntax.CatchClause>(node).guard);
                    } else if (node instanceof Syntax.SwitchCase) {
                        return leftMost(castTo<Syntax.SwitchCase>(node).test);
                    } else if (node instanceof Syntax.MemberExpression) {
                        return leftMost(castTo<Syntax.MemberExpression>(node).object);
                    } else if (node instanceof Syntax.CallExpression) {
                        return leftMost(castTo<Syntax.CallExpression>(node).callee);
                    } else if (node instanceof Syntax.NewExpression) {
                        return leftMost(castTo<Syntax.NewExpression>(node).callee);
                    } else if (node instanceof Syntax.ConditionalExpression) {
                        return leftMost(castTo<Syntax.ConditionalExpression>(node).test);
                    } else if (node instanceof Syntax.LogicalExpression) {
                        return leftMost(castTo<Syntax.LogicalExpression>(node).left);
                    } else if (node instanceof Syntax.UpdateExpression) {
                        return leftMost(castTo<Syntax.UpdateExpression>(node).argument);
                    } else if (node instanceof Syntax.AssignmentExpression) {
                        return leftMost(castTo<Syntax.AssignmentExpression>(node).left);
                    } else if (node instanceof Syntax.BinaryExpression) {
                        return leftMost(castTo<Syntax.BinaryExpression>(node).left);
                    } else if (node instanceof Syntax.UnaryExpression) {
                        return leftMost(castTo<Syntax.UnaryExpression>(node).argument);
                    } else if (node instanceof Syntax.SequenceExpression) {
                        return leftMost(castTo<Syntax.SequenceExpression>(node).expressions) || node;
                    } else if (node instanceof Syntax.ArrowFunctionExpression) {
                        return leftMost(castTo<Syntax.ArrowFunctionExpression>(node).id);
                    } else if (node instanceof Syntax.FunctionExpression) {
                        return leftMost(castTo<Syntax.FunctionExpression>(node).id);
                    } else if (node instanceof Syntax.Property) {
                        return leftMost(castTo<Syntax.Property>(node).value);
                    } else if (node instanceof Syntax.ObjectExpression) {
                        return leftMost(castTo<Syntax.ObjectExpression>(node).properties) || node;
                    } else if (node instanceof Syntax.ArrayExpression) {
                        return leftMost(castTo<Syntax.ArrayExpression>(node).elements) || node;
                    } else if (node instanceof Syntax.ThisExpression) {
                        return node;
                    } else if (node instanceof Syntax.VariableDeclarator) {
                        return leftMost(castTo<Syntax.VariableDeclarator>(node).id);
                    } else if (node instanceof Syntax.VariableDeclaration) {
                        return leftMost(castTo<Syntax.VariableDeclaration>(node).declarations) || node;
                    } else if (node instanceof Syntax.ForInStatement) {
                        return leftMost(castTo<Syntax.ForInStatement>(node).left);
                    } else if (node instanceof Syntax.ForStatement) {
                        return leftMost(castTo<Syntax.ForStatement>(node).init);
                    } else if (node instanceof Syntax.DoWhileStatement) {
                        return leftMost(castTo<Syntax.DoWhileStatement>(node).test);
                        ;
                    } else if (node instanceof Syntax.WhileStatement) {
                        return leftMost(castTo<Syntax.WhileStatement>(node).test);
                    } else if (node instanceof Syntax.TryStatement) {
                        return leftMost(castTo<Syntax.TryStatement>(node).block);
                    } else if (node instanceof Syntax.ThrowStatement) {
                        return leftMost(castTo<Syntax.ThrowStatement>(node).argument);
                    } else if (node instanceof Syntax.ReturnStatement) {
                        return leftMost(castTo<Syntax.ReturnStatement>(node).argument) || node;
                    } else if (node instanceof Syntax.SwitchStatement) {
                        return leftMost(castTo<Syntax.SwitchStatement>(node).discriminant);
                    } else if (node instanceof Syntax.WithStatement) {
                        return leftMost(castTo<Syntax.WithStatement>(node).object);
                    } else if (node instanceof Syntax.ContinueStatement) {
                        return leftMost(castTo<Syntax.ContinueStatement>(node).label) || node;
                    } else if (node instanceof Syntax.BreakStatement) {
                        return leftMost(castTo<Syntax.BreakStatement>(node).label) || node;
                    } else if (node instanceof Syntax.LabeledStatement) {
                        return leftMost(castTo<Syntax.LabeledStatement>(node).label);
                    } else if (node instanceof Syntax.IfStatement) {
                        return leftMost(castTo<Syntax.IfStatement>(node).test);
                    } else if (node instanceof Syntax.ExpressionStatement) {
                        return leftMost(castTo<Syntax.ExpressionStatement>(node).expression);
                    } else if (node instanceof Syntax.BlockStatement) {
                        return leftMost(castTo<Syntax.BlockStatement>(node).body);
                    } else if (node instanceof Syntax.EmptyStatement) {
                        return undefined;
                    } else {
                        return undefined;
                    }
                }();

                leftMostHash.set(h, res);
                return res;
            }
        }
    }
}