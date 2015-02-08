/// <reference path="cena.ts" />
module lib.ast.importer {
    export module stack {
        import castTo = lib.utils.castTo;
        import cena = lib.ast.importer.cena;
        export interface FunctionStack {
            functionName: string;
            id: cena.Identifier;
        }
        var currentFunction: string;

        function captureStack(nd: cena.Node, data: Map<string, FunctionStack[]>): cena.Node {
            if (!lib.utils.isNullOrUndefined(nd) && nd.type === "FunctionExpression") {
                var fun: cena.FunctionExpression = castTo<cena.FunctionExpression>(nd);
                data.set(fun.id.name, []);
                currentFunction = fun.id.name;
            }
            if (nd.type === "VariableDeclarator") {
                var decl: cena.VariableDeclarator = castTo<cena.VariableDeclarator>(nd);
                data.get(currentFunction).push({
                    functionName: currentFunction,
                    id: decl.id
                });
            }
            return nd;
        }

        function decorateFunction(nd: cena.Node, data: Map<string, FunctionStack[]>): cena.Node {
            if (!lib.utils.isNullOrUndefined(nd) && nd.type === "FunctionExpression") {
                var fun: cena.FunctionExpression = castTo<cena.FunctionExpression>(nd);
                fun.marker.stack = data.get(fun.id.name);
            }
            return nd;
        }

        function removeDeclarations(nd: cena.Node, data: Map<string, FunctionStack[]>): cena.Node {

            if (nd.type === "VariableDeclarator") {
                var decl: cena.VariableDeclarator = castTo<cena.VariableDeclarator>(nd);
                nd.deleted = true;
            }
            return nd;
        }


        export function mark(prog: cena.Node) {
            var data = new Map<string, FunctionStack[]>();
            prog.postOrderTraverse(captureStack, data);
            prog.postOrderTraverse(decorateFunction, data);
            return prog.postOrderTraverse(removeDeclarations, data);
        }

        export function removeDecls(prog: cena.Node): esprima.Syntax.Program {
            var e = prog.toEsprima();
            var v = lib.ast.traverse.replace(e, {
                enter: function(node) {
                    if (node.type === "VariableDeclaration") this.remove();
                }
            });
            return v;
        }
    }
}