
/// <reference path="cena.ts" />
module lib.ast.importer {
    export module memory {
        import castTo = lib.utils.castTo;
        import cena = lib.ast.importer.cena;
        function markNeeded(nd: cena.Node, deep? : boolean) {
            nd.marker = _.extend(nd.marker, {
                neededInstruction: true
            });
            if (deep) {
                if (nd.hasChildren()) {
                    _.each(nd.children, (nd: cena.Node) => markNeeded(nd, deep));
                }
            }
        }
        function visitor(nd: cena.Node, data: Map<string, boolean>) : cena.Node {
            if (!lib.utils.isNullOrUndefined(nd.parent) && nd.parent.type === "FunctionExpression") {
                var fun: cena.FunctionExpression = castTo<cena.FunctionExpression>(nd);
                if (!_.contains(fun.attributes, "__global__")) {
                    markNeeded(fun, true);
                }
            }
            if (nd.type === "SubscriptExpression") {
                var sub: cena.SubscriptExpression = castTo<cena.SubscriptExpression>(nd);
                markNeeded(sub);
                _.each(sub.children, (nd : cena.Node) => markNeeded(nd, true));
                markNeeded(sub.parent, true);
            } else if (nd.marker.neededInstruction || _.any(nd.children, (child: cena.Node) => !lib.utils.isUndefined(child) && _.has(child, "neededInstruction"))) {
                markNeeded(nd);
                _.each(nd.children, (nd: cena.Node) => markNeeded(nd, true));
                markNeeded(nd.parent);
                }
            return nd;
        }

        export function mark(prog: cena.Node) {
            var data = new Map<string, boolean>();
            return prog.postOrderTraverse(visitor, data);
        }
    }
}