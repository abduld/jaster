
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
                markNeeded(sub.parent);
            } else if (nd.type === "Identifier") {
                var id: cena.Identifier = castTo<cena.Identifier>(nd);
                if (id.marker.neededInstruction || _.any(id.children, (child: cena.Node) => !lib.utils.isUndefined(child) && _.has(child, "neededInstruction"))) {
                    markNeeded(id);
                    data.set(id.name, true);
                }
                if (data.has(id.name)) {
                    markNeeded(id);
                    _.each(id.children, (nd: cena.Node) => markNeeded(nd, true));
                }
                markNeeded(id.parent);
            } else if (nd.type === "BinaryExpression") {
                var bop: cena.BinaryExpression = castTo<cena.BinaryExpression>(nd);
                if (bop.marker.neededInstruction || _.any(bop.children, (child: cena.Node) => !lib.utils.isUndefined(child) && _.has(child, "neededInstruction"))) {
                    markNeeded(bop);
                    _.each(bop.children, (nd: cena.Node) => markNeeded(nd, true));
                    markNeeded(bop.parent);
                }
            } else if (nd.type === "AssignmentExpression") {
                var ass: cena.AssignmentExpression = castTo<cena.AssignmentExpression>(nd);
                if (ass.marker.neededInstruction || _.any(bop.children, (child: cena.Node) => !lib.utils.isUndefined(child) && _.has(child, "neededInstruction"))) {
                    markNeeded(ass);
                    _.each(ass.children, (nd: cena.Node) => markNeeded(nd, true));
                    markNeeded(ass.parent);
                }
            } else {
            if (nd.marker.neededInstruction || _.any(nd.children, (child: cena.Node) => !lib.utils.isUndefined(child) && _.has(child, "neededInstruction"))) {
                markNeeded(nd);
                _.each(nd.children, (nd: cena.Node) => markNeeded(nd, true));
                markNeeded(nd.parent);
                }
            }
            return nd;
        }

        export function mark(prog: cena.Node) {
            var data = new Map<string, boolean>();
            return prog.postOrderTraverse(visitor, data);
        }
    }
}