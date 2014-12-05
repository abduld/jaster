
/// <reference path="cena.ts" />
module lib.ast.importer {
    export module memory {
        import castTo = lib.utils.castTo;
        import cena = lib.ast.importer.cena;
        function markNeeded(nd: cena.Node) {
            nd.marker = _.extend(nd.marker, {
                neededinstruction: true
            });
        }
        function visitor(nd: cena.Node, data: Map<string, boolean>) : cena.Node {
            
            if (nd.type === "SubscriptExpression") {
                var sub: cena.SubscriptExpression = castTo<cena.SubscriptExpression>(nd);
                markNeeded(sub);
                _.each(sub.children, markNeeded);
                markNeeded(sub.parent);
            } else if (nd.type === "Identifier") {
                var id: cena.Identifier = castTo<cena.Identifier>(nd);
                if (id.marker.neededInstruction) {
                    data.set(id.name, true);
                } 
                if (data.has(id.name)) {
                    markNeeded(id);
                    _.each(id.children, markNeeded);
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