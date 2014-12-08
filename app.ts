
/// <reference path="./typings/assert/assert.d.ts" />
/// <reference path="./typings/lodash/lodash.d.ts" />
/// <reference path="./typings/parallel/parallel.d.ts" />
/// <reference path="./typings/q/Q.d.ts" />
/// <reference path="./typings/requirejs/require.d.ts" />
/// <reference path="./lib/ref.ts" />

module app {
    import c = lib.c;
    export class Greeter {
        constructor() {
        }

        start() {
            lib.utils.globals.output = lib.ast.importer.cena.fromCena(lib.example.mp1);
        }

        stop() {
        }

    }
}

var ast = lib.ast.importer.cena.fromCena(lib.example.mp1);
lib.ast.importer.memory.mark(ast);
//lib.ast.importer.stack.mark(ast);

var res = lib.ast.gen.generate(
    ast.toEsprima(),
    // we might have to do some extra think here (see https://github.com/estools/escodegen/wiki/Source-Map-Usage )
    { sourceMap: true, sourceMapWithCode: true, comment: true }
);

console.log(res.code);