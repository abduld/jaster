
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

window.onload = () => {
    lib.ast.importer.cena.fromCena(lib.example.mp1)
};

lib.ast.importer.cena.fromCena(lib.example.mp1)