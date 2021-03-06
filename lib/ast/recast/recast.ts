/// <reference path="../types/types.ts" />
/// <reference path="private.ts" />
/// <reference path="options.ts" />
/// <reference path="printer.ts" />
/// <reference path="types.ts" />
module lib.ast {
    export module recast {

        export function print(node, options) {
            return new Printer(options).print(node);
        }

        export function prettyPrint(node, options) {
            return new Printer(options).printGenerically(node);
        }

        function defaultWriteback(output) {
            console.log(output);
        }

        export function run(code, transformer, options) {
            var writeback = options && options.writeback || defaultWriteback;
            transformer(parse(code, options), function(node) {
                writeback(print(node, options).code);
            });
        }

    }
}
