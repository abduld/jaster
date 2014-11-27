
module lib.ast {
    export module recast {

        export function print(node, options) {
            return new Printer(options).print(node);
        }

        export function prettyPrint(node, options) {
            return new Printer(options).printGenerically(node);
        }

        export function run(transformer, options) {
            return runFile(process.argv[2], transformer, options);
        }

        function runFile(path, transformer, options) {
            require("fs").readFile(path, "utf-8", function(err, code) {
                if (err) {
                    console.error(err);
                    return;
                }

                runString(code, transformer, options);
            });
        }

        function defaultWriteback(output) {
            process.stdout.write(output);
        }

        function runString(code, transformer, options) {
            var writeback = options && options.writeback || defaultWriteback;
            transformer(parse(code, options), function(node) {
                writeback(print(node, options).code);
            });
        }

    }
}