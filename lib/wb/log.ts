

module lib {
    export module wb {
        export function wbLog(state, stack, ...args) {
            console.log(_.map(args, (arg) => arg.toString()).join(" "));
        }
    }
}