module lib.utils {
    export module timer {


        export function now():number {
            return performance.now();
        }
    }
}
