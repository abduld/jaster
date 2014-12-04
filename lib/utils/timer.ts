module lib.utils {
    export module timer {

        var now_ = performance.now ||
                /*
                 performance.mozNow ||
                 performance.msNow ||
                 performance.oNow ||
                 performance.webkitNow ||
                 */
            function () {
                return new Date().getTime();
            };

        export function now():number {
            return now_();
        }
    }
}
