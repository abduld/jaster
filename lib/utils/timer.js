var lib;
(function (lib) {
    var utils;
    (function (utils) {
        var timer;
        (function (timer) {
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
            function now() {
                return now_();
            }
            timer.now = now;
        })(timer = utils.timer || (utils.timer = {}));
    })(utils = lib.utils || (lib.utils = {}));
})(lib || (lib = {}));
