/// <reference path="./logger.ts" />
/// <reference path="./assert.ts" />
/// <reference path="./rand.ts" />
/// <reference path="./guuid.ts" />
/// <reference path="./exception.ts" />
/// <reference path="./error.ts" />
/// <reference path="./timer.ts" />
/// <reference path="./testing.ts" />
define(["require", "exports", "./logger", "./assert", "./guuid"], function (require, exports, Logger_, Assert_, Guuid_) {
    var Utils = (function () {
        function Utils() {
            this.Logger = Logger_;
            this.Assert = Assert_;
            this.Guuid = Guuid_;
        }
        return Utils;
    })();
    ;
    var utils = new Utils();
    return utils;
});
